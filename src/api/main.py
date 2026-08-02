import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Ensure models can be imported from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sasrec import SASRec

# ==========================================
# 1. GLOBAL STATE & CONFIGURATION
# ==========================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
WEIGHTS_DIR = "models/weights"

# Global Variables
engines = {
    "sasrec_model": None,
    "idx_to_movie": {},
    "movie_to_idx": {},
    "movie_dict": {},
    "tmdb_map": {},
    "genome_map": {}
}

class SequenceRequest(BaseModel):
    user_history: list[int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI Lifespan manager.
    Handles startup and shutdown events cleanly without DeprecationWarnings.
    """
    print("🚀 Initializing AI Engines...")
    
    # 1. Load Mappings
    try:
        with open(os.path.join(PROCESSED_DIR, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            engines["movie_to_idx"] = mappings['movie_to_idx']
            engines["idx_to_movie"] = {v: k for k, v in mappings['movie_to_idx'].items()}
            
        with open(os.path.join(PROCESSED_DIR, 'movie_meta_dict.pkl'), 'rb') as f:
            engines["movie_dict"] = pickle.load(f)
    except FileNotFoundError as e:
        print(f"⚠️ Mapping files missing: {e}.")

    # 2. Load TMDB Links
    try:
        print("🎬 Loading TMDB IDs...")
        links_df = pd.read_csv(os.path.join(RAW_DIR, 'link.csv'))
        engines["tmdb_map"] = dict(zip(links_df.movieId, links_df.tmdbId.fillna(0).astype(int)))
    except Exception as e:
        print(f"⚠️ link.csv missing: {e}")

    # 3. Load Genome Tags (For the Netflix UI Hover Info)
    try:
        print("🧬 Loading Genome Tags (This takes a few seconds)...")
        g_scores = pd.read_csv(os.path.join(RAW_DIR, 'genome-scores.csv'))
        g_tags = pd.read_csv(os.path.join(RAW_DIR, 'genome-tags.csv'))
        
        # Sort by relevance to find the top 2 tags for every movie
        top_scores = g_scores.sort_values(['movieId', 'relevance'], ascending=[True, False]).groupby('movieId').head(2)
        top_tags_merged = pd.merge(top_scores, g_tags, on='tagId')
        
        # Create a dictionary mapping MovieID to a list of its top 2 tags
        engines["genome_map"] = top_tags_merged.groupby('movieId')['tag'].apply(list).to_dict()
    except Exception as e:
        print(f"⚠️ Genome tags missing: {e}. Falling back to empty tags.")

    # 4. Load Sequential Engine (SASRec)
    print("🧠 Loading Sequential Engine (SASRec)...")
    device = torch.device("cpu")
    sasrec_model = SASRec(
        num_items=22884, max_seq_len=50, hidden_dim=128, 
        num_heads=4, num_blocks=3, dropout_rate=0.0, device=device
    )
    
    try:
        sasrec_model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'sasrec_final_model.pth'), map_location=device))
        sasrec_model.eval()
        engines["sasrec_model"] = sasrec_model
    except FileNotFoundError:
        print("⚠️ sasrec_final_model.pth not found! Predictions will be random.")

    print("✅ All systems online!")
    
    yield # API is running
    
    print("🛑 Shutting down AI Engines...")
    engines.clear()

app = FastAPI(
    title="MovieLens SASRec API", 
    description="Serving SASRec and Metadata.",
    lifespan=lifespan
)

# ⚡ ENABLE CORS (CRITICAL FOR NETFLIX FRONTEND)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all frontend domains to access the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. ENDPOINTS
# ==========================================

@app.get("/search")
async def search_movies(q: str):
    """Searches for movies by title to populate the Netflix Search Bar."""
    results = []
    movie_dict = engines["movie_dict"]
    tmdb_map = engines["tmdb_map"]
    genome_map = engines["genome_map"]

    for raw_id, text in movie_dict.items():
        title = text.split(". Genres:")[0].replace("Movie Title: ", "")
        
        # Simple text matching
        if q.lower() in title.lower():
            tmdb_id = tmdb_map.get(raw_id, 0)
            tags = genome_map.get(raw_id, [])
            
            results.append({
                "movie_id": int(raw_id),
                "title": title,
                "tmdb_id": int(tmdb_id) if tmdb_id != 0 else None,
                "tags": tags
            })
            
        # Return top 15 matches to keep UI snappy
        if len(results) >= 15:
            break
            
    return {"results": results}

@app.post("/recommend/sequence")
async def get_sequence_recommendations(req: SequenceRequest):
    """Predicts the next movie and returns rich TMDB metadata for posters."""
    if not req.user_history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
        
    sasrec_model = engines["sasrec_model"]
    if sasrec_model is None:
        raise HTTPException(status_code=500, detail="SASRec Engine offline")

    movie_to_idx = engines["movie_to_idx"]
    
    # 1. Convert Raw MovieLens IDs from the frontend into Model Token IDs (1 to 22884)
    # This safely drops any movie the AI didn't see during training
    token_history = [movie_to_idx[mid] for mid in req.user_history if mid in movie_to_idx]
    
    if not token_history:
        raise HTTPException(status_code=400, detail="No valid AI-trained movies found in history.")

    # Pad or truncate sequence to exactly 50 items
    seq = token_history[-50:]
    padded_seq = np.zeros(50, dtype=np.int64)
    padded_seq[-len(seq):] = seq
    
    tensor_seq = torch.LongTensor(np.array([padded_seq]))
    all_items = torch.arange(1, 22884 + 1)
    
    with torch.no_grad():
        logits = sasrec_model.predict(tensor_seq, all_items)
        _, top_indices = torch.topk(logits[0], k=15) # Get top 15 for Netflix row
        
    # Convert to native python ints
    top_items = (top_indices.numpy() + 1).tolist()
    
    recommendations = []
    idx_to_movie = engines["idx_to_movie"]
    movie_dict = engines["movie_dict"]
    tmdb_map = engines["tmdb_map"]
    genome_map = engines["genome_map"]

    for item_token in top_items:
        raw_id = int(idx_to_movie.get(item_token, 0)) 
        
        # Skip if somehow mapped to 0 or movie already in history
        if raw_id == 0 or raw_id in req.user_history:
            continue
            
        title_full = movie_dict.get(raw_id, f"Unknown Movie (ID: {raw_id})")
        title = title_full.split(". Genres:")[0].replace("Movie Title: ", "")
        
        tmdb_id = tmdb_map.get(raw_id, 0)
        tags = genome_map.get(raw_id, [])
        
        recommendations.append({
            "movie_id": int(raw_id), 
            "title": title,
            "tmdb_id": int(tmdb_id) if tmdb_id != 0 else None,
            "tags": tags
        })
        
    return {
        "engine_used": "SASRec Transformer (Sequential)",
        "recommended_movies": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    # Pass the 'app' object directly instead of a string to fix pathing issues
    uvicorn.run(app, host="127.0.0.1", port=8000)