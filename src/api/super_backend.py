import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Ensure models can be imported from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==========================================
# 0. PICKLE SHIMS (CRITICAL FOR LOADING ARTIFACTS)
# ==========================================
# We MUST define this class here in the main execution script so that 
# when Suhas's model tries to unpickle the movie database, Python knows what this class is.
class MovieMetadataDB:
    """Minimal reconstruction of the training-side MovieMetadataDB."""
    def get(self, key, default=None):
        return self.data.get(key, default)
    def __getitem__(self, key):
        return self.data[key]
    def __contains__(self, key):
        return key in self.data
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def keys(self):
        return self.data.keys()
    def items(self):
        return self.data.items()
    def values(self):
        return self.data.values()

# ⚡ THE PROFESSIONAL FIX: Inject the class into BOTH the __main__ module and sys.modules['__main__']
# so pickle can find it regardless of how uvicorn starts the app.
import __main__
setattr(__main__, 'MovieMetadataDB', MovieMetadataDB)
if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'MovieMetadataDB', MovieMetadataDB)

# Import the models AFTER defining the shim
from models.sasrec import SASRec
from models.manushri.manushri_model import Model1 as ManushriModel
from models.suhas.suhas_model import Model2Recommender as SuhasModel

# ==========================================
# 1. GLOBAL STATE & CONFIGURATION
# ==========================================
engines = {
    "manushri": None,
    "suhas": None,
    "rajdeep": None,
    "movie_to_idx": {}, 
    "idx_to_movie": {}, 
    "movie_dict": {},
    "tmdb_map": {}
}

class OnboardingRequest(BaseModel):
    ratings: list[list[float]] 

class SequenceRequest(BaseModel):
    user_history: list[int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing Super Backend (3-Model Ensemble)...")
    
    # --- 1. Load Core Mappings ---
    try:
        with open('data/processed/mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            engines["movie_to_idx"] = mappings['movie_to_idx']
            engines["idx_to_movie"] = {v: k for k, v in mappings['movie_to_idx'].items()}
        with open('data/processed/movie_meta_dict.pkl', 'rb') as f:
            engines["movie_dict"] = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Missing core mappings: {e}")

    try:
        links_df = pd.read_csv('data/raw/link.csv')
        engines["tmdb_map"] = dict(zip(links_df.movieId, links_df.tmdbId.fillna(0).astype(int)))
    except Exception as e:
        print(f"⚠️ Missing TMDB map: {e}")

    # --- 2. Load Manushri (Model 1) ---
    print("🧠 Loading Manushri's ALS Model...")
    try:
        engines["manushri"] = ManushriModel(artifacts_dir="src/models/manushri/artifacts")
    except Exception as e:
        print(f"⚠️ Failed to load Manushri's Model: {e}")

    # --- 3. Load Suhas (Model 2) ---
    print("🧠 Loading Suhas's Multi-Modal Network...")
    try:
        engines["suhas"] = SuhasModel(device='cpu')
    except Exception as e:
        print(f"⚠️ Failed to load Suhas's Model: {e}")

    # --- 4. Load Rajdeep (Model 3) ---
    print("🧠 Loading Rajdeep's SASRec Transformer...")
    try:
        device = torch.device("cpu")
        sasrec = SASRec(
            num_items=22884, max_seq_len=50, hidden_dim=128, 
            num_heads=4, num_blocks=3, dropout_rate=0.0, device=device
        )
        sasrec.load_state_dict(torch.load('models/weights/sasrec_final_model.pth', map_location=device))
        sasrec.eval()
        engines["rajdeep"] = sasrec
    except Exception as e:
        print(f"⚠️ Failed to load Rajdeep's Model: {e}")

    print("✅ Super Backend Online!")
    yield 
    engines.clear()

app = FastAPI(title="RecFlix Super Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 2. THE SUPER ORCHESTRATOR ENDPOINTS
# ==========================================
@app.post("/recommend/homepage")
async def generate_homepage(req: OnboardingRequest):
    if len(req.ratings) < 5:
        raise HTTPException(status_code=400, detail="Please provide at least 5 ratings to start.")

    manushri = engines.get("manushri")
    suhas = engines.get("suhas")
    rajdeep = engines.get("rajdeep")
    movie_dict = engines["movie_dict"]

    if not manushri or not suhas or not rajdeep:
         raise HTTPException(status_code=500, detail="One or more AI engines are offline.")

    raw_user_ratings = [(int(mid), float(rating)) for mid, rating in req.ratings]
    user_rated_mids = [mid for mid, _ in raw_user_ratings]
    
    # ---------------------------------------------------------
    # STEP 1: MANUSHRI (The Baseline & Candidate Generator)
    # ---------------------------------------------------------
    # Filter out movies Manushri's model hasn't seen to prevent her ValueError
    m1_user_ratings = [(mid, min(rating, 5.0)) for mid, rating in raw_user_ratings if mid in manushri.movie_to_idx]
    
    if not m1_user_ratings:
        raise HTTPException(status_code=400, detail="Please select older/more popular movies. The ALS Baseline doesn't recognize these.")

    # Her model requires exactly 15 ratings. Pad with the first valid movie if needed.
    while len(m1_user_ratings) < 15:
        m1_user_ratings.append(m1_user_ratings[0]) 
        
    m1_user_ratings = m1_user_ratings[:15] # Ensure exactly 15

    print("Executing Model 1 (Manushri)...")
    m1_output = manushri.recommend_from_movie_ratings(m1_user_ratings, top_n=15, candidate_k=200)
    
    row_1_recs = m1_output["recommended_for_you"]
    top_200_mids = m1_output["top_200_movie_ids"]

    # ---------------------------------------------------------
    # STEP 2: SUHAS (The Categorizer)
    # ---------------------------------------------------------
    print("Executing Model 2 (Suhas)...")
    best_movie_tuple = max(raw_user_ratings, key=lambda x: x[1])
    anchor_movie_id = best_movie_tuple[0]
    best_movie_title = movie_dict.get(anchor_movie_id, "").split(". Genres:")[0].replace("Movie Title: ", "")
    
    # Cap 5.1 back to 5.0 for Suhas's User Embedding MLP
    suhas_history_ids = [(mid, min(rating, 5.0)) for mid, rating in raw_user_ratings]

    # Bypass run_production_demo and call recommend directly with IDs!
    m2_output = suhas.recommend(
        user_rated_items=suhas_history_ids,
        top_400_mids=top_200_mids,
        syw_target_mid=anchor_movie_id,
        n_syw=15,
        n_got=15
    )

    # ---------------------------------------------------------
    # STEP 3: RAJDEEP (The Sequential Momentum Engine)
    # ---------------------------------------------------------
    print("Executing Model 3 (Rajdeep)...")
    movie_to_idx = engines["movie_to_idx"]
    idx_to_movie = engines["idx_to_movie"]
    
    token_history = [movie_to_idx[mid] for mid in user_rated_mids if mid in movie_to_idx]
    
    seq = token_history[-50:]
    padded_seq = np.zeros(50, dtype=np.int64)
    padded_seq[-len(seq):] = seq
    tensor_seq = torch.LongTensor(np.array([padded_seq]))
    
    # Run against the FULL catalog of items (1 to 22884)
    all_items = torch.arange(1, 22884 + 1)
    
    with torch.no_grad():
        logits = rajdeep.predict(tensor_seq, all_items)
        _, top_indices = torch.topk(logits[0], k=30) # Fetch 30 just in case we need to filter
        
    top_sasrec_tokens = (top_indices.numpy() + 1).tolist()
    
    row_2_recs = []
    for token in top_sasrec_tokens:
        raw_id = int(idx_to_movie.get(token, 0))
        if raw_id != 0 and raw_id not in user_rated_mids:
            title = movie_dict.get(raw_id, "").split(". Genres:")[0].replace("Movie Title: ", "")
            row_2_recs.append({"movie_id": raw_id, "title": title})
        if len(row_2_recs) >= 15: # Stop when we have enough for a Netflix row
            break

    # ---------------------------------------------------------
    # FINAL ASSEMBLY
    # ---------------------------------------------------------
    print("✅ Assembly Complete. Returning Homepage Payload.")
    
    def format_row(movie_list):
        formatted = []
        for m in movie_list:
            mid = m.get("movieId") or m.get("movie_id")
            title = m.get("title") or movie_dict.get(mid, "").split(". Genres:")[0].replace("Movie Title: ", "")
            tmdb_id = engines["tmdb_map"].get(mid, 0)
            formatted.append({"movie_id": mid, "title": title, "tmdb_id": int(tmdb_id) if tmdb_id != 0 else None})
        return formatted

    display_title = best_movie_title if best_movie_title else "Your Recent Activity"

    return {
        "status": "success",
        "rows": [
            {
                "title": "Recommended For You",
                "tag": "ALS Taste Match",
                "movies": format_row(row_1_recs)
            },
            {
                "title": "Keep the Momentum Going",
                "tag": "SASRec Sequence Prediction",
                "movies": format_row(row_2_recs)
            },
            {
                "title": f"Because You Watched {display_title}",
                "tag": "Similarity Engine",
                "movies": format_row(m2_output["since_you_watched"])
            },
            {
                "title": "Go Off Trail",
                "tag": "Exploration Engine",
                "movies": format_row(m2_output["go_off_trail"])
            },
            {
                "title": "Movie Marathon",
                "tag": "High Likeability",
                "movies": format_row(m2_output["movie_marathon"])
            }
        ]
    }

@app.get("/search")
async def search_movies(q: str):
    results = []
    movie_dict = engines["movie_dict"]
    tmdb_map = engines["tmdb_map"]

    for raw_id, text in movie_dict.items():
        title = text.split(". Genres:")[0].replace("Movie Title: ", "")
        if q.lower() in title.lower():
            tmdb_id = tmdb_map.get(raw_id, 0)
            results.append({
                "movie_id": int(raw_id),
                "title": title,
                "tmdb_id": int(tmdb_id) if tmdb_id != 0 else None
            })
        if len(results) >= 15:
            break
    return {"results": results}

@app.post("/recommend/sequence")
async def get_sequence_recommendations(req: SequenceRequest):
    if not req.user_history:
        raise HTTPException(status_code=400, detail="History cannot be empty")
        
    sasrec_model = engines["rajdeep"]
    if sasrec_model is None:
        raise HTTPException(status_code=500, detail="SASRec Engine offline")

    movie_to_idx = engines["movie_to_idx"]
    idx_to_movie = engines["idx_to_movie"]
    movie_dict = engines["movie_dict"]
    tmdb_map = engines["tmdb_map"]
    
    token_history = [movie_to_idx[mid] for mid in req.user_history if mid in movie_to_idx]
    
    if not token_history:
        raise HTTPException(status_code=400, detail="No valid AI-trained movies found in history.")

    seq = token_history[-50:]
    padded_seq = np.zeros(50, dtype=np.int64)
    padded_seq[-len(seq):] = seq
    tensor_seq = torch.LongTensor(np.array([padded_seq]))
    all_items = torch.arange(1, 22884 + 1)
    
    with torch.no_grad():
        logits = sasrec_model.predict(tensor_seq, all_items)
        _, top_indices = torch.topk(logits[0], k=15) 
        
    top_items = (top_indices.numpy() + 1).tolist()
    
    recommendations = []
    for item_token in top_items:
        raw_id = int(idx_to_movie.get(item_token, 0)) 
        if raw_id == 0 or raw_id in req.user_history:
            continue
            
        title_full = movie_dict.get(raw_id, f"Unknown Movie (ID: {raw_id})")
        title = title_full.split(". Genres:")[0].replace("Movie Title: ", "")
        tmdb_id = tmdb_map.get(raw_id, 0)
        
        recommendations.append({
            "movie_id": int(raw_id), 
            "title": title,
            "tmdb_id": int(tmdb_id) if tmdb_id != 0 else None
        })
        
    return {
        "engine_used": "SASRec Transformer (Sequential)",
        "recommended_movies": recommendations
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("super_backend:app", host="127.0.0.1", port=8000, reload=True)
