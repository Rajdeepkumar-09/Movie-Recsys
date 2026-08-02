import os
import torch
import numpy as np
import pandas as pd
import math
import pickle
from torch.utils.data import DataLoader, Dataset
from models.sasrec import SASRec

class EvalDataset(Dataset):
    def __init__(self, processed_dir, limit=500):
        # Load the sequences and the ground truth test targets
        self.seqs = np.load(os.path.join(processed_dir, 'train_seqs.npy'))
        self.targets = np.load(os.path.join(processed_dir, 'test_targets.npy'))
        
        # ⚡ Constrain to exactly 500 users to match your friend's benchmark
        if limit is not None:
            self.seqs = self.seqs[:limit]
            self.targets = self.targets[:limit]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.LongTensor(self.seqs[idx][1:]), self.targets[idx][1]

def load_category_mappings(raw_dir, processed_dir):
    print("Loading genre and category mappings...")
    
    # 1. Load Token -> Raw Movie ID mapping
    with open(os.path.join(processed_dir, 'mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
        idx_to_movie = {v: k for k, v in mappings['movie_to_idx'].items()}

    # 2. Load Raw Movie ID -> Primary Genre mapping
    movies_df = pd.read_csv(os.path.join(raw_dir, 'movie.csv'))
    movie_to_primary_genre = {}
    for _, row in movies_df.iterrows():
        # We take the first genre (e.g. "Action|Comedy" -> "Action")
        genres = str(row['genres']).split('|')
        movie_to_primary_genre[row['movieId']] = genres[0] if genres else "Unknown"

    # 3. Create direct Token -> Genre mapping for ultra-fast evaluation
    token_to_genre = {}
    for token, movie_id in idx_to_movie.items():
        token_to_genre[token] = movie_to_primary_genre.get(movie_id, "Unknown")
        
    return token_to_genre

def evaluate_similarity():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    PROCESSED_DIR = "data/processed"
    RAW_DIR = "data/raw"
    WEIGHTS_DIR = "models/weights"
    
    NUM_ITEMS = 22884
    MAX_SEQ_LEN = 50
    HIDDEN_DIM = 128
    NUM_HEADS = 4    
    NUM_BLOCKS = 3   
    BATCH_SIZE = 256
    USERS_TO_EVALUATE = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. LOAD DATA, MAPPINGS & MODEL
    # ==========================================
    token_to_genre = load_category_mappings(RAW_DIR, PROCESSED_DIR)
    
    dataset = EvalDataset(PROCESSED_DIR, limit=USERS_TO_EVALUATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SASRec(
        num_items=NUM_ITEMS, max_seq_len=MAX_SEQ_LEN, hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS, dropout_rate=0.0, device=device
    ).to(device)

    model_path = os.path.join(WEIGHTS_DIR, 'sasrec_final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded. Evaluating Category Interaction on {USERS_TO_EVALUATE} users...")

    # ==========================================
    # 3. EVALUATION LOOP (CATEGORY MATCHING)
    # ==========================================
    category_hits = 0.0
    category_ndcg = 0.0
    category_mrr = 0.0
    total_users = 0
    
    all_items = torch.arange(1, NUM_ITEMS + 1, device=device)

    with torch.no_grad():
        for seqs, targets in dataloader:
            seqs = seqs.to(device)
            targets = targets.numpy()
            
            logits = model.predict(seqs, all_items)
            _, top_indices = torch.topk(logits, k=10, dim=-1)
            
            top_items = top_indices.cpu().numpy() + 1 
            
            for i in range(len(targets)):
                target_token = targets[i]
                top_10 = top_items[i]
                
                # Look up what category the user *actually* wanted to watch
                target_category = token_to_genre.get(target_token, "UnknownTarget")
                
                # Check if our model successfully recommended that category
                for rank, rec_token in enumerate(top_10):
                    rec_category = token_to_genre.get(rec_token, "UnknownRec")
                    
                    if rec_category == target_category and target_category != "Unknown":
                        category_hits += 1
                        # Mathematical discounting based on how high up the list the category match was
                        category_ndcg += 1 / math.log2((rank + 1) + 1)
                        category_mrr += 1 / (rank + 1)
                        break # We only count the highest-ranked category match
                    
            total_users += len(targets)

    final_hr = category_hits / total_users
    final_ndcg = category_ndcg / total_users
    final_mrr = category_mrr / total_users

    # ⚡ PRINTING FORMATTED LIKE YOUR FRIEND'S LOG
    print("\n" + "=" * 60)
    print("SIMILARITY & CATEGORY EVALUATION METRICS")
    print("=" * 60)
    print(f"Users Evaluated: {total_users}")
    print(f"Category Hit Ratio@10:    {final_hr:.4f}")
    print(f"Category NDCG@10:         {final_ndcg:.4f}")
    print(f"Category MRR@10:          {final_mrr:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_similarity()