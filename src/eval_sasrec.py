import os
import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from models.sasrec import SASRec

class EvalDataset(Dataset):
    def __init__(self, processed_dir):
        # Load the sequences and the ground truth test targets
        self.seqs = np.load(os.path.join(processed_dir, 'train_seqs.npy'))
        self.targets = np.load(os.path.join(processed_dir, 'test_targets.npy'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # seqs array is [user_idx, item1, item2, ...] -> we slice [1:] to get just items
        # targets array is [user_idx, target_item] -> we take index 1 for the target
        return torch.LongTensor(self.seqs[idx][1:]), self.targets[idx][1]

def evaluate():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    PROCESSED_DIR = "data/processed"
    WEIGHTS_DIR = "models/weights"
    
    # Must match the exact architecture from train.py
    NUM_ITEMS = 22884
    MAX_SEQ_LEN = 50
    HIDDEN_DIM = 128
    NUM_HEADS = 4    # ⚡ UPDATED: From Grid Search
    NUM_BLOCKS = 3   # ⚡ UPDATED: From Grid Search
    BATCH_SIZE = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on device: {device}")

    # ==========================================
    # 2. LOAD DATA & MODEL
    # ==========================================
    dataset = EvalDataset(PROCESSED_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SASRec(
        num_items=NUM_ITEMS, 
        max_seq_len=MAX_SEQ_LEN, 
        hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, 
        num_blocks=NUM_BLOCKS, 
        dropout_rate=0.0, # Disable dropout for deterministic evaluation
        device=device
    ).to(device)

    model_path = os.path.join(WEIGHTS_DIR, 'sasrec_final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Lock the model into evaluation mode
    
    print("Model loaded. Starting rigorous full-catalog evaluation...")

    # ==========================================
    # 3. EVALUATION LOOP (HR@10, NDCG@10, MRR, Coverage)
    # ==========================================
    hits = 0.0
    ndcg = 0.0
    mrr = 0.0
    total_users = 0
    recommended_items_set = set() # ⚡ To track Catalog Coverage
    
    # Pre-create candidate tensor for all items (Token IDs: 1 to 22884)
    all_items = torch.arange(1, NUM_ITEMS + 1, device=device)

    with torch.no_grad():
        for seqs, targets in dataloader:
            seqs = seqs.to(device)
            targets = targets.numpy()
            
            # Extract the final hidden representation of the user sequence
            # log2feats returns (batch, seq_len, hidden_dim). We only want the last timestep: [:, -1, :]
            user_feats = model.log2feats(seqs)[:, -1, :] 
            
            # Look up embeddings for all 22,884 movies
            item_embs = model.item_emb(all_items) # (NUM_ITEMS, hidden_dim)
            
            # Matrix multiplication to score every user against every movie
            # user_feats: (batch, hidden_dim) | item_embs.T: (hidden_dim, NUM_ITEMS)
            logits = user_feats.matmul(item_embs.transpose(-1, -2)) # -> (batch, NUM_ITEMS)
            
            # Get the top 10 highest scored movies for each user in the batch
            _, top_indices = torch.topk(logits, k=10, dim=-1)
            
            # Indices are 0-based relative to the `all_items` tensor, so we add 1 to get actual Item IDs
            top_items = top_indices.cpu().numpy() + 1 
            
            for i in range(len(targets)):
                target_item = targets[i]
                top_10 = top_items[i]
                
                # ⚡ TRACK COVERAGE: Add all recommended items to our unique set
                recommended_items_set.update(top_10)
                
                # Check if the actual next movie they watched is in the top 10
                if target_item in top_10:
                    hits += 1
                    
                    # Calculate Normalized Discounted Cumulative Gain (rewarding higher ranks)
                    rank = np.where(top_10 == target_item)[0][0] + 1
                    ndcg += 1 / math.log2(rank + 1)
                    mrr += 1 / rank  # ⚡ NEW: Calculate Mean Reciprocal Rank
                    
            total_users += len(targets)
            
            if total_users % 25600 == 0:
                print(f"Processed {total_users}/{len(dataset)} users...")

    final_hr = hits / total_users
    final_ndcg = ndcg / total_users
    final_mrr = mrr / total_users
    
    # ⚡ Calculate Coverage: (Unique Items Recommended) / (Total Items in Catalog)
    catalog_coverage = len(recommended_items_set) / NUM_ITEMS

    print("\n" + "=" * 40)
    print("🏆 ADVANCED SASREC EVALUATION RESULTS")
    print("=" * 40)
    print(f"Hit Ratio @ 10   : {final_hr:.4f}")
    print(f"NDCG @ 10        : {final_ndcg:.4f}")
    print(f"MRR @ 10         : {final_mrr:.4f}")
    print(f"Catalog Coverage : {catalog_coverage:.4f} ({len(recommended_items_set)} unique movies)")
    print("=" * 40)

if __name__ == '__main__':
    evaluate()