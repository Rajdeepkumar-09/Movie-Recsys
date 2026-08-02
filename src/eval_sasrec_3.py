import os
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from models.sasrec import SASRec

class EvalDataset(Dataset):
    def __init__(self, processed_dir, limit=None):
        # Load the sequences and the ground truth test targets
        self.seqs = np.load(os.path.join(processed_dir, 'train_seqs.npy'))
        self.targets = np.load(os.path.join(processed_dir, 'test_targets.npy'))
        
        # Slice the dataset to evaluate only a specific number of users
        if limit is not None:
            self.seqs = self.seqs[:limit]
            self.targets = self.targets[:limit]
            
        # Build a set of all movies each user has EVER watched in the training data.
        # We need this so we don't accidentally pick a movie they already watched as a "negative" distractor.
        self.user_history = defaultdict(set)
        for i in range(len(self.seqs)):
            user_idx = self.seqs[i][0]
            # Add all non-zero items from the sequence
            items = [item for item in self.seqs[i][1:] if item != 0]
            self.user_history[user_idx].update(items)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        user_idx = self.seqs[idx][0]
        # Return sequence, target item, and the user's ID
        return torch.LongTensor(self.seqs[idx][1:]), self.targets[idx][1], user_idx

def evaluate():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    PROCESSED_DIR = "data/processed"
    WEIGHTS_DIR = "models/weights"
    
    NUM_ITEMS = 22884
    MAX_SEQ_LEN = 50
    HIDDEN_DIM = 128
    NUM_HEADS = 4    
    NUM_BLOCKS = 3   
    BATCH_SIZE = 256
    
    USERS_TO_EVALUATE = 500
    NUM_NEGATIVES = 999 # The number of distractors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. LOAD DATA & MODEL
    # ==========================================
    dataset = EvalDataset(PROCESSED_DIR, limit=USERS_TO_EVALUATE)
    
    # We must evaluate one user at a time (batch_size=1) for this negative sampling logic
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SASRec(
        num_items=NUM_ITEMS, max_seq_len=MAX_SEQ_LEN, hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS, dropout_rate=0.0, device=device
    ).to(device)

    model_path = os.path.join(WEIGHTS_DIR, 'sasrec_final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded. Evaluating on exactly {USERS_TO_EVALUATE} users with {NUM_NEGATIVES} negative samples each...")

    # ==========================================
    # 3. EVALUATION LOOP (WITH NEGATIVE SAMPLING)
    # ==========================================
    hits = 0.0
    ndcg = 0.0
    mrr = 0.0
    total_users = 0
    
    all_possible_items = set(range(1, NUM_ITEMS + 1))

    with torch.no_grad():
        for seqs, target_item_tensor, user_idx in dataloader:
            seqs = seqs.to(device)
            target_item = target_item_tensor.item()
            user_idx = user_idx.item()
            
            # 1. Generate 999 Negative Items
            # A negative item is a movie the user has NEVER seen.
            seen_items = dataset.user_history[user_idx]
            # Exclude seen items AND the target item from the candidate pool
            valid_negatives = list(all_possible_items - seen_items - {target_item})
            
            # Randomly pick 999
            negatives = random.sample(valid_negatives, NUM_NEGATIVES)
            
            # 2. Combine the True item with the 999 fakes to create the 1,000 candidates
            candidates = [target_item] + negatives
            candidates_tensor = torch.LongTensor(candidates).to(device)
            
            # 3. Get Model Predictions
            # We ask the model to predict scores ONLY for the 1,000 candidates, not all 22,000
            user_vector = model.log2feats(seqs)[:, -1, :] # Shape: (1, 128)
            item_embs = model.item_emb(candidates_tensor) # Shape: (1000, 128)
            
            # Calculate similarity scores
            logits = user_vector.matmul(item_embs.transpose(-1, -2)).squeeze() # Shape: (1000,)
            
            # 4. Rank the 1,000 candidates
            # Get the indices of the top 10 highest scored candidates
            _, top_10_indices = torch.topk(logits, k=10)
            top_10_indices = top_10_indices.cpu().numpy()
            
            # Because we put the target_item at index 0 of the candidates list,
            # if the index 0 is in the top_10_indices, it's a hit!
            if 0 in top_10_indices:
                hits += 1
                rank = np.where(top_10_indices == 0)[0][0] + 1
                ndcg += 1 / math.log2(rank + 1)
                mrr += 1 / rank 
                    
            total_users += 1

    final_hr = hits / total_users
    final_ndcg = ndcg / total_users
    final_mrr = mrr / total_users

    print("\n" + "=" * 60)
    print("EVALUATION METRICS (1 POSITIVE vs 999 NEGATIVES)")
    print("=" * 60)
    print(f"Users Evaluated: {total_users}")
    print(f"Hit Ratio@10:    {final_hr:.4f}")
    print(f"NDCG@10:         {final_ndcg:.4f}")
    print(f"MRR@10:          {final_mrr:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    evaluate()