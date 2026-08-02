import os
import time
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import DataLoader
from models.sasrec import SASRecDataset, SASRec

def run_grid_search():
    # ==========================================
    # 1. GRID SEARCH SPACE
    # ==========================================
    learning_rates = [0.001, 0.0005]
    dropouts = [0.2, 0.1]
    num_heads_list = [2, 4]
    num_blocks_list = [2, 3]
    
    # Generate all 16 combinations
    grid = list(itertools.product(learning_rates, dropouts, num_heads_list, num_blocks_list))
    
    # CONSTANTS
    NUM_EPOCHS = 15        # ⚡ Reduced for rapid prototyping
    NUM_ITEMS = 22884
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 256
    HIDDEN_DIM = 128
    PROCESSED_DIR = "data/processed"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting SASRec Grid Search on {device}. Total combinations to test: {len(grid)}")

    # ==========================================
    # 2. LOAD DATA & FUSED EMBEDDINGS
    # ==========================================
    dataset = SASRecDataset(processed_dir=PROCESSED_DIR, num_items=NUM_ITEMS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    w2v_path = os.path.join(PROCESSED_DIR, 'fused_embeddings.npy')
    if os.path.exists(w2v_path):
        pretrained_matrix = np.load(w2v_path)
        print(f"✅ Loaded SOTA Fused Embeddings!")
    else:
        print("⚠️ Fused Matrix not found!")
        return

    # Tracking the winner
    best_loss = float('inf')
    best_params = None

    # ==========================================
    # 3. AUTOMATED TRAINING LOOP
    # ==========================================
    for idx, (lr, dropout, heads, blocks) in enumerate(grid):
        print("\n" + "="*50)
        print(f"🧪 EXPERIMENT [{idx+1}/{len(grid)}] | LR: {lr} | Drop: {dropout} | Heads: {heads} | Blocks: {blocks}")
        print("="*50)
        
        # Initialize a fresh SASRec model for this experiment
        model = SASRec(
            num_items=NUM_ITEMS, 
            max_seq_len=MAX_SEQ_LEN, 
            hidden_dim=HIDDEN_DIM, 
            num_heads=heads, 
            num_blocks=blocks, 
            dropout_rate=dropout, 
            device=device,
            pretrained_item_emb=pretrained_matrix
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        model.train()
        final_epoch_loss = 0.0
        
        for epoch in range(1, NUM_EPOCHS + 1):
            total_loss = 0.0
            
            for log_seqs, pos_targets, neg_targets in dataloader:
                log_seqs, pos_targets, neg_targets = log_seqs.to(device), pos_targets.to(device), neg_targets.to(device)
                
                pos_logits, neg_logits = model(log_seqs, pos_targets, neg_targets)
                pos_labels, neg_labels = torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
                
                loss_pos = criterion(pos_logits, pos_labels)
                loss_neg = criterion(neg_logits, neg_labels)
                
                mask = (log_seqs != 0).float()
                loss = (loss_pos + loss_neg) * mask
                loss = loss.sum() / mask.sum()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            final_epoch_loss = total_loss / len(dataloader)
            
            # Only print the 15th epoch to keep the terminal clean
            if epoch == NUM_EPOCHS:
                print(f"🏁 Final Epoch Loss: {final_epoch_loss:.4f}")
        
        # Check if this is the new best model
        if final_epoch_loss < best_loss:
            best_loss = final_epoch_loss
            best_params = (lr, dropout, heads, blocks)
            
            # Save the winning weights
            save_path = os.path.join("models/weights", 'sasrec_grid_winner.pth')
            torch.save(model.state_dict(), save_path)
            print("🌟 NEW BEST MODEL FOUND! Weights Saved.")

    # ==========================================
    # 4. GRID SEARCH COMPLETE
    # ==========================================
    print("\n" + "🏆"*20)
    print("SASREC GRID SEARCH COMPLETE!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Winning Parameters -> LR: {best_params[0]} | Drop: {best_params[1]} | Heads: {best_params[2]} | Blocks: {best_params[3]}")
    print("🏆"*20)

if __name__ == '__main__':
    run_grid_search()