import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.sasrec import SASRecDataset, SASRec

def train_final():
    PROCESSED_DIR = "data/processed"
    WEIGHTS_DIR = "models/weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # 🏆 WINNING PARAMETERS FROM GRID SEARCH
    NUM_ITEMS = 22884      
    MAX_SEQ_LEN = 50       
    BATCH_SIZE = 256       
    HIDDEN_DIM = 128       
    NUM_HEADS = 4          # From Grid Search
    NUM_BLOCKS = 3         # From Grid Search
    DROPOUT_RATE = 0.1     # From Grid Search
    INITIAL_LR = 0.001     # From Grid Search
    NUM_EPOCHS = 100       
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Final Training on {device} with Cosine Annealing!")

    dataset = SASRecDataset(processed_dir=PROCESSED_DIR, num_items=NUM_ITEMS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # LOAD SOTA FUSED EMBEDDINGS
    w2v_path = os.path.join(PROCESSED_DIR, 'fused_embeddings.npy')
    if os.path.exists(w2v_path):
        pretrained_matrix = np.load(w2v_path)
        print("✅ Loaded SOTA Fused Embeddings!")
    else:
        pretrained_matrix = None
        print("⚠️ Fused Matrix not found! Using random weights.")

    model = SASRec(
        num_items=NUM_ITEMS, max_seq_len=MAX_SEQ_LEN, hidden_dim=HIDDEN_DIM, 
        num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS, dropout_rate=DROPOUT_RATE, 
        device=device, pretrained_item_emb=pretrained_matrix
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # ⚡ THE SECRET SAUCE: Cosine Annealing Learning Rate Scheduler
    # This will slowly curve your learning rate down to 0, forcing the model 
    # to make micro-adjustments in the final epochs to perfect the NDCG ranking!
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    model.train()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        total_loss = 0.0
        
        for batch_idx, (log_seqs, pos_targets, neg_targets) in enumerate(dataloader):
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
        
        # ⚡ STEP THE SCHEDULER AFTER EACH EPOCH
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        
        # Print every epoch to watch the LR drop
        print(f"✅ Epoch {epoch:03d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
    save_path = os.path.join(WEIGHTS_DIR, 'sasrec_final_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"💾 Final Optimized Weights saved to {save_path}")

if __name__ == '__main__':
    train_final()