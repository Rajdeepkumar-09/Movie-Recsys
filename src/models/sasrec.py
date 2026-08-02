import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATA LOADER & NEGATIVE SAMPLING
# ==========================================
class SASRecDataset(Dataset):
    def __init__(self, processed_dir, num_items):
        """
        Loads the chronological sequences and dynamically generates 
        positive and negative targets for next-item prediction.
        """
        super().__init__()
        # Load the 2D array [User, Item_1, Item_2, ..., Item_50]
        self.data = np.load(os.path.join(processed_dir, 'train_seqs.npy'))
        self.num_items = num_items # 22884 unique movies
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        user = row[0]
        
        # The true sequence of movies watched
        tokens = row[1:] 
        
        # To predict the sequence, we shift it. 
        # Input (seq): [0, Item_1, Item_2, Item_3]
        # Target (pos): [Item_1, Item_2, Item_3, Item_4]
        seq = np.zeros_like(tokens)
        seq[1:] = tokens[:-1]
        pos = tokens
        
        # Generate negative samples (random movies they haven't watched right now)
        neg = np.zeros_like(tokens)
        for i in range(len(tokens)):
            if tokens[i] != 0:
                # Randomly sample an item ID as a negative example
                random_neg = np.random.randint(1, self.num_items + 1)
                # Note: In strict production, we ensure random_neg isn't in user's history.
                # For speed in training millions of rows, random uniform is standard practice.
                neg[i] = random_neg
                
        return torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(neg)

# ==========================================
# 2. TRANSFORMER BLOCKS
# ==========================================
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_dim)
        outputs = inputs.transpose(-1, -2) # Conv1d expects (batch, channels, length)
        outputs = self.dropout1(self.relu(self.conv1(outputs)))
        outputs = self.dropout2(self.conv2(outputs))
        return outputs.transpose(-1, -2)

class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.ffn = PointWiseFeedForward(hidden_dim, dropout_rate)
        
        self.layernorm1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.layernorm2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, attention_mask):
        # 1. Multi-Head Self Attention
        attn_out, _ = self.attention(
            query=x, key=x, value=x, 
            attn_mask=attention_mask,
            need_weights=False
        )
        # Residual connection + LayerNorm
        x = self.layernorm1(x + self.dropout(attn_out))
        
        # 2. Point-wise Feed Forward
        ffn_out = self.ffn(x)
        # Residual connection + LayerNorm
        x = self.layernorm2(x + self.dropout(ffn_out))
        return x

# ==========================================
# 3. MAIN SASREC ARCHITECTURE
# ==========================================
class SASRec(nn.Module):
    def __init__(self, num_items, max_seq_len, hidden_dim, num_heads, num_blocks, dropout_rate, device, pretrained_item_emb=None):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.device = device
        
        # Embeddings: 0 is the padding token, so we need num_items + 1 slots
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout_rate) for _ in range(num_blocks)
        ])
        
        self.apply(self._init_weights)
        
        # INJECTION: Apply pre-trained Word2Vec weights AFTER standard initialization
        if pretrained_item_emb is not None:
            print("🚀 INJECTION: Loading pre-trained Word2Vec embeddings into SASRec!")
            self.item_emb.weight.data.copy_(torch.from_numpy(pretrained_item_emb))
            # Note: We leave requires_grad=True so SASRec can "fine-tune" the Word2Vec weights during training
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            # Xavier uniform initialization for stable embedding gradients
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)

    def log2feats(self, log_seqs):
        """
        Converts the sequence of item IDs into contextualized 
        hidden representations using Self-Attention.
        """
        seq_len = log_seqs.shape[1]
        
        # 1. Look up item embeddings
        seq_emb = self.item_emb(log_seqs)
        seq_emb *= self.item_emb.embedding_dim ** 0.5 # Scale embeddings
        
        # 2. Add Positional Embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand_as(log_seqs)
        pos_emb = self.pos_emb(positions)
        
        x = self.emb_dropout(seq_emb + pos_emb)
        
        # 3. Create Causal Mask (Ensures model only looks at the past, not the future)
        # 0 = allowed to attend, -inf = masked/ignored
        mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device), diagonal=1).bool()
        
        # 4. Pass through Transformer Blocks
        for block in self.blocks:
            x = block(x, attention_mask=mask)
            
        return x

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """
        Training Forward Pass: Predicts scores for positive items (what they actually watched)
        and negative items (what they didn't watch).
        """
        # Get contextualized history representations
        log_feats = self.log2feats(log_seqs) # (batch, seq_len, hidden_dim)
        
        # Get embeddings for the targets
        pos_embs = self.item_emb(pos_seqs) # (batch, seq_len, hidden_dim)
        neg_embs = self.item_emb(neg_seqs) # (batch, seq_len, hidden_dim)
        
        # Calculate dot-product scores
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits

    def predict(self, log_seqs, item_indices):
        """
        Inference Pass: Given a user's sequence, score a list of candidate items.
        (Used for validation and FAISS embedding extraction).
        """
        # Only take the representation of the final timestamp to predict the next item
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :] # (batch, hidden_dim)
        
        item_embs = self.item_emb(item_indices) # (num_items, hidden_dim)
        
        # Matrix multiplication to get scores for all candidates
        logits = final_feat.matmul(item_embs.transpose(-1, -2))
        return logits