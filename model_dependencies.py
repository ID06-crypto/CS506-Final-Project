import torch.nn as nn
import torch
from torch.utils.data import Dataset

class POS_Transformer (nn.Module):
    def __init__ (self, embed_dim, num_heads, hidden_dim1, hidden_dim2, num_pos_tags):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_pos_tags = num_pos_tags

        self.MHA = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        self.MLP = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(self.hidden_dim2, self.embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)
        self.unembed = nn.Linear(self.embed_dim, self.num_pos_tags)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x, attention_mask):
        attention_out = self.MHA(x, x, x, key_padding_mask=attention_mask)[0]
        attention_out = self.dropout(attention_out)
        x = self.layernorm1(attention_out + x)
        mlp_out = self.MLP(x)
        x = self.layernorm2(mlp_out + x)
        cls_output = x[:, 0, :] 
        return self.unembed(cls_output)
    
class POSDataset(Dataset):
    def __init__(self, data, pos_to_idx):
        self.data = data
        self.pos_to_idx = pos_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'embeddings': item['embedding'],
            'attention_mask': item['padding_mask'],
            'labels': torch.tensor(self.pos_to_idx[item['pos']], dtype=torch.long)
        }