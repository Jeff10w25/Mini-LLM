"""pos_encoding.py — various positional encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    """Sinusoidal Encoding based on the paper `Attention is all you need` where
    PE(p, 2i)   = sin(p/(10000^(2i/d))) -> even num
    PE(p, 2i+1) = cos(p/(10000^(2i/d))) -> odd num
    i = d/2
    
    Input/Output: [B, seq_len, embed_dim] ->  
        
    Args:
        cfg (ModelConfig): Model hyperparameters (seq_len, embed_dim, _dropout)
    """
    def __init__(self, max_len: int, embed_dim: int, dropout: float = 0.1, n: int = 10_000):
        super().__init__()
        pos_encodings = torch.empty(max_len, embed_dim)
        p = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(0, embed_dim, 2)
        denom_val = p / (n ** (i / embed_dim))
        pos_encodings[:, ::2] = denom_val.sin()  # ::2 means start=0:stop=end:step=2, even num
        pos_encodings[:, 1::2] = denom_val.cos() # 1::2 means start=1:stop=end:step=2, odd num
        self.register_buffer("pos_encodings", pos_encodings)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: torch.Tensor):
        # Return only positional encoding upto seq_len of X
        return self.dropout(self.pos_encodings[:X.size(-2)])
    
    
class RotaryPE()

if __name__ == '__main__':
    # testing sinusoidal positional encoding
    torch.manual_seed(42)
    torch.set_printoptions(precision=4)
    max_len = 5
    embed_dim = 6
    X = torch.randn(max_len, embed_dim)
    sin_enc = SinusoidalPE(max_len, embed_dim)
    print(sin_enc(X))