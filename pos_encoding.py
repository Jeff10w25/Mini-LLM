"""pos_encoding.py — various positional encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding based on the paper `Attention is all you need` where
    PE(p, 2i)   = sin(p/(10000^(2i/d))) , for even index in embeddings
    PE(p, 2i+1) = cos(p/(10000^(2i/d))) , for odd index in embeddings
    i = d/2 , dimension / 2
    
    Input/Output: [B, seq_len, embed_dim] ->  [seq_len, embed_dim]
    """
    def __init__(self, max_len: int, embed_dim: int, dropout: float = 0.1, n: int = 10_000):
        super().__init__()
        pos_encodings = torch.empty(max_len, embed_dim) # [max_len, embed_dim]
        pos = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        i = torch.arange(0, embed_dim, 2) # [embed_dim, ]
        denom_val = pos / (n ** (2 * i / embed_dim))
        pos_encodings[:, ::2] = denom_val.sin()  # ::2 means start=0:stop=end:step=2, even num
        pos_encodings[:, 1::2] = denom_val.cos() # 1::2 means start=1:stop=end:step=2, odd num
        self.register_buffer("pos_encodings", pos_encodings)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return only positional encoding upto seq_len of X
        return self.dropout(self.pos_encodings[:X.size(-2)])
    
class FrequencyPositionalEncoding(nn.Module):
    """Frequency positional encoding encodes position as inverse frequencies, 
    which will be used to compute and apply RoPE. The output is the angle of each positions.
    Inverse_freq = 10000^(-2(i-1)/dim), i = 1, 2, ... embed_dim // 2
    Pos_encoding = p * inverse_freq, p = 0, 1, ... seq_len - 1
    
    Input/Output: [seq_len, ] ->  [seq_len, embed_dim]
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        # compute θ_m = 1 / 10000^(2t/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len)  # [seq_len, ]
        # frequency positional encoding (outer product of pos and inv_freq)
        pos_encodings = torch.outer(pos, self.inv_freq) # [seq_len, embed_dim/2]
        # dupe each element along inv_freq dimensions 
        pos_encodings = pos_encodings.repeat_interleave(2, dim=-1) # [seq_len, embed_dim]
        return pos_encodings
    
class RoPE:
    """Rotary positional encoding 
    
        
    Input/Output: [B, seq_len, embed_dim] ->  [seq_len, embed_dim]
    """
    def __init__(self, freq_pos_enc: torch.Tensor):
        super().__init__()
        # θ_m = 1 / 10000^(2m/d)
        self.freq_pos_enc = freq_pos_enc
        self.rotate_dim = freq_pos_enc.shape[-1]
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        
        return
        
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        
        return
    

if __name__ == '__main__':
    # testing sinusoidal positional encoding
    torch.manual_seed(42)
    torch.set_printoptions(precision=4)
    max_len = 5
    embed_dim = 6
    X = torch.randn(max_len, embed_dim)
    sin_enc = SinusoidalPE(max_len, embed_dim)
    print(sin_enc(X))