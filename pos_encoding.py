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
    
class FrequencyPE(nn.Module):
    """ Frequency positional encoding encodes position as inverse frequencies, 
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
        # halves repeated the angle 
        pos_encodings = torch.cat((pos_encodings, pos_encodings), dim=-1) # [seq_len, embed_dim]
        return pos_encodings
    
class RoPE(nn.Module):
    """Rotary positional encoding, LLaMa styled 
    Let x be query or key vector, split into two halves:
    u1 = dims [0, D/2],  u2 = dims [D/2, D]
    Each dim m of u1 pairs with dim m of u2, sharing angle θ_m: 
    
        [u1'] = [ u1·cos(θ1) - u2·sin(θ1) ]
        [u2']   [ u1·sin(θ1) + u2·cos(θ1) ]
    
    Vectorized form (rotate_half = swap each pair's elements, negate the first):
        x_rot = x · cos(θ)  +  rotate_half(x) · sin(θ)
        rotate_half(x) = [-u2, u1]  , e.g. [-e,-f,-g,-h, a, b, c, d] for D=8
        
    Input/Output: [B, H, S, D] ->  [B, H, S, D]
    """
    def __init__(self, pos_enc: torch.Tensor):
        super().__init__()
        # precompute cos, sin of pos_enc
        self.register_buffer("cos_cache", pos_enc.cos())
        self.register_buffer("sin_cache", pos_enc.sin())
        self.rotate_dim = pos_enc.shape[-1]
    
    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        # only use value up to seq_len of x
        cos, sin = self.cos_cache[:x.shape[-2]], self.sin_cache[:x.shape[-2]]
        rot_x = x * cos + self._rotate_half(x) * sin
        return rot_x
        
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        u1, u2 = torch.chunk(x, 2, dim=-1) # u1: [B, H, S, D//2], u2: [B, H, S, D//2]
        return torch.cat((-u2, u1), dim=-1) # -> [B, H, S, D]
    

if __name__ == '__main__':
    # testing sinusoidal positional encoding
    torch.manual_seed(42)
    torch.set_printoptions(precision=4)
    max_len = 5
    embed_dim = 6
    X = torch.randn(max_len, embed_dim)
    sin_enc = SinusoidalPE(max_len, embed_dim)
    print(sin_enc(X))
    # Sanity test of _rotate_half(x)
    x = torch.arange(8).float()
    print(RoPE._rotate_half(x))
    # Testing RoPE
    batch_size = 1
    num_heads = 1
    seq_len = 5
    head_dim = 6

    # random queries and keys
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # frequency-encode positions 1 - 512
    pos = FrequencyPE(head_dim)
    pos_enc = pos(512)
    rope = RoPE(pos_enc)
    # encode absolute positions into queries and keys
    q_rot = rope.rotate(q)
    k_rot = rope.rotate(k)

    # take inner product of queries and keys to obtain attention
    attn = q_rot @ k_rot.transpose(-2, -1)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
    print(f"Rotated: {q_rot}, \nNormal: {q}")
    """ Rotated: tensor([[[
        [ 0.1877, -0.3576, -0.3165,  0.5886, -0.8905,  0.4098],
        [-1.3067, -0.0944,  0.3494, -0.8925, -0.1739,  0.2340],
        [-1.6243,  1.2672, -0.3644,  3.6948,  0.1915,  0.8171],
        [-1.4492,  0.3114, -1.4238,  0.3241,  0.2633, -0.0559],
        [ 1.1357, -0.2911, -0.4298,  0.9133, -1.5190,  0.5177]]]]), 
        Normal: tensor([[[
        [ 0.1877, -0.3576, -0.3165,  0.5886, -0.8905,  0.4098],
        [-1.4570, -0.1023,  0.3499,  0.6173, -0.1693,  0.2332], 
        [ 4.0356,  1.2795, -0.3609, -0.0606,  0.0733,  0.8187],
        [ 1.4805,  0.3449, -1.4241, -0.1163,  0.2176, -0.0467],
        [-1.4335, -0.5665, -0.4253,  0.2625, -1.4391,  0.5214]]]])"""