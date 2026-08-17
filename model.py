"""
model.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_encoding import FrequencyPE, RoPE

class MultiHeadAttention(nn.Module):
    """Causal Multi-headed causal self-attention with RoPE positional Encoding
    
    Input/Output idx [B, T] → logits [B, T, vocab_size].
    
    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self, pos_enc: FrequencyPE, seq_len: int, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("Embedded dimensions must be divisible by number of heads")
        self.rope = RoPE(pos_enc)
        self.h = n_heads
        self.d = embed_dim // n_heads
        self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len), diagonal=0))
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3) # Fused qkv projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        ''' reshape into separate n amount of heads
        [batch, seq_len, d_model] -> [batch, seq_len, h, d]
        [batch, seq_len, h, d] -> [batch, h, seq_len, d]'''
        X_viewed = X.view(*X.shape[:-1], self.h, self.d)
        X_transpose = X_viewed.transpose(-3, -2)
        return X_transpose
    
    def merge_heads(self, X: torch.Tensor) -> Tensor:
        '''Merge all heads back to original
        [batch, h, seq_len, d] -> [batch, seq_len, h, d]
        [batch, seq_len, h, d] -> [batch, seq_len, d_model]'''
        X_transposed = X.transpose(-3, -2)
        X_reshaped = X_transposed.reshape(*X_transposed.shape[:-2], self.h * self.d)
        return X_reshaped

    def forward(self, X: torch.Tensor, padding_mask= None, causal_attn=True) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(X)                   # [B, seq_len, d_model * 3]
        query, key, value = qkv.chunk(3, dim=-1) # [B, seq_len, d_model] each
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)
        # apply RoPE rotation
        q_rot = self.rope.rotate(q)
        k_rot = self.rope.rotate(q)
        scores = q_rot @ k_rot.transpose(-2, -1)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -torch.inf)
        if causal_attn:
            scores = scores.masked_fill(self.causal_mask == 0, -torch.inf)
        
        weights = torch.softmax(scores / (self.d ** 0.5), dim=-1)
        Z = self.dropout(weights) @ v
        output = self.out_proj(self.merge_heads(Z))
        return (output, weights)
    
class SwiGLUFFN(nn.Module):
    """Feed-Forward layer with SiLU activation function

    Input/Output idx [B, T] → logits [B, T, vocab_size].

    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self, embed_dim: int, ff_dims: int, dropout: float = 0.1):
        super().__init__()
        self.gate_up = nn.Linear(embed_dim, ff_dims * 2, bias=False) # Fused W_gate and W_up
        self.down = nn.Linear(ff_dims, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(X)
        gate, up = gate_up.chunk(2, dim=-1)  # [B, seq_len, ff_dims] each
        gate = F.silu(gate)
        return self.dropout(self.down(gate * up)) # [B, seq_len, embed_dim]

class Block(nn.Module):
    """Transformer decoder block: 
    
    Input/Output: idx [B, T] → logits [B, T, vocab_size].
    
    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self):
        self
        
class SmolGPT(nn.Module):
    """Transformer decoder block: 
        
    Input/Output: idx [B, T] → logits [B, T, vocab_size].
        
    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self, embed_dim, n_heads):
        pos = FrequencyPE(embed_dim)
        pos_enc = pos(embed_dim // n_heads)
        
    def forward(self,):
        return
    
if __name__=='main':
    pass