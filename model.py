"""
model.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from pos_encoding import FrequencyPE, RoPE

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor
    from config import ModelConfig

class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with RoPE positional encoding.

    Fused QKV projection, then split into heads, apply RoPE rotation, and
    compute scaled dot product attention with a causal mask.

    Args:
        pos_enc: Frequency positional encoding [seq_len, d_head]
        seq_len: Max sequence length
        embed_dim: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        weights_out: If True, return (output, attention_weights) else return output
    """
    def __init__(
        self, 
        pos_enc: Tensor, 
        seq_len: int, 
        embed_dim: int, 
        n_heads: int, 
        dropout: float = 0.1, 
        weights_out=False
        ):
        
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("Embedded dimensions must be divisible by number of heads")
        self.weights_out = weights_out
        self.rope = RoPE(pos_enc)
        self.h = n_heads
        self.d = embed_dim // n_heads
        self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool())
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3) # Fused qkv projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X: Tensor) -> Tensor:
        """Reshape into separate n amount of heads
        [batch, seq_len, d_model] -> [batch, seq_len, h, d]
        [batch, seq_len, h, d] -> [batch, h, seq_len, d]"""
        X_viewed = X.view(*X.shape[:-1], self.h, self.d)
        X_transpose = X_viewed.transpose(-3, -2)
        return X_transpose
    
    def merge_heads(self, X: Tensor) -> Tensor:
        """Merge all heads back to original
        [batch, h, seq_len, d] -> [batch, seq_len, h, d]
        [batch, seq_len, h, d] -> [batch, seq_len, d_model]"""
        X_transposed = X.transpose(-3, -2)
        X_reshaped = X_transposed.reshape(*X_transposed.shape[:-2], self.h * self.d)
        return X_reshaped

    def forward(
        self, 
        X: Tensor, 
        padding_mask: Tensor | None = None, 
        causal_attn: bool = True
        ) -> Tensor | tuple[Tensor, Tensor]:
        
        qkv = self.qkv_proj(X)                   # [B, seq_len, d_model * 3]
        query, key, value = qkv.chunk(3, dim=-1) # [B, seq_len, d_model] each
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)
        # apply RoPE rotation
        q_rot = self.rope.rotate(q)
        k_rot = self.rope.rotate(k)
        scores = q_rot @ k_rot.transpose(-2, -1)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -torch.inf)
        if causal_attn:
            T = X.shape[-2]
            scores = scores.masked_fill(~self.causal_mask[:T, :T], -torch.inf)
        
        weights = torch.softmax(scores / (self.d ** 0.5), dim=-1)
        Z = self.dropout(weights) @ v
        output = self.out_proj(self.merge_heads(Z))
        if self.weights_out: # if need weights, otherwise just returns output
            return (output, weights)
        return output
    
class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward, SiLU-gated two-path MLP with a fused gate+up projection
    
    Applies  down(SiLU(x @ W_gate) * (x @ W_up))  using a single fused
    gate+up projection and one down projection

    Args:
        embed_dim: Hidden dimension
        ff_dims: Expansion dimension inside FFN
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, ff_dims: int, dropout: float = 0.1):
        super().__init__()
        self.gate_up = nn.Linear(embed_dim, ff_dims * 2, bias=False) # Fused W_gate and W_up
        self.down = nn.Linear(ff_dims, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        gate_up = self.gate_up(X)
        gate, up = gate_up.chunk(2, dim=-1)  # [B, seq_len, ff_dims] each
        gate = F.silu(gate)
        return self.dropout(self.down(gate * up)) # [B, seq_len, embed_dim]

class TransformerBlock(nn.Module):
    """Pre-norm decoder block, residual self-attention + SwiGLU FFN
    
    Applies the pre-RMSNorm residual pattern:
        X = X + Attn(RMSNorm1(X))
        X = X + FFN(RMSNorm2(X))
        
    Args:
        pos_enc: Frequency positional encoding [seq_len, d_head]
        seq_len: Max sequence length 
        embed_dim: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self, 
        pos_enc: Tensor, 
        seq_len: int, 
        embed_dim: int, 
        n_heads: int, 
        dropout: float = 0.1
        ):
        
        super().__init__()
        # Optimal SwiGLU expansion ratio is around 2.67. 
        # Compute in integer math only to avoid floating point precision.
        # And also make sure to expand to value that is divisible by 64 for hardware optimization
        ff_dim = (8 * embed_dim) // 3 // 64 * 64  
        self.attn = MultiHeadAttention(pos_enc, seq_len, embed_dim, n_heads, dropout)
        self.ffn = SwiGLUFFN(embed_dim, ff_dim, dropout)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        
    def forward(self, X: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        X = X + self.attn(self.norm1(X), padding_mask)   # only select attn without weights
        X = X + self.ffn(self.norm2(X))
        return X
        
        
class MiniGPT(nn.Module):
    """Causal decoder-transformer block

    Embeds token ids, passes through n_layers transformer blocks, applies
    a final RMSNorm, and projects to vocab logits with a tied weight head

    Args:
        cfg (ModelConfig): Model hyperparameters.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        pos = FrequencyPE(cfg.embed_dim // cfg.n_heads)
        pos_enc = pos(cfg.seq_len)
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim) 
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                n_heads=cfg.n_heads,
                seq_len=cfg.seq_len,
                pos_enc=pos_enc,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.embed_dim)
        self.output = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)  # bias=False to share weights
        self.output.weight = self.embed.weight  # tying the lm head weights to the embedding weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Init weight to std=0.02 to prevent the loss from starting too high"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)
            
    def forward(self, input_ids: Tensor) -> Tensor:
        text_embeds = self.embed(input_ids)
        for layer in self.layers:
            text_embeds = layer(text_embeds)
        return self.output(self.norm(text_embeds))
    
    
if __name__ == "__main__":
    cfg = config.ModelConfig()
    model = MiniGPT(cfg)
    tied = id(model.embed.weight) == id(model.output.weight)
    print("Tied Embed and LM head weights =", tied)   # → True if tied correctly, on the same memory address
    total_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Total Parameters: {total_params:,}") # should be around 50M parameters