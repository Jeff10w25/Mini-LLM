"""
model.py
"""
import torch

class MultiHeadAttention(nn.Module):
    """Causal Multi-headed causal self-attention with RoPE positional Encoding
    
    Input/Output idx [B, T] → logits [B, T, vocab_size].
    
    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self, ):
        self.n_heads = None
        
    def forward(self,):
        return
    
class MLP(nn.Module):
    """Feed-Forward layer with SeLU activation function

    Input/Output idx [B, T] → logits [B, T, vocab_size].

    Args:
        cfg (ModelConfig): Model hyperparameters (vocab_size, n_layers,
            n_heads, embed_dim, block_size, dropout)
    """
    def __init__(self,):
        self

    def forward(self,) -> torch.Tensor:
        return

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
    def __init__(self,):
        self
        
    def forward(self,):
        return