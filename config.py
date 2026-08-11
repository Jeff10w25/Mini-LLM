"""
config.py provides 
"""
from __future__ import annotations   
from dataclasses import dataclass 

@dataclass
class ModelConfig:
    vocab_size: int     = 50000
    embed_dim: int      = 512
    n_heads: int        = 6
    n_layers: int       = 12
    seq_len: int        = 512
    hd_dropout: float   = 0.1
    lm_dropout: float   = 0.0

@dataclass
class TrainConfig:
    seed: int       = 42
    lr: float       = 3e-4
    batch_size: int = 32
    max_iters: int  = 100
    ckpt_dir: str   = "/drive/etc"  # for checkpointing in Kaggle
    
@dataclass
class DataConfig:
    train_path: str     = "/data/"
    val_path: str       = "/data/"
    seq_len: str        = 512
    stride: int         = 1
    val_split: float    = 0.05
    test_split: float   = 0.05