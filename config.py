"""
config.py provides 
"""
from __future__ import annotations   
from dataclasses import dataclass 

@dataclass
class ModelConfig:
    vocab_size: int = 50256
    embed_dim: int  = 512
    n_heads: int    = 8
    n_layers: int   = 8
    seq_len: int    = 512
    dropout: float  = 0.1


@dataclass
class TrainConfig:
    seed: int       = 42
    lr: float       = 3e-4
    batch_size: int = 32
    max_iters: int  = 100
    ckpt_dir: str   = "/drive/etc"  # for checkpointing in Kaggle
    
@dataclass
class DataConfig:
    data_source: str        = "HuggingFaceFW/fineweb-edu"
    data_source_name: str   = "sample-10BT"
    cache_dir: str          = "data/hf_cache/"
    train_bin_path: str     = "data/train_corpus.bin"
    valid_bin_path: str     = "data/valid_corpus.bin"
    max_train_token: int    = 950_000_000   
    max_valid_token: int    = 50_000_000  # 5% of total tokens for validation
    chunk_size: int         = 10_000_000  # Tokenization chunking so it can fit in RAM
    seq_len: int            = 512
    stride: int             = 512