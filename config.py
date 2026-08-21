"""
config.py provides 
"""
from __future__ import annotations   
from dataclasses import dataclass 

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    embed_dim: int  = 128 # 640
    n_heads: int    = 4 # 10
    n_layers: int   = 4 # 12
    seq_len: int    = 128 # 1024
    dropout: float  = 0.1
@dataclass
class TrainConfig:
    device: str     = "cuda"
    seed: int       = 42
    lr: float       = 3e-4
    grad_accum: int = 4
    max_iters: int  = 20000
    eval_every: int = 500
    ckpt_dir: str   = "checkpoints/"  # for checkpointing in Kaggle
@dataclass
class DataConfig:
    data_source: str        = "HuggingFaceFW/fineweb-edu"
    data_source_name: str   = "sample-10BT"
    cache_dir: str          = "data/hf_cache/"
    corpus_bin_path: str    = "data/full_corpus.bin"
    train_bin_path: str     = "data/train_corpus.bin"
    valid_bin_path: str     = "data/valid_corpus.bin"
    total_token: int        = 100_000_000
    chunk_size: int         = 1_000_000  # Tokenization chunking so it can fit in RAM
    val_monitor_token: int  = 100_000
    train_split: float      = 0.95
    valid_split: float      = 0.05  # 5% of total tokens for validation
    batch_size: int         = 16 # 32
    seq_len: int            = 128 # 512
    stride: int             = 128 # 512
    num_workers: int        = 4  # CPU=0, GPU=4
    pin_memory: bool        = True  # CPU=False, GPU=True
    persistent_workers: bool= True  # CPU=False, GPU=True
    
# Build 2 models on Kaggle. 1 for smoke test and 1 for actual test.
PRESETS = {
    "smoke-7M": {
        "model": dict(
            vocab_size=50257, 
            embed_dim=128, 
            n_heads=4, 
            n_layers=4, 
            seq_len=128
        ),
        "data": dict(
            data_source_name="sample-10BT",
            total_token=20_000_000,
            batch_size=16,
            seq_len=128,
            stride=128,
            val_monitor_token=100_000,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        "train": dict(
            lr=3e-4,
            max_iters=500,
            eval_every=50,
            ckpt_dir="checkpoints/smoke/",
        ),
    },
    "mini-90M": {
        "model": dict(
            vocab_size=50257, 
            embed_dim=640, 
            n_heads=10, 
            n_layers=12, 
            seq_len=1024
        ),
        "data": dict(
            data_source_name="sample-10BT",
            total_token=2_000_000_000,
            batch_size=48,
            seq_len=1024,
            stride=1024,
            val_monitor_token=1_000_000,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        ),
        "train": dict(
            lr=3e-4,
            max_iters=40_000,
            eval_every=500,
            ckpt_dir="checkpoints/mini/",
        ),
    },
}