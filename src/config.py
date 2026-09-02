"""
config.py - dataclass configs for model/train/data/generator.

Values come from configs/*.json at runtime (single source of truth); the
field defaults here are only the fallback for keys a JSON does not set.
See config.from_json / config.to_json below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar, Type

@dataclass
class ModelConfig:
    device: str     = "cuda"
    vocab_size: int = 50257
    embed_dim: int  = 640 # 640
    n_heads: int    = 10 # 10
    n_layers: int   = 12 # 12
    seq_len: int    = 1024 # 1024
    dropout: float  = 0.1

@dataclass
class TrainConfig:
    seed: int       = 42
    lr: float       = 3e-4
    grad_accum: int = 8
    max_iters: int  = 15000
    eval_every: int = 500
    ckpt_dir: str   = "checkpoints/mini/"

@dataclass
class DataConfig:
    data_source: str        = "HuggingFaceFW/fineweb-edu"
    data_source_name: str   = "sample-10BT"
    cache_dir: str          = "data/hf_cache/"
    corpus_bin_path: str    = "data/full_corpus.bin"
    train_bin_path: str     = "data/train_corpus.bin"
    valid_bin_path: str     = "data/valid_corpus.bin"
    total_token: int        = 2_000_000_000
    chunk_size: int         = 10_000_000  # tokenization chunking so it can fit in RAM
    val_monitor_token: int  = 500_000  # token used for eval during training
    train_split: float      = 0.95
    valid_split: float      = 0.05  # 5% of total tokens for validation
    batch_size: int         = 16 # 16
    seq_len: int            = 1024 # 512
    stride: int             = 1024 # 512
    num_workers: int        = 4  # CPU=0, GPU=4
    pin_memory: bool        = True  # CPU=False, GPU=True
    persistent_workers: bool= True  # CPU=False, GPU=True

@dataclass
class GeneratorConfig:
    seed: int           = 42
    device: str         = "cuda"
    ckpt_path: str      = "checkpoints/mini/step_6500.pt"
    output_dir: str     = "samples/"
    temperature: float  = 1.0
    top_k: int | None   = None
    top_p: float | None = None
    max_tokens: int     = 2000
    banned_tokens: int  = 50256 # endoftext token
    keep: int | None    = None  # sliding-window length; None -> full seq_len
    anneal: str | None  = None  # "temp" | "top_k" | "top_p" | None
    caching: bool       = True  # use the KV cache
    to_json: bool       = False  # append the run to a generations_*.jsonl

T = TypeVar("T")

def from_json(cls: Type[T], path: str | Path) -> T:
    """Load a config dataclass from JSON; missing keys fall back to defaults."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})

def to_json(cfg, path: str | Path) -> None:
    """Save a config dataclass to JSON (so a run records exactly what it used)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
