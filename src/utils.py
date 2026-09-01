"""utils.py - shared helpers used across the training pipeline."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

def ddp_setup():
    """Initialize the process group only when actually running DDP (torchrun)."""
    os.environ.setdefault("USE_LIBUV", "0")     # Windows torchrun workaround
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return world_size

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)
        
def apply_overrides(sections: dict, overrides: list[str]):
    for ov in overrides:
        path, _, val = ov.partition("=") # "data.num_workers", "=", "8"
        section_name, _, field = path.partition(".") # "data", ".", "num_workers"
        cfg = sections[section_name]
        if not hasattr(cfg, field):
            raise ValueError(f"Unknown field {section_name}.{field}")
        setattr(cfg, field, type(getattr(cfg, field))(val))
        
def pretty_view(path):
    runs = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in runs:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print("---")        
    
def r0print(*args, **kwargs):
    """Print only from rank 0 (single-GPU/CPU prints normally)."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)

def save_json(path: str, data: dict, indent: int = 2):
    """Atomically write a JSON file (safe against partial writes)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, path)

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def seed_everything(seed: int):
    """Set the RNG seed for torch/numpy/python reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA/cuDNN backend flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False