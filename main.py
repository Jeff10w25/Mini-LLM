""" main.py — entry point for training, launch with
    `torchrun --nproc_per_node=2 main.py`   (DDP, multi-GPU)
    `python main.py`                        (single CPU/GPU)
"""
import os
import argparse

import torch
import torch.distributed as dist

import config
import data
import model
from train import Trainer

def r0print(*args, **kwargs):
    """Print only from rank 0 (1xGPU or CPU prints normally)"""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs) 

def ddp_setup():
    """Initialize the process group only when actually running DDP (torchrun)."""
    os.environ.setdefault("USE_LIBUV", "0")     # Windows torchrun workaround
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return world_size

def main():
    world_size = ddp_setup()
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))

    # model preset (smoke-7M vs mini-91M)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["smoke-7M", "mini-91M"], default="smoke-7M")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()
    preset = config.PRESETS[args.model]
    model_cfg = config.ModelConfig(**preset["model"])
    data_cfg = config.DataConfig(**preset["data"])
    train_cfg = config.TrainConfig(**preset["train"])
    
    # ensure dirs exist (torch.save / np.memmap don't create parents)
    os.makedirs(train_cfg.ckpt_dir, exist_ok=True)  # checkpoints/smoke/
    os.makedirs(data_cfg.cache_dir, exist_ok=True)  # data/hf_cache/
    os.makedirs(os.path.dirname(data_cfg.corpus_bin_path), exist_ok=True)  # data/
    os.makedirs(os.path.dirname(data_cfg.train_bin_path), exist_ok=True)        
    os.makedirs(os.path.dirname(data_cfg.valid_bin_path), exist_ok=True)

    # data pipeline, handles DistributedSampler internally
    if world_size > 1:
        data_cfg.batch_size //= world_size  # split batch across ranks
    pipeline = data.DataPipeline(data_cfg)
    train_loader, valid_loader, train_sampler = pipeline.make_pipeline()

    # model init
    train_model = model.MiniGPT(model_cfg)   

    # trainer, wraps in DDP, torch.compile, moves to device internally
    trainer = Trainer(train_model, train_cfg, train_loader, valid_loader, train_sampler)
    if args.compile and torch.cuda.is_available():
        r0print("Compiling with torch.compile...")
        trainer.model = torch.compile(trainer.model) 

    # only print on 1 process
    r0print(f"Rank = {gpu_id}, Model = {args.model}, "
            f"Model params = {sum(p.numel() for p in set(train_model.parameters())):,}")

    # train loop. If DDP, ensures all ranks reaches the end process
    try:
        trainer.train(args.resume)
    except KeyboardInterrupt:
        r0print("User Interrupted, cleaning up...")
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()