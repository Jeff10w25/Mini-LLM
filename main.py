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


def ddp_setup():
    """Initialize the process group ONLY when actually running DDP (torchrun)."""
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
    args = parser.parse_args()
    preset = config.PRESETS[args.model]
    model_cfg = config.ModelConfig(**preset["model"])
    data_cfg = config.DataConfig(**preset["data"])
    train_cfg = config.TrainConfig(**preset["train"])

    # data pipeline, handles DistributedSampler internally
    if world_size > 1:
        data_cfg.batch_size //= world_size   # split batch across ranks
    pipeline = data.DataPipeline(data_cfg)
    train_loader, valid_loader, train_sampler = pipeline.make_pipeline()

    # model init
    train_model = model.MiniGPT(model_cfg)

    # trainer, wraps in DDP + moves to device internally
    trainer = Trainer(train_model, train_cfg, train_loader, valid_loader, train_sampler)

    # only print on 1 process
    if gpu_id == 0:
        print(f"Rank = {gpu_id}, Model = {args.model}, "
            f"Model params = {sum(p.numel() for p in set(train_model.parameters())):,}")

    # train loop
    trainer.train(args.resume)

    # if DDP, makes all ranks wait until every rank reaches this line then end the process
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()