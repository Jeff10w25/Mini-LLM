"""main.py - entry point for training.

Usage / launch modes:
    python main.py --model <preset>                        single CPU/GPU
    torchrun --nproc_per_node=2 main.py --model <preset>   DDP, multi-GPU

Flags:
    --model <preset>    Which preset to run. Choices: smoke-7M, mini-90M. Default: smoke-7M.
    --resume <path>     Resume training from a checkpoint (.pt file). Omit to start from step 0.
    --compile           Enable torch.compile (best on GPU + Linux, 
                        skip on Windows because not support out of the box).
    --override <>
"""
import os
import argparse

import torch
import torch.distributed as dist

import config
import data
import model
from train import Trainer
import utils 


def main():
    # setup DDP
    world_size = utils.ddp_setup()
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))

    # model preset (smoke-7M vs mini-91M)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["smoke-7M", "mini-90M"], default="smoke-7M")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override a config field, e.g. --override data.num_workers=8 "
            "train.max_iters=2000 model.seq_len=512",
    )

    # build config
    args = parser.parse_args()
    preset = config.PRESETS[args.model]
    model_cfg = config.ModelConfig(**preset["model"])
    data_cfg = config.DataConfig(**preset["data"])
    train_cfg = config.TrainConfig(**preset["train"])
    
    # seed for reproducibility
    utils.seed_everything(train_cfg.seed)
    
    # apply --override after building configs
    sections = {"model": model_cfg, "data": data_cfg, "train": train_cfg}
    utils.apply_overrides(sections, args.override)
    
    # ensure dirs exist (torch.save / np.memmap don't create parents)
    utils.ensure_dirs(
        train_cfg.ckpt_dir, # checkpoints/smoke/
        data_cfg.cache_dir, # data/hf_cache/
        os.path.dirname(data_cfg.corpus_bin_path) # create /data
    )

    # data pipeline, handles DistributedSampler internally
    if world_size > 1:
        data_cfg.batch_size //= world_size  # split batch across ranks
    pipeline = data.DataPipeline(data_cfg)
    train_loader, valid_loader, train_sampler, valid_sampler = pipeline.make_pipeline()

    # model init
    train_model = model.MiniGPT(model_cfg)   

    # trainer, wraps in DDP, torch.compile, moves to device internally
    trainer = Trainer(train_model, train_cfg, train_loader, valid_loader, train_sampler, valid_sampler)
    if args.compile and torch.cuda.is_available():
        utils.r0print("Compiling with torch.compile...")
        trainer.model = torch.compile(trainer.model) 

    # only print on 1 process
    utils.r0print(f"Rank = {gpu_id}, Model = {args.model}, "
            f"Model params = {sum(p.numel() for p in set(train_model.parameters())):,}")

    # train loop. If DDP, ensures all ranks reaches the end process
    try:
        history = trainer.train(args.resume)
    except KeyboardInterrupt:
        utils.r0print("User Interrupted, cleaning up...")
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()