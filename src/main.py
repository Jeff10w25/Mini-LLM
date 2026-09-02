"""main.py - canonical training entrypoint. Configs are loaded from JSON.

Usage:
    python main.py                                   single CPU/GPU
    torchrun --nproc_per_node=2 main.py              DDP, multi-GPU

Config source of truth = configs/*.json (one set). For quick runs use flags:
    python main.py --steps 500                        smoke run (caps max_iters)
    python main.py --override data.num_workers=0      override one field
    python main.py --resume checkpoints/mini/step_8000.pt

Flags:
    --model-json / --data-json / --train-json   Paths to config JSONs (defaults under configs/).
    --resume <path>     Resume training from a checkpoint (.pt). Omit to start from step 0.
    --compile           Enable torch.compile (best on GPU + Linux; skip on Windows).
    --steps <n>         Smoke shortcut: run max_iters = n without editing the JSON.
    --override <k=v>    Override a config field, e.g. --override data.num_workers=8
"""
import argparse
import os

import torch
import torch.distributed as dist

import config
import data
import model
import utils
from train import Trainer


def load_configs(model_json: str, data_json: str, train_json: str):
    """Build the three config dataclasses from JSON (values win, defaults for missing keys)."""
    model_cfg = config.from_json(config.ModelConfig, model_json)
    data_cfg = config.from_json(config.DataConfig, data_json)
    train_cfg = config.from_json(config.TrainConfig, train_json)
    return model_cfg, data_cfg, train_cfg


def main():
    world_size = utils.ddp_setup()
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))

    parser = argparse.ArgumentParser(description="Train MiniGPT (configs loaded from JSON)")
    parser.add_argument("--model-json", default="configs/model.json")
    parser.add_argument("--data-json", default="configs/data.json")
    parser.add_argument("--train-json", default="configs/train.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--steps", type=int, default=None,
                        help="Smoke shortcut: cap max_iters (does not edit the JSON)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override a config field, e.g. --override data.num_workers=8 train.max_iters=2000",
    )
    args = parser.parse_args()

    model_cfg, data_cfg, train_cfg = load_configs(
        args.model_json, 
        args.data_json, 
        args.train_json
    )

    # seed for reproducibility
    utils.seed_everything(train_cfg.seed)

    # apply --override after loading, then the --steps smoke shortcut last
    sections = {"model": model_cfg, "data": data_cfg, "train": train_cfg}
    utils.apply_overrides(sections, args.override)
    if args.steps is not None:
        train_cfg.max_iters = args.steps

    # a training window must always fit inside the model's context length
    if data_cfg.seq_len > model_cfg.seq_len:
        raise ValueError(
            f"data seq_len ({data_cfg.seq_len}) exceeds model seq_len ({model_cfg.seq_len}); "
            "a training window would overrun the model context"
        )

    # ensure dirs exist (torch.save / np.memmap don't create parents)
    utils.ensure_dirs(
        train_cfg.ckpt_dir,           # e.g. checkpoints/mini/
        data_cfg.cache_dir,           # data/hf_cache/
        os.path.dirname(data_cfg.corpus_bin_path),  # data/
    )

    # data pipeline (handles DistributedSampler internally)
    if world_size > 1:
        data_cfg.batch_size //= world_size  # split batch across ranks
    pipeline = data.DataPipeline(data_cfg)
    train_loader, valid_loader, train_sampler = pipeline.make_pipeline()

    # model init
    train_model = model.MiniGPT(model_cfg)

    # trainer (moves to device; wraps DDP when world_size > 1)
    trainer = Trainer(train_model, train_cfg)
    if args.compile and torch.cuda.is_available():
        utils.r0print("Compiling with torch.compile...")
        trainer.model = torch.compile(trainer.model)

    utils.r0print(
        f"Rank = {gpu_id}, Model params = {sum(p.numel() for p in set(train_model.parameters())):,}, "
        f"max_iters = {train_cfg.max_iters}"
    )

    # train loop; DDP ensures all ranks reach the end together
    try:
        history = trainer.train(
            train_loader, valid_loader, train_sampler, resume_path=args.resume
        )
    except KeyboardInterrupt:
        utils.r0print("User Interrupted, cleaning up...")
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
