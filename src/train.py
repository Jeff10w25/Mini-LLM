"""
train.py - Trainer: AMP, gradient accumulation, DDP, resume, eval, checkpointing.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from utils import r0print, save_json

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor, nn
    from torch.utils.data.distributed import DistributedSampler
    from datasets import DataLoader
    from config import TrainConfig
    
    
class Trainer:
    """Trainer with AMP, resume, eval, and optional DDP

    Moves the model to the device, wraps it in DDP when world_size > 1, and
    manages the optimizer, LR scheduler, GradScaler, and checkpointing
    Handles CPU / single-GPU / multi-GPU DDP automatically

    Args:
        model: The model to train
        cfg: Training hyperparameters (device, lr, max_iters, eval_every, ckpt_dir)
        train_loader: Training batches (already sharded for DDP)
        val_loader: Validation batches (shuffle=False)
        train_sampler: Train sampler for set_epoch on restart (None when not using DDP)
        valid_sampler: Valid sampler (None when not using DDP)
    """
    def __init__(
        self, 
        model: nn.Module, 
        cfg: TrainConfig, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        train_sampler: DistributedSampler | None,
        valid_sampler: DistributedSampler | None
        ):
        
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_cuda = torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if self.is_cuda else "cpu"
        self.cfg = cfg 
        # handle compiled model (has _orig_mod.), if not compiled then get the model
        model = getattr(model, "_orig_mod", model).to(self.device)
        
        # only wrap in DDP if using more than 1 GPU
        if self.is_cuda and self.world_size > 1:
            self.model = DDP(model, device_ids=[self.gpu_id])
            self.scaler = torch.amp.GradScaler()
        # use 1 GPU
        elif self.is_cuda and self.world_size == 1:
            self.model = model
            self.scaler = torch.amp.GradScaler()
        # use CPU
        else:
            self.model = model
            self.scaler = None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, fused=self.is_cuda)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.max_iters)
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {
            "train_loss": [], 
            "val_loss": [],
            "val_ppl": [],
            "step": [],
            "lr": [], 
        }
        
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (avg_loss, perplexity) over the whole loader.

        Uses reduction="sum" + manual division so the average is exact
        even if the last batch is ragged.
        """
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            # xentropy use value [B * seq_len, vocab_size], target [B * seq_len]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()
        if self.world_size > 1:
            t = torch.tensor([total_loss, total_tokens], device=self.device)  # this rank's sum
            dist.all_reduce(t, op=dist.ReduceOp.SUM)  # t = sum across ALL ranks 
            total_loss, total_tokens = t[0].item(), int(t[1].item())
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)  # perplexity loss

        return avg_loss, ppl
    
    def _raw_model(self) -> nn.Module:
        """Unwrap DDP (.module) and torch.compile (_orig_mod) for save/load checkpoint
        Handles all combinations, CPU, GPU, GPU+DDP, GPU+DDP+torch.compile
        """
        model = self.model
        model = getattr(model, "_orig_mod", model)   # strip torch.compile (outer layer)
        model = getattr(model, "module", model)      # strip DDP (inner layer), if present
        model = model.to(self.device)
        return model
    
    def save_checkpoint(self, step: int):
        """Save model, optimizer, scheduler, scaler, step, config, and history.
        Writes the unwrapped model's state_dict so checkpoints are
        portable across DDP/compile. Creates the checkpoint dir if missing.
        """
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        torch.save({
            "model": self._raw_model().state_dict(), # save 
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "step": step,
            "cfg": self.cfg,
            "history": self.history
        }, f"{self.cfg.ckpt_dir}/step_{step}.pt")
        
    def load_checkpoint_path(self, path: str) -> int:
        """Load a checkpoint (weights, optimizer, scheduler, scaler, history).
        Returns the step it was saved at, so training resumes from there.
        """
        ckpt_dict = torch.load(path, weights_only=False, map_location=self.device)
        self._raw_model().load_state_dict(ckpt_dict["model"], strict=False)
        self.optimizer.load_state_dict(ckpt_dict["optimizer"])
        self.scheduler.load_state_dict(ckpt_dict["scheduler"])
        if ckpt_dict.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt_dict["scaler"])
        self.history = ckpt_dict["history"]
        
        step = ckpt_dict["step"]
        lr = self.scheduler.get_last_lr()[0]          # LR after the loaded scheduler state
        r0print(f"Resume training from step {step} | lr {lr:.6f} | "
            f"train_loss {self.history['train_loss'][-1]:.4f}")
        return ckpt_dict["step"]
    
    def load_valid_checkpoint(self, resume_path: str) -> int:
        """Verify that checkpoint path is valid, other wise raise error and exit.
        Returns the starting step"""
        if resume_path is not None:
            try:
                r0print("Loading from checkpoint...")
                start = self.load_checkpoint_path(resume_path)
                return start
            except Exception as e:
                r0print(f"Failed to load checkpoint {resume_path}: {e}, " 
                        f"provide a valid checkpoint path or omit --resume to start from scratch.")
                r0print("Exiting...")
                raise SystemExit(1)   
        else: 
            start = 0
        return start
        
    def train_step(self, batch: tuple[Tensor, Tensor], accum: int) -> float:
        """Run one forward/backward/optimizer step on a batch, returning the loss.
        Uses fp16 autocast + GradScaler on GPU, plain fp32 on CPU
        """
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        
        if self.is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(X)
                # xentropy use all value across all batches to compute loss, must flatten first 2 dim to (B * seq_len, vocab_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            self.scaler.scale(loss / accum).backward()
        else:
            logits = self.model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / accum).backward()
        return loss.item()

    def train(self, resume_path: str | None = None) -> dict:
        """Run the training loop for max_iters steps, eval and save on eval_every.

        Gradient accumulation happens inside each step. Effective batch = batch_size * grad_accum
        Optionally resumes from a checkpoint. Restarts the data iterator on
        StopIteration (calling set_epoch when a sampler is present). Eval and
        checkpointing run only on rank 0. Returns the history dict. Total, training, and eval time
        is timed separately to monitor the expected tokens/sec.
        The training history is saved in JSON format at checkpoint directory.
        """
        start = self.load_valid_checkpoint(resume_path) # starting step
        self.data_iter = iter(self.train_loader) # make iterable 
        self.model.train()
        print(f"Start Training on {self.device}...\n") # print on all rank available
        
        # per-interval time (reset at each save)
        train_s = 0.0
        eval_s = 0.0
        t_interval = time.perf_counter()
        
        # train loop
        for step in range(start + 1, self.cfg.max_iters + 1):
            t_train = time.perf_counter()
            step_loss = 0.0
            self.optimizer.zero_grad() # once per effective batch
            # gradient accumulation, combine batches to make effective batch
            for i in range(1, self.cfg.grad_accum + 1):  
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.train_loader)
                    # tells what shuffle sampler will use in case of training many epochs, set epoch only on restart
                    if self.train_sampler is not None:
                        self.train_sampler.set_epoch(step // len(self.train_sampler))  
                    batch = next(self.data_iter)  
                # skip all_reduce on early micro-batches, all reduce once on the last micro batches in grad accum
                if self.world_size > 1 and i % self.cfg.grad_accum != 0:
                    with self.model.no_sync():
                        step_loss += self.train_step(batch, self.cfg.grad_accum)
                else:
                    step_loss += self.train_step(batch, self.cfg.grad_accum)
                    
            if self.is_cuda:
                self.scaler.step(self.optimizer)             # once per effective batch
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.scheduler.step()
            train_s += time.perf_counter() - t_train
            # compute loss without accum and append lr, step, loss to history
            loss = step_loss / self.cfg.grad_accum
            self.history["step"].append(step)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            self.history["train_loss"].append(loss)
            
            # eval and save to checkpoint path
            if step % self.cfg.eval_every == 0:
                # eval loop
                t_eval = time.perf_counter()
                val_loss, val_ppl= self.evaluate(self.val_loader)
                eval_s += time.perf_counter() - t_eval
                
                self.history["val_loss"].append(val_loss)
                self.history["val_ppl"].append(val_ppl)
                if self.gpu_id == 0:
                    self.save_checkpoint(step)
                    
                self.model.train() # set model back to train mode after eval
                
                total_s = time.perf_counter() - t_interval
                r0print(f"Step {step}/{self.cfg.max_iters}:")
                r0print(f"train_loss {loss:.4f} | val_loss {val_loss:.4f} | ppl {val_ppl:.2f}")
                r0print(f"Time: total {total_s:.1f}s | train {train_s:.1f}s | eval {eval_s:.1f}s")
                # reset training and eval time to 0
                train_s = 0.0
                eval_s = 0.0
                t_interval = time.perf_counter()
                
        # optional save to json files once training finished
        save_json(os.path.join(self.cfg.ckpt_dir, "history.json"), self.history)
        return self.history
                
if __name__ == '__main__':
    import data as data, config as config, model as model
    print(f"threads: {torch.get_num_threads()}, cores: {os.cpu_count()}")
    pipeline = data.DataPipeline(config.DataConfig())
    train_loader, valid_loader, train_sampler, valid_sampler = pipeline.make_pipeline()
    
    model = model.MiniGPT(config.ModelConfig())
    if torch.cuda.is_available():
        model = torch.compile(model)    
    trainer = Trainer(model, config.TrainConfig(), train_loader, valid_loader, train_sampler, valid_sampler)
    resume = "checkpoints/step_10000.pt"
    history = trainer.train(resume)
    
    # x = torch.randint(0, 50257, (32, 128))
    # model.train()

    # t0 = time.time()
    # with torch.no_grad():
    #     logits = model(x)
    # print(f"forward: {time.time()-t0:.2f}s")

    # t0 = time.time()
    # with torch.no_grad():
    #     logits = model(x)   # second call — warm
    # print(f"forward (warm): {time.time()-t0:.2f}s")
