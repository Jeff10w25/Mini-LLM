"""
train.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import math
import os
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import config

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor, nn
    from torch.utils.data.distributed import DistributedSampler
    from datasets import DataLoader
    from config import TrainConfig

def r0print(*args, **kwargs):
    """Print only from rank 0 (1xGPU or CPU prints normally)"""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs) 
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
    """
    def __init__(
        self, 
        model: nn.Module, 
        cfg: TrainConfig, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        train_sampler: DistributedSampler | None
        ):
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_cuda = torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}" if self.is_cuda else "cpu"
        self.cfg = cfg 
        # handle compiled model (has _orig_mod.), if not compiled then get the model
        model = getattr(model, "_orig_mod", model).to(self.device)
        
        # only wrap in DDP if using more than 1 GPU
        if self.is_cuda and world_size > 1:
            self.model = DDP(model, device_ids=[self.gpu_id])
            self.scaler = torch.amp.GradScaler()
        # use 1 GPU
        elif self.is_cuda and world_size == 1:
            self.model = model
            self.scaler = torch.amp.GradScaler()
        # use CPU
        else:
            self.model = model
            self.scaler = None
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.max_iters)
        self.train_sampler = train_sampler
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
        total_loss, total_correct, total_tokens = 0.0, 0, 0
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
            total_correct += (logits.argmax(-1) == y).sum().item() 
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)  # perplexity loss
        return avg_loss, ppl
    
    def _raw_model(self) -> nn.Module:
        """Unwrap DDP (.module) and torch.compile (_orig_mod) for save/load checkpoint
        Handles all combinations, CPU, GPU, GPU+DDP, GPU+DDP+torch.compile
        """
        model = getattr(self.model, "module", self.model)
        model = getattr(model, "_orig_mod", model)
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
        
    def load_checkpoint(self, path: str) -> int:
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
        return ckpt_dict["step"]
        
    def train_step(self, batch: tuple[Tensor, Tensor]) -> float: 
        """Run one forward/backward/optimizer step on a batch, returning the loss.
        Uses fp16 autocast + GradScaler on GPU, plain fp32 on CPU
        """
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        
        if self.is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(X)
                # xentropy use all value across all batches to compute loss, must flatten first 2 dim to (B * seq_len, vocab_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)                      
            self.scaler.update()
        else:
            logits = self.model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            self.optimizer.step()               
        
        self.scheduler.step()
        return loss.item()

    def train(self, resume_path: str | None = None) -> dict:
        """Run the training loop for max_iters steps, eval and save on eval_every.

        Optionally resumes from a checkpoint. Restarts the data iterator on
        StopIteration (calling set_epoch when a sampler is present). Eval and
        checkpointing run only on rank 0. Returns the history dict. Eval and save
        is also timed using t1-t0.
        """
        if resume_path is not None:
            try:
                r0print("Loading from checkpoint...")
                start = self.load_checkpoint(resume_path) 
                r0print("Checkpoint loaded")
            except FileNotFoundError:
                r0print(f"Checkpoint path doesn't exist. Got {resume_path}")      
        else: 
            start = 0
            
        self.data_iter = iter(self.train_loader) # make iterable 
        self.model.train()
        print(f"Start Training on {self.device}...\n") # print on all rank available
        
        t0 = time.time() # t0
        for step in range(start + 1, self.cfg.max_iters + 1):
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                if self.train_sampler is not None:
                    # tells what shuffle sampler will use in case of training many epochs, set epoch only on restart
                    self.train_sampler.set_epoch(step // len(self.train_sampler))  
                batch = next(self.data_iter)  
            loss = self.train_step(batch) 
            self.history["step"].append(step)
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            self.history["train_loss"].append(loss)
            
            if self.gpu_id == 0 and step % self.cfg.eval_every == 0:
                val_loss, val_ppl= self.evaluate(self.val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_ppl"].append(val_ppl)
                self.save_checkpoint(step)
                t1 = time.time() # t1
                print(f"Step {step}/{self.cfg.max_iters}: "
                    f"train_loss {loss:.4f} | val_loss {val_loss:.4f} | "
                    f"ppl {val_ppl:.2f} | Time {t1-t0:.2f}s")
                self.model.train() # set model back to train mode after eval
                t0 = time.time() # reassign t0
        return self.history
                
if __name__ == '__main__':
    import data, config, model
    print(f"threads: {torch.get_num_threads()}, cores: {os.cpu_count()}")
    pipeline = data.DataPipeline(config.DataConfig())
    train_loader, valid_loader, train_sampler = pipeline.make_pipeline()
    
    model = model.MiniGPT(config.ModelConfig())
    if torch.cuda.is_available():
        model = torch.compile(model)    
    trainer = Trainer(model, config.TrainConfig(), train_loader, valid_loader, train_sampler)
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
