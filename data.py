"""data.py - token arrays, TokenDataset, and dataset construction."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch
import os
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from dotenv import load_dotenv

import config
from utils import r0print

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor
    from datasets import IterableDataset
    from config import DataConfig

if not os.environ.get("HF_TOKEN"):
    load_dotenv()
class TokenDataset(Dataset):
    """TokenDataset for DataLoader
    
    Takes fully tokenized corpus using np.memmap to prevent writing full corpus to memory. 
    Then returns target, shifted right by 1 token. This is used to do next token prediction.
        
    Args:
        path: Fully tokenized corpus path. E.g. /data/corpus.bin
        window_len: The context size that will be fed to the model
        stride: How far apart __getitem__ will sample the next idx. Defaults to seq_len.
                Stride < seq_len is only use to grow the corpus if there isn't enough data.
                Or if the data download from HuggingFace is a bottleneck.
    """
    def __init__(
        self, 
        path: str, 
        seq_len: int, 
        stride: int | None = None, 
        max_tokens: int | None = None
        ):
        
        self.token = np.memmap(path, dtype=np.uint16, mode="r+") 
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.length = len(self.token) if max_tokens is None else max_tokens
        
    def __len__(self) -> int:
        return (self.length - self.seq_len) // self.stride

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx >= len(self):
            raise ValueError("Index is out of range.")
        start = idx * self.stride
        end = start + self.seq_len
        window = torch.from_numpy(self.token[start:end + 1]).long() # 1 memmap read
        X = window[:-1] # [start : end]
        y = window[1:] # [start+1 : end+1]
        return X, y

class DataPipeline:
    """DataPipeline to prepare data for training
    
    Builds the tokenized corpus from HuggingFace via streaming into
    compact uint16 .bin files, then create train/validation DataLoaders and train/validation sampler.
    The corpus is built once and reused across sessions (skip tokenization if they already exist).

    Tiktoken tokenizer is used for tokenization.
    (r50k_base → n_vocab ≈ 50257) 
    
    Attributes:
        cfg (DataConfig): Data hyperparameters.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.tokenizer = tiktoken.get_encoding("r50k_base")

    def batch_tokenize(
        self, 
        stream: IterableDataset, 
        out_path: str, 
        max_token: int, 
        chunk_size: int
        ) -> int:
        """Tokenize corpus in chunk"""
        assert self.tokenizer.n_vocab <= 65535 # check that vocab_size is smaller than uint16
        
        total_token = 0
        buffer = []
        r0print("Starting tokenization...")
        with open(out_path, "wb") as f:
            for batch in stream.iter(batch_size=1000):
                encoded = self.tokenizer.encode_batch(batch["text"], allowed_special={"<|endoftext|>"})
                
                for ids in encoded:
                    """flatten the doc and append EOS token to the end each document
                    to separate the document"""
                    buffer.extend(ids)
                    buffer.append(self.tokenizer.eot_token)
                
                remaining = max_token - total_token
                if len(buffer) >= remaining:        # Final buffer.
                    buffer = buffer[:remaining]     # trim to match exact max tokens
                    total_token += len(buffer)
                    np.asarray(buffer, dtype=np.uint16).tofile(f)
                    break   
                                
                if len(buffer) >= chunk_size:       # normal buffer, saves data once it reach desires chunk size
                    total_token += len(buffer)
                    np.asarray(buffer, dtype=np.uint16).tofile(f)
                    r0print(f"Accumulated tokens: {total_token:,}/{max_token:,}")
                    buffer.clear() # reset buffer size to 0

        return total_token
        
    def build_corpus(self) -> int:
        """Stream corpus from HuggingFace -> batch tokenize -> numpy array -> saves to corpus.bin in output path"""
        stream = load_dataset(
            self.cfg.data_source,
            name=self.cfg.data_source_name,
            split="train",
            streaming=True,
            cache_dir=self.cfg.cache_dir,
        )
        total = self.cfg.total_token
        corpus_len = self.batch_tokenize(stream, self.cfg.corpus_bin_path, total, self.cfg.chunk_size)
        r0print(f"Corpus built. Total {corpus_len:,} tokens (target {total:,})")
        return corpus_len
            
    def split_corpus(self):
        """Split corpus.bin into train/valid .bin"""
        corpus = np.memmap(self.cfg.corpus_bin_path, dtype=np.uint16, mode="r")
        train_tokens = int(self.cfg.train_split * self.cfg.total_token)
        valid_tokens = int(self.cfg.valid_split * self.cfg.total_token)
        corpus[:train_tokens].tofile(self.cfg.train_bin_path) # first n tokens to train
        corpus[train_tokens:self.cfg.total_token].tofile(self.cfg.valid_bin_path) # the rest to valid
        r0print(f"Split: train {train_tokens:,}, valid {valid_tokens:,} tokens")   
        
    def check_corpus_exist(self):
        """Verify whether corpus already exist, if exists will skip tokenization"""
        if os.path.exists(self.cfg.train_bin_path) and os.path.exists(self.cfg.valid_bin_path):
            # already has train/valid split
            r0print("Train/Validation corpus already exists, skipping...")
            return
        elif os.path.exists(self.cfg.corpus_bin_path):
            # no train/valid split but has full corpus
            self.split_corpus()
        else:
            # has none of the corpus
            self.build_corpus()
            self.split_corpus()
            
    @staticmethod
    def check_corpus_size(bin_path: str, expected: int):
        """Verify that corpus token size match the expected token"""
        num_tokens = os.path.getsize(bin_path) // 2  # uint16 = 2 bytes/token
        diff = 100 * abs(num_tokens - expected) / expected
        ok = diff < 1 # acceptable < 1% token mismatch
        msg = f"{bin_path}: {num_tokens:,} tokens (expected {expected:,}, diff {diff:.2f}%)"
        r0print(f"{msg} -> {'OK' if ok else 'MISMATCH'}")

    def _verify_sizes(self):
        total = self.cfg.total_token
        train = int(self.cfg.train_split * total)
        self.check_corpus_size(self.cfg.train_bin_path, train)
        self.check_corpus_size(self.cfg.valid_bin_path, total - train)
        
    def make_dataset(self) -> tuple[TokenDataset, TokenDataset]:
        """Build train/validation datasets"""
        train_ds = TokenDataset(path=self.cfg.train_bin_path, seq_len=self.cfg.seq_len, stride=self.cfg.stride)
        valid_ds = TokenDataset(path=self.cfg.valid_bin_path, seq_len=self.cfg.seq_len, stride=self.cfg.stride, max_tokens=self.cfg.val_monitor_token)
        return train_ds, valid_ds

    def make_loader(self) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
        """Build train/validation dataloader. Also handles DDP using DistributedSampler"""
        train_ds, valid_ds = self.make_dataset()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) 
        rank = int(os.environ.get("LOCAL_RANK", 0)) 
        
        if world_size > 1: # Have more than 1 GPU
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
            shuffle = False
        else:
            train_sampler = None
            valid_sampler = None
            shuffle = True
            
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,  # Use sampler to shuffle the data instead                
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
        )
    
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,           
            sampler=valid_sampler,        
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
        )
        r0print("Train/Validation DataLoader loaded")
        return train_loader, valid_loader, train_sampler, valid_sampler
    
    def make_pipeline(self) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
        """Make full pipeline. Only need to call this method"""
        self.check_corpus_exist()
        self._verify_sizes()
        return self.make_loader()
        
if __name__ == "__main__":
    # Sanity check on all functions
    load_dotenv()
    # tokenize a sentence -> Dataset
    # tokenizer = tiktoken.get_encoding("p50k_base")
    # input_ids = tokenizer.encode("The cat was underperforming in Q2 so it was put on PIP")
    # arr = np.array(input_ids, dtype=np.uint16)
    # ds = TokenDataset.__new__(TokenDataset)   # bypass __init__ for the test...
    # print(arr, "\n")
    
    # Load and tokenize test on full corpus
    pipeline = DataPipeline(config.DataConfig)
    train_loader, valid_loader = pipeline.make_pipeline()
    
    print(len(train_loader), len(valid_loader))