"""data.py — token arrays, TokenDataset, and dataset construction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import config

class TokenDataset(Dataset):
    """
    TokenDataset takes fully tokenized corpus using np.memmap to prevent writing corpus to memory. 
    Then returns target, shifted right by 1 token. This is used to do next token prediction.
    
    Input/Output: __getitem__(idx) -> (X[idx], y[idx])
        
    Attributes:
        path (str): Fully tokenized corpus path. E.g. /data/corpus.bin
        window_len (int): The context size that will be fed to the model.
        stride (int): How far apart __getitem__ will sample the next idx. Defaults to 1.
    """
    def __init__(self, path: str, seq_len: int, stride: int = 1):
        self.token = np.memmap(path, dtype=np.int64, mode="r") 
        self.seq_len = seq_len
        self.stride = stride  

    def __len__(self) -> int:
        return (len(self.encoded_text) - self.seq_len) // self.stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise ValueError("Index is out of range.")
        start = idx * self.stride
        end = start + self.seq_len
        X = torch.from_numpy(self.token[start : end])
        y = torch.from_numpy(self.token[start+1 : end+1])
        return X, y
    
def load_and_tokenize(text_path: str, out_path: str, tokenizer) -> torch.Tensor:
    """Read raw corpus -> numpy array -> saves to .bin in output path"""
    text = open(text_path, encoding="utf-8").read()
    input_ids = tokenizer.encode(text)  
    np.array(input_ids, dtype=np.int64).tofile(out_path)
    
def make_dataset() -> tuple[TokenDataset, TokenDataset]:
    """Build train/validation datasets"""
    cfg = config.DataConfig()
    train_ds = TokenDataset(path=cfg.train_path,seq_len=cfg.seq_len, stride=cfg.stride)
    valid_ds = TokenDataset(path=cfg.valid_path, seq_len=cfg.seq_len, stride=cfg.stride)
    return train_ds, valid_ds


if __name__ == "__main__":
    # Sanity check on all functions
    import tiktoken

    # tokenize a sentence -> Dataset
    tok = tiktoken.get_encoding("gpt2")
    input_ids = tok.encode("The cat was underperforming in Q2 so it was put on PIP")
    arr = np.array(input_ids, dtype=np.int64)
    ds = TokenDataset.__new__(TokenDataset)   # bypass __init__ for the test...
    
    # Load and tokenize test on full corpus
    load_and_tokenize("data/corpus.txt", "data/corpus.bin", tok)