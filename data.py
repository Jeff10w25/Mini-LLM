"""data.py — token arrays, TokenDataset, and dataset construction."""

from __future__ import annotations

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from dotenv import load_dotenv

import config

# Load HF_TOKEN to allows faster download from HuggingFace
load_dotenv()

class TokenDataset(Dataset):
    """
    TokenDataset takes fully tokenized corpus using np.memmap to prevent writing full corpus to memory. 
    Then returns target, shifted right by 1 token. This is used to do next token prediction.
    
    Input/Output: __getitem__(idx) -> (X[idx], y[idx])
        
    Attributes:
        path (str): Fully tokenized corpus path. E.g. /data/corpus.bin
        window_len (int): The context size that will be fed to the model.
        stride (int): How far apart __getitem__ will sample the next idx. Defaults to seq_len.
                    Stride < seq_len is only use to grow the corpus if there isn't enough data.
                    Or if the data download from HuggingFace is a bottleneck.
    """
    def __init__(self, path: str, seq_len: int, stride: int = 512):
        self.token = np.memmap(path, dtype=np.uint16, mode="r") 
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


def batch_tokenize(stream, out_path, max_token, chunk_size, tokenizer):
    """Tokenize corpus in chunk
    """
    
    assert tokenizer.n_vocab <= 65535 # check that vocab_size is smaller than uint16
    
    total_token = 0
    buffer = []
    with open(out_path, "wb") as f:
        for batch in stream.iter(batch_size=10000):
            encoded = tokenizer.encode_batch(batch["text"])
            
            for ids in encoded:
                """flatten the doc and append EOS token to the end each document
                to separate the document"""
                buffer.extend(ids)
                buffer.append(tokenizer.eot_token)
            
            if len(buffer) >= chunk_size:
                total_token += len(buffer)
                arr = np.asarray(buffer, dtype=np.uint16)
                arr.tofile(f)
                print(f"Token amount has reaches buffer size!, saving..." )
                print(f" Accumulated tokens: {total_token:,}/{max_token:,}")
                buffer.clear() # reset buffer size to 0
            
            if total_token >= max_token:
                break

    return total_token
    
def load_and_tokenize(tokenizer) -> torch.Tensor:
    """Stream corpus from HuggingFace -> batch tokenize -> numpy array -> saves to .bin in output path"""
    cfg = config.DataConfig()
    train_max_token_needed = cfg.max_train_token * cfg.stride // cfg.seq_len
    valid_max_token_needed = cfg.max_valid_token * cfg.stride // cfg.seq_len
    
    # Take first n amount of train_max_token_needed
    train_stream = load_dataset(
        cfg.data_source, 
        name=cfg.data_source_name, 
        streaming=True, 
        split="train", 
        cache_dir=cfg.cache_dir
        ).take(train_max_token_needed)
    
    # Skip first n amount of train_max_token_needed
    valid_stream = load_dataset(
        cfg.data_source, 
        name=cfg.data_source_name, 
        streaming=True, 
        split="train", 
        cache_dir=cfg.cache_dir
        ).skip(train_max_token_needed)
    
    train_len = batch_tokenize(train_stream, cfg.train_bin_path, train_max_token_needed, cfg.chunk_size, tokenizer)
    valid_len = batch_tokenize(valid_stream, cfg.valid_bin_path, valid_max_token_needed, cfg.chunk_size, tokenizer)
            
    print(f"Training set loaded. Total {len(train_len)} tokens")
    print(f"Validation set loaded. Total {len(valid_len)} tokens")
        
    
def make_dataset() -> tuple[TokenDataset, TokenDataset]:
    """Build train/validation datasets"""
    cfg = config.DataConfig()
    train_ds = TokenDataset(path=cfg.train_bin_path,seq_len=cfg.seq_len, stride=cfg.stride)
    valid_ds = TokenDataset(path=cfg.valid_bin_path, seq_len=cfg.seq_len, stride=cfg.stride)
    return train_ds, valid_ds


if __name__ == "__main__":
    # Sanity check on all functions
    import tiktoken

    # tokenize a sentence -> Dataset
    tokenizer = tiktoken.get_encoding("p50k_base")
    input_ids = tokenizer.encode("The cat was underperforming in Q2 so it was put on PIP")
    arr = np.array(input_ids, dtype=np.uint16)
    ds = TokenDataset.__new__(TokenDataset)   # bypass __init__ for the test...
    print(arr, "\n")
    
    # Load and tokenize test on full corpus
    load_and_tokenize(tokenizer)
    train, valid = make_dataset()
    print(train.shape, valid.shape)