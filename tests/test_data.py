import numpy as np
import torch
import pytest
from data import TokenDataset

def _write_corpus(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(path)

def test_dataset_length(tmp_path):
    path = tmp_path / "corpus.bin"
    _write_corpus(path, list(range(20)))
    ds = TokenDataset(str(path), seq_len=5, stride=5)
    assert len(ds) == (20 - 5) // 5 == 3

def test_dataset_shift(tmp_path):
    path = tmp_path / "corpus.bin"
    _write_corpus(path, list(range(10)))
    ds = TokenDataset(str(path), seq_len=4, stride=4)
    X, y = ds[0]
    assert torch.allclose(X, torch.tensor([0, 1, 2, 3]).long())
    assert torch.allclose(y, torch.tensor([1, 2, 3, 4]).long())

def test_dataset_index_out_of_range(tmp_path):
    path = tmp_path / "corpus.bin"
    _write_corpus(path, list(range(10)))
    ds = TokenDataset(str(path), seq_len=4, stride=4)
    with pytest.raises(ValueError):
        ds[1000]