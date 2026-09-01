import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import config
from model import MiniGPT

@pytest.fixture(scope="session")
def tiny_cfg():
    """A tiny config so model tests run in seconds, not minutes."""
    return config.ModelConfig(
        vocab_size=50257, embed_dim=32, n_heads=4, n_layers=2, seq_len=16
    )

@pytest.fixture(scope="session")
def tiny_model(tiny_cfg):
    return MiniGPT(tiny_cfg)
