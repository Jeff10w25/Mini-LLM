import torch
from pos_encoding import FrequencyPE, RoPE, SinusoidalPE

def test_sinusoidal_shape():
    pe = SinusoidalPE(max_len=10, embed_dim=8)
    out = pe(torch.randn(2, 5, 8))
    assert out.shape == (5, 8)

def test_frequency_pe_shape():
    pe = FrequencyPE(embed_dim=6)
    enc = pe(12)
    assert enc.shape == (12, 6)

def test_rotate_half():
    x = torch.arange(8).float()
    out = RoPE._rotate_half(x)
    assert torch.allclose(out, torch.tensor([-4, -5, -6, -7, 0, 1, 2, 3]).float())

def test_rope_keeps_shape():
    pe = FrequencyPE(embed_dim=6)
    rope = RoPE(pe(8))
    x = torch.randn(1, 1, 5, 6)
    assert rope.rotate(x).shape == x.shape

def test_rope_registers_cos_sin_buffers():
    pe = FrequencyPE(embed_dim=6)
    rope = RoPE(pe(8))
    assert hasattr(rope, "cos_cache") and hasattr(rope, "sin_cache")