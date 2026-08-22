import torch
from model import MultiHeadAttention, SwiGLUFFN, TransformerBlock
from pos_encoding import FrequencyPE

def test_swiglu_output_shape():
    m = SwiGLUFFN(embed_dim=128, ff_dims=341)
    out = m(torch.randn(2, 16, 128))
    assert out.shape == (2, 16, 128)

def test_swiglu_gate_up_even_out_features():
    for d in (128, 512, 640):
        block = SwiGLUFFN(d, (8 * d) // 3 // 64 * 64)
        assert block.gate_up.out_features % 2 == 0

def test_mha_output_shape(tiny_cfg):
    pos = FrequencyPE(tiny_cfg.embed_dim // tiny_cfg.n_heads)
    mha = MultiHeadAttention(
        pos(tiny_cfg.seq_len), 
        tiny_cfg.seq_len,
        tiny_cfg.embed_dim, 
        tiny_cfg.n_heads
    )
    out = mha(torch.randn(2, tiny_cfg.seq_len, tiny_cfg.embed_dim))
    assert out.shape == (2, tiny_cfg.seq_len, tiny_cfg.embed_dim)

def test_transformer_block_output_shape(tiny_cfg):
    pos = FrequencyPE(tiny_cfg.embed_dim // tiny_cfg.n_heads)
    block = TransformerBlock(
        pos(tiny_cfg.seq_len), 
        tiny_cfg.seq_len,
        tiny_cfg.embed_dim, 
        tiny_cfg.n_heads
    )
    out = block(torch.randn(2, tiny_cfg.seq_len, tiny_cfg.embed_dim))
    assert out.shape == (2, tiny_cfg.seq_len, tiny_cfg.embed_dim)

def test_model_output_logits(tiny_model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.seq_len))
    out = tiny_model(x)
    assert out.shape == (2, tiny_cfg.seq_len, tiny_cfg.vocab_size)

def test_weight_tied(tiny_model):
    assert tiny_model.output.weight is tiny_model.embed.weight

def test_init_std(tiny_model):
    std = tiny_model.embed.weight.std().item()
    assert 0.005 < std < 0.1

def test_causal_mask_is_lower_triangular(tiny_cfg):
    pos = FrequencyPE(tiny_cfg.embed_dim // tiny_cfg.n_heads)
    mha = MultiHeadAttention(
        pos(tiny_cfg.seq_len), 
        tiny_cfg.seq_len,
        tiny_cfg.embed_dim, 
        tiny_cfg.n_heads
    )
    mask = mha.causal_mask
    assert mask.triu(1).sum().item() == 0