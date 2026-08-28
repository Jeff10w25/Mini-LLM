"""
test_generation.py - end-to-end generation tests: KV-cache correctness,
sliding-window bounds, rebuild correctness, and long-run stability.

Uses a small model with seq_len=1024 (same context length as the 90M
training config) so the window/rebuild logic is exercised realistically
without the 90M's runtime cost.
"""
import torch
import pytest
import config
from model import MiniGPT, KVCache
from generate import Generator


@pytest.fixture(scope="module")
def gen_cfg_1024(tmp_path_factory):
    """A tiny model config with the 90M's real context length."""
    return config.ModelConfig(
        vocab_size=50257, embed_dim=64, n_heads=4, n_layers=2, seq_len=1024
    )

@pytest.fixture(scope="module")
def eval_model_1024(gen_cfg_1024):
    return MiniGPT(gen_cfg_1024).eval()

@pytest.fixture(scope="module")
def make_generator(tmp_path_factory, gen_cfg_1024, eval_model_1024):
    """Factory: build a Generator backed by the tiny model's weights on CPU."""
    def _make(max_tokens=64):
        ckpt = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
        torch.save({"model": eval_model_1024.state_dict()}, ckpt)
        gen_cfg = config.GeneratorConfig(
            device="cpu", ckpt_path=str(ckpt), max_tokens=max_tokens,
        )
        gen = Generator(gen_cfg_1024, gen_cfg)
        gen.model.eval()
        return gen
    return _make

def _make_cache(cfg):
    return KVCache(
        cfg.n_layers, cfg.n_heads, cfg.seq_len,
        cfg.embed_dim // cfg.n_heads, "cpu",
    )

def test_cache_matches_nocache(make_generator):
    """Greedy + same seed: cached and uncached must produce identical text."""
    gen = make_generator(max_tokens=32)
    prompt = "The quick brown fox jumps"

    torch.manual_seed(0)
    out_cache = gen.generate(prompt, caching=True, keep=512)

    torch.manual_seed(0)
    out_nocache = gen.generate(prompt, caching=False, keep=512)

    assert out_cache == out_nocache

@pytest.mark.slow
def test_window_never_exceeds_seq_len(make_generator, gen_cfg_1024):
    """Every forward call's input must stay <= seq_len, even past the boundary."""
    gen = make_generator(max_tokens=gen_cfg_1024.seq_len * 3)   # 3072 tokens, ~5 rebuilds
    seen = []
    orig_fwd = gen.model.forward

    def spy(input_ids, *args, **kwargs):
        seen.append(input_ids.shape[1])
        return orig_fwd(input_ids, *args, **kwargs)

    gen.model.forward = spy
    try:
        gen.generate("The cat sat on the mat", caching=True, keep=512)
    finally:
        gen.model.forward = orig_fwd

    assert seen, "model was never called"
    assert max(seen) <= gen_cfg_1024.seq_len

def test_rebuild_matches_fresh_prefill(gen_cfg_1024, eval_model_1024):
    """After a rebuild, the window's logits must equal a from-scratch prefill."""
    keep = 512
    prompt = torch.randint(0, gen_cfg_1024.vocab_size, (1, 200)).long()
    cache = _make_cache(gen_cfg_1024)

    with torch.inference_mode():
        # prefill + decode a few tokens to fill context
        eval_model_1024(prompt, kv_cache=cache, offset=0)
        cache.advance(prompt.shape[1])
        ids = prompt[:, -1:]
        total = torch.cat([prompt, ids], dim=-1)
        for _ in range(5):
            logits = eval_model_1024(ids, kv_cache=cache, offset=cache.pos)
            ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache.advance(1)
            total = torch.cat([total, ids], dim=-1)

        # rebuild: window re-prefilled into a fresh cache
        window = total[:, -keep:]
        cache2 = _make_cache(gen_cfg_1024)
        logits_rebuilt = eval_model_1024(window, kv_cache=cache2, offset=0)

        # fresh uncached forward over the same window
        logits_fresh = eval_model_1024(window)

    torch.testing.assert_close(
        logits_rebuilt[:, -1, :], logits_fresh[:, -1, :], atol=1e-5, rtol=1e-5
    )

@pytest.mark.slow
def test_long_generation_no_crash(make_generator, gen_cfg_1024):
    """Many rebuilds (10+) must run without error and produce text."""
    gen = make_generator(max_tokens=gen_cfg_1024.seq_len * 10)   # 10240 tokens, ~20 rebuilds
    out = gen.generate("Once upon a time", caching=True, keep=512)
    assert isinstance(out, str) and len(out) > 0

@pytest.mark.slow
def test_keep_variants_differ(make_generator, gen_cfg_1024):
    """Different keep sizes must produce different text (windows differ)."""
    gen = make_generator(max_tokens=gen_cfg_1024.seq_len * 2)   # 2 rebuilds minimum
    prompt = "Deep in the forest, a strange light appeared"

    torch.manual_seed(0)
    out_half = gen.generate(prompt, caching=True, keep=gen_cfg_1024.seq_len // 2)
    torch.manual_seed(0)
    out_quarter = gen.generate(prompt, caching=True, keep=gen_cfg_1024.seq_len // 4)

    assert out_half != out_quarter   # different windows -> different text