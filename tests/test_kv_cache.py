"""Deterministic tensor-level verification that the KV cache reproduces
the uncached forward pass exactly.

The idea: a correct KV cache must produce identical logits to a full
uncached forward pass, because it is only a recomputation optimization.
Run with model in eval mode so dropout cannot introduce path-dependent noise.

Comparison structure (matching the canonical prefill/decode split):
    1. Prefill: cached prefill logits == uncached full-sequence logits.
    2. Decode:  cached 1-token decode logits == uncached logits for the same
    token appended to the full sequence.
"""
import torch
import pytest
from model import KVCache


@pytest.fixture(scope="module")
def eval_model(tiny_model):
    """The tiny model in eval mode (disables dropout - essential for this test)."""
    return tiny_model.eval()


@pytest.fixture(scope="module")
def prompt_ids(tiny_cfg):
    """Short prompt (seq_len//2) so decode steps stay within seq_len."""
    torch.manual_seed(0)
    return torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.seq_len // 2)).long()


def _make_cache(tiny_cfg, device="cpu"):
    return KVCache(
        n_layers=tiny_cfg.n_layers,
        n_heads=tiny_cfg.n_heads,
        seq_len=tiny_cfg.seq_len,
        d_head=tiny_cfg.embed_dim // tiny_cfg.n_heads,
        device=device,
    )


def test_prefill_matches_uncached(eval_model, tiny_cfg, prompt_ids):
    """Prefill (full prompt through cache) == uncached full forward, last position."""
    with torch.inference_mode():
        full_logits = eval_model(prompt_ids)                      # [1, S, V] no cache

        cache = _make_cache(tiny_cfg)
        prompt_logits = eval_model(prompt_ids, kv_cache=cache, offset=0)
        cache.advance(prompt_ids.size(1))

    assert prompt_logits.shape == full_logits.shape
    torch.testing.assert_close(
        full_logits[:, -1, :],   # last prompt position, uncached
        prompt_logits[:, -1, :], # same position, cached prefill
        atol=1e-5,
        rtol=1e-5,
    )


def test_decode_matches_uncached(eval_model, tiny_cfg, prompt_ids):
    """Single-token cached decode == same token appended in a full forward."""
    # prefill the cache
    cache = _make_cache(tiny_cfg)
    with torch.inference_mode():
        eval_model(prompt_ids, kv_cache=cache, offset=0)
        cache.advance(prompt_ids.size(1))

        # the "next" token: last prompt token re-fed (deterministic, no sampling)
        next_input = prompt_ids[:, -1:]                            # [1, 1]
        cached_logits = eval_model(next_input, kv_cache=cache, offset=cache.pos)

        # uncached: same tokens as one full sequence
        extended = torch.cat([prompt_ids, next_input], dim=-1)     # [1, S+1]
        full_extended_logits = eval_model(extended)

    torch.testing.assert_close(
        full_extended_logits[:, -1, :],  # position S, uncached
        cached_logits[:, -1, :],         # position S, cached decode
        atol=1e-5,
        rtol=1e-5,
    )


def test_multi_step_decode_matches_uncached(eval_model, tiny_cfg, prompt_ids):
    """Every decode step must match the growing uncached sequence, not just the first."""
    cache = _make_cache(tiny_cfg)
    ids = prompt_ids.clone()
    n_steps = tiny_cfg.seq_len // 4

    with torch.inference_mode():
        # prefill
        eval_model(ids, kv_cache=cache, offset=0)
        cache.advance(ids.size(1))

        for _ in range(n_steps):
            # cached: feed one token, decode at its absolute position
            nxt = torch.randint(0, tiny_cfg.vocab_size, (1, 1)).long()
            cached_logits = eval_model(nxt, kv_cache=cache, offset=cache.pos)
            cache.advance(1)

            # uncached: same full sequence grown by one token
            ids = torch.cat([ids, nxt], dim=-1)
            full_logits = eval_model(ids)

            torch.testing.assert_close(
                full_logits[:, -1, :],
                cached_logits[:, -1, :],
                atol=1e-5,
                rtol=1e-5,
            )
