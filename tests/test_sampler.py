import torch
from generate import TextGenSampler



def test_greedy_returns_argmax_shape():
    logits = torch.randn(2, 100)
    out = TextGenSampler(temperature=0.0)(logits)
    assert out.shape == (2, 1)

def test_top_k_masked_softmax_renormalizes():
    """top_k then softmax, mass must sum to 1."""
    logits = torch.randn(1, 100)
    sampler = TextGenSampler(top_k=20)
    masked = sampler._apply_top_k(logits)
    probs = sampler._apply_softmax(masked)
    assert torch.allclose(probs.sum(-1), torch.ones(1), atol=1e-5)

def test_top_p_output_sum_to_one():
    probs = torch.rand(1, 50) 
    probs = probs / probs.sum(dim=-1)
    sampler = TextGenSampler(top_p=0.5)
    out = sampler._apply_top_p(probs)
    torch.allclose(out.sum(-1),torch.ones(1), atol=1e-5)

def test_top_p_keeps_at_least_one_token():
    """if top token prob >= top_p, it must not produce an empty set."""
    probs = torch.zeros(1, 10)
    probs[0, 0] = 0.97          
    probs[0, 1:] = 0.03 / 9
    sampler = TextGenSampler(top_p=0.95)
    out = sampler._apply_top_p(probs)
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-5)
    assert (out > 0).sum().item() >= 1 
    
def test_top_p_no_nan():
    """the 0/0 NaN crash."""
    probs = torch.zeros(1, 10)
    probs[0, 0] = 1.0           
    sampler = TextGenSampler(top_p=0.95)
    out = sampler._apply_top_p(probs)
    assert not torch.isnan(out).any()
    
def test_top_k_top_p_combined_safe():
    """The combination must never produce NaN or empty."""
    logits = torch.randn(1, 100)
    sampler = TextGenSampler(top_k=50, top_p=0.95)
    masked = sampler._apply_top_k(logits)
    probs = sampler._apply_softmax(masked)
    out = sampler._apply_top_p(probs)
    assert not torch.isnan(out).any()
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-5)
    
def test_sampler_full_pipeline_no_nan():
    """Various combo must not output NaN"""
    logits = torch.randn(1, 100)
    for temp, k, p in [(0.8, 50, 0.95), (0.6, 20, None), (1.0, None, 0.95), (1.0, None, None)]:
        s = TextGenSampler(temp, k, p)
        out = s(logits)
        assert not torch.isnan(out).any()
        assert out.shape == (1, 1)