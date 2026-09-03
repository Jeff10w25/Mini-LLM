# Mini-LLM

A GPT-style causal decoder-only transformer, built from scratch in PyTorch and trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~2B tokens).

## Zero shot benchmark results

MiniGPT (90M, step_15000) vs GPT-2 small (124M), evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
**bold = better**

**Accuracy** (report **acc_norm** when available, else **acc**, higher is better)

| Task | MiniGPT (90M) | GPT-2 small (124M) | Δ |
| :--- | ---: | ---: | ---: |
| Lambada | 0.192 ± 0.005 | **0.326** ± 0.007 | −0.134 |
| HellaSwag | 0.282 ± 0.004 | **0.311** ± 0.005 | −0.029 |
| ARC-Easy | **0.416** ± 0.010 | 0.395 ± 0.010 | **+0.021** |
| ARC-Challenge | **0.239** ± 0.012 | 0.227 ± 0.012 | **+0.012** |
| PIQA | 0.591 ± 0.011 | **0.625** ± 0.011 | −0.034 |
| WinoGrande | 0.511 ± 0.014 | **0.516** ± 0.014 | −0.005 |

**Perplexity** (whole-corpus, lower is better)

| Task | MiniGPT (90M) | GPT-2 small (124M) | Δ |
| :--- | ---: | ---: | ---: |
| Lambada | 358.3 | **40.1** | +318.2 |
| Wikitext (word) | 64.1 | **37.4** | +26.7 |

On accuracy MiniGPT is competitive on most tasks (except Lambada) and even beats GPT-2 on ARC-Easy and ARC-Challenge.

**Why the perplexity score is much higher?**

1. **Data scale**: MiniGPT trained on 2B tokens vs GPT-2's 10B+, so its
   long-tail word knowledge is thinner.
2. **Domain shift**: `wikitext` (and to a degree Lambada) is out of
   distribution vs the educational fineweb-edu corpus. Wikitext
   `word_perplexity` (64 vs 37, ~1.7×) is the cleaner comparison.
3. **Model is still improving**: the earlier `step_13000` scores are worse
   on every generative metric (Lambada ppl 411.5 / acc 0.191, Wikitext word_ppl
   66.3), so `step_15000` is reported and the model is still improving with no signs of overfitting.

## Model Architecture

| | |
|---|---|
| Vocab / context | 50,257 tokens (BPE, r50k) / seq_len **1024** |
| Hidden / heads / layers | embed_dim **640** / **10** heads / **12** layers |
| Attention | fused QKV -> **RoPE** rotary positions -> causal masked **MHA** |
| Feed-forward | **SwiGLU** (SiLU-gated), ~2.67× expansion, no bias |
| Normalization | pre-norm **RMSNorm** before attn + FFN |
| Output | final RMSNorm and tied LM head (shared embedding weights) |

Highlights:
- **RoPE** (Rotary positional encoding) fixed position table and compute via rotation.
- **Fused QKV** projection and **fused SwiGLU gate+up** for fewer matmuls.
- **Pre-norm + residual** transformer blocks.

## Features

- KV cache + sliding-window generation: reduces load on memory and handles beyond context generation
- Text sampling with temperature / top-k / top-p + repetition annealing
- JSON configs as the single source of truth (`configs/*.json`)
- Generated text quality analyzer (repetition rate)
- lm-eval benchmark harness for MiniGPT + GPT-2
- pytest suite (fast/slow markers)

## Project Structure

```text
src/
├── main.py            # training entrypoint (loads configs/*.json)
├── train.py           # Trainer: AMP, grad-accum, DDP, resume, eval, ckpt
├── model.py           # MiniGPT, MultiHeadAttention, KVCache, SwiGLU
├── pos_encoding.py    # RoPE + sinusoidal variants
├── data.py            # FineWeb-Edu streaming -> tokenized .bin + DataLoader
├── generate.py        # Generator + sampler (KV cache, sliding window, CLI)
├── analyzer.py        # summarize samples/*.jsonl into a comparison table
├── config.py          # dataclasses + from_json/to_json
├── utils.py           # shared helpers
└── benchmark/         # bench.py + MiniGPT_LM wrapper (lm-eval)
configs/               # model.json, data.json, train.json, gen.json, bench.json
tests/                 # pytest (fast + slow)
checkpoints/           # trained checkpoints (gitignored)
samples/               # generation experiment logs (JSONL)
```

## Setup

All configs live in `configs/*.json` (single source of truth). But can use CLI flags to override them. Recommend training on GPU if available, will require different torch version.

```bash
pip install -r requirements.txt

# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Training

```bash
# smoke run (caps max_iters without editing JSON)
python src/main.py --steps 500

# real run, resume from a checkpoint
python src/main.py --resume checkpoints/mini/step_15000.pt

# override any config field
python src/main.py --override data.num_workers=8 train.max_iters=20000

# enable torch.compile (GPU + Linux recommended; skip on Windows)
python src/main.py --compile

# multi-GPU training with torch.distributed and torch.compile (recommend for kaggle 2xT4)
torchrun --nproc_per_node=2 src/main.py --resume checkpoints/mini/step_15000.pt --compile
```

## Text Generation

```bash
python src/generate.py --prompt "Once upon a time" --max-tokens 300 \
    --temperature 0.8 --top-k 50 --top-p 0.95
```

## Evaluation with lm-evaluation-harness

```bash
# run from the repo root (wrapper uses relative paths)
python -m src\benchmark\bench.py
```

## Testing

```bash
# test all
python -m pytest tests/

# test only not slow one
python -m pytest tests/ -m "not slow"
```

## License

MIT