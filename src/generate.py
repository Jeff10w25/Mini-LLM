"""generate.py - TextGenSampler + Generator (engine) and a JSON+argparse CLI.

Run:
    python generate.py --prompt "Once upon a time" --max-tokens 200 --temperature 0.8 --top-k 50

Config source of truth = configs/model.json (model) + configs/gen.json (sampling).
CLI flags override gen.json for a single run; nothing is written back.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import argparse
import json
import os
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils
from model import MiniGPT, KVCache

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor
    from config import ModelConfig, GeneratorConfig


class TextGenSampler:
    """Sampler for next-token decoding.

    Applies, in order: banned-token masking -> temperature -> top-k -> softmax
    -> top-p. temperature <= 0 selects greedy argmax and ignores the rest.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        banned_tokens: int | None = None,
    ):
        if (top_k is not None) and top_k <= 0:
            raise ValueError("Top-k cannot be 0 or float")
        if (top_p is not None) and not (0.0 < top_p < 1.0):
            raise ValueError("Top-p must be value between 0 and 1")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.banned_tokens = banned_tokens

    def __call__(self, logits: Tensor):
        if self.temperature <= 0:
            # greedy sampling, ignore top_p and top_k and select highest logit's token
            return logits.argmax(dim=-1, keepdim=True)

        if self.banned_tokens is not None:
            logits = logits.clone()
            logits[..., self.banned_tokens] = -torch.inf

        logits = self._apply_temperature(logits)
        if self.top_k is not None:
            logits = self._apply_top_k(logits)

        probs = self._apply_softmax(logits)
        if self.top_p is not None:
            probs = self._apply_top_p(probs)

        return torch.multinomial(probs, 1)

    def _apply_softmax(self, logits: Tensor):
        return F.softmax(logits, dim=-1)

    def _apply_temperature(self, logits: Tensor):
        if self.temperature > 0:  # just to prevent dividing with 0
            return logits / self.temperature

    def _apply_top_k(self, logits: Tensor):
        """Take top_k val/idx, return mask with all value except top_k val/idx negative infinity"""
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.full_like(logits, -torch.inf)
        mask.scatter_(index=top_k_idx, src=top_k_vals, dim=-1)
        return mask

    def _apply_top_p(self, probs: Tensor):
        """Take top_p val/idx, return top_p val/idx with normalized probabilities. Other idx prob is 0"""
        if 0.0 < self.top_p < 1.0:
            # sort probs in descending order and do cumulative sum
            probs_val, probs_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(probs_val, dim=-1)
            #   cumsum      : [0.50, 0.80, 0.95, 0.99, 1.00]
            #   - probs_val : [0.50, 0.30, 0.15, 0.04, 0.01]
            #   = before    : [0.00, 0.50, 0.80, 0.95, 0.99] so that it keep the token that just pass the threshold too
            keep = (cumsum - probs_val) <= self.top_p
            keep[:, 0] = True
            # make a mask size of probs and scatter the probs_val value at top_p_idx
            probs_val = probs_val.masked_fill(~keep, 0.0)
            mask = torch.zeros_like(probs)
            mask.scatter_(index=probs_idx, src=probs_val, dim=-1)
            # renormalize by dividing the cutting spot value so that it sums to 1
            mask = mask / mask.sum(dim=-1, keepdim=True)
        else:
            return probs
        return mask


class Generator:
    def __init__(self, model_cfg: ModelConfig, gen_cfg: GeneratorConfig):
        self.device = gen_cfg.device
        model_state = torch.load(gen_cfg.ckpt_path, map_location=self.device, weights_only=False)["model"]
        self.model = MiniGPT(model_cfg).to(gen_cfg.device)
        self._raw_model().load_state_dict(model_state, strict=False)
        self.model.eval()  # evaluation mode
        self.max_tokens = gen_cfg.max_tokens
        self.sampler = TextGenSampler(gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.banned_tokens)
        self.tokenizer = tiktoken.get_encoding("r50k_base")
        self.model_cfg = model_cfg
        self.gen_cfg = gen_cfg
        self._current_preset = None

    def is_repeating(
        self,
        total_ids: Tensor,
        n: int = 4,
        tail: int = 32,
        threshold: float = 0.3,
    ) -> bool:
        tokens = total_ids[0, -tail:].tolist()
        seen = set()
        repeats = 0
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if gram in seen:
                repeats += 1
            else:
                seen.add(gram)
        return (repeats / (len(tokens) - n + 1)) > threshold

    def gen_annealing(self, total_ids: Tensor, annealing: str | None = None):
        if annealing is None:
            return
        base_temp, base_k, base_p = self.sampler.temperature, self.sampler.top_k, self.sampler.top_p
        max_temp, max_k, max_p = 2.0, 1000, 0.99

        if self.is_repeating(total_ids):
            # if n_gram repeating, loosen everything that's active
            if annealing == "temp":
                self.sampler.temperature = min(self.sampler.temperature * 1.1, max_temp)
            elif annealing == "top_k" and self.sampler.top_k is not None:
                self.sampler.top_k = min(int(self.sampler.top_k * 1.5), max_k)
            elif annealing == "top_p" and self.sampler.top_p is not None:
                self.sampler.top_p = min(self.sampler.top_p + 0.05, max_p)
        else:
            # if no longer repeating, decrease toward base value
            if annealing == "temp":
                self.sampler.temperature = max(base_temp, self.sampler.temperature / 1.1)
            elif annealing == "top_k" and self.sampler.top_k is not None:
                self.sampler.top_k = max(base_k, self.sampler.top_k // 1.5)
            elif annealing == "top_p" and self.sampler.top_p is not None:
                self.sampler.top_p = max(base_p, self.sampler.top_p - 0.05)

    def _raw_model(self):
        model = self.model
        model = getattr(model, "_orig_mod", model)   # strip torch.compile (outer layer)
        model = getattr(model, "module", model)      # strip DDP (inner layer), if present
        model = model.to(self.device)
        return model

    def save_output(self, entry: dict, save_path: str | None = None):
        """Append one run's record as a JSON line in the output dir."""
        if save_path is None:
            stem = os.path.splitext(os.path.basename(self.gen_cfg.ckpt_path))[0]
            save_path = os.path.join(self.gen_cfg.output_dir, f"generations_{stem}.jsonl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def generate(
        self,
        prompt: str,
        caching: bool = True,
        keep: int | None = None,
        to_json: bool = False,
        gen_annealing: str | None = None,
    ) -> str:
        # No instance side effects: run params are resolved to locals and the sampler
        # state is restored afterwards, so calling generate() again starts clean and
        # does not overwrite self.max_tokens or leak drifted anneal params.
        sampler_state = (self.sampler.temperature, self.sampler.top_k, self.sampler.top_p)

        use_full_window = keep is None        # no window -> generate seq_len tokens
        window = self.model_cfg.seq_len if use_full_window else keep
        assert window <= self.model_cfg.seq_len  # cannot keep more than seq_len tokens
        budget = self.model_cfg.seq_len if use_full_window else self.gen_cfg.max_tokens
        print(f"keep={window}, max_tokens={budget}")

        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # start with the prompt
        total_ids = input_ids
        generated = prompt_ids
        kv_cache = KVCache(
            self.model_cfg.n_layers,
            self.model_cfg.n_heads,
            self.model_cfg.seq_len,
            self.model_cfg.embed_dim // self.model_cfg.n_heads,
            self.gen_cfg.device,
        ) if caching else None

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                for _ in range(budget):
                    offset = kv_cache.pos if caching else 0
                    logits = self.model(input_ids, kv_cache=kv_cache, offset=offset)
                    next_token = self.sampler(logits[:, -1, :])
                    if caching:
                        kv_cache.advance(input_ids.shape[1])           # advance by # tokens
                        offset = kv_cache.pos
                        total_ids = torch.cat([total_ids, next_token], dim=-1)
                        input_ids = next_token
                        if total_ids.shape[1] >= self.model_cfg.seq_len:
                            # prefill with new truncated text_ids and new empty cache again
                            kv_cache = KVCache(
                                self.model_cfg.n_layers,
                                self.model_cfg.n_heads,
                                self.model_cfg.seq_len,
                                self.model_cfg.embed_dim // self.model_cfg.n_heads,
                                self.gen_cfg.device,
                            )
                            total_ids = total_ids[:, -window:]      # keep last `window` tokens
                            input_ids = total_ids                 # next loop call pre-fills the window
                            offset = 0
                    else:
                        input_ids = torch.cat([input_ids, next_token], dim=-1)  # [1, S+1]
                        if input_ids.shape[1] >= self.model_cfg.seq_len:  # if about to exceed seq_len
                            input_ids = input_ids[:, -window:]
                    generated.append(int(next_token[-1].item()))           # store token id as int

                    # temperature annealing check every step (mutates the sampler only
                    # for this run; state is restored in `finally` below)
                    if gen_annealing is not None:
                        self.gen_annealing(total_ids, annealing=gen_annealing)

                full_text = self.tokenizer.decode(generated)
        finally:
            # restore sampler params so anneal drift never leaks into later calls
            self.sampler.temperature, self.sampler.top_k, self.sampler.top_p = sampler_state

        t1 = time.perf_counter()
        print(f"Generation time: {t1 - t0:.3f}s\n")
        print(full_text)

        # if saving to JSON (records the configured sampler values actually used)
        if to_json:
            self.save_output({
                "preset": self._current_preset,        # optional: set before calling
                "temperature": sampler_state[0],
                "top_k": sampler_state[1],
                "top_p": sampler_state[2],
                "anneal": gen_annealing,
                "keep": window,
                "max_tokens": budget,
                "time_s": round(t1 - t0, 3),
                "prompt": prompt,
                "output": full_text,
            })

        return full_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT (configs from JSON).")
    parser.add_argument("--model-json", default="configs/model.json")
    parser.add_argument("--gen-json", default="configs/gen.json")
    parser.add_argument("--prompt",
                        default="There was a time in 1920 in Paris where half of the population")
    parser.add_argument("--ckpt", default=None, help="override GeneratorConfig.ckpt_path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--banned-token", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--keep", type=int, default=None,
                        help="sliding-window length (default from gen.json keep)")
    parser.add_argument("--anneal", default=None, choices=[None, "temp", "top_k", "top_p"],
                        help="loosen sampler params when n-gram repetition is detected")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    parser.add_argument("--to-json", action="store_true",
                        help="append the run to generations_*.jsonl in gen.json output_dir")
    args = parser.parse_args()

    model_cfg = config.from_json(config.ModelConfig, args.model_json)
    gen_cfg = config.from_json(config.GeneratorConfig, args.gen_json)

    # CLI flags override the JSON for this run only (nothing is written back)
    overrides = {
        "ckpt":        "ckpt_path",
        "device":      "device",
        "temperature": "temperature",
        "top_k":       "top_k",
        "top_p":       "top_p",
        "banned_token":"banned_tokens",
        "max_tokens":  "max_tokens",
    }
    for flag, field in overrides.items():
        val = getattr(args, flag)
        if val is not None:
            setattr(gen_cfg, field, val)

    seed = args.seed if args.seed is not None else gen_cfg.seed
    utils.seed_everything(seed)

    # runtime toggles default from gen.json; CLI flags override them.
    keep = args.keep if args.keep is not None else gen_cfg.keep
    anneal = args.anneal if args.anneal is not None else gen_cfg.anneal
    caching = gen_cfg.caching and not args.no_cache
    to_json = gen_cfg.to_json or args.to_json

    gen = Generator(model_cfg, gen_cfg)
    gen.generate(
        args.prompt,
        caching=caching,
        keep=keep,
        to_json=to_json,
        gen_annealing=anneal,
    )


if __name__ == "__main__":
    main()
