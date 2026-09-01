from __future__ import annotations
from typing import TYPE_CHECKING

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
    """Sampler 
    """
    def __init__(
        self, 
        temperature: float = 1.0, 
        top_k: int | None = None,
        top_p: float | None = None,
        banned_tokens: int | None = None
        ): 
        
        if (top_k is not None) and top_k <= 0:
            raise ValueError("Top-k cannot be 0 or float")
        if (top_p is not None) and not (0.0 < top_p < 1.0):
            raise ValueError("Top-p must be value between 0 and 1")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.banned_tokens = banned_tokens
        # elif isinstance(banned_tokens, int):        # single int
        #     self.banned_tokens = {banned_tokens}
        # else:                                       # handle list/tuple/set
        #     self.banned_tokens = set(banned_tokens)

    def __call__(self, logits: Tensor):
        # apply order is temperature -> top_k -> softmax -> top_p
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
        self.model.eval() # evaluation mode
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
        threshold: float = 0.3
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
        gen_annealing: str | None = None
        ) -> str:
        
        if keep is None: # if has no moving window just generate the max seq_len tokens
            self.max_tokens = self.model_cfg.seq_len
        else:
            assert keep <= self.model_cfg.seq_len  # cannot keep more than the seq_len length
        print(f"keep={keep}, max_tokens={self.max_tokens}")
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # start with the prompt
        total_ids = input_ids
        generated = prompt_ids
        kv_cache = KVCache(
            self.model_cfg.n_layers,
            self.model_cfg.n_heads,
            self.model_cfg.seq_len,
            self.model_cfg.embed_dim // self.model_cfg.n_heads,
            self.gen_cfg.device
        ) if caching else None
        
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(self.max_tokens):
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
                            self.gen_cfg.device
                        )  
                        total_ids = total_ids[:, -keep:]      # window (keep tokens)
                        input_ids = total_ids                 # next loop call pre-fills the window
                        offset = 0                   # = keep    
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)  # [1, S+1]
                    if input_ids.shape[1] >= self.model_cfg.seq_len: # if about to exceed seq_len
                        input_ids = input_ids[:, -keep:]
                generated.append(int(next_token[-1].item()))           # store token id as int
                
                # temperature annealing check every step
                if gen_annealing is not None:
                    self.gen_annealing(total_ids, annealing=gen_annealing)
            
            full_text = self.tokenizer.decode(generated)
            t1 = time.perf_counter()
            print(f"Generation time: {t1-t0:.3f}s\n")
            print(full_text)
            
            # if saving to JSON
            if to_json:
                self.save_output({
                    "preset": self._current_preset,        # optional: set before calling
                    "temperature": self.sampler.temperature,
                    "top_k": self.sampler.top_k,
                    "top_p": self.sampler.top_p,
                    "keep": keep,
                    "max_tokens": self.max_tokens,
                    "time_s": round(t1 - t0, 3),
                    "prompt": prompt,
                    "output": full_text,
                })
            
        return full_text

def main():
    preset = config.PRESETS
    model_cfg = config.ModelConfig(**preset["mini-90M"]["model"])
    gen_presets = config.GEN_PRESETS 
    
if __name__ == "__main__":
    preset = config.PRESETS
    model_cfg = config.ModelConfig(**preset["mini-90M"]["model"])
    gen_presets = config.GEN_PRESETS 
    
    # trying step 3500, 6500, 8000, 13000 model
    checkpoints = [
        "checkpoints/mini/step_3500.pt",
        "checkpoints/mini/step_8000.pt",
        "checkpoints/mini/step_13000.pt",
        "checkpoints/mini/step_15000.pt",
    ]
    # also try different starting prompt
    prompts = [
        "Let's tell a stories about a researcher working on these issues.",
        "The three main causes of climate change are:",
        "What is the capital of France and why does it matter historically?",
        "She opened the letter and immediately knew that the news was",
    ]
    
    # for model in checkpoints:
    #     gen_cfg = config.GeneratorConfig(ckpt_path=model)
    #     text_gen = Generator(model_cfg, gen_cfg)
    #     for gen_type, preset in gen_presets.items():
    #         text_gen._current_preset = gen_type
    #         text_gen.sampler.temperature = preset["gen"]["temperature"]
    #         text_gen.sampler.top_k = preset["gen"]["top_k"]
    #         text_gen.sampler.top_p = preset["gen"]["top_p"]
    #         for prompt in prompts:
    #             print(f"\n Using {gen_type}, with KV cache, keep seq_len/2\n")
    #             print(f"Temperature: {text_gen.sampler.temperature}, Top_k: {text_gen.sampler.top_k}, Top_p: {text_gen.sampler.top_p}\n")
    #             print(f"Starting prompt: {prompt}")
    #             utils.seed_everything(gen_cfg.seed)
    #             text_gen.generate(
    #                 prompt, 
    #                 caching=True, 
    #                 keep=model_cfg.seq_len//2, 
    #                 to_json=True,
    #                 temp_annealing=True
    #             )
                
    # short test 
    model_cfg = config.from_json(config.ModelConfig, "configs/model.json")
    gen_cfg = config.from_json(config.GeneratorConfig, "configs/gen.json")

    text_gen = Generator(model_cfg, gen_cfg)
    prompt = "There was a time in 1920 in Paris where half of the population"
    anneal_list = ["temp"]
    
    for anneal in anneal_list:
        utils.seed_everything(gen_cfg.seed)
        # text_gen.sampler.temperature = 0.95
        # text_gen.sampler.top_k = 100
        # text_gen.sampler.top_p = 0.9
        # text_gen.max_tokens = 10240
        # text_gen.sampler.banned_tokens = 50256
        text_gen.generate(
            prompt,
            caching=True, 
            keep=int(model_cfg.seq_len // 2), 
            to_json=False,
            gen_annealing=anneal
        )
        # utils.seed_everything(gen_cfg.seed)
        # text_gen.generate(
        #     prompt,
        #     caching=True, 
        #     keep=model_cfg.seq_len//2, 
        #     to_json=False,
        #     gen_annealing="temp"
        # )
        # utils.seed_everything(gen_cfg.seed)
        # text_gen.generate(
        #     prompt,
        #     caching=True, 
        #     keep=model_cfg.seq_len//2, 
        #     to_json=False,
        #     gen_annealing="top_k"
        # )
        # utils.seed_everything(gen_cfg.seed)
        # text_gen.generate(
        #     prompt,
        #     caching=True, 
        #     keep=model_cfg.seq_len//2, 
        #     to_json=False,
        #     gen_annealing="top_p"
        # )

    # utils.pretty_view("samples\generations_step_3500.jsonl")