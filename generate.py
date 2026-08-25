from __future__ import annotations
from typing import TYPE_CHECKING

import time
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import config, utils
from model import MiniGPT, KVCache

if TYPE_CHECKING:
    # only read by your IDE/Type Checker, completely ignored at runtime
    from torch import Tensor, nn
    from config import ModelConfig, GeneratorConfig

def generate_text(sampler, input_text, model, tokenizer, n_token=100, print_in_loops=False):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.inference_mode():
        for _ in range(n_token):
            logit = model(input_ids).logits
            next_token = sampler(logit[:, -1, :])
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if print_in_loops:
                next_word = tokenizer.batch_decode(
                    next_token,                  
                    skip_special_tokens=True,      
                    clean_up_tokenization_spaces=True
                )
                print(next_word[0], end="")
        out_text = tokenizer.batch_decode(
            input_ids,                  
            skip_special_tokens=True,      
            clean_up_tokenization_spaces=True
        )
    return out_text

class TextGenSampler:
    """Sampler 
    """
    def __init__(
        self, 
        temperature: float = 1.0, 
        top_k: int | None = None,
        top_p: float | None = None
        ): 
        
        if (top_k is not None) and top_k <= 0:
            raise ValueError("Top-k cannot be 0 or float")
        if (top_p is not None) and not (0.0 < top_p < 1.0):
            raise ValueError("Top-p must be value between 0 and 1")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def __call__(self, logits):
        # apply order is temperature -> top_k -> softmax -> top_p
        if self.temperature <= 0:
            # greedy sampling, ignore top_p and top_k and select highest logit's token
            return logits.argmax(dim=-1, keepdim=True)
        
        logits = self._apply_temperature(logits)
        if self.top_k is not None:
            logits = self._apply_top_k(logits)
            
        probs = self._apply_softmax(logits)
        if self.top_p is not None:
            probs = self._apply_top_p(probs)

        return torch.multinomial(probs, 1)

    def _apply_softmax(self, logits):
        return F.softmax(logits, dim=-1)
    
    def _apply_temperature(self, logits):
        if self.temperature > 0:  # just to prevent dividing with 0
            return logits / self.temperature

    def _apply_top_k(self, logits):
        """Take top_k val/idx, return mask with all value except top_k val/idx negative infinity"""
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        mask = torch.full_like(logits, -torch.inf)
        mask.scatter_(index=top_k_idx, src=top_k_vals, dim=-1)
        return mask
    
    def _apply_top_p(self, probs):
        """Take top_p val/idx, return top_p val/idx with normalized probabilities. Other idx prob is 0"""
        if 0.0 < self.top_p < 1.0:
            # sort probs in descending order and do cumulative sum
            probs_val, probs_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(probs_val, dim=-1)
            # take the first element idx that cumsum >= top_p e.g. top_p=0.7, take 0.72
            cut = (cumsum >= self.top_p).nonzero()[0, -1]
            top_p_idx = probs_idx[:, :cut]
            # make a mask size of probs and scatter the probs_val value at top_p_idx
            mask = torch.full_like(probs, 0.)
            mask.scatter_(index=top_p_idx, src=probs_val, dim=-1)
            # renormalize by dividing the cutting spot value so that it sums to 1
            mask = mask / cumsum[:, cut]
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
        self.sampler = TextGenSampler(gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p)
        self.tokenizer = tiktoken.get_encoding("r50k_base")
        self.model_cfg = model_cfg
        self.gen_cfg = gen_cfg
        
    def _raw_model(self):
        model = self.model
        model = getattr(model, "_orig_mod", model)   # strip torch.compile (outer layer)
        model = getattr(model, "module", model)      # strip DDP (inner layer), if present
        model = model.to(self.device)
        return model
    
    def generate(self, prompt: str, print_in_loops: bool = False, caching: bool = True) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # start with the prompt
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
                    input_ids = next_token          # -> [1, 1]
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)  # [1, S+1]

                generated.append(int(next_token[-1].item()))           # store token id as int
            full_text = self.tokenizer.decode(generated)
            # print(generated)
            print(full_text)
            t1 = time.perf_counter()
            print(f"Generation time: {t1-t0:.3f}s")
        return full_text

    
if __name__ == "__main__":
    preset = config.PRESETS
    model_cfg = config.ModelConfig(**preset["mini-90M"]["model"])
    gen_cfg = config.GeneratorConfig()
    train_cfg = config.TrainConfig()
    text_gen = Generator(model_cfg, gen_cfg)
    prompt = "It is a sunny day today right"
    
    utils.seed_everything(train_cfg.seed)
    print("\nNo KV cache\n")
    text_gen.generate(prompt, print_in_loops=True, caching=False)
    
    utils.seed_everything(train_cfg.seed)
    print("\nWith KV cache\n")
    text_gen.generate(prompt, print_in_loops=True, caching=True)
    
    # gen_model = model.MiniGPT(cfg)
    