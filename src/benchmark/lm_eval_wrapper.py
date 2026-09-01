import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import GPT2TokenizerFast
import config
from model import MiniGPT


@register_model("minigpt")
class MiniGPT_LM(LM):
    eot_token_id = 50256
    eot_token = "<|endoftext|>"
    max_gen_toks = 512

    def __init__(self, ckpt_path, device="cuda", batch_size=1):
        super().__init__()
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        cfg = config.from_json(config.ModelConfig, "configs/model.json")
        self._model = MiniGPT(cfg).to(device)
        self._model.load_state_dict(
            {k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in state["model"].items()}, strict=True)
        self._model.eval()
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # r50k, harness-compatible
        self._device, self._batch_size = device, batch_size

    @property
    def device(self):     return self._device
    @property
    def batch_size(self): return self._batch_size
    @property
    def max_length(self): return self._model.model_cfg.seq_len if hasattr(self._model, "model_cfg") else 1024
    @property
    def vocab_size(self): return 50257
    @property
    def tokenizer(self):  return self._tokenizer

    @torch.no_grad()
    def _model_call(self, inps):
        """inps: [batch, seq] ids -> raw logits [batch, seq, vocab] (harness does the shift)."""
        return self._model(inps)

    @torch.no_grad()
    def _model_generate(self, context, max_length, stop=None, **gen_kwargs):
        out = context
        for _ in range(max_length - context.shape[1]):
            logits = self._model(out)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=-1)
            if (nxt == self.eot_token_id).all():
                break
        return out
    
    @torch.no_grad()
    def _logits(self, ids):
        """[B, S] ids -> raw logits [B, S, vocab]."""
        return self._model(ids)

    def loglikelihood(self, requests):
        """requests: iterable of Instance(args=(context, continuation)) -> [(cont_logprob, is_greedy), ...]"""
        res = []
        for request in requests:
            context, continuation = request.args
            if len(continuation) == 0:
                res.append((0.0, True))
                continue
            ctx = self.tokenizer.encode(context)
            cont = self.tokenizer.encode(continuation)
            if len(ctx) + len(cont) > self.max_length:       # truncate context from the left
                ctx = ctx[-(self.max_length - len(cont)):]
            ids = torch.tensor([ctx + cont], dtype=torch.long, device=self.device)
            logits = self._logits(ids)
            logp = F.log_softmax(logits.float(), dim=-1)
            idx = torch.arange(len(cont), device=self.device) + (len(ctx) - 1)  # prediction positions
            cont_toks = torch.tensor(cont, dtype=torch.long, device=self.device)
            tok_logp = logp[0, idx, cont_toks]
            is_greedy = bool((logits[0, idx].argmax(-1) == cont_toks).all().item())
            res.append((tok_logp.sum().item(), is_greedy))
        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError   # only needed for perplexity

    def generate_until(self, requests):
        raise NotImplementedError   # only needed for generative tasks 