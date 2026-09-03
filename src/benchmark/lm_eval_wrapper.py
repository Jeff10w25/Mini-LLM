from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import utils 
from transformers import GPT2TokenizerFast
import config
from model import MiniGPT

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance
    from torch import Tensor


class MiniGPT_LM(LM):
    eot_token_id = 50256
    eot_token = "<|endoftext|>"
    max_gen_toks = 512

    def __init__(
        self, 
        ckpt_path: str, 
        device: str = "cpu", 
        batch_size: int = 1
        ):
        
        super().__init__()
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = config.from_json(config.ModelConfig, "configs/model.json")
        self._model = MiniGPT(cfg).to(device)
        self._model.load_state_dict(
            {k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in state["model"].items()}, strict=True)
        self._model.eval()
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # r50k, harness-compatible
        self._device = device
        self._batch_size = batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._model.model_cfg.seq_len if hasattr(self._model, "model_cfg") else 1024

    @property
    def vocab_size(self) -> int:
        return 50257

    @property
    def tokenizer(self) -> GPT2TokenizerFast:
        return self._tokenizer

    @torch.no_grad()
    def _model_call(self, inps: Tensor) -> Tensor:
        """inps: [batch, seq] ids -> raw logits [batch, seq, vocab] (harness does the shift)."""
        return self._model(inps)

    @torch.no_grad()
    def _model_generate(
        self,
        context: Tensor,
        max_length: int,
    ) -> Tensor:
        out = context
        for _ in range(max_length - context.shape[1]):
            logits = self._model(out)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=-1)
            if (nxt == self.eot_token_id).all():
                break
        return out

    @torch.no_grad()
    def _logits(self, ids: Tensor) -> Tensor:
        """[B, S] ids -> raw logits [B, S, vocab]."""
        return self._model(ids)

    def loglikelihood(
        self, 
        requests: Iterable[Instance]
    ) -> list[tuple[float, bool]]:
        """requests: iterable of Instance(args=(context, continuation)) -> [(cont_logprob, is_greedy), ...].
        Batches requests into forward passes of self._batch_size via right-padding, so the
        --batch_size flag is actually used (one forward per batch, not per request)."""
        reqs = list(requests)
        res = [None] * len(reqs)
        pad = 0   # pad id; right-padding can't leak into causal real-token logprobs
        work = []  # (index_in_res, ctx_tokens, cont_tokens)
        for i, request in enumerate(reqs):
            context, continuation = request.args
            ctx_toks = self.tokenizer.encode(context)
            cont_toks = self.tokenizer.encode(continuation)
            if len(cont_toks) == 0:
                res[i] = (0.0, True)
                continue
            if len(ctx_toks) + len(cont_toks) > self.max_length:  # truncate context from the left
                ctx_toks = ctx_toks[-(self.max_length - len(cont_toks)):]
            work.append((i, ctx_toks, cont_toks))

        bs = max(1, int(getattr(self, "_batch_size", 1) or 1))
        for start in tqdm(range(0, len(work), bs), desc="loglikelihood", unit="batch"):
            batch = work[start:start + bs]
            M = max(len(c) + len(k) for _, c, k in batch)   # pad all rows to the longest in the batch
            rows, meta = [], []                             # meta: (index_in_res, nctx, cont_toks)
            
            for i, ctx_toks, cont_toks in batch:
                rows.append(ctx_toks + cont_toks + [pad] * (M - len(ctx_toks) - len(cont_toks)))
                meta.append((i, len(ctx_toks), cont_toks))
            ids = torch.tensor(rows, dtype=torch.long, device=self.device)   # [B, M]
            logits = self._logits(ids)                                       # [B, M, vocab]
            logp = F.log_softmax(logits.float(), dim=-1)
            
            for j, (i, nctx, cont_toks) in enumerate(meta):
                ncont = len(cont_toks)
                if ncont == 0:
                    res[i] = (0.0, True)
                    continue
                pred = torch.arange(ncont, device=self.device) + (nctx - 1)   # positions predicting cont
                cont = torch.tensor(cont_toks, dtype=torch.long, device=self.device)
                tok_logp = logp[j, pred, cont]
                is_greedy = bool((logits[j, pred].argmax(-1) == cont).all().item())
                res[i] = (tok_logp.sum().item(), is_greedy)
        return res

    def loglikelihood_rolling(
        self, 
        requests: Iterable[Instance], 
        disable_tqdm: bool = False
    ) -> list[float]:
        """Rolling (full-context) log-likelihood per document, for perplexity tasks
        (e.g. wikitext). Returns one summed log-prob per request, as a list of floats.

        Mirrors lm_eval's HFLM algorithm exactly so the resulting perplexity is
        directly comparable to the GPT-2 reference: each request carries its text in
        `.args`, windows come from utils.get_rolling_token_windows + make_disjoint_window
        with the same prefix/eot token and max_seq_len, and only the trailing
        len(continuation) logits are scored.
        """
        max_len = int(self.max_length)
        strings = [req.args[0] for req in requests]   # each Instance.args == (text,)
        total_logprob = [0.0] * len(strings)

        # windows: (req_idx, inp_tokens, ncont, cont_tokens)
        windows = []
        for idx, string in enumerate(strings):
            toks = self.tokenizer.encode(string)
            if not toks:
                continue
            for ctx, cont in map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=toks,
                    prefix_token=self.eot_token_id,   # conditions the very first token
                    max_seq_len=max_len,
                    context_len=1,
                ),
            ):
                ncont = len(cont)
                if ncont == 0:
                    continue
                combined = ctx + cont
                # causal shift: drop the last combined token from the model input so we
                # predict each cont token from its left context; left-truncate to fit.
                inp = combined[-(max_len + 1):][:-1]
                windows.append((idx, inp, ncont, cont))

        bs = max(1, int(getattr(self, "_batch_size", 1) or 1))
        pad = 0
        for start in tqdm(
            range(0, len(windows), bs),
            desc="rolling ppl",
            unit="batch",
            disable=bool(disable_tqdm),
        ):
            batch = windows[start:start + bs]
            M = max(len(inp) for _, inp, _, _ in batch)
            rows, meta = [], []
            for idx, inp, ncont, cont in batch:
                rows.append(inp + [pad] * (M - len(inp)))
                meta.append((idx, len(inp), ncont, cont))
            ids = torch.tensor(rows, dtype=torch.long, device=self.device)
            logp = F.log_softmax(self._logits(ids).float(), dim=-1)
            for j, (idx, lin, ncont, cont) in enumerate(meta):
                # trailing ncont logits predict the continuation tokens
                pred = torch.arange(lin - ncont, lin, device=self.device)
                c = torch.tensor(cont, dtype=torch.long, device=self.device)
                total_logprob[idx] += float(logp[j, pred, c].sum().item())
        return total_logprob

    def generate_until(self, requests: Iterable[Instance]) -> list[str]:
        raise NotImplementedError   # only needed for generative tasks

# Register with lm_eval. Prevent importing this module more than once
try:
    register_model("minigpt")(MiniGPT_LM)
except ValueError:
    pass  
