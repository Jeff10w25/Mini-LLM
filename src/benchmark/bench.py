import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
from typing import Any
from lm_eval import simple_evaluate
from lm_eval_wrapper import MiniGPT_LM  
from lm_eval.tasks import TaskManager


def main():
    parser = argparse.ArgumentParser(description="lm-eval benchmark for MiniGPT")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="max requests per task (None = full)")
    parser.add_argument("--ckpt", default="checkpoints/mini/step_15000.pt")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--bootstrap_iters", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument(
        "--tasks", nargs="*",
        default=["lambada_openai", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"])
    args = parser.parse_args()

    results = simple_evaluate(
        model=MiniGPT_LM(args.ckpt, device=args.device, batch_size=args.batch_size),
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        verbosity=args.verbosity,
        bootstrap_iters=args.bootstrap_iters,
        task_manager=TaskManager(include_path="configs/"),
    )

    def first(*vals: Any) -> Any:     # first non-None value (keeps a legit 0.0)
        for v in vals:
            if v is not None:
                return v
        return None

    UNAVAILABLE = {"N/A", "", "NONE", "NAN"}
    for task, r in results["results"].items():
        acc   = first(r.get("acc,none"),  r.get("acc"))
        acc_n = first(r.get("acc_norm,none"), r.get("acc_norm"))
        err   = first(r.get("acc_stderr,none"), r.get("acc_stderr"))
        err_n = first(r.get("acc_norm_stderr,none"), r.get("acc_norm_stderr"))
        # perplexity-style metrics (lambada uses "perplexity", wikitext "word_perplexity")
        ppl_metrics = [
            (k, r[k]) for k in sorted(r)
            if "perplexity" in k and not k.endswith("_stderr") and not k.endswith("_stderr,none")
        ]
        line = f"{task:16s}"
        if acc_n is not None:
            line += f" acc_norm: {float(acc_n):.3f}"
            if err_n is not None and str(err_n).strip().upper() not in UNAVAILABLE:
                line += f" ± {float(err_n):.3f}"
        if acc is not None:
            line += f" acc: {float(acc):.3f}"
            if err is not None and str(err).strip().upper() not in UNAVAILABLE:
                line += f" ± {float(err):.3f}"
        for k, v in ppl_metrics:
            if v is None:
                continue
            perr = r.get(k + "_stderr")
            label = "ppl" if "word_" not in k else "word_ppl"
            line += f" {label}: {float(v):.3f}"
            if perr is not None and str(perr).strip().upper() not in UNAVAILABLE:
                line += f" ± {float(perr):.3f}"
        if acc is None and acc_n is None and not ppl_metrics:
            line += f"  (no acc/ppl; keys={sorted(r.keys())})"
        print(line)


if __name__ == "__main__":
    main()
    
    
