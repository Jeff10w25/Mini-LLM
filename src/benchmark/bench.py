import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from lm_eval import simple_evaluate
from lm_eval_wrapper import MiniGPT_LM  
from lm_eval.tasks import TaskManager


def main():
    CKPT = "checkpoints/mini/step_15000.pt"
    # tasks where a 90M is competitive; limit=50 = smoke test, remove for full
    results = simple_evaluate(
        model=MiniGPT_LM(CKPT, device="cpu", batch_size=1),
        tasks=["lambada_openai", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"],
        num_fewshot=0,
        limit=None,                    # delete this line for the real run
        batch_size=1,
        verbosity=1,
        bootstrap_iters=0,
        task_manager=TaskManager(include_path="configs/"),
    )

    def first(*vals):            # first non-None value (keeps a legit 0.0)
        for v in vals:
            if v is not None:
                return v
        return None

    for task, r in results["results"].items():
        acc = first(r.get("acc,none"), r.get("acc"),
                    r.get("acc_norm,none"), r.get("acc_norm"))
        line = f"{task:16s}"
        if acc is not None:
            line += f" acc: {float(acc):.3f}"
        else:
            line += f"  (no acc metric; keys={sorted(r.keys())})"
        print(line)

if __name__ == "__main__":
    main()