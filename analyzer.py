"""
analyze.py - read generations*.jsonl, compute summary metrics, print a comparison table.

Usage:
    python analyze.py                          # all samples/*.jsonl
    python analyze.py samples/generations_step_6500.jsonl
"""
import json
import sys
from pathlib import Path


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that repeat. 0 = fully novel, ~1 = degenerate loop."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    seen = set()
    repeats = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats / (len(tokens) - n + 1)

def unique_ratio(text: str) -> float:
    """Fraction of unique tokens. 1.0 = all different, low = repetitive."""
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def avg_token_len(text: str) -> float:
    tokens = text.split()
    return sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0

def summarize(entry: dict) -> dict:
    text = entry.get("output", "")
    return {
        "rep4": repetition_rate(text, 4),
        "rep8": repetition_rate(text, 8),   # stricter: whole-phrase loops
        "unique": unique_ratio(text),
        "avg_tok_len": avg_token_len(text),
        "n_chars": len(text),
    }

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def print_table(runs: list[dict]):
    print(f"{'preset':<12} {'ckpt':<15} {'rep4':>6} {'rep8':>6} {'unique':>7} {'chars':>7} {'time_s':>7}")
    print("-" * 75)
    for r in sorted(runs, key=lambda x: (x.get("rep4", 0.0), -x.get("unique", 0.0))):
        m = summarize(r)
        print(
            f"{str(r.get('preset')):<12} "
            f"{str(r.get('ckpt', '')):<15} "
            f"{m['rep4']:>6.3f} {m['rep8']:>6.3f} {m['unique']:>7.3f} "
            f"{m['n_chars']:>7} {r.get('time_s', 0):>7.1f}"
        )

def main():
    args = sys.argv[1:]
    if not args:
        args = sorted(str(p) for p in Path("samples").glob("generations_*.jsonl"))
    paths = args

    for path in paths:
        runs = load_jsonl(path)
        if not runs:
            print(f"{path}: empty")
            continue
        print(f"\n=== {path} ({len(runs)} runs) ===")
        print_table(runs)

        # per-prompt breakdown, if multiple prompts were used
        # per-prompt x per-preset breakdown
        prompts = {r.get("prompt") for r in runs}
        presets = sorted({r.get("preset") for r in runs})
        if len(prompts) > 1:
            print("\n  per-prompt x preset (rep4 / unique / n):")
            label_w = 45
            col_w = 14
            print("    " + " " * label_w + "".join(f"{p:<{col_w}}" for p in presets))
            for p in sorted(prompts):
                label = p[:label_w]
                cells = []
                for pr in presets:
                    subset = [r for r in runs if r.get("prompt") == p and r.get("preset") == pr]
                    if not subset:
                        cells.append(f"{'-':<{col_w}}")
                        continue
                    avg_rep = sum(summarize(r)["rep4"] for r in subset) / len(subset)
                    avg_uniq = sum(summarize(r)["unique"] for r in subset) / len(subset)
                    cells.append(f"{avg_rep:.2f}/{avg_uniq:.2f}/{len(subset):<{col_w}}")
                print(f"    {label:<{label_w}}" + "".join(cells))


if __name__ == "__main__":
    main()