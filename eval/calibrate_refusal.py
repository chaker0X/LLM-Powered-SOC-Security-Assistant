"""Pick the refusal-gate threshold from the saved retrieval evaluation.

Run:  python eval/calibrate_refusal.py        (after `make eval-retrieval`)

No index or model is needed: it reads the best dense cosine score of every question,
saved by eval/retrieval_eval.py in eval/results/retrieval_eval.json.

Rule: keep EVERY in-scope question answered (a wrong refusal hides real knowledge from
the analyst), and refuse as many out-of-scope questions as possible under that limit:
    threshold = lowest in-scope score - MARGIN, rounded down to 0.01
Out-of-scope questions that still pass the gate are caught by the LLM's own refusal rule
(system prompt rule 1), the second safety net. The table also shows the threshold with
the best balanced accuracy, to discuss the trade-off in the report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
MARGIN = 0.005


def rates(threshold: float, in_scope: list[float], oos: list[float]) -> tuple[float, float]:
    """(share of in-scope questions answered, share of out-of-scope questions refused)."""
    answered = sum(s >= threshold for s in in_scope) / len(in_scope)
    refused = sum(s < threshold for s in oos) / len(oos)
    return answered, refused


def recall_first_threshold(in_scope: list[float], margin: float = MARGIN) -> float:
    return math.floor((min(in_scope) - margin) * 100) / 100


def best_balanced_threshold(in_scope: list[float], oos: list[float]) -> float:
    candidates = sorted(set(in_scope + oos))
    # Ties: prefer the lower threshold (answers more in-scope questions).
    return max(candidates, key=lambda t: (sum(rates(t, in_scope, oos)), -t))


def main() -> None:
    data = json.loads((RESULTS / "retrieval_eval.json").read_text())
    calib = data["dense_score_calibration"]
    lines = [
        "# Refusal gate calibration",
        "",
        f"{data['n_in_scope']} in-scope and {data['n_out_of_scope']} out-of-scope questions. "
        "Score = best dense cosine over the whole corpus.",
        "",
        "| Embedder | Rule | Threshold | In-scope answered | Out-of-scope refused |",
        "|---|---|---|---|---|",
    ]
    chosen = {}
    for model, c in calib.items():
        ins, oos = c["in_scope"], c["out_of_scope"]
        t_rec = recall_first_threshold(ins)
        t_bal = best_balanced_threshold(ins, oos)
        chosen[model] = t_rec
        for rule, t in (("**chosen**: answer every in-scope question", t_rec),
                        ("best balanced accuracy", t_bal)):
            a, r = rates(t, ins, oos)
            lines.append(f"| {model} | {rule} | {t:.3f} | {a:.0%} | {r:.0%} |")
    lines += ["", "Put the chosen values in `config.yaml` under `retrieval.min_score`:", "", "```yaml",
              "min_score:"] + [f"  {m}: {t:.2f}" for m, t in chosen.items()] + ["```", ""]
    lines += ["Caveat: only a few out-of-scope questions, so these thresholds are rough. "
              "Re-run after adding questions to eval/retrieval_questions.jsonl."]
    out = RESULTS / "refusal_calibration.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
