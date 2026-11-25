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

Two honesty checks:
  leave-one-out  the chosen threshold answers 100% of the in-scope questions by
                 construction, because it is fitted on them. Re-fitting it without each
                 question in turn estimates how often a NEW in-scope question is answered.
  hard questions out-of-scope questions marked "hard" sound like security but the corpus
                 cannot answer them (live CVE data, company-internal facts). A cosine gate
                 is expected to let many through; they are reported apart from the easy ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
QUESTIONS = ROOT / "eval" / "retrieval_questions.jsonl"
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


def leave_one_out_answered(in_scope: list[float], margin: float = MARGIN) -> float:
    """Share of in-scope questions answered by a threshold fitted without them."""
    kept = sum(s >= recall_first_threshold(in_scope[:i] + in_scope[i + 1:], margin)
               for i, s in enumerate(in_scope))
    return kept / len(in_scope)


def split_easy_hard(scores: list[float], ids: list[str], hard_ids: set[str]) -> tuple[list[float], list[float]]:
    easy = [s for s, i in zip(scores, ids) if i not in hard_ids]
    hard = [s for s, i in zip(scores, ids) if i in hard_ids]
    return easy, hard


def refused_share(threshold: float, scores: list[float]) -> str:
    return f"{sum(s < threshold for s in scores) / len(scores):.0%} (n={len(scores)})" if scores else "n/a"


def main() -> None:
    data = json.loads((RESULTS / "retrieval_eval.json").read_text())
    calib = data["dense_score_calibration"]
    questions = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    hard_ids = {q["id"] for q in questions if q.get("hard")}
    stale = any("out_of_scope_ids" not in c for c in calib.values())

    lines = [
        "# Refusal gate calibration",
        "",
        f"{data['n_in_scope']} in-scope and {data['n_out_of_scope']} out-of-scope questions in the saved "
        "evaluation. Score = best dense cosine over the whole corpus.",
        "",
        "| Embedder | Rule | Threshold | In-scope answered | In-scope answered, leave-one-out "
        "| Easy out-of-scope refused | Hard out-of-scope refused |",
        "|---|---|---|---|---|---|---|",
    ]
    chosen = {}
    for model, c in calib.items():
        ins, oos = c["in_scope"], c["out_of_scope"]
        easy, hard = (oos, []) if stale else split_easy_hard(oos, c["out_of_scope_ids"], hard_ids)
        t_rec = recall_first_threshold(ins)
        t_bal = best_balanced_threshold(ins, oos)
        chosen[model] = t_rec
        for rule, t, loo in (("**chosen**: answer every in-scope question", t_rec,
                              f"{leave_one_out_answered(ins):.0%}"),
                             ("best balanced accuracy", t_bal, "–")):
            lines.append(f"| {model} | {rule} | {t:.3f} | {rates(t, ins, oos)[0]:.0%} | {loo} "
                         f"| {refused_share(t, easy)} | {refused_share(t, hard)} |")

    lines += ["", "Safety margin of the chosen threshold (lowest in-scope score minus threshold):", ""]
    lines += [f"- {m}: {min(c['in_scope']) - chosen[m]:.3f}" for m, c in calib.items()]
    lines += ["", "Put the chosen values in `config.yaml` under `retrieval.min_score`:", "", "```yaml",
              "min_score:"] + [f"  {m}: {t:.2f}" for m, t in chosen.items()] + ["```", ""]
    if stale:
        lines += [f"**The saved evaluation predates the {len(hard_ids)} hard out-of-scope questions.** "
                  "Re-run `python eval/retrieval_eval.py`, then this script, to fill the last column.", ""]
    lines += ["Caveats: thresholds come from a small, author-written question set. The leave-one-out "
              "column is the better estimate for new questions. Technique-ID questions do not use this "
              "gate (exact-match shortcut in `rag.py`)."]
    out = RESULTS / "refusal_calibration.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
