"""Compare retrieval setups on the labelled question set.

Run:  python eval/retrieval_eval.py           (or `make eval-retrieval`)

Metrics (computed on documents, after removing duplicate chunks of the same document):
  hit@k  share of questions with at least one correct document in the top k
  MRR    mean of 1/rank of the first correct document (0 if none in the top 10)
chunk_hit@4  share of questions where one of the first 4 *chunks* (exactly what the LLM
             receives with top_k = 4) comes from a correct document. This is the number that
             matters for answers; the document-level metrics are robust to re-chunking.

Out-of-scope questions have no correct document. For them we record the best dense
cosine score, to calibrate the refusal threshold (week 2).

Caveat for the report: the questions were written by the project author, which can bias
results; they were phrased without copying document wording to limit this.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soc_assistant.retrieval.hybrid import Retriever, load_chunks  # noqa: E402

QUESTIONS = ROOT / "eval" / "retrieval_questions.jsonl"
RESULTS = ROOT / "eval" / "results"
BGE = "BAAI/bge-small-en-v1.5"
GRANITE = "ibm-granite/granite-embedding-small-english-r2"
CONFIGS = [  # (label, mode, embedding model)
    ("BM25", "bm25", BGE),
    ("Dense bge-small", "dense", BGE),
    ("Dense granite-small-r2", "dense", GRANITE),
    ("Hybrid BM25+bge", "hybrid", BGE),
    ("Hybrid BM25+granite", "hybrid", GRANITE),
]
KS = (1, 4, 10)


def ranked_docs(hits) -> list[str]:
    seen, out = set(), []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.add(h.chunk.doc_id)
            out.append(h.chunk.doc_id)
    return out


def evaluate(retriever: Retriever, mode: str, questions: list[dict]) -> dict:
    per_q, by_type = [], defaultdict(list)
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=30, mode=mode)
        latencies.append(time.perf_counter() - t0)
        docs = ranked_docs(hits)[:10]
        gold = set(q["gold"])
        rank = next((i for i, d in enumerate(docs, start=1) if d in gold), None)
        chunk_rank = next((i for i, h in enumerate(hits, start=1) if h.chunk.doc_id in gold), None)
        row = {"id": q["id"], "type": q["type"], "rank": rank, "chunk_rank": chunk_rank,
               "top3": docs[:3]}
        per_q.append(row)
        by_type[q["type"]].append(row)

    def summary(rows):
        n = len(rows)
        out = {f"hit@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n for k in KS}
        out["mrr"] = sum(1 / r["rank"] for r in rows if r["rank"]) / n
        out["chunk_hit@4"] = sum(1 for r in rows if r["chunk_rank"] and r["chunk_rank"] <= 4) / n
        out["n"] = n
        return out

    return {
        "overall": summary(per_q),
        "by_type": {t: summary(rows) for t, rows in sorted(by_type.items())},
        "latency_ms": 1000 * statistics.median(latencies),
        "per_question": per_q,
    }


def top_dense_scores(retriever: Retriever, questions: list[dict]) -> list[float]:
    return [retriever.dense.search(q["question"], 1)[0][1] for q in questions]


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    in_scope = [q for q in questions if q["gold"]]
    oos = [q for q in questions if not q["gold"]]
    chunks = load_chunks()
    retrievers = {m: Retriever(m, chunks) for m in (BGE, GRANITE)}
    for r in retrievers.values():      # warm-up: first query loads the model weights
        r.search("warm up", k=1, mode="hybrid")

    results = {}
    for label, mode, model in CONFIGS:
        results[label] = evaluate(retrievers[model], mode, in_scope)
        o = results[label]["overall"]
        print(f"{label:24s} hit@1 {o['hit@1']:.2f}  hit@4 {o['hit@4']:.2f}  hit@10 {o['hit@10']:.2f}  "
              f"MRR {o['mrr']:.3f}  chunk_hit@4 {o['chunk_hit@4']:.2f}  "
              f"({results[label]['latency_ms']:.0f} ms/query)")

    calib = {}
    for model, r in retrievers.items():
        ins, out = top_dense_scores(r, in_scope), top_dense_scores(r, oos)
        calib[model] = {"in_scope": ins, "out_of_scope": out,
                        "out_of_scope_ids": [q["id"] for q in oos]}

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_eval.json").write_text(json.dumps(
        {"n_in_scope": len(in_scope), "n_out_of_scope": len(oos), "configs": results,
         "dense_score_calibration": calib}, indent=2))
    write_markdown(results, calib, len(in_scope), len(oos), {q["id"]: q for q in questions})
    print(f"\nSaved {RESULTS / 'retrieval_eval.md'} and retrieval_eval.json")


def write_markdown(results: dict, calib: dict, n_in: int, n_oos: int, qmap: dict) -> None:
    lines = [
        "# Retrieval evaluation",
        "",
        f"{n_in} in-scope questions (gold = correct documents), {n_oos} out-of-scope questions.",
        "hit@k and MRR are per document (duplicate chunks of one document removed). "
        "**chunk hit@4** counts the first 4 chunks, exactly what the LLM receives (top_k = 4).",
        "",
        "| Setup | chunk hit@4 | hit@1 | hit@4 | hit@10 | MRR | ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        o = r["overall"]
        lines.append(f"| {label} | **{o['chunk_hit@4']:.2f}** | {o['hit@1']:.2f} | {o['hit@4']:.2f} | {o['hit@10']:.2f} | "
                     f"{o['mrr']:.3f} | {r['latency_ms']:.0f} |")
    types = list(next(iter(results.values()))["by_type"])
    lines += ["", "## chunk hit@4 by question type", "",
              "| Setup | " + " | ".join(f"{t} (n={results[next(iter(results))]['by_type'][t]['n']})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
    for label, r in results.items():
        lines.append(f"| {label} | " + " | ".join(f"{r['by_type'][t]['chunk_hit@4']:.2f}" for t in types) + " |")
    lines += ["", "## Best dense cosine score (for the refusal threshold)", "",
              "| Embedder | in-scope min / median | out-of-scope median / max |", "|---|---|---|"]
    for model, c in calib.items():
        i, o = c["in_scope"], c["out_of_scope"]
        lines.append(f"| {model} | {min(i):.3f} / {statistics.median(i):.3f} | "
                     f"{statistics.median(o):.3f} / {max(o):.3f} |")
    best = max(results, key=lambda l: (results[l]["overall"]["chunk_hit@4"], results[l]["overall"]["mrr"]))
    lines += ["", f"## Misses of the best setup ({best}): no correct chunk among the 4 sent to the LLM", "",
              "| id | question | first correct chunk | first correct doc | retrieved top 3 docs |",
              "|---|---|---|---|---|"]
    for row in results[best]["per_question"]:
        if not row["chunk_rank"] or row["chunk_rank"] > 4:
            lines.append(f"| {row['id']} | {qmap[row['id']]['question']} | {row['chunk_rank'] or '>30'} | "
                         f"{row['rank'] or '>10'} | "
                         f"{', '.join(row['top3'])} |")
    (RESULTS / "retrieval_eval.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
