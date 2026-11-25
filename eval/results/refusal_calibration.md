# Refusal gate calibration

48 in-scope and 12 out-of-scope questions in the saved evaluation. Score = best dense cosine over the whole corpus.

| Embedder | Rule | Threshold | In-scope answered | In-scope answered, leave-one-out | Easy out-of-scope refused | Hard out-of-scope refused |
|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | **chosen**: answer every in-scope question | 0.600 | 100% | 98% | 83% (n=12) | n/a |
| BAAI/bge-small-en-v1.5 | best balanced accuracy | 0.667 | 96% | – | 100% (n=12) | n/a |
| ibm-granite/granite-embedding-small-english-r2 | **chosen**: answer every in-scope question | 0.820 | 100% | 100% | 92% (n=12) | n/a |
| ibm-granite/granite-embedding-small-english-r2 | best balanced accuracy | 0.842 | 94% | – | 100% (n=12) | n/a |

Safety margin of the chosen threshold (lowest in-scope score minus threshold):

- BAAI/bge-small-en-v1.5: 0.009
- ibm-granite/granite-embedding-small-english-r2: 0.010

Put the chosen values in `config.yaml` under `retrieval.min_score`:

```yaml
min_score:
  BAAI/bge-small-en-v1.5: 0.60
  ibm-granite/granite-embedding-small-english-r2: 0.82
```

**The saved evaluation predates the 18 hard out-of-scope questions.** Re-run `python eval/retrieval_eval.py`, then this script, to fill the last column.

Caveats: thresholds come from a small, author-written question set. The leave-one-out column is the better estimate for new questions. Technique-ID questions do not use this gate (exact-match shortcut in `rag.py`).
