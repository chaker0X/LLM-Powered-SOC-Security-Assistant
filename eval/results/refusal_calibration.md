# Refusal gate calibration

48 in-scope and 12 out-of-scope questions. Score = best dense cosine over the whole corpus.

| Embedder | Rule | Threshold | In-scope answered | Out-of-scope refused |
|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | **chosen**: answer every in-scope question | 0.600 | 100% | 83% |
| BAAI/bge-small-en-v1.5 | best balanced accuracy | 0.667 | 96% | 100% |
| ibm-granite/granite-embedding-small-english-r2 | **chosen**: answer every in-scope question | 0.820 | 100% | 92% |
| ibm-granite/granite-embedding-small-english-r2 | best balanced accuracy | 0.842 | 94% | 100% |

Put the chosen values in `config.yaml` under `retrieval.min_score`:

```yaml
min_score:
  BAAI/bge-small-en-v1.5: 0.60
  ibm-granite/granite-embedding-small-english-r2: 0.82
```

Caveat: only a few out-of-scope questions, so these thresholds are rough. Re-run after adding questions to eval/retrieval_questions.jsonl.
