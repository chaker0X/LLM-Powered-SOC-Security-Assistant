# SOC Assistant — local LLM + RAG copilot for security analysts

> 🚧 Work in progress: academic project (AI applied to cybersecurity, P11).

A fully **local** conversational assistant that helps SOC analysts:

- **interpret security alerts**: incident summaries enriched with MITRE ATT&CK
- **find response procedures**: retrieval over public incident-response playbooks
- **trust the answers**: every claim cites its source, and the assistant refuses when the evidence is weak
- **resist prompt injection**: a trained classifier plus prompt isolation defend against malicious text hidden in alerts or documents

## Architecture

```
alert / question
      │
 [1] injection guard (fine-tuned classifier)
 [2] hybrid retrieval (BM25 + dense, Chroma)  ← ATT&CK · playbooks · alerts
 [3] refusal gate (low retrieval score → "not enough evidence")
 [4] prompt builder (retrieved data fenced as untrusted)
 [5] local LLM via Ollama (qwen3:4b-instruct-2507, Iris Xe iGPU via Vulkan)
 [6] citation validator (cited IDs ⊆ retrieved IDs)
      │
answer + sources (Streamlit UI)
```

## Quick start

```bash
# 1. Local LLM runtime (https://ollama.com)
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 2. Python env (creates .venv, installs deps + the package in editable mode)
make setup

# 3. Check the LLM works and measure its speed
make smoke
make test

# 4. Build the knowledge base (downloads ~60 MB, embeds ~4.8k chunks on CPU)
make data      # pinned public sources -> data/raw
make corpus    # parse + chunk -> data/processed/chunks.jsonl
make index     # embeddings -> Chroma (data/processed/chroma)

# 5. Chat
make app       # http://localhost:8501
```

## Knowledge base

All sources are public, pinned to a commit in [`data/sources.yaml`](data/sources.yaml), and
downloaded by `make data` (raw files are not redistributed in this repo).

| Source | Content | Docs | License |
|---|---|---|---|
| [MITRE ATT&CK](https://attack.mitre.org) Enterprise v20.1 | one doc per technique: description, mitigations, detection analytics | 706 | ATT&CK Terms of Use |
| [CERT-SG IRM](https://github.com/certsocietegenerale/IRM) | 22 incident response methodologies (6-step cheat sheets) | 22 | CC BY 3.0 |
| [AWS IR playbooks](https://github.com/aws-samples/aws-incident-response-playbooks) | cloud incident playbooks | 10 | MIT-0 |
| [Counteractive IR template](https://github.com/counteractive/incident-response-plan-template) | ransomware, phishing, defacement playbooks | 3 | Apache-2.0 |
| [Incident-Playbook](https://github.com/austinsonger/Incident-Playbook) | ATT&CK-mapped playbooks and checklists | 11 | MIT |

Template files that are still mostly `TODO` placeholders are detected and skipped.

**Retrieval** combines BM25 (keywords, exact technique IDs) and dense embeddings (meaning)
with Reciprocal Rank Fusion. See [`eval/results/retrieval_eval.md`](eval/results/retrieval_eval.md)
for the BM25 vs dense vs hybrid comparison (`make eval-retrieval`).

### Local vs cloud

| Part | Where it runs |
|---|---|
| App, LLM, vector store, demo | **Local**: no alert data leaves the machine |
| Injection-detector fine-tuning | Cloud notebook GPU (Colab/Kaggle); the trained model is used locally |
| Optional faster LLM during development | Any OpenAI-compatible API (`llm.provider: api` in `config.yaml`), **off by default** |

## Roadmap

- [x] Phase 0: project skeleton, local LLM smoke test
- [x] Week 1: ingestion (ATT&CK, playbooks), hybrid retrieval, retrieval eval, basic chat with sources
- [ ] Week 2: synthetic alerts, incident summaries, citations, refusal gate
  - [x] refusal gate, threshold calibrated per embedding model
  - [x] citation validator
  - [ ] synthetic alerts
  - [ ] incident summaries
- [ ] Week 3: prompt-injection detector (TF-IDF+LR baseline vs fine-tuned transformer)
- [ ] Week 4: evaluation (retrieval hit@k/MRR, groundedness, attack success with/without guard), demo

## Results

**Retrieval** (48 labelled questions; full tables and misses in
[`eval/results/retrieval_eval.md`](eval/results/retrieval_eval.md)). *Chunk hit@4* = a
correct source is among the 4 chunks actually sent to the LLM; the other metrics are
per document.

| Setup | chunk hit@4 | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| BM25 | 0.79 | 0.58 | 0.83 | 0.696 |
| Dense bge-small | 0.79 | 0.62 | 0.83 | 0.707 |
| Dense granite-small-r2 | 0.73 | 0.58 | 0.77 | 0.672 |
| **Hybrid BM25 + bge (default)** | 0.85 | **0.75** | **0.92** | **0.818** |
| Hybrid BM25 + granite | **0.88** | 0.71 | 0.90 | 0.792 |

BM25 finds exact technique IDs (8/8) where dense search fails (3/8); dense search is better
on paraphrased symptoms and procedures. Fusing both gets the best of each. The two hybrids
differ by one question on chunk hit@4 (noise on 48 questions); bge is kept for its better
ranking (MRR) and faster indexing.

**Local LLM** (i5-1155G7 + Iris Xe): see [`eval/results/bench_llm_summary.md`](eval/results/bench_llm_summary.md).
Qwen3-4B-Instruct-2507 on the iGPU is the default because it reliably cites sources.

## Limitations & ethics

Synthetic alerts, a small CPU-friendly model, and small evaluation sets (the retrieval
questions were written by the author). This is a decision-support prototype, not a
replacement for analyst judgement.

## Attribution

This project uses MITRE ATT&CK®. © The MITRE Corporation. This work is reproduced and
distributed with the permission of The MITRE Corporation. Incident Response Methodologies
by CERT Société Générale (CC BY 3.0). Other playbooks are used under their MIT, MIT-0 and
Apache-2.0 licenses (see `data/sources.yaml`).
