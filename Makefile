PY ?= .venv/bin/python
TORCH_CPU = torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

.PHONY: setup smoke test data corpus index eval-retrieval app all-data alerts

setup:            ## create venv, install CPU-only torch, deps and the package (editable)
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install $(TORCH_CPU)
	$(PY) -m pip install -r requirements.txt -e .

smoke:            ## check the LLM answers and measure speed
	$(PY) scripts/smoke_llm.py

test:             ## run unit tests
	$(PY) -m pytest -q

data:             ## download the pinned public sources (data/sources.yaml) into data/raw
	$(PY) -m soc_assistant.ingest.sources

corpus:           ## parse + chunk the sources into data/processed/{docs,chunks}.jsonl
	$(PY) -m soc_assistant.ingest.build

index:            ## embed the chunks into Chroma (both embedders, for the comparison)
	$(PY) -m soc_assistant.retrieval.index --models BAAI/bge-small-en-v1.5 ibm-granite/granite-embedding-small-english-r2

all-data: data corpus index

alerts:           ## synthetic labelled alerts -> data/alerts/alerts.jsonl
	$(PY) scripts/gen_alerts.py

eval-retrieval:   ## BM25 vs dense vs hybrid on eval/retrieval_questions.jsonl
	$(PY) eval/retrieval_eval.py

app:              ## start the chat UI on http://localhost:8501
	$(PY) -m streamlit run app/streamlit_app.py
