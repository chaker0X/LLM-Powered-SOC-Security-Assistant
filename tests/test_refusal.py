import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from soc_assistant.ingest.documents import Chunk
from soc_assistant.llm.prompts import NO_CONTEXT
from soc_assistant.rag import RagPipeline, min_score_for
from soc_assistant.retrieval.hybrid import Hit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from calibrate_refusal import best_balanced_threshold, rates, recall_first_threshold  # noqa: E402

MODEL = "BAAI/bge-small-en-v1.5"


class FakeRetriever:
    def __init__(self, best_score):
        self.best_score = best_score
        self.dense = SimpleNamespace(embedder=SimpleNamespace(model_name=MODEL))

    def search_with_score(self, query, k=4, mode="hybrid"):
        chunk = Chunk(chunk_id="d#0", doc_id="d", title="Doc", section="", text="text",
                      source_id="s", url="u", license="MIT")
        return [Hit(chunk, 1.0, self.best_score, 1, 1)], self.best_score


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, stream=True):
        self.calls += 1
        yield "Answer [S1]."


def test_threshold_is_per_embedding_model():
    cfg = {"min_score": {MODEL: 0.6, "other": 0.8}}
    assert min_score_for(MODEL, cfg) == 0.6
    assert min_score_for("other", cfg) == 0.8
    assert min_score_for("any", {"min_score": 0.5}) == 0.5      # single value still accepted
    with pytest.raises(ValueError):
        min_score_for("unknown", cfg)


def test_configured_thresholds_cover_the_app_embedders():
    for model in (MODEL, "ibm-granite/granite-embedding-small-english-r2"):
        assert 0 < min_score_for(model) < 1


def test_weak_evidence_is_refused_without_calling_the_llm():
    llm = FakeLLM()
    pipe = RagPipeline(retriever=FakeRetriever(best_score=0.20), llm=llm)
    answer, retrieval = pipe.answer("best pizza in Paris?")
    assert retrieval.refused
    assert answer == NO_CONTEXT
    assert llm.calls == 0


def test_strong_evidence_reaches_the_llm():
    llm = FakeLLM()
    pipe = RagPipeline(retriever=FakeRetriever(best_score=0.90), llm=llm)
    answer, retrieval = pipe.answer("What is T1059.001?")
    assert not retrieval.refused
    assert answer == "Answer [S1]."
    assert llm.calls == 1


def test_calibration_rules():
    in_scope, oos = [0.61, 0.70, 0.80], [0.40, 0.50, 0.65]
    t = recall_first_threshold(in_scope)
    assert t == 0.60
    assert rates(t, in_scope, oos) == (1.0, 2 / 3)                 # every in-scope answered
    # 0.61, 0.65 and 0.70 all score 1 + 2/3: the tie goes to the lowest threshold.
    assert best_balanced_threshold(in_scope, oos) == 0.61
