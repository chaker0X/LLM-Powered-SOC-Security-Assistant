import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from soc_assistant.ingest.documents import Chunk
from soc_assistant.llm.prompts import NO_CONTEXT
from soc_assistant.rag import RagPipeline, min_score_for, pin_techniques, technique_ids_in
from soc_assistant.retrieval.hybrid import Hit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from calibrate_refusal import (  # noqa: E402
    best_balanced_threshold, leave_one_out_answered, rates, recall_first_threshold, split_easy_hard)

MODEL = "BAAI/bge-small-en-v1.5"


def _chunk(doc_id, n=0, source_id="s"):
    return Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, title=doc_id, section="", text="text",
                 source_id=source_id, url="u", license="MIT")


PLAYBOOK = _chunk("aws:IRP-Ransomware")
POWERSHELL = [_chunk("attack:T1059.001", 0, "attack"), _chunk("attack:T1059.001", 1, "attack")]


class FakeRetriever:
    def __init__(self, best_score, hits=None):
        self.best_score = best_score
        self.chunks = [PLAYBOOK, *POWERSHELL]
        self.hits = hits if hits is not None else [Hit(PLAYBOOK, 1.0, best_score, 1, 1)]
        self.dense = SimpleNamespace(embedder=SimpleNamespace(model_name=MODEL))

    def search_with_score(self, query, k=4, mode="hybrid"):
        return self.hits[:k], self.best_score


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
    assert "cosine 0.20" in retrieval.reason


def test_strong_evidence_reaches_the_llm():
    llm = FakeLLM()
    pipe = RagPipeline(retriever=FakeRetriever(best_score=0.90), llm=llm)
    answer, retrieval = pipe.answer("How do I contain ransomware?")
    assert not retrieval.refused
    assert answer == "Answer [S1]."
    assert llm.calls == 1


def test_technique_ids_are_found_in_any_case_without_duplicates():
    assert technique_ids_in("detect t1059.001, T1059.001 and T1110 (not T12345)") == ["T1059.001", "T1110"]


def test_known_technique_id_passes_the_gate_and_is_pinned_first():
    llm = FakeLLM()
    hits = [Hit(PLAYBOOK, 1.0, 0.3, 1, 1), Hit(POWERSHELL[1], 0.5, 0.3, 2, 2)]
    pipe = RagPipeline(retriever=FakeRetriever(best_score=0.30, hits=hits), llm=llm)
    answer, retrieval = pipe.answer("How do I detect t1059.001?")
    assert not retrieval.refused and llm.calls == 1          # cosine 0.30 alone would be refused
    assert retrieval.known_ids == ["T1059.001"]
    assert [h.chunk.chunk_id for h in retrieval.hits] == ["attack:T1059.001#1", "aws:IRP-Ransomware#0"]


def test_unknown_technique_id_is_refused_without_calling_the_llm():
    llm = FakeLLM()
    pipe = RagPipeline(retriever=FakeRetriever(best_score=0.95), llm=llm)
    answer, retrieval = pipe.answer("What is T9999.123?")
    assert retrieval.refused and answer == NO_CONTEXT and llm.calls == 0
    assert retrieval.unknown_ids == ["T9999.123"]
    assert "T9999.123 is not a technique" in retrieval.reason


def test_pinned_technique_missing_from_retrieval_uses_its_first_chunk():
    pinned = pin_techniques([Hit(PLAYBOOK, 1.0, 0.7, 1, 1)], [POWERSHELL[0]], k=4)
    assert [h.chunk.chunk_id for h in pinned] == ["attack:T1059.001#0", "aws:IRP-Ransomware#0"]
    assert pinned[0].dense_score is None


def test_calibration_rules():
    in_scope, oos = [0.61, 0.70, 0.80], [0.40, 0.50, 0.65]
    t = recall_first_threshold(in_scope)
    assert t == 0.60
    assert rates(t, in_scope, oos) == (1.0, 2 / 3)                 # every in-scope answered
    # 0.61, 0.65 and 0.70 all score 1 + 2/3: the tie goes to the lowest threshold.
    assert best_balanced_threshold(in_scope, oos) == 0.61


def test_leave_one_out_exposes_the_overfitted_threshold():
    # Fitted without 0.61, the threshold becomes 0.69, so that question would be refused.
    assert leave_one_out_answered([0.61, 0.70, 0.80]) == 2 / 3


def test_hard_questions_are_reported_apart():
    assert split_easy_hard([0.4, 0.7, 0.5], ["oos01", "oos13", "oos02"], {"oos13"}) == ([0.4, 0.5], [0.7])
