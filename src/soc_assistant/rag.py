"""The question-answering pipeline: retrieve -> refusal gate -> build prompt -> stream the LLM answer.

Week 2 adds the refusal gate: when even the best match in the knowledge base is weak
(best dense cosine below `retrieval.min_score`), the pipeline answers with the fixed
refusal sentence and never calls the LLM, so it cannot invent an answer. The citation
validator (llm/citations.py) checks the answer afterwards. The injection guard comes
before retrieval in week 3.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from soc_assistant.config import load_config
from soc_assistant.llm.client import LLMClient
from soc_assistant.llm.prompts import NO_CONTEXT, build_messages
from soc_assistant.retrieval.hybrid import Hit, Retriever


def min_score_for(model_name: str, cfg: dict | None = None) -> float:
    """Refusal threshold of one embedding model (`retrieval.min_score` in config.yaml)."""
    value = (cfg or load_config()["retrieval"])["min_score"]
    if not isinstance(value, dict):
        return float(value)
    if model_name not in value:
        raise ValueError(f"No refusal threshold for {model_name!r} in config.yaml retrieval.min_score. "
                         "Run eval/calibrate_refusal.py and add it.")
    return float(value[model_name])


@dataclass
class Retrieval:
    hits: list[Hit]
    best_score: float    # best dense cosine over the whole corpus
    threshold: float

    @property
    def refused(self) -> bool:
        return self.best_score < self.threshold


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.cfg = load_config()["retrieval"]
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()
        self.threshold = min_score_for(self.retriever.dense.embedder.model_name, self.cfg)

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> Retrieval:
        hits, best = self.retriever.search_with_score(
            question, k=k or self.cfg["top_k"], mode=mode or self.cfg["mode"])
        return Retrieval(hits, best, self.threshold)

    def stream_answer(self, question: str, retrieval: Retrieval) -> Iterator[str]:
        if retrieval.refused:
            yield NO_CONTEXT
            return
        yield from self.llm.chat(build_messages(question, retrieval.hits), stream=True)

    def answer(self, question: str, **kw) -> tuple[str, Retrieval]:
        retrieval = self.retrieve(question, **kw)
        return "".join(self.stream_answer(question, retrieval)), retrieval
