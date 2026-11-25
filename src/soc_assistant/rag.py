"""The question-answering pipeline: retrieve -> refusal gate -> build prompt -> stream the LLM answer.

Refusal gate: when even the best match in the knowledge base is weak (best dense cosine
below `retrieval.min_score`), the pipeline answers with the fixed refusal sentence and
never calls the LLM, so it cannot invent an answer.

Technique IDs bypass that score. Embedding models blur exact identifiers, so a question
like "T1059.001?" can get a low cosine although the corpus has that exact technique.
Instead: an ATT&CK ID that exists in the corpus pins its document at the top of the
sources and the question is answered; a well-formed ID that does not exist is refused.

The citation validator (llm/citations.py) checks the answer afterwards. The injection
guard comes before retrieval in week 3.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from soc_assistant.config import load_config
from soc_assistant.ingest.documents import TECHNIQUE_RE, Chunk
from soc_assistant.llm.client import LLMClient
from soc_assistant.llm.prompts import NO_CONTEXT, build_messages
from soc_assistant.retrieval.hybrid import Hit, Retriever

ATTACK_SOURCE = "attack"     # source id of MITRE ATT&CK in data/sources.yaml
_TECHNIQUE = re.compile(TECHNIQUE_RE.pattern, re.IGNORECASE)   # analysts also type "t1059"


def min_score_for(model_name: str, cfg: dict | None = None) -> float:
    """Refusal threshold of one embedding model (`retrieval.min_score` in config.yaml)."""
    value = (cfg or load_config()["retrieval"])["min_score"]
    if not isinstance(value, dict):
        return float(value)
    if model_name not in value:
        raise ValueError(f"No refusal threshold for {model_name!r} in config.yaml retrieval.min_score. "
                         "Run eval/calibrate_refusal.py and add it.")
    return float(value[model_name])


def technique_ids_in(text: str) -> list[str]:
    """ATT&CK technique IDs in a question, upper-cased, in order, without duplicates."""
    return list(dict.fromkeys(m.upper() for m in _TECHNIQUE.findall(text)))


def first_chunks_by_technique(chunks: list[Chunk]) -> dict[str, Chunk]:
    """'T1059.001' -> first chunk of the ATT&CK document of that technique."""
    out: dict[str, Chunk] = {}
    for c in chunks:
        if c.source_id == ATTACK_SOURCE:
            out.setdefault(c.doc_id.split(":", 1)[1], c)
    return out


def pin_techniques(hits: list[Hit], first_chunk: list[Chunk], k: int) -> list[Hit]:
    """Put the best-ranked chunk of each named technique first (its first chunk if
    retrieval did not find it), then the other hits in their original order."""
    pinned = [next((h for h in hits if h.chunk.doc_id == c.doc_id), Hit(c, 0.0, None, None, None))
              for c in first_chunk[:k]]
    taken = {h.chunk.chunk_id for h in pinned}
    return (pinned + [h for h in hits if h.chunk.chunk_id not in taken])[:k]


@dataclass
class Retrieval:
    hits: list[Hit]
    best_score: float    # best dense cosine over the whole corpus
    threshold: float
    known_ids: list[str] = field(default_factory=list)     # technique IDs found in the corpus
    unknown_ids: list[str] = field(default_factory=list)   # well-formed IDs that do not exist

    @property
    def refused(self) -> bool:
        if self.known_ids:
            return False     # exact match: stronger evidence than any cosine score
        return bool(self.unknown_ids) or self.best_score < self.threshold

    @property
    def reason(self) -> str:
        if self.known_ids:
            return "exact ATT&CK technique match: " + ", ".join(self.known_ids)
        if self.unknown_ids:
            return f"{', '.join(self.unknown_ids)} is not a technique in the knowledge base (MITRE ATT&CK)"
        if self.refused:
            return f"best match cosine {self.best_score:.2f} < threshold {self.threshold:.2f}"
        return ""


class RagPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.cfg = load_config()["retrieval"]
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()
        self.threshold = min_score_for(self.retriever.dense.embedder.model_name, self.cfg)
        self.techniques = first_chunks_by_technique(self.retriever.chunks)

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> Retrieval:
        k = k or self.cfg["top_k"]
        ids = technique_ids_in(question)
        known = [t for t in ids if t in self.techniques]
        unknown = [t for t in ids if t not in self.techniques]
        # With a named technique, look deeper so its best-matching section can be pinned.
        n = max(k, self.cfg["candidates"]) if known else k
        hits, best = self.retriever.search_with_score(question, k=n, mode=mode or self.cfg["mode"])
        if known:
            hits = pin_techniques(hits, [self.techniques[t] for t in known], k)
        return Retrieval(hits[:k], best, self.threshold, known, unknown)

    def stream_answer(self, question: str, retrieval: Retrieval) -> Iterator[str]:
        if retrieval.refused:
            yield NO_CONTEXT
            return
        yield from self.llm.chat(build_messages(question, retrieval.hits), stream=True)

    def answer(self, question: str, **kw) -> tuple[str, Retrieval]:
        retrieval = self.retrieve(question, **kw)
        return "".join(self.stream_answer(question, retrieval)), retrieval
