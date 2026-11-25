"""Check the citations in an LLM answer against the sources it was actually given.

The prompt numbers the retrieved chunks S1..Sk and asks the model to cite them as [S1].
A small model can still cite a source that does not exist ([S7] with 4 sources), or
write facts with no citation at all. This module reports:
  invalid   cited ids that were not given to the model (a sign of hallucination)
  coverage  share of sentences that carry at least one citation
  valid     the answer is the refusal sentence, or it cites at least one source and
            every cited id exists
Accepted citation forms: [S1], [S1][S3], [S1, S2], [S1; S2].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from soc_assistant.llm.prompts import NO_CONTEXT, source_label

_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_ID = re.compile(r"S(\d+)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITE_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[\s*S\d+[^\]]*\])+)", re.IGNORECASE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
MIN_SENTENCE_CHARS = 20   # shorter pieces ("Response steps:") are headings, not claims


@dataclass
class CitationReport:
    cited: list[str] = field(default_factory=list)     # unique ids, in order of first use
    invalid: list[str] = field(default_factory=list)
    refused: bool = False
    sentences: int = 0
    cited_sentences: int = 0

    @property
    def coverage(self) -> float:
        return self.cited_sentences / self.sentences if self.sentences else 0.0

    @property
    def valid(self) -> bool:
        return self.refused or (bool(self.cited) and not self.invalid)

    def problems(self) -> list[str]:
        if self.refused:
            return []
        out = []
        if self.invalid:
            out.append("cites sources that were not provided: " + ", ".join(f"[{i}]" for i in self.invalid))
        if not self.cited:
            out.append("no source is cited")
        elif self.coverage < 0.5:
            out.append(f"only {self.cited_sentences} of {self.sentences} sentences cite a source")
        return out


def cited_ids(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for group in _GROUP.finditer(text):
        for num in _ID.findall(group.group(1)):
            seen.setdefault(source_label(int(num)), None)
    return list(seen)


def split_sentences(text: str) -> list[str]:
    """Sentences and list items. A citation written after the full stop ("... disk. [S1]")
    is moved before the stop, so it stays with the sentence it supports."""
    pieces: list[str] = []
    for line in text.splitlines():
        line = _CITE_AFTER_STOP.sub(lambda m: m.group(2) + m.group(1), _LIST_MARKER.sub("", line).strip())
        pieces += [p.strip() for p in _SENTENCE_END.split(line) if p.strip()]
    return [p for p in pieces if len(_GROUP.sub("", p).strip()) >= MIN_SENTENCE_CHARS]


def check_citations(answer: str, n_sources: int) -> CitationReport:
    cited = cited_ids(answer)
    if NO_CONTEXT in answer and not cited:
        return CitationReport(refused=True)
    allowed = {source_label(i) for i in range(1, n_sources + 1)}
    sentences = split_sentences(answer)
    return CitationReport(
        cited=cited,
        invalid=[c for c in cited if c not in allowed],
        sentences=len(sentences),
        cited_sentences=sum(1 for s in sentences if _GROUP.search(s)),
    )
