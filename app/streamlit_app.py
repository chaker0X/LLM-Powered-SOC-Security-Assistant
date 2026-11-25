"""SOC Assistant: chat UI.

Run:  streamlit run app/streamlit_app.py        (or `make app`)

Streamlit re-runs this whole script on every click, so heavy objects (embedding model,
indexes, LLM client) are created once with @st.cache_resource.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soc_assistant.config import load_config  # noqa: E402
from soc_assistant.llm.citations import check_citations  # noqa: E402
from soc_assistant.llm.client import LLMClient  # noqa: E402
from soc_assistant.llm.prompts import source_label  # noqa: E402
from soc_assistant.rag import RagPipeline  # noqa: E402
from soc_assistant.retrieval.hybrid import MODES, Retriever  # noqa: E402

CFG = load_config()
EMBEDDERS = ["BAAI/bge-small-en-v1.5", "ibm-granite/granite-embedding-small-english-r2"]
SOURCE_NAMES = {
    "attack": "MITRE ATT&CK",
    "counteractive": "Counteractive IR playbook",
    "austinsonger": "Incident-Playbook",
    "aws": "AWS IR playbook",
    "certsg": "CERT-SG IRM",
}

st.set_page_config(page_title="SOC Assistant", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the knowledge base and embedding model…")
def get_pipeline(embedder: str) -> RagPipeline:
    return RagPipeline(retriever=Retriever(embedder), llm=LLMClient())


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for s in sources:
            score = f" · cosine {s['dense_score']:.2f}" if s["dense_score"] is not None else ""
            st.markdown(f"**[{s['label']}] {s['title']}**" + (f" › {s['section']}" if s["section"] else ""))
            st.caption(f"{SOURCE_NAMES.get(s['source_id'], s['source_id'])} · {s['license']}{score} · "
                       f"[open source]({s['url']})")
            st.text(s["text"])


def render_check(level: str, text: str) -> None:
    """Refusal-gate / citation-validator verdict shown under an answer."""
    {"ok": st.success, "warn": st.warning, "info": st.info}[level](text)


# ---- Sidebar -----------------------------------------------------------------------
with st.sidebar:
    st.header("🛡️ SOC Assistant")
    st.caption("Local RAG over MITRE ATT&CK and public incident-response playbooks.")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index(CFG["retrieval"]["mode"]))
    embedder = st.selectbox("Embedding model", EMBEDDERS,
                            index=EMBEDDERS.index(CFG["embeddings"]["model"])
                            if CFG["embeddings"]["model"] in EMBEDDERS else 0)
    top_k = st.slider("Sources sent to the LLM (top-k)", 1, 8, CFG["retrieval"]["top_k"],
                      help="More sources = more context but a slower answer on this laptop.")
    st.caption(f"LLM: `{LLMClient().model}` via {CFG['llm']['provider']}")
    if st.button("Clear chat"):
        st.session_state.messages = []

pipeline = get_pipeline(embedder)

# ---- Chat history --------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("check"):
            render_check(*msg["check"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("timing"):
            st.caption(msg["timing"])

# ---- New question ----------------------------------------------------------------------
question = st.chat_input("Ask about an alert, a technique (e.g. T1059.001) or a response procedure…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        retrieval = pipeline.retrieve(question, k=top_k, mode=mode)
        t_retrieval = time.perf_counter() - t0
        sources = [{
            "label": source_label(i), "title": h.chunk.title, "section": h.chunk.section,
            "source_id": h.chunk.source_id, "license": h.chunk.license, "url": h.chunk.url,
            "text": h.chunk.text, "dense_score": h.dense_score,
        } for i, h in enumerate(retrieval.hits, start=1)]

        t1 = time.perf_counter()
        check = None
        try:
            answer = st.write_stream(pipeline.stream_answer(question, retrieval))
            if retrieval.refused:
                check = ("info", f"Refused by the evidence gate: best match cosine "
                                 f"{retrieval.best_score:.2f} < threshold {retrieval.threshold:.2f}. "
                                 "The LLM was not called.")
            else:
                report = check_citations(answer, len(sources))
                check = (("ok", f"Citations valid ({len(report.cited)} sources cited, "
                                f"{report.coverage:.0%} of sentences cited).")
                         if report.valid and not report.problems()
                         else ("warn", "Check this answer: " + "; ".join(report.problems()) + "."))
        except requests.exceptions.ConnectionError:
            answer = "⚠️ Cannot reach the LLM. Is Ollama running? (`ollama serve`)"
            st.error(answer)
        except requests.exceptions.RequestException as e:
            answer = f"⚠️ LLM request failed: {e}"
            st.error(answer)
        timing = (f"retrieval {t_retrieval * 1000:.0f} ms · answer {time.perf_counter() - t1:.1f} s · "
                  f"mode {mode} · best cosine {retrieval.best_score:.2f}")
        if check:
            render_check(*check)
        render_sources(sources)
        st.caption(timing)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "timing": timing, "check": check})
