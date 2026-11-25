"""SOC Assistant: chat and alert-analysis UI.

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

from soc_assistant.alerts import AlertAnalysis, AlertAnalyzer, load_alerts  # noqa: E402
from soc_assistant.config import load_config  # noqa: E402
from soc_assistant.llm.citations import check_citations  # noqa: E402
from soc_assistant.llm.client import LLMClient  # noqa: E402
from soc_assistant.llm.prompts import source_label  # noqa: E402
from soc_assistant.rag import RagPipeline  # noqa: E402
from soc_assistant.retrieval.hybrid import MODES, Hit, Retriever  # noqa: E402

CFG = load_config()
EMBEDDERS = ["BAAI/bge-small-en-v1.5", "ibm-granite/granite-embedding-small-english-r2"]
SOURCE_NAMES = {
    "attack": "MITRE ATT&CK",
    "counteractive": "Counteractive IR playbook",
    "austinsonger": "Incident-Playbook",
    "aws": "AWS IR playbook",
    "certsg": "CERT-SG IRM",
}
VIEWS = ["Ask a question", "Analyze an alert"]
LLM_DOWN = "⚠️ Cannot reach the LLM. Is Ollama running? (`ollama serve`)"

st.set_page_config(page_title="SOC Assistant", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the knowledge base and embedding model…")
def get_pipeline(embedder: str) -> RagPipeline:
    return RagPipeline(retriever=Retriever(embedder), llm=LLMClient())


@st.cache_data
def get_alerts() -> list[dict]:
    return load_alerts()


def to_sources(hits: list[Hit]) -> list[dict]:
    return [{"label": source_label(i), "title": h.chunk.title, "section": h.chunk.section,
             "source_id": h.chunk.source_id, "license": h.chunk.license, "url": h.chunk.url,
             "text": h.chunk.text, "dense_score": h.dense_score}
            for i, h in enumerate(hits, start=1)]


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
    {"ok": st.success, "warn": st.warning, "info": st.info, "error": st.error}[level](text)


# ---- Sidebar -----------------------------------------------------------------------
with st.sidebar:
    st.header("🛡️ SOC Assistant")
    st.caption("Local RAG over MITRE ATT&CK and public incident-response playbooks.")
    view = st.radio("Mode", VIEWS)
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index(CFG["retrieval"]["mode"]))
    embedder = st.selectbox("Embedding model", EMBEDDERS,
                            index=EMBEDDERS.index(CFG["embeddings"]["model"])
                            if CFG["embeddings"]["model"] in EMBEDDERS else 0)
    top_k = st.slider("Sources sent to the LLM (top-k)", 1, 8, CFG["retrieval"]["top_k"],
                      help="More sources = more context but a slower answer on this laptop.")
    st.caption(f"LLM: `{LLMClient().model}` via {CFG['llm']['provider']}")
    if view == VIEWS[0] and st.button("Clear chat"):
        st.session_state.messages = []

pipeline = get_pipeline(embedder)


# ---- Alert analysis ------------------------------------------------------------------
def render_analysis(res: AlertAnalysis, label: dict, seconds: float) -> None:
    st.caption(f"Search query built from the alert: {res.query}")
    sources = to_sources(res.retrieval.hits)
    if res.refused:
        render_check("info", f"Refused by the evidence gate: {res.retrieval.reason}. The LLM was not called.")
        render_sources(sources)
        return
    if res.error:
        render_check("error", f"Check this answer: {res.error}.")
        st.code(res.raw)
        return

    inc = res.incident
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.metric("ATT&CK technique", inc.technique_id, inc.technique_name, delta_color="off")
    c2.metric("Severity", inc.severity.upper())
    c3.metric("Instructions in alert", "yes" if inc.suspicious_instructions else "no")
    if inc.suspicious_instructions:
        render_check("error", "The model found text addressed to an AI or to the analyst inside the alert. "
                              "Treat it as hostile and do not follow it.")
    st.markdown(inc.summary)
    st.markdown("**Response steps**\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(inc.response_steps, 1)))
    if res.problems:
        render_check("warn", "Check this summary: " + "; ".join(res.problems) + ".")
    else:
        render_check("ok", f"Citations valid ({len(res.citations.cited)} sources cited) and the technique "
                           "appears in the retrieved sources.")
    render_sources(sources)

    with st.expander("Ground truth (synthetic label, never shown to the LLM)"):
        hit = "✅" if inc.technique_id == label["technique_id"] else "❌"
        st.markdown(f"Technique: **{label['technique_id']}** {hit}  \n"
                    f"Injection payload hidden in the alert: **{'yes' if label['injected'] else 'no'}**"
                    + (f" (field `{label['injection_field']}`)" if label["injected"] else ""))
    st.caption(f"answer {seconds:.1f} s · mode {mode} · best cosine {res.retrieval.best_score:.2f}")


if view == VIEWS[1]:
    st.subheader("Analyze an alert")
    records = get_alerts()
    options = {f"{r['alert']['alert_id']} · {r['alert']['rule']['name']}": r for r in records}
    rec = options[st.selectbox(f"Synthetic alert ({len(records)} available)", list(options))]
    alert_id = rec["alert"]["alert_id"]
    with st.expander("Raw alert, as sent by the SIEM", expanded=True):
        st.json(rec["alert"])

    if st.button("Analyze", type="primary"):
        t0 = time.perf_counter()
        try:
            with st.spinner("Retrieving sources and writing the incident summary (about a minute on CPU)…"):
                result = AlertAnalyzer(pipeline).analyze(rec["alert"], k=top_k, mode=mode)
            st.session_state.alert_result = (alert_id, result, time.perf_counter() - t0)
        except requests.exceptions.ConnectionError:
            st.error(LLM_DOWN)
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ LLM request failed: {e}")

    stored = st.session_state.get("alert_result")
    if stored and stored[0] == alert_id:
        render_analysis(stored[1], rec["label"], stored[2])
    st.stop()


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
        sources = to_sources(retrieval.hits)

        t1 = time.perf_counter()
        check = None
        try:
            answer = st.write_stream(pipeline.stream_answer(question, retrieval))
            if retrieval.refused:
                check = ("info", f"Refused by the evidence gate: {retrieval.reason}. "
                                 "The LLM was not called.")
            else:
                report = check_citations(answer, len(sources))
                check = (("ok", f"Citations valid ({len(report.cited)} sources cited, "
                                f"{report.coverage:.0%} of sentences cited).")
                         if report.valid and not report.problems()
                         else ("warn", "Check this answer: " + "; ".join(report.problems()) + "."))
        except requests.exceptions.ConnectionError:
            answer = LLM_DOWN
            st.error(answer)
        except requests.exceptions.RequestException as e:
            answer = f"⚠️ LLM request failed: {e}"
            st.error(answer)
        timing = (f"retrieval {t_retrieval * 1000:.0f} ms · answer {time.perf_counter() - t1:.1f} s · "
                  f"mode {mode} · best cosine {retrieval.best_score:.2f}"
                  + (f" · {retrieval.reason}" if retrieval.known_ids else ""))
        if check:
            render_check(*check)
        render_sources(sources)
        st.caption(timing)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "timing": timing, "check": check})
