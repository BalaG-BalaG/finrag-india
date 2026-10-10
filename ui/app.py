"""Streamlit demo UI.  streamlit run ui/app.py

Three panes matter for the viva:
  1. the decision banner (ANSWER / FLAGGED / ABSTAIN) with the groundedness score
  2. the claim table - every atomic claim, its verdict, and the exact evidence row
  3. guarded vs unguarded side-by-side, which is the contribution made visible
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finrag.pipeline import FinRAG            # noqa: E402
from finrag.retrieve.hybrid import RetrievalConfig   # noqa: E402
from finrag.store import db                   # noqa: E402

st.set_page_config(page_title="FinRAG-India", layout="wide")
st.title("Hallucination-aware RAG for Indian Financial Reports")

with st.sidebar:
    st.header("Settings")
    k_final = st.slider("Context chunks (k)", 2, 12, 6)
    use_rerank = st.checkbox("Cross-encoder reranker", True)
    use_lexical = st.checkbox("Lexical (BM25) arm", True)
    expand_parent = st.checkbox("Small-to-big parent expansion", True)
    compare = st.checkbox("Compare guarded vs unguarded", True)
    try:
        with db.connect() as c:
            st.caption(f"Corpus: {db.corpus_stats(c)['documents']} docs")
    except Exception as e:
        st.error(f"DB unavailable: {e}")

q = st.text_input("Question", "What was the Gross NPA ratio as at 31 March 2024?")

BADGE = {"answer": ("VERIFIED", "#1a7f37"), "answer_flagged": ("FLAGGED - unverified claims", "#9a6700"),
         "abstain": ("ABSTAINED", "#b42318")}


def render(out: dict, title: str):
    st.subheader(title)
    # The unguarded arm never runs verification, so its groundedness is a
    # placeholder, not a measurement. Labelling it VERIFIED would claim a check
    # that never happened.
    if out.get("reason") == "guard_disabled":
        label, colour, g_txt = "NOT CHECKED — no guardrail", "#555", "n/a"
    else:
        label, colour = BADGE.get(out["decision"], (out["decision"], "#444"))
        g = out.get("groundedness")
        # None means no claims were ever scored (the model self-abstained, or a
        # pre-generation gate fired), which is different from a measured zero.
        g_txt = f"{g:.2f}" if g is not None else "not scored"
    st.markdown(
        f"<div style='padding:8px 12px;border-radius:6px;background:{colour};color:#fff;"
        f"display:inline-block'><b>{label}</b> &nbsp; groundedness "
        f"{g_txt} &nbsp; | &nbsp; {out['n_claims']} claims, "
        f"{out['n_unsupported']} unsupported &nbsp; | &nbsp; {out['iterations']} pass(es), "
        f"{out['latency_ms']} ms</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown(out["answer"])
    if out["claims"]:
        st.markdown("**Claim-level verification**")
        st.dataframe([{
            "claim": c["text"][:90], "type": c["kind"],
            "verdict": "supported" if c["supported"] else "UNSUPPORTED",
            "score": round(c["score"], 2),
            "evidence": (c["evidence"][0].get("evidence", "")[:90] if c["evidence"] else ""),
            "why": c["reason"][:110],
        } for c in out["claims"]], use_container_width=True, hide_index=True)
    with st.expander(f"Retrieved context ({len(out['contexts'])} chunks)"):
        for c in out["contexts"]:
            st.markdown(f"**[C{c['index']}]** `{c['element_type']}` p.{c['page_no']} "
                        f"| {c.get('section_path') or ''} | unit: {c.get('unit_name')} "
                        f"| scope: {c.get('scope')}")
            st.code(c["content"][:1200])


if st.button("Ask", type="primary") and q.strip():
    rc = RetrievalConfig(use_rerank=use_rerank, use_lexical=use_lexical,
                         expand_parent=expand_parent, k_final=k_final)
    with st.spinner("Retrieving, generating, verifying..."):
        with db.connect() as conn:
            rag = FinRAG(conn, rc=rc)
            guarded = rag.answer(q, guard=True)
            plain = rag.answer(q, guard=False) if compare else None
    if plain:
        a, b = st.columns(2)
        with a:
            render(guarded, "With hallucination-awareness layer")
        with b:
            render(plain, "Baseline RAG (no guardrail)")
    else:
        render(guarded, "Answer")
