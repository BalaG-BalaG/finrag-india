"""The full hallucination-aware RAG loop.

    question
      -> retrieve (hybrid + rerank + parent expansion)
      -> pre-generation gate            [abstain early if evidence is weak]
      -> generate (cited, unit-locked)
      -> atomic claim extraction
      -> claim verification (numeric / derived / textual)
      -> groundedness score
      -> policy: answer | re-retrieve (loop) | flag | abstain

Everything the loop did is returned in `trace`, which is what the UI renders and
what Langfuse records; the same dict is the eval harness's input, so the demo and
the numbers in the report can never diverge.
"""
from __future__ import annotations

import json
import time

from .config import CFG
from .generate.llm import get_llm
from .guard import policy
from .guard.claims import extract_claims, is_abstention
from .guard.verifier import groundedness, verify_claims
from .guard.repair import repair_derived
from .retrieve.hybrid import RetrievalConfig, format_context, retrieve
from .store import db

if CFG.langfuse_enabled:                # optional tracing, off unless configured
    from langfuse import observe
else:
    def observe(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco


class FinRAG:
    def __init__(self, conn, llm=None, rc: RetrievalConfig | None = None, cfg=CFG):
        self.conn = conn
        self.llm = llm if llm is not None else get_llm()
        self.rc = rc or RetrievalConfig()
        self.cfg = cfg

    @observe(name="finrag_answer")
    def answer(self, question: str, guard: bool = True, log: bool = True) -> dict:
        t0 = time.time()
        trace = {"question": question, "iterations": [], "config": self.cfg.as_dict(),
                 "guard_enabled": guard,
                 "retrieval_config": self.rc.__dict__.copy()}
        q, extra_filters = question, None
        decision = None
        contexts: list[dict] = []
        claims = []
        g = {"score": 0.0, "n_claims": 0, "n_unsupported": 0, "numeric_ok": True,
             "n_numeric": 0, "unsupported": []}

        for it in range(self.cfg.max_iters + 1):
            step: dict = {"iteration": it, "query": q}
            r = retrieve(self.conn, q, self.rc, extra_filters)
            contexts = r["contexts"]
            step["arms"] = r["arms"]
            step["top_rerank"] = round(r.get("top_rerank", 0.0), 4)
            step["contexts"] = [{"chunk_id": c["chunk_id"], "element_type": c["element_type"],
                                 "page_no": c["page_no"], "section_path": c.get("section_path"),
                                 "score": round(c.get("rerank_score", 0.0), 4),
                                 "preview": c["content"][:180]} for c in contexts]

            if guard:
                gate = policy.pre_generation_gate(r)
                if gate is not None:
                    decision = gate
                    step["gate"] = gate.reason
                    trace["iterations"].append(step)
                    break

            ctx_str = format_context(contexts)
            draft = self.llm.answer(question, ctx_str)
            step["draft"] = draft

            if draft.startswith("INSUFFICIENT_CONTEXT") or is_abstention(draft):
                decision = policy.Decision(policy.ABSTAIN, None, "model_self_abstained",
                                           draft, next_action="stop")
                trace["iterations"].append(step)
                break

            if not guard:                                   # baseline arm: no guardrail
                decision = policy.Decision(policy.ANSWER, 1.0, "guard_disabled", draft,
                                           next_action="stop")
                trace["iterations"].append(step)
                break

            claims, fallback = extract_claims(draft, self.llm)
            verify_claims(claims, contexts, llm=self.llm, conn=self.conn)
            repair = repair_derived(question, draft, claims, contexts)
            if repair is not None:
                draft = repair["answer"]
                claims.append(repair["claim"])
                step["repair"] = repair["note"]
            g = groundedness(claims)
            step["claims"] = [c.to_dict() for c in claims]
            step["groundedness"] = g
            step["claim_extractor_fallback"] = fallback

            decision = policy.decide(draft, claims, g, it, self.cfg)
            step["decision"] = decision.decision
            step["decision_reason"] = decision.reason
            trace["iterations"].append(step)

            if decision.next_action != "reretrieve":
                break
            q, extra_filters = policy.rewrite_query(question, claims, r["plan"])
            step["rewritten_query"] = q

        latency = int((time.time() - t0) * 1000)
        out = {
            "question": question,
            "decision": decision.decision if decision else policy.ABSTAIN,
            "answer": decision.answer if decision else "",
            "groundedness": decision.groundedness if decision else 0.0,
            "reason": decision.reason if decision else "",
            "flagged_spans": decision.flagged_spans if decision else [],
            "claims": [c.to_dict() for c in claims],
            "citations": [{"index": i + 1, "chunk_id": c["chunk_id"], "page_no": c.get("page_no"),
                           "section": c.get("section_path"), "doc_id": c.get("doc_id")}
                          for i, c in enumerate(contexts)],
            "contexts": [{"index": i + 1, **{k: c.get(k) for k in
                          ("chunk_id", "content", "page_no", "element_type", "section_path",
                           "unit_name", "scope", "entity", "fiscal_year")}}
                         for i, c in enumerate(contexts)],
            "n_claims": g["n_claims"], "n_unsupported": g["n_unsupported"],
            "iterations": len(trace["iterations"]), "latency_ms": latency, "trace": trace,
        }
        if log:
            try:
                db.log_query(self.conn, question=question, config=json.dumps(self.rc.__dict__),
                             decision=out["decision"], groundedness=out["groundedness"],
                             n_claims=out["n_claims"], n_unsupported=out["n_unsupported"],
                             iterations=out["iterations"], latency_ms=latency,
                             answer=out["answer"][:4000],
                             payload=json.dumps({"citations": out["citations"]}))
            except Exception:
                pass
        return out
