"""FastAPI backend.

    uvicorn finrag.api:app --reload --port 8000

Endpoints
    GET  /health          liveness + corpus stats
    POST /ask             the guarded pipeline; returns decision, claims, evidence
    POST /ask/compare     same question through guarded and unguarded arms (demo gold)
    POST /ingest          ingest a PDF already on disk
    GET  /stats           chunk/document counts by arm
    GET  /log             recent queries with their groundedness + decision
"""
from __future__ import annotations
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CFG
from .ingest.run_ingest import ingest_pdf
from .pipeline import FinRAG
from .retrieve.hybrid import RetrievalConfig
from .store import db

app = FastAPI(title="FinRAG-India", version="0.1.0",
              description="Hallucination-aware RAG for Indian financial report Q&A")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    guard: bool = True
    k_final: int | None = None
    pipeline: str = "structured"
    fiscal_year: str | None = None
    scope: str | None = None


class IngestRequest(BaseModel):
    path: str
    entity: str
    doc_type: str = "company_ar"
    fiscal_year: str = ""
    scope: str | None = None
    max_pages: int | None = None
    also_naive: bool = False


def _rc(req: AskRequest) -> RetrievalConfig:
    rc = RetrievalConfig(pipeline=req.pipeline)
    if req.k_final:
        rc.k_final = req.k_final
    return rc


@app.get("/health")
def health():
    try:
        with db.connect() as conn:
            return {"status": "ok", "corpus": db.corpus_stats(conn),
                    "embed_backend": CFG.embed_backend,
                    "embed_model": CFG.embed_model if CFG.embed_backend == "st" else "hash (offline)",
                    "llm": "echo-offline" if os.getenv("LLM_OFFLINE") == "1" else CFG.llm_model,
                    "verify_backend": CFG.verify_backend}
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {e}")


@app.post("/ask")
def ask(req: AskRequest):
    filters = {k: v for k, v in (("fiscal_year", req.fiscal_year), ("scope", req.scope)) if v}
    with db.connect() as conn:
        rag = FinRAG(conn, rc=_rc(req))
        out = rag.answer(req.question, guard=req.guard)
    out.pop("trace", None)
    return out


@app.post("/ask/compare")
def ask_compare(req: AskRequest):
    """Side-by-side guarded vs unguarded - the demo that makes the contribution visible."""
    with db.connect() as conn:
        rag = FinRAG(conn, rc=_rc(req))
        guarded = rag.answer(req.question, guard=True)
        plain = rag.answer(req.question, guard=False)
    for o in (guarded, plain):
        o.pop("trace", None)
    return {"question": req.question, "guarded": guarded, "unguarded": plain}


@app.post("/ingest")
def ingest(req: IngestRequest):
    with db.connect() as conn:
        db.init_schema(conn, CFG.embed_dim)
        return ingest_pdf(conn, req.path, req.entity, req.doc_type, req.fiscal_year,
                          req.scope, req.max_pages, req.also_naive)


@app.get("/stats")
def stats():
    with db.connect() as conn:
        return db.corpus_stats(conn)


@app.get("/log")
def log(limit: int = 25):
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT ts, question, decision, groundedness, n_claims, n_unsupported,
                              iterations, latency_ms FROM query_log ORDER BY id DESC LIMIT %s""",
                    (limit,))
        return cur.fetchall()
