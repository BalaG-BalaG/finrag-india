"""CLI: PDF -> parse -> structure-aware chunk -> embed -> pgvector.

    python -m finrag.ingest.run_ingest --pdf data/raw/rbi_ar_2024.pdf \
        --entity "Reserve Bank of India" --doc-type rbi_annual_report \
        --fiscal-year FY2023-24 --also-naive

--also-naive additionally indexes the SAME pdf with fixed 512-token windows under
pipeline='naive', which is the control arm for the ablation study.
"""
from __future__ import annotations

import argparse
import hashlib
import os

from ..config import CFG
from ..embed import embed_passages
from ..store import db
from .metadata import ParsedTable, TextBlock, detect_fiscal_years, detect_scope
from .pdf_parse import parse_pdf, raw_text
from .table_chunker import chunk_table, stitch_continuations
from .text_chunker import chunk_text_blocks, naive_chunk


def doc_id_for(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]


def ingest_pdf(conn, path: str, entity: str, doc_type: str = "company_ar",
               fiscal_year: str = "", scope: str | None = None, max_pages: int | None = None,
               also_naive: bool = False, engine: str | None = None, batch: int = 64) -> dict:
    doc_id = doc_id_for(path)
    doc = parse_pdf(path, max_pages=max_pages, engine=engine)

    tables = stitch_continuations([e for e in doc.elements if isinstance(e, ParsedTable)])
    blocks = [e for e in doc.elements if isinstance(e, TextBlock)]
    head = " ".join(b.text for b in blocks[:20])
    fiscal_year = fiscal_year or (detect_fiscal_years(head)[:1] or [""])[0]
    scope = scope or detect_scope(head)

    db.upsert_document(conn, doc_id, path, os.path.basename(path), entity, doc_type,
                       fiscal_year, scope, doc.n_pages, doc.engine)

    chunks = chunk_text_blocks(blocks, doc_id, entity, scope)
    for t in tables:
        chunks += chunk_table(t, doc_id, entity, scope)

    if also_naive:
        txt, page_map = raw_text(path, max_pages=max_pages)
        chunks += naive_chunk(txt, doc_id, entity, page_map)

    n = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        db.insert_chunks(conn, part, embed_passages([c.embed_text for c in part]))
        n += len(part)
    db.build_indexes(conn)

    counts: dict[str, int] = {}
    for c in chunks:
        counts[f"{c.pipeline}:{c.element_type}"] = counts.get(f"{c.pipeline}:{c.element_type}", 0) + 1
    return {"doc_id": doc_id, "pages": doc.n_pages, "engine": doc.engine,
            "tables": len(tables), "chunks": n, "breakdown": counts,
            "fiscal_year": fiscal_year, "scope": scope}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, nargs="+")
    ap.add_argument("--entity", required=True)
    ap.add_argument("--doc-type", default="company_ar")
    ap.add_argument("--fiscal-year", default="")
    ap.add_argument("--scope", default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--engine", default=None, choices=[None, "pdfplumber", "docling"])
    ap.add_argument("--also-naive", action="store_true")
    ap.add_argument("--init", action="store_true", help="create schema first")
    a = ap.parse_args()

    with db.connect() as conn:
        if a.init:
            db.init_schema(conn, CFG.embed_dim)
        for p in a.pdf:
            print(ingest_pdf(conn, p, a.entity, a.doc_type, a.fiscal_year, a.scope,
                             a.max_pages, a.also_naive, a.engine))
        print(db.corpus_stats(conn))


if __name__ == "__main__":
    main()
