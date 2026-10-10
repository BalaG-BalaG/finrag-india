"""pgvector storage + the three retrieval arms (dense, lexical, numeric).

Fusion and reranking live in finrag.retrieve.hybrid; this module only owns SQL.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from ..config import CFG
from ..ingest.metadata import Chunk
from ..numeric import extract_numbers

SCHEMA_PATH = "sql/schema.sql"


@contextmanager
def connect(dsn: str | None = None):
    conn = psycopg.connect(dsn or CFG.pg_dsn, row_factory=dict_row)
    try:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # Fresh database: the extension doesn't exist yet, so the vector type
            # can't be registered. Create it, then register.
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema(conn, dim: int | None = None, schema_path: str = SCHEMA_PATH) -> None:
    sql = open(schema_path).read().replace(":dim", str(dim or CFG.embed_dim))
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def upsert_document(conn, doc_id: str, path: str, title: str, entity: str, doc_type: str,
                    fiscal_year: str, scope: str | None, n_pages: int, engine: str) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO documents (doc_id, path, title, entity, doc_type, fiscal_year, scope, n_pages, engine)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_id) DO UPDATE SET path=EXCLUDED.path, title=EXCLUDED.title,
                entity=EXCLUDED.entity, doc_type=EXCLUDED.doc_type, fiscal_year=EXCLUDED.fiscal_year,
                scope=EXCLUDED.scope, n_pages=EXCLUDED.n_pages, engine=EXCLUDED.engine
        """, (doc_id, path, title, entity, doc_type, fiscal_year, scope, n_pages, engine))
    conn.commit()


def insert_chunks(conn, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    rows = []
    for c, e in zip(chunks, embeddings):
        rows.append((c.chunk_id, c.doc_id, c.element_type, c.pipeline, c.content, c.embed_text,
                     c.page_no, c.section_path, c.parent_id, c.table_id, c.scope, c.unit_name,
                     c.unit_multiplier, c.entity, c.fiscal_year, json.dumps(c.meta), e))
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO chunks (chunk_id, doc_id, element_type, pipeline, content, embed_text,
                page_no, section_path, parent_id, table_id, scope, unit_name, unit_multiplier,
                entity, fiscal_year, meta, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
            ON CONFLICT (chunk_id) DO UPDATE SET content=EXCLUDED.content,
                embed_text=EXCLUDED.embed_text, embedding=EXCLUDED.embedding, meta=EXCLUDED.meta
        """, rows)
    conn.commit()
    index_numeric_facts(conn, chunks)
    return len(rows)


def index_numeric_facts(conn, chunks: list[Chunk]) -> int:
    """Explode every number out of every chunk into numeric_facts (scale-normalised)."""
    rows = []
    for c in chunks:
        cols = c.meta.get("columns") or []
        for n in extract_numbers(c.content):
            mult = c.unit_multiplier if (not n.is_percent and n.scale is None) else None
            val_abs = n.absolute(mult)
            col_label = ""
            if c.element_type == "table_row" and cols:
                seg = c.content.split("|", 1)[-1]
                best = -1
                for j, col in enumerate(cols):
                    p = seg.find(col)
                    if 0 <= p <= n.start and p > best:
                        best, col_label = p, col
            rows.append((c.chunk_id, c.doc_id, n.raw, n.value, val_abs, n.decimals,
                         n.is_percent, c.unit_name, c.meta.get("row_label", ""), col_label,
                         c.fiscal_year, c.scope, c.page_no))
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.execute("DELETE FROM numeric_facts WHERE chunk_id = ANY(%s)",
                    ([c.chunk_id for c in chunks],))
        cur.executemany("""INSERT INTO numeric_facts (chunk_id, doc_id, raw, value, value_abs,
            decimals, is_percent, unit_name, row_label, col_label, fiscal_year, scope, page_no)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows)
    conn.commit()
    return len(rows)


def build_indexes(conn) -> None:
    """Call after bulk load: HNSW builds far faster on a populated table."""
    with conn.cursor() as cur:
        cur.execute("ANALYZE chunks")
    conn.commit()


def _filter_sql(filters: dict | None) -> tuple[str, list]:
    where, params = [], []
    f = filters or {}
    where.append("pipeline = %s")
    params.append(f.get("pipeline", "structured"))
    if f.get("fiscal_year"):
        where.append("(fiscal_year = %s OR fiscal_year = '')")
        params.append(f["fiscal_year"])
    if f.get("scope"):
        where.append("(scope = %s OR scope IS NULL)")
        params.append(f["scope"])
    if f.get("entity"):
        where.append("entity ILIKE %s")
        params.append(f"%{f['entity']}%")
    if f.get("doc_id"):
        where.append("doc_id = %s")
        params.append(f["doc_id"])
    if f.get("element_types"):
        where.append("element_type = ANY(%s)")
        params.append(list(f["element_types"]))
    return (" AND ".join(where) if where else "TRUE"), params


COLS = ("chunk_id, doc_id, element_type, content, page_no, section_path, parent_id, table_id, "
        "scope, unit_name, unit_multiplier, entity, fiscal_year, meta")


def dense_search(conn, qvec: list[float], k: int, filters: dict | None = None) -> list[dict]:
    where, params = _filter_sql(filters)
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(CFG.hnsw_ef_search),))
        cur.execute(f"""
            SELECT {COLS}, 1 - (embedding <=> %s::vector) AS dense_score
            FROM chunks WHERE {where}
            ORDER BY embedding <=> %s::vector LIMIT %s
        """, [qvec] + params + [qvec, k])
        return cur.fetchall()


def lexical_search(conn, query: str, k: int, filters: dict | None = None) -> list[dict]:
    """BM25-style arm: ts_rank_cd over the generated tsvector, OR-ed websearch query."""
    where, params = _filter_sql(filters)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {COLS}, ts_rank_cd(tsv, q) AS lex_score
            FROM chunks, websearch_to_tsquery('english', %s) q
            WHERE {where} AND tsv @@ q
            ORDER BY lex_score DESC LIMIT %s
        """, [query] + params + [k])
        return cur.fetchall()


def numeric_search(conn, value: float, decimals: int = 2, is_percent: bool = False,
                   filters: dict | None = None, k: int = 20) -> list[dict]:
    """Index-backed lookup of a literal figure, with rounding tolerance.

    Used by the verifier when a claim cites a number that the retrieved context
    does not contain: 'does this number exist anywhere in the corpus, or did the
    model invent it?' -> distinguishes a retrieval miss from a true fabrication.
    """
    tol = 0.5 * 10 ** (-decimals) * 1.000001
    where, params = _filter_sql(filters)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT c.chunk_id, c.content, c.page_no, c.doc_id, c.element_type, c.section_path,
                   n.raw, n.value, n.row_label, n.col_label, n.unit_name, n.is_percent
            FROM numeric_facts n JOIN chunks c ON c.chunk_id = n.chunk_id
            WHERE {where.replace('pipeline', 'c.pipeline').replace('fiscal_year', 'c.fiscal_year')
                        .replace('scope', 'c.scope').replace('entity', 'c.entity')
                        .replace('doc_id', 'c.doc_id').replace('element_type', 'c.element_type')}
              AND n.is_percent = %s AND abs(abs(n.value) - %s) <= %s
            LIMIT %s
        """, params + [is_percent, abs(value), tol, k])
        return cur.fetchall()


def get_chunks(conn, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute(f"SELECT {COLS} FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
        return cur.fetchall()


def get_parents(conn, parent_ids: list[str]) -> dict[str, dict]:
    return {r["chunk_id"]: r for r in get_chunks(conn, [p for p in parent_ids if p])}


def log_query(conn, **kw) -> None:
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO query_log (question, config, decision, groundedness, n_claims,
            n_unsupported, iterations, latency_ms, answer, payload)
            VALUES (%(question)s,%(config)s::jsonb,%(decision)s,%(groundedness)s,%(n_claims)s,
                    %(n_unsupported)s,%(iterations)s,%(latency_ms)s,%(answer)s,%(payload)s::jsonb)""",
                    kw)
    conn.commit()


def corpus_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""SELECT pipeline, element_type, count(*) AS n FROM chunks
                       GROUP BY 1,2 ORDER BY 1,2""")
        by_type = cur.fetchall()
        cur.execute("SELECT count(*) AS n FROM documents")
        docs = cur.fetchone()["n"]
        cur.execute("SELECT count(*) AS n FROM numeric_facts")
        nf = cur.fetchone()["n"]
    return {"documents": docs, "numeric_facts": nf, "chunks": by_type}
