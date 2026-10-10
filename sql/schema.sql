-- FinRAG-India storage layer (PostgreSQL 16/17 + pgvector >= 0.6)
-- Design notes for Review-2:
--   * chunks.embedding  : vector(:dim), cosine distance operator class
--   * HNSW (m=16, ef_construction=64) not IVFFlat -> no training step, better
--     recall at high filter selectivity; ef_search tuned per query at runtime
--   * tsv              : Postgres full-text vector -> BM25-style lexical arm
--     (ts_rank_cd) so exact tokens like "Gross NPA", "23,456.78" are matchable
--   * numeric_facts    : every number parsed out of a chunk, stored with its
--     absolute (scale-normalised) value -> lets the verifier do an index-backed
--     numeric lookup instead of scanning text
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    title         TEXT,
    entity        TEXT,
    doc_type      TEXT,                  -- rbi_annual_report | company_ar | sebi_filing
    fiscal_year   TEXT,                  -- normalised FY2023-24
    scope         TEXT,                  -- consolidated | standalone | null
    n_pages       INT,
    engine        TEXT,
    ingested_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    element_type    TEXT NOT NULL,       -- text | table | table_rowgroup | table_row
    pipeline        TEXT NOT NULL DEFAULT 'structured',  -- structured | naive (ablation arm)
    content         TEXT NOT NULL,
    embed_text      TEXT NOT NULL,
    page_no         INT,
    section_path    TEXT,
    parent_id       TEXT,                -- small-to-big pointer (row -> table)
    table_id        TEXT,
    scope           TEXT,
    unit_name       TEXT,
    unit_multiplier DOUBLE PRECISION,
    entity          TEXT,
    fiscal_year     TEXT,
    meta            JSONB DEFAULT '{}'::jsonb,
    embedding       vector(:dim),
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED
);

-- Approximate nearest neighbour. Build AFTER bulk load for speed.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin      ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_trgm         ON chunks USING gin (embed_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_pipeline_idx ON chunks (pipeline);
CREATE INDEX IF NOT EXISTS chunks_filter_idx   ON chunks (fiscal_year, scope, element_type);
CREATE INDEX IF NOT EXISTS chunks_parent_idx   ON chunks (parent_id);
CREATE INDEX IF NOT EXISTS chunks_doc_idx      ON chunks (doc_id);

-- One row per number found in a chunk; value_abs is scale-normalised to rupees.
CREATE TABLE IF NOT EXISTS numeric_facts (
    id           BIGSERIAL PRIMARY KEY,
    chunk_id     TEXT REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    doc_id       TEXT,
    raw          TEXT,
    value        DOUBLE PRECISION,
    value_abs    DOUBLE PRECISION,
    decimals     INT,
    is_percent   BOOLEAN,
    unit_name    TEXT,
    row_label    TEXT,
    col_label    TEXT,
    fiscal_year  TEXT,
    scope        TEXT,
    page_no      INT
);
CREATE INDEX IF NOT EXISTS nf_value_idx  ON numeric_facts (value);
CREATE INDEX IF NOT EXISTS nf_abs_idx    ON numeric_facts (value_abs);
CREATE INDEX IF NOT EXISTS nf_chunk_idx  ON numeric_facts (chunk_id);

-- Query log: every answer, its groundedness score and the abstention decision.
CREATE TABLE IF NOT EXISTS query_log (
    id            BIGSERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ DEFAULT now(),
    question      TEXT,
    config        JSONB,
    decision      TEXT,        -- answer | answer_flagged | abstain
    groundedness  DOUBLE PRECISION,
    n_claims      INT,
    n_unsupported INT,
    iterations    INT,
    latency_ms    INT,
    answer        TEXT,
    payload       JSONB
);
