-- ===========================================================================
--  FinRAG-India — database walkthrough for Review-2
--
--  Run it in one go so nothing is typed live in front of the panel:
--
--    Get-Content sql\demo_db.sql | docker exec -i finrag-india-db-1 psql -U postgres -d finrag
--
--  Each section is labelled with the claim it evidences.
-- ===========================================================================

\pset pager off
\pset border 2

\echo ''
\echo '==========================================================================='
\echo ' 1.  STORAGE LAYER — pgvector is installed      (evidences scope 4.2)'
\echo '==========================================================================='
SELECT extname AS extension, extversion AS version
FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY extname;

\echo ''
\echo '==========================================================================='
\echo ' 2.  WHAT WAS INGESTED'
\echo '==========================================================================='
SELECT doc_id, entity, fiscal_year, n_pages, engine FROM documents ORDER BY n_pages DESC;

\echo ''
\echo '==========================================================================='
\echo ' 3.  CHUNK INVENTORY — structured vs naive, in ONE database'
\echo '      This is what makes the chunking ablation a controlled comparison:'
\echo '      same documents, same storage, only the chunker differs.'
\echo '==========================================================================='
SELECT pipeline, element_type, count(*) AS chunks
FROM chunks GROUP BY pipeline, element_type ORDER BY pipeline, element_type;

SELECT count(*) FILTER (WHERE pipeline = 'structured') AS structured,
       count(*) FILTER (WHERE pipeline = 'naive')      AS naive,
       count(*)                                        AS total
FROM chunks;

\echo ''
\echo '==========================================================================='
\echo ' 4.  STRUCTURE-AWARE OUTPUT — recovered hierarchy + inherited units'
\echo '      The ">" in row_label is hierarchy recovered from PDF x-coordinates.'
\echo '      unit_name was inherited from the "(Rs. in crore)" banner.'
\echo '==========================================================================='
SELECT page_no,
       unit_name,
       left(meta->>'row_label', 42) AS row_label,
       left(content, 58)            AS serialised_row
FROM chunks
WHERE element_type = 'table_row' AND meta->>'row_label' LIKE '%>%'
ORDER BY page_no
LIMIT 8;

\echo ''
\echo '==========================================================================='
\echo ' 5.  THE CONTRAST — the same corpus under the naive baseline chunker'
\echo '      Numbers detached from their headers. This is why Recall@5 = 0.'
\echo '==========================================================================='
SELECT left(content, 300) AS naive_window FROM chunks WHERE pipeline = 'naive' LIMIT 1;

\echo ''
\echo '==========================================================================='
\echo ' 6.  NUMERIC FACT INDEX — scale normalisation made concrete'
\echo '      value     = the figure as printed'
\echo '      value_abs = the same figure normalised to absolute rupees'
\echo '      This is what lets the verifier compare a crore figure against a'
\echo '      lakh figure, and tell a retrieval miss from a fabrication.'
\echo '==========================================================================='
SELECT left(row_label, 34) AS row_label, raw, value, value_abs, unit_name
FROM numeric_facts
WHERE unit_name IS NOT NULL
ORDER BY value_abs DESC
LIMIT 8;

SELECT count(*) AS numeric_facts_indexed FROM numeric_facts;

\echo ''
\echo '==========================================================================='
\echo ' 7.  QUERY LOG — every decision the guardrail made, with its score'
\echo '==========================================================================='
SELECT decision,
       count(*)                                   AS n,
       round(avg(groundedness)::numeric, 3)       AS avg_groundedness,
       round(avg(n_unsupported)::numeric, 2)      AS avg_unsupported_claims
FROM query_log GROUP BY decision ORDER BY n DESC;

\echo ''
\echo '==========================================================================='
\echo ' 8.  SCHEMA AND INDEXES — vector(384), HNSW, GIN, B-tree'
\echo '==========================================================================='
\d chunks

\echo ''
\echo 'Walkthrough complete.'
\echo ''
