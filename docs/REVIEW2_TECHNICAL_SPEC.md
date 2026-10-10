# Review-2 Technical Specification
### Answering the panel: "the scope statements are generic, the technical terms and concepts are missing"

For each scope item: **what the panel saw**, **the named techniques that were missing**, **the parameters/algorithms**, and **the file that implements it**. Every term below is implemented in the repository, not aspirational.

---

## 1. Structure-Aware Ingestion

> *Review-1 wording:* "specialized chunking algorithms designed to preserve the row, column, and hierarchical integrity of financial tables"

### 1.1 Table detection: lattice vs stream extraction
Two complementary strategies, run per page:

| Strategy | Signal used | Settings | When it wins |
|---|---|---|---|
| **Lattice** | explicit ruling lines | `vertical_strategy="lines"`, `snap_tolerance=4`, `join_tolerance=4`, `intersection_tolerance=5` | bordered statements (most RBI/bank tables) |
| **Stream** | whitespace + x-alignment clustering | `vertical_strategy="text"`, `text_tolerance=2`, `min_words_vertical=2` | borderless notes and annexures |

**Lattice-first precedence with region subtraction**: lattice detections claim their bounding boxes first; a stream detection is accepted only if its box does not intersect an already-claimed region. This prevents the same table being emitted twice with two different grids.

**False-positive rejection** (`_grid_score`): a candidate grid is discarded unless
fill ratio ≥ 0.45, width ∈ [2, 25], and ≥ 20 % of rows are numeric-dominant. Without this, whitespace clustering on a prose page yields a 15 × 49 mostly-empty "table" — this was an actual observed failure during development.

**Optional Docling backend**: layout model + **TableFormer** cell-structure recovery for spanning cells and scanned pages, behind `PARSER=docling` with automatic fallback.

### 1.2 The four structural signals that naive chunking destroys

1. **Header-band detection** (`detect_header_rows`) — scans down until the first numeric-dominant row; the rows above are the header band.
2. **Multi-level header flattening with span forward-fill** (`flatten_headers`) — a blank cell inherits from its left neighbour, recovering merged/spanning cells. Output is a **column path**: `"As at 31 March 2024 > Consolidated"`.
3. **Row-label hierarchy from PDF geometry** (`row_indents`, `row_label_path`) — pdfplumber strips leading spaces, so indentation is measured from the **x-coordinate of the first cell's leftmost word**, bucketed into depth levels. Yields `"Interest earned > Interest on advances"`, `"Advances > Term loans"`. *This is the concrete answer to "hierarchical integrity".*
4. **Unit-banner detection and inheritance** (`detect_unit_scale`, `caption_and_unit`) — `(Rs. in crore)` / `Amounts in Lakhs` are located by geometry (≤ 90 pt above / ≤ 26 pt below the table box) and attached to every cell as `unit_multiplier`.

### 1.3 Multi-page continuation stitching
`stitch_continuations` merges a fragment into its predecessor when column count matches **and** (header row repeats ∨ first row is numeric-dominant ∨ caption matches `(contd.)`), carrying `page_end` and the indent vector across the join.

### 1.4 Dual-granularity indexing (small-to-big / parent-child)
Each table produces three chunk types:

| Type | Content | Purpose |
|---|---|---|
| `table_row` | one serialised fact per row | **embedded** — one vector = one fact, maximum precision |
| `table_rowgroup` | header + 6 rows | local comparison context |
| `table` | full markdown grid | **parent** — what the LLM actually reads |

Row serialisation turns a grid row into a self-contained sentence:
`Advances > Term loans | 31 March 2024 (crore): 18,90,123.45 ; 31 March 2023 (crore): 16,54,321.09`

A `table_row` match is swapped for its parent at retrieval time, so the model sees the header band and neighbouring rows.

### 1.5 Contextual chunk headers
Every chunk's embedded text is prefixed with `entity | section | caption | unit | scope | page` (contextual-retrieval pattern) so the vector carries provenance, not just content.

### 1.6 Metadata enrichment
- **Fiscal-year normalisation**: `FY23-24`, `2023-24`, `year ended 31 March 2024`, `31.03.2024` → canonical `FY2023-24` (Indian April–March convention). Overlapping-span suppression prevents `FY2023-24` being re-read as bare `FY2023`.
- **Scope classification**: consolidated / standalone by marker counting.
- **Multi-period rows are deliberately left unpinned** (`fiscal_year=""`) so a year filter cannot hide a row that legitimately holds three years.

**Files:** `ingest/pdf_parse.py`, `ingest/table_chunker.py`, `ingest/text_chunker.py`, `ingest/metadata.py`, `numeric.py`

---

## 2. Vector Retrieval & Storage

> *Review-1 wording:* "pgvector database architecture to store document embeddings and execute efficient similarity searches"

### 2.1 Index choice, with the reason
**HNSW** (`m=16`, `ef_construction=64`, `vector_cosine_ops`), not IVFFlat:

| | IVFFlat | HNSW |
|---|---|---|
| Training step | required (needs representative data first) | none |
| Recall under metadata filtering | degrades — probes hit filtered-out lists | holds up |
| Build cost | lower | higher |

`hnsw.ef_search` is set **per query** via `set_config(..., true)` (transaction-local), giving a runtime recall/latency dial without an index rebuild.

### 2.2 Embedding model: asymmetric bi-encoder
`BAAI/bge-small-en-v1.5`, 384-d, L2-normalised (so cosine distance = 1 − dot product). Queries get the instruction prefix `"Represent this sentence for searching relevant passages: "`; passages do not. A deterministic hashing vectoriser (`EMBED_BACKEND=hash`) keeps CI offline and reproducible.

### 2.3 Hybrid retrieval: three arms
1. **Dense** — HNSW cosine kNN, `k=40`
2. **Lexical** — Postgres `tsvector` + `ts_rank_cd` + `websearch_to_tsquery`, `k=40`. This is what matches `"CRAR"`, `"Gross NPA"` and literal figures that dense vectors smear together. GIN index; `pg_trgm` for fuzzy label matching.
3. **Numeric** — `numeric_facts` table: every number exploded out of every chunk with `value_abs` (scale-normalised to rupees), B-tree indexed. Used by the verifier to ask *"does this figure exist anywhere in the corpus?"* — which separates a **retrieval miss** from a **fabrication**.

### 2.4 Fusion: Reciprocal Rank Fusion
`score(d) = Σ_arms w / (k + rank_arm(d))`, `k=60`. Rank-based, so cosine similarity and `ts_rank_cd` never need score normalisation — their scales are not comparable.

### 2.5 Two-stage retrieval
Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) over the top 30 fused candidates, sigmoid-normalised. A bi-encoder embeds query and passage separately; a cross-encoder attends over both jointly, which is why it resolves adjacent-fiscal-year confusions the bi-encoder cannot.

### 2.6 Diversity and expansion
- **MMR** (`λ=0.7`) on token Jaccard, so the context window isn't six near-duplicate rows.
- **Small-to-big parent expansion** (§1.4).
- **Pre-filtering, not post-filtering**: FY/scope/entity filters are pushed into the SQL `WHERE` so kNN searches the filtered set. Comparative questions are detected and deliberately **not** year-filtered — filtering a YoY question to one year guarantees failure.

**Files:** `sql/schema.sql`, `store/db.py`, `retrieve/hybrid.py`, `embed.py`

---

## 3. Active Hallucination Guardrails

> *Review-1 wording:* "calculates a groundedness score ... by verifying individual atomic claims"

### 3.1 Typed atomic claims
Decomposition is **hybrid**: a deterministic sentence/clause splitter always runs; an LLM pass refines it. If the LLM returns malformed JSON we keep the deterministic claims — this is the explicit mitigation for the *cascading claim-extraction failure* noted in the FinGround row of the literature survey. A **coverage guarantee** ensures every numeric token in the answer belongs to some claim.

| Type | Example | Verifier |
|---|---|---|
| `numeric` | "Gross NPA stood at 2.24 per cent" | deterministic matching |
| `derived` | "a decline of 54 basis points" | arithmetic recomputation |
| `textual` | "attributed to higher recoveries" | NLI entailment / LLM judge |

### 3.2 Numeric verification — four mechanisms

**(a) Rounding-aware tolerance.** Tolerance = half a unit of the **last digit the answer states**. Context `1,234.56` supports `1,234.6` and `1,235`, but not `1,200`, and *not* `1,234.5678` (over-precision = invented digits). This directly addresses the "over-cautious abstention" demerit in the literature survey: legitimate rounding is not a hallucination.

**(b) Scale normalisation.** `1.2 lakh crore` matches a cell `1,20,345` in a `(Rs. in crore)` table; `1,20,345 lakh` does not. Catches the lakh/crore conversion error.

**(c) Line-level evidence localisation.** Evidence is the **physical table row** containing the matched number, not a character window. A fuzzy ±240-char window made every row in a table look like evidence for every claim — an observed and fixed defect.

**(d) Row-competition test.** If another row in the same chunk matches the claim's metric vocabulary *better* than the grounded row, the match is rejected as `wrong_row`. This catches the canonical error named in the Review-1 findings section — *"distinguishing between a 1.2 % Net NPA and a 1.5 % Gross NPA within the same document chunk"*. Worked example:

```
Claim  : "Net NPA ratio stood at 2.24 per cent"
Match  : | Gross NPA ratio (%) | 2.24 | 2.78 |   sim = 0.67
Better : | Net NPA ratio (%)   | 0.57 | 0.67 |   sim = 1.00   -> REJECT (wrong row)
```

### 3.3 Retrieval miss vs fabrication
Unmatched figure → probe the corpus-wide `numeric_facts` index:
- present elsewhere → **retrieval miss** → re-retrieval can fix it
- absent → **fabrication** → must abstain

Without this distinction every retrieval failure is misdiagnosed as a hallucination and the abstention rate explodes.

### 3.4 Derived-claim recomputation
The stated delta is reproduced from operands, with two restrictions that matter:
- **operand pool restricted** to rows whose label matches the claim's metric;
- **operation fixed by the claim's own cue word** (`bps` → bps change, `growth %` → pct change, `times` → ratio).

Without both, a large operand pool "explains" any number by coincidence — during testing an unrestricted pool validated a fabricated "88 basis points". Percent-ness is read from the **row label** (`Gross NPA ratio (%)`), not the token, since table cells carry no `%` sign.

### 3.5 Groundedness score
```
G = Σ wᵢ · scoreᵢ / Σ wᵢ ,   w = 2.0 for numeric/derived, 1.0 for textual
```
Plus a boolean `numeric_ok` gate: **any** unsupported numeric claim blocks a silent answer regardless of `G`, because in financial QA a wrong figure is the failure that matters.

**Files:** `guard/claims.py`, `guard/verifier.py`

---

## 4. Abstention Protocols

> *Review-1 wording:* "strict safety thresholds ... (e.g. < 0.95)"

This is **selective prediction**, and the threshold is **calibrated, not guessed**.

### 4.1 Two pre-generation gates
Refusing before generating is cheaper and safer: (i) empty retrieval → abstain; (ii) top-1 cross-encoder score < `τ_rerank` → "no relevant evidence" abstain. The gate is **scale-aware** — `τ_rerank` is calibrated on cross-encoder scores, so it is skipped when RRF scores are in play.

### 4.2 Post-generation band
```
G ≥ τ_high                → ANSWER
τ_low ≤ G < τ_high        → REPAIR: re-retrieve and regenerate (≤ max_iters)
                            still unmet → ANSWER_FLAGGED
G < τ_low                 → ABSTAIN (structured refusal)
any unsupported numeric   → hard gate: never answered silently
```

### 4.3 Repair loop
Query rewriting pulls metric/period vocabulary out of the **failed claims** into the lexical arm and relaxes the FY filter (a comparison was probably over-filtered). Bounded by `max_iters=2` — the literature survey's "agents stuck in infinite retrieval loops" demerit.

### 4.4 Graded output, not binary
`ANSWER_FLAGGED` strikes through the exact unsupported span and appends the machine-generated reason, rather than discarding a 90 %-correct answer. Structured refusals state **reason / what was searched / what would resolve it**.

### 4.5 Calibration (the answer to "why 0.95?")
`eval/calibrate.py` sweeps (τ_low, τ_high) on the **dev split** and selects the point maximising accuracy subject to selective risk ≤ target (default 5 %). If no point is feasible it says so explicitly rather than returning a meaningless threshold — an infeasible sweep means the confidence signal itself needs work.

**Files:** `guard/policy.py`, `pipeline.py`, `eval/calibrate.py`

---

## 5. Automated Evaluation

> *Review-1 wording:* "using established frameworks (such as RAGAS)"

### 5.1 Why RAGAS alone is insufficient
RAGAS Faithfulness is an LLM judge — fuzzy about digits, exactly where this project needs precision. RAGAS is the **second** opinion; the deterministic numeric verifier is the primary signal. Abstentions are **excluded** from faithfulness and reported separately, because averaging them in silently rewards refusing to answer.

### 5.2 The benchmark: IFHB
Generated programmatically from parsed table cells, so every item carries **gold value + gold evidence chunk_id** — you cannot compute retrieval metrics without gold chunk labels. Deterministic dev/test split.

| Type | Targets |
|---|---|
| `lookup` | single-cell factual accuracy |
| `yoy` | multi-step numeric reasoning |
| `unanswerable` | honest abstention |
| `scale_trap` | lakh/crore conversion hallucination |
| `scope_trap` | consolidated/standalone confusion |

### 5.3 Metrics — selective prediction, not plain accuracy
A system that abstains on everything scores 0 % hallucination, so accuracy alone is the wrong headline:

- **Coverage** = answered / answerable
- **Selective risk** = wrong / answered (hallucination rate on what we *do* answer)
- **Risk–coverage curve and AURC** — threshold-free measure of how well groundedness *ranks* good answers above bad ones
- **Abstention confusion matrix**: correct / wrong / honest-abstain / over-abstain
- **Retrieval, separately**: Recall@k, MRR, nDCG@10 — so a failure is attributable to retriever vs generator
- **Dump guard**: predictions > 1200 chars or > 12 figures score incorrect. Without it, a naive baseline that dumps its 512-token window "contains" the gold figure and scores as accurate — a false comparison this benchmark exists to avoid.

### 5.4 Ablation arms
`B0_naive_noguard` (512-token windows, dense only) → `B1_struct_noguard` → `B2_struct_guard` (full), plus `A1_no_lexical`, `A2_no_rerank`, `A3_no_parent`. Each isolates one contribution.

**Files:** `eval/build_benchmark.py`, `eval/metrics.py`, `eval/run_eval.py`, `eval/calibrate.py`, `eval/ragas_eval.py`

---

## Novelty statement (one paragraph for the viva)

Existing financial-RAG hallucination detectors either need white-box logit access (unusable with commercial APIs) or rely on an LLM judge (slow, and imprecise about digits). This work contributes a **deterministic, interpretable numeric grounding layer for Indian financial reports** combining: rounding-aware tolerance tied to the precision the answer itself asserts; lakh/crore scale normalisation; **row-competition conflict detection** that catches a figure grounded against the wrong row of the right table; **retrieval-miss vs fabrication disambiguation** via a corpus-wide numeric index, which drives a repair loop instead of a blanket refusal; and a **calibrated two-threshold abstention policy** evaluated with selective-prediction metrics on a purpose-built Indian benchmark with scale-trap and scope-trap adversarial items. The failure modes addressed (Ind AS consolidated/standalone duality, lakh–crore dual notation, April–March fiscal-year convention) are structurally absent from FinanceBench and PHANTOM.
