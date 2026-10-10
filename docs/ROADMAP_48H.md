# 48-Hour Roadmap to Review-2 (50 % completion)

**Today is Fri 18 Sep 2026.** Plan assumes Review-2 on Sun 21 Sep. Times are indicative — the *checkpoints* matter, not the clock.

**The strategic point:** the code in this repository already implements the pipeline. Your 48 hours are not for building from zero — they are for **swapping the synthetic PDF for real RBI/NSE filings, running the real models, and producing the numbers and slides**. Do not rewrite what already passes tests.

---

## Hour 0 — Ground rules

1. **Every claim in your slides must be backed by a file or a number you produced.** That is what "generic" meant in Review-1.
2. **Freeze scope at 14:00 on Day 2.** Whatever works then is what you demo.
3. **Commit every 2 hours.** `git commit -am "checkpoint: <what works now>"`.
4. Run `make test` before every commit. 35 tests currently pass; keep them passing.

---

# DAY 1 — Real data, real models, real numbers

## Block 1 (2 h) — Environment up, smoke test green

```bash
cd finrag-india
cp .env.example .env
docker compose up -d db                 # Postgres 17 + pgvector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline smoke test — proves the whole loop before real models are involved
export PYTHONPATH=src EMBED_BACKEND=hash LLM_OFFLINE=1 VERIFY_BACKEND=none
python tools/make_sample_pdf.py
python -m finrag.ingest.run_ingest --pdf data/raw/sample_bank_ar.pdf \
  --entity "Bharat National Bank" --fiscal-year FY2023-24 --also-naive --init
pytest tests -q
python eval/build_benchmark.py && python eval/run_eval.py --arms all --split test
```

**CHECKPOINT 1:** 35 tests pass; `eval/results/comparison.md` exists.
*If the DB won't start:* install Postgres locally and `CREATE EXTENSION vector;` — the code only needs a DSN.

## Block 2 (3 h) — Ingest real Indian filings

Download **6–10 PDFs** (this is the single highest-value task of the 48 hours):

| Source | What to get | Why |
|---|---|---|
| rbi.org.in | RBI Annual Report 2023-24 + 2022-23 | dense statistical tables, the regulatory anchor |
| Bank investor-relations pages | SBI, HDFC Bank, ICICI annual reports | consolidated **and** standalone statements |
| nseindia.com / bseindia.com | 2–3 quarterly results PDFs | short docs, fast iteration |

Start with **page ranges**, not whole 300-page reports:

```bash
python -m finrag.ingest.run_ingest \
  --pdf data/raw/rbi_ar_2024.pdf --entity "Reserve Bank of India" \
  --doc-type rbi_annual_report --fiscal-year FY2023-24 --max-pages 40 --also-naive
```

Then **inspect what the parser produced** — this is where real PDFs differ from synthetic ones:

```python
PYTHONPATH=src python -c "
from finrag.ingest.pdf_parse import parse_pdf
from finrag.ingest.metadata import ParsedTable
d = parse_pdf('data/raw/rbi_ar_2024.pdf', max_pages=40)
for t in [e for e in d.elements if isinstance(e, ParsedTable)][:10]:
    print(t.table_id, t.extractor, '| unit:', t.unit_name, '| cap:', t.caption[:60])
    print('   indents:', t.row_indents[:8]); print('   ', t.grid[0][:5])
"
```

**Likely fixes needed (budget 1 h):**
- Tables missed → loosen `min_fill` in `_grid_score`, or `PARSER=docling` (`pip install docling`, downloads models once).
- Unit banner missed → widen `look_above` in `caption_and_unit`.
- Indentation flat → the report uses bullets/dashes instead of indents; extend `row_label_path`.

**CHECKPOINT 2:** `GET /stats` shows ≥ 2000 structured chunks with non-null `unit_name` on financial tables. Screenshot this.

## Block 3 (2 h) — Real embeddings + reranker

```bash
export EMBED_BACKEND=st EMBED_MODEL=BAAI/bge-small-en-v1.5 EMBED_DIM=384
# schema is created with vector(:dim) — if you change model, change EMBED_DIM and re-init
python -m finrag.ingest.run_ingest --pdf data/raw/*.pdf --entity "..." --also-naive --init
```

First run downloads ~130 MB (bge-small) + ~90 MB (MiniLM reranker). **Do this before you need it** — do not discover the download at 23:00.

*No GPU?* CPU is fine at this corpus size. *No disk/network?* `EMBED_BACKEND=hash` still demos the full logic; say so honestly in the slides.

**CHECKPOINT 3:** retrieval returns the right table row for 5 questions you ask by hand.

## Block 4 (2 h) — Connect a real LLM

Cheapest reliable route is **Ollama locally**:
```bash
ollama pull qwen2.5:7b-instruct
export LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen2.5:7b-instruct
export VERIFY_BACKEND=llm LLM_OFFLINE=0
```
Any OpenAI-compatible endpoint works — just change the three variables.

```bash
PYTHONPATH=src python -c "
from finrag.store import db; from finrag.pipeline import FinRAG
with db.connect() as c:
    r = FinRAG(c).answer('What was the Gross NPA ratio of scheduled commercial banks in 2023-24?')
    print(r['decision'], r['groundedness']); print(r['answer'])"
```

**CHECKPOINT 4:** one real grounded answer with `[C#]` citations, page numbers, and a groundedness score.

## Block 5 (2 h) — Build the real benchmark

```bash
python eval/build_benchmark.py --out eval/benchmark_ifhb.jsonl --max-items 200
```

Then **hand-audit 25 items** (30 min, non-negotiable — it converts "auto-generated" into "curated and graded", which is what your Review-1 objective promised). Open the JSONL, fix any wrong `gold_value`, delete nonsense questions, and record: *"25 of 200 items manually audited; 3 corrected."*

Add ~10 hand-written multi-hop questions that the generator cannot produce ("Which segment contributed the largest share of revenue, and by how much did it grow?").

**CHECKPOINT 5:** `eval/benchmark_ifhb.jsonl` with ≥ 120 items across 5 types, audit note recorded.

**END OF DAY 1 — you should be able to say:** real Indian filings ingested, real models running, benchmark built.

---

# DAY 2 — Numbers, calibration, demo, slides

## Block 6 (2 h) — Full ablation study

```bash
python eval/run_eval.py --arms all --split test --out-dir eval/results
cat eval/results/comparison.md
```

This produces **your headline table**. Expect roughly:

| Arm | What it shows |
|---|---|
| B0_naive_noguard | the strawman — naive chunking destroys retrieval |
| B1_struct_noguard | structure-aware chunking lifts accuracy; hallucinations remain |
| B2_struct_guard | guardrail cuts hallucination rate and AURC; costs some coverage |
| A1/A2/A3 | each component's individual contribution |

**Report honestly.** If the guardrail costs coverage, say so — the coverage/risk trade-off *is* the finding. A panel trusts a measured trade-off far more than a suspiciously perfect result.

## Block 7 (1.5 h) — Calibrate the thresholds

```bash
python eval/calibrate.py --records eval/results/B2_struct_guard_records.json --target-risk 5
# put the chosen TAU_LOW / TAU_HIGH in .env, then re-run B2 to confirm
```

This is your direct answer to *"why 0.95?"* — you now have a sweep, a target risk, and a chosen operating point. Put the risk–coverage curve on a slide.

## Block 8 (1 h) — RAGAS second opinion (optional but strong)

```bash
python -m venv .venv-eval && source .venv-eval/bin/activate
pip install -r requirements-eval.txt        # pins langchain-community<0.4 (ragas 0.4.x breaks otherwise)
OPENAI_API_KEY=... python eval/ragas_eval.py --records eval/results/B2_struct_guard_records.json
```
*If you have no API key:* skip it, and say in the slides that RAGAS integration is implemented and scheduled for Review-3. Do not fake numbers.

## Block 9 (2 h) — Demo surface

```bash
PYTHONPATH=src uvicorn finrag.api:app --port 8000 &   # /docs gives Swagger UI
streamlit run ui/app.py
```

Rehearse **four questions, in this order**:
1. **Answerable lookup** → VERIFIED badge, claim table, cited page.
2. **YoY comparison** → derived claim recomputed from operands.
3. **Unanswerable** ("carbon emission intensity") → structured refusal. *This is your money shot.*
4. **Guarded vs unguarded side-by-side** → the baseline invents, the guarded arm abstains.

Record a 3-minute screen capture as insurance against live-demo failure.

## Block 10 (2 h) — Slides

Suggested 12 slides:

| # | Slide | Content |
|---|---|---|
| 1 | Title | unchanged |
| 2 | Gap restated | one line: Indian filings break Western-benchmark assumptions |
| 3 | **Architecture** | the real one — three retrieval arms, guardrail, repair loop |
| 4 | **Ingestion internals** | lattice/stream, header flattening, indent→hierarchy, unit inheritance (§1 of the tech spec) |
| 5 | **Storage & retrieval** | HNSW m=16/ef=64, RRF k=60, cross-encoder, small-to-big (§2) |
| 6 | **Claim verification** | the row-competition worked example — Net NPA vs Gross NPA (§3.2d) |
| 7 | **Abstention policy** | the two-threshold state machine (§4) |
| 8 | **Benchmark** | 5 item types, scale traps, scope traps, gold evidence ids |
| 9 | **Results** | the ablation table |
| 10 | **Calibration** | risk–coverage curve, chosen operating point |
| 11 | Demo | live or recorded |
| 12 | Remaining work | Review-3 plan |

**Rule:** every slide 4–7 must name techniques and parameters. That is precisely what Review-1 was missing.

## Block 11 (1.5 h) — Report update + freeze

Rewrite §4 (Scope) and §7 (Methodology) of your report using `docs/REVIEW2_TECHNICAL_SPEC.md` — it is written to be pasted in. Add a "Progress against objectives" table:

| Objective (Review-1) | Status | Evidence |
|---|---|---|
| Domain-specific ingestion pipeline | **Done** | `ingest/`, N tables parsed, 9 chunking tests |
| AI guardrails / abstention | **Done** | `guard/`, 16 guardrail tests, ablation B1 vs B2 |
| Localized benchmark | **Done (v1)** | `eval/benchmark_ifhb.jsonl`, N items, 25 audited |
| Rigorous evaluation | **In progress** | ablation + calibration done; RAGAS pending |
| FastAPI + UI + CI | **Done** | `api.py`, `ui/app.py`, `.github/workflows/ci.yml` |

That is comfortably ≥ 50 %.

---

## Time-boxed fallbacks (use them without guilt)

| If this breaks | Do this |
|---|---|
| Real PDFs parse badly | demo on 2 clean PDFs + the synthetic one; show the failure case as "identified limitation, Review-3 work" |
| No LLM endpoint | `LLM_OFFLINE=1`; the guardrail logic is what you're presenting, and it's model-agnostic |
| Embedding download fails | `EMBED_BACKEND=hash` |
| Docker unavailable | local Postgres + `CREATE EXTENSION vector` |
| Ablation run too slow | `--limit 40 --arms B1_struct_noguard,B2_struct_guard` |
| Streamlit misbehaves | demo via `/docs` Swagger UI |

---

## Three questions the panel will ask — and your answers

**"How is this different from any RAG tutorial?"**
Four mechanisms no tutorial has: rounding-aware numeric tolerance tied to asserted precision, lakh/crore scale normalisation, row-competition conflict detection, and retrieval-miss vs fabrication disambiguation driving a repair loop instead of a blanket refusal.

**"Why 0.95 / how did you pick the threshold?"**
I didn't pick it — I swept (τ_low, τ_high) on the dev split and chose the point maximising accuracy subject to selective risk ≤ 5 %. Here is the risk–coverage curve and the AURC.

**"Your system abstains a lot. Isn't that just refusing to work?"**
That's why I report coverage and selective risk, not accuracy alone. Arm B1 answers more and is wrong more often; B2 trades X % coverage for a Y-point drop in hallucination rate. In financial QA that trade is the product decision, and it's now measurable rather than accidental.

---

## Beyond Review-2 (put this on slide 12)

1. Scale corpus to 30+ filings; rebuild HNSW and report index build time and p95 latency.
2. NLI verifier (`VERIFY_BACKEND=nli`) vs LLM judge — cost/latency/agreement comparison.
3. Langfuse tracing dashboards (`LANGFUSE_ENABLED=1`; decorators are already in `pipeline.py`).
4. Cross-document multi-hop (bank A vs bank B, FY over FY).
5. Inter-annotator agreement on a 100-item human-verified benchmark subset.
6. Compare against FinanceBench/PHANTOM items to quantify the India-specific gap empirically.
