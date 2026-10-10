# FinRAG-India

**Hallucination-aware retrieval-augmented generation for Indian financial report Q&A.**

A question-answering system over RBI annual reports and Indian company filings, built around a
**deterministic numeric verification layer** that checks every figure an LLM produces against the
source table cell it claims to come from — and abstains when it cannot.

No second language model sits in the numeric verification path. Every check is arithmetic, so the
verdict is reproducible, auditable, and works with any generation backend including closed APIs.

---

## Headline result

| | Naive chunking, no guard | Structure-aware, no guard | Structure-aware + guard |
|---|---|---|---|
| Answer accuracy | 33.3% | **55.6%** | 22.2% |
| **Hallucination rate** | 70.0% | 42.5% | **5.0%** |
| Coverage | 96.3% | 96.3% | 29.6% |
| Selective risk | 75.7% | 53.1% | **25.0%** |
| Honest abstention | 2 / 13 | 7 / 13 | **13 / 13** |
| Recall@5 | 0.0 | **100.0** | **100.0** |

*40-item test subsample (27 answerable, 13 unanswerable) of the IFHB benchmark, RBI Annual Report
2023-24 + a controlled synthetic filing. Generator: `qwen2.5:3b-instruct` via Ollama, temperature 0.*

**Read the coverage row.** A system that abstains on everything scores 0% hallucination, so the
reduction only means something next to the coverage cost. The current operating point is
over-conservative, and [§ Limitations](#limitations) says so.

Retrieval, measured independently on the full test split:

| Metric | Lexical hashing | bge-small-en-v1.5 | Δ |
|---|---|---|---|
| Recall@1 | 36.51 | **73.02** | +36.5 (2.0×) |
| Recall@5 | 74.60 | **86.51** | +11.9 |
| MRR | 0.528 | **0.796** | +0.268 |
| nDCG@10 | 0.585 | **0.817** | +0.232 |

---

## Why a generic RAG pipeline fails here

| Failure mode in Indian financial QA | Mechanism in this repo |
|---|---|
| 512-token windows shred table rows away from their headers | lattice/stream table detection, multi-level header flattening with span fill, row hierarchy recovered from PDF x-coordinates, unit-banner inheritance, multi-page stitching |
| Dense vectors cannot separate 2.24% from 2.78% | three-arm hybrid — dense + `ts_rank_cd` lexical + numeric-fact index — fused by RRF, reranked by a cross-encoder |
| The model quotes a **real** number from the **wrong row** | **row-competition test**: a match is rejected when another row in the same chunk fits the claim's metric better |
| RBI annexes invert the axes — years on rows, metrics in columns | **period-competition test**: layout is detected and the matched row's *period* is compared instead of its metric |
| Lakh/crore misread is a 100× error | scale normalisation to absolute rupees on both sides of every comparison |
| Legitimate rounding flagged as fabrication | tolerance = half a unit of the **last digit the answer asserts** — 2.2 ✓ for a source of 2.24, 2.2413 ✗ |
| A retrieval miss looks identical to a fabrication | corpus-wide numeric-fact index: present elsewhere = recoverable miss, absent everywhere = invented |
| "I don't know" is never said | two-threshold selective-prediction policy with a hard numeric gate and a bounded repair loop |

The period-keyed layout problem is, as far as the 15 papers surveyed go, undocumented. It was found
empirically because it broke the verifier.

---

## Quickstart — 5 minutes, fully offline

No model downloads, no API keys, deterministic. This is the path to use if you are reviewing the repo.

```bash
docker compose up -d db
pip install -r requirements.txt

export PYTHONPATH=src EMBED_BACKEND=hash LLM_OFFLINE=1 VERIFY_BACKEND=none
# Windows PowerShell: $env:PYTHONPATH="src"; $env:EMBED_BACKEND="hash"; ...

python tools/make_sample_pdf.py
python -m finrag.ingest.run_ingest --pdf data/raw/sample_bank_ar.pdf \
    --entity "Bharat National Bank" --fiscal-year FY2023-24 --also-naive --init
pytest tests -q                       # 46 tests
python eval/build_benchmark.py
python eval/run_eval.py --arms all --split test
```

`make all` runs the same sequence.

### With real models

```bash
export EMBED_BACKEND=st EMBED_MODEL=BAAI/bge-small-en-v1.5 EMBED_DIM=384
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b-instruct \
       LLM_API_KEY=ollama LLM_OFFLINE=0 VERIFY_BACKEND=llm

uvicorn finrag.api:app --port 8000    # Swagger UI at /docs
streamlit run ui/app.py               # guarded vs baseline, side by side
```

CPU inference costs 66–250 s per query. Budget accordingly.

---

## Where each claim lives in the code

If you are checking this against the dissertation, these are the files to open.

| Claim | File | Function |
|---|---|---|
| Rounding-aware, scale-aware numeric matching | `src/finrag/numeric.py` | `numbers_match` |
| Row hierarchy recovered from PDF geometry | `src/finrag/ingest/pdf_parse.py` | `row_indents` |
| Header flattening, small-to-big chunking | `src/finrag/ingest/table_chunker.py` | — |
| Typed atomic claim decomposition | `src/finrag/guard/claims.py` | `extract_claims`, `split_claims_rule` |
| Wrong-row / wrong-period detection | `src/finrag/guard/verifier.py` | `_row_conflict` |
| Groundedness score + the `numeric_ok` boolean | `src/finrag/guard/verifier.py` | `groundedness` |
| Abstention band and the hard numeric gate | `src/finrag/guard/policy.py` | `decide` |
| Deterministic repair of verified operands | `src/finrag/guard/repair.py` | `repair_derived` |
| Selective-prediction metrics, AURC | `eval/metrics.py` | — |
| Threshold calibration sweep | `eval/calibrate.py` | — |

---

## Repository layout

```
src/finrag/
  numeric.py            Indian number parsing, scale- and rounding-aware matching
  config.py             every threshold and model name, read from env
  embed.py              bi-encoder / cross-encoder (+ an offline hash backend for CI)
  ingest/               PDF -> structure-aware chunks
    pdf_parse.py          lattice vs stream detection, captions, unit banners, indents
    table_chunker.py      header flattening, row hierarchy, small-to-big chunks
    text_chunker.py       section-aware packing + the naive baseline arm
    metadata.py           fiscal-year normalisation, scope detection
  store/db.py           pgvector HNSW, tsvector lexical index, numeric_facts table
  retrieve/hybrid.py    query analysis -> RRF -> rerank -> MMR -> parent expansion
  generate/llm.py       OpenAI-compatible client + the citation-constrained prompt
  guard/                THE RESEARCH CORE
    claims.py             typed atomic claim decomposition
    verifier.py           numeric / derived / textual verification
    policy.py             abstention state machine
    repair.py             deterministic recomputation from verified operands
  pipeline.py           the full loop, with a trace record per query
  api.py                FastAPI
eval/                   benchmark builder, metrics, ablations, calibration, RAGAS adapter
sql/                    schema + a psql walkthrough of the storage layer
tests/                  46 unit tests
tools/verifier_check.py live regression suite against the real indexed corpus
ui/app.py               Streamlit: guarded vs unguarded, claim-level verdicts
docs/                   technical specification
```

---

## Testing

Three tiers, each catching what the others cannot.

**Tier 1 — 46 unit tests**, deterministic, no database.

| File | n | Pins |
|---|---|---|
| `test_numeric.py` | 7 | Indian grouping; periods not read as quantities; precision-aware tolerance; lakh/crore swap |
| `test_chunking.py` | 9 | span-filled headers; geometric indent hierarchy; multi-period rows; continuation stitching |
| `test_guard.py` | 16 | one test per documented hallucination mode, including *legitimate rounding is NOT flagged* |
| `test_retrieval.py` | 4 | RRF ordering; comparative questions exempt from year filtering; MMR dedup |
| `test_repair.py` | 10 | 4 correct computation, **6 correct refusal** |

**Tier 2 — live regression suite** (`tools/verifier_check.py`): six hand-written claims with known
verdicts, run against the real indexed corpus. Moved 4/6 → 6/6 after two verifier fixes. This tier
caught the period-keyed layout failure, which was invisible to fixtures.

**Tier 3 — benchmark**: 133 items, six ablation arms.

**CI**: GitHub Actions provisions a pgvector service container and runs unit tests plus a full
ingest → retrieve → guard → evaluate smoke test on every push, using deterministic offline backends.

---

## The benchmark

**IFHB v1** — 133 items generated from parsed table cells, each carrying a gold value *and* gold
evidence chunk ids, which is what makes retrieval metrics computable. Five item types:

- `lookup` — direct cell retrieval
- `yoy` — comparative, requires computation
- `unanswerable` — the document does not disclose it
- `scale_trap` — asks to confirm a real figure stated in the **wrong unit**
- `scope_trap` — asks for a statement scope the document does not publish

The last two are the India-specific contribution. Neither FinanceBench nor PHANTOM contains an
equivalent item, because US GAAP filings have no lakh/crore ambiguity and no consolidated/standalone
duality within one document.

---

## Limitations

Stated plainly, because the numbers above are not the whole picture.

- **Over-abstention is 70.4%** and unresolved. An audit of the 8 refusals on answerable dev-split
  questions found 5 justified (the model quoted both operands and never subtracted) and 3 false
  rejections caused by verifier defects. Two of the three are fixed.
- **End-to-end metrics are on a 40-item subsample**, because CPU inference costs 66–250 s per query.
- **Benchmark gold is programmatic.** 3 of 6 comparative items were invalid — row labels were list
  markers rather than metric names. Found by audit; the human verification pass is outstanding.
- **Two runs of 20 questions at temperature 0 produced different results.** Repeated runs are
  required at this sample size.
- **RBI's two-column prose interleaves** during text extraction, degrading narrative chunks. Tables
  are unaffected.
- **One 3B generator, two documents.** No generalisation claim is made.
- **RAGAS is integrated but unrun** — it needs credit on an external API, and it was always intended
  as a secondary signal, since RAGAS faithfulness is itself an LLM judge and imprecise about digits.

---

## Citation

```bibtex
@misc{balaji2026finragindia,
  author = {Balaji P},
  title  = {FinRAG-India: Hallucination-aware Retrieval-Augmented Generation
            for Indian Financial Report Question Answering},
  year   = {2026},
  note   = {M.Sc. Artificial Intelligence and Machine Learning dissertation,
            Vellore Institute of Technology},
  howpublished = {\url{https://github.com/BalaG-BalaG/finrag-india}}
}
```

Dissertation-I, PAAML698 · Department of Computer Applications, SCORE, VIT Vellore
Guide: Dr. Jenifa G, Assistant Professor Sr. Grade 2

## License

MIT — see [LICENSE](LICENSE).
