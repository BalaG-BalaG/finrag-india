"""Retrieval: query analysis -> dense + lexical -> RRF -> cross-encoder -> MMR -> parent expansion.

Pipeline stages (each one is individually ablatable via RetrievalConfig flags, which
is what produces the Review-2 ablation table):
  S1 query analysis   : pull FY / scope / entity / metric out of the question and turn
                        them into SQL metadata filters (pre-filtering, not post-filtering,
                        so HNSW+filter recall stays high)
  S2 dense arm        : pgvector HNSW cosine kNN over contextualised chunk text
  S3 lexical arm      : Postgres ts_rank_cd; catches exact tokens ("Gross NPA", "CRAR")
                        and literal figures that dense vectors smear together
  S4 fusion           : Reciprocal Rank Fusion, score = sum 1/(k + rank), k=60
  S5 rerank           : cross-encoder (query, passage) scoring - the single biggest
                        precision win on adjacent-year confusions
  S6 diversify        : MMR so the context window isn't 6 near-duplicate rows
  S7 expand           : small-to-big - a matched table_row is swapped for its parent
                        table so the LLM sees the header band and neighbouring rows
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import CFG
from ..embed import embed_queries, rerank
from ..ingest.metadata import detect_fiscal_years, detect_scope
from ..store import db

_METRIC_HINTS = [
    "gross npa", "net npa", "crar", "capital adequacy", "provision coverage", "casa",
    "net interest margin", "nim", "net profit", "profit after tax", "pat", "total income",
    "operating profit", "earnings per share", "eps", "book value", "dividend", "roa", "roe",
    "total assets", "advances", "deposits", "borrowings", "cash flow", "revenue from operations",
    "gdp", "inflation", "repo rate", "crr", "slr", "forex reserves", "credit growth",
]


@dataclass
class RetrievalConfig:
    use_dense: bool = True
    use_lexical: bool = True
    use_rrf: bool = True
    use_rerank: bool = True
    use_mmr: bool = True
    expand_parent: bool = True
    pipeline: str = "structured"
    k_final: int = CFG.k_final


@dataclass
class QueryPlan:
    question: str
    filters: dict = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    lexical_query: str = ""
    comparative: bool = False


def analyse_query(question: str, strict_filters: bool = False) -> QueryPlan:
    q = question.lower()
    fys = detect_fiscal_years(question)
    scope = detect_scope(question)
    metrics = [m for m in _METRIC_HINTS if m in q]
    comparative = bool(re.search(r"\b(yoy|y-o-y|year[- ]on[- ]year|compared|versus|vs\.?|growth|change|increase|decrease)\b", q))
    filters: dict = {"pipeline": "structured"}
    # A single-year question can be pre-filtered; a comparison must NOT be (it needs both years).
    if strict_filters and len(fys) == 1 and not comparative:
        filters["fiscal_year"] = fys[0]
    if scope:
        filters["scope"] = scope
    lex = " ".join(metrics) if metrics else re.sub(r"[^\w\s%.-]", " ", question)
    return QueryPlan(question=question, filters=filters, metrics=metrics,
                     lexical_query=lex.strip(), comparative=comparative)


def rrf_fuse(ranked_lists: list[list[dict]], k: int = None, weights: list[float] | None = None) -> list[dict]:
    """Reciprocal Rank Fusion: rank-based, so dense cosine and ts_rank_cd never
    need score normalisation (their scales are not comparable)."""
    k = k or CFG.rrf_k
    weights = weights or [1.0] * len(ranked_lists)
    agg: dict[str, dict] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, row in enumerate(lst, start=1):
            cid = row["chunk_id"]
            cur = agg.setdefault(cid, {**row, "rrf": 0.0, "ranks": {}})
            cur["rrf"] += w / (k + rank)
            cur["ranks"][f"r{len(cur['ranks'])}"] = rank
            for f in ("dense_score", "lex_score"):
                if f in row:
                    cur[f] = row[f]
    return sorted(agg.values(), key=lambda r: -r["rrf"])


def mmr_select(cands: list[dict], k: int, lam: float = None) -> list[dict]:
    """Maximal Marginal Relevance on token Jaccard - cheap, no second embedding pass."""
    lam = CFG.mmr_lambda if lam is None else lam
    def toks(r):
        return set(re.findall(r"[a-z0-9.%]+", (r.get("content") or "").lower()))
    chosen: list[dict] = []
    pool = list(cands)
    while pool and len(chosen) < k:
        best, best_score = None, -1e9
        for r in pool:
            rel = r.get("rerank_score", r.get("rrf", 0.0))
            div = max((len(toks(r) & toks(c)) / (len(toks(r) | toks(c)) or 1) for c in chosen), default=0.0)
            s = lam * rel - (1 - lam) * div
            if s > best_score:
                best, best_score = r, s
        chosen.append(best)
        pool.remove(best)
    return chosen


def retrieve(conn, question: str, rc: RetrievalConfig | None = None,
             extra_filters: dict | None = None) -> dict:
    rc = rc or RetrievalConfig()
    plan = analyse_query(question)
    plan.filters["pipeline"] = rc.pipeline
    if extra_filters:
        plan.filters.update(extra_filters)

    lists, arms = [], {}
    if rc.use_dense:
        qv = embed_queries([question])[0]
        d = db.dense_search(conn, qv, CFG.k_dense, plan.filters)
        lists.append(d)
        arms["dense"] = len(d)
    if rc.use_lexical and plan.lexical_query:
        try:
            l = db.lexical_search(conn, plan.lexical_query, CFG.k_lexical, plan.filters)
        except Exception:
            l = []
        lists.append(l)
        arms["lexical"] = len(l)

    if not lists:
        return {"plan": plan, "candidates": [], "contexts": [], "arms": arms}
    fused = rrf_fuse(lists) if (rc.use_rrf and len(lists) > 1) else [
        {**r, "rrf": 1.0 / (i + 1)} for i, r in enumerate(lists[0])]
    cands = fused[:CFG.k_fused]

    if rc.use_rerank and cands:
        scores = rerank(question, [c["content"] for c in cands])
        for c, s in zip(cands, scores):
            c["rerank_score"] = float(s)
        cands.sort(key=lambda r: -r["rerank_score"])
    else:
        for c in cands:
            c["rerank_score"] = c.get("rrf", 0.0)

    top = mmr_select(cands, rc.k_final) if rc.use_mmr else cands[:rc.k_final]

    contexts = top
    if rc.expand_parent:
        parents = db.get_parents(conn, [t.get("parent_id") for t in top])
        seen, expanded = set(), []
        for t in top:
            pid = t.get("parent_id")
            src = parents.get(pid) if pid and pid in parents else t
            key = src["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            expanded.append({**src, "rerank_score": t.get("rerank_score", 0.0),
                             "matched_chunk_id": t["chunk_id"],
                             "matched_row": (t.get("meta") or {}).get("row_label", "")})
        contexts = expanded

    return {"plan": plan, "candidates": cands, "contexts": contexts, "arms": arms,
            # tau_rerank is calibrated on cross-encoder scores; RRF scores live on a
            # different scale, so the gate must know which produced this number.
            "scored_by": "cross_encoder" if (rc.use_rerank and CFG.use_reranker) else "rrf",
            "top_rerank": max([c.get("rerank_score", 0.0) for c in cands], default=0.0)}


def format_context(contexts: list[dict]) -> str:
    """Numbered evidence blocks; the LLM must cite these ids, which is what makes
    claim-level verification possible downstream."""
    out = []
    for i, c in enumerate(contexts, start=1):
        head = " | ".join([b for b in [c.get("entity"), c.get("section_path"),
                                       c.get("scope"), c.get("unit_name") and f"({c['unit_name']})",
                                       f"p.{c.get('page_no')}"] if b])
        out.append(f"[C{i}] {head}\n{c['content']}")
    return "\n\n".join(out)
