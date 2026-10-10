"""Single source of truth for every tunable. Nothing is hard-coded downstream."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


def _f(k: str, d: float) -> float:
    return float(os.getenv(k, d))


def _i(k: str, d: int) -> int:
    return int(os.getenv(k, d))


@dataclass
class Config:
    pg_dsn: str = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/finrag")

    # --- embeddings ---
    embed_backend: str = os.getenv("EMBED_BACKEND", "st")       # st | hash
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = _i("EMBED_DIM", 384)
    embed_query_prefix: str = os.getenv(
        "EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

    # --- retrieval ---
    k_dense: int = _i("K_DENSE", 40)
    k_lexical: int = _i("K_LEXICAL", 40)
    rrf_k: int = _i("RRF_K", 60)            # reciprocal rank fusion constant
    k_fused: int = _i("K_FUSED", 30)        # candidates handed to the reranker
    k_final: int = _i("K_FINAL", 6)         # chunks put in the prompt
    hnsw_ef_search: int = _i("HNSW_EF_SEARCH", 100)
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = os.getenv("USE_RERANKER", "1") == "1"
    expand_to_parent: bool = os.getenv("EXPAND_PARENT", "1") == "1"
    mmr_lambda: float = _f("MMR_LAMBDA", 0.7)

    # --- generation ---
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    temperature: float = _f("TEMPERATURE", 0.0)
    max_tokens: int = _i("MAX_TOKENS", 700)

    # --- guardrail ---
    verify_backend: str = os.getenv("VERIFY_BACKEND", "llm")    # none | llm | nli
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
    nli_entail_threshold: float = _f("NLI_ENTAIL", 0.60)
    numeric_rel_tol: float = _f("NUMERIC_REL_TOL", 0.0)         # 0 = rounding tolerance only
    w_numeric: float = _f("W_NUMERIC", 2.0)                     # numeric claims weigh double

    # --- abstention policy (calibrate on dev split; see eval/calibrate.py) ---
    tau_rerank: float = _f("TAU_RERANK", 0.20)   # min top-1 rerank score to even generate
    tau_high: float = _f("TAU_HIGH", 0.90)       # >= -> answer
    tau_low: float = _f("TAU_LOW", 0.60)         # <  -> abstain; between -> re-retrieve/flag
    hard_numeric_gate: bool = os.getenv("HARD_NUMERIC_GATE", "1") == "1"
    max_iters: int = _i("MAX_ITERS", 2)

    # --- observability ---
    langfuse_enabled: bool = os.getenv("LANGFUSE_ENABLED", "0") == "1"

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("llm_api_key", None)
        return d


CFG = Config()
