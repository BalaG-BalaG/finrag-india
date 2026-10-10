"""Embedding + reranking backends.

EMBED_BACKEND=st   -> sentence-transformers bi-encoder (bge-small-en-v1.5, 384-d,
                      asymmetric: queries get an instruction prefix, passages do not),
                      L2-normalised so cosine distance == 1 - dot product.
EMBED_BACKEND=hash -> deterministic hashing vectoriser. No downloads, no torch:
                      used by CI and by the offline smoke test so the whole
                      pipeline is runnable on a laptop with no internet.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from .config import CFG

_TOK = re.compile(r"[a-z0-9][a-z0-9.,%]*")


def _hash_embed(texts: list[str], dim: int) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * dim
        toks = _TOK.findall((t or "").lower())
        for i, tok in enumerate(toks):
            for gram in (tok, tok[:4], f"{toks[i-1]}_{tok}" if i else tok):
                h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
                v[h % dim] += 1.0 if h % 2 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


@lru_cache(maxsize=1)
def _st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CFG.embed_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    if CFG.embed_backend == "hash":
        return _hash_embed(texts, CFG.embed_dim)
    vecs = _st_model().encode(texts, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_queries(texts: list[str]) -> list[list[float]]:
    if CFG.embed_backend == "hash":
        return _hash_embed(texts, CFG.embed_dim)
    prefixed = [CFG.embed_query_prefix + t for t in texts]
    vecs = _st_model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CFG.rerank_model)


def rerank(query: str, passages: list[str]) -> list[float]:
    """Cross-encoder relevance scores. Falls back to lexical overlap when the
    model is unavailable (offline CI), so the pipeline never hard-fails."""
    if not passages:
        return []
    if not CFG.use_reranker or CFG.embed_backend == "hash":
        q = set(_TOK.findall(query.lower()))
        return [len(q & set(_TOK.findall(p.lower()))) / (len(q) or 1) for p in passages]
    try:
        scores = _cross_encoder().predict([(query, p) for p in passages])
        return [1 / (1 + math.exp(-float(s))) for s in scores]   # logits -> 0..1
    except Exception:
        q = set(_TOK.findall(query.lower()))
        return [len(q & set(_TOK.findall(p.lower()))) / (len(q) or 1) for p in passages]
