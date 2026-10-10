"""Section-aware prose chunking + the naive baseline used for the ablation.

`chunk_text_blocks`  -> structured arm: respects section headings, packs to a token
                        budget with sentence-boundary overlap, prepends a contextual
                        header (entity | section path | page).
`naive_chunk`        -> baseline arm: fixed 512-token sliding window over the raw
                        text dump, tables flattened to whitespace (i.e. exactly what
                        the report criticises). Both arms are stored in the SAME
                        table with pipeline='structured' / 'naive' so Review-2 can
                        show a like-for-like ablation.
"""
from __future__ import annotations

import hashlib
import re

from .metadata import Chunk, TextBlock, detect_fiscal_years, detect_scope

_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(₹])")


def approx_tokens(text: str) -> int:
    """~4 chars/token; avoids a tiktoken dependency in CI."""
    return max(1, len(text) // 4)


def _cid(*p: str) -> str:
    return hashlib.sha1("|".join(p).encode()).hexdigest()[:16]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text or "") if s.strip()]


def pack_sentences(sents: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sents:
        st = approx_tokens(s)
        if cur and cur_tok + st > max_tokens:
            out.append(" ".join(cur))
            back, tok = [], 0
            for prev in reversed(cur):                 # sentence-boundary overlap
                t = approx_tokens(prev)
                if tok + t > overlap_tokens:
                    break
                back.insert(0, prev)
                tok += t
            cur, cur_tok = back, tok
        cur.append(s)
        cur_tok += st
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text_blocks(blocks: list[TextBlock], doc_id: str, entity: str,
                      doc_scope: str | None, max_tokens: int = 350,
                      overlap_tokens: int = 60) -> list[Chunk]:
    by_section: dict[str, list[TextBlock]] = {}
    for b in blocks:
        by_section.setdefault(b.section_path, []).append(b)

    chunks: list[Chunk] = []
    for section, items in by_section.items():
        text = "\n".join(i.text for i in items).strip()
        if not text:
            continue
        page = items[0].page_no
        for k, piece in enumerate(pack_sentences(split_sentences(text), max_tokens, overlap_tokens)):
            banner = f"{entity} | {section} | p.{page}"
            fys = detect_fiscal_years(f"{section} {piece}")
            chunks.append(Chunk(
                chunk_id=_cid(doc_id, section, str(page), str(k)), doc_id=doc_id,
                element_type="text", content=piece, embed_text=f"{banner}\n{piece}",
                page_no=page, section_path=section,
                scope=detect_scope(f"{section} {piece}") or doc_scope,
                entity=entity, fiscal_year=fys[0] if fys else "", pipeline="structured",
                meta={"fiscal_years": fys, "tokens": approx_tokens(piece)},
            ))
    return chunks


def naive_chunk(raw_text: str, doc_id: str, entity: str, page_map: list[tuple[int, int]] | None = None,
                window_tokens: int = 512, stride_tokens: int = 448) -> list[Chunk]:
    """Fixed-size sliding window, no structure awareness (baseline arm B0)."""
    words = raw_text.split()
    w_per_tok = 0.75
    win, stride = int(window_tokens * w_per_tok), int(stride_tokens * w_per_tok)
    chunks: list[Chunk] = []
    for k, start in enumerate(range(0, max(1, len(words)), stride)):
        piece = " ".join(words[start:start + win])
        if not piece.strip():
            continue
        page = 1
        if page_map:
            for pg, off in page_map:
                if off <= start:
                    page = pg
        chunks.append(Chunk(
            chunk_id=_cid(doc_id, "naive", str(k)), doc_id=doc_id, element_type="text",
            content=piece, embed_text=piece, page_no=page, entity=entity,
            pipeline="naive", meta={"window": [start, start + win]},
        ))
        if start + win >= len(words):
            break
    return chunks
