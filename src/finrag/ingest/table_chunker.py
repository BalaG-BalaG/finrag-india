"""Structure-aware chunking for financial tables.

Concretely, this replaces "512-token sliding window" with:
  1. header-band detection + multi-level header flattening (column path
     "Assets > FY2023-24 > Consolidated"),
  2. row-label hierarchy recovery from leading indentation ("Advances > Term loans"),
  3. unit-scale inheritance from the caption/unit banner ("(Rs. in crore)"),
  4. multi-page continuation stitching,
  5. dual granularity indexing (small-to-big):
        * table_row      -> one serialised fact per row  (precision, what we embed)
        * table_rowgroup -> ~N rows + header            (local comparison context)
        * table          -> full markdown grid          (parent, what the LLM reads)
Every child chunk carries parent_id so retrieval can expand row -> full table.
"""
from __future__ import annotations

import hashlib
import re

from ..numeric import parse_number, detect_unit_scale, format_indian
from .metadata import Chunk, ParsedTable, detect_fiscal_years, detect_scope

_NUMISH = re.compile(r"\d")


def _cid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _clean(cell: str | None) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def _numeric_ratio(row: list[str]) -> float:
    cells = [c for c in row if _clean(c)]
    if not cells:
        return 0.0
    return sum(1 for c in cells if parse_number(c) is not None) / len(cells)


def detect_header_rows(grid: list[list[str]], max_header: int = 4) -> int:
    """A header row is text-dominant and sits above the first numeric-dominant row."""
    n_header = 0
    for i, row in enumerate(grid[:max_header]):
        if _numeric_ratio(row) >= 0.4:
            break
        if i > 0 and not any(_clean(c) for c in row):
            break
        n_header = i + 1
    if n_header == 0 and grid:
        n_header = 1
    # A trailing all-blank-but-one header row is usually a spanner: keep it.
    return min(n_header, len(grid) - 1) if len(grid) > 1 else 1


def flatten_headers(grid: list[list[str]], n_header: int) -> list[str]:
    """Merge a multi-row header band into one path per column.

    Blank cells inherit from the left (merged/spanning cells), which is how
    '(Rs. in crore)' banners and 'As at 31 March' spanners are recovered.
    """
    width = max(len(r) for r in grid) if grid else 0
    band = [list(r) + [""] * (width - len(r)) for r in grid[:n_header]]
    for row in band:                       # forward-fill spans
        last = ""
        for j in range(width):
            v = _clean(row[j])
            if v:
                last = v
            else:
                row[j] = last
    paths = []
    for j in range(width):
        parts, seen = [], set()
        for row in band:
            v = _clean(row[j])
            if v and v.lower() not in seen:
                seen.add(v.lower())
                parts.append(v)
        paths.append(" > ".join(parts) if parts else f"col{j}")
    return paths


def _indent(cell: str | None) -> int:
    raw = (cell or "").replace("\n", " ")
    return len(raw) - len(raw.lstrip(" "))


def row_label_path(grid: list[list[str]], n_header: int, r: int,
                   indents: list[int] | None = None) -> str:
    """Rebuild 'Advances > Term loans'.

    Indent depth comes from the PDF geometry when available (pdfplumber strips
    leading spaces), otherwise from leading whitespace in the cell string.
    A parent is the nearest row above with a strictly smaller depth, or a
    numeric-free 'section' row at the same depth.
    """
    def label(i: int) -> str:
        return _clean(grid[i][0] if grid[i] else "")

    def depth(i: int) -> int:
        if indents and i < len(indents):
            return indents[i]
        return _indent(grid[i][0] if grid[i] else "")

    cur, path = depth(r), [label(r)]
    for i in range(r - 1, n_header - 1, -1):
        if not label(i):
            continue
        ind = depth(i)
        is_section = _numeric_ratio(grid[i][1:]) == 0.0
        if ind < cur or (is_section and ind <= cur):
            path.insert(0, label(i))
            cur = ind
            if ind == 0 and not is_section:
                break
    return " > ".join([p for p in path if p])


def stitch_continuations(tables: list[ParsedTable]) -> list[ParsedTable]:
    """Merge 'table continued on next page' fragments (same width + header)."""
    out: list[ParsedTable] = []
    for t in tables:
        prev = out[-1] if out else None
        prev_end = (prev.page_end or prev.page_no) if prev else 0
        if prev is not None and prev_end <= t.page_no <= prev_end + 1:
            same_width = prev.grid and t.grid and len(prev.grid[0]) == len(t.grid[0])
            hdr_prev = [_clean(c).lower() for c in prev.grid[0]] if prev.grid else []
            hdr_cur = [_clean(c).lower() for c in t.grid[0]] if t.grid else []
            repeated_header = hdr_prev == hdr_cur
            headerless = _numeric_ratio(t.grid[0]) >= 0.4 if t.grid else False
            contd = bool(re.search(r"\(?\bcont(?:d|inued)\b\.?\)?", t.caption, re.I))
            if same_width and (repeated_header or headerless or contd):
                add_from = 1 if repeated_header else 0
                prev.grid.extend(t.grid[add_from:])
                if prev.row_indents and t.row_indents:
                    prev.row_indents.extend(t.row_indents[add_from:])
                prev.page_end = t.page_no
                continue
        out.append(t)
    return out


def table_to_markdown(grid: list[list[str]], n_header: int) -> str:
    width = max(len(r) for r in grid)
    cols = flatten_headers(grid, n_header)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * width]
    for row in grid[n_header:]:
        cells = [_clean(c) for c in (list(row) + [""] * (width - len(row)))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def serialise_row(cols: list[str], row: list[str], label_path: str,
                  unit_name: str | None) -> str:
    """One row -> one self-contained sentence, so a single embedding = a single fact.

    'Net profit > Attributable to owners | FY2023-24 (Rs. in crore): 12,345.67 ;
     FY2022-23 (Rs. in crore): 9,876.54'
    """
    unit = f" ({unit_name})" if unit_name else ""
    pairs = []
    for j in range(1, min(len(cols), len(row))):
        val = _clean(row[j])
        if not val:
            continue
        head = cols[j] if j < len(cols) else f"col{j}"
        pairs.append(f"{head}{unit if _NUMISH.search(val) else ''}: {val}")
    return f"{label_path} | " + " ; ".join(pairs) if pairs else label_path


def chunk_table(t: ParsedTable, doc_id: str, entity: str, doc_scope: str | None,
                rows_per_group: int = 6, pipeline: str = "structured") -> list[Chunk]:
    if not t.grid or len(t.grid) < 2:
        return []
    n_header = t.n_header_rows or detect_header_rows(t.grid)
    cols = flatten_headers(t.grid, n_header)
    header_line = " | ".join(cols)

    unit_name, unit_mult = t.unit_name, t.unit_multiplier
    if unit_name is None:
        unit_name, unit_mult = detect_unit_scale(t.caption + " " + header_line)
    banner = f"{t.caption} {t.section_path} {header_line}"
    scope = detect_scope(banner) or doc_scope
    fys = detect_fiscal_years(banner)
    page_tag = f"p.{t.page_no}" + (f"-{t.page_end}" if t.page_end and t.page_end != t.page_no else "")

    def ctx(extra: str = "") -> str:
        """Contextual chunk header (Anthropic-style contextual retrieval)."""
        bits = [entity, t.section_path, t.caption, extra,
                f"({unit_name})" if unit_name else "", scope or "", page_tag]
        return " | ".join([b for b in bits if b])

    parent_id = _cid(doc_id, t.table_id, "table")
    md = table_to_markdown(t.grid, n_header)
    chunks = [Chunk(
        chunk_id=parent_id, doc_id=doc_id, element_type="table",
        content=f"{t.caption}\n\n{md}" if t.caption else md,
        embed_text=f"{ctx()}\n{md}", page_no=t.page_no, section_path=t.section_path,
        table_id=t.table_id, scope=scope, unit_name=unit_name, unit_multiplier=unit_mult,
        entity=entity, fiscal_year=(fys[0] if len(fys) == 1 else ""), pipeline=pipeline,
        meta={"n_rows": len(t.grid) - n_header, "n_cols": len(cols),
              "columns": cols, "page_end": t.page_end, "fiscal_years": fys},
    )]

    body = t.grid[n_header:]
    for i, row in enumerate(body):
        if not any(_clean(c) for c in row):
            continue
        r_abs = i + n_header
        label = row_label_path(t.grid, n_header, r_abs, t.row_indents)
        if not label or _numeric_ratio(row[1:]) == 0.0:
            continue                                   # pure section heading row
        sent = serialise_row(cols, list(row), label, unit_name)
        row_fys = detect_fiscal_years(f"{header_line} {sent}") or fys
        chunks.append(Chunk(
            chunk_id=_cid(doc_id, t.table_id, "row", str(r_abs)), doc_id=doc_id,
            element_type="table_row", content=sent, embed_text=f"{ctx(label)}\n{sent}",
            page_no=t.page_no, section_path=t.section_path, parent_id=parent_id,
            table_id=t.table_id, scope=scope, unit_name=unit_name, unit_multiplier=unit_mult,
            entity=entity, fiscal_year=(row_fys[0] if len(row_fys) == 1 else ""), pipeline=pipeline,
            meta={"row_index": r_abs, "row_label": label, "columns": cols,
                  "fiscal_years": row_fys},
        ))

    for g in range(0, len(body), rows_per_group):
        block = body[g:g + rows_per_group]
        if not any(any(_clean(c) for c in r) for r in block):
            continue
        md_g = table_to_markdown([list(x) for x in t.grid[:n_header]] + [list(r) for r in block], n_header)
        chunks.append(Chunk(
            chunk_id=_cid(doc_id, t.table_id, "grp", str(g)), doc_id=doc_id,
            element_type="table_rowgroup", content=md_g, embed_text=f"{ctx()}\n{md_g}",
            page_no=t.page_no, section_path=t.section_path, parent_id=parent_id,
            table_id=t.table_id, scope=scope, unit_name=unit_name, unit_multiplier=unit_mult,
            entity=entity, fiscal_year=(fys[0] if len(fys) == 1 else ""), pipeline=pipeline,
            meta={"row_span": [g, g + len(block)], "columns": cols},
        ))
    return chunks


def describe_cell(cols: list[str], row_label: str, col_index: int, value: float,
                  unit_name: str | None) -> str:
    """Human-readable cell address, used by the verifier's evidence strings."""
    col = cols[col_index] if col_index < len(cols) else f"col{col_index}"
    return f"{row_label} / {col} = {format_indian(value)}{' ' + unit_name if unit_name else ''}"
