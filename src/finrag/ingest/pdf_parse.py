"""PDF -> ParsedDoc (reading-order list of TextBlock / ParsedTable).

Two interchangeable backends (set PARSER=pdfplumber|docling):

* pdfplumber (default, zero model download)
    - lattice extraction  : vertical/horizontal ruling lines  -> "lines" strategy
    - stream extraction   : whitespace/x-alignment clustering -> "text" strategy
    - both are tried per page and scored; the denser, better-formed grid wins
    - table bounding boxes are subtracted from the page before text extraction,
      so a table's digits never leak into a prose chunk (a classic corruption)
* docling (optional)
    - layout model + TableFormer cell-structure recovery; handles spanning cells
      and borderless tables far better on scanned/irregular RBI pages

Both emit the same dataclasses, so the rest of the pipeline is backend-agnostic.
"""
from __future__ import annotations

import os
import re

from ..numeric import detect_unit_scale
from .metadata import ParsedDoc, ParsedTable, TextBlock

LATTICE = {"vertical_strategy": "lines", "horizontal_strategy": "lines",
           "snap_tolerance": 4, "join_tolerance": 4, "intersection_tolerance": 5}
STREAM = {"vertical_strategy": "text", "horizontal_strategy": "text",
          "text_tolerance": 2, "intersection_tolerance": 8, "min_words_vertical": 2}

_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+|Note\s+\d+|Schedule\s+[0-9IVX]+|Annex(?:ure)?\s+[A-Z0-9]+)[.)\s-]+)?"
    r"([A-Z][A-Za-z0-9&(),'’\-/ ]{3,90})$"
)
_HEADINGY = re.compile(r"^(?:balance sheet|profit and loss|statement of|notes? (?:to|forming)|"
                       r"cash flow|schedule|annex|independent auditor|significant accounting)", re.I)


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not (3 <= len(s) <= 95) or s.endswith((".", ";", ",")):
        return False
    if _HEADINGY.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return True
    return bool(_HEADING.match(s)) and len(s.split()) <= 12


def _grid_score(grid, min_fill: float = 0.45) -> float:
    """Reject sparse/ragged false positives (the classic 'stream' failure mode).

    A real financial table is dense (most cells filled), rectangular, and has at
    least one numeric-dominant row. Whitespace clustering on a prose page produces
    wide, mostly-empty grids - those score negative and are dropped.
    """
    if not grid or len(grid) < 2:
        return -1.0
    width = max(len(r) for r in grid)
    if width < 2 or width > 25:
        return -1.0
    cells = len(grid) * width
    filled = sum(1 for r in grid for c in r if (c or "").strip())
    fill = filled / cells
    if fill < min_fill:
        return -1.0
    ragged = sum(1 for r in grid if len(r) != width) / len(grid)
    numeric_rows = sum(1 for r in grid if _numeric_row(r)) / len(grid)
    if numeric_rows < 0.2:
        return -1.0
    numeric_cells = sum(1 for r in grid for c in r
                        if (c or "").strip() and any(ch.isdigit() for ch in c))
    if numeric_cells < 6 or numeric_cells / max(filled, 1) < 0.25:
        return -1.0
    header_named = sum(1 for c in grid[0] if (c or "").strip())
    if header_named < 2:
        return -1.0
    return fill * (1 - ragged) * (1 + numeric_rows) * min(len(grid), 30) ** 0.5


def _numeric_row(row) -> bool:
    vals = [(c or "").strip() for c in row]
    vals = [v for v in vals if v]
    if len(vals) < 2:
        return False
    return sum(1 for v in vals if re.search(r"\d", v)) / len(vals) >= 0.5


def parse_pdf_pdfplumber(path: str, max_pages: int | None = None) -> ParsedDoc:
    import pdfplumber

    doc = ParsedDoc(path=path, n_pages=0, engine="pdfplumber")
    section = ""
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        doc.n_pages = len(pdf.pages)
        for pno, page in enumerate(pages, start=1):
            # Lattice first: ruling lines are hard evidence of a table. Stream is a
            # fallback for borderless tables and only fills regions lattice missed.
            boxes: list[tuple] = []
            kept: list[tuple] = []
            for name, settings in (("lattice", LATTICE), ("stream", STREAM)):
                found = []
                try:
                    for tb in page.find_tables(table_settings=settings):
                        grid = tb.extract()
                        found.append((_grid_score(grid), tb.bbox, grid, tb))
                except Exception:
                    continue
                for score, bbox, grid, tb in sorted(found, key=lambda x: -x[0]):
                    if score <= 0:
                        continue
                    x0, top, x1, bottom = bbox
                    if any(not (x1 <= b[0] or x0 >= b[2] or bottom <= b[1] or top >= b[3]) for b in boxes):
                        continue                  # already covered by a better detection
                    boxes.append(bbox)
                    kept.append((bbox, grid, name, tb))

            region = page
            for bbox in boxes:                     # carve tables out of the prose layer
                try:
                    region = region.outside_bbox(bbox)
                except Exception:
                    pass
            text = region.extract_text(x_tolerance=1.5, y_tolerance=3) or ""

            buf: list[str] = []
            for line in text.split("\n"):
                s = line.strip()
                if not s or re.fullmatch(r"(page\s*)?\|?\s*\d{1,4}\s*\|?", s, re.I):
                    continue
                if looks_like_heading(s):
                    if buf:
                        doc.elements.append(TextBlock(pno, " ".join(buf), section))
                        buf = []
                    section = s if len(s) < 60 else section
                    continue
                buf.append(s)
            if buf:
                doc.elements.append(TextBlock(pno, " ".join(buf), section))

            lines = page_lines(page)
            for i, (bbox, grid, name, tbl) in enumerate(sorted(kept, key=lambda k: k[0][1])):
                grid = [[(c or "") for c in row] for row in grid]
                caption, unit_name, unit_mult = caption_and_unit(lines, bbox)
                if unit_name is None:
                    unit_name, unit_mult = detect_unit_scale(" ".join(" ".join(r) for r in grid[:2]))
                indents = row_indents(page, tbl)
                doc.elements.append(ParsedTable(
                    table_id=f"p{pno}t{i}", page_no=pno, grid=grid, caption=caption,
                    section_path=section, unit_name=unit_name, unit_multiplier=unit_mult,
                    row_indents=indents, extractor=name,
                ))
    return doc


def page_lines(page) -> list[dict]:
    """Reconstruct visual lines with geometry: {top, x0, x1, text}."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
    rows: dict[int, list] = {}
    for w in words:
        rows.setdefault(round(w["top"] / 3.0), []).append(w)
    lines = []
    for _, ws in sorted(rows.items()):
        ws.sort(key=lambda w: w["x0"])
        lines.append({"top": min(w["top"] for w in ws), "x0": min(w["x0"] for w in ws),
                      "x1": max(w["x1"] for w in ws),
                      "text": " ".join(w["text"] for w in ws).strip()})
    return lines


_CAPTION_RE = re.compile(
    r"^(table|statement|schedule|annex(ure)?|appendix|chart|exhibit|note)\s*[-–]?\s*[0-9IVX]+", re.I)


def caption_and_unit(lines: list[dict], bbox, look_above: float = 90.0,
                     look_below: float = 26.0) -> tuple[str, str | None, float | None]:
    """Nearest caption line above the table + the unit banner ('(Rs. in crore)'),
    which in Indian reports sits on its own line above or just below the table."""
    x0, top, x1, bottom = bbox
    above = [l for l in lines if top - look_above <= l["top"] < top]
    below = [l for l in lines if bottom < l["top"] <= bottom + look_below]

    caption = ""
    for l in reversed(above):                       # nearest first
        if _CAPTION_RE.match(l["text"]):
            caption = l["text"]
            break
    if not caption and above:
        for l in reversed(above):
            t = l["text"]
            if 6 <= len(t) <= 140 and not re.fullmatch(r"[\d\s,.()%-]+", t):
                caption = t
                break

    unit_name = unit_mult = None
    for l in reversed(above + below):
        name, mult = detect_unit_scale(l["text"])
        if name:
            unit_name, unit_mult = name, mult
            break
    if not unit_name and caption:
        unit_name, unit_mult = detect_unit_scale(caption)
    return caption[:160], unit_name, unit_mult


def row_indents(page, table) -> list[int]:
    """Indentation depth of each row's first cell, recovered from word x-positions.

    pdfplumber strips leading spaces, so '  Interest on advances' loses its indent
    and the row hierarchy ('Interest earned > Interest on advances') would be lost.
    We measure the actual left edge of the first cell's text instead and bucket it
    into depth levels - this is what makes row_label_path work on real PDFs.
    """
    try:
        rows = table.rows
    except Exception:
        return []
    words = page.extract_words(x_tolerance=1.5, y_tolerance=3)
    lefts: list[float | None] = []
    for r in rows:
        cells = [c for c in r.cells if c]
        if not cells:
            lefts.append(None)
            continue
        cx0, ctop, cx1, cbottom = cells[0]
        inside = [w for w in words
                  if w["x0"] >= cx0 - 1 and w["x1"] <= cx1 + 1
                  and w["top"] >= ctop - 1 and w["bottom"] <= cbottom + 1]
        lefts.append(min((w["x0"] for w in inside), default=None))
    base = min([l for l in lefts if l is not None], default=0.0)
    out = []
    for l in lefts:
        out.append(0 if l is None else int(round(max(0.0, l - base) / 6.0)))
    return out


def parse_pdf_docling(path: str, max_pages: int | None = None) -> ParsedDoc:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DocItemLabel, TableItem, TextItem

    conv = DocumentConverter().convert(path)
    dl = conv.document
    doc = ParsedDoc(path=path, n_pages=len(getattr(dl, "pages", {}) or {}), engine="docling")
    section = ""
    ti = 0
    for item, _level in dl.iterate_items():
        page = 1
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", 1)
        if max_pages and page > max_pages:
            break
        if isinstance(item, TableItem):
            df = item.export_to_dataframe(doc=dl)
            grid = [[str(c) for c in df.columns.tolist()]] + df.astype(str).values.tolist()
            caption = item.caption_text(dl) or ""
            unit_name, unit_mult = detect_unit_scale(caption + " " + " ".join(grid[0]))
            doc.elements.append(ParsedTable(
                table_id=f"dl{ti}", page_no=page, grid=grid, caption=caption,
                section_path=section, unit_name=unit_name, unit_multiplier=unit_mult))
            ti += 1
        elif isinstance(item, TextItem):
            label = getattr(item, "label", None)
            if label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}:
                continue
            if label in {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}:
                section = item.text.strip()
                continue
            if item.text.strip():
                doc.elements.append(TextBlock(page, item.text.strip(), section))
    return doc


def parse_pdf(path: str, max_pages: int | None = None, engine: str | None = None) -> ParsedDoc:
    engine = engine or os.getenv("PARSER", "pdfplumber")
    if engine == "docling":
        try:
            return parse_pdf_docling(path, max_pages)
        except Exception as e:                      # graceful fallback keeps the demo alive
            print(f"[parse] docling failed ({e}); falling back to pdfplumber")
    return parse_pdf_pdfplumber(path, max_pages)


def raw_text(path: str, max_pages: int | None = None) -> tuple[str, list[tuple[int, int]]]:
    """Flat text dump + (page_no, word_offset) map, for the naive baseline arm."""
    import pdfplumber

    parts, page_map, n_words = [], [], 0
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages[:max_pages] if max_pages else pdf.pages, start=1):
            t = page.extract_text() or ""
            page_map.append((pno, n_words))
            n_words += len(t.split())
            parts.append(t)
    return "\n".join(parts), page_map
