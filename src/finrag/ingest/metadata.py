"""Shared data models + metadata detectors (fiscal year, statement scope)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

_MON = r"(?:march|mar\.?)"
_FY_PATTERNS = [
    (re.compile(r"\bFY\s?'?(\d{4})\s?[-–/]\s?(\d{2,4})\b", re.I), "span"),
    (re.compile(r"\b((?:19|20)\d{2})\s?[-–/]\s?(\d{2})\b"), "span"),
    (re.compile(r"\bFY\s?'?(\d{2,4})\b", re.I), "end"),
    (re.compile(rf"\b(?:ended|ending|as\s+at|as\s+on|as\s+of)\s+(?:on\s+)?(?:31(?:st)?\s+{_MON}|{_MON}\s+31(?:st)?),?\s+(\d{{4}})\b", re.I), "end"),
    (re.compile(r"\b31[./-]0?3[./-](\d{4})\b"), "end"),
    # bare period headers: "31 March 2024" / "March 31, 2024" as a column label
    (re.compile(rf"\b31(?:st)?\s+{_MON},?\s+(\d{{4}})\b", re.I), "end"),
    (re.compile(rf"\b{_MON}\s+31(?:st)?,?\s+(\d{{4}})\b", re.I), "end"),
]


def _fy(end_year: int) -> str:
    return f"FY{end_year - 1}-{end_year % 100:02d}"


def detect_fiscal_years(text: str) -> list[str]:
    """Normalise every period mention to 'FY2023-24' (Indian FY = 1 Apr - 31 Mar).

    Patterns are ordered most-specific first and each match consumes its character
    span, so "FY2023-24" is read once as a full span and not a second time as the
    bare year "FY2023" (which would wrongly yield FY2022-23).
    """
    found: list[tuple[int, str]] = []
    consumed: list[tuple[int, int]] = []
    for rx, kind in _FY_PATTERNS:
        for m in rx.finditer(text or ""):
            if any(m.start() < e and s < m.end() for s, e in consumed):
                continue
            if kind == "span":
                y1, y2 = int(m.group(1)), int(m.group(2)) % 100
                if (y1 + 1) % 100 != y2:
                    continue
                fy = _fy(y1 + 1)
            else:
                y = int(m.group(1))
                y = y + 2000 if y < 100 else y
                if not (1990 <= y <= 2100):
                    continue
                fy = _fy(y)
            consumed.append((m.start(), m.end()))
            found.append((m.start(), fy))
    seen, out = set(), []
    for _, fy in sorted(found, key=lambda x: x[0]):
        if fy not in seen:
            seen.add(fy)
            out.append(fy)
    return out


def detect_scope(text: str) -> str | None:
    """Indian annual reports carry BOTH statement sets; mixing them is a top error source."""
    t = (text or "").lower()
    c, s = t.count("consolidated"), t.count("standalone")
    if c == s:
        return None
    return "consolidated" if c > s else "standalone"


@dataclass
class TextBlock:
    page_no: int
    text: str
    section_path: str = ""


@dataclass
class ParsedTable:
    table_id: str
    page_no: int
    grid: list[list[str]]                 # raw cell grid, header rows included
    caption: str = ""
    section_path: str = ""
    unit_name: str | None = None          # 'crore', 'lakh', 'million' ...
    unit_multiplier: float | None = None
    page_end: int | None = None           # set when a multi-page table was stitched
    n_header_rows: int = 1
    row_indents: list[int] = field(default_factory=list)   # geometric indent depth per row
    extractor: str = ""                   # lattice | stream | docling


@dataclass
class ParsedDoc:
    path: str
    n_pages: int
    elements: list = field(default_factory=list)   # TextBlock | ParsedTable, reading order
    engine: str = "pdfplumber"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    element_type: str                     # text | table | table_rowgroup | table_row
    content: str                          # what the LLM reads
    embed_text: str                       # what is embedded + lexically indexed (contextual header + content)
    page_no: int
    section_path: str = ""
    parent_id: str | None = None          # small-to-big: table_row -> table
    table_id: str | None = None
    scope: str | None = None
    unit_name: str | None = None
    unit_multiplier: float | None = None
    entity: str = ""
    fiscal_year: str = ""
    pipeline: str = "structured"          # structured | naive  (ablation tag)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
