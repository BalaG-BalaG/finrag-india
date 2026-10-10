"""Indian-format numeric parsing, scale normalisation and rounding-aware matching.

Why this exists: dense embeddings and LLM judges are both weak at exact numerals.
Everything numeric in this project (table chunk metadata, the claim verifier, the
numeric-exact-match eval metric) goes through this one deterministic module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SCALES: dict[str, float] = {
    "lakh crore": 1e12, "lakh crores": 1e12,
    "crore": 1e7, "crores": 1e7, "cr": 1e7, "cr.": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "thousand": 1e3, "thousands": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9,
    "trillion": 1e12,
}
_SCALE_ALT = r"lakh\s+crores?|crores?|cr\.?|lakhs?|lacs?|thousands?|millions?|mn|billions?|bn|trillion"

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
# Spans that contain digits but are NOT quantities: periods, dates, standards, citations.
_MASKS = [
    r"\bFY\s?'?\d{2,4}(?:\s?[-–/]\s?\d{2,4})?\b",
    r"\b(?:19|20)\d{2}\s?[-–/]\s?\d{2,4}\b",
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}\b",
    r"\bInd\s?AS\s?\d+\b", r"\bIFRS\s?\d+\b", r"\bQ[1-4]\b", r"\b[1-4]Q\b", r"\bH[12]\b",
    r"\b(?:note|schedule|section|page|p\.|table|chart|box|annex|form)\s?(?:no\.?\s?)?\d+[A-Za-z]?\b",
    r"\[C\d+\]", r"\bBasel\s+I{1,3}\b", r"\b(?:tier|level|stage|phase)[-\s]?\d\b",
]
_MASK_RE = re.compile("|".join(_MASKS), re.I)

_NUM_RE = re.compile(
    rf"(?<![\w.])(?<![A-Za-z][-–])"
    rf"(?P<lp>\()?\s?(?P<sign>[-−–])?\s?(?:₹|Rs\.?|INR)?\s?"
    rf"(?P<num>(?:\d{{1,3}}(?:,\d{{2,3}})+|\d+)(?:\.\d+)?)"
    rf"(?P<rp>\))?"
    rf"(?:(?P<pct>\s?%|\s?per\s?cent\b|\s?percent\b)|\s?(?P<scale>{_SCALE_ALT})(?![A-Za-z]))?",
    re.I,
)
_UNIT_RE = re.compile(
    rf"(?:₹|\brs\.?|\binr\b|\brupees\b|\bamounts?\b|\bfigures\b)[^\n]{{0,25}}?\b(?P<s>{_SCALE_ALT})(?![A-Za-z])", re.I
)


@dataclass(frozen=True)
class Num:
    raw: str
    value: float            # as written (sign applied, inline scale NOT applied)
    decimals: int           # precision the author chose -> drives rounding tolerance
    scale: float | None     # inline scale word multiplier, if any ("1.2 lakh crore" -> 1e12)
    is_percent: bool
    is_year: bool
    start: int
    end: int

    def absolute(self, default_scale: float | None = None) -> float:
        return self.value * (self.scale or default_scale or 1.0)


def _scale_of(word: str | None) -> float | None:
    if not word:
        return None
    return SCALES.get(re.sub(r"\s+", " ", word.lower().strip()))


def detect_unit_scale(text: str) -> tuple[str | None, float | None]:
    """Find a table/statement unit declaration such as '(₹ in crore)' or 'Rs. in Lakhs'."""
    m = _UNIT_RE.search(text or "")
    if not m:
        return None, None
    name = re.sub(r"\s+", " ", m.group("s").lower().rstrip("."))
    return name, _scale_of(name)


def parse_number(cell: str) -> float | None:
    """Parse a single table cell: '1,23,456.78', '(1,234.5)', '12.5%', '-', 'Nil'."""
    if cell is None:
        return None
    s = cell.strip()
    if s in {"", "-", "–", "—", "--", "NA", "N.A.", "n/a", "N/A"}:
        return None
    if s.lower() in {"nil", "nil."}:
        return 0.0
    nums = [n for n in extract_numbers(s, keep_years=True)]
    if len(nums) != 1 or (nums[0].end - nums[0].start) < 0.6 * len(s):
        return None
    return nums[0].value


def extract_numbers(text: str, keep_years: bool = False) -> list[Num]:
    """All quantity mentions in free text. Periods/dates/standard numbers are masked out."""
    if not text:
        return []
    masked = _MASK_RE.sub(lambda m: " " * len(m.group()), text) if not keep_years else text
    out: list[Num] = []
    for m in _NUM_RE.finditer(masked):
        raw_num = m.group("num")
        value = float(raw_num.replace(",", ""))
        neg = bool(m.group("sign")) or (bool(m.group("lp")) and bool(m.group("rp")))
        decimals = len(raw_num.split(".")[1]) if "." in raw_num else 0
        is_year = (
            "," not in raw_num and decimals == 0 and 1900 <= value <= 2100
            and not m.group("pct") and not m.group("scale")
        )
        if is_year and not keep_years:
            continue
        out.append(Num(
            raw=m.group().strip(), value=-value if neg else value, decimals=decimals,
            scale=_scale_of(m.group("scale")), is_percent=bool(m.group("pct")),
            is_year=is_year, start=m.start(), end=m.end(),
        ))
    return out


def numbers_match(ans: Num, ctx: Num, ctx_default_scale: float | None = None, rel_tol: float = 0.0) -> bool:
    """Is the answer's number grounded in the context number?

    * rounding-aware: tolerance = half a unit of the LAST digit the answer states,
      so ctx 1,234.56 supports '1,234.6' and '1,235' but not '1,200'.
    * scale-aware: '1.2 lakh crore' matches a cell '1,20,345' in a '(₹ crore)' table,
      while '1,20,345 lakh' does NOT (classic lakh/crore scale hallucination).
    * sign is compared on magnitude (prose says 'loss of 1,234', tables say '(1,234)').
    """
    if ctx.is_percent and not ans.is_percent:
        return False
    half_unit = 0.5 * 10 ** (-ans.decimals) * (1 + 1e-9)
    a_raw, c_raw = abs(ans.value), abs(ctx.value)
    if ans.is_percent:                       # % cells in tables often omit the sign
        return abs(a_raw - c_raw) <= max(half_unit, rel_tol * c_raw)
    c_scale = ctx.scale or ctx_default_scale
    if ans.scale and c_scale:                # both scales known -> compare absolute rupees
        a_abs, c_abs = a_raw * ans.scale, c_raw * c_scale
        return abs(a_abs - c_abs) <= max(half_unit * ans.scale, rel_tol * c_abs)
    return abs(a_raw - c_raw) <= max(half_unit, rel_tol * c_raw)


def format_indian(value: float, decimals: int = 2) -> str:
    """12345678.9 -> '1,23,45,678.90' (2-2-3 grouping)."""
    neg, value = value < 0, abs(value)
    whole, frac = f"{value:.{decimals}f}".split(".") if decimals else (f"{value:.0f}", "")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    s = ",".join(groups + [tail]) if groups else tail
    return ("-" if neg else "") + s + (f".{frac}" if decimals else "")
