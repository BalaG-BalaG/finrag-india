"""Atomic claim decomposition.

An "atomic claim" here is the smallest independently checkable unit of the answer.
We type them, because different types need different verifiers:

  NUMERIC     "Gross NPA stood at 3.24 per cent in FY2023-24"
              -> (metric, period, scope, value, unit) tuple; verified arithmetically
                 against the retrieved cells. No LLM judge needed, no tolerance drift.
  DERIVED     "a decline of 62 basis points year-on-year"
              -> value not expected verbatim in context; recomputed from two
                 grounded operands (this is how YoY deltas stop being hallucinations).
  TEXTUAL     "the bank attributed the improvement to higher recoveries"
              -> verified by NLI entailment or an LLM judge.

Extraction is hybrid: a deterministic sentence/number splitter always runs (it can
never fail), and an optional LLM pass refines it. The FinGround-style cascading
failure the literature warns about is handled by `fallback_used`: if the LLM
extractor returns malformed JSON we keep the deterministic claims rather than
silently verifying nothing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from ..numeric import Num, extract_numbers

CITE = re.compile(r"\[C(\d+)\]")


def strip_citations(text: str) -> str:
    """Remove [C#] markers without leaving the double spaces / ' .' artefacts that
    would stop the flagged span from matching the answer text later."""
    t = CITE.sub("", text or "")
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()
_DERIVED_CUE = re.compile(
    r"\b(increase|decrease|decline|growth|rose|fell|grew|dropped|change|delta|"
    r"yoy|y-o-y|year[- ]on[- ]year|basis points?|bps|times|multiple|ratio of)\b", re.I)
_HEDGE = re.compile(r"\b(insufficient|not (?:available|stated|disclosed|found)|cannot|unable|"
                    r"no (?:data|information|figure))\b", re.I)


@dataclass
class Claim:
    claim_id: str
    text: str
    kind: str                       # numeric | derived | textual
    numbers: list[dict] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    metric: str = ""
    period: str = ""
    scope: str = ""
    # filled by the verifier
    supported: bool | None = None
    score: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _num_dict(n: Num) -> dict:
    return {"raw": n.raw, "value": n.value, "decimals": n.decimals, "scale": n.scale,
            "is_percent": n.is_percent, "start": n.start, "end": n.end}


def split_claims_rule(answer: str) -> list[Claim]:
    """Deterministic decomposition: sentence split, then one claim per clause that
    carries at least one quantity (clauses joined by ';' or ', and' are separated)."""
    claims: list[Claim] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(₹\[])", answer or "") if s.strip()]
    idx = 0
    for sent in sentences:
        if _HEDGE.search(sent):
            continue
        clauses = [c.strip() for c in re.split(r";|,\s+(?:and|while|whereas)\s+", sent) if c.strip()]
        for clause in clauses:
            cited = sorted({int(m.group(1)) for m in CITE.finditer(clause)})
            body = strip_citations(clause)
            nums = extract_numbers(body)
            if not body or len(body) < 12:
                continue
            kind = "textual"
            if nums:
                kind = "derived" if _DERIVED_CUE.search(body) else "numeric"
            idx += 1
            claims.append(Claim(claim_id=f"c{idx}", text=body, kind=kind,
                                numbers=[_num_dict(n) for n in nums], cited=cited))
    return claims


EXTRACT_PROMPT = """You decompose a financial answer into atomic, independently verifiable claims.

Rules:
- One claim = one fact. Never merge two figures into one claim.
- Keep every number EXACTLY as written (do not round, do not convert lakh/crore).
- kind = "numeric" if the figure should appear verbatim in a source table/sentence.
- kind = "derived" if the figure is computed (YoY change, growth %, bps difference, ratio).
- kind = "textual" for non-numeric assertions.
- metric/period/scope: short strings, "" if not stated.
- Ignore sentences that only say data is unavailable.

Return ONLY a JSON array, no prose, no markdown fences:
[{"text":"...","kind":"numeric","metric":"Gross NPA ratio","period":"FY2023-24","scope":"consolidated"}]

ANSWER:
"""


def split_claims_llm(answer: str, llm) -> tuple[list[Claim], bool]:
    """LLM-refined decomposition. Returns (claims, fallback_used)."""
    try:
        raw = llm.complete(EXTRACT_PROMPT + answer, max_tokens=800, temperature=0.0)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        data = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        if not isinstance(data, list) or not data:
            raise ValueError("empty")
        out = []
        for i, d in enumerate(data, start=1):
            text = str(d.get("text", "")).strip()
            if not text:
                continue
            nums = extract_numbers(strip_citations(text))
            kind = d.get("kind") or ("numeric" if nums else "textual")
            if nums:
                kind = "derived" if _DERIVED_CUE.search(text) else "numeric"
            elif kind in ("numeric", "derived"):
                kind = "textual"
            out.append(Claim(claim_id=f"c{i}", text=strip_citations(text), kind=kind,
                             numbers=[_num_dict(n) for n in nums],
                             cited=sorted({int(m.group(1)) for m in CITE.finditer(text)}),
                             metric=str(d.get("metric", "")), period=str(d.get("period", "")),
                             scope=str(d.get("scope", ""))))
        if not out:
            raise ValueError("no claims")
        return out, False
    except Exception:
        return split_claims_rule(answer), True


def extract_claims(answer: str, llm=None) -> tuple[list[Claim], bool]:
    """Union strategy: rule-based claims are always present; LLM claims are merged in.

    A number that appears in the answer but in NO claim is itself a red flag, so we
    guarantee coverage: every numeric token in the answer belongs to some claim.
    """
    rule = split_claims_rule(answer)
    if llm is None:
        return rule, True
    llm_claims, fallback = split_claims_llm(answer, llm)
    if fallback:
        return rule, True
    covered = {round(n["value"], 6) for c in llm_claims for n in c.numbers}
    idx = len(llm_claims)
    for c in rule:
        missing = [n for n in c.numbers if round(n["value"], 6) not in covered]
        if missing:
            idx += 1
            c.claim_id = f"c{idx}"
            llm_claims.append(c)
            covered.update(round(n["value"], 6) for n in missing)
    return llm_claims, False


def is_abstention(answer: str) -> bool:
    return bool(_HEDGE.search(answer or "")) and len(extract_numbers(answer or "")) == 0
