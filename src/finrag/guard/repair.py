"""Repair, rather than refuse, a specific and common failure class.

Observed failure: asked "by how much did X change from FY2022-23 to FY2023-24?",
a small model frequently quotes BOTH years' figures correctly and then simply does
not subtract. The guardrail sees no claim carrying the requested delta, scores the
answer unsupported, and abstains.

That is the wrong response. Nothing was hallucinated - the model produced two
correctly grounded operands and stopped one arithmetic step short. The evidence
needed to answer is already verified.

So: when the question asks for a computed quantity, and two operands for the SAME
metric at the DIFFERENT periods named in the question have been verified, compute
the result deterministically and append it as a derived claim whose evidence is the
two operand claims.

Safety properties, which are what make this a guardrail feature rather than a hole:
  1. Operands must already be `supported` - we never compute from an unverified
     figure, so repair cannot introduce a number the verifier has not grounded.
  2. The arithmetic is done in Python, not by the LLM, so it cannot be wrong in the
     way LLM arithmetic is wrong.
  3. The operation is chosen by the question's own cue words, not guessed.
  4. Period assignment must be unambiguous; ambiguity means no repair.
  5. The appended sentence is explicitly labelled as computed, and cites both
     operands, so a reader can check the derivation.
"""
from __future__ import annotations

import re

from ..ingest.metadata import detect_fiscal_years
from ..numeric import extract_numbers, format_indian
from .claims import Claim

# What kind of computation the question is asking for.
_ASKS_BPS = re.compile(r"\bbasis points?\b|\bbps\b", re.I)
_ASKS_PCT = re.compile(r"per ?cent(?:age)? (?:change|growth|increase|decrease|decline)|"
                       r"\bgrowth rate\b|\bby what per ?cent", re.I)
_ASKS_DIFF = re.compile(r"\b(change|changed|difference|increase|decrease|decline|grew|rose|fell|"
                        r"growth|higher|lower|more|less)\b", re.I)


def wants_computation(question: str) -> str | None:
    """Which operation the question asks for, or None if it asks for a level."""
    if _ASKS_BPS.search(question):
        return "bps_change"
    if _ASKS_PCT.search(question):
        return "pct_change"
    if _ASKS_DIFF.search(question):
        return "abs_diff"
    return None


def _verified_operands(claims: list[Claim]) -> list[tuple[str, float, int, bool, Claim]]:
    """(period, value, decimals, is_percent, source_claim) for each grounded figure.

    A figure is usable only if its claim passed verification AND the claim text
    names exactly one period and one figure, so the pairing is unambiguous.
    """
    out = []
    for c in claims:
        if not c.supported or c.kind not in ("numeric", "derived"):
            continue
        periods = detect_fiscal_years(c.text)
        nums = [n for n in extract_numbers(c.text)]
        if len(periods) != 1 or len(nums) != 1:
            continue
        n = nums[0]
        out.append((periods[0], n.value, n.decimals, n.is_percent, c))
    return out


def _percent_context(claim: Claim) -> bool:
    """A table cell holding 14.28 in a row labelled 'CRAR (%)' is a percentage even
    though the claim text may not repeat the sign; read it from the evidence row."""
    for ev in (claim.evidence or []):
        blob = f"{ev.get('row_label', '')} {ev.get('evidence', '')}"
        if re.search(r"%|per ?cent", blob, re.I):
            return True
    return False


def repair_derived(question: str, answer: str, claims: list[Claim],
                   contexts: list[dict]) -> dict | None:
    """Return {'answer', 'claim', 'note'} if a repair was made, else None."""
    op = wants_computation(question)
    if op is None:
        return None

    # The answer must not already contain the computed quantity, or we would be
    # appending a duplicate (and possibly contradicting the model).
    q_periods = detect_fiscal_years(question)
    if len(q_periods) != 2:
        return None
    p_from, p_to = q_periods[0], q_periods[1]

    operands = _verified_operands(claims)
    by_period: dict[str, tuple[float, int, bool, Claim]] = {}
    for period, value, dec, is_pct, src in operands:
        if period in by_period and by_period[period][0] != value:
            return None                      # two different values for one period
        by_period[period] = (value, dec, is_pct, src)

    if p_from not in by_period or p_to not in by_period:
        return None

    v_from, d_from, pct_from, c_from = by_period[p_from]
    v_to, d_to, pct_to, c_to = by_period[p_to]
    is_pct = (pct_from or _percent_context(c_from)) and (pct_to or _percent_context(c_to))

    if op == "bps_change":
        if not is_pct:
            return None                      # basis points only apply to percentages
        value = (v_to - v_from) * 100
        shown = f"{value:+.0f} basis points"
    elif op == "pct_change":
        if v_from == 0:
            return None
        value = (v_to - v_from) / abs(v_from) * 100
        shown = f"{value:+.2f} per cent"
    else:
        value = v_to - v_from
        shown = (f"{value:+.2f} percentage points" if is_pct
                 else f"{'+' if value >= 0 else '-'}{format_indian(abs(value), max(d_from, d_to))}")

    # Already stated? Then there is nothing to repair.
    for n in extract_numbers(answer):
        if abs(abs(n.value) - abs(value)) <= 0.5 * 10 ** (-n.decimals) * 1.000001:
            return None

    direction = "an increase" if value > 0 else ("a decrease" if value < 0 else "no change")
    sentence = (f"Computed from the verified figures: {p_to} ({v_to}) minus {p_from} ({v_from}) "
                f"gives {shown}, i.e. {direction} over the period.")

    claim = Claim(
        claim_id=f"c{len(claims) + 1}", text=sentence, kind="derived",
        numbers=[], cited=sorted({e.get("context_index") for c in (c_from, c_to)
                                  for e in (c.evidence or []) if e.get("context_index")}),
    )
    claim.supported = True
    claim.score = 1.0
    claim.evidence = [{
        "operation": op, "computed": round(value, 4),
        "operands": [{"period": p_from, "value": v_from, "claim_id": c_from.claim_id},
                     {"period": p_to, "value": v_to, "claim_id": c_to.claim_id}],
        "evidence": "deterministic arithmetic over two verifier-grounded operands",
    }]
    claim.reason = ""

    return {"answer": answer.rstrip() + "\n\n" + sentence, "claim": claim,
            "note": f"repaired: computed {op} = {round(value, 4)} from grounded "
                    f"{p_from}={v_from} and {p_to}={v_to}"}
