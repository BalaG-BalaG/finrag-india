"""Abstention policy = selective prediction over a two-threshold band.

    g  >= tau_high                     -> ANSWER
    tau_low <= g < tau_high            -> repair: re-retrieve (rewritten query /
                                          relaxed filters / more candidates) and
                                          regenerate, up to max_iters; if the band
                                          is still unmet -> ANSWER_FLAGGED, which
                                          returns the answer with the ungrounded
                                          spans struck out and labelled
    g  <  tau_low                      -> ABSTAIN with a structured refusal
    any unsupported NUMERIC claim      -> hard gate: never answered silently,
                                          regardless of g (HARD_NUMERIC_GATE)

Two pre-generation gates also exist, because refusing before you generate is
cheaper and safer than refusing after:
    * empty retrieval                  -> abstain immediately
    * top-1 cross-encoder < tau_rerank -> "no relevant evidence" abstain

The thresholds are NOT magic numbers: eval/calibrate.py sweeps them on the dev
split and picks the operating point that maximises answer accuracy subject to a
target hallucination rate; the risk-coverage curve and AURC are reported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import CFG
from .claims import Claim

ANSWER = "answer"
FLAGGED = "answer_flagged"
ABSTAIN = "abstain"


@dataclass
class Decision:
    decision: str
    groundedness: float | None
    reason: str
    answer: str
    flagged_spans: list[dict] = field(default_factory=list)
    next_action: str = ""            # stop | reretrieve
    retry_hint: dict = field(default_factory=dict)


REFUSAL_TEMPLATE = (
    "I can't verify this from the indexed documents, so I'm not going to state a figure.\n\n"
    "Reason: {reason}\n"
    "Searched: {searched}\n"
    "What would resolve it: {fix}")


def pre_generation_gate(retrieval: dict) -> Decision | None:
    if not retrieval.get("contexts"):
        return Decision(ABSTAIN, None, "empty_retrieval",
                        REFUSAL_TEMPLATE.format(
                            reason="no chunk in the index matched this question",
                            searched="dense + lexical arms returned 0 candidates",
                            fix="ingest the relevant filing, or rephrase using the metric name as printed in the report"),
                        next_action="stop")
    top = retrieval.get("top_rerank", 1.0)
    if retrieval.get("scored_by", "cross_encoder") == "cross_encoder" and top < CFG.tau_rerank:
        return Decision(ABSTAIN, None, f"low_relevance(top_rerank={top:.3f}<{CFG.tau_rerank})",
                        REFUSAL_TEMPLATE.format(
                            reason=f"the best-matching passage scored {top:.2f}, below the relevance floor",
                            searched=f"{len(retrieval.get('candidates', []))} candidates after fusion",
                            fix="name the entity, statement and fiscal year explicitly (e.g. 'SBI standalone Gross NPA FY2023-24')"),
                        next_action="stop")
    return None


def rewrite_query(question: str, claims: list[Claim], plan) -> tuple[str, dict]:
    """Repair-loop query rewriting: pull the metric/period words out of the claims
    that failed and push them into the lexical arm; relax the FY filter (a
    comparison question was probably over-filtered)."""
    terms: list[str] = []
    for c in claims:
        if c.supported:
            continue
        terms += [w for w in re.findall(r"[A-Za-z]{4,}", c.metric or c.text)][:6]
        if c.period:
            terms.append(c.period)
    seen, uniq = set(), []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            uniq.append(t)
    new_q = (question + " " + " ".join(uniq[:12])).strip()
    relaxed = {k: v for k, v in (getattr(plan, "filters", {}) or {}).items() if k != "fiscal_year"}
    return new_q, relaxed


def strike_ungrounded(answer: str, claims: list[Claim]) -> tuple[str, list[dict]]:
    """Mark the exact unsupported spans instead of throwing the whole answer away."""
    spans = []
    marked = answer
    for c in claims:
        if c.supported:
            continue
        snippet = c.text.strip()[:160]
        spans.append({"claim_id": c.claim_id, "kind": c.kind, "text": snippet,
                      "reason": c.reason})
        if snippet and snippet in marked:
            marked = marked.replace(snippet, f"~~{snippet}~~ [UNVERIFIED: {c.reason}]", 1)
        else:                       # span reworded by the extractor: mark the figure itself
            for n in (c.numbers or []):
                raw = n.get("raw", "")
                if raw and raw in marked:
                    marked = marked.replace(raw, f"~~{raw}~~ [UNVERIFIED: {c.reason}]", 1)
                    break
    return marked, spans


def decide(answer: str, claims: list[Claim], g: dict, iteration: int,
           cfg=CFG) -> Decision:
    score = g["score"]
    numeric_fail = cfg.hard_numeric_gate and not g["numeric_ok"]

    if score >= cfg.tau_high and not numeric_fail:
        return Decision(ANSWER, score, "grounded", answer, next_action="stop")

    if score < cfg.tau_low or (numeric_fail and score < cfg.tau_high):
        if iteration < cfg.max_iters:
            return Decision(ABSTAIN if score < cfg.tau_low else FLAGGED, score,
                            "below_band_retry", answer, next_action="reretrieve")

    if score < cfg.tau_low:
        bad = g["unsupported"][:3]
        return Decision(ABSTAIN, score, "below_tau_low",
                        REFUSAL_TEMPLATE.format(
                            reason=f"groundedness {score:.2f} < {cfg.tau_low}; "
                                   f"{g['n_unsupported']} of {g['n_claims']} claims unsupported"
                                   + (f" ({bad[0]['reason']})" if bad else ""),
                            searched=f"{iteration + 1} retrieval pass(es)",
                            fix="the figure may be in a filing that is not indexed, or in a "
                                "note/annexure that needs a page-range ingest"),
                        next_action="stop")

    if iteration < cfg.max_iters:
        return Decision(FLAGGED, score, "in_band_retry", answer, next_action="reretrieve")

    marked, spans = strike_ungrounded(answer, claims)
    if numeric_fail:
        marked = ("NUMERIC CLAIM UNVERIFIED - do not use the struck-through figures.\n\n" + marked)
    return Decision(FLAGGED, score, "flagged_after_retries", marked, flagged_spans=spans,
                    next_action="stop")
