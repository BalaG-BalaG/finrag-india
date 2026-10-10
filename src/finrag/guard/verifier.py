"""Claim verification + groundedness scoring.

Routing by claim type:

1. NUMERIC (`verify_numeric_claim`) - deterministic, no LLM.
   For each figure we look for a matching number in the retrieved context under
   (a) rounding tolerance = half a unit of the last digit the answer states, and
   (b) lakh/crore/million scale normalisation. A match is accepted only if the
   EVIDENCE LINE (the actual table row / sentence containing that number, not a
   fuzzy character window) is the best label-match for the claim.

   Row-competition test: if some OTHER row in the same chunk matches the claim's
   metric words better than the row we grounded on, the grounding is rejected as
   `wrong_row`. This catches the canonical financial error - "Net NPA was 2.24%"
   being 'supported' by the Gross NPA cell two rows above it in the same table.

   Retrieval-miss vs fabrication: with no context match we probe the corpus-wide
   numeric_facts index. Found elsewhere -> retrieval miss (re-retrieval can fix
   it); found nowhere -> fabricated (must abstain). The policy layer treats these
   differently, which stops the abstention rate from exploding.

2. DERIVED (`verify_derived_claim`) - the stated delta/growth/bps is RECOMPUTED
   from operands drawn only from rows whose label matches the claim's metric,
   using the operation named by the claim's own cue word. Without both
   restrictions a large operand pool "explains" any number by coincidence.

3. TEXTUAL - NLI entailment (cross-encoder) or LLM judge per VERIFY_BACKEND,
   with lexical overlap as the offline fallback.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from ..config import CFG
from ..numeric import Num, extract_numbers, numbers_match
from .claims import Claim

_STOP = set("the a an of in on at to for and or is was are were be been by with as from that "
            "this it its their our per cent percent crore crores lakh lakhs rs inr stood year "
            "ended ending during about approximately roughly reported stands versus compared".split())

_OP_CUES = [
    (re.compile(r"\b(basis points?|bps)\b", re.I), "bps_change"),
    (re.compile(r"\b(times|multiple)\b", re.I), "ratio"),
    (re.compile(r"(per ?cent|%|percentage)\s*(growth|change|increase|decrease|decline|rise|fall)|"
                r"\b(grew|rose|fell|declined|increased|decreased|growth)\b[^.]{0,40}?(per ?cent|%)", re.I),
     "pct_change"),
]

_PERIOD_LABEL = re.compile(
    r"^\s*(FY\s?)?\d{4}\s*[-–/]\s*\d{2,4}\s*$"        # 2023-24, FY2023-24
    r"|^\s*(FY\s?)?(19|20)\d{2}\s*$"                   # 2024, FY2024
    r"|^\s*\d{1,2}\s+\w+\s+\d{4}\s*$"                  # 31 March 2024
    r"|^\s*\w+\s+\d{1,2},?\s*\d{4}\s*$", re.I)         # March 31, 2024


def _periods_in(text: str) -> set[str]:
    """Normalised period tokens, for comparing a claim's year against a row's year."""
    from ..ingest.metadata import detect_fiscal_years
    return set(detect_fiscal_years(text))

def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def _as_num(d: dict) -> Num:
    return Num(raw=d["raw"], value=d["value"], decimals=d["decimals"], scale=d.get("scale"),
               is_percent=d.get("is_percent", False), is_year=False,
               start=d.get("start", 0), end=d.get("end", 0))


def line_span(text: str, pos: int) -> str:
    """The physical line containing `pos`: one markdown table row, or one sentence."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    line = text[start: end if end != -1 else len(text)]
    if len(line) > 400:                       # prose: fall back to a sentence window
        s = max(text.rfind(". ", 0, pos) + 1, start)
        e = text.find(". ", pos)
        line = text[s: e if e != -1 else min(len(text), pos + 300)]
    return line.strip()


def _label_of(line: str) -> str:
    """Leading row label of a table row; the whole line otherwise."""
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        return cells[0] if cells else line
    if "|" in line:                           # serialised row: "Label | col: val ; ..."
        return line.split("|", 1)[0]
    return line


def _sim(claim_text: str, label: str) -> float:
    cw = _content_words(claim_text)
    if not cw:
        return 1.0
    return len(cw & _content_words(label)) / len(cw)

def _metric_text(claim_text: str, entity: str | None) -> str:
    """Remove the entity name from a claim before comparing it to a row label.

    'Bharat National Bank's Unallocated was 3,21,098' has three words identifying
    the company and one naming the metric. Since the company name never appears in
    a row label, leaving it in drags the similarity score down and correct answers
    get rejected.
    """
    t = claim_text
    for tok in (entity or "").split():
        t = re.sub(rf"\b{re.escape(tok)}(?:'s)?\b", " ", t, flags=re.I)
    return t


def _context_lines(ctx: dict) -> list[str]:
    return [l for l in (ctx.get("content") or "").split("\n") if l.strip()]


def _row_conflict(claim_text: str, ctx: dict, matched_line: str,
                  matched_value: float) -> dict | None:
    """Does a different DATA ROW in this chunk describe the claim's metric better?

    Two table layouts exist and they need different checks:

    * metric-keyed (row labels are metric names, e.g. 'Gross NPA ratio'):
      compare the claim's metric words against each row label. A better-matching
      row means we grounded on the wrong one.
    * period-keyed (row labels are years, e.g. '2023-24'): the metric lives in the
      COLUMN header, so row labels carry no metric information and the comparison
      above is meaningless. Instead check the period: the year on the matched row
      must be the year the claim refers to.

    Only real data rows compete. Captions, footnotes ('Note:', 'Source:') and the
    markdown separator are not rows, and a caption naturally scores high because it
    names the table.
    """
    matched_label = _label_of(matched_line).strip()

    if _PERIOD_LABEL.match(matched_label):
        claim_periods = _periods_in(claim_text)
        row_periods = _periods_in(matched_label)
        if claim_periods and row_periods and not (claim_periods & row_periods):
            return {"better_row": f"period {sorted(claim_periods)[0]}",
                    "matched_row": matched_label, "better_sim": 1.0, "matched_sim": 0.0,
                    "matched_value": matched_value, "kind": "wrong_period"}
        return None

    matched_sim = _sim(claim_text, matched_label)
    best_line, best_sim = matched_line, matched_sim
    for line in _context_lines(ctx):
        if line.strip() == matched_line.strip() or not re.search(r"\d", line):
            continue
        if not line.lstrip().startswith("|") or set(line) <= set("|- "):
            continue                                   # caption / separator, not a row
        if re.match(r"\|?\s*(note|source)\b", line, re.I):
            continue                                   # footnote row
        s = _sim(claim_text, _label_of(line))
        if s > best_sim + 1e-9:
            best_line, best_sim = line, s
    if best_line.strip() != matched_line.strip():
        return {"better_row": _label_of(best_line)[:80], "better_sim": round(best_sim, 3),
                "matched_row": matched_label[:80], "matched_sim": round(matched_sim, 3),
                "matched_value": matched_value, "kind": "wrong_row"}
    return None


def _ground_figure(claim: Claim, fig: Num, contexts: list[dict],
                   min_label_sim: float) -> tuple[dict | None, dict | None]:
    """Return (hit, rejection). A hit is a verbatim, correctly-labelled match."""
    best_hit, best_reject = None, None
    for ci, ctx in enumerate(contexts, start=1):
        # Strip the entity name: it identifies the company, not the metric, and it
        # never appears in a row label, so it only dilutes the similarity score.
        ctext = _metric_text(claim.text, ctx.get("entity"))
        default_scale = ctx.get("unit_multiplier")
        for cand in extract_numbers(ctx["content"], keep_years=True):
            if not numbers_match(fig, cand, default_scale, CFG.numeric_rel_tol):
                continue
            line = line_span(ctx["content"], cand.start)
            label = _label_of(line)
            # The metric name may live in the caption or section rather than the row
            # label (tables keyed by year down the side), so compare against both.
            context_label = " ".join(filter(None, [
                label, ctx.get("section_path", ""),
                (ctx.get("content") or "").split("\n", 1)[0][:120]]))
            sim = max(_sim(ctext, label), _sim(ctext, context_label))
            record = {"chunk_id": ctx["chunk_id"], "context_index": ci,
                      "page_no": ctx.get("page_no"), "matched": cand.raw,
                      "claim_value": fig.raw, "label_sim": round(sim, 3),
                      "row_label": label[:100], "evidence": line[:400]}
            if sim < min_label_sim:
                record["rejected"] = "label_mismatch"
                if best_reject is None or sim > best_reject.get("label_sim", -1):
                    best_reject = record
                continue
            conflict = _row_conflict(ctext, ctx, line, cand.value)
            if conflict:
                record["rejected"] = "wrong_row"
                record["conflict"] = conflict
                if best_reject is None or best_reject.get("rejected") == "label_mismatch":
                    best_reject = record
                continue
            if best_hit is None or sim > best_hit["label_sim"]:
                best_hit = record
    return best_hit, best_reject


def verify_numeric_claim(claim: Claim, contexts: list[dict], conn=None,
                         min_label_sim: float = 0.34) -> Claim:
    figures = [_as_num(n) for n in claim.numbers]
    if not figures:
        claim.kind = "textual"
        return claim

    hits, misses = [], []
    for fig in figures:
        hit, reject = _ground_figure(claim, fig, contexts, min_label_sim)
        if hit:
            hits.append(hit)
            continue
        m = {"claim_value": fig.raw, "rejected": reject}
        if reject is None and conn is not None:
            try:
                from ..store import db
                rows = db.numeric_search(conn, fig.value, fig.decimals, fig.is_percent, k=3)
                m["exists_in_corpus"] = [{"chunk_id": r["chunk_id"], "page_no": r["page_no"],
                                          "row_label": r.get("row_label", ""), "raw": r["raw"]}
                                         for r in rows]
            except Exception:
                m["exists_in_corpus"] = []
        misses.append(m)

    claim.evidence = hits
    claim.score = len(hits) / len(figures)
    claim.supported = claim.score == 1.0
    claim.__dict__["misses"] = misses
    if not claim.supported:
        bad = misses[0]
        rj = bad.get("rejected") or {}
        if rj.get("rejected") == "wrong_row":
            c = rj["conflict"]
            if c.get("kind") == "wrong_period":
                claim.reason = (f"figure {bad['claim_value']} is the {c['matched_row']} value, "
                                f"but the claim refers to {c['better_row']} -> wrong-period grounding")
            else:
                claim.reason = (f"figure {bad['claim_value']} sits in row '{c['matched_row']}', but the "
                                f"claim's metric matches row '{c['better_row']}' better -> wrong-row grounding")
        elif rj.get("rejected") == "label_mismatch":
            claim.reason = (f"figure {bad['claim_value']} found in row '{rj['row_label']}', whose label "
                            f"does not match the claimed metric (sim {rj['label_sim']})")
        elif bad.get("exists_in_corpus"):
            claim.reason = (f"figure {bad['claim_value']} is absent from the retrieved context but exists "
                            f"elsewhere in the corpus -> retrieval miss")
        else:
            claim.reason = f"figure {bad['claim_value']} found nowhere in the corpus -> fabricated"
    return claim


def _operation_for(text: str) -> str:
    for rx, op in _OP_CUES:
        if rx.search(text):
            return op
    return "abs_diff"


_PCT_MARK = re.compile(r"%|per ?cent|ratio|rate\b", re.I)


def _is_percent_in_context(cand: Num, line: str) -> bool:
    """A cell holding 2.24 in a row labelled 'Gross NPA ratio (%)' IS a percentage,
    even though the token itself carries no % sign. Percent-ness lives in the row
    label / column header, so read it from there."""
    return cand.is_percent or bool(_PCT_MARK.search(_label_of(line)))


def _relevant_operands(claim: Claim, contexts: list[dict],
                       min_sim: float = 0.34) -> list[tuple[Num, dict]]:
    """Only numbers from rows whose label matches the claim's metric may be operands."""
    pool: list[tuple[Num, dict]] = []
    for ci, ctx in enumerate(contexts, start=1):
        for cand in extract_numbers(ctx["content"], keep_years=True):
            line = line_span(ctx["content"], cand.start)
            if _sim(claim.text, _label_of(line)) < min_sim:
                continue
            pool.append((cand, {"chunk_id": ctx["chunk_id"], "context_index": ci,
                                "page_no": ctx.get("page_no"), "raw": cand.raw,
                                "row_label": _label_of(line)[:80],
                                "is_percent": _is_percent_in_context(cand, line)}))
    return pool


def verify_derived_claim(claim: Claim, contexts: list[dict], conn=None) -> Claim:
    """Ground verbatim what can be grounded; recompute only the genuinely derived figures."""
    figures = [_as_num(n) for n in claim.numbers]
    if not figures:
        claim.kind = "textual"
        return claim

    op = _operation_for(claim.text)
    pool = _relevant_operands(claim, contexts)
    supported, evidence, misses = 0, [], []

    for fig in figures:
        hit, _ = _ground_figure(claim, fig, contexts, 0.34)
        if hit:                                    # operand quoted verbatim
            supported += 1
            evidence.append(hit)
            continue
        target = abs(fig.value)
        tol = max(0.5 * 10 ** (-fig.decimals) * 1.000001, 1e-9)
        found = None
        for a, ma in pool:
            for b, mb in pool:
                if a is b or a.value == 0:
                    continue
                if op == "bps_change":
                    if not (ma.get("is_percent") and mb.get("is_percent")):
                        continue
                    v = abs(b.value - a.value) * 100
                elif op == "pct_change":
                    v = abs((b.value - a.value) / a.value) * 100
                elif op == "ratio":
                    v = abs(b.value / a.value)
                else:
                    v = abs(b.value - a.value)
                if abs(v - target) <= tol:
                    found = {"operation": op, "operands": [a.raw, b.raw], "computed": round(v, 4),
                             "claimed": fig.raw, "sources": [ma, mb]}
                    break
            if found:
                break
        if found:
            supported += 1
            evidence.append(found)
        else:
            misses.append({"claim_value": fig.raw, "operation": op,
                           "operand_rows": sorted({m["row_label"] for _, m in pool})[:4]})

    claim.score = supported / len(figures)
    claim.supported = claim.score == 1.0
    claim.evidence = evidence
    claim.__dict__["misses"] = misses
    if not claim.supported:
        m = misses[0]
        claim.reason = (f"derived figure {m['claim_value']} could not be reproduced as {m['operation']} "
                        f"of any two grounded figures from rows {m['operand_rows']}")
    return claim


@lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CFG.nli_model)


def verify_textual_claim(claim: Claim, contexts: list[dict], llm=None) -> Claim:
    backend = CFG.verify_backend
    ctx_texts = [c["content"] for c in contexts]
    if backend == "nli":
        try:
            import numpy as np
            model = _nli()
            logits = np.asarray(model.predict([(t, claim.text) for t in ctx_texts]))
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            labels = list(getattr(model.config, "id2label", {}).values())
            ent = next((i for i, l in enumerate(labels) if "entail" in str(l).lower()), -1)
            col = probs[:, ent]
            best, j = float(col.max()), int(col.argmax())
            claim.score, claim.supported = best, best >= CFG.nli_entail_threshold
            claim.evidence = [{"chunk_id": contexts[j]["chunk_id"], "context_index": j + 1,
                               "entailment": round(best, 3), "evidence": ctx_texts[j][:400]}]
            if not claim.supported:
                claim.reason = f"NLI entailment {best:.2f} < {CFG.nli_entail_threshold}"
            return claim
        except Exception:
            pass
    if backend == "llm" and llm is not None:
        prompt = ("Decide if the CLAIM is fully supported by the CONTEXT. Answer ONLY JSON: "
                  '{"supported": true|false, "context_index": <int>, "reason": "<12 words>"}\n\n'
                  "CONTEXT:\n" + "\n\n".join(f"[C{i}] {t}" for i, t in enumerate(ctx_texts, 1)) +
                  f"\n\nCLAIM: {claim.text}")
        try:
            raw = llm.complete(prompt, max_tokens=120, temperature=0.0)
            d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            claim.supported = bool(d.get("supported"))
            claim.score = 1.0 if claim.supported else 0.0
            ci = int(d.get("context_index") or 0)
            if 1 <= ci <= len(contexts):
                claim.evidence = [{"chunk_id": contexts[ci - 1]["chunk_id"], "context_index": ci,
                                   "evidence": ctx_texts[ci - 1][:400]}]
            claim.reason = "" if claim.supported else str(d.get("reason", "judge: unsupported"))
            return claim
        except Exception:
            pass
    cw = _content_words(claim.text)
    best_i, best = 0, 0.0
    for i, t in enumerate(ctx_texts):
        ov = len(cw & _content_words(t)) / (len(cw) or 1)
        if ov > best:
            best_i, best = i, ov
    claim.score, claim.supported = best, best >= 0.6
    claim.evidence = [{"chunk_id": contexts[best_i]["chunk_id"], "context_index": best_i + 1,
                       "lexical_overlap": round(best, 3),
                       "evidence": ctx_texts[best_i][:400]}] if contexts else []
    if not claim.supported:
        claim.reason = f"lexical support {best:.2f} below 0.60"
    return claim


def verify_claims(claims: list[Claim], contexts: list[dict], llm=None, conn=None) -> list[Claim]:
    for c in claims:
        if not contexts:
            c.supported, c.score, c.reason = False, 0.0, "no context retrieved"
            continue
        if c.kind == "numeric":
            verify_numeric_claim(c, contexts, conn=conn)
        elif c.kind == "derived":
            verify_derived_claim(c, contexts, conn=conn)
        else:
            verify_textual_claim(c, contexts, llm=llm)
    return claims


def groundedness(claims: list[Claim], w_numeric: float | None = None) -> dict:
    """Weighted groundedness; numeric/derived claims carry w_numeric x the weight of
    textual ones, because in financial QA a wrong figure is the failure that matters."""
    w_numeric = CFG.w_numeric if w_numeric is None else w_numeric
    if not claims:
        return {"score": 0.0, "n_claims": 0, "n_unsupported": 0, "numeric_ok": True,
                "n_numeric": 0, "unsupported": []}
    num = den = 0.0
    unsupported, n_numeric = [], 0
    for c in claims:
        is_num = c.kind in ("numeric", "derived")
        w = w_numeric if is_num else 1.0
        n_numeric += int(is_num)
        num += w * float(c.score or 0.0)
        den += w
        if not c.supported:
            unsupported.append({"claim_id": c.claim_id, "kind": c.kind, "text": c.text,
                                "score": round(float(c.score or 0), 3), "reason": c.reason})
    return {"score": round(num / den, 4) if den else 0.0, "n_claims": len(claims),
            "n_unsupported": len(unsupported),
            "numeric_ok": all(c.supported for c in claims if c.kind in ("numeric", "derived")),
            "n_numeric": n_numeric, "unsupported": unsupported}
