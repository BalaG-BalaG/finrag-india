"""Build the Indian Financial Hallucination Benchmark (IFHB) from the ingested corpus.

Hand-writing 150 Q&A pairs takes days. Instead we generate them programmatically
from the parsed table cells - every item carries its gold value AND its gold
evidence chunk_id, because the cell is where it came from - then a human spot-checks
a sample. This is the same "programmatic gold + human audit" construction used by
FinanceBench-style datasets, and it is what makes retrieval metrics (Recall@k, MRR,
nDCG) computable at all: you cannot score retrieval without gold chunk labels.

Five item types, each targeting a documented failure mode:

  lookup        single cell           -> baseline factual accuracy
  yoy           two cells + arithmetic-> multi-step numeric reasoning
  unanswerable  metric absent         -> honest abstention (the key safety metric)
  scale_trap    asks in the WRONG unit-> lakh/crore conversion hallucination
  scope_trap    asks consolidated when
                only standalone exists-> consolidated/standalone confusion

Items marked `answerable: false` MUST be abstained on; scoring an answer to one of
those as correct is exactly the hallucination we are measuring.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from finrag.numeric import format_indian, parse_number
from finrag.store import db

random.seed(7)

ABSENT_METRICS = [
    "carbon emission intensity", "number of ATMs in Kerala", "median employee tenure",
    "cost of digital marketing", "value of gold loans written off", "CEO travel expenses",
    "number of cyber incidents reported", "spend on CSR education projects",
]


_PERIOD_RE = re.compile(
    r"(FY\s?\d{4}[-–]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})")


def split_column(col: str) -> tuple[str, str]:
    """'Revenue FY2023-24' -> ('Revenue', 'FY2023-24'). A column is only comparable
    with another column of the SAME measure and a DIFFERENT period - otherwise a
    'YoY' item would compare Revenue against Result of the same year."""
    col = re.sub(r"\(.*?\)", " ", col).strip()
    m = _PERIOD_RE.search(col)
    if not m:
        return col.strip(), ""
    measure = (col[:m.start()] + " " + col[m.end():]).strip(" |-,")
    return re.sub(r"\s+", " ", measure), m.group(1).strip()


def _clean_metric(label: str) -> str:
    return re.sub(r"\s*\(%\)|\s*\(Rs\.?[^)]*\)", "", label).strip()


def build(conn, out_path: str, max_items: int = 200) -> dict:
    with conn.cursor() as cur:
        cur.execute("""SELECT chunk_id, doc_id, content, page_no, entity, scope, unit_name,
                              unit_multiplier, meta, parent_id, section_path
                       FROM chunks WHERE element_type='table_row' AND pipeline='structured'""")
        rows = cur.fetchall()

    items: list[dict] = []
    qid = 0

    for r in rows:
        meta = r["meta"] or {}
        label = _clean_metric(meta.get("row_label", ""))
        if not label:
            continue
        pairs = re.findall(r"([^|;]+?):\s*([\d,()%.\-]+)(?:\s|;|$)", r["content"].split("|", 1)[-1])
        cells = []
        for col, val in pairs:
            v = parse_number(val)
            if v is None:
                continue
            measure, period = split_column(col.strip())
            cells.append((period or col.strip(), v, val.strip(), measure))
        if not cells:
            continue
        unit = r["unit_name"] or ("per cent" if "%" in meta.get("row_label", "") else "")
        entity, scope = r["entity"], r["scope"]

        qid += 1
        period, value, raw, measure0 = cells[0]
        items.append({
            "qid": f"IFHB-{qid:04d}", "type": "lookup", "answerable": True,
            "question": f"What was {entity}'s {label} for {period}"
                        + (f" ({scope})" if scope else "") + "?",
            "gold_value": value, "gold_text": raw, "gold_unit": unit,
            "gold_period": period, "gold_scope": scope, "gold_metric": label,
            "gold_chunk_ids": [r["chunk_id"]] + ([r["parent_id"]] if r["parent_id"] else []),
            "gold_page": r["page_no"], "doc_id": r["doc_id"],
        })

        comparable = [c for c in cells[1:]
                      if c[3] == measure0 and c[0] != period and c[1] not in (0, None)]
        if comparable:
            (p1, v1) = (period, value)
            (p2, v2) = comparable[0][0], comparable[0][1]
            if True:
                qid += 1
                is_pct_metric = "%" in meta.get("row_label", "")
                if is_pct_metric:
                    delta = round((v1 - v2) * 100, 2)
                    q = (f"By how many basis points did {entity}'s {label} change from {p2} to {p1}?")
                    gold, gunit = delta, "bps"
                else:
                    delta = round((v1 - v2) / v2 * 100, 2)
                    q = f"What was the year-on-year percentage change in {entity}'s {label} from {p2} to {p1}?"
                    gold, gunit = delta, "per cent"
                items.append({
                    "qid": f"IFHB-{qid:04d}", "type": "yoy", "answerable": True, "question": q,
                    "gold_value": gold, "gold_text": f"{gold}", "gold_unit": gunit,
                    "gold_period": f"{p2}->{p1}", "gold_scope": scope, "gold_metric": label,
                    "gold_operands": [v1, v2],
                    "gold_chunk_ids": [r["chunk_id"]] + ([r["parent_id"]] if r["parent_id"] else []),
                    "gold_page": r["page_no"], "doc_id": r["doc_id"],
                })

        if True:
            if unit in ("crore", "lakh"):
                wrong = "lakh" if unit == "crore" else "crore"
                qid += 1
                items.append({
                    "qid": f"IFHB-{qid:04d}", "type": "scale_trap", "answerable": True,
                    "question": f"Confirm that {entity}'s {label} for {period} was "
                                f"Rs. {format_indian(value)} {wrong}. Is that correct?",
                    "gold_value": value, "gold_text": raw, "gold_unit": unit,
                    "gold_period": period, "gold_scope": scope, "gold_metric": label,
                    "trap": f"stated unit '{wrong}' contradicts the table unit '{unit}'",
                    "gold_chunk_ids": [r["chunk_id"]] + ([r["parent_id"]] if r["parent_id"] else []),
                    "gold_page": r["page_no"], "doc_id": r["doc_id"],
                })

        if scope in ("consolidated", "standalone") and cells:
            other = "standalone" if scope == "consolidated" else "consolidated"
            with conn.cursor() as cur:
                cur.execute("""SELECT count(*) AS n FROM chunks WHERE element_type='table_row'
                               AND scope=%s AND meta->>'row_label' ILIKE %s""",
                            (other, f"%{label[:18]}%"))
                exists = cur.fetchone()["n"]
            if not exists:
                qid += 1
                items.append({
                    "qid": f"IFHB-{qid:04d}", "type": "scope_trap", "answerable": False,
                    "question": f"What was {entity}'s {other} {label} for {period}?",
                    "gold_value": None, "gold_text": "", "gold_unit": "",
                    "gold_period": period, "gold_scope": other, "gold_metric": label,
                    "trap": f"only the {scope} figure is disclosed; the {other} one is not",
                    "gold_chunk_ids": [], "gold_page": None, "doc_id": r["doc_id"],
                })

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT entity FROM documents")
        entities = [e["entity"] for e in cur.fetchall()] or ["the bank"]
    for m in ABSENT_METRICS:
        qid += 1
        items.append({
            "qid": f"IFHB-{qid:04d}", "type": "unanswerable", "answerable": False,
            "question": f"What was {random.choice(entities)}'s {m} for FY2023-24?",
            "gold_value": None, "gold_text": "", "gold_unit": "", "gold_period": "FY2023-24",
            "gold_scope": None, "gold_metric": m, "gold_chunk_ids": [], "gold_page": None,
            "doc_id": None,
        })

    random.shuffle(items)
    items = items[:max_items]
    for i, it in enumerate(items):                    # deterministic dev/test split
        it["split"] = "dev" if i % 3 == 0 else "test"

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    dist: dict[str, int] = {}
    for it in items:
        dist[it["type"]] = dist.get(it["type"], 0) + 1
    return {"path": out_path, "n": len(items), "by_type": dist,
            "dev": sum(1 for i in items if i["split"] == "dev"),
            "test": sum(1 for i in items if i["split"] == "test")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/benchmark_ifhb.jsonl")
    ap.add_argument("--max-items", type=int, default=200)
    a = ap.parse_args()
    with db.connect() as conn:
        print(build(conn, a.out, a.max_items))


if __name__ == "__main__":
    main()
