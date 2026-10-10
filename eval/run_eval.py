"""Run the benchmark through one or more ablation arms and emit metrics + a report.

    python eval/run_eval.py --benchmark eval/benchmark_ifhb.jsonl --arms all --split test

Arms (each isolates one contribution, which is what a Review-2 panel wants to see):
    B0_naive_noguard     512-token windows, dense only, no guardrail   <- the strawman
    B1_struct_noguard    structure-aware chunking, hybrid, no guardrail
    B2_struct_guard      + hallucination-awareness layer               <- full system
    A1_no_lexical        full system minus the BM25 arm
    A2_no_rerank         full system minus the cross-encoder
    A3_no_parent         full system minus small-to-big expansion
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from finrag.pipeline import FinRAG                      # noqa: E402
from finrag.retrieve.hybrid import RetrievalConfig      # noqa: E402
from finrag.store import db                             # noqa: E402
import metrics as M                                     # noqa: E402

ARMS = {
    "B0_naive_noguard":  (RetrievalConfig(use_lexical=False, use_rrf=False, use_rerank=False,
                                          use_mmr=False, expand_parent=False, pipeline="naive"), False),
    "B1_struct_noguard": (RetrievalConfig(), False),
    "B2_struct_guard":   (RetrievalConfig(), True),
    "A1_no_lexical":     (RetrievalConfig(use_lexical=False), True),
    "A2_no_rerank":      (RetrievalConfig(use_rerank=False), True),
    "A3_no_parent":      (RetrievalConfig(expand_parent=False), True),
}


def load(path: str, split: str | None) -> list[dict]:
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [i for i in items if not split or i.get("split") == split]


def run_arm(conn, items: list[dict], rc: RetrievalConfig, guard: bool) -> tuple[list[dict], dict]:
    rag = FinRAG(conn, rc=rc)
    records = []
    for it in items:
        t0 = time.time()
        out = rag.answer(it["question"], guard=guard, log=False)
        answered = out["decision"] in ("answer", "answer_flagged")
        correct = (M.numeric_match(out["answer"], it.get("gold_value")) if answered else False)
        its = out["trace"]["iterations"]
        last = its[-1] if its else {}
        draft = last.get("draft", "")
        draft_g = (last.get("groundedness") or {}).get("score")
        records.append({
            "qid": it["qid"], "type": it["type"], "answerable": it["answerable"],
            "question": it["question"], "decision": out["decision"],
            "groundedness": out["groundedness"], "answer": out["answer"],
            "numeric_correct": correct, "gold_value": it.get("gold_value"),
            "gold_chunk_ids": it.get("gold_chunk_ids", []),
            "retrieved_chunk_ids": [c["chunk_id"] for c in out["citations"]],
            "n_claims": out["n_claims"], "n_unsupported": out["n_unsupported"],
            "iterations": out["iterations"], "latency_ms": int((time.time() - t0) * 1000),
            "reason": out["reason"],
            "draft": draft,
            "draft_groundedness": draft_g,
            "draft_correct": M.numeric_match(draft, it.get("gold_value")) if draft else False,
        })
    return records, M.aggregate(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="eval/benchmark_ifhb.jsonl")
    ap.add_argument("--arms", default="B0_naive_noguard,B1_struct_noguard,B2_struct_guard")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="eval/results")
    a = ap.parse_args()

    items = load(a.benchmark, a.split if a.split != "all" else None)
    if a.limit:
        items = items[:a.limit]
    arm_names = list(ARMS) if a.arms == "all" else [x.strip() for x in a.arms.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)

    summary = {}
    with db.connect() as conn:
        for name in arm_names:
            rc, guard = ARMS[name]
            print(f"\n=== {name}  (n={len(items)}) ===")
            recs, agg = run_arm(conn, items, rc, guard)
            summary[name] = agg
            json.dump(recs, open(f"{a.out_dir}/{name}_records.json", "w", encoding="utf-8"), indent=2)
            print(json.dumps({k: v for k, v in agg.items()
                              if k not in ("risk_coverage_curve", "by_type")}, indent=2))

    json.dump(summary, open(f"{a.out_dir}/summary.json", "w", encoding="utf-8"), indent=2)
    table = M.compare(summary)
    open(f"{a.out_dir}/comparison.md", "w", encoding="utf-8").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
