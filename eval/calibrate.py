"""Pick tau_low / tau_high on the DEV split instead of guessing them.

Objective: maximise answer accuracy subject to selective risk <= target (default 5%).
Reports the full sweep so the report can show WHY the operating point was chosen -
this is the difference between "threshold e.g. < 0.95" (a guess) and a calibrated,
defensible operating point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import metrics as M                                      # noqa: E402


def sweep(records: list[dict], target_risk: float = 5.0) -> dict:
    grid = [round(x / 20, 2) for x in range(0, 21)]
    rows = []
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            sim = []
            for r in records:
                g = r.get("draft_groundedness")
                if g is None:            # self-abstained or gated: no draft to release
                    sim.append({**r, "decision": "abstain"})
                    continue
                d = "answer" if g >= hi else ("abstain" if g < lo else "answer_flagged")
                sim.append({**r, "decision": d, "numeric_correct": r["draft_correct"]})
                #g = float(r.get("groundedness") or 0.0)
                #d = "answer" if g >= hi else ("abstain" if g < lo else "answer_flagged")
                #sim.append({**r, "decision": d})
            m = M.aggregate(sim)
            rows.append({"tau_low": lo, "tau_high": hi,
                         "accuracy": m["answer_accuracy_pct"] or 0.0,
                         "selective_risk": m["selective_risk_pct"] or 0.0,
                         "coverage": m["coverage_pct"] or 0.0,
                         "honest_abstention": m["honest_abstention_pct"] or 0.0})
    feasible = [r for r in rows if r["selective_risk"] <= target_risk and r["coverage"] > 0]
    if feasible:
        best = max(feasible, key=lambda r: (r["accuracy"], r["coverage"]))
        note = "feasible: operating point meets the risk target"
    else:
        # No threshold reaches the target. Report that honestly instead of returning a
        # meaningless 0/0: pick the lowest-risk point that still answers something, and
        # flag that the confidence signal itself needs work (better verifier or retriever).
        answering = [r for r in rows if r["coverage"] > 0]
        best = min(answering or rows, key=lambda r: (r["selective_risk"], -r["accuracy"]))
        note = ("INFEASIBLE: no threshold pair reaches the target selective risk. "
                "Groundedness is not separating correct from incorrect answers well enough - "
                "improve the verifier or the retriever before tuning thresholds further.")
    return {"target_risk_pct": target_risk, "chosen": best, "feasible": bool(feasible),
            "note": note, "feasible_points": len(feasible), "sweep": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="eval/results/B2_struct_guard_records.json")
    ap.add_argument("--target-risk", type=float, default=5.0)
    ap.add_argument("--out", default="eval/results/calibration.json")
    a = ap.parse_args()
    recs = json.load(open(a.records, encoding="utf-8"))
    res = sweep(recs, a.target_risk)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2)
    print(res["note"])
    print(json.dumps(res["chosen"], indent=2))
    print(f"\nSet these in .env:\n  TAU_LOW={res['chosen']['tau_low']}\n  TAU_HIGH={res['chosen']['tau_high']}")


if __name__ == "__main__":
    main()
