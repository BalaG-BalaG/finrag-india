"""Metrics for a selective-prediction QA system.

Plain accuracy is the wrong headline metric here, because a system that abstains on
everything scores 0% hallucination. We therefore report the standard
selective-prediction triple - coverage, selective risk, and the risk-coverage curve
(AURC) - alongside an abstention confusion matrix, plus retrieval quality separately
so a failure can be attributed to the retriever or to the generator.

  coverage        = answered / total answerable
  selective risk  = wrong / answered            (the hallucination rate on what we DO answer)
  AURC            = area under the risk-coverage curve, lower is better; it is the
                    threshold-free summary of how well groundedness ranks good
                    answers above bad ones
  honest abstention = abstained / total unanswerable   (higher is better)
  over-abstention   = abstained / total answerable     (the cost of safety)
"""
from __future__ import annotations

import math

from finrag.numeric import extract_numbers


def numeric_match(pred_text: str, gold_value: float | None, decimals: int | None = None,
                  rel_tol: float = 0.005, max_numbers: int = 12, max_chars: int = 1200) -> bool:
    """Is the gold figure present in the prediction, at the precision it was asked for?

    Dump guard: an "answer" that reproduces a whole page will contain the gold figure
    by construction. Without this guard a naive baseline that dumps its 512-token
    window scores as accurate, which is exactly the false comparison this benchmark
    exists to avoid. Predictions longer than `max_chars` or carrying more than
    `max_numbers` figures are scored incorrect regardless of content.
    """
    if gold_value is None:
        return False
    if len(pred_text or "") > max_chars:
        return False
    nums = extract_numbers(pred_text or "", keep_years=True)
    if len(nums) > max_numbers:
        return False
    for n in nums:
        d = decimals if decimals is not None else n.decimals
        tol = max(0.5 * 10 ** (-d) * 1.000001, abs(gold_value) * rel_tol)
        if abs(abs(n.value) - abs(gold_value)) <= tol:
            return True
    return False


def classify(record: dict) -> str:
    """One of: correct | wrong | honest_abstain | over_abstain."""
    answered = record["decision"] in ("answer", "answer_flagged")
    if record["answerable"]:
        if not answered:
            return "over_abstain"
        return "correct" if record["numeric_correct"] else "wrong"
    return "wrong" if answered else "honest_abstain"


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    ans_items = [r for r in records if r["answerable"]]
    un_items = [r for r in records if not r["answerable"]]
    cls = [classify(r) for r in records]
    c = {k: cls.count(k) for k in ("correct", "wrong", "honest_abstain", "over_abstain")}

    answered = [r for r in records if r["decision"] in ("answer", "answer_flagged")]
    answered_ans = [r for r in ans_items if r["decision"] in ("answer", "answer_flagged")]
    wrong_answered = sum(1 for r in answered if classify(r) == "wrong")

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else None

    out = {
        "n": n, "n_answerable": len(ans_items), "n_unanswerable": len(un_items),
        "counts": c,
        "coverage_pct": pct(len(answered_ans), len(ans_items)),
        "selective_risk_pct": pct(wrong_answered, len(answered)),
        "answer_accuracy_pct": pct(c["correct"], len(ans_items)),
        "hallucination_rate_pct": pct(wrong_answered, n),
        "honest_abstention_pct": pct(sum(1 for r in un_items if r["decision"] == "abstain"),
                                     len(un_items)),
        "over_abstention_pct": pct(c["over_abstain"], len(ans_items)),
        "flagged_pct": pct(sum(1 for r in records if r["decision"] == "answer_flagged"), n),
        "mean_latency_ms": round(sum(r.get("latency_ms", 0) for r in records) / n, 1) if n else 0,
        "p95_latency_ms": (sorted(r.get("latency_ms", 0) for r in records)[int(0.95 * (n - 1))]
                           if n else 0),
        "mean_iterations": round(sum(r.get("iterations", 1) for r in records) / n, 2) if n else 0,
    }
    out.update(risk_coverage(records))
    out.update(retrieval_metrics(records))
    by_type: dict[str, dict] = {}
    for r in records:
        t = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "wrong": 0,
                                           "honest_abstain": 0, "over_abstain": 0})
        t["n"] += 1
        t[classify(r)] += 1
    out["by_type"] = by_type
    return out


def risk_coverage(records: list[dict]) -> dict:
    """Sweep the confidence score; report the risk-coverage curve and its area."""
    scored = [r for r in records if r["answerable"]]
    if not scored:
        return {"aurc": None, "risk_coverage_curve": []}
    ranked = sorted(scored, key=lambda r: -float(r.get("groundedness") or 0.0))
    curve, wrong = [], 0
    risks = []
    for i, r in enumerate(ranked, start=1):
        if not r["numeric_correct"]:
            wrong += 1
        cov = i / len(ranked)
        risk = wrong / i
        risks.append(risk)
        curve.append({"coverage": round(cov, 4), "risk": round(risk, 4),
                      "threshold": round(float(r.get("groundedness") or 0.0), 4)})
    return {"aurc": round(sum(risks) / len(risks), 4), "risk_coverage_curve": curve}


def _dcg(rels: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def retrieval_metrics(records: list[dict], ks=(1, 3, 5, 10)) -> dict:
    """Recall@k / MRR / nDCG@k against the gold chunk ids from the benchmark builder."""
    rel = [r for r in records if r.get("gold_chunk_ids")]
    if not rel:
        return {"retrieval": None}
    out: dict[str, float] = {}
    for k in ks:
        hits = 0
        for r in rel:
            got = set(r.get("retrieved_chunk_ids", [])[:k])
            if got & set(r["gold_chunk_ids"]):
                hits += 1
        out[f"recall@{k}"] = round(100.0 * hits / len(rel), 2)
    rr, ndcg = [], []
    for r in rel:
        got = r.get("retrieved_chunk_ids", [])
        gold = set(r["gold_chunk_ids"])
        rank = next((i + 1 for i, c in enumerate(got) if c in gold), None)
        rr.append(1.0 / rank if rank else 0.0)
        rels = [1 if c in gold else 0 for c in got[:10]]
        ideal = sorted(rels, reverse=True)
        ndcg.append(_dcg(rels) / _dcg(ideal) if any(ideal) else 0.0)
    out["mrr"] = round(sum(rr) / len(rr), 4)
    out["ndcg@10"] = round(sum(ndcg) / len(ndcg), 4)
    return {"retrieval": out}


def compare(arms: dict[str, dict]) -> str:
    """Markdown comparison table across ablation arms - drops straight into the report."""
    cols = [("answer_accuracy_pct", "Acc%"), ("hallucination_rate_pct", "Halluc%"),
            ("coverage_pct", "Cov%"), ("selective_risk_pct", "SelRisk%"),
            ("honest_abstention_pct", "HonAbst%"), ("over_abstention_pct", "OverAbst%"),
            ("aurc", "AURC"), ("mean_latency_ms", "ms")]
    lines = ["| Arm | " + " | ".join(c[1] for c in cols) + " | R@5 | MRR |",
             "|" + "---|" * (len(cols) + 3)]
    for name, m in arms.items():
        r = m.get("retrieval") or {}
        vals = [str(m.get(c[0], "-")) for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) +
                     f" | {r.get('recall@5', '-')} | {r.get('mrr', '-')} |")
    return "\n".join(lines)
