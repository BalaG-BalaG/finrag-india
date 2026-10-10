"""RAGAS adapter (kept in its own venv - see requirements-eval.txt).

    python eval/ragas_eval.py --records eval/results/B2_struct_guard_records.json

Why RAGAS is a SECOND opinion here, not the primary metric:
  * RAGAS Faithfulness is an LLM judge; it is fuzzy about digits, which is exactly
    where this project needs precision. Our deterministic numeric verifier is the
    primary faithfulness signal; RAGAS corroborates it on the textual dimension and
    gives the report a comparable, published metric.
  * Abstentions have no claims to judge, so they are excluded from faithfulness and
    reported separately - averaging them in would silently reward refusing.

Install note (this bites people): ragas 0.4.x fails to import with
langchain-community 0.4.x (`langchain_community.chat_models.vertexai`), so pin
`langchain-community<0.4` in the eval environment.

    pip install -r requirements-eval.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os


async def score(records: list[dict], model: str, api_key: str | None) -> dict:
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision, ContextRecall,
                                           Faithfulness)

    client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    llm = llm_factory(model, client=client)
    emb = embedding_factory(provider="openai", model="text-embedding-3-small", client=client)

    faith, arel = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb)
    cprec, crec = ContextPrecision(llm=llm), ContextRecall(llm=llm)

    answered = [r for r in records if r["decision"] in ("answer", "answer_flagged")]
    out: dict[str, list[float]] = {"faithfulness": [], "answer_relevancy": [],
                                   "context_precision": [], "context_recall": []}
    for r in answered:
        ctx = r.get("contexts") or []
        if not ctx:
            continue
        ref = str(r.get("gold_text") or r.get("gold_value") or "")
        try:
            out["faithfulness"].append((await faith.ascore(
                user_input=r["question"], response=r["answer"], retrieved_contexts=ctx)).value)
            out["answer_relevancy"].append((await arel.ascore(
                user_input=r["question"], response=r["answer"])).value)
            if ref:
                out["context_precision"].append((await cprec.ascore(
                    user_input=r["question"], reference=ref, retrieved_contexts=ctx)).value)
                out["context_recall"].append((await crec.ascore(
                    user_input=r["question"], retrieved_contexts=ctx, reference=ref)).value)
        except Exception as e:
            print(f"  [skip {r['qid']}] {e}")

    means = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in out.items()}
    means["n_scored"] = len(out["faithfulness"])
    means["n_abstained_excluded"] = len(records) - len(answered)
    return means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="eval/results/B2_struct_guard_records.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="eval/results/ragas.json")
    a = ap.parse_args()
    recs = json.load(open(a.records))
    res = asyncio.run(score(recs, a.model, a.api_key))
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
