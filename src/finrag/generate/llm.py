"""LLM client (any OpenAI-compatible endpoint) + the grounded-answer prompt.

Prompt design decisions that matter for the guardrail:
 * every sentence must carry a [C#] citation  -> claims become checkable
 * quote figures EXACTLY as printed           -> no silent rounding/scale conversion
 * unit banner is restated per context block  -> stops crore/lakh conversion errors
 * explicit 'INSUFFICIENT_CONTEXT' escape     -> gives the model a legal way to abstain
   before the guardrail has to force it (cheaper, and improves honest-abstention rate)
 * temperature 0 + optional self-consistency  -> determinism for reproducible evals
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import CFG

SYSTEM = """You are a financial analyst assistant answering ONLY from the supplied context.

HARD RULES
1. Use ONLY the numbered context blocks. Never use prior knowledge about any company or the RBI.
2. Copy every figure EXACTLY as printed in the context, including commas and decimals.
   Never convert between lakh, crore, million or billion. Never round.
3. State the unit and the period with every figure, e.g. "Rs. 12,345.67 crore in FY2023-24".
4. Cite the block you used after EVERY sentence, like [C2]. Sentences without a citation are invalid.
5. Never mix consolidated and standalone figures. Never mix fiscal years.
6. If the context does not contain the answer, reply exactly:
   INSUFFICIENT_CONTEXT: <one line naming the specific figure that is missing>
7. For a computed quantity (YoY change, growth %, bps), first show the arithmetic on one line
   using only cited figures, then give the result.
"""

USER_TMPL = """CONTEXT
{context}

QUESTION
{question}

Answer in at most {max_sentences} sentences. Every sentence must cite a context block."""


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        from openai import OpenAI
        self.model = model or CFG.llm_model
        self.client = OpenAI(base_url=base_url or CFG.llm_base_url,
                             api_key=api_key or CFG.llm_api_key or "none")

    def complete(self, prompt: str, system: str | None = None, max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(
            model=self.model, messages=msgs,
            temperature=CFG.temperature if temperature is None else temperature,
            max_tokens=max_tokens or CFG.max_tokens)
        return (r.choices[0].message.content or "").strip()

    def answer(self, question: str, context: str, max_sentences: int = 5) -> str:
        return self.complete(USER_TMPL.format(context=context, question=question,
                                              max_sentences=max_sentences),
                             system=SYSTEM)


class EchoLLM:
    """Offline stand-in used by tests/CI/demos when no LLM endpoint is reachable.

    It picks the context line with the highest content-word overlap with the
    question and echoes it with a citation; below an overlap floor it emits
    INSUFFICIENT_CONTEXT. It is deliberately *imperfect* (it can still pick the
    wrong row), which is the point: the guardrail must catch its mistakes rather
    than be tested against a model that never errs.
    """
    model = "echo-offline"
    _STOP = set("what was the of in on at to for and or is are a an as by with during year "
                "ended how much many did does bank report reported figure value".split())

    def complete(self, prompt: str, system: str | None = None, **kw) -> str:
        return "[]"

    def _words(self, t: str) -> set:
        import re
        return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in self._STOP}

    def answer(self, question: str, context: str, max_sentences: int = 5) -> str:
        import re
        qw = self._words(question)
        best, best_ov, idx = "", 0.0, 1
        for block in context.split("\n\n"):
            m = re.match(r"\[C(\d+)\]", block)
            ci = int(m.group(1)) if m else 1
            for line in block.split("\n")[1:]:      # skip the [C#] provenance header
                if not re.search(r"\d", line) or line.startswith("|---"):
                    continue
                ov = len(qw & self._words(line)) / (len(qw) or 1)
                if ov > best_ov:
                    best, best_ov, idx = line.strip(), ov, ci
        if best_ov < 0.25:
            return ("INSUFFICIENT_CONTEXT: the retrieved passages do not contain the "
                    "requested figure.")
        return f"{best} [C{idx}]"


def get_llm(offline: bool | None = None):
    if offline is None:
        offline = os.getenv("LLM_OFFLINE", "0") == "1"
    return EchoLLM() if offline else LLMClient()
