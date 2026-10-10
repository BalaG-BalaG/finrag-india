"""Repair turns a specific recoverable failure into an answer without weakening any
safety guarantee. These tests pin both halves: it fires when it should, and - more
importantly - it refuses whenever an operand is not already verifier-grounded.
"""
from finrag.guard.claims import Claim
from finrag.guard.repair import repair_derived, wants_computation

CTX = [{
    "chunk_id": "k1", "page_no": 7, "unit_multiplier": None, "section_path": "Asset Quality",
    "content": ("| Ratio | FY2023-24 | FY2022-23 |\n"
                "| Gross NPA ratio (%) | 2.24 | 2.78 |\n"
                "| Net profit | 57,766.67 | 42,469.13 |"),
    "meta": {},
}]


def grounded(text: str, cid: str = "c1") -> Claim:
    c = Claim(claim_id=cid, text=text, kind="numeric")
    c.supported, c.score = True, 1.0
    c.evidence = [{"chunk_id": "k1", "context_index": 1, "row_label": "Gross NPA ratio (%)",
                   "evidence": "| Gross NPA ratio (%) | 2.24 | 2.78 |"}]
    return c


def ungrounded(text: str, cid: str = "c1") -> Claim:
    c = Claim(claim_id=cid, text=text, kind="numeric")
    c.supported, c.score, c.reason = False, 0.0, "fabricated"
    return c


def test_operation_detected_from_question_cues():
    assert wants_computation("By how many basis points did X change?") == "bps_change"
    assert wants_computation("What was the percentage change in X?") == "pct_change"
    assert wants_computation("How did X change from FY2022-23 to FY2023-24?") == "abs_diff"
    assert wants_computation("What was X in FY2023-24?") is None


def test_bps_change_computed_from_two_grounded_operands():
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              grounded("Gross NPA in FY2022-23 was 2.78 per cent", "c2")]
    rep = repair_derived(q, "draft", claims, CTX)
    assert rep is not None
    assert rep["claim"].evidence[0]["computed"] == -54.0
    assert rep["claim"].supported and rep["claim"].kind == "derived"


def test_percentage_change_computed():
    q = "What was the percentage change in Net profit from FY2022-23 to FY2023-24?"
    claims = [grounded("Net profit in FY2023-24 was 57,766.67 crore", "c1"),
              grounded("Net profit in FY2022-23 was 42,469.13 crore", "c2")]
    rep = repair_derived(q, "draft", claims, CTX)
    assert rep is not None
    assert abs(rep["claim"].evidence[0]["computed"] - 36.02) < 0.01


def test_repair_cites_both_operands_for_audit():
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              grounded("Gross NPA in FY2022-23 was 2.78 per cent", "c2")]
    ops = repair_derived(q, "draft", claims, CTX)["claim"].evidence[0]["operands"]
    assert {o["claim_id"] for o in ops} == {"c1", "c2"}


# --- the safety half: repair must never manufacture support ---

def test_no_repair_when_an_operand_is_unverified():
    """The whole guarantee: arithmetic over a fabricated figure is still a hallucination."""
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              ungrounded("Gross NPA in FY2022-23 was 3.99 per cent", "c2")]
    assert repair_derived(q, "draft", claims, CTX) is None


def test_no_repair_for_a_level_question():
    q = "What was the Gross NPA ratio in FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              grounded("Gross NPA in FY2022-23 was 2.78 per cent", "c2")]
    assert repair_derived(q, "draft", claims, CTX) is None


def test_no_repair_when_only_one_period_is_grounded():
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1")]
    assert repair_derived(q, "draft", claims, CTX) is None


def test_no_duplicate_when_model_already_computed_it():
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    draft = "Gross NPA fell by 54 basis points."
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              grounded("Gross NPA in FY2022-23 was 2.78 per cent", "c2")]
    assert repair_derived(q, draft, claims, CTX) is None


def test_no_bps_repair_on_a_non_percentage_metric():
    """Basis points are only meaningful between two percentages."""
    q = "By how many basis points did Net profit change from FY2022-23 to FY2023-24?"
    c1 = Claim(claim_id="c1", text="Net profit in FY2023-24 was 57,766.67 crore", kind="numeric")
    c2 = Claim(claim_id="c2", text="Net profit in FY2022-23 was 42,469.13 crore", kind="numeric")
    for c in (c1, c2):
        c.supported, c.score = True, 1.0
        c.evidence = [{"chunk_id": "k1", "context_index": 1, "row_label": "Net profit",
                       "evidence": "| Net profit | 57,766.67 | 42,469.13 |"}]
    assert repair_derived(q, "draft", [c1, c2], CTX) is None


def test_no_repair_on_conflicting_values_for_one_period():
    q = "By how many basis points did Gross NPA change from FY2022-23 to FY2023-24?"
    claims = [grounded("Gross NPA in FY2023-24 was 2.24 per cent", "c1"),
              grounded("Gross NPA in FY2023-24 was 2.31 per cent", "c2"),
              grounded("Gross NPA in FY2022-23 was 2.78 per cent", "c3")]
    assert repair_derived(q, "draft", claims, CTX) is None
