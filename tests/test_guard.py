"""The guardrail tests are the ones that matter: each asserts a specific,
documented hallucination mode is caught (or that a legitimate answer is not)."""
from finrag.guard.claims import extract_claims, is_abstention, split_claims_rule
from finrag.guard.policy import ABSTAIN, ANSWER, FLAGGED, Decision, decide, strike_ungrounded
from finrag.guard.verifier import groundedness, verify_claims

CTX = [{
    "chunk_id": "k1", "page_no": 7, "unit_multiplier": None, "section_path": "Asset Quality",
    "content": ("| Ratio | As at 31 March 2024 | As at 31 March 2023 |\n"
                "| Gross NPA ratio (%) | 2.24 | 2.78 |\n"
                "| Net NPA ratio (%) | 0.57 | 0.67 |\n"
                "| Provision coverage ratio (%) | 91.89 | 90.14 |"),
    "meta": {},
}]


def run(answer):
    claims, _ = extract_claims(answer, None)
    verify_claims(claims, CTX)
    return claims, groundedness(claims)


def test_correct_figure_is_supported():
    _, g = run("The Gross NPA ratio stood at 2.24 per cent as at 31 March 2024 [C1].")
    assert g["score"] == 1.0 and g["numeric_ok"]


def test_fabricated_figure_is_caught():
    claims, g = run("The Gross NPA ratio stood at 3.91 per cent as at 31 March 2024 [C1].")
    assert not g["numeric_ok"] and "fabricated" in claims[0].reason


def test_wrong_row_grounding_is_caught():
    """2.24 exists in the context - but it is the GROSS NPA row, not the Net NPA row."""
    claims, g = run("The Net NPA ratio stood at 2.24 per cent as at 31 March 2024 [C1].")
    assert not g["numeric_ok"] and "wrong-row" in claims[0].reason


def test_legitimate_rounding_is_accepted_not_flagged():
    _, g = run("The Gross NPA ratio was about 2.2 per cent [C1].")
    assert g["numeric_ok"], "over-cautious abstention on a valid rounding"


def test_over_precision_is_rejected():
    _, g = run("The Gross NPA ratio was 2.2413 per cent [C1].")
    assert not g["numeric_ok"]


def test_derived_claim_recomputed_from_operands():
    _, g = run("The Gross NPA ratio fell by 54 basis points year-on-year [C1].")
    assert g["numeric_ok"], "a correctly computed delta must not be called a hallucination"


def test_wrong_derived_claim_is_caught():
    _, g = run("The Gross NPA ratio fell by 88 basis points year-on-year [C1].")
    assert not g["numeric_ok"]


def test_no_context_means_no_support():
    claims, _ = extract_claims("Gross NPA was 2.24 per cent [C1].", None)
    verify_claims(claims, [])
    assert not claims[0].supported


def test_claim_splitter_separates_clauses_and_keeps_citations():
    claims = split_claims_rule("Gross NPA was 2.24 per cent [C1]; Net NPA was 0.57 per cent [C2].")
    assert len(claims) == 2 and claims[0].cited == [1] and claims[1].cited == [2]


def test_numeric_weighting_dominates_groundedness():
    claims, _ = extract_claims(
        "Asset quality improved during the year [C1]. Gross NPA was 9.99 per cent [C1].", None)
    verify_claims(claims, CTX)
    g = groundedness(claims, w_numeric=2.0)
    assert g["score"] < 0.5 and not g["numeric_ok"]


def test_policy_bands():
    class C:
        tau_high, tau_low, max_iters, hard_numeric_gate = 0.90, 0.60, 0, True
    good = {"score": 0.95, "numeric_ok": True, "n_claims": 2, "n_unsupported": 0, "unsupported": []}
    mid = {"score": 0.75, "numeric_ok": True, "n_claims": 2, "n_unsupported": 1, "unsupported": []}
    bad = {"score": 0.20, "numeric_ok": False, "n_claims": 2, "n_unsupported": 2, "unsupported": []}
    assert decide("a", [], good, 0, C).decision == ANSWER
    assert decide("a", [], mid, 0, C).decision == FLAGGED
    assert decide("a", [], bad, 0, C).decision == ABSTAIN


def test_hard_numeric_gate_blocks_a_high_score_with_a_bad_number():
    class C:
        tau_high, tau_low, max_iters, hard_numeric_gate = 0.90, 0.60, 0, True
    g = {"score": 0.93, "numeric_ok": False, "n_claims": 3, "n_unsupported": 1, "unsupported": []}
    assert decide("a", [], g, 0, C).decision != ANSWER


def test_retry_then_stop_is_bounded():
    class C:
        tau_high, tau_low, max_iters, hard_numeric_gate = 0.90, 0.60, 2, True
    g = {"score": 0.75, "numeric_ok": True, "n_claims": 2, "n_unsupported": 1, "unsupported": []}
    assert decide("a", [], g, 0, C).next_action == "reretrieve"
    assert decide("a", [], g, 2, C).next_action == "stop"


def test_unsupported_spans_are_struck_not_deleted():
    claims, _ = extract_claims("Gross NPA was 9.99 per cent [C1].", None)
    verify_claims(claims, CTX)
    marked, spans = strike_ungrounded("Gross NPA was 9.99 per cent.", claims)
    assert "~~" in marked and spans and "UNVERIFIED" in marked


def test_model_self_abstention_detected():
    assert is_abstention("INSUFFICIENT_CONTEXT: the figure is not disclosed.")
    assert not is_abstention("Gross NPA was 2.24 per cent.")

def test_llm_mislabel_cannot_route_a_figure_to_the_textual_verifier():
    """A numeric claim typed 'textual' would skip deterministic checking entirely."""
    class StubLLM:
        def complete(self, *a, **k):
            return '[{"text": "The ratio fell from 2.78 per cent [C1]", "kind": "textual"}]'
    claims, fallback = extract_claims("The ratio fell from 2.78 per cent [C1].", StubLLM())
    assert not fallback
    assert claims[0].kind in ("numeric", "derived")
