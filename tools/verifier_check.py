"""Checks the verifier directly: we write the answer, it gives a verdict.
No LLM involved, so it's fast and repeatable. Run before and after a fix."""
import sys
sys.path.insert(0, "src")

from finrag.store import db
from finrag.retrieve.hybrid import retrieve, RetrievalConfig
from finrag.guard.claims import extract_claims
from finrag.guard.verifier import verify_claims

# (answer sentence, should it be accepted?)
CASES = [
    # Type B: years down the side (RBI Table VIII.8)
    ("The Reserve Bank detected 17,613 counterfeit notes in 2023-24.", True),
    ("Counterfeit notes detected at other banks in 2022-23 were 2,15,304.", True),
    ("The Reserve Bank detected 19,500 counterfeit notes in 2023-24.", False),  # made-up number
    ("The Reserve Bank detected 10,465 counterfeit notes in 2023-24.", False),  # real number, wrong year
    # Type A: metric names down the side (must keep working)
    ("Bharat National Bank's Gross NPA ratio was 2.24 per cent as at 31 March 2024.", True),
    ("Bharat National Bank's Net NPA ratio was 2.24 per cent as at 31 March 2024.", False),  # wrong row
]

ok = 0
with db.connect() as c:
    for sentence, expected in CASES:
        contexts = retrieve(c, sentence, RetrievalConfig())["contexts"]
        claims, _ = extract_claims(sentence + " [C1]", None)
        verify_claims(claims, contexts, conn=c)
        got = all(cl.supported for cl in claims) if claims else False
        right = got == expected
        ok += right
        print(f"{'PASS' if right else 'FAIL'}  expected={'accept' if expected else 'reject'}  "
              f"got={'accept' if got else 'reject'}")
        print(f"      {sentence}")
        for cl in claims:
            if not cl.supported:
                print(f"      why: {cl.reason[:110]}")
print(f"\n{ok}/{len(CASES)} correct")