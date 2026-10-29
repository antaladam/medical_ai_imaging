"""Offline test of the text functions in scripts/evaluation/memorisation.py (hand-computed answers).

Usage: python -I scripts/tests/test_memorisation.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import memorisation as mem  # noqa: E402

ok = True


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok &= check("tokens: lower case, punctuation and XXXX tokens removed, other x-words kept",
            mem.tokens("The XXXX heart, is NORMAL. xxxx-ray x") == ["the", "heart", "is", "normal", "ray", "x"])
ok &= check("tokens: 'exxon' is not a de-identification token", mem.tokens("Exxon") == ["exxon"])
# reference 3-grams: (the heart is) (heart is normal) -> 2; text contains only the first -> 0.5
ok &= check("containment: share of the reference's 3-grams found in the text",
            mem.containment("The heart is normal.", "Here the heart is enlarged") == 0.5)
ok &= check("containment: duplicate 3-grams counted once",
            mem.containment("a b c a b c", "a b c") == 1 / 3)  # distinct: abc, bca, cab; only abc found
ok &= check("containment: reference too short -> nan", math.isnan(mem.containment("heart normal", "heart normal")))
ok &= check("containment: identical text -> 1.0",
            mem.containment("No acute cardiopulmonary disease.", "no acute cardiopulmonary disease") == 1.0)

idf = {"heart": 1.0, "normal": 2.0, "the": 0.0}
ok &= check("tf-idf cosine: identical -> 1, disjoint -> 0, all-zero weights -> nan",
            abs(mem.tfidf_cosine("heart normal", "normal heart", idf) - 1) < 1e-12
            and mem.tfidf_cosine("heart", "normal", idf) == 0.0
            and math.isnan(mem.tfidf_cosine("the", "the", idf)))

answer = "Lung Opacity: present\n**Cardiomegaly**: absent\n- Fracture: uncertain\nThe heart is normal.\nLung Opacity is seen."
ok &= check("strip_checklist removes checklist lines only (same rule as the scoring)",
            mem.strip_checklist(answer) == "The heart is normal.\nLung Opacity is seen.")

same = {("0",): ["CXR1", "CXR2", "CXR3", "CXR10", "CXR11"]}
ok &= check("baseline pool: the study itself and shown reports excluded",
            mem.baseline_pool("CXR1", ("0",), same, {"CXR3"}) == ["CXR2", "CXR10", "CXR11"])
big = {("0",): [f"CXR{i}" for i in range(1, 100)]}
p1 = mem.baseline_pool("CXR5", ("0",), big, set())
ok &= check("baseline pool: 20 drawn, reproducible per study, never the study itself",
            len(p1) == 20 and p1 == mem.baseline_pool("CXR5", ("0",), big, set()) and "CXR5" not in p1)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
