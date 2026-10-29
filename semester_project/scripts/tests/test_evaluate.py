"""Offline test of the E2–E4 metric functions and the study alignment in
scripts/evaluation/evaluate.py, on small synthetic data with answers computed by hand.

Usage: python -I scripts/tests/test_evaluate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from common import FINDINGS  # noqa: E402
import evaluate as ev  # noqa: E402

ok = True
F = len(FINDINGS)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def labels(*present):
    """(F,) bool with the given finding indices set."""
    v = np.zeros(F, bool)
    v[list(present)] = True
    return v


# --- precision@k (E2) ---
# 3 queries, k = 2. Query 0 has finding 0; its neighbours: one with 0, one normal -> 0.5.
# Query 1 has finding 0; neighbours both with 0 -> 1.0. Finding 0 precision = (0.5 + 1.0) / 2 = 0.75.
# Query 2 is normal; neighbours: one normal, one with finding 1 -> normal share 0.5.
yq = np.array([labels(0), labels(0), labels()])
yn = np.array([[labels(0), labels()], [labels(0), labels(0, 2)], [labels(), labels(1)]])
ytr = np.array([labels(0), labels(), labels(), labels()])  # prevalence of finding 0 = 0.25, normal = 0.75
p = ev.precision_at_k(yq, yn, ytr)
ok &= check("precision@k: per finding over queries that have it",
            p[FINDINGS[0]]["queries"] == 2 and abs(p[FINDINGS[0]]["precision"] - 0.75) < 1e-12)
ok &= check("precision@k: lift = precision / train prevalence", abs(p[FINDINGS[0]]["lift"] - 3.0) < 1e-12)
ok &= check("precision@k: finding with no positive query -> nan, 0 queries",
            p[FINDINGS[3]]["queries"] == 0 and np.isnan(p[FINDINGS[3]]["precision"]))
ok &= check("precision@k: normal row (normal queries, share of normal neighbours)",
            p["normal"]["queries"] == 1 and abs(p["normal"]["precision"] - 0.5) < 1e-12
            and abs(p["normal"]["prevalence"] - 0.75) < 1e-12)

# --- copying (E3) ---
# One finding column matters (0); 4 studies.
# truth     1 0 0 1
# knn       1 1 0 0   -> wrong at studies 1 and 3
# system    1 1 1 1   -> agrees at 0, 1; disagrees at 2, 3. Agree all = 2/4; agree when wrong = 1/2.
y = np.zeros((4, F), bool)
kd = np.zeros((4, F), bool)
sb = np.zeros((4, F), bool)
y[[0, 3], 0] = True
kd[[0, 1], 0] = True
sb[:, 0] = True
c = ev.copying(sb, kd, y)
ok &= check("copying: agreement overall for one finding", abs(c[FINDINGS[0]]["agree"] - 0.5) < 1e-12)
ok &= check("copying: agreement where the neighbours' decision is wrong",
            c[FINDINGS[0]]["neighbours_wrong"] == 2 and abs(c[FINDINGS[0]]["agree_when_wrong"] - 0.5) < 1e-12)
# pooled: other 5 findings all 0 everywhere -> 20 more agreeing pairs, none wrong
ok &= check("copying: pooled over findings",
            c["pooled"]["pairs"] == 24 and abs(c["pooled"]["agree"] - 22 / 24) < 1e-12
            and c["pooled"]["neighbours_wrong"] == 2 and abs(c["pooled"]["agree_when_wrong"] - 0.5) < 1e-12)
ok &= check("copying: finding where neighbours are never wrong -> nan", np.isnan(c[FINDINGS[1]]["agree_when_wrong"]))

# --- present rate by number of positive neighbours (E3) ---
n_pos = np.array([[0, 1], [2, 5], [0, 3]])
sbin = np.array([[True, False], [True, True], [False, False]])
pr = ev.present_by_positive_neighbours(sbin, n_pos)
ok &= check("present by positive neighbours: bins 0 / 1 / 2 / 3+",
            pr["0"]["pairs"] == 2 and pr["0"]["present_rate"] == 0.5
            and pr["1"]["pairs"] == 1 and pr["1"]["present_rate"] == 0.0
            and pr["2"]["pairs"] == 1 and pr["2"]["present_rate"] == 1.0
            and pr["3+"]["pairs"] == 2 and pr["3+"]["present_rate"] == 0.5)

# --- citation agreement (E4) ---
lab = {"A": labels(0), "B": labels(), "C": labels(0), "D": labels(), "Q1": labels(0), "Q2": labels()}
nb = {"Q1": ["A", "B", "C", "D"], "Q2": ["A", "B", "C", "D"]}
f0, f1 = FINDINGS[0], FINDINGS[1]
verdicts = [
    ("Q1", f0, "present", ["A", "B", "C"]),  # 2 of 3 cited have f0
    ("Q2", f0, "present", []),               # present without citation
    ("Q1", f1, "absent", ["B"]),             # citation under absent: counted only
    ("Q2", f1, "uncertain", []),
]
ca = ev.citation_agreement(verdicts, lab, nb)
o = ca[f0]
ok &= check("citations: present calls and present without citation",
            o["present_calls"] == 2 and o["present_without_citation"] == 1)
ok &= check("citations: share of cited cases with the finding", o["cited"] == 3 and abs(o["share"] - 2 / 3) < 1e-12)
# baseline over both present calls: 4 + 4 neighbours, 2 + 2 with f0 -> 0.5
ok &= check("citations: baseline = share of all given neighbours", abs(o["baseline"] - 0.5) < 1e-12)
ok &= check("citations: citations under absent / uncertain counted, not scored",
            ca["citations_under_absent_or_uncertain"] == 1 and ca[f1]["present_calls"] == 0
            and np.isnan(ca[f1]["share"]))

# --- alignment and per-finding metrics ---
ok &= check("common studies: intersection in CXR number order",
            ev.common_studies({"CXR10": 1, "CXR2": 1, "CXR3": 1}, {"CXR3": 0, "CXR10": 0, "CXR7": 0})
            == ["CXR3", "CXR10"])
yy = np.array([labels(0), labels(0), labels(), labels()])
bb = np.array([labels(0), labels(), labels(0), labels()])
sens = ev.per_finding(yy, bb, "sensitivity")
spec = ev.per_finding(yy, bb, "specificity")
ok &= check("per_finding on binary verdicts: sensitivity and specificity",
            sens[0] == 0.5 and spec[0] == 0.5 and np.isnan(sens[1]) and spec[1] == 1.0)
au = ev.per_finding(yy, np.array([[0.9] * F, [0.2] * F, [0.5] * F, [0.1] * F]), "auroc")
ok &= check("per_finding AUROC (positives 0.9, 0.2 vs. negatives 0.5, 0.1 -> 3/4)", au[0] == 0.75 and np.isnan(au[1]))
ok &= check("outcome: hit / miss / false alarm / correct absent",
            [ev.outcome(True, True), ev.outcome(True, False), ev.outcome(False, True), ev.outcome(False, False)]
            == ["hit", "miss", "false alarm", "correct absent"])
se, sp = ev.sens_spec_from_outcomes(["hit", "miss", "miss", "false alarm", "correct absent", "correct absent",
                                     "correct absent"])
ok &= check("sensitivity / specificity counted from outcomes (1/3, 3/4)", abs(se - 1 / 3) < 1e-12 and sp == 0.75)
ok &= check("outcomes without positives -> sensitivity nan", np.isnan(ev.sens_spec_from_outcomes(["correct absent"])[0]))
ok &= check("comparisons: exactly one primary, G - S0 on AUROC",
            [c[:4] for c in ev.COMPARISONS if c[0] == 1] == [(1, "G", "S0", "auroc")])

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
