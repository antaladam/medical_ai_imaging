"""Offline test of scripts/g/g_system.py (no API calls).

Usage: python -I scripts/tests/test_g_system.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "g"))
import g_system as G  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


nb = [{"study": f"CXR{i}", "similarity": 0.98 - i / 1000, "report": f"FINDINGS: report {i}."} for i in range(1, 11)]
g = G.build_prompt("g", 1, "Chest pain.", nb)
gi = G.build_prompt("g_ind", 1, "Chest pain.", nb)
gm = G.build_prompt("g_img", 1, "Chest pain.", nb)
check("G contains the indication, the definitions and all 10 cases",
      "Clinical indication: Chest pain." in g and G.DEFINITIONS in g and all(n["study"] in g for n in nb))
import re  # noqa: E402


def unnumbered(text):  # step numbers shift when a step is removed; compare the steps themselves
    return [re.sub(r"^\d+\. ", "", ln) for ln in text.splitlines()]


removed = [ln for ln in unnumbered(g) if ln not in unnumbered(gi)]
added = [ln for ln in unnumbered(gi) if ln not in unnumbered(g)]
check("G-ind = G minus the indication line and the step that refers to it (nothing else)",
      sorted(removed) == sorted(["Clinical indication: Chest pain.", "Take the clinical indication into account."])
      and added == [])
check("G-img has no indication and no cases, but the same definitions and output step",
      "Clinical indication" not in gm and "Similar cases" not in gm and "CXR1" not in gm
      and G.DEFINITIONS in gm and G.PROMPTS[1]["step_output"] in gm)
check("missing indication is stated, not left blank", "Clinical indication: not provided" in G.build_prompt("g", 1, "", nb))

good = {"findings": [{"finding": f, "status": "absent", "probability": 0.1, "cited_case_ids": ["CXR1"],
                      "rationale": "x"} for f in G.FINDINGS], "impression": "normal"}
check("valid response -> no problems", G.validate(good, {"CXR1"}) == [])
bad = {"findings": good["findings"][:5] + [dict(good["findings"][5], probability=1.5, cited_case_ids=["CXR999"])],
       "impression": ""}
p = G.validate(bad, {"CXR1"})
check("probability out of range and uncited case ids are reported", any("out of range" in x for x in p)
      and any("CXR999" in x for x in p))
check("a missing finding is reported", G.validate({"findings": good["findings"][:5]}, {"CXR1"}) != [])

with open(G.PROC / "splits.csv") as fh:
    splits = {r["study"]: r["split"] for r in csv.DictReader(fh)}
ret = G.Retriever(splits)
train0 = ret.train[0]
try:
    ret.top(train0); check("leakage guard", False)
except AssertionError:
    check("a train study as query is refused (leakage guard)", True)

with open(G.PROJECT / "data" / "runs" / "r0" / "val.csv") as fh:
    r0 = {r["study"]: r["neighbours"].split(";")[:10] for r in csv.DictReader(fh)}
same = all([n["study"] for n in ret.top(s)] == r0[s] for s in r0)
check(f"G's 10 neighbours equal R0's first 10 for all {len(r0)} validation studies", same)
sims = [n["similarity"] for n in ret.top(next(iter(r0)))]
check("neighbours sorted by similarity, highest first", sims == sorted(sims, reverse=True))

# --- prompt version 2 ---
import json  # noqa: E402

g2 = G.build_prompt("g", 2, "Chest pain.", nb)
check("v2 lists cases by ID without position numbers", "Case CXR1 (similarity" in g2 and "[1]" not in g2)
check("v2 asks for citation by ID and two-decimal probabilities",
      "Cite cases by their ID exactly as written, e.g. CXR316." in g2 and "two decimals" in g2)
diff = sorted(set(unnumbered(g2)) ^ set(unnumbered(g)))
check("v2 differs from v1 only in the case lines and the output step",
      all(ln.startswith(("Case CXR", "[", "For each of the six findings")) for ln in diff))
# The refactor must not change version 1: rebuild every cached prompt-1 prompt and compare.
cached = sorted((G.PROJECT / "data" / "runs" / "g__prompt1" / "val").glob("*.json"))
same = True
for f in cached:
    rec = json.loads(f.read_text(encoding="utf-8"))
    n = [{"study": x["study"], "similarity": x["similarity"], "report": G.report_text(x["study"])}
         for x in rec["neighbours"]]
    same &= G.build_prompt("g", 1, rec["indication"], n) == rec["prompt"]
check(f"prompt 1 rebuilt identically for all {len(cached)} cached pilot answers", cached and same)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
