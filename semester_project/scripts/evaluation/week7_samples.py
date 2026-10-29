"""Draw the week-7 review samples and write review sheets (week7_design.md W1, W3; decided
2026-10-18). Test set, cached outputs only; seed 2026.

W1 failure review (23 errors, from data/runs/eval/test/per_study.csv):
  G misses 4, G false alarms 4 (stated); G wrong / S0 right 2 and S0 wrong / G right 2
  (threshold readings); G copied a wrong neighbours' decision while G-img was right 3 (stated;
  the kNN@10 decision of evaluate.py E3); each starter 1 miss + 1 false alarm (stated);
  G-img's invented-citation answers CXR206, CXR3080 (one row each, finding = the first cited one).
  Each stratum: distinct studies, not already in an earlier stratum.
W3 hand check of G: 15 G test answers, 8 with at least one "present", the rest with none.

Writes data/runs/review/failure_review_sample.csv + failure_review_sheet.md and
g_rationale_sample.csv + g_rationale_sheet.md. Judgements go in failure_review.csv and
g_rationale_review.csv.

Usage: python -I scripts/evaluation/week7_samples.py
"""
import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
RUNS = PROJECT / "data" / "runs"
PROC = PROJECT / "data" / "processed"
REPORTS = PROJECT / "data" / "raw" / "iu_xray" / "reports" / "ecgen-radiology"
OUT = RUNS / "review"
SEED = 2026
ARMS = {"P1-adapted-ns": "p1_adapted_ns__checklist", "P2-adapted": "p2_adapted__checklist",
        "P3-adapted": "p3_adapted__checklist", "G": "g__prompt2", "G−ind": "g_ind__prompt2", "G-img": "g_img__prompt2"}
K = 10


def sections(study: str) -> dict:
    root = ET.parse(REPORTS / f"{study[3:]}.xml").getroot()
    return {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}


def cache(system: str, study: str) -> dict:
    return json.loads((RUNS / ARMS[system] / "test" / f"{study}.json").read_text(encoding="utf-8"))


def answer_text(system: str, rec: dict) -> str:
    if system in ("G", "G−ind", "G-img"):
        p = rec.get("parsed") or {}
        lines = [f"- **{f['finding']}**: {f['status']} (p = {f['probability']:.2f}); cites {f['cited_case_ids'] or '—'}; "
                 f"{f['rationale']}" for f in p.get("findings", [])]
        return "\n".join(lines + [f"- **Impression:** {p.get('impression', '')}"])
    return rec.get("response_text", "")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    with open(PROC / "ground_truth.csv") as fh:
        gt = {r["study"]: {f: r[f] == "1" for f in FINDINGS} for r in csv.DictReader(fh)}
    with open(RUNS / "eval" / "test" / "per_study.csv", encoding="utf-8") as fh:
        ps = list(csv.DictReader(fh))
    out = {(r["study"], r["finding"], r["system"], r["reading"]): r for r in ps}
    thr10 = json.loads((RUNS / "eval" / "test" / "results.json").read_text())["copying"]["thresholds_k10"]

    used = set()

    def draw(cands, n, label):
        cands = sorted({c for c in cands if c[0] not in used}, key=lambda c: (int(c[0][3:]), c[1], c[2]))
        pick = rng.sample(cands, min(n, len(cands)))
        used.update(c[0] for c in pick)
        return [(label, *c) for c in pick]

    def rows(system, reading, oc):
        return [(r["study"], r["finding"], system, reading) for r in ps
                if r["system"] == system and r["reading"] == reading and r["outcome"] == oc]

    def wrong(s, f, system, reading):
        return out[(s, f, system, reading)]["outcome"] in ("miss", "false alarm")

    sample = []
    sample += draw(rows("G", "stated", "miss"), 4, "G miss")
    sample += draw(rows("G", "stated", "false alarm"), 4, "G false alarm")
    pairs = [(r["study"], r["finding"]) for r in ps if r["system"] == "G" and r["reading"] == "threshold"]
    sample += draw([(s, f, "G", "threshold") for s, f in pairs
                    if wrong(s, f, "G", "threshold") and not wrong(s, f, "S0", "threshold")], 2, "G wrong, S0 right")
    sample += draw([(s, f, "S0", "threshold") for s, f in pairs
                    if wrong(s, f, "S0", "threshold") and not wrong(s, f, "G", "threshold")], 2, "S0 wrong, G right")
    copied = []
    for s in {p[0] for p in pairs}:
        nb = [n["study"] for n in cache("G", s)["neighbours"]][:K]
        for f in FINDINGS:
            knn = sum(gt[n][f] for n in nb) / K >= thr10[f]
            g = out[(s, f, "G", "stated")]["verdict"] == "1"
            if g == knn and knn != gt[s][f] and not wrong(s, f, "G-img", "stated"):
                copied.append((s, f, "G", "stated"))
    sample += draw(copied, 3, "G copied wrong neighbours, G-img right")
    for p in ("P1-adapted-ns", "P2-adapted", "P3-adapted"):
        sample += draw(rows(p, "stated", "miss"), 1, f"{p} miss")
        sample += draw(rows(p, "stated", "false alarm"), 1, f"{p} false alarm")
    for s in ("CXR206", "CXR3080"):
        first = next(x["finding"] for x in cache("G-img", s)["parsed"]["findings"] if x["cited_case_ids"])
        sample.append(("G-img invented citation", s, first, "G-img", "stated"))
        used.add(s)
    assert len(sample) == 23, len(sample)

    with open(OUT / "failure_review_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "stratum", "study", "finding", "system", "reading", "truth", "verdict", "outcome", "score"])
        for i, (lab, s, f, system, reading) in enumerate(sample, 1):
            r = out[(s, f, system, reading)]
            w.writerow([i, lab, s, f, system, reading, r["truth"], r["verdict"], r["outcome"], r["score"]])
    md = ["# Failure review sheet (W1) — test\n",
          "Categories (week7_design.md W1): missed finding / invented finding / definition mismatch / copied "
          "neighbour / text-led / hedged / label noise / other. Judged from text only.\n"]
    for i, (lab, s, f, system, reading) in enumerate(sample, 1):
        r, sec = out[(s, f, system, reading)], sections(s)
        md += [f"## Case {i}: {lab} — {s}, {f}, {system} ({reading})\n",
               f"Truth {r['truth']}, verdict {r['verdict']} ({r['outcome']}), score {r['score'] or '–'}. "
               f"Labels of the study: {', '.join(x for x in FINDINGS if gt[s][x]) or 'none'}.\n",
               f"**Indication:** {sec.get('INDICATION') or '—'}\n",
               f"**Report findings:** {sec.get('FINDINGS') or '—'}\n",
               f"**Report impression:** {sec.get('IMPRESSION') or '—'}\n"]
        if system != "S0":
            md += [f"**{system} answer:**\n", answer_text(system, cache(system, s)), ""]
        if system in ("G", "S0"):
            g = cache("G", s)
            if system == "S0":
                md += ["**G's answer:**\n", answer_text("G", g), ""]
            md.append("**G's 10 neighbours** (labels; report):\n")
            for n in g["neighbours"]:
                ns = sections(n["study"])
                md.append(f"- {n['study']} ({n['similarity']:.3f}) [{', '.join(x for x in FINDINGS if gt[n['study']][x]) or 'none'}]: "
                          f"{ns.get('FINDINGS', '')} {ns.get('IMPRESSION', '')}")
            md.append("")
        if system == "S0":
            md.append(f"S0 score for {f}: {r['score']}.\n")
    (OUT / "failure_review_sheet.md").write_text("\n".join(md), encoding="utf-8")

    # W3
    rng3 = random.Random(SEED)
    with open(RUNS / ARMS["G"] / "test_checklist.csv", encoding="utf-8") as fh:
        rows3 = sorted(csv.DictReader(fh), key=lambda r: int(r["study"][3:]))
    pos = [r["study"] for r in rows3 if any(r[f] == "1" for f in FINDINGS)]
    neg = [r["study"] for r in rows3 if not any(r[f] == "1" for f in FINDINGS)]
    pick = sorted(rng3.sample(pos, 8) + rng3.sample(neg, 7), key=lambda s: int(s[3:]))
    with open(OUT / "g_rationale_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "finding", "status", "probability", "cited", "rationale"])
        for s in pick:
            for x in cache("G", s)["parsed"]["findings"]:
                w.writerow([s, x["finding"], x["status"], x["probability"], ";".join(x["cited_case_ids"]), x["rationale"]])
    md = ["# G rationale hand check (W3) — test\n",
          "Per finding: does the status agree with its own rationale (agree / disagree / unclear)? Per \"present\" "
          "call: basis of the rationale (image / cited cases / indication / mixed).\n"]
    for s in pick:
        g = cache("G", s)
        md += [f"## {s}\n", f"**Indication given to G:** {g['indication'] or '—'}\n", answer_text("G", g), "",
               "**Cited cases** (labels; report):\n"]
        cited = sorted({c for x in g["parsed"]["findings"] for c in x["cited_case_ids"]}, key=lambda c: int(c[3:]))
        for c in cited:
            cs = sections(c)
            md.append(f"- {c} [{', '.join(x for x in FINDINGS if gt[c][x]) or 'none'}]: {cs.get('FINDINGS', '')} "
                      f"{cs.get('IMPRESSION', '')}")
        md.append("")
    (OUT / "g_rationale_sheet.md").write_text("\n".join(md), encoding="utf-8")
    print(f"W1: {len(sample)} cases; W3: {len(pick)} answers ({len(pick) * 6} findings) -> {OUT}")


if __name__ == "__main__":
    main()
