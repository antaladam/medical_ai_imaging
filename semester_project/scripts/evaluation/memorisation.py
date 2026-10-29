"""W2: after-the-fact memorisation check (plan §8; week7_design.md, decided 2026-10-18).

Question: does a system's text match the test study's OWN report more than other reports with
the same labels? If so, it may recall the IU report instead of reading the case.

Rules (fixed before any overlap was computed):
- Systems: the six Gemini systems (P1-adapted-ns, P2-adapted, P3-adapted, G, G-ind, G-img).
- System text: G family = impression + all rationales; starters = the answer without its
  checklist lines (parse_checklist.LINE, the same definition as the scoring).
- Reference = the report's FINDINGS + IMPRESSION, without INDICATION (G received the indication).
- Normalisation: lower case; "XXXX" de-identification tokens and punctuation removed.
- Primary measure: containment of word 3-grams = share of the reference's distinct 3-grams that
  occur in the system text. Secondary: TF-IDF cosine of word unigrams (IDF over all reports).
- Baseline per test study: the same measure against 20 other reports with the identical six-label
  vector (all splits; the study itself excluded; seed 2026 per study, so every system gets the
  same pool). Also excluded from the pool: every report any system was shown for that study
  (G's 10 neighbours, P3's 6 retrieved cases, P2's retrieved case), so that copying from a shown
  case cannot raise the baseline. Fewer than 20 candidates -> all of them; studies with fewer
  than 5 are left out and counted.
- Statistics per system: mean (own - baseline), paired bootstrap 95% CI over studies (2,000,
  seed 2026); share of studies whose own overlap exceeds the maximum of its pool (chance level
  about 1 / (pool size + 1), ~5% with 20).

Writes data/runs/eval/test/memorisation.md and memorisation.csv (per study and system).

Usage: python -I scripts/evaluation/memorisation.py
"""
import csv
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
sys.path.insert(0, str(Path(__file__).resolve().parent))  # evaluation/: parse_checklist
import metrics  # noqa: E402
from common import FINDINGS  # noqa: E402
from parse_checklist import LINE  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
RUNS = PROJECT / "data" / "runs"
PROC = PROJECT / "data" / "processed"
REPORTS = PROJECT / "data" / "raw" / "iu_xray" / "reports" / "ecgen-radiology"
OUT = RUNS / "eval" / "test"
SEED, POOL, MIN_POOL, N_BOOT = 2026, 20, 5, 2000
SYSTEMS = {"P1-adapted-ns": "p1_adapted_ns__checklist", "P2-adapted": "p2_adapted__checklist",
           "P3-adapted": "p3_adapted__checklist", "G": "g__prompt2", "G−ind": "g_ind__prompt2",
           "G-img": "g_img__prompt2"}
G_FAMILY = {"G", "G−ind", "G-img"}


# --- text functions (tested in scripts/tests/test_memorisation.py) --------------------------------
def tokens(text: str) -> list[str]:
    t = re.sub(r"\bx{2,}\b", " ", (text or "").lower())
    return re.findall(r"[a-z0-9]+", t)


def ngrams(toks: list[str], n: int = 3) -> set:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def containment(reference: str, text: str, n: int = 3) -> float:
    """Share of the reference's distinct n-grams that occur in the text; nan if it has none."""
    ref = ngrams(tokens(reference), n)
    return len(ref & ngrams(tokens(text), n)) / len(ref) if ref else float("nan")


def tfidf_cosine(a: str, b: str, idf: dict) -> float:
    va, vb = Counter(tokens(a)), Counter(tokens(b))
    wa = {w: c * idf.get(w, 0.0) for w, c in va.items()}
    wb = {w: c * idf.get(w, 0.0) for w, c in vb.items()}
    dot = sum(wa[w] * wb.get(w, 0.0) for w in wa)
    na, nb = math.sqrt(sum(v * v for v in wa.values())), math.sqrt(sum(v * v for v in wb.values()))
    return dot / (na * nb) if na and nb else float("nan")


def strip_checklist(answer: str) -> str:
    """The answer without its checklist lines (same line definition as the scoring)."""
    return "\n".join(ln for ln in (answer or "").splitlines()
                     if not LINE.match(re.sub(r"[#*_`|]", "", ln).strip()))


def baseline_pool(study: str, label_key, same_label: dict, exclude: set) -> list[str]:
    """Up to POOL other studies with the same label vector, seeded per study."""
    cand = sorted((s for s in same_label[label_key] if s != study and s not in exclude), key=lambda s: int(s[3:]))
    rng = random.Random(f"{SEED}-{study}")
    return cand if len(cand) <= POOL else rng.sample(cand, POOL)


# --- data --------------------------------------------------------------------------------------------
def reference_text(study: str) -> str:
    root = ET.parse(REPORTS / f"{study[3:]}.xml").getroot()
    assert root.find("uId").get("id") == study
    sec = {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}
    return " ".join(sec[k] for k in ("FINDINGS", "IMPRESSION") if sec.get(k))


def system_text(name: str, rec: dict) -> str:
    if name in G_FAMILY:
        p = rec.get("parsed") or {}
        return " ".join([p.get("impression", "")] + [f.get("rationale", "") for f in p.get("findings", [])])
    return strip_checklist(rec.get("response_text", ""))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(PROC / "splits.csv") as fh:
        split_of = {r["study"]: r["split"] for r in csv.DictReader(fh)}
    with open(PROC / "ground_truth.csv") as fh:
        key_of = {r["study"]: tuple(r[f] for f in FINDINGS) for r in csv.DictReader(fh)}
    ref = {s: reference_text(s) for s in key_of}
    ref = {s: t for s, t in ref.items() if ngrams(tokens(t))}  # reports with at least one 3-gram
    same_label = {}
    for s in ref:
        same_label.setdefault(key_of[s], []).append(s)
    df = Counter(w for t in ref.values() for w in set(tokens(t)))
    idf = {w: math.log(len(ref) / c) for w, c in df.items()}

    test = sorted((s for s, sp in split_of.items() if sp == "test"), key=lambda s: int(s[3:]))
    recs = {name: {s: json.loads((RUNS / arm / "test" / f"{s}.json").read_text(encoding="utf-8")) for s in test}
            for name, arm in SYSTEMS.items()}
    shown = {s: {n["study"] for n in recs["G"][s]["neighbours"]}
             | {x["study"] for x in recs["P3-adapted"][s].get("retrieved") or []}
             | ({recs["P2-adapted"][s]["retrieved_study"]} if recs["P2-adapted"][s].get("retrieved_study") else set())
             for s in test}

    pools, skipped = {}, []
    for s in test:
        if s not in ref:
            skipped.append((s, "no reference text"))
            continue
        pool = baseline_pool(s, key_of[s], same_label, shown[s])
        if len(pool) < MIN_POOL:
            skipped.append((s, f"pool {len(pool)}"))
            continue
        pools[s] = pool
    studies = sorted(pools, key=lambda s: int(s[3:]))

    rows, summary = [], {}
    for name in SYSTEMS:
        own3, base3, own1, base1, beats = [], [], [], [], []
        for s in studies:
            txt = system_text(name, recs[name][s])
            o3 = containment(ref[s], txt)
            b3 = [containment(ref[p], txt) for p in pools[s]]
            o1 = tfidf_cosine(ref[s], txt, idf)
            b1 = [tfidf_cosine(ref[p], txt, idf) for p in pools[s]]
            own3.append(o3), base3.append(float(np.nanmean(b3))), beats.append(o3 > np.nanmax(b3))
            own1.append(o1), base1.append(float(np.nanmean(b1)))
            rows.append([s, name, len(pools[s]), f"{o3:.4f}", f"{np.nanmean(b3):.4f}", f"{np.nanmax(b3):.4f}",
                         int(o3 > np.nanmax(b3)), f"{o1:.4f}", f"{np.nanmean(b1):.4f}"])
        d3, d1 = np.array(own3) - np.array(base3), np.array(own1) - np.array(base1)
        b = metrics.bootstrap(lambda i: np.array([np.nanmean(d3[i]), np.nanmean(d1[i])]), len(studies), N_BOOT, SEED)
        chance = float(np.mean([1 / (len(pools[s]) + 1) for s in studies]))
        summary[name] = {"own3": np.nanmean(own3), "base3": np.nanmean(base3), "d3": np.nanmean(d3),
                         "d3_ci": metrics.ci(b[:, 0]), "beats": float(np.mean(beats)), "chance": chance,
                         "own1": np.nanmean(own1), "base1": np.nanmean(base1), "d1": np.nanmean(d1),
                         "d1_ci": metrics.ci(b[:, 1])}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "memorisation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "system", "pool", "own_3gram", "pool_mean_3gram", "pool_max_3gram", "own_beats_max",
                    "own_tfidf", "pool_mean_tfidf"])
        w.writerows(rows)
    md = ["# Memorisation check (W2) — test\n",
          "Method: week7_design.md W2; `scripts/evaluation/memorisation.py`. Own = the study's own report "
          "(FINDINGS + IMPRESSION); pool = up to 20 other reports with the identical label vector, excluding "
          "every report shown to any system for that study.\n",
          f"Studies: {len(studies)} of {len(test)} (left out: "
          + (", ".join(f"{s} ({r})" for s, r in skipped) or "none") + ").\n",
          "| System | 3-gram containment own / pool mean | Own − pool [95% CI] | Own above the pool's maximum "
          "(chance) | TF-IDF cosine own / pool mean | Own − pool [95% CI] |",
          "|---|---|---|---|---|---|"]
    for name, m in summary.items():
        md.append(f"| {name} | {m['own3']:.3f} / {m['base3']:.3f} | {m['d3']:+.3f} [{m['d3_ci'][0]:+.3f}, "
                  f"{m['d3_ci'][1]:+.3f}] | {m['beats']:.1%} ({m['chance']:.1%}) | {m['own1']:.3f} / {m['base1']:.3f} | "
                  f"{m['d1']:+.3f} [{m['d1_ci'][0]:+.3f}, {m['d1_ci'][1]:+.3f}] |")
    (OUT / "memorisation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
