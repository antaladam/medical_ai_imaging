"""Pilot overview: each system's binary findings on the 50 pilot studies vs. the ground truth.

The pilot has only 10 positive findings (natural prevalence), so this shows behaviour (how often
each system calls a finding, false alarms on normal studies, misses), not reliable accuracy.
Starters: their own findings checklist (checklist question, parse_checklist.py; the CheXbert labels were
invalid, diary 2026-10-17, and their summary is kept as pilot_summary.md). S0 and R0 (context): their
validation scores at the thresholds fixed on train (data/runs/s0|r0/selection.json).

Usage: python -I scripts/evaluation/pilot_summary.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
RUNS = PROJECT / "data" / "runs"
STARTERS = ["p1_adapted_ns__checklist", "p2_adapted__checklist", "p3_adapted__checklist", "g__prompt1"]  # G: g_export.py


def read(path):
    with open(path, encoding="utf-8") as fh:
        return {r["study"]: r for r in csv.DictReader(fh)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    splits = read(PROJECT / "data" / "processed" / "splits.csv")
    pilot = sorted((s for s, r in splits.items() if r["pilot"] == "1"), key=lambda s: int(s[3:]))
    gt = read(PROJECT / "data" / "processed" / "ground_truth.csv")
    truth = {s: {f for f in FINDINGS if gt[s][f] == "1"} for s in pilot}

    pred = {}
    for arm in STARTERS:
        lab = read(RUNS / arm / "val_checklist.csv")
        assert set(pilot) <= set(lab), f"{arm}: pilot studies missing"
        pred[arm] = {s: {f for f in FINDINGS if lab[s][f] == "1"} for s in pilot}
    sel = json.loads((RUNS / "s0" / "selection.json").read_text())
    s0 = read(RUNS / "s0" / f"{sel['weights']}__{sel['prep']}__val.csv")
    pred["S0"] = {s: {f for f in FINDINGS if float(s0[s][sel["label_map"][f]]) >= sel["thresholds"][f]} for s in pilot}
    sel = json.loads((RUNS / "r0" / "selection.json").read_text())
    r0 = read(RUNS / "r0" / "val.csv")
    pred["R0"] = {s: {f for f in FINDINGS if float(r0[s][f]) >= sel["thresholds"][f]} for s in pilot}

    normal = [s for s in pilot if not truth[s]]
    n_pos = sum(len(t) for t in truth.values())
    print(f"Pilot: {len(pilot)} studies, {len(normal)} with none of the six, {n_pos} positive findings: "
          + ", ".join(f"{f} {sum(f in truth[s] for s in pilot)}" for f in FINDINGS) + "\n")
    print("| System | Findings called | Correct (of " + str(n_pos) + ") | False alarms | Normal studies called normal (of "
          + str(len(normal)) + ") | Calls per finding (LO / CM / AT / LL / PE / FR) |")
    print("|---|---|---|---|---|---|")
    for arm, p in pred.items():
        called = sum(len(p[s]) for s in pilot)
        tp = sum(len(p[s] & truth[s]) for s in pilot)
        per = " / ".join(str(sum(f in p[s] for s in pilot)) for f in FINDINGS)
        print(f"| {arm} | {called} | {tp} | {called - tp} | {sum(not p[s] for s in normal)} | {per} |")

    print("\nPer positive finding (✓ = found):\n")
    print("| Study | Finding | " + " | ".join(pred) + " |")
    print("|---|---|" + "---|" * len(pred))
    for s in pilot:
        for f in sorted(truth[s]):
            print(f"| {s} | {f} | " + " | ".join("✓" if f in pred[a][s] else "·" for a in pred) + " |")


if __name__ == "__main__":
    main()
