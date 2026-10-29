"""Draw the checklist hand-check sample and write review sheets (protocol: diary 2026-10-18).

Per arm: 15 pilot answers, seed 2026; up to 8 with at least one "present" in the checklist,
the rest from answers with none. Writes data/runs/review/checklist_review_<arm>.md (each
sampled answer in full, with its parsed checklist) and checklist_review_sample.csv (the sample).
The judgements themselves are entered in data/runs/review/checklist_review.csv.

Usage: python -I scripts/evaluation/checklist_review.py
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
RUNS = PROJECT / "data" / "runs"
OUT = RUNS / "review"
ARMS = ["p1_adapted_ns__checklist", "p2_adapted__checklist", "p3_adapted__checklist"]
N, N_POS, SEED = 15, 8, 2026


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    sample_rows = []
    for arm in ARMS:
        with open(RUNS / arm / "val_checklist.csv", encoding="utf-8") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: int(r["study"][3:]))
        pos = [r for r in rows if any(r[f] == "1" for f in FINDINGS)]
        neg = [r for r in rows if not any(r[f] == "1" for f in FINDINGS)]
        pick = rng.sample(pos, min(N_POS, len(pos)))
        pick += rng.sample(neg, N - len(pick))
        pick.sort(key=lambda r: int(r["study"][3:]))
        lines = [f"# Checklist review — {arm}\n",
                 "Judge each finding: agree / disagree / unclear (protocol: diary 2026-10-18).\n"]
        for r in pick:
            rec = json.loads((RUNS / arm / "val" / f"{r['study']}.json").read_text(encoding="utf-8"))
            lines.append(f"\n## {r['study']}\n")
            lines.append("Parsed checklist: " + "; ".join(f"{f}: {r['checklist_' + f]}" for f in FINDINGS) + "\n")
            lines.append("```text\n" + (rec["response_text"] or "").strip() + "\n```\n")
            sample_rows.append([arm, r["study"]] + [r["checklist_" + f] for f in FINDINGS])
        (OUT / f"checklist_review_{arm}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"{arm}: {len(pick)} answers ({min(N_POS, len(pos))} with a 'present'; {len(pos)} available)")
    with open(OUT / "checklist_review_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "study"] + [f"checklist_{f}" for f in FINDINGS])
        w.writerows(sample_rows)


if __name__ == "__main__":
    main()
