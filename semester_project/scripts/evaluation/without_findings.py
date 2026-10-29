"""Sensitivity check: G − S0 and G − G-img macro AUROC without Lung Lesion (and Fracture).

Promised before the test run (results §6, 2026-10-17: "macro averages are also reported without
them if it changes a conclusion"); computed after it, 2026-10-18 (results §5.5, limit 5). Same
paired bootstrap as evaluate.py (2,000 study resamples, seed 2026). Reads per_study.csv only.

Writes data/runs/eval/<split>/without_findings.md.

Usage: python -I scripts/evaluation/without_findings.py test
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
import metrics  # noqa: E402
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
N_BOOT, SEED = 2000, 2026
SUBSETS = [("all six", FINDINGS),
           ("without Lung Lesion", [f for f in FINDINGS if f != "Lung Lesion"]),
           ("without Lung Lesion and Fracture", [f for f in FINDINGS if f not in ("Lung Lesion", "Fracture")])]
PAIRS = [("G", "threshold", "S0", "threshold"), ("G", "threshold", "G-img", "stated")]  # readings only locate the scores


def main(split: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = PROJECT / "data" / "runs" / "eval" / split
    with open(d / "per_study.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def mat(system, reading, field):
        m = {}
        for r in rows:
            if r["system"] == system and r["reading"] == reading:
                m.setdefault(r["study"], {})[r["finding"]] = float(r[field])
        st = sorted(m, key=lambda s: int(s[3:]))
        return st, np.array([[m[s][f] for f in FINDINGS] for s in st])

    st, y = mat("G", "threshold", "truth")
    md = [f"# Macro AUROC differences without hard findings — {split}\n",
          "`scripts/evaluation/without_findings.py`; paired bootstrap, 2,000 study resamples, seed 2026.\n",
          "| Difference | " + " | ".join(n for n, _ in SUBSETS) + " |", "|---" * (len(SUBSETS) + 1) + "|"]
    for a, ra, b, rb in PAIRS:
        sa, xa = mat(a, ra, "score")
        sb, xb = mat(b, rb, "score")
        assert sa == st == sb, "studies differ between systems"
        cells = []
        for _, sub in SUBSETS:
            js = [FINDINGS.index(f) for f in sub]

            def stat(i):
                return np.mean([metrics.auroc(y[i, j], xa[i, j]) - metrics.auroc(y[i, j], xb[i, j]) for j in js])
            lo, hi = metrics.ci(metrics.bootstrap(stat, len(st), N_BOOT, SEED))
            cells.append(f"{stat(np.arange(len(st))):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    (d / "without_findings.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
