"""G family on validation: the pooled decision threshold, and a validation report per arm
(results §2.5, §4.4; g_design.md).

Rules (fixed 2026-10-18, before the full validation runs were seen):
- Threshold = ONE per arm, pooled across the six findings: Youden's J over all (study, finding)
  pairs of the stated probabilities (metrics.youden_threshold; ties -> higher threshold), the
  rule shared by all tuned systems (results §5). Predicted positive = probability >= threshold.
- Input = g_export.py's table (data/runs/<arm>/val_checklist.csv); a missing probability is 0.0
  there and is counted. Ground truth = data/processed/ground_truth.csv.
- Reported per finding: AUROC of the probability (bootstrap 95% CI over studies, 2,000
  resamples, seed 2026), sensitivity / specificity / F1 at the pooled threshold, and the same for
  the stated status (binary = "present", the starters' rule). Also: schema compliance, how often
  the two binary readings disagree, and token use (thinking) from the cached answers.
Validation only: the test split is not read here.

Writes data/runs/<arm>/selection.json (threshold + validation results); the report goes to stdout.

Usage: python -I scripts/g/g_select.py <arm> [<arm> ...]     (arm e.g. g__prompt2)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
import metrics  # noqa: E402
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
RUNS = PROJECT / "data" / "runs"
N_BOOT, SEED = 2000, 2026
SPLIT = "val"


def fmt(x, d=3):
    return "–" if np.isnan(x) else f"{x:.{d}f}"


def per_finding_auroc(y, s, idx=slice(None)):
    return np.array([metrics.auroc(y[idx, j], s[idx, j]) for j in range(len(FINDINGS))])


def main(arms: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(PROJECT / "data" / "processed" / "ground_truth.csv") as fh:
        gt = {r["study"]: [r[f] == "1" for f in FINDINGS] for r in csv.DictReader(fh)}
    with open(PROJECT / "data" / "processed" / "splits.csv") as fh:
        expected = {r["study"] for r in csv.DictReader(fh) if r["split"] == SPLIT}

    for arm in arms:
        with open(RUNS / arm / f"{SPLIT}_checklist.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        studies = [r["study"] for r in rows]
        assert set(studies) == expected, f"{arm}: {len(studies)} answers, expected all {len(expected)} {SPLIT} studies"
        y = np.array([gt[s] for s in studies])
        p = np.array([[float(r[f"prob_{f}"]) for f in FINDINGS] for r in rows])
        stated = np.array([[r[f] == "1" for f in FINDINGS] for r in rows])
        missing = np.array([[r[f"checklist_{f}"] == "missing" for f in FINDINGS] for r in rows])

        thr = metrics.youden_threshold(y.ravel(), p.ravel())
        pooled = metrics.threshold_metrics(y.ravel(), p.ravel(), thr)
        a = per_finding_auroc(y, p)
        b = metrics.bootstrap(lambda i: per_finding_auroc(y, p, i), len(studies), N_BOOT, SEED)
        disagree = int(((p >= thr) != stated).sum())

        think = []
        for s in studies:
            rec = json.loads((RUNS / arm / SPLIT / f"{s}.json").read_text(encoding="utf-8"))
            think.append(rec["usage"]["thinking"] or 0)
        think = np.array(think)

        print(f"## {arm} — {SPLIT} ({len(studies)} studies)\n")
        print(f"- Answers: {sum(r['outcome'] == 'answer' for r in rows)} valid JSON, "
              f"{sum(r['outcome'] != 'answer' for r in rows)} invalid; "
              f"{sum(r['block_at_start'] == '0' for r in rows)} with validation problems; "
              f"{int(missing.sum())} missing statuses")
        print(f"- Thinking tokens per answer: median {int(np.median(think))}, mean {int(think.mean())}, "
              f"max {int(think.max())}; > 20,000: {int((think > 20000).sum())}")
        print(f"- **Pooled threshold {thr:.4f}** (Youden J over {y.size} pairs, {int(y.sum())} positive): "
              f"sensitivity {fmt(pooled['sensitivity'], 2)}, specificity {fmt(pooled['specificity'], 2)}")
        print(f"- Threshold reading vs. stated status disagree on {disagree} of {y.size} pairs\n")
        print("| Finding | Positives | AUROC [95% CI] | Sens. @thr | Spec. @thr | F1 @thr "
              "| Sens. stated | Spec. stated | F1 stated |")
        print("|---|---|---|---|---|---|---|---|---|")
        res = {}
        for j, f in enumerate(FINDINGS):
            m = metrics.threshold_metrics(y[:, j], p[:, j], thr)
            ms = metrics.threshold_metrics(y[:, j], stated[:, j].astype(float), 1.0)
            lo, hi = metrics.ci(b[:, j])
            print(f"| {f} | {int(y[:, j].sum())} | {fmt(a[j])} [{fmt(lo)}, {fmt(hi)}] | "
                  f"{fmt(m['sensitivity'], 2)} | {fmt(m['specificity'], 2)} | {fmt(m['f1'], 2)} | "
                  f"{fmt(ms['sensitivity'], 2)} | {fmt(ms['specificity'], 2)} | {fmt(ms['f1'], 2)} |")
            res[f] = {"positives": int(y[:, j].sum()), "auroc": a[j], "auroc_ci95": [lo, hi],
                      "at_threshold": m, "stated": ms}
        mlo, mhi = metrics.ci(np.nanmean(b, axis=1))
        macro = {k: {m: float(np.nanmean([res[f][k][m] for f in FINDINGS])) for m in ("sensitivity", "specificity", "f1")}
                 for k in ("at_threshold", "stated")}
        print(f"| **Macro** | | **{fmt(np.nanmean(a))}** [{fmt(mlo)}, {fmt(mhi)}] | "
              + " | ".join(fmt(macro[k][m], 2) for k in ("at_threshold", "stated")
                           for m in ("sensitivity", "specificity", "f1")) + " |\n")

        out = {"arm": arm, "threshold": thr,
               "threshold_rule": "one per arm: Youden J over all (study, finding) pairs of the stated probabilities "
                                 "on validation, ties -> higher threshold; positive = probability >= threshold",
               "split": SPLIT, "n_studies": len(studies), "pooled_at_threshold": pooled,
               "macro_auroc": float(np.nanmean(a)), "macro_auroc_ci95": [mlo, mhi], "macro": macro,
               "threshold_vs_stated_disagreements": disagree, "missing_statuses": int(missing.sum()),
               "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "study"}, f"{SPLIT}_results": res}
        with open(RUNS / arm / "selection.json", "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"wrote {RUNS / arm / 'selection.json'}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
