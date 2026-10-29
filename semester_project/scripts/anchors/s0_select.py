"""S0: choose weights + preprocessing on train, tune per-finding thresholds on train, report
train and validation results.

Rules fixed before scoring (diary 2026-10-17):
- Candidates: the three DenseNets (chex, mimic_nb, mimic_ch) x {crop, pad}. Selected = highest
  macro AUROC over the six findings on train (S0 never saw IU data; results §2.3b).
- resnet50-res512-all is reported only (its Lung Lesion output is untrained; scored here as
  max(Nodule, Mass)).
- Thresholds: per finding, Youden's J on train (metrics.youden_threshold).
Paired bootstrap (2,000 resamples of train studies) of each config's macro AUROC minus the
winner's, to show whether the choice is meaningful; the rule's choice stands either way.

Writes data/runs/s0/selection.json.

Usage: python -I scripts/anchors/s0_select.py
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
PROC = PROJECT / "data" / "processed"
RUNS = PROJECT / "data" / "runs" / "s0"
LABEL = {"Lung Opacity": "Lung Opacity", "Cardiomegaly": "Cardiomegaly", "Atelectasis": "Atelectasis",
         "Lung Lesion": "Lung Lesion", "Pleural Effusion": "Effusion", "Fracture": "Fracture"}
CANDIDATES = [(w, p) for w in ("densenet121-res224-chex", "densenet121-res224-mimic_nb",
                               "densenet121-res224-mimic_ch") for p in ("crop", "pad")]
REFERENCE = [("resnet50-res512-all", p) for p in ("crop", "pad")]
N_BOOT, SEED = 2000, 2026


def load_truth(split: str):
    with open(PROC / "splits.csv") as fh:
        studies = [r["study"] for r in csv.DictReader(fh) if r["split"] == split]
    with open(PROC / "ground_truth.csv") as fh:
        gt = {r["study"]: r for r in csv.DictReader(fh)}
    studies.sort(key=lambda s: int(s[3:]))
    return studies, np.array([[gt[s][f] == "1" for f in FINDINGS] for s in studies])


def load_scores(weights: str, prep: str, split: str, studies) -> np.ndarray:
    with open(RUNS / f"{weights}__{prep}__{split}.csv") as fh:
        rows = {r["study"]: r for r in csv.DictReader(fh)}
    assert set(rows) == set(studies), f"{weights} {prep} {split}: studies differ from the split"
    out = []
    for s in studies:
        r = rows[s]
        if weights.startswith("resnet"):  # untrained Lung Lesion output -> proxy
            r = dict(r, **{"Lung Lesion": max(float(r["Nodule"]), float(r["Mass"]))})
        out.append([float(r[LABEL[f]]) for f in FINDINGS])
    return np.array(out)


def per_finding_auroc(y, s, idx=slice(None)):
    return np.array([metrics.auroc(y[idx, j], s[idx, j]) for j in range(len(FINDINGS))])


def fmt(x, d=3):
    return "–" if x is None or np.isnan(x) else f"{x:.{d}f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252; -I ignores PYTHONIOENCODING
    studies, y = load_truth("train")
    configs = CANDIDATES + REFERENCE
    scores = {c: load_scores(*c, "train", studies) for c in configs}
    auc = {c: per_finding_auroc(y, scores[c]) for c in configs}
    winner = max(CANDIDATES, key=lambda c: auc[c].mean())

    print(f"## Train ({len(studies)} studies) — AUROC per config\n")
    print("| Weights | Prep | " + " | ".join(FINDINGS) + " | Macro | Δ macro vs. selected [95% CI] |")
    print("|" + "---|" * (len(FINDINGS) + 4))
    boot = {}
    for c in configs:
        if c == winner:
            d = "selected"
        else:
            diffs = metrics.bootstrap(lambda i: per_finding_auroc(y, scores[c], i).mean()
                                      - per_finding_auroc(y, scores[winner], i).mean(),
                                      len(studies), N_BOOT, SEED)
            lo, hi = metrics.ci(diffs)
            boot[f"{c[0]}__{c[1]}"] = [lo, hi]
            d = f"{auc[c].mean() - auc[winner].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        tag = " (reference)" if c in REFERENCE else ""
        print(f"| {c[0].replace('densenet121-res224-', 'DenseNet ')}{tag} | {c[1]} | "
              + " | ".join(fmt(a) for a in auc[c]) + f" | **{fmt(auc[c].mean())}** | {d} |")

    s = scores[winner]
    thr = {f: metrics.youden_threshold(y[:, j], s[:, j]) for j, f in enumerate(FINDINGS)}
    selection = {
        "weights": winner[0], "prep": winner[1], "label_map": LABEL,
        "thresholds": thr, "threshold_rule": "Youden J on train, ties -> higher threshold",
        "train_auroc": dict(zip(FINDINGS, auc[winner].tolist())),
        "train_macro_auroc": float(auc[winner].mean()),
        "all_configs_train_macro_auroc": {f"{w}__{p}": float(auc[(w, p)].mean()) for w, p in configs},
        "bootstrap_delta_macro_vs_selected_ci95": boot,
        "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "study"},
    }

    for split in ("train", "val"):
        st, yy = (studies, y) if split == "train" else load_truth("val")
        ss = s if split == "train" else load_scores(*winner, "val", st)
        a = per_finding_auroc(yy, ss)
        b = metrics.bootstrap(lambda i: per_finding_auroc(yy, ss, i), len(st), N_BOOT, SEED)
        print(f"\n## Selected S0 = {winner[0]} / {winner[1]} — {split} ({len(st)} studies)"
              + (" — thresholds tuned here (in-sample)" if split == "train" else "") + "\n")
        print("| Finding | Positives | AUROC [95% CI] | Threshold | Sensitivity | Specificity | F1 |")
        print("|---|---|---|---|---|---|---|")
        res = {}
        for j, f in enumerate(FINDINGS):
            m = metrics.threshold_metrics(yy[:, j], ss[:, j], thr[f])
            lo, hi = metrics.ci(b[:, j]) if yy[:, j].sum() else (np.nan, np.nan)
            print(f"| {f} | {int(yy[:, j].sum())} | {fmt(a[j])} [{fmt(lo)}, {fmt(hi)}] | {thr[f]:.4f} | "
                  f"{fmt(m['sensitivity'], 2)} | {fmt(m['specificity'], 2)} | {fmt(m['f1'], 2)} |")
            res[f] = dict(m, auroc=a[j], auroc_ci95=[lo, hi], positives=int(yy[:, j].sum()))
        macro_ci = metrics.ci(np.nanmean(b, axis=1))
        print(f"| **Macro** | | **{fmt(np.nanmean(a))}** [{fmt(macro_ci[0])}, {fmt(macro_ci[1])}] | | "
              f"{fmt(np.nanmean([res[f]['sensitivity'] for f in FINDINGS]), 2)} | "
              f"{fmt(np.nanmean([res[f]['specificity'] for f in FINDINGS]), 2)} | "
              f"{fmt(np.nanmean([res[f]['f1'] for f in FINDINGS]), 2)} |")
        selection[f"{split}_results"] = res

    with open(RUNS / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(f"\nwrote {RUNS / 'selection.json'}")


if __name__ == "__main__":
    main()
