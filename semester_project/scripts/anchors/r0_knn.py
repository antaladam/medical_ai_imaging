"""R0: retrieval alone, no LLM (plan §3). Score per finding = share of the k nearest training
studies (BiomedCLIP cosine similarity of the frontal images) that have the finding.

Rules fixed before running (diary 2026-10-17):
- Index = train split only (data/processed/biomedclip_frontal.npy, shared with G).
- k: one value for all findings, from K_GRID, chosen by leave-one-out on train (each train study
  scored from the other 2,803) = highest macro AUROC; ties -> smaller k. The grid first ended at
  100, which turned out to be the maximum with AUROC still rising, so it was extended upwards
  (train only; diary 2026-10-17).
- Thresholds: per finding, Youden's J on the leave-one-out train scores at that k.
- Unweighted vote: BiomedCLIP similarities sit in a narrow band (results §2.3).
Leakage guards: a train study never retrieves itself (asserted); queries from another split
are asserted absent from the index.

Writes data/runs/r0/selection.json and data/runs/r0/<split>.csv (train = leave-one-out):
study, the six scores, and the k neighbour study IDs (for retrieval quality and copying-rate
analyses later). The test split is refused unless --test is given (plan §6).

Usage: python -I scripts/anchors/r0_knn.py <split>[,<split>...] [--test]     (train is always run)
"""
import argparse
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
OUT = PROJECT / "data" / "runs" / "r0"
K_GRID = [1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]  # extended: see docstring
N_BOOT, SEED = 2000, 2026


def fmt(x, d=3):
    return "–" if np.isnan(x) else f"{x:.{d}f}"


def neighbours(q: np.ndarray, index: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    sim = q @ index.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def per_finding_auroc(y, s, idx=slice(None)):
    return np.array([metrics.auroc(y[idx, j], s[idx, j]) for j in range(len(FINDINGS))])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="?", default="")
    ap.add_argument("--test", action="store_true", help="allow the test split (week 6)")
    args = ap.parse_args()
    extra = [s for s in args.splits.split(",") if s and s != "train"]
    if "test" in extra and not args.test:
        sys.exit("refusing the test split without --test (plan §6: once, in week 6)")

    emb = np.load(PROC / "biomedclip_frontal.npy")
    with open(PROC / "biomedclip_frontal_ids.csv") as fh:
        row_of = {r["study"]: int(r["row"]) for r in csv.DictReader(fh)}
    with open(PROC / "splits.csv") as fh:
        split_of = {r["study"]: r["split"] for r in csv.DictReader(fh)}
    with open(PROC / "ground_truth.csv") as fh:
        gt = {r["study"]: [r[f] == "1" for f in FINDINGS] for r in csv.DictReader(fh)}

    def part(name):
        st = sorted((s for s, sp in split_of.items() if sp == name), key=lambda s: int(s[3:]))
        return st, emb[[row_of[s] for s in st]], np.array([gt[s] for s in st])

    train, e_tr, y_tr = part("train")
    assert np.allclose(np.linalg.norm(e_tr, axis=1), 1, atol=1e-5)

    # Leave-one-out on train
    nn_tr = neighbours(e_tr, e_tr, max(K_GRID), exclude_self=True)
    assert not (nn_tr == np.arange(len(train))[:, None]).any(), "leakage: a study retrieved itself"
    print(f"## R0 leave-one-out on train ({len(train)} studies) — AUROC by k\n")
    print("| k | " + " | ".join(FINDINGS) + " | Macro |")
    print("|" + "---|" * (len(FINDINGS) + 2))
    curve = {}
    for k in K_GRID:
        a = per_finding_auroc(y_tr, y_tr[nn_tr[:, :k]].mean(1))
        curve[k] = a
        print(f"| {k} | " + " | ".join(fmt(x) for x in a) + f" | **{fmt(a.mean())}** |")
    k = max(K_GRID, key=lambda k: (round(curve[k].mean(), 12), -k))  # ties -> smaller k

    s_tr = y_tr[nn_tr[:, :k]].mean(1)
    thr = {f: metrics.youden_threshold(y_tr[:, j], s_tr[:, j]) for j, f in enumerate(FINDINGS)}
    selection = {"k": k, "k_grid": K_GRID, "loo_macro_auroc_by_k": {kk: float(curve[kk].mean()) for kk in K_GRID},
                 "thresholds": thr, "threshold_rule": "Youden J on leave-one-out train scores, ties -> higher threshold",
                 "index": "train split, BiomedCLIP frontal embeddings (data/processed/biomedclip_frontal.npy)",
                 "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "study"}}
    OUT.mkdir(parents=True, exist_ok=True)

    results = {"train": (train, y_tr, s_tr, nn_tr[:, :k])}
    for name in extra:
        st, e, y = part(name)
        assert not set(st) & set(train), f"leakage: {name} study in the retrieval index"
        nn = neighbours(e, e_tr, k, exclude_self=False)
        results[name] = (st, y, y_tr[nn].mean(1), nn)

    for name, (st, y, s, nn) in results.items():
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["study"] + FINDINGS + ["neighbours"])
            for i, q in enumerate(st):
                w.writerow([q] + [f"{v:.4f}" for v in s[i]] + [";".join(train[j] for j in nn[i])])
        a = per_finding_auroc(y, s)
        b = metrics.bootstrap(lambda i: per_finding_auroc(y, s, i), len(st), N_BOOT, SEED)
        label = "leave-one-out, thresholds tuned here (in-sample)" if name == "train" else ""
        print(f"\n## R0 (k = {k}) — {name} ({len(st)} studies) {label}\n")
        print("| Finding | Positives | AUROC [95% CI] | Threshold | Sensitivity | Specificity | F1 |")
        print("|---|---|---|---|---|---|---|")
        res = {}
        for j, f in enumerate(FINDINGS):
            m = metrics.threshold_metrics(y[:, j], s[:, j], thr[f])
            lo, hi = metrics.ci(b[:, j])
            print(f"| {f} | {int(y[:, j].sum())} | {fmt(a[j])} [{fmt(lo)}, {fmt(hi)}] | {thr[f]:.3f} | "
                  f"{fmt(m['sensitivity'], 2)} | {fmt(m['specificity'], 2)} | {fmt(m['f1'], 2)} |")
            res[f] = dict(m, auroc=a[j], auroc_ci95=[lo, hi], positives=int(y[:, j].sum()))
        mlo, mhi = metrics.ci(np.nanmean(b, axis=1))
        print(f"| **Macro** | | **{fmt(np.nanmean(a))}** [{fmt(mlo)}, {fmt(mhi)}] | | "
              + " | ".join(fmt(np.nanmean([res[f][m] for f in FINDINGS]), 2)
                           for m in ("sensitivity", "specificity", "f1")) + " |")
        selection[f"{name}_results"] = res

    with open(OUT / "selection.json", "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(f"\nwrote {OUT / 'selection.json'}")


if __name__ == "__main__":
    main()
