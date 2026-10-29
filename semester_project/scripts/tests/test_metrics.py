"""Offline test of scripts/metrics.py against brute-force definitions.

Usage: python -I scripts/tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics  # noqa: E402

ok = True


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def brute_auroc(y, s):
    pos, neg = s[y], s[~y]
    return np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])


def brute_youden(y, s):
    best_t, best_j = None, -2
    for t in sorted(set(s), reverse=True):  # high to low: first max = highest t
        p = s >= t
        j = (p & y).sum() / y.sum() - (p & ~y).sum() / (~y).sum()
        if j > best_j + 1e-12:
            best_t, best_j = t, j
    return best_t


rng = np.random.default_rng(1)
for trial in range(200):
    n = int(rng.integers(5, 60))
    y = rng.random(n) < rng.uniform(0.1, 0.6)
    if y.all() or not y.any():
        continue
    s = rng.integers(0, 6, n) / 5.0 if trial % 2 else rng.random(n)  # every 2nd trial has ties
    ok &= abs(metrics.auroc(y, s) - brute_auroc(y, s)) < 1e-12
    ok &= metrics.youden_threshold(y, s) == brute_youden(y, s)
ok = check("auroc and youden_threshold match brute force on 200 random cases (half with ties)", ok)

y = np.array([1, 1, 0, 0, 1, 0], bool)
s = np.array([.9, .4, .4, .1, .6, .7])
m = metrics.threshold_metrics(y, s, 0.5)
ok &= check("threshold_metrics counts", (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2))
ok &= check("threshold_metrics rates", abs(m["sensitivity"] - 2 / 3) < 1e-12
            and abs(m["specificity"] - 2 / 3) < 1e-12 and abs(m["f1"] - 2 / 3) < 1e-12)
ok &= check("score == threshold counts as positive", metrics.threshold_metrics(y, s, 0.4)["tp"] == 3)
ok &= check("auroc perfect / reversed / one class",
            metrics.auroc([0, 0, 1], [0, 1, 2]) == 1.0 and metrics.auroc([1, 0], [0, 1]) == 0.0
            and np.isnan(metrics.auroc([1, 1], [0, 1])))
ok &= check("youden tie-break takes the higher threshold",
            metrics.youden_threshold([0, 1, 0, 1], [0.1, 0.5, 0.6, 0.9]) == 0.9)

b = metrics.bootstrap(lambda idx: idx.mean(), 100, n_boot=50, seed=3)
ok &= check("bootstrap is reproducible with a seed",
            np.array_equal(b, metrics.bootstrap(lambda idx: idx.mean(), 100, n_boot=50, seed=3)))
lo, hi = metrics.ci([np.nan] + list(range(101)))
ok &= check("ci ignores nan and gives the 2.5/97.5 percentiles",
            abs(lo - 2.5) < 1e-9 and abs(hi - 97.5) < 1e-9)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
