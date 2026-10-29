"""Evaluation metrics shared by all systems (plan §5): AUROC, threshold metrics, the Youden
threshold used for every tuned system, and a paired bootstrap over studies.

scikit-learn is not installed; these are small and tested against brute force
(scripts/tests/test_metrics.py).
"""
import numpy as np
from scipy.stats import rankdata


def auroc(y, s) -> float:
    """Area under the ROC curve = P(score of a random positive > random negative), ties 1/2.
    nan if only one class is present."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)  # average ranks for ties
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_metrics(y, s, t) -> dict:
    """Predicted positive = score >= t."""
    y, p = np.asarray(y, bool), np.asarray(s, float) >= t
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec, "precision": ppv, "f1": f1}


def youden_threshold(y, s) -> float:
    """The score t maximising sensitivity + specificity - 1 for 'positive = score >= t'.
    Candidates are the observed scores; ties in J go to the higher t (higher specificity)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("Youden threshold needs both classes")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp, fp = np.cumsum(y_sorted), np.cumsum(~y_sorted)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]  # last row of each tied score block
    t, j = s_sorted[last], tp[last] / n1 - fp[last] / n0
    return float(t[np.argmax(j)])  # argmax takes the first = highest t among equal J


def bootstrap(stat, n: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """stat(idx) for n_boot resamples of range(n) with replacement (studies resampled, so
    paired when stat compares systems on the same idx). nan results are kept."""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])


def ci(values, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    a = (1 - level) / 2
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))
