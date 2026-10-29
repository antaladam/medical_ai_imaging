"""Q3 evaluation: every system on one split, scored as decided in results §5.4 (E1–E5,
eval_design.md, decided 2026-10-18). No API calls; reads the cached outputs only.

Systems and binary readings (results §5.4, "already fixed"):
- S0, R0: score >= their train-tuned per-finding thresholds ("threshold"); scores -> AUROC.
- P1-adapted-ns, P2-adapted, P3-adapted: their checklist ("stated": present -> 1; absent,
  uncertain, missing, conflict -> 0).
- G, G-ind: "stated" (same rule) and "threshold" (probability >= the pooled validation threshold,
  g_select.py); probabilities -> AUROC. G-img: "stated"; probabilities -> AUROC (no threshold).
Unanswered / unparsable outputs are rows of all 0 in the systems' tables; they are counted.

E1 comparisons (difference = first - second, paired bootstrap over studies, on the studies both
systems have): see COMPARISONS; #1 is the primary comparison.
E2 precision@k: per finding, over queries that have it, the share of the k retrieved training
cases that also have it; train prevalence as baseline (lift = precision / prevalence); normal
queries in a separate row. Each system's own k and k = 1.
E3 copying rate: the neighbours' decision = share of G's 10 neighbours with the finding >=
a per-finding Youden threshold tuned by leave-one-out on train at k = 10 (R0's train
neighbours). Agreement of each system with that decision, overall and where it is wrong;
G-img (no neighbours seen) is the control. Also the "present" rate by number of positive
neighbours.
E4 citation agreement: for G's and G-ind's "present" verdicts, the share of cited cases with the
finding vs. the share of all 10 neighbours with it; "present" without citation and citations
under absent / uncertain are counted.

On validation, the starters and G-img have only the 50 pilot studies; each system is reported on
its own studies and every comparison on the common ones (this checks the script; the numbers are
not results). The test split needs --test (plan §6: once) and every system must have all studies.

Writes data/runs/eval/<split>/report.md, results.json and per_study.csv (every individual check:
study, finding, truth, system, reading, verdict, outcome = hit / miss / false alarm / correct
absent, score; counting it reproduces the reported sensitivity and specificity, asserted).

Usage: python -I scripts/evaluation/evaluate.py <split> [--test]
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
RUNS = PROJECT / "data" / "runs"
PROC = PROJECT / "data" / "processed"
N_BOOT, SEED = 2000, 2026
K_G = 10
NF = len(FINDINGS)
S0_TABLE = "densenet121-res224-mimic_nb__pad__{split}.csv"  # the selected S0 (results §5.1)
STARTERS = {"P1-adapted-ns": "p1_adapted_ns__checklist", "P2-adapted": "p2_adapted__checklist",
            "P3-adapted": "p3_adapted__checklist"}
G_ARMS = {"G": "g__prompt2", "G−ind": "g_ind__prompt2", "G-img": "g_img__prompt2"}
BINARY_METRICS = ("sensitivity", "specificity", "f1")

# (number, first, second, metric, reading of first, reading of second); reading None = scores (AUROC)
COMPARISONS = [
    (1, "G", "S0", "auroc", None, None),
    (2, "G", "R0", "auroc", None, None),
    (3, "G", "G−ind", "auroc", None, None),
    (3, "G", "G−ind", "f1", "stated", "stated"),
    (4, "G", "G-img", "auroc", None, None),
    (4, "G", "G-img", "f1", "stated", "stated"),
    (5, "G-img", "S0", "auroc", None, None),
] + [(n, "G", p, m, "stated", "stated") for n, p in zip((6, 7, 8), STARTERS) for m in BINARY_METRICS]


# --- metric functions (tested in scripts/tests/test_evaluate.py) -----------------------------------
def per_finding(y, x, metric: str) -> np.ndarray:
    """(F,) metric per finding. x = scores for "auroc", else a 0/1 binary array."""
    if metric == "auroc":
        return np.array([metrics.auroc(y[:, j], x[:, j]) for j in range(y.shape[1])])
    return np.array([metrics.threshold_metrics(y[:, j], x[:, j].astype(float), 1.0)[metric]
                     for j in range(y.shape[1])])


def precision_at_k(y_query, y_neigh, y_train) -> dict:
    """y_query (n, F) bool; y_neigh (n, k, F) bool, the labels of each query's k neighbours;
    y_train (N, F) bool for the prevalence baseline. E2."""
    share = y_neigh.mean(axis=1)
    prev = y_train.mean(axis=0)
    out = {}
    for j, f in enumerate(FINDINGS):
        q = y_query[:, j]
        p = float(share[q, j].mean()) if q.any() else float("nan")
        out[f] = {"queries": int(q.sum()), "precision": p, "prevalence": float(prev[j]),
                  "lift": p / prev[j] if prev[j] else float("nan")}
    normal = ~y_query.any(axis=1)
    p = float((~y_neigh.any(axis=2)).mean(axis=1)[normal].mean()) if normal.any() else float("nan")
    prev_n = float((~y_train.any(axis=1)).mean())
    out["normal"] = {"queries": int(normal.sum()), "precision": p, "prevalence": prev_n, "lift": p / prev_n}
    return out


def copying(sys_bin, knn_dec, y) -> dict:
    """Agreement of a system's binary verdicts with the neighbours' decision, per finding and
    pooled, overall and where the neighbours' decision is wrong. All (n, F) bool. E3."""
    agree, wrong = sys_bin == knn_dec, knn_dec != y

    def rates(a, w):
        return {"pairs": int(a.size), "agree": float(a.mean()) if a.size else float("nan"),
                "neighbours_wrong": int(w.sum()),
                "agree_when_wrong": float(a[w].mean()) if w.any() else float("nan")}
    out = {f: rates(agree[:, j], wrong[:, j]) for j, f in enumerate(FINDINGS)}
    out["pooled"] = rates(agree.ravel(), wrong.ravel())
    return out


def present_by_positive_neighbours(sys_bin, n_pos) -> dict:
    """Share of "present" verdicts by number of positive neighbours (0, 1, 2, 3+), pooled over
    findings. sys_bin (n, F) bool; n_pos (n, F) int. E3."""
    out = {}
    for label, sel in (("0", n_pos == 0), ("1", n_pos == 1), ("2", n_pos == 2), ("3+", n_pos >= 3)):
        out[label] = {"pairs": int(sel.sum()),
                      "present_rate": float(sys_bin[sel].mean()) if sel.any() else float("nan")}
    return out


def citation_agreement(verdicts, label_of: dict, neighbours_of: dict) -> dict:
    """verdicts: (study, finding, status, cited IDs) per finding answer. label_of: study -> (F,)
    bool; neighbours_of: study -> the IDs given to the model. E4."""
    out = {f: {"present_calls": 0, "present_without_citation": 0, "cited": 0, "cited_with_finding": 0,
               "neighbours": 0, "neighbours_with_finding": 0} for f in FINDINGS}
    cited_not_present = 0
    for study, f, status, cited in verdicts:
        if status != "present":
            cited_not_present += bool(cited)
            continue
        j, o = FINDINGS.index(f), out[f]
        o["present_calls"] += 1
        o["present_without_citation"] += not cited
        o["cited"] += len(cited)
        o["cited_with_finding"] += sum(bool(label_of[c][j]) for c in cited)
        o["neighbours"] += len(neighbours_of[study])
        o["neighbours_with_finding"] += sum(bool(label_of[c][j]) for c in neighbours_of[study])
    for o in out.values():
        o["share"] = o["cited_with_finding"] / o["cited"] if o["cited"] else float("nan")
        o["baseline"] = o["neighbours_with_finding"] / o["neighbours"] if o["neighbours"] else float("nan")
    out["citations_under_absent_or_uncertain"] = cited_not_present
    return out


def outcome(truth: bool, verdict: bool) -> str:
    """One check of a verdict against the ground truth."""
    return ("hit" if verdict else "miss") if truth else ("false alarm" if verdict else "correct absent")


def sens_spec_from_outcomes(outcomes) -> tuple[float, float]:
    """Sensitivity and specificity counted from a list of outcome() strings."""
    n = {k: outcomes.count(k) for k in ("hit", "miss", "false alarm", "correct absent")}
    pos, neg = n["hit"] + n["miss"], n["false alarm"] + n["correct absent"]
    return (n["hit"] / pos if pos else float("nan")), (n["correct absent"] / neg if neg else float("nan"))


def common_studies(*systems) -> list[str]:
    """Studies every given system has, in CXR number order."""
    keys = set.intersection(*(set(s) for s in systems))
    return sorted(keys, key=lambda s: int(s[3:]))


# --- loading -----------------------------------------------------------------------------------------
def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_cache(arm: str, split: str, studies) -> dict:
    return {s: json.loads((RUNS / arm / split / f"{s}.json").read_text(encoding="utf-8")) for s in studies}


def load_systems(split: str, label_of: dict) -> dict:
    """name -> {"readings": {reading: {study: (F,) bool}}, "scores": {study: (F,)} or None,
    "unanswered": int, "missing": int}."""
    systems = {}
    s0 = json.loads((RUNS / "s0" / "selection.json").read_text())
    rows = read_csv(RUNS / "s0" / S0_TABLE.format(split=split))
    sc = {r["study"]: np.array([float(r[s0["label_map"][f]]) for f in FINDINGS]) for r in rows}
    thr = np.array([s0["thresholds"][f] for f in FINDINGS])
    systems["S0"] = {"scores": sc, "readings": {"threshold": {s: v >= thr for s, v in sc.items()}},
                     "unanswered": 0, "missing": 0}

    r0 = json.loads((RUNS / "r0" / "selection.json").read_text())
    rows = read_csv(RUNS / "r0" / f"{split}.csv")
    # The table rounds vote shares to 4 decimals (2/150 -> 0.0133 < the threshold 0.01333...), which
    # flips studies sitting exactly at a threshold. Scores are recomputed exactly from the stored
    # neighbours and checked against the table.
    sc = {r["study"]: np.array([label_of[n] for n in r["neighbours"].split(";")]).mean(axis=0) for r in rows}
    assert all(len(r["neighbours"].split(";")) == r0["k"] for r in rows)
    assert all(abs(sc[r["study"]][j] - float(r[f])) < 6e-5 for r in rows for j, f in enumerate(FINDINGS)), \
        "R0: recomputed vote shares differ from the table"
    thr = np.array([r0["thresholds"][f] for f in FINDINGS])
    systems["R0"] = {"scores": sc, "readings": {"threshold": {s: v >= thr for s, v in sc.items()}},
                     "unanswered": 0, "missing": 0}

    for name, arm in {**STARTERS, **G_ARMS}.items():
        path = RUNS / arm / f"{split}_checklist.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        entry = {"readings": {"stated": {r["study"]: np.array([r[f] == "1" for f in FINDINGS]) for r in rows}},
                 "scores": None, "unanswered": sum(r["outcome"] != "answer" for r in rows),
                 "missing": sum(r[f"checklist_{f}"] in ("missing", "conflict") for r in rows for f in FINDINGS),
                 "arm": arm}
        if name in G_ARMS:
            entry["scores"] = {r["study"]: np.array([float(r[f"prob_{f}"]) for f in FINDINGS]) for r in rows}
            sel = RUNS / arm / "selection.json"
            if name != "G-img":
                t = json.loads(sel.read_text())["threshold"]
                entry["readings"]["threshold"] = {s: v >= t for s, v in entry["scores"].items()}
                entry["threshold"] = t
        systems[name] = entry
    return systems


def stack(d: dict, studies) -> np.ndarray:
    return np.array([d[s] for s in studies])


def ci(values) -> tuple[float, float]:
    """metrics.ci, or (nan, nan) when every resample is undefined (e.g. no positives at all)."""
    v = np.asarray(values, float)
    return metrics.ci(v) if (~np.isnan(v)).any() else (float("nan"), float("nan"))


# --- report helpers ----------------------------------------------------------------------------------
def fmt(x, d=3):
    return "–" if x is None or np.isnan(x) else f"{x:.{d}f}"


def fmt_ci(v, lo, hi, d=3, signed=False):
    f = (lambda x: "–" if np.isnan(x) else f"{x:+.{d}f}") if signed else (lambda x: fmt(x, d))
    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--test", action="store_true", help="allow the test split (week 6)")
    args = ap.parse_args()
    if args.split == "test" and not args.test:
        sys.exit("refusing the test split without --test (plan §6: once, in week 6)")
    split = args.split

    split_of = {r["study"]: r["split"] for r in read_csv(PROC / "splits.csv")}
    label_of = {r["study"]: np.array([r[f] == "1" for f in FINDINGS]) for r in read_csv(PROC / "ground_truth.csv")}
    expected = sorted((s for s, sp in split_of.items() if sp == split), key=lambda s: int(s[3:]))
    train = sorted((s for s, sp in split_of.items() if sp == "train"), key=lambda s: int(s[3:]))
    y_train = stack(label_of, train)
    systems = load_systems(split, label_of)
    for name, sysd in systems.items():
        have = set(next(iter(sysd["readings"].values())))
        assert have <= set(expected), f"{name}: studies outside the {split} split"
        if split == "test":
            assert have == set(expected), f"{name}: {len(have)} of {len(expected)} test studies"
    out = {"split": split, "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "study", "paired": True}}
    md = [f"# Q3 evaluation — {split}\n",
          "Method: results §5.4 (E1–E5). Generated by `scripts/evaluation/evaluate.py`; do not edit.\n"]
    if split == "val":
        md.append("**Validation run: checks the script end to end. Starters and G-img have only the 50 "
                  "pilot studies. These numbers are not results.**\n")

    # --- per-system tables -----------------------------------------------------------------------
    md.append("## 1. Systems\n")
    md.append("| System | Studies | Unanswered | Missing verdicts | Readings | Threshold |")
    md.append("|---|---|---|---|---|---|")
    for name, s in systems.items():
        n = len(next(iter(s["readings"].values())))
        md.append(f"| {name} | {n} | {s['unanswered']} | {s['missing']} | {', '.join(s['readings'])} | "
                  f"{fmt(s['threshold'], 2) if 'threshold' in s else ('per finding' if name in ('S0', 'R0') else '–')} |")
    md.append("")
    out["systems"] = {}
    per_study = []  # one row per study x finding x system x reading (the individual checks)
    for name, s in systems.items():
        st = common_studies(next(iter(s["readings"].values())))
        y = stack(label_of, st)
        res = {"studies": len(st), "unanswered": s["unanswered"], "missing": s["missing"], "readings": {}}
        md.append(f"### {name} ({len(st)} studies)\n")
        head = "| Finding | Positives |"
        if s["scores"] is not None:
            sc = stack(s["scores"], st)
            a = per_finding(y, sc, "auroc")
            b = metrics.bootstrap(lambda i: per_finding(y[i], sc[i], "auroc"), len(st), N_BOOT, SEED)
            cis = [ci(b[:, j]) for j in range(NF)]
            mci = ci(np.nanmean(b, axis=1))
            res["auroc"] = {f: {"value": a[j], "ci95": list(cis[j])} for j, f in enumerate(FINDINGS)}
            res["macro_auroc"] = {"value": float(np.nanmean(a)), "ci95": list(mci)}
            head += " AUROC [95% CI] |"
        readings = list(s["readings"])
        head += "".join(f" Sens. / spec. / F1 ({r}) |" for r in readings)
        md.append(head)
        md.append("|" + "---|" * (head.count("|") - 1))
        vals = {}
        for r in readings:
            bn = stack(s["readings"][r], st)
            vals[r] = {m: per_finding(y, bn, m) for m in BINARY_METRICS}
            res["readings"][r] = {f: {m: vals[r][m][j] for m in BINARY_METRICS} for j, f in enumerate(FINDINGS)}
            for j, f in enumerate(FINDINGS):
                rows_f = [[st[i], f, int(y[i, j]), name, r, int(bn[i, j]), outcome(y[i, j], bn[i, j]),
                           f"{s['scores'][st[i]][j]:.6f}" if s["scores"] is not None else ""]
                          for i in range(len(st))]
                se, sp = sens_spec_from_outcomes([x[6] for x in rows_f])
                assert np.allclose([se, sp], [vals[r]["sensitivity"][j], vals[r]["specificity"][j]], equal_nan=True), \
                    f"{name}/{r}/{f}: per-study outcomes disagree with the reported metrics"
                per_study.extend(rows_f)
            res["readings"][r]["macro"] = {m: float(np.nanmean(vals[r][m])) for m in BINARY_METRICS}
        for j, f in enumerate(FINDINGS):
            row = f"| {f} | {int(y[:, j].sum())} |"
            if s["scores"] is not None:
                row += f" {fmt_ci(a[j], *cis[j])} |"
            row += "".join(" " + " / ".join(fmt(vals[r][m][j], 2) for m in BINARY_METRICS) + " |" for r in readings)
            md.append(row)
        row = "| **Macro** | |"
        if s["scores"] is not None:
            row += f" **{fmt_ci(float(np.nanmean(a)), *mci)}** |"
        row += "".join(" " + " / ".join(fmt(res["readings"][r]["macro"][m], 2) for m in BINARY_METRICS) + " |"
                       for r in readings)
        md.append(row + "\n")
        out["systems"][name] = res

    # --- E1 comparisons ----------------------------------------------------------------------------
    md.append("## 2. Pre-declared comparisons (E1; first − second, paired bootstrap 95% CI)\n")
    md.append("#1 is the primary comparison; #2–#8 are secondary.\n")
    md.append("| # | Comparison | Metric | Reading | Studies | Macro difference [95% CI] | "
              + " | ".join(FINDINGS) + " |")
    md.append("|" + "---|" * (6 + NF))
    out["comparisons"] = []
    for num, a_name, b_name, metric, ra, rb in COMPARISONS:
        if a_name not in systems or b_name not in systems:
            md.append(f"| {num} | {a_name} − {b_name} | {metric} | {ra or 'scores'} | 0 (system missing) | – | "
                      + " | ".join("–" for _ in FINDINGS) + " |")
            continue
        A, B = systems[a_name], systems[b_name]
        xa_d = A["scores"] if ra is None else A["readings"][ra]
        xb_d = B["scores"] if rb is None else B["readings"][rb]
        st = common_studies(xa_d, xb_d)
        y, xa, xb = stack(label_of, st), stack(xa_d, st), stack(xb_d, st)
        d = per_finding(y, xa, metric) - per_finding(y, xb, metric)
        b = metrics.bootstrap(lambda i: per_finding(y[i], xa[i], metric) - per_finding(y[i], xb[i], metric),
                              len(st), N_BOOT, SEED)
        cis = [ci(b[:, j]) for j in range(NF)]
        mci = ci(np.nanmean(b, axis=1))
        md.append(f"| {num}{' (primary)' if num == 1 else ''} | {a_name} − {b_name} | {metric} | "
                  f"{ra or 'scores'} | {len(st)} | **{fmt_ci(float(np.nanmean(d)), *mci, signed=True)}** | "
                  + " | ".join(fmt_ci(d[j], *cis[j], d=2, signed=True) for j in range(NF)) + " |")
        out["comparisons"].append({
            "number": num, "first": a_name, "second": b_name, "metric": metric, "reading": ra or "scores",
            "studies": len(st), "macro_difference": float(np.nanmean(d)), "macro_ci95": list(mci),
            "per_finding": {f: {"difference": d[j], "ci95": list(cis[j])} for j, f in enumerate(FINDINGS)}})
    md.append("")

    # --- E2 precision@k ----------------------------------------------------------------------------
    neighbours = {}  # system -> {study: [IDs, best first]}
    rows = read_csv(RUNS / "r0" / f"{split}.csv")
    neighbours["R0"] = {r["study"]: r["neighbours"].split(";") for r in rows}
    if "G" in systems:
        g_cache = load_cache(G_ARMS["G"], split, systems["G"]["readings"]["stated"])
        neighbours["G"] = {s: [n["study"] for n in rec["neighbours"]] for s, rec in g_cache.items()}
    if "P2-adapted" in systems:
        c = load_cache(STARTERS["P2-adapted"], split, systems["P2-adapted"]["readings"]["stated"])
        neighbours["P2-adapted"] = {s: [rec["retrieved_study"]] for s, rec in c.items() if rec.get("retrieved_study")}
    if "P3-adapted" in systems:
        c = load_cache(STARTERS["P3-adapted"], split, systems["P3-adapted"]["readings"]["stated"])
        neighbours["P3-adapted"] = {
            s: [x["study"] for x in sorted(rec["retrieved"], key=lambda x: -x["similarity"])]
            for s, rec in c.items() if rec.get("retrieved")}
    for name, nb in neighbours.items():
        assert all(split_of[n] == "train" for ids in nb.values() for n in ids), f"{name}: a neighbour outside train"
    md.append("## 3. Retrieval quality (E2)\n")
    md.append("Per finding, over test studies with the finding: share of the k retrieved training cases "
              "that also have it; lift = that share / the finding's train prevalence (1 = random). "
              "Normal row: normal queries, share of normal neighbours.\n")
    if "G" in neighbours:
        same = sum(set(neighbours["G"][s]) == set(neighbours["R0"][s][:K_G]) for s in neighbours["G"])
        md.append(f"Check: G's 10 neighbours equal R0's first 10 for {same} of {len(neighbours['G'])} studies.\n")
        out["g_neighbours_equal_r0_first10"] = [same, len(neighbours["G"])]
    md.append("| System | k | Queries | " + " | ".join(FINDINGS) + " | Normal |")
    md.append("|" + "---|" * (4 + NF))
    out["precision_at_k"] = {}
    for name, nb in neighbours.items():
        own_k = min(len(v) for v in nb.values())
        for k in sorted({own_k, 1}, reverse=True):
            st = common_studies(nb)
            yq = stack(label_of, st)
            yn = np.array([[label_of[n] for n in nb[s][:k]] for s in st])
            p = precision_at_k(yq, yn, y_train)
            out["precision_at_k"][f"{name}@{k}"] = p
            md.append(f"| {name} | {k} | {len(st)} | "
                      + " | ".join(f"{fmt(p[f]['precision'], 2)} (×{fmt(p[f]['lift'], 1)}, n={p[f]['queries']})"
                                   for f in FINDINGS + ["normal"]) + " |")
    md.append("| Train prevalence | | | " + " | ".join(
        fmt(next(iter(out["precision_at_k"].values()))[f]["prevalence"], 3) for f in FINDINGS + ["normal"]) + " |\n")

    # --- E3 copying rate ---------------------------------------------------------------------------
    if "G" in neighbours:
        tr_rows = {r["study"]: r["neighbours"].split(";")[:K_G] for r in read_csv(RUNS / "r0" / "train.csv")}
        vote_tr = np.array([[label_of[n] for n in tr_rows[s]] for s in train]).mean(axis=1)
        thr10 = np.array([metrics.youden_threshold(y_train[:, j], vote_tr[:, j]) for j in range(NF)])
        st_g = common_studies(neighbours["G"])
        npos = {s: np.array([label_of[n] for n in neighbours["G"][s]]).sum(axis=0) for s in st_g}
        knn = {s: npos[s] / K_G >= thr10 for s in st_g}
        md.append("## 4. Copying rate (E3)\n")
        md.append("Neighbours' decision = share of G's 10 neighbours with the finding ≥ a per-finding "
                  "Youden threshold (leave-one-out on train, k = 10): "
                  + ", ".join(f"{f} {thr10[j]:.2f}" for j, f in enumerate(FINDINGS))
                  + ". G-img never saw the neighbours (control).\n")
        md.append("| System | Reading | Studies | Pairs | Agree (all) | Neighbours wrong | Agree when wrong | "
                  "Present rate at 0 / 1 / 2 / 3+ positive neighbours |")
        md.append("|---|---|---|---|---|---|---|---|")
        out["copying"] = {"thresholds_k10": dict(zip(FINDINGS, thr10.tolist()))}
        for name in ("G", "G−ind", "G-img"):
            if name not in systems:
                continue
            for r, bins in systems[name]["readings"].items():
                st = common_studies(bins, knn)
                sb, kd, y = stack(bins, st), stack(knn, st), stack(label_of, st)
                c = copying(sb, kd, y)
                pr = present_by_positive_neighbours(sb, stack(npos, st))
                out["copying"][f"{name}/{r}"] = {"studies": len(st), **c, "present_by_positive_neighbours": pr}
                md.append(f"| {name} | {r} | {len(st)} | {c['pooled']['pairs']} | {fmt(c['pooled']['agree'], 3)} | "
                          f"{c['pooled']['neighbours_wrong']} | {fmt(c['pooled']['agree_when_wrong'], 3)} | "
                          + " / ".join(f"{fmt(pr[b]['present_rate'], 2)} (n={pr[b]['pairs']})" for b in pr) + " |")
        if "G-img" in systems:
            st = common_studies(systems["G"]["readings"]["stated"], systems["G-img"]["readings"]["stated"], knn)
            md.append(f"\nPer finding, stated verdicts, agreement when the neighbours are wrong, on the {len(st)} "
                      "studies G and G-img share:\n")
            md.append("| System | " + " | ".join(FINDINGS) + " |")
            md.append("|" + "---|" * (1 + NF))
            for name in ("G", "G-img"):
                c = copying(stack(systems[name]["readings"]["stated"], st), stack(knn, st), stack(label_of, st))
                md.append(f"| {name} | " + " | ".join(
                    f"{fmt(c[f]['agree_when_wrong'], 2)} (n={c[f]['neighbours_wrong']})" for f in FINDINGS) + " |")
        md.append("")

    # --- E4 citation agreement ---------------------------------------------------------------------
    md.append("## 5. Citation label agreement (E4)\n")
    md.append("For \"present\" verdicts: share of cited cases with the finding vs. share of all 10 "
              "neighbours given (baseline).\n")
    md.append("| System | Finding | Present calls | Without citation | Cited | Share with finding | Baseline |")
    md.append("|---|---|---|---|---|---|---|")
    out["citations"] = {}
    for name in ("G", "G−ind"):
        if name not in systems:
            continue
        cache = load_cache(G_ARMS[name], split, systems[name]["readings"]["stated"])
        nb_of = {s: [n["study"] for n in rec["neighbours"]] for s, rec in cache.items()}
        verdicts = [(s, x["finding"], x["status"], x["cited_case_ids"])
                    for s, rec in cache.items() for x in (rec["parsed"] or {}).get("findings", [])]
        c = citation_agreement(verdicts, label_of, nb_of)
        out["citations"][name] = c
        for f in FINDINGS:
            o = c[f]
            md.append(f"| {name} | {f} | {o['present_calls']} | {o['present_without_citation']} | {o['cited']} | "
                      f"{fmt(o['share'], 2)} | {fmt(o['baseline'], 2)} |")
        md.append(f"| {name} | citations under absent / uncertain | | | {c['citations_under_absent_or_uncertain']} | | |")
    md.append("")

    dest = RUNS / "eval" / split
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.md").write_text("\n".join(md), encoding="utf-8")
    with open(dest / "results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(dest / "per_study.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "finding", "truth", "system", "reading", "verdict", "outcome", "score"])
        w.writerows(per_study)
    print("\n".join(md))
    print(f"\nwrote {dest / 'report.md'}, results.json and per_study.csv ({len(per_study)} checks)")


if __name__ == "__main__":
    main()
