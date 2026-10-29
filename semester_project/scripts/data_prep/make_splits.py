"""Train / validation / test split of the 3,604 ground-truth studies (one study per patient,
results.md §2.2, so a study split is a patient split).

Test (400): positive-enriched. Findings are processed rarest first; positives not yet in the
test set are sampled until each finding has >= MIN_POS, then the set is filled at random.
Validation (400): random from the rest (natural prevalence); the 50-study pilot is drawn from it.
Train: everything else; it is the only source for retrieval indexes (P2, P3, R0, G).

Usage: python -I scripts/data_prep/make_splits.py <processed_dir>
"""
import csv
import random
import sys
from pathlib import Path

SEED, N_TEST, N_VAL, N_PILOT, MIN_POS = 2026, 400, 400, 50, 35


def main(proc: Path) -> None:
    with open(proc / "ground_truth.csv") as fh:
        rows = list(csv.DictReader(fh))
    findings = [c for c in rows[0] if c != "study"]
    gt = {r["study"]: {f for f in findings if r[f] == "1"} for r in rows}
    studies = sorted(gt, key=lambda s: int(s[3:]))
    rng = random.Random(SEED)

    test = set()
    for f in sorted(findings, key=lambda f: sum(f in gt[s] for s in studies)):
        pool = [s for s in studies if f in gt[s] and s not in test]
        rng.shuffle(pool)
        while sum(f in gt[s] for s in test) < MIN_POS:
            test.add(pool.pop())
    rest = [s for s in studies if s not in test]
    rng.shuffle(rest)
    test |= set(rest[:N_TEST - len(test)])
    rest = [s for s in studies if s not in test]
    rng.shuffle(rest)
    val = set(rest[:N_VAL])
    train = set(rest[N_VAL:])
    pilot = set(rng.sample(sorted(val, key=lambda s: int(s[3:])), N_PILOT))

    # leakage guards (plan §2): disjoint, complete
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(studies) and pilot <= val
    assert len(test) == N_TEST and len(val) == N_VAL

    with open(proc / "splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "split", "pilot"])
        for s in studies:
            split = "train" if s in train else "val" if s in val else "test"
            w.writerow([s, split, int(s in pilot)])

    print(f"{'split':8s} {'n':>5s} " + " ".join(f"{f[:12]:>12s}" for f in findings) + f" {'none':>6s}")
    for name, part in (("train", train), ("val", val), ("pilot", pilot), ("test", test)):
        counts = [sum(f in gt[s] for s in part) for f in findings]
        none = sum(not gt[s] for s in part)
        print(f"{name:8s} {len(part):5d} " + " ".join(f"{c:12d}" for c in counts) + f" {none:6d}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
