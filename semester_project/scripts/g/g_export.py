"""Export G-family answers to the same per-study table the starters have (parse_checklist.py).

Rules (fixed 2026-10-18, before any pilot result was seen):
- A finding's status is used if it occurs exactly once in the parsed answer; otherwise "missing".
- An answer that is not valid JSON: every finding "missing".
- Binary = status "present" (absent, uncertain, missing -> 0), the starters' rule (results §3.5).
- prob_<finding> = the stated probability; a missing probability is written as 0.0 and counted.
- Other validation problems (e.g. a cited ID that was not among the neighbours) do not change
  the statuses; they are counted.

Writes data/runs/<arm>/<split>_checklist.csv: study, outcome, block_at_start (1 = no problems),
checklist_<finding>, <finding> (binary), prob_<finding>.

Usage: python -I scripts/g/g_export.py <split> <arm> [<arm> ...]     (arm e.g. g__prompt1)
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)


def main(split: str, arms: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for arm in arms:
        files = sorted((PROJECT / "data" / "runs" / arm / split).glob("*.json"), key=lambda p: int(p.stem[3:]))
        assert files, f"no cached answers for {arm}/{split}"
        rows, n_invalid, n_missing, n_noprob, n_problem = [], 0, 0, 0, 0
        for f in files:
            rec = json.loads(f.read_text(encoding="utf-8"))
            parsed = rec.get("parsed")
            n_invalid += parsed is None
            n_problem += bool(rec.get("problems"))
            found = {}
            for x in (parsed or {}).get("findings", []):
                found.setdefault(x.get("finding"), []).append(x)
            status, prob = {}, {}
            for g in FINDINGS:
                one = found.get(g, [])
                status[g] = one[0]["status"] if len(one) == 1 else "missing"
                p = one[0].get("probability") if len(one) == 1 else None
                prob[g] = float(p) if p is not None else 0.0
                n_missing += status[g] == "missing"
                n_noprob += p is None
            rows.append([rec["study"], "answer" if parsed is not None else "invalid_json", int(not rec.get("problems"))]
                        + [status[g] for g in FINDINGS] + [int(status[g] == "present") for g in FINDINGS]
                        + [f"{prob[g]:.4f}" for g in FINDINGS])
        out = PROJECT / "data" / "runs" / arm / f"{split}_checklist.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["study", "outcome", "block_at_start"] + [f"checklist_{g}" for g in FINDINGS] + FINDINGS
                       + [f"prob_{g}" for g in FINDINGS])
            w.writerows(rows)
        print(f"{arm}: {len(rows)} answers -> {out.name}; invalid JSON {n_invalid}; answers with problems "
              f"{n_problem}; missing statuses {n_missing}; missing probabilities {n_noprob}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
