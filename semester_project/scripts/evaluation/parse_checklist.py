"""Parse the findings checklist at the top of each answer (checklist question) into binary findings.

Rules, fixed before any checklist answer was seen (diary 2026-10-17):
- Markdown characters # * _ ` | are ignored, as are leading bullet or number markers.
- A checklist line starts with a finding's name, then ":" or a dash, then present / absent /
  uncertain (case-insensitive); text after the answer word is allowed and ignored.
- Per finding: exactly one answer (or several that agree) -> that answer; none -> missing;
  several that disagree -> conflict.
- Binary: present -> 1; absent, uncertain (plan §5), missing, conflict -> 0.
- A malformed-function-call outcome -> all missing.
- block_at_start: the first six non-empty lines are the six checklist lines (compliance).

Writes data/runs/<arm>/<split>_checklist.csv: study, outcome, block_at_start, the checklist answer for each
finding, and its binary label.

Usage: python -I scripts/evaluation/parse_checklist.py <split> <arm> [<arm> ...]
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import FINDINGS  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
LINE = re.compile(r"^\s*(?:[-•–]\s+|\d+[.)]\s+)?(?P<finding>" + "|".join(re.escape(f) for f in FINDINGS)
                  + r")\s*[:\-–—]\s*(?P<verdict>present|absent|uncertain)\b", re.IGNORECASE)
CANON = {f.lower(): f for f in FINDINGS}


def parse(text: str) -> tuple[dict, bool]:
    lines = [re.sub(r"[#*_`|]", "", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    found = {f: set() for f in FINDINGS}
    first6 = []
    for i, ln in enumerate(lines):
        m = LINE.match(ln)
        if m:
            f = CANON[m["finding"].lower()]
            found[f].add(m["verdict"].lower())
            if i < 6:
                first6.append(f)
    verdicts = {f: ("missing" if not v else v.pop() if len(v) == 1 else "conflict") for f, v in found.items()}
    return verdicts, sorted(first6) == sorted(FINDINGS)


def main(split: str, arms: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for arm in arms:
        files = sorted((PROJECT / "data" / "runs" / arm / split).glob("*.json"), key=lambda p: int(p.stem[3:]))
        assert files, f"no cached answers for {arm}/{split}"
        rows = []
        for f in files:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("outcome") == "malformed_function_call":
                verdicts, at_start = {g: "missing" for g in FINDINGS}, False
            else:
                verdicts, at_start = parse(rec.get("response_text"))
            rows.append([rec["study"], rec.get("outcome", "answer"), int(at_start)]
                        + [verdicts[g] for g in FINDINGS] + [int(verdicts[g] == "present") for g in FINDINGS])
        out = PROJECT / "data" / "runs" / arm / f"{split}_checklist.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["study", "outcome", "block_at_start"] + [f"checklist_{g}" for g in FINDINGS] + FINDINGS)
            w.writerows(rows)
        cells = [v for r in rows for v in r[3:3 + len(FINDINGS)]]
        print(f"{arm}: {len(rows)} answers -> {out.name}; block at start {sum(r[2] for r in rows)}/{len(rows)}; "
              f"missing {cells.count('missing')}, conflict {cells.count('conflict')} of {len(cells)} checklist entries")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
