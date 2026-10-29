"""Label the systems' free-text answers with CheXbert (plan §5: the same labeller path for every
system in the starter comparison).

Rules, fixed before any labels were seen (diary 2026-10-17):
1. A malformed-function-call outcome counts as "no findings reported": empty text, all negative.
2. Light markdown clean-up, same for every system: the characters # * _ ` | are removed (CheXbert
   was trained on plain report text); all words are kept.
3. CheXbert reads at most 512 tokens and the report labeller truncates beyond that. Many answers
   are longer (P1-adapted-ns: all; P2-adapted: 60%), so every answer is split into sentences
   packed into chunks of at most 512 tokens, each chunk is labelled, and the classes are combined
   per observation: positive > uncertain > negative > blank. An answer that fits is one chunk,
   labelled exactly as a report would be. A sentence longer than a chunk is truncated (counted).
4. Binary label = positive only (uncertain -> negative, plan §5); our six findings are CheXbert's
   heads of the same name, as in the check against the ground truth (results §2.4b).

Writes data/runs/<arm>/<split>_labels.csv: study, outcome, n_chunks, truncated_sentences, the raw
combined class of all 14 observations (0 blank, 1 positive, 2 negative, 3 uncertain), and the
binary label of all 14.

Usage: python scripts/outdated/label_outputs.py <split> <arm> [<arm> ...]
  (run with HF_HOME set to the project cache, like chexbert_labels.py)
"""
import csv
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))  # chexbert_labels
from chexbert_labels import CONDITIONS, load_model  # noqa: E402

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
MAX_TOKENS = 512
RANK = {1: 3, 3: 2, 2: 1, 0: 0}  # combine chunks: positive > uncertain > negative > blank


def answer_text(rec: dict) -> str:
    if rec.get("outcome") == "malformed_function_call":
        return ""  # rule 1 (response_text holds agno's error message, not an answer)
    return re.sub(r"[#*_`|]", " ", rec.get("response_text") or "")


def chunks(text: str, tok) -> tuple[list[str], int]:
    """Sentences (split at . ! ? or line breaks) packed greedily into <= MAX_TOKENS chunks."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    out, cur, cur_len, truncated = [], [], 2, 0  # 2 = [CLS] + [SEP]
    for s in sentences:
        n = len(tok.tokenize(s))
        if n + 2 > MAX_TOKENS:
            truncated += 1
        if cur and cur_len + n > MAX_TOKENS:
            out.append(" ".join(cur))
            cur, cur_len = [], 2
        cur.append(s)
        cur_len += n
    if cur:
        out.append(" ".join(cur))
    return out or [""], truncated


@torch.no_grad()
def label(texts: list[str], model, tok, device) -> list[list[int]]:
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
    outs = [o.argmax(1).tolist() for o in model(enc["input_ids"], enc["attention_mask"])]
    return [[outs[c][j] for c in range(len(CONDITIONS))] for j in range(len(texts))]


def main(split: str, arms: list[str]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(device)
    for arm in arms:
        files = sorted((PROJECT / "data" / "runs" / arm / split).glob("*.json"), key=lambda p: int(p.stem[3:]))
        assert files, f"no cached answers for {arm}/{split}"
        rows = []
        for f in files:
            rec = json.loads(f.read_text(encoding="utf-8"))
            parts, truncated = chunks(answer_text(rec), tok)
            raw = [max(col, key=RANK.get) for col in zip(*label(parts, model, tok, device))]
            rows.append([rec["study"], rec.get("outcome", "answer"), len(parts), truncated]
                        + raw + [int(r == 1) for r in raw])
        out = PROJECT / "data" / "runs" / arm / f"{split}_labels.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["study", "outcome", "n_chunks", "truncated_sentences"]
                       + [f"raw_{c}" for c in CONDITIONS] + CONDITIONS)
            w.writerows(rows)
        multi = sum(r[2] > 1 for r in rows)
        print(f"{arm}: {len(rows)} answers labelled -> {out.name}; {multi} split into >1 chunk; "
              f"{sum(r[3] for r in rows)} over-long sentences truncated")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
