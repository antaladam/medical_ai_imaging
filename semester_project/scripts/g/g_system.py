"""G, G-ind and G-img: the built grounded image + text system (design: g_design.md, decided
2026-10-18; results.md §4.4).

  patient's frontal image -> BiomedCLIP (R0's index: train split) -> 10 most similar train studies
  -> ONE Gemini call: the patient's image + the clinical indication + the 10 neighbours' reports
  -> JSON enforced by the API's response schema: per finding status / probability / cited case IDs
     / rationale, plus an impression.

Arms: g (as above), g_ind (identical, indication line omitted), g_img (image only: no indication,
no neighbours). Model gemini-3-flash-preview at its default temperature and thinking level; no
tools, no output cap. Every response is cached in full with the exact prompt; every call goes
through the metered client (request budget, USD 10 per-run limit).

Prompt versions (D6: at most 3, developed on the 50 pilot studies): **prompt 2 is frozen** (2026-10-18);
non-pilot runs refuse any other version. Answers are kept
per version in data/runs/<arm>__prompt<N>/<split>/<study>.json.

Usage: python -I scripts/g/g_system.py <arm> <split> [--pilot] [--limit N] [--prompt N]
                                     [--max-requests N] [--max-usd X] [--test]
"""
import argparse
import csv
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import (DEFAULT_MAX_USD, FINDINGS, PROJECT, load_api_key, metered_client, read_cache,  # noqa: E402
                    requests_made, run_spend, set_call_context, set_request_budget, with_retry, write_cache)
from google.genai import types  # noqa: E402

MODEL = "gemini-3-flash-preview"
K = 10  # D1
FROZEN_PROMPT = 2  # frozen 2026-10-18 (D6); validation and test runs must use it
ARMS = ("g", "g_ind", "g_img")
PROC = PROJECT / "data" / "processed"
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"
REPORTS = PROJECT / "data" / "raw" / "iu_xray" / "reports" / "ecgen-radiology"


class FindingOut(BaseModel):
    finding: Literal["Lung Opacity", "Cardiomegaly", "Atelectasis", "Lung Lesion", "Pleural Effusion", "Fracture"]
    status: Literal["present", "absent", "uncertain"]
    probability: float = Field(description="Probability (0 to 1) that the finding is present in this patient")
    cited_case_ids: list[str]
    rationale: str


class Output(BaseModel):
    findings: list[FindingOut]
    impression: str


# --- prompt (version 1) -----------------------------------------------------------------------
DEFINITIONS = """Findings and their definitions:
- Lung Opacity: airspace opacity, consolidation, infiltrate or airspace disease in the lung. A discrete nodule or mass is not Lung Opacity (it is Lung Lesion).
- Cardiomegaly: an enlarged heart. A borderline or upper-normal heart size counts as absent.
- Atelectasis: collapse or incomplete expansion of lung tissue, including subsegmental or plate-like atelectasis.
- Lung Lesion: a nodule, mass or cavity in the lung. A calcified granuloma does not count.
- Pleural Effusion: fluid in the pleural space.
- Fracture: a fracture of the ribs, clavicles, spine or any other visible bone. Healed or old fractures count."""

PROMPTS = {1: {
    "intro": ("You are a decision-support tool for a radiologist. Review this patient's frontal chest X-ray "
              "and report on six findings. A radiologist checks your output; give your best assessment."),
    "steps_with_cases": [
        "Read the patient's image systematically first.",
        f"Then look at the {K} most similar cases from a reference database, listed below with their reports. "
        "They are other patients whose images look similar to this one. Use them as supporting evidence to "
        "confirm or question what you see in the patient's image, not as the answer: a finding in a similar "
        "case does not mean that this patient has it.",
    ],
    "steps_image_only": ["Read the patient's image systematically."],
    "step_indication": "Take the clinical indication into account.",
    "step_output": ("For each of the six findings, give its status (present, absent or uncertain), the probability "
                    "(0 to 1) that it is present in this patient, the IDs of the similar cases that support your "
                    "call (an empty list if none), and a rationale of one or two sentences. Then give a short "
                    "impression. Do not explain what the findings are."),
    "case_line": "[{i}] {study} (similarity {sim:.3f}): {report}",
}}
# Version 2 (diary 2026-10-18): prompt 1 led to citations by list position in 3 / 50 pilot answers
# and to coarse probabilities (198 / 300 exactly 0.05). Only these two things change.
PROMPTS[2] = dict(
    PROMPTS[1],
    step_output=("For each of the six findings, give its status (present, absent or uncertain), the probability "
                 "(0 to 1) that it is present in this patient, the IDs of the similar cases that support your "
                 "call (an empty list if none), and a rationale of one or two sentences. Then give a short "
                 "impression. Do not explain what the findings are. "
                 "Cite cases by their ID exactly as written, e.g. CXR316. "
                 "Give the probability with two decimals as your honest estimate; for subtle, borderline or "
                 "partly obscured appearances, use intermediate values rather than a fixed low value."),
    case_line="Case {study} (similarity {sim:.3f}): {report}",
)


def build_prompt(arm: str, version: int, indication: str, neighbours: list[dict]) -> str:
    p = PROMPTS[version]
    steps = list(p["steps_image_only"] if arm == "g_img" else p["steps_with_cases"])
    if arm == "g":
        steps.append(p["step_indication"])
    steps.append(p["step_output"])
    parts = [p["intro"], DEFINITIONS, "How to work:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))]
    if arm == "g":
        parts.append(f"Clinical indication: {indication or 'not provided'}")
    if arm != "g_img":
        parts.append("Similar cases (most similar first):\n" + "\n".join(
            p["case_line"].format(i=i, study=n["study"], sim=n["similarity"], report=n["report"] or "no report text")
            for i, n in enumerate(neighbours, 1)))
    parts.append("The patient's frontal chest X-ray follows.")
    return "\n\n".join(parts)


# --- data ---------------------------------------------------------------------------------------
def report_sections(study: str) -> dict:
    root = ET.parse(REPORTS / f"{study[3:]}.xml").getroot()
    assert root.find("uId").get("id") == study
    return {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}


def report_text(study: str) -> str:
    sec = report_sections(study)
    return " ".join(f"{k}: {sec[k]}" for k in ("FINDINGS", "IMPRESSION") if sec.get(k))


class Retriever:
    """BiomedCLIP cosine top-K over the train split (R0's index)."""

    def __init__(self, splits: dict):
        emb = np.load(PROC / "biomedclip_frontal.npy")
        with open(PROC / "biomedclip_frontal_ids.csv") as fh:
            self.row = {r["study"]: int(r["row"]) for r in csv.DictReader(fh)}
        self.emb = emb
        self.train = sorted((s for s, sp in splits.items() if sp == "train"), key=lambda s: int(s[3:]))
        self.index = emb[[self.row[s] for s in self.train]]

    def top(self, study: str, k: int = K) -> list[dict]:
        assert study not in set(self.train), f"leakage: {study} is in the retrieval index"
        sim = self.index @ self.emb[self.row[study]]
        order = np.argsort(-sim, kind="stable")[:k]
        return [{"study": self.train[i], "similarity": float(sim[i])} for i in order]


def validate(parsed: dict, allowed_ids: set) -> list[str]:
    """Problems with a parsed response (empty list = valid)."""
    problems = []
    names = [f.get("finding") for f in parsed.get("findings", [])]
    if sorted(names) != sorted(FINDINGS):
        problems.append(f"findings not exactly the six: {names}")
    for f in parsed.get("findings", []):
        if not 0.0 <= float(f.get("probability", -1)) <= 1.0:
            problems.append(f"{f.get('finding')}: probability out of range")
        bad = [c for c in f.get("cited_case_ids", []) if c not in allowed_ids]
        if bad:
            problems.append(f"{f.get('finding')}: cited cases not provided {bad}")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", type=int, default=FROZEN_PROMPT, choices=sorted(PROMPTS))
    ap.add_argument("--max-requests", type=int)
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    ap.add_argument("--test", action="store_true", help="allow the test split (week 6)")
    args = ap.parse_args()
    if not args.pilot and args.prompt != FROZEN_PROMPT:
        sys.exit(f"prompt {args.prompt} is not the frozen prompt ({FROZEN_PROMPT}); only pilot runs may use it (D6)")
    if args.split == "test" and not args.test:
        sys.exit("refusing the test split without --test (plan §6: once, in week 6)")
    set_request_budget(args.max_requests, args.max_usd)

    with open(PROC / "splits.csv") as fh:
        rows = list(csv.DictReader(fh))
    splits = {r["study"]: r["split"] for r in rows}
    with open(PROC / "frontal.csv") as fh:
        image_of = {r["study"]: r["image"] for r in csv.DictReader(fh)}
    queries = [r["study"] for r in rows if r["split"] == args.split and (not args.pilot or r["pilot"] == "1")]
    queries.sort(key=lambda s: int(s[3:]))
    if args.limit:
        queries = queries[:args.limit]
    retriever = Retriever(splits)
    arm = f"{args.arm}__prompt{args.prompt}"
    client = metered_client(load_api_key())
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Output)

    for i, s in enumerate(queries, 1):
        if read_cache(arm, args.split, s):
            continue
        neighbours = [] if args.arm == "g_img" else retriever.top(s)
        for n in neighbours:
            n["report"] = report_text(n["study"])
        indication = report_sections(s).get("INDICATION", "") if args.arm == "g" else ""
        prompt = build_prompt(args.arm, args.prompt, indication, neighbours)
        image = types.Part.from_bytes(data=(IMAGES / f"{image_of[s]}.png").read_bytes(), mime_type="image/png")
        set_call_context(arm=arm, split=args.split, study=s)
        r = with_retry(lambda: client.models.generate_content(model=MODEL, contents=[prompt, image], config=config))
        try:
            parsed = json.loads(r.text)
            problems = validate(parsed, {n["study"] for n in neighbours})
        except (json.JSONDecodeError, TypeError) as e:
            parsed, problems = None, [f"not valid JSON: {e}"]
        u = r.usage_metadata
        write_cache(arm, args.split, s, {
            "arm": arm, "system": args.arm, "prompt_version": args.prompt, "study": s, "image": image_of[s],
            "indication": indication, "neighbours": [{k: n[k] for k in ("study", "similarity")} for n in neighbours],
            "prompt": prompt, "model_requested": MODEL, "model_version": r.model_version,
            "response_text": r.text, "parsed": parsed, "problems": problems,
            "usage": {"input": u.prompt_token_count, "output": u.candidates_token_count,
                      "thinking": u.thoughts_token_count, "total": u.total_token_count},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"{i}/{len(queries)} {s} {'OK' if not problems else 'PROBLEMS ' + '; '.join(problems)[:120]} "
              f"tokens in/out/think {u.prompt_token_count}/{u.candidates_token_count}/{u.thoughts_token_count}",
              flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"API calls this run: {requests_made()}; cost this run: ${run_spend():.4f}", flush=True)
