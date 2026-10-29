"""P2-adapted: the `vision_rag` starter run as a batch job on IU X-Ray.

Kept as shipped (rag_tutorials/vision_rag/vision_rag.py):
  - embed -> top-1 cosine retrieval (np.dot, argmax) -> generate
  - image preparation: resize_image (max 1568*1568 px; never triggers on IU, all images 512 px wide)
  - the answer prompt text, with the fixed findings question
Adaptations (each recorded in results.md):
  1. Batch runner instead of the Streamlit UI.
  2. Embedder: Gemini Embedding 2 at default settings instead of Cohere embed-v4.0
     (decision: no Cohere account). Images embedded one per request, no task type.
  3. Query: the patient's frontal image instead of a text question (image -> image retrieval).
  4. Generation input: the patient's image AND the retrieved image (as shipped, only the
     retrieved image reached the model, which would describe a different patient). The prompt's
     "the following image" becomes "the following images" plus one sentence naming them.
  5. Generation model: gemini-3-flash-preview instead of the retired gemini-2.5-flash.
Index: train-split frontal images only. Leakage assertion: no query study is in the index.

Usage:
  python -I scripts/starters/p2_adapted.py index [--limit N] [--max-requests N]
  python -I scripts/starters/p2_adapted.py run <split> [--pilot] [--limit N] [--smoke] [--max-requests N]
"""
import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_MAX_USD, PROJECT, run_spend, arm_name, load_api_key, question, metered_client, read_cache, requests_made, set_call_context, set_request_budget, with_retry, write_cache  # noqa: E402
from google.genai import types  # noqa: E402

ARM = "p2_adapted"
EMBED_MODEL = "gemini-embedding-2"
GEN_MODEL = "gemini-3-flash-preview"
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"
EMB_DIR = PROJECT / "data" / "runs" / ARM / "embeddings"

# --- copied from vision_rag.py (lines 80-91) ---
max_pixels = 1568 * 1568


def resize_image(pil_image: PIL.Image.Image) -> None:
    org_width, org_height = pil_image.size
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        pil_image.thumbnail((int(org_width * scale_factor), int(org_height * scale_factor)))
# --- end copy ---

# vision_rag.py lines 359-363, with adaptation 4 (marked)
PROMPT = (
    "Answer the question based on the following images. "  # adapted: "image" -> "images"
    "The first image is the patient's chest X-ray; the second image is the most similar image "
    "retrieved from the database. "  # adapted: added sentence naming the two images
    "Be as elaborate as possible giving extra relevant information.\n"
    "Don't use markdown formatting in the response.\n"
    "Please provide enough context for your answer.\n\n"
    "Question: {question}"
)




def load_frontal():
    with open(PROJECT / "data" / "processed" / "frontal.csv") as fh:
        frontal = {r["study"]: r["image"] for r in csv.DictReader(fh)}
    with open(PROJECT / "data" / "processed" / "splits.csv") as fh:
        splits = list(csv.DictReader(fh))
    return frontal, splits


def image_bytes(image: str) -> bytes:
    pil = PIL.Image.open(IMAGES / f"{image}.png")
    fmt = pil.format or "PNG"
    resize_image(pil)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def embed_image(client, study: str, image: str) -> np.ndarray:
    path = EMB_DIR / f"{study}.npy"
    if path.exists():
        return np.load(path)
    part = types.Part.from_bytes(data=image_bytes(image), mime_type="image/png")
    r = with_retry(lambda: client.models.embed_content(model=EMBED_MODEL, contents=[part]))
    vec = np.asarray(r.embeddings[0].values, dtype=np.float32)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec


def cmd_index(args, client):
    frontal, splits = load_frontal()
    train = [r["study"] for r in splits if r["split"] == "train"]
    if args.limit:
        train = train[:args.limit]
    for i, s in enumerate(train, 1):
        set_call_context(arm=ARM, split="train", study=s)
        embed_image(client, s, frontal[s])
        if i % 100 == 0 or i == len(train):
            print(f"indexed {i}/{len(train)}", flush=True)


def load_index(frontal, splits):
    train = [r["study"] for r in splits if r["split"] == "train"]
    have = [s for s in train if (EMB_DIR / f"{s}.npy").exists()]
    if len(have) < len(train):
        print(f"WARNING: index incomplete ({len(have)}/{len(train)} train images embedded)")
    return have, np.stack([np.load(EMB_DIR / f"{s}.npy") for s in have])


def cmd_run(args, client):
    frontal, splits = load_frontal()
    queries = [r["study"] for r in splits if r["split"] == args.split and (not args.pilot or r["pilot"] == "1")]
    if args.limit:
        queries = queries[:args.limit]
    index_ids, index = load_index(frontal, splits)
    assert not set(queries) & set(index_ids), "leakage: query study in the retrieval index"
    n_train = sum(r["split"] == "train" for r in splits)
    arm = arm_name(ARM, args.question)  # answers per question version; embeddings stay in ARM
    if len(index_ids) < n_train:
        if not args.smoke:
            sys.exit("Index incomplete; run `index` first (or use --smoke for a test run).")
        arm = arm + "_smoke"  # partial-index test output never goes to the real cache
        print(f"SMOKE TEST: partial index, output to data/runs/{arm}/")

    for i, s in enumerate(queries, 1):
        if read_cache(arm, args.split, s):
            continue
        set_call_context(arm=arm, split=args.split, study=s)
        q = embed_image(client, s, frontal[s])
        scores = q @ index.T  # as shipped: np.dot, argmax (top-1, no threshold)
        top = int(np.argmax(scores))
        hit_study = index_ids[top]
        contents = [PROMPT.format(question=question(args.question)),
                    PIL.Image.open(IMAGES / f"{frontal[s]}.png"),
                    PIL.Image.open(IMAGES / f"{frontal[hit_study]}.png")]
        r = with_retry(lambda: client.models.generate_content(model=GEN_MODEL, contents=contents))
        u = r.usage_metadata
        write_cache(arm, args.split, s, {
            "arm": arm, "study": s, "image": frontal[s],
            "retrieved_study": hit_study, "retrieved_image": frontal[hit_study],
            "similarity": float(scores[top]),
            "model_requested": GEN_MODEL, "model_version": r.model_version,
            "embed_model": EMBED_MODEL,
            "prompt": contents[0], "question_version": args.question, "response_text": r.text,
            "usage": {"input": u.prompt_token_count, "output": u.candidates_token_count,
                      "thinking": u.thoughts_token_count, "total": u.total_token_count},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"{i}/{len(queries)} {s} -> {hit_study} ({scores[top]:.3f}) "
              f"tokens in/out/think {u.prompt_token_count}/{u.candidates_token_count}/{u.thoughts_token_count}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("--limit", type=int)
    b = sub.add_parser("run"); b.add_argument("split"); b.add_argument("--pilot", action="store_true")
    b.add_argument("--limit", type=int)
    b.add_argument("--smoke", action="store_true", help="allow a partial index; output to <arm>_smoke")
    b.add_argument("--question", choices=["original", "checklist"], default="checklist",
                   help="findings question (checklist: six checklist lines first; original: the first runs)")
    for sp in (a, b):
        sp.add_argument("--max-requests", type=int, help="stop after this many API attempts")
        sp.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD, help="stop once this run has cost this much (default 10)")
    args = ap.parse_args()
    set_request_budget(args.max_requests, args.max_usd)
    client = metered_client(load_api_key())  # counts + logs every call; carries the timeout
    (cmd_index if args.cmd == "index" else cmd_run)(args, client)


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"API attempts this run: {requests_made()}; cost this run: ${run_spend():.4f}", flush=True)
