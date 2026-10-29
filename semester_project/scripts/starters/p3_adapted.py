"""P3-adapted (and P3-orig with --shipped-cap): the `multimodal_agentic_rag` starter run as a
batch job on IU X-Ray.

The starter's own code is imported and called (rag_tutorials/multimodal_agentic_rag/backend):
rag_store.py (embedding, blending, scoring, retrieval packet), agentic_rag_agent/agent.py
(build_agent: the ADK agent, its instruction, tools and generation config) and app_state.py
(the RAG_STORE the agent's inspect tool reads).

Kept as shipped:
  - index entries as built by MultimodalRagStore.add_file_source: image embedding
    (gemini-embedding-2, 768 dimensions, inline bytes) blended 0.68 / 0.32 with the embedding of
    the annotation text "<title>. <notes>" (prefix "task: retrieval document"); title = the
    uploaded file's name, notes = the study's report;
  - the /ask path of server.py: retrieve top_k = 6 first, then run the agent with its
    retrieval tool replaced by one that returns that packet; the same user message text;
  - the agent (gemini-3-flash-preview, temperature 0.25, max 900 output tokens, both tools).
Adaptations (each recorded in results.md):
  1. Batch runner instead of the FastAPI server and React UI.
  2. Retrieval query = the patient's frontal-image embedding (the store's own image-embedding
     path) instead of the embedded text question. Scoring and ranking follow search() exactly
     (cosine, rounded to 4 decimals, best chunk per source, top 6); the 3-D PCA projection for
     the UI is skipped (it does not affect the matches).
  3. Index = train-split frontal images; notes = the study's FINDINGS + IMPRESSION. Built once
     and cached (data/runs/p3_orig/index/), then loaded into the store.
  4. The agent's model gets the project's metered client (same model ID): every call counted
     and logged (common.metered_client). No other change to the agent.
  5. P3-adapted only (default; deviation, results.md): the agent's max_output_tokens (900) is
     removed, i.e. the model's default, as for P1 and P2. With the checklist question the
     answering call thinks ~860 tokens and the cap (which includes thinking) cut 45 / 50 pilot
     checklists (diary 2026-10-18). `--shipped-cap` keeps 900 = P3-orig (folders p3_orig...).
The generator never sees the patient's image (as shipped): only the retrieved reports.
Leakage assertion: no query study is in the index. The index and query embeddings are shared
by both variants and stay in data/runs/p3_orig/.

Usage:
  python -I scripts/starters/p3_adapted.py index [--limit N] [--max-requests N]
  python -I scripts/starters/p3_adapted.py run <split> [--pilot] [--limit N] [--shipped-cap]
                                               [--question original|checklist] [--max-requests N]
"""
import argparse
import asyncio
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_MAX_USD, PROJECT, run_spend, arm_name, load_api_key, question, metered_client, read_cache, requests_made, set_call_context, set_request_budget, with_retry, write_cache  # noqa: E402

STARTER = PROJECT.parent / "rag_tutorials" / "multimodal_agentic_rag" / "backend"
os.environ["GOOGLE_API_KEY"] = load_api_key()  # app_state builds its store from this at import
sys.path.insert(0, str(STARTER))
import rag_store as rs  # noqa: E402
from app_state import RAG_STORE  # noqa: E402
from agentic_rag_agent.agent import build_agent  # noqa: E402
from google.adk.models import Gemini  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

ARM = "p3_orig"  # index, query embeddings and P3-orig answers; P3-adapted answers: "p3_adapted"
TOP_K = 6  # server.py AskRequest default
APP_NAME, USER_ID = "multimodal_agentic_rag", "demo-user"  # server.py
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"
REPORTS = PROJECT / "data" / "raw" / "iu_xray" / "reports" / "ecgen-radiology"
INDEX_DIR = PROJECT / "data" / "runs" / ARM / "index"
QUERY_DIR = PROJECT / "data" / "runs" / ARM / "query_embeddings"


def load_splits():
    with open(PROJECT / "data" / "processed" / "frontal.csv") as fh:
        frontal = {r["study"]: r["image"] for r in csv.DictReader(fh)}
    with open(PROJECT / "data" / "processed" / "splits.csv") as fh:
        splits = list(csv.DictReader(fh))
    return frontal, splits


def report_notes(study: str) -> str:
    """FINDINGS + IMPRESSION, the same report text CheXbert labelled, with section names."""
    root = ET.parse(REPORTS / f"{study[3:]}.xml").getroot()
    assert root.find("uId").get("id") == study
    sec = {a.get("Label"): (a.text or "").strip() for a in root.iter("AbstractText")}
    parts = [f"{k}: {sec[k]}" for k in ("FINDINGS", "IMPRESSION") if sec.get(k)]
    return " ".join(parts)


def embed_train_study(study: str, image: str) -> dict:
    """As in MultimodalRagStore.add_file_source, but vectors cached on disk."""
    path = INDEX_DIR / f"{study}.npz"
    title = f"{image}.png"  # the uploaded file's name
    display_text = rs._clean_text(report_notes(study)) or f"{title} (image/png) embedded natively in Gemini Embedding 2."
    if not path.exists():
        data = (IMAGES / f"{image}.png").read_bytes()
        media_vector, embedding_path = with_retry(lambda: RAG_STORE._embed_file(data, "image/png", title, display_text))
        annotation_text = rs._clean_text(f"{title}. {display_text}")
        annotation_vector = with_retry(lambda: RAG_STORE._embed_text(annotation_text, "task: retrieval document"))
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, media=np.asarray(media_vector, np.float64), annotation=np.asarray(annotation_vector, np.float64),
                 embedding_path=embedding_path)
    z = np.load(path)
    return {"title": title, "display_text": display_text, "media": z["media"].tolist(),
            "annotation": z["annotation"].tolist(), "embedding_path": str(z["embedding_path"])}


def cmd_index(args):
    frontal, splits = load_splits()
    train = [r["study"] for r in splits if r["split"] == "train"]
    if args.limit:
        train = train[:args.limit]
    for i, s in enumerate(train, 1):
        set_call_context(arm=ARM, split="train", study=s)
        embed_train_study(s, frontal[s])
        if i % 100 == 0 or i == len(train):
            print(f"indexed {i}/{len(train)}", flush=True)


def load_store(frontal, splits) -> list[str]:
    """Fill RAG_STORE with the cached train entries, as add_file_source would have."""
    train = [r["study"] for r in splits if r["split"] == "train"]
    missing = [s for s in train if not (INDEX_DIR / f"{s}.npz").exists()]
    if missing:
        sys.exit(f"Index incomplete ({len(train) - len(missing)}/{len(train)}); run `index` first.")
    RAG_STORE.sources, RAG_STORE.chunks = [], []
    for s in train:
        e = embed_train_study(s, frontal[s])
        vector = rs._blend_vectors(e["media"], e["annotation"])
        meta = {"mime_type": "image/png", "embedding_path": e["embedding_path"], "annotation_blended": True,
                "study": s}
        RAG_STORE.sources.append(rs.RackSource(id=s, title=e["title"], modality="image",
                                               summary=e["display_text"][:220], chunks=1, metadata=meta))
        RAG_STORE.chunks.append(rs.RackChunk(id=f"{s}-1", source_id=s, title=e["title"], modality="image",
                                             text=e["display_text"], vector=vector, metadata=meta))
    return train


def query_vector(study: str, image: str) -> list[float]:
    path = QUERY_DIR / f"{study}.npy"
    if not path.exists():
        data = (IMAGES / f"{image}.png").read_bytes()
        v, _ = with_retry(lambda: RAG_STORE._embed_file(data, "image/png", f"{image}.png", ""))
        QUERY_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(v, np.float64))
    return np.load(path).tolist()


def search_by_vector(qv: list[float], top_k: int) -> dict:
    """MultimodalRagStore.search() from 'source_by_id = ...' on, with a given query vector."""
    source_by_id = {source.id: source for source in RAG_STORE.sources}
    source_matches = {}
    for chunk in RAG_STORE.chunks:
        score = round(rs._cosine(qv, chunk.vector), 4)
        current = source_matches.get(chunk.source_id)
        if not current or score > current["score"]:
            source = source_by_id.get(chunk.source_id)
            if not source:
                continue
            source_matches[chunk.source_id] = {
                "id": source.id, "source_id": source.id, "title": source.title,
                "modality": source.modality, "text": chunk.text, "score": score,
                "metadata": {"best_chunk": chunk.id, **chunk.metadata},
            }
    matches = sorted(source_matches.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return {"matches": matches}


async def run_agent(packet: dict, q: str, shipped_cap: bool) -> tuple[str, list]:
    """server.py _run_adk_agent, with the metered client on the agent's model.
    A new client per study: its async connections belong to the event loop that created them,
    and each study runs in its own asyncio.run loop (reusing one failed: "Event loop is
    closed", diary 2026-10-17)."""
    client = metered_client(load_api_key())
    def retrieve_relevant_context(query: str, top_k: int = 6) -> dict:
        """Return the exact retrieval packet already embedded for this request."""
        return packet

    agent = build_agent(retrieve_relevant_context)
    agent.model = Gemini(model=agent.model, client=client)  # adaptation 4: same model ID
    assert agent.generate_content_config.max_output_tokens == 900  # as shipped
    if not shipped_cap:
        agent.generate_content_config.max_output_tokens = None  # adaptation 5 (P3-adapted)
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=f"Question: {q}\nUse the retrieval tool result for this exact question.")],
    )
    final_text, trace = "", []
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        parts = (event.content.parts if getattr(event, "content", None) and event.content.parts else [])
        for part in parts:
            if getattr(part, "function_call", None):
                trace.append({"author": event.author, "function_call": part.function_call.name,
                              "args": dict(part.function_call.args or {})})
            elif getattr(part, "function_response", None):
                trace.append({"author": event.author, "function_response": part.function_response.name})
        text = "".join(p.text for p in parts if getattr(p, "text", None))  # server.py _event_text
        if text:
            final_text = text
            trace.append({"author": event.author, "text_chars": len(text)})
    return final_text, trace


def cmd_run(args):
    frontal, splits = load_splits()
    queries = [r["study"] for r in splits if r["split"] == args.split and (not args.pilot or r["pilot"] == "1")]
    if args.limit:
        queries = queries[:args.limit]
    index_ids = load_store(frontal, splits)
    assert not set(queries) & set(index_ids), "leakage: query study in the retrieval index"
    arm = arm_name(ARM if args.shipped_cap else "p3_adapted", args.question)  # index / query vectors stay in ARM
    q = question(args.question)

    for i, s in enumerate(queries, 1):
        if read_cache(arm, args.split, s):
            continue
        set_call_context(arm=arm, split=args.split, study=s)
        results = search_by_vector(query_vector(s, frontal[s]), TOP_K)
        assert all(m["id"] in set(index_ids) for m in results["matches"])
        packet = RAG_STORE.retrieval_payload(results)
        answer, trace = with_retry(lambda: asyncio.run(run_agent(packet, q, args.shipped_cap)))
        write_cache(arm, args.split, s, {
            "arm": arm, "study": s, "image": frontal[s],
            "retrieved": [{"study": m["id"], "title": m["title"], "similarity": m["score"]} for m in results["matches"]],
            "retrieval_packet": packet,
            "model_requested": "gemini-3-flash-preview",
            "max_output_tokens": 900 if args.shipped_cap else None,
            "question": q, "question_version": args.question, "response_text": answer, "agent_trace": trace,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"{i}/{len(queries)} {s} top-1 {results['matches'][0]['id']} ({results['matches'][0]['score']}) "
              f"answer {len(answer)} chars, {sum('function_call' in t for t in trace)} tool calls", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("--limit", type=int)
    b = sub.add_parser("run"); b.add_argument("split"); b.add_argument("--pilot", action="store_true")
    b.add_argument("--limit", type=int)
    b.add_argument("--shipped-cap", action="store_true", help="P3-orig: keep the shipped 900-token output cap")
    b.add_argument("--question", choices=["original", "checklist"], default="checklist",
                   help="findings question (checklist: six checklist lines first; original: the first runs)")
    for sp in (a, b):
        sp.add_argument("--max-requests", type=int, help="stop after this many API calls")
        sp.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD, help="stop once this run has cost this much (default 10)")
    args = ap.parse_args()
    set_request_budget(args.max_requests, args.max_usd)
    RAG_STORE.client = metered_client(load_api_key())  # the store's embedding calls are metered too
    (cmd_index if args.cmd == "index" else cmd_run)(args)


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"API calls this run: {requests_made()}; cost this run: ${run_spend():.4f}", flush=True)
