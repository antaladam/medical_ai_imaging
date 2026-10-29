"""P1-adapted: the `ai_medical_imaging_agent` starter run as a batch job on IU X-Ray.

Kept as shipped (starter_ai_agents/ai_medical_imaging_agent/ai_medical_imaging.py):
  - agno Agent with a Gemini model, markdown=True
  - the 5-section prompt, verbatim (including section 5's instruction to use DuckDuckGo)
  - image preparation: resize to width 500 keeping the aspect ratio, saved as PNG, passed as an
    agno Image from a file path
Adaptations (each recorded in results.md):
  1. Batch runner instead of the Streamlit UI. Each study gets its own temporary file (the
     shipped app reused one shared "temp_resized_image.png"; that only matters for concurrent
     users, not for the output).
  2. DuckDuckGo web search OFF in the main run (plan §3: reproducibility). `--web-search`
     turns it on for the planned 50-study sensitivity run (separate arm folder).
  3. The fixed findings question is appended to the prompt (P1 has no question input).
  4. Model: gemini-3-flash-preview instead of the retired gemini-2.5-pro (Pro -> Flash).
Library: agno 3.x (the starter requires agno>=2.2.10).

Arm P1-adapted-ns (`--no-search-section`, folder p1_adapted_ns; deviation, results.md):
  5. Section 5 ("Research Context", whose only purpose is the DuckDuckGo search) is removed from
     the prompt; sections 1-4 and the closing format line stay verbatim. Reason: with search off
     (adaptation 2), the instruction to use the missing tool made 62% of P1-adapted's pilot
     studies end in MALFORMED_FUNCTION_CALL (diary 2026-10-17). Everything else is identical.
P1-adapted (verbatim prompt) is kept as evidence of that failure and is not run on test.

Usage: python -I scripts/starters/p1_adapted.py <split> [--pilot] [--limit N] [--web-search]
                                                [--no-search-section] [--max-requests N]
"""
import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEFAULT_MAX_USD, PROJECT, run_spend, QUESTION, arm_name, load_api_key, question, metered_client, read_cache, requests_made, set_call_context, set_request_budget, with_retry, write_cache  # noqa: E402
from agno.agent import Agent  # noqa: E402
from agno.media import Image as AgnoImage  # noqa: E402
from agno.models.google import Gemini  # noqa: E402

GEN_MODEL = "gemini-3-flash-preview"
IMAGES = PROJECT / "data" / "raw" / "iu_xray" / "images"

# --- verbatim from ai_medical_imaging.py lines 58-94 ---
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""
# --- end verbatim ---

QUERY = query + "\n" + QUESTION + "\n"  # adaptation 3: findings question appended

# Adaptation 5 (arm P1-adapted-ns): the whole of section 5, exactly as it appears above.
SECTION_5 = """### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

"""
assert query.count(SECTION_5) == 1
QUERY_NS = query.replace(SECTION_5, "") + "\n" + QUESTION + "\n"




def run_checked(agent: Agent, path: Path, prompt: str):
    """agno does not raise on API errors: it returns a run with status ERROR and the error JSON
    as its content. Raise instead, so with_retry retries and nothing invalid is cached.
    Exception: a malformed function call is the model's own behaviour (the prompt orders a web
    search but no tool is available, diary 2026-10-17). It is returned as an outcome and cached,
    like a refusal (plan §3), after agno's own single guidance retry."""
    run = agent.run(prompt, images=[AgnoImage(filepath=path)])
    status = getattr(run.status, "value", run.status)
    if status == "ERROR" and MALFORMED in str(run.content):
        return run
    if status != "COMPLETED" or not (run.metrics and run.metrics.output_tokens):
        raise RuntimeError(f"agno run failed (status {status}): {str(run.content)[:300]}")
    return run


MALFORMED = "MALFORMED_FUNCTION_CALL"


def make_agent(api_key: str, web_search: bool) -> Agent:
    tools = []
    if web_search:
        from agno.tools.duckduckgo import DuckDuckGoTools
        tools = [DuckDuckGoTools()]
    # metered client (common.py): counts and logs every real call, incl. agno's internal
    # guidance retries; it also carries the per-request timeout.
    return Agent(model=Gemini(id=GEN_MODEL, client=metered_client(api_key)),
                 tools=tools, markdown=True)


def resized_png(image: str, tmpdir: Path) -> Path:
    # as shipped: width 500, height from aspect ratio, PIL default resample, saved as PNG
    img = PILImage.open(IMAGES / f"{image}.png")
    width, height = img.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized = img.resize((new_width, new_height))
    path = tmpdir / f"{image}_resized.png"
    resized.save(path)
    return path


def message_log(run) -> list:
    out = []
    for m in run.messages or []:
        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append({"role": m.role, "content": content,
                    "images": len(m.images or []) if hasattr(m, "images") else 0,
                    "tool_calls": m.tool_calls if getattr(m, "tool_calls", None) else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--web-search", action="store_true")
    ap.add_argument("--no-search-section", action="store_true", help="arm P1-adapted-ns (prompt without section 5)")
    ap.add_argument("--question", choices=["original", "checklist"], default="checklist",
                    help="findings question (checklist: six checklist lines first; original: the first runs)")
    ap.add_argument("--max-requests", type=int, help="stop after this many API attempts")
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD, help="stop once this run has cost this much (default 10)")
    args = ap.parse_args()
    set_request_budget(args.max_requests, args.max_usd)
    if args.web_search and args.no_search_section:
        sys.exit("--web-search and --no-search-section are different arms; choose one")
    arm = arm_name("p1_adapted_web" if args.web_search else "p1_adapted_ns" if args.no_search_section
                   else "p1_adapted", args.question)
    base = query.replace(SECTION_5, "") if args.no_search_section else query
    prompt = base + "\n" + question(args.question) + "\n"  # adaptation 3: the question appended
    if args.question == "original":  # the original runs' prompts, unchanged
        assert prompt == (QUERY_NS if args.no_search_section else QUERY)

    with open(PROJECT / "data" / "processed" / "frontal.csv") as fh:
        frontal = {r["study"]: r["image"] for r in csv.DictReader(fh)}
    with open(PROJECT / "data" / "processed" / "splits.csv") as fh:
        queries = [r["study"] for r in csv.DictReader(fh)
                   if r["split"] == args.split and (not args.pilot or r["pilot"] == "1")]
    if args.limit:
        queries = queries[:args.limit]

    agent = make_agent(load_api_key(), args.web_search)
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(queries, 1):
            if read_cache(arm, args.split, s):
                continue
            path = resized_png(frontal[s], Path(tmp))
            set_call_context(arm=arm, split=args.split, study=s)
            run = with_retry(lambda: run_checked(agent, path, prompt))
            m = run.metrics
            failed = getattr(run.status, "value", run.status) == "ERROR"
            write_cache(arm, args.split, s, {
                "arm": arm, "study": s, "image": frontal[s],
                "outcome": "malformed_function_call" if failed else "answer",
                "model_requested": GEN_MODEL, "model_reported": getattr(run, "model", None),
                "web_search": args.web_search,
                "prompt_variant": "no_section_5" if args.no_search_section else "verbatim",
                "question_version": args.question,
                "response_text": run.content,
                "messages": message_log(run),
                "usage": {"input": getattr(m, "input_tokens", None), "output": getattr(m, "output_tokens", None),
                          "thinking": getattr(m, "reasoning_tokens", None), "total": getattr(m, "total_tokens", None)},
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            print(f"{i}/{len(queries)} {s} {'MALFORMED_FUNCTION_CALL' if failed else 'answer'} tokens in/out/think "
                  f"{getattr(m, 'input_tokens', None)}/{getattr(m, 'output_tokens', None)}/{getattr(m, 'reasoning_tokens', None)}",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"API attempts this run: {requests_made()}; cost this run: ${run_spend():.4f}", flush=True)
