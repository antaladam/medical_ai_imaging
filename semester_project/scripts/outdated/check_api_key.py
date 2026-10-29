"""Check the Gemini API key: does it work, and can it reach the models the starters use?

Sends one tiny text request per generation model and one tiny embedding request. Prints only
model names, success/failure and error messages; never the key.

Usage: python -I scripts/outdated/check_api_key.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import load_api_key  # noqa: E402
from google import genai  # noqa: E402

GEN_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"]
EMBED_MODEL = "gemini-embedding-2"


def main() -> None:
    client = genai.Client(api_key=load_api_key())
    listed = {m.name.removeprefix("models/") for m in client.models.list()}
    for m in GEN_MODELS + [EMBED_MODEL]:
        print(f"{m:26s} listed: {m in listed}")
    for m in GEN_MODELS:
        try:
            r = client.models.generate_content(model=m, contents="Reply with the single word OK.")
            print(f"{m:26s} generate: OK ({(r.text or '').strip()[:10]!r}, served by {r.model_version})")
        except Exception as e:
            print(f"{m:26s} generate: FAILED {type(e).__name__}: {str(e)[:200]}")
    try:
        r = client.models.embed_content(model=EMBED_MODEL, contents="test")
        print(f"{EMBED_MODEL:26s} embed: OK (dim {len(r.embeddings[0].values)})")
    except Exception as e:
        print(f"{EMBED_MODEL:26s} embed: FAILED {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
