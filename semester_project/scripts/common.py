"""Shared helpers for the API-based systems: API key loading, the fixed findings question,
retry/quota handling with a request budget, and response caching.

The key is read from semester_project/.env (line GOOGLE_API_KEY=...) and never printed.
"""
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FINDINGS = ["Lung Opacity", "Cardiomegaly", "Atelectasis", "Lung Lesion", "Pleural Effusion", "Fracture"]
# Same question for all starter arms (plan §3).
QUESTION = "Which of these findings are present: " + ", ".join(FINDINGS) + "?"
# Checklist question (diary 2026-10-17/18): asks for a findings checklist at the top of the
# answer, because CheXbert misread the free-text answers. The starters' own prompts are
# unchanged. (Called "question v2", folders "__q2", in the first runs; renamed 2026-10-18.)
QUESTION_CHECKLIST = (QUESTION + ' Begin your answer with exactly six lines, one per finding, in the form '
                      '"<finding>: present", "<finding>: absent" or "<finding>: uncertain". '
                      'Then continue as instructed.')


def question(version: str) -> str:
    return {"original": QUESTION, "checklist": QUESTION_CHECKLIST}[version]


def arm_name(base: str, version: str) -> str:
    """Original-question runs keep their folders; checklist runs go to <base>__checklist."""
    return base if version == "original" else f"{base}__checklist"


def load_api_key(name: str = "GOOGLE_API_KEY") -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = PROJECT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == name:
                os.environ[name] = value.strip().strip('"').strip("'")
                return os.environ[name]
    raise RuntimeError(f"{name} not found in environment or {env}")


REQUEST_TIMEOUT_S = 180  # per API request; without it a dropped connection hangs forever

# Error classes (matched against "<ExceptionType>: <message>"):
QUOTA = ("429", "RESOURCE_EXHAUSTED")                  # rate limit or quota
DAILY_QUOTA = ("PerDay", "per day")                    # a daily quota: no point waiting
TRANSIENT = ("500", "503", "504", "UNAVAILABLE", "DEADLINE", "timed out", "timeout", "Timeout",
             "ReadTimeout", "ConnectTimeout", "RemoteProtocolError")
MAX_QUOTA_RETRIES = 2      # a per-minute limit clears within a minute; more retries only burn quota
MAX_TRANSIENT_RETRIES = 5  # overload/timeouts: 60, 120, 240, 300, 300 s


class QuotaExhausted(SystemExit):
    """Raised to stop a run cleanly when an API quota is used up. Cached results are kept;
    rerunning the same command later continues where it stopped."""


class RequestBudgetReached(SystemExit):
    """Raised when a run reaches its --max-requests budget."""


class SpendLimitReached(SystemExit):
    """Raised when a run's own logged cost reaches its --max-usd limit."""


# Prices: paid tier, Standard, ai.google.dev/gemini-api/docs/pricing, checked 2026-10-17.
# Used by the per-run spend limit and by spend.py (the ledger), so both use one formula.
GEN_PRICES = {"gemini-3-flash-preview": (0.50, 3.00)}  # USD per 1M tokens: input, output incl. thinking
EMBED_PER_IMAGE = {"gemini-embedding-2": 0.00012}
EMBED_TEXT_PER_M = {"gemini-embedding-2": 0.20}  # USD per 1M text tokens
CHARS_PER_TOKEN = 4  # the embed response has no token count: estimate (tiny amounts)
DEFAULT_MAX_USD = 10.0  # per run (decision 2026-10-18); runners: --max-usd


def call_cost(entry: dict) -> float:
    """USD cost of one logged call. Failed calls carry no usage and are not billed.
    Billable output = total - input, at the output rate (thinking is billed as output)."""
    if not entry["ok"]:
        return 0.0
    if entry["method"] == "generate":
        p_in, p_out = GEN_PRICES[entry["model"]]
        u = entry["usage"]
        return (u["input"] * p_in + (u["total"] - u["input"]) * p_out) / 1e6
    images = entry.get("images", entry["items"])  # older entries: images only (P2)
    return (images * EMBED_PER_IMAGE[entry["model"]]
            + entry.get("text_chars", 0) / CHARS_PER_TOKEN * EMBED_TEXT_PER_M[entry["model"]] / 1e6)


_requests_made = 0
_max_requests = None
_run_spend = 0.0
_max_usd = DEFAULT_MAX_USD


def set_request_budget(max_requests, max_usd=DEFAULT_MAX_USD):
    global _max_requests, _max_usd
    _max_requests, _max_usd = max_requests, max_usd


def requests_made() -> int:
    return _requests_made


def run_spend() -> float:
    return _run_spend


def count_request(model=None) -> None:
    """Check the run's limits, then count one real API call; called by the metered client
    before every HTTP request (so a library's internal retries count too). Failed calls count
    as well, since they can count toward the provider's quota. The spend limit uses the cost of
    the calls already made, so a run can overshoot it by at most one call."""
    global _requests_made
    if _max_requests is not None and _requests_made >= _max_requests:
        raise RequestBudgetReached(
            f"Request budget reached ({_requests_made} API calls). Results so far are cached.")
    if _max_usd is not None and _run_spend >= _max_usd:
        raise SpendLimitReached(
            f"Spend limit reached: this run cost ${_run_spend:.4f} (limit ${_max_usd:.2f}). "
            "Results so far are cached; rerun to continue.")
    if model is not None and model not in GEN_PRICES and model not in EMBED_PER_IMAGE:
        raise RuntimeError(f"no price for model {model!r}; add it to common.py before calling it")
    _requests_made += 1


# --- Metered client: every real API call is counted and logged with its token usage ---
# Reason (diary 2026-10-17): agno retries a malformed function call internally with a guidance
# message, so one runner-level attempt can be several billed calls, and calls that end in an
# error are never cached. The log, not the cache, is therefore the spend ledger.
CALL_LOG = PROJECT / "data" / "runs" / "logs" / "api_calls.jsonl"
_call_context = {}


def set_call_context(**kw) -> None:
    """Tag the following API calls in the log (e.g. arm=..., split=..., study=...)."""
    _call_context.clear()
    _call_context.update(kw)


def _log_call(entry: dict) -> None:
    import time
    global _run_spend
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **_call_context, **entry}
    with open(CALL_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _run_spend += call_cost(entry)


def metered_client(api_key: str):
    """A google-genai Client whose generate_content / embed_content calls are counted against
    the request budget and logged to CALL_LOG. Pass it to agno with Gemini(client=...)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
    _meter(client.models)
    _meter_async(client.aio.models)  # Google ADK (P3) calls the async API
    return client


def _log_generate(model, r=None, error=None) -> None:
    if error is not None:
        _log_call({"method": "generate", "model": model, "ok": False,
                   "error": f"{type(error).__name__}: {str(error)[:200]}"})
        return
    u = r.usage_metadata
    fin = r.candidates[0].finish_reason if r.candidates else None
    _log_call({"method": "generate", "model": model, "ok": True,
               "model_version": r.model_version,
               "finish_reason": getattr(fin, "name", str(fin)) if fin is not None else None,
               "usage": {"input": u.prompt_token_count, "output": u.candidates_token_count,
                         "thinking": u.thoughts_token_count, "total": u.total_token_count}})


def _embed_items(contents) -> dict:
    """Images are billed per image, text per token: record which (text by character count)."""
    items = contents if isinstance(contents, list) else [contents]
    texts = [c for c in items if isinstance(c, str)]
    return {"items": len(items), "images": len(items) - len(texts),
            "text_chars": sum(len(t) for t in texts)}


def _unmetered(name):
    def refuse(*args, **kwargs):
        raise RuntimeError(f"{name} is not metered; add metering before using it")
    return refuse


def _meter(models) -> None:
    """Wrap a client's sync `models` object in place (separate for the offline test)."""
    gen, emb = models.generate_content, models.embed_content

    def generate_content(*args, **kwargs):
        count_request(kwargs.get("model"))
        try:
            r = gen(*args, **kwargs)
        except Exception as e:
            _log_generate(kwargs.get("model"), error=e)
            raise
        _log_generate(kwargs.get("model"), r)
        return r

    def embed_content(*args, **kwargs):
        count_request(kwargs.get("model"))
        items = _embed_items(kwargs.get("contents"))
        try:
            r = emb(*args, **kwargs)
        except Exception as e:
            _log_call({"method": "embed", "model": kwargs.get("model"), "ok": False, **items,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"})
            raise
        _log_call({"method": "embed", "model": kwargs.get("model"), "ok": True, **items})
        return r

    models.generate_content, models.embed_content = generate_content, embed_content
    models.generate_content_stream = _unmetered("generate_content_stream")


def _meter_async(models) -> None:
    """Wrap a client's async `aio.models` object in place: generate_content only."""
    gen = models.generate_content

    async def generate_content(*args, **kwargs):
        count_request(kwargs.get("model"))
        try:
            r = await gen(*args, **kwargs)
        except Exception as e:
            _log_generate(kwargs.get("model"), error=e)
            raise
        _log_generate(kwargs.get("model"), r)
        return r

    models.generate_content = generate_content
    models.generate_content_stream = _unmetered("aio generate_content_stream")
    models.embed_content = _unmetered("aio embed_content")


def with_retry(fn):
    """Call fn (one API request, made through the metered client). Retry transient errors with
    backoff; retry a 429 at most twice (per-minute limits); stop the whole run on a daily quota
    or on repeated 429s. The request budget is enforced per real API call by the metered client
    (RequestBudgetReached is a SystemExit, so it passes through here)."""
    import time
    quota_retries = transient_retries = 0
    while True:
        try:
            return fn()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if any(c in msg for c in QUOTA):
                if any(c in msg for c in DAILY_QUOTA):
                    raise QuotaExhausted(f"Daily API quota exhausted; stopping. Rerun later to continue.\n{msg[:400]}")
                if quota_retries >= MAX_QUOTA_RETRIES:
                    raise QuotaExhausted(f"API quota still exhausted after {quota_retries} retries; stopping.\n{msg[:400]}")
                quota_retries += 1
                wait = 65
            elif any(c in msg for c in TRANSIENT):
                if transient_retries >= MAX_TRANSIENT_RETRIES:
                    raise
                wait = min(60 * 2 ** transient_retries, 300)
                transient_retries += 1
            else:
                raise
            print(f"  retry in {wait}s after: {msg[:120]}", flush=True)
            time.sleep(wait)


def cache_path(arm: str, split: str, study: str) -> Path:
    return PROJECT / "data" / "runs" / arm / split / f"{study}.json"


def read_cache(arm: str, split: str, study: str):
    p = cache_path(arm, split, study)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_cache(arm: str, split: str, study: str, record: dict) -> None:
    p = cache_path(arm, split, study)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
