"""Offline test of common.with_retry and the metered client with fake errors and a fake API
(no API calls, sleeps patched out).

Usage: python -I scripts/tests/test_retry.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

time.sleep = lambda s: None  # no real waiting


def failing(message, succeed_after=None):
    calls = {"n": 0}

    def fn():
        common.count_request()  # what the metered client does before every real call
        calls["n"] += 1
        if succeed_after is not None and calls["n"] > succeed_after:
            return "ok"
        raise RuntimeError(message)
    return fn, calls


GEN, EMB = "gemini-3-flash-preview", "gemini-embedding-2"


def reset(budget=None, max_usd=None):
    common._requests_made = 0
    common._run_spend = 0.0
    common.set_request_budget(budget, max_usd)


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    return cond


ok = True
DAILY = "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE = "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier"

reset(); fn, c = failing(DAILY)
try:
    common.with_retry(fn); ok &= check("daily quota stops", False)
except common.QuotaExhausted:
    ok &= check("daily quota stops immediately (1 attempt)", c["n"] == 1)

reset(); fn, c = failing(MINUTE)
try:
    common.with_retry(fn); ok &= check("per-minute quota", False)
except common.QuotaExhausted:
    ok &= check("per-minute 429: 2 retries, then stop (3 attempts)", c["n"] == 3)

reset(); fn, c = failing(MINUTE, succeed_after=1)
ok &= check("per-minute 429 that clears: succeeds on retry", common.with_retry(fn) == "ok" and c["n"] == 2)

reset(); fn, c = failing("503 UNAVAILABLE high demand", succeed_after=3)
ok &= check("503 retried until success", common.with_retry(fn) == "ok" and c["n"] == 4)

reset(); fn, c = failing("503 UNAVAILABLE high demand")
try:
    common.with_retry(fn); ok &= check("503 forever", False)
except RuntimeError:
    ok &= check("503 forever: 5 retries, then error (6 attempts)", c["n"] == 6)

reset(); fn, c = failing("ReadTimeout: timed out", succeed_after=1)
ok &= check("timeout retried", common.with_retry(fn) == "ok" and c["n"] == 2)

reset(); fn, c = failing("400 INVALID_ARGUMENT bad request")
try:
    common.with_retry(fn); ok &= check("non-retryable", False)
except RuntimeError:
    ok &= check("non-retryable error raised immediately (1 attempt)", c["n"] == 1)

reset(budget=3); fn, c = failing("503 UNAVAILABLE high demand")
try:
    common.with_retry(fn); ok &= check("budget", False)
except common.RequestBudgetReached:
    ok &= check("request budget stops after 3 calls", c["n"] == 3 and common.requests_made() == 3)

def one_call(v):
    common.count_request()
    return v


reset(budget=2)
common.with_retry(lambda: one_call("a")); common.with_retry(lambda: one_call("b"))
try:
    common.with_retry(lambda: one_call("c")); ok &= check("budget across calls", False)
except common.RequestBudgetReached:
    ok &= check("request budget counts across calls", common.requests_made() == 2)

# --- metered client, with a fake `models` object (no network) ---
import json  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace as NS  # noqa: E402

common.CALL_LOG = Path(tempfile.mkdtemp()) / "api_calls.jsonl"
usage = NS(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30, total_token_count=150)
fake = NS(generate_content=lambda **kw: NS(usage_metadata=usage, model_version="m-v",
                                          candidates=[NS(finish_reason=NS(name="STOP"))]),
          embed_content=lambda **kw: NS(embeddings=[]))
common._meter(fake)
reset(budget=3)
common.set_call_context(arm="t", study="CXR1")
fake.generate_content(model=GEN); fake.embed_content(model=EMB, contents=["a", "b"])
log = [json.loads(line) for line in common.CALL_LOG.read_text(encoding="utf-8").splitlines()]
ok &= check("metered calls are logged with context, usage and finish reason",
            len(log) == 2 and log[0]["study"] == "CXR1" and log[0]["usage"]["total"] == 150
            and log[0]["finish_reason"] == "STOP" and log[1]["items"] == 2)


def library_internal_retry():  # like agno's guidance retry: 2 real calls inside one attempt
    fake.generate_content(model=GEN)
    return fake.generate_content(model=GEN)


try:
    common.with_retry(library_internal_retry); ok &= check("internal retries counted", False)
except common.RequestBudgetReached:
    ok &= check("a library's internal retries count against the budget (stops at 3 calls)",
                common.requests_made() == 3)

def broken(**kw):
    raise RuntimeError("503 UNAVAILABLE")


fake2 = NS(generate_content=broken, embed_content=broken)
common._meter(fake2)
reset()
try:
    fake2.generate_content(model=GEN)
except RuntimeError:
    pass
last = json.loads(common.CALL_LOG.read_text(encoding="utf-8").splitlines()[-1])
ok &= check("failed calls are logged as not ok, without usage", last["ok"] is False and "usage" not in last)

# --- async path (Google ADK) and unmetered paths ---
import asyncio  # noqa: E402


async def fake_async_generate(**kw):
    return NS(usage_metadata=usage, model_version="m-v", candidates=[NS(finish_reason=NS(name="STOP"))])


aio = NS(generate_content=fake_async_generate, generate_content_stream=None, embed_content=None)
common._meter_async(aio)
reset(budget=1)
asyncio.run(aio.generate_content(model=GEN))
last = json.loads(common.CALL_LOG.read_text(encoding="utf-8").splitlines()[-1])
ok &= check("async generate_content is metered and logged", common.requests_made() == 1 and last["usage"]["total"] == 150)
try:
    asyncio.run(aio.generate_content(model=GEN)); ok &= check("async budget", False)
except common.RequestBudgetReached:
    ok &= check("async calls count against the budget", True)
blocked = 0
for fn in (aio.generate_content_stream, aio.embed_content, fake.generate_content_stream):
    try:
        fn(model=GEN)
    except RuntimeError:
        blocked += 1
ok &= check("unmetered paths (streaming, async embed) refuse to run", blocked == 3)
ok &= check("embed items split into images and text characters",
            common._embed_items(["abc", NS()]) == {"items": 2, "images": 1, "text_chars": 3})

# --- per-run dollar limit (decision 2026-10-18) ---
cost = common.call_cost({"ok": True, "method": "generate", "model": GEN,
                         "usage": {"input": 100, "total": 150}})
ok &= check("generation cost = input x $0.50/M + (total - input) x $3.00/M",
            abs(cost - (100 * 0.50 + 50 * 3.00) / 1e6) < 1e-15)
ok &= check("embedding cost: images per image, text per estimated token",
            abs(common.call_cost({"ok": True, "method": "embed", "model": EMB, "items": 2, "images": 1,
                                  "text_chars": 400}) - (0.00012 + 100 * 0.20 / 1e6)) < 1e-15)
ok &= check("failed calls cost nothing", common.call_cost({"ok": False, "method": "generate", "model": GEN}) == 0)

big = NS(prompt_token_count=1_000_000, candidates_token_count=0, thoughts_token_count=0,
         total_token_count=1_000_000)  # $0.50 per call
fake3 = NS(generate_content=lambda **kw: NS(usage_metadata=big, model_version="m-v",
                                           candidates=[NS(finish_reason=NS(name="STOP"))]),
           embed_content=lambda **kw: NS(embeddings=[]))
common._meter(fake3)
reset(max_usd=1.2)
made = 0
try:
    for _ in range(10):
        fake3.generate_content(model=GEN)
        made += 1
    ok &= check("spend limit", False)
except common.SpendLimitReached:
    # $0.50, $1.00 (both < 1.2: next call allowed), $1.50 (>= 1.2: stop before the 4th)
    ok &= check("spend limit stops the run before the next call, overshooting by at most one call",
                made == 3 and abs(common.run_spend() - 1.5) < 1e-9)

reset()
calls_before = common.requests_made()
lines_before = len(common.CALL_LOG.read_text(encoding="utf-8").splitlines())
try:
    fake3.generate_content(model="some-unpriced-model"); ok &= check("unpriced model", False)
except RuntimeError:
    ok &= check("a model without a price is refused before the call (not counted, not logged)",
                common.requests_made() == calls_before
                and len(common.CALL_LOG.read_text(encoding="utf-8").splitlines()) == lines_before)
ok &= check("default per-run limit is USD 10", common.DEFAULT_MAX_USD == 10.0)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
