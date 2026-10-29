"""API spend ledger: the cost of every billed API call, from the metered call log.

Every real API call goes through common.metered_client, which appends it with its token usage to
data/runs/logs/api_calls.jsonl, including calls that end in an error and calls a library makes
internally (agno's guidance retries). Failed HTTP calls (429 / 5xx) carry no usage and are not
billed.

Prices: paid tier, Standard (not Batch), checked on ai.google.dev/gemini-api/docs/pricing on
2026-10-17. Billable output = total - input, at the output rate (thinking tokens are billed as
output). Text embeddings: tokens estimated as characters / 4. The project's monthly cap is USD 45 (set on the Google account). This ledger is computed
from the API's own token counts; the Google billing console remains the authority.

Usage: python -I scripts/tools/spend.py [--since YYYY-MM-DD]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py, metrics.py
from common import EMBED_PER_IMAGE, GEN_PRICES, call_cost  # noqa: E402  (one price table)

from common import PROJECT  # noqa: E402  (the project path is computed only in common.py)
CALL_LOG = PROJECT / "data" / "runs" / "logs" / "api_calls.jsonl"
CAP_USD = 45.0
# Billed calls made before the call log existed (paid tier started 2026-10-17):
UNLOGGED = [("2026-10-17", "p1_adapted / val (2 calls, MALFORMED_FUNCTION_CALL; = the logged CXR8 repeat: 2,986 input tokens)", 2, 0.0015)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-17")
    since = ap.parse_args().since

    calls, cost, failed = defaultdict(int), defaultdict(float), 0
    if CALL_LOG.exists():
        for line in CALL_LOG.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["time"] < since:
                continue
            if not e["ok"]:
                failed += 1
                continue
            key = f"{e.get('arm', '?')} / {e.get('split', '?')} / {e['method']}"
            if e["model"] not in GEN_PRICES and e["model"] not in EMBED_PER_IMAGE:
                sys.exit(f"no price for model {e['model']!r}; add it to common.py")
            cost[key] += call_cost(e)
            calls[key] += 1
    for day, key, n, usd in UNLOGGED:
        if day >= since:
            calls[key + " [unlogged]"] += n
            cost[key + " [unlogged]"] += usd

    print(f"Spend since {since}:\n")
    print(f"{'arm / split / method':58s} {'calls':>6s} {'USD':>8s}")
    for k in sorted(calls):
        print(f"{k:58s} {calls[k]:6d} {cost[k]:8.4f}")
    total = sum(cost.values())
    print(f"{'TOTAL':58s} {sum(calls.values()):6d} {total:8.4f}   "
          f"({100 * total / CAP_USD:.2f}% of the USD {CAP_USD:.0f} monthly cap)")
    print(f"Failed calls (not billed): {failed}")


if __name__ == "__main__":
    main()
