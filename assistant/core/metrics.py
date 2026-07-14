"""Per-request metrics: append to JSONL, summarize with cost.

deepseek-v4-pro pricing (USD per 1M tokens):
  cache-hit input  : $0.003625
  cache-miss input : $0.435
  output           : $0.870
Cache-hit input is ~120x cheaper, which is why the system prompt and static
context are kept as a stable message prefix (DeepSeek caches it).
"""

from __future__ import annotations

import json
from pathlib import Path

PRICE_CACHE_HIT_IN = 0.003625 / 1_000_000
PRICE_CACHE_MISS_IN = 0.435 / 1_000_000
PRICE_OUT = 0.87 / 1_000_000


def request_cost_usd(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    miss = max(0, prompt_tokens - cached_tokens)
    return (
        cached_tokens * PRICE_CACHE_HIT_IN
        + miss * PRICE_CACHE_MISS_IN
        + completion_tokens * PRICE_OUT
    )


def record(metrics_path: Path, entry: dict) -> None:
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def load(metrics_path: Path) -> list[dict]:
    if not metrics_path.is_file():
        return []
    rows: list[dict] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize(metrics_path: Path) -> dict:
    rows = load(metrics_path)
    n = len(rows)
    if n == 0:
        return {"count": 0}
    lat = [float(r.get("latency_ms", 0)) for r in rows]
    pt = [int(r.get("prompt_tokens", 0)) for r in rows]
    ct = [int(r.get("completion_tokens", 0)) for r in rows]
    cached = [int(r.get("cached_tokens", 0)) for r in rows]
    costs = [
        request_cost_usd(
            int(r.get("prompt_tokens", 0)),
            int(r.get("cached_tokens", 0)),
            int(r.get("completion_tokens", 0)),
        )
        for r in rows
    ]
    return {
        "count": n,
        "latency_p50_ms": round(_percentile(lat, 0.50)),
        "latency_p95_ms": round(_percentile(lat, 0.95)),
        "avg_prompt_tokens": round(sum(pt) / n),
        "avg_completion_tokens": round(sum(ct) / n),
        "avg_cached_tokens": round(sum(cached) / n),
        "avg_cost_usd": sum(costs) / n,
        "total_cost_usd": sum(costs),
    }
