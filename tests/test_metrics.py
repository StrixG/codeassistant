"""Metrics: cost formula and JSONL summary aggregation."""

from __future__ import annotations

from pathlib import Path

from assistant.core import metrics


def test_cost_formula_matches_pricing():
    # 1000 cached-in, 1000 miss-in, 100 out.
    cost = metrics.request_cost_usd(prompt_tokens=2000, cached_tokens=1000, completion_tokens=100)
    expected = (
        1000 * 0.003625 / 1e6
        + 1000 * 0.435 / 1e6
        + 100 * 0.87 / 1e6
    )
    assert abs(cost - expected) < 1e-12


def test_cache_hit_is_cheaper_than_miss():
    all_hit = metrics.request_cost_usd(1000, 1000, 0)
    all_miss = metrics.request_cost_usd(1000, 0, 0)
    assert all_miss > all_hit * 100  # ~120x cheaper per spec


def test_summary_percentiles(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    for lat in (100, 200, 300, 400, 500):
        metrics.record(p, {"latency_ms": lat, "prompt_tokens": 10,
                           "completion_tokens": 1, "cached_tokens": 0})
    s = metrics.summarize(p)
    assert s["count"] == 5
    assert s["latency_p50_ms"] == 300
    assert s["latency_p95_ms"] >= 400


def test_empty_metrics(tmp_path: Path):
    assert metrics.summarize(tmp_path / "none.jsonl") == {"count": 0}
