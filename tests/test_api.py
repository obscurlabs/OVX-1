"""FastAPI helpers and the bounded warm-demo benchmark."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from voicerag import api
from voicerag.api import percentile_summary
from voicerag.contracts import QueryRequest


def test_percentile_summary_uses_the_required_task_metrics():
    assert percentile_summary([1.0, 2.0, 3.0, 4.0, 5.0]) == {
        "avg": 3.0,
        "p50": 3.0,
        "p70": 3.8,
        "p100": 5.0,
    }


def test_percentile_summary_handles_an_empty_measurement():
    assert percentile_summary([]) == {"avg": 0.0, "p50": 0.0, "p70": 0.0, "p100": 0.0}


class FakePipeline:
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []

    def answer(self, request: QueryRequest) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace()


def test_warm_benchmark_reports_required_metrics_without_llm(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setitem(api.state, "ready", True)
    monkeypatch.setitem(api.state, "pipeline", pipeline)

    report = api.benchmark()

    assert {"avg", "p50", "p70", "p100", "count", "mode"} <= set(report)
    assert report["count"] == 30
    assert report["mode"] == "warm-cache demo"
    assert all(request.fast_only for request in pipeline.requests)
    # 15 warm-ups + 30 timed queries, each with the fixed bilingual suite.
    assert len(pipeline.requests) == 45


def test_only_one_warm_benchmark_may_run_at_once(monkeypatch):
    monkeypatch.setitem(api.state, "ready", True)
    monkeypatch.setitem(api.state, "pipeline", FakePipeline())
    assert api._benchmark_lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException, match="already running"):
            api.benchmark()
    finally:
        api._benchmark_lock.release()
