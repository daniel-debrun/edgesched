from __future__ import annotations

import json
import math

import pytest

from edgesched.metrics import DispatchRecord, Trace, percentile, prometheus_text, summarize
from edgesched.model import Job


def _job(stream, release, deadline, started=None, finished=None, batch=1, dropped=False):
    j = Job(stream, "m", release, deadline)
    j.started, j.finished, j.batch_size, j.dropped = started, finished, batch, dropped
    return j


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_summary_counts_misses_drops_and_censoring():
    jobs = [
        _job("a", 0.0, 0.010, 0.000, 0.004, batch=2),  # hit, response 4 ms
        _job("a", 0.010, 0.020, 0.012, 0.022, batch=2),  # late by 2 ms
        _job("a", 0.020, 0.030, dropped=True),  # dropped
        _job("b", 0.050, 0.090, 0.050, 0.060),  # hit, response 10 ms
        _job("b", 0.080, 0.095),  # unfinished, deadline passed -> miss
        _job("b", 0.095, 0.200),  # unfinished, deadline beyond horizon -> censored
    ]
    trace = Trace(
        jobs=jobs,
        dispatches=[
            DispatchRecord("m", 0.0, 0.004, 2),
            DispatchRecord("m", 0.012, 0.022, 2),
            DispatchRecord("m", 0.050, 0.060, 1),
        ],
        start=0.0,
        end=0.100,
    )
    s = summarize(trace, "edf")
    a, b = s.streams
    assert (a.released, a.completed, a.missed, a.dropped) == (3, 2, 2, 1)
    assert (b.released, b.completed, b.missed) == (2, 1, 1)
    assert a.p50_ms == pytest.approx(8.0)
    assert a.max_lateness_ms == pytest.approx(2.0)
    assert a.mean_batch == pytest.approx(2.0)
    assert s.overall.released == 5
    assert s.overall.miss_rate == pytest.approx(3 / 5)
    assert s.utilization == pytest.approx(0.024 / 0.100)
    assert s.mean_batch_size == pytest.approx(5 / 3)
    assert s.throughput_rps == pytest.approx(30.0)


def test_exports_are_well_formed():
    trace = Trace([_job("cam", 0.0, 0.01, 0.0, 0.005)], [DispatchRecord("m", 0, 0.005, 1)], 0, 1)
    s = summarize(trace, "fifo")
    data = json.loads(s.to_json())
    assert data["overall"]["released"] == 1
    assert "cam" in s.table() and "ALL" in s.table()
    prom = prometheus_text(s)
    assert "# TYPE edgesched_deadline_miss_ratio gauge" in prom
    assert 'edgesched_jobs_released_total{policy="fifo",stream="cam"} 1.0' in prom
    for line in prom.strip().splitlines():
        assert line.startswith("#") or line.startswith("edgesched_")
