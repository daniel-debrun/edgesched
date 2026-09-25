"""Deadline and latency metrics computed from a job trace.

The simulator and the runtime both produce a :class:`Trace`, so every number
in a report is computed by the same code regardless of where it came from.

Accounting rules:

* A job *misses* if it was dropped, finished after its deadline, or was still
  unfinished at the end of the horizon with its deadline already passed.
* Jobs unfinished at the end whose deadline lies beyond the horizon are
  censored (excluded from all counts).
* Response time and lateness are computed over completed jobs only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from edgesched.model import Job


@dataclass(frozen=True)
class DispatchRecord:
    model: str
    start: float
    end: float
    batch_size: int


@dataclass
class Trace:
    jobs: list[Job] = field(default_factory=list)
    dispatches: list[DispatchRecord] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]. NaN for empty input."""
    if not values:
        return math.nan
    s = sorted(values)
    pos = (q / 100.0) * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class StreamStats:
    name: str
    released: int
    completed: int
    missed: int
    dropped: int
    miss_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_lateness_ms: float
    p99_lateness_ms: float
    mean_batch: float


@dataclass
class Summary:
    policy: str
    horizon_s: float
    overall: StreamStats
    streams: list[StreamStats]
    utilization: float
    throughput_rps: float
    dispatches: int
    mean_batch_size: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def table(self) -> str:
        return format_table(self)


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _stats(name: str, jobs: Iterable[Job], end: float) -> StreamStats:
    released = completed = missed = dropped = 0
    responses: list[float] = []
    lateness: list[float] = []
    batch_sum = 0
    for j in jobs:
        if j.finished is None and not j.dropped:
            if j.deadline > end:
                continue
            released += 1
            missed += 1
            continue
        released += 1
        if j.dropped:
            dropped += 1
            missed += 1
            continue
        assert j.finished is not None
        completed += 1
        responses.append(j.finished - j.release)
        lateness.append(j.finished - j.deadline)
        batch_sum += j.batch_size
        if j.finished > j.deadline:
            missed += 1
    ms = 1e3
    return StreamStats(
        name=name,
        released=released,
        completed=completed,
        missed=missed,
        dropped=dropped,
        miss_rate=missed / released if released else math.nan,
        p50_ms=percentile(responses, 50) * ms,
        p95_ms=percentile(responses, 95) * ms,
        p99_ms=percentile(responses, 99) * ms,
        max_lateness_ms=max(lateness) * ms if lateness else math.nan,
        p99_lateness_ms=percentile(lateness, 99) * ms,
        mean_batch=batch_sum / completed if completed else math.nan,
    )


def summarize(trace: Trace, policy: str = "") -> Summary:
    horizon = max(trace.end - trace.start, 1e-12)
    by_stream: dict[str, list[Job]] = {}
    for j in trace.jobs:
        by_stream.setdefault(j.stream, []).append(j)
    streams = [_stats(n, by_stream[n], trace.end) for n in sorted(by_stream)]
    overall = _stats("ALL", trace.jobs, trace.end)
    busy = sum(
        max(0.0, min(d.end, trace.end) - max(d.start, trace.start)) for d in trace.dispatches
    )
    n_disp = len(trace.dispatches)
    batched = sum(d.batch_size for d in trace.dispatches)
    return Summary(
        policy=policy,
        horizon_s=horizon,
        overall=overall,
        streams=streams,
        utilization=busy / horizon,
        throughput_rps=overall.completed / horizon,
        dispatches=n_disp,
        mean_batch_size=batched / n_disp if n_disp else math.nan,
    )


def _fmt(v: float, spec: str) -> str:
    return "-" if isinstance(v, float) and math.isnan(v) else format(v, spec)


def format_table(summary: Summary) -> str:
    """Fixed-width per-stream table followed by device-level numbers."""
    header = (
        f"{'stream':<18}{'released':>9}{'missed':>8}{'miss%':>8}"
        f"{'p50ms':>9}{'p95ms':>9}{'p99ms':>9}{'maxlate':>9}{'batch':>7}"
    )
    lines = [f"policy: {summary.policy}", header, "-" * len(header)]
    for s in [*summary.streams, summary.overall]:
        if s is summary.overall:
            lines.append("-" * len(header))
        lines.append(
            f"{s.name:<18}{s.released:>9}{s.missed:>8}{_fmt(100 * s.miss_rate, '.2f'):>8}"
            f"{_fmt(s.p50_ms, '.2f'):>9}{_fmt(s.p95_ms, '.2f'):>9}{_fmt(s.p99_ms, '.2f'):>9}"
            f"{_fmt(s.max_lateness_ms, '.1f'):>9}{_fmt(s.mean_batch, '.2f'):>7}"
        )
    lines.append(
        f"utilization {summary.utilization:.3f}  throughput {summary.throughput_rps:.1f} req/s  "
        f"dispatches {summary.dispatches}  mean batch {_fmt(summary.mean_batch_size, '.2f')}"
    )
    return "\n".join(lines)


def prometheus_text(summary: Summary, prefix: str = "edgesched") -> str:
    """Render a summary in the Prometheus text exposition format (no client library needed)."""
    out: list[str] = []

    def metric(name: str, kind: str, help_: str, samples: list[tuple[dict[str, str], float]]):
        full = f"{prefix}_{name}"
        out.append(f"# HELP {full} {help_}")
        out.append(f"# TYPE {full} {kind}")
        for labels, value in samples:
            lab = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
            val = "NaN" if math.isnan(value) else repr(float(value))
            out.append(f"{full}{{{lab}}} {val}" if lab else f"{full} {val}")

    pol = {"policy": summary.policy}
    streams = summary.streams
    metric(
        "jobs_released_total",
        "counter",
        "Released jobs.",
        [({**pol, "stream": s.name}, s.released) for s in streams],
    )
    metric(
        "jobs_missed_total",
        "counter",
        "Jobs that missed their deadline (incl. dropped).",
        [({**pol, "stream": s.name}, s.missed) for s in streams],
    )
    metric(
        "deadline_miss_ratio",
        "gauge",
        "Missed / released.",
        [({**pol, "stream": s.name}, s.miss_rate) for s in streams],
    )
    metric(
        "response_time_seconds",
        "gauge",
        "Response time percentiles.",
        [
            ({**pol, "stream": s.name, "quantile": q}, v / 1e3)
            for s in streams
            for q, v in (("0.5", s.p50_ms), ("0.95", s.p95_ms), ("0.99", s.p99_ms))
        ],
    )
    metric(
        "device_utilization_ratio", "gauge", "Busy time / horizon.", [(pol, summary.utilization)]
    )
    metric("mean_batch_size", "gauge", "Jobs per dispatch.", [(pol, summary.mean_batch_size)])
    return "\n".join(out) + "\n"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
