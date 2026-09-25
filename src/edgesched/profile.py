"""Latency profiles, execution-time noise, and a profiler for real callables.

A profile answers one question for the scheduler: how long will a batch of
``b`` requests for this model take? Two forms are provided:

* :class:`LinearProfile`, ``latency(b) = a + c * b``. On accelerators ``a``
  (kernel launch, memory transfer, framework overhead) usually dominates for
  small models, which is why batching helps.
* :class:`TableProfile`, measured points with linear interpolation between them
  and linear extrapolation from the last two points.

:func:`measure` times a real callable across batch sizes and produces both.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LatencyProfile(Protocol):
    def predict(self, batch_size: int) -> float: ...

    def scaled(self, factor: float) -> LatencyProfile: ...


@dataclass(frozen=True)
class LinearProfile:
    """``latency(b) = a + c * b`` seconds."""

    a: float
    c: float = 0.0

    def __post_init__(self) -> None:
        if self.a < 0 or self.c < 0:
            raise ValueError("profile coefficients must be non-negative")
        if self.a + self.c <= 0:
            raise ValueError("profile must predict a positive latency")

    def predict(self, batch_size: int) -> float:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return self.a + self.c * batch_size

    def scaled(self, factor: float) -> LinearProfile:
        return LinearProfile(self.a * factor, self.c * factor)


@dataclass(frozen=True)
class TableProfile:
    """Measured latencies keyed by batch size, linearly interpolated."""

    points: tuple[tuple[int, float], ...]

    def __init__(self, points: Mapping[int, float] | Iterable[tuple[int, float]]):
        items = points.items() if isinstance(points, Mapping) else points
        pts = tuple(sorted((int(b), float(t)) for b, t in items))
        if not pts:
            raise ValueError("TableProfile needs at least one point")
        if any(b < 1 or t <= 0 for b, t in pts):
            raise ValueError("batch sizes must be >= 1 and latencies > 0")
        object.__setattr__(self, "points", pts)

    def predict(self, batch_size: int) -> float:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        bs = [b for b, _ in self.points]
        ts = [t for _, t in self.points]
        if len(bs) == 1:
            return ts[0] * batch_size / bs[0] if batch_size > bs[0] else ts[0]
        i = bisect.bisect_left(bs, batch_size)
        if i < len(bs) and bs[i] == batch_size:
            return ts[i]
        if i == 0:
            return ts[0]
        if i == len(bs):
            i = len(bs) - 1
        b0, b1, t0, t1 = bs[i - 1], bs[i], ts[i - 1], ts[i]
        slope = max(0.0, (t1 - t0) / (b1 - b0))
        return max(ts[0], t0 + slope * (batch_size - b0))

    def scaled(self, factor: float) -> TableProfile:
        return TableProfile([(b, t * factor) for b, t in self.points])


@dataclass(frozen=True)
class NoiseModel:
    """Multiplicative execution-time noise for simulation.

    ``actual = predicted * exp(N(0, sigma))``, and with probability
    ``tail_prob`` an additional factor ``tail_scale * exp(N(0, sigma))`` models
    the long tail (allocator stalls, thermal throttling, driver hiccups).
    """

    sigma: float = 0.05
    tail_prob: float = 0.0
    tail_scale: float = 2.0

    def sample(self, predicted: float, rng: random.Random) -> float:
        factor = math.exp(rng.gauss(0.0, self.sigma)) if self.sigma > 0 else 1.0
        if self.tail_prob > 0 and rng.random() < self.tail_prob:
            factor *= self.tail_scale
        return predicted * factor


def fit_linear(points: Mapping[int, float]) -> LinearProfile:
    """Least-squares fit of ``a + c*b``, clamped to non-negative coefficients."""
    bs = list(points)
    ts = [points[b] for b in bs]
    if len(bs) == 1:
        return LinearProfile(ts[0], 0.0)
    mean_b, mean_t = statistics.fmean(bs), statistics.fmean(ts)
    var_b = sum((b - mean_b) ** 2 for b in bs)
    c = sum((b - mean_b) * (t - mean_t) for b, t in zip(bs, ts)) / var_b
    c = max(0.0, c)
    a = max(0.0, mean_t - c * mean_b)
    if a + c <= 0:
        a = min(ts)
    return LinearProfile(a, c)


def _quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return math.nan
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class ProfileResult:
    """Timings for one model across batch sizes. All values in seconds."""

    model: str
    samples: dict[int, list[float]]
    quantile: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> dict[int, dict[str, float]]:
        return {
            b: {
                "mean": statistics.fmean(s),
                "p50": _quantile(s, 0.5),
                "p90": _quantile(s, 0.9),
                "p99": _quantile(s, 0.99),
                "max": max(s),
            }
            for b, s in sorted(self.samples.items())
        }

    def table(self) -> TableProfile:
        return TableProfile({b: _quantile(s, self.quantile) for b, s in self.samples.items()})

    def linear(self) -> LinearProfile:
        return fit_linear({b: _quantile(s, self.quantile) for b, s in self.samples.items()})

    def to_json(self) -> dict[str, Any]:
        lin = self.linear()
        return {
            "model": self.model,
            "unit": "s",
            "quantile": self.quantile,
            "points": {str(b): t for b, t in self.table().points},
            "fit": {"a": lin.a, "c": lin.c},
            "stats": {str(b): v for b, v in self.stats().items()},
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")


def load_profile_json(path: str | Path, kind: str = "table") -> LatencyProfile:
    """Load a profile written by :meth:`ProfileResult.save` (or by hand / trtexec script).

    ``kind`` selects ``"table"`` (measured points) or ``"linear"`` (the fit).
    Required keys: ``points`` (batch size -> seconds) or ``fit`` (a, c); ``unit``
    may be ``"s"`` or ``"ms"``.
    """
    data = json.loads(Path(path).read_text())
    scale = 1e-3 if data.get("unit", "s") == "ms" else 1.0
    if kind == "linear" and "fit" in data:
        return LinearProfile(data["fit"]["a"] * scale, data["fit"]["c"] * scale)
    if "points" in data:
        return TableProfile({int(b): float(t) * scale for b, t in data["points"].items()})
    if "fit" in data:
        return LinearProfile(data["fit"]["a"] * scale, data["fit"]["c"] * scale)
    raise ValueError(f"{path}: profile JSON needs 'points' or 'fit'")


def measure(
    fn: Callable[[Any], Any],
    make_input: Callable[[int], Any],
    batch_sizes: Iterable[int] = (1, 2, 4, 8),
    *,
    name: str = "model",
    warmup: int = 5,
    iters: int = 30,
    quantile: float = 0.5,
    sync: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ProfileResult:
    """Time ``fn(make_input(b))`` for each batch size.

    ``sync`` is called after each invocation before stopping the clock; pass
    ``torch.cuda.synchronize`` when profiling asynchronous GPU kernels.
    """
    samples: dict[int, list[float]] = {}
    for b in batch_sizes:
        x = make_input(b)
        for _ in range(warmup):
            fn(x)
            if sync:
                sync()
        times = []
        for _ in range(iters):
            t0 = clock()
            fn(x)
            if sync:
                sync()
            times.append(clock() - t0)
        samples[b] = times
    return ProfileResult(name, samples, quantile, meta={"warmup": warmup, "iters": iters})
