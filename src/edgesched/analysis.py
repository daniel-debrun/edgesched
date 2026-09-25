r"""Admission control and schedulability analysis for one non-preemptive executor.

All tests assume a single work-conserving executor that runs one batch at a
time, non-preemptively, under EDF ordering (``EDF`` or ``EDFBatch``). They say
nothing about FIFO, round-robin or the Triton-style batcher. ``C_m(b)`` is the
profile's predicted latency for a batch of ``b``; the tests are only as good as
that number, so on a real device use a high percentile (or a safety factor).

**Utilization** (necessary). ``U = sum_i C_i(1) / T_i``. If ``U > 1`` no
policy keeps up without batching. ``batched_utilization`` (streams sharing a
batchable model and a period served as full batches) is reported for
information only; it is optimistic.

**No batching** (``batching="none"``). Non-preemptive EDF demand bound
(Jeffay, Stanat and Martel, RTSS 1991; George, Rivierre and Spuri, INRIA
RR-2966, 1996). Schedulable if ``U < 1`` and at every absolute deadline
``t <= L``::

    dbf(t) + B(t) <= t
    dbf(t) = sum_i eta_i(t) * C_i(1),   eta_i(t) = max(0, floor((t - D_i) / T_i) + 1)
    B(t)   = max { C_j(1) : D_j > t }    (0 if none)
    L      = max(D_max, (sum_i (T_i - D_i) C_i / T_i + max B) / (1 - U))

The integer-time result uses ``C_j - 1`` in ``B``; in continuous time we use
``C_j``, which is conservative. Release jitter is folded in as ``D_i - J_i``.

**Same-deadline batching** (``batching="same_deadline"``, rule RA, policy
``EDFBatch(rule="same_deadline")``). Streams declare a ``group``: same
batchable model, period, deadline and release tick, no jitter. Only jobs of
one group release are batched, and batches are maximal. Each group is charged
as one task with ``W_g = floor(n_g / B) C(B) + C(n_g mod B)`` per release, and
its blocking is ``C(min(n_g, B))``; the test above is then applied to the
groups. Ungrouped streams are singleton groups.

**Cross-deadline batching** (``batching="cross_deadline"``, rule RB, policy
``EDFBatch()``). Any same-model jobs may share a batch. Charging a batch only
as blocking is **unsound**: a batch that carries later-deadline "riders" also
consumes time inside the busy window of earlier deadlines. Counterexample
(regression-tested): ``C_m(1)=1, C_m(2)=1+c``, stream Y (C=5, T=D=100), A and B
on ``m`` (T=D=10 and T=D=20), X1, X2 (C=2, T=D=10). The blocking-only test holds
at t=10 (5 + 5 <= 10), but with Y released at 0, A and B at eps and X1, X2 at
2 eps, the batch {A, B} at t=5 pushes X2 to 10 + c. The sound test adds a rider
term::

    dbf(t) + B(t) + R <= t
    B(t) = max { C_m(min(B_m, N_m)) : D_j > t }
    R    = sum over streams i on batchable models shared by >= 2 streams of delta_m
    delta_m = max_b [C_m(b) - C_m(b - 1)]         (requires delta_m <= C_m(1))

and uses ``R`` in ``L`` as well. The argument: a batch with ``k`` jobs of
deadline ``<= d`` and ``r`` riders costs at most ``k C(1) + r delta``, and with
constrained deadlines at most one job per stream can ride into a window.

The RA/RB tests follow the analysis in a working paper on multi-model inference
scheduling that accompanies this project (in preparation; proof sketches, not
yet peer reviewed). The test suite cross-checks every admitted random task
set against the simulator.

Preemption is not modelled because accelerator kernels (TensorRT engines,
cuDNN calls) cannot be preempted mid-execution; the scheduler only chooses
what runs next.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from edgesched.model import Arrival, Criticality, ModelSpec, TaskSpec

_EPS = 1e-9
BATCH_RULES = ("none", "same_deadline", "cross_deadline")


@dataclass
class AnalysisResult:
    admitted: bool
    utilization: float
    utilization_batched: float
    rule: str = "none"
    utilization_charged: float = math.nan
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    high_criticality_admitted: bool | None = None
    points_checked: int = 0

    def __str__(self) -> str:
        verdict = "ADMITTED" if self.admitted else "REJECTED"
        lines = [
            f"{verdict} (batching rule: {self.rule})",
            f"  utilization (unbatched): {self.utilization:.3f}",
            f"  utilization charged by the test: {self.utilization_charged:.3f}",
            f"  utilization (batch-aware, optimistic): {self.utilization_batched:.3f}",
        ]
        lines += [f"  check {k}: {'pass' if v else 'fail'}" for k, v in self.checks.items()]
        lines += [f"  reason: {r}" for r in self.reasons]
        if self.high_criticality_admitted is not None:
            hi = "admitted" if self.high_criticality_admitted else "rejected"
            lines.append(f"  high-criticality subset alone: {hi}")
        return "\n".join(lines)


def utilization(models: Mapping[str, ModelSpec], tasks: Sequence[TaskSpec]) -> float:
    return sum(models[t.model].latency(1) / t.period for t in tasks)


def batched_utilization(models: Mapping[str, ModelSpec], tasks: Sequence[TaskSpec]) -> float:
    groups: dict[tuple[str, float], int] = {}
    for t in tasks:
        key = (t.model, round(t.period, 9))
        groups[key] = groups.get(key, 0) + 1
    total = 0.0
    for (model_name, period), k in groups.items():
        total += _batched_cost(models[model_name], k) / period
    return total


def _batched_cost(model: ModelSpec, n: int) -> float:
    """``floor(n/B) C(B) + C(n mod B)``: cost of serving ``n`` simultaneous jobs maximally."""
    cap = model.effective_max_batch
    full, rest = divmod(n, cap)
    return full * model.latency(cap) + (model.latency(rest) if rest else 0.0)


def batch_increment(model: ModelSpec) -> float:
    """``delta_m``: the largest cost of adding one job to a batch."""
    cap = model.effective_max_batch
    return max((model.latency(b) - model.latency(b - 1) for b in range(2, cap + 1)), default=0.0)


@dataclass(frozen=True)
class _Entity:
    """A demand source: one stream, or one same-deadline group."""

    name: str
    period: float
    deadline: float
    cost: float
    blocking: float


def _normalize_rule(batching: str | bool) -> str:
    if batching is True:
        return "cross_deadline"
    if batching is False:
        return "none"
    if batching not in BATCH_RULES:
        raise ValueError(f"batching must be one of {BATCH_RULES}")
    return batching


def _entities(
    models: Mapping[str, ModelSpec], tasks: Sequence[TaskSpec], rule: str
) -> tuple[list[_Entity], float, list[str]]:
    """Build demand entities and the rider term for a rule. Returns (entities, R, errors)."""
    errors: list[str] = []
    if rule == "same_deadline":
        groups: dict[str, list[TaskSpec]] = {}
        singles: list[TaskSpec] = []
        for t in tasks:
            (groups.setdefault(t.group, []) if t.group else singles).append(t)
        ents = []
        for name, members in groups.items():
            first = members[0]
            key = (first.model, first.period, first.relative_deadline, first.phase)
            for m in members:
                if (m.model, m.period, m.relative_deadline, m.phase) != key:
                    errors.append(
                        f"group {name!r}: members must share model, period, deadline, phase"
                    )
                if m.jitter > 0 or m.arrival is not Arrival.PERIODIC:
                    errors.append(f"group {name!r}: members must be periodic with zero jitter")
            model = models[first.model]
            n = len(members)
            ents.append(
                _Entity(
                    f"group:{name}",
                    first.period,
                    first.relative_deadline,
                    _batched_cost(model, n),
                    model.latency(min(n, model.effective_max_batch)),
                )
            )
        for t in singles:
            c = models[t.model].latency(1)
            ents.append(_Entity(t.name, t.period, t.relative_deadline - t.jitter, c, c))
        return ents, 0.0, errors

    users: dict[str, int] = {}
    for t in tasks:
        users[t.model] = users.get(t.model, 0) + 1
    rider = 0.0
    ents = []
    for t in tasks:
        model = models[t.model]
        c1 = model.latency(1)
        blocking = c1
        if rule == "cross_deadline" and model.batchable and users[t.model] >= 2:
            blocking = model.latency(min(model.effective_max_batch, users[t.model]))
            delta = batch_increment(model)
            if delta > c1 + _EPS:
                errors.append(
                    f"model {t.model!r}: batch increment {delta * 1e3:.3f} ms exceeds C(1) "
                    f"{c1 * 1e3:.3f} ms, cross-deadline batching is not certified"
                )
            rider += delta
        ents.append(_Entity(t.name, t.period, t.relative_deadline - t.jitter, c1, blocking))
    return ents, rider, errors


def np_edf_demand_test(
    models: Mapping[str, ModelSpec],
    tasks: Sequence[TaskSpec],
    *,
    batching: str | bool = "none",
    max_points: int = 200_000,
) -> tuple[bool, str, int, float]:
    """Run the demand-bound test for a batching rule.

    Returns ``(ok, reason, points_checked, charged_utilization)``.
    """
    rule = _normalize_rule(batching)
    if not tasks:
        return True, "", 0, 0.0
    for t in tasks:
        if t.relative_deadline > t.period + 1e-12:
            return False, f"{t.name}: test assumes constrained deadlines (D <= T)", 0, math.nan
        if t.relative_deadline - t.jitter <= 0:
            return False, f"{t.name}: jitter consumes the whole deadline", 0, math.nan

    ents, rider, errors = _entities(models, tasks, rule)
    u = sum(e.cost / e.period for e in ents)
    if errors:
        return False, errors[0], 0, u
    if u >= 1.0 - 1e-12:
        return False, f"charged utilization {u:.3f} >= 1", 0, u

    max_block = max(e.blocking for e in ents)
    d_max = max(e.deadline for e in ents)
    slack_sum = sum((e.period - e.deadline) * e.cost / e.period for e in ents)
    bound = max(d_max, (slack_sum + max_block + rider) / (1.0 - u))

    points: set[float] = set()
    for e in ents:
        k = 0
        while (t := e.deadline + k * e.period) <= bound + _EPS:
            points.add(t)
            if len(points) > max_points:
                return False, f"inconclusive: more than {max_points} test points", len(points), u
            k += 1
    ordered = sorted(points)
    for t in ordered:
        demand = sum(
            (math.floor((t - e.deadline) / e.period + _EPS) + 1) * e.cost
            for e in ents
            if t >= e.deadline - _EPS
        )
        blocking = max((e.blocking for e in ents if e.deadline > t + _EPS), default=0.0)
        if demand + blocking + rider > t + 1e-12:
            rider_txt = f" + riders {rider * 1e3:.3f} ms" if rider else ""
            return (
                False,
                f"demand bound violated at t={t * 1e3:.3f} ms: demand {demand * 1e3:.3f} ms "
                f"+ blocking {blocking * 1e3:.3f} ms{rider_txt} > {t * 1e3:.3f} ms",
                len(ordered),
                u,
            )
    return True, "", len(ordered), u


def analyze(
    models: Mapping[str, ModelSpec],
    tasks: Sequence[TaskSpec],
    *,
    batching: str | bool = "none",
) -> AnalysisResult:
    """Admit or reject a task set for EDF-family scheduling under a batching rule.

    ``batching`` is ``"none"``, ``"same_deadline"`` or ``"cross_deadline"``
    (``True`` means ``"cross_deadline"``). Use a policy's ``batch_rule``
    attribute to pick the rule that certifies it.
    """
    rule = _normalize_rule(batching)
    u = utilization(models, tasks)
    ub = batched_utilization(models, tasks)
    demand_ok, reason, npts, charged = np_edf_demand_test(models, tasks, batching=rule)
    reasons = [reason] if not demand_ok and reason else []
    necessary_ok = (ub if rule != "none" else u) <= 1.0 + 1e-12
    result = AnalysisResult(
        admitted=demand_ok and necessary_ok,
        utilization=u,
        utilization_batched=ub,
        rule=rule,
        utilization_charged=charged,
        reasons=reasons,
        checks={"utilization<=1": necessary_ok, "np_edf_demand_bound": demand_ok},
        points_checked=npts,
    )
    if not result.admitted:
        hi = [t for t in tasks if t.criticality is Criticality.HIGH]
        if hi and len(hi) < len(tasks):
            result.high_criticality_admitted = analyze(models, hi, batching=rule).admitted
    return result


def admit(
    models: Mapping[str, ModelSpec],
    admitted: Sequence[TaskSpec],
    candidate: TaskSpec,
    *,
    batching: str | bool = "none",
) -> AnalysisResult:
    """Incremental admission: would ``admitted + [candidate]`` still pass?"""
    return analyze(models, [*admitted, candidate], batching=batching)
