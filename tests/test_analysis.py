from __future__ import annotations

import math
import random

import pytest

from edgesched.analysis import admit, analyze, batch_increment, batched_utilization
from edgesched.metrics import summarize
from edgesched.model import Criticality, ModelSpec, TaskSpec
from edgesched.policies import EDF, EDFBatch
from edgesched.profile import LinearProfile, TableProfile
from edgesched.sim import Simulator

MS = 1e-3


def _model(name: str, a_ms: float, c_ms: float = 0.0, max_batch: int = 1) -> ModelSpec:
    return ModelSpec(name, LinearProfile(a_ms * MS, c_ms * MS), max_batch, max_batch > 1)


def _sim(models, tasks, policy, duration=1.0, seed=0):
    trace = Simulator(
        models, tasks, policy, duration=duration, seed=seed, noise=False, late_policy="run"
    ).run()
    return summarize(trace, policy.name)


def test_light_task_set_is_admitted():
    models = {"a": _model("a", 1.0), "b": _model("b", 1.0)}
    tasks = [TaskSpec("t1", "a", 4 * MS), TaskSpec("t2", "b", 5 * MS)]
    r = analyze(models, tasks)
    assert r.admitted, r
    assert r.utilization == pytest.approx(0.45)


def test_overloaded_task_set_is_rejected():
    models = {"a": _model("a", 3.0), "b": _model("b", 3.0)}
    tasks = [TaskSpec("t1", "a", 5 * MS), TaskSpec("t2", "b", 5 * MS)]
    r = analyze(models, tasks)
    assert not r.admitted
    assert any("utilization" in reason for reason in r.reasons)


def test_non_preemptive_blocking_rejects_low_utilization_set_that_misses():
    # U ~ 0.5, but a 1.5 ms non-preemptive job can block a job with a 2 ms deadline.
    models = {"fast": _model("fast", 1.0), "slow": _model("slow", 1.5)}
    tasks = [TaskSpec("fast", "fast", 2 * MS, phase=0.1 * MS), TaskSpec("slow", "slow", 100 * MS)]
    r = analyze(models, tasks)
    assert r.utilization < 0.6
    assert not r.admitted
    assert "blocking" in r.reasons[0]
    assert _sim(models, tasks, EDF(), duration=0.5).overall.missed > 0


def test_jitter_consumes_deadline():
    models = {"p": _model("p", 1.0)}
    assert analyze(models, [TaskSpec("p", "p", 10 * MS, deadline=3 * MS)]).admitted
    jittery = [TaskSpec("p", "p", 10 * MS, deadline=2 * MS, jitter=2 * MS)]
    assert not analyze(models, jittery).admitted


def test_cross_deadline_batching_charges_bigger_blocking():
    models = {"mlp": _model("mlp", 2.0, 0.5, max_batch=8), "p": _model("p", 1.0)}
    tasks = [TaskSpec(f"a{i}", "mlp", 40 * MS) for i in range(4)]
    tasks.append(TaskSpec("p", "p", 4 * MS))
    # unbatched: at t=4 ms, 1 ms demand + 2.5 ms blocking fits
    assert analyze(models, tasks, batching="none").admitted
    # batched: a batch of 4 lasts 4 ms, so 1 ms + 4 ms blocking > 4 ms
    assert not analyze(models, tasks, batching="cross_deadline").admitted


# --- Proposition: blocking-only charging of cross-deadline batches is unsound ---------

EPS = 0.01  # ms
RIDER_C = 0.5  # ms, C_m(2) - C_m(1)


def _counterexample():
    models = {
        "m": ModelSpec("m", LinearProfile((1 - RIDER_C) * MS, RIDER_C * MS), 2, True),
        "y": _model("y", 5.0),
        "x": _model("x", 2.0),
    }
    tasks = [
        TaskSpec("Y", "y", 100 * MS),
        TaskSpec("A", "m", 10 * MS, phase=EPS * MS),
        TaskSpec("B", "m", 20 * MS, phase=EPS * MS),
        TaskSpec("X1", "x", 10 * MS, phase=2 * EPS * MS),
        TaskSpec("X2", "x", 10 * MS, phase=2 * EPS * MS),
    ]
    return models, tasks


def _blocking_only_test(models, tasks) -> bool:
    """The unsound variant: batch cost charged only as blocking, no rider term."""
    users = {t.model: sum(u.model == t.model for u in tasks) for t in tasks}
    horizon = 200 * MS
    points = sorted({t.relative_deadline + k * t.period for t in tasks for k in range(40)})
    for p in (p for p in points if p <= horizon):
        demand = sum(
            (math.floor((p - t.relative_deadline) / t.period + 1e-9) + 1)
            * models[t.model].latency(1)
            for t in tasks
            if p >= t.relative_deadline - 1e-9
        )
        blocking = max(
            (
                models[t.model].latency(min(models[t.model].effective_max_batch, users[t.model]))
                for t in tasks
                if t.relative_deadline > p + 1e-9
            ),
            default=0.0,
        )
        if demand + blocking > p + 1e-12:
            return False
    return True


def test_counterexample_blocking_only_test_would_admit():
    models, tasks = _counterexample()
    assert _blocking_only_test(models, tasks)


@pytest.mark.parametrize("lookahead", [0, 1])
def test_counterexample_misses_under_edf_batch_and_is_not_admitted(lookahead):
    models, tasks = _counterexample()
    summary = _sim(models, tasks, EDFBatch(lookahead=lookahead), duration=0.2)
    x2 = next(s for s in summary.streams if s.name == "X2")
    assert x2.missed > 0, summary.table()
    assert x2.max_lateness_ms == pytest.approx(RIDER_C - 2 * EPS, abs=1e-6)

    result = analyze(models, tasks, batching="cross_deadline")
    assert not result.admitted
    assert "riders" in result.reasons[0]


def test_counterexample_without_batching_is_admitted_and_meets_deadlines():
    models, tasks = _counterexample()
    assert analyze(models, tasks, batching="none").admitted
    assert _sim(models, tasks, EDF(), duration=0.2).overall.missed == 0
    # rule RA never batches A with B (different deadlines, no shared group)
    same = EDFBatch(rule="same_deadline")
    assert analyze(models, tasks, batching="same_deadline").admitted
    assert _sim(models, tasks, same, duration=0.2).overall.missed == 0


def test_batch_increment_condition_is_enforced():
    # C(1) = 1 ms, C(2) = 3 ms: delta = 2 ms > C(1) -> cross-deadline not certified
    models = {"m": ModelSpec("m", TableProfile({1: 1 * MS, 2: 3 * MS}), 2, True)}
    tasks = [TaskSpec("a", "m", 100 * MS), TaskSpec("b", "m", 100 * MS)]
    assert batch_increment(models["m"]) == pytest.approx(2 * MS)
    r = analyze(models, tasks, batching="cross_deadline")
    assert not r.admitted and "not certified" in r.reasons[0]


# --- Same-deadline (group) batching ---------------------------------------------------


def test_same_deadline_groups_are_admitted_where_unbatched_is_not():
    models = {"actor": _model("actor", 2.0, 0.5, max_batch=4), "p": _model("p", 1.0)}
    tasks = [TaskSpec(f"a{i}", "actor", 10 * MS, group="actors") for i in range(4)]
    tasks.append(TaskSpec("p", "p", 5 * MS, phase=0.3 * MS))
    assert not analyze(models, tasks, batching="none").admitted  # U = 1.2
    r = analyze(models, tasks, batching="same_deadline")
    assert r.admitted, r
    assert r.utilization_charged == pytest.approx(0.6)
    assert _sim(models, tasks, EDFBatch(rule="same_deadline")).overall.missed == 0
    assert _sim(models, tasks, EDF()).overall.missed > 0


def test_invalid_groups_are_not_certified():
    models = {"actor": _model("actor", 1.0, 0.5, max_batch=4)}
    jittery = [TaskSpec(f"a{i}", "actor", 10 * MS, group="g", jitter=1 * MS) for i in range(2)]
    r = analyze(models, jittery, batching="same_deadline")
    assert not r.admitted and "zero jitter" in r.reasons[0]
    mixed = [TaskSpec("a", "actor", 10 * MS, group="g"), TaskSpec("b", "actor", 20 * MS, group="g")]
    assert not analyze(models, mixed, batching="same_deadline").admitted


def test_batched_utilization_is_lower_for_shared_models():
    models = {"mlp": _model("mlp", 2.0, 0.5, max_batch=4)}
    tasks = [TaskSpec(f"a{i}", "mlp", 10 * MS) for i in range(4)]
    assert batched_utilization(models, tasks) == pytest.approx(4.0 / 10)
    assert analyze(models, tasks).utilization == pytest.approx(10.0 / 10)


def test_admit_incremental_and_criticality_report():
    models = {"fast": _model("fast", 1.0), "slow": _model("slow", 5.0)}
    base = [TaskSpec("fast", "fast", 3 * MS, criticality=Criticality.HIGH)]
    assert analyze(models, base).admitted
    r = admit(models, base, TaskSpec("slow", "slow", 50 * MS))
    assert not r.admitted
    assert r.high_criticality_admitted is True


def test_arbitrary_deadline_and_bad_rule_rejected():
    models = {"a": _model("a", 1.0)}
    r = analyze(models, [TaskSpec("t", "a", 10 * MS, deadline=20 * MS)])
    assert not r.admitted and "constrained" in r.reasons[0]
    with pytest.raises(ValueError):
        analyze(models, [TaskSpec("t", "a", 10 * MS)], batching="sometimes")


# --- Sufficiency cross-check against the simulator ------------------------------------


def _random_task_set(rng: random.Random):
    periods_ms = [5, 10, 20, 25, 40, 50, 100]
    eps = rng.choice([0.0, 0.01, 0.05])
    target_u = rng.uniform(0.2, 0.95)
    models: dict[str, ModelSpec] = {}
    tasks: list[TaskSpec] = []
    n_entities = rng.randint(2, 6)
    shares = [rng.random() for _ in range(n_entities)]
    for i in range(n_entities):
        period = rng.choice(periods_ms)
        budget = max(0.05, target_u * shares[i] / sum(shares) * period)
        deadline = rng.uniform(0.5, 1.0) * period
        phase = rng.choice([0.0, eps, 2 * eps, rng.uniform(0, period)])
        kind = rng.random()
        if kind < 0.35:  # a release-aligned group on a batchable model
            n = rng.randint(2, 4)
            cap = rng.choice([2, 4])
            c1 = budget / n
            models[f"g{i}"] = _model(f"g{i}", c1 * 0.7, c1 * 0.3, max_batch=cap)
            tasks += [
                TaskSpec(
                    f"g{i}_{k}",
                    f"g{i}",
                    period * MS,
                    deadline * MS,
                    phase=phase * MS,
                    group=f"g{i}",
                )
                for k in range(n)
            ]
        elif kind < 0.6:  # join an existing shared batchable model with a different period
            shared = next((m for m in models if m.startswith("s")), None)
            if shared is None:
                shared = f"s{i}"
                models[shared] = _model(shared, budget * 0.6, budget * 0.4, max_batch=2)
            tasks.append(TaskSpec(f"t{i}", shared, period * MS, deadline * MS, phase=phase * MS))
        else:
            models[f"m{i}"] = _model(f"m{i}", budget)
            jitter = rng.choice([0.0, 0.2])
            tasks.append(
                TaskSpec(
                    f"t{i}",
                    f"m{i}",
                    period * MS,
                    max(deadline, 0.5) * MS,
                    phase=phase * MS,
                    jitter=jitter * MS,
                )
            )
    return models, tasks


def test_admitted_sets_have_no_misses_in_simulation():
    """Every task set a rule admits must meet all deadlines under the policy it certifies."""
    rng = random.Random(2026)
    admitted = {"none": 0, "same_deadline": 0, "cross_deadline": 0}
    for trial in range(160):
        models, tasks = _random_task_set(rng)
        candidates = (
            (EDF(), "none"),
            (EDFBatch(rule="same_deadline"), "same_deadline"),
            (EDFBatch(lookahead=rng.choice([0, 1, 2])), "cross_deadline"),
        )
        for policy, rule in candidates:
            if not analyze(models, tasks, batching=rule).admitted:
                continue
            admitted[rule] += 1
            s = _sim(models, tasks, policy, duration=0.6, seed=trial)
            assert s.overall.missed == 0, (policy, rule, tasks, s.table())
    assert all(v >= 15 for v in admitted.values()), admitted
