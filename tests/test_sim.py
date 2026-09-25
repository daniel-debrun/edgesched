from __future__ import annotations

import itertools
import time

import pytest

from edgesched.config import load_workload
from edgesched.metrics import summarize
from edgesched.model import Arrival, Job, TaskSpec
from edgesched.policies import EDF, POLICIES, Dispatch, Policy
from edgesched.sim import Simulator, generate_releases, simulate

from .conftest import MS, WORKLOADS


def _fingerprint(trace):
    return [
        (j.stream, j.release, j.started, j.finished, j.batch_size, j.dropped) for j in trace.jobs
    ]


def test_release_generation_counts_phase_and_jitter():
    tasks = [
        TaskSpec("a", "m", 10 * MS, phase=2 * MS),
        TaskSpec("b", "m", 25 * MS, jitter=3 * MS),
    ]
    jobs = generate_releases(tasks, 0.1, seed=3)
    a = [j for j in jobs if j.stream == "a"]
    b = [j for j in jobs if j.stream == "b"]
    assert len(a) == 10 and a[0].release == pytest.approx(0.002)
    assert len(b) == 4
    for k, j in enumerate(b):
        assert k * 0.025 <= j.release <= k * 0.025 + 0.003
        assert j.deadline == pytest.approx(k * 0.025 + 0.025)
    assert [j.release for j in jobs] == sorted(j.release for j in jobs)


def test_sporadic_gaps_are_at_least_the_period():
    task = TaskSpec("s", "m", 10 * MS, arrival=Arrival.SPORADIC, sporadic_mean_extra=5 * MS)
    rel = [j.release for j in generate_releases([task], 2.0, seed=1)]
    gaps = [b - a for a, b in itertools.pairwise(rel)]
    assert min(gaps) >= 0.010 - 1e-12
    assert len(rel) < 200


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_simulator_is_deterministic_under_seed(policy):
    wl = load_workload(WORKLOADS / "synthetic_mix.yaml")
    t1 = simulate(wl, policy, duration=3.0, seed=11)
    t2 = simulate(wl, policy, duration=3.0, seed=11)
    t3 = simulate(wl, policy, duration=3.0, seed=12)
    assert _fingerprint(t1) == _fingerprint(t2)
    assert _fingerprint(t1) != _fingerprint(t3)


def test_same_seed_gives_same_arrivals_across_policies():
    wl = load_workload(WORKLOADS / "multi_agent_cell.yaml")
    r1 = [(j.stream, j.release) for j in simulate(wl, "edf", duration=2.0).jobs]
    r2 = [(j.stream, j.release) for j in simulate(wl, "round_robin", duration=2.0).jobs]
    assert r1 == r2


def test_nonoverlapping_execution_and_timestamps(models):
    tasks = [TaskSpec(f"a{i}", "mlp", 10 * MS) for i in range(3)] + [
        TaskSpec("cam", "cnn", 33 * MS, jitter=2 * MS),
        TaskSpec("plan", "planner", 100 * MS, deadline=100 * MS),
    ]
    trace = Simulator(models, tasks, POLICIES["edf_batch"](), duration=2.0, seed=0).run()
    spans = sorted((d.start, d.end) for d in trace.dispatches)
    for (_, e0), (s1, _) in itertools.pairwise(spans):
        assert s1 >= e0 - 1e-12
    for j in trace.jobs:
        if j.finished is not None:
            assert j.release <= j.started <= j.finished
    assert summarize(trace).mean_batch_size > 1.0


def test_drop_vs_run_late(models):
    # 2x overloaded single stream
    tasks = [TaskSpec("p", "planner", 5 * MS)]
    drop = summarize(Simulator(models, tasks, EDF(), duration=1.0, late_policy="drop").run())
    run = summarize(Simulator(models, tasks, EDF(), duration=1.0, late_policy="run").run())
    assert drop.overall.dropped > 0
    assert run.overall.dropped == 0
    assert run.overall.missed > 0
    assert run.overall.max_lateness_ms > 100  # backlog grows without bound when running late


def test_policy_bugs_are_detected(models):
    class Bad(Policy):
        def choose(self, queue, now, state):
            return Dispatch("mlp", (Job("ghost", "mlp", 0.0, 1.0),))

    with pytest.raises(RuntimeError, match="not in the ready queue"):
        Simulator(models, [TaskSpec("a", "mlp", 10 * MS)], Bad(), duration=0.1).run()


def test_sixty_seconds_of_ten_streams_is_fast():
    wl = load_workload(WORKLOADS / "synthetic_mix.yaml")
    t0 = time.perf_counter()
    trace = simulate(wl, "edf_batch", duration=60.0)
    elapsed = time.perf_counter() - t0
    assert len(trace.jobs) > 15_000
    assert elapsed < 10.0  # typically ~1 s; generous bound for slow CI runners
