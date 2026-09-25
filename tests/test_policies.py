from __future__ import annotations

import random

import pytest

from edgesched.model import Criticality, ModelSpec
from edgesched.policies import (
    EDF,
    FIFO,
    POLICIES,
    DeviceState,
    Dispatch,
    DynamicBatchTimeout,
    EDFBatch,
    FixedPriority,
    RoundRobin,
    Wait,
    make_policy,
)
from edgesched.profile import LinearProfile

from .conftest import MS, make_job


def test_empty_queue_waits(state):
    for name in POLICIES:
        decision = make_policy(name).choose([], 0.0, state)
        assert isinstance(decision, Wait)


def test_edf_picks_earliest_deadline_then_criticality(state):
    a = make_job("a", "mlp", 0.0, 0.050)
    b = make_job("b", "cnn", 0.001, 0.020)
    c = make_job("c", "mlp", 0.002, 0.020, criticality=Criticality.HIGH)
    d = EDF().choose([a, b, c], 0.003, state)
    assert isinstance(d, Dispatch) and d.jobs == (c,)


def test_edf_order_over_a_drained_queue(state):
    jobs = [make_job(f"s{i}", "mlp", 0.0, dl) for i, dl in enumerate([0.03, 0.01, 0.02, 0.005])]
    queue, order = list(jobs), []
    while queue:
        d = EDF().choose(queue, 0.0, state)
        order.append(d.jobs[0].deadline)
        queue.remove(d.jobs[0])
    assert order == sorted(order)


def test_fifo_and_fixed_priority(state):
    a = make_job("a", "mlp", 0.002, 0.010, priority=0)
    b = make_job("b", "mlp", 0.001, 0.100, priority=5)
    assert FIFO().choose([a, b], 0.003, state).jobs == (b,)
    assert FixedPriority().choose([a, b], 0.003, state).jobs == (a,)


def test_round_robin_cycles_streams(state):
    queue = [make_job(s, "mlp", 0.0, 1.0) for s in ("x", "y", "z", "x")]
    rr = RoundRobin()
    served = []
    for _ in range(4):
        d = rr.choose(queue, 0.0, state)
        served.append(d.jobs[0].stream)
        queue.remove(d.jobs[0])
    assert served == ["x", "y", "z", "x"]


def test_edf_batch_groups_same_model_up_to_max_batch(state):
    queue = [make_job(f"a{i}", "mlp", 0.0, 0.020) for i in range(10)]
    d = EDFBatch().choose(queue, 0.0, state)
    assert d.model == "mlp"
    assert len(d.jobs) == 8


def test_edf_batch_does_not_batch_non_batchable(state):
    queue = [make_job(f"p{i}", "planner", 0.0, 0.5) for i in range(3)]
    assert len(EDFBatch().choose(queue, 0.0, state).jobs) == 1


def test_edf_batch_lookahead_limits_batch(state):
    # cnn(1) = 7 ms, planner = 10 ms -> 17 ms <= 18 ms; cnn(2) = 9 ms -> 19 ms > 18 ms.
    head = make_job("cam0", "cnn", 0.0, 0.015)
    other_cam = make_job("cam1", "cnn", 0.0, 0.030)
    planner = make_job("plan", "planner", 0.0, 0.018)
    d = EDFBatch(lookahead=1).choose([head, other_cam, planner], 0.0, state)
    assert d.jobs == (head,)
    d0 = EDFBatch(lookahead=0).choose([head, other_cam, planner], 0.0, state)
    assert len(d0.jobs) == 2


def test_edf_batch_late_head_still_batches_jobs_that_can_meet(state):
    # Run-late mode: the head is already late, so it constrains nothing. A candidate
    # that would miss at batch size 2 is skipped; a later one that meets is added.
    now = 0.010
    late_head = make_job("a", "cnn", 0.0, 0.005)
    doomed = make_job("b", "cnn", 0.0, 0.012)  # cnn(2) = 9 ms -> finishes at 19 ms
    ok = make_job("c", "cnn", 0.0, 0.050)
    d = EDFBatch().choose([late_head, doomed, ok], now, state)
    assert d.jobs == (late_head, ok)


def _predicted_invariants(queue, now, state, decision, lookahead):
    """Check the EDFBatch safety invariant against a non-batched dispatch of the head."""
    ordered = sorted(queue, key=lambda j: (j.deadline, -int(j.criticality), j.release, j.id))
    head = ordered[0]
    assert decision.jobs[0] is head
    n = len(decision.jobs)
    finish_batch = now + state.predict(head.model, n)
    finish_alone = now + state.predict(head.model, 1)
    if finish_alone <= head.deadline:
        assert finish_batch <= head.deadline
    for j in decision.jobs[1:]:
        assert finish_batch <= j.deadline
    others = [j for j in ordered if j.model != head.model][:lookahead]
    t_alone, t_batch = finish_alone, finish_batch
    for k in others:
        t_alone += state.predict(k.model, 1)
        t_batch += state.predict(k.model, 1)
        if t_alone <= k.deadline:
            assert t_batch <= k.deadline


@pytest.mark.parametrize("lookahead", [0, 1, 3])
def test_edf_batch_never_creates_predicted_miss_randomized(lookahead):
    rng = random.Random(1234 + lookahead)
    batched_something = 0
    for _ in range(2000):
        models = {
            f"m{i}": ModelSpec(
                f"m{i}",
                LinearProfile(rng.uniform(0.5, 8) * MS, rng.uniform(0, 3) * MS),
                max_batch=rng.choice([1, 2, 4, 8]),
                batchable=rng.random() < 0.8,
            )
            for i in range(3)
        }
        state = DeviceState(models, safety_factor=rng.choice([1.0, 1.2]))
        now = 1.0
        queue = [
            make_job(
                f"s{k}",
                rng.choice(list(models)),
                now - rng.uniform(0, 0.01),
                now + rng.uniform(-0.002, 0.04),
                criticality=Criticality(rng.randint(0, 1)),
            )
            for k in range(rng.randint(1, 12))
        ]
        decision = EDFBatch(lookahead=lookahead).choose(queue, now, state)
        assert isinstance(decision, Dispatch)
        assert len(decision.jobs) <= state.max_batch(decision.model)
        assert all(j.model == decision.model for j in decision.jobs)
        _predicted_invariants(queue, now, state, decision, lookahead)
        batched_something += len(decision.jobs) > 1
    assert batched_something > 100  # the test exercises real batching


def test_dynamic_batch_waits_then_flushes(state):
    policy = DynamicBatchTimeout(max_queue_delay=0.002, preferred_batch_sizes={"mlp": [4]})
    queue = [make_job(f"a{i}", "mlp", 0.0, 1.0) for i in range(2)]
    d = policy.choose(queue, 0.001, state)
    assert isinstance(d, Wait) and d.until == pytest.approx(0.002)
    d = policy.choose(queue, 0.002, state)
    assert isinstance(d, Dispatch) and len(d.jobs) == 2
    queue += [make_job(f"b{i}", "mlp", 0.0005, 1.0) for i in range(3)]
    d = policy.choose(queue, 0.0006, state)
    assert isinstance(d, Dispatch) and len(d.jobs) == 4


def test_dynamic_batch_non_batchable_dispatches_immediately(state):
    policy = DynamicBatchTimeout(max_queue_delay=1.0)
    d = policy.choose([make_job("p", "planner", 0.0, 1.0)], 0.0, state)
    assert isinstance(d, Dispatch)


def test_dispatch_validation(state):
    with pytest.raises(ValueError):
        Dispatch("mlp", ())
    with pytest.raises(ValueError):
        Dispatch("mlp", (make_job("a", "cnn", 0, 1),))
    with pytest.raises(ValueError):
        make_policy("nope")


def test_same_deadline_rule_batches_only_one_group_release(state):
    g = [make_job(f"a{i}", "mlp", 0.0, 0.010, group="actors") for i in range(3)]
    other_release = make_job("a3", "mlp", 0.001, 0.010, group="actors")
    other_group = make_job("b0", "mlp", 0.0, 0.010, group="other")
    ungrouped = make_job("c0", "mlp", 0.0, 0.010)
    rider = make_job("a4", "mlp", 0.0, 0.020, group="actors")
    queue = [*g, other_release, other_group, ungrouped, rider]
    policy = EDFBatch(rule="same_deadline")
    assert policy.choose(queue, 0.002, state).jobs == tuple(g)
    assert policy.choose([ungrouped, g[0]], 0.0, state).jobs == (g[0],)
    assert policy.choose([ungrouped, other_group], 0.0, state).jobs == (other_group,)
    with pytest.raises(ValueError):
        EDFBatch(rule="whenever")
