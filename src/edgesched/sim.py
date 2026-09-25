"""Discrete-event simulator of one non-preemptive accelerator.

The simulated device runs one batch at a time. Each time it becomes free (or
a job is released while it is idle, or a policy's ``Wait`` expires) the
policy is asked for the next decision. Executed latency is the model's
profile prediction, optionally perturbed by its :class:`NoiseModel`.

Determinism: releases are generated up front from a per-stream RNG derived
from the seed, so every policy sees the identical arrival sequence for a
given seed. Execution noise uses a separate RNG stream.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from edgesched.config import Workload
from edgesched.metrics import DispatchRecord, Trace
from edgesched.model import Arrival, Job, ModelSpec, TaskSpec, rate_monotonic_priorities
from edgesched.policies import DeviceState, Dispatch, Policy


def generate_releases(tasks: Sequence[TaskSpec], duration: float, seed: int) -> list[Job]:
    """All jobs released in ``[0, duration)``, sorted by release time."""
    prios = rate_monotonic_priorities(list(tasks))
    jobs: list[Job] = []
    for idx, task in enumerate(tasks):
        rng = random.Random(seed * 1_000_003 + idx)
        nominal = task.phase
        while True:
            release = nominal + (rng.uniform(0.0, task.jitter) if task.jitter > 0 else 0.0)
            if nominal >= duration:
                break
            if release < duration:
                jobs.append(
                    Job(
                        stream=task.name,
                        model=task.model,
                        release=release,
                        deadline=nominal + task.relative_deadline,
                        priority=prios[task.name],
                        criticality=task.criticality,
                        group=task.group,
                    )
                )
            gap = task.period
            if task.arrival is Arrival.SPORADIC and task.sporadic_mean_extra > 0:
                gap += rng.expovariate(1.0 / task.sporadic_mean_extra)
            nominal += gap
    jobs.sort(key=lambda j: (j.release, j.id))
    return jobs


class Simulator:
    """Simulate ``tasks`` on ``models`` under ``policy``.

    Args:
        late_policy: ``"drop"`` discards a queued job once its deadline has
            passed; ``"run"`` keeps it and runs it late.
        safety_factor: Multiplies predictions the policy sees (not the
            simulated execution time).
        noise: Apply each model's noise model to execution times.

    Deadlines are measured from the nominal (unjittered) release, so release
    jitter consumes deadline budget, as it does for a camera frame that
    arrives late.
    """

    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        tasks: Sequence[TaskSpec],
        policy: Policy,
        *,
        duration: float,
        seed: int = 0,
        late_policy: str = "drop",
        safety_factor: float = 1.0,
        noise: bool = True,
    ) -> None:
        if late_policy not in ("drop", "run"):
            raise ValueError("late_policy must be 'drop' or 'run'")
        for t in tasks:
            if t.model not in models:
                raise ValueError(f"stream {t.name!r} references unknown model {t.model!r}")
        self.models = dict(models)
        self.tasks = list(tasks)
        self.policy = policy
        self.duration = duration
        self.seed = seed
        self.late_policy = late_policy
        self.safety_factor = safety_factor
        self.noise = noise

    @classmethod
    def from_workload(cls, workload: Workload, policy: Policy | str, **overrides) -> Simulator:
        if isinstance(policy, str):
            policy = workload.policy(policy)
        opts = {
            "duration": workload.sim.duration,
            "seed": workload.sim.seed,
            "late_policy": workload.sim.late_policy,
            "safety_factor": workload.sim.safety_factor,
        }
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return cls(workload.models, workload.tasks, policy, **opts)

    def run(self) -> Trace:
        self.policy.reset()
        releases = generate_releases(self.tasks, self.duration, self.seed)
        exec_rng = random.Random(self.seed * 7_919 + 17)
        state = DeviceState(self.models, safety_factor=self.safety_factor)
        trace = Trace(jobs=releases, start=0.0, end=self.duration)
        drop = self.late_policy == "drop"

        queue: list[Job] = []
        n, i, now = len(releases), 0, 0.0
        while now < self.duration:
            while i < n and releases[i].release <= now:
                queue.append(releases[i])
                i += 1
            if drop and queue:
                kept = []
                for j in queue:
                    if j.deadline <= now:
                        j.dropped = True
                    else:
                        kept.append(j)
                queue = kept
            next_release = releases[i].release if i < n else math.inf
            if not queue:
                if next_release == math.inf:
                    break
                now = next_release
                continue

            decision = self.policy.choose(queue, now, state)
            if not isinstance(decision, Dispatch):
                until = decision.until if decision.until is not None else math.inf
                nxt = min(next_release, until if until > now else math.inf)
                if nxt == math.inf:
                    break
                now = nxt
                continue

            model = self.models[decision.model]
            batch = decision.jobs
            if len(batch) > model.effective_max_batch:
                raise RuntimeError(f"{self.policy!r} exceeded max_batch for {model.name}")
            chosen = {id(j) for j in batch}
            remaining = [j for j in queue if id(j) not in chosen]
            if len(remaining) != len(queue) - len(batch):
                raise RuntimeError(f"{self.policy!r} dispatched a job not in the ready queue")
            queue = remaining

            latency = model.latency(len(batch))
            if self.noise and model.noise is not None:
                latency = model.noise.sample(latency, exec_rng)
            end = now + latency
            for j in batch:
                j.started, j.finished, j.batch_size = now, end, len(batch)
            trace.dispatches.append(DispatchRecord(model.name, now, end, len(batch)))
            state.last_model = model.name
            state.dispatches += 1
            now = end

        return trace


def simulate(workload: Workload, policy: Policy | str, **overrides) -> Trace:
    """Convenience wrapper: build a :class:`Simulator` from a workload and run it."""
    return Simulator.from_workload(workload, policy, **overrides).run()
