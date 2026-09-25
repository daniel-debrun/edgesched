"""Policy interface shared by the simulator and the real runtime.

A policy sees the ready queue, the current time and a :class:`DeviceState`,
and returns either a :class:`Dispatch` (run these jobs of one model as one
batch, now) or a :class:`Wait` (do nothing until ``until`` or until a new job
arrives). Policies never touch clocks, threads or backends, which is what lets
the exact same objects drive the discrete-event simulator and the
:class:`~edgesched.runtime.Arbiter`.

The executor is non-preemptive: once a batch starts it runs to completion.
Policies only choose what runs next.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from edgesched.model import Job, ModelSpec


@dataclass(frozen=True)
class Dispatch:
    """Run ``jobs`` (all of ``model``) as a single batch."""

    model: str
    jobs: tuple[Job, ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("Dispatch needs at least one job")
        if any(j.model != self.model for j in self.jobs):
            raise ValueError("all jobs in a batch must target the same model")


@dataclass(frozen=True)
class Wait:
    """Leave the device idle until ``until`` (absolute time) or the next arrival."""

    until: float | None = None


Decision = Dispatch | Wait


@dataclass
class DeviceState:
    """What a policy may know about the device.

    ``predict`` applies ``safety_factor`` and the per-model online correction
    ``scale`` (maintained by the runtime; 1.0 in simulation unless set).
    """

    models: Mapping[str, ModelSpec]
    safety_factor: float = 1.0
    scale: dict[str, float] = field(default_factory=dict)
    last_model: str | None = None
    dispatches: int = 0

    def predict(self, model: str, batch_size: int) -> float:
        base = self.models[model].latency(batch_size)
        return base * self.safety_factor * self.scale.get(model, 1.0)

    def max_batch(self, model: str) -> int:
        return self.models[model].effective_max_batch


class Policy(abc.ABC):
    """Base class for scheduling policies."""

    name: str = "policy"
    #: Which admission test certifies this policy's batching: ``"none"``,
    #: ``"same_deadline"`` or ``"cross_deadline"`` (see :mod:`edgesched.analysis`).
    batch_rule: str = "none"

    @abc.abstractmethod
    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Decision:
        """Pick the next batch. ``queue`` contains only released, undispatched jobs."""

    def reset(self) -> None:  # noqa: B027 - optional hook for stateful policies
        """Clear internal state between runs."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def edf_key(job: Job) -> tuple[float, int, float, int]:
    """Earliest deadline, then higher criticality, then earlier release, then id."""
    return (job.deadline, -int(job.criticality), job.release, job.id)
