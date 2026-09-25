"""Task model: models, streams, and jobs.

Times are in seconds throughout the library. Config files use milliseconds and
are converted at load time (see :mod:`edgesched.config`).
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concurrent.futures import Future

    from edgesched.profile import LatencyProfile, NoiseModel


class Criticality(enum.IntEnum):
    """Criticality level of a stream. Higher values are more critical."""

    LOW = 0
    HIGH = 1

    @classmethod
    def parse(cls, value: str | int | Criticality) -> Criticality:
        if isinstance(value, Criticality):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.upper()]


@dataclass(frozen=True)
class ModelSpec:
    """A model that can be executed on the device.

    Attributes:
        name: Unique model name.
        profile: Predicted execution time as a function of batch size.
        max_batch: Largest batch the backend accepts.
        batchable: Whether requests from different streams may share a batch.
        noise: Optional execution-time noise model, used only by the simulator.
    """

    name: str
    profile: LatencyProfile
    max_batch: int = 1
    batchable: bool = False
    noise: NoiseModel | None = None

    def __post_init__(self) -> None:
        if self.max_batch < 1:
            raise ValueError(f"{self.name}: max_batch must be >= 1")

    @property
    def effective_max_batch(self) -> int:
        return self.max_batch if self.batchable else 1

    def latency(self, batch_size: int) -> float:
        return self.profile.predict(batch_size)


class Arrival(str, enum.Enum):
    PERIODIC = "periodic"
    SPORADIC = "sporadic"


@dataclass(frozen=True)
class TaskSpec:
    """A stream of inference requests (a real-time task in scheduling terms).

    Attributes:
        name: Unique stream name.
        model: Name of the :class:`ModelSpec` this stream invokes.
        period: Release period (periodic) or minimum inter-arrival time (sporadic).
        deadline: Relative deadline. Defaults to the period (implicit deadline).
        priority: Fixed priority, lower is more important. ``None`` means
            rate-monotonic (derived from the period) for fixed-priority policies.
        phase: Offset of the first release.
        jitter: Maximum release jitter; each release is delayed by U(0, jitter).
        criticality: Used for tie-breaking and in admission reports.
        arrival: Periodic or sporadic. Sporadic gaps are ``period + Exp(mean=sporadic_mean_extra)``.
        sporadic_mean_extra: Mean extra gap for sporadic streams.
        group: Optional batching group. Streams in one group share a model,
            period, deadline and release tick (e.g. MAPPO actors stepped by one
            environment tick). Same-deadline batching only combines jobs of one
            group release, which is what makes it certifiable (see
            :mod:`edgesched.analysis`).
    """

    name: str
    model: str
    period: float
    deadline: float | None = None
    priority: int | None = None
    phase: float = 0.0
    jitter: float = 0.0
    criticality: Criticality = Criticality.LOW
    arrival: Arrival = Arrival.PERIODIC
    sporadic_mean_extra: float = 0.0
    group: str | None = None

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"{self.name}: period must be > 0")
        if self.deadline is not None and self.deadline <= 0:
            raise ValueError(f"{self.name}: deadline must be > 0")
        if self.jitter < 0 or self.phase < 0:
            raise ValueError(f"{self.name}: phase and jitter must be >= 0")

    @property
    def relative_deadline(self) -> float:
        return self.period if self.deadline is None else self.deadline

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.period


_job_ids = itertools.count()


@dataclass(eq=False)
class Job:
    """One inference request.

    ``release``/``started``/``finished`` mirror the produced/read timestamps a
    shared-buffer slot carries, extended with the scheduling-relevant ones.
    """

    stream: str
    model: str
    release: float
    deadline: float
    priority: int = 0
    criticality: Criticality = Criticality.LOW
    group: str | None = None
    payload: Any = None
    id: int = field(default_factory=lambda: next(_job_ids))
    started: float | None = None
    finished: float | None = None
    batch_size: int = 0
    dropped: bool = False
    future: Future[Any] | None = field(default=None, repr=False)

    @property
    def response_time(self) -> float | None:
        return None if self.finished is None else self.finished - self.release

    @property
    def lateness(self) -> float | None:
        return None if self.finished is None else self.finished - self.deadline

    @property
    def missed(self) -> bool:
        return self.dropped or (self.finished is not None and self.finished > self.deadline)


def rate_monotonic_priorities(tasks: list[TaskSpec]) -> dict[str, int]:
    """Explicit priorities where given, otherwise rank by period (shorter = higher)."""
    order = sorted(tasks, key=lambda t: (t.period, t.relative_deadline, t.name))
    prio: dict[str, int] = {}
    for rank, task in enumerate(order):
        prio[task.name] = task.priority if task.priority is not None else rank
    return prio
