"""Unbatched baseline policies: round-robin, FIFO, fixed priority, EDF."""

from __future__ import annotations

from collections.abc import Sequence

from edgesched.model import Job
from edgesched.policies.base import DeviceState, Dispatch, Policy, Wait, edf_key


def _single(job: Job) -> Dispatch:
    return Dispatch(job.model, (job,))


class RoundRobin(Policy):
    """Serve streams in a fixed cyclic order, one request at a time.

    This is what a ``while not stopped: for w in workers: w.step()`` loop does
    when every worker calls its model directly: no notion of deadlines, and a
    stream with a 10 ms deadline waits behind a 200 ms planner if the planner
    happens to be next in the cycle.
    """

    name = "round_robin"

    def __init__(self) -> None:
        self._last: str | None = None

    def reset(self) -> None:
        self._last = None

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        streams = sorted({j.stream for j in queue})
        nxt = next((s for s in streams if self._last is None or s > self._last), streams[0])
        self._last = nxt
        job = min((j for j in queue if j.stream == nxt), key=lambda j: (j.release, j.id))
        return _single(job)


class FIFO(Policy):
    """Earliest release first."""

    name = "fifo"

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        return _single(min(queue, key=lambda j: (j.release, j.id)))


class FixedPriority(Policy):
    """Non-preemptive fixed priority. With default priorities this is rate-monotonic."""

    name = "fixed_priority"

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        return _single(min(queue, key=lambda j: (j.priority, j.release, j.id)))


class EDF(Policy):
    """Non-preemptive earliest-deadline-first, batch size 1."""

    name = "edf"

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        return _single(min(queue, key=edf_key))
