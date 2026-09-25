"""Batching policies: deadline-aware EDFBatch and a Triton-style dynamic batcher."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from edgesched.model import Job
from edgesched.policies.base import DeviceState, Dispatch, Policy, Wait, edf_key


class EDFBatch(Policy):
    """Non-preemptive EDF with batching. Two batching rules are available.

    ``rule="cross_deadline"`` (default; rule RB). Let ``h`` be the
    earliest-deadline job and ``m`` its model. ``h`` is always dispatched. Let
    ``K`` be the ``lookahead`` earliest-deadline ready jobs of *other* models,
    assumed to run next, one at a time, in EDF order. Same-model candidates are
    visited in EDF order and candidate ``c`` is added to a batch of size ``n``
    only if, with ``f = now + L_m(n + 1)``:

    1. ``c`` is predicted to meet its deadline: ``f <= d_c``;
    2. no job already in the batch that was predicted to meet at size ``n``
       would miss at size ``n + 1``;
    3. no job in ``K`` that is predicted to meet when ``h`` is dispatched
       alone is predicted to miss when the batch finishes at ``f``.

    Growth stops at ``max_batch`` or at the first violation of (2) or (3); a
    candidate failing only (1) is skipped. Under the latency predictions,
    batching never turns a predicted hit for ``h`` or a lookahead job into a
    predicted miss (tested). That local guard is *not* a schedulability
    guarantee: jobs beyond the lookahead, or released during the batch, can
    still be pushed past their deadline by the riders (jobs with later
    deadlines that joined the batch). The admission test for this rule
    charges that rider overhead explicitly (``batching="cross_deadline"`` in
    :func:`edgesched.analysis.analyze`).

    ``rule="same_deadline"`` (rule RA). Only jobs from the head's declared
    group (``TaskSpec.group``) with the same release tick and the same
    absolute deadline are added, in criticality order, up to ``max_batch``,
    with no lookahead guard. Batches are maximal, which is what the
    ``batching="same_deadline"`` admission test requires. Jobs without a group
    are never batched under this rule.

    Predictions come from :meth:`DeviceState.predict`, so a ``safety_factor``
    above 1 makes the RB guard conservative against execution-time noise.
    """

    name = "edf_batch"
    _TOL = 1e-12

    def __init__(self, lookahead: int = 1, rule: str = "cross_deadline") -> None:
        if lookahead < 0:
            raise ValueError("lookahead must be >= 0")
        if rule not in ("cross_deadline", "same_deadline"):
            raise ValueError("rule must be 'cross_deadline' or 'same_deadline'")
        self.lookahead = lookahead
        self.rule = rule
        self.batch_rule = rule
        self.name = "edf_batch" if rule == "cross_deadline" else "edf_batch_same_deadline"

    def __repr__(self) -> str:
        return f"EDFBatch(lookahead={self.lookahead}, rule={self.rule!r})"

    def _same_deadline_batch(self, head: Job, ordered: Sequence[Job], cap: int) -> Dispatch:
        batch = [head]
        if head.group is not None:
            for j in ordered[1:]:
                if len(batch) >= cap:
                    break
                if (
                    j.group == head.group
                    and j.model == head.model
                    and abs(j.deadline - head.deadline) <= self._TOL
                    and abs(j.release - head.release) <= self._TOL
                ):
                    batch.append(j)
        return Dispatch(head.model, tuple(batch))

    @staticmethod
    def _serial_meets(start: float, jobs: Sequence[Job], state: DeviceState) -> list[bool]:
        t, out = start, []
        for j in jobs:
            t += state.predict(j.model, 1)
            out.append(t <= j.deadline)
        return out

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        ordered = sorted(queue, key=edf_key)
        head = ordered[0]
        model = head.model
        cap = state.max_batch(model)
        batch = [head]
        if cap == 1:
            return Dispatch(model, (head,))
        if self.rule == "same_deadline":
            return self._same_deadline_batch(head, ordered, cap)

        others = [j for j in ordered if j.model != model][: self.lookahead]
        baseline = self._serial_meets(now + state.predict(model, 1), others, state)
        finish = now + state.predict(model, 1)

        for cand in ordered[1:]:
            if len(batch) >= cap:
                break
            if cand.model != model:
                continue
            new_finish = now + state.predict(model, len(batch) + 1)
            if new_finish > cand.deadline:
                continue
            if any(finish <= j.deadline < new_finish for j in batch):
                break
            meets = self._serial_meets(new_finish, others, state)
            if any(base and not ok for base, ok in zip(baseline, meets)):
                break
            batch.append(cand)
            finish = new_finish
        return Dispatch(model, tuple(batch))


class DynamicBatchTimeout(Policy):
    """Approximation of Triton's dynamic batcher, as a datacenter baseline.

    Per model, requests queue in arrival order. A batch is released when a
    preferred batch size can be formed, or when the oldest request has waited
    ``max_queue_delay``; in the latter case everything queued (up to
    ``max_batch``) goes. Among models that are ready to fire, the one with the
    oldest request runs first. Deadlines are ignored, as in Triton.
    """

    name = "dynamic_batch"
    batch_rule = "cross_deadline"

    def __init__(
        self,
        max_queue_delay: float = 0.002,
        preferred_batch_sizes: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.max_queue_delay = max_queue_delay
        self.preferred = dict(preferred_batch_sizes or {})

    def __repr__(self) -> str:
        return f"DynamicBatchTimeout(max_queue_delay={self.max_queue_delay})"

    def choose(self, queue: Sequence[Job], now: float, state: DeviceState) -> Dispatch | Wait:
        if not queue:
            return Wait()
        by_model: dict[str, list[Job]] = {}
        for j in sorted(queue, key=lambda j: (j.release, j.id)):
            by_model.setdefault(j.model, []).append(j)

        best: tuple[float, int, str, list[Job]] | None = None
        wake = math.inf
        for model, jobs in by_model.items():
            cap = state.max_batch(model)
            prefs = sorted(p for p in self.preferred.get(model, (cap,)) if 1 <= p <= cap)
            formable = [p for p in prefs if p <= len(jobs)]
            oldest = jobs[0]
            if cap == 1:
                size = 1
            elif formable:
                size = formable[-1]
            elif now >= oldest.release + self.max_queue_delay:
                size = min(len(jobs), cap)
            else:
                wake = min(wake, oldest.release + self.max_queue_delay)
                continue
            key = (oldest.release, oldest.id, model, jobs[:size])
            if best is None or key[:2] < best[:2]:
                best = key
        if best is not None:
            return Dispatch(best[2], tuple(best[3]))
        return Wait(until=wake)
