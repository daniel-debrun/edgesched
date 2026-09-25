"""In-process arbiter that owns an accelerator.

Producer threads call :meth:`Arbiter.submit` (returns a future) or
:meth:`Arbiter.infer` (blocks). One dispatcher thread holds the queue, asks the
policy what to run, and executes the chosen batch on the model's backend. The
policy objects are the same ones the simulator uses.

Only one batch runs at a time. That is the point: on a single edge GPU,
concurrent submissions from several threads are interleaved by the driver
anyway, but in an order nobody chose.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from edgesched.analysis import AnalysisResult, analyze
from edgesched.metrics import DispatchRecord, Trace
from edgesched.model import Job, ModelSpec, TaskSpec, rate_monotonic_priorities
from edgesched.policies import DeviceState, Dispatch, Policy
from edgesched.runtime.backends import Backend


class DeadlineMissed(Exception):
    """Set on a future whose job was dropped because its deadline passed while queued."""


class AdmissionError(RuntimeError):
    def __init__(self, result: AnalysisResult):
        super().__init__("stream rejected by admission control:\n" + str(result))
        self.result = result


@dataclass(frozen=True)
class StreamHandle:
    """Convenience handle bound to one registered stream."""

    executor: Any
    task: TaskSpec

    def submit(self, payload: Any, release: float | None = None) -> Future[Any]:
        return self.executor.submit(self.task.name, payload, release=release)

    def infer(self, payload: Any, release: float | None = None, timeout: float | None = None):
        return self.executor.infer(self.task.name, payload, release=release, timeout=timeout)


class _TraceRecorder:
    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.dispatches: list[DispatchRecord] = []
        self.t0 = clock()

    def reset(self) -> None:
        with self.lock:
            self.jobs, self.dispatches, self.t0 = [], [], self.clock()

    def snapshot(self) -> Trace:
        with self.lock:
            now = self.clock()
            return Trace(list(self.jobs), list(self.dispatches), start=self.t0, end=now)


class Arbiter:
    """Deadline-aware dispatcher for real backends.

    Args:
        models: Model specs (latency profiles are used for predictions).
        backends: One backend per model name.
        policy: Any :class:`~edgesched.policies.Policy`.
        late_policy: ``"run"`` executes late jobs anyway; ``"drop"`` fails their
            future with :class:`DeadlineMissed` once the deadline has passed.
        admission: ``"off"``, ``"warn"`` or ``"enforce"`` when registering streams.
        adapt: Maintain a per-model EWMA of observed/predicted latency and
            feed it to the policy's predictions.
    """

    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        backends: Mapping[str, Backend],
        policy: Policy,
        *,
        late_policy: str = "run",
        safety_factor: float = 1.0,
        admission: str = "warn",
        adapt: bool = True,
        adapt_alpha: float = 0.1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        missing = set(models) - set(backends)
        if missing:
            raise ValueError(f"no backend for models: {sorted(missing)}")
        if late_policy not in ("run", "drop"):
            raise ValueError("late_policy must be 'run' or 'drop'")
        if admission not in ("off", "warn", "enforce"):
            raise ValueError("admission must be 'off', 'warn' or 'enforce'")
        self.models = dict(models)
        self.backends = dict(backends)
        self.policy = policy
        self.late_policy = late_policy
        self.admission = admission
        self.adapt = adapt
        self.adapt_alpha = adapt_alpha
        self.clock = clock
        self.state = DeviceState(self.models, safety_factor=safety_factor)
        self.tasks: dict[str, TaskSpec] = {}
        self._priorities: dict[str, int] = {}
        self._queue: list[Job] = []
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._recorder = _TraceRecorder(clock)

    def register_stream(self, task: TaskSpec) -> StreamHandle:
        if task.model not in self.models:
            raise ValueError(f"unknown model {task.model!r}")
        with self._cond:
            if task.name in self.tasks:
                raise ValueError(f"stream {task.name!r} already registered")
            if self.admission != "off":
                result = analyze(
                    self.models, [*self.tasks.values(), task], batching=self.policy.batch_rule
                )
                if not result.admitted:
                    if self.admission == "enforce":
                        raise AdmissionError(result)
                    warnings.warn(
                        f"stream {task.name!r} fails admission: {'; '.join(result.reasons)}",
                        stacklevel=2,
                    )
            self.tasks[task.name] = task
            self._priorities = rate_monotonic_priorities(list(self.tasks.values()))
        return StreamHandle(self, task)

    def submit(self, stream: str, payload: Any, release: float | None = None) -> Future[Any]:
        """Enqueue one request. ``release`` defaults to now (on the arbiter clock)."""
        task = self.tasks[stream]
        fut: Future[Any] = Future()
        rel = self.clock() if release is None else release
        job = Job(
            stream=stream,
            model=task.model,
            release=rel,
            deadline=rel + task.relative_deadline,
            priority=self._priorities[stream],
            criticality=task.criticality,
            group=task.group,
            payload=payload,
            future=fut,
        )
        with self._cond:
            if not self._running:
                raise RuntimeError(
                    "arbiter is not running; call start() or use it as a context manager"
                )
            self._queue.append(job)
            self._cond.notify()
        with self._recorder.lock:
            self._recorder.jobs.append(job)
        return fut

    def infer(
        self, stream: str, payload: Any, release: float | None = None, timeout: float | None = None
    ) -> Any:
        return self.submit(stream, payload, release=release).result(timeout)

    def start(self) -> Arbiter:
        with self._cond:
            if self._running:
                return self
            self._running = True
        self.policy.reset()
        self._recorder.reset()
        self._thread = threading.Thread(target=self._loop, name="edgesched-dispatcher", daemon=True)
        self._thread.start()
        return self

    def stop(self, drain: bool = True, timeout: float | None = 10.0) -> None:
        """Stop accepting work. Queued jobs run first with ``drain``, else they are cancelled."""
        with self._cond:
            self._running = False
            if not drain:
                for j in self._queue:
                    if j.future is not None:
                        j.future.cancel()
                    j.dropped = True
                self._queue.clear()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def __enter__(self) -> Arbiter:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def trace(self) -> Trace:
        return self._recorder.snapshot()

    def reset_trace(self) -> None:
        self._recorder.reset()

    def _drop_late(self, now: float) -> None:
        kept = []
        for j in self._queue:
            if j.deadline <= now:
                j.dropped = True
                if j.future is not None:
                    j.future.set_exception(DeadlineMissed(f"{j.stream} job {j.id} dropped"))
            else:
                kept.append(j)
        self._queue = kept

    def _next_decision(self) -> Dispatch | None:
        with self._cond:
            while True:
                if not self._queue:
                    if not self._running:
                        return None
                    self._cond.wait()
                    continue
                now = self.clock()
                if self.late_policy == "drop":
                    self._drop_late(now)
                    if not self._queue:
                        continue
                decision = self.policy.choose(list(self._queue), now, self.state)
                if isinstance(decision, Dispatch):
                    chosen = {id(j) for j in decision.jobs}
                    self._queue = [j for j in self._queue if id(j) not in chosen]
                    return decision
                if decision.until is None:
                    if not self._running:
                        for j in self._queue:
                            if j.future is not None:
                                j.future.cancel()
                        self._queue.clear()
                        return None
                    self._cond.wait()
                else:
                    self._cond.wait(max(0.0, decision.until - now))

    def _loop(self) -> None:
        while True:
            decision = self._next_decision()
            if decision is None:
                return
            self._execute(decision)

    def _execute(self, decision: Dispatch) -> None:
        jobs = decision.jobs
        n = len(jobs)
        start = self.clock()
        try:
            outputs = self.backends[decision.model].run_batch([j.payload for j in jobs])
            if len(outputs) != n:
                raise ValueError(f"backend returned {len(outputs)} outputs for {n} inputs")
            error: BaseException | None = None
        except Exception as e:
            outputs, error = [], e
        end = self.clock()

        for i, j in enumerate(jobs):
            j.started, j.finished, j.batch_size = start, end, n
            j.payload = None
            if j.future is None:
                continue
            if error is None:
                j.future.set_result(outputs[i])
            else:
                j.future.set_exception(error)
        with self._recorder.lock:
            self._recorder.dispatches.append(DispatchRecord(decision.model, start, end, n))
        self.state.last_model = decision.model
        self.state.dispatches += 1
        if self.adapt and error is None:
            predicted = self.models[decision.model].latency(n)
            ratio = min(4.0, max(0.25, (end - start) / predicted))
            old = self.state.scale.get(decision.model, 1.0)
            self.state.scale[decision.model] = old + self.adapt_alpha * (ratio - old)


class DirectLockExecutor:
    """Baseline: every caller runs its model directly under one global lock.

    This is the "threads and a mutex around the GPU" pattern. There is no
    queue, no batching and no deadline awareness; whichever thread wins the
    lock goes next. Exposes the same ``register_stream``/``infer``/``trace``
    surface as :class:`Arbiter` so the two can be compared with the same metrics.
    """

    def __init__(
        self, backends: Mapping[str, Backend], clock: Callable[[], float] = time.perf_counter
    ):
        self.backends = dict(backends)
        self.clock = clock
        self.tasks: dict[str, TaskSpec] = {}
        self._lock = threading.Lock()
        self._recorder = _TraceRecorder(clock)

    def register_stream(self, task: TaskSpec) -> StreamHandle:
        self.tasks[task.name] = task
        return StreamHandle(self, task)

    def submit(self, stream: str, payload: Any, release: float | None = None) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            fut.set_result(self.infer(stream, payload, release=release))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def infer(
        self, stream: str, payload: Any, release: float | None = None, timeout: float | None = None
    ) -> Any:
        task = self.tasks[stream]
        rel = self.clock() if release is None else release
        job = Job(
            stream, task.model, rel, rel + task.relative_deadline, criticality=task.criticality
        )
        with self._lock:
            start = self.clock()
            out = self.backends[task.model].run_batch([payload])[0]
            end = self.clock()
        job.started, job.finished, job.batch_size = start, end, 1
        with self._recorder.lock:
            self._recorder.jobs.append(job)
            self._recorder.dispatches.append(DispatchRecord(task.model, start, end, 1))
        return out

    def start(self) -> DirectLockExecutor:
        self._recorder.reset()
        return self

    def stop(self, drain: bool = True, timeout: float | None = None) -> None:
        return None

    def __enter__(self) -> DirectLockExecutor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def trace(self) -> Trace:
        return self._recorder.snapshot()

    def reset_trace(self) -> None:
        self._recorder.reset()
