from __future__ import annotations

import threading
import time

import pytest

from edgesched.metrics import summarize
from edgesched.model import ModelSpec, TaskSpec
from edgesched.policies import EDF, FIFO, DynamicBatchTimeout, EDFBatch
from edgesched.profile import LinearProfile
from edgesched.runtime import (
    AdmissionError,
    Arbiter,
    CallableBackend,
    DeadlineMissed,
    DirectLockExecutor,
    default_collate,
    default_split,
)

MS = 1e-3


class RecordingBackend:
    """Doubles each input; sleeps a fixed time per batch; records batch sizes."""

    def __init__(self, sleep: float = 0.002):
        self.sleep = sleep
        self.batches: list[list[int]] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, batch):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.sleep)
        self.batches.append(list(batch))
        with self._lock:
            self.active -= 1
        return [2 * x for x in batch]


def _models(max_batch=8):
    return {
        "mlp": ModelSpec("mlp", LinearProfile(2 * MS, 0.1 * MS), max_batch, max_batch > 1),
        "other": ModelSpec("other", LinearProfile(1 * MS)),
    }


def test_futures_resolve_with_correctly_split_outputs():
    rec = RecordingBackend(sleep=0.001)
    backends = {"mlp": CallableBackend(rec), "other": CallableBackend(lambda b: [x + 1 for x in b])}
    with Arbiter(_models(), backends, EDFBatch(), admission="off") as arb:
        arb.register_stream(TaskSpec("s", "mlp", 50 * MS))
        other = arb.register_stream(TaskSpec("o", "other", 50 * MS))
        futs = [arb.submit("s", i) for i in range(50)]
        assert [f.result(timeout=5) for f in futs] == [2 * i for i in range(50)]
        assert other.infer(41, timeout=5) == 42
    flat = [x for b in rec.batches for x in b]
    assert sorted(flat) == list(range(50))
    assert rec.max_active == 1


def test_batching_happens_under_concurrent_load():
    rec = RecordingBackend(sleep=0.005)
    models = _models()
    backends = {"mlp": CallableBackend(rec), "other": CallableBackend(lambda b: b)}
    arb = Arbiter(models, backends, EDFBatch(), admission="off").start()
    handles = [arb.register_stream(TaskSpec(f"agent{i}", "mlp", 20 * MS)) for i in range(6)]
    results: dict[int, list[int]] = {}

    def producer(i, h):
        results[i] = [h.infer(i * 100 + k, timeout=5) for k in range(15)]

    threads = [threading.Thread(target=producer, args=(i, h)) for i, h in enumerate(handles)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    arb.stop()
    for i in range(6):
        assert results[i] == [2 * (i * 100 + k) for k in range(15)]
    assert max(len(b) for b in rec.batches) > 1
    summary = summarize(arb.trace(), "edf_batch")
    assert summary.overall.completed == 90
    assert summary.mean_batch_size > 1.0


def test_unbatched_policy_runs_batch_size_one():
    rec = RecordingBackend(sleep=0.001)
    with Arbiter(
        _models(), {"mlp": CallableBackend(rec), "other": rec}, EDF(), admission="off"
    ) as a:
        a.register_stream(TaskSpec("s", "mlp", 10 * MS))
        futs = [a.submit("s", i) for i in range(10)]
        [f.result(timeout=5) for f in futs]
    assert all(len(b) == 1 for b in rec.batches)


def test_dynamic_batch_timeout_flushes_in_runtime():
    rec = RecordingBackend(sleep=0.0)
    policy = DynamicBatchTimeout(max_queue_delay=0.02, preferred_batch_sizes={"mlp": [4]})
    with Arbiter(
        _models(), {"mlp": CallableBackend(rec), "other": rec}, policy, admission="off"
    ) as a:
        a.register_stream(TaskSpec("s", "mlp", 100 * MS))
        t0 = time.perf_counter()
        assert a.infer("s", 3, timeout=5) == 6  # alone: waits for the queue delay
        assert time.perf_counter() - t0 >= 0.015
        futs = [a.submit("s", i) for i in range(4)]
        assert [f.result(timeout=5) for f in futs] == [0, 2, 4, 6]
    assert [len(b) for b in rec.batches] == [1, 4]


def test_drop_late_sets_deadline_missed():
    gate = threading.Event()

    def slow(batch):
        gate.wait(5)
        return list(batch)

    backends = {"mlp": CallableBackend(slow), "other": CallableBackend(lambda b: b)}
    with Arbiter(_models(1), backends, FIFO(), late_policy="drop", admission="off") as arb:
        arb.register_stream(TaskSpec("s", "mlp", 5 * MS))
        first = arb.submit("s", 1)
        time.sleep(0.01)
        doomed = arb.submit("s", 2)
        time.sleep(0.02)  # doomed's 5 ms deadline passes while the device is busy
        gate.set()
        assert first.result(timeout=5) == 1
        with pytest.raises(DeadlineMissed):
            doomed.result(timeout=5)
    assert summarize(arb.trace()).overall.dropped == 1


def test_backend_exception_propagates_to_all_futures():
    def boom(batch):
        raise RuntimeError("kernel failed")

    backends = {"mlp": CallableBackend(boom), "other": CallableBackend(boom)}
    with Arbiter(_models(), backends, EDFBatch(), admission="off") as arb:
        arb.register_stream(TaskSpec("s", "mlp", 10 * MS))
        fut = arb.submit("s", 1)
        with pytest.raises(RuntimeError, match="kernel failed"):
            fut.result(timeout=5)
        arb.register_stream(TaskSpec("t", "other", 10 * MS))
        with pytest.raises(RuntimeError, match="kernel failed"):
            arb.infer("t", 1, timeout=5)


def test_admission_enforce_and_warn():
    models = {
        "fast": ModelSpec("fast", LinearProfile(1 * MS)),
        "slow": ModelSpec("slow", LinearProfile(5 * MS)),
    }
    backends = {m: CallableBackend(lambda b: b) for m in models}
    arb = Arbiter(models, backends, EDF(), admission="enforce")
    arb.register_stream(TaskSpec("fast", "fast", 3 * MS))
    with pytest.raises(AdmissionError) as err:
        arb.register_stream(TaskSpec("slow", "slow", 50 * MS))
    assert "blocking" in str(err.value)
    assert "slow" not in arb.tasks
    warn = Arbiter(models, backends, EDF(), admission="warn")
    warn.register_stream(TaskSpec("fast", "fast", 3 * MS))
    with pytest.warns(UserWarning, match="fails admission"):
        warn.register_stream(TaskSpec("slow", "slow", 50 * MS))


def test_submit_requires_running_and_known_stream():
    arb = Arbiter(_models(), {m: CallableBackend(lambda b: b) for m in ("mlp", "other")}, EDF())
    arb.register_stream(TaskSpec("s", "mlp", 10 * MS))
    with pytest.raises(RuntimeError, match="not running"):
        arb.submit("s", 1)
    with pytest.raises(KeyError):
        arb.submit("missing", 1)
    with pytest.raises(ValueError, match="no backend"):
        Arbiter(_models(), {"mlp": CallableBackend(lambda b: b)}, EDF())


def test_online_adaptation_tracks_slow_backend():
    rec = RecordingBackend(sleep=0.010)  # profile says ~2 ms
    with Arbiter(
        _models(), {"mlp": CallableBackend(rec), "other": rec}, EDF(), admission="off"
    ) as a:
        a.register_stream(TaskSpec("s", "mlp", 100 * MS))
        for i in range(20):
            a.infer("s", i, timeout=5)
    assert a.state.scale["mlp"] > 2.0


def test_direct_lock_executor_serializes_and_records():
    rec = RecordingBackend(sleep=0.002)
    ex = DirectLockExecutor({"mlp": CallableBackend(rec)}).start()
    h = ex.register_stream(TaskSpec("s", "mlp", 10 * MS))
    threads = [threading.Thread(target=lambda: [h.infer(1) for _ in range(5)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rec.max_active == 1
    assert summarize(ex.trace()).overall.completed == 20


def test_collate_split_helpers():
    np = pytest.importorskip("numpy")
    batch = default_collate([np.zeros(3), np.ones(3)])
    assert batch.shape == (2, 3)
    parts = default_split(batch, 2)
    assert len(parts) == 2 and parts[1].sum() == 3
    assert default_collate([1, 2]) == [1, 2]
    with pytest.raises(ValueError):
        default_split([1, 2, 3], 2)


def test_torch_backend_batches_real_module():
    torch = pytest.importorskip("torch")
    from edgesched.runtime import TorchBackend

    torch.manual_seed(0)
    module = torch.nn.Linear(4, 2)
    backend = TorchBackend(module, device="cpu")
    xs = [torch.randn(4) for _ in range(3)]
    outs = backend.run_batch(xs)
    with torch.inference_mode():
        for x, y in zip(xs, outs):
            assert torch.allclose(module(x), y, atol=1e-6)


def test_torch_backend_waits_for_inputs_produced_on_default_stream():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from edgesched.runtime import TorchBackend

    backend = TorchBackend(torch.nn.Identity(), device="cuda")
    x = torch.zeros(1024, device="cuda")
    torch.cuda._sleep(20_000_000)  # keep the default stream busy so fill_ is still queued
    x.fill_(1.0)
    assert backend.run_batch([x])[0].sum().item() == 1024


def test_tensorrt_backend_runs_dynamic_batch_engine(tmp_path):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    np = pytest.importorskip("numpy")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from edgesched.runtime import TensorRTBackend

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    x = net.add_input("x", trt.float32, (-1, 4))
    two = net.add_constant((1, 4), trt.Weights(np.full((1, 4), 2.0, np.float32)))
    y = net.add_elementwise(x, two.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    y.name = "y"
    net.mark_output(y)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 4), (4, 4), (8, 4))
    config.add_optimization_profile(profile)
    engine = tmp_path / "double.engine"
    engine.write_bytes(bytes(builder.build_serialized_network(net, config)))

    backend = TensorRTBackend(str(engine))
    assert backend.max_batch == 8
    for n in (1, 3, 8):
        xs = [torch.randn(4) for _ in range(n)]
        outs = backend.run_batch(xs)
        assert all(torch.allclose(o, 2 * x) for o, x in zip(outs, xs))
    with pytest.raises(ValueError, match="max batch"):
        backend.run_batch([torch.randn(4)] * 9)
