"""Real-runtime demo: PyTorch or TensorRT models, several producer threads, one arbiter.

Streams (mirroring workloads/multi_agent_cell.yaml, but with real models):
  cam_front, cam_rear  conv encoder      30 Hz, deadline 33 ms, shared model (batchable)
  agent0..agent3       MAPPO-style MLP   50 Hz, deadline 20 ms, shared model (batchable)
  planner              wide MLP           5 Hz, deadline 200 ms

On CUDA the CNN input is larger and the planner runs over a sequence of tokens,
so that the models load a desktop GPU instead of finishing in microseconds.

Each stream is a thread that behaves like a robot worker: at every tick it
calls ``infer`` and blocks until the result is back. If a call overruns so far
that whole ticks pass, those ticks are recorded as dropped (missed) jobs.

Modes compared:
  naive      every thread calls its model directly under one global lock
  round_robin  Arbiter + round-robin over streams (polling-loop baseline)
  fifo       Arbiter + FIFO
  edf        Arbiter + non-preemptive EDF
  edf_batch  Arbiter + EDFBatch (cross-deadline rule)
  edf_batch_same_deadline  Arbiter + EDFBatch batching only the actors' shared tick

Latency profiles C(b) are measured at startup by timing each backend's
``run_batch`` in isolation (collate, host-to-device copy, forward, copy back),
which is what the arbiter times per dispatch. The script then reports
how the dispatch latencies observed under contention compare with C(b).

    python examples/runtime_demo.py --duration 20 --json results/cpu_demo.json
    python examples/runtime_demo.py --device cuda --backend tensorrt --engine-dir engines
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

from edgesched import zoo
from edgesched.metrics import Summary, percentile, summarize
from edgesched.model import Criticality, Job, ModelSpec, TaskSpec
from edgesched.policies import make_policy
from edgesched.profile import measure
from edgesched.runtime import (
    Arbiter,
    DeadlineMissed,
    DirectLockExecutor,
    TensorRTBackend,
    TorchBackend,
)

MS = 1e-3
PLANNER_TOKENS = 4096


def build_models(device: str):
    if device == "cpu":
        return {
            "cnn": (*zoo.tiny_cnn(image_size=128, width=32), 4, True),
            "actor": (*zoo.actor_mlp(obs_dim=64, hidden=512), 8, True),
            "planner": (*zoo.planner_mlp(in_dim=1024, hidden=2048), 1, False),
        }
    planner, make_tokens = zoo.planner_mlp(in_dim=1024, hidden=4096)
    return {
        "cnn": (*zoo.tiny_cnn(image_size=896, width=128), 4, True),
        "actor": (*zoo.actor_mlp(obs_dim=64, hidden=512), 8, True),
        "planner": (
            planner,
            lambda b: make_tokens(b * PLANNER_TOKENS).reshape(b, PLANNER_TOKENS, -1),
            1,
            False,
        ),
    }


def build_engine(module, make_input, max_batch: int, path: Path) -> None:
    """Export to ONNX and build a TensorRT engine with a 1..max_batch profile."""
    import tensorrt as trt

    x = make_input(max_batch)
    onnx_path = path.with_suffix(".onnx")
    dynamic = ({0: torch.export.Dim("batch", min=1, max=max_batch)},) if max_batch > 1 else None
    torch.onnx.export(
        module,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=["y"],
        dynamic_shapes=dynamic,
        dynamo=True,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(
            f"ONNX parse failed: {[parser.get_error(i) for i in range(parser.num_errors)]}"
        )
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    rest = tuple(x.shape[1:])
    name = network.get_input(0).name
    profile.set_shape(name, (1, *rest), (max_batch, *rest), (max_batch, *rest))
    config.add_optimization_profile(profile)
    path.write_bytes(bytes(builder.build_serialized_network(network, config)))


def make_backends(raw, device: str, backend: str, engine_dir: Path | None):
    if backend == "torch":
        return {name: TorchBackend(module, device=device) for name, (module, *_) in raw.items()}
    assert engine_dir is not None, "--engine-dir is required with --backend tensorrt"
    engine_dir.mkdir(parents=True, exist_ok=True)
    backends = {}
    for name, (module, make_input, max_batch, batchable) in raw.items():
        path = engine_dir / f"{name}.engine"
        if not path.exists():
            build_engine(module, make_input, max_batch if batchable else 1, path)
        backends[name] = TensorRTBackend(str(path), device=device)
    return backends


def build_streams(rate_scale: float) -> list[TaskSpec]:
    hi = Criticality.HIGH
    streams = [
        TaskSpec("cam_front", "cnn", 1 / (30 * rate_scale), 33 * MS, criticality=hi),
        TaskSpec("cam_rear", "cnn", 1 / (30 * rate_scale), 33 * MS, phase=11 * MS, criticality=hi),
        TaskSpec("planner", "planner", 1 / (5 * rate_scale), 200 * MS, phase=3 * MS),
    ]
    streams += [
        TaskSpec(
            f"agent{i}",
            "actor",
            1 / (50 * rate_scale),
            20 * MS,
            phase=5 * MS,
            criticality=hi,
            group="actors",
        )
        for i in range(4)
    ]
    return streams


def profile_models(raw, backends, meta, profile_dir: Path | None) -> dict[str, ModelSpec]:
    specs = {}
    for name, (_module, make_input, max_batch, batchable) in raw.items():
        sizes = [b for b in (1, 2, 4, 8) if b <= max_batch]
        result = measure(
            backends[name].run_batch,
            lambda b, _mk=make_input: list(_mk(b).unbind(0)),
            sizes,
            name=name,
            warmup=10,
            iters=40,
        )
        result.meta.update(meta)
        if profile_dir is not None:
            profile_dir.mkdir(parents=True, exist_ok=True)
            result.save(profile_dir / f"{meta['prefix']}_{name}.json")
        stats = result.stats()
        print(
            f"profile {name:8s} "
            + "  ".join(f"b={b}: {s['p50'] * 1e3:.2f} ms" for b, s in stats.items())
        )
        specs[name] = ModelSpec(name, result.table(), max_batch=max_batch, batchable=batchable)
    return specs


def run_mode(mode, specs, raw, backends, streams, duration, warmup, late_policy):
    if mode == "naive":
        executor = DirectLockExecutor(backends)
    else:
        executor = Arbiter(
            specs, backends, make_policy(mode), late_policy=late_policy, admission="off"
        )
    executor.start()
    handles = {t.name: executor.register_stream(t) for t in streams}
    inputs = {t.name: raw[t.model][1](1)[0] for t in streams}
    start = time.perf_counter() + 0.05
    measure_from = start + warmup
    end = measure_from + duration
    skipped: list[Job] = []
    lock = threading.Lock()

    def worker(task: TaskSpec) -> None:
        t0 = start + task.phase
        k = 0
        while True:
            tick = t0 + k * task.period
            if tick >= end:
                return
            delay = tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with contextlib.suppress(DeadlineMissed):  # already recorded as dropped
                handles[task.name].infer(inputs[task.name], release=tick)
            now = time.perf_counter()
            nxt = max(k + 1, math.ceil((now - t0) / task.period))
            for missed_k in range(k + 1, nxt):
                rel = t0 + missed_k * task.period
                if measure_from <= rel < end:
                    job = Job(task.name, task.model, rel, rel + task.relative_deadline)
                    job.dropped = True
                    with lock:
                        skipped.append(job)
            k = nxt

    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in streams]
    for th in threads:
        th.start()
    time.sleep(max(0.0, measure_from - time.perf_counter()))
    executor.reset_trace()
    for th in threads:
        th.join()
    trace = executor.trace()
    executor.stop()
    trace.jobs = [j for j in trace.jobs if j.release >= measure_from] + skipped
    trace.start, trace.end = measure_from, end
    latencies: dict[tuple[str, int], list[float]] = defaultdict(list)
    for d in trace.dispatches:
        if d.start >= measure_from:
            latencies[(d.model, d.batch_size)].append(d.end - d.start)
    return summarize(trace, mode), latencies


def _hi_miss(s: Summary) -> float:
    hi = [x for x in s.streams if x.name != "planner"]
    return sum(x.missed for x in hi) / max(1, sum(x.released for x in hi))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--duration", type=float, default=20.0, help="measured seconds per mode")
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--backend", choices=["torch", "tensorrt"], default="torch")
    p.add_argument("--engine-dir", type=Path, default=None, help="TensorRT engine cache")
    p.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    p.add_argument("--rate-scale", type=float, default=1.0, help="multiply all stream rates")
    p.add_argument(
        "--modes", default="naive,round_robin,fifo,edf,edf_batch,edf_batch_same_deadline"
    )
    p.add_argument("--reps", type=int, default=3, help="interleaved repetitions per mode")
    p.add_argument(
        "--late-policy",
        choices=["drop", "run"],
        default="drop",
        help="arbiter modes only; the naive lock has no queue to drop from",
    )
    p.add_argument("--profile-dir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    raw = build_models(args.device)
    backends = make_backends(raw, args.device, args.backend, args.engine_dir)
    meta = {
        "prefix": args.device if args.backend == "torch" else f"{args.device}_{args.backend}",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": args.device,
        "backend": args.backend,
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if args.device == "cuda":
        meta.update({"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
    if args.backend == "tensorrt":
        import tensorrt

        meta["tensorrt"] = tensorrt.__version__
    specs = profile_models(raw, backends, meta, args.profile_dir)
    streams = build_streams(args.rate_scale)
    u = sum(specs[t.model].latency(1) / t.period for t in streams)
    print(f"nominal utilization from measured p50 profiles: {u:.2f}\n")

    load_before = os.getloadavg()
    modes = args.modes.split(",")
    runs: dict[str, list[Summary]] = {m: [] for m in modes}
    latencies: dict[tuple[str, int], list[float]] = defaultdict(list)
    for rep in range(args.reps):
        for mode in modes:
            s, lat = run_mode(
                mode, specs, raw, backends, streams, args.duration, args.warmup, args.late_policy
            )
            runs[mode].append(s)
            for key, values in lat.items():
                latencies[key] += values
            print(f"rep {rep} {mode}: miss {100 * s.overall.miss_rate:.2f}%")
    load_after = os.getloadavg()
    print(f"\nload average (1/5/15 min) before {load_before}, after {load_after}\n")

    rows = {}
    for mode, reps in runs.items():
        median_run = sorted(reps, key=lambda x: x.overall.miss_rate)[len(reps) // 2]
        print(median_run.table())
        print()
        rows[mode] = {
            "miss_pct": [100 * x.overall.miss_rate for x in reps],
            "hi_miss_pct": [100 * _hi_miss(x) for x in reps],
            "agent_p99_ms": [
                max(y.p99_ms for y in x.streams if y.name.startswith("agent")) for x in reps
            ],
            "cam_p99_ms": [
                max(y.p99_ms for y in x.streams if y.name.startswith("cam")) for x in reps
            ],
            "utilization": [x.utilization for x in reps],
            "mean_batch": [x.mean_batch_size for x in reps],
        }

    print(f"median over {args.reps} interleaved reps of {args.duration:g} s each")
    print(
        f"{'mode':<25}{'miss%':>8}{'hi-crit miss%':>15}{'agent p99 ms':>14}"
        f"{'cam p99 ms':>12}{'util':>7}{'batch':>7}"
    )
    for mode, r in rows.items():
        med = {k: statistics.median(v) for k, v in r.items()}
        print(
            f"{mode:<25}{med['miss_pct']:>8.2f}{med['hi_miss_pct']:>15.2f}"
            f"{med['agent_p99_ms']:>14.2f}{med['cam_p99_ms']:>12.2f}"
            f"{med['utilization']:>7.2f}{med['mean_batch']:>7.2f}"
        )

    print("\ndispatch latency under contention vs profile C(b), all modes and reps")
    print(f"{'model':<9}{'b':>3}{'C(b) ms':>10}{'p50 ms':>9}{'p99 ms':>9}{'max ms':>9}{'n':>7}")
    profile_fit = []
    for (model, b), values in sorted(latencies.items()):
        row = {
            "model": model,
            "batch": b,
            "predicted_ms": specs[model].latency(b) / MS,
            "p50_ms": percentile(values, 50) / MS,
            "p99_ms": percentile(values, 99) / MS,
            "max_ms": max(values) / MS,
            "n": len(values),
        }
        profile_fit.append(row)
        print(
            f"{model:<9}{b:>3}{row['predicted_ms']:>10.3f}{row['p50_ms']:>9.3f}"
            f"{row['p99_ms']:>9.3f}{row['max_ms']:>9.3f}{row['n']:>7}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {k: str(v) for k, v in vars(args).items()},
            "meta": meta,
            "profile_fit": profile_fit,
            "nominal_utilization": u,
            "load_average_before": load_before,
            "load_average_after": load_after,
            "cpu_count": os.cpu_count(),
            "runs": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
