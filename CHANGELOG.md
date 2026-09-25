# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `TensorRTBackend` for single-input, single-output engines with a dynamic batch profile.
- `examples/runtime_demo.py` (renamed from `cpu_runtime_demo.py`): `--device cuda`,
  `--backend tensorrt`, a round-robin mode, and a report comparing dispatch latencies
  under contention with the profile `C(b)`. Profiles now time the backend's full
  `run_batch` path instead of the forward pass alone.
- RTX 4080 runtime results (PyTorch and TensorRT) in `docs/results/`.

### Fixed
- `TorchBackend` on CUDA ran on a non-blocking stream without waiting for the
  caller's stream, so it could read inputs before the kernels producing them finished.

## [0.1.0] - 2026-09-16

### Added
- Task model: `ModelSpec`, `TaskSpec` (period, deadline, priority, phase, jitter,
  criticality, periodic/sporadic, release group), `Job` with release/start/finish timestamps.
- Latency profiles: linear `a + c*b`, interpolated tables, lognormal-tail noise,
  `measure()` profiler for any callable, JSON save/load.
- Policies shared by simulator and runtime: `RoundRobin`, `FIFO`, `FixedPriority`,
  `EDF`, `EDFBatch` with two batching rules (cross-deadline with a lookahead
  guard; same-deadline within a declared release group), `DynamicBatchTimeout`
  (Triton-style baseline).
- Admission control: utilization test and non-preemptive EDF demand-bound tests
  for no batching, same-deadline batching and cross-deadline batching (with the
  rider term; the blocking-only variant is unsound and its counterexample is a
  regression test).
- Seeded discrete-event simulator of one non-preemptive accelerator with drop-late
  and run-late modes.
- Runtime `Arbiter` (futures, batching, drop-late, admission, online latency
  correction), `DirectLockExecutor` baseline, callable/PyTorch/ONNX Runtime
  backends, TensorRT interface stub.
- Metrics: per-stream miss rate, response-time percentiles, lateness, utilization,
  batch size, throughput; table, JSON and Prometheus text output.
- CLI: `edgesched simulate | bench | analyze | profile`.
- Workloads `multi_agent_cell`, `vla_plus_control`, `synthetic_mix`; simulated
  sweep results and figures; CPU runtime demo; ROS 2 node sketch; pilot_core
  integration guide; Jetson `trtexec` profiling script.
