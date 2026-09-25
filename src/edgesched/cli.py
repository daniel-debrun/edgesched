"""Command-line interface: ``edgesched {simulate,bench,analyze,profile}``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from edgesched.analysis import analyze, utilization
from edgesched.config import load_workload
from edgesched.metrics import prometheus_text, summarize
from edgesched.policies import POLICIES
from edgesched.sim import simulate


def _policies(names: Sequence[str]) -> list[str]:
    if not names or "all" in names:
        return list(POLICIES)
    unknown = [n for n in names if n not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies {unknown}; choose from {sorted(POLICIES)}")
    return list(names)


def cmd_simulate(args: argparse.Namespace) -> int:
    wl = load_workload(args.workload)
    if args.scale != 1.0:
        wl = wl.scaled(args.scale)
    results = {}
    for name in _policies(args.policy):
        trace = simulate(
            wl,
            name,
            duration=args.duration,
            seed=args.seed,
            late_policy=args.late_policy,
            safety_factor=args.safety_factor,
        )
        summary = summarize(trace, name)
        results[name] = summary.to_dict()
        if args.prometheus:
            print(prometheus_text(summary), end="")
        else:
            print(summary.table())
            print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    wl = load_workload(args.workload)
    if args.scale != 1.0:
        wl = wl.scaled(args.scale)
    result = analyze(wl.models, wl.tasks, batching=args.batching)
    print(f"workload: {wl.name} ({len(wl.tasks)} streams, batching={args.batching})")
    print(result)
    return 0 if result.admitted else 2


def cmd_bench(args: argparse.Namespace) -> int:
    from edgesched.bench import aggregate, markdown_table, run_benchmark, utilization_grid

    utils = utilization_grid(args.util_min, args.util_max, args.util_step)
    seeds = list(range(args.seeds))
    for path in args.workloads:
        points, wl = run_benchmark(
            path,
            _policies(args.policies),
            utils,
            seeds,
            args.duration,
            args.jobs,
            args.out_dir,
            None if args.no_plot else args.figure_dir,
        )
        u0 = utilization(wl.models, wl.tasks)
        print(f"## {wl.name} (nominal U0={u0:.3f}); miss rate %, * = admitted by analysis")
        print(markdown_table(aggregate(points)))
        print()
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from edgesched.profile import measure

    module_name, _, attr = args.target.partition(":")
    if not attr:
        raise SystemExit("--target must look like package.module:factory")
    factory = getattr(importlib.import_module(module_name), attr)
    model, make_input = factory()
    fn, sync, meta = model, None, {"target": args.target}
    if type(model).__module__.startswith("torch"):
        import torch

        if args.threads:
            torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        model = model.to(device).eval()
        base_input = make_input
        if device.type == "cuda":
            sync = torch.cuda.synchronize

        def make_input(b: int, _mk=base_input):
            return _mk(b).to(device)

        def fn(x, _m=model):
            with torch.inference_mode():
                return _m(x)

        meta.update(
            {"torch": torch.__version__, "device": str(device), "threads": torch.get_num_threads()}
        )
    sizes = [int(s) for s in args.batch_sizes.split(",")]
    result = measure(
        fn,
        make_input,
        sizes,
        name=args.name or attr,
        warmup=args.warmup,
        iters=args.iters,
        quantile=args.quantile,
        sync=sync,
    )
    result.meta.update(meta)
    for b, st in result.stats().items():
        print(f"b={b:<3} p50 {st['p50'] * 1e3:8.3f} ms  p99 {st['p99'] * 1e3:8.3f} ms")
    lin = result.linear()
    print(f"fit: latency(b) = {lin.a * 1e3:.3f} + {lin.c * 1e3:.3f} * b  ms")
    if args.out:
        result.save(args.out)
        print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgesched", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="simulate a workload under one or more policies")
    s.add_argument("workload")
    s.add_argument("--policy", "-p", action="append", default=[], help="policy name or 'all'")
    s.add_argument("--duration", type=float)
    s.add_argument("--seed", type=int)
    s.add_argument("--scale", type=float, default=1.0, help="multiply all latency profiles")
    s.add_argument("--late-policy", choices=["drop", "run"])
    s.add_argument("--safety-factor", type=float)
    s.add_argument("--json", help="write summaries to this path")
    s.add_argument("--prometheus", action="store_true", help="print Prometheus text format")
    s.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="schedulability / admission test")
    a.add_argument("workload")
    a.add_argument(
        "--batching",
        choices=["none", "same_deadline", "cross_deadline"],
        default="none",
        help="batching rule to certify (edf: none, edf_batch: cross_deadline)",
    )
    a.add_argument("--scale", type=float, default=1.0)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("bench", help="sweep utilization and compare policies")
    b.add_argument("workloads", nargs="+")
    b.add_argument("--policies", nargs="*", default=["all"])
    b.add_argument("--util-min", type=float, default=0.3)
    b.add_argument("--util-max", type=float, default=1.2)
    b.add_argument("--util-step", type=float, default=0.1)
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--duration", type=float)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--out-dir", default="results")
    b.add_argument("--figure-dir", default="docs/figures")
    b.add_argument("--no-plot", action="store_true")
    b.set_defaults(func=cmd_bench)

    pr = sub.add_parser("profile", help="measure a model across batch sizes and fit a profile")
    pr.add_argument(
        "--target", required=True, help="module:factory returning (callable, make_input)"
    )
    pr.add_argument("--batch-sizes", default="1,2,4,8")
    pr.add_argument("--iters", type=int, default=30)
    pr.add_argument("--warmup", type=int, default=5)
    pr.add_argument("--quantile", type=float, default=0.5)
    pr.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    pr.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    pr.add_argument("--name")
    pr.add_argument("--out")
    pr.set_defaults(func=cmd_profile)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
