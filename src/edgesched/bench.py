"""Utilization sweeps over workloads and policies, with tables and plots.

Utilization is swept by scaling every latency profile in the workload by
``target_u / U0``, where ``U0`` is the workload's nominal unbatched
utilization. Rates and deadlines stay fixed, so blocking grows with load the
way it would on a slower device or with bigger models.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from edgesched.analysis import BATCH_RULES, analyze, utilization
from edgesched.config import Workload, load_workload
from edgesched.metrics import summarize
from edgesched.model import Criticality
from edgesched.policies import POLICIES
from edgesched.sim import simulate


@dataclass(frozen=True)
class BenchPoint:
    workload: str
    policy: str
    target_util: float
    scale: float
    seed: int
    miss_rate: float
    high_crit_miss_rate: float
    p99_ms: float
    measured_util: float
    mean_batch: float
    admitted: dict[str, bool]


def _run_one(args: tuple[str, str, float, int, float]) -> BenchPoint:
    path, policy, target_u, seed, duration = args
    base = load_workload(path)
    scale = target_u / utilization(base.models, base.tasks)
    wl = base.scaled(scale)
    trace = simulate(wl, policy, seed=seed, duration=duration)
    summary = summarize(trace, policy)
    hi_streams = {t.name for t in wl.tasks if t.criticality is Criticality.HIGH}
    hi_stats = [s for s in summary.streams if s.name in hi_streams]
    hi_released = sum(s.released for s in hi_stats)
    hi_missed = sum(s.missed for s in hi_stats)
    return BenchPoint(
        workload=wl.name,
        policy=policy,
        target_util=round(target_u, 4),
        scale=scale,
        seed=seed,
        miss_rate=summary.overall.miss_rate,
        high_crit_miss_rate=hi_missed / hi_released if hi_released else math.nan,
        p99_ms=summary.overall.p99_ms,
        measured_util=summary.utilization,
        mean_batch=summary.mean_batch_size,
        admitted={r: analyze(wl.models, wl.tasks, batching=r).admitted for r in BATCH_RULES},
    )


def utilization_grid(lo: float, hi: float, step: float) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 6) for i in range(n + 1)]


def sweep(
    workload_path: str | Path,
    policies: Sequence[str],
    utils: Sequence[float],
    seeds: Sequence[int] = (0, 1, 2),
    duration: float | None = None,
    jobs: int = 1,
) -> list[BenchPoint]:
    path = str(workload_path)
    dur = duration if duration is not None else load_workload(path).sim.duration
    tasks = [(path, p, u, s, dur) for u in utils for p in policies for s in seeds]
    if jobs <= 1:
        return [_run_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_run_one, tasks, chunksize=4))


@dataclass(frozen=True)
class AggregateRow:
    policy: str
    target_util: float
    miss_rate: float
    miss_rate_min: float
    miss_rate_max: float
    high_crit_miss_rate: float
    p99_ms: float
    mean_batch: float
    admitted: dict[str, bool]


def _policy_order(names) -> list[str]:
    known = [p for p in POLICIES if p in names]
    return known + sorted(set(names) - set(known))


def aggregate(points: Sequence[BenchPoint]) -> list[AggregateRow]:
    groups: dict[tuple[str, float], list[BenchPoint]] = {}
    for p in points:
        groups.setdefault((p.policy, p.target_util), []).append(p)
    rows = []
    for (policy, u), pts in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):

        def mean(field: str, pts: list[BenchPoint] = pts) -> float:
            vals = [getattr(p, field) for p in pts if not math.isnan(getattr(p, field))]
            return statistics.fmean(vals) if vals else math.nan

        rows.append(
            AggregateRow(
                policy=policy,
                target_util=u,
                miss_rate=mean("miss_rate"),
                miss_rate_min=min(p.miss_rate for p in pts),
                miss_rate_max=max(p.miss_rate for p in pts),
                high_crit_miss_rate=mean("high_crit_miss_rate"),
                p99_ms=mean("p99_ms"),
                mean_batch=mean("mean_batch"),
                admitted={r: all(p.admitted[r] for p in pts) for r in BATCH_RULES},
            )
        )
    return rows


def markdown_table(rows: Sequence[AggregateRow], field: str = "miss_rate") -> str:
    """Utilization rows x policy columns (percent), plus the analysis verdicts."""
    policies = _policy_order({r.policy for r in rows})
    utils = sorted({r.target_util for r in rows})
    lookup = {(r.policy, r.target_util): r for r in rows}
    head = "| U | " + " | ".join(policies) + " | test: " + " | test: ".join(BATCH_RULES) + " |"
    sep = "|---:|" + "---:|" * len(policies) + ":---:|" * len(BATCH_RULES)
    lines = [head, sep]
    for u in utils:
        cells = []
        for p in policies:
            r = lookup.get((p, u))
            v = getattr(r, field) if r else math.nan
            cells.append("-" if math.isnan(v) else f"{100 * v:.2f}")
        any_row = next(r for r in rows if r.target_util == u)
        verdicts = ["admit" if any_row.admitted[r] else "reject" for r in BATCH_RULES]
        lines.append(f"| {u:.2f} | " + " | ".join(cells + verdicts) + " |")
    return "\n".join(lines)


def write_results(points: Sequence[BenchPoint], out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = aggregate(points)
    (out / f"{name}_points.json").write_text(json.dumps([asdict(p) for p in points], indent=1))
    md = [
        f"# {name}: simulated deadline-miss rate (%) vs utilization",
        "",
        "Simulated, not measured. Mean over seeds. The `test:` columns are the",
        "non-preemptive EDF admission verdicts for the scaled task set under each",
        "batching rule: `none` certifies `edf`, `same_deadline` certifies",
        "`edf_batch_same_deadline`, `cross_deadline` certifies `edf_batch`. The tests",
        "use nominal latencies while the simulator adds execution-time noise, so an",
        "admitted row can still show small miss rates. They say nothing about FIFO,",
        "round-robin, fixed priority or the Triton-style batcher.",
        "",
        "## All streams",
        "",
        markdown_table(rows, "miss_rate"),
        "",
        "## High-criticality streams",
        "",
        markdown_table(rows, "high_crit_miss_rate"),
        "",
    ]
    path = out / f"{name}_table.md"
    path.write_text("\n".join(md))
    return path


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def plot(points: Sequence[BenchPoint], path: str | Path, title: str) -> Path:
    """Two panels (all streams, high-criticality streams), one line per policy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - optional extra
        raise ImportError("plotting needs `pip install edgesched[plot]`") from e

    rows = aggregate(points)
    policies = _policy_order({r.policy for r in rows})
    admitted = [r.target_util for r in rows if r.admitted["none"]]
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=surface)
    panels = [("miss_rate", "All streams"), ("high_crit_miss_rate", "High-criticality streams")]
    for ax, (field, label) in zip(axes, panels):
        ax.set_facecolor(surface)
        for i, pol in enumerate(policies):
            pr = [r for r in rows if r.policy == pol]
            ax.plot(
                [r.target_util for r in pr],
                [100 * getattr(r, field) for r in pr],
                color=SERIES_COLORS[i % len(SERIES_COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markersize=5,
                linewidth=2,
                label=pol,
            )
        if admitted:
            ax.axvline(max(admitted), color=muted, linestyle=":", linewidth=1.2)
            ax.annotate(
                "EDF test (no batching)\nadmits up to here",
                xy=(max(admitted), 0.97),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=8,
                color=muted,
            )
        ax.set_title(label, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("nominal utilization (unbatched)", color=muted)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=muted)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(grid)
    axes[0].set_ylabel("deadline-miss rate (%)", color=muted)
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=ink,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(title, color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)
    return path


def run_benchmark(
    workload_path: str | Path,
    policies: Sequence[str],
    utils: Sequence[float],
    seeds: Sequence[int],
    duration: float | None,
    jobs: int,
    out_dir: str | Path,
    figure_dir: str | Path | None,
) -> tuple[list[BenchPoint], Workload]:
    wl = load_workload(workload_path)
    points = sweep(workload_path, policies, utils, seeds, duration, jobs)
    write_results(points, out_dir, wl.name)
    if figure_dir is not None:
        dur = duration if duration is not None else wl.sim.duration
        title = (
            f"{wl.name}: simulated deadline-miss rate vs utilization "
            f"({len(seeds)} seeds x {dur:g} s each)"
        )
        plot(points, Path(figure_dir) / f"{wl.name}_miss_vs_util.png", title)
    return points, wl
