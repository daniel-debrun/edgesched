"""Load a device workload (models + streams) from YAML or a dict.

Units in config files are milliseconds and hertz; everything is converted to
seconds on load. Schema::

    name: multi_agent_cell
    device: jetson-orin-nano            # free-form label
    models:
      actor_mlp:
        max_batch: 8
        batchable: true
        profile: {type: linear, a_ms: 1.5, c_ms: 0.3}
        # or {type: table, points_ms: {1: 1.8, 4: 2.7}}
        # or {type: json, path: ../profiles/actor_mlp.json, kind: table}
        noise: {sigma: 0.05, tail_prob: 0.01, tail_scale: 2.0}
    streams:
      - {name: agent0, model: actor_mlp, rate_hz: 50, deadline_ms: 20,
         jitter_ms: 1, phase_ms: 0, priority: 1, criticality: high}
    sim: {duration_s: 30, late_policy: drop, safety_factor: 1.0}
    policies:
      dynamic_batch: {max_queue_delay_ms: 2}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from edgesched.model import Arrival, Criticality, ModelSpec, TaskSpec
from edgesched.policies import Policy, make_policy
from edgesched.profile import (
    LatencyProfile,
    LinearProfile,
    NoiseModel,
    TableProfile,
    load_profile_json,
)

MS = 1e-3


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SimOptions:
    duration: float = 30.0
    late_policy: str = "drop"
    safety_factor: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.late_policy not in ("drop", "run"):
            raise ConfigError("late_policy must be 'drop' or 'run'")


@dataclass(frozen=True)
class Workload:
    name: str
    models: dict[str, ModelSpec]
    tasks: list[TaskSpec]
    device: str = "unspecified"
    sim: SimOptions = field(default_factory=SimOptions)
    policy_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [t.name for t in self.tasks]
        if len(names) != len(set(names)):
            raise ConfigError("stream names must be unique")
        for t in self.tasks:
            if t.model not in self.models:
                raise ConfigError(f"stream {t.name!r} references unknown model {t.model!r}")

    def scaled(self, factor: float) -> Workload:
        """Same workload with every latency profile multiplied by ``factor``."""
        models = {n: replace(m, profile=m.profile.scaled(factor)) for n, m in self.models.items()}
        return replace(self, models=models)

    def policy(self, name: str) -> Policy:
        return make_policy(name, **self.policy_params.get(name, {}))


def _profile(spec: Mapping[str, Any], base: Path, model: str) -> LatencyProfile:
    kind = spec.get("type", "linear")
    if kind == "linear":
        return LinearProfile(float(spec["a_ms"]) * MS, float(spec.get("c_ms", 0.0)) * MS)
    if kind == "table":
        return TableProfile({int(b): float(t) * MS for b, t in spec["points_ms"].items()})
    if kind == "json":
        path = Path(spec["path"])
        return load_profile_json(
            path if path.is_absolute() else base / path, spec.get("kind", "table")
        )
    raise ConfigError(f"model {model!r}: unknown profile type {kind!r}")


def _ms_keys(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k.endswith("_ms"):
            out[k[:-3]] = float(v) * MS
        elif k == "preferred_batch_sizes":
            out[k] = {m: [int(x) for x in sizes] for m, sizes in v.items()}
        else:
            out[k] = v
    return out


def _task(raw: Mapping[str, Any]) -> TaskSpec:
    try:
        name, model = raw["name"], raw["model"]
    except KeyError as e:
        raise ConfigError(f"stream is missing required key {e}") from None
    if "rate_hz" in raw:
        period = 1.0 / float(raw["rate_hz"])
    elif "period_ms" in raw:
        period = float(raw["period_ms"]) * MS
    else:
        raise ConfigError(f"stream {name!r}: needs rate_hz or period_ms")
    deadline = float(raw["deadline_ms"]) * MS if "deadline_ms" in raw else None
    return TaskSpec(
        name=name,
        model=model,
        period=period,
        deadline=deadline,
        priority=raw.get("priority"),
        phase=float(raw.get("phase_ms", 0.0)) * MS,
        jitter=float(raw.get("jitter_ms", 0.0)) * MS,
        criticality=Criticality.parse(raw.get("criticality", "low")),
        arrival=Arrival(raw.get("arrival", "periodic")),
        sporadic_mean_extra=float(raw.get("sporadic_mean_extra_ms", 0.0)) * MS,
        group=raw.get("group"),
    )


def load_workload_dict(data: Mapping[str, Any], base_dir: str | Path = ".") -> Workload:
    base = Path(base_dir)
    if "models" not in data or "streams" not in data:
        raise ConfigError("workload needs 'models' and 'streams'")
    models: dict[str, ModelSpec] = {}
    for name, m in data["models"].items():
        noise = NoiseModel(**m["noise"]) if m.get("noise") else None
        models[name] = ModelSpec(
            name=name,
            profile=_profile(m.get("profile", {}), base, name),
            max_batch=int(m.get("max_batch", 1)),
            batchable=bool(m.get("batchable", False)),
            noise=noise,
        )
    tasks = [_task(t) for t in data["streams"]]
    sim_raw = dict(data.get("sim", {}))
    sim = SimOptions(
        duration=float(sim_raw.get("duration_s", 30.0)),
        late_policy=sim_raw.get("late_policy", "drop"),
        safety_factor=float(sim_raw.get("safety_factor", 1.0)),
        seed=int(sim_raw.get("seed", 0)),
    )
    params = {k: _ms_keys(v or {}) for k, v in (data.get("policies") or {}).items()}
    return Workload(
        name=data.get("name", "workload"),
        models=models,
        tasks=tasks,
        device=data.get("device", "unspecified"),
        sim=sim,
        policy_params=params,
    )


def load_workload(path: str | Path) -> Workload:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return load_workload_dict(data, base_dir=path.parent)
