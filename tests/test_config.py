from __future__ import annotations

import json

import pytest

from edgesched.config import ConfigError, load_workload, load_workload_dict
from edgesched.model import Criticality
from edgesched.policies import DynamicBatchTimeout
from edgesched.profile import TableProfile

from .conftest import WORKLOADS


@pytest.mark.parametrize("name", ["multi_agent_cell", "vla_plus_control", "synthetic_mix"])
def test_shipped_workloads_load(name):
    wl = load_workload(WORKLOADS / f"{name}.yaml")
    assert wl.name == name
    assert wl.tasks and wl.models


def test_units_and_fields_are_converted():
    wl = load_workload(WORKLOADS / "multi_agent_cell.yaml")
    agent = next(t for t in wl.tasks if t.name == "agent0")
    assert agent.period == pytest.approx(0.020)
    assert agent.relative_deadline == pytest.approx(0.020)
    assert agent.jitter == 0.0
    assert agent.group == "actors"
    camera = next(t for t in wl.tasks if t.name == "camera")
    assert camera.jitter == pytest.approx(0.002)
    assert agent.criticality is Criticality.HIGH
    assert wl.models["actor_mlp"].latency(4) == pytest.approx(0.0022)
    policy = wl.policy("dynamic_batch")
    assert isinstance(policy, DynamicBatchTimeout)
    assert policy.max_queue_delay == pytest.approx(0.002)
    assert policy.preferred == {"actor_mlp": [4]}
    assert wl.scaled(2.0).models["actor_mlp"].latency(4) == pytest.approx(0.0044)


def test_json_profile_reference(tmp_path):
    (tmp_path / "prof.json").write_text(
        json.dumps({"unit": "s", "points": {"1": 0.01, "2": 0.015}})
    )
    wl = load_workload_dict(
        {
            "models": {
                "m": {
                    "max_batch": 2,
                    "batchable": True,
                    "profile": {"type": "json", "path": "prof.json"},
                }
            },
            "streams": [{"name": "s", "model": "m", "period_ms": 50}],
        },
        base_dir=tmp_path,
    )
    assert isinstance(wl.models["m"].profile, TableProfile)
    assert wl.models["m"].latency(2) == pytest.approx(0.015)


@pytest.mark.parametrize(
    "data, match",
    [
        ({"models": {}}, "models"),
        (
            {
                "models": {"m": {"profile": {"a_ms": 1}}},
                "streams": [{"name": "s", "model": "x", "rate_hz": 1}],
            },
            "unknown model",
        ),
        (
            {"models": {"m": {"profile": {"a_ms": 1}}}, "streams": [{"name": "s", "model": "m"}]},
            "rate_hz",
        ),
        ({"models": {"m": {"profile": {"type": "cubic"}}}, "streams": []}, "profile type"),
        (
            {
                "models": {"m": {"profile": {"a_ms": 1}}},
                "streams": [],
                "sim": {"late_policy": "maybe"},
            },
            "late_policy",
        ),
    ],
)
def test_invalid_configs_raise(data, match):
    with pytest.raises(ConfigError, match=match):
        load_workload_dict(data)
