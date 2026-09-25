from __future__ import annotations

from pathlib import Path

import pytest

from edgesched.model import Job, ModelSpec
from edgesched.policies import DeviceState
from edgesched.profile import LinearProfile

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "workloads"

MS = 1e-3


def make_job(stream: str, model: str, release: float, deadline: float, **kw) -> Job:
    return Job(stream=stream, model=model, release=release, deadline=deadline, **kw)


@pytest.fixture
def models() -> dict[str, ModelSpec]:
    return {
        "mlp": ModelSpec("mlp", LinearProfile(1.0 * MS, 0.25 * MS), max_batch=8, batchable=True),
        "cnn": ModelSpec("cnn", LinearProfile(5.0 * MS, 2.0 * MS), max_batch=4, batchable=True),
        "planner": ModelSpec("planner", LinearProfile(10.0 * MS)),
    }


@pytest.fixture
def state(models) -> DeviceState:
    return DeviceState(models)
