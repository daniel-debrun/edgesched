"""Scheduling policies. The same objects drive the simulator and the runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edgesched.policies.base import Decision, DeviceState, Dispatch, Policy, Wait, edf_key
from edgesched.policies.batching import DynamicBatchTimeout, EDFBatch
from edgesched.policies.simple import EDF, FIFO, FixedPriority, RoundRobin

POLICIES: dict[str, Callable[..., Policy]] = {
    "round_robin": RoundRobin,
    "fifo": FIFO,
    "fixed_priority": FixedPriority,
    "edf": EDF,
    "edf_batch": EDFBatch,
    "edf_batch_same_deadline": lambda **kw: EDFBatch(rule="same_deadline", **kw),
    "dynamic_batch": DynamicBatchTimeout,
}


def make_policy(name: str, **kwargs: Any) -> Policy:
    """Instantiate a policy by registry name."""
    try:
        factory = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
    return factory(**kwargs)


__all__ = [
    "EDF",
    "FIFO",
    "POLICIES",
    "Decision",
    "DeviceState",
    "Dispatch",
    "DynamicBatchTimeout",
    "EDFBatch",
    "FixedPriority",
    "Policy",
    "RoundRobin",
    "Wait",
    "edf_key",
    "make_policy",
]
