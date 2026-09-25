"""edgesched: deadline-aware inference scheduling for models sharing one edge accelerator."""

from edgesched.analysis import AnalysisResult, admit, analyze
from edgesched.config import Workload, load_workload, load_workload_dict
from edgesched.metrics import Summary, Trace, prometheus_text, summarize
from edgesched.model import Criticality, Job, ModelSpec, TaskSpec
from edgesched.policies import (
    EDF,
    FIFO,
    DeviceState,
    Dispatch,
    DynamicBatchTimeout,
    EDFBatch,
    FixedPriority,
    Policy,
    RoundRobin,
    Wait,
    make_policy,
)
from edgesched.profile import LinearProfile, NoiseModel, TableProfile, load_profile_json, measure
from edgesched.sim import Simulator, simulate

__version__ = "0.1.0"

__all__ = [
    "EDF",
    "FIFO",
    "AnalysisResult",
    "Criticality",
    "DeviceState",
    "Dispatch",
    "DynamicBatchTimeout",
    "EDFBatch",
    "FixedPriority",
    "Job",
    "LinearProfile",
    "ModelSpec",
    "NoiseModel",
    "Policy",
    "RoundRobin",
    "Simulator",
    "Summary",
    "TableProfile",
    "TaskSpec",
    "Trace",
    "Wait",
    "Workload",
    "__version__",
    "admit",
    "analyze",
    "load_profile_json",
    "load_workload",
    "load_workload_dict",
    "make_policy",
    "measure",
    "prometheus_text",
    "simulate",
    "summarize",
]
