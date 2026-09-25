# Integrating edgesched with a pilot_core-style worker stack

pilot_core-style stacks run each stage (camera, segmentation, SLAM, control) as a
worker with a `step()` method, connected by shared-memory ring buffers whose slots
carry `produced_ts` / `read_ts`. Workers are polled in a loop. Every worker that
touches the GPU calls its engine directly, so GPU work is ordered by whichever
thread or process gets there first. There are no deadlines and no arbitration.

The change is small: route **only the inference call** through an arbiter. Buffers,
workers and the control loop stay as they are.

## 1. Declare the stream next to the worker

Each worker that runs a model declares its rate, deadline and criticality. The
deadline is the latency budget the worker actually has: for segmentation feeding a
30 Hz control loop it is roughly one frame; for a planner it can be much looser.

```python
from edgesched.model import Criticality, TaskSpec

SEGMENTATION_STREAM = TaskSpec(
    name="segmentation",
    model="seg_trt",
    period=1 / 30,
    deadline=0.030,
    criticality=Criticality.HIGH,
)
```

## 2. One arbiter per device, created in `start.py`

```python
from edgesched.config import load_workload
from edgesched.runtime import Arbiter

workload = load_workload("config/edgesched_device.yaml")   # models + profiles
arbiter = Arbiter(workload.models, backends, workload.policy("edf_batch"),
                  late_policy="drop", admission="enforce").start()
seg = arbiter.register_stream(SEGMENTATION_STREAM)          # raises if not schedulable
```

With `admission="enforce"`, a worker whose stream would make the set
unschedulable fails at startup, with the violated condition in the error.

## 3. Replace the direct engine call in `step()`

Before:

```python
def step(self):
    slot = self.frames.read_latest()
    mask = self.engine.infer(slot.payload)
    self.masks.write(mask, produced_ts=time.time())
```

After:

```python
def step(self):
    slot = self.frames.read_latest()
    release = perf_counter_from_wall(slot.produced_ts)      # deadline counts from capture
    try:
        mask = self.seg.infer(slot.payload, release=release)
    except DeadlineMissed:
        return                                              # stale frame, skip it
    self.masks.write(mask, produced_ts=time.time())
```

Using the slot's `produced_ts` as the release time is what makes the deadline
meaningful: a frame that sat in the ring buffer for 20 ms has 20 ms less budget.
The arbiter uses `time.perf_counter`, so convert once
(`perf_counter() - (time.time() - produced_ts)`).

## 4. Multi-process stacks

`Arbiter` is in-process. If workers are separate processes (as with pilot_core's
shared-memory buffers), v0.1 needs the models that share a GPU to live in one
process that hosts the arbiter, with the other workers passing tensors through the
existing ring buffers. A multi-process arbiter over shared memory is on the roadmap.

## 5. Measure before trusting

Build profiles on the device (`scripts/jetson_profile.sh`), then run
`edgesched analyze` and `edgesched simulate` on the resulting workload file. If the
analysis rejects the set, the simulator's per-stream table shows who misses and why
(usually a long non-preemptive model blocking a short-deadline stream).
