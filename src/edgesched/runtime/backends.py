"""Execution backends for the runtime arbiter.

A backend turns a list of per-request inputs into a list of per-request
outputs, running them as one batch. ``run_batch`` must be *synchronous*: when
it returns, the work is done. The arbiter's timing, deadline accounting and
online profile correction all assume this, so GPU backends synchronize their
stream before returning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    def run_batch(self, inputs: Sequence[Any]) -> list[Any]: ...


def _type_root(x: Any) -> str:
    return type(x).__module__.split(".")[0]


def default_collate(inputs: Sequence[Any]) -> Any:
    """Stack torch tensors or numpy arrays along a new leading dim; otherwise return a list."""
    first = inputs[0]
    root = _type_root(first)
    if root == "torch":
        import torch

        return torch.stack(list(inputs))
    if root == "numpy":
        import numpy as np

        return np.stack(list(inputs))
    return list(inputs)


def default_split(output: Any, n: int) -> list[Any]:
    """Split a batched output along its leading dimension into ``n`` items."""
    if isinstance(output, list | tuple):
        if len(output) != n:
            raise ValueError(f"backend returned {len(output)} outputs for a batch of {n}")
        return list(output)
    if len(output) != n:
        raise ValueError(f"backend output leading dim {len(output)} != batch size {n}")
    return [output[i] for i in range(n)]


class CallableBackend:
    """Wrap any Python callable.

    With ``batched=True`` the callable receives ``collate(inputs)`` and must
    return something ``split`` can cut into per-request outputs. With
    ``batched=False`` it is called once per input (the batch is still a single
    non-preemptive dispatch from the scheduler's point of view).
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        batched: bool = True,
        collate: Callable[[Sequence[Any]], Any] = default_collate,
        split: Callable[[Any, int], list[Any]] = default_split,
    ) -> None:
        self.fn = fn
        self.batched = batched
        self.collate = collate
        self.split = split

    def run_batch(self, inputs: Sequence[Any]) -> list[Any]:
        if not self.batched:
            return [self.fn(x) for x in inputs]
        return self.split(self.fn(self.collate(inputs)), len(inputs))


class TorchBackend:
    """Run an ``nn.Module`` under ``torch.inference_mode``.

    Inputs are per-request tensors without a batch dimension; they are stacked,
    moved to ``device`` and run. On CUDA the batch runs on a dedicated
    ``torch.cuda.Stream`` which is synchronized before returning. Outputs are
    returned on CPU unless ``outputs_on_device=True``.
    """

    def __init__(self, module: Any, device: str | None = None, *, outputs_on_device: bool = False):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.module = module.to(self.device).eval()
        self.outputs_on_device = outputs_on_device
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

    def run_batch(self, inputs: Sequence[Any]) -> list[Any]:
        torch = self.torch
        with torch.inference_mode():
            if self.stream is None:
                out = self.module(torch.stack(list(inputs)).to(self.device))
            else:
                # The backend stream is non-blocking, so it must wait for work the
                # caller queued on the default stream (e.g. kernels producing the inputs).
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.stream):
                    batch = torch.stack(list(inputs)).to(self.device, non_blocking=True)
                    out = self.module(batch)
                self.stream.synchronize()
        if not self.outputs_on_device:
            out = out.cpu()
        return list(out.unbind(0))


class OnnxRuntimeBackend:
    """ONNX Runtime session (install the ``onnx`` extra). Single input, first output."""

    def __init__(
        self, path: str, providers: Sequence[str] | None = None, input_name: str | None = None
    ):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError("OnnxRuntimeBackend needs `pip install edgesched[onnx]`") from e
        self.np = np
        self.session = ort.InferenceSession(
            path, providers=list(providers or ort.get_available_providers())
        )
        self.input_name = input_name or self.session.get_inputs()[0].name

    def run_batch(self, inputs: Sequence[Any]) -> list[Any]:
        batch = self.np.stack([self.np.asarray(x) for x in inputs])
        out = self.session.run(None, {self.input_name: batch})[0]
        return default_split(out, len(inputs))


class TensorRTBackend:
    """Serialized TensorRT engine with one input and one output, batched on dim 0.

    The engine needs an optimization profile (profile 0) whose max shape sets
    ``max_batch``. Device buffers for ``max_batch`` are allocated once as torch
    tensors; each call copies the stacked inputs in, sets the input shape, runs
    ``execute_async_v3`` on the backend's stream and synchronizes. Inputs are
    per-request tensors (CPU or CUDA) without a batch dimension. Outputs are
    returned on CPU unless ``outputs_on_device=True``.
    """

    def __init__(self, engine_path: str, device: str = "cuda", *, outputs_on_device: bool = False):
        import tensorrt as trt
        import torch

        dtypes = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        self.torch = torch
        self.device = torch.device(device)
        self.outputs_on_device = outputs_on_device
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if n not in ins]
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(f"{engine_path}: expected one input and one output, got {names}")
        self.input_name = ins[0]
        max_shape = tuple(self.engine.get_tensor_profile_shape(self.input_name, 0)[2])
        self.max_batch = max_shape[0]
        self.context.set_input_shape(self.input_name, max_shape)
        buffers = {}
        for name in names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = dtypes[self.engine.get_tensor_dtype(name)]
            buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
            # Batch is the outermost dim, so a batch of n is a prefix of the buffer.
            self.context.set_tensor_address(name, buffers[name].data_ptr())
        self.input, self.output = buffers[ins[0]], buffers[outs[0]]
        self.stream = torch.cuda.Stream(self.device)

    def run_batch(self, inputs: Sequence[Any]) -> list[Any]:
        torch = self.torch
        n = len(inputs)
        if n > self.max_batch:
            raise ValueError(f"batch of {n} exceeds the engine's max batch {self.max_batch}")
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.input[:n].copy_(torch.stack(list(inputs)), non_blocking=True)
            self.context.set_input_shape(self.input_name, tuple(self.input[:n].shape))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        self.stream.synchronize()
        # Copy out: the output buffer is reused by the next call.
        out = self.output[:n].clone() if self.outputs_on_device else self.output[:n].cpu()
        return list(out.unbind(0))
