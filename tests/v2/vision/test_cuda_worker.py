"""Tests for CUDAWorker (mast.vision.cuda_worker).

Strategy: CUDAWorker imports torch only inside the worker thread entry (_run),
so we monkeypatch sys.modules["torch"] with a stub BEFORE calling start().
This lets us test lifecycle, queue back-pressure, error propagation, and
idempotent shutdown without a real CUDA device.

A separate skipped-by-default test exercises the real CUDA path when
torch.cuda.is_available() is True.
"""
from __future__ import annotations

# ── path bootstrap (canonical block from tests/v2/vision/test_mock_backend.py)
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import asyncio  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.vision.cuda_worker import CUDAWorker, _SENTINEL  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Torch stub factory — installs a fake `torch` module into sys.modules so
# CUDAWorker._run() can run without a real torch+CUDA install.
# ─────────────────────────────────────────────────────────────────────────


class _FakeOOM(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError."""


def _make_torch_stub(*, sleep_per_infer: float = 0.0):
    """Build a stub module satisfying the surface CUDAWorker uses:
        torch._inductor.config.cpp_wrapper = True
        torch.cuda.set_device(device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.OutOfMemoryError
        torch.inference_mode()  (context manager)
        torch.zeros(*shape, device=...)        — for warm-up dummy
    """
    torch_mod = types.ModuleType("torch")
    inductor_mod = types.ModuleType("torch._inductor")
    inductor_config = types.SimpleNamespace(cpp_wrapper=False)
    inductor_mod.config = inductor_config
    torch_mod._inductor = inductor_mod

    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.set_device = lambda dev: None
    cuda_mod.synchronize = lambda: None
    cuda_mod.empty_cache = lambda: None
    cuda_mod.OutOfMemoryError = _FakeOOM
    torch_mod.cuda = cuda_mod

    class _InferenceModeCM:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_mod.inference_mode = lambda: _InferenceModeCM()

    class _FakeCpuTensor:
        """Stub returned by .cpu() — supports .numpy()."""

        def __init__(self, array: np.ndarray):
            self._array = array

        def numpy(self) -> np.ndarray:
            return self._array

    class _FakeGpuTensor:
        """Stub returned by torch.zeros / .cuda() — supports model call."""

        def __init__(self, shape=(1, 3, 224, 224)):
            self.shape = shape

        def cuda(self, device, non_blocking=False):  # noqa: ARG002
            if sleep_per_infer:
                time.sleep(sleep_per_infer)
            return self

        def cpu(self):
            # Return a small deterministic array — the test doesn't care
            # about the values, just the round-trip.
            return _FakeCpuTensor(np.zeros((1,), dtype=np.float32))

    torch_mod.zeros = lambda *shape, **kw: _FakeGpuTensor(shape=shape)  # noqa: ARG005
    return torch_mod, _FakeGpuTensor, _FakeOOM


def _install_torch_stub(monkeypatch, **kw):
    stub, gpu_cls, oom_cls = _make_torch_stub(**kw)
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setitem(sys.modules, "torch._inductor", stub._inductor)
    monkeypatch.setitem(sys.modules, "torch.cuda", stub.cuda)
    return stub, gpu_cls, oom_cls


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — warm-up failure propagates through start()
# ─────────────────────────────────────────────────────────────────────────


def test_worker_warmup_failure_propagates(monkeypatch):
    """Per task spec: if model_factory raises, start() must re-raise so the
    caller can decide whether to fall back to a CPU path."""
    _install_torch_stub(monkeypatch)

    def bad_factory():
        raise RuntimeError("simulated CUDA init failure")

    loop = asyncio.new_event_loop()
    try:
        w = CUDAWorker(bad_factory)
        with pytest.raises(RuntimeError, match="simulated CUDA init failure"):
            w.start(loop)
        # After warm-up failure the thread should have exited cleanly.
        assert w._thread is not None
        w._thread.join(timeout=2.0)
        assert not w._thread.is_alive()
    finally:
        loop.close()


def test_worker_warmup_import_error_propagates(monkeypatch):
    """If torch import itself fails, start() must re-raise ImportError.
    This simulates the v2-venv-without-torch case."""
    # Block any 'torch' import by inserting a finder that raises.
    monkeypatch.setitem(sys.modules, "torch", None)  # importing None raises

    def factory():  # pragma: no cover — never reached because torch fails first
        return lambda x: x

    loop = asyncio.new_event_loop()
    try:
        w = CUDAWorker(factory)
        with pytest.raises((ImportError, TypeError)):
            # importing a module that maps to None in sys.modules raises
            # ImportError (CPython behaviour).
            w.start(loop)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — full lifecycle: start → infer → shutdown
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_lifecycle(monkeypatch):
    """Mock model_factory returns a fake model; infer returns; shutdown drains."""
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    forward_calls: list[int] = []

    class FakeModel:
        def __call__(self, x):
            forward_calls.append(1)
            return gpu_cls()  # output is also a fake tensor with .cpu().numpy()

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: FakeModel())
    w.start(loop)

    # warm-up forward counts as 1 call
    assert len(forward_calls) == 1, "warm-up forward must run exactly once"

    # Submit two inferences
    x = gpu_cls()
    r1 = await w.infer(x)
    r2 = await w.infer(x)
    assert isinstance(r1, np.ndarray)
    assert isinstance(r2, np.ndarray)
    # warm-up + 2 infers = 3 forward calls
    assert len(forward_calls) == 3

    await w.shutdown(drain_timeout=5.0)
    assert w._thread is not None
    assert not w._thread.is_alive(), "thread should be joined after shutdown"


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — queue full back-pressure
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_queue_full(monkeypatch):
    """Submit more items than the queue can hold; the surplus submission must
    raise RuntimeError("CUDA worker queue full") within the put() timeout.

    Queue maxsize is 64. We block the worker on a threading.Event so it does
    NOT drain the queue, then enqueue 64 items (fills the queue) plus one
    more — which must hit Full inside the 2s put-timeout.
    """
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    # Worker waits on this event before processing each item — keeps the
    # queue full for as long as we want.
    gate = threading.Event()

    class GatedModel:
        def __call__(self, x):
            gate.wait(timeout=10.0)  # release at end of test
            return gpu_cls()

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: GatedModel())
    # Warm-up forward runs through immediately because we set the gate first
    gate.set()
    w.start(loop)
    # Now clear the gate so subsequent infers block
    gate.clear()

    try:
        x = gpu_cls()

        # Submit 64 items: the worker pulls the first immediately (and blocks
        # on gate.wait), so 63 fill the queue and 1 is in-flight = 64 total
        # in-flight. The 65th submission will see the queue full and hit the
        # 2s put-timeout. We use to_thread because put() is synchronous.
        async def submit():
            return await w.infer(x)

        # Fill the queue completely: 65 in-flight (1 worker-popped + 64 queued).
        # Once the worker pops the first item and blocks on gate.wait, the
        # queue has capacity for exactly 64 more items.
        filler_tasks = [asyncio.create_task(submit()) for _ in range(65)]
        # Give the worker a moment to pop the first item off the queue
        await asyncio.sleep(0.2)
        # Sanity: queue should now be at capacity (64) with 1 item in worker
        assert w._in_q.qsize() == 64, (
            f"expected qsize=64 after 65 submits, got {w._in_q.qsize()}"
        )

        # The 66th submission must fail with "queue full" after the 2s
        # put-timeout inside infer(). Calling infer() directly — the
        # synchronous put will block the event loop for 2s then raise.
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="queue full"):
            await w.infer(x)
        elapsed = time.monotonic() - t0
        # The internal put timeout is 2.0s; allow generous slack on slow CI.
        assert 1.5 <= elapsed <= 5.0, (
            f"infer() should fail after ~2s put-timeout, took {elapsed:.2f}s"
        )

        # Cleanup: release the gate so the worker drains and filler_tasks
        # resolve before shutdown.
        gate.set()
        await asyncio.gather(*filler_tasks, return_exceptions=True)
    finally:
        gate.set()  # ensure worker is unblocked even if assertions failed
        await w.shutdown(drain_timeout=10.0)


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — shutdown is idempotent
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_shutdown_idempotent(monkeypatch):
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    class FakeModel:
        def __call__(self, x):
            return gpu_cls()

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: FakeModel())
    w.start(loop)

    # First shutdown — real work
    await w.shutdown(drain_timeout=5.0)
    assert w._thread is not None
    assert not w._thread.is_alive()
    first_thread_id = w._thread.ident

    # Second shutdown — no-op, must not raise or restart the thread
    await w.shutdown(drain_timeout=5.0)
    assert w._thread is not None
    assert w._thread.ident == first_thread_id  # same thread object


# ─────────────────────────────────────────────────────────────────────────
# Test 5 — infer error from model is propagated via fut.set_exception
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_infer_error_propagates(monkeypatch):
    """A model that raises mid-inference must surface its exception to
    the awaiter, NOT crash the worker thread."""
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    call_count = {"n": 0}

    class FlakyModel:
        def __call__(self, x):
            call_count["n"] += 1
            # warm-up call (n=1) succeeds; later calls fail
            if call_count["n"] == 1:
                return gpu_cls()
            raise ValueError("simulated model failure")

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: FlakyModel())
    w.start(loop)

    with pytest.raises(ValueError, match="simulated model failure"):
        await w.infer(gpu_cls())

    # Worker thread must still be alive — it does not die on a per-task exception
    assert w._thread is not None and w._thread.is_alive()

    # And subsequent infers can be submitted (will also raise ValueError)
    with pytest.raises(ValueError, match="simulated model failure"):
        await w.infer(gpu_cls())

    await w.shutdown(drain_timeout=5.0)


# ─────────────────────────────────────────────────────────────────────────
# Test 6 — CUDA OOM triggers empty_cache and propagates as exception
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_oom_triggers_empty_cache(monkeypatch):
    """OOM path: empty_cache must be called, exception must reach the awaiter."""
    stub, gpu_cls, oom_cls = _install_torch_stub(monkeypatch)

    empty_cache_calls: list[int] = []
    stub.cuda.empty_cache = lambda: empty_cache_calls.append(1)

    call_count = {"n": 0}

    class OOMModel:
        def __call__(self, x):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return gpu_cls()  # warm-up OK
            raise oom_cls("simulated OOM")

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: OOMModel())
    w.start(loop)

    with pytest.raises(oom_cls, match="simulated OOM"):
        await w.infer(gpu_cls())

    assert empty_cache_calls, "empty_cache must run after OOM"
    await w.shutdown(drain_timeout=5.0)


# ─────────────────────────────────────────────────────────────────────────
# Test 7 — start() cannot be called twice
# ─────────────────────────────────────────────────────────────────────────


def test_worker_double_start_rejected(monkeypatch):
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    class FakeModel:
        def __call__(self, x):
            return gpu_cls()

    loop = asyncio.new_event_loop()
    try:
        w = CUDAWorker(lambda: FakeModel())
        w.start(loop)
        with pytest.raises(RuntimeError, match="already called"):
            w.start(loop)

        # Cleanup
        async def _shut():
            await w.shutdown(drain_timeout=5.0)

        loop.run_until_complete(_shut())
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# Test 8 — cpp_wrapper is set BEFORE first forward (pytorch#163061)
# ─────────────────────────────────────────────────────────────────────────


def test_worker_sets_cpp_wrapper_before_warmup(monkeypatch):
    """Per pytorch#163061: cpp_wrapper MUST be set before the first forward.
    We verify that by recording the order: at the moment the warm-up forward
    runs, cpp_wrapper must already be True.
    """
    stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    observed: dict[str, bool] = {}

    class RecordingModel:
        def __call__(self, x):
            observed["cpp_wrapper_at_warmup"] = stub._inductor.config.cpp_wrapper
            return gpu_cls()

    loop = asyncio.new_event_loop()
    try:
        w = CUDAWorker(lambda: RecordingModel())
        w.start(loop)
        assert observed.get("cpp_wrapper_at_warmup") is True, (
            "cpp_wrapper must be True before warm-up forward (pytorch#163061)"
        )

        async def _shut():
            await w.shutdown(drain_timeout=5.0)

        loop.run_until_complete(_shut())
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# Test 9 — infer rejected before start / after shutdown
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_infer_before_start_raises(monkeypatch):
    w = CUDAWorker(lambda: (_ for _ in ()).throw(AssertionError("unused")))
    with pytest.raises(RuntimeError, match="not started"):
        await w.infer(object())


@pytest.mark.asyncio
async def test_worker_infer_after_shutdown_raises(monkeypatch):
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    class FakeModel:
        def __call__(self, x):
            return gpu_cls()

    loop = asyncio.get_running_loop()
    w = CUDAWorker(lambda: FakeModel())
    w.start(loop)
    await w.shutdown(drain_timeout=5.0)

    with pytest.raises(RuntimeError, match="not started or already stopped"):
        await w.infer(gpu_cls())


# ─────────────────────────────────────────────────────────────────────────
# Test 10 — optional real-CUDA smoke test (skipped if not available)
# ─────────────────────────────────────────────────────────────────────────


def _real_cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _real_cuda_available(),
    reason="real CUDA device not available in v2 venv",
)
@pytest.mark.asyncio
async def test_worker_real_cuda_smoke():  # pragma: no cover — env-gated
    """End-to-end smoke test with real PyTorch on a real CUDA device.
    Only runs when torch.cuda.is_available() is True."""
    import torch  # type: ignore[import-not-found]

    # Smallest possible model: identity-like 1x1 conv.
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    def factory():
        m = TinyModel().cuda().eval()
        return m

    loop = asyncio.get_running_loop()
    w = CUDAWorker(factory)
    w.start(loop)
    try:
        x = torch.zeros(1, 3, 224, 224, device="cpu")
        result = await w.infer(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 224, 224)
    finally:
        await w.shutdown(drain_timeout=10.0)


# ─────────────────────────────────────────────────────────────────────────
# Sanity: _SENTINEL is an opaque singleton
# ─────────────────────────────────────────────────────────────────────────


def test_sentinel_is_opaque_singleton():
    # Identity check — _SENTINEL is module-scoped, two imports see the same
    # object. This matters because the worker uses `is _SENTINEL`.
    from mast.vision.cuda_worker import _SENTINEL as s2
    assert _SENTINEL is s2


# ─────────────────────────────────────────────────────────────────────────
# Fix 4 — worker thread is a daemon: a wedged warm-up must not block exit
# ─────────────────────────────────────────────────────────────────────────


def test_worker_thread_is_daemon(monkeypatch):
    """The worker thread must be a daemon so a hung warm-up (one that never
    sets _ready and never drains a sentinel) cannot keep the interpreter alive
    at process exit (a non-daemon zombie blocks teardown forever)."""
    _stub, gpu_cls, _oom = _install_torch_stub(monkeypatch)

    class FakeModel:
        def __call__(self, x):
            return gpu_cls()

    loop = asyncio.new_event_loop()
    try:
        w = CUDAWorker(lambda: FakeModel())
        w.start(loop)
        assert w._thread is not None
        assert w._thread.daemon is True

        async def _shut():
            await w.shutdown(drain_timeout=5.0)

        loop.run_until_complete(_shut())
    finally:
        loop.close()
