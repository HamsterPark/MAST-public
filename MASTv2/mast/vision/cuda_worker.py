"""Dedicated CUDA worker thread bridging asyncio main loop to PyTorch.

Per guide §5.1 (docs/v2/compass_artifact_wf-dce9d451-2b6b-4840-bb8b-c85203d4e7fe_text_markdown.md):

Why a dedicated thread (not loop.run_in_executor(ThreadPoolExecutor(1), ...)):
  - executor does not guarantee long-lived worker existence (may be destroyed
    when idle), cannot drain in-flight CUDA work on shutdown, and cancelling
    a concurrent.futures.Future of a detached thread cannot actually interrupt
    the running task.
  - Self-managed thread + queue.Queue gives us precise control over warm-up,
    shutdown, and error propagation.
  - loop.call_soon_threadsafe is the canonical cross-thread bridge per the
    asyncio docs.

Critical GIL workaround:
  pytorch/pytorch#163061 — by default, torch.compile-emitted kernels hold the
  GIL. We MUST set torch._inductor.config.cpp_wrapper = True BEFORE the first
  forward (maintainer @jansel: "main blocker for on-by-default"; reporter
  syuoni: "the cpp wrapper should be turned on before the first execution of
  the torch.compile function, otherwise it's not effective").

Lifecycle:
  __init__ → start(loop) → repeated await infer(x_cpu) → await shutdown()

CUDA-unavailable fallback policy:
  If torch is missing OR the model_factory raises (ImportError / CUDA init
  failure / OOM during warm-up), start() re-raises the captured exception.
  The CUDAWorker does NOT silently degrade — the caller (e.g. VisionModule)
  decides whether to fall back to asyncio.to_thread on a CPU model, which
  preserves the v1 calling pattern. See:
      - VisionModule.MockBackend (mast.vision._mock_backend) — torch-free
      - VisionModule.LegacyBackend (mast.vision._legacy_wrapper) — torch CPU
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

# Module-level sentinel used to signal shutdown to the worker thread.
# A bare object() is fine: identity comparison (`is`) keeps payloads opaque,
# avoids accidental matches with user inputs, and is cheap.
_SENTINEL: object = object()


class CUDAWorker:
    """Single dedicated thread that owns the CUDA primary context.

    Lifecycle:
        w = CUDAWorker(model_factory, device=0)
        w.start(loop)                       # blocks until warm-up done
        result = await w.infer(x_cpu)       # tensor in, np.ndarray out
        await w.shutdown(drain_timeout=10)

    Notes on torch types:
        We avoid importing torch at module load — keeps this file importable
        even when the v2 venv has no torch installed (e.g. CI test stubs).
        x_cpu is duck-typed: anything with `.cuda(device, non_blocking=...)`
        works. The model_factory return value is also duck-typed (must be
        callable).
    """

    def __init__(self, model_factory: Callable[[], Any], device: int = 0):
        self._model_factory = model_factory
        self._device = device
        self._in_q: queue.Queue = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        # Captured exception from the warm-up phase, surfaced by start().
        self._error: BaseException | None = None
        # Flips True once shutdown() has been called once (idempotency).
        self._shutdown_done: bool = False

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the worker thread and block until warm-up succeeds.

        Raises whatever the model_factory / torch warm-up raised (ImportError,
        RuntimeError for CUDA unavailable, torch.cuda.OutOfMemoryError, ...).
        On any warm-up failure the worker thread exits cleanly — no zombie.
        """
        if self._thread is not None:
            raise RuntimeError("CUDAWorker.start() already called")
        self._loop = loop
        # daemon=True: if warm-up wedges (a hung CUDA init / driver stall that
        # never trips the 30s budget below — the worker never sets _ready and
        # never reaches the sentinel-drained shutdown), the thread must NOT keep
        # the interpreter alive at exit. A non-daemon zombie here blocks process
        # teardown forever. The clean-shutdown drain path (shutdown()) still
        # join()s it; daemon only governs the abandoned-on-timeout case.
        self._thread = threading.Thread(
            target=self._run,
            name=f"cuda-worker-dev{self._device}",
            daemon=True,
        )
        self._thread.start()
        # 30s budget covers: import torch, cuBLAS handle, cuDNN autotune,
        # weight download (if any). Tunable per deployment. On timeout the
        # daemon worker is abandoned (it cannot block exit) and we raise so the
        # caller can fall back to a CPU path.
        if not self._ready.wait(timeout=30.0):
            raise RuntimeError(
                f"CUDA worker on device {self._device} failed to warm up in 30s"
            )
        if self._error is not None:
            # Re-raise the original exception so the caller can decide whether
            # to fall back to a CPU path. Per task spec: caller decides.
            raise self._error

    def _run(self) -> None:
        """Worker thread entry point. Warm-up then process loop."""
        try:
            import torch  # local — keeps module importable without torch
            # CRITICAL: pytorch#163061 GIL workaround. MUST be set before the
            # first torch.compile forward, otherwise it has no effect.
            torch._inductor.config.cpp_wrapper = True  # type: ignore[attr-defined]
            torch.cuda.set_device(self._device)
            model = self._model_factory()
            with torch.inference_mode():
                dummy = torch.zeros(
                    1, 3, 224, 224, device=f"cuda:{self._device}"
                )
                _ = model(dummy)
                torch.cuda.synchronize()
        except BaseException as e:  # noqa: BLE001 — propagate any failure
            self._error = e
            self._ready.set()
            return

        self._ready.set()
        log.info("CUDA worker ready on device %d", self._device)

        # ── Main processing loop ─────────────────────────────────────────
        while True:
            item = self._in_q.get()
            if item is _SENTINEL:
                log.info("CUDA worker shutdown signal received")
                break
            fut, x_cpu = item
            try:
                x = x_cpu.cuda(self._device, non_blocking=True)
                with torch.inference_mode():
                    y = model(x)
                torch.cuda.synchronize()
                # IMPORTANT: convert to numpy on the worker thread.
                # Never hand a live CUDA tensor across the threading boundary
                # — caching allocator is process-wide but stream-ordering is
                # the user's responsibility. CPU ndarray is trivially safe.
                result = y.cpu().numpy()
                self._post(fut, "set_result", result)
            except torch.cuda.OutOfMemoryError as e:
                # Best-effort reclaim before propagating so subsequent infers
                # have a chance on a smaller batch. Caller sees the OOM.
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 — empty_cache is best-effort
                    log.exception("torch.cuda.empty_cache() failed after OOM")
                self._post(fut, "set_exception", e)
            except BaseException as e:  # noqa: BLE001 — any infer failure
                log.exception("CUDA worker inference task failed")
                self._post(fut, "set_exception", e)

    def _post(self, fut: asyncio.Future, method: str, arg: Any) -> None:
        """Schedule fut.set_result(arg) / fut.set_exception(arg) on the loop.

        loop.call_soon_threadsafe is the canonical cross-thread bridge per
        the asyncio docs:
            "To schedule a callback from another OS thread, the
             loop.call_soon_threadsafe() method should be used."
        """
        assert self._loop is not None
        # Guard against the loop being already closed during shutdown races.
        if self._loop.is_closed():
            log.warning(
                "CUDA worker: event loop already closed; dropping %s", method
            )
            return
        try:
            self._loop.call_soon_threadsafe(getattr(fut, method), arg)
        except RuntimeError:
            # Loop closed between is_closed() check and call_soon_threadsafe.
            log.exception("CUDA worker: call_soon_threadsafe failed")

    # ─────────────────────────────────────────────────────────────────────
    # Async API
    # ─────────────────────────────────────────────────────────────────────

    async def infer(self, x_cpu) -> Any:
        """Submit a CPU tensor for inference. Returns CPU ndarray.

        Raises RuntimeError if the in-queue is full (back-pressure). The
        caller can wrap with asyncio.wait_for(...) for a deadline.
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("CUDA worker not started or already stopped")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        try:
            # timeout=2.0 protects against deadlock if the worker is wedged.
            # 2s is well above ms-scale vision target; a stalled queue here
            # means something is seriously wrong upstream.
            self._in_q.put((fut, x_cpu), timeout=2.0)
        except queue.Full as e:
            raise RuntimeError("CUDA worker queue full") from e
        return await fut

    async def shutdown(self, drain_timeout: float = 10.0) -> None:
        """Drain in-flight work and join the worker thread.

        Idempotent: a second call is a no-op. drain_timeout bounds how long
        we wait for in-flight CUDA work to finish; if it expires, we log and
        return (the thread is a daemon, so it cannot block process exit and
        PyTorch tears down on interpreter shutdown).
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._thread is None:
            return
        # The sentinel goes onto the same FIFO queue, so it is processed
        # after any pending infers. This is the "drain" semantics.
        try:
            self._in_q.put(_SENTINEL, timeout=2.0)
        except queue.Full:
            # If the queue is full of pending work, we still want to shut
            # down. Drop one item and retry once.
            log.warning(
                "CUDA worker shutdown: in-queue full, dropping head to inject sentinel"
            )
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_q.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                log.error("CUDA worker shutdown: could not inject sentinel")
                return
        # join() blocks the calling thread. We want to keep the async loop
        # responsive while waiting, so we offload to a default executor.
        await asyncio.to_thread(self._thread.join, drain_timeout)
        if self._thread.is_alive():
            log.error(
                "CUDA worker did not drain in %.1fs (thread still alive)",
                drain_timeout,
            )


__all__ = ["CUDAWorker"]
