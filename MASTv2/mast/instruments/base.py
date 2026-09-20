"""MotionController / MotionAxis abstractions for optical-bench hardware.

Every concrete driver (PZTC nm003, PI GCS, Thorlabs Kinesis) implements
these interfaces so the registry, the delay-line wrapper and the optics
skills never care which vendor is behind an axis.

Safety model — soft travel limits are LAYER-0:

``MotionAxis.move_abs`` / ``move_rel`` are concrete template methods on the
base class. They clamp-check the target against ``AxisConfig`` travel
limits and raise :class:`TravelLimitError` BEFORE delegating to the
driver's ``_move_abs_raw``. Drivers only ever see validated targets, so no
caller — skill, agent, GUI, or a hallucinated LLM parameter — can command
a move outside the configured range. This mirrors the SafetyGuard
philosophy: hard checks live below the LLM, not beside it.

Threading: one controller = one serial link = one lock. All axis calls on
a controller serialise through ``controller._io_lock`` so concurrent skill
threads cannot interleave protocol transactions.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = [
    "InstrumentError",
    "InstrumentUnavailable",
    "TravelLimitError",
    "MotionTimeout",
    "AxisConfig",
    "AxisStatus",
    "MotionAxis",
    "MotionController",
]


# ── errors ────────────────────────────────────────────────────────────────


class InstrumentError(RuntimeError):
    """Base error for all instrument-driver failures."""


class InstrumentUnavailable(InstrumentError):
    """Hardware/driver stack not usable (no port, no DLL, no device).

    The graceful-degradation signal: callers catch this to report
    "unavailable" instead of crashing on machines without the bench.
    """


class TravelLimitError(InstrumentError):
    """Requested target lies outside the configured soft travel limits."""


class MotionTimeout(InstrumentError):
    """Axis did not reach the target / finish homing within the deadline."""


# ── configuration & status ────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisConfig:
    """Static description of one axis, from the device inventory (config).

    Positions are in the axis's *native unit* (``unit``). Soft limits are
    mandatory: a driver refuses to construct an axis without a finite
    travel range — "unlimited" optical hardware does not exist, and an
    unconfigured range must fail loudly rather than move blindly.
    """

    name: str                 # logical axis name, e.g. "x", "delay"
    channel: int | str        # controller channel: PZTC 1-3, PI "A"/"1", ...
    min_pos: float            # soft travel limit, native unit (inclusive)
    max_pos: float            # soft travel limit, native unit (inclusive)
    unit: str = "um"          # native unit label, documentation + UI only
    default_speed: float | None = None   # native unit / s, driver-specific
    settle_s: float = 0.0     # extra settle wait after on-target, seconds

    def __post_init__(self) -> None:
        if not (self.min_pos < self.max_pos):
            raise ValueError(
                f"axis {self.name!r}: min_pos {self.min_pos} must be < "
                f"max_pos {self.max_pos}"
            )


@dataclass
class AxisStatus:
    """Snapshot of one axis. ``on_target`` is None when the controller
    cannot report it (open-loop hardware)."""

    position: float
    moving: bool
    on_target: bool | None = None
    homed: bool | None = None
    raw: dict = field(default_factory=dict)


# ── axis ──────────────────────────────────────────────────────────────────


class MotionAxis(ABC):
    """One movable axis. Subclasses implement the ``_*_raw`` primitives;
    the public API (limit checks, wait loops) lives here and is final in
    spirit — drivers must NOT override ``move_abs`` / ``move_rel``.
    """

    #: default polling cadence for wait-until-on-target loops (seconds)
    POLL_INTERVAL_S = 0.05
    #: default motion deadline when the caller gives none (seconds)
    DEFAULT_TIMEOUT_S = 60.0

    def __init__(self, config: AxisConfig, controller: "MotionController"):
        self.config = config
        self._controller = controller

    # -- public API (template methods, limit-checked) ----------------------

    def move_abs(
        self,
        position: float,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> AxisStatus:
        """Move to *position* (native unit). Raises :class:`TravelLimitError`
        if the target is outside the soft limits; blocks until on-target
        when ``wait`` (raising :class:`MotionTimeout` on deadline)."""
        target = float(position)
        self._check_limits(target)
        with self._controller._io_lock:
            self._move_abs_raw(target)
        if wait:
            return self.wait_until_settled(target, timeout=timeout)
        return self.get_status()

    def move_rel(
        self,
        delta: float,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> AxisStatus:
        """Relative move. Resolved against a fresh position reading so the
        limit check covers the true final target."""
        current = self.get_position()
        return self.move_abs(current + float(delta), wait=wait, timeout=timeout)

    def get_position(self) -> float:
        with self._controller._io_lock:
            return self._get_position_raw()

    def get_status(self) -> AxisStatus:
        with self._controller._io_lock:
            return self._get_status_raw()

    def stop(self) -> None:
        """Halt motion. Never limit-checked, never blocking — this is the
        panic path and must stay callable in any state."""
        with self._controller._io_lock:
            self._stop_raw()

    def home(self, *, wait: bool = True, timeout: float | None = None) -> AxisStatus:
        """Reference/zero the axis (drivers without a homing concept raise
        InstrumentError). Position afterwards is defined by the hardware."""
        with self._controller._io_lock:
            self._home_raw()
        if wait:
            deadline = time.monotonic() + (timeout or self.DEFAULT_TIMEOUT_S)
            while time.monotonic() < deadline:
                status = self.get_status()
                if not status.moving and (status.homed is not False):
                    return status
                time.sleep(self.POLL_INTERVAL_S)
            raise MotionTimeout(
                f"axis {self.config.name!r}: homing did not finish within "
                f"{timeout or self.DEFAULT_TIMEOUT_S:.1f}s"
            )
        return self.get_status()

    def wait_until_settled(
        self, target: float, *, timeout: float | None = None
    ) -> AxisStatus:
        """Poll until the controller reports on-target/not-moving, then apply
        the configured extra ``settle_s``. Falls back to a position-delta
        criterion when the hardware cannot report on-target."""
        deadline = time.monotonic() + (timeout or self.DEFAULT_TIMEOUT_S)
        status = self.get_status()
        while time.monotonic() < deadline:
            status = self.get_status()
            if status.on_target is True:
                break
            if status.on_target is None and not status.moving:
                break
            time.sleep(self.POLL_INTERVAL_S)
        else:
            raise MotionTimeout(
                f"axis {self.config.name!r}: move to {target} not settled "
                f"within {timeout or self.DEFAULT_TIMEOUT_S:.1f}s "
                f"(pos={status.position}, moving={status.moving})"
            )
        if self.config.settle_s > 0:
            time.sleep(self.config.settle_s)
            status = self.get_status()
        return status

    # -- limit enforcement (Layer-0) ---------------------------------------

    def _check_limits(self, target: float) -> None:
        cfg = self.config
        if not (cfg.min_pos <= target <= cfg.max_pos):
            raise TravelLimitError(
                f"axis {cfg.name!r}: target {target} {cfg.unit} outside "
                f"soft limits [{cfg.min_pos}, {cfg.max_pos}] {cfg.unit}"
            )

    # -- driver primitives (hold no locks; base class already locked) ------

    @abstractmethod
    def _move_abs_raw(self, target: float) -> None: ...

    @abstractmethod
    def _get_position_raw(self) -> float: ...

    @abstractmethod
    def _get_status_raw(self) -> AxisStatus: ...

    @abstractmethod
    def _stop_raw(self) -> None: ...

    def _home_raw(self) -> None:
        raise InstrumentError(
            f"axis {self.config.name!r}: driver has no homing support"
        )


# ── controller ────────────────────────────────────────────────────────────


class MotionController(ABC):
    """One physical controller box (may drive several axes).

    Lifecycle: construction is cheap and NEVER touches hardware;
    :meth:`connect` opens the link (raising :class:`InstrumentUnavailable`
    on machines without it); :meth:`close` is idempotent. ``axis()`` on a
    non-connected controller connects lazily.
    """

    def __init__(self) -> None:
        self._io_lock = threading.RLock()
        self._axes: dict[str, MotionAxis] = {}
        self._connected = False

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        with self._io_lock:
            if self._connected:
                return
            self._connect_raw()
            self._axes = self._build_axes()
            self._connected = True

    def close(self) -> None:
        with self._io_lock:
            if not self._connected:
                return
            try:
                self._close_raw()
            finally:
                self._axes = {}
                self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- axis access ---------------------------------------------------------

    def axes(self) -> dict[str, MotionAxis]:
        if not self._connected:
            self.connect()
        return dict(self._axes)

    def axis(self, name: str) -> MotionAxis:
        if not self._connected:
            self.connect()
        try:
            return self._axes[name]
        except KeyError:
            raise InstrumentError(
                f"{type(self).__name__}: no axis {name!r}; "
                f"configured axes: {sorted(self._axes)}"
            ) from None

    def stop_all(self) -> None:
        """Panic-stop every axis; best-effort, collects nothing."""
        for ax in list(self._axes.values()):
            try:
                ax.stop()
            except Exception:  # noqa: BLE001 - panic path must not raise
                pass

    # -- driver primitives ----------------------------------------------------

    @abstractmethod
    def _connect_raw(self) -> None:
        """Open the physical link. Raise InstrumentUnavailable when absent."""

    @abstractmethod
    def _close_raw(self) -> None: ...

    @abstractmethod
    def _build_axes(self) -> dict[str, MotionAxis]:
        """Instantiate MotionAxis objects for the configured channels."""

    # context-manager sugar
    def __enter__(self) -> "MotionController":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
