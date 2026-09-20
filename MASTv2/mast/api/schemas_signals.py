"""Pydantic models for the signal-channels endpoint (信号捕获 + FFT redesign).

Single source of types for ``GET /api/experimental/signals``. The shapes mirror
the live IC skills that already exist and are merely RELAYED by the route:

* ``SignalChannel`` ⇐ ``ListSignalChannels`` result rows (``Signals_NamesGet``):
  the 128 acquirable Nanonis signals as ``{index, name}``, plus a ``unit`` the
  route fills in heuristically (current channels are amperes).
* ``TimebaseOption`` ⇐ ``GetOsciTimebases`` rows (``Osci1T_TimebaseGet``): the
  hardware oscilloscope sample-rate timebases as ``{index, dt_s, fs_hz}``.
* the window functions / output modes are STATIC UI enums (the FFT is computed
  client-side / in the capture path over an already-acquired trace), surfaced
  here so the frontend renders the same choices the old Gradio tab offered
  (Hann / Hamming / Rect; amplitude |FFT| / power-spectral-density).

Per the house rules every field is optional-with-default where the live backend
may be absent, and the response carries ``degraded`` so the UI can fall back to
the static default channel list without ever seeing a 500.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── building blocks ──────────────────────────────────────────────────────


class SignalChannel(BaseModel):
    """One acquirable Nanonis signal channel (a row of ``Signals_NamesGet``)."""

    index: int
    name: str
    # Best-effort physical unit. Current channels read in amperes; other
    # channels' true unit depends on the Nanonis signal config, so we leave it
    # blank rather than guess wrong (the UI shows the raw value then).
    unit: Optional[str] = None
    is_current: bool = False


class TimebaseOption(BaseModel):
    """One Osci1T oscilloscope timebase (a row of ``Osci1T_TimebaseGet``).

    ``dt_s`` is the per-sample interval; ``fs_hz = 1/dt`` is the sample rate."""

    index: int
    dt_s: float
    fs_hz: float


class WindowOption(BaseModel):
    """A static FFT window function the UI can offer (value + display label)."""

    value: str
    label: str


class OutputModeOption(BaseModel):
    """A static FFT output mode (amplitude |FFT| vs power-spectral-density)."""

    value: str
    label: str


# ── response ─────────────────────────────────────────────────────────────


class SignalChannelsResponse(BaseModel):
    """``GET /api/experimental/signals`` — acquirable channels + Osci timebases
    + the static FFT window / output enums for the capture+FFT tab.

    ``degraded=True`` means the live Nanonis was not reachable and ``channels`` /
    ``timebases`` are the sensible static fallbacks (never a 500). ``source`` is
    ``"live"`` when the names came off the instrument, else ``"default"``."""

    channels: list[SignalChannel] = Field(default_factory=list)
    n_channels: int = 0
    #: 仪器自己声明的信号路数。与 ``n_channels`` 不同 = 名单没解全。
    declared_n: Optional[int] = None
    #: 名单被截断。**这不是「这台机器只有这么多信号」**：一个按名字找通道的
    #: 下游（monitoring.aux_channels）在截断的名单上会把 86 号的 lock-in 报成
    #: 「本机没有」—— 一句关于硬件的话，却是一次解析失败的产物。下拉框据此提示
    #: 用户「名单不完整」，而不是让他相信通道不存在。
    truncated: bool = False
    current_indices: list[int] = Field(default_factory=list)

    timebases: list[TimebaseOption] = Field(default_factory=list)
    current_timebase_index: Optional[int] = None
    osci_available: bool = False

    windows: list[WindowOption] = Field(default_factory=list)
    output_modes: list[OutputModeOption] = Field(default_factory=list)

    source: str = "default"
    degraded: bool = True
    detail: Optional[str] = None
