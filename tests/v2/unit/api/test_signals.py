"""Contract tests for the signal-channels endpoint (信号捕获 + FFT redesign).

``GET /api/experimental/signals`` lists the acquirable Nanonis signal channels +
Osci timebases + the static FFT enums for the new capture tab. Guarantees:

  * standalone (no live core wired): 200 + a SENSIBLE DEFAULT channel list,
    ``degraded=True`` / ``source="default"`` — never a 500;
  * the static FFT enums (windows / output modes) are ALWAYS present, even
    degraded, so the UI can render them offline;
  * when a live ExecutionContext is wired (here a fake pool/state/registry that
    returns ``ListSignalChannels`` / ``GetOsciTimebases`` results), the route
    RELAYS the live names + units + Osci timebases (``degraded=False`` /
    ``source="live"``);
  * the Osci timebases degrade INDEPENDENTLY: live channels + an unloaded Osci1T
    module ⇒ live channels but ``osci_available=False`` / ``timebases=[]``.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.signals import router


def _client(ctx: AppContext | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx if ctx is not None else AppContext()
    app.include_router(router, prefix="/api")
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── degraded / standalone ────────────────────────────────────────────────────
def test_signals_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/experimental/signals")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["source"] == "default"
    # sensible default list rendered, with a current channel pre-flagged
    assert body["n_channels"] > 0
    assert body["n_channels"] == len(body["channels"])
    assert body["current_indices"], "a default current channel should be flagged"
    cur_idx = body["current_indices"][0]
    cur = next(c for c in body["channels"] if c["index"] == cur_idx)
    assert cur["is_current"] is True
    assert cur["unit"] == "A"
    # Osci unavailable offline
    assert body["osci_available"] is False
    assert body["timebases"] == []
    assert body["current_timebase_index"] is None


def test_static_fft_enums_always_present(client: TestClient) -> None:
    body = client.get("/api/experimental/signals").json()
    win_vals = {w["value"] for w in body["windows"]}
    assert win_vals == {"hann", "hamming", "rect"}
    out_vals = {o["value"] for o in body["output_modes"]}
    assert out_vals == {"magnitude", "power"}
    # labels are present for UI rendering
    assert all(w["label"] for w in body["windows"])
    assert all(o["label"] for o in body["output_modes"])


def test_default_units_parsed_from_names(client: TestClient) -> None:
    body = client.get("/api/experimental/signals").json()
    by_name = {c["name"]: c for c in body["channels"]}
    assert by_name["Bias (V)"]["unit"] == "V"
    assert by_name["Z (m)"]["unit"] == "m"
    assert by_name["Current (A)"]["unit"] == "A"


# ── live relay (fake ExecutionContext backend) ───────────────────────────────
class _Res:
    def __init__(self, success: bool, data: dict | None = None, error: str | None = None):
        self.success = success
        self.data = data or {}
        self.error = error


class _FakeRegistry:
    """Stand-in skill registry whose run() returns canned skill results."""

    def __init__(self, results: dict, osci_raises: bool = False):
        self._results = results
        self._osci_raises = osci_raises


class _FakePool:
    pass


class _FakeState:
    pass


def _live_ctx(channel_names, current_indices, timebases=None, *,
              osci_unavailable=False) -> AppContext:
    """Wire a ctx whose ExecutionContext.run is monkeypatch-replaced to relay
    fake ListSignalChannels / GetOsciTimebases results."""
    ctx = AppContext()
    ctx.connection_pool = _FakePool()
    ctx.state = _FakeState()
    # ``skill_registry`` is a read-only property; integration wires it via
    # AppContext.wire(). The route also accepts a plain ``registry`` attr, which
    # is what the live MASTApp sets — use that here.
    ctx.registry = _FakeRegistry({})
    ctx._chan = channel_names
    ctx._cur = current_indices
    ctx._tb = timebases
    ctx._osci_unavailable = osci_unavailable
    return ctx


def _patch_run(monkeypatch, ctx) -> None:
    """Replace ExecutionContext.run so no real hardware / pool is touched."""
    import mast.core.execution_context as ec_mod

    def _fake_run(self, skill_name, params):
        if skill_name == "ListSignalChannels":
            channels = [{"index": i, "name": n} for i, n in enumerate(ctx._chan)]
            return _Res(True, {
                "channels": channels,
                "n_channels": len(channels),
                "declared_n": getattr(ctx, "_declared", None),
                "truncated": bool(getattr(ctx, "_declared", None)
                                  and ctx._declared > len(channels)),
                "current_indices": ctx._cur,
            })
        if skill_name == "GetOsciTimebases":
            if ctx._osci_unavailable:
                return _Res(False, error="Osci1T module is not loaded")
            return _Res(True, {
                "timebases": ctx._tb or [],
                "n_timebases": len(ctx._tb or []),
                "current_index": 0,
            })
        return _Res(False, error=f"unknown skill {skill_name}")

    monkeypatch.setattr(ec_mod.ExecutionContext, "run", _fake_run, raising=True)


def test_signals_live_relay(monkeypatch) -> None:
    ctx = _live_ctx(
        channel_names=["Current (A)", "Bias (V)", "Z (m)", "Input 1 (V)"],
        current_indices=[0],
        timebases=[
            {"index": 0, "dt_s": 5e-5, "fs_hz": 20000.0},
            {"index": 1, "dt_s": 1e-4, "fs_hz": 10000.0},
        ],
    )
    _patch_run(monkeypatch, ctx)
    c = _client(ctx)
    r = c.get("/api/experimental/signals")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["source"] == "live"
    assert body["n_channels"] == 4
    assert body["current_indices"] == [0]
    assert body["channels"][0]["name"] == "Current (A)"
    assert body["channels"][0]["unit"] == "A"
    assert body["channels"][0]["is_current"] is True
    assert body["channels"][1]["unit"] == "V"
    # Osci timebases relayed
    assert body["osci_available"] is True
    assert len(body["timebases"]) == 2
    assert body["timebases"][0]["fs_hz"] == 20000.0
    assert body["current_timebase_index"] == 0


def test_signals_live_channels_osci_unloaded(monkeypatch) -> None:
    """Live channels but Osci1T module unloaded ⇒ channels live, timebases empty,
    osci_available False — the two degrade independently."""
    ctx = _live_ctx(
        channel_names=["Current (A)", "Bias (V)"],
        current_indices=[0],
        osci_unavailable=True,
    )
    _patch_run(monkeypatch, ctx)
    c = _client(ctx)
    body = c.get("/api/experimental/signals").json()
    assert body["degraded"] is False
    assert body["source"] == "live"
    assert body["osci_available"] is False
    assert body["timebases"] == []
    assert body["current_timebase_index"] is None


def test_signals_live_channel_read_fails_falls_back(monkeypatch) -> None:
    """ExecutionContext wired but ListSignalChannels returns failure ⇒ the route
    hands back the default list (degraded) rather than an empty channel set."""
    ctx = AppContext()
    ctx.connection_pool = _FakePool()
    ctx.state = _FakeState()
    ctx.registry = _FakeRegistry({})

    import mast.core.execution_context as ec_mod

    monkeypatch.setattr(
        ec_mod.ExecutionContext, "run",
        lambda self, name, params: _Res(False, error="no hardware"),
        raising=True,
    )
    c = _client(ctx)
    body = c.get("/api/experimental/signals").json()
    assert body["degraded"] is True
    assert body["source"] == "default"
    assert body["n_channels"] > 0


# ── 名单截断（的前提：下拉框不能骗人） ─────────────────────────


def test_a_truncated_name_table_is_reported_not_swallowed(monkeypatch):
    """声明的通道数大于解析出的名称数时，端点应明确报告截断；不能把解析缺失误报为硬件缺失。"""
    ctx = _live_ctx([f"Sig {i}" for i in range(7)], [])
    ctx._declared = 16
    _patch_run(monkeypatch, ctx)
    b = _client(ctx).get("/api/experimental/signals").json()
    assert b["n_channels"] == 7
    assert b["declared_n"] == 16
    assert b["truncated"] is True


def test_a_complete_name_table_is_not_flagged(monkeypatch):
    """反面：正常时不许报截断，否则这个标志就永远在响、等于没有。"""
    ctx = _live_ctx([f"Sig {i}" for i in range(128)], [0])
    ctx._declared = 128
    _patch_run(monkeypatch, ctx)
    b = _client(ctx).get("/api/experimental/signals").json()
    assert b["n_channels"] == 128
    assert b["truncated"] is False


def test_an_instrument_that_declares_nothing_is_not_called_truncated(monkeypatch):
    """声明数读不出来时是「不知道」，不是「被截断」。

    把未知当成故障，和把故障当成正常，是同一种错的两个方向。
    """
    ctx = _live_ctx([f"Sig {i}" for i in range(16)], [])
    _patch_run(monkeypatch, ctx)          # 不设 _declared
    b = _client(ctx).get("/api/experimental/signals").json()
    assert b["declared_n"] is None
    assert b["truncated"] is False
