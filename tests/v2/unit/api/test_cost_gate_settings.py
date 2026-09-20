"""The cost ceilings are operator-adjustable, end to end.

A new persisted setting has to appear in THREE independent whitelists before it
works, and missing one fails **silently** in a different way each time:

  * ``SettingsStore.KNOWN_KEYS`` — the write is dropped, the API still returns 200,
    and the value is simply gone on the next read. (This exact gap once broke the
    admin PIN gate.)
  * ``schemas.py`` — the value persists but the GET never shows it, so the UI
    renders an empty box over a live setting.
  * ``schemas_settings_admin_write.py`` — the POST body field is rejected before it
    reaches the store.

None of those raise. So this file walks the whole path — POST → store → GET → **and
the two functions that actually consume the value** — because "it saved" and "the
gate is using it" are different claims and only the second one matters.

Defaults (2026-07-30): per-run $80, per-day $300. Both are BACKSTOPS, not quotas: a
legitimate multi-stage pipeline measured ~$5.19, and the runaway they exist to stop
was $30.66/hour. Tuning them close to observed history would abort long campaigns
mid-thought, which is both likelier and more annoying than an expensive run.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.app import create_app  # noqa: E402
from mast.api.routes.orchestrator import (  # noqa: E402
    _RUN_BUDGET_USD,
    _effective_run_budget_usd,
)
from mast.core.runtime import CoreRuntime  # noqa: E402
from mast.core.wake_scheduler import DEFAULT_DAILY_BUDGET_USD  # noqa: E402
from mast.webui.settings_store import SettingsStore  # noqa: E402

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    """A real SettingsStore on a temp file — never the operator's."""
    return SettingsStore(tmp_path / "ui_settings.json")


@pytest.fixture()
def client(store):
    app = create_app()
    # Same injection point the sibling hardware-modules API test uses.
    app.state.ctx._settings_store = store  # noqa: SLF001
    return TestClient(app)


class _Host:
    """Stands in for the app/runtime object the two readers take."""

    def __init__(self, store):
        self._settings = store


# ════════════════════════════════════════════════════════════════════
# Defaults
# ════════════════════════════════════════════════════════════════════

class TestDefaults:
    def test_the_defaults_are_the_documented_ones(self):
        assert _RUN_BUDGET_USD == 80.0
        assert DEFAULT_DAILY_BUDGET_USD == 300.0

    def test_the_daily_ceiling_is_well_above_the_per_run_one(self):
        """They bound different things: one run going long is normal; several runs'
        worth of spend in a day is the shape worth stopping. A daily ceiling at or
        below the per-run one would make the per-run gate unreachable."""
        assert DEFAULT_DAILY_BUDGET_USD > _RUN_BUDGET_USD * 2

    def test_the_default_leaves_real_headroom_over_measured_work(self):
        """A legitimate 4-agent run with a revision cycle measured $5.19. A ceiling
        hugging that number turns a long campaign into a mid-thought abort."""
        assert _RUN_BUDGET_USD > 5.19 * 5

    def test_the_default_still_catches_the_measured_runaway(self):
        """The whole point. The recorded runaway was $30.66 in 60 minutes — and
        before this gate was wired, the only bound was recursion_limit (~$50)."""
        assert _RUN_BUDGET_USD > 30.66
        assert _RUN_BUDGET_USD < 200, "a ceiling this high stops being a ceiling"

    def test_an_unconfigured_host_gets_the_default(self):
        assert _effective_run_budget_usd(_Host(None)) == _RUN_BUDGET_USD
        assert CoreRuntime._effective_daily_budget_usd(_Host(None)) == \
            DEFAULT_DAILY_BUDGET_USD


# ════════════════════════════════════════════════════════════════════
# The whole adjustable path
# ════════════════════════════════════════════════════════════════════

class TestOperatorCanAdjustThem:
    def test_a_written_run_budget_is_read_back_by_the_gate(self, client, store):
        """POST → store → GET → **and the function the supervisor actually calls**.
        The last hop is the one that matters: a value that persists but nothing reads
        is a setting in name only."""
        r = client.post("/api/settings", json={"orchestrator_run_budget_usd": 42.5})
        assert r.status_code == 200
        assert client.get("/api/settings").json()["orchestrator_run_budget_usd"] == 42.5
        assert _effective_run_budget_usd(_Host(store)) == 42.5

    def test_a_written_daily_budget_is_read_back_by_the_scheduler(self, client, store):
        r = client.post("/api/settings", json={"daily_budget_usd": 175.0})
        assert r.status_code == 200
        assert client.get("/api/settings").json()["daily_budget_usd"] == 175.0
        assert CoreRuntime._effective_daily_budget_usd(_Host(store)) == 175.0

    def test_the_activation_gate_is_adjustable_too(self, client, store):
        client.post("/api/settings", json={"orchestrator_activation_gating": True})
        assert client.get("/api/settings").json()["orchestrator_activation_gating"] is True

    def test_both_budgets_survive_a_restart(self, tmp_path):
        """The store is the durable one; a ceiling that resets on restart is a
        ceiling the operator has to keep re-setting and will eventually stop trusting."""
        p = tmp_path / "s.json"
        SettingsStore(p).update(orchestrator_run_budget_usd=12.0,
                                daily_budget_usd=99.0)
        again = SettingsStore(p)
        assert _effective_run_budget_usd(_Host(again)) == 12.0
        assert CoreRuntime._effective_daily_budget_usd(_Host(again)) == 99.0

    def test_setting_one_does_not_disturb_the_other(self, client, store):
        client.post("/api/settings", json={"orchestrator_run_budget_usd": 20.0})
        client.post("/api/settings", json={"daily_budget_usd": 60.0})
        got = client.get("/api/settings").json()
        assert got["orchestrator_run_budget_usd"] == 20.0
        assert got["daily_budget_usd"] == 60.0


class TestZeroMeansOff:
    def test_zero_turns_the_run_gate_off_rather_than_ending_every_run(self, store):
        """0 is what an unconfigured numeric setting looks like, so for a CEILING it
        has to mean "no ceiling". Reading it as a $0 budget would end every run on its
        first hop and look like a broken orchestrator."""
        store.update(orchestrator_run_budget_usd=0)
        assert _effective_run_budget_usd(_Host(store)) == 0.0

    def test_zero_turns_the_daily_gate_off(self, store):
        store.update(daily_budget_usd=0)
        assert CoreRuntime._effective_daily_budget_usd(_Host(store)) == 0.0

    def test_a_negative_value_is_off_not_a_tiny_ceiling(self, store):
        """Clamping a negative to something small would invent a ceiling the operator
        never asked for and abort every run."""
        store.update(orchestrator_run_budget_usd=-5)
        assert _effective_run_budget_usd(_Host(store)) == 0.0

    def test_an_unparseable_value_falls_back_to_the_default(self, store):
        store.update(orchestrator_run_budget_usd="lots")
        assert _effective_run_budget_usd(_Host(store)) == _RUN_BUDGET_USD


class TestWakeQuotaSetting:
    """The per-experiment auto-wake ceiling — THE loop bound for product-driven waking.

    Every other guard in the system is per-run, and a wake starts a NEW run (fresh
    recursion_limit, fresh visit_count, fresh budget). Every step of a PW⇄PR wake
    cycle SUCCEEDS, so StallGuard — which keys off repeated FAILURE signatures —
    cannot see it either. This counter is the only thing that can.
    """

    def test_the_default_is_the_documented_one(self):
        from mast.core.wake_scheduler import DEFAULT_MAX_WAKES_PER_DAY

        assert DEFAULT_MAX_WAKES_PER_DAY == 6
        assert CoreRuntime._effective_max_wakes_per_day(_Host(None)) == 6

    def test_a_written_value_is_read_back(self, client, store):
        r = client.post("/api/settings", json={"wake_max_per_day": 3})
        assert r.status_code == 200
        assert client.get("/api/settings").json()["wake_max_per_day"] == 3
        assert CoreRuntime._effective_max_wakes_per_day(_Host(store)) == 3

    def test_zero_means_no_automatic_waking_at_all(self, store):
        """Opposite convention from the USD ceilings, deliberately. An operator typing
        0 into a SAFETY COUNTER means "do not do this"; 0 in a numeric CEILING is just
        what an unconfigured setting looks like and must mean "no ceiling"."""
        store.update(wake_max_per_day=0)
        assert CoreRuntime._effective_max_wakes_per_day(_Host(store)) == 0

    def test_unlimited_is_not_reachable_from_settings(self, store):
        """The scheduler accepts a negative as "disable the check" — useful in tests,
        and it must NOT be reachable from the settings page. Offering a one-click
        "unlimited" would be offering to remove the only bound that exists on a wake
        loop. Someone who needs headroom raises the number instead."""
        store.update(wake_max_per_day=-1)
        assert CoreRuntime._effective_max_wakes_per_day(_Host(store)) == 0, \
            "a negative reached the scheduler — the loop breaker can be switched off"

    def test_a_garbage_value_falls_back_to_the_default(self, store):
        store.update(wake_max_per_day="lots")
        assert CoreRuntime._effective_max_wakes_per_day(_Host(store)) == 6

    def test_the_runtime_actually_passes_it_to_the_scheduler(self):
        """A setting that is read but never handed to the thing it configures is a
        setting in name only — the failure this whole test file exists to catch."""
        import inspect

        src = source_of(CoreRuntime._ensure_wake_scheduler)
        assert "max_wakes_per_day=" in src and "_effective_max_wakes_per_day" in src


class TestTheThreeWhitelistsAgree:
    """Each of the three lists fails silently and differently when it is missed, so
    the agreement itself is worth asserting rather than rediscovering."""

    KEYS = ("orchestrator_run_budget_usd", "daily_budget_usd",
            "orchestrator_activation_gating", "wake_max_per_day")

    def test_the_store_persists_every_key(self):
        from mast.webui.settings_store import KNOWN_KEYS

        for k in self.KEYS:
            assert k in KNOWN_KEYS, \
                f"{k} missing from KNOWN_KEYS — writes are a SILENT no-op"

    def test_the_read_schema_exposes_every_key(self):
        from mast.api.schemas import SettingsResponse

        for k in self.KEYS:
            assert k in SettingsResponse.model_fields, \
                f"{k} missing from the read schema — the UI would render an empty box"

    def test_the_write_schema_accepts_every_key(self):
        from mast.api.schemas_settings_admin_write import SettingsWriteRequest

        for k in self.KEYS:
            assert k in SettingsWriteRequest.model_fields, \
                f"{k} missing from the write schema — the POST field is dropped"

    def test_the_generated_openapi_carries_them(self, client):
        """Guards the step that is easy to forget after a schema change: regenerate
        openapi.json + schema.d.ts, or the frontend's typed client silently lacks the
        field and `tsc` passes anyway."""
        spec = client.app.openapi()
        props = spec["components"]["schemas"]["SettingsResponse"]["properties"]
        for k in self.KEYS:
            assert k in props
