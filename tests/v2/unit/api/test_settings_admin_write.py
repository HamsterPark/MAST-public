"""Wave A contract tests for domain ``settings_admin_write``.

The router is NOT yet mounted in mast.api.app (integration wires that), so
each test builds a throwaway FastAPI app and includes the router under /api.

Guarantees asserted:
  * every endpoint returns 200 + a body matching its response_model;
  * with no live core wired (standalone AppContext) every registry-backed
    endpoint DEGRADES — empty-but-valid, ``degraded: true``, never 500;
  * the unified settings write hits a pure-stdlib SettingsStore (isolated tmp
    config dir) and actually persists whitelisted keys + drops the rest;
  * when a real ConfigOverrideRegistry IS wired (pointed at a tmp dir) the
    per-agent / knowledge / guidance / encyclopedia / quick-prompt writes
    persist + hot-reload and the read reflects them — proving the passthrough.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.settings_admin_write import router


def _client(user_root: str | None = None) -> TestClient:
    app = FastAPI()
    app.state.ctx = AppContext(user_root=user_root)
    app.include_router(router, prefix="/api")
    return TestClient(app)


def _client_with_registry(tmp_path, *, config=None) -> tuple[TestClient, object]:
    """Throwaway app with a REAL registry pointed at a tmp dir (leader-style
    wiring: attribute set on ctx). Optionally a fake live config."""
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "overrides")
    app = FastAPI()
    ctx = AppContext(user_root=str(tmp_path))
    ctx.override_registry = reg
    if config is not None:
        ctx.config = config
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    return TestClient(app), reg


@pytest.fixture()
def client() -> TestClient:
    return _client()


# ── Unified settings write ───────────────────────────────────────────────────
def test_settings_write_persists_whitelisted(tmp_path) -> None:
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"theme": "Dark", "font_scale": "大"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["degraded"] is False
    assert body["persisted"]["theme"] == "Dark"
    assert body["persisted"]["font_scale"] == "大"
    # no live config wired → nothing applied to config.llm, but NOT degraded
    assert body["applied"] == []


def test_a_key_the_model_does_not_have_is_refused_not_dropped(tmp_path) -> None:
    """2026-08-10 反转的一条：以前这里断言 ``"bogus"`` 被**静默丢掉**而整笔
    仍然 ``ok:true``。

    那正是当天在 6.2.6 上骗过用户的行为 —— ``POST
    {"tool_refine_min_chars": 1234}`` 得到 ``ok:true, rejected:{}``，而
    ``GET`` 回来是 ``None``。「收下了但没写」和「写了」在响应里长得一模一样。

    ``SettingsWriteRequest`` 现在是 ``extra='forbid'``：模型上没有的键 422，
    **整笔不写**。代价写在这里，别当成意外：同一个 body 里那两个合法的键
    也不会存进去 —— 一次拒绝是整笔的，半写半不写更难查。
    """
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"theme": "Dark", "font_scale": "大", "bogus": "x"})
    assert r.status_code == 422, f"静默丢弃回来了：{r.status_code} {r.text[:200]}"
    assert "bogus" in r.text
    # 整笔不写：合法的那两个也不能溜进去
    from mast.webui.settings_store import SettingsStore, default_config_dir
    assert SettingsStore(default_config_dir(tmp_path)).load() == {}


def test_settings_write_ignores_none(tmp_path) -> None:
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"model_alias": None})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "model_alias" not in body["persisted"]


def test_settings_write_applies_to_live_config(tmp_path) -> None:
    """When a live config IS wired, model_alias + thinking are pushed to
    config.llm via use()/set_thinking()."""

    class _LLM:
        def __init__(self):
            self.model = ""
            self.used = None
            self.think = None

        def use(self, name):
            self.used = name
            self.model = name

        def set_thinking(self, level):
            self.think = level

    class _Cfg:
        def __init__(self):
            self.llm = _LLM()

    cfg = _Cfg()
    app = FastAPI()
    ctx = AppContext(user_root=str(tmp_path))
    ctx.config = cfg
    app.state.ctx = ctx
    app.include_router(router, prefix="/api")
    c = TestClient(app)

    r = c.post("/api/settings", json={"model_alias": "glm-5.2", "thinking": "high"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert set(body["applied"]) == {"model_alias", "thinking"}
    assert cfg.llm.used == "glm-5.2"
    assert cfg.llm.think == "high"


# ── Per-agent model override ──────────────────────────────────────────────────
def test_agent_model_override_degrades_unwired(client: TestClient) -> None:
    r = client.post(
        "/api/agents/instrument_control/model-override",
        json={"model": "glm-5.2", "thinking": "high"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["degraded"] is True
    assert body["agent_id"] == "instrument_control"


def test_agent_model_override_with_live_registry(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    r = c.post(
        "/api/agents/literature/model-override",
        json={"model": "kimi-k2.6", "thinking": "max"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["override"] == {"model": "kimi-k2.6", "thinking": "max"}
    # persisted into the registry
    assert reg.get_agent_override("literature") == {"model": "kimi-k2.6", "thinking": "max"}

    # partial merge: only thinking changes, model stays
    r2 = c.post("/api/agents/literature/model-override", json={"thinking": "low"})
    assert r2.json()["override"] == {"model": "kimi-k2.6", "thinking": "low"}


# ── Knowledge edit ────────────────────────────────────────────────────────────
def test_knowledge_get_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/admin/knowledge/workflows")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "workflows"
    assert body["degraded"] is True
    assert body["has_override"] is False


def test_knowledge_write_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/admin/knowledge/workflows", json={"data": {"x": 1}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert body["reloaded"] is False


def test_knowledge_write_with_live_registry(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    w = c.post(
        "/api/admin/knowledge/fault_diagnosis",
        json={"data": {"tip_crash": {"tip": "withdraw"}}},
    )
    assert w.status_code == 200
    wb = w.json()
    # `reloaded` is False because nothing subscribes to the override hot-reload
    # (it was a hardcoded True until 2026-08-03). The write still persisted.
    assert wb["ok"] is True and wb["degraded"] is False and wb["reloaded"] is False
    assert wb["override"] == {"tip_crash": {"tip": "withdraw"}}

    g = c.get("/api/admin/knowledge/fault_diagnosis")
    gb = g.json()
    assert gb["has_override"] is True
    assert gb["override"] == {"tip_crash": {"tip": "withdraw"}}

    # empty payload ⇒ reset that ktype (override dropped)
    d = c.post("/api/admin/knowledge/fault_diagnosis", json={"data": {}})
    assert d.json()["ok"] is True
    assert c.get("/api/admin/knowledge/fault_diagnosis").json()["has_override"] is False


# ── Guidance edit ─────────────────────────────────────────────────────────────
def test_guidance_get_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/admin/guidance/scan_area")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_guidance_write_with_live_registry(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    w = c.post(
        "/api/admin/guidance/scan_area",
        json={"data": {"hint": "slow down near steps"}},
    )
    assert w.status_code == 200
    wb = w.json()
    # `reloaded` is False because nothing subscribes to the override hot-reload
    # (it was a hardcoded True until 2026-08-03). The write still persisted.
    assert wb["ok"] is True and wb["reloaded"] is False
    assert wb["override"] == {"hint": "slow down near steps"}
    # stored under skill_extra in guidance_overrides.json
    assert reg.get_guidance_override("scan_area") == {"hint": "slow down near steps"}

    g = c.get("/api/admin/guidance/scan_area")
    assert g.json()["has_override"] is True

    d = c.post("/api/admin/guidance/scan_area", json={"data": {}})
    assert d.json()["ok"] is True
    assert c.get("/api/admin/guidance/scan_area").json()["has_override"] is False


# ── Encyclopedia config edit ──────────────────────────────────────────────────
def test_encyclopedia_get_unknown_section_degrades(client: TestClient) -> None:
    r = client.get("/api/encyclopedia/config/not_a_section")
    assert r.status_code == 200
    assert r.json()["degraded"] is True


def test_encyclopedia_get_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/encyclopedia/config/domains")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "domains"
    assert body["degraded"] is True


def test_encyclopedia_write_with_live_registry(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    w = c.post(
        "/api/encyclopedia/config/domains",
        json={"data": [{"id": "stm", "label": "STM"}]},
    )
    assert w.status_code == 200
    wb = w.json()
    # `reloaded` is False because nothing subscribes to the override hot-reload
    # (it was a hardcoded True until 2026-08-03). The write still persisted.
    assert wb["ok"] is True and wb["reloaded"] is False
    assert wb["override"] == [{"id": "stm", "label": "STM"}]
    # stored under canonical key "domains"
    assert reg.get_encyclopedia_overrides()["domains"] == [{"id": "stm", "label": "STM"}]

    # "intents" maps to canonical "intent_mapping"
    wi = c.post(
        "/api/encyclopedia/config/intents",
        json={"data": [{"intent": "scan", "skill": "scan_area"}]},
    )
    assert wi.json()["ok"] is True
    assert "intent_mapping" in reg.get_encyclopedia_overrides()

    g = c.get("/api/encyclopedia/config/domains")
    assert g.json()["has_override"] is True

    d = c.post("/api/encyclopedia/config/domains", json={"data": []})
    assert d.json()["ok"] is True
    assert c.get("/api/encyclopedia/config/domains").json()["has_override"] is False


# ── Quick prompts ─────────────────────────────────────────────────────────────
def test_quick_prompts_get_degrades_unwired(client: TestClient) -> None:
    r = client.get("/api/admin/quick-prompts")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["prompts"] == [] and body["count"] == 0


def test_quick_prompts_write_degrades_unwired(client: TestClient) -> None:
    r = client.post("/api/admin/quick-prompts", json={"prompts": [{"label": "x"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True


def test_quick_prompts_write_with_live_registry(tmp_path) -> None:
    c, reg = _client_with_registry(tmp_path)
    prompts = [{"label": "扫描", "text": "scan a clean area"}]
    w = c.post("/api/admin/quick-prompts", json={"prompts": prompts})
    assert w.status_code == 200
    wb = w.json()
    # `reloaded` is False because nothing subscribes to the override hot-reload
    # (it was a hardcoded True until 2026-08-03). The write still persisted.
    assert wb["ok"] is True and wb["reloaded"] is False
    assert wb["prompts"] == prompts
    assert reg.get_quick_prompts() == prompts

    g = c.get("/api/admin/quick-prompts")
    gb = g.json()
    assert gb["count"] == 1 and gb["has_override"] is True

    # empty list ⇒ reset to code defaults (override file deleted)
    d = c.post("/api/admin/quick-prompts", json={"prompts": []})
    assert d.json()["ok"] is True
    assert c.get("/api/admin/quick-prompts").json()["has_override"] is False


# ── Vision tip-quality thresholds (live-apply to the process holder) ──────────
@pytest.fixture(autouse=True)
def _reset_vision_thresholds():
    """The threshold holder is process-global — reset it after every test so a
    live-apply never leaks into another test."""
    yield
    from mast.vision.thresholds import set_thresholds
    set_thresholds(None)


def test_settings_write_vision_thresholds_persists_and_live_applies(tmp_path) -> None:
    from mast.vision.thresholds import VisionThresholds, get_thresholds

    c = _client(user_root=str(tmp_path))
    th = {"coarse_good_threshold": 0.4, "multi_apex_p_max": 0.65, "m0_quality_min": 65.0}
    r = c.post("/api/settings", json={"vision_thresholds": th})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["persisted"]["vision_thresholds"] == th
    assert "vision_thresholds" in body["applied"]
    # the live holder reflects the new cuts (next tip assessment sees them, no
    # model reload); fields not sent keep their defaults.
    active = get_thresholds()
    assert active.coarse_good_threshold == 0.4
    assert active.multi_apex_p_max == 0.65
    assert active.m0_quality_min == 65.0
    assert active.contam_p_max == VisionThresholds().contam_p_max


def test_settings_write_without_vision_thresholds_leaves_holder(tmp_path) -> None:
    from mast.vision.thresholds import get_thresholds, set_thresholds

    set_thresholds({"coarse_good_threshold": 0.42})
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"theme": "Dark"})
    assert r.status_code == 200
    assert "vision_thresholds" not in r.json()["applied"]
    assert get_thresholds().coarse_good_threshold == 0.42  # unrelated write, untouched


# ── Classical (network-free) tip-quality thresholds — same live-apply pattern ──
@pytest.fixture(autouse=True)
def _reset_classical_thresholds():
    yield
    from mast.vision.classical_thresholds import set_classical_thresholds
    set_classical_thresholds(None)


def test_settings_write_classical_thresholds_persists_and_live_applies(tmp_path) -> None:
    from mast.vision.classical_thresholds import ClassicalThresholds, get_classical_thresholds

    c = _client(user_root=str(tmp_path))
    th = {"tq_double_threshold": 0.30, "tq_instability_max": 0.6, "oscillation_threshold": 5.0}
    r = c.post("/api/settings", json={"classical_thresholds": th})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["persisted"]["classical_thresholds"] == th
    assert "classical_thresholds" in body["applied"]
    active = get_classical_thresholds()
    assert active.tq_double_threshold == 0.30
    assert active.tq_instability_max == 0.6
    assert active.oscillation_threshold == 5.0
    assert active.iz_barrier_min == ClassicalThresholds().iz_barrier_min   # unset → default


def test_settings_write_classical_thresholds_clamps_out_of_range(tmp_path) -> None:
    from mast.vision.classical_thresholds import get_classical_thresholds

    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"classical_thresholds": {"tq_instability_max": 9.0}})
    assert r.status_code == 200
    assert get_classical_thresholds().tq_instability_max == 1.0   # clamped to FIELD_BOUNDS


# ── Tunnelling-current monitor knobs — same live-read holder ─────────────────
@pytest.fixture(autouse=True)
def _reset_monitor_thresholds():
    yield
    from mast.monitoring.thresholds import set_monitor_thresholds
    set_monitor_thresholds(None)


def test_settings_write_current_monitor_persists_and_live_applies(tmp_path) -> None:
    """Without the KNOWN_KEYS entry this silently no-ops — the failure mode is a
    settings page that appears to save and changes nothing."""
    from mast.monitoring.thresholds import MonitorThresholds, get_monitor_thresholds

    c = _client(user_root=str(tmp_path))
    knobs = {"cm_segment_s": 2.0, "cm_keep_gb": 8.0, "cm_rms_warn_a": 5e-12}
    r = c.post("/api/settings", json={"current_monitor": knobs})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["persisted"]["current_monitor"] == knobs
    assert "current_monitor" in body["applied"]

    active = get_monitor_thresholds()
    assert active.cm_segment_s == 2.0
    assert active.cm_keep_gb == 8.0
    assert active.cm_keep_hours == MonitorThresholds().cm_keep_hours   # unset → default


def test_settings_write_current_monitor_can_switch_acquisition_off(tmp_path) -> None:
    from mast.monitoring.thresholds import get_monitor_thresholds

    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"current_monitor": {"cm_enabled": 0.0}})
    assert r.status_code == 200
    assert get_monitor_thresholds().enabled is False


def test_settings_write_current_monitor_clamps_out_of_range(tmp_path) -> None:
    from mast.monitoring.thresholds import get_monitor_thresholds

    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"current_monitor": {"cm_segment_s": 999.0}})
    assert r.status_code == 200
    assert get_monitor_thresholds().cm_segment_s == 10.0           # FIELD_BOUNDS max


def test_settings_write_current_monitor_is_not_pin_guarded(tmp_path) -> None:
    """The PIN protects handing dangerous capabilities to agents; retuning a
    read-only monitor is an operations action, like resetting the usage ledger."""
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"current_monitor": {"cm_keep_hours": 12.0}})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "pin" not in (r.json().get("note") or "").lower()


def test_settings_get_echoes_current_monitor(tmp_path) -> None:
    """Round trip through GET /api/settings — a write the read cannot see would
    leave the UI showing stale knobs."""
    from mast.api.routes.settings import router as read_router

    c = _client(user_root=str(tmp_path))
    c.post("/api/settings", json={"current_monitor": {"cm_keep_gb": 6.0}})

    app = FastAPI()
    app.state.ctx = AppContext(user_root=str(tmp_path))
    app.include_router(read_router, prefix="/api")
    body = TestClient(app).get("/api/settings").json()
    assert body["current_monitor"] == {"cm_keep_gb": 6.0}


# ── Experiment default-parameter preferences (#147, live-apply to holder) ─────
@pytest.fixture(autouse=True)
def _reset_experiment_prefs():
    """The prefs holder is process-global — reset after every test so a
    live-apply never leaks into another test."""
    yield
    from mast.agents._shared.experiment_prefs import set_prefs
    set_prefs(None)


def test_settings_write_experiment_defaults_persists_and_live_applies(tmp_path) -> None:
    from mast.agents._shared.experiment_prefs import get_prefs

    c = _client(user_root=str(tmp_path))
    prefs = {"scan_size_nm": 50, "setpoint_pa": 100, "bias_v": 0.5, "notes": "常温"}
    r = c.post("/api/settings", json={"experiment_defaults": prefs})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["persisted"]["experiment_defaults"] == prefs
    assert "experiment_defaults" in body["applied"]
    # the live holder reflects the sanitized prefs (IC / experiment_design see
    # them on the next turn, no graph rebuild)
    active = get_prefs()
    assert active["scan_size_nm"] == 50.0
    assert active["setpoint_pa"] == 100.0
    assert active["bias_v"] == 0.5
    assert active["notes"] == "常温"


def test_settings_write_experiment_defaults_sanitizes_junk(tmp_path) -> None:
    from mast.agents._shared.experiment_prefs import get_prefs

    c = _client(user_root=str(tmp_path))
    # bias clamped to 10, unknown key dropped by the holder sanitiser
    r = c.post("/api/settings", json={"experiment_defaults": {"bias_v": 99, "junk": "x"}})
    assert r.status_code == 200
    active = get_prefs()
    assert active == {"bias_v": 10.0}


def test_settings_write_without_experiment_defaults_leaves_holder(tmp_path) -> None:
    from mast.agents._shared.experiment_prefs import get_prefs, set_prefs

    set_prefs({"scan_size_nm": 30})
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"theme": "Dark"})
    assert r.status_code == 200
    assert "experiment_defaults" not in r.json()["applied"]
    assert get_prefs() == {"scan_size_nm": 30.0}  # unrelated write, untouched


def test_settings_write_instrument_profile_persists_and_live_applies(tmp_path) -> None:
    from mast.core import instrument_profile as ip

    ip.set_profile({}); ip.set_persist_sink(None)
    c = _client(user_root=str(tmp_path))
    profile = {"retract_motor_dir": "z-", "retract_total_steps": 5000,
               "lockin_signal_index": 8, "lockin_mod_amp_v": 0.02}
    r = c.post("/api/settings", json={"instrument_profile": profile})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["persisted"]["instrument_profile"]["retract_motor_dir"] == "z-"
    assert "instrument_profile" in body["applied"]
    # the live holder reflects the config (retract/approach skills + the injecting
    # middleware see it on the next turn, no graph rebuild)
    assert ip.get_config("retract_motor_dir") == "z-"
    assert ip.get_retract_dir_code() == 5           # z- → Nanonis motor code 5
    assert ip.get_config("retract_total_steps") == 5000
    ip.set_profile({}); ip.set_persist_sink(None)


def test_settings_write_instrument_profile_sanitizes_junk(tmp_path) -> None:
    from mast.core import instrument_profile as ip

    ip.set_profile({}); ip.set_persist_sink(None)
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"instrument_profile": {
        "retract_step_max": 99999,        # clamped to 1000
        "retract_motor_dir": "sideways",  # invalid enum → dropped
        "junk": "x"}})                     # unknown → dropped
    assert r.status_code == 200
    prof = ip.get_profile()
    assert prof.get("retract_step_max") == 1000
    assert "retract_motor_dir" not in prof
    assert "junk" not in prof
    ip.set_profile({}); ip.set_persist_sink(None)


def test_settings_write_without_instrument_profile_leaves_holder(tmp_path) -> None:
    from mast.core import instrument_profile as ip

    ip.set_profile({"retract_motor_dir": "z-"}); ip.set_persist_sink(None)
    c = _client(user_root=str(tmp_path))
    r = c.post("/api/settings", json={"theme": "Dark"})
    assert r.status_code == 200
    assert "instrument_profile" not in r.json()["applied"]
    assert ip.get_config("retract_motor_dir") == "z-"  # unrelated write, untouched
    ip.set_profile({}); ip.set_persist_sink(None)
