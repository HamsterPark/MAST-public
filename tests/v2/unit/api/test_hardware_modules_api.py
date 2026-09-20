"""The hardware-module settings API: read, write, and the rebuild that makes it real.

The subtle failure this file exists to prevent: a toggle that PERSISTS but is never
LIVE. The agent's tool list is frozen at graph-build time, so flipping a switch
writes the JSON, updates the holder, turns the UI green — and the running agent
keeps exactly the tools it had. The write route must therefore ask the runtime to
rebuild, and must report honestly when it cannot.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mast.api.app import create_app
from mast.skills import hardware_modules as hm
from mast.webui.settings_store import SettingsStore


@pytest.fixture(autouse=True)
def _restore_holder():
    before = hm.enabled_ids()
    yield
    hm.set_enabled(before)


# 2026-07-13: hardware_modules moved behind the admin PIN (高级 → 能力开关).
# Switching a module ON hands the agent DANGEROUS skills — a laser, an RF amplifier,
# probes that can collide — which is not the same class of action as changing the
# font size. So every write in this file now carries the PIN, and two new tests pin
# the guard itself.
_PIN = "1234"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: tmp_path)
    from mast.api.admin_pin import set_pin
    ok, reason = set_pin(_PIN)
    assert ok, reason

    app = create_app()
    ctx = app.state.ctx
    ctx._settings_store = SettingsStore(tmp_path)
    hm.set_enabled({})
    return TestClient(app)


def _post_modules(client, modules: dict, pin: str | None = _PIN):
    body: dict = {"hardware_modules": modules}
    if pin is not None:
        body["admin_pin"] = pin
    return client.post("/api/settings", json=body)


class _FakeRuntime:
    """Stands in for CoreRuntime. Records whether the rebuild was requested."""

    def __init__(self, note="（agent 工具表后台重建中，数秒后生效）", boom=False):
        self.rebuild_calls = 0
        self._note = note
        self._boom = boom

    def _request_composite_rebuild(self) -> str:
        self.rebuild_calls += 1
        if self._boom:
            raise RuntimeError("orchestrator is wedged")
        return self._note


# ─────────────────────────────────────────────────────────────────────────────
# GET
# ─────────────────────────────────────────────────────────────────────────────

def test_get_lists_every_module_off_by_default(client):
    r = client.get("/api/settings/hardware-modules")
    assert r.status_code == 200
    body = r.json()
    assert len(body["modules"]) == len(hm.MODULES)
    assert body["enabled_count"] == 0
    assert all(not m["enabled"] for m in body["modules"])
    assert body["gated_skill_count"] == len(hm.SKILL_OWNER)
    # Each one tells the operator what it needs and what it costs.
    for m in body["modules"]:
        assert m["hardware"].strip()
        assert m["skill_count"] >= 1


def test_get_reflects_the_live_holder_not_just_the_store(client):
    """The gate reads the HOLDER. If this endpoint read the store instead, it could
    show a state that is not the one actually in force."""
    hm.set_enabled({"laser": True})
    body = client.get("/api/settings/hardware-modules").json()
    assert body["enabled_count"] == 1
    assert next(m for m in body["modules"] if m["id"] == "laser")["enabled"] is True


def test_the_literal_route_is_not_shadowed(client):
    """'hardware-modules' must resolve to the module list, never to a settings key
    lookup. (A path-param route shadowing a literal one once turned the group-chat
    read endpoints into a healthy-looking empty list.)"""
    body = client.get("/api/settings/hardware-modules").json()
    assert "modules" in body, f"路由被遮蔽了，返回的是 {sorted(body)[:5]}"


# ─────────────────────────────────────────────────────────────────────────────
# POST — persist + hold + rebuild
# ─────────────────────────────────────────────────────────────────────────────

def test_write_persists_and_updates_the_holder(client):
    r = _post_modules(client, {"laser": True})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "hardware_modules" in body["applied"]
    assert hm.enabled_ids() == frozenset({"laser"}), "holder 没跟着更新——门控读的就是它"
    assert client.app.state.ctx.settings_store.get("hardware_modules") == {"laser": True}


def test_write_requests_a_tool_list_rebuild(client):
    """THE test. Without this, the switch is cosmetic until the next restart."""
    rt = _FakeRuntime()
    client.app.state.ctx.live_app = rt
    r = _post_modules(client, {"kelvin": True})
    assert rt.rebuild_calls == 1, (
        "改硬件模块后没有请求重建 agent 工具表。开关会写进 JSON、holder 会更新、"
        "UI 会变绿——而运行中的 agent 的工具表一个都没变，直到重启。"
    )
    assert r.json()["rebuild_note"] == "（agent 工具表后台重建中，数秒后生效）"


def test_rebuild_declined_mid_task_is_reported_not_swallowed(client):
    """The runtime refuses to rebuild while a task runs (swapping the graph under a
    live run). That is correct — but the operator must be TOLD, or they will think
    the toggle took effect."""
    rt = _FakeRuntime(note="（任务运行中，agent 工具表暂不重建——任务结束后重试或重启生效）")
    client.app.state.ctx.live_app = rt
    body = _post_modules(client, {"laser": True}).json()
    assert body["ok"] is True                    # it DID persist
    assert "任务运行中" in body["rebuild_note"]   # …and it says it is not live yet


def test_rebuild_failure_is_reported_not_a_500(client):
    rt = _FakeRuntime(boom=True)
    client.app.state.ctx.live_app = rt
    body = _post_modules(client, {"laser": True}).json()
    assert body["ok"] is True
    assert "重启" in body["rebuild_note"]


def test_no_live_runtime_says_so(client):
    """Headless API / tests: nothing to rebuild. Persisted, holder set — but say
    plainly that it is not live, rather than implying it is."""
    body = _post_modules(client, {"laser": True}).json()
    assert body["ok"] is True
    assert "下次启动生效" in body["rebuild_note"]


def test_other_settings_writes_do_not_trigger_a_rebuild(client):
    """A rebuild is not free. Only a hardware_modules change may cause one."""
    rt = _FakeRuntime()
    client.app.state.ctx.live_app = rt
    client.post("/api/settings", json={"font_scale": "大"})
    assert rt.rebuild_calls == 0
    assert client.post("/api/settings", json={"font_scale": "大"}).json()["rebuild_note"] == ""


def test_the_store_replaces_wholesale_so_the_ui_must_send_everything(client):
    """Pins the trap the UI has to work around. SettingsStore REPLACES the dict; it
    does not merge. If the frontend ever posts only the toggled key, every other
    module silently switches off. HardwareModulesSection.toggle() sends the full map
    precisely because of this — the assertion below is what makes that requirement
    real rather than a comment."""
    _post_modules(client, {"laser": True, "kelvin": True})
    assert hm.enabled_ids() == frozenset({"laser", "kelvin"})

    # A "partial" write — exactly what a naive UI would send.
    _post_modules(client, {"osci_hr": True})
    assert hm.enabled_ids() == frozenset({"osci_hr"}), (
        "如果这里变成 {laser, kelvin, osci_hr}，说明 store 改成了合并语义——"
        "那么前端的 whole-replace guard 就是多余的；反之（现在这样）它是必须的。"
    )


def test_garbage_write_does_not_enable_everything(client):
    _post_modules(client, {"laser": True})
    # A dict of the right shape but nonsense ids.
    _post_modules(client, {"nope": True, "also_nope": True})
    assert hm.enabled_ids() == frozenset(), "未知 id 不应启用任何模块"


def test_settings_get_echoes_the_persisted_map(client):
    _post_modules(client, {"laser": True})
    assert client.get("/api/settings").json()["hardware_modules"] == {"laser": True}


# ─────────────────────────────────────────────────────────────────────────────
# The PIN guard (2026-07-13) — it used to be dead code that minted a token nobody
# checked. It now gates exactly the two keys that GRANT THE AGENT CAPABILITY.
# ─────────────────────────────────────────────────────────────────────────────

def test_a_module_cannot_be_switched_on_without_the_pin(client):
    """The refusal must be a REFUSAL — not a refusal that quietly applied anyway."""
    r = _post_modules(client, {"laser": True}, pin=None)
    body = r.json()
    assert body["ok"] is False
    assert body["pin_required"] is True
    assert body["pin_reason"] == "empty"
    assert hm.enabled_ids() == frozenset(), "写被拒绝了，模块却启用了"
    assert client.app.state.ctx.settings_store.get("hardware_modules") in (None, {})

    r = _post_modules(client, {"laser": True}, pin="9999")
    assert r.json()["pin_reason"] == "wrong"
    assert hm.enabled_ids() == frozenset()


def test_the_pin_is_never_persisted(client):
    """It is compared against a SHA-256 and dropped. A PIN that ends up in
    settings.json is a PIN written in plaintext to a file nobody guards."""
    _post_modules(client, {"laser": True})
    saved = client.app.state.ctx.settings_store.load()
    assert "admin_pin" not in saved
    assert _PIN not in str(saved)


def test_ordinary_settings_still_need_no_pin(client):
    """Only the two capability keys are guarded. Getting the font size wrong costs
    you nothing; making the PIN mandatory everywhere would just teach people to keep
    it in a sticky note next to the keyboard."""
    assert client.post("/api/settings", json={"font_scale": "大"}).json()["ok"] is True
