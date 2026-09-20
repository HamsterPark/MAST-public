"""针尖方案覆写:一个存在、会被读、却**写不进去**的机制。

## 2026-08-10

`tip_conditioning_policy.py` 与 `tip_conditioning_resolver.py` 都写着:覆写存在
`SettingsStore["tip_conditioning_overrides"]`,**优先级最高**。那个键确实在
`settings_store.py` 的 `KNOWN_KEYS` 里,resolver 每次 resolve 都读它。

**但它从来不是 `SettingsWriteRequest` 上的字段** —— 全仓没有任何 API 写得进这个键
(`instrument_init` 只写自己那一个)。三张白名单必须同时点头才写得进去
(KNOWN_KEYS / 读 schema / 写 schema),而这一条缺的是第三张。
这正是当时改不了针尖包络的原因。

「没有写入口」和「写了没生效」在界面上长得一模一样:都是「我填了,没反应」。

## 这个文件测什么

**端到端**:POST 进去 → resolver 真的用到了它。判据不是「存盘成功」,而是
**一次拒绝判定被它改变了** —— 那才证明它到达了实际做决定的那一层。
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.settings_admin_write import router

_PIN = "246813"

#: 一根 qPlus 针的事实。包络就在这一档上(2026-08-10 起 max_poke_depth_m = 5 nm)。
_QPLUS = {"id": 7, "name": "qp-e2e", "material": "W", "fabrication": "etched",
          "form": "qplus"}


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """一个完全隔离的设置目录 + 一枚 tmp 里的 admin PIN。

    PIN 文件默认住在 ``project_root()/config`` —— **真实用户的目录**。不重定向
    的话这个测试会往用户机器上写一枚 PIN(tests_polluting_real_user_data,
    本仓已四次)。
    """
    from mast.api import admin_pin
    from mast.webui import settings_store as ss

    pin_file = tmp_path / "admin_pin.txt"
    monkeypatch.setattr(admin_pin, "_pin_path", lambda: pin_file)
    ok, why = admin_pin.set_pin(_PIN)
    assert ok, why
    assert admin_pin.pin_is_set() is True

    app = FastAPI()
    app.state.ctx = AppContext(user_root=str(tmp_path))
    app.include_router(router, prefix="/api")

    # resolver 读的是**进程级**那个 store;不发布的话它会去读
    # project_root()/config —— 又是同一个污染面。
    #
    # 发布的必须是 **ctx 自己那一个实例**,不是「指向同一个目录的第二个 store」:
    # 第一版这么写过,于是写盘成功而 resolver 的判定纹丝不动 —— 它手里那份
    # `_data` 是写入之前的快照。生产里没有这个问题(`runtime.py` 把
    # `self._settings` 同时发布并交给 API,注释里逐字写着「而不是第二个持有陈旧
    # 快照的实例」),所以那是**测试的错**,不是产品的。
    store = app.state.ctx.settings_store
    ss.set_process_store(store)
    try:
        yield TestClient(app), store
    finally:
        ss.reset_process_store()


def _refusals(depth_m: float) -> list[str]:
    """在当前(进程级)覆写下,扎这么深会不会被拒。"""
    from mast.core.tip_conditioning_resolver import resolve_conditioning

    res = resolve_conditioning(
        ("poke_deep_depth_m",),
        explicit={"poke_deep_depth_m": -abs(depth_m)},
        facts=_QPLUS,
    )
    return list(res.refusals)


# ══════════════════════════════════════════════════════════════════════
# 端到端:写进去 → resolver 真的用到了它
# ══════════════════════════════════════════════════════════════════════

def test_the_override_reaches_the_decision_that_refuses_a_poke(rig):
    """判据是**一次拒绝判定被它改变了**,不是「存盘成功」。

    3 nm 的下压在出厂 qPlus 包络(5 nm)下是允许的;把包络覆写成 2 nm 之后,
    同一个 3 nm 必须被拒 —— 而且拒绝原文里出现的是**覆写后的**那个上限。
    """
    client, _store = rig

    assert _refusals(3e-9) == [], "前提变了:3 nm 在出厂包络下本来就被拒,这条测不出东西"

    r = client.post("/api/settings", json={
        "admin_pin": _PIN,
        "tip_conditioning_overrides": {"max_poke_depth_m": 2.0e-9},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    assert body["persisted"].get("tip_conditioning_overrides") == {
        "max_poke_depth_m": 2.0e-9}

    after = _refusals(3e-9)
    assert after, "覆写存进去了,但 resolver 的判定一点没变 —— 它没到达做决定的那一层"
    assert any("2.000e-09" in x for x in after), after


def test_a_write_without_the_pin_is_refused(rig):
    """它能**放宽硬件包络**(max_abs_pulse_v / max_poke_depth_m 就在这张表里),
    所以进了 GUARDED_KEYS。误改一次就直接放大物理动作的上限。"""
    from mast.api.admin_pin import GUARDED_KEYS

    assert "tip_conditioning_overrides" in GUARDED_KEYS
    client, store = rig
    r = client.post("/api/settings", json={
        "tip_conditioning_overrides": {"max_abs_pulse_v": 10.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["pin_required"] is True, body
    assert store.get("tip_conditioning_overrides") in (None, {}), "没过 PIN 却写进去了"


def test_the_three_whitelists_all_know_about_it():
    """三张白名单必须同时点头 —— 缺任何一张,写入都是**静默** no-op。

    2026-08-10 缺的是写 schema:键在 KNOWN_KEYS 里、resolver 每次都读它、
    两处注释都说它优先级最高,而没有任何请求体装得下它。
    """
    from mast.api.schemas import SettingsResponse
    from mast.api.schemas_settings_admin_write import SettingsWriteRequest
    from mast.webui.settings_store import KNOWN_KEYS

    key = "tip_conditioning_overrides"
    assert key in KNOWN_KEYS, "存储层白名单"
    assert key in SettingsWriteRequest.model_fields, "写 schema —— 2026-08-10 缺的是这张"
    assert key in SettingsResponse.model_fields, "读 schema(设置界面回显不出来)"


def test_the_resolver_still_reads_the_key_this_test_writes():
    """标识符从真源取:两边各查一次同一个名字。

    键名写错的话,上面每一条都会「通过」—— 写进一个没人读的键,再断言它没生效,
    而没生效正是我们要证伪的东西。**用自己编的名字查存在性永远验证不了存在性。**
    """
    from pathlib import Path

    import mast.core.tip_conditioning_resolver as resolver_mod

    src = Path(resolver_mod.__file__).read_text(encoding="utf-8")
    assert '"tip_conditioning_overrides"' in src, (
        "resolver 不再读这个键了 —— 这个文件测的是一条已经不存在的链")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
