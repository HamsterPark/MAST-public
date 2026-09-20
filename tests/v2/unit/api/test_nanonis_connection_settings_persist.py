"""「保存 Nanonis 连接配置」这个按钮,到底存下了什么。

背景(2026-08-10 实测):``nanonis_host`` 和四个端口**都在 KNOWN_KEYS 里**、
**都在读 schema 里**,唯独**不在 ``SettingsWriteRequest`` 上**。pydantic 默认
``extra='ignore'`` ⇒ ``model_dump(exclude_none=True)`` 是 ``{}`` ⇒
``store.update()`` 什么也没做 ⇒ 响应仍然是 ``ok=True, degraded=False`` ⇒
前端 toast **绿字「已保存 Nanonis 连接配置。」**。``ui_settings.json`` 连建都没建。

**为什么一直没被发现**:同一块面板上的「重新连接」走的是另一个端点
(``POST /api/nanonis/connect``)和另一套字段名(``host`` / ``port_main``),
那条路是通的 —— 所以当场试连能成功,只有「下次开机还记得吗」是假的,
而那一问要等到下次开机才问得出来。

所以验收不是「字段加上了」,是:
**POST 之后 GET 回来是新值,且重建 store(= 重启)之后还是新值。**

跑法::

    $env:PYTHONPATH='<repo>\\MASTv2'
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/api/test_nanonis_connection_settings_persist.py -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.settings import router as read_router
from mast.api.routes.settings_admin_write import router as write_router
from mast.api.schemas_settings_admin_write import SettingsWriteRequest
from mast.webui.settings_store import KNOWN_KEYS, SettingsStore, default_config_dir

#: 前端 ``HardwareManager.tsx`` 的保存按钮构造的那个 body(host + 四端口)。
_BODY = {
    "nanonis_host": "10.0.0.99",
    "nanonis_port_main": 7001,
    "nanonis_port_monitor": 7002,
    "nanonis_port_data": 7003,
    "nanonis_port_emergency": 7004,
}


def _client(tmp_path, *, config=None) -> TestClient:
    app = FastAPI()
    ctx = AppContext(user_root=str(tmp_path))
    if config is not None:
        ctx.config = config
    app.state.ctx = ctx
    app.include_router(read_router, prefix="/api")
    app.include_router(write_router, prefix="/api")
    return TestClient(app)


def _settings_file(tmp_path) -> Path:
    return default_config_dir(tmp_path) / "ui_settings.json"


# ── 端到端:POST → GET → 重启 ─────────────────────────────────────────────────
def test_post_then_get_then_restart(tmp_path):
    """三段都要过。第二段和第三段问的不是同一件事:

    * GET 回来是新值   —— 证明这一次写进了 store;
    * 重建 store 还在  —— 证明它**落了盘**。缺陷版本两段都答「旧值」,
      而修一半(只改内存不落盘)会在第二段绿、第三段红。
    """
    c = _client(tmp_path)
    r = c.post("/api/settings", json=_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["degraded"] is False
    assert body["rejected"] == {}
    persisted = body["persisted"]
    for k, v in _BODY.items():
        assert persisted.get(k) == v, f"{k} 没被 store 收下:{persisted.get(k)!r}"

    got = c.get("/api/settings").json()
    for k, v in _BODY.items():
        assert got.get(k) == v, f"GET /api/settings 回来的 {k} 还是旧值:{got.get(k)!r}"

    # 「重启」:一个全新的 SettingsStore 读同一个目录 —— 新进程看到的就是这个。
    reborn = SettingsStore(default_config_dir(tmp_path)).load()
    for k, v in _BODY.items():
        assert reborn.get(k) == v, f"重启之后 {k} 丢了:{reborn.get(k)!r}"


def test_the_file_gets_created_at_all(tmp_path):
    """缺陷版本连 ``ui_settings.json`` 都没建 —— 那是最直接的一条痕迹。"""
    f = _settings_file(tmp_path)
    assert not f.exists()
    _client(tmp_path).post("/api/settings", json=_BODY)
    assert f.exists(), "POST 之后设置文件都没建,这次写入是个彻底的 no-op"
    on_disk = json.loads(f.read_text(encoding="utf-8"))
    assert {k: on_disk.get(k) for k in _BODY} == _BODY


def test_live_config_follows_the_save(tmp_path):
    """存盘的同时也要改活着的 config。

    ``/api/nanonis/connect`` 改 live config 但**不存盘**;这条路径以前存盘但
    **不改 live config**。两边加起来,用户存了新主机、这次会话继续用旧的,
    中间任何一次自动重连都还在拨老地址。
    """
    class _Nano:
        host = "127.0.0.1"
        port_main = 6501
        port_monitor = 6502
        port_data = 6503
        port_emergency = 6504

    class _Cfg:
        nanonis = _Nano()
        llm = None

    cfg = _Cfg()
    r = _client(tmp_path, config=cfg).post("/api/settings", json=_BODY)
    assert r.status_code == 200 and r.json()["ok"] is True
    assert cfg.nanonis.host == "10.0.0.99"
    assert cfg.nanonis.port_main == 7001
    assert cfg.nanonis.port_emergency == 7004
    assert set(_BODY) <= set(r.json()["applied"])


# ── 拒绝:非法端口一个字节都不写 ──────────────────────────────────────────────
@pytest.mark.parametrize("bad", [0, -1, 70000, 65536])
def test_out_of_range_port_is_refused_and_nothing_is_written(tmp_path, bad):
    """越界端口的失败是**延迟**的:存下去不会当场报错,下次开机
    ``apply_to_config`` 把它 setattr 进 config.nanonis,然后连接失败,
    而错在哪只有翻设置文件才看得出来。所以在这里就拒,并且整笔不写。"""
    c = _client(tmp_path)
    c.post("/api/settings", json={"nanonis_port_main": 7001})   # 先有一个好值
    before = _settings_file(tmp_path).read_text(encoding="utf-8")

    r = c.post("/api/settings", json={"nanonis_host": "1.2.3.4",
                                      "nanonis_port_main": bad})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False
    assert "nanonis_port_main" in body["rejected"]
    assert body["persisted"] == {}
    # 同一笔里那个**合法**的 host 也不能溜进去 —— 「拒绝」是整笔的
    assert _settings_file(tmp_path).read_text(encoding="utf-8") == before
    assert c.get("/api/settings").json()["nanonis_host"] is None


def test_a_refusal_is_not_reported_as_ok(tmp_path):
    """``ok=False, degraded=False`` 是拒绝的形状。前端只看 ``degraded`` 的话,
    拒绝会显示成绿色的「已保存」—— 与修好之前那句谎话一模一样。
    这条钉后端的形状;前端那一半由 test_hardware_manager_reports_refusals 守。"""
    r = _client(tmp_path).post("/api/settings", json={"nanonis_port_data": 0}).json()
    assert (r["ok"], r["degraded"]) == (False, False)


# ── 前端送的键名 = 后端收的键名 ──────────────────────────────────────────────
def _hardware_manager_src() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        f = p / "frontend" / "src" / "components" / "admin" / "HardwareManager.tsx"
        if f.is_file():
            return f.read_text(encoding="utf-8")
        p = p.parent
    pytest.skip("frontend/ not present in this checkout")


def test_the_keys_the_form_sends_are_the_keys_the_schema_accepts():
    """两侧都从真源取名字,各查一次。

    自己编一份名单去查「这些字段在不在」是查不出问题的:名单里的笔误和「全部
    通过」长得一模一样。所以端口名从 TSX 的 ``PORT_ROLES`` 里正则抓,host 名从
    ``body.nanonis_host =`` 那一行抓,再拿去问 pydantic 模型和 KNOWN_KEYS。
    """
    src = _hardware_manager_src()
    sent = set(re.findall(r'setting:\s*"(\w+)"', src))
    sent |= set(re.findall(r"body\.(\w+)\s*=", src))
    sent = {k for k in sent if k.startswith("nanonis_")}
    assert sent, "在 HardwareManager.tsx 里没抓到任何 nanonis_* 键 —— 正则失效了"
    assert len(sent) == 5, f"抓到 {sorted(sent)},预期 host + 四个端口"

    missing_write = sorted(sent - set(SettingsWriteRequest.model_fields))
    assert not missing_write, (
        f"表单在 POST 这些键,而写 schema 上没有它们 —— pydantic 会静默丢掉,"
        f"响应仍然 ok=True:{missing_write}")
    missing_known = sorted(sent - set(KNOWN_KEYS))
    assert not missing_known, (
        f"这些键不在 KNOWN_KEYS 里,SettingsStore.update 会静默丢掉:{missing_known}")
