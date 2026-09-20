"""设置键应通过 HTTP 写入、读回，并在重新打开存储后仍然存在。

字段名单只能证明声明存在，无法证明校验、授权、live-apply 与落盘链路可达。
测试从声明与已知键的交集派生用例，要求每个可写键都有样本，且未知键不能假成功。"""
from __future__ import annotations

import hashlib
import types
import typing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.admin_pin import GUARDED_KEYS
from mast.api.context import AppContext
from mast.api.routes.settings import router as read_router
from mast.api.routes.settings_admin_write import router as write_router
from mast.api.schemas import SettingsResponse
from mast.api.schemas_settings_admin_write import SettingsWriteRequest
from mast.webui.settings_store import KNOWN_KEYS, SettingsStore, default_config_dir

_PIN = "9182"


# ── 名单：从真源派生 ─────────────────────────────────────────────────────────
def _writable_keys() -> set[str]:
    """存得进 store **且** 收得下的键 —— 也就是「声称可写」的那一批。

    交集而不是任一侧：``admin_pin`` 在写 schema 上但刻意不在 ``KNOWN_KEYS`` 里
    （它是一次性凭据，进了白名单会被明文写进 ui_settings.json）。
    """
    return set(SettingsWriteRequest.model_fields) & set(KNOWN_KEYS)


def _refused_keys() -> set[str]:
    """store 认、而 API 不收的键 —— 每一个都必须**明确拒绝**，不能报成功。

    这正是缺陷那一天的形状：这批键当时是 18 个，一次 POST 得到 ok:true。
    """
    return set(KNOWN_KEYS) - set(SettingsWriteRequest.model_fields)


# ── 样本值：只有推不出来的才手写 ─────────────────────────────────────────────
#
# 手写的原因分两类，都写在旁边：
#   * 路由/核心对取值有结构要求（档位表、参数组、copy|hardlink）；
#   * 那个位置的字符串是**枚举**，随手编一个词会在将来某个校验加上时红得莫名其妙。
_SAMPLES: dict[str, object] = {
    # 结构化
    # 恰好一个兜底档（``upper_size_m`` 留空）—— 少了或多了都会被 sanitize 拒。
    "scan_policy": {"tiers": [
        {"name": "细", "upper_size_m": 2e-8, "pixels": 256, "line_time_s": 0.5,
         "setpoint_a": None, "p_gain": None, "time_constant_s": None},
        {"name": "兜底", "upper_size_m": None, "pixels": 128, "line_time_s": 1.0,
         "setpoint_a": None, "p_gain": None, "time_constant_s": None},
    ]},
    "zctrl_presets": [{"name": "探针样本组", "p_gain": "3p", "i_gain": "180n"}],
    "ingest_copy_mode": "hardlink",     # 只有 copy | hardlink 两个合法值
    # 枚举式字符串
    "thinking": "off",
    "voice_mode": "wake",
    "autonomy_mode": "semi",
    "font_scale": "大",
    "theme": "Light",
    # 端口必须落在 1..65535
    "nanonis_port_main": 7101,
    "nanonis_port_monitor": 7102,
    "nanonis_port_data": 7103,
    "nanonis_port_emergency": 7104,
    # 「关掉」这一侧才是有意义的样本：0 会被 core/runtime.py:_pos 换成出厂默认，
    # 所以路由拒 0；这里用一个明显不是默认值的数。
    "chat_model_calls_per_run": 250,
    "chat_tool_calls_per_run": 500,
}


def _origin_types(ann: object) -> list[object]:
    """把 ``Optional[X]`` / ``X | None`` 摊平成 ``[X]``（丢掉 NoneType）。"""
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        return [a for a in typing.get_args(ann) if a is not type(None)]
    return [ann]


def _derive(key: str, ann: object) -> object:
    """从字段标注派生一个「明显不是默认值」的样本。

    bool 一律取 ``False``：这一组的出厂默认全是「开」，取 False 才能同时抓到
    「写进去了」和「有没有被谁又刷回默认」。
    """
    for base in _origin_types(ann):
        origin = typing.get_origin(base) or base
        if origin is bool:
            return False
        if origin is int:
            return 4242
        if origin is float:
            return 42.5
        if origin is str:
            return f"probe-{key}"
        if origin is dict:
            args = typing.get_args(base)
            val: object = "probe"
            if args:
                tail = args[-1]
                val = {float: 1.25, bool: True, int: 7}.get(tail, "probe")  # type: ignore[arg-type]
            return {"probe_key": val}
    raise AssertionError(
        f"{key}: 标注 {ann!r} 推不出样本值 —— 请在 _SAMPLES 里补一个。"
        f"（这里刻意不 skip：一个静静跳过的键和一个通过的键长得一模一样。）")


def sample_for(key: str) -> object:
    if key in _SAMPLES:
        return _SAMPLES[key]
    return _derive(key, SettingsWriteRequest.model_fields[key].annotation)


# ── 台架 ─────────────────────────────────────────────────────────────────────
@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """一个隔离的 config 目录 + 一个设好 PIN 的 admin_pin.txt。

    ``MAST2_PROJECT_ROOT`` 和 ``user_root`` 指同一处：PIN 走 ``project_root()``，
    设置文件走 ``user_root`` —— 分开指会得到一个「PIN 查得到、设置写别处」的台架，
    而那种台架的绿色什么也不证明。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    cfg = default_config_dir(tmp_path)
    (cfg / "admin_pin.txt").write_text(
        hashlib.sha256(_PIN.encode("utf-8")).hexdigest(), encoding="utf-8")

    app = FastAPI()
    app.state.ctx = AppContext(user_root=str(tmp_path))
    app.include_router(read_router, prefix="/api")
    app.include_router(write_router, prefix="/api")
    return TestClient(app), Path(tmp_path)


def _body_for(key: str) -> dict:
    body: dict = {key: sample_for(key)}
    if key in GUARDED_KEYS:
        body["admin_pin"] = _PIN
    return body


# ── 闸门自检 ─────────────────────────────────────────────────────────────────
def test_the_gate_is_looking_at_something():
    """一条匹配不到任何东西的闸门会一直绿，和「确实没问题」输出一模一样。"""
    writable = _writable_keys()
    assert len(writable) >= 40, f"只找到 {len(writable)} 个可写键 —— 真源多半取错了"
    # 2026-08-10 补进来的那一批必须真的在里面（点名，不只靠计数）。
    for k in ("tool_refine_min_chars", "chat_model_calls_per_run",
              "ingest_enabled", "conv_export_enabled"):
        assert k in writable, f"{k} 不在可写名单里 —— 这条闸门这次覆盖不到它"
    for k in writable:
        sample_for(k)      # 每个键都要能取到样本，缺一个就在这里红


def test_pin_rig_actually_gates(rig):
    """台架自检：PIN 真的在起作用。

    如果 ``verify_pin`` 因为路径没对上而恒真，下面所有 PIN 键的绿色都是假的 ——
    所以这里故意送错 PIN，必须被拒。
    """
    client, _ = rig
    key = sorted(GUARDED_KEYS & _writable_keys())[0]
    r = client.post("/api/settings", json={key: sample_for(key), "admin_pin": "0000"})
    assert r.status_code == 200
    assert r.json()["pin_required"] is True, "PIN 台架没生效，后面的绿色不算数"


# ── 主闸门：写进去 → 读回来 → 重启后还在 ─────────────────────────────────────
@pytest.mark.parametrize("key", sorted(_writable_keys()))
def test_a_claimed_writable_key_actually_lands(rig, key):
    """三段问的不是同一件事：

    * ``ok`` / ``persisted``  —— 这一次写有没有被 store 收下；
    * ``GET`` 回来是新值      —— 界面重挂之后看到的是不是它（缺陷版本这里答 None）；
    * 重建 store 还在         —— 它**落了盘**。只改内存不落盘会在第二段绿、第三段红。
    """
    client, root = rig
    want = sample_for(key)

    r = client.post("/api/settings", json=_body_for(key))
    assert r.status_code == 200, f"{key}: {r.status_code} {r.text[:300]}"
    body = r.json()
    assert body["rejected"] == {}, f"{key} 被拒：{body['rejected']}"
    assert body["pin_required"] is False, f"{key}: PIN 未通过"
    assert body["ok"] is True and body["degraded"] is False, f"{key}: {body}"
    assert body["persisted"].get(key) == want, (
        f"{key} 没被 store 收下 —— 响应说成功，persisted 里却是 "
        f"{body['persisted'].get(key)!r}")

    if key in SettingsResponse.model_fields:
        got = client.get("/api/settings").json()
        assert got.get(key) == want, (
            f"GET /api/settings 回来的 {key} 是 {got.get(key)!r}，不是刚写的 {want!r}")

    reborn = SettingsStore(default_config_dir(root)).load()
    assert reborn.get(key) == want, (
        f"重启之后 {key} 丢了：{reborn.get(key)!r} —— 这次写没落盘")


# ── 反方向：写不进的键必须**说**它写不进 ─────────────────────────────────────
def test_an_unknown_key_is_not_reported_as_saved(rig):
    """未知设置键必须明确拒绝，不能静默忽略输入并返回已保存。"""
    client, root = rig
    r = client.post("/api/settings", json={"tool_refine_min_chars_typo": 1234})
    assert r.status_code != 200, (
        f"送了一个不存在的键却拿到 {r.status_code} —— 响应体：{r.text[:300]}")
    assert not (default_config_dir(root) / "ui_settings.json").exists(), \
        "什么都不该被写"


def test_the_measured_defect_itself(rig):
    """设置写入后应读回相同值，直接验证配置字段的往返持久化。"""
    client, _ = rig
    r = client.post("/api/settings", json={"tool_refine_min_chars": 1234})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/api/settings").json()["tool_refine_min_chars"] == 1234


# ── 会被内核悄悄换成另一个数的取值：在这里拒 ─────────────────────────────────
@pytest.mark.parametrize("key,bad", [
    ("chat_model_calls_per_run", 0),
    ("chat_model_calls_per_run", -5),
    ("chat_tool_calls_per_run", 0),
    ("tool_refine_min_chars", 0),
    ("chat_model_calls_per_thread", -1),
    ("ingest_copy_mode", "hardlnk"),      # 打错一个字母
])
def test_a_value_the_core_would_silently_replace_is_refused(rig, key, bad):
    """这一组的坏值**存得下去，而且不会当场报错** —— 然后被换掉：

    * ``core/runtime.py:_pos()`` 对 per_run 的 ``v <= 0`` 返回**出厂默认**，
      填 0 想「关掉上限」，拿到的是 30 —— 正是砍断长任务的那个数；
    * ``_ingest_copy_mode()`` 对不认识的字符串返回 ``"copy"``，hardlink
      静静地没生效。

    两种情况下 ``GET`` 回来的都是用户填的那个值，而生效的是另一个 ——
    在任何界面上都看不出来。所以在写入口拒，并且整笔不写。
    """
    client, root = rig
    r = client.post("/api/settings", json={key: bad})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["degraded"] is False
    assert key in body["rejected"], f"{key}={bad!r} 没被拒：{body}"
    assert body["persisted"] == {}
    assert not (default_config_dir(root) / "ui_settings.json").exists(), \
        "被拒的写不该留下任何字节"


def test_a_refusal_does_not_take_the_good_keys_with_it_silently(rig):
    """拒绝是**整笔**的 —— 同一个 body 里那个合法的键也不写。

    半写半不写才是最难查的：GET 回来一半新一半旧，而 toast 只说了一句话。
    """
    client, root = rig
    r = client.post("/api/settings", json={"theme": "Light",
                                           "chat_tool_calls_per_run": 0})
    assert r.json()["ok"] is False
    assert SettingsStore(default_config_dir(root)).load() == {}


def test_the_keys_that_only_take_effect_after_a_restart_say_so(rig):
    """存下来了 ≠ 已经在用了。

    ``chat_model_calls_per_run`` 在建图时读一次就冻住，这一轮对话仍然用老上限。
    不说这句话，用户改完看到绿色的「已保存」，然后照样死在 30 上 —— 那和
    这次修掉的那句谎话是同一种。「没存进去」和「存了还没生效」必须是两句话。
    """
    client, _ = rig
    note = client.post("/api/settings",
                       json={"chat_model_calls_per_run": 250}).json()["rebuild_note"]
    assert "生效" in note, f"没有说什么时候生效：{note!r}"
    # 反方向：live-read 的键不该被贴上「要重启」——那会让人白重启一次。
    quiet = client.post("/api/settings", json={"theme": "Light"}).json()["rebuild_note"]
    assert quiet == "", f"live-read 的键不该说要重启：{quiet!r}"


@pytest.mark.parametrize("key", sorted(_refused_keys()) or ["<空>"])
def test_a_key_without_a_write_entry_is_refused_not_swallowed(rig, key):
    """已知但没有写入入口的键必须明确拒绝。
    测试只要求结果如实反映未写入，不限定使用哪个 HTTP 状态码。"""
    if key == "<空>":
        pytest.skip("当前没有「存得进写不进」的键")
    client, root = rig
    r = client.post("/api/settings", json={key: "任意值"})
    if r.status_code == 200:
        body = r.json()
        assert body.get("ok") is not True, (
            f"{key} 写不进去，响应却是 ok:true —— 这就是那句谎话")
        assert body.get("rejected"), f"{key} 被拒了但没说原因"
    reborn = SettingsStore(default_config_dir(root)).load()
    assert key not in reborn, f"{key} 居然写进去了 —— 它不该有写入口"
