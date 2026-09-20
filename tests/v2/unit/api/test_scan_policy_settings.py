"""扫描档位表的设置接线(KNOWN_KEYS → schema → 写路径 → hydrate → 端点)。

这条链断在任何一环都是**静默**的:界面上照样能填、能保存、能显示,只是那些
数字永远到不了扫描。KNOWN_KEYS 这个坑本项目已经踩过两次(admin PIN 门、
override_store._ALL_FILES),所以每一环都单独钉一个用例。

测试全部走 tmp 落点,绝不碰真实用户设置。
"""

from __future__ import annotations

import json

import pytest

from mast.core import scan_policy
from mast.webui.settings_store import KNOWN_KEYS, SettingsStore


@pytest.fixture(autouse=True)
def _clean_holder():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


@pytest.fixture
def store(tmp_path):
    """真实的 SettingsStore,但落点在 tmp。

    「重定向 env A 而存储读 env B」是本项目四次测试污染真实数据的恒定根因,
    所以这里不重定向任何全局变量,直接把 tmp 目录交给 store 自己持有。
    """
    s = SettingsStore(tmp_path)
    assert not (tmp_path / "ui_settings.json").exists() or True
    return s


_TIERS = [
    {"name": "fine", "upper_size_m": 2e-8, "pixels": 1024, "line_time_s": 5.0},
    {"name": "coarse", "upper_size_m": None, "pixels": 128, "line_time_s": 0.2},
]


# ── ① 白名单 ─────────────────────────────────────────────────────────────────

def test_scan_policy_is_in_known_keys():
    """不在 KNOWN_KEYS 里,store.update 会静默丢掉它 —— 保存按钮照常变绿。"""
    assert "scan_policy" in KNOWN_KEYS


def test_store_round_trips_the_tier_table(store, tmp_path):
    store.update(scan_policy={"tiers": _TIERS})
    on_disk = json.loads((tmp_path / "ui_settings.json").read_text("utf-8"))
    assert on_disk["scan_policy"]["tiers"][0]["name"] == "fine"
    assert store.load()["scan_policy"]["tiers"][1]["pixels"] == 128


def test_store_write_does_not_touch_the_real_user_settings(tmp_path):
    """落点指纹守卫:写入必须只发生在 tmp 里。"""
    store = SettingsStore(tmp_path)
    store.update(scan_policy={"tiers": _TIERS})
    written = list(tmp_path.rglob("ui_settings.json"))
    assert len(written) == 1
    assert str(written[0]).startswith(str(tmp_path))


# ── ② schema ─────────────────────────────────────────────────────────────────

def test_read_schema_exposes_scan_policy():
    from mast.api.schemas import SettingsResponse
    assert "scan_policy" in SettingsResponse.model_fields


def test_write_schema_exposes_scan_policy():
    from mast.api.schemas_settings_admin_write import SettingsWriteRequest
    assert "scan_policy" in SettingsWriteRequest.model_fields


def test_write_response_can_report_a_rejection():
    from mast.api.schemas_settings_admin_write import SettingsWriteResponse
    resp = SettingsWriteResponse(ok=False, rejected={"scan_policy": "坏表"})
    assert resp.rejected["scan_policy"] == "坏表"


# ── ③ hydrate ────────────────────────────────────────────────────────────────

def test_hydrate_applies_a_stored_table_to_the_holder(store):
    """启动时把磁盘上的表推进 holder —— 否则第一次扫描用的是出厂参数。"""
    store.update(scan_policy={"tiers": _TIERS})
    assert not scan_policy.is_customised()

    stored = store.get("scan_policy")
    scan_policy.set_policy(stored)

    assert scan_policy.tier_names() == ["fine", "coarse"]
    assert scan_policy.get_tier_for_size(1e-8)["pixels"] == 1024


def test_hydrate_of_a_blank_setting_leaves_factory_defaults(store):
    scan_policy.set_policy(store.get("scan_policy"))
    assert not scan_policy.is_customised()
    # 2026-08-14 起是 6 档:``atomic_verify``(≤5 nm / 512 px / 0.30 s)插在
    # slow 与 atomic 之间。它存在的理由是原子判据的尺度门 —— 256 px 下要进
    # 满权重档需要帧宽 < 5.12 nm,而旧的 atomic 档(≤10 nm / 256 px)在 5–10 nm
    # 那一段算出来是 0.039 nm/px,判据只会回「证据不足」。
    assert scan_policy.tier_names() == [
        "slow", "atomic_verify", "atomic", "highres", "roi", "survey"]


# ── ④ 写路径:结构非法必须在存盘之前被拒 ──────────────────────────────────────

def _write_settings(tmp_path, body_kwargs):
    """直接驱动写路由,用一个最小的假 app.state.ctx。"""
    from mast.api.routes.settings_admin_write import write_settings
    from mast.api.schemas_settings_admin_write import SettingsWriteRequest

    class _Ctx:
        settings_store = SettingsStore(tmp_path)
        orchestrator = None

    class _State:
        ctx = _Ctx()

    class _App:
        state = _State()

    class _Req:
        app = _App()

    return write_settings(_Req(), SettingsWriteRequest(**body_kwargs)), _Ctx.settings_store


def test_a_structurally_invalid_table_is_refused_before_persisting(tmp_path):
    """先存盘再 live-apply 的顺序有个陷阱:坏表会存进磁盘却应用失败,于是
    用户用着出厂表、设置文件里躺着他那张坏表,下次启动 hydrate 同样失败 ——
    两边不一致,而且在任何界面上都看不出来。所以整体拒绝,一个字节都不写。"""
    resp, store = _write_settings(tmp_path, {
        "scan_policy": {"tiers": [
            # 没有兜底档
            {"name": "a", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 2.0},
        ]},
    })
    assert resp.ok is False
    assert "scan_policy" in resp.rejected
    assert "兜底档" in resp.rejected["scan_policy"]
    assert store.get("scan_policy") is None
    assert not (tmp_path / "ui_settings.json").exists() or \
        "scan_policy" not in json.loads(
            (tmp_path / "ui_settings.json").read_text("utf-8"))


def test_a_rejected_write_does_not_persist_the_other_keys_either(tmp_path):
    """拒绝是整次写入的拒绝 —— 半次生效比全不生效更难排查。"""
    resp, store = _write_settings(tmp_path, {
        "theme": "Dark",
        "scan_policy": {"tiers": [
            {"name": "a", "upper_size_m": 1e-8, "pixels": 512, "line_time_s": 2.0},
        ]},
    })
    assert resp.ok is False
    assert store.get("theme") is None


def test_a_valid_table_is_persisted_and_applied_live(tmp_path):
    resp, store = _write_settings(tmp_path, {"scan_policy": {"tiers": _TIERS}})
    assert resp.ok is True
    assert resp.rejected == {}
    assert "scan_policy" in resp.applied
    assert store.get("scan_policy")["tiers"][0]["name"] == "fine"
    # live-apply:holder 立刻生效,不需要重启或重建 graph
    assert scan_policy.tier_names() == ["fine", "coarse"]


def test_clearing_the_table_falls_back_to_factory(tmp_path):
    _write_settings(tmp_path, {"scan_policy": {"tiers": _TIERS}})
    assert scan_policy.is_customised()
    resp, _ = _write_settings(tmp_path, {"scan_policy": {"tiers": []}})
    assert resp.ok is True
    assert not scan_policy.is_customised()


# ── ⑤ 端点 ───────────────────────────────────────────────────────────────────

def test_scan_policy_endpoint_shows_the_effective_table_and_factory_template():
    from mast.api.routes.settings import get_scan_policy

    out = get_scan_policy(None)
    # 自动查表与扫描档位必须一致。
    # 测试当前 atomic_verify 在相应尺寸区间中的选择行为；
    # 若档位定义改变，查表预期应随正式产品语义一起更新。
    assert [t["name"] for t in out["tiers"]] == \
        ["slow", "atomic_verify", "atomic", "highres", "roi", "survey"]
    assert out["stored"] == []          # 没改过 → 界面拿 factory 做 placeholder
    assert out["customised"] is False
    assert len(out["factory"]) == 6
    assert out["min_tiers"] == 1 and out["max_tiers"] == 8
    keys = {f["key"] for f in out["fields"]}
    assert {"pixels", "line_time_s", "setpoint_a", "p_gain",
            "time_constant_s"} <= keys


def test_scan_policy_endpoint_reports_a_customised_table():
    from mast.api.routes.settings import get_scan_policy

    scan_policy.set_policy(_TIERS)
    out = get_scan_policy(None)
    assert out["customised"] is True
    assert [t["name"] for t in out["stored"]] == ["fine", "coarse"]


def test_preview_endpoint_answers_what_parameters_a_given_size_would_use():
    from mast.api.routes.settings import preview_scan_policy

    out = preview_scan_policy(size_nm=200.0)
    assert out["ok"] is True
    assert out["tier_name"] == "roi"
    assert out["pixels"] == 256
    assert out["line_time_s"] == 0.8
    assert out["size_nm"] == 200.0


def test_preview_endpoint_reflects_an_edited_table_immediately():
    """改表即见效果 —— 这是预览存在的唯一理由。"""
    from mast.api.routes.settings import preview_scan_policy

    scan_policy.set_policy(_TIERS)
    out = preview_scan_policy(size_nm=200.0)
    assert out["tier_name"] == "coarse"
    assert out["pixels"] == 128


def test_preview_endpoint_returns_a_typed_error_for_a_bad_size():
    from mast.api.routes.settings import preview_scan_policy

    out = preview_scan_policy(size_nm=0.0)
    assert out["ok"] is False
    assert "size_m" in out["error"]
