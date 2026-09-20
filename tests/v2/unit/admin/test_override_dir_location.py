"""管理员覆写目录不该住在安装目录里（KNOWN_ISSUES §3.2）。

原来的解析是 ``Path(__file__).resolve().parents[3] / "config" / "overrides"``。
冻结版里 ``__file__`` 是 ``C:\\MAST\\_internal\\mast\\admin\\override_store.py``，
于是它落在 **``C:\\MAST\\config\\overrides\\`` —— 安装目录内部**，而升级正是往那里
拷文件。它靠的是「安装包里恰好没有同名文件」活着，而这条已经不成立过一次
（2026-08-03 冒烟在打包产物里留下了一份测试用 ±2.52 µm 的 safety_limits.json）。

两件事必须同时成立，缺一条就是把一个安全 bug 换成另一个：

1. 目标目录跟着**用户数据根**（``MAST2_PROJECT_ROOT`` / data_dir.txt）走；
2. 已经装好的机器上，旧位置那份**实测**限值必须被带过来 —— 否则一台真机会在升级
   后安静地退回 ±1.5 µm 出厂占位值，那是一次**静默放宽**。
"""

from __future__ import annotations

import json

import pytest

from mast.admin import override_store as store
from mast.admin.override_store import SAFETY_LIMITS, ConfigOverrideRegistry


@pytest.fixture(autouse=True)
def _clean_singleton():
    ConfigOverrideRegistry.reset()
    yield
    ConfigOverrideRegistry.reset()


def test_default_dir_follows_the_user_data_root(tmp_path, monkeypatch):
    """启动器把 data_dir.txt 导出成 MAST2_PROJECT_ROOT —— 覆写要跟着它。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    assert store._default_dir() == tmp_path / "config" / "overrides"


def test_default_dir_is_resolved_per_call_not_at_import(tmp_path, monkeypatch):
    """模块级常量会把 import 那一刻的值冻住 —— 本仓栽过好几次的形状。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "a"))
    first = store._default_dir()
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "b"))
    assert store._default_dir() != first


def test_legacy_overrides_are_migrated_not_silently_dropped(tmp_path, monkeypatch):
    """真机场景：C:\\MAST\\config\\overrides 里有实测包络，新位置是空的。"""
    legacy = tmp_path / "install" / "config" / "overrides"
    legacy.mkdir(parents=True)
    (legacy / SAFETY_LIMITS).write_text(
        json.dumps({"xy_max_m": 2.52e-6, "xy_min_m": -2.52e-6}), encoding="utf-8"
    )
    data_root = tmp_path / "data"
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(data_root))
    monkeypatch.setattr(store, "_LEGACY_DIR", legacy)

    reg = ConfigOverrideRegistry.get()

    assert reg.get_safety_limits()["xy_max_m"] == 2.52e-6
    assert (data_root / "config" / "overrides" / SAFETY_LIMITS).exists()
    # COPY, not move: a rollback to the previous build must still find its config.
    assert (legacy / SAFETY_LIMITS).exists()


def test_migration_never_overwrites_an_existing_target_file(tmp_path, monkeypatch):
    legacy = tmp_path / "install" / "config" / "overrides"
    legacy.mkdir(parents=True)
    (legacy / SAFETY_LIMITS).write_text('{"bias_max_v": 1.0}', encoding="utf-8")
    data_root = tmp_path / "data"
    target = data_root / "config" / "overrides"
    target.mkdir(parents=True)
    (target / SAFETY_LIMITS).write_text('{"bias_max_v": 7.0}', encoding="utf-8")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(data_root))
    monkeypatch.setattr(store, "_LEGACY_DIR", legacy)

    reg = ConfigOverrideRegistry.get()

    assert reg.get_safety_limits()["bias_max_v"] == 7.0


def test_migration_only_touches_known_files(tmp_path, monkeypatch):
    legacy = tmp_path / "install" / "config" / "overrides"
    (legacy / "_history").mkdir(parents=True)
    (legacy / SAFETY_LIMITS).write_text("{}", encoding="utf-8")
    (legacy / "stray.json").write_text("{}", encoding="utf-8")
    (legacy / "_history" / "old.json").write_text("{}", encoding="utf-8")
    data_root = tmp_path / "data"
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(data_root))
    monkeypatch.setattr(store, "_LEGACY_DIR", legacy)

    ConfigOverrideRegistry.get()

    target = data_root / "config" / "overrides"
    assert (target / SAFETY_LIMITS).exists()
    assert not (target / "stray.json").exists()
    assert not (target / "_history" / "old.json").exists()


def test_explicit_dir_is_never_migrated_into(tmp_path, monkeypatch):
    """测试/工具传显式目录时必须拿到一个干净的、只含它自己内容的 registry。"""
    legacy = tmp_path / "install" / "config" / "overrides"
    legacy.mkdir(parents=True)
    (legacy / SAFETY_LIMITS).write_text('{"bias_max_v": 1.0}', encoding="utf-8")
    monkeypatch.setattr(store, "_LEGACY_DIR", legacy)

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path / "explicit")

    assert reg.get_safety_limits() == {}
    assert not (tmp_path / "explicit" / SAFETY_LIMITS).exists()


def test_dev_checkout_resolves_to_the_same_place_as_before(monkeypatch):
    """开发树里两种解析必须同一个目录 —— 这次搬家不该动开发者的配置。"""
    monkeypatch.delenv("MAST2_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MAST_PROJECT_ROOT", raising=False)
    assert store._default_dir() == store._LEGACY_DIR
