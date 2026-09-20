"""可空阈值字段:**越界丢弃,不夹紧**;``None`` 就是「判不了」。

验证可空标定字段的拒绝行为：未知或越界输入不能被伪装成已标定阈值。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/vision/test_scan_prep_nullable_thresholds.py -q

## 为什么这一族要跟既有字段不一样

``from_mapping`` 对既有数值字段是**夹紧**(``min(hi, max(lo, v))``)。夹紧对一个
「有出厂默认、总要有个值」的旋钮是可以接受的;对一个**判据阈值**不行:

* 一个越界的阈值被静默改成边界值,而调用方以为自己设的是原值;
* 这两个字段的出厂值是 ``None`` = **判不了**。把越界值夹进合法区间,等于凭空
  造出一个从没标定过的判据 —— 而本仓的立场是「造一个出来就是伪造标定」。

⚠️ **既有字段的夹紧行为没有跟着改**,那是另一个决定,需要单独论证。本文件同时
钉住这一点:两种行为并存是有意的,不是漏改。
"""

from __future__ import annotations

import logging

import pytest

from mast.vision.scan_prep_thresholds import (
    CALIBRATABLE_FROM_DISTRIBUTION,
    DEFAULT_PROFILE,
    FIELD_BOUNDS,
    KNOB_LABELS,
    PROFILES,
    ScanPrepThresholds,
    _NULLABLE_FIELDS,
    knob_catalog,
    resolve,
)

PAIR = ("corrugation_high_pm", "corrugation_ref_scan_nm")


# ═══════════════════════════════════════════════════════════════════════
# 1. 出厂就是「判不了」
# ═══════════════════════════════════════════════════════════════════════

def test_factory_default_is_none_which_means_undecidable():
    th = ScanPrepThresholds()
    for key in PAIR:
        assert getattr(th, key) is None, f"{key} 出厂不该有值 —— 那就是伪造标定"
    for key in PAIR:
        assert getattr(PROFILES[DEFAULT_PROFILE], key) is None


def test_the_pair_is_declared_as_one_group():
    """两个字段是一组 —— 它们必须同时进 :data:`_NULLABLE_FIELDS` 与边界表。"""
    assert set(PAIR) == set(_NULLABLE_FIELDS)
    for key in PAIR:
        assert key in FIELD_BOUNDS, "可空字段也要有拒绝线(只是拒绝而不是夹紧)"
        assert key in KNOB_LABELS, "没有标签,设置 UI 会显示裸键名"


# ═══════════════════════════════════════════════════════════════════════
# 2. 夹紧禁止
# ═══════════════════════════════════════════════════════════════════════

def _assert_out_of_range_is_dropped_not_clamped(caplog):
    hi = FIELD_BOUNDS["corrugation_high_pm"][1]
    with caplog.at_level(logging.WARNING, logger="mast.vision.scan_prep_thresholds"):
        th = ScanPrepThresholds.from_mapping({"corrugation_high_pm": 1e9})
    assert th.corrugation_high_pm is None, (
        "越界的阈值被夹进了合法区间 —— 调用方会以为自己设的是原值,"
        "而这道门其实用的是边界值")
    assert th.corrugation_high_pm != hi
    assert any("丢弃" in r.message or "丢弃" in r.getMessage()
               for r in caplog.records), "丢了一个值却一声不吭"


def test_out_of_range_is_dropped_not_clamped(caplog):
    _assert_out_of_range_is_dropped_not_clamped(caplog)


def test_below_the_floor_is_dropped_too(caplog):
    with caplog.at_level(logging.WARNING, logger="mast.vision.scan_prep_thresholds"):
        th = ScanPrepThresholds.from_mapping({"corrugation_ref_scan_nm": 0.0})
    assert th.corrugation_ref_scan_nm is None
    assert caplog.records


def test_nan_and_non_numbers_are_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="mast.vision.scan_prep_thresholds"):
        th = ScanPrepThresholds.from_mapping({
            "corrugation_high_pm": float("nan"),
            "corrugation_ref_scan_nm": "一百",
        })
    assert th.corrugation_high_pm is None and th.corrugation_ref_scan_nm is None
    assert len(caplog.records) >= 2


def test_a_legal_value_goes_through_unchanged():
    th = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 40.0, "corrugation_ref_scan_nm": 100.0})
    assert th.corrugation_high_pm == 40.0
    assert th.corrugation_ref_scan_nm == 100.0


def test_explicit_null_clears_an_inherited_value():
    """显式 ``null`` = 「这个 profile 不声明这个口径」= 判不了(保守的一侧)。"""
    base = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 40.0, "corrugation_ref_scan_nm": 100.0})
    cleared = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": None}, base=base)
    assert cleared.corrugation_high_pm is None
    assert cleared.corrugation_ref_scan_nm == 100.0, "没提到的那半不该被动"


def test_existing_fields_still_clamp():
    """既有字段的夹紧**没有跟着改** —— 两种行为并存是有意的,不是漏改。"""
    lo, hi = FIELD_BOUNDS["step_sep"]
    assert ScanPrepThresholds.from_mapping({"step_sep": 1e9}).step_sep == hi
    assert ScanPrepThresholds.from_mapping({"step_sep": -5.0}).step_sep == lo


# ═══════════════════════════════════════════════════════════════════════
# 3. 下游拿到的是「没有这个键」,不是 0
# ═══════════════════════════════════════════════════════════════════════

def test_numeric_mapping_omits_unset_nullables_rather_than_reporting_zero():
    m = ScanPrepThresholds().numeric_mapping()
    for key in PAIR:
        assert key not in m, "「没标定」被报成了一个数 —— 那是看不出来的兜底"
    # 下游就是这么用它的(`skills/builtins/scan_prep.py` 的阈值清单)
    "; ".join(f"`{k}`={v:g}" for k, v in sorted(m.items()))


def test_numeric_mapping_includes_them_once_set():
    m = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 40.0}).numeric_mapping()
    assert m["corrugation_high_pm"] == 40.0


def test_knob_catalog_carries_them_as_nullable():
    cat = {k["key"]: k for k in knob_catalog()}
    for key in PAIR:
        entry = cat[key]
        assert entry["nullable"] is True
        assert entry["default"] is None and entry["value"] is None, (
            "未标定的阈值在 UI 上显示成 0 会让人以为这道门开着(它其实是关的)")
        assert entry["label_zh"] and entry["hint_zh"]
        assert entry["max"] > entry["min"] > 0
    assert cat["step_sep"]["nullable"] is False


def test_knob_catalog_reflects_a_set_value():
    th = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 40.0}, base=resolve(DEFAULT_PROFILE))
    cat = {k["key"]: k for k in knob_catalog(th)}
    assert cat["corrugation_high_pm"]["value"] == 40.0
    assert cat["corrugation_ref_scan_nm"]["value"] is None


def test_only_the_threshold_is_calibratable_from_a_distribution():
    """阈值能从分布标;**口径声明不能** —— 那是关于模型选择的陈述。"""
    assert "corrugation_high_pm" in CALIBRATABLE_FROM_DISTRIBUTION
    assert "corrugation_ref_scan_nm" not in CALIBRATABLE_FROM_DISTRIBUTION


# ═══════════════════════════════════════════════════════════════════════
# 4. 变异验证
# ═══════════════════════════════════════════════════════════════════════

def test_mutation_removing_the_nullable_branch_turns_the_clamp_test_red(
        monkeypatch, caplog):
    """把可空分支拿掉 ⇒ 1e9 会被夹到上界。夹紧禁止那条必须当场红。"""
    import mast.vision.scan_prep_thresholds as spt

    monkeypatch.setattr(spt, "_NULLABLE_FIELDS", frozenset())
    # 1) 变异已应用:那个越界值现在真的被夹住了
    hi = FIELD_BOUNDS["corrugation_high_pm"][1]
    assert spt.ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 1e9}).corrugation_high_pm == hi
    # 2) 测试红了
    with pytest.raises(AssertionError):
        _assert_out_of_range_is_dropped_not_clamped(caplog)
