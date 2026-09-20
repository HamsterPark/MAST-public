"""``AssessFrameCorrugation`` 的技能外壳:阈值取用、三态、以及 **default 禁止**。

判据本体在 ``tests/v2/unit/vision/test_corrugation_gate.py``。这里测的是外壳:
真写一个 ``.sxm`` 到磁盘再让技能从头读一遍,重点是四件只有外壳管得到的事。

跑:
    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/skills/builtins/test_frame_corrugation.py -q

1. **``threshold_pm`` 不许有 default**,而且这一条**必须从生产入口验**
   (schema → coerce → execute)。直接搭一个 dict 调 ``execute`` 会绕过 pydantic
   填默认值那一层 —— 6.2.13 的 26 条单测就是这么全绿着放跑了一个真机故障。
2. **阈值与它的视野同源**:要么都不给(走 profile),要么一起给。只给一个 ⇒
   判不了,而**不会**拿 profile 的另一半来补。
3. **三态在 ``data.verdict`` 里,``success`` 恒为 True**(只要文件读得动)。
   把「判据没通过」表达成技能失败,会让 composite 的必做步骤直接中止整条流程。
4. 旁证(``frame_usable`` / ``nan_frac`` / ``bad_row_frac``)要在,而且不参与判决。
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from mast.core.types import SafetyLevel, SkillCategory
from mast.skills.builtins.frame_corrugation import (
    AssessFrameCorrugation,
    resolve_threshold_pair,
)
from mast.vision.scan_prep_thresholds import PROFILES, ScanPrepThresholds

PX = 128
FRAME_M = 100e-9                      # 复测视野钉死在 100 nm(设计 D2)
PROFILE = "test-corrugation"


def _write_sxm(path, *, fwd, bwd=None, scan_dir="down", bias=0.4,
               range_m=FRAME_M, channel="Z", setpoint="500.0000E-12"):
    """一个最小但真实的 Nanonis ``.sxm``(大端 float32)。"""
    ny, nx = fwd.shape
    direction = "both" if bwd is not None else "fwd"
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 14.08.2026\n"
        ":REC_TIME:\n12:00:00\n"
        f":BIAS:\n\t{bias:.6E}\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_RANGE:\n{range_m:>19.6E}{range_m:>19.6E}\n"
        f":SCAN_DIR:\n{scan_dir}\n"
        f":Z-CONTROLLER>SETPOINT:\n{setpoint}\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        f"\t14\t{channel}\tm\t{direction}\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    blob = b"".join(
        struct.pack(">%df" % a.size, *a.astype(np.float32).ravel().tolist())
        for a in ([fwd] if bwd is None else [fwd, bwd]))
    path.write_bytes(header.encode("utf-8") + b"\x1a\x04" + blob)
    return path


def _rough(n=PX, corrugation_m=100e-12, seed=0):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    h = (corrugation_m * np.sin(2 * np.pi * x / (n / 15.0))
         + 0.3 * corrugation_m * np.sin(2 * np.pi * y / (n / 11.0)))
    return 1e-9 + h + 0.05 * corrugation_m * np.random.default_rng(
        seed).standard_normal(h.shape)


def _frame_file(tmp_path, **kw):
    return _write_sxm(tmp_path / "frame.sxm", fwd=_rough(**kw))


@pytest.fixture
def calibrated_profile(monkeypatch):
    """一个「标定过」的样品 profile —— 阈值 40 pm @ 100 nm。"""
    th = ScanPrepThresholds.from_mapping(
        {"corrugation_high_pm": 40.0, "corrugation_ref_scan_nm": 100.0})
    th = replace(th, name=PROFILE, provenance="测试用,不是真标定")
    monkeypatch.setitem(PROFILES, PROFILE, th)
    return th


# ── 生产入口:schema → coerce → execute ──────────────────────────────────

def _through_the_tool_schema(**model_args) -> dict:
    """走模型真正会走的那条路。**填默认值那一层就在这里。**"""
    from mast.agents._shared.skill_adapter import (
        _coerce_si_params,
        _schema_from_metadata,
    )

    meta = AssessFrameCorrugation().metadata()
    kw = _schema_from_metadata(meta)(**model_args).model_dump()
    kw.pop("tool_call_id", None)
    out, errs = _coerce_si_params(meta, kw)
    assert not errs, f"SI 转换报错: {errs}"
    return out


def _run_from_production_entry(**model_args):
    return AssessFrameCorrugation().execute(
        None, _through_the_tool_schema(**model_args))


# ═══════════════════════════════════════════════════════════════════════
# 1. 元数据
# ═══════════════════════════════════════════════════════════════════════

def test_metadata_is_analysis_and_auto():
    meta = AssessFrameCorrugation().metadata()
    assert meta.category is SkillCategory.ANALYSIS
    assert meta.safety_level is SafetyLevel.AUTO
    assert meta.preconditions == [], "只读文件的技能不该要求任何硬件前置条件"


def test_threshold_has_no_default_in_the_spec():
    """有 default,「没传」就会变成「传了那个数」,查 profile 那行永远到不了。"""
    specs = {p.name: p for p in AssessFrameCorrugation().metadata().parameters}
    for name in ("threshold_pm", "ref_scan_nm"):
        assert specs[name].required is False
        assert specs[name].default is None, (
            f"{name} 有了 default —— 那会把「没传」物化成「传了」")


def test_the_pm_and_nm_params_are_plain_floats_not_si_text():
    """**不许**给它们写 ``unit=`` —— 那会让 schema 收字符串并按 SI 前缀解析。

    ``threshold_pm`` 已经以 pm 计。带上 ``unit="pm"`` 之后模型很自然会写
    ``"40p"``,而 ``parse_quantity`` 会把它读成 4e-11 —— 差 1e12,而且直接改判决。
    单位写在名字和描述里,值就是普通浮点数。
    """
    from mast.agents._shared.skill_adapter import _si_params

    meta = AssessFrameCorrugation().metadata()
    assert not (set(_si_params(meta)) & {"threshold_pm", "ref_scan_nm"})
    kw = _through_the_tool_schema(scan_path="x", threshold_pm=40.0,
                                  ref_scan_nm=100.0)
    assert kw["threshold_pm"] == 40.0 and kw["ref_scan_nm"] == 100.0


# ═══════════════════════════════════════════════════════════════════════
# 2. default 禁止:从生产入口验「查 profile 那一行真的走到了」
# ═══════════════════════════════════════════════════════════════════════

def _assert_omitting_the_threshold_reaches_the_profile(tmp_path):
    res = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE)
    assert res.success is True
    assert res.data["threshold_source"] == f"profile:{PROFILE}", (
        "没传 threshold_pm,却没走到查 profile 那一行 —— "
        "「没传」被谁变成了「传了」")
    assert res.data["threshold_pm"] == 40.0
    assert res.data["profile_name"] == PROFILE
    assert res.data["verdict"] == "high"


def test_omitting_the_threshold_reaches_the_profile(tmp_path, calibrated_profile):
    _assert_omitting_the_threshold_reaches_the_profile(tmp_path)


def test_mutation_giving_the_threshold_a_default_turns_that_test_red(
        tmp_path, calibrated_profile, monkeypatch):
    """给 ``threshold_pm`` 补一个 default ⇒ 上面那条必须当场红。"""
    orig = AssessFrameCorrugation.metadata

    def mutant(self):
        meta = orig(self)
        for spec in meta.parameters:
            if spec.name in ("threshold_pm", "ref_scan_nm"):
                spec.default = 4000.0 if spec.name == "threshold_pm" else 100.0
        return meta

    monkeypatch.setattr(AssessFrameCorrugation, "metadata", mutant)
    # 1) 变异已应用:模型没传,schema 却把它填上了
    assert _through_the_tool_schema(scan_path="x")["threshold_pm"] == 4000.0
    # 2) 测试红了
    with pytest.raises(AssertionError):
        _assert_omitting_the_threshold_reaches_the_profile(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# 3. 阈值与它的视野同源
# ═══════════════════════════════════════════════════════════════════════

def test_explicit_pair_wins_over_the_profile(tmp_path, calibrated_profile):
    res = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE,
        threshold_pm=400.0, ref_scan_nm=100.0)
    assert res.data["threshold_source"] == "explicit"
    assert res.data["threshold_pm"] == 400.0
    assert res.data["verdict"] == "normal"


def test_half_a_pair_is_undecidable_and_never_borrows_from_the_profile(
        tmp_path, calibrated_profile):
    """显式阈值 + profile 的视野 = 两次不同标定拼出来的判据。拒绝。"""
    res = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE,
        threshold_pm=400.0)
    assert res.data["verdict"] == "undecidable"
    assert res.data["ref_scan_nm"] is None, "profile 的视野被借来补了显式阈值"
    assert "哪个视野" in res.data["reason"]


def test_resolve_threshold_pair_is_symmetric(calibrated_profile):
    th = PROFILES[PROFILE]
    assert resolve_threshold_pair(None, None, th) == (40.0, 100.0, f"profile:{PROFILE}")
    assert resolve_threshold_pair(40.0, 100.0, th) == (40.0, 100.0, "explicit")
    assert resolve_threshold_pair(40.0, None, th) == (40.0, None, "explicit")
    assert resolve_threshold_pair(None, 100.0, th) == (None, 100.0, "explicit")


def test_an_uncalibrated_profile_is_undecidable_not_a_pass(tmp_path):
    """出厂 profile 没有这两个数 ⇒ 判不了。**不是**「没有上限所以正常」。

    ⚠️ 这一句要指向「去标定」,不是「去补一个视野声明」—— 判不了的几种成因
    各指不同的下一步,合成一句就等于把几条路合并成一条。
    """
    res = _run_from_production_entry(scan_path=str(_frame_file(tmp_path)))
    assert res.success is True
    assert res.data["verdict"] == "undecidable"
    assert "起伏门没有标定" in res.data["reason"]
    assert "不是「没有上限所以都算正常」" in res.data["reason"]


# ═══════════════════════════════════════════════════════════════════════
# 4. 三态 / 尺度门 / 旁证
# ═══════════════════════════════════════════════════════════════════════

def test_scale_mismatch_is_undecidable(tmp_path, calibrated_profile):
    """帧是 200 nm 而阈值标在 100 nm ⇒ 判不了,**不换算**。"""
    p = _write_sxm(tmp_path / "wide.sxm", fwd=_rough(), range_m=200e-9)
    res = _run_from_production_entry(scan_path=str(p), profile=PROFILE)
    assert res.data["verdict"] == "undecidable"
    assert res.data["this_scan_nm"] == pytest.approx(200.0)
    assert res.data["ref_scan_nm"] == 100.0
    assert "视野对不上" in res.data["reason"]


def test_a_dead_flat_frame_is_undecidable_but_the_skill_still_succeeds(
        tmp_path, calibrated_profile):
    p = _write_sxm(tmp_path / "dead.sxm", fwd=np.zeros((PX, PX)))
    res = _run_from_production_entry(scan_path=str(p), profile=PROFILE)
    assert res.success is True, "「判不了」不是「这件事没做成」"
    assert res.data["verdict"] == "undecidable"
    assert res.data["frame_usable"] is False
    assert res.data["unusable_reason"]
    assert res.data["value_pm"] is None


def test_low_frame_abstains(tmp_path, calibrated_profile):
    p = _write_sxm(tmp_path / "flat.sxm", fwd=_rough(corrugation_m=5e-12))
    res = _run_from_production_entry(scan_path=str(p), profile=PROFILE)
    assert res.data["verdict"] == "low"


def test_side_evidence_is_present_and_does_not_decide(tmp_path, calibrated_profile):
    """半张图是**输入**,不是错误:未扫完的那一行不许把旁证打成「算不出来」。

    ``detect_scan_artifacts`` 碰上一个 NaN 行会直接抛(拟合返回 NaN,pydantic 拒收),
    所以这里先用全仓那一份 ``acquired_row_mask`` 裁掉没扫完的行。裁行会改这个数 ⇒
    ``bad_row_rows_used`` 必须跟着报出去(同名不同预处理 = 不同的量)。
    """
    fwd = _rough()
    fwd[3, :] = np.nan
    p = _write_sxm(tmp_path / "holes.sxm", fwd=fwd, bwd=np.fliplr(_rough(seed=1)))
    res = _run_from_production_entry(scan_path=str(p), profile=PROFILE)
    d = res.data
    assert d["frame_usable"] is True
    assert d["nan_frac"] == pytest.approx(1.0 / PX, rel=0.2)
    assert d["bad_row_frac"] is not None
    assert d["bad_row_rows_used"] == PX - 1
    assert d["rows"] == PX and d["cols"] == PX
    assert d["bias_v"] is not None and d["setpoint_a"] is not None
    assert d["verdict"] in ("high", "normal", "low", "undecidable")


def test_every_number_carries_its_pipeline_and_z_calibration(
        tmp_path, calibrated_profile):
    d = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE).data
    assert d["detrend"] == "ols" and d["statistic"] == "std"
    assert "标定" in d["z_cal_note"] and "目标仪器" in d["z_cal_note"]
    assert "行偏置" in d["blind_to"]
    assert d["provenance"], "阈值的出身要跟着进每一份报告"


def test_verdict_is_never_a_tip_conclusion(tmp_path, calibrated_profile):
    d = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE).data
    assert d["verdict"] != "bad_tip"
    assert "复测" in d["reason"]


def test_data_is_json_friendly(tmp_path, calibrated_profile):
    """``SkillResult.data`` 会进 ToolMessage —— 不许夹带 ndarray。"""
    d = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), profile=PROFILE).data
    json.dumps(d)
    assert not any(isinstance(v, np.ndarray) for v in d.values())


# ═══════════════════════════════════════════════════════════════════════
# 5. 「这件事没做成」才是技能失败
# ═══════════════════════════════════════════════════════════════════════

def test_missing_file_is_a_skill_failure(tmp_path):
    res = AssessFrameCorrugation().execute(
        None, {"scan_path": str(tmp_path / "nope.sxm")})
    assert res.success is False and "文件不存在" in res.error


def test_missing_channel_is_a_skill_failure(tmp_path):
    res = _run_from_production_entry(
        scan_path=str(_frame_file(tmp_path)), channel="Current")
    assert res.success is False and "通道" in res.error
