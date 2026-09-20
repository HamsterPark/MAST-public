"""扫描预处理技能和 SXM 取向辅助的合成集成测试。

算法判据由 vision/test_scan_prep.py 覆盖。本文件生成最小 SXM 文件并验证读取、
处理和报告；autouse fixture 将所有图像产物隔离到临时目录，并核验重定向生效。
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
from mast.skills.builtins.scan_prep import AnalyzeScanImage, AutoProcessScanBatch
from mast.core.types import SafetyLevel, SkillCategory

PX = 64


@pytest.fixture(autouse=True)
def _isolate_figures(tmp_path, monkeypatch):
    d = tmp_path / "figures"
    d.mkdir()
    monkeypatch.setenv("MAST_FIGURES_DIR", str(d))
    from mast.agents._shared.data_paths import figures_dir

    assert figures_dir() == d, "重定向没生效 —— 技能会往真实图池里写东西"
    return d


def _write_sxm(path, *, fwd, bwd=None, scan_dir="down", bias=2.0,
               range_m=1e-8, channel="Z"):
    """写一个最小但真实的 Nanonis ``.sxm``(大端 float32,头部按真文件的格式)。"""
    ny, nx = fwd.shape
    direction = "both" if bwd is not None else "fwd"
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 01.01.2099\n"
        ":REC_TIME:\n12:00:00\n"
        f":BIAS:\n\t{bias:.6E}\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_RANGE:\n{range_m:>19.6E}{range_m:>19.6E}\n"
        f":SCAN_DIR:\n{scan_dir}\n"
        ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n"
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


def _ramp(n=PX, seed=0, offset=0.0):
    """一帧可辨认的图:沿 x 的斜坡 + 噪声,这样镜像/翻转看得出来。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    rng = np.random.default_rng(seed)
    return 1e-11 * x / n + 4e-12 * y / n + offset + rng.normal(0, 1e-13, (n, n))


# ── .sxm 取向辅助 ───────────────────────────────────────────────────────────

def test_backward_frame_is_un_mirrored(tmp_path):
    """反扫沿 −x 采集,存下来是镜像的。不 ``fliplr`` 回来,任何正反扫比较都是拿一张
    图和它自己的镜像比,得到的数字没有意义。"""
    fwd = _ramp()
    bwd = np.fliplr(fwd)                       # 硬件写下去的就是镜像的那一份
    p = _write_sxm(tmp_path / "a.sxm", fwd=fwd, bwd=bwd)

    raw = read_sxm(str(p))
    assert np.allclose(raw["channels"]["Z"]["backward"], bwd, atol=1e-15), (
        "read_sxm 的行为被改了 —— 它应该原样返回硬件写下的块")

    fr = sxm_oriented_frames(raw, "Z")
    assert np.allclose(fr["forward"], fwd, atol=1e-15)
    assert np.allclose(fr["backward"], fwd, atol=1e-15), "反扫没有被镜像回来"


def test_scan_dir_up_puts_row_zero_at_the_top(tmp_path):
    """``:SCAN_DIR: up`` 时第一条采集的线是画面的**下**边。不翻回来,报告里的
    「第 241 行」会指到反方向去。"""
    fwd = _ramp()
    down = sxm_oriented_frames(read_sxm(str(
        _write_sxm(tmp_path / "d.sxm", fwd=fwd, scan_dir="down"))), "Z")
    up = sxm_oriented_frames(read_sxm(str(
        _write_sxm(tmp_path / "u.sxm", fwd=fwd, scan_dir="up"))), "Z")
    assert np.allclose(down["forward"], fwd, atol=1e-15)
    assert np.allclose(up["forward"], np.flipud(fwd), atol=1e-15)


def test_oriented_frames_carry_the_scale(tmp_path):
    p = _write_sxm(tmp_path / "s.sxm", fwd=_ramp(), range_m=1e-8, bias=-1.5)
    fr = sxm_oriented_frames(read_sxm(str(p)), "Z")
    assert fr["width_nm"] == pytest.approx(10.0)
    assert fr["nm_per_px"] == pytest.approx(10.0 / PX)
    assert fr["bias_v"] == pytest.approx(-1.5)
    assert fr["setpoint_a"] == pytest.approx(20e-12)
    assert fr["unit"] == "m"
    assert fr["scan_dir"] == "down"


def test_missing_channel_is_reported_not_guessed(tmp_path):
    p = _write_sxm(tmp_path / "s.sxm", fwd=_ramp())
    fr = sxm_oriented_frames(read_sxm(str(p)), "Current")
    assert fr["forward"] is None


def test_channel_lookup_is_case_insensitive(tmp_path):
    p = _write_sxm(tmp_path / "s.sxm", fwd=_ramp(), channel="Z")
    assert sxm_oriented_frames(read_sxm(str(p)), "z")["forward"] is not None


# ── 技能外壳 ────────────────────────────────────────────────────────────────

def test_metadata_is_a_read_only_analysis_skill():
    for skill in (AnalyzeScanImage(), AutoProcessScanBatch()):
        meta = skill.metadata()
        assert meta.category is SkillCategory.ANALYSIS
        assert meta.safety_level is SafetyLevel.AUTO, (
            "图像分析不碰硬件,不该要人批准")
        assert meta.description
        assert not meta.preconditions, "读文件的技能不该依赖硬件状态"
        # profile 参数必须在,否则「按样品体系切换阈值」在技能这一层就断了
        assert any(p.name == "threshold_profile" for p in meta.parameters)


def test_analyze_one_frame(tmp_path):
    p = _write_sxm(tmp_path / "one.sxm", fwd=_ramp(), bwd=np.fliplr(_ramp()))
    r = AnalyzeScanImage().execute(None, {"scan_path": str(p), "save_png": True,
                                          "output_dir": str(tmp_path / "out")})
    assert r.success, r.error
    assert r.data["plan"]["method"] in ("plane", "poly2", "line", "masked_line")
    assert r.data["metrics"]["nm_per_px"] == pytest.approx(10.0 / PX)
    assert r.data["png_path"] and (tmp_path / "out").exists()
    # 阈值出身必须跟着结论走
    from mast.vision.scan_prep_thresholds import DEFAULT_PROFILE, resolve
    assert DEFAULT_PROFILE == "generic-uncommissioned"
    assert r.data["plan"]["threshold_profile"] == DEFAULT_PROFILE
    assert r.data["plan"]["threshold_provenance"] == resolve().provenance
    assert "未标定" in (r.summary or "")
    assert DEFAULT_PROFILE in (r.summary or "")


def test_analyze_result_data_is_json_serialisable(tmp_path):
    import json

    p = _write_sxm(tmp_path / "one.sxm", fwd=_ramp())
    r = AnalyzeScanImage().execute(None, {"scan_path": str(p)})
    json.dumps(r.data)                        # 不许抛(要进 ToolMessage / 记录)


def test_missing_file_fails_cleanly():
    r = AnalyzeScanImage().execute(None, {"scan_path": "no/such/file.sxm"})
    assert not r.success
    assert "不存在" in r.error


def test_missing_channel_fails_with_the_list_of_channels(tmp_path):
    p = _write_sxm(tmp_path / "one.sxm", fwd=_ramp())
    r = AnalyzeScanImage().execute(None, {"scan_path": str(p), "channel": "Current"})
    assert not r.success
    assert "Current" in r.error and "Z" in r.error, (
        "报错里要说清楚这个文件到底有哪些通道")


def test_unknown_profile_does_not_fail_the_skill(tmp_path):
    """认不出的 profile 回落到默认并在 provenance 里说清楚 —— 不是让技能失败。"""
    p = _write_sxm(tmp_path / "one.sxm", fwd=_ramp())
    r = AnalyzeScanImage().execute(None, {"scan_path": str(p),
                                          "threshold_profile": "au111-nope"})
    assert r.success
    assert "au111-nope" in r.data["plan"]["threshold_provenance"]


def test_batch_writes_a_report_and_harmonises_the_group(tmp_path):
    """同 (视野, 偏压) 的一组图必须用同一种处理 —— 否则对比度差异会被读成样品变了。"""
    src = tmp_path / "scans"
    src.mkdir()
    # 三张平坦帧(会选 plane)+ 一张有行漂移的(会选 line)。四张同视野同偏压,
    # 多数票应该把那一张拉回 plane。
    for i in range(3):
        _write_sxm(src / f"flat_{i}.sxm", fwd=_ramp(seed=i))
    rng = np.random.default_rng(9)
    drift = np.cumsum(rng.normal(0, 3e-11, PX))[:, None]
    _write_sxm(src / "drift_3.sxm", fwd=_ramp(seed=3) + drift)

    r = AutoProcessScanBatch().execute(None, {
        "folder": str(src), "output_dir": str(tmp_path / "out"), "render": False})
    assert r.success, r.error
    assert r.data["n_files"] == 4
    report = tmp_path / "out" / "_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "generic-uncommissioned" in text
    assert "未标定" in text and "未经任何样品或仪器验证" in text
    assert "atomic_phase" in text, "报告里必须说明哪几列是转发既有判据的"
    assert len(set(f["method"] for f in r.data["frames"])) == 1, (
        f"批次一致性没生效:{[(f['file'], f['method']) for f in r.data['frames']]}")
    assert "批次一致性" in text, "被拉齐的那一张必须在报告里说明它是被拉齐的"


def test_batch_on_an_empty_folder_fails_cleanly(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    r = AutoProcessScanBatch().execute(None, {"folder": str(d)})
    assert not r.success
    assert ".sxm" in r.error


def test_batch_skips_unreadable_files_instead_of_dying(tmp_path):
    src = tmp_path / "scans"
    src.mkdir()
    _write_sxm(src / "good.sxm", fwd=_ramp())
    (src / "broken.sxm").write_bytes(b"not an sxm at all")
    r = AutoProcessScanBatch().execute(None, {
        "folder": str(src), "output_dir": str(tmp_path / "out"), "render": False})
    assert r.success, r.error
    assert r.data["n_files"] == 1
    assert any("broken.sxm" in s for s in r.data["skipped"])


def test_batch_summary_stays_short_enough_for_a_tool_message(tmp_path):
    """``data`` 会被适配层塞进模型上下文 —— 逐张的完整依据放报告,不放这里。"""
    import json

    src = tmp_path / "scans"
    src.mkdir()
    for i in range(6):
        _write_sxm(src / f"f_{i}.sxm", fwd=_ramp(seed=i))
    r = AutoProcessScanBatch().execute(None, {
        "folder": str(src), "output_dir": str(tmp_path / "out"), "render": False})
    assert r.success
    assert len(r.summary or "") < 1200
    assert len(json.dumps(r.data)) < 20000


def test_render_is_optional_and_its_failure_never_fails_the_analysis(tmp_path,
                                                                    monkeypatch):
    """画不出图不该让分析失败 —— 分析的产物是判断,不是那张 PNG。"""
    import mast.data.visualization as viz

    def _boom(*a, **kw):
        raise RuntimeError("no matplotlib on this box")

    monkeypatch.setattr(viz, "plot_flattened_scan", _boom)
    p = _write_sxm(tmp_path / "one.sxm", fwd=_ramp())
    r = AnalyzeScanImage().execute(None, {"scan_path": str(p), "save_png": True,
                                          "output_dir": str(tmp_path / "out")})
    assert r.success
    assert r.data["png_path"] == ""


# ── 标定工具(换样品体系时看分布定阈值) ────────────────────────────────────

def test_commission_reports_distributions_and_never_writes_settings(tmp_path):
    from mast.vision import scan_prep_commission as C
    from mast.vision.scan_prep_thresholds import config_path

    src = tmp_path / "scans"
    src.mkdir()
    for i in range(6):
        _write_sxm(src / f"f_{i}.sxm", fwd=_ramp(seed=i))
    batch = C.collect([str(p) for p in sorted(src.glob("*.sxm"))])
    report = C.analyse(batch)
    text = C.render(batch, report)

    assert report["n_frames"] == 6
    assert "generic-uncommissioned" in text
    assert "未标定" in text and "合成示例" in text
    assert "指标分布" in text and "不从分布标定" in text
    assert not config_path().exists(), (
        "标定工具绝不写配置 —— 建议是决策的输入,不是决策")


def test_commission_marks_which_knobs_may_not_be_calibrated_from_a_distribution():
    """``line_gain`` / ``bow_gain`` / ``step_purity`` 是关于模型选择与几何的陈述,
    工具只报双侧样本数,不给建议值。"""
    from mast.vision import scan_prep_commission as C

    report = C.analyse({"thresholds": {}, "frames": []})
    by_knob = {e["knob"]: e for e in report["metrics"] if e.get("knob")}
    for k in ("line_gain", "bow_gain", "step_purity"):
        assert by_knob[k]["calibratable"] is False
    assert by_knob["step_sep"]["calibratable"] is True


def test_commission_refuses_to_suggest_when_the_batch_has_no_two_clumps(tmp_path):
    """一个分类阈值只有在这批数确实分成两团时才有意义。分不开就明说,不给数字。"""
    from mast.vision import scan_prep_commission as C

    src = tmp_path / "scans"
    src.mkdir()
    for i in range(8):                       # 八张几乎一样的平坦帧
        _write_sxm(src / f"f_{i}.sxm", fwd=_ramp(seed=i))
    report = C.analyse(C.collect([str(p) for p in sorted(src.glob("*.sxm"))]))
    step_sep = next(e for e in report["metrics"] if e.get("knob") == "step_sep")
    assert step_sep["suggested"] is None
    assert "保留当前值" in step_sep["note"]


def test_the_skills_are_registered_for_the_frozen_build():
    """没被 ``skills.builtins.__init__`` import 到的技能,打包后会静默消失。"""
    import mast.skills.builtins as B

    assert B.AnalyzeScanImage is AnalyzeScanImage
    assert B.AutoProcessScanBatch is AutoProcessScanBatch
    assert "AnalyzeScanImage" in B.__all__
    assert "AutoProcessScanBatch" in B.__all__
