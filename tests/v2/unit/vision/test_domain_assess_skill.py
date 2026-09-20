"""``AssessDomainPhase`` 外壳(``mast.skills.builtins.domain_assess``)。

外壳的职责只有三件:读 ``.sxm``、解析像素尺度与**帧角**、把两个纯函数串起来。
所以这里测的是那三件事**没有在半路上把信息弄丢**,尤其:

* 帧角从 header 直读 —— ``sxm_frame_meta`` 不转发角度(设计陷阱 1),走那条路的帧
  角度已经没了;
* header 里**没有** ``:SCAN_ANGLE:`` 时必须是 ``unknown_frame_angle``,**不是 0°**;
* 没有参照系时输出「指纹 + no_reference」,而且技能**成功**(不是失败)。

真 ``.sxm`` 在 ``tmp_path`` 里现写(大端 float32 + 真头部格式),不碰任何真实数据;
参照系目录用 ``MAST2_PROJECT_ROOT`` 重定向 —— 那是 ``_runtime_paths.project_root``
**实际读的**那个环境变量。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import _domain_synth as S
import numpy as np
import pytest

from mast.skills.builtins.domain_assess import NEXT_STEP, AssessDomainPhase
from mast.vision import domain_phase as D
from mast.vision import domain_reference as R

FRAME_M = S.NMPP * S.PIXELS * 1e-9          # 5 nm


def write_sxm(path, arr, *, range_m: float = FRAME_M, angle_deg: float | None = 0.0):
    """最小但真实的 Nanonis ``.sxm``。``angle_deg=None`` ⇒ header 里没有角度。"""
    ny, nx = arr.shape
    angle_line = "" if angle_deg is None else f":SCAN_ANGLE:\n\t{angle_deg:.3E}\n"
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 14.08.2026\n"
        ":REC_TIME:\n12:00:00\n"
        ":BIAS:\n\t-1.200000E+0\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_OFFSET:\n{0.0:>19.6E}{0.0:>19.6E}\n"
        f":SCAN_RANGE:\n{range_m:>19.6E}{range_m:>19.6E}\n"
        + angle_line +
        ":SCAN_DIR:\ndown\n"
        ":Z-CONTROLLER>SETPOINT:\n50.0000E-12\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    blob = struct.pack(">%df" % arr.size, *arr.astype(np.float32).ravel().tolist())
    Path(path).write_bytes(header.encode("utf-8") + b"\x1a\x04" + blob)
    return Path(path)


@pytest.fixture
def refs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    R.clear_cache()
    d = tmp_path / "config" / R.DIRNAME
    d.mkdir(parents=True)
    yield d
    R.clear_cache()


def run(path, **params):
    p = {"scan_path": str(path), "channel": "Z"}
    p.update(params)
    return AssessDomainPhase().execute(None, p)


def lattice_sxm(tmp_path, *, theta=17.0, scan_angle=0.0, seed=1, name="a.sxm",
                header_angle: float | None = None):
    img = S.lattice_frame(theta_deg=theta, scan_angle_deg=scan_angle, seed=seed,
                          noise=2e-12)
    return write_sxm(tmp_path / name, img,
                     angle_deg=(scan_angle if header_angle is None else header_angle))


# ── 1. 没有参照系:成功 + 指纹 + no_reference ══════════════════════════════

def test_without_a_reference_it_still_succeeds_and_reports_the_fingerprint(
        tmp_path, refs_dir):
    res = run(lattice_sxm(tmp_path))
    assert res.success, res.error
    assert res.data["verdict"] == "undetermined"
    assert res.data["verdict_reason"] == "no_reference"
    assert res.data["label"] is None
    assert len(res.data["fingerprint"]) == 3
    assert res.data["scan_angle_deg"] == pytest.approx(0.0)
    assert res.data["atomic_passed"] is True
    assert res.data["next_step"] == NEXT_STEP["no_reference"]
    # 指纹是三元组 (样品系角度, 周期nm, 相对功率),JSON 友好。
    json.dumps(res.data)
    for ang, per, rel in res.data["fingerprint"]:
        assert 0.0 <= ang < 180.0 and 0.2 < per < 0.3 and 0.0 <= rel <= 1.0


def test_the_skill_never_invents_a_label(tmp_path, refs_dir):
    """R6 的通过判据:普查阶段出现任何 label 都说明有硬编码漏进来了。"""
    for i in range(4):
        res = run(lattice_sxm(tmp_path, theta=10.0 * i, seed=i, name=f"s{i}.sxm"))
        assert res.data["label"] is None
        assert res.data["verdict"] == "undetermined"


# ── 2. 帧角 ════════════════════════════════════════════════════════════════

def test_missing_scan_angle_is_undetermined_not_zero(tmp_path, refs_dir):
    """header 里没有 ``:SCAN_ANGLE:`` ⇒ 拒判,``scan_angle_deg`` 落 null。"""
    path = write_sxm(tmp_path / "noangle.sxm",
                     S.lattice_frame(theta_deg=17.0, seed=1, noise=2e-12),
                     angle_deg=None)
    res = run(path)
    assert res.success
    assert res.data["verdict_reason"] == "unknown_frame_angle"
    assert res.data["scan_angle_deg"] is None       # **不是 0.0**
    assert res.data["fingerprint"] == []            # 半个指纹不许流出去
    assert res.data["n_peaks"] == 3                 # 但诊断读数照报
    assert "不按 0° 处理" in res.data["next_step"]


def test_frame_angle_comes_from_the_header_and_lands_in_the_sample_frame(
        tmp_path, refs_dir):
    """转过的扫描框:样品系指纹不变,而帧系角**真的**跟着转了。"""
    a = run(lattice_sxm(tmp_path, scan_angle=0.0, seed=2, name="a0.sxm"))
    b = run(lattice_sxm(tmp_path, scan_angle=30.0, seed=3, name="a30.sxm"))
    assert a.data["scan_angle_deg"] == pytest.approx(0.0)
    assert b.data["scan_angle_deg"] == pytest.approx(30.0)
    sa = sorted(t[0] for t in a.data["fingerprint"])
    sb = sorted(t[0] for t in b.data["fingerprint"])
    assert sa == pytest.approx(sb, abs=0.5)         # 样品系:不变
    assert min(abs((x - y + 90) % 180 - 90)
               for x in a.data["peaks_frame_deg"]
               for y in b.data["peaks_frame_deg"]) > 20.0   # 帧系:转了


def test_sxm_frame_meta_still_drops_the_angle(tmp_path):
    """陷阱 1 的现状钉子:``sxm_frame_meta`` **不**转发角度,所以外壳读 header。

    哪天它转发了(设计开放问题 1),这条会红 —— 那时把外壳改成走它,并删掉这条。
    """
    from mast.io.nanonis_files import read_sxm, sxm_frame_meta

    path = lattice_sxm(tmp_path, scan_angle=30.0, name="meta.sxm")
    header = read_sxm(str(path))["header"]
    assert "scan_angle" in header, "header 里就没有角度,那是另一个问题"
    assert "scan_angle" not in sxm_frame_meta(header)


# ── 3. 有参照系:label / mixed / no_match ═══════════════════════════════════

def _reference_body(fps: dict, **match):
    m = {"w_angle": 1.0, "w_period": 1.0, "match_tol": 0.06,
         "ambiguity_margin": 0.03, "mixed_coverage_min": 0.8}
    m.update(match)
    return {
        "schema": 1, "version": "v001", "sample": "synthetic",
        "created": "2026-08-14", "confirmed_by": "test",
        "provenance": "测试生成的合成参照系,不代表任何真实样品。",
        "symmetry_deg": 60.0, "labels": list(fps),
        "prototypes": {k: {"peaks": [list(t) for t in v]} for k, v in fps.items()},
        "match": m,
    }


@pytest.fixture
def two_domain_reference(refs_dir):
    def _fp(theta, seed):
        return D.extract_fingerprint(
            S.lattice_frame(theta_deg=theta, seed=seed, noise=2e-12),
            nm_per_px=S.NMPP, scan_angle_deg=0.0).triples()

    body = _reference_body({"A": _fp(0.0, 1), "B": _fp(25.0, 2)})
    (refs_dir / "synthetic-2026-08-14-v001.json").write_text(
        json.dumps(body, ensure_ascii=False), encoding="utf-8")
    R.clear_cache()
    return body


def test_labels_a_frame_when_a_reference_exists(tmp_path, two_domain_reference):
    a = run(lattice_sxm(tmp_path, theta=0.0, seed=3, name="da.sxm"))
    b = run(lattice_sxm(tmp_path, theta=25.0, seed=4, name="db.sxm"))
    assert (a.data["verdict"], a.data["label"]) == ("A", "A")
    assert (b.data["verdict"], b.data["label"]) == ("B", "B")
    assert a.data["reference_version"] == "v001"
    assert a.data["distances"]["A"] < a.data["distances"]["B"]
    assert "畴 A" in a.summary


def test_a_boundary_frame_is_mixed(tmp_path, two_domain_reference):
    both = (S.lattice_frame(theta_deg=0.0, seed=11, noise=2e-12)
            + S.lattice_frame(theta_deg=25.0, seed=12, noise=2e-12))
    res = run(write_sxm(tmp_path / "mixed.sxm", both))
    assert res.data["verdict"] == "mixed"
    assert res.data["label"] is None
    assert min(res.data["coverage"].values()) >= 0.8
    assert "畴界就在这一帧" in res.summary


def test_a_third_cluster_is_escalated_not_labelled(tmp_path, two_domain_reference):
    res = run(lattice_sxm(tmp_path, theta=40.0, seed=5, name="third.sxm"))
    assert res.data["verdict"] == "undetermined"
    assert res.data["verdict_reason"] == "no_match"
    assert "交给人看" in res.data["next_step"]


def test_a_missing_version_is_refused_not_silently_replaced(
        tmp_path, two_domain_reference):
    res = run(lattice_sxm(tmp_path, theta=0.0, seed=3, name="ver.sxm"),
              reference_version="v999")
    assert res.data["verdict_reason"] == "no_reference"
    assert res.data["reference_requested"] == "v999"
    assert res.data["reference_version"] is None


# ── 4. 拒判要说清楚,而且技能仍然成功 ═══════════════════════════════════════

def test_coarse_frame_refuses_to_judge_instead_of_reporting_absence(
        tmp_path, refs_dir):
    """50 nm / 256 px:``scale_gate`` —— 不是「这里没有畴」。"""
    img = S.lattice_frame(nmpp=50.0 / 256, seed=7)
    res = run(write_sxm(tmp_path / "coarse.sxm", img, range_m=50e-9))
    assert res.success
    assert res.data["verdict_reason"] == "scale_gate"
    assert res.data["scale"] == "off"
    assert "换更小的视野" in res.data["next_step"]
    assert res.data["nm_per_px"] == pytest.approx(50.0 / 256, rel=1e-6)


def test_a_frame_without_atoms_is_not_a_new_domain(tmp_path, refs_dir):
    res = run(write_sxm(tmp_path / "noise.sxm", S.noise_frame(seed=0)))
    assert res.success
    assert res.data["verdict_reason"] == "no_atomic_phase"
    assert res.data["fingerprint"] == []
    assert "不是新畴" in res.data["next_step"]


def test_every_reason_the_shell_can_emit_has_a_next_step():
    """闭集对账:判据能给的每一条码,外壳都得给得出下一步。"""
    assert set(D.UNDETERMINED_REASONS) <= set(NEXT_STEP)
    assert all(v.strip() for v in NEXT_STEP.values())


# ── 5. 「这件事没做成」才算技能失败 ═════════════════════════════════════════

def test_missing_file_is_a_skill_failure(tmp_path):
    res = run(tmp_path / "nope.sxm")
    assert not res.success and "不存在" in res.error


def test_missing_channel_is_a_skill_failure(tmp_path, refs_dir):
    res = run(lattice_sxm(tmp_path, name="ch.sxm"), channel="Nonexistent")
    # 通道名对不上时退到文件里的第一个通道(与既有 assess 壳同形状),
    # 所以这里只要求它不炸;真正没有通道的文件在下一条测。
    assert res.success


def test_a_file_with_no_channels_is_a_skill_failure(tmp_path, refs_dir):
    bad = tmp_path / "empty.sxm"
    bad.write_bytes(b":SCAN_PIXELS:\n         0         0\n:SCANIT_END:\n\n\x1a\x04")
    res = run(bad)
    assert not res.success
    assert "通道" in res.error


def test_metadata_is_read_only_and_auto(tmp_path):
    from mast.core.types import SafetyLevel, SkillCategory

    md = AssessDomainPhase().metadata()
    assert md.name == "AssessDomainPhase"
    assert md.safety_level is SafetyLevel.AUTO
    assert md.category in (SkillCategory.READ, SkillCategory.ANALYSIS)
    assert md.composition_level == 1
    names = {p.name for p in md.parameters}
    assert "scan_path" in names
    # 参照系版本可指定,但**没有**「阈值」类参数 —— 那些只能来自参照系文件。
    assert not names & {"match_tol", "symmetry_deg", "ambiguity_margin",
                        "w_angle", "w_period"}
