"""锐度的全部 verdict 均需显式分类；未知或低于采样能力的判据必须弃权，不能判为不合格。"""
from pathlib import Path

import numpy as np
import pytest

from mast.skills.builtins.tip_sharpness import (
    SHARPNESS_VERDICTS,
    AssessTipSharpness,
)
from mast.skills.composite.forge_au_tip import (
    _SHARPNESS_VERDICT_KIND,
    sharpness_verdict_kind,
)


def _make_sxm(path: Path, frame: np.ndarray, *, size_m=(1e-7, 1e-7)) -> Path:
    """最小可读 .sxm（与 test_frame_validity_guard.py 同款）。"""
    ny, nx = frame.shape
    header = (
        ":SCAN_PIXELS:\n" f"{nx} {ny}\n"
        ":SCAN_OFFSET:\n" "0.000000E+0 0.000000E+0\n"
        ":SCAN_RANGE:\n" f"{size_m[0]:.6E} {size_m[1]:.6E}\n"
        ":SCAN_ANGLE:\n" "0.000000E+0\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    data = np.asarray(frame, dtype=">f4")
    path.write_bytes(header.encode() + b"\x1a\x04" + data.tobytes() + data.tobytes())
    return path


def _one_pixel_step(n: int = 128) -> np.ndarray:
    """合成单像素台阶用于验证采样不足时的判据边界。"""
    rng = np.random.default_rng(0)
    frame = np.zeros((n, n), dtype=np.float64)
    frame[:, n // 2:] = 2.36e-10                      # 一个 Au(111) 单原子台阶
    return frame + rng.normal(0.0, 3e-12, (n, n))     # pm 级噪声，别做成死平帧


# ── 1 / 2. 没有孤儿状态，且认不出的算「判不了」 ──────────────────────────────
def test_every_verdict_the_skill_can_return_is_classified():
    """技能新增一个 verdict 而忘了教验收 ⇒ 这条红。

    这正是当初漏掉 `measured` 的那个形状：白名单读起来像在防护，
    其实它把「没想到的情况」默默判成了失败。
    """
    unclassified = [v for v in SHARPNESS_VERDICTS
                    if v not in _SHARPNESS_VERDICT_KIND]
    assert not unclassified, (
        f"这些 verdict 验收没有显式表态，会掉进默认支: {unclassified}。"
        f"请在 _SHARPNESS_VERDICT_KIND 里给它们各自的归属。")


@pytest.mark.parametrize("verdict, kind", [
    ("no_step", "undecidable"),
    ("measured", "undecidable"),     # ← bug 当时这个是 "fail"
    ("unresolved", "undecidable"),
    ("sharp", "pass"),
    ("blunt", "fail"),
])
def test_verdict_kinds(verdict, kind):
    assert sharpness_verdict_kind(verdict) == kind


@pytest.mark.parametrize("weird", ["", None, "SHARP", "who_knows", 0])
def test_unknown_verdict_is_undecidable_not_failure(weird):
    """认不出 ⇒ 判不了。**绝不能**是 fail。

    「判不了」和「不合格」在 ForgeAuTip 里导向完全不同的动作：不合格要继续
    打脉冲、重扎；判不了要去换一张能判的图。把前者当后者，等于拿一根可能
    好好的针去撞一个测量问题。
    """
    assert sharpness_verdict_kind(weird) != "fail"
    # 判定值按声明规则解析大小写。
    assert sharpness_verdict_kind("SHARP") == "pass"


# ── 3. 阈值细过采样极限 ⇒ unresolved，不是 sharp ────────────────────────────
def test_threshold_below_sampling_floor_abstains(tmp_path):
    """判定阈值低于两个采样点时弃权，不依据无法分辨的尺度做正负判决。"""
    p = _make_sxm(tmp_path / "step.sxm", _one_pixel_step(128))
    r = AssessTipSharpness().execute(None, {"scan_path": str(p),
                                            "sharp_edge_nm": 1.2})
    assert r.success, r.error
    assert r.data.get("has_step"), "这张合成帧本来就该有台阶，测试前提没成立"
    assert r.data["verdict"] == "unresolved", (
        f"阈值 1.2 nm 落在采样极限 1.5625 nm 以下，应当弃权；"
        f"实得 {r.data['verdict']}（edge={r.data.get('edge_resolution_nm')} nm）")
    assert "nm/px" in (r.summary or ""), "弃权要给出**照着做的下一步**，不是一句判不了"


def test_threshold_above_sampling_floor_still_decides(tmp_path):
    """同一张图，阈值 3 nm > 采样极限 1.5625 nm ⇒ 判得了，而且该判 sharp。

    有这一条，上面那条才不是「把判据关掉了」——弃权只发生在真的判不了时。
    """
    p = _make_sxm(tmp_path / "step2.sxm", _one_pixel_step(128))
    r = AssessTipSharpness().execute(None, {"scan_path": str(p),
                                            "sharp_edge_nm": 3.0})
    assert r.success, r.error
    assert r.data["verdict"] == "sharp", (
        f"阈值高于采样极限时应当正常判定，实得 {r.data['verdict']}")
    assert sharpness_verdict_kind(r.data["verdict"]) == "pass"


def test_no_threshold_is_undecidable_not_failure(tmp_path):
    """不传阈值 ⇒ `measured` ⇒ 验收看成「判不了」。

    这一条直接钉住那个 bug：ForgeAuTip 就是这么调的。
    """
    p = _make_sxm(tmp_path / "step3.sxm", _one_pixel_step(128))
    r = AssessTipSharpness().execute(None, {"scan_path": str(p)})
    assert r.success, r.error
    assert r.data["verdict"] == "measured"
    assert sharpness_verdict_kind(r.data["verdict"]) == "undecidable"
