# -*- coding: utf-8 -*-
"""尺度逃生参数必须透传到原子相判别，并保留角向集中度的保护。

采用独立构造的六重平面波帧检验过渡尺度：允许 reduced scale 时可计算，
结果仍携带尺度警告；纯噪声不能借此绕过集中度检查。
"""
from __future__ import annotations

import numpy as np
import pytest

from mast.skills.builtins import atomic_lattice as AL


def _hex_frame(n: int = 256, period_px: float = 8.0) -> np.ndarray:
    """三组夹角为 60° 的平面波构成独立合成六重晶格。"""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    k = 2.0 * np.pi / float(period_px)
    out = np.zeros((n, n), float)
    for deg in (0.0, 60.0, 120.0):
        t = np.deg2rad(deg)
        out += np.cos(k * (x * np.cos(t) + y * np.sin(t)))
    return out


class _Ctx:
    """技能只用 params，不碰仪器。"""


@pytest.fixture()
def frame(monkeypatch):
    """把 _load_frame 换成合成帧，nm/px 固定在过渡带里。"""
    f = _hex_frame()
    nmpp = 9.0 / 256.0          # 独立合成的过渡尺度
    monkeypatch.setattr(AL, "_load_frame", lambda *a, **k: (f, f, nmpp, ""))
    return f


def _run(params):
    return AL.AnalyseAtomicLattice().execute(_Ctx(), dict(params))


# ── 逃生门本身 ────────────────────────────────────────────────────────────

def test_the_scale_escape_exists_on_this_skill(frame):
    """参数表里得真有它 —— 没有的话，唯一的出路就是整道关闸。"""
    names = {p.name for p in AL.AnalyseAtomicLattice().metadata().parameters}
    assert "allow_reduced_scale" in names
    assert "require_atomic" in names


def test_default_still_refuses_a_transition_scale_frame(frame):
    """默认不放行 —— 这条闸的存在理由没被削弱。"""
    r = _run({"scan_path": "x.sxm"})
    assert r.success is False
    assert "scale_reduced" in (r.error or "")


def test_the_escape_lets_a_real_lattice_through(frame):
    """传了 allow_reduced_scale 就能出数。"""
    r = _run({"scan_path": "x.sxm", "allow_reduced_scale": True})
    assert r.success is True, r.error
    assert (r.data or {}).get("period_mean_nm")


def test_using_the_escape_puts_the_caveat_on_the_number(frame):
    """用了逃生门，``scale_reduced`` 必须跟着结果走。

    否则一个 0.03 nm/px 上量到的周期，和一个 0.01 nm/px 上量到的，在结果里
    长得一模一样 —— 而它们的证据强度差一档。
    """
    r = _run({"scan_path": "x.sxm", "allow_reduced_scale": True})
    warn = " ".join((r.data or {}).get("warnings") or [])
    assert "scale_reduced" in warn
    assert (r.data or {}).get("scale_gate") == "reduced"


def test_no_caveat_when_the_escape_was_not_needed(monkeypatch):
    """尺度够的帧不该被贴上「过渡带」的警告 —— 警告要有区分力。"""
    f = _hex_frame(period_px=24.0)
    monkeypatch.setattr(AL, "_load_frame", lambda *a, **k: (f, f, 0.01, ""))
    r = _run({"scan_path": "x.sxm", "allow_reduced_scale": True})
    assert r.success is True, r.error
    warn = " ".join((r.data or {}).get("warnings") or [])
    assert "scale_reduced" not in warn
    assert (r.data or {}).get("scale_gate") == "full"


# ── 拒绝语不能自打嘴巴 ────────────────────────────────────────────────────

def test_a_scale_only_refusal_does_not_claim_tip_jitter(frame):
    """只差尺度时，拒绝理由应解释尺度限制，不得断言针尖抖动。"""
    r = _run({"scan_path": "x.sxm"})
    assert r.success is False
    assert "针尖抖动" not in (r.error or ""), (
        "只差尺度却断言是针尖抖动 —— 而同一句里的角向集中度说的是反话")
    assert "allow_reduced_scale" in (r.error or ""), "得告诉人另一条路在哪"
    assert (r.data or {}).get("scale_only") is True


def test_a_real_absence_still_says_tip_jitter(monkeypatch):
    """真的没有原子相时，那句话要留着 —— 修的是错判，不是把话删了。"""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(256, 256))
    monkeypatch.setattr(AL, "_load_frame", lambda *a, **k: (f, f, 0.01, ""))
    r = _run({"scan_path": "x.sxm"})
    assert r.success is False
    assert "针尖抖动" in (r.error or "")
    assert (r.data or {}).get("scale_only") is False


def test_the_escape_does_not_disable_the_concentration_gate(monkeypatch):
    """**这是这条修复的重点**：逃生门只放尺度，不放角向集中度。

    纯噪声帧即使在过渡尺度上、即使传了 allow_reduced_scale，也必须被拦下 ——
    否则这个开关就等价于 require_atomic=false，等于什么也没修。
    """
    rng = np.random.default_rng(1)
    f = rng.normal(size=(256, 256))
    monkeypatch.setattr(AL, "_load_frame", lambda *a, **k: (f, f, 9.0 / 256.0, ""))
    r = _run({"scan_path": "x.sxm", "allow_reduced_scale": True})
    assert r.success is False, "尺度逃生门放走了一帧纯噪声"
    assert (r.data or {}).get("scale_only") is False


def test_require_atomic_false_still_bypasses_everything(monkeypatch):
    """老的整体逃生门保持原样 —— 有人已经在用它了。"""
    rng = np.random.default_rng(2)
    f = rng.normal(size=(256, 256))
    monkeypatch.setattr(AL, "_load_frame", lambda *a, **k: (f, f, 0.01, ""))
    r = _run({"scan_path": "x.sxm", "require_atomic": False})
    # 量不量得出晶格是另一回事，但**不能**因为原子相判别而被拒
    assert "没有通过原子相判别" not in (r.error or "")


# ── 常量本身 ──────────────────────────────────────────────────────────────

def test_scale_only_reasons_is_a_subset_of_the_closed_set():
    """新出局词加进 atomic_phase 时，这里漏判是安全的，错判进来才危险。"""
    from mast.vision import atomic_phase as AP

    assert AL._SCALE_ONLY_REASONS <= set(AP.ALL_REASONS), (
        "_SCALE_ONLY_REASONS 里有 atomic_phase 根本不会给出的词 —— "
        "那说明它是照记忆写的，不是照那张闭集写的")
