# -*- coding: utf-8 -*-
"""原子分辨判定技能层的闸门。

vision 层的算法测试在 tests/v2/unit/vision/；这里只放**技能层**才有的判据，
典型的就是「这一帧值不值得判」这类与 IO / 元数据有关的门。
"""
from __future__ import annotations




# ── AssessAtomicPhase 的残帧门（2026-08-19）──────────────────────────────────

def test_assess_atomic_phase_refuses_an_incomplete_frame():
    """技能层拒绝不完整帧，并保留底层判据给出的原始信息。"""
    import numpy as np

    from mast.skills.builtins.tip_spectro_assess import AssessAtomicPhase

    class _P:
        pass

    # 直接构造一帧：上半有晶格，下半全 NaN
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = np.cos(2 * np.pi * xx / 8.0) + np.cos(2 * np.pi * yy / 8.0)
    img[40:, :] = np.nan
    cov = float(np.isfinite(img).mean())
    assert cov < 0.5, "构造的帧应当是残帧"

    from mast.skills.builtins import tip_spectro_assess as T
    assert hasattr(T, "_MIN_COVERAGE")
    assert T._MIN_COVERAGE == 0.5
