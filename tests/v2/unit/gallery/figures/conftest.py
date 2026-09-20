# -*- coding: utf-8 -*-
"""出图测试的共用件（设计文档 §10）。

索引一律用**真构建**（``build.run_build``）从合成原始文件生成，标记用 ``marks.patch_marks`` 写：
出图读的 ``ang cx cy x y t mt zo ex`` 都来自构建对真文件格式的解析 —— 手写 index.json 等于替构建答题，
方向约定错了也测不出来。上一级 conftest 的 ``gallery_dir`` / ``synth`` 照用。

合成场景的方向约定只写在 :meth:`FigLab.scene` 一处：帧内 (x′ 右, y′ 上)，行 0 = 上沿；
SCAN_ANGLE 为正 = 扫描框相对压电坐标顺时针转 ⇒ 压电偏移 d = Rot(−θ)·x′。
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from mast.gallery import build, index, marks
from mast.gallery import config as gcfg

DAY_LINE = "2001/200109/20010910"
DAY_STITCH = "2001/200109/20010911"
DAY_ROT = "2001/200109/20010907"

#: 合成晶格的两个倒格矢（nm⁻¹，压电坐标 x 右 y 上）：周期 0.40 nm @ 20°、0.55 nm @ 110°。
LATTICE_G = (2.5 * np.array([math.cos(math.radians(20)), math.sin(math.radians(20))]),
             (1 / 0.55) * np.array([math.cos(math.radians(110)), math.sin(math.radians(110))]))


@pytest.fixture(autouse=True)
def _no_figure_job_left_running():
    """出图后台线程不许活过测试（与上一级 conftest 对构建线程的要求相同）。"""
    yield
    from mast.gallery.figures import service

    service._reset_for_tests()


def epoch(s: str) -> float:
    return time.mktime(time.strptime(s, "%d.%m.%Y %H:%M:%S"))


def stamp(t: float) -> str:
    return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(t))


class FigLab:
    """合成原始文件 → 真构建 → 标记。"""

    G = LATTICE_G
    epoch = staticmethod(epoch)

    def __init__(self, root, state, synth):
        self.root, self.state, self.synth = root, state, synth

    # ── 构建与标记 ────────────────────────────────────────────────────
    def build(self) -> dict:
        roots, errors = gcfg.normalise_roots([{"name": "SPM", "path": str(self.root)}])
        assert errors == []
        gcfg.save_config(gcfg.GalleryConfig(roots=roots, workers=2))
        res = build.run_build()
        assert res["phase"] == "done", res
        return index.read_index()

    @staticmethod
    def mark(items=None, series=None) -> None:
        marks.patch_marks({"items": items or {}, "series": series or {}})

    def figure_meta(self, key: str) -> dict:
        import json

        cat, base = key.split("/", 1)
        return json.loads((self.state / "figures" / cat / (base + ".figure.json")).read_text("utf-8"))

    # ── 原始文件 ──────────────────────────────────────────────────────
    def spectrum(self, day, name, *, t, xy_nm, V, I_pA, li_y_fA=None, li_x_fA=None, zoff_pm=0.0, sweeps=2,
                 lockin="ON") -> str:
        cols = ["Bias calc (V)", "Current [AVG] (A)", "Current [AVG] [bwd] (A)"]
        data = [V, I_pA * 1e-12, I_pA * 1e-12]
        if li_x_fA is not None:
            cols.append("LI Demod 1 X [AVG] (A)")
            data.append(li_x_fA * 1e-15)
        if li_y_fA is not None:
            cols.append("LI Demod 1 Y [AVG] (A)")
            data.append(li_y_fA * 1e-15)
        hdr = {"X (m)": "%.9E" % (xy_nm[0] * 1e-9), "Y (m)": "%.9E" % (xy_nm[1] * 1e-9),
               "Z offset (m)": "%.6E" % (zoff_pm * 1e-12), "Start time": stamp(t), "Saved Date": stamp(t + 30),
               "Bias Spectroscopy>Number of sweeps": str(sweeps), "Lock-in>Lock-in status": lockin}
        self.synth.dat(self.root / day / name, cols, np.column_stack(data), header=hdr, mtime_ns=int((t + 30) * 1e9))
        return f"SPM/{day}/{name}"

    def frame(self, day, name, z_top_m, *, t, range_nm, offset_nm=(0.0, 0.0), angle=0.0, scan_dir="down",
              bias=1.0, setpoint_a=1e-10) -> str:
        lt = time.localtime(t)
        self.synth.sxm(self.root / day / name, z_top_m, rec_date=time.strftime("%d.%m.%Y", lt),
                       rec_time=time.strftime("%H:%M:%S", lt), range_nm=range_nm, offset_nm=offset_nm, angle=angle,
                       scan_dir=scan_dir, bias=bias, setpoint_a=setpoint_a, mtime_ns=int((t + 60) * 1e9))
        return f"SPM/{day}/{name}"

    # ── 合成场景 ──────────────────────────────────────────────────────
    @staticmethod
    def scene(th_deg, *, px=128, w=5.0, c=(0.0, 0.0), shift=(0.0, 0.0), defect=(0.8, 0.5), seed=0) -> np.ndarray:
        """一帧 Z（pm，行 0 = 上沿）：晶格 + 一个暗缺陷，整体随 ``shift`` 漂移；帧心在压电 ``c``，转角 ``th_deg``。"""
        rng = np.random.default_rng(seed)
        g1, g2 = LATTICE_G
        xp = ((np.arange(px) + 0.5) / px - 0.5) * w                    # 帧内 x′（右）
        yp = (0.5 - (np.arange(px) + 0.5) / px) * w                    # 帧内 y′（上）
        XP, YP = np.meshgrid(xp, yp)
        a = math.radians(th_deg)
        X = c[0] + XP * math.cos(a) + YP * math.sin(a) - shift[0]      # d = Rot(−θ)·x′，再扣掉样品漂移
        Y = c[1] - XP * math.sin(a) + YP * math.cos(a) - shift[1]
        z = 20 * (np.cos(2 * np.pi * (g1[0] * X + g1[1] * Y)) + 0.8 * np.cos(2 * np.pi * (g2[0] * X + g2[1] * Y)))
        z = z - 60 * np.exp(-((X - defect[0]) ** 2 + (Y - defect[1]) ** 2) / (2 * 0.35 ** 2))
        return z + rng.normal(0, 0.5, z.shape)

    def line_dataset(self, *, drift_nm=0.08, step_nm=0.2, n=6, angle_deg=30.0) -> dict:
        """一条线四个区组：A 正序、B 逆序且整体漂移 drift_nm、C 三个站位 + 一条落在两站位正中间、D 评为排除。"""
        rng = np.random.default_rng(11)
        u = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
        perp = np.array([-u[1], u[0]])
        p0 = np.array([12.0, -7.0])
        base = epoch("10.09.2001 15:00:00")
        V = np.linspace(2.0, -2.0, 81)
        centre = p0 + u * step_nm * (n - 1) / 2
        for k, t in (("a", base + 300), ("c", base + 3700)):
            self.frame(DAY_LINE, f"stm{k}_0001.sxm", rng.normal(0, 1e-11, (32, 32)), t=t, range_nm=(3.0, 3.0),
                       offset_nm=tuple(centre))
        truth: dict[str, int] = {}
        counter = [0]

        def put(station, t, along=0.0, keep=True):
            counter[0] += 1
            p = p0 + u * (station * step_nm + along + rng.normal(0, 0.002)) + perp * rng.normal(0, 0.002)
            amp = 1 + 0.1 * station
            sid = self.spectrum(DAY_LINE, "line_rep%05d.dat" % counter[0], t=t, xy_nm=p, V=V,
                                I_pA=30 * np.tanh(1.5 * V) * amp + rng.normal(0, 0.02, V.size),
                                li_y_fA=14 * 45 / np.cosh(1.5 * V) ** 2 * amp + rng.normal(0, 0.5, V.size),
                                li_x_fA=rng.normal(0, 3, V.size))
            if keep:
                truth[sid] = station
            return sid

        A = [put(i, base + 600 + 60 * i) for i in range(n)]
        B = [put(st, base + 2000 + 60 * i, along=drift_nm) for i, st in enumerate(reversed(range(n)))]
        C = [put(0, base + 4000), put(1, base + 4060)]
        outlier = put(1.5, base + 4120, keep=False)
        C += [outlier, put(2, base + 4180)]
        D = [put(3, base + 5000), put(4, base + 5060)]
        series = {
            "SA": {"name": "测试线 · 区组0 · L9 过缺陷", "ids": A, "r": 2, "ts": 1},
            "SB": {"name": "测试线 · 区组1 · L9 过缺陷", "ids": B, "r": 2, "ts": 1},
            "SC": {"name": "测试线 · 区组2 · L9 过缺陷（中断）", "ids": C, "r": 1, "ts": 1},
            "SD": {"name": "测试线 · 区组3 · L9 过缺陷", "ids": D, "r": -1, "ts": 1},
        }
        return dict(truth=truth, outlier=outlier, series=series, A=A, B=B, C=C, D=D, u=u, n=n)

    def stitch_dataset(self) -> dict:
        """一个目录里的单根谱：两条同范围核心、一条向上接（重叠 3 点）、一条向下只在端点相接、一条 PSD、一条系列成员。"""
        rng = np.random.default_rng(5)
        base = epoch("11.09.2001 10:00:00")
        anchor = self.frame(DAY_STITCH, "img_0012.sxm", rng.normal(0, 1e-11, (32, 32)), t=base - 1800,
                            range_nm=(4.0, 4.0))

        def seg(name, V, scale, zoff, t):
            return self.spectrum(DAY_STITCH, name, t=t, xy_nm=(0.2, 0.1), V=V, I_pA=100 * np.tanh(2 * V) / scale,
                                 li_y_fA=14 * 200 / np.cosh(2 * V) ** 2 / scale, li_x_fA=rng.normal(0, 0.5, V.size),
                                 zoff_pm=zoff)

        core = [seg("rep00001.dat", np.linspace(1.0, -1.0, 41), 1.0, 0.0, base),
                seg("rep00002.dat", np.linspace(1.0, -1.0, 41), 1.0, 0.0, base + 120)]
        up = seg("rep00005.dat", np.linspace(0.9, 2.0, 23), 3.0, 50.0, base + 240)
        down = seg("rep00007.dat", np.linspace(-1.0, -2.0, 21), 2.0, 150.0, base + 360)
        f = np.logspace(0, 3, 40)
        self.synth.dat(self.root / DAY_STITCH / "psd_0003.dat", ["Frequency (Hz)", "Z PSD"],
                       np.column_stack([f, 1.0 / f]), header={"Experiment": "Spectrum", "Start time": stamp(base + 400)},
                       mtime_ns=int((base + 430) * 1e9))
        member = seg("rep00009.dat", np.linspace(1.0, -1.0, 41), 1.0, 0.0, base + 500)
        return dict(anchor=anchor, core=core, up=up, down=down, psd=f"SPM/{DAY_STITCH}/psd_0003.dat", member=member)

    def rotation_dataset(self, *, angles=(20.0, 50.0, 80.0, 20.0, 50.0, 80.0), px=128, w=5.0) -> dict:
        """转角系列：帧心每帧挪 (0.1, −0.05) nm，样品（晶格 + 缺陷）每帧漂 (0.03, −0.02) nm。"""
        base = epoch("07.09.2001 20:00:00")
        c0 = np.array([30.0, -40.0])
        d0 = c0 + np.array([0.3, -0.2])
        ids, defect = [], []
        for k, th in enumerate(angles):
            c = c0 + k * np.array([0.1, -0.05])
            shift = k * np.array([0.03, -0.02])
            z = self.scene(th, px=px, w=w, c=tuple(c), shift=tuple(shift), defect=tuple(d0), seed=k)
            ids.append(self.frame(DAY_ROT, "rot_%04d.sxm" % (k + 1), z * 1e-12, t=base + 400 * k, range_nm=(w, w),
                                  offset_nm=tuple(c), angle=th))
            defect.append(d0 + shift)
        return dict(ids=ids, defect=defect, angles=list(angles), px=px, w=w)


@pytest.fixture()
def lab(tmp_path, gallery_dir, synth) -> FigLab:
    return FigLab(tmp_path / "SPM", gallery_dir, synth)
