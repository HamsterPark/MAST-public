# -*- coding: utf-8 -*-
"""多帧晶格定标 —— 把「压电畸变」与「热漂移」真正分开。

单帧做不到这件事
════════════════
``lattice_calibration.calibrate_from_lattice`` 从一帧里解出仿射矩阵，其中的
剪切项 **归属未定**：压电管的非正交与慢轴热漂移在一帧里produce完全相同的
图像畸变，没有任何单帧判据能区分。曾经声称正反扫能分离，
那是错的（见 ``test_forward_backward_does_not_claim_to_separate_drift``）。

多帧为什么可以
════════════
设图像坐标 ``u``（按标称标度，nm），Nanonis 的扫描角 ``θ`` 把它旋进扫描器系
``s = R(θ)·u``，压电把标称位移映射成实际位移 ``p = A·s``。**A 固定在扫描器
坐标系里**（它是压电管的几何属性，不随扫描角变）。样品上的晶格 ``cos(K_lab·p)``
在图像上就是::

    I(u) = cos( (R(θ)ᵀ Aᵀ K_lab) · u )

所以图像里量到的倒格矢 ``K_img(θ) = R(θ)ᵀ·Aᵀ·K_lab``。

热漂移速度 ``v`` 固定在**实验室系**。慢轴第 n 行的时刻正比于 ``u_y``，于是样品
额外位移 ``v·τ·u_y``，探测点相对样品变成 ``[A R(θ) − v τ ê_yᵀ]·u``，图像倒格矢::

    K_img(θ) = R(θ)ᵀ·G − c·ê_y        G ≡ Aᵀ K_lab,  c ≡ τ (v · K_lab)

**关键在这里**：``G`` 那一项随 θ 旋转，``c·ê_y`` 那一项恒定压在图像慢轴上、
且 ``c`` 与 θ 无关（``v`` 与 ``K_lab`` 都在实验室系里）。两项对 θ 的依赖不同
⇒ 扫若干个不同的 θ 就能把它们分开。这是单帧信息量不够、而多帧够的确切原因，
不是「多扫几张取平均更准」那种意义上的多帧。

未知 6 个（G¹ 2 + G² 2 + c₁ + c₂），每个角度给 4 个方程 ⇒ **两个角度即恰定，
三个以上过定**，残差成为模型是否成立的检验。角度要岔开：θ 全挤在一起时
``R(θ)ᵀ`` 几乎不变，方程退化（``angle_conditioning`` 会拦）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mast.vision.lattice_calibration import (
    LatticeResult,
    find_lattice_peaks,
    first_order_period_nm,
    solve_affine,
)

__all__ = [
    "FrameObservation",
    "MultiFrameCalibration",
    "AtomicConsistency",
    "collect_observation",
    "calibrate_multi_angle",
    "assess_atomic_consistency",
    "angle_conditioning",
]

#: 角度张开度低于这个值就拒绝求解。cond 数按设计矩阵算，2 个正交角度 ≈ 1。
_MIN_ANGLE_SPREAD_DEG = 15.0

#: 帧间晶格周期的相对一致性阈值 —— 超过就不是同一个晶格。
_CONSISTENCY_PERIOD_TOL = 0.06

#: 帧间晶格取向的一致性阈值（度）。漂移会让取向缓慢转，但一帧内转不了几度。
_CONSISTENCY_ANGLE_TOL_DEG = 4.0

#: 一对基矢的长度相对差上限。六角晶格三个方向的 |K| 本该相等，压电畸变让它们
#: 差百分之几 —— 20% 只挡得住「配到了别的结构」，挡不住正常畸变。
_PAIR_LENGTH_TOL = 0.20

# 跨帧周期相对偏差用于排除锁错衍射阶的输入。
# 该容差是软件默认值，不能代替目标仪器的畸变标定或保证任意畸变都不会误拒。
_PERIOD_OUTLIER_TOL = 0.25

#: 收一组格矢所需的最低角向集中度。真晶格 97–7645，针尖抖动 1.8–3.3 —— 两者
#: 之间差两个数量级，所以这个门槛极不敏感；它挡的是白噪声凑出来的假峰。
_MIN_CONCENTRATION = 20.0

#: 这些原因说明**帧不可用**，而不是「这帧上没有晶格」。区别是实质性的：
#: 残帧不能作为「晶格是假的」的证据，而可用帧上量不到晶格可以。
_UNUSABLE_REASONS = frozenset({"incomplete_frame", "too_small", "not_2d",
                               "all_nan", "no_finite_data"})


def _R(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


@dataclass
class FrameObservation:
    """一帧上量到的两个独立倒格矢，连同它的扫描角。"""

    angle_deg: float
    K1: np.ndarray                    # 1/nm，图像坐标系
    K2: np.ndarray
    period_mean_nm: float
    period_spread: float = 0.0
    lattice_angle_deg: float = 0.0
    label: str = ""
    nm_per_px: float = 0.0
    line_time_s: float = 0.0


@dataclass
class MultiFrameCalibration:
    ok: bool = False
    reason: str = ""
    n_frames: int = 0
    angle_spread_deg: float = 0.0
    condition_number: float = float("inf")

    # 压电部分
    x_scale: float | None = None
    y_scale: float | None = None
    shear_deg: float | None = None

    # 漂移部分 —— 单位是「每个慢轴单位长度的相位滑移」换算回的速度
    drift_nm_per_s: float | None = None
    drift_direction_deg: float | None = None
    drift_shear_deg: float | None = None

    residual_rel: float = float("nan")
    per_frame_residual: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


@dataclass
class AtomicConsistency:
    #: consistent / inconsistent / absent / undetermined —— 四个态指向四种
    #: 不同的下一步动作，见 ``assess_atomic_consistency`` 里的说明。
    verdict: str = "undetermined"
    n_frames: int = 0
    n_atomic: int = 0
    #: 帧本身用不了（残帧、标度不符）—— 它们不构成「没有晶格」的证据
    n_unusable: int = 0
    #: 帧可用、但量不到晶格 —— 这才是证据
    n_no_lattice: int = 0
    period_spread: float = float("nan")
    angle_spread_deg: float = float("nan")
    reason: str = ""
    per_frame: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def angle_conditioning(angles_deg) -> tuple[float, float]:
    """返回 (角度张开度, 设计矩阵条件数)。

    张开度用的是**倍角**的圆方差：倒格矢 ±K 不可分，所以 θ 与 θ+180° 对这个
    问题是同一个角度，直接取 max-min 会把 (0°, 179°) 当成张得很开。
    """
    a = np.asarray(list(angles_deg), float)
    if a.size < 2:
        return 0.0, float("inf")
    z = np.exp(2j * np.radians(a))
    spread = float(np.degrees(np.arccos(np.clip(abs(z.mean()), 0, 1))) )
    # 设计矩阵：每个角度贡献 R(θ)ᵀ 与 -ê_y 两块
    rows = []
    for th in a:
        R = _R(th).T
        rows.append(np.hstack([R, np.array([[0.0], [-1.0]])]))
    A = np.vstack(rows)
    try:
        cond = float(np.linalg.cond(A))
    except Exception:  # noqa: BLE001
        cond = float("inf")
    return spread, cond


def collect_observation(image, nm_per_px: float, angle_deg: float,
                        *, label: str = "",
                        line_time_s: float = 0.0) -> "FrameObservation | None":
    """从一帧里取出两个独立倒格矢（单位 1/nm）。取不到就返回 ``None`` —— **不猜**。

    ``LatticePeak.kx/ky`` 是**像素**单位（谱中心为原点），这里统一换算成 1/nm：
    多帧之间视场大小可以不同，像素单位跨帧不可比，混用会得到一个量纲自洽但
    物理错误的解。
    """
    img = np.asarray(image, float)
    pk = find_lattice_peaks(img, float(nm_per_px))
    if not pk.ok or pk.n_peaks < 4:
        return None
    # ── 峰提取不判「这是不是真晶格」 ────────────────────────────────────
    # ``find_lattice_peaks`` 只找局部极大，纯白噪声上它照样能凑出 6 个峰
    # （实测：随机场上 n_peaks=6、周期散布 0.17）。分工是清楚的 —— 判真伪
    # 是 ``atomic_phase`` 的角向集中度的事 —— 但**这个函数的输出会被喂进
    # 定标方程**，所以它必须自己先过那一关，否则白噪声会以「一组格矢」的
    # 身份进入最小二乘。阈值来自 atomic_phase 的合成晶格与带通抖动对照。
    try:
        from mast.vision.atomic_phase import assess_atomic_phase
        ap = assess_atomic_phase(img, nm_per_px=float(nm_per_px),
                                 allow_reduced_scale=True)
        if float(ap.angular_concentration) < _MIN_CONCENTRATION:
            return None
    except Exception:  # noqa: BLE001 — 判据不可用时不因此丢掉一帧
        pass
    span_nm = float(min(img.shape)) * float(nm_per_px)
    if span_nm <= 0:
        return None
    K = _independent_pair(pk, span_nm)
    if K is None:
        return None
    # 周期由实际配对的格矢计算，而非所有候选峰的平均周期。
    # 候选集合可能含低频污染，报告值必须与拟合实际使用的格矢一致。
    n1, n2 = float(np.linalg.norm(K[0])), float(np.linalg.norm(K[1]))
    per = 2.0 / (n1 + n2) if (n1 + n2) > 0 else 0.0
    spread = abs(n1 - n2) / max(n1, n2, 1e-12)
    ang_deg = math.degrees(math.atan2(K[0][1], K[0][0]))
    return FrameObservation(
        angle_deg=float(angle_deg), K1=K[0], K2=K[1],
        period_mean_nm=float(per),
        period_spread=float(spread),
        lattice_angle_deg=float(ang_deg),
        label=label, nm_per_px=float(nm_per_px), line_time_s=float(line_time_s),
    )


def _independent_pair(pk: LatticeResult, span_nm: float):
    """挑一对夹角接近 60°/120° 的独立基矢，返回 1/nm 单位的两个矢量。"""
    reps = []
    for q in pk.peaks:
        v = np.array([q.kx, q.ky], float) / span_nm
        if v[1] < 0 or (abs(v[1]) < 1e-12 and v[0] < 0):
            v = -v            # ±K 等价，只留上半平面代表元
        if not any(np.linalg.norm(v - w) < 0.05 * max(np.linalg.norm(v), 1e-9)
                   for w in reps):
            reps.append(v)
    if len(reps) < 2:
        return None
    # 六角晶格基矢配对同时要求角度与长度相容。
    # 仅看夹角可能把低频漂移或反馈残留配入晶格；长度约束应在拟合之前排除这类污染。
    best, best_err = None, 1e9
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                continue
            if abs(na - nb) / max(na, nb) > _PAIR_LENGTH_TOL:
                continue          # 长度差太多 —— 不是同一套格矢
            ang = math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1, 1))))
            err = min(abs(ang - 60.0), abs(ang - 120.0))
            if err < best_err:
                best_err, best = err, (a, b)
    if best is None or best_err > 15.0:
        return None
    a, b = best
    # 统一成夹角 120° 的一对 —— ``solve_affine`` 的第三个方程就是 cos120°，
    # 给它 60° 的一对会解出一个残差为零的假剪切（见那边的注释）。
    if float(a @ b) > 0.0:
        b = -b
    return a, b


# --- 多角度求解 -----------------------------------------------------------

def _match_to_reference(obs, G1, G2):
    """把这一帧的峰旋回公共系，挑出与 ``G1``/``G2`` 最接近的那一对。

    帧间对应不能靠「第一个峰配第一个峰」—— 扫描角一变峰在数组里的次序就变，
    而配错一对会收敛到一个残差很小、却把两个晶格方向对调了的解。
    """
    R = _R(obs.angle_deg)
    cands = []
    for v in (obs.K1, obs.K2, obs.K1 + obs.K2, obs.K1 - obs.K2):
        for sgn in (1.0, -1.0):
            cands.append(sgn * (R @ v))

    def pick(G):
        d = [float(np.linalg.norm(c - G)) for c in cands]
        i = int(np.argmin(d))
        return cands[i], d[i]

    g1, d1 = pick(G1)
    g2, d2 = pick(G2)
    if np.linalg.norm(g1 - g2) < 1e-9:
        return None, float("inf")
    return (R.T @ g1, R.T @ g2), max(d1, d2)


def calibrate_multi_angle(observations, surface: str = "Au(111)",
                          *, line_time_s: float = 0.0,
                          slow_axis_nm: float = 0.0) -> MultiFrameCalibration:
    """由多个扫描角的帧解出压电畸变与热漂移。

    见模块注释的推导 ``K_img(theta) = R(theta)^T G - c e_y``。对每个基矢 j 这是
    3 未知（``G_x, G_y, c``）、每帧 2 方程的线性最小二乘。
    """
    out = MultiFrameCalibration()
    obs = [o for o in observations if o is not None]
    out.n_frames = len(obs)
    if len(obs) < 2:
        out.reason = "need_two_angles"
        return out

    spread, cond = angle_conditioning([o.angle_deg for o in obs])
    out.angle_spread_deg, out.condition_number = spread, cond
    if spread < _MIN_ANGLE_SPREAD_DEG:
        out.reason = "angles_too_close"
        out.warnings.append(
            "扫描角只张开 %.1f 度（要求 >= %.0f）：R(theta) 几乎不变，压电项与"
            "漂移项在方程里分不开，解出来的分离是噪声。"
            % (spread, _MIN_ANGLE_SPREAD_DEG))
        return out

    d = first_order_period_nm(surface)
    if not d:
        out.reason = "unknown_surface"
        return out

    # 跨帧周期一致性在匹配之前检查。
    # 匹配器可能用其他衍射阶吸收错误周期，使拟合残差仍然很小；残差不能代替此门。
    # 以中位数作基准，避免待排除的离群帧拉偏参考周期。
    periods = np.array([o.period_mean_nm for o in obs], float)
    med = float(np.median(periods[periods > 0])) if np.any(periods > 0) else 0.0
    if med > 0:
        keep, dropped = [], []
        for o in obs:
            rel = abs(o.period_mean_nm - med) / med
            (keep if rel <= _PERIOD_OUTLIER_TOL else dropped).append((o, rel))
        if dropped:
            out.warnings.append(
                "剔除 %d 帧：周期与其余帧差 %s（中位 %.4f nm）—— 多半锁在了别的"
                "衍射阶上。**留着它拟合仍然会成功**，只是答案是错的。"
                % (len(dropped),
                   "、".join("%.0f%%" % (r * 100) for _, r in dropped), med))
            out.detail["dropped_frames"] = [
                {"angle_deg": o.angle_deg, "period_nm": o.period_mean_nm,
                 "rel_dev": r, "label": o.label} for o, r in dropped]
            obs = [o for o, _ in keep]
            out.n_frames = len(obs)
            if len(obs) < 2:
                out.reason = "too_few_frames_after_outlier_removal"
                out.warnings.append(
                    "剔除之后只剩 %d 帧，解不了。这些帧的晶格测量互相矛盾 —— "
                    "先确认它们是同一块表面、同一根针尖。" % len(obs))
                return out
            spread, cond = angle_conditioning([o.angle_deg for o in obs])
            out.angle_spread_deg, out.condition_number = spread, cond
            if spread < _MIN_ANGLE_SPREAD_DEG:
                out.reason = "angles_too_close_after_outlier_removal"
                out.warnings.append(
                    "剔除离群帧后剩下的角度只张开 %.1f°，分不开压电与漂移。"
                    % spread)
                return out

    R0 = _R(obs[0].angle_deg)
    G1, G2 = R0 @ obs[0].K1, R0 @ obs[0].K2

    matched, cs = None, [0.0, 0.0]
    for _ in range(6):          # 匹配 <-> 最小二乘交替，几轮即稳定
        rows = []
        for o in obs:
            m, _dist = _match_to_reference(o, G1, G2)
            if m is None:
                out.reason = "peak_matching_failed"
                return out
            rows.append((o, m[0], m[1]))
        matched = rows
        newG, cs = [], []
        for j in (0, 1):
            A, b = [], []
            for o, k1, k2 in rows:
                K = k1 if j == 0 else k2
                th = math.radians(o.angle_deg)
                c_, s_ = math.cos(th), math.sin(th)
                # K_x =  c*Gx + s*Gy
                # K_y = -s*Gx + c*Gy - c_drift
                A.append([c_, s_, 0.0])
                b.append(K[0])
                A.append([-s_, c_, -1.0])
                b.append(K[1])
            sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
            newG.append(np.array([sol[0], sol[1]]))
            cs.append(float(sol[2]))
        done = (np.linalg.norm(newG[0] - G1) + np.linalg.norm(newG[1] - G2)) < 1e-9
        G1, G2 = newG
        if done:
            break
    c1, c2 = cs

    per, scale = [], max(float(np.linalg.norm(G1)), 1e-12)
    for o, k1, k2 in matched:
        Rt = _R(o.angle_deg).T
        e1 = Rt @ G1 - np.array([0.0, c1]) - k1
        e2 = Rt @ G2 - np.array([0.0, c2]) - k2
        per.append(float(max(np.linalg.norm(e1), np.linalg.norm(e2)) / scale))
    out.per_frame_residual = per
    out.residual_rel = float(np.mean(per)) if per else float("nan")

    # 压电：与 theta 无关的 G 交给单帧那套对称仿射求解
    W, res = solve_affine(G1, G2, 1.0 / d)
    if W is None:
        out.reason = "affine_no_solution"
        return out
    M = np.linalg.inv(W)
    out.x_scale = float(np.linalg.norm(M[:, 0]))
    out.y_scale = float(np.linalg.norm(M[:, 1]))
    cosang = float(M[:, 0] @ M[:, 1] / (out.x_scale * out.y_scale))
    out.shear_deg = float(90.0 - math.degrees(math.acos(np.clip(cosang, -1, 1))))

    # 漂移：c_j = tau * (v . K_lab^j)
    out.drift_shear_deg = float(math.degrees(math.atan2(
        (abs(c1) + abs(c2)) / 2.0, max(float(np.linalg.norm(G1)), 1e-12))))
    # 慢轴单位距离所需时间应使用每行时间除以单像素物理步距。
    # 若误除整个慢轴视野，会差一个行数因子；应将推算漂移代回完整帧时长核对量纲。
    px_nm = float(obs[0].nm_per_px or 0.0)
    if line_time_s > 0 and px_nm > 0:
        tau_per_nm = float(line_time_s) / px_nm
        Klab = (W @ G1, W @ G2)
        A2 = np.vstack([Klab[0], Klab[1]])
        try:
            v = np.linalg.solve(A2, np.array([c1, c2]) / tau_per_nm)
            out.drift_nm_per_s = float(np.linalg.norm(v))
            out.drift_direction_deg = float(math.degrees(math.atan2(v[1], v[0])))
            # 代回场景做量纲自检：一帧时长 x 漂移速度，若超过视场本身，
            # 那这个速度不可能是真的（图会糊得认不出晶格）。
            if slow_axis_nm > 0 and px_nm > 0:
                frame_s = float(line_time_s) * (slow_axis_nm / px_nm)
                travel = out.drift_nm_per_s * frame_s
                out.detail["drift_per_frame_nm"] = travel
                if travel > slow_axis_nm:
                    out.warnings.append(
                        "漂移速度自检不过：按它算，一帧(%.0f s)要漂 %.1f nm，"
                        "而视场只有 %.1f nm —— 真这样的话这些帧上不会有晶格。"
                        "只采信 drift_shear_deg。"
                        % (frame_s, travel, slow_axis_nm))
        except Exception:  # noqa: BLE001 -- 奇异就是定不出来，不编一个
            out.warnings.append("漂移方程奇异，只报剪切当量、不报 nm/s")
    else:
        out.warnings.append(
            "没给 line_time_s（或帧里没有像素标度），漂移只能报成剪切当量 "
            "drift_shear_deg，报不出 nm/s。")

    # update 不是赋值 —— 上面的量级自检已经往 detail 里写过东西了，
    # 直接赋值会把它悄悄抹掉（第一版就是这样，自检结果永远是缺的）。
    out.detail.update({
        "G1": [float(G1[0]), float(G1[1])],
        "G2": [float(G2[0]), float(G2[1])],
        "c1": c1, "c2": c2, "affine_residual": float(res),
        "expected_period_nm": float(d),
        "angles_deg": [float(o.angle_deg) for o in obs],
    })
    out.ok = True
    if out.residual_rel > 0.08:
        out.warnings.append(
            "跨角度残差 %.1f%% 偏大：模型（刚性压电 + 匀速漂移）没有完全描述这批"
            "帧，分离结果只能当量级看。" % (out.residual_rel * 100))
    return out


# --- 多帧原子相一致性 -----------------------------------------------------

def assess_atomic_consistency(frames, nm_per_px: float,
                              *, angles_deg=None) -> AtomicConsistency:
    """同一区域重复扫若干帧，判断那个晶格是不是**真的**。

    单帧的角向集中度能排除白噪声与准周期抖动，但排除不掉「针尖在这一帧里恰好
    以某个空间频率抖」—— 那种假象在一帧内可以非常像晶格。真晶格的判据是它
    **跨帧重现同一组格矢**：抖动的频率与方向帧间不重复，晶格的重复。
    """
    out = AtomicConsistency()
    frames = list(frames)
    angs = list(angles_deg) if angles_deg else [0.0] * len(frames)
    obs, rows = [], []
    for i, img in enumerate(frames):
        o = collect_observation(img, nm_per_px,
                                angs[i] if i < len(angs) else 0.0,
                                label="frame%d" % i)
        row = {"index": i, "ok": o is not None}
        if o is not None:
            obs.append(o)
            row.update({"period_nm": o.period_mean_nm,
                        "period_spread": o.period_spread,
                        "lattice_angle_deg": o.lattice_angle_deg})
        else:
            # 为什么没量到，决定了缺帧算不算证据 —— 残帧什么都不证明。
            pk = find_lattice_peaks(np.asarray(img, float), float(nm_per_px))
            row["reason"] = pk.reason or "no_peaks"
            row["why"] = ("unusable" if pk.reason in _UNUSABLE_REASONS
                          else "no_lattice")
        rows.append(row)
    out.per_frame = rows
    out.n_frames = len(frames)
    out.n_atomic = len(obs)
    out.n_unusable = sum(1 for r in rows if r.get("why") == "unusable")
    out.n_no_lattice = sum(1 for r in rows if r.get("why") == "no_lattice")
    if out.n_frames < 2:
        out.reason = "need_two_frames"
        return out
    if len(obs) < 2:

        # 四态对应不同证据与下一步：consistent 表示多帧晶格相容；
        # inconsistent 表示已测晶格互不相容；absent 表示有效帧均未检测到晶格；
        # undetermined 表示残帧、标度等问题令测量不足，需要重新采集。
        # 不能把采集不完整归为晶格不一致，也不能把明确缺失说成判不了。
        if len(obs) == 0:
            if out.n_unusable == 0 and out.n_no_lattice >= 2:
                out.verdict = "absent"
                out.reason = "no_lattice_in_%d_usable_frames" % out.n_no_lattice
                out.warnings.append(
                    "%d 帧都完整、都没有晶格 —— 这是一个**明确的否定**，不是"
                    "「判不了」。下一步是换成像条件或修针尖，不是重扫。"
                    % out.n_no_lattice)
            else:
                out.verdict = "undetermined"
                out.reason = "no_usable_frame"
                out.warnings.append(
                    "%d/%d 帧用不了（残帧 / 标度不符），一帧晶格都没量到 —— "
                    "先把帧扫完整。" % (out.n_unusable, out.n_frames))
        elif out.n_no_lattice >= 2 and out.n_unusable == 0:
            # 帧本身可用、却量不到晶格 —— 这时孤零零那一帧的晶格才真可疑。
            out.verdict = "inconsistent"
            out.reason = ("only_1_of_%d_usable_frames_shows_a_lattice"
                          % (out.n_no_lattice + len(obs)))
            out.warnings.append(
                "%d 帧完整可用却量不到晶格，只有 1 帧有 —— 那一帧的晶格可疑："
                "针尖时好时坏，或者它本身就是假象。"
                % out.n_no_lattice)
        else:
            out.verdict = "undetermined"
            out.reason = "only_%d_frames_show_a_lattice" % len(obs)
            if out.n_unusable:
                out.warnings.append(
                    "%d/%d 帧根本用不了（残帧 / 标度不符），判不了一致性 —— "
                    "这是**采集**没完成，不是晶格有问题。重扫完整帧再判。"
                    % (out.n_unusable, out.n_frames))
        return out

    per = np.array([o.period_mean_nm for o in obs], float)
    out.period_spread = float(np.std(per) / max(float(np.mean(per)), 1e-12))
    # 取向要在扣掉各自扫描角之后比 —— 转了台子当然会转
    lab = np.array([o.lattice_angle_deg + o.angle_deg for o in obs], float)
    z = np.exp(1j * np.radians(lab * 6.0))        # 六角 60 度周期
    out.angle_spread_deg = float(
        np.degrees(np.arccos(np.clip(abs(z.mean()), 0, 1))) / 6.0)

    ok_p = out.period_spread <= _CONSISTENCY_PERIOD_TOL
    ok_a = out.angle_spread_deg <= _CONSISTENCY_ANGLE_TOL_DEG
    if ok_p and ok_a:
        out.verdict = "consistent"
        if len(obs) < out.n_frames:
            out.warnings.append(
                "%d/%d 帧量到晶格，量到的那些互相一致。缺的那几帧是真丢了分辨"
                "还是只是那一帧质量差，这里判不了。" % (len(obs), out.n_frames))
    else:
        out.verdict = "inconsistent"
        out.reason = ("period_spread=%.3f angle_spread=%.2fdeg"
                      % (out.period_spread, out.angle_spread_deg))
        out.warnings.append(
            "跨帧不重现同一组格矢 —— 每帧自己看着像晶格，但它们不是同一个晶格。"
            "针尖抖动 / 双针尖是最常见的来源。")
    return out
