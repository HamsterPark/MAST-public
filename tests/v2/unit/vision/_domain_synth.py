"""畴指纹的合成语料(测试专用,不进生产包)。

参数量级照 ``atomic_phase`` 的既有形态:原子行间距 ~0.25 nm、起伏 10 pm、噪声
1-10 pm、5 nm / 256 px(= 0.0195 nm/px,刚好落在满权重尺度档)。

## 帧角约定(整个语料的地基)

样品系里晶格的方向是 ``theta_deg``;扫描框转 ``scan_angle_deg`` 之后,在**帧坐标**
里它看起来转了 ``-scan_angle_deg``。所以这里按 ``phi = theta - scan_angle`` 画,
判据端按 ``k_angle_sample = k_angle_frame + scan_angle`` 还原 —— 两边合起来才是
恒等。

⚠️ **这条约定是合成语料自己定的,不是从仪器测来的。** 所以
``test_frame_rotation_does_not_change_the_fingerprint`` 证明的是「归一化确实被执行
了」,不是「符号是对的」。符号只有真机能答(见 ``domain_phase`` 模块注释 / 设计
§6 R4)。把这两句话混起来,就会拿一个自己造的约定去「验证」自己。

角度定义与 ``herringbone.stripe_peak`` 一致:从**快扫轴(+列)**量到**行号增大**
的方向,mod 180。
"""

from __future__ import annotations

import math

import numpy as np

ROW_SPACING_NM = 0.2494          # 原子行间距的典型量级
FRAME_NM = 5.0
PIXELS = 256
NMPP = FRAME_NM / PIXELS         # 0.01953 nm/px —— 满权重档
CORRUGATION_M = 10e-12           # 10 pm
NOISE_M = 2e-12                  # 2 pm

#: 六角/矩形的对称度(样品事实,语料里显式写出来 —— 代码那边永远不许猜)
HEX_SYMMETRY_DEG = 60.0
RECT_SYMMETRY_DEG = 90.0


def lattice_frame(
    *,
    period_nm: float = ROW_SPACING_NM,
    theta_deg: float = 0.0,
    scan_angle_deg: float = 0.0,
    symmetry_deg: float = HEX_SYMMETRY_DEG,
    second_period_nm: float | None = None,
    n: int = PIXELS,
    nmpp: float = NMPP,
    amp: float = CORRUGATION_M,
    noise: float = NOISE_M,
    seed: int = 0,
    shear_px_per_row: float = 0.0,
    row_offset_m: float = 0.0,
    random_phase: bool = True,
    amp_jitter_frac: float = 0.0,
    tilt_m_per_nm: float = 0.0,
) -> np.ndarray:
    """一帧晶格高度图(米)。

    ``symmetry_deg=60`` ⇒ 三组波矢相隔 60°(六角面的实际对称性);
    ``symmetry_deg=90`` ⇒ 两组相隔 90°(矩形),``second_period_nm`` 给第二个周期
    (留空 = 与第一个相同,即正方)。

    ``shear_px_per_row`` 模拟慢轴漂移(每往下一行整行沿快扫方向平移一点);
    ``row_offset_m`` 模拟逐行随机偏移(1/f、蠕变)——**它是标准平场要减掉的那个东西**;
    ``random_phase`` 让同一个畴的不同帧不是同一张图(否则「同畴距离」测的是复制品)。

    ``amp_jitter_frac`` 是**针尖各向异性**:同一个畴的几个晶格方向被画出来的衬度
    不一样,而且随针尖状态帧帧在变。纯正弦的功率谱与相位无关,所以**只有它**能让
    三个布拉格峰的强弱顺序真的翻过来 —— 「只取 argmax 当主方向」那条被否方案的
    致命性要靠它才演示得出来(见 ``test_single_argmax_direction_flips_by_60deg_on_hex``)。
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    if shear_px_per_row:
        x = x + y * float(shear_px_per_row)
    xs, ys = x * float(nmpp), y * float(nmpp)

    if float(symmetry_deg) == 60.0:
        offsets = (0.0, 60.0, 120.0)
        periods = (period_nm,) * 3
    elif float(symmetry_deg) == 90.0:
        offsets = (0.0, 90.0)
        periods = (period_nm, second_period_nm or period_nm)
    else:  # pragma: no cover — 语料只用这两种,别的对称性得先想清楚物理
        raise ValueError(f"没有为 symmetry_deg={symmetry_deg} 定义波矢集合")

    h = np.zeros((n, n), dtype=np.float64)
    for off, per in zip(offsets, periods):
        # 帧坐标里的方向 = 样品系方向 − 扫描框角度(见模块注释)。
        phi = math.radians(float(theta_deg) + off - float(scan_angle_deg))
        k = 2.0 * math.pi / float(per)
        phase = rng.uniform(0.0, 2.0 * math.pi) if random_phase else 0.0
        w = (1.0 + rng.uniform(-1.0, 1.0) * float(amp_jitter_frac)
             if amp_jitter_frac else 1.0)
        h += w * np.cos(k * (xs * math.cos(phi) + ys * math.sin(phi)) + phase)
    h = h / float(len(offsets)) * float(amp)

    if row_offset_m:
        h = h + rng.normal(0.0, float(row_offset_m), (n, 1))
    if tilt_m_per_nm:
        h = h + xs * float(tilt_m_per_nm)
    if noise:
        h = h + rng.normal(0.0, float(noise), h.shape)
    return h


def noise_frame(*, n: int = PIXELS, sigma: float = NOISE_M, seed: int = 0) -> np.ndarray:
    """纯高斯白噪声 —— 「这儿什么都没有」。"""
    return np.random.default_rng(seed).normal(0.0, float(sigma), (n, n))


def bandpass_noise_frame(*, period_nm: float = ROW_SPACING_NM, n: int = PIXELS,
                         nmpp: float = NMPP, frac: float = 0.15,
                         amp: float = CORRUGATION_M, seed: int = 0) -> np.ndarray:
    """带通白噪声 —— 针尖抖动/反馈振铃造出的**准周期**条纹。

    整条判据链最难的对照组:它在原子带里产生一个合格的谱峰(峰强度类判据一个都
    拦不住),只有「离散布拉格点 vs 弥散环」分得开。构造照抄 ``atomic_phase`` 的
    对照组,免得两边的反例其实不是同一个东西。
    """
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.normal(0.0, 1.0, (n, n)))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    fr = np.hypot(fy, fx)
    f0 = float(nmpp) / float(period_nm)
    band = np.exp(-((fr - f0) ** 2) / (2.0 * (float(frac) * f0) ** 2))
    return np.real(np.fft.ifft2(F * band)) * float(amp)


def dead_flat_frame(*, n: int = PIXELS, value: float = 0.0) -> np.ndarray:
    """死平帧(反馈关掉/没接上)——「测出来是零」和「没测」必须是两句话。"""
    return np.full((n, n), float(value), dtype=np.float64)


def upsample2(image: np.ndarray) -> np.ndarray:
    """双线性升采样 ×2 —— **零信息注入**。

    像素数翻倍、``nm_per_px`` 减半、不带任何新信息。判据在这上面变了多少,
    就是它在量自己多少(§5.3 检验 ①)。
    """
    h = np.asarray(image, dtype=np.float64)
    ny, nx = h.shape
    # 新格点落在旧格点之间:0, 0.5, 1, 1.5, ... 边界外用边缘值(clip)。
    ry = np.clip(np.arange(2 * ny) * 0.5, 0, ny - 1)
    rx = np.clip(np.arange(2 * nx) * 0.5, 0, nx - 1)
    y0 = np.floor(ry).astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)
    wy = (ry - y0)[:, None]
    x0 = np.floor(rx).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    wx = (rx - x0)[None, :]
    top = h[y0][:, x0] * (1 - wx) + h[y0][:, x1] * wx
    bot = h[y1][:, x0] * (1 - wx) + h[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """块平均降采样 —— **信息剥夺**(§5.3 检验 ②)。物理尺寸不变 ⇒ nm/px ×factor。"""
    h = np.asarray(image, dtype=np.float64)
    f = int(factor)
    ny, nx = h.shape[0] // f * f, h.shape[1] // f * f
    return h[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def same_domain_batch(*, n_frames: int = 8, theta_deg: float = 17.0,
                      period_nm: float = ROW_SPACING_NM,
                      scan_angle_deg: float = 0.0, **kw) -> list[np.ndarray]:
    """同一个畴的一批帧:噪声 1-10 pm、剪切 0-0.3 px/行、行偏移、随机相位全变。

    **这一组的最大距离就是分离度的分母** —— 分母比分子重要。
    """
    out = []
    for i in range(int(n_frames)):
        out.append(lattice_frame(
            period_nm=period_nm, theta_deg=theta_deg,
            scan_angle_deg=scan_angle_deg,
            noise=(1.0 + 9.0 * i / max(1, n_frames - 1)) * 1e-12,
            shear_px_per_row=0.3 * i / max(1, n_frames - 1),
            row_offset_m=3e-12 * (i % 3),
            seed=100 + i, **kw))
    return out
