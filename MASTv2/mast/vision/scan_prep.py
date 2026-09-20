"""扫描图自动预处理：提取指标、选择处理方式并返回可追溯的说明。

本模块的纯函数不读取配置或仪器。阈值由 ScanPrepThresholds 显式传入；
公开版默认 profile 未标定，仅用于合成示例。实际数据应使用另行验证的 profile。

原子相、针尖突变、正反扫一致性与坏行判据分别委托给 atomic_phase、tip_change、
tip_metrics 和 scan_artifacts。fine_periodic_snr 只控制显示色阶，不宣称存在晶格。

平场选择比较扣平面、扣二阶曲面与逐行拟合的残差；更大的拟合空间会降低残差，
因此选择增益应明显高于纯噪声因自由度变化造成的改善。台阶保护同时检查能级分离
与行纯度，避免把严格沿行的分层误当成真实台阶。

三个反例由合成测试覆盖：行中位数统计量的类别重叠、中值高通产生精确零而压低 MAD、
以及仅凭直方图峰数无法区分台阶和行向分层。均值高通与行纯度用于避免这些问题。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import numpy.typing as npt

from mast.vision.scan_prep_thresholds import ScanPrepThresholds

logger = logging.getLogger(__name__)

#: 处理方式 → 人话。
METHOD_LABEL: dict[str, str] = {
    "plane": "扣平面",
    "poly2": "扣二阶曲面",
    "line": "逐行一阶平场",
    "masked_line": "只在主 terrace 上拟合的逐行平场",
}

METHODS: tuple[str, ...] = ("plane", "poly2", "line", "masked_line")


# ═══════════════════════════════════════════════════════════════════════
# 基本量
# ═══════════════════════════════════════════════════════════════════════

def _mad(x: npt.ArrayLike) -> float:
    """1.4826 × 中位绝对偏差。空/全非有限 → nan。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def poly_subtract(img: npt.ArrayLike, order: int = 1) -> np.ndarray:
    """减去最小二乘拟合的二维多项式曲面(order=1 即平面)。NaN 安全。

    与 ``mast.data.processors.plane_subtract`` 的区别:那个只有一阶,而且遇到 NaN
    会把整幅图算成 NaN(``lstsq`` 吃到 NaN)。未完成的扫描是常态,所以这里只用有限像素
    拟合,再把曲面从**整幅**图上减掉。
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"poly_subtract 需要二维数组,拿到 {a.shape}")
    order = int(max(0, order))
    ny, nx = a.shape
    y, x = np.mgrid[0:ny, 0:nx]
    m = np.isfinite(a)
    n_terms = (order + 1) * (order + 2) // 2
    if m.sum() < n_terms + 1:
        return a - (np.nanmean(a) if m.any() else 0.0)
    terms = [x ** j * y ** i for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.column_stack([t[m].ravel().astype(np.float64) for t in terms])
    coef, *_ = np.linalg.lstsq(A, a[m].ravel(), rcond=None)
    surface = np.zeros_like(a)
    for c, t in zip(coef, terms):
        surface = surface + c * t
    return a - surface


def line_subtract(img: npt.ArrayLike, order: int = 1,
                  mask: npt.NDArray[np.bool_] | None = None) -> np.ndarray:
    """逐行减去一条多项式。

    ``mask`` 把**拟合**限制在选中的像素上(主 terrace),而**修正仍施加到整行** ——
    这就是台阶能活下来的原因。可用像素太少的行不去拿垃圾拟合,而是从邻行的系数
    线性插值补上(否则一条几乎全 NaN 的行会甩出一个荒谬的斜率,再被减到整行上)。

    仓里既有的 ``data.processors.line_by_line_level`` 没有 mask 参数,也不处理
    NaN —— 遇到有台阶或未扫完的帧会把台阶吃掉/整行变 NaN。
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"line_subtract 需要二维数组,拿到 {a.shape}")
    order = int(max(0, order))
    ny, nx = a.shape
    x = np.arange(nx, dtype=np.float64)
    need = max(order + 2, int(0.15 * nx))
    coeffs: list[np.ndarray | None] = [None] * ny
    for i in range(ny):
        row = a[i]
        m = np.isfinite(row)
        if mask is not None:
            m = m & np.asarray(mask[i], dtype=bool)
        if int(m.sum()) >= need:
            coeffs[i] = np.polyfit(x[m], row[m], order)
    valid = [i for i, c in enumerate(coeffs) if c is not None]
    if not valid:
        finite = np.isfinite(a)
        return a - (np.nanmean(a) if finite.any() else 0.0)
    C = np.array([coeffs[i] for i in valid], dtype=np.float64)
    filled = np.column_stack([np.interp(np.arange(ny), valid, C[:, k])
                              for k in range(order + 1)])
    baseline = np.array([np.polyval(filled[i], x) for i in range(ny)])
    return a - baseline


def dominant_terrace_mask(flat: npt.ArrayLike, roughness: float,
                          separation: float) -> npt.NDArray[np.bool_]:
    """属于**主** terrace 的像素 —— masked 逐行平场的拟合域。

    主 terrace = 平滑后的高度直方图里最高的那个峰;半宽取
    ``max(3×粗糙度, 峰间距/3)``,于是既包住这一层的起伏,又够不到隔壁那一层。
    """
    from scipy import ndimage

    a = np.asarray(flat, dtype=np.float64)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.ones(a.shape, dtype=bool)
    lo, hi = np.percentile(v, [0.3, 99.7])
    if not (hi > lo):
        return np.isfinite(a)
    hist, edges = np.histogram(v, bins=256, range=(float(lo), float(hi)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), 4)
    main = float(centers[int(np.argmax(smooth))])
    rough = float(roughness) if np.isfinite(roughness) else 0.0
    sep = float(separation) if np.isfinite(separation) else 0.0
    half = max(3.0 * rough, sep / 3.0 if sep > 0 else 0.0, 1e-13)
    return np.isfinite(a) & (np.abs(a - main) < half)


def row_correlation(img: npt.ArrayLike) -> np.ndarray:
    """每一行与下一行的相关系数。数据太少的行给 NaN,**不是** 0。

    (给 0 会把「这行没数据」和「这行与邻行完全不相关」混成一件事,而中位数会被
    前者拖低 —— 一张扫了一半的图会因此被标成「噪声帧」。)
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return np.array([], dtype=np.float64)
    m = np.isfinite(a)
    ok = m.mean(axis=1) > 0.5
    cnt = np.maximum(m.sum(axis=1), 1)
    # Deliberately NOT np.nanmean/np.nanstd: an all-NaN row makes both emit a
    # RuntimeWarning per call, and a batch of unfinished scans then buries the
    # console in warnings that say nothing the `ok` mask does not already say.
    filled = np.where(m, a, 0.0)
    mean = filled.sum(axis=1) / cnt
    b = np.where(m, a - mean[:, None], 0.0)
    s = np.sqrt((b * b).sum(axis=1) / cnt) + 1e-30
    c = np.sum(b[:-1] * b[1:], axis=1) / (a.shape[1] * s[:-1] * s[1:])
    c = np.asarray(c, dtype=np.float64)
    c[~(ok[:-1] & ok[1:])] = np.nan
    return c


def acquired_row_span(img: npt.ArrayLike, min_finite_frac: float = 0.5
                      ) -> tuple[int, int]:
    """真正采到数据的那一段行 ``[r0, r1)`` —— 最长的连续「大部分像素有限」行段。

    为什么需要:未扫完的帧下半部是整片 NaN。把它填成中位数再交给
    :func:`mast.vision.tip_change.detect_tip_change`(它不吃 NaN),那片人造的常数
    平台会在边界上造出一个巨大的行 DC 跳变 —— 一个**由填充制造的**「针尖突变」。
    所以转发之前先把分析限制在采到的那一段上,并在结果里说明用了哪一段。
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return (0, 0)
    ok = np.isfinite(a).mean(axis=1) > float(min_finite_frac)
    best = (0, 0)
    start = None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def fine_periodic_peak(img: npt.ArrayLike, nm_per_px: float,
                       th: ScanPrepThresholds) -> dict:
    """受限带 FFT 峰，对局部环背景打分，返回 snr、period_nm 与 angle_deg。

    这里只决定色阶，晶格结论由 atomic_phase 给出。扫描轴附近由可配置的
    axis_guard_deg 屏蔽，以降低扫描线条纹的影响；沿扫描轴的周期结构也可能因此
    被忽略。该取舍只改变显示范围，不修改输入数据，也不构成晶格验收。
    """
    out = {"snr": 0.0, "period_nm": float("nan"), "angle_deg": float("nan")}
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 16:
        return out
    if not (nm_per_px and np.isfinite(nm_per_px) and nm_per_px > 0):
        return out
    ny, nx = a.shape
    finite = np.isfinite(a)
    if not finite.any():
        return out
    g = np.nan_to_num(a - float(a[finite].mean()), nan=0.0, posinf=0.0, neginf=0.0)
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    dy, dx = yy - cy, xx - cx
    # 归一化成「周期/像素」,矩形帧也成立。
    fy, fx = dy / float(ny), dx / float(nx)
    freq = np.hypot(fy, fx)
    with np.errstate(divide="ignore", invalid="ignore"):
        period_px = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-30), np.inf)
    period_nm = period_px * float(nm_per_px)
    # 角度取在**频率**空间(fy, fx)而不是像素索引空间:方帧上两者相同,矩形帧上只有
    # 前者是物理方向。
    ang = np.degrees(np.arctan2(fy, fx))
    axis_dist = np.abs(((ang + 90.0) % 180.0) - 90.0)      # 0 = x 轴, 90 = y 轴
    guard = float(th.axis_guard_deg)
    band = (
        (period_nm >= float(th.fine_period_min_nm))
        & (period_nm <= float(th.fine_period_max_nm))
        & (period_px >= 3.0)                                # <3 px 的周期是采样噪声
        & (axis_dist > guard)
        & (np.abs(axis_dist - 90.0) > guard)
    )
    if int(band.sum()) < 50:
        return out
    pi = np.unravel_index(int(np.argmax(np.where(band, F, 0.0))), F.shape)
    f_peak = float(freq[pi])
    if f_peak <= 0:
        return out
    ring = band & (np.abs(freq - f_peak) < max(1.0 / max(ny, nx), 0.08 * f_peak))
    bg = float(np.median(F[ring])) if ring.any() else 0.0
    if not np.isfinite(bg) or bg <= 0:
        return out
    return {"snr": float(F[pi] / bg),
            "period_nm": float(period_nm[pi]),
            "angle_deg": float(ang[pi])}


def height_levels(flat: npt.ArrayLike) -> dict:
    """高度直方图上的能级:``{n_peaks, separation, row_purity}``(高度单位同输入)。

    ``row_purity`` 是这一条的全部价值:在两个最强峰之间的谷底切一刀,问**有多少行
    完全落在同一侧**(>95% 或 <5%)。真台阶横跨画面 → 大多数行同时含两个高度 → 纯度低;
    针尖突变 / z 漂移严格按行切 → 纯度高。只数峰会把后者当成台阶去保护,
    然后用宽色阶掩盖精细结构；合成台阶与行向分层测试覆盖这一区别。
    """
    from scipy import ndimage
    from scipy.signal import find_peaks

    a = np.asarray(flat, dtype=np.float64)
    out = {"n_peaks": 0, "separation": 0.0, "row_purity": float("nan")}
    if a.ndim != 2:
        return out
    v = a[np.isfinite(a)]
    if v.size <= 100:
        return out
    lo, hi = np.percentile(v, [0.3, 99.7])
    if not (hi > lo):
        return out
    hist, edges = np.histogram(v, bins=256, range=(float(lo), float(hi)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), 4)
    peaks, _ = find_peaks(smooth, prominence=float(smooth.max()) * 0.10, distance=10)
    out["n_peaks"] = int(len(peaks))
    if len(peaks) <= 1:
        return out
    out["separation"] = float(centers[peaks].max() - centers[peaks].min())
    p_lo, p_hi = sorted(peaks[np.argsort(smooth[peaks])[-2:]])
    valley = p_lo + int(np.argmin(smooth[p_lo:p_hi + 1]))
    upper = a > centers[valley]
    ok = np.isfinite(a)
    fracs = []
    for i in range(a.shape[0]):
        if int(ok[i].sum()) > 10:
            fracs.append(float(upper[i][ok[i]].mean()))
    if fracs:
        f = np.asarray(fracs)
        out["row_purity"] = float(np.mean((f < 0.05) | (f > 0.95)))
    return out


# ═══════════════════════════════════════════════════════════════════════
# 一帧的全部测量
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FrameMetrics:
    """一帧上量到的全部东西。纯标量 + 小 dict,可直接进 ``SkillResult.data``。

    单位约定:高度类的量以**输入数组的单位**为准(``.sxm`` 的 Z 通道是米),
    带 ``_pm`` 后缀的字段已换算成皮米。
    """

    # 几何 / 完整性
    shape: tuple[int, int] = (0, 0)
    nm_per_px: float | None = None
    nan_frac: float = 0.0
    dead_rows: int = 0
    #: 实际参与转发判据的行段 ``[r0, r1)`` —— 见 :func:`acquired_row_span`。
    analysis_rows: tuple[int, int] = (0, 0)

    # 模型选择
    plane_rms_pm: float = 0.0
    line_gain: float = 1.0
    bow_gain: float = 1.0
    roughness_pm: float = 0.0

    # 表面形貌
    n_peaks: int = 0
    step_sep_pm: float = 0.0
    sep_over_rough: float = 0.0
    row_purity: float = float("nan")

    # 精细周期结构(**只决定色阶**)
    fine_periodic_snr: float = 0.0
    fine_period_nm: float = float("nan")
    fine_angle_deg: float = float("nan")

    # 帧质量
    rowcorr_median: float = float("nan")

    # ── 转发既有判据的结论(每个都可能是 None = 没算成) ──
    atomic: dict | None = None        # mast.vision.atomic_phase
    tip_change: dict | None = None    # mast.vision.tip_change
    artifacts: dict | None = None     # mast.vision.scan_artifacts
    fb_instability: float | None = None  # mast.vision.tip_metrics

    #: 哪些转发判据没算成,以及为什么。空 = 全都算了。
    delegate_errors: dict = field(default_factory=dict)

    # 内部量:masked 逐行平场要用(单位同输入)
    _roughness: float = 0.0
    _separation: float = 0.0

    def to_dict(self) -> dict:
        """JSON 友好的扁平 dict(去掉下划线开头的内部量)。"""
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["shape"] = list(self.shape)
        d["analysis_rows"] = list(self.analysis_rows)
        return d


def _safe(name: str, fn, errors: dict):
    """跑一个转发判据;失败只记一行,绝不把整个分析拖垮。"""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — 转发判据挂了不该让预处理失败
        errors[name] = f"{type(exc).__name__}: {exc}"
        logger.debug("scan_prep 转发判据 %s 失败", name, exc_info=True)
        return None


def measure_frame(z: npt.ArrayLike, *, bwd: npt.ArrayLike | None = None,
                  nm_per_px: float | None = None,
                  thresholds: ScanPrepThresholds | None = None) -> FrameMetrics:
    """量一帧。纯函数:不读文件、不读配置、不碰硬件。

    ``z`` 是正扫,``bwd`` 是**已经镜像回来**的反扫(见
    :func:`mast.io.nanonis_files.sxm_oriented_frames`;不镜像的话正反扫比较是拿一张图
    和它自己的镜像比)。
    """
    th = thresholds or ScanPrepThresholds()
    a = np.asarray(z, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"measure_frame 需要二维数组,拿到 {a.shape}")
    ny, nx = a.shape
    finite = np.isfinite(a)

    nan_frac = float(1.0 - finite.mean())
    dead_rows = int(np.sum(~finite.any(axis=1)))

    plane = poly_subtract(a, 1)
    poly2 = poly_subtract(a, 2)
    lined = line_subtract(a, 1)
    s_plane = float(np.nanstd(plane)) if finite.any() else 0.0
    s_poly2 = float(np.nanstd(poly2)) if finite.any() else 0.0
    s_line = float(np.nanstd(lined)) if finite.any() else 0.0

    # 像素级粗糙度:**均值**滤波高通。中值滤波会让残差出现大量精确的 0
    # (中心像素可能就是窗口中位数)，MAD 因此偏低并夸大峰间距/粗糙度比。
    from scipy import ndimage
    hp = plane - ndimage.uniform_filter(
        np.where(finite, plane, np.nan), (1, 9))
    rough = _mad(hp)
    rough = 0.0 if not np.isfinite(rough) else float(rough)

    levels = height_levels(plane)
    sep = float(levels["separation"])

    # ── 转发既有判据。先把分析限制在真正采到数据的那一段行上。 ──
    errors: dict = {}
    r0, r1 = acquired_row_span(a)
    span = lined[r0:r1] if r1 > r0 else lined
    span_bwd = None
    if bwd is not None:
        b = np.asarray(bwd, dtype=np.float64)
        if b.shape == a.shape:
            span_bwd = line_subtract(b, 1)[r0:r1] if r1 > r0 else line_subtract(b, 1)
        else:
            errors["fb_instability"] = (
                f"正反扫形状不一致: {a.shape} vs {b.shape}")

    def _fill(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if np.isfinite(arr).all():
            return arr
        med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
        return np.nan_to_num(arr, nan=float(med) if np.isfinite(med) else 0.0)

    atomic = None
    if nm_per_px:
        def _atomic():
            from mast.vision.atomic_phase import assess_atomic_phase
            res = assess_atomic_phase(span, nm_per_px=float(nm_per_px))
            return {
                "passed": bool(res.passed),
                "scale": res.scale,
                "period_fast_axis_nm": res.period_fast_axis_nm,
                "period_radial_nm": res.period_nm,
                "angular_concentration": float(res.angular_concentration),
                "fft_sharpness": float(res.fft_sharpness),
                "snr": float(res.snr),
                "reasons": list(res.reasons),
                "warnings": list(res.warnings),
            }
        atomic = _safe("atomic", _atomic, errors)
    else:
        errors["atomic"] = "没有像素尺度(nm/px),任何「有没有原子相」的结论都没有根据"

    def _tipchange():
        from mast.vision.tip_change import detect_tip_change
        res = detect_tip_change(_fill(span), nm_per_px=nm_per_px)
        return {
            "changed": bool(res.changed),
            # 行号换算回**整帧**坐标 —— 报告里的行号必须能对上原图。
            "change_row": (None if res.change_row is None
                           else int(res.change_row) + int(r0)),
            "score": float(res.score),
            "threshold": float(res.threshold),
            "lod": res.lod,
            "calib": res.calib,
            # 每个通道的校准分 —— 分歧时要看的就是这个(哪一路最接近阈值)。
            "channel_scores": {k: float(v) for k, v in
                               (res.channel_scores or {}).items()},
        }
    tip_change = _safe("tip_change", _tipchange, errors)

    def _artifacts():
        from mast.vision.scan_artifacts import detect_scan_artifacts
        res = detect_scan_artifacts(
            _fill(span), bwd=(None if span_bwd is None else _fill(span_bwd)))
        return {
            "has_artifact": bool(res.has_artifact),
            "oscillation": bool(res.oscillation),
            "oscillation_severity": float(res.oscillation_severity),
            "drift_px": res.drift_px,
            "bad_row_frac": float(res.bad_row_frac),
            "spike_frac": float(res.spike_frac),
        }
    artifacts = _safe("artifacts", _artifacts, errors)

    fb = None
    if span_bwd is not None:
        def _fb():
            # 直接用 _fwd_bwd_instability,**不**走 assess_tip_classical:后者开头的
            # `std < 1e-9` 早退守卫是按无量纲/pm 输入写的,而 .sxm 的 Z 通道是米
            # 固定的量纲守卫可能误拒米制输入；此处的归一化互相关与输入单位无关。
            from mast.vision.tip_metrics import _fwd_bwd_instability
            return float(_fwd_bwd_instability(_fill(span), _fill(span_bwd)))
        fb = _safe("fb_instability", _fb, errors)

    peak = fine_periodic_peak(lined, nm_per_px or 0.0, th)
    rc = row_correlation(lined)
    rc_med = float(np.nanmedian(rc)) if rc.size and np.isfinite(rc).any() else float("nan")

    return FrameMetrics(
        shape=(int(ny), int(nx)),
        nm_per_px=(float(nm_per_px) if nm_per_px else None),
        nan_frac=nan_frac,
        dead_rows=dead_rows,
        analysis_rows=(int(r0), int(r1)),
        plane_rms_pm=s_plane * 1e12,
        line_gain=float(s_plane / s_line) if s_line > 0 else 1.0,
        bow_gain=float(s_plane / s_poly2) if s_poly2 > 0 else 1.0,
        roughness_pm=rough * 1e12,
        n_peaks=int(levels["n_peaks"]),
        step_sep_pm=sep * 1e12,
        sep_over_rough=float(sep / rough) if rough > 0 else 0.0,
        row_purity=float(levels["row_purity"]),
        fine_periodic_snr=float(peak["snr"]),
        fine_period_nm=float(peak["period_nm"]),
        fine_angle_deg=float(peak["angle_deg"]),
        rowcorr_median=rc_med,
        atomic=atomic,
        tip_change=tip_change,
        artifacts=artifacts,
        fb_instability=fb,
        delegate_errors=errors,
        _roughness=rough,
        _separation=sep,
    )


# ═══════════════════════════════════════════════════════════════════════
# 决策
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FlattenPlan:
    """一帧的处理方案 + 为什么。``why`` 是决策依据,``notes`` 是给人的提醒。"""

    method: str = "plane"
    clip: tuple[float, float] = (1.0, 99.0)
    why: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    step_like: bool = False
    fine_structure: bool = False
    profile: str = ""
    provenance: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "method_label": METHOD_LABEL.get(self.method, self.method),
            "clip_percentile": list(self.clip),
            "why": list(self.why),
            "notes": list(self.notes),
            "step_like": self.step_like,
            "fine_structure": self.fine_structure,
            "threshold_profile": self.profile,
            "threshold_provenance": self.provenance,
        }


def plan_for(m: FrameMetrics, thresholds: ScanPrepThresholds | None = None,
             *, override: str | None = None) -> FlattenPlan:
    """测量 → 处理方案 + 人能读的依据。"""
    th = thresholds or ScanPrepThresholds()
    why: list[str] = []
    notes: list[str] = []

    multilevel = (m.n_peaks >= int(th.step_peaks)
                  and m.sep_over_rough > float(th.step_sep))
    row_split = bool(np.isfinite(m.row_purity) and m.row_purity > float(th.step_purity))
    step_like = bool(multilevel and not row_split)
    curved = m.bow_gain > float(th.bow_gain)
    needs_line = m.line_gain > float(th.line_gain)
    fine = m.fine_periodic_snr > float(th.fine_periodic_snr)

    if multilevel and row_split:
        notes.append(
            f"直方图上有两个高度能级,但 {m.row_purity:.0%} 的行整行落在其中一个上 —— "
            f"这是**行向分层**(针尖突变或 z 漂移),不是台阶边缘。所以按普通帧平掉,"
            f"而不是当台阶保护起来(保护它会顺带用宽色阶把精细结构压没)。")

    # ── 平场方式 ──
    if override and override != "auto":
        if override not in METHODS:
            raise ValueError(f"未知的 flatten 方式 {override!r},可选 {METHODS}")
        method = override
        why.append(f"处理方式由调用方指定为 `{method}`({METHOD_LABEL[method]}),"
                   f"自动判定被跳过")
    elif step_like and needs_line:
        method = "masked_line"
        why.append(
            f"有台阶({m.n_peaks} 个高度峰,间距 {m.sep_over_rough:.1f} 倍粗糙度,"
            f"行纯度 {m.row_purity:.2f} ≤ {th.step_purity})且行间漂移显著"
            f"(line_gain {m.line_gain:.2f} > {th.line_gain}) → 每一行只在主 terrace 上"
            f"拟合,台阶才活得下来")
    elif step_like:
        method = "poly2" if curved else "plane"
        why.append(
            f"有台阶({m.n_peaks} 个峰,{m.sep_over_rough:.1f} 倍粗糙度)但各行本来就平"
            f"(line_gain {m.line_gain:.2f} < {th.line_gain}) → 保守处理,只"
            f"{METHOD_LABEL[method]}")
    elif needs_line:
        method = "line"
        why.append(
            f"单一平坦区域,行间漂移占主导(line_gain {m.line_gain:.2f} > "
            f"{th.line_gain}) → 逐行一阶平场")
    elif curved:
        method = "poly2"
        why.append(f"面是弯的(bow_gain {m.bow_gain:.2f} > {th.bow_gain},压电弯曲/蠕变)"
                   f" → 基线改用二阶曲面")
    else:
        method = "plane"
        why.append(f"扣平面之后已经够平(line_gain {m.line_gain:.2f},"
                   f"bow_gain {m.bow_gain:.2f}) → 只扣平面")

    # ── 色阶:精细结构优先于台阶 ──
    # 分辨出来的精细起伏是更稀有的收获,而宽色阶会把它压成一团;台阶高度即使被压,
    # 也还能从 step_sep_pm 这个数字读到。
    if fine:
        clip = th.clip_lattice
        why.append(
            f"带内有周期 {m.fine_period_nm:.3f} nm 的精细结构(受限带 FFT 峰对局部环"
            f"背景 SNR {m.fine_periodic_snr:.0f} > {th.fine_periodic_snr}) → 色阶收紧到 "
            f"{clip[0]}–{clip[1]} 百分位,把起伏展开")
    elif step_like:
        clip = th.clip_step
        why.append(f"色阶放宽到 {clip[0]}–{clip[1]} 百分位,保住 {m.step_sep_pm:.0f} pm "
                   f"的台阶高度")
    else:
        clip = th.clip_default
        why.append(f"无精细周期结构、无台阶 → 默认色阶 {clip[0]}–{clip[1]} 百分位")

    notes.extend(_frame_notes(m, th, fine))
    return FlattenPlan(method=method, clip=clip, why=tuple(why), notes=tuple(notes),
                       step_like=step_like, fine_structure=fine,
                       profile=th.name, provenance=th.provenance)


def _frame_notes(m: FrameMetrics, th: ScanPrepThresholds, fine: bool) -> list[str]:
    """值得告诉人的事。**关于样品/针尖的结论一律用转发判据的原话。**"""
    notes: list[str] = []

    if m.nan_frac > float(th.nan_annotate):
        notes.append(f"扫描未完成:{m.nan_frac:.0%} 的画面没有采到"
                     f"({m.dead_rows} 个空行)。"
                     + (f"下面的判据只在第 {m.analysis_rows[0]}–{m.analysis_rows[1]} 行"
                        f"(真正采到的那一段)上算。"
                        if m.analysis_rows[1] - m.analysis_rows[0] < m.shape[0] else ""))

    # 原子相 —— 三态:有 / 没有 / 判不了。
    a = m.atomic
    if a is not None:
        if a["passed"]:
            per = a["period_fast_axis_nm"]
            notes.append(
                f"原子相判据通过(mast.vision.atomic_phase):快扫方向周期 "
                f"{per:.3f} nm,角向集中度 {a['angular_concentration']:.0f}。"
                f"引用晶格常数请用快轴这个数 —— 径向周期 "
                f"{(a['period_radial_nm'] or float('nan')):.3f} nm 会被慢轴漂移拉偏。")
        elif "scale_gate" in a["reasons"] or "unknown_pixel_size" in a["reasons"]:
            notes.append(
                f"原子相**判不了**,不是「没有」:像素尺度 "
                f"{(m.nm_per_px or float('nan')):.4f} nm/px 在尺度门之外"
                f"(reasons={a['reasons']})。要结论就换更小的视野再扫一帧。"
                + (f" 顺带一提,受限带 FFT 里确实有周期 {m.fine_period_nm:.3f} nm 的"
                   f"结构(SNR {m.fine_periodic_snr:.0f}) —— 那只够用来决定色阶,"
                   f"不足以支持「有原子分辨」。" if fine else ""))
        elif fine:
            notes.append(
                f"受限带 FFT 里有周期 {m.fine_period_nm:.3f} nm 的结构"
                f"(SNR {m.fine_periodic_snr:.0f},只用于色阶),但原子相判据**没过**"
                f"(reasons={a['reasons']}) —— 峰强度分不开针尖抖动造出的准周期条纹,"
                f"以 atomic_phase 的结论为准。")
    elif fine:
        notes.append(
            f"受限带 FFT 里有周期 {m.fine_period_nm:.3f} nm 的结构"
            f"(SNR {m.fine_periodic_snr:.0f})。**这只用来决定色阶**;"
            f"原子相判据没跑成"
            f"({m.delegate_errors.get('atomic', '未知原因')}),所以不能说这是晶格。")

    # 针尖突变 —— 转发 v2 的分与阈,包括它的 lod。
    tc = m.tip_change
    if tc is not None:
        if tc["changed"]:
            notes.append(
                f"扫描中途针尖变了(mast.vision.tip_change v2:校准 z {tc['score']:.1f} "
                f"> 阈 {tc['threshold']:.1f},{tc['calib']}),大约在第 "
                f"{tc['change_row']} 行。这一行以下是另一根针尖成的像,"
                f"定量分析前先裁掉。")
        elif tc.get("lod") is not None:
            scores = tc.get("channel_scores") or {}
            best = max(scores.items(), key=lambda kv: kv[1]) if scores else None
            notes.append(
                f"没检出针尖突变(校准 z {tc['score']:.1f} < 阈 {tc['threshold']:.1f}),"
                f"而这一帧本可以看见 ≥ {tc['lod']:.3g}(输入单位)的行 DC 跳变。"
                + (f"最接近的通道是 `{best[0]}`({best[1]:.1f})。" if best else "")
                + "未检出不代表没有变化；幅度或噪声型变化仍可能低于当前阈值。")

    if np.isfinite(m.rowcorr_median) and m.rowcorr_median < float(th.rowcorr_poor):
        notes.append(f"噪声帧:相邻行相关中位数只有 {m.rowcorr_median:.2f}")

    art = m.artifacts
    if art is not None:
        if art["bad_row_frac"] > float(th.bad_row_frac_annotate):
            notes.append(f"{art['bad_row_frac']:.1%} 的扫描线受扰"
                         f"(mast.vision.scan_artifacts)")
        if art["oscillation"]:
            notes.append(f"反馈振荡:轴上谱峰强度 {art['oscillation_severity']:.1f}"
                         f"(mast.vision.scan_artifacts) —— 调 Z 控制器增益")

    if m.fb_instability is not None:
        if m.fb_instability < float(th.fb_instability_max):
            notes.append(
                f"正反扫一致(不稳定度 {m.fb_instability:.2f} < "
                f"{th.fb_instability_max}):图上的结构是真的")
        else:
            notes.append(
                f"正反扫不一致(不稳定度 {m.fb_instability:.2f} ≥ "
                f"{th.fb_instability_max}) —— 细节存疑。"
                f"(这个量用的是允许横向位移的互相关,已经补偿了压电迟滞造成的"
                f"快轴偏移;零位移的裸相关在真机上会饱和,不能拿来判。)")

    for name, err in (m.delegate_errors or {}).items():
        notes.append(f"判据 `{name}` 没跑成:{err}")
    return notes


def _row_median_level(a: np.ndarray) -> np.ndarray:
    """减去每行的中位数(NaN 安全,不发 RuntimeWarning)。"""
    out = np.asarray(a, dtype=np.float64).copy()
    m = np.isfinite(out)
    for i in range(out.shape[0]):
        if m[i].any():
            out[i] -= float(np.median(out[i][m[i]]))
    return out


def apply_flatten(z: npt.ArrayLike, method: str, m: FrameMetrics) -> np.ndarray:
    """把选定的处理方式施加到一帧上。"""
    a = np.asarray(z, dtype=np.float64)
    if method == "plane":
        return poly_subtract(a, 1)
    if method == "poly2":
        return poly_subtract(a, 2)
    if method == "line":
        return line_subtract(a, 1)
    if method == "masked_line":
        # 主 terrace 是从**去掉逐行偏置之后**的图上找的。理由:masked_line 只在
        # 「有台阶 AND 行漂移显著」时才被选中,而行漂移一旦大过台阶高度,扣平面残差
        # 的高度直方图就被漂移糊掉,「最高的那个峰」选出来的是漂移的中位数而不是主
        # terrace —— 掩膜于是选中一堆跨越两个能级的像素,拟合被台阶带偏,台阶反而被
        # 削掉。合成实测:1 nm 量级的行漂移下,不去偏置的掩膜让重建残差从 5 pm 涨到
        # 148 pm。掩膜**只用来挑参与拟合的像素**,真正的修正仍然拟合在原图上,所以
        # 这一步不会把台阶洗掉。
        base = _row_median_level(poly_subtract(a, 1))
        return line_subtract(a, 1, mask=dominant_terrace_mask(
            base, m._roughness, m._separation))
    raise ValueError(f"未知的 flatten 方式 {method!r},可选 {METHODS}")


# ═══════════════════════════════════════════════════════════════════════
# 批次一致性
# ═══════════════════════════════════════════════════════════════════════

def harmonise_batch(items: list[tuple[object, FlattenPlan]],
                    thresholds: ScanPrepThresholds | None = None
                    ) -> list[FlattenPlan]:
    """同一组图按多数票统一处理方式。``items`` 是 ``[(分组键, 方案), …]``。

    为什么要这件事:两帧之间的对比度差异必须来自**样品**。同尺寸同偏压的一组图里,
    一张扣平面、另一张逐行平场,看图的人会读成样品变了。

    **有真台阶的帧豁免** —— 它们要的是保护性处理(masked_line),被多数票剥掉就等于
    把台阶平掉。它们也不参与投票:一张台阶帧的 line_gain 反映的是台阶,不是行漂移。
    """
    th = thresholds or ScanPrepThresholds()
    out = list(p for _k, p in items)
    groups: dict[object, list[int]] = {}
    for i, (key, _p) in enumerate(items):
        groups.setdefault(key, []).append(i)

    for key, idx in groups.items():
        if len(idx) < int(th.group_min):
            continue
        free = [i for i in idx if not out[i].step_like]
        if len(free) < int(th.group_min):
            continue
        methods = [out[i].method for i in free]
        winner = max(set(methods), key=methods.count)
        n_win = methods.count(winner)
        if n_win == len(methods):
            continue
        for i in free:
            if out[i].method == winner:
                continue
            out[i] = replace(
                out[i], method=winner,
                why=out[i].why + (
                    f"批次一致性:改用 `{winner}`({METHOD_LABEL[winner]}) —— "
                    f"分组 {key} 里 {len(free)} 张可投票的帧中有 {n_win} 张选了它。"
                    f"打算互相比较的图必须用同一种处理,否则对比度差异会被读成样品变了。",))
    return out


__all__ = [
    "METHODS",
    "METHOD_LABEL",
    "FlattenPlan",
    "FrameMetrics",
    "acquired_row_span",
    "apply_flatten",
    "dominant_terrace_mask",
    "fine_periodic_peak",
    "harmonise_batch",
    "height_levels",
    "line_subtract",
    "measure_frame",
    "plan_for",
    "poly_subtract",
    "row_correlation",
]
