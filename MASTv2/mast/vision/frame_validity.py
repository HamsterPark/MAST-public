"""针尖相关判据共用的帧有效性前置检查。

没有有效起伏的帧不能支持针尖质量或区域平坦程度的结论。不同去趋势实现
可能留下不同量级的舍入残差，因此集中执行同一有效性检查，避免各消费者
分别用 std == 0 判断。

拒判表达数据不足，不能解释成针尖不合格。rows、cols、nm_per_px、
scan_path 等元信息仍可报告；实际算得的 corrugation_rms_m 也应保留，
以区分零与未测量。roundness_score、area_px、circularity、分辨率和
区域位置等依赖有效信号的结论则必须拒判。

这里检查缺少可用起伏，不把起伏很小直接等同于无信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: 拒判时对用户/模型说的话。**这是本模块的产品** —— 它把「判不了」和
#: 「判出来不好」分开。改这句之前先读模块 docstring 的那一节。
DEAD_FLAT_REASON = (
    "这一帧是死平的（去趋势后起伏为 0）——不是针尖的问题，是数据的问题。"
    "换一块地方重扫一张再判；不要据此去修针尖。"
)

#: 第二档：去趋势残差相对于原始峰峰值的比值下限。**低于它就是浮点舍入，不是形貌。**
#:
#: 使用无量纲比值，避免固定米数阈值随量程和目标结构高度改变有效性。
#:
#: 阈值 1e-7 从合成帧实测的分隔挑出来（``_detrend`` 残差 / 原始 ptp）：
#:
#:   死平·倾斜平面 (1 nm 量程)    1.56e-08   ← 纯舍入
#:   死平·倾斜平面 (1 µm 量程)    1.52e-08   ← 同一个数，证明它确实无量纲
#:   真实·1 pm 噪声（已在噪声底以下） 9.65e-06
#:   真实·20 pm 噪声              1.97e-04
#:   真实·Au(111) 236 pm 台阶     4.64e-04
#:
#: 1e-7 在死平那一侧留 6.5×，在**最弱**的真实帧那一侧留 97×。
#:
#: 这一档覆盖去趋势后仍有浮点残差的倾斜平面；第一档只处理残差精确为零。
#: 上述数表来自合成输入，不替代目标仪器与样品上的有效性验证。
FLAT_RATIO_MIN: float = 1e-7


@dataclass(frozen=True)
class FrameVerdict:
    """一帧能不能拿来做针尖判定。

    ``usable`` 与 ``reason`` 刻意分开：``usable=True`` 时 ``reason`` 是空串，
    调用方不必去解析一句话来判断该不该继续。
    """

    #: 能不能判。False = 这帧不含信息，任何基于信号的结论都不该给。
    usable: bool
    #: 拒判原因（中文，直接给用户看）。usable=True 时为空串。
    reason: str
    #: 去趋势后的起伏 RMS（米）。**拒判时也给** —— 它就是测出来的那个 0。
    corrugation_rms_m: float
    #: 去趋势后的图。给调用方复用，省一次去趋势（也保证大家判的是同一份数据）。
    detrended: Any = None

    def meta(self, **extra: Any) -> dict:
        """拒判时仍然答得上来的那些字段。

        调用方把它塞进 ``SkillResult.data``，这样「判不了」这件事本身
        也带着可核对的证据，而不是一个空 dict。
        """
        out: dict[str, Any] = {
            "frame_usable": self.usable,
            "corrugation_rms_m": self.corrugation_rms_m,
        }
        if not self.usable:
            out["unusable_reason"] = self.reason
        out.update({k: v for k, v in extra.items() if v is not None})
        return out


def judge_frame(image: Any) -> FrameVerdict:
    """判断高度图是否包含可用于针尖评估的信息。

    复用 tip_metrics._detrend 的行中值、平面拟合与 float32 路径，
    避免各消费者使用不同的去趋势方法而产生不同有效性结论。"""
    from mast.vision.tip_metrics import _detrend

    arr = np.asarray(image, dtype=np.float64)
    # 一维输入（单条扫描线，或 CheckLineQuality 的 fwd/bwd 行数组）提升成 (1, N)。
    # 平面拟合在单行上退化成直线拟合 —— 正是一条扫描线该做的去趋势。实测：
    # 常数行 → 0，倾斜平面的任一行 → 0（一条直线被直线拟合减干净），
    # 有台阶有噪声的行 → 6.2e-11。三种都答对，所以不需要第二套判据。
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] < 2:
        return FrameVerdict(
            usable=False,
            reason=f"这一帧不是二维图像或太小（shape={arr.shape}）——数据的问题，不是针尖的问题。",
            corrugation_rms_m=0.0)

    if not np.isfinite(arr).any():
        return FrameVerdict(
            usable=False,
            reason="这一帧整幅都不是有限数值（全 NaN/Inf）——数据的问题，不是针尖的问题。",
            corrugation_rms_m=0.0)

    # NaN 洞不该让整帧作废：真机上行尾/未扫完的部分就是 NaN。用有限值算起伏。
    filled = np.where(np.isfinite(arr), arr, np.nanmean(arr[np.isfinite(arr)]))
    try:
        h = _detrend(filled)
    except Exception as exc:  # noqa: BLE001 — 去趋势自己坏了也是「判不了」
        return FrameVerdict(
            usable=False,
            reason=f"这一帧无法去趋势（{type(exc).__name__}: {exc}）——数据的问题，不是针尖的问题。",
            corrugation_rms_m=0.0)

    std = float(np.std(h))

    # 第一档：去趋势残差精确为零。
    if not np.isfinite(std) or std <= 0.0:
        return FrameVerdict(usable=False, reason=DEAD_FLAT_REASON,
                            corrugation_rms_m=0.0, detrended=h)

    # 第二档：残差相对原始峰峰值小到只可能是浮点舍入（见 FLAT_RATIO_MIN）。
    # 一张**带倾斜**的死平帧残差是 ~2e-15 而不是 0，第一档拦不住它。
    finite = filled[np.isfinite(filled)]
    ptp = float(np.ptp(finite)) if finite.size else 0.0
    if ptp > 0.0 and (std / ptp) < FLAT_RATIO_MIN:
        return FrameVerdict(
            usable=False,
            reason=(f"这一帧是死平的（去趋势后残差只有原始起伏的 {std / ptp:.2e}，"
                    f"低于 {FLAT_RATIO_MIN:.0e} —— 那是浮点舍入，不是形貌）"
                    f"——不是针尖的问题，是数据的问题。"
                    "换一块地方重扫一张再判；不要据此去修针尖。"),
            corrugation_rms_m=std, detrended=h)

    return FrameVerdict(usable=True, reason="", corrugation_rms_m=std, detrended=h)


def acquired_row_mask(*frames: Any) -> "np.ndarray":
    """返回全部输入帧共同拥有的完整已采集行掩膜。

    原始 .sxm 中未采集行可为 NaN，Scan_FrameDataGrab 的未写入缓冲可为全零；
    两种形式都排除。这里只接受原始扫描帧，不应用于零值有物理意义的变换结果。

    每行须全部有限，而非只含任意有限值；正在写入的部分行会污染后续平面或
    直线拟合。多帧输入取交集，确保正反扫等逐点比较使用两侧共同的数据。

    裁行属于预处理，会改变后续统计量。报告应说明是否裁行，不把未采集区域
    参与计算造成的 NaN 或低分误写成针尖不合格。
    """
    masks = []
    for fr in frames:
        a = np.asarray(fr)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.size == 0:
            return np.zeros(0, dtype=bool)
        finite = np.isfinite(a).all(axis=1)
        # 全零行 = 活体缓冲里「还没写过」的那一片(见上面那张表)。
        #
        # ⚠️ **只在有非零行作对照时才这么读。** 整帧全零是另一件事:反馈关掉/
        # 没接上时「**测出来就是零**」—— 而「测出来是零」和「没测」必须是两句话
        # (``_domain_synth.dead_flat_frame`` 的 docstring 原话,
        #  ``test_dead_flat_is_undetermined`` 钉着它)。
        # 一整帧零里没有任何东西能把这两者分开,所以那时不裁,让下游按
        # ``dead_flat`` 去说 —— 那句话比「什么都没扫到」更接近现场。
        zero_row = (np.nan_to_num(a) == 0.0).all(axis=1)
        if zero_row.any() and not zero_row.all():
            finite = finite & ~zero_row
        masks.append(finite)
    if not masks:
        return np.zeros(0, dtype=bool)
    n = min(m.size for m in masks)
    out = np.ones(n, dtype=bool)
    for m in masks:
        out &= m[:n]
    return out


__all__ = ["DEAD_FLAT_REASON", "FLAT_RATIO_MIN", "FrameVerdict", "judge_frame",
           "acquired_row_mask"]
