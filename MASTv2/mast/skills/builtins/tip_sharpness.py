"""从一张扫描图判针尖尖不尖 —— 台阶边缘有多陡。

用于修针收尾的图像判据:尖针尖看到的台阶(尤其是垂直快扫方向的台阶)
应当接近阶跃函数,但不会与阶跃函数完全一致。

量化它的是 10-90 上升宽度(``edge_resolution``):尖针尖几个像素就把台阶走完,钝
针尖或双针尖把它抹开。判据本体在 ``mast.vision.tip_metrics`` 里早就有了 —— 只是
一直没有技能把它拿出来,于是「台阶够不够陡」这件事只能靠人看图。

**没有清晰台阶时报 None,不报数**。平坦区上的「边缘宽度」是在测噪声,一个编出来
的数字会让流程以为自己验收过了。三态(尖 / 不够尖 / 这张图上判不了)必须能分辨。

## ``no_step`` 的两种含义 —— 读结果的人必须知道

判据认的是「相干的梯度离群点」:台阶边缘处的梯度要显著高于图上典型的梯度
(``gmax/gmed >= 6``)。合成扫描测试(128 px 帧):

  * 对噪声很稳健 —— 噪声从台阶高度的 0.5% 加到 5%,结论几乎不变。
  * 对**台阶密度**敏感 —— 2 级台阶的帧上边缘宽到 2 px 仍测得出;4 级台阶的帧上
    超过 1 px 就测不出了。边缘像素占比一高,「离群」本身就不成立。
  * 对**边缘宽度**敏感 —— 这正是要测的量,但宽到 4 px 以上就彻底判不了。

所以 ``no_step`` 既可能是「这块地方本来就没有台阶」,**也可能是「针尖钝到把台阶
抹平了」**。两者在这一个数字上无法区分:要分辨,看同一帧的 ``fft_sharpness`` 和
``corrugation_rms_m`` —— 真有台阶而针尖钝,起伏仍在,只是边缘糊了。**绝不能把
``no_step`` 当作「针尖没问题」**。
"""
from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

#: `AssessTipSharpness` 能返回的**全部** verdict。调用方必须对每一个都表态。
#:
#: 这个常量存在的理由:2026-08-11 之前 `ForgeAuTip` 的验收写的是
#: ``verdict in ("good","sharp","ok","pass")`` —— 一个**白名单**,于是新增/未预料
#: 的状态自动落进「不合格」那一侧。`"measured"`(量到了但没给阈值)就是这样被当成
#: 不合格的,而该流程从不传阈值 ⇒ 验收结构上不可能通过。
#:
#: 白名单读起来像在防护,其实它把「没想到的情况」默默判成了失败。
#: 有了这份词汇表,`tests/.../test_sharpness_verdict_coverage.py` 才能钉住
#: 「每个状态都被显式分类」,新增一个状态而忘了教调用方时测试会红。
SHARPNESS_VERDICTS: tuple[str, ...] = (
    "no_step",      # 图里没有台阶 —— 判不了
    "measured",     # 量到了,但调用方没给阈值 —— 判不了
    "unresolved",   # 阈值细过这张图的采样极限 —— 判不了
    "sharp",        # 够尖
    "blunt",        # 不够尖
)

logger = logging.getLogger(__name__)


class AssessTipSharpness(BaseSkill):
    """台阶边缘的锐利程度 + 正反扫描线稳定度,从一张 .sxm 算出来。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessTipSharpness",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "针尖有多锐，从一张已保存的 .sxm 上量：最陡台阶边缘的 10-90 上升宽度（锐针尖几个 px "
                "就能分开一个台阶；钝针或双针会把它抹开），外加正/反扫不稳定度与 FFT 锐度。只读。"
                "画面里没有清晰台阶时 edge_resolution_nm 为 null —— 那表示「这张图判不出来」，"
                "**不是**「锐」。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description="要分析的 .sxm 文件路径。",
                    required=True),
                ParameterSpec(
                    name="channel", type="str",
                    description="形貌通道（标准是 'Z'）。",
                    required=False, default="Z"),
                ParameterSpec(
                    name="sharp_edge_nm", type="float", unit="nm",
                    description="边缘宽度到这个值或更小，就算针尖锐。留空则只报数、不下结论 —— "
                                "合适的阈值取决于像素尺寸和表面。",
                    required=False, min_value=0.01, max_value=100.0),
            ],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["tip", "sharpness", "step", "analysis", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        scan_path = str(params["scan_path"])
        channel_name = params.get("channel") or "Z"

        if not Path(scan_path).exists():
            return SkillResult(skill_name="AssessTipSharpness", success=False,
                               error=f"文件不存在: {scan_path}")
        try:
            import numpy as np

            from mast.io.mosaic import parse_xy_meta
            from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
            from mast.vision.frame_validity import judge_frame
            from mast.vision.tip_metrics import (
                _edge_resolution,
                _fft_sharpness,
                _fwd_bwd_instability,
            )
        except ImportError as exc:
            return SkillResult(skill_name="AssessTipSharpness", success=False,
                               error=f"缺依赖: {exc}")

        try:
            scan = read_sxm(scan_path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="AssessTipSharpness", success=False,
                               error=f".sxm 读取失败: {exc}")

        # 使用 sxm_oriented_frames 统一几何归位，再比较正反扫。
        # 反扫原始数组沿快轴镜像，直接比较会把坐标差异当成不稳定，
        # 也可能因近似镜像对称的结构产生虚假一致性。
        oriented = sxm_oriented_frames(scan, channel_name)
        if oriented.get("forward") is None:
            # 指名的通道没有 → 退回文件里的第一个通道（旧行为，原样保留）。
            channels = scan.get("channels", {}) or {}
            first = next(iter(channels), None)
            if first is None:
                return SkillResult(
                    skill_name="AssessTipSharpness", success=False,
                    error=f"文件里没有可用通道(要的是 {channel_name!r})")
            oriented = sxm_oriented_frames(scan, first)
        fwd, bwd = oriented.get("forward"), oriented.get("backward")
        if fwd is None:
            return SkillResult(skill_name="AssessTipSharpness", success=False,
                               error="通道里没有正扫/反扫数据")

        arr = np.asarray(fwd, dtype=np.float64)
        bwd_arr = (np.asarray(bwd, dtype=np.float64)
                   if bwd is not None and np.asarray(bwd).shape == arr.shape
                   else None)

        nm_per_px = None
        try:
            meta = parse_xy_meta(scan.get("header", {}) or {})
            if meta and arr.shape[1]:
                nm_per_px = (float(meta["w"]) / arr.shape[1]) * 1e9
        except Exception:  # noqa: BLE001
            nm_per_px = None

        # 复用 vision 层判据，但在此入口使用适合当前图像尺度的预处理。
        # 不能让另一条判据链中的幅度早退条件隐式筛掉此处需要分析的细微结构。
        # 前置有效性由 mast.vision.frame_validity 共用，避免多个入口各自判断而产生分歧。
        frame = judge_frame(arr)
        if not frame.usable:
            return SkillResult(
                skill_name="AssessTipSharpness", success=False,
                error=frame.reason,
                data=frame.meta(scan_path=scan_path, nm_per_px=nm_per_px,
                                rows=int(arr.shape[0]), cols=int(arr.shape[1])))
        h, std = frame.detrended, frame.corrugation_rms_m
        hn = h / std

        edge_px, edge_nm = _edge_resolution(hn, nm_per_px)
        sharp, _res_nm, has_lat = _fft_sharpness(hn, nm_per_px)
        instab = (_fwd_bwd_instability(arr, bwd_arr)
                  if bwd_arr is not None else None)

        data = {
            "scan_path": scan_path,
            "edge_resolution_nm": edge_nm,
            "edge_resolution_px": edge_px,
            "fwd_bwd_instability": instab,
            "fft_sharpness": float(sharp),
            "has_lattice": bool(has_lat),
            "corrugation_rms_m": std,
            "nm_per_px": nm_per_px,
            "has_step": edge_nm is not None or edge_px is not None,
        }

        thr = params.get("sharp_edge_nm")
        if edge_nm is None:
            # 平坦帧上量「边缘宽度」是在量噪声 —— 判不了就说判不了。
            data["verdict"] = "no_step"
            summary = ("这张图里没有清晰台阶,判不了针尖锐利度 —— "
                       "换一块有台阶的地方再扫一张。")
        elif thr is None:
            data["verdict"] = "measured"
            summary = f"台阶边缘 10-90 宽度 {edge_nm:.2f} nm（未给判定阈值）"
        else:
            # 判定阈值必须处于当前采样能够分辨的尺度。
            # 10–90 宽度阈值低于两个采样点时，判别主要反映像素栅格，不能据此验收针尖。
            # 无法判定时报告所需的像素尺度，让调用方取得适合判定的新图。
            floor_nm = (2.0 * float(nm_per_px)) if nm_per_px else None
            if floor_nm is not None and float(thr) < floor_nm:
                data["verdict"] = "unresolved"
                data["sharp_edge_nm"] = float(thr)
                data["sampling_floor_nm"] = floor_nm
                summary = (
                    f"判不了(不是不合格):阈值 {float(thr):.2f} nm 比这张图的采样"
                    f"极限 {floor_nm:.2f} nm 还小({nm_per_px:.3f} nm/px × 2)。"
                    f"量到的 {edge_nm:.2f} nm 只反映像素大小。"
                    f"把验收图扫细到 **≤ {float(thr) / 2:.3f} nm/px**"
                    f"(同视野加像素,或缩小视野)再判。")
            else:
                sharp = float(edge_nm) <= float(thr)
                data["verdict"] = "sharp" if sharp else "blunt"
                data["sharp_edge_nm"] = float(thr)
                summary = (f"台阶边缘 {edge_nm:.2f} nm "
                           f"{'≤' if sharp else '>'} 阈值 {float(thr):.2f} nm — "
                           f"{'够尖' if sharp else '还不够尖'}")

        return SkillResult(skill_name="AssessTipSharpness", success=True,
                           data=data, summary=summary)


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(AssessTipSharpness, context_provider)
