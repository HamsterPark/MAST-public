"""观察当前扫描帧而不停扫，提供进度与新采集行的统计量。

报告采集推进、粗糙度、量程、斜率、行间相关和触顶比例，不在此处代替专门的
原子线或撞针判据，也不引入统一的未标定质量阈值。

since_line 是游标：下一次传入返回的 last_line_index，只观察新采集行。
没有新行可能表示停止、卡顿或换行等待，必须由调用方处理，不能反复把同一帧
当成新的测量证据。尚未采集的行需依据帧有效性规则处理。"""
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

logger = logging.getLogger(__name__)

#: Z 触顶的判据：一行里有这么多点贴着该行的极值，就认为压电到头了。
_SATURATION_TOL = 1e-12
#: 行间相关低于这个数就提醒「像是针尖不稳 / 有条纹」。**是提醒不是判定** ——
#: 真实平坦表面上相邻行相关本来就高（0.9+），而一片纯噪声区也可能低而无害。
_CORR_HINT = 0.5


class WatchScanLines(BaseSkill):
    """扫描进行中，看一眼已经扫出来的行 —— 不停扫、不重扫。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="WatchScanLines",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在**不停止**正在运行的扫描的前提下看它一眼：进行到哪儿了、"
                "自某条游标行以来是否还在推进、以及新扫出来的那些行长什么样"
                "（粗糙度、量程、倾斜、行间相关性、piezo 饱和）。扫一帧要花几分钟，"
                "而这个只花一次 TCP 往返 —— 于是「这一帧还值不值得扫完」不再是"
                "一个要等几分钟才有答案的问题。它报的是**测量值**加上被标注出来的"
                "观察 —— 判语归判据类技能（AssessAtomicLines、CheckScanForCrash）。"
                "把 last_line_index 当作游标传回 since_line 即可；"
                "新增 0 行意味着扫描停了、卡住了、或者正扫在一行中间 —— "
                "这是一个必须有人处理的信号。"
            ),
            parameters=[
                ParameterSpec(
                    name="direction", type="int",
                    description="1 = 正扫，0 = 反扫（对应 .sxm 的 forward / backward）。",
                    required=False, default=1, min_value=0, max_value=1),
                ParameterSpec(
                    name="since_line", type="int",
                    description=("游标：只统计行号比它「更新」的那些行。-1 = 全部已扫出的行。"
                                 "把上一次回包里的 last_line_index 传回来即可。"),
                    required=False, default=-1, min_value=-1, max_value=100000),
                ParameterSpec(
                    name="channel_index", type="int",
                    description=("强行指定**信号索引**（不是缓冲位）。-1 = 自己去 "
                                 "Scan_BufferGet 找 Z。"),
                    required=False, default=-1, min_value=-1, max_value=127),
                ParameterSpec(
                    name="max_lines", type="int",
                    description="最多统计这么多条最新的行（省算力）。0 = 不限。",
                    required=False, default=64, min_value=0, max_value=4096),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["scan", "read", "live", "lines", "infrastructure"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np

        from mast.skills.builtins._scan_readout import (
            grab_frame, resolve_readout,
        )
        from mast.vision.atomic_lines import usable_rows

        direction = int(params.get("direction", 1))
        since = int(params.get("since_line", -1))
        forced = int(params.get("channel_index", -1))
        max_lines = int(params.get("max_lines", 64) or 0)

        calls: list = []
        ro = resolve_readout(context, calls, forced_channel=forced)
        if not ro.ok:
            return SkillResult(skill_name="WatchScanLines", success=False,
                               error=ro.why, data=ro.as_dict(), nanonis_calls=calls)

        img, why = grab_frame(context, ro.z_index, direction, calls)
        if img is None:
            return SkillResult(
                skill_name="WatchScanLines", success=False,
                error=("取不到通道 %d 的帧：%s　（扫描缓冲里的通道是 %s —— "
                       "这个参数要的是**信号索引**，不是缓冲位）"
                       % (ro.z_index, why, ro.channels)),
                data=ro.as_dict(), nanonis_calls=calls)

        a = np.asarray(img, dtype=float)
        if a.ndim != 2:
            return SkillResult(skill_name="WatchScanLines", success=False,
                               error="帧不是二维：shape=%s" % (a.shape,),
                               data=ro.as_dict(), nanonis_calls=calls)

        filled = usable_rows(a)
        idx = np.flatnonzero(filled)
        n_done = int(idx.size)
        n_total = int(a.shape[0])
        data = dict(ro.as_dict())
        data.update({
            "direction": direction,
            "n_lines_done": n_done,
            "n_lines_total": n_total,
            "fraction_done": (n_done / n_total) if n_total else None,
        })

        if n_done == 0:
            data["advancing"] = False
            data["n_lines_new"] = 0
            data["observations"] = [
                "帧缓冲里一行都还没有 —— 扫描可能刚起、或者根本没在扫。"
                "这不等于「表面是平的」。"]
            return SkillResult(skill_name="WatchScanLines", success=True, data=data,
                               summary=data["observations"][0], nanonis_calls=calls)

        # 扫描方向决定新行在哪一头：未扫区在低号侧 ⇒ 新行是 idx 的开头。
        low_unscanned = not bool(filled[0])
        last_line = int(idx.min() if low_unscanned else idx.max())
        data["last_line_index"] = last_line
        data["scan_direction_hint"] = ("行号从大到小填（未扫区在低号侧）"
                                       if low_unscanned else "行号从小到大填")

        if since < 0:
            new_idx = idx
        elif low_unscanned:
            new_idx = idx[idx < since]
        else:
            new_idx = idx[idx > since]
        data["n_lines_new"] = int(new_idx.size)
        data["advancing"] = bool(new_idx.size > 0) if since >= 0 else None

        use = new_idx if new_idx.size else idx
        if max_lines and use.size > max_lines:
            use = use[:max_lines] if low_unscanned else use[-max_lines:]
        rows = a[use]

        # 每行去掉自己的均值再统计 —— 不然整帧的倾斜会盖住行内的起伏
        centred = rows - rows.mean(axis=1, keepdims=True)
        rms = float(np.median(np.sqrt((centred ** 2).mean(axis=1))))
        rng = float(np.median(rows.max(axis=1) - rows.min(axis=1)))
        # 行间相关：稳定表面上相邻行长得像；针尖不稳 / 有条纹时它掉得最快
        corr = None
        if rows.shape[0] >= 2:
            cs = []
            order = np.argsort(use)
            r2 = rows[order]
            for k in range(r2.shape[0] - 1):
                x, y = centred[order][k], centred[order][k + 1]
                sx, sy = float(x.std()), float(y.std())
                if sx > 0 and sy > 0:
                    cs.append(float(np.mean(x * y) / (sx * sy)))
            corr = float(np.median(cs)) if cs else None
        # 触顶：贴着自己极值的点占多少
        hi = rows.max(axis=1, keepdims=True)
        lo = rows.min(axis=1, keepdims=True)
        sat = float(np.mean((np.abs(rows - hi) < _SATURATION_TOL)
                            | (np.abs(rows - lo) < _SATURATION_TOL)))
        # 慢轴斜率：相邻行均值的差 —— 漂移 / 倾斜
        means = rows.mean(axis=1)[np.argsort(use)]
        slope = float(np.median(np.diff(means))) if means.size >= 2 else None

        data.update({
            "n_lines_measured": int(use.size),
            "rms_roughness_m": rms,
            "range_m": rng,
            "line_to_line_corr": corr,
            "saturated_fraction": sat,
            "slope_m_per_line": slope,
        })

        obs: list[str] = []
        if since >= 0 and new_idx.size == 0:
            obs.append(
                "自上次以来**一行新的都没有** —— 扫描停了、卡了，或者正好在换行。"
                "这条要有人接：接着对同一张不再更新的图打分，看到的都是旧的。")
        if corr is not None and corr < _CORR_HINT:
            obs.append("行间相关只有 %.2f（提醒，不是判定）—— 稳定表面上相邻行"
                       "通常 0.9 以上；低到这里常见于针尖不稳或成条纹。" % corr)
        if sat > 0.02:
            obs.append("有 %.1f%% 的点贴着自己那一行的极值 —— 可能压电到头了。" % (100 * sat))
        if not obs:
            obs.append("已扫 %d/%d 行；新行 %s；粗糙度中位 %.3g m、量程 %.3g m。"
                       % (n_done, n_total,
                          ("%d 条" % new_idx.size) if since >= 0 else "（没给游标）",
                          rms, rng))
        data["observations"] = obs

        return SkillResult(skill_name="WatchScanLines", success=True, data=data,
                           summary=obs[0], nanonis_calls=calls)


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return [wrap_skill(WatchScanLines, context_provider)]


__all__ = ["WatchScanLines"]
