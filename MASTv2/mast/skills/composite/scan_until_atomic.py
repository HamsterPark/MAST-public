# -*- coding: utf-8 -*-
"""按预算重复扫描并逐帧判断原子分辨。

保留当前工作点，可按配置移动到未被扰动区域；像素尺度必须支持所选判据。
扫描失败不存盘、不判定；路径重复或读数重复不计作新样本；判据失败不等于原子相缺失。
报告区分 attempts 与 frames_judged。使用 FullScan 内部等待和完整帧检查，避免复用终态进度。"""
from __future__ import annotations

import logging
from typing import Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)

__all__ = ["ScanUntilAtomicResolution", "JUDGED_VERDICTS"]

#: history 里**对一帧图做出了判定**的那几个 verdict。
#:
#: 与之相对的是记账行:``relocated``(换了地方)、``no_file``(没存出文件)、
#: ``no_scan``(这一次根本没采到)、``no_new_frame``(交回来的还是判过的那一张)。
#: 把它们当成「扫了一帧」正是 2026-08-24 那个 bug 的形状。
#:
#: ⚠️ **单一真源。**``achieve_atomic`` 的两道分诊读的也是这三个词,那边从这里
#: import —— 抄第二份的话,这边加一个 verdict 而那边不知道,后果是「没采到」
#: 被当成针尖的证据去修针。
JUDGED_VERDICTS: frozenset = frozenset({"atomic", "absent", "undetermined"})


def _read_pixels(ctx) -> "int | None":
    """当前扫描缓冲的像素数。读不到就返回 None —— **不猜 256**。"""
    try:
        if ctx is None:
            return None
        d = getattr(ctx.run("GetScanBuffer", {}), "data", None) or {}
        n = d.get("pixels")
        return int(n) if isinstance(n, (int, float)) and n > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("读扫描缓冲失败: %s", exc)
        return None


def _read_linear_speed(ctx) -> "float | None":
    """当前的前向线速度（m/s）。读不到返回 None ⇒ 退回档位表。"""
    try:
        if ctx is None:
            return None
        d = getattr(ctx.run("GetScanSpeed", {}), "data", None) or {}
        v = d.get("fwd_speed_m_s")
        return float(v) if isinstance(v, (int, float)) and v > 0 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("读扫描速度失败: %s", exc)
        return None


def _shrink_into_full_gate(size_m: float, pixels: int) -> "float | None":
    """把视场缩到 nm/px 进 full 档；已经在档里、或缩不动就返回 None。

    档位边界取自 :data:`mast.vision.atomic_phase.SCALE_FULL_NMPP` —— **不抄一个
    0.02 进来**。判据和成帧只能有一个真源，抄一份就是两处各漂各的。

    缩到刚好压线是不够的（浮点往返会把 0.02 变成 0.020000000000000004，
    而闸门是**严格小于**），所以留一点余量。
    """
    from mast.vision.atomic_phase import SCALE_FULL_NMPP

    if not (size_m > 0 and pixels > 0):
        return None
    nmpp = (size_m * 1e9) / pixels
    if nmpp < SCALE_FULL_NMPP:
        return None                       # 已经在 full 档里
    target = SCALE_FULL_NMPP * 0.95
    new_size = target * pixels * 1e-9
    # 缩得太狠就不缩了：一个比晶格常数还小的视场上没有可判的东西。
    if new_size < 1e-9:
        return None
    return new_size


class ScanUntilAtomicResolution(CompositeSkillGraph):
    """反复扫同一块区域，直到判据认定出现了原子分辨。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ScanUntilAtomicResolution",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "反复扫同一区域并逐帧判定，直到出现原子分辨或次数用完。"
                "不改参数、不动针尖 —— 这是代价最低的那一档，先试它。"
                "**这是阶梯里的一档，不是入口。**用户只说「我想要原子分辨」"
                "而没有指名要重扫时，调 AchieveAtomicResolution —— 它会先量"
                "现状、决定从哪一档起步，并在这一档不够时自己升级。"
            ),
            parameters=[
                ParameterSpec(
                    name="max_attempts", type="int", required=False, default=6,
                    min_value=1, max_value=50,
                    description="最多扫几帧",
                ),
                ParameterSpec(
                    name="center_x_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 X；不填就用当前扫描框",
                ),
                ParameterSpec(
                    name="center_y_m", type="float", unit="m", required=False,
                    default=None, description="扫描中心 Y；不填就用当前扫描框",
                ),
                ParameterSpec(
                    name="scan_size_m", type="float", unit="m", required=False,
                    default=None, min_value=1e-10, max_value=1e-6,
                    description="视场边长；不填就用当前扫描框",
                ),
                ParameterSpec(
                    name="line_time_s", type="float", unit="s", required=False,
                    default=None, min_value=1e-4, max_value=600.0,
                    description="每线时间；不填走出厂档位表（8 nm → atomic 档 1.2 s）",
                ),
                ParameterSpec(
                    name="concentration_min", type="float", required=False,
                    default=60.0, min_value=1.0, max_value=1e5,
                    description=("角向集中度下限，仅用于粗筛；"
                                 "完整判定还必须通过 peaks_not_one_lattice 检查。"),
                ),
                ParameterSpec(
                    name="require_full_frame", type="bool", required=False,
                    default=True,
                    description=("残帧不参与判定。**默认 True**：残帧上 NaN 区域会被"
                                 "当成常数，在谱上造出与扫描无关的结构，"
                                 "而判据照样给分"),
                ),
                ParameterSpec(
                    name="allow_reduced_scale", type="bool", required=False,
                    default=False,
                    description=("允许「采样不足但周期清楚」的帧算作原子分辨。"
                                 "8 nm/256 px = 0.031 nm/px 时每周期只有 7 个点，"
                                 "判据默认判 undetermined —— 那对「成像质量」是对的，"
                                 "但对「晶格测量」过严：FFT 峰位精度取决于视场里有"
                                 "多少个周期，不取决于每周期多少像素。做定标时开它"),
                ),
                ParameterSpec(
                    name="keep_current_speed", type="bool", required=False,
                    default=True,
                    description=("按仪器**当前**的线速度算每线时间，而不是走档位表。"
                                 "档位表给 5 nm 帧 4 s/线 = 34 分钟一帧；这台机器"
                                 "实际出过原子分辨的是 6.51 nm/s = 0.77 s/线。"
                                 "显式给了 line_time_s 时这一项不起作用"),
                ),
                ParameterSpec(
                    name="prefer_full_scale_gate", type="bool", required=False,
                    default=True,
                    description=("必要时缩小视场，让 nm/px 进判据的 full 档"
                                 "（< 0.02）。8 nm/256 px = 0.0313 落在过渡带里，"
                                 "判据只会给 undetermined；缩到 5 nm 就进 full 档，"
                                 "结论不用开逃生门。显式给了 scan_size_m 时不动"),
                ),
                ParameterSpec(
                    name="relocate_after_attempts", type="int", required=False,
                    default=0, min_value=0, max_value=50,
                    description=("连续这么多帧没出原子分辨就**换一块地方**"
                                 "（0 = 从不换，保持「原地多扫几帧」的原意）。"
                                 "扎过针/打过脉冲之后原地重扫多少帧都还是那个坑，"
                                 "这时候要换地方"),
                ),
                ParameterSpec(
                    name="relocate_step_m", type="float", unit="m", required=False,
                    default=3e-8, min_value=1e-9, max_value=1e-6,
                    description="每次换区沿对角线移动的距离，默认 30 nm，可按任务配置",
                ),
                ParameterSpec(
                    name="save_every_frame", type="bool", required=False,
                    default=True,
                    description="每帧都存盘（失败的帧也是数据 —— 它们是「没出来」的证据）",
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=1800.0,
            composition_level=3,
            tags=["atomic", "scan", "retry", "composite"],
        )

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        n_max = int(params.get("max_attempts") or 6)
        # 集中度只用于粗筛，不能独立区分晶格与其他周期结构。
        # 最终检查还使用起伏与一阶峰半径散布，避免用单个阈值代替形貌证据。
        conc_min = float(params.get("concentration_min") or 60.0)
        require_full = bool(params.get("require_full_frame", True))
        save_every = bool(params.get("save_every_frame", True))
        allow_reduced = bool(params.get("allow_reduced_scale", False))
        keep_speed = bool(params.get("keep_current_speed", True))
        want_full_gate = bool(params.get("prefer_full_scale_gate", True))
        reloc_after = int(params.get("relocate_after_attempts") or 0)
        reloc_step = float(params.get("relocate_step_m") or 3e-8)

        # 扫描框：不填就问仪器。**不假设一个** —— 假设出来的那个框多半不是
        # 用户正在看的地方（``OptimizeResolution_BO`` 就这么扫了半年原点）。
        cx = params.get("center_x_m")
        cy = params.get("center_y_m")
        size = params.get("scan_size_m")
        if cx is None or cy is None or size is None:
            fd = {}
            try:
                ctx = getattr(executor, "context", None)
                if ctx is not None:
                    fr = ctx.run("GetScanFrame", {})
                    fd = getattr(fr, "data", None) or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("ScanUntilAtomicResolution: 读扫描框失败: %s", exc)
            cx = cx if cx is not None else fd.get("center_x_m")
            cy = cy if cy is not None else fd.get("center_y_m")
            size = size if size is not None else fd.get("width_m")
        if cx is None or cy is None or not size:
            executor.set_partial("aborted_reason", "no_scan_frame")
            logger.error("ScanUntilAtomicResolution: 读不到扫描框，且调用方没给 —— "
                         "**不猜一个框去扫**。")
            return

        # ── 成帧计划：像素尺度进 full 档、线速度跟仪器 ────────────────
        plan_notes: list[str] = []
        ctx = getattr(executor, "context", None)
        pixels = _read_pixels(ctx)
        if want_full_gate and params.get("scan_size_m") is None and pixels:
            new_size = _shrink_into_full_gate(float(size), int(pixels))
            if new_size is not None:
                plan_notes.append(
                    "视场 %.1f → %.1f nm，让 nm/px 从 %.4f 降到 %.4f（进 full 档）"
                    % (size * 1e9, new_size * 1e9,
                       size * 1e9 / pixels, new_size * 1e9 / pixels))
                size = new_size
        line_time = params.get("line_time_s")
        if line_time is None and keep_speed:
            v = _read_linear_speed(ctx)
            if v and v > 0:
                line_time = float(size) / float(v)
                plan_notes.append(
                    "每线 %.3f s（按仪器当前线速度 %.3g m/s 算，不走档位表）"
                    % (line_time, v))
        for _n in plan_notes:
            logger.info("ScanUntilAtomicResolution 成帧计划：%s", _n)
        executor.set_partial("frame_plan", list(plan_notes))
        executor.set_partial("planned_size_m", float(size))
        if line_time is not None:
            executor.set_partial("planned_line_time_s", float(line_time))

        done = int(executor.progress.partial_data.get("attempts_done", 0))
        history = list(executor.progress.partial_data.get("history", []))
        executor.set_total_steps(4 * n_max)

        # 每次 attempt 都必须对应独立采集；新文件名或新时间戳不能证明缓冲已更新。
        seen_paths = {str(h["path"]) for h in history if h.get("path")}
        n_scan_ok = int(executor.progress.partial_data.get("scans_ok", 0))
        n_scan_fail = int(executor.progress.partial_data.get("scans_failed", 0))
        n_repeat = int(executor.progress.partial_data.get("repeat_frames", 0))
        #: 采到了帧、判据却没跑成的次数。与上面三个分开:这不是「没采到」,
        #: 也不是「没有原子分辨」—— 是这一帧**还没有判定**。
        n_assess_fail = int(executor.progress.partial_data.get("assess_failed", 0))

        for i in range(done, n_max):
            tag = "try_%d" % (i + 1)
            scan_params = {
                "center_x_m": float(cx), "center_y_m": float(cy),
                "width_m": float(size), "height_m": float(size),
            }
            if line_time is not None:
                scan_params["line_time_s"] = float(line_time)

            yield CompositeStep(
                step_id=f"{tag}:scan", skill_name="FullScan",
                params=scan_params, optional=True, checkpoint_after=False,
                tags=("attempt", f"attempt={i + 1}", "scan"),
            )
            if executor.progress.aborted:
                return

            # 扫描步骤失败时不保存旧缓冲，也不进行本次判定。
            scan_res = executor.sub_results.get(f"{tag}:scan")
            if scan_res is None:
                why = str(executor.progress.failed_reasons.get(f"{tag}:scan")
                          or "扫描这一步没有留下结果（失败或被断点跳过）")
                n_scan_fail += 1
                # ⚠️ verdict 用 ``no_scan``,**不是** ``absent``/``undetermined``。
                # 「没扫成」既不是「没有原子分辨」也不是「判不了」—— 后两者都是
                # 关于**这一帧**的判定,而这里根本没有这一帧。上层
                # (``achieve_atomic`` 的两道分诊、下面的换地方计数)只认那三个词,
                # 于是这一行既不会被当成针尖的证据,也不会去触发一次换地方。
                history.append({"attempt": i + 1, "verdict": "no_scan",
                                "error": why})
                executor.set_partial("history", list(history))
                executor.set_partial("attempts_done", i + 1)
                executor.set_partial("scans_failed", n_scan_fail)
                logger.warning(
                    "ScanUntilAtomicResolution %d/%d: **这一次没有采集** —— %s。"
                    "不存盘、不判定（存盘会把上一帧当成新帧存出去）",
                    i + 1, n_max, why)
                continue
            n_scan_ok += 1
            executor.set_partial("scans_ok", n_scan_ok)
            _sdat = getattr(scan_res, "data", None) or {}

            # **FullScan 不存盘** —— 它只 Configure/SetSpeed/Start/Wait。
            # 要判定就要有文件，所以存盘这一步归本技能自己发。
            if save_every:
                yield CompositeStep(
                    step_id=f"{tag}:save", skill_name="SaveScan", params={},
                    optional=True, checkpoint_after=False,
                    tags=("attempt", f"attempt={i + 1}", "save"),
                )
                if executor.progress.aborted:
                    return

            save_res = executor.sub_results.get(f"{tag}:save")
            path = (getattr(save_res, "data", None) or {}).get("saved_path")
            if not path:
                yield CompositeStep(
                    step_id=f"{tag}:latest", skill_name="GetLatestScanFile",
                    params={}, optional=True, checkpoint_after=False,
                    tags=("attempt", f"attempt={i + 1}", "latest"),
                )
                lr = executor.sub_results.get(f"{tag}:latest")
                path = (getattr(lr, "data", None) or {}).get("path")
            if not path:
                # 没有文件就没法判 —— 记下来继续，而不是把这一轮算作「没有原子」。
                history.append({"attempt": i + 1, "verdict": "no_file"})
                executor.set_partial("history", list(history))
                executor.set_partial("attempts_done", i + 1)
                continue

            # ── 这张图**判过没有** ────────────────────────────────────────
            #
            # ``GetLatestScanFile`` 那条兜底路径尤其危险:存盘没给出路径时它交出
            # 「最新的一张」—— 而那多半就是**上一次 attempt 存的那一张**。
            # 于是同一帧被判第二次,报告里多出一次「扫了一帧」。
            # 路径重复是**免费而确定**的证据:同一个文件不可能是两次采集。
            if str(path) in seen_paths:
                n_repeat += 1
                history.append({"attempt": i + 1, "verdict": "no_new_frame",
                                "path": str(path)})
                executor.set_partial("history", list(history))
                executor.set_partial("attempts_done", i + 1)
                executor.set_partial("repeat_frames", n_repeat)
                logger.warning(
                    "ScanUntilAtomicResolution %d/%d: 交回来的还是 %s —— "
                    "**这一帧已经判过**,不再判第二次", i + 1, n_max, path)
                continue
            seen_paths.add(str(path))

            # 仅传技能实际支持的参数，防止参数校验拒绝整个判定步骤。
            assess_params = {"scan_path": str(path),
                             "concentration_min": conc_min,
                             "allow_reduced_scale": allow_reduced}
            yield CompositeStep(
                step_id=f"{tag}:assess", skill_name="AssessAtomicResolution",
                params=assess_params, optional=True, checkpoint_after=True,
                tags=("attempt", f"attempt={i + 1}", "assess"),
            )
            if executor.progress.aborted:
                return

            ar = executor.sub_results.get(f"{tag}:assess")
            if ar is None:
                # ── 判据这一步**根本没跑成** ⇒ 这不是「没有原子分辨」 ────────
                #
                # 判定步同样是 ``optional=True``,而下面那行 ``or ... else "absent"``
                # 会把一个空 ``data`` 折叠成一条 ``absent`` —— 「读不到」被念成
                # 「量到了,没有」,而 ``absent`` 是**关于针尖的证据**:上层据此
                # 升级去修针。同一个形状本仓一天里出现过五次,而它就在上面那个
                # bug 往下三行。
                why = str(executor.progress.failed_reasons.get(f"{tag}:assess")
                          or "判定这一步没有留下结果")
                n_assess_fail += 1
                history.append({"attempt": i + 1, "verdict": "no_verdict",
                                "path": str(path), "error": why})
                executor.set_partial("history", list(history))
                executor.set_partial("attempts_done", i + 1)
                executor.set_partial("assess_failed", n_assess_fail)
                logger.warning(
                    "ScanUntilAtomicResolution %d/%d: 采到了 %s,但**判据没跑成** "
                    "—— %s。这不是「没有原子分辨」", i + 1, n_max, path, why)
                continue
            ad = getattr(ar, "data", None) or {}
            verdict = ad.get("verdict") or ("atomic" if ad.get("atomic") else "absent")
            # 残帧不许算数：NaN 区域在谱上会造出与扫描无关的结构，而判据照样给分。
            cov = ad.get("coverage")
            if require_full and isinstance(cov, (int, float)) and cov < 0.98:
                verdict = "undetermined"
                ad = dict(ad)
                ad["reasons"] = list(ad.get("reasons") or []) + ["incomplete_frame"]
            # 读数与上一帧完全相同则记录重复，不计作新样本；文件路径不同不能绕过此检查。
            prev = next((h for h in reversed(history)
                         if h.get("verdict") in JUDGED_VERDICTS), None)
            conc_now = ad.get("angular_concentration")
            same_as_prev = bool(
                prev is not None and conc_now is not None
                and prev.get("concentration") == conc_now
                and prev.get("coverage") == ad.get("coverage"))

            row = {
                "attempt": i + 1, "verdict": verdict, "path": str(path),
                "concentration": conc_now,
                "coverage": ad.get("coverage"),
                # 「这一帧是**这一次**采来的」的证据,由 FullScan 自己报。
                # 只记录、不当闸门:它们缺席时(None)什么都推不出来。
                "scan_lines_done": _sdat.get("scan_lines_done"),
                "wait_outcome": _sdat.get("wait_outcome") or None,
            }
            if same_as_prev:
                row["same_readings_as_previous"] = True
                n_repeat += 1
                executor.set_partial("repeat_frames", n_repeat)
                logger.warning(
                    "ScanUntilAtomicResolution %d/%d: 读数与上一帧**逐位相同** "
                    "(集中度 %s、覆盖 %s)—— 这多半是同一块数据被存了两份,"
                    "不是第二次采集", i + 1, n_max, conc_now, ad.get("coverage"))
            history.append(row)
            executor.set_partial("history", list(history))
            executor.set_partial("attempts_done", i + 1)
            logger.info("ScanUntilAtomicResolution %d/%d: %s (集中度 %s)",
                        i + 1, n_max, verdict, ad.get("angular_concentration"))

            if verdict == "atomic":
                executor.set_partial("found_at", i + 1)
                executor.set_partial("found_path", str(path))
                return

            # 连续未找到原子分辨时可换到未被处理扰动的位置，再按预算复评。
            if reloc_after:
                n_absent = sum(1 for h in history[-reloc_after:]
                               if h.get("verdict") == "absent")
                if n_absent >= reloc_after:
                    cx = float(cx) + reloc_step
                    cy = float(cy) + reloc_step
                    history.append({"attempt": i + 1, "verdict": "relocated",
                                    "center_x_m": cx, "center_y_m": cy})
                    executor.set_partial("history", list(history))
                    executor.set_partial("relocations",
                                         int(executor.progress.partial_data
                                             .get("relocations", 0)) + 1)
                    logger.info("ScanUntilAtomicResolution: 连续 %d 帧判为没有 —— "
                                "换到 (%.4g, %.4g)", reloc_after, cx, cy)

        executor.set_partial("exhausted", True)

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        history = list(pd.get("history", []))
        found = pd.get("found_at")
        n_undet = sum(1 for h in history if h.get("verdict") == "undetermined")
        # 报告区分循环次数与真正判过的独立帧数。
        judged = [h for h in history if h.get("verdict") in JUDGED_VERDICTS]
        n_fresh = sum(1 for h in judged if not h.get("same_readings_as_previous"))
        out = {
            "found": bool(found),
            "found_at_attempt": found,
            "found_path": pd.get("found_path"),
            "attempts": int(pd.get("attempts_done", 0)),
            "history": history,
            "n_undetermined": n_undet,
            # ── 「N 次 attempt = N 次真采集」的账 ────────────────────────
            # ``attempts`` 是循环圈数,回答不了「到底采了几帧」。这四个数才是:
            #   scans_ok       扫描这一步真的成功了几次
            #   scans_failed   没成/没执行几次(那几次**没有存盘、没有判定**)
            #   repeat_frames  报了成功却没给出新样本几次(同一路径,或读数逐位相同)
            #   frames_judged  真正判过的、互不相同的帧数 ← 报告里该说的那个数
            "scans_ok": int(pd.get("scans_ok", 0)),
            "scans_failed": int(pd.get("scans_failed", 0)),
            "repeat_frames": int(pd.get("repeat_frames", 0)),
            #   assess_failed  采到了帧、判据却没跑成几次(**不是**「没有原子分辨」)
            "assess_failed": int(pd.get("assess_failed", 0)),
            "frames_judged": n_fresh,
            # 这次**实际**用的成帧参数（不是调用方传的那些）——
            # 「为什么这帧是 5 nm 不是 8 nm」必须能事后回答。
            "frame_plan": list(pd.get("frame_plan") or []),
            "planned_size_m": pd.get("planned_size_m"),
            "planned_line_time_s": pd.get("planned_line_time_s"),
            "relocations": int(pd.get("relocations", 0)),
        }
        if not found:
            # 「这一次 attempt 什么样本都没拿到」的次数。它和「没有原子分辨」
            # 是两回事,而以前两者在报告里长得一模一样。
            n_nosample = out["scans_failed"] + out["repeat_frames"]
            if pd.get("aborted_reason") == "no_scan_frame":
                out["advice"] = "读不到扫描框，一帧都没扫。给 center_x_m/center_y_m/scan_size_m。"
            elif not judged and out["assess_failed"]:
                # 采到了帧,但**一帧都没判成**。这同样不是关于针尖的证据。
                out["advice"] = (
                    "采到了 %d 帧,但**判据一次都没跑成**（%d 次）—— "
                    "所以这里**没有任何关于针尖的证据**，别据此去修针。"
                    "先查判定这一步为什么失败（文件读不了？依赖缺了？），"
                    "那几帧还在盘上，修好之后可以直接重判。"
                    % (out["scans_ok"], out["assess_failed"]))
            elif not judged and n_nosample:
                # **一帧都没采到。**这不是关于针尖的任何证据 —— 说成「没有原子
                # 分辨」会让上层去修一根可能本来就好的针。
                out["advice"] = (
                    "**一帧都没有真的采到**（扫描失败 %d 次、交回已判过的帧 %d 次），"
                    "所以这里**没有任何关于针尖的证据** —— 别据此去修针。"
                    "先查扫描这一步为什么没成（超时？被停？落点被 crash_guard 拒了？），"
                    "再谈原子分辨。" % (out["scans_failed"], out["repeat_frames"]))
            elif n_undet and n_undet == len(judged):
                # 全是「判不了」时说「没有原子分辨」是错的 —— 那是采集没完成。
                #
                # ⚠️ 分母是**判过的行数**,不是 ``len(history)``：history 里还有
                # ``relocated`` / ``no_scan`` / ``no_new_frame`` 这些记账行,
                # 拿它当分母时只要发生过一次换地方,这条判定就永远不成立 ——
                # 一道**看着在防护、其实从没触发过**的分支。
                out["advice"] = (
                    "%d 帧全部**判不了**（多半是残帧）。这不是「没有原子分辨」，"
                    "是每一帧都没扫完 —— 先查扫描为什么中断，再谈针尖。" % n_undet)
            else:
                tail = ""
                if not out["relocations"]:
                    tail = ("这一轮未换区。可设置 relocate_after_attempts=2，"
                            "移到未被处理扰动的位置后重新评估。")
                if n_nosample:
                    tail += (
                        "⚠️ 这一轮有 **%d 次 attempt 没有产出新的一帧**"
                        "（扫描失败 %d 次、没给出新样本 %d 次）—— 下面那个帧数是"
                        "**真判过的互不相同的帧**，不是 attempt 数。"
                        "这一档的全部理由是「针尖构型会自发变化，这一帧没有不代表"
                        "下一帧没有」，判同一帧两次兑现不了那条理由。"
                        % (n_nosample, out["scans_failed"], out["repeat_frames"]))
                out["advice"] = (
                    "扫了 %d 帧仍无原子分辨。%s下一档代价：OptimizeResolution_BO "
                    "（搜索 bias/setpoint，会离开当前工作点）；再下一档才是修针。"
                    % (out["frames_judged"], tail))
        return out
