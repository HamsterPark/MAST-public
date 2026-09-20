"""Scan-frame grab + crash detection skills — G1 governed wrappers.

P5 (2026-06-26): the declarative composite IR (`SpecComposite`) can only call
REGISTERED skills, but three hand-written composites (FullScan / TrackDrift /
PreScanCheck) ran a raw ``context.safe_call("Scan_FrameDataGrab", ...)`` + inline
numpy directly inside their plan generators — *out of graph*. Nothing else in the
skill tree wrapped ``Scan_FrameDataGrab``. These two governed skills close that
gap so the same work becomes a declarative ``step`` in a JSON composite, WITHOUT
adding a raw-TCP / eval node to the spec layer (the「数据而非代码」rail, R3):

  * ``GrabScanFrameData`` — read one channel's frame samples and persist them to a
    ``.npy``; returns the PATH (never the array — keeps tensors out of the
    checkpointer and the spec variable context). For drift / line-quality flows.
  * ``CheckScanForCrash`` — grab the probe channels itself and apply the
    near-zero-variance / NaN tip-crash test, returning only scalars. A faithful,
    self-contained replacement for ``FullScan._check_scan_data``.
"""
from __future__ import annotations

import logging
import time

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _grab_channel_array(context, channel_index: int, direction: int,
                        *, shape_2d: bool = False):
    """Grab one channel's frame as a float64 ndarray (or ``None``).

    Mirrors the parsing the hand-written composites used: ``Scan.FrameDataGrab``
    returns ``(err, raw_bytes, parsed)`` and ``parsed[2]`` is the sample list.
    Returns ``(array_or_None, record)`` so callers can record the TCP call.

    ``shape_2d`` keeps the ``(rows, cols)`` image shape. Crash detection wants
    the flat form (it only needs the variance); anything that PERSISTS the frame
    wants the image — see :class:`GrabScanFrameData`.
    """
    from mast.io.nanonis_files import parse_frame_grab
    rec = context.safe_call("Scan_FrameDataGrab", channel_index, direction)
    if rec.error or rec.return_value is None:
        return None, rec
    # Shared robust parse: the real Nanonis body is a heterogeneous
    # [name_len, name, rows, cols, data_2D, dir] list — a bare np.asarray on it
    # raised "inhomogeneous shape" on EVERY real scan.
    arr = parse_frame_grab(rec.return_value, shape_2d=shape_2d)
    if arr is None and shape_2d:
        # 2-D is a PREFERENCE, not a precondition: parse_frame_grab returns None
        # when it cannot form an image (a flat body carrying no rows/cols header
        # — stubs, and instruments that answer without it). Re-parse the SAME
        # reply flat rather than issuing a second TCP read; losing samples that
        # arrived fine would be the worse failure.
        arr = parse_frame_grab(rec.return_value)
    return arr, rec


def _frames_dir():
    from mast._runtime_paths import project_root
    d = project_root() / "experiments" / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


class GrabScanFrameData(BaseSkill):
    """Read one scan channel's frame samples and persist them to a ``.npy``."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GrabScanFrameData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "抓取某一路扫描通道的整帧采样（Scan_FrameDataGrab），"
                "存成一个 .npy 文件；返回的是文件**路径**（不是数组本身）。"
                "direction=1 取正扫，0 取反扫。通道 0 是"
                "第一路采集通道（通常是形貌）；通道 14 是 "
                "Z-controller 信号。"
            ),
            parameters=[
                ParameterSpec(
                    name="channel_index", type="int",
                    description="采集通道索引（0 = 第一路 / 形貌）",
                    required=True, min_value=0,
                ),
                ParameterSpec(
                    name="direction", type="int",
                    description="1 = 正扫，0 = 反扫",
                    required=False, default=1, allowed_values=[0, 1],
                ),
                ParameterSpec(
                    name="save_path", type="str",
                    description=("可选：显式指定 .npy 输出路径；默认 = "
                                 "<data>/experiments/frames/"),
                    required=False, default="",
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "frame", "read", "g1"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np
        channel_index = int(params["channel_index"])
        direction = int(params.get("direction", 1))
        save_path = (params.get("save_path") or "").strip()

        # shape_2d=True — this skill PERSISTS the frame, and a persisted frame
        # must keep its image shape.
        #
        # It did not. `parse_frame_grab` ravels to 1-D unless asked otherwise,
        # so every .npy this skill ever wrote was a flat (rows*cols,) array. The
        # rows/cols were right there in the reply body and were discarded on the
        # way to disk, which broke the whole scan→figure→report chain one step
        # after the measurement: `plot_scan` correctly refuses a 1-D array, so
        # the report had no image. Verified end-to-end 2026-07-28 — the run
        # produced a full draft whose Results section reads "**No figure is
        # included in this report.** The saved frame was stored flattened as a
        # 1-D array of shape (65536,) rather than a 2-D (256, 256) image".
        # Every number in that report was real; only the picture was missing.
        arr, rec = _grab_channel_array(context, channel_index, direction,
                                       shape_2d=True)
        calls = [rec] if rec is not None else []
        if arr is None:
            return SkillResult(
                skill_name="GrabScanFrameData", success=False,
                error=(rec.error if rec is not None and rec.error
                       else "Scan_FrameDataGrab returned no usable samples"),
                nanonis_calls=calls,
            )
        try:
            from pathlib import Path
            if save_path:
                out = Path(save_path)
                out.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Run-unique, NOT a fixed name.
                #
                # It was `frame_ch{N}_dir{D}.npy` — one path shared by every run
                # this process ever does. Two consequences, both observed in a
                # real end-to-end run (2026-07-28):
                #   * data_processing resolved that path BEFORE this run's grab
                #     landed and analysed the PREVIOUS run's leftover — it
                #     reported "frame_ch0_dir1.npy 实际为一维 8 元素全零数组"
                #     and sent the whole task back for a re-scan;
                #   * and when the grab did land, it DESTROYED the earlier run's
                #     frame.
                # This repo has learned the same lesson twice already: composite
                # sidecars were keyed by name until a finished AutoApproach's
                # file made every later approach "succeed" instantly, and
                # `_persist_frame_png` is timestamped for exactly the
                # history-falsification reason #78 raised. A shared mutable path
                # is the defect; a guard on top of it is not the fix.
                # A millisecond stamp ALONE does not make the name unique: two
                # grabs issued back to back land in the same millisecond
                # (measured: 2000 consecutive reads of time.time() yield two
                # distinct values), and then the second one overwrites the first
                # — the exact defect this block exists to prevent. Take the
                # first free name instead of trusting the clock.
                stamp = int(time.time() * 1000) & 0xFFFFFFFF
                base = _frames_dir() / f"frame_ch{channel_index}_dir{direction}_{stamp:x}"
                out = base.with_suffix(".npy")
                n = 1
                while out.exists():
                    out = base.with_name(f"{base.name}_{n:02d}").with_suffix(".npy")
                    n += 1
            np.save(out, arr)
            frame_path = str(out)
        except Exception as exc:  # noqa: BLE001 — surface a clean failure
            return SkillResult(
                skill_name="GrabScanFrameData", success=False,
                error=f"failed to save frame .npy: {exc}",
                nanonis_calls=calls,
            )
        return SkillResult(
            skill_name="GrabScanFrameData", success=True,
            data={"frame_path": frame_path, "n_samples": int(arr.size),
                  "shape": list(arr.shape)},
            nanonis_calls=calls,
        )


class LoadScanFrameFromFile(BaseSkill):
    """从保存的 .sxm 读取指定通道正反扫二维数组，分别写入 .npy 并返回路径。
    
    实时缓冲只反映当前缓冲区，不能用于恢复任意历史帧。存盘帧统一通过
    sxm_oriented_frames 取出：反扫按采集顺序保存，需要水平方向对齐；向上扫描
    还需统一行方向。未经对齐的相关性比较可能把时间方向伪影当成相同形貌。
    
    只返回路径，不把数组放入 checkpoint；保持 (rows, cols) 形状，以便现有画图和
    CheckLineQuality 等消费者使用。请求通道不存在时明确失败并列出可用通道，
    不能自动改用不同物理量。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LoadScanFrameFromFile",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从一个**已保存**的 .sxm 文件中，抽出某一路通道的正扫与反扫 "
                "2-D 帧，各写成一个 .npy；返回的是两个**路径**"
                "（绝不返回数组本身）。它是 GrabScanFrameData 的离线对应版 —— "
                "后者只能读实时缓冲。这两个路径可以直接喂给 "
                "CheckLineQuality、或任何接受 .npy 的判据。若该通道或反扫"
                "方向不存在，它会**大声报错** —— 绝不拿另一路通道来顶替。"
            ),
            parameters=[
                ParameterSpec(
                    name="scan_path", type="str",
                    description="已保存的 .sxm 文件的路径",
                    required=True,
                ),
                ParameterSpec(
                    name="channel", type="str",
                    description=("通道名，按 .sxm 头里写的那样，例如 "
                                 "'Z'（形貌）。通道不存在 = 报错，并列出这个"
                                 "文件里实际有哪些通道。"),
                    required=False, default="Z",
                ),
                ParameterSpec(
                    name="save_dir", type="str",
                    description=("可选：输出目录；默认 = "
                                 "<data>/experiments/frames/"),
                    required=False, default="",
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["scan", "frame", "read", "offline", "sxm"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np
        from pathlib import Path

        name = "LoadScanFrameFromFile"
        scan_path = (params.get("scan_path") or "").strip()
        channel = (params.get("channel") or "Z").strip()
        save_dir = (params.get("save_dir") or "").strip()

        if not scan_path:
            return SkillResult(skill_name=name, success=False,
                               error="scan_path 是必填的")
        # 单一入口:.sxm 只经 io.nanonis_files 读(项目规约 的目录边界规矩)。
        try:
            from mast.io.nanonis_files import read_sxm
            parsed = read_sxm(scan_path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f"读不了 {scan_path}: {type(exc).__name__}: {exc}")

        channels = parsed.get("channels") or {}
        if not channels:
            return SkillResult(skill_name=name, success=False,
                               error=f"{scan_path} 里没有任何通道数据")
        ch = channels.get(channel)
        if not isinstance(ch, dict):
            return SkillResult(
                skill_name=name, success=False,
                error=(f"这个 .sxm 里没有通道 {channel!r}。它有:"
                       f"{sorted(channels)} —— 请指名要哪一个。"
                       "(不替你挑:挑错通道量出来的数看起来完全正常,"
                       "只是量的是另一个物理量。)"),
                data={"available_channels": sorted(channels)})

        fwd, bwd = ch.get("forward"), ch.get("backward")
        # 反扫缺失是**一类独立的失败**,不能和「通道不存在」合并:一次只存了正扫的
        # 扫描根本无法回答正反扫一致性,而那正是调用方要问的问题。
        missing = [d for d, v in (("forward", fwd), ("backward", bwd)) if v is None]
        if missing:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"通道 {channel!r} 缺 {'/'.join(missing)} 方向的数据 —— "
                       "这一帧回答不了正反扫一致性(只扫了一个方向的图就是这样)。"),
                data={"channel": channel, "missing_directions": missing})

        # ── 几何归位:反扫镜像 + :SCAN_DIR: up 的行序 ────────────────────────
        #
        # **不在这里自己写 fliplr/flipud。** 上面的 `ch["forward"|"backward"]` 只用来
        # 回答「这两个方向在不在」(那两条错误消息要精确到方向名);真正交出去的
        # 数组一律来自 sxm_oriented_frames —— 同一个几何问题全仓只有那一份答案。
        #
        # 走到这里 fwd/bwd 都非 None,所以那个函数里「只有反扫的通道就把它当正扫
        # 交回」的分支不可能触发,拿回来的一定还是这两个方向本身。
        try:
            from mast.io.nanonis_files import sxm_oriented_frames
            oriented = sxm_oriented_frames(parsed, channel)
            o_fwd, o_bwd = oriented.get("forward"), oriented.get("backward")
        except Exception as exc:  # noqa: BLE001
            return SkillResult(
                skill_name=name, success=False,
                error=(f"几何归位失败({type(exc).__name__}: {exc})—— "
                       "宁可失败也不交未归位的帧:未镜像的反扫会让正反扫一致性"
                       "判据整个反过来(越坏分越高)。"))
        if o_fwd is None or o_bwd is None:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"通道 {channel!r} 几何归位后拿不到正/反扫两个方向 —— "
                       "不交未归位的帧。"))

        f = np.asarray(o_fwd, dtype=np.float64)
        b = np.asarray(o_bwd, dtype=np.float64)
        if f.ndim != 2 or b.ndim != 2:
            return SkillResult(
                skill_name=name, success=False,
                error=(f"正/反扫不是二维: forward{f.shape} backward{b.shape}。"
                       "持久化的帧必须保持图像形状 —— 拍平的 .npy 会让下游作图"
                       "与逐行判据全部失效。"))
        if f.shape != b.shape:
            return SkillResult(
                skill_name=name, success=False,
                error=f"正反扫形状不一致: {f.shape} vs {b.shape},无法逐点比较。")

        try:
            if save_dir:
                out_dir = Path(save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = _frames_dir()
            # 与 GrabScanFrameData 同一条理由:固定名 = 一条被每次运行共享的可变路径,
            # 上一次的帧会被这一次悄悄覆盖,而下游读到的是别人的数据。取第一个空位。
            stem = Path(scan_path).stem
            safe_ch = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel)
            stamp = int(time.time() * 1000) & 0xFFFFFFFF
            base = out_dir / f"{stem}_{safe_ch}_{stamp:x}"
            n = 0
            while (base.with_name(f"{base.name}_fwd").with_suffix(".npy").exists()
                   or base.with_name(f"{base.name}_bwd").with_suffix(".npy").exists()):
                n += 1
                base = out_dir / f"{stem}_{safe_ch}_{stamp:x}_{n:02d}"
            fwd_path = base.with_name(f"{base.name}_fwd").with_suffix(".npy")
            bwd_path = base.with_name(f"{base.name}_bwd").with_suffix(".npy")
            np.save(fwd_path, f)
            np.save(bwd_path, b)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f"写 .npy 失败: {type(exc).__name__}: {exc}")

        return SkillResult(
            skill_name=name, success=True,
            data={"fwd_path": str(fwd_path), "bwd_path": str(bwd_path),
                  "channel": channel, "shape": list(f.shape),
                  "available_channels": sorted(channels),
                  # 说出来。读者不该靠猜来判断这两个 .npy 能不能直接逐点比 ——
                  # 之前它们**不能**,而没有任何一个字段说得出这件事。
                  "orientation": "sample_frame",
                  "orientation_note": ("反扫已 fliplr 回样品坐标、:SCAN_DIR: up 已 "
                                       "flipud —— 两个数组可以直接逐点/互相关比较。"),
                  "source": scan_path},
        )


class CheckScanForCrash(BaseSkill):
    """Post-scan tip-crash detection across one or more channels.

    Grabs each probed channel and flags a crash if ANY has near-zero variance
    (``ptp < 1e-25``) or NaN — a faithful, self-contained replacement for the
    raw out-of-graph check ``FullScan`` ran. The skill itself always SUCCEEDS
    (the analysis ran); the verdict is in ``data`` for the composite to branch
    on. Never reports "no crash" when it merely failed to read: ``status`` ∈
    ``{ok, crash, skipped}``.
    """

    _CRASH_RANGE_EPS = 1e-25

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CheckScanForCrash",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从刚采到的这一帧里检测撞针：抓取几路探针通道，"
                "只要有任何一路的方差接近零、或出现 NaN，就标记为撞针。"
                "返回 crash_indicator + status（ok/crash/skipped）+ 每路通道的"
                "判语。Channels = 逗号分隔的索引列表（默认 '0,14'："
                "形貌 + Z-controller 信号 —— 撞针会把 Z 压平，哪怕"
                "通道 0 看起来还说得过去）。"
            ),
            parameters=[
                ParameterSpec(
                    name="channels", type="str",
                    description="要探测的采集通道索引，逗号分隔",
                    required=False, default="0,14",
                ),
                ParameterSpec(
                    name="direction", type="int",
                    description="1 = 正扫，0 = 反扫",
                    required=False, default=1, allowed_values=[0, 1],
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "crash", "safety", "analysis", "g1"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np
        raw_channels = str(params.get("channels", "0,14"))
        direction = int(params.get("direction", 1))
        channels: list[int] = []
        for tok in raw_channels.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    channels.append(int(tok))
                except ValueError:
                    pass
        if not channels:
            channels = [0, 14]

        per_channel: dict[str, str] = {}
        all_calls = []
        readable_any = False
        crash = False
        crash_channel = None
        crash_range = None

        for ch in channels:
            label = f"ch{ch}"
            try:
                arr, rec = _grab_channel_array(context, ch, direction)
            except Exception:  # noqa: BLE001 — one bad channel must not abort the probe
                per_channel[label] = "error"
                continue
            if rec is not None:
                all_calls.append(rec)
            if arr is None:
                per_channel[label] = "no_data"
                continue
            readable_any = True
            data_range = float(np.ptp(arr))
            has_nan = bool(np.any(np.isnan(arr)))
            if data_range < self._CRASH_RANGE_EPS or has_nan:
                per_channel[label] = "crash"
                if not crash:
                    crash = True
                    crash_channel = label
                    crash_range = data_range
            else:
                per_channel[label] = "ok"

        if crash:
            status = "crash"
        elif readable_any:
            status = "ok"
        else:
            status = "skipped"  # nothing readable — inconclusive, never "ok"
            logger.warning("CheckScanForCrash: no channel returned usable data (%s)",
                           per_channel)

        return SkillResult(
            skill_name="CheckScanForCrash", success=True,
            data={
                "crash_indicator": crash,
                "status": status,
                "per_channel": per_channel,
                "crash_channel": crash_channel,
                "data_range": crash_range,
                "channels_probed": channels,
            },
            nanonis_calls=all_calls,
        )


class ComputeDriftVector(BaseSkill):
    """Estimate sample drift by cross-correlating the current scan frame against
    a reference .npy image; returns the drift in METERS.

    Self-contained governed wrapper for the raw grab + scipy.correlate2d that
    TrackDrift_ReferenceScan ran out-of-graph. Mirrors its ``_compute_drift``:
    normalise both frames, cross-correlate, peak offset × pixel size → metres.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ComputeDriftVector",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "抓取当前的扫描帧，与一张参考 .npy 图像做互相关，"
                "以估计样品漂移；结果以**米**为单位返回"
                "（drift_x_m, drift_y_m）。漂移跟踪类 workflow 会用到它。"
            ),
            parameters=[
                ParameterSpec(name="ref_path", type="str",
                              description="参考图像 .npy 的路径（2-D）",
                              required=True),
                ParameterSpec(name="scan_width_m", type="float", unit="m",
                              description="扫描框宽度，单位米（用于 px→m 换算）",
                              required=True, min_value=0.0),
                ParameterSpec(name="channel_index", type="int",
                              description="从哪一路通道抓取当前帧",
                              required=False, default=0, min_value=0),
                ParameterSpec(name="direction", type="int",
                              description="1 = 正扫，0 = 反扫",
                              required=False, default=1, allowed_values=[0, 1]),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["scan", "drift", "analysis", "g1"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import numpy as np
        ref_path = str(params["ref_path"])
        scan_width_m = float(params["scan_width_m"])
        ch = int(params.get("channel_index", 0))
        direction = int(params.get("direction", 1))

        try:
            ref = np.load(ref_path).astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="ComputeDriftVector", success=False,
                               error=f"cannot load reference image: {exc}")
        arr, rec = _grab_channel_array(context, ch, direction)
        calls = [rec] if rec is not None else []
        if arr is None:
            return SkillResult(skill_name="ComputeDriftVector", success=False,
                               error="could not grab current scan frame",
                               nanonis_calls=calls)
        if arr.size != ref.size:
            # Mirror the Python composite: a size mismatch ⇒ no usable drift,
            # report 0 rather than fail (the caller may still proceed).
            return SkillResult(
                skill_name="ComputeDriftVector", success=True,
                data={"drift_x_m": 0.0, "drift_y_m": 0.0,
                      "note": f"size mismatch ref={ref.size} cur={arr.size}"},
                nanonis_calls=calls)
        try:
            from scipy.signal import correlate2d
            cur = arr.reshape(ref.shape) - float(arr.mean())
            ref0 = ref - float(ref.mean())
            corr = correlate2d(ref0, cur, mode="same")
            peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
            cy, cx = ref.shape[0] // 2, ref.shape[1] // 2
            dy_px = int(peak[0]) - cy
            dx_px = int(peak[1]) - cx
            pixel_size = scan_width_m / ref.shape[1] if ref.shape[1] > 0 else 1e-9
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="ComputeDriftVector", success=False,
                               error=f"cross-correlation failed: {exc}",
                               nanonis_calls=calls)
        return SkillResult(
            skill_name="ComputeDriftVector", success=True,
            data={"drift_x_m": dx_px * pixel_size, "drift_y_m": dy_px * pixel_size,
                  "shift_x_px": dx_px, "shift_y_px": dy_px},
            nanonis_calls=calls)


class ParseRegions(BaseSkill):
    """Parse + validate + normalise a JSON region list for batch scanning.

    Lets a declarative composite ``foreach`` over operator-specified regions:
    the spec layer has no JSON parser (safe_eval is data-only), so this governed
    skill does ``json.loads`` + bounds-checking and returns a clean list of
    region dicts (each guaranteed to carry center_x_m / center_y_m / width_m /
    height_m / angle_deg / label). Mirrors BatchRegionsScan._parse_regions.
    """

    _MAX_REGIONS = 64
    _CENTER_LIMIT_M = 1e-3
    _SIZE_MIN_M = 1e-10
    _SIZE_MAX_M = 1e-5

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ParseRegions",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "解析 + 校验一个由扫描区域组成的 JSON 数组（每个形如 "
                "{center_x_m, center_y_m, width_m, height_m, [angle_deg], [label]}），"
                "转成 composite 可以 foreach 遍历的归一化列表。坏 "
                "JSON、>64 个区域、中心/尺寸越界，都会被拒绝。"
            ),
            parameters=[
                ParameterSpec(name="regions", type="str",
                              description="由区域对象组成的 JSON 数组（单位米）",
                              required=True),
            ],
            estimated_duration_s=0.2,
            composition_level=0,
            tags=["scan", "regions", "analysis", "g1"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        import json
        raw = params.get("regions", "")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name="ParseRegions", success=False,
                               error=f"invalid regions JSON: {exc}")
        if not isinstance(data, list):
            return SkillResult(skill_name="ParseRegions", success=False,
                               error="regions must be a JSON array")
        if len(data) > self._MAX_REGIONS:
            return SkillResult(skill_name="ParseRegions", success=False,
                               error=f"too many regions ({len(data)} > {self._MAX_REGIONS})")
        out: list[dict] = []
        for i, r in enumerate(data):
            if not isinstance(r, dict):
                return SkillResult(skill_name="ParseRegions", success=False,
                                   error=f"region {i} is not an object")
            try:
                cx = float(r["center_x_m"]); cy = float(r["center_y_m"])
                w = float(r["width_m"]); h = float(r["height_m"])
            except (KeyError, TypeError, ValueError) as exc:
                return SkillResult(skill_name="ParseRegions", success=False,
                                   error=f"region {i} missing/bad field: {exc}")
            if abs(cx) > self._CENTER_LIMIT_M or abs(cy) > self._CENTER_LIMIT_M:
                return SkillResult(skill_name="ParseRegions", success=False,
                                   error=f"region {i} center out of range (±{self._CENTER_LIMIT_M} m)")
            if not (self._SIZE_MIN_M <= w <= self._SIZE_MAX_M) or not (
                    self._SIZE_MIN_M <= h <= self._SIZE_MAX_M):
                return SkillResult(skill_name="ParseRegions", success=False,
                                   error=f"region {i} size out of range")
            out.append({"center_x_m": cx, "center_y_m": cy, "width_m": w,
                        "height_m": h, "angle_deg": float(r.get("angle_deg", 0.0)),
                        "label": str(r.get("label", f"R{i + 1}"))})
        return SkillResult(skill_name="ParseRegions", success=True,
                           data={"regions": out, "count": len(out)})
