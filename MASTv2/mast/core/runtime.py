"""Gradio-free core runtime — the construction of MAST's live core singletons.

AUTO-EXTRACTED from mast/gui/app.py (MASTApp) by tools/_extract_core_runtime.py.
This is the cutover home for the core (pool / safety / state / registry /
executor / storage / cognition / monitor / orchestrator / chat engine), with NO
Gradio dependency, so the FastAPI service can run after the Gradio UI is deleted.

Do not hand-edit large swaths; re-run the extractor if app.py construction
changes. ``mast/gui/app.py`` no longer exists (deleted in the TS rewrite), so
this module is now the canonical, hand-maintained source — there is nothing
left to re-extract from.
"""

from __future__ import annotations

import atexit
import logging
import sys
import threading as _threading_mod
import threading as _threading
import time as _t
import time
from pathlib import Path

from mast._runtime_paths import project_root
from mast.config import (
    MASTConfig,
    MODEL_PRESETS,
    THINKING_PRESETS,
    model_thinking_mode,
)
from mast.agents._shared.models import GLM_5_1, GLM_5_2, MINIMAX_M3, OPUS_4_7, SONNET_4_6, HAIKU_4_5
from mast.core.operating_mode import safe_mode_active


def _chat_effective_safety_limits(raw):
    """私聊 agent 图要用的**生效**包络（KNOWN_ISSUES §2.16）。

    走的是手动执行路径**同一个函数** —— 默认值 → 合并管理员覆写 → 按已登记的仪器
    事实收紧。私聊这条线原来直接传 ``config.safety`` 原件，于是同一个 agent 在私聊里
    被出厂包络约束，而它的工具 schema 通告的却是用户声明的那份
    （``skill_adapter._envelope_for`` 一直是读覆写的）。

    管理员覆写与执行闸门若使用不同的包络，工具 schema 声明合法的参数仍可能
    被拒绝，且拒绝说明与模型看到的范围矛盾。两条路径必须共享生效包络。

    ⚠️ 不能改成「把 build() 收到的 ``registry`` 转发给 SafetyGateMiddleware」——
    那两个 ``registry`` **同名不同物**：``build(registry=…)`` 收的是 ``SkillRegistry``，
    而 ``SafetyGateMiddleware(registry=…)`` 要的是 ``ConfigOverrideRegistry``。
    转发过去会在 ``get_safety_limits()`` 上抛 AttributeError，被 ``except`` 吞掉，
    行为一模一样地坏，只多一行 warning。

    失败一律退回原件：一个坏掉的覆写文件绝不能让私聊建不起来。
    """
    from mast.config import SafetyLimits

    base = raw if raw is not None else SafetyLimits()
    try:
        from mast.core.safety import _get_effective_limits

        return _get_effective_limits(base)
    except Exception as exc:  # noqa: BLE001
        logger.warning("私聊安全包络合并失败，退回原始配置: %s", exc)
        return base

logger = logging.getLogger(__name__)


# How much of a skill's result payload one v1 action row may hold. A skill's
# ``data`` can carry whole sample traces (TipShapeWithReadback's z/current
# arrays), so the records layer BOUNDS it — it does not drop it. Before
# 2026-07-27 the whole dict was thrown away and every action read ``"data": {}``,
# which reads exactly like a skill that returned nothing ().
_V1_DATA_MAX_CHARS = 20000
_V1_STR_CAP = 2000
_V1_SEQ_CAP = 200
# Trajectory exit_status vocabulary the v2 schema actually accepts
# (`CHECK (exit_status IN ('success','aborted','failed','timeout'))`). Callers
# say "completed"; that value made the UPDATE fail its CHECK constraint and the
# failure was swallowed at DEBUG level, so no finished run ever got an exit
# status (2026-07-27, same silent-write family as).
_TRAJ_EXIT_STATUS = {
    "success": "success", "completed": "success", "complete": "success",
    "done": "success", "ok": "success",
    "aborted": "aborted", "abort": "aborted", "cancelled": "aborted",
    "canceled": "aborted", "stopped": "aborted",
    "failed": "failed", "failure": "failed", "error": "failed",
    "timeout": "timeout", "timed_out": "timeout",
}


# Fire-and-forget MUST mean "does not propagate", never "is not observable".
# The 2026-07-27 forensics found the whole agent training log dead for a full
# day behind `logger.debug(..., exc_info=True)`: every INSERT failed a foreign
# key, at a 100% rate, and the service log carried not one line about it. These
# swallow sites now log the first occurrence LOUDLY and then throttle, so a
# systematic failure is impossible to miss while a flaky one can't spam.
_SWALLOW_QUIET_S = 60.0
_SWALLOW_LOCK = _threading.Lock()
_SWALLOW_STATE: dict[str, list] = {}   # key -> [total_count, last_logged_monotonic]


def _log_swallowed(key: str, msg: str, *args) -> None:
    """Log a swallowed exception: first one at WARNING with the traceback, then
    at most one line per ``_SWALLOW_QUIET_S`` carrying the running total."""
    now = _t.monotonic()
    with _SWALLOW_LOCK:
        st = _SWALLOW_STATE.setdefault(key, [0, 0.0])
        st[0] += 1
        count, last = st[0], st[1]
        due = count == 1 or (now - last) >= _SWALLOW_QUIET_S
        if due:
            st[1] = now
    if not due:
        return
    if count == 1:
        logger.warning(msg, *args, exc_info=True)
    else:
        logger.warning("%s [%d occurrences of '%s' so far]",
                       msg % args if args else msg, count, key)


def _finite(*candidates):
    """First candidate that is a real finite number, else None. NaN/inf are
    rejected at the source — a non-finite coordinate poisons the map's extent
    maths (same rule as mast.io.exp_map._coerce_float)."""
    import math
    for v in candidates:
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


# Files a skill can produce that the v2 store knows how to describe. Anything
# else is left alone rather than filed under a guessed format.
_SCAN_FILE_FORMATS = {".sxm": "sxm", ".dat": "dat", ".3ds": "3ds", ".h5": "h5",
                      ".png": "png", ".npy": "npy", ".parquet": "parquet",
                      ".tif": "tiff", ".tiff": "tiff"}
_SCAN_FILE_MAX_BYTES = 512 * 1024 * 1024
_ARTIFACTS_PER_SKILL = 50


def _artifact_paths(payload: dict) -> list[str]:
    """Every file THIS skill call really produced, de-duplicated.

    Covers the single-artifact shape (`data.path` / `file_path` / `sxm_path`,
    already resolved into `payload['artifact_path']`) AND the multi-file shapes
    a composite reports — `scanned_paths`, and the per-region records. A region
    that FAILED contributes nothing: BatchRegionsScan has been seen carrying an
    `sxm_path` on a region the safety gate rejected (an earlier region's file),
    and treating that as this region's output is how a rejected region ends up
    corroborating a result."""
    import os
    out: list[str] = []
    seen: set[str] = set()

    def _add(p) -> None:
        if not p:
            return
        s = str(p)
        key = os.path.normcase(os.path.normpath(s))
        if key not in seen:
            seen.add(key)
            out.append(s)

    _add(payload.get("artifact_path"))
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("scanned_paths", "saved_paths"):
            seq = data.get(key)
            if isinstance(seq, (list, tuple)):
                for p in list(seq)[:_ARTIFACTS_PER_SKILL]:
                    _add(p)
        regions = data.get("regions")
        if isinstance(regions, list):
            for r in regions[:_ARTIFACTS_PER_SKILL]:
                if isinstance(r, dict) and r.get("success", True):
                    _add(r.get("sxm_path") or r.get("path") or r.get("file_path"))
    return out[:_ARTIFACTS_PER_SKILL]


def _load_quarantine_shas() -> set:
    """隔离区索引里已有的 sha256（去重用）。索引损坏就当空的。"""
    import json
    out: set[str] = set()
    try:
        from mast.core.experiment_paths import quarantine_dir
        with open(quarantine_dir() / "index.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = row.get("sha256")
                if s:
                    out.add(str(s))
    except OSError:
        pass
    return out


def _register_scan_files(repos, action_id: str, paths) -> int:
    """File the artifacts of one action as v2 ``scan_files`` rows (with their
    sha256, so the record can prove the file on disk is still the one that was
    produced). Returns how many were registered.

    The v2 store held 0 scan_files for a session that wrote 9 .sxm — the paths
    existed only in a chat transcript that was itself truncated (/). Only files that ACTUALLY EXIST are registered: a path an
    agent mentioned but never wrote must stay absent from the record, because
    'is it in scan_files' is exactly the question the claim cross-check asks."""
    import hashlib
    import os
    n = 0
    for p in paths or ():
        try:
            fmt = _SCAN_FILE_FORMATS.get(os.path.splitext(str(p))[1].lower())
            if not fmt or not os.path.isfile(p):
                continue
            size = os.path.getsize(p)
            if size <= 0 or size > _SCAN_FILE_MAX_BYTES:
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            repos.scan_files.register(
                produced_by_action_id=action_id, sha256=h.hexdigest(),
                size_bytes=size, current_path=str(p), format_kind=fmt,
                parser_spec="nanonis-sxm" if fmt == "sxm" else fmt)
            n += 1
        except Exception:  # noqa: BLE001 — one bad file can't lose the others
            _log_swallowed("scan_file_register", "scan_file register failed for %s", p)
    return n


def _marker_subrecords(data) -> list[dict]:
    """Per-item positioned records inside a composite's result, normalised.

    Returns ``[]`` unless the result carries a list whose entries have BOTH a
    centre x and y — i.e. only when the skill really reported where each item
    happened. Never guesses a position: a marker at the wrong place is worse
    than no marker.

    ``meta`` 与 ``kind`` 是 2026-08-14 加的两个转发口(S3 畴搜索设计 §3.4)：

    * **``meta``** —— 此前这个函数把每条子记录压成 9 个固定键，**别的一律丢掉**。
      于是「skill 在 ``SkillResult.data`` 如实报告、由 recorder 落库」这条分层原则
      对任何**带自己那套 meta** 的多点技能都走不通：畴指纹 / verdict / 参照系版本
      全都进不了地图。而整条记录链是 fire-and-forget，症状不是报错，是
      ``meta.fingerprint`` 永远是 ``None``，同时技能自己的返回体里那个字段一直
      好好的。三条线(畴普查 / 多点谱学 / 跨点针尖核查)同坑。
    * **``kind``** —— 只接受 :data:`~mast.io.exp_map.KIND_STYLE` 里**已经存在**的
      kind，认不出来就退回调用方按技能名分类的结果。这道白名单是刻意的：marker
      kind 是**双端镜像**的(后端 ``KIND_STYLE`` ↔ 前端配色/图例)，让子记录随便声明
      一个新 kind，失败模式是地图上静默变灰——没有报错，只是那批点看不出是什么。
    """
    if not isinstance(data, dict):
        return []
    from mast.io.exp_map import KIND_STYLE

    out: list[dict] = []
    for key in ("regions", "points"):
        seq = data.get(key)
        if not isinstance(seq, list):
            continue
        for i, r in enumerate(seq):
            if not isinstance(r, dict):
                continue
            x = _finite(r.get("center_x_m"), r.get("x_m"), r.get("x"))
            y = _finite(r.get("center_y_m"), r.get("y_m"), r.get("y"))
            if x is None or y is None:
                continue
            meta = r.get("meta")
            kind = str(r.get("kind") or "")
            out.append({
                "x_m": x, "y_m": y,
                "w_m": _finite(r.get("width_m"), r.get("w_m")),
                "h_m": _finite(r.get("height_m"), r.get("h_m")),
                "index": r.get("index", i + 1),
                "label": r.get("label") or "",
                # Absent success flag → unknown-but-reached, i.e. done; an
                # EXPLICIT False is what draws a failed marker.
                "success": r.get("success", True),
                "error": r.get("error"),
                "artifact_path": (r.get("sxm_path") or r.get("path")
                                  or r.get("file_path")),
                # None (not {}) 时调用方原样跳过，不去覆盖既有的那几个 meta 键。
                "meta": meta if isinstance(meta, dict) else None,
                "kind": kind if kind in KIND_STYLE else None,
            })
        if out:
            break
    return out


_LATERAL_COARSE_DIRECTIONS = ("x+", "x-", "y+", "y-")

#: Skills whose SUCCESSFUL result means "the stage slid sideways". ``MotorMove``
#: is the raw primitive; ``RelocateCoarseXY`` is the guarded composite, which
#: issues ``Motor_StartMove`` through ``safe_call`` DIRECTLY (the same reason
#: RetractForSampleChange does: it must not inherit MotorMove's hardcoded
#: semantics). That bypass means the recorder never sees a ``MotorMove``
#: payload for it — so without this second name the stage would move, the
#: coordinate frame would die, and ``coord_epoch`` would never advance. Any
#: future composite that steps the lateral motor itself must be added here.
_LATERAL_COARSE_SKILLS = ("motormove", "relocatecoarsexy")


#: The one field a FAILED result may use to still reach the odometer.
#:
#: It means, and may only be written to mean: "the stage physically took this
#: many lateral steps, whatever the overall verdict of this skill turns out to
#: be". A skill that cannot say that must not write it. See
#: :func:`lateral_coarse_move_info` for why this is a separate key rather than a
#: relaxation of the ``success`` gate.
_STEPS_TAKEN_KEY = "lateral_steps_taken"


def lateral_coarse_move_info(payload: dict) -> dict | None:
    """Return direction and steps for a lateral coarse move that occurred.

    Lateral moves change the surface addressed by piezo coordinates, advancing
    coord_epoch. Z-only moves do not.

    A successful result uses echoed data, then request parameters as fallback.
    A failed result is recorded only when it explicitly reports a positive
    _STEPS_TAKEN_KEY count. Never infer completed movement from requested
    parameters on a failure, and never record a dry run as physical movement.

    The evidence is the acknowledged motor chunks reported by the skill. Without
    position feedback this is an odometer record, not an independent position
    measurement. A later failure cannot erase an earlier reported move.
    """
    if not isinstance(payload, dict):
        return None
    if str(payload.get("skill") or "").strip().lower() not in _LATERAL_COARSE_SKILLS:
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    if data.get("dry_run") is True or params.get("dry_run") is True:
        return None

    succeeded = payload.get("success") is True
    if succeeded:
        direction = str(data.get("direction") or params.get("direction") or "").strip()
        steps = _finite(data.get("steps"), params.get("steps"))
    else:
        direction = str(data.get("direction") or "").strip()
        taken = _finite(data.get(_STEPS_TAKEN_KEY))
        if taken is None or taken <= 0:
            return None
        steps = taken
    if direction not in _LATERAL_COARSE_DIRECTIONS:
        return None
    return {"direction": direction,
            "steps": int(steps) if steps is not None else None,
            "partial": not succeeded}


def is_crash_report(payload: dict) -> bool:
    """仅在 crash_indicator is True 时记录撞针，不使用 success 代替判决。
    
    一次成功的检测可以发现撞针；success 只说明检查是否执行成功。
    crash_indicator 的 None 表示未知，不能折叠成已撞或未撞。
    例如振幅判据在激励关闭、激励状态未知或缺基线时缺少有效比较条件，必须弃权。
    检测结果与执行结果分开，避免生成假撞针禁区或把未知冒充安全。"""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    return data.get("crash_indicator") is True


def crash_point(payload: dict) -> tuple[float, float] | None:
    """Where a crash happened, from the PAYLOAD alone — or None.

    ``None`` here means "this payload does not say where", which covers both
    "not a crash report" and "a crash with no readable position". The caller
    must NOT read that as "no crash": ask :func:`is_crash_report` for that.
    Keeping the two apart is the whole point — a crash we cannot place is still
    a crash, and it gets recorded as one (see ``_record_map_marker``), just
    without a keep-out circle.

    Keyed on the RESULT carrying ``crash_indicator``, not on a skill name and
    not on the tip-crash tracker: the tracker is an in-memory 30-minute
    block-list for the fast path, whereas this marker is the permanent,
    auditable record of a spot that damaged the tip. The two are allowed to
    disagree — the tracker forgets, the map does not."""
    if not is_crash_report(payload):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    x = _finite(data.get("center_x_m"), params.get("center_x_m"), params.get("x_m"))
    y = _finite(data.get("center_y_m"), params.get("center_y_m"), params.get("y_m"))
    if x is None or y is None:
        return None
    return (float(x), float(y))


#: 撞针判据的作用域。见 :func:`crash_scope`。
_POINT_SCOPE = "point"
_FRAME_SCOPE = "frame"


def crash_scope(payload: dict) -> str:
    """这次撞针判据对**一个点**负责,还是对**一整帧**负责 —— ``"point"`` / ``"frame"``。

    决定的是「payload 说不出坐标时该退到哪」,而两者的**不确定度差一个量级**:

    * **point**(qPlus 振幅):针尖撞在它**当时所在的那一点**,不确定度 ≈ 0
      ⇒ 退到针尖位置是**对的**;
    * **frame**(post-scan 数据方差):撞点可能在帧内**任何地方**,不确定度 ≈ 帧的
      半对角 ⇒ 退到帧中心只有在「半对角 ≤ 避让半径」时才站得住(见
      :func:`crash_position` 的尺寸闸)。

    判据用**技能自己声明的** ``data["crash_scope"]`` 优先 —— 给未来的判据一条
    干净的表达方式,免得下一个人去改下面那条形状启发式。没声明就看**正面证据**:
    振幅类判据报的是一个点上的标量对(``amplitude`` + ``baseline``)。

    **认不出一律算 ``frame``**,这是刻意的失败方向:``frame`` 那一档要过尺寸闸,
    最坏是落 unlocated;而错判成 ``point`` 会给出一个**假装精确**的圈。
    这里不按技能名分类 —— ``crash_point`` 的契约从一开始就是「keyed on the RESULT」。
    """
    if not isinstance(payload, dict):
        return _FRAME_SCOPE
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    declared = str(data.get("crash_scope") or "").strip().lower()
    if declared in (_POINT_SCOPE, _FRAME_SCOPE):
        return declared
    if (_finite(data.get("amplitude")) is not None
            and _finite(data.get("baseline")) is not None):
        return _POINT_SCOPE
    return _FRAME_SCOPE


def crash_position(payload: dict, state, *, crash_r_m: float | None = None):
    """撞针记在哪 —— ``(xy | None, pos_src, unlocated_reason)``。**一级都不许编。**

    优先级(``point`` 档比 ``frame`` 档多第 2 级):

    1. ``payload`` 自己报的坐标 —— ``pos_src="payload"``;
    2. 仅 ``point`` 档:快照里的**针尖位置** —— ``pos_src="tip_position"``。
       点式判据唯一真正对的位点;
    3. 快照里的**扫描框中心** —— ``pos_src="state_scan_frame"``,**但要过尺寸闸**;
    4. 都不行 ⇒ ``(None, "unlocated", 理由)``。

    ## 尺寸闸:大帧时「帧中心」不是不精确,是**错的**

    帧中心画的圈半径是 ``crash_r_m``,而真实撞点可能在帧内任何地方,最远是半对角::

        100 nm 帧 → 半对角 ≈  71 nm  < 150 nm ⇒ 撞点仍在圈内,可以用
        500 nm 帧 → 半对角 ≈ 354 nm  > 150 nm ⇒ **撞点证明得出来可能在圈外**

    后者那个圈**两头都错**:它没圈住真正撞过的点,却圈掉了一块**没撞过的好表面**
    (选点器会绕开它)。一个证明得出来可能不含撞点的圈,比承认「不知道在哪」更坏 ——
    它给的是假信心。所以半对角 > 半径时**宁可落 unlocated**。

    ``crash_r_m`` 由调用方注入(取自 ``AnalysisConfig.radius_for("crash")``,即
    ``DAMAGE_KINDS`` 那张表)—— **这里不写第二个 150**。注入而不是自己去读,也让这个
    函数保持纯函数、可直接测。``None`` = 这台仪器根本没有撞针避让圈,那就没有「假圈」
    可言,尺寸闸自然不适用。

    ``unlocated_reason`` 对用户是三句不同的话:
    ``no_position``(不知道在哪)、``frame_too_large``(知道大概在哪但不敢画圈)、
    ``frame_size_unknown``(读不到帧多大 ⇒ 证不出这个圈站得住)。
    """
    xy = crash_point(payload)
    if xy is not None:
        return xy, "payload", None

    if crash_scope(payload) == _POINT_SCOPE:
        tx = _finite(getattr(state, "x_pos_m", None))
        ty = _finite(getattr(state, "y_pos_m", None))
        if tx is not None and ty is not None:
            return (float(tx), float(ty)), "tip_position", None

    fx = _finite(getattr(state, "scan_center_x_m", None))
    fy = _finite(getattr(state, "scan_center_y_m", None))
    if fx is None or fy is None:
        return None, "unlocated", "no_position"

    if crash_r_m is not None and float(crash_r_m) > 0:
        w = _finite(getattr(state, "scan_width_m", None))
        h = _finite(getattr(state, "scan_height_m", None))
        if w is None and h is None:
            # 帧多大都不知道 ⇒ 证不出这个圈盖得住撞点。不猜。
            return None, "unlocated", "frame_size_unknown"
        w = float(w if w is not None else h)
        h = float(h if h is not None else w)
        if ((w * w + h * h) ** 0.5) / 2.0 > float(crash_r_m):
            return None, "unlocated", "frame_too_large"

    return (float(fx), float(fy)), "state_scan_frame", None


# Guards the tip-quality halt slot. Module level (not per-instance) so the
# vision publisher thread never races a lazily-created lock; there is one
# CoreRuntime per process.
_TIP_HALT_LOCK = _threading.Lock()


#: ⑰-C1(2026-08-09)之前这里还有 ``_safe_mode_suppresses_tip_halt()`` —— SAFE 模式
#: 下跳过**视觉**来源 CRITICAL 触发的中止。它被**包含掉**了:视觉判定现在在**任何
#: 模式、任何场景**下都不再中止 composite,SAFE 只是其中一种情形。
#:
#: 它那段「视觉 vs 电流监控该怎么区分」的推理没有丢 —— 整段搬进了
#: :func:`tip_halt_source`,那是现在**唯一**一处做这个区分的地方(⑰-C1 之前同一段
#: 判别写了三遍:SAFE 抑制、修针抑制、和 ``raise_tip_halt(source=...)`` 的实参)。


def tip_halt_source(payload: dict) -> str:
    """这条 CRITICAL ``tip_quality_drop`` 是谁判出来的:``"vision"`` 还是
    ``"current_monitor"``。**判据只有这一份。**

    两个生产者用**同一个 event kind**,含义却完全不同:

      - **vision**(``signal`` = tip_change / learned_quality / …):「针尖顶端比
        刚才差了」—— 一个关于**形貌**的猜测,由模型做出,而针尖可能正被用户
        蓄意改变;
      - **current_monitor**(``source`` = ``current_monitor``,或 ``signal`` 以
        ``current_`` 开头 / 在 ``physical_current_signals()`` 里):前放到轨、
        测量链死了、远超结电流的台阶 —— **物理越界**,不是形貌猜测。

    所以区分不能只按 event kind 做，必须读取 payload。

    **标记读不出来时一律算 ``"vision"``**(历史生产者),而 ⑰-C1 之后 vision =
    「永不中止」—— 也就是失败方向是**安静**。这与 SAFE 与修针豁免一直用的默认同向:
    明天新加的一个判定在有人明确把它归进物理类之前,不会获得中止实验的权力。
    """
    try:
        if str(payload.get("source") or "") == "current_monitor":
            return "current_monitor"
        signal = str(payload.get("signal") or "")
        if signal.startswith("current_"):
            return "current_monitor"
        from mast.core.tip_intent import physical_current_signals

        if signal and signal in physical_current_signals():
            return "current_monitor"
        return "vision"
    except Exception:  # noqa: BLE001 — publisher thread; 安静是失败方向
        return "vision"


def _tip_work_suppresses_tip_halt(payload: dict) -> str:
    """返回蓄意修针期间应豁免瞬变告警的技能名，否则返回空串。

    仅处理电流监控来源的瞬变类信号；视觉来源由上层单独处理。持续贴轨和
    冻结从不因修针而豁免。此处读取仪器令牌，不依赖监控分段的技能归属，
    因为两者可能存在时间窗口差异。

    豁免只影响步边界中止，事件记录、通知与诊断仍保留。判据由
    core.tip_intent 统一提供。
    """
    try:
        from mast.core.tip_intent import active_tip_work, exempt_during_tip_work

        # 只有**显式归进瞬变类**的才豁免。持续类、以及明天新加而没人归类的信号，
        # 一律照 halt —— 白名单不是黑名单。
        signal = str(payload.get("signal") or "")
        return active_tip_work() if exempt_during_tip_work(signal) else ""
    except Exception:  # noqa: BLE001 — publisher thread; halting is the safe default
        return ""


def _notice_vision_verdict(ev, payload: dict) -> None:
    """视觉针尖判定「本来会在这里中止流程」—— 记一行,然后放行。永不抛。

    为什么这里要写台账,而 ``buffer_hitl`` 已经为同一条事件写过一行:**两行答的是
    两个问题**。buffer_hitl 那行说的是「一条关键事件到了 agent 状态里,没有升级成
    确认框」;这一行说的是「**这条判定本来会中止一条正在跑的 composite**」——
    「本来会拦我几次」要的是后面这个数。

    两者也不总是同时存在:buffer_hitl 只在有 agent 带着 buffer 在跑时才有;这个钩子
    跑在视觉发布线程上,手动路径的扫描也照样经过。少了这一行,那些场景下的「本来会
    拦」就没有任何落点。

    ``subject`` 用 ``tip_halt:vision`` 前缀,与 buffer_hitl 的 ``buffer:<kind>``
    分得开 —— 两个数不该被混成一个。
    """
    signal = payload.get("signal")
    reason = str(payload.get("summary_zh") or payload.get("advice")
                 or payload.get("summary") or "视觉判定针尖状态恶化")
    logger.info(
        "视觉针尖 CRITICAL 记录但不中止（signal=%s）：%s。"
        "⑰-C1 之后视觉判定在任何场景下都不再中止 composite；"
        "物理越界（贴轨 / 冻结 / 巨幅瞬变）照旧中止。事件照常记录并进面板。",
        signal, reason)
    try:
        from mast.core.diagnostics import record as _diag

        _diag("notice_only", "tip_halt:vision",
              f"视觉针尖判定 —— 只记录不中止流程：{reason}",
              signal=signal,
              event_id=str(getattr(ev, "event_id", "") or ""),
              seqno=getattr(ev, "seqno", None),
              cause_ref=getattr(ev, "cause_ref", None))
    except Exception:  # noqa: BLE001 — 发布线程;台账写不进去绝不能反噬事件分发
        logger.debug("视觉判定通知写入失败(已忽略)", exc_info=True)


def make_tip_halt_hook(app):
    """A BufferService critical hook that arms the composite halt ().

    Fires on CRITICAL ``tip_quality_drop`` ONLY. A WARN-level drop is an
    observation, not a reason to stop a plan mid-flight, and every other event
    kind has its own path (E_STOP has the abort latch). Runs inline on the
    vision publisher thread, so it does nothing but take a lock and write a
    dict — and never raises, because an exception here breaks event fanout for
    every other consumer."""
    from mast.buffer.schemas import Severity as _Sev, VisionEventType as _VET

    def _hook(ev) -> None:
        try:
            if getattr(ev, "kind", None) is not _VET.TIP_QUALITY_DROP:
                return
            if _coerce_sev(getattr(ev, "severity", None)) is not _Sev.CRITICAL:
                return
            payload = getattr(ev, "payload", None) or {}

            # 视觉针尖判定仅记录和通知，不在任何模式中触发此处的中止链。
            # 事件仍进入缓冲、记录和诊断，供 ReadHardwareEvents 与面板使用。
            # current_monitor 的物理来源仍按独立规则处理；修针瞬态可豁免，持续异常仍拦截。
            # 来源分类统一由 tip_halt_source 决定，不能把通知策略当成移除物理防护。
            if tip_halt_source(payload) == "vision":
                _notice_vision_verdict(ev, payload)
                return
            # ── 蓄意修针豁免（此处只剩电流监控来源）──
            # 修针流程会故意改变针尖，并且每扎一次就扫图确认 —— 视觉必然判到突变；
            # 脉冲和扎针每一下也必然在电流上打出远超结电流的台阶。不豁免的话，流程
            # 会在自己造出的证据上被中止（而且是扎完刚看完簇那个步边界，一次修针只
            # 做一半）。
            #
            # ⑰-C2（2026-08-09）：豁免范围从「只跳过视觉来源」扩到「视觉来源 +
            # 电流监控的**瞬变类**」。**持续类（贴轨 / 冻结）照旧 halt** —— 修针
            # 不该造成持续贴轨。判据与翻盘条件见
            # `_tip_work_suppresses_tip_halt` 与 `mast.core.tip_intent`。
            tip_work = _tip_work_suppresses_tip_halt(payload)
            if tip_work:
                logger.info(
                    "正在执行 %s（蓄意修针）：跳过本条 CRITICAL 触发的复合技能中止"
                    "（signal=%s）。这是本职动作的签名，不是事故；持续型物理越界"
                    "（贴轨 / 冻结）仍会照常中止。事件照常记录并进面板。",
                    tip_work, payload.get("signal"))
                return
            # `summary_zh` FIRST — it is the key scan_monitor actually writes
            # ("扫描中途针尖状态突变（已采集 71 行中第 ~9 行…）——12% 处"). `advice`
            # and `summary` are written by NOTHING in this repo, so the halt
            # reason fell through to the generic sentence every single time and
            # the specific, actionable verdict never reached the operator or the
            # agent (grep: 0 producers, 2026-07-28).
            app.raise_tip_halt(
                str(payload.get("summary_zh") or payload.get("advice")
                    or payload.get("summary") or "电流监控判定物理越界"),
                event_id=str(getattr(ev, "event_id", "") or ""),
                seqno=getattr(ev, "seqno", None),
                # ⑰-C1 之后走到这里的**只可能**是电流监控来源(视觉在上面就返回了),
                # 所以这个实参是常量而不再是一份重复的判别。原来那份三元式是
                # `tip_halt_source()` 判据的第三个副本 —— 三份必须保持相等的判别,
                # 正是本仓反复踩过的形状。
                source="current_monitor")
        except Exception:  # noqa: BLE001 — publisher thread; never propagate
            logger.exception("tip-halt hook failed")

    return _hook


def _attach_halt_check(ctx, app, run_id: str) -> None:
    """Give an ExecutionContext the run-scoped ``check_halt`` composites poll.

    Best-effort: a context that won't take the attribute simply has no halt
    check, which degrades to today's behaviour (composite runs to completion)
    rather than breaking the run."""
    try:
        ctx.check_halt = app._make_halt_check(run_id)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("check_halt attach failed for run %r", run_id)


def _attach_marker_sink(ctx, app) -> None:
    """让 composite 子步骤使用与 agent 工具边界相同的地图标记记录器。
    
    子步骤走 ExecutionContext.run，不经过 wrap_skill 的 post hook。
    缺少这条接线时，脉冲等表面操作不会留下地图标记，后续清洁区域选择可能
    错误地复用已改变的位置。挂接采用尽力而为策略；失败不改变技能执行结果。"""
    try:
        ctx.marker_sink = app._record_map_marker  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("marker_sink attach failed")


def _coerce_sev(sev):
    """VisionEvent.severity as a Severity member (it may arrive as a str)."""
    from mast.buffer.schemas import Severity as _Sev
    if isinstance(sev, _Sev):
        return sev
    try:
        return _Sev(str(sev).lower())
    except Exception:  # noqa: BLE001 — an unparseable severity is not critical
        return _Sev.INFO


def _refine_marker_from_saved_file(marker, data: dict):
    """Replace a spectroscopy marker's readback position with the exact xy from
    the saved .dat header. Returns ``(marker, extra_meta)``.

    ``AcquireSTS`` takes no position parameter — it measures wherever the tip
    already is — so its marker would otherwise be placed from the ≤1 s-old
    cached snapshot. But Nanonis writes the true stage coordinate into the file
    it just saved, and ``_attach_saved_dat`` has already resolved that path into
    the result. Reading it here rather than inside the skill keeps file parsing
    a *record* concern: the skill's job is the measurement.

    Only ever narrows: the header is consulted for point-spectroscopy markers,
    and ``extract_dat_position`` returns None unless the coordinate parses and
    passes the ±1 mm plausibility guard, in which case the readback position
    stands."""
    if marker is None or getattr(marker, "kind", "") != "sts":
        return marker, {}
    path = (data or {}).get("path")
    if not isinstance(path, str) or not path.lower().endswith(".dat"):
        return marker, {}
    try:
        from mast.io.exp_map import extract_dat_position
        pos = extract_dat_position(path)
    except Exception:  # noqa: BLE001 — a bad file can never lose the marker
        return marker, {}
    if pos is None:
        return marker, {}
    from dataclasses import replace as _replace
    return (_replace(marker, x_m=pos[0], y_m=pos[1]),
            {"pos_src": "dat_header", "file": path})


def _pos_provenance(marker, params: dict, state, data: dict | None = None) -> dict:
    """Where a marker's coordinate CAME FROM — commanded, frame, or readback.

    Three very different epistemic statuses were being recorded identically
    ():

      ``param``          the skill was TOLD to go here. An aim.
      ``scan_frame``     the live scan frame's centre. Where the raster is
                         defined, NOT where the tip is.
      ``skill_readback`` the skill read the hardware position at the instant it
                         acted, and reported it. The strongest source there is
                         for an in-place operation.
      ``tip_readback``   wherever the cached snapshot said the tip was. The
                         skill carried no position at all and could not read
                         one, so this is up to a refresh interval stale.

    The 15:03 AcquireSTS ran with ``params={"save_basename": ""}`` — no
    position whatsoever — so its marker is a *readback*: (1432.6, 1354.4) nm,
    56 nm from the (1400, 1400) frame centre the agent reported, and the same
    point the 16:12 spectrum later landed on. The record was right both times;
    nothing in it said "this is where the tip was, not where anyone aimed".

    Derived by comparing the marker's own output against each candidate source
    rather than by re-deriving marker_from_skill's precedence — so this can't
    drift away from what that function actually did. Also stamps snapshot
    staleness: a marker placed from a stale snapshot is a position that may be
    minutes old."""
    from mast.io.exp_map import _param_length_m

    out: dict = {}
    mx, my = getattr(marker, "x_m", None), getattr(marker, "y_m", None)
    if mx is None or my is None:
        return out
    px = _param_length_m(params or {}, "x_m", "x", "x_nm")
    py = _param_length_m(params or {}, "y_m", "y", "y_nm")
    dx = _finite((data or {}).get("x_m"))
    dy = _finite((data or {}).get("y_m"))
    if px == mx and py == my:
        src = "param"
    elif dx is not None and dx == mx and dy == my:
        src = "skill_readback"
    elif (state is not None
          and getattr(state, "scan_center_x_m", None) == mx
          and getattr(state, "scan_center_y_m", None) == my):
        src = "scan_frame"
    elif (state is not None
          and getattr(state, "x_pos_m", None) == mx
          and getattr(state, "y_pos_m", None) == my):
        src = "tip_readback"
    else:
        src = "unknown"
    out["pos_src"] = src
    if src in ("scan_frame", "tip_readback") and getattr(state, "stale", False):
        # The snapshot itself said its values were carried forward from an
        # older read (no hardware read succeeded in the last refresh).
        out["pos_stale"] = True
        out["pos_as_of"] = str(getattr(state, "timestamp", "") or "")
    return out


def _registered_category(registry, skill: str):
    """注册表里这个技能的 ``SkillMetadata.category``，查不到就 None。

    给 ``classify_skill`` 用（S4 STS 设计 §1.4a/O7）：名字规则靠子串匹配，追不上新
    技能，而 ``SkillCategory.ANALYSIS``（「Data processing, no hardware interaction」）
    是一条不会被名字绕过的判据。

    **查不到一律 None = 只按名字判**，不是「当成分析类」。这个方向是刻意的：
    ``_never_positions`` 的注释里那条代价不对称仍然成立 —— 误记看得见，漏记是一次
    真实的扎针从实验记录里消失。所以注册表缺席（还没进注册表的 spec 技能、手动活动
    侦测、测试替身）只回退到既有行为，绝不多删标记。

    模块级函数而不是 ``CoreRuntime`` 的方法：``_record_map_marker`` 在测试里是拿
    **鸭子类型的 self** 调的（``SimpleNamespace(_storage=…, _state=…)``，见
    ``test_forensics_20260727_records._rt`` 的注释「they only touch these attrs」）。
    第一版写成方法，那些替身上没有这个方法 ⇒ AttributeError ⇒ 被外层 try/except 吞
    掉 ⇒ **整个记录静默消失**，12 条取证测试当场红。所以这条路径只许读 self 的
    **属性**。

    约 11 µs 一次（实测 462 个技能全查 5.2 ms），相对一次技能执行可忽略，所以不缓存
    —— 缓存会在 composite 热注册改掉一个名字的类别时读到旧值。
    """
    if registry is None or not skill:
        return None
    try:
        return registry._get_metadata(registry.get(skill)).category
    except Exception:  # noqa: BLE001 — 记账是 best-effort，查不到就按名字判
        return None


def _log_one_marker(storage, *, kind: str, x_m, y_m, w_m=None, h_m=None,
                    angle_deg: float = 0.0, label: str = "",
                    skill_name: str = "", status: str = "done",
                    exp_id=None, sample_id=None, meta=None,
                    advance: bool = True) -> None:
    """Write one map marker + advance the planned route. Per-marker try/except
    so one bad row can't lose the rest of a batch."""
    from mast.io.exp_map import KIND_STYLE
    try:
        storage.log_marker(
            kind=kind, x_m=x_m, y_m=y_m, w_m=w_m, h_m=h_m,
            angle_deg=float(angle_deg or 0.0),
            label=label or KIND_STYLE.get(kind, ("", "", skill_name))[2],
            skill_name=skill_name, status=status, source="skill",
            experiment_id=exp_id, sample_id=sample_id,
            meta={k: v for k, v in (meta or {}).items() if v is not None})
    except Exception:  # noqa: BLE001
        _log_swallowed("map_marker_row", "map marker row failed for %s", skill_name)
        return
    if not advance:
        return
    # Advance the planned route — the operation reached a planned step, so the
    # nav-style ghost route shrinks toward the destination.
    try:
        from mast.io.plan_overlay import get_plan_overlay
        get_plan_overlay().advance(x_m, y_m)
    except Exception:  # noqa: BLE001
        pass


def _v1_shrink(value, _depth: int = 0):
    """Coerce one value into something JSON/`asdict`-safe and bounded."""
    if _depth > 4:
        return "<nested>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _V1_STR_CAP else value[:_V1_STR_CAP] + "…"
    if isinstance(value, dict):
        out = {str(k): _v1_shrink(v, _depth + 1)
               for k, v in list(value.items())[:_V1_SEQ_CAP]}
        if len(value) > _V1_SEQ_CAP:
            out["_truncated_keys"] = len(value) - _V1_SEQ_CAP
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [_v1_shrink(v, _depth + 1) for v in seq[:_V1_SEQ_CAP]]
        if len(seq) > _V1_SEQ_CAP:
            out.append(f"<+{len(seq) - _V1_SEQ_CAP} more>")
        return out
    return str(value)[:_V1_STR_CAP]


def _v1_result_data(data) -> dict:
    """Bound a skill's result ``data`` for storage in one v1 action row.

    Returns ``{}`` ONLY when the skill really returned nothing — a payload that
    is too big is stored partially with an explicit ``_truncated`` marker, never
    silently emptied (an empty dict is indistinguishable from "no result").
    """
    import json as _json
    if not isinstance(data, dict) or not data:
        return {}
    out = _v1_shrink(data)
    if not isinstance(out, dict):  # defensive — _v1_shrink keeps dict-ness
        return {"value": out}
    try:
        blob = _json.dumps(out, default=str)
    except Exception:  # noqa: BLE001
        return {"_unserialisable": str(type(data).__name__)}
    if len(blob) > _V1_DATA_MAX_CHARS:
        small = {k: v for k, v in out.items()
                 if v is None or isinstance(v, (bool, int, float, str))}
        small["_truncated"] = (f"{len(blob)} chars of result data exceeded the "
                               f"{_V1_DATA_MAX_CHARS}-char row budget; "
                               f"{len(out) - len(small)} non-scalar field(s) omitted")
        return small
    return out


def _v1_nanonis_calls(calls) -> list:
    """Bound a skill's NanonisCallRecord list for storage (return_value can be a
    whole scan/trace array). Keeps the records themselves — only the payload of
    each call is shrunk — so `if action.nanonis_calls:` becomes true when TCP
    calls really happened ( → citations/manager.py)."""
    if not calls:
        return []
    from mast.core.types import NanonisCallRecord
    out = []
    for c in list(calls)[:_V1_SEQ_CAP]:
        if not isinstance(c, NanonisCallRecord):
            continue
        try:
            out.append(NanonisCallRecord(
                method=str(getattr(c, "method", "") or ""),
                args=tuple(_v1_shrink(list(getattr(c, "args", ()) or ()))),
                kwargs=_v1_shrink(getattr(c, "kwargs", {}) or {}),
                return_value=_v1_shrink(getattr(c, "return_value", None)),
                error=str(getattr(c, "error", "") or "")[:_V1_STR_CAP],
                elapsed_s=float(getattr(c, "elapsed_s", 0.0) or 0.0),
                timestamp=str(getattr(c, "timestamp", "") or ""),
            ))
        except Exception:  # noqa: BLE001 — one odd record can't lose the rest
            continue
    return out


# ── module-level helpers (extracted) ──

def _ensure_buffer_loop(app):
    """Return a long-lived asyncio loop running on a daemon thread.

    Created once per process and cached on ``app._buffer_loop``. The
    BufferService captures ``asyncio.get_running_loop()`` in ``start()`` and
    schedules every WAL write via ``run_coroutine_threadsafe(..., that_loop)``.
    Previously the GUI did ``asyncio.run(buf.start())`` — that spins up a loop,
    runs start(), then CLOSES the loop, so ``buf._loop`` was a dead loop and
    every subsequent WAL append was silently dropped (``not loop.is_closed()``
    guard short-circuited) . Giving the buffer a loop that stays alive for
    the process makes WAL persistence actually work.
    """
    import asyncio
    import threading
    loop = getattr(app, "_buffer_loop", None)
    if loop is not None and not loop.is_closed():
        return loop
    loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run_loop, name="mast-buffer-loop", daemon=True)
    t.start()
    app._buffer_loop = loop
    return loop


def _ensure_buffer_for_gui(app) -> "object":
    """Return the live BufferService, kicking a NON-BLOCKING init if needed.

    The instance is bound to ``app._buffer`` so successive Tab refreshes reuse
    it. WAL is enabled at ``experiments/vision_buffer.wal.sqlite`` so the panel
    keeps showing history across restarts. The service runs on a LONG-LIVED
    daemon-thread event loop (see ``_ensure_buffer_loop``) so its WAL writer
    keeps running — #64.

    CRITICAL — never block the caller. This is invoked from ``gr.Timer`` ticks
    (Vision Pulse + Vision Buffer, every 1.5 s) that run on the shared Gradio
    queue worker pool. The old code did ``fut.result(timeout=10)`` inline, so a
    slow/locked WAL froze that worker for up to 10 s; worse, if ``start()``
    raised, ``app._buffer`` stayed ``None`` and EVERY subsequent tick re-entered
    and re-blocked, retry-storming the queue into a whole-UI freeze. We now fire
    ``start()`` on the persistent loop and publish ``app._buffer`` from a done
    callback; the caller gets ``None`` ("idle") until it is ready. All callers
    (``_read_recent_events`` / ``_pull``) already render an idle panel on
    ``None`` — graceful degradation, never a hang.
    """
    if app._buffer is not None:
        return app._buffer

    import time as _t
    # Single-flight + post-failure cooldown so a broken WAL can't retry-storm
    # the 1.5 s ticks. Guarded by a lock because ticks run on a threadpool.
    lock = getattr(app, "_buffer_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        if app._buffer is not None:
            return app._buffer
        if getattr(app, "_buffer_starting", False):
            return None
        last_fail = getattr(app, "_buffer_failed_at", 0.0)
        if last_fail and (_t.monotonic() - last_fail) < 30.0:
            return None
        try:
            import asyncio
            from mast.buffer.service import BufferService
            wal_path = project_root() / "experiments" / "vision_buffer.wal.sqlite"
            buf = BufferService(wal_path=wal_path, history_size=200)
            loop = _ensure_buffer_loop(app)
            app._buffer_starting = True
            fut = asyncio.run_coroutine_threadsafe(buf.start(), loop)

            def _on_started(f, _app=app, _buf=buf, _wal=wal_path):
                try:
                    f.result()
                    _app._buffer = _buf
                    # Wire the GUI's BufferService as the GLOBAL active buffer so
                    # the scan-progress vision monitor actually starts on the GUI
                    # path: the imaging skill calls get_active_buffer() to decide
                    # whether to spawn ScanVisionMonitor; previously ONLY
                    # pipeline/main.py called set_active_buffer(), so on the GUI
                    # path get_active_buffer() was None and the tip-/scan-quality
                    # vision model NEVER activated during a scan (user-reported).
                    try:
                        from mast.buffer.active import set_active_buffer
                        set_active_buffer(_buf)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("set_active_buffer failed: %s", exc)
                    # 修复项: physical E_STOP path. A published E_STOP event sets
                    # the orchestrator abort Event the instant it is emitted
                    # (synchronous hook on the publisher thread — no asyncio
                    # loop / LLM-call-boundary dependency), so a long-running
                    # composite aborts at its next check_abort() between steps.
                    try:
                        from mast.buffer.schemas import VisionEventType as _VET

                        def _estop_sets_abort(ev, _a=_app):
                            if ev.kind is not _VET.E_STOP:
                                return
                            ab = getattr(_a, "_orch_abort", None)
                            already = ab is not None and ab.is_set()
                            _rsn = str((ev.payload or {}).get("reason") or "?")
                            _det = str((ev.payload or {}).get("detail") or "")
                            _why = f"E_STOP 事件(reason={_rsn}){f': {_det}' if _det else ''}"
                            if ab is not None:
                                from mast.core.execution_context import mark_abort
                                mark_abort(ab, _why)
                                # LATCH: an E_STOP survives the next run-task's
                                # clear(). Before 2026-07-28 any new group run
                                # cleared this Event outright, re-arming the
                                # write commands of an in-flight PRIVATE-chat
                                # composite that the E_STOP had just halted.
                                try:
                                    _a._orch_abort_emergency = True
                                    _a._orch_abort_why = _why
                                except Exception:  # noqa: BLE001
                                    pass
                            try:
                                _lk = getattr(_a, "_orch_run_aborts_lock", None)
                                _runs = getattr(_a, "_orch_run_aborts", None) or {}
                                if _lk is not None:
                                    with _lk:
                                        for _rev in _runs.values():
                                            _rev.set()
                            except Exception:  # noqa: BLE001
                                pass
                            # Reverse bridge: also stop the manual/executor
                            # path (legacy composites poll executor's Event).
                            ex_ab = getattr(getattr(_a, "_executor", None),
                                            "_abort_event", None)
                            if ex_ab is not None:
                                ex_ab.set()
                            # Log once per latch — an E_STOP storm must not do
                            # per-event sync log I/O on the publisher thread.
                            if not already:
                                logger.warning(
                                    "E_STOP event → abort set (orchestrator%s, "
                                    "reason=%s)",
                                    "+executor" if ex_ab is not None else "",
                                    (ev.payload or {}).get("reason"),
                                )
                        _buf.register_critical_hook(_estop_sets_abort)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("E_STOP abort hook wiring failed: %s", exc)
                    # A separate run-scoped signal lets composites stop at
                    # step boundaries. It is not the global E_STOP latch,
                    # which blocks all new instrument actions. Keeping the
                    # mechanisms distinct preserves recovery operations.
                    try:
                        _buf.register_critical_hook(make_tip_halt_hook(_app))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("tip-halt hook wiring failed: %s", exc)
                    # The orchestrator was first built at setup() when _buffer was
                    # still None, so the instrument_control agent has NO buffer
                    # tools (read_latest_tip_status / get_scan_progress). Rebuild
                    # ONCE now that the buffer is live so those tools exist. This
                    # callback runs on the buffer-loop daemon thread (NOT a Gradio
                    # worker), so the rebuild never freezes the UI. Skip if a task
                    # is mid-flight (don't swap the graph under a running run).
                    try:
                        if not getattr(_app, "_orch_buffer_wired", False):
                            _task_busy = bool(
                                ((getattr(_app, "_agents_api_state", None) or {})
                                 .get("task") or {}).get("active"))
                            if not _task_busy and _app._orchestrator is not None:
                                _app._build_orchestrator()
                                _app._orch_buffer_wired = True
                                logger.info("Orchestrator rebuilt with live buffer "
                                            "→ IC agent gained buffer-read tools")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("post-buffer orchestrator rebuild skipped: %s", exc)
                    logger.info("BufferService started for Vision-buffer Tab "
                                "(async, wal=%s, persistent loop)", _wal)
                except Exception as exc:  # noqa: BLE001
                    _app._buffer_failed_at = _t.monotonic()
                    logger.warning("BufferService init failed: %s", exc)
                finally:
                    _app._buffer_starting = False

            fut.add_done_callback(_on_started)
        except Exception as exc:  # noqa: BLE001
            app._buffer_starting = False
            app._buffer_failed_at = _t.monotonic()
            logger.warning("BufferService kick failed: %s", exc)
    finally:
        if lock is not None:
            lock.release()
    # None until the done-callback publishes app._buffer on a later tick.
    return app._buffer


# ── context-compaction markers for BACKGROUND runs ──────
# The foreground 群聊 bridge (api/routes/orchestrator.py) does the same job with
# its own copies — it deliberately imports nothing from agents/** — but runtime.py
# already builds the middleware, so here the key comes from the source of truth.
def _background_compaction_event(msgs) -> "dict | None":
    """The compaction facts stamped on this state update's summary message."""
    try:
        from mast.agents._shared.compaction_mw import COMPACTION_META_KEY
    except Exception:  # noqa: BLE001 — no middleware ⇒ no compaction to report
        return None
    for m in (msgs or []):
        ak = getattr(m, "additional_kwargs", None)
        if isinstance(ak, dict):
            event = ak.get(COMPACTION_META_KEY)
            if isinstance(event, dict):
                return event
    return None


def _background_compaction_line(event: dict) -> str:
    """One operator-facing line, stating only the counts that actually arrived."""
    removed, kept = event.get("removed"), event.get("kept")
    if isinstance(removed, int) and removed > 0 and isinstance(kept, int):
        return (f"上下文压缩：较早的 {removed} 条消息已被摘要替代，"
                f"最近 {kept} 条保留原文")
    if isinstance(removed, int) and removed > 0:
        return f"上下文压缩：较早的 {removed} 条消息已被摘要替代"
    return "上下文压缩：此处较早的对话已被摘要替代"


def _tune_checkpoint_conn(conn) -> None:
    """WAL + bounded busy timeout on a checkpoint SQLite connection.

    Two other stores in this repo already do this (``chat/store.py``,
    ``logging/storage.py``); the orchestrator checkpointer did not, on the
    strength of a comment asserting that langgraph serialises its writers. It
    does not, and this database has several: a group run, a private chat and the
    background-run manager. WAL lets a reader and a writer coexist; the busy
    timeout turns "database is locked" — raised mid-run, inside the checkpointer
    — into a bounded wait. Best-effort: a pragma that will not apply must never
    stop the checkpointer from being built.
    """
    for pragma in ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=5000"):
        try:
            conn.execute(pragma)
        except Exception as exc:  # noqa: BLE001
            logger.debug("checkpoint pragma %r skipped: %s", pragma, exc)


def _time_mod_time() -> float:
    import time as _t
    return _t.time()


class CoreRuntime:
    """Gradio-free construction of the live MAST core."""

    _MODEL_ID_TO_UI = {
        "kimi-k2.6": "kimi-k2.6",
        "kimi-k2.7-code": "kimi-k2.7-code",
        "kimi-k3": "kimi-k3",
        "kimi-k2.5": "kimi-k2.6",          # legacy alias → closest UI choice
        "moonshot-v1-128k": "kimi-k2.6",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek-v4-pro",   # legacy alias (model retired)
        OPUS_4_7: "sonnet-4.6",   # no opus chip in the picker
        SONNET_4_6: "sonnet-4.6",
        HAIKU_4_5: "haiku-4.5",
        MINIMAX_M3: "minimax-m3",
        GLM_5_1: "glm-5.1",
        GLM_5_2: "glm-5.2",
    }


    _AGENTS_DEFAULT_THINKING = {
        "_supervisor": "max", "research_director": "max", "literature": "max",
        "experiment_design": "max", "instrument_control": "max",
        "data_processing": "max", "paper_writing": "max",
        "paper_review": "max", "buffer_summarizer": "max",
    }


    _AGENTS_IDS = (
        "_supervisor", "research_director",
        "literature", "experiment_design", "instrument_control",
        "data_processing", "paper_writing", "paper_review", "buffer_summarizer",
    )


    # 私聊入口。**两条入口给的工具面必须一样** —— 群聊有、私聊没有，会让
    # 「能不能做这件事」取决于用户点开的是哪个页面（ask-the-operator 刚栽过）。
    # 前端的 agent 选择器还没列出 research_director（见 registry.tsx），所以这一条
    # 目前是「接好了但还没有入口」；反过来（有入口没接线）会直接 ValueError。
    _CHAT_AGENT_MODULES = {
        "research_director": "mast.agents.research_director.graph",
        "literature": "mast.agents.literature.graph",
        "experiment_design": "mast.agents.experiment_design.graph",
        "data_processing": "mast.agents.data_processing.graph",
        "paper_writing": "mast.agents.paper_writing.graph",
        "paper_review": "mast.agents.paper_review.graph",
    }


    _UI_TO_MODEL_ID = {
        "kimi-k2.6": "kimi-k2.6", "kimi-k2.7-code": "kimi-k2.7-code",
        "kimi-k3": "kimi-k3",
        "deepseek-v4-pro": "deepseek-v4-pro",
        "sonnet-4.6": SONNET_4_6, "haiku-4.5": HAIKU_4_5,
        "deepseek-r1.5": "deepseek-v4-pro",
        "minimax-m3": MINIMAX_M3,
        "glm-5.1": GLM_5_1, "glm-5.2": GLM_5_2,
    }


    def __init__(self, config: MASTConfig):
        self.config = config
        self._pool = None
        self._registry = None
        self._executor = None
        self._planner = None
        self._storage = None
        self._monitor = None
        # Scan-map: monotonic ts of the last skill ACTIVITY (any skill, set by
        # the safety+skill recorders), so the manual-activity watcher can tell a
        # hand-driven state change apart from a skill-driven one — including
        # non-positioned skills like SetBias/SetSetpoint that change state but
        # don't drop a map marker.
        self._map_last_skill_ts: float = 0.0
        # 关停钩子：进程内会往连接池发命令的**外部入口**（外部 agent 网关的作业
        # 线程等）在这里登记，``shutdown()`` 第一步就调它们 —— 见 add_shutdown_hook。
        self._shutdown_hooks: list = []
        self._scan_map_timer = None  # live-refresh timer, set in build_ui()
        self._env_alarm_log: list[dict] = []   # recent environment over-limit events
        self._experiment_log = None
        self._safety = None
        self._state = None
        self._claude_client = None
        self._plan_store = None
        self._voice_client = None
        self._quickask = None
        # Real 6-agent orchestrator graph; populated lazily in setup()
        # iff the chat-model key (Kimi/DeepSeek/Anthropic) is available.
        # When None, /agents/sup/run_task falls back to MissionPlanner.
        self._orchestrator = None
        # True-parallel background runs (break the LangGraph super-step barrier
        # WITHOUT changing graph semantics): a long/independent task runs as a
        # SEPARATE orchestrator invocation on its own thread_id + checkpointer, so
        # the foreground chat stays responsive. Both lazily built on first use.
        self._background_runs = None       # BackgroundRunManager | None
        self._bg_orchestrator = None       # isolated backgroundable-agent graph
        # Per-agent-type run-duration history (item ②): feeds the smarter auto-
        # background decision (a consistently-slow type is worth detaching) and
        # persists across restarts. Lazily built by _ensure_run_stats. None until.
        self._run_stats = None             # RunStatsStore | None
        # Cognition layer (memory + phase-sharding + dreaming over the experiment
        # DB); built in setup() once self._storage exists. None until then.
        self._cognition = None
        # Coarse abort flag polled by _stream_orchestrator between super-steps
        # (set by POST /agents/sup/task_abort) AND shared into every
        # ExecutionContext so a running composite's check_abort() sees it
        # between sub-steps (修复项 — previously the context got a private dead
        # Event and the GUI abort never reached a running composite). Created
        # eagerly so the E_STOP buffer hook, _orch_context and _sup_run_task
        # can never race on lazy creation; _sup_run_task clear()s it per run.
        import threading as _threading
        self._orch_abort = _threading.Event()
        # Is the GLOBAL abort a latched EMERGENCY (E_STOP / watchdog), or an
        # ordinary "stop this run"? Starting a new group run clear()s the shared
        # Event, and that clear used to be indiscriminate: an E_STOP that had
        # just halted an in-flight PRIVATE-chat composite was silently undone the
        # moment anyone started a group task, re-arming its write commands
        # in place (审计 致命一(b)). An emergency latch now
        # survives a new run and needs an explicit clear.
        self._orch_abort_emergency: bool = False

        # 记录挂闩原因；空串表示未知。调用方不得把未知来源猜成用户中止。
        self._orch_abort_why: str = ""
        # Per-RUN stop Events. The shared _orch_abort is the GLOBAL emergency
        # latch and must keep its identity (the E_STOP hook and every
        # ExecutionContext hold it); a run's own stop lives here, so two
        # concurrent runs can be stopped independently instead of clobbering one
        # another's single slot.
        self._orch_run_aborts: dict[str, _threading.Event] = {}
        self._orch_run_aborts_lock = _threading.Lock()
        # Id of the orchestrator run currently streaming. Set by run-task before
        # it streams; read by _orch_context so composite step-progress sidecars
        # are scoped to THIS run and can never be resumed by a later one.
        self._orch_run_id: str = ""
        # Process-level checkpointer for the orchestrator graph. Created once in
        # _build_orchestrator (InMemorySaver) so interrupt()/resume works.
        self._orch_checkpointer = None
        # ── DANGEROUS-skill HITL gate (Step 1-4) ──────────────────────────
        # When _stream_orchestrator sees LangGraph yield a '__interrupt__' chunk
        # (HumanInTheLoopMiddleware paused the IC subgraph awaiting human
        # approval of a DANGEROUS skill) it publishes a normalized pending entry
        # here and BLOCKS the worker thread until the operator resolves it via
        # POST /interrupts/<event_id>/resolve. Blocking in app.py is allowed —
        # the no-sleep/no-block invariant constrains agents/**/graph.py nodes,
        # NOT this GUI worker. Structure:
        #   _orch_interrupts["lock"]      threading.Lock guarding the dicts
        #   _orch_interrupts["pending"]   event_id -> normalized pending dict
        #                                 (shape consumed by the React UI's
        #                                  normalizeInterrupt: event_id/agent_id/
        #                                  skill/params/rationale/allowed_decisions/
        #                                  kind/thread_id)
        #   _orch_interrupts["resolved"]  event_id -> LangGraph Decision dict
        #                                 (filled by /resolve, drained by worker)
        #   _orch_interrupts["events"]    event_id -> threading.Event (worker
        #                                 waits on it; /resolve sets it)
        import threading as _th_init
        self._orch_interrupts = {
            "lock": _th_init.Lock(),
            "pending": {},
            "resolved": {},
            "events": {},
        }
        # In-process Agents-API state — the operator-control + live-task store the
        # old MASTApp built in _mount_agents_api. Was NOT migrated, so interject /
        # hold / run-task task-slot / artifact edits all silently no-op'd (every
        # reader does getattr(self,"_agents_api_state",None) → None). Initialised
        # here so:
        #   * _orch_control_provider drains operator interjections each super-step
        #   * routes/agents_control hold/release/interject mutate a real dict
        #   * routes/orchestrator populates ["task"] (active/agent/handoffs) →
        #     CoreRuntime.agents_snapshot projects it for GET /api/agents/snapshot
        #   * routes/artifacts_edit persists operator edits into ["artifact_edits"]
        self._agents_api_state = {
            "lock": _th_init.RLock(),
            "holds": {},            # agent_id -> bool (operator pause)
            "interjects": [],       # list[{agent_id, text, t}] drained once
            "artifact_edits": {},   # artifact_id -> {body, t}
            "artifact_edit_history": {},  # artifact_id -> list[{body, t}]
            "task": {},             # current run-task slot (routes/orchestrator)
        }
        # True while a supervisor task is streaming — idle gate for background
        # dreaming (skip consolidation cycles while the orchestrator is busy) and
        # for the wake scheduler (never decide anything while the mainline is
        # deciding — two decision streams would fight).
        self._orch_running = False
        # Cost ceiling for the CURRENT foreground run. The orchestrator graph is
        # built once and reused, but a budget belongs to a run, so the graph gets a
        # stable probe (_budget_remaining_usd) and this attribute is what the probe
        # reads. None = no run in flight / no ceiling configured → the gate stays
        # inert, exactly as it behaved before it was wired up.
        self._run_meter = None
        # Wakes parked agents when what they waited for lands. Its own daemon thread
        # (it may block — it is not a graph node), started in setup().
        self._wake_scheduler = None
        self._system_check_results: list[dict] = []
        self._last_scan_path: str | None = None
        # Persistent UI settings (JSON-backed). Created in setup(); kept as a
        # declared attribute so handlers can null-check it before setup runs.
        self._settings = None
        # Single reusable temp .wav path for assistant TTS playback. One file is
        # rewritten per reply and cleaned up at exit instead of leaking a fresh
        # NamedTemporaryFile(delete=False) on every spoken reply .
        self._tts_wav_path: str | None = None
        # Phase 7 (RISK A.5): in-process BufferService for the Vision tab.
        # Lazily initialised on first Vision-buffer Tab refresh so unused
        # installs don't pay the WAL setup cost.
        self._buffer = None
        # Long-lived asyncio loop (daemon thread) that owns the BufferService so
        # its WAL writer keeps running for the process lifetime . Lazily
        # created by _ensure_buffer_loop on first Vision-buffer use.
        self._buffer_loop = None
        # Async-init guards so the BufferService warm-up NEVER blocks a Gradio
        # queue worker (the 1.5 s vision timers call _ensure_buffer_for_gui on
        # every tick). _buffer_starting gates concurrent kicks; _buffer_failed_at
        # holds a monotonic cooldown stamp so a broken/locked WAL can't retry-
        # storm the queue every 1.5 s and freeze the whole UI. See
        # _ensure_buffer_for_gui.
        import threading as _threading_mod
        self._buffer_starting = False
        self._buffer_failed_at = 0.0
        self._buffer_lock = _threading_mod.Lock()
        # Re-entry registry for _offload(): keys of handlers whose blocking work
        # is currently running on a daemon thread, so a second click can't stack
        # another worker-pinning job (see _offload).
        self._offload_busy: set[str] = set()
        self._offload_lock = _threading_mod.Lock()
        # Serialize _build_orchestrator across its (now several) off-worker
        # callers — the buffer-loop _on_started rebuild, the per-agent override
        # offload thread, and the anyio.to_thread _set_model route can otherwise
        # rebuild the LangGraph concurrently and corrupt shared state.
        self._orch_build_lock = _threading_mod.Lock()
        # -- Deferred agent-tool-table rebuild  -------------
        # A rebuild is refused while a task is streaming, and that part is
        # correct: ExecutionContext.run looks the skill class up in the registry
        # on EVERY sub-step (execution_context.py:366), so swapping mid-run
        # would change the second half of a composite whose first half already
        # ran under the old code.
        #
        # What was NOT correct: "refused" was the END of the story.
        # _request_composite_rebuild returned a Chinese sentence and nobody ever
        # came back. The operator changed an envelope, saw the "task running"
        # message, and the agents kept the old tool table until someone thought
        # to restart. Now it is remembered and drained the moment the task ends.
        self._pending_agent_rebuild: dict | None = None
        self._pending_rebuild_lock = _threading_mod.Lock()
        # One-time flag: rebuild the orchestrator after the async BufferService
        # comes up so the instrument_control agent gains its buffer-read tools
        # (built at setup() when _buffer was still None). See _ensure_buffer_for_gui.
        self._orch_buffer_wired = False
        # Set True whenever a per-agent model/thinking override is written; the
        # coalesced "orchestrator_rebuild" offload loops until it's False so a
        # change made DURING an in-flight rebuild isn't silently dropped.
        self._orch_override_dirty = False


    def _setting(self, key: str, default=None):
        """One persisted setting, read LIVE, with no way to spell the read wrong.

        Prefers ``self._settings`` — the instance ``POST /api/settings`` writes
        through (``bootstrap`` wires ``ctx.settings_store = rt._settings``), so a
        change made a second ago is already visible here. Falls back to
        ``settings_store_for_runtime()`` for callers reached before/without
        ``setup()``; that still reads the operator's file, it just isn't the same
        object as a concurrent writer's.

        Why a helper instead of the one-liner at each call site: six call sites in
        this file spelled it ``SettingsStore().get(key)`` — no ``config_dir``,
        which is a required positional — inside a broad ``except``. Every one of
        them raised TypeError and returned its fallback, measured 2026-08-10, and
        because the fallbacks equalled the defaults nothing ever looked wrong.
        A helper cannot be called with the wrong number of arguments.
        """
        st = getattr(self, "_settings", None)
        if st is None:
            from mast.webui.settings_store import settings_store_for_runtime
            st = settings_store_for_runtime()
        return st.get(key, default)

    def _current_operating_mode(self) -> str:
        """Current global operating mode ("safe" | "semi" | "auto").

        SOURCE OF TRUTH = the persisted SettingsStore (``autonomy_mode`` key).
        The API's ``ctx.settings_store`` shares this store BY IDENTITY (bootstrap
        wires ``settings_store=rt._settings``), so a ``POST /api/settings`` write
        is picked up here live — no agent rebuild. Threaded to the IC agent's
        tip-processing gate as ``get_mode`` (SafetyGate Layer-0d / pulse HITL /
        belief); ``OperatingMode.coerce()`` in the middlewares parses the string.
        Fails open to "auto" (prior behaviour) if settings are absent/corrupt."""
        st = getattr(self, "_settings", None)
        if st is None:
            return "auto"
        try:
            return st.get("autonomy_mode", "auto") or "auto"
        except Exception:
            return "auto"


    def setup(self) -> None:
        """Initialize all MAST components."""
        # ── Persistent UI settings (closes the "settings reset on restart" gap) ──
        # Hydrate the live MASTConfig from <root>/config/ui_settings.json BEFORE
        # anything reads it (ConnectionPool below uses self.config.nanonis; the
        # LLM/voice clients later read self.config.llm/voice). Best-effort: a
        # missing/corrupt file degrades to defaults without raising. Theme/font
        # are client-side (CSS/localStorage) and are seeded into the components
        # in build_ui from self._settings.load(), not applied here.
        try:
            from mast.webui.settings_store import (
                SettingsStore,
                default_config_dir,
                set_process_store,
            )
            self._settings = SettingsStore(default_config_dir(project_root()))
            # Publish it. Code with no ``self`` to reach for (the tip-conditioning
            # resolver, module-level helpers) asks settings_store_for_runtime()
            # and must get THIS object — the one POST /api/settings mutates —
            # rather than a second instance holding its own stale snapshot.
            set_process_store(self._settings)
            applied = self._settings.apply_to_config(self.config)
            if applied:
                logger.info("UI settings hydrated from disk: %s",
                            ", ".join(sorted(applied)))
        except Exception as exc:
            logger.warning("SettingsStore init/apply failed (using defaults): %s", exc)
            self._settings = None

        # Vision tip-quality thresholds: hydrate the process-level holder from the
        # persisted vision_thresholds so tip discrimination uses the operator's
        # cuts from the first assessment. Separate best-effort block — the vision
        # layer reads the holder lock-free and never imports settings. (module.py
        # has no top-level torch, so this import stays cheap.)
        try:
            from mast.vision.thresholds import set_thresholds
            if self._settings is not None:
                set_thresholds(self._settings.get("vision_thresholds"))
        except Exception as exc:
            logger.debug("vision thresholds hydrate skipped: %s", exc)

        # Classical tip-quality tool knobs: same hydrate as vision_thresholds so the
        # network-free tools use the operator's per-instrument cuts from the start.
        try:
            from mast.vision.classical_thresholds import set_classical_thresholds
            if self._settings is not None:
                set_classical_thresholds(self._settings.get("classical_thresholds"))
        except Exception as exc:
            logger.debug("classical thresholds hydrate skipped: %s", exc)

        # Operating mode: bind the live-read holder the NON-agent layers use
        # (vision publisher thread, buffer service, composite skills). The agent
        # path keeps its explicit get_mode callables; this is the same value,
        # reachable from code that has no agent context. Binding is what makes
        # SAFE's "the tip is fine" contract hold at the verdict producers instead
        # of only at the agent's door. Unbound (tests, headless pipeline.main)
        # means "unknown" → no override, i.e. pre-2026-08-01 behaviour.
        try:
            from mast.core.operating_mode import bind_mode_source
            bind_mode_source(self._current_operating_mode)
        except Exception as exc:
            logger.debug("operating-mode holder bind skipped: %s", exc)

        # Current-monitor knobs: same live-read holder pattern. Hydrated before
        # the daemon starts so the very first segment uses the operator's
        # settings — including cm_enabled, which decides whether it runs at all.
        try:
            from mast.monitoring.thresholds import set_monitor_thresholds
            if self._settings is not None:
                set_monitor_thresholds(self._settings.get("current_monitor"))
        except Exception as exc:
            logger.debug("current monitor thresholds hydrate skipped: %s", exc)

        # Environment-history knobs: same live-read holder. Must be hydrated
        # before the environment monitor starts, because eh_enabled decides
        # whether its history sink accumulates at all and eh_raw_keep_days is
        # the only knob in this system that DELETES anything.
        try:
            from mast.envhistory.thresholds import set_env_history_thresholds
            if self._settings is not None:
                set_env_history_thresholds(self._settings.get("env_history"))
        except Exception as exc:
            logger.debug("env history thresholds hydrate skipped: %s", exc)

        # conduct 指挥线程的旋钮:同一个 live-read holder。必须在
        # ``start_service`` 之前 hydrate —— ``cd_enabled`` 决定的是那条线程**建不
        # 建**,而不是建好之后跑不跑。默认 0(关),关着的时候连 store 都不建,
        # 于是「关着」逐字节等于这个功能落地之前。
        try:
            from mast.conduct.settings import set_conduct_knobs
            if self._settings is not None:
                set_conduct_knobs(self._settings.get("conduct"))
        except Exception as exc:
            logger.debug("conduct knobs hydrate skipped: %s", exc)

        # Experiment default-parameter preferences : hydrate the process-level
        # holder from the persisted experiment_defaults so IC / experiment_design see
        # the operator's preferred scan size / speed / setpoint / bias from the first
        # turn. Live-read holder like vision_thresholds; the agents never import
        # settings. Best-effort — a missing/blank value just leaves the holder empty
        # (middleware no-op).
        try:
            from mast.agents._shared.experiment_prefs import set_prefs
            if self._settings is not None:
                set_prefs(self._settings.get("experiment_defaults"))
        except Exception as exc:
            logger.debug("experiment defaults hydrate skipped: %s", exc)

        # Instrument profile (换样品退针方向 / lock-in dI/dV 参数 / 到样品标定值):
        # hydrate the holder from persisted settings AND inject a persist sink so a
        # learned dI/dV-at-contact written at runtime (set_calibration on a verified
        # 进针) survives to the next run/session. Live-read holder like the two
        # above; the skill layer reads it, the middleware injects it. Best-effort —
        # a missing value leaves spec defaults in effect.
        try:
            from mast.core import instrument_profile as _iprof
            if self._settings is not None:
                _iprof.set_profile(self._settings.get("instrument_profile"))
                _iprof.set_persist_sink(
                    lambda profile: self._settings.update(instrument_profile=profile))
        except Exception as exc:
            logger.debug("instrument profile hydrate skipped: %s", exc)

        # 粗动驱动电压的本机声明:hydrate 进 live-read holder。
        # **失败记 warning 而不是 debug**:hydrate 失败 = 声明读不到 = 一切驱动写入
        # 与粗动移动都会被拒绝。这是 fail-closed 的正确方向,但用户会看到「明明填过
        # 却说没声明」,所以必须在日志里留下痕迹,而不是安静地退回未声明态。
        try:
            from mast.core import coarse_drive as _cdrive
            if self._settings is not None:
                _cdrive.set_declaration(self._settings.get("coarse_drive"))
                _cdrive.set_persist_sink(
                    lambda decl: self._settings.update(coarse_drive=decl))
        except Exception as exc:
            logger.warning("coarse drive declaration hydrate FAILED (粗动将被拒绝): %s", exc)

        # 扫描参数档位表(按尺度):hydrate 进 live-read holder,让 ScanAt /
        # scan_resolver 从第一次调用就用用户的表。**这一处刻意把失败记成
        # warning 而不是 debug**:表存在但加载失败意味着系统正用出厂参数跑用户
        # 以为已经改过的实验,而两者的差别在界面上看不出来。
        try:
            from mast.core import scan_policy as _spolicy
            if self._settings is not None:
                stored = self._settings.get("scan_policy")
                if stored:
                    _spolicy.set_policy(stored)
        except Exception as exc:
            logger.warning(
                "扫描档位表加载失败,本次运行将使用出厂默认参数: %s", exc)

        # 自定义 Z 参数组只保存按名解析所需的用户配置，与 instrument_profile
        # 和扫描档位表分别负责各自来源；sink 持久化 CreateZCtrlPreset 的结果。
        # AutoApprovalNoticeMiddleware 在操作后记录解析后的参数，不是操作前
        # 审批。若需拒绝异常数值，应在写入前通过明确的量级约束实现。
        try:
            from mast.core import zctrl_presets as _zpresets
            if self._settings is not None:
                _zpresets.set_presets(self._settings.get("zctrl_presets"))
                _zpresets.set_persist_sink(
                    lambda presets: self._settings.update(zctrl_presets=presets)
                )
        except Exception as exc:
            logger.warning("自定义 Z 参数组加载失败: %s", exc)

        # Vision warmup: preload the learned stm_quality_v1 scorer (frozen DINOv3 +
        # ridge head, ~87 s cold) in the BACKGROUND so the first unified assess()
        # isn't slow. Daemon thread — never blocks startup; best-effort (a missing
        # scorer just stays lazy and assess(use_learned=…) degrades gracefully). The
        # scorer is a separate process-wide singleton; later inference is lock-
        # guarded inside VisionModule, so the background load is safe.
        try:
            import threading as _th_warm

            def _warm_vision() -> None:
                try:
                    from mast.vision.quality_model import get_scorer
                    get_scorer().preload()
                    logger.info("stm_quality_v1 scorer preloaded (vision warmup)")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("vision scorer preload skipped: %s", exc)

            _th_warm.Thread(target=_warm_vision, name="mast-vision-warmup", daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.debug("vision warmup thread not started: %s", exc)

        # Optional hardware modules (KPFM / 多探针 / OsciHR / …). Hydrate the
        # holder BEFORE _build_orchestrator() (line ~722) — the tool-list gate
        # reads it while wrapping skills, so hydrating after the build would give
        # the agent one session with the wrong tool list. Every module defaults
        # OFF, so a missing/failed read is the safe state, not a broken one.
        try:
            from mast.skills.hardware_modules import set_enabled
            if self._settings is not None:
                set_enabled(self._settings.get("hardware_modules"))
        except Exception as exc:
            logger.debug("hardware modules hydrate skipped: %s", exc)

        # Advanced capabilities (script-file I/O / quit Nanonis / multi-pass files /
        # blocking wait). Same ordering requirement: BEFORE _build_orchestrator(), and
        # every one defaults OFF so a failed read is the safe state.
        try:
            from mast.skills.advanced_capabilities import (
                set_enabled as set_advanced_enabled,
            )
            if self._settings is not None:
                set_advanced_enabled(self._settings.get("advanced_capabilities"))
        except Exception as exc:
            logger.debug("advanced capabilities hydrate skipped: %s", exc)

        try:
            import mast.core.nanonis_patch as _patch  # noqa: F401
            logger.info("Nanonis monkey-patch applied")
        except Exception as exc:
            logger.warning("Could not apply nanonis patch: %s", exc)

        connected = 0
        try:
            from mast.core.connection import ConnectionPool
            self._pool = ConnectionPool(self.config.nanonis)
            results = self._pool.connect_all()
            connected = sum(1 for v in results.values() if v)
            logger.info("ConnectionPool: %d/%d ports connected", connected, len(results))
        except Exception as exc:
            logger.warning("Could not create ConnectionPool: %s", exc)

        # 启动过程不得静默覆盖仪器上的扫描速度，即使当前没有扫描也不能写默认值。
        # 默认档位只用于 MAST 自己发起的扫描，由 scan_policy 与 ConfigureScan 处理。
        # 启动无仪器写入的契约由 test_startup_no_instrument_writes.py 验证。

        try:
            from mast.core.safety import SafetyGuard
            from mast.core.state import InstrumentState
            self._safety = SafetyGuard(self.config.safety)
            if self._pool is not None:
                self._state = InstrumentState(self._pool)
                # Run state polling on a daemon thread so GUI handlers
                # always read from ``state.snapshot()`` (instant, cached)
                # instead of having ``state.refresh()`` block the main
                # thread with 8 sequential safe_call(monitor) round-trips.
                try:
                    self._state.start_background_refresh(interval_s=1.0)
                except Exception as exc:
                    logger.warning("Could not start state background refresh: %s", exc)
        except Exception as exc:
            logger.warning("Could not create safety/state: %s", exc)

        try:
            from mast.core.registry import SkillRegistry
            self._registry = SkillRegistry()
            count = self._registry.discover()
            logger.info("SkillRegistry: %d skills", count)
            # Register declarative composite specs the user built/cloned in the
            # 复杂技能 tab so they become real, runnable skills (seeds built-in
            # templates on a fresh install). Best-effort — never blocks startup.
            try:
                from mast.skills.composite.loader import load_spec_skills
                loaded = load_spec_skills(self._registry)
                if loaded:
                    logger.info("Registered %d declarative composite(s): %s",
                                len(loaded), ", ".join(loaded))
            except Exception as exc:
                logger.warning("Could not load declarative composites: %s", exc)
            # P3-A: curated agent @tools (DP analysis / LIT search) become
            # registry skills so workflow step nodes can call them through
            # the same ExecutionContext.run choke point.
            try:
                from mast.skills.composite.tool_skills import (
                    register_workflow_tool_skills,
                )
                register_workflow_tool_skills(self._registry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent tool-skills registration failed: %s", exc)
            # P5-A: user-authored custom skills from the DATA root (explicit
            # enabled.json allowlist — dropping a .py never auto-executes).
            try:
                from mast.skills.custom_loader import load_custom_skills
                load_custom_skills(self._registry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("custom skills load failed: %s", exc)
            # 覆盖层**最后**加载 —— 它的定义就是「最后说了算」。放在
            # set_live_registry 之前，是为了让首帧 UI 目录和 agent 工具表从一开始
            # 就是覆盖之后的状态（否则开机瞬间的目录是内置版，随后又变）。
            #
            # 失败用 error 而不是 warning：「以为覆盖了其实没有」是这个功能的头号
            # 事故形态，它必须比邻居那几条更响。
            try:
                from mast.skills.overlay.loader import apply_overlays
                _ovl = apply_overlays(self._registry)
                if _ovl.changed or _ovl.failed:
                    logger.info("技能覆盖层：%s", _ovl.describe())
            except Exception as exc:  # noqa: BLE001
                logger.error("技能覆盖层加载失败（本次全部走内置版）：%s",
                             exc, exc_info=True)
            # 修复项: wire the live registry + agent-tool refresh so the 复杂技能
            # tab's clone/restore/delete hot-(un)register without a restart.
            # OUTSIDE the load try-block — a failed spec load must not silently
            # disable hot-registration for the whole session (review nit).
            try:
                from mast.webui.composite_panel import (
                    set_agent_refresh, set_live_registry,
                )
                set_live_registry(self._registry)
                set_agent_refresh(self._request_composite_rebuild)
            except Exception as exc:
                logger.warning("composite hot-register wiring failed: %s", exc)
        except Exception as exc:
            logger.warning("Could not create SkillRegistry: %s", exc)

        try:
            from mast.core.executor import SkillExecutor
            if all(c is not None for c in [self._pool, self._registry, self._safety, self._state]):
                self._executor = SkillExecutor(
                    pool=self._pool, registry=self._registry,
                    safety=self._safety, state=self._state,
                )
                # Start the tip-crash safety watchdog (current anomaly →
                # SafeRetract via emergency port). 审查 HIGH: it was
                # implemented but never started on the v2 path, leaving the
                # hardware safety net dead-wired. It polls harmlessly (gets
                # errors) until Nanonis is connected, then guards the tip.
                self._ensure_watchdog()
        except Exception as exc:
            logger.warning("Could not create SkillExecutor: %s", exc)

        # Tunnelling-current monitor: segmented acquisition on the data role,
        # on by default (``cm_enabled`` = 1.0).
        #
        # 它**不取仪器令牌**，所以不会跟别的入口抢仲裁 —— 但「不取令牌」不等于
        # 「不发命令」。2026-08-02 更正：这里原本写着「Read-only and advisory —
        # it takes no instrument token and issues no commands」，后半句是假的。
        # ``_ScopePump.configure()`` 启动时会往示波器写四条：
        # ``*_Run`` / ``*_ChSet``（改指隧道电流）/ ``*_TrigSet``（强制 Immediate）
        # / ``*_TimebaseSet``。停止时按原值还原（见 ``pump.py`` 的 ``restore``），
        # 但触发模式还不回去 —— ``Osci1T_TrigGet`` 在 nanonis_spm 1.0.9 里响应
        # 规格是空的、必然误解析，读不到原值就无从还原。
        #
        # **2026-08-09 起占的是哪一台变了**：默认策略从 Osci1T 换成 **Osci2T**
        # （``cm_use_osci2t``，见 monitoring/thresholds.py 的说明；模块没加载会
        # 自动退回 1T）。采样率一模一样，换的是「一次往返拿回多少」。
        # 对用户的净效果：**Osci1T 还给他了，改成长期占着 Osci2T** —— 后者
        # 更值钱（双通道），这是这次切换里唯一对他不利的一面，如实记在这里。
        # 第二通道写回他原来那一路，我们不往上放自己的东西
        # （``Osci2TPump._channels_to_write``）。
        #
        # Own try/except so a monitoring fault cannot stop the runtime coming up.
        try:
            from mast.monitoring.service import start_service as _start_current_monitor
            _start_current_monitor(self)
        except Exception as exc:
            logger.warning("Could not start the current monitor: %s", exc)

        try:
            from mast.logging.storage import ExperimentStorage
            from mast.logging.experiment_log import ExperimentLog, set_active_log
            self._storage = ExperimentStorage(self.config.db_path)
            self._experiment_log = ExperimentLog(self._storage)
            # Register the singleton so low-level skills (StartScan etc.)
            # can build basenames from the active experiment / sample.
            set_active_log(self._experiment_log)
            # Manual-activity watcher: records hand-driven Nanonis frame/tip
            # moves into the experiment-record map (requirement 3). Best-effort.
            try:
                self._start_map_activity_watcher()
            except Exception as exc:  # noqa: BLE001
                logger.debug("map activity watcher not started: %s", exc)
            # Graceful teardown on a normal/signal-driven exit: close the Nanonis
            # pool + stop daemons so the fragile TCP ports aren't abandoned
            # mid-transaction. atexit won't fire on a hard
            # TerminateProcess — the /api/admin/shutdown endpoint + launcher cover
            # that path — but it does cover normal exits / SIGTERM.
            try:
                import atexit
                atexit.register(self.shutdown)
            except Exception:  # noqa: BLE001
                pass
            # Cognition layer: memory + phase-sharding + dreaming, all over the
            # SAME experiment DB (so a project's cognitive context exports with
            # its record). Best-effort — the GUI degrades gracefully if absent.
            try:
                from mast.agents._shared.cognition import CognitionContext
                self._cognition = CognitionContext(self.config.db_path, author="user")
                # Upgrade the rule-based phase summariser + dream consolidator to
                # real LLM versions (provider-portable, plain .invoke). Best-effort:
                # both backends fall back to rule-based if the LLM raises.
                try:
                    from mast.agents._shared.cognition_llm import (
                        make_llm_dream_consolidator, make_llm_phase_summarizer,
                    )
                    from mast.agents._shared.models import make_chat_model
                    _cog_llm = make_chat_model("orchestrator", max_tokens=2048)
                    self._cognition.set_summarizer(make_llm_phase_summarizer(_cog_llm))
                    self._cognition.set_consolidator(make_llm_dream_consolidator(_cog_llm))
                except Exception as _cog_llm_exc:  # noqa: BLE001
                    logger.debug("cognition LLM injection skipped: %s", _cog_llm_exc)
                # Start the async background "dreaming" consolidation pass. It only
                # READS the experiment record + WRITES memory rows (never touches
                # the instrument), so it's safe to run alongside live tasks; it is
                # idle-gated to skip cycles while the orchestrator is busy and is a
                # daemon thread (dies with the process). On-demand dreaming is also
                # available from the 记忆 tab.
                self._cognition.start_dreaming(
                    should_dream=lambda: not getattr(self, "_orch_running", False))
            except Exception as exc:
                logger.warning("CognitionContext init failed: %s", exc)
                self._cognition = None
            # Wake scheduler — same idle predicate as dreaming, same daemon-thread
            # shape, and started here beside it because they answer the same
            # question ("what may the system do while nobody is driving it?").
            # Never wakes anything unless a park exists, so an install that never
            # enables activation gating pays one no-op tick a minute.
            try:
                self._ensure_wake_scheduler()
            except Exception as exc:  # noqa: BLE001
                logger.warning("wake scheduler start failed: %s", exc)
            # Restore the scope pointer from the previous session.
            #
            # 2026-07-28: this used to scan for the first ``status == 'running'``
            # row and adopt it. Experiments no longer have a lifecycle state
            # (「有的实验可能过了十年重启」), so "which one is current" is answered
            # by the explicit persistent pointer instead of being guessed from a
            # column that everything else also wrote to. restore_scope() validates
            # the pointer, drops a sample that no longer belongs to its experiment,
            # writes the corrected pointer back (self-healing), and seeds once on
            # first launch after the upgrade. It never raises.
            if self._storage is not None and self._experiment_log is not None:
                try:
                    sc = self._experiment_log.restore_scope()
                    if sc.experiment_id:
                        logger.info("Restored scope: 实验 %s (%s) / 样品 %s",
                                    sc.experiment_name, sc.experiment_id[:8],
                                    sc.sample_name or "(未选)")
                except Exception as exc:
                    logger.debug("Could not restore scope: %s", exc)

            # 当前针尖读进 holder(注入中间件与 skill 层都 live-read 它)。
            #
            # 没有 restore_scope 那样的校验/自愈链,因为不需要:当前针尖不是一个
            # 可能悬空的指针,而是「唯一 removed_at IS NULL 的那一行」—— 查询
            # 本身就是真源(见 storage.py 建表注释)。
            if self._storage is not None:
                try:
                    from mast.logging import tip_registry as _tips
                    row = _tips.hydrate(self._storage)
                    if row:
                        logger.info("当前针尖: %s (%s/%s)",
                                    row.get("name") or row.get("id", "")[:8],
                                    row.get("material") or "材料未记录",
                                    row.get("form") or "stm_wire")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("读取当前针尖失败(按未登记处理): %s", exc)
        except Exception as exc:
            logger.warning("Could not create storage: %s", exc)

        # v2 records (live): open the v2 store + a session experiment so live
        # skill runs land as v2 actions and tip-shape figures reach the Records
        # tab (the writer was previously dormant). Best-effort — never blocks.
        try:
            from mast.logging.v2.live import open_live_v2
            self._v2_repos, self._v2_eid = open_live_v2()
        except Exception as exc:
            logger.warning("live v2 records init failed: %s", exc)
            self._v2_repos, self._v2_eid = None, None

        # Training/usage trajectory sink (#8 P1). Wire it to the live v2 repos so
        # the skill/agent-turn recorders (_skill_trace_recorder / _turn_trace_
        # recorder) actually persist. In the Gradio→TS rewrite nobody set
        # _trace_sink, so the whole trajectory log was silently dead (review
        # 2026-07-03). Best-effort; _active_traj is opened per task/turn.
        self._trace_sink = None
        self._active_traj = None
        try:
            if self._v2_repos is not None:
                from mast.logging.v2.trace_sink import QueuedTraceSink
                self._trace_sink = QueuedTraceSink(self._v2_repos)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("trace sink init failed: %s", exc)
            self._trace_sink = None

        # Data ingest sink (2026-07-28): copies Nanonis artefacts into the
        # current sample's raw/ folder. Single background daemon thread; the
        # skill return path only does one put_nowait, so a 200 MB .3ds copy can
        # never stall a scan or the UI. Design:
        # docs/v2/design/experiment_folder_persistence.md §7
        self._ingest = None
        try:
            self._init_experiment_folders()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("experiment folder init failed: %s", exc)

        try:
            from mast.environment.monitor import EnvironmentMonitor
            from mast.environment.autodetect import build_environment_sensors
            # Real DL-7 vacuum gauges / Lakeshore temperature monitors are
            # auto-detected (or read from environment_sensors.json); kinds not
            # found fall back to "unavailable" placeholders so a hardware-free
            # install still runs (requirement #7 — no gauge/thermometer no crash).
            # Nanonis-backed sensors first: their names must be reserved so
            # build_environment_sensors() does not also emit a placeholder under
            # the same name (the dedup would then rename the REAL one).
            nanonis_sensors = self._build_nanonis_env_sensors()
            sensors = build_environment_sensors(
                reserved_names=[s.name() for s in nanonis_sensors])
            sensors.extend(nanonis_sensors)
            # 环境读数同时写 CSV（实验级按天轮转 + 当前样品切片），让用户能
            # 用 Excel/Origin 直接画温度曲线。sink 自己吞异常并自我禁用 60 秒 ——
            # 监控循环的首要职责是告警，写文件失败不能让它停摆。
            self._env_csv = None
            try:
                if self._env_csv_enabled():
                    from mast.environment.csv_sink import EnvironmentCsvSink
                    self._env_csv = EnvironmentCsvSink(self._env_csv_scope)
            except Exception as exc:  # noqa: BLE001
                logger.debug("env csv sink init failed: %r", exc)
            # 环境历史记录器：把 2 秒读数流聚合成永久的统计桶，给噪声谱定期存
            # 快照，并给 environment_log 的原始行加上保留期（这张表此前只增不减
            # 且零读取方）。它**不新起采样线程** —— sink 挂在下面这条 2 秒循环
            # 上，谱挂在电流监控的段流上，清扫是到期才起的有界线程。
            # 设计：docs/v2/design/environment_history.md
            self._env_history = self._build_env_history_recorder(sensors)
            sinks = [s for s in (self._env_csv,
                                 getattr(self._env_history, "sink", None)) if s]
            self._monitor = EnvironmentMonitor(
                sensors=sensors, storage=self._storage, interval_s=2.0,
                on_alarm=self._on_env_alarm,
                sinks=sinks or None,
                scope_provider=self._current_env_scope,
            )
            # Feed the coarse-motion vacuum interlock. Injected as a callable so
            # mast.core.vacuum_interlock never imports the environment layer and
            # stays a pure function under test. It carries the sensor's CLASS
            # NAME on purpose: a placeholder reports value=0.0 with
            # status="unavailable", and an interlock that reads only the number
            # would turn "no gauge installed" into "perfect vacuum".
            try:
                from mast.core import vacuum_interlock as _vac
                _vac.set_pressure_source(self._latest_pressure_sample)
                _vac.set_audit_sink(self._log_vacuum_interlock_event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("vacuum interlock wiring failed (粗动将被拒绝): %s", exc)
            # Temperature for the coarse-move odometer. Recorded beside every
            # relocation because the same step count travels several times
            # further at 300 K than at 4 K — two entries without temperatures
            # cannot be compared with each other.
            try:
                from mast.core import coarse_map_provider as _cmp
                _cmp.set_temperature_source(self._latest_temperature_k)
                # And the stage-scale map itself, so RelocateCoarseXY can check
                # its destination against where the stage has already been
                # without importing the runtime.
                _cmp.set_marker_source(self._coarse_map_inputs)
            except Exception as exc:  # noqa: BLE001
                logger.debug("coarse map provider wiring failed: %s", exc)
            # 温度的**公共**只读口。同一形状的注入：技能层从
            # ``mast.core.temperature`` 取数，因此 ``mast/skills/**`` 不必 import
            # 活的 app 对象。没接上时那边回 ``no_source``（与「仪器不给数」分开）。
            try:
                from mast.core import temperature as _temp
                _temp.set_source(lambda ch=None: self.latest_temperature(ch))
                _temp.set_channels_source(self.temperature_channels)
            except Exception as exc:  # noqa: BLE001
                logger.debug("temperature provider wiring failed: %s", exc)
            # Start the background archive+alarm loop only when at least one REAL
            # hardware sensor is present, so an instrument-free install doesn't
            # spam environment_log with placeholder "unavailable" rows. The Lab
            # Console still reads sensors live on each panel refresh either way.
            if self._has_real_env_sensors(sensors):
                self._monitor.start()
                logger.info("EnvironmentMonitor loop started (%d sensor(s), real hardware present)",
                            len(sensors))
            else:
                logger.info("EnvironmentMonitor created with %d placeholder sensor(s) "
                            "(no real hardware detected; background loop idle)", len(sensors))
            # Stop the loop + close serial handles at process exit (review 2.1.13
            # #24) so a graceful shutdown doesn't leak COM ports / the thread.
            def _stop_env_monitor():
                m = getattr(self, "_monitor", None)
                if m is not None:
                    try:
                        m.stop()
                    except Exception:
                        pass
            atexit.register(_stop_env_monitor)
        except Exception as exc:
            logger.warning("Could not create EnvironmentMonitor: %s", exc)
            self._monitor = None

        # PlanStore for expert-mode plan management.
        #
        # ``plans_dir`` 从 2026-07-29 起只是**退路**：有实验归属的计划把正文写成
        # 该实验文件夹里的一个文档（``plans/<doc-dir>/vNNN.md``，定义修订才发版本，
        # 进度走 progress.jsonl + progress.md）。只有没有实验归属、或实验行已经不
        # 存在的计划才落这里。保留它是因为「计划总得有个地方放」，不是因为还有谁
        # 从这个目录读。设计：docs/v2/design/document_and_library_management.md §3.7
        try:
            from mast.planning.plan_store import PlanStore
            self._plan_store = PlanStore(
                db_path=self.config.db_path,
                plans_dir=self.config.experiments_dir / "plans",
            )
            logger.info("PlanStore created (fallback plans dir: %s)",
                        self.config.experiments_dir / "plans")
        except Exception as exc:
            logger.warning("Could not create PlanStore: %s", exc)

        # ── conduct 指挥线程 (多天 conduct;**默认关**) ─────────────────
        #
        # 生命周期挂点与 ``executor.start_watchdog`` 同级,但**权限根本不同**:
        # watchdog 是紧急救济,有权走裸 ``urgent_call``;Director 是常规驱动,
        # 每一步都走 ``ExecutionContext.run`` 的完整安全管道(registry → 状态刷新
        # → SafetyGuard → 快照 → instrument_lock),**永远没有裸 TCP 权**。
        # 把这两句分开写,是因为「反正都是代码线程驱动仪器」这个念头会把它们混成
        # 一个,而混起来的那一刻 conduct 就获得了绕过安全门的能力。
        #
        # 放在这里是因为它要 registry / state / pool 都在(上面几段建好的),而且
        # 要在 ``start_service`` 里**先做重启清算再起线程** —— 顺序反过来的话,
        # 线程会在清算之前先推进一步,而那一步的世界状态正是「进程刚死过一次」。
        #
        # 自己的 try/except:conduct 起不来绝不能拦住 runtime 起来。
        try:
            from mast.conduct.service import start_service as _start_conduct
            _start_conduct(self)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not start the conduct director: %s", exc)

        if self.config.llm.api_key:
            try:
                from mast.llm.client import ClaudeClient
                self._claude_client = ClaudeClient(self.config.llm)

                # Build the real 7-agent orchestrator graph from
                # ``mast/agents/orchestrator/graph.py``. Wires LIT, XD,
                # IC, DP, PW, PR as sub-graphs around a supervisor that
                # uses Kimi K2.6 to route. IC needs a ``context_provider``
                # callable returning an ``ExecutionContext`` — we supply
                # one bound to the live pool/state/registry so SafetyGate
                # middleware sees real hardware limits.
                self._build_orchestrator()
                # Build the unified conversation engine for the main Chat tab:
                # a real private chat (私聊) directly onto the IC agent graph
                # (multi-conversation, context compaction, cross-conversation
                # memory). This REPLACES the legacy MissionPlanner on the chat
                # path — both 私聊 and 群聊 now share the same agent engine.
                self._build_chat_engine()
                # Read-only single-turn helper for the 查询助手 tab. It uses an
                # independent ClaudeClient instance so that switching the main
                # chat's model / thinking level doesn't disturb it.
                if all(c is not None for c in [self._executor, self._registry, self._state]):
                    try:
                        from mast.llm.client import ClaudeClient as _CC
                        from mast.llm.quickask import QuickAskAgent
                        qa_client = _CC(self.config.llm)
                        self._quickask = QuickAskAgent(
                            client=qa_client,
                            executor=self._executor,
                            registry=self._registry,
                            state=self._state,
                            experiment_log=self._experiment_log,
                            plan_store=self._plan_store,
                        )
                        # Restore the persisted 查询助手 model (independent of the
                        # main chat model) so it also survives a restart.
                        if self._settings is not None:
                            _qa_persisted = self._settings.get("qa_model")
                            if _qa_persisted:
                                try:
                                    qa_client.use(_qa_persisted)
                                    logger.info("QuickAsk model restored: %s", _qa_persisted)
                                except Exception as exc:
                                    logger.debug("QA model restore failed: %s", exc)
                        logger.info("QuickAskAgent created (read-only single-turn helper)")
                    except Exception as exc:
                        logger.warning("Could not create QuickAskAgent: %s", exc)
            except Exception as exc:
                logger.warning("Could not create LLM components: %s", exc)

        # Voice client (DashScope TTS + ASR). Optional — runs without it if no key.
        if self.config.voice.enabled and self.config.voice.api_key:
            try:
                from mast.voice import DashScopeVoiceClient
                cache_dir = self.config.experiments_dir / self.config.voice.cache_dir_name
                self._voice_client = DashScopeVoiceClient(
                    api_key=self.config.voice.api_key,
                    tts_model=self.config.voice.tts_model,
                    asr_model=self.config.voice.asr_model,
                    default_voice=self.config.voice.default_voice,
                    cache_dir=cache_dir,
                )
                logger.info(
                    "Voice client ready: tts=%s asr=%s voice=%s",
                    self.config.voice.tts_model, self.config.voice.asr_model,
                    self.config.voice.default_voice,
                )
            except Exception as exc:
                logger.warning("Could not create voice client: %s", exc)
                self._voice_client = None


    def reconnect(self) -> int:
        """Attempt to (re)connect to Nanonis. Returns number of ports connected."""
        from mast.core.connection import ConnectionPool
        from mast.core.state import InstrumentState
        from mast.core.safety import SafetyGuard

        # ORPHAN-THREAD FIX (): stop the OLD InstrumentState's
        # background refresh BEFORE we close the pool and replace the state.
        # That daemon thread calls state.refresh() → pool.safe_call() every 1 s
        # against the pool we are about to close_all(); if left running it spins
        # on the dead pool, and ConnectionPool's watchdog tries to
        # reconnect the very port we are tearing down (corrupting the fragile
        # Nanonis TCP port). Stop it first; we start a fresh one for the new
        # state below.
        old_state = getattr(self, "_state", None)
        if old_state is not None:
            try:
                old_state.stop_background_refresh()
            except Exception as exc:
                logger.debug("stop old state refresh failed: %s", exc)

        # Same orphan-thread hazard, same fix: the current monitor polls the
        # pool on the data role. Stop it before close_all() and restart it at
        # the end so it re-probes the scope against the new pool.
        try:
            from mast.monitoring.service import stop_service as _stop_current_monitor
            _stop_current_monitor()
        except Exception as exc:
            logger.debug("stop current monitor failed: %s", exc)

        # 同理：环境历史的 Z burst 也在 data 角色上发 TCP。它不是常驻线程（到期
        # 才起、干完就退），所以这里只需要让在飞的那一次收手，不必也不该把整个
        # 记录器停掉 —— 标量统计桶在重连期间照常累积。
        try:
            eh = getattr(self, "_env_history", None)
            if eh is not None:
                eh.abort_hardware_work()
        except Exception as exc:  # noqa: BLE001
            logger.debug("abort env-history burst failed: %s", exc)

        if self._pool is not None:
            try:
                self._pool.close_all()
            except Exception:
                pass

        try:
            self._pool = ConnectionPool(self.config.nanonis)
            results = self._pool.connect_all()
            connected = sum(1 for v in results.values() if v)
            logger.info("Reconnect: %d/%d ports connected", connected, len(results))
        except Exception as exc:
            logger.warning("Reconnect failed: %s", exc)
            self._pool = None
            return 0

        if self._pool is not None:
            if self._safety is None:
                self._safety = SafetyGuard(self.config.safety)
            self._state = InstrumentState(self._pool)
            # Start the refresh thread for the NEW state so GUI handlers keep
            # reading a recent snapshot() (mirrors setup()'s 1 s interval).
            # Without this, after a reconnect every snapshot() was frozen at the
            # InstrumentState() default until the next full restart.
            try:
                self._state.start_background_refresh(interval_s=1.0)
            except Exception as exc:
                logger.warning("Could not start state background refresh: %s", exc)

            if self._executor is not None:
                self._executor._pool = self._pool
                self._executor._state = self._state
            elif self._registry is not None and self._safety is not None:
                from mast.core.executor import SkillExecutor
                self._executor = SkillExecutor(
                    pool=self._pool, registry=self._registry,
                    safety=self._safety, state=self._state,
                )

        # The watchdog holds a pool reference; the reconnect just swapped the
        # pool, so restart it against the new one.
        self._ensure_watchdog()

        # The monitor's getters are closures over the runtime, so it picks up
        # the new pool by itself; restarting simply makes it re-probe the scope.
        try:
            from mast.monitoring.service import start_service as _start_current_monitor
            _start_current_monitor(self)
        except Exception as exc:
            logger.debug("restart current monitor failed: %s", exc)
        return connected


    def _ensure_watchdog(self) -> None:
        """Start (or restart) the tip-crash SafetyWatchdog on the current pool.

        Idempotent: if one is already running it is stopped first so it never
        polls a stale pool. Best-effort — never blocks startup/reconnect."""
        ex = getattr(self, "_executor", None)
        if ex is None or not hasattr(ex, "start_watchdog"):
            return
        try:
            if getattr(ex, "_watchdog", None) is not None:
                ex.stop_watchdog()
            ex.start_watchdog()
            logger.info("SafetyWatchdog started (current-anomaly → emergency SafeRetract)")
        except Exception as exc:
            logger.warning("Could not start SafetyWatchdog: %s", exc)

    # tip halt 是针对单次 run 的一次性步边界信号，与全局 E_STOP 闩不同。
    # 首个匹配的 composite 步边界消费它；它不阻止新技能，也不阻止退针等恢复动作。
    #
    # raise_tip_halt 缺少显式 run_id 时使用 _orch_run_id，而聊天入口可能使用
    # ConversationEngine.active_run_id；两者不匹配时该入口无法消费 halt。
    # 这条边界与看门狗、告警投递分别承担不同职责：看门狗可直接请求退针，
    # 投递在模型调用前通知，halt 只在 composite 步边界起作用。普通 Scan
    # 调用内部没有这种步边界。
    #
    # 若调整为全局信号，必须同时评估退针、换样等恢复流程的豁免和信号期限，
    # 避免在恢复流程的第一个步边界把它阻止。相关词汇表由 buffer_hitl 和
    # instrument_lock 维护，并有覆盖测试。

    def raise_tip_halt(self, reason: str, *, run_id: str | None = None,
                       event_id: str = "", seqno=None, source: str = "") -> None:
        """Stop the composite running in *run_id* at its next step boundary.

        Called from a BufferService critical hook, i.e. INLINE ON THE VISION
        PUBLISHER THREAD — so it only takes a lock and writes a dict. Never
        raises: a failure here must not break event fanout.

        ``source`` records WHO armed it ("vision" / "current_monitor"). Without
        it the consumer has no way to tell a tip verdict from a physical-safety
        finding — the two arrive as the same event kind — so an operator who
        switches to SAFE while a halt is already armed would either keep a
        tip-repair halt SAFE promised not to give, or (worse, if we dropped them
        all) lose a current-saturation halt that must fire in every mode."""
        try:
            rid = run_id if run_id is not None else (getattr(self, "_orch_run_id", "") or "")
            with _TIP_HALT_LOCK:
                self._tip_halt = {"run_id": rid, "reason": str(reason or ""),
                                  "event_id": event_id, "seqno": seqno,
                                  "source": str(source or "vision"),
                                  "at": _t.time()}
            logger.warning(
                "tip_quality_drop → composite halt armed for run %r (event %s): %s",
                rid, event_id or "?", reason)
        except Exception:  # noqa: BLE001 — publisher thread; never propagate
            logger.exception("raise_tip_halt failed")

    def consume_tip_halt(self, run_id: str) -> str:
        """Take the pending halt for *run_id* and clear it. Returns the reason
        text, or ``""`` when there is nothing pending for this run."""
        try:
            with _TIP_HALT_LOCK:
                h = getattr(self, "_tip_halt", None)
                if not h or h.get("run_id") != (run_id or ""):
                    return ""
                # A halt armed BEFORE the operator switched to SAFE is still
                # sitting here. Drop it if it is a vision tip verdict — SAFE says
                # do not stop the experiment over the tip — but keep anything the
                # current monitor armed: that is the tip railed into the surface
                # or a dead measurement chain, and SAFE does not switch off
                # physical protection.
                if h.get("source") != "current_monitor" and safe_mode_active():
                    self._tip_halt = None
                    logger.info(
                        "SAFE 模式：丢弃切换前已武装的视觉针尖 halt（事件 %s）",
                        h.get("event_id") or "?")
                    return ""
                self._tip_halt = None
            ev = h.get("event_id") or "?"
            return (f"tip_quality_drop: {h.get('reason') or '视觉判定针尖状态恶化'}"
                    f"（视觉事件 {ev}）—— 已在当前步骤边界停止本流程，未回滚；"
                    f"{self._tip_halt_remedy()}")
        except Exception:  # noqa: BLE001 — a broken halt must not stop work
            _log_swallowed("tip_halt_consume", "consume_tip_halt failed")
            return ""

    def _tip_halt_remedy(self) -> str:
        """What to DO about the halted tip — which depends on the mode.

        「明明是safe模式，还是要修针尖」. The halt text ended
        with a flat "修针尖后可重跑" in every mode, and SAFE mode is exactly the
        mode in which MAST will not repair a tip: the belief block tells the
        agent not to, and SafetyGate hard-blocks the pulse. So the agent was
        handed an instruction the system itself refuses to carry out — it
        retried, got halted again, and the operator watched a safe-mode session
        argue with itself about tip conditioning.

        Never raises: this is called from an error path.
        """
        try:
            from mast.core.types import OperatingMode
            mode = OperatingMode.coerce(self._current_operating_mode())
        except Exception:  # noqa: BLE001 — unknown mode → the neutral wording
            return "修针尖后可重跑。"
        if mode is OperatingMode.SAFE:
            return ("当前为**安全模式**，MAST 不会自动修针尖（电脉冲/tip shaping 均被"
                    "拒绝）——请人工处理针尖，或切换到 semi/auto 模式后重试。"
                    "不要在本模式下反复重扫或尝试修针。")
        if mode is OperatingMode.SEMI:
            return ("当前为**半自动模式**：电脉冲修针需人工确认；可先尝试浅层机械 "
                    "tip shaping，或请用户处理后重跑。")
        return "修针尖后可重跑。"

    def clear_tip_halt(self) -> None:
        """Drop any pending halt (operator resolved the interrupt / new run)."""
        with _TIP_HALT_LOCK:
            self._tip_halt = None

    def tip_halt_status(self) -> dict | None:
        """The pending halt, or None. Read-only — does NOT consume."""
        with _TIP_HALT_LOCK:
            h = getattr(self, "_tip_halt", None)
            return dict(h) if h else None

    def _make_halt_check(self, run_id: str):
        """The ``check_halt`` a composite's ExecutionContext consults."""
        return lambda: self.consume_tip_halt(run_id)

    def _v2_scope(self) -> tuple[str | None, str | None]:
        """The (experiment_id, sample_id) of the LIVE V2 records session.

        These are the ids the v2 tables' foreign keys point at. The v1
        ExperimentLog's ids look identical in a log line (both are opaque
        strings) but live in a different store, and writing one where the other
        is expected is a `FOREIGN KEY constraint failed`, not a type error.
        The v2 sample is read once from the session experiment and cached."""
        eid = getattr(self, "_v2_eid", None)
        if not eid:
            return None, None
        sid = getattr(self, "_v2_sid", None)
        if sid is None:
            sid = ""
            repos = getattr(self, "_v2_repos", None)
            try:
                row = repos.experiments.get(eid) if repos is not None else None
                sid = (row or {}).get("sample_id") or ""
            except Exception:  # noqa: BLE001 — a missing sample is not fatal
                sid = ""
            self._v2_sid = sid
        return eid, (sid or None)

    def begin_trajectory(self, thread_id: str, **kw) -> str | None:
        """Open a training trajectory for a task/turn and mark it active so the
        skill/agent-turn recorders attach to it. Returns the id (or None if the
        sink isn't wired). Best-effort — never raises into the task path."""
        sink = getattr(self, "_trace_sink", None)
        if sink is None:
            return None
        try:
            # V2 ids, NOT the v1 ExperimentLog's. `trajectories.experiment_id`
            # REFERENCES the v2 `experiments` table and the store runs with
            # `PRAGMA foreign_keys = ON`, so handing it the v1 experiment UUID
            # made every INSERT die on a FOREIGN KEY constraint — silently, at
            # DEBUG level, behind a locally-minted ULID the caller happily used.
            # Result: the agent training log recorded ZERO rows for a whole day
            # of runs, and 100% of the time an experiment was open — the only
            # situation worth recording (). The one
            # trajectory that survived was created 18 s before that day's
            # experiment existed, when the v1 id happened to be None.
            eid, sid = self._v2_scope()
            tid = sink.begin_trajectory(
                thread_id=thread_id, experiment_id=eid, sample_id=sid, **kw)
            self._active_traj = tid
            self._active_thread_id = thread_id
            return tid
        except Exception:  # pragma: no cover - best-effort
            _log_swallowed("begin_trajectory", "begin_trajectory failed for thread %s",
                           thread_id)
            return None

    def end_trajectory(self, exit_status: str = "completed", **kw) -> None:
        """Close the active training trajectory and clear it. Best-effort."""
        sink = getattr(self, "_trace_sink", None)
        traj = getattr(self, "_active_traj", None)
        self._active_traj = None
        self._active_thread_id = None
        if sink is None or traj is None:
            return
        # Callers say "completed"; the schema's CHECK accepts only
        # ('success','aborted','failed','timeout'), so every FINISHED run's
        # close was rejected by the constraint and swallowed — the same
        # silent-write family as the foreign-key bug above. Map the caller's
        # vocabulary onto the stored one; an unrecognised label stores NULL
        # ("unknown") rather than a plausible-looking guess, and says so.
        raw = str(exit_status or "")
        mapped = _TRAJ_EXIT_STATUS.get(raw.strip().lower())
        if mapped is None and raw:
            logger.warning("end_trajectory: unknown exit_status %r — storing "
                           "NULL rather than guessing", raw)
        try:
            sink.end_trajectory(traj, exit_status=mapped, **kw)
        except Exception:  # pragma: no cover - best-effort
            _log_swallowed("end_trajectory", "end_trajectory failed for %s", traj)

    def start_experimental_monitor(self, channel: str, interval_s: float = 5.0) -> dict:
        """Long-term single-channel monitor: read *channel* every *interval_s* on a
        daemon thread and append to a CSV. Returns the status dict. This hook was
        never wired in the TS runtime, so the '长期监控' feature always failed. Single monitor — starting a new one stops the old."""
        import csv
        import threading
        import time as _t

        self.stop_experimental_monitor()
        pool = getattr(self, "_pool", None)
        if pool is None:
            self._exp_monitor = {"running": False, "channel": str(channel),
                                 "error": "no hardware connection"}
            return self._exp_monitor
        csv_path = ""
        try:
            # 长期监控的 CSV 归当前实验（2026-07-28）。以前一律落在
            # <project_root>/experiments/monitors/，与实验完全脱钩 —— 事后没人
            # 说得出某个 monitor_*.csv 是哪次实验的。没有活跃实验时回退旧路径，
            # 功能不因此失效。
            mon_dir = None
            scope = self._ensure_scope_dirs()
            if scope:
                mon_dir = scope[0] / "env" / "signals"
            if mon_dir is None:
                from mast._runtime_paths import project_root
                mon_dir = project_root() / "experiments" / "monitors"
            mon_dir.mkdir(parents=True, exist_ok=True)
            safe_ch = "".join(c for c in str(channel) if c.isalnum() or c in "-_") or "ch"
            csv_path = str(mon_dir / f"monitor_{safe_ch}_{int(_t.time())}.csv")
        except Exception:  # noqa: BLE001
            csv_path = ""
        stop_evt = threading.Event()
        st: dict = {"running": True, "channel": str(channel),
                    "interval_s": float(interval_s), "count": 0, "last_value": None,
                    "unit": "", "last_t": "", "csv_path": csv_path, "error": "",
                    "_stop": stop_evt}
        self._exp_monitor = st

        def _read():
            ch = str(channel).strip().lower()
            try:
                if ch in ("current", "current (a)", "i"):
                    rec = pool.safe_call("Current_Get", role="monitor")
                    unit = "A"
                else:
                    idx = int(float(channel))
                    rec = pool.safe_call("Signals_ValGet", idx, 1, role="monitor")
                    unit = ""
                if getattr(rec, "error", ""):
                    return None, unit
                rv = getattr(rec, "return_value", None)
                if isinstance(rv, (list, tuple)) and len(rv) > 2:
                    inner = rv[2]
                    if isinstance(inner, (list, tuple)) and inner:
                        return float(inner[0]), unit
                    if isinstance(inner, (int, float)):
                        return float(inner), unit
                return None, unit
            except Exception:  # noqa: BLE001
                return None, ""

        def _loop():
            try:
                if csv_path:
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(["timestamp", "channel", "value", "unit"])
            except Exception:  # noqa: BLE001
                pass
            while not stop_evt.is_set():
                val, unit = _read()
                ts = _t.strftime("%Y-%m-%d %H:%M:%S")
                if val is not None:
                    st["count"] = int(st.get("count", 0)) + 1
                    st["last_value"] = val
                    st["unit"] = unit
                    st["last_t"] = ts
                    try:
                        if csv_path:
                            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([ts, channel, val, unit])
                    except Exception:  # noqa: BLE001
                        pass
                end = _t.monotonic() + max(0.5, float(interval_s))
                while _t.monotonic() < end and not stop_evt.is_set():
                    _t.sleep(0.2)
            st["running"] = False

        th = threading.Thread(target=_loop, name="ExperimentalMonitor", daemon=True)
        st["_thread"] = th
        th.start()
        import atexit
        atexit.register(lambda: stop_evt.set())
        return st

    @staticmethod
    def build_coarse_map_config():
        """``CoarseMapConfig`` for the stage-scale map, from this rig's profile.

        Sibling of :meth:`build_map_analysis_config`, and deliberately separate:
        that one describes a ±1.5 µm square in metres, this one describes
        millimetres of stage travel in STEPS. Mixing the two scales into one
        config object would invite exactly the unit confusion the step/metre
        split exists to prevent.

        Never raises — a caller always gets a usable config from spec defaults."""
        from mast.core import instrument_profile as ip
        from mast.io.coarse_map import CoarseMapConfig

        def _num(key, fallback, cast=float):
            try:
                v = ip.get_config(key, None)
                return cast(v) if v is not None else fallback
            except (TypeError, ValueError):
                return fallback

        return CoarseMapConfig(
            site_spacing_steps=_num("xy_site_spacing_steps", 200, int),
            axis_step_budget=_num("xy_axis_step_budget", 5000, int),
            step_uncertainty_frac=_num("xy_step_uncertainty_frac", 0.3),
            # Annotation only. Passed through so the UI can print "≈ 12 µm"
            # beside the step count; nothing in the planner touches it.
            step_m=_num("xy_motor_step_m", None) or None,
            default_move_steps=_num("xy_site_spacing_steps", 200, int),
        )

    def build_map_analysis_config(self):
        """``AnalysisConfig`` for the scan-map analysis, from this rig's profile.

        The one place the instrument profile, the piezo safety limits and the
        live scan frame are turned into the numbers ``mast.io.map_analysis``
        reasons with. It replaced a standalone in-memory grid (SurfaceNavigator)
        whose "used"/"forbidden" areas only existed if the agent remembered to
        declare them, bore no relation to the footprints actually scanned, and
        vanished on restart. Everything now derives from the recorded markers, so
        the agent and the operator are reading the same surface.

        The construction itself lives in ``mast.core.map_scope`` so the skill
        layer builds the SAME config: a composite that decides where to fire the
        next pulse has to be reading the map the operator is looking at, and two
        constructions of "this rig's avoidance radii" would drift apart.

        Never raises: on any failure the caller still gets a usable config built
        from spec defaults.

        限值取自运行中的 SafetyGuard，包含已合并的管理员覆写。
        地图边界、候选区域和 SafetyGate 必须使用同一生效范围；
        不能让地图停留在原始配置，从而错误估算覆盖率或建议提前粗动换区。
        """
        from mast.core.map_scope import analysis_config
        guard = getattr(self, "_safety", None)
        limits = getattr(guard, "_limits", None) if guard is not None else None
        if limits is None:  # guard 还没建（setup 之前 / 降级路径）
            limits = getattr(self.config, "safety", None) or getattr(
                self.config, "safety_limits", None)
        return analysis_config(getattr(self, "_state", None), safety=limits)

    def stop_experimental_monitor(self) -> dict:
        """Stop the long-term monitor (idempotent). Returns the last status."""
        st = getattr(self, "_exp_monitor", None)
        if isinstance(st, dict):
            ev = st.get("_stop")
            if ev is not None:
                try:
                    ev.set()
                except Exception:  # noqa: BLE001
                    pass
            st["running"] = False
            return st
        return {"running": False}

    def _current_scan_dirs(self) -> tuple:
        """Plausible dirs where Nanonis writes .sxm/.dat/.3ds — the map activity
        watcher scans these for manual scans/spectra. Was never populated in the
        TS runtime, so `_scan_search_dirs` stayed empty and NO manual spectrum was
        ever detected."""
        dirs: list[str] = []
        try:
            sess = self._resolve_session_dir()
            if sess:
                dirs.append(str(sess))
        except Exception:  # noqa: BLE001
            pass
        try:
            from mast._runtime_paths import project_root
            pr = project_root()
            for sub in ("working-sessions", "experiments"):
                d = pr / sub
                try:
                    if d.is_dir():
                        dirs.append(str(d))
                except OSError:
                    pass
        except Exception:  # noqa: BLE001
            pass
        seen: set[str] = set()
        out: list[str] = []
        for d in dirs:
            if d and d not in seen:
                seen.add(d)
                out.append(d)
        return tuple(out)

    # ── per-run stop Events (审计 致命一(b)) ─────────
    def run_abort_event(self, run_id: str):
        """The stop Event for ONE orchestrator run. Created on first ask.

        The global ``_orch_abort`` stays what it always was — the emergency
        latch every ExecutionContext and the E_STOP hook hold by identity. What
        it must NOT also be is "the stop button for whichever run happens to be
        streaming": two concurrent runs shared that one slot, so aborting either
        stopped both and starting either re-armed both.
        """
        import threading as _threading

        rid = str(run_id or "")
        if not rid:
            return None
        with self._orch_run_aborts_lock:
            ev = self._orch_run_aborts.get(rid)
            if ev is None:
                ev = _threading.Event()
                self._orch_run_aborts[rid] = ev
                # Bounded: a long session must not accumulate one Event per run
                # forever. Oldest-first, and never the live one.
                if len(self._orch_run_aborts) > 64:
                    live = str(getattr(self, "_orch_run_id", "") or "")
                    for k in list(self._orch_run_aborts)[:-32]:
                        if k not in (rid, live):
                            self._orch_run_aborts.pop(k, None)
            return ev

    def abort_run(self, run_id: str = "") -> bool:
        """Stop ONE run (or the live one when *run_id* is empty). Not global."""
        rid = str(run_id or getattr(self, "_orch_run_id", "") or "")
        ev = self.run_abort_event(rid) if rid else None
        if ev is None:
            return False
        ev.set()
        logger.info("run %s aborted (per-run Event)", rid)
        return True

    def emergency_latch_state(self) -> dict:
        """只读查询紧急停止闩及其原因，不抛异常。

        状态和原因必须一起返回，不能让调用方自行猜测中止来源。
        """
        try:
            ab = getattr(self, "_orch_abort", None)
            return {
                "latched": bool(getattr(self, "_orch_abort_emergency", False)),
                "abort_set": bool(ab is not None and ab.is_set()),
                "why": str(getattr(self, "_orch_abort_why", "") or ""),
            }
        except Exception:  # noqa: BLE001
            return {"latched": False, "abort_set": False, "why": ""}

    def clear_emergency_latch(self, why: str = "") -> bool:
        """Explicitly release a latched emergency stop; return whether one was set.

        Releasing the latch is separate from starting a run, so a new task cannot
        silently re-enable hardware writes. The explicit API entry point is
        POST /api/safety/clear-emergency.
        """
        was = bool(getattr(self, "_orch_abort_emergency", False))
        prev_why = str(getattr(self, "_orch_abort_why", "") or "")
        self._orch_abort_emergency = False
        self._orch_abort_why = ""
        try:
            self._orch_abort.clear()
        except Exception:  # noqa: BLE001
            pass
        # 每一个 per-run 事件也要放开:急停当时把它们全 set 了,只清全局的话
        # 旧 run 的上下文照样被拦,症状与没解一模一样。
        try:
            with self._orch_run_aborts_lock:
                for _ev in self._orch_run_aborts.values():
                    _ev.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            ex_ab = getattr(getattr(self, "_executor", None), "_abort_event", None)
            if ex_ab is not None:
                ex_ab.clear()
        except Exception:  # noqa: BLE001
            pass
        if was:
            logger.warning("emergency abort latch cleared (was: %s)%s",
                           prev_why or "未留原因", f" — {why}" if why else "")
        return was

    def emergency_stop(self, why: str = "") -> dict:
        """One-click hardware EMERGENCY STOP for the UI E-STOP button.

        Before this there was NO hardware emergency-stop path from the API —
        'abort' only set a software flag and left the tip wherever it was (review
        2026-07-03). This: (1) aborts autonomous runs, (2) stops powered motion
        (auto-approach + coarse motor + scan), (3) retracts the tip (emergency
        port, main fallback), and (4) emits a buffer E_STOP. Never raises.

        ``why`` —— 谁按的、为什么。缺省是 UI 急停按钮那句；外部 agent 网关传
        「外部 agent ext:<名> 触发急停：…」。闩上的原因应准确保留触发来源。"""
        result = {"aborted": False, "stopped_motion": False, "retracted": False,
                  "errors": []}
        _why = (str(why or "").strip()[:200]) or "用户按下了急停(E-STOP)"
        try:
            ab = getattr(self, "_orch_abort", None)
            if ab is not None:
                from mast.core.execution_context import mark_abort
                mark_abort(ab, _why)
                # LATCH it: an emergency stop is not undone by someone starting
                # the next task. Only clear_emergency_latch() releases this.
                self._orch_abort_emergency = True
                self._orch_abort_why = _why
                result["aborted"] = True
            # Every per-run stop too — an E-STOP stops everything, by definition.
            try:
                with self._orch_run_aborts_lock:
                    for _ev in self._orch_run_aborts.values():
                        _ev.set()
            except Exception:  # noqa: BLE001
                pass
            ex_ab = getattr(getattr(self, "_executor", None), "_abort_event", None)
            if ex_ab is not None:
                ex_ab.set()
            store = getattr(self, "_orch_interrupts", None)
            if store:
                lock = store.get("lock")
                import contextlib
                with (lock or contextlib.nullcontext()):
                    for ev in (store.get("events", {}) or {}).values():
                        ev.set()
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"abort: {exc}")

        pool = getattr(self, "_pool", None)
        if pool is not None:
            # Stop powered motion FIRST (so a retract isn't out-stepped).
            #
            # LITERAL verbs, held in thunks. This used to be a table of tuples splatted
            # into `pool.safe_call(*call, …)` — which meant the three verbs the
            # EMERGENCY ABORT itself issues were invisible to the abort-policy checker,
            # the one test whose entire job is to prove that every stop MAST performs is
            # still permitted once an abort is latched. They happened to be allow-listed;
            # nothing verified it. The abort path is the last place that should be
            # outside its own guard.
            # urgent_call, NOT safe_call: the role lock spans a whole socket
            # round-trip, so a caller wedged on a dead peer used to park the
            # E-STOP behind it forever — the fallback to role="main" queued
            # behind the exact stall it exists to rescue (dispatch audit
            # 2026-07-28 致命三 "最要命"). urgent_call waits 2 s and then
            # force-closes that role's socket to unstick it.
            _urgent = getattr(pool, "urgent_call", None) or pool.safe_call
            # 急停口被显式停用时(端口=0),下面三个「停运动」动词发不出去 ——
            # 说出来,别让用户从三条 ConnectionError 里自己拼。
            # 这是显式做出的取舍,
            # 但取舍的后果要写在结果里,不能只写在某人的记忆里。
            try:
                if not getattr(getattr(pool, "_config", None), "port_emergency", 1):
                    result["errors"].append(
                        "急停口已停用(port_emergency=0):停运动的三个动词发不出去,"
                        "退针改走 main 且需要等它的角色锁。**要立即停机请用仪器面板"
                        "上的硬急停。**")
                    result["emergency_port_disabled"] = True
            except Exception:  # noqa: BLE001
                pass
            for verb, thunk in (
                ("AutoApproach_OnOffSet",
                 lambda: _urgent("AutoApproach_OnOffSet", 0, role="emergency")),
                ("Motor_StopMove",
                 lambda: _urgent("Motor_StopMove", role="emergency")),
                ("Scan_Action",
                 lambda: _urgent("Scan_Action", 1, 0, role="emergency")),
            ):
                try:
                    thunk()
                    result["stopped_motion"] = True
                except Exception as exc:  # noqa: BLE001
                    result["errors"].append(f"{verb}: {exc}")
            # Retract the tip: emergency port, fall back to main.
            for role in ("emergency", "main"):
                try:
                    rec = _urgent("ZCtrl_Withdraw", 1, -1, role=role)
                    if not getattr(rec, "error", ""):
                        result["retracted"] = True
                        break
                    result["errors"].append(f"withdraw({role}): {rec.error}")
                except Exception as exc:  # noqa: BLE001
                    result["errors"].append(f"withdraw({role}): {exc}")

        try:
            from mast.buffer.active import get_active_buffer
            buf = get_active_buffer()
            if buf is not None:
                from mast.buffer.schemas import make_e_stop
                # Reason must be one of E_STOP_REASONS — an operator-pressed
                # stop is the "user" reason ("operator" was rejected by the
                # factory, so the E_STOP event was silently dropped, 2026-07-10).
                buf.emit_event(make_e_stop(
                    "user", "operator emergency stop", seqno=buf.next_seq()))
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"estop_event: {exc}")

        logger.critical("EMERGENCY STOP invoked → %s", result)
        return result

    def shutdown(self) -> None:
        """Graceful teardown for a clean service stop.

        Stops every daemon that holds the Nanonis TCP pool (watchdog, state
        refresh, env monitor, map watcher, experimental monitor) THEN closes the
        pool, so the fragile ports aren't left mid-transaction. A hard
        TerminateProcess (which the launcher's no-/F taskkill falls through to on
        a windowless frozen service) skips ALL of this and can corrupt the port
        until Nanonis restarts. Idempotent + best-effort."""
        # 登记过的外部入口**最先**停（外部 agent 网关：停止接新作业、请在跑的作业
        # 停下、有界等待）。必须排在下面解绑运行模式之前：解绑之后的语义是「未知
        # 模式放行」，一个还在跑的外部作业会在这段窗口里失去 SAFE 的保护；也必须
        # 早于 close_all —— 还在往正被拆掉的池里发命令的线程，正是脆弱的 Nanonis
        # 端口被搞坏的方式。钩子自己负责「请它停 + 等一会 + 如实记日志，不强杀」。
        for _hook_name, _hook in list(getattr(self, "_shutdown_hooks", None) or ()):
            try:
                _hook()
            except Exception as exc:  # noqa: BLE001 — 一个钩子坏了不能挡住关停
                logger.warning("shutdown hook %s failed: %s", _hook_name, exc)
        # Release the process-level operating-mode holder FIRST: it points at a
        # bound method of this runtime, so leaving it armed keeps this instance's
        # settings deciding tip verdicts for whatever runs next in the process
        # (a second runtime, or the rest of a test session).
        try:
            from mast.core.operating_mode import bind_mode_source
            bind_mode_source(None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown unbind operating mode: %s", exc)
        try:
            sched = getattr(self, "_wake_scheduler", None)
            if sched is not None:
                sched.stop()
                self._wake_scheduler = None
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown stop wake scheduler: %s", exc)
        # conduct 指挥线程要在池关掉之前停:它会在自己的线程里调
        # ``ExecutionContext.run``,而一个还在往正在拆掉的池里发命令的线程,
        # 正是那几个脆弱的 Nanonis 端口被搞坏的方式。
        #
        # ⚠️ 它**可能停不下来** —— 卡在一次 TCP 事务里的线程不会因为我们请它停就
        # 返回,而强杀会永久损坏端口。``stop_service`` 因此只是请它停 + 等一会 +
        # 如实记一条日志,**不强杀**(设计 §3-7 的诚实短板)。
        try:
            from mast.conduct.service import stop_service as _stop_conduct
            _stop_conduct(drop=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown stop conduct director: %s", exc)
        try:
            ex = getattr(self, "_executor", None)
            if ex is not None and hasattr(ex, "stop_watchdog"):
                ex.stop_watchdog()
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown stop_watchdog: %s", exc)
        try:
            st = getattr(self, "_state", None)
            if st is not None:
                st.stop_background_refresh()
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown stop state refresh: %s", exc)
        try:
            m = getattr(self, "_monitor", None)
            if m is not None and hasattr(m, "stop"):
                m.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown stop env monitor: %s", exc)
        try:
            ev = getattr(self, "_map_watch_stop", None)
            if ev is not None:
                ev.set()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.stop_experimental_monitor()
        except Exception:  # noqa: BLE001
            pass
        # Must precede pool.close_all() below: the monitor polls the data role,
        # and a thread still calling into a pool being torn down is how the
        # fragile Nanonis ports get corrupted.
        try:
            from mast.monitoring.service import stop_service as _stop_current_monitor
            _stop_current_monitor()
        except Exception:  # noqa: BLE001
            pass
        # Drain the data-ingest queue. Note this is a courtesy, NOT a
        # correctness requirement: every manifest/CSV/chat write is incremental
        # and atomic, so a hard kill that skips this leaves the experiment folder
        # fully usable — only the last few in-flight copies are lost, and the
        # watcher picks those up on the next run because they aren't in seen[].
        try:
            ing = getattr(self, "_ingest", None)
            if ing is not None:
                ing.flush(timeout=5.0)
                ing.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            csv_sink = getattr(self, "_env_csv", None)
            if csv_sink is not None:
                csv_sink.close()
        except Exception:  # noqa: BLE001
            pass
        # 环境历史：落定未满的统计桶，并有界等待清扫 / Z burst 线程。放在
        # pool.close_all() 之前 —— burst 会在 data 角色上发 TCP，一个还在调用
        # 正被拆掉的 pool 的线程正是脆弱的 Nanonis 端口被搞坏的方式。
        try:
            eh = getattr(self, "_env_history", None)
            if eh is not None:
                eh.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            from mast.core.scan_registry import save_state
            save_state()
        except Exception:  # noqa: BLE001
            pass
        # 最后导一次对话。同样是 courtesy 而非正确性要求 —— 已导出的部分在磁盘
        # 上已经完整，这里只是把最后 60 秒的转录也带上。
        try:
            self._chat_export_tick()
        except Exception:  # noqa: BLE001
            pass
        try:
            unsub = getattr(self, "_scope_unsub", None)
            if callable(unsub):
                unsub()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "_pool", None) is not None:
                self._pool.close_all()
        except Exception as exc:  # noqa: BLE001
            logger.debug("shutdown close_all: %s", exc)
        logger.info("CoreRuntime shutdown complete (daemons stopped, pool closed)")

    def add_shutdown_hook(self, fn, *, name: str = "") -> None:
        """登记一个在 ``shutdown()`` **最开始**调用的无参回调。

        给那些自己起线程、经 ``ExecutionContext`` 往连接池发命令、而 runtime 本身
        不认识的入口用（外部 agent 网关的作业管理器）。同一个回调重复登记只留一份。
        """
        hooks = getattr(self, "_shutdown_hooks", None)
        if hooks is None:
            hooks = []
            self._shutdown_hooks = hooks
        if any(f is fn for _n, f in hooks):
            return
        hooks.append((name or getattr(fn, "__name__", "hook"), fn))

    # ── run cost ceiling (2026-07-30) ────────────────────────────────────────
    def begin_run_budget(self, budget_usd: float) -> None:
        """Arm the per-run USD ceiling for the run that is about to start.

        Called by the run-task bridge right before it drives the graph. The
        orchestrator's gate has existed since and was never armed by
        anyone, so a run's only cost bound was recursion_limit (~$50 worth). See
        ``mast.billing.run_meter`` for the accounting caveat: the meter reports
        system-wide spend since this moment, which over-attributes concurrent work
        to this run — the safe direction for a ceiling.

        ``budget_usd <= 0`` disarms it (the honest inert default), never a $0
        ceiling that would end every run on its first hop.
        """
        try:
            from mast.billing.run_meter import RunMeter

            b = float(budget_usd or 0.0)
            self._run_meter = RunMeter(budget_usd=b) if b > 0 else None
            if self._run_meter is not None:
                logger.info("run budget armed: $%.2f", b)
        except Exception as exc:  # noqa: BLE001 — billing never blocks a run
            logger.debug("begin_run_budget failed (gate left inert): %s", exc)
            self._run_meter = None

    def end_run_budget(self) -> None:
        """Disarm the ceiling when the run finishes.

        Must be called on EVERY exit path (success, abort, exception), otherwise a
        stale meter keeps counting and the next run inherits a budget that looks
        already-spent. The bridge does this in a ``finally``.
        """
        self._run_meter = None

    def _budget_remaining_usd(self):
        """Remaining USD for the current run, or ``None`` if unknown/unarmed.

        The stable callable handed to ``orchestrator.build(budget_probe=…)``: the
        graph is built once and reused across runs, so it cannot hold a meter
        directly. ``None`` means "do not touch the gate" — a billing read failure
        must not fabricate a budget in either direction.
        """
        meter = getattr(self, "_run_meter", None)
        if meter is None:
            return None
        try:
            return meter.remaining_usd()
        except Exception as exc:  # noqa: BLE001
            logger.debug("budget probe failed: %s", exc)
            return None

    def reset_watchdog(self) -> bool:
        """Re-arm the SafetyWatchdog after operator recovery.

        The anomaly latch is one-shot: once a tip-crash retract fires and is
        confirmed, the watchdog stops monitoring until reset. Nothing used to
        call reset(), so the safety net stayed disarmed for the rest of the
        session. Starting a new orchestrator task (operator has taken control
        and is proceeding) is a recovery point — clear the latch so the net
        re-arms. Best-effort; returns True iff a live watchdog was reset."""
        ex = getattr(self, "_executor", None)
        wd = getattr(ex, "_watchdog", None) if ex is not None else None
        if wd is None or not hasattr(wd, "reset"):
            return False
        try:
            was_latched = getattr(wd, "is_anomaly_triggered", False)
            wd.reset()
            if was_latched:
                logger.info("SafetyWatchdog re-armed (latched anomaly cleared on "
                            "new-task recovery)")
            # Also clear the executor abort Event so a prior E_STOP does not keep
            # the very next composite from running (it too was never cleared).
            ab = getattr(ex, "_abort_event", None)
            if ab is not None:
                ab.clear()
            return True
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("reset_watchdog failed: %s", exc)
            return False


    def _orch_control_provider(self) -> dict:
        """Drain pending operator interjections for the running task.

        Called by the supervisor node each super-step (non-blocking). Returns
        ``{"interjections": [text, ...]}`` and CLEARS the consumed queue so an
        interjection is delivered exactly once. Reads the in-process Agents-API
        state populated by POST /agents/<id>/interject.
        """
        st = getattr(self, "_agents_api_state", None)
        if not st:
            return {}
        # Atomic swap-drain under the lock: take the current list and replace it
        # with a fresh empty one in one critical section, so an interject append
        # racing this drain is never lost . The previous read-then-clear had
        # a window where an append between the two lines was silently dropped.
        lock = st.get("lock")
        if lock is not None:
            with lock:
                pending = st.get("interjects") or []
                st["interjects"] = []
                _live_task_id = str(((st.get("task") or {}) if isinstance(
                    st.get("task"), dict) else {}).get("id") or "")
        else:  # pragma: no cover - lock always present after _mount_agents_api
            pending = st.get("interjects") or []
            st["interjects"] = []
            _live_task_id = str(((st.get("task") or {}) if isinstance(
                st.get("task"), dict) else {}).get("id") or "")
        if not pending:
            return {}
        # DROP entries queued against a DIFFERENT task (2026-07-28). The queue is
        # a process-level singleton that run-task never cleaned, so an
        # interjection that landed in a window with no consumer left — the run's
        # final LLM call, an abort, an exception — was drained by the first hop
        # of the NEXT task. With "@instrument_control" in it, that stale text
        # became a deterministic hard route into a brand-new experiment. Entries
        # with no task_id are legacy/unstamped and still delivered.
        if _live_task_id:
            fresh = [j for j in pending
                     if not j.get("task_id") or j.get("task_id") == _live_task_id]
            if len(fresh) != len(pending):
                stale = [j for j in pending if j not in fresh]
                logger.warning(
                    "Orchestrator control: dropping %d interjection(s) queued "
                    "against a previous task (now running %s): %s",
                    len(stale), _live_task_id,
                    [(j.get("task_id"), (j.get("text") or "")[:40]) for j in stale])
                try:
                    from mast.core.diagnostics import record as _diag
                    _diag("interject_stale_dropped", _live_task_id,
                          "上一个任务遗留的插话未被送达（已丢弃，不会污染新任务）",
                          count=len(stale))
                except Exception:  # noqa: BLE001
                    pass
            pending = fresh
        if not pending:
            return {}
        # Preserve the operator's TARGET agent BOTH in the text (honest intent,
        # visible to the agent) AND as a STRUCTURED ``directed_targets`` hint the
        # supervisor dispatches to DETERMINISTICALLY (a hard @agent route). The
        # old text-only "(指向 X)" left the LLM router free to ignore the operator's
        # choice — the "@agent 不管用" report . It still cannot hard-scope
        # delivery MID-batch (the LangGraph super-step barrier delivers it when the
        # running batch hands back), but once delivered it reaches THAT agent.
        texts = []
        directed: list[str] = []
        for j in pending:
            t = (j.get("text") or "").strip()
            if not t:
                continue
            aid = j.get("agent_id")
            if aid and aid not in ("_supervisor", "__all__"):
                texts.append(f"(指向 {aid}) {t}")
                directed.append(aid)
            else:
                texts.append(t)
        # NOTE: the consumed list was already swapped out atomically above; do
        # NOT clear st["interjects"] again here — a second clear would wipe any
        # interjection that arrived WHILE we were formatting these .
        if texts:
            logger.info("Orchestrator control: delivering %d interjection(s)%s", len(texts),
                        f" (directed → {directed})" if directed else "")
        return {"interjections": texts, "directed_targets": directed}

    # ── True-parallel background runs (break the super-step barrier) ──────────
    def _auto_background_enabled(self) -> bool:
        """Conservative auto-background toggle — the supervisor's ``background_gate``.

        DEFAULT OFF: absent/false setting → False → the orchestrator's routing is
        byte-for-byte unchanged. Read LIVE from settings so the operator can flip
        ``orchestrator_auto_background`` without a restart. Fail-safe to OFF."""
        st = getattr(self, "_settings", None)
        try:
            return bool(st.get("orchestrator_auto_background")) if st is not None else False
        except Exception:  # noqa: BLE001
            return False

    # ── wake scheduler (2026-07-30) ──────────────────────────────────────────
    def _wake_ask(self, park: dict, arrived: list, versions: dict) -> dict:
        """Ask a parked agent whether the arrival is enough to wake for.

        Runs on the scheduler's own thread — never in a graph node and never on the
        EventBus publishing thread — because it makes a model call.
        """
        from mast.agents._shared.activation import ask_should_wake, humanize_age
        from mast.agents._shared.models import make_chat_model

        import time as _t

        model = make_chat_model("orchestrator", max_tokens=2048)
        waited = _t.time() - float(park.get("created_at") or _t.time())
        left = float(park.get("deadline_at") or 0.0) - _t.time()
        decision = ask_should_wake(
            model,
            agent=str(park.get("agent") or ""),
            waiting_for=list(park.get("waiting_for") or []),
            # No live state exists when the system is idle — that is the whole
            # situation. Disk is the authority anyway; the agent reads bodies with
            # load_document either way.
            state=dict(park.get("artifact_snapshot") or {}),
            arrived=list(arrived),
            waited_human=humanize_age(waited),
            declines=int(park.get("declines") or 0),
            deadline_human=(f"还有 {humanize_age(left)}" if left > 0 else "已超时"),
            instruction=str(park.get("instruction") or ""),
        )
        # EVERY decision is written down, including — especially — "chose not to
        # wake". A park that quietly ages with no record is indistinguishable from a
        # stuck one, which is the failure this whole mechanism has to avoid being.
        self._wake_escalate(
            "woke" if decision.get("action") == "start" else "declined", park,
            f"到位:{'、'.join(arrived)};理由:{decision.get('reason') or '(未说明)'}")
        return decision

    def _wake_spawn(self, park: dict) -> str:
        """Start the detached run for a woken park. Returns its run_id ("" on failure).

        The woken run is ``{interactive:0, instrument:0, hitl:0}`` like any background
        run — it just also gets ``seeded_from = mainline snapshot``, which is the one
        axis the old foreground/background split could not express.
        """
        mgr = self._ensure_background_manager()
        if mgr is None:
            return ""
        try:
            from mast.agents._shared.artifact_channel import carried_from

            seed = carried_from(park.get("artifact_snapshot") or {})
        except Exception:  # noqa: BLE001
            seed = {}
        instruction = str(park.get("instruction") or "").strip()
        if not instruction:
            waits = "、".join(park.get("waiting_for") or [])
            instruction = (f"你之前在等 {waits},现在它已经到位了 —— "
                           "接着把当时的工作做完。")
        try:
            rec = mgr.spawn(instruction=instruction,
                            agents=(str(park.get("agent") or ""),),
                            title=f"唤醒:{park.get('agent')}",
                            seed_artifacts=seed,
                            park_id=str(park.get("park_id") or ""))
            return str(rec.get("run_id") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake spawn failed: %s", exc)
            return ""

    def _wake_goal_check(self, park: "dict | None"):
        """这份 park 服务的科研纲领，它的目标达成了没有。

        由 :class:`~mast.core.wake_scheduler.WakeScheduler` 注入调用 —— 调度器
        本身**不 import** ``mast.goals`` / ``mast.conduct``（有 AST 结构测试钉
        着）：那条路的整个意义是在系统空闲、没有任何 agent 上下文的时候也能跑，
        把 agents 栈拉进那条线程与它的目的相反。

        归属**只认 park 上冻结的那一个**（``_persist_park`` 建 park 时从
        ``research_campaign`` 冻的）。不猜：v1 实验行的 ``v2_campaign_id`` 只指
        向 GUI 作用域纲领，而 research_director 建的纲领与任何实验都没有链接 ——
        猜错的后果是把 A 纲领的判据套到 B 的 park 上，静默地不唤醒。

        **永不抛。** 抛出去只会杀掉这一趟 tick；回 ``None`` 让调度器照旧走它
        原来的路（也就是这道闸不存在时的行为）。
        """
        cid = str((park or {}).get("campaign_id") or "").strip()
        if not cid:
            return None
        try:
            import json as _json

            from mast.agents._shared.data_paths import (
                experiment_db_path,
                v2_experiment_db_path,
            )
            from mast.goals import evaluate_done_when, normalise_done_when
            from mast.goals.sources import make_collector
            from mast.logging.v2.repos import build_repos
            from mast.logging.v2.storage import open_store
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: goal modules unavailable (%s)", exc)
            return None

        try:
            repos = build_repos(open_store(v2_experiment_db_path()))
            row = repos.campaigns.get(cid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: campaign %s unreadable (%s)", cid, exc)
            return None
        if not row:
            return None

        # 人或 RD 已经把这条纲领收口了 ⇒ 它下面的 park 不该再花钱醒。
        status = str(row.get("status") or "")
        if status in ("completed", "aborted"):
            try:
                from mast.goals import DONE, GoalVerdict

                return GoalVerdict(verdict=DONE,
                                   reason=f"纲领已 {status}（不是判据，是状态）")
            except Exception:  # noqa: BLE001
                return None

        # ``goal_json`` 是一列 JSON 文本，不是解好的 dict。读不懂 ⇒ 回 None
        # （照旧唤醒），**不是**当成「没有判据也就算了」的某种达成。
        try:
            goal = _json.loads(row.get("goal_json") or "{}") or {}
        except Exception:  # noqa: BLE001
            logger.debug("wake: campaign %s 的 goal_json 读不懂", cid)
            return None
        if not isinstance(goal, dict):
            return None
        spec, errs = normalise_done_when(goal.get("done_when"))
        if errs or spec is None:
            return None          # 没写判据 / 写坏了 ⇒ 照旧唤醒
        collect = make_collector(
            baseline=goal.get("baseline"),
            conduct_db=experiment_db_path(),
            v2_db=v2_experiment_db_path(),
            campaign_id=cid,
            # 空闲进程里没人能向用户提问 —— 「问不了」是判不了，不是「没确认」。
            askable=False,
        )
        try:
            return evaluate_done_when(spec, collect)
        except Exception as exc:  # noqa: BLE001
            logger.debug("wake: goal evaluation failed for %s (%s)", cid, exc)
            return None

    def _wake_escalate(self, kind: str, park: "dict | None", detail: str) -> None:
        """Surface a wake-scheduler event to the operator.

        Deliberately does NOT try to push a live notification. This fires in exactly
        the situation where a push cannot arrive: the system is idle, so an SSE frame
        has no receiver and a transcript line sits in a conversation nobody has open.
        A notification nobody receives is not a delivery, and pretending otherwise is
        how this whole mechanism would become the silent channel it is at risk of
        being.

        What actually delivers is the BOARD: an expired park keeps its row demanding
        attention in the 待唤醒 panel until a human acknowledges it, whenever they next
        look. The board write already happened (``sweep_expired``) before this is
        called; this adds the operator-readable audit line, which is the design's
        third non-negotiable — every "the agent chose not to wake" must be on record
        with its reason.
        """
        agent = (park or {}).get("agent") or "?"
        logger.warning("wake scheduler → operator [%s] %s: %s", kind, agent, detail)
        try:
            from mast.core.diagnostics import record as _diag_record

            # "note" — this is a breadcrumb, not one of the system's refusal layers.
            # Same ledger the 621 fail_silent_end events live in, on purpose: an
            # operator investigating "why did nothing happen overnight" should find
            # the wake decisions in the place they already look.
            _diag_record("note", f"wake:{kind}", detail,
                         agent=agent,
                         park_id=(park or {}).get("park_id", ""),
                         waiting_for=list((park or {}).get("waiting_for") or []),
                         declines=(park or {}).get("declines", 0))
        except Exception as exc:  # noqa: BLE001 — audit is best-effort
            logger.debug("wake diagnostic not recorded: %s", exc)

    def _ensure_wake_scheduler(self):
        """Build + start the wake scheduler once. None if it cannot be built."""
        if getattr(self, "_wake_scheduler", None) is not None:
            return self._wake_scheduler
        try:
            from mast.core.wake_scheduler import WakeScheduler
        except Exception as exc:  # noqa: BLE001
            logger.warning("wake scheduler unavailable: %s", exc)
            return None
        sched = WakeScheduler(
            should_wake=lambda: not getattr(self, "_orch_running", False),
            ask=self._wake_ask,
            spawn=self._wake_spawn,
            escalate=self._wake_escalate,
            goal_check=self._wake_goal_check,
            daily_budget_usd=self._effective_daily_budget_usd(),
            max_wakes_per_day=self._effective_max_wakes_per_day(),
        )
        sched.subscribe_to_events()
        sched.start()
        self._wake_scheduler = sched
        return sched

    def _effective_daily_budget_usd(self) -> float:
        st = getattr(self, "_settings", None)
        try:
            v = st.get("daily_budget_usd") if st is not None else None
            if v is not None:
                f = float(v)
                return f if f > 0 else 0.0
        except Exception:  # noqa: BLE001
            pass
        from mast.core.wake_scheduler import DEFAULT_DAILY_BUDGET_USD

        return DEFAULT_DAILY_BUDGET_USD

    def _effective_max_wakes_per_day(self) -> int:
        """Per-experiment auto-wake ceiling, read live from settings.

        **Never returns a negative value, and that is the point.** The scheduler
        treats a negative as "disable the check entirely", which is useful in tests
        but must not be reachable from the settings page: this counter is the ONLY
        bound on a product-driven wake loop. Every other guard in the system is
        per-run, and a wake starts a NEW run — fresh recursion_limit, fresh
        visit_count, fresh budget — with every step succeeding, so StallGuard cannot
        see the cycle either. Offering "unlimited" as a one-click setting would mean
        offering to remove the only thing standing between a PW⇄PR ping-pong and the
        bill for it.

        So the settings path clamps at 0, and 0 means **no automatic waking at all**
        (an operator who types 0 into a safety counter means "do not do this" — the
        opposite convention from the USD ceilings, where 0 is what an unconfigured
        numeric setting looks like and therefore has to mean "no ceiling"). Someone
        who genuinely needs more headroom raises the number.
        """
        st = getattr(self, "_settings", None)
        try:
            v = st.get("wake_max_per_day") if st is not None else None
            if v is not None:
                return max(0, int(v))
        except Exception:  # noqa: BLE001
            pass
        from mast.core.wake_scheduler import DEFAULT_MAX_WAKES_PER_DAY

        return DEFAULT_MAX_WAKES_PER_DAY

    def _activation_gating_enabled(self) -> bool:
        """Environment-driven activation gating — the supervisor's ``activation_gate``.

        DEFAULT OFF, and deliberately so: when ON, the supervisor may decline to
        dispatch an agent (parking it until its inputs exist). That is the behaviour
        the design itself flags as the most likely to become "a beautiful silent
        death channel", so it does not arrive switched on. Read LIVE from
        ``orchestrator_activation_gating`` so a flip needs no restart; fail-safe OFF.
        """
        st = getattr(self, "_settings", None)
        try:
            return bool(st.get("orchestrator_activation_gating")) if st is not None else False
        except Exception:  # noqa: BLE001
            return False

    def _ensure_background_manager(self):
        """Lazily build the BackgroundRunManager (results merged into the durable
        ConversationStore). None only if the manager class can't be imported."""
        if self._background_runs is not None:
            return self._background_runs
        try:
            from mast.core.background_runs import BackgroundRunManager
        except Exception as exc:  # pragma: no cover — stdlib-only module
            logger.warning("background run manager unavailable: %s", exc)
            return None
        store = getattr(self, "_conv_store", None)

        def _sink(cid: str, kind: str, agent_id: str, role: str, text: str) -> None:
            # Best-effort merge into the SAME durable transcript as the foreground
            # (append_message is atomic + busy-timeout-serialised → concurrent-safe).
            if store is None:
                return
            store.append_message(cid, kind=kind, agent_id=agent_id, role=role, text=text)

        stats = self._ensure_run_stats()
        self._background_runs = BackgroundRunManager(
            run_fn=self._background_run_fn, transcript_sink=_sink,
            # item ②: record each finished run's real duration so the auto-
            # background policy can learn which task types are consistently slow.
            duration_sink=(stats.record if stats is not None else None),
            # item ③: fire a foreground supervisor hint (+ any EXPLICITLY declared
            # follow-up) when a run finishes cleanly. NEVER auto-executes anything.
            completion_sink=self._on_background_complete)
        logger.info("BackgroundRunManager ready (true-parallel background runs, "
                    "duration-learning=%s)", "on" if stats is not None else "off")
        return self._background_runs

    # ── item ②: run-duration learning ────────────────────────────────────────
    # A type must have >= MIN_SAMPLES completed runs before its history is trusted
    # (below that: None → the static whitelist decides), and its median run must
    # exceed THRESHOLD to be judged "slow enough to be worth detaching".
    _BG_SLOW_THRESHOLD_S = 45.0
    _BG_SLOW_MIN_SAMPLES = 5

    def _ensure_run_stats(self):
        """Lazily build the persistent RunStatsStore (loads prior-session history
        so the signal is warm from the first turn). None if it can't be built."""
        store = getattr(self, "_run_stats", None)
        if store is not None:
            return store
        try:
            from mast.core.run_stats import RunStatsStore
            exp_dir = getattr(self.config, "experiments_dir", None)
            path = (Path(exp_dir) / "background_run_stats.json") if exp_dir else None
            self._run_stats = RunStatsStore(path=path)
        except Exception as exc:  # noqa: BLE001 — never block the background path
            logger.debug("run stats store unavailable: %s", exc)
            self._run_stats = None
        return self._run_stats

    def _bg_is_slow_type(self, agent_type: str):
        """Duration advisor for ``_split_auto_background`` (item ②). Returns
        True/False when we have enough history, else None (prefer-missing → the
        static whitelist decides). Never raises."""
        store = self._ensure_run_stats()
        if store is None:
            return None
        try:
            return store.is_slow(agent_type, threshold_s=self._BG_SLOW_THRESHOLD_S,
                                 min_samples=self._BG_SLOW_MIN_SAMPLES)
        except Exception:  # noqa: BLE001
            return None

    # ── item ③: cross-run dependency (controlled) ────────────────────────────
    # Cap the queued completion hints so a burst of finishing runs can't grow the
    # interjection queue without bound.
    _BG_COMPLETION_NOTE_CAP = 20

    def _on_background_complete(self, run) -> None:
        """completion_sink target: inject an ADVISORY note into the FOREGROUND
        supervisor's interjection queue when a background run finishes cleanly —
        surfacing its output + any EXPLICITLY declared follow-up so the operator/
        supervisor can decide the next step.

        Hard contract (item ③): this NEVER executes anything — no tool, no skill,
        and above all no instrument_control hardware call. It only appends TEXT the
        router sees next super-step ("产出自动就位,不自动化整条链"). It is delivered
        only while a foreground task is actively streaming (otherwise nobody drains
        it and the result is already merged into the durable transcript). Best-
        effort — a failure here must never crash the manager's worker thread."""
        st = getattr(self, "_agents_api_state", None)
        if not isinstance(st, dict):
            return
        task = st.get("task") if isinstance(st.get("task"), dict) else {}
        if not task.get("active"):
            return  # no live supervisor to hint; the result is in the transcript
        try:
            from mast.core.background_runs import build_completion_hint
            text = build_completion_hint(run)
        except Exception as exc:  # noqa: BLE001
            logger.debug("completion hint build failed: %s", exc)
            return
        import contextlib
        lock = st.get("lock")
        with (lock or contextlib.nullcontext()):
            q = st.setdefault("interjects", [])
            # agent_id="_supervisor" → delivered as a PLAIN (non-directed) note the
            # router weighs and decides on; nothing is auto-dispatched.
            q.append({"agent_id": "_supervisor", "text": text, "t": time.time(),
                      "kind": "background_completion"})
            if len(q) > self._BG_COMPLETION_NOTE_CAP:
                del q[: len(q) - self._BG_COMPLETION_NOTE_CAP]
        logger.info("background run %s complete → foreground supervisor hint queued",
                    getattr(run, "run_id", "?"))

    def _build_background_orchestrator(self):
        """Build (once, cached) an ISOLATED orchestrator for background tasks.

        Differs from the foreground orchestrator on exactly the axes that make a
        detached run safe + non-interfering:
          * include_agents = the BACKGROUNDABLE set (NO instrument_control — the
            sole hardware agent stays foreground/interactive), so context_provider
            is unneeded and two background runs can never contend for hardware;
          * control_provider=None — a background run must NOT drain the operator's
            interjection queue (that belongs to the foreground);
          * its OWN InMemorySaver — the foreground's checkpoint state is never
            touched, so a background run can never pollute the live MASTState;
          * enable_hitl=False — no IC ⇒ no DANGEROUS skills ⇒ HITL is a no-op
            anyway, and a background run has no operator-resume path.
        Reuses the shared compaction/tool-pair/memory middleware + call limits +
        supervisor & agent models. Returns None if it can't be built (no key)."""
        if self._bg_orchestrator is not None:
            return self._bg_orchestrator
        with self._orch_build_lock:
            if self._bg_orchestrator is not None:
                return self._bg_orchestrator
            try:
                from langgraph.checkpoint.memory import InMemorySaver
                from mast.agents.orchestrator import graph as _orch_graph
                sup_model = self._resolve_supervisor_model()
                try:
                    overrides = self._resolve_override_models()
                except Exception:
                    overrides = {}
                # DERIVED from the admission whitelist, not transcribed
                # (2026-08-21). This tuple used to be a hand-written copy of it,
                # and a copy of a roster drifts the first time the roster grows:
                # research_director was added to BACKGROUNDABLE and this list
                # stayed at five, so the agent existed, passed admission, and
                # then had no node to be dispatched to — a "nothing happens"
                # with no error anywhere. Order follows _AGENT_NAMES so the
                # wiring reads in pipeline order.
                from mast.core.background_runs import BackgroundRunManager
                bg_agents = tuple(
                    a for a in _orch_graph._AGENT_NAMES
                    if a in BackgroundRunManager.BACKGROUNDABLE)
                self._bg_orchestrator = _orch_graph.build(
                    buf=self._buffer,
                    supervisor_model=sup_model,
                    context_provider=None,               # no IC in the background set
                    control_provider=None,               # never steal foreground interjects
                    agent_model_overrides=overrides or None,
                    include_agents=bg_agents,
                    checkpointer=InMemorySaver(),        # isolated from the foreground
                    enable_hitl=False,
                    agent_extra_middleware=self._group_agent_middleware(),
                    agent_call_limits=self._chat_call_limits(),
                )
                logger.info("Background orchestrator built (%d backgroundable agents: %s, "
                            "isolated checkpointer, no control_provider)",
                            len(bg_agents), ",".join(bg_agents))
            except Exception as exc:  # noqa: BLE001
                logger.warning("background orchestrator build failed: %s", exc)
                self._bg_orchestrator = None
        return self._bg_orchestrator

    @staticmethod
    def _seed_notice(seeded: dict, snapshot_at: float) -> str:
        """The line that tells a woken agent its inherited pointers are a SNAPSHOT.

        A snapshot goes stale between spawn and execution — the mainline may have
        written a newer version of any of these in between. Letting the agent assume
        the pointers are live would have it reason from an old summary while a newer
        document sits on disk, and nothing would reveal the mismatch.

        So it is told two things plainly: when the snapshot was taken, and that
        ``load_document`` reads the CURRENT text regardless. Not dressing a snapshot
        up as live data is the whole of the honesty here.
        """
        import time as _t

        from mast.agents._shared.artifact_channel import field_label

        when = ""
        if snapshot_at:
            age = max(0.0, _t.time() - snapshot_at)
            when = (f"（快照时间:{int(age // 60)} 分钟前）" if age >= 60
                    else "（快照时间:刚刚）")
        names = "、".join(field_label(f) for f in sorted(seeded)) or "（无）"
        return (
            "## 你是被「唤醒」启动的\n"
            f"下面这些上游产物的指针是从主线**复制**过来的快照{when}:{names}。\n"
            "- 指针可能已经过时:主线可能在快照之后又存了新版本。\n"
            "- **`load_document(doc_id)` 读到的永远是磁盘上的最新版**,正文以它为准。\n"
            "- 需要确认环境现在到底有什么,用 `survey_environment()`。"
        )

    def _build_instrument_loop_v2(self):
        """IC 的 v2 私聊循环，装配完整硬件安全中间件。
        
        注入项与私聊 IC 图逐项对应；安全限值使用合并后的有效值，使工具 schema
        与执行闸门范围一致。安全组件缺失时拒绝装配，不能运行不完整的控制循环。"""
        return self.build_instrument_loop_for()

    def build_instrument_loop_for(self, *, registry=None, model=None,
                                  system_suffix: str = "",
                                  max_model_calls: int | None = None,
                                  max_tool_calls: int | None = None,
                                  enable_hitl: bool = True):
        """公开入口：与私聊 IC **同一套安全件**，只放开可换的几个槽。

        为谁而设：离线驱动（benchmark harness、回放、脚本化评测）要拿着**完整的**
        硬件安全栈跑 IC，但换一个模型、收窄一份技能表、在系统提示尾部声明本次的
        工具面、或把调用上限调到不绑定。以前它们只能摸 ``_chat_context_provider`` /
        ``_current_operating_mode`` / ``_chat_call_limits`` 三个私有属性自己装 ——
        私有名字一改，harness 静默装出一个少安全件的 IC（`build_instrument_loop`
        会抛，但抛在 harness 里而不是这里）。

        ``registry=None`` / ``model=None`` = 私聊的默认；``max_*_calls=None`` = 用户
        在设置里调的那份上限。安全件（限值合并、记录器、HITL）**不可换**。
        """
        from mast.agentruntime.ic_assembly import build_instrument_loop

        # ``_chat_call_limits`` speaks the LangGraph-era vocabulary (per-run caps under
        # ``*_per_run``, cumulative thread caps under ``max_*_calls``); the v2 loop's
        # CallLimits has only per-run caps. Translate — passing the dict through as-is
        # raised ``TypeError: unexpected keyword 'max_model_calls_per_run'`` (found
        # 2026-08-28 by the STM-Bench driver; engine_v2 is off by default, so the
        # private-chat path had never been built). Thread caps have no v2 slot yet.
        raw = dict(self._chat_call_limits())

        def _cap(value, default: int) -> int:
            # settings semantics (call_limits._norm_run): 0 = no limit, and it says so
            try:
                v = int(value) if value not in (None, "") else int(default)
            except (TypeError, ValueError):
                v = int(default)
            return v if v > 0 else 10 ** 9

        limits = {
            "max_model_calls": _cap(raw.get("max_model_calls_per_run"), 30),
            "max_tool_calls": _cap(raw.get("max_tool_calls_per_run"), 80),
        }
        if max_model_calls is not None:
            limits["max_model_calls"] = int(max_model_calls)
        if max_tool_calls is not None:
            limits["max_tool_calls"] = int(max_tool_calls)
        return build_instrument_loop(
            buf=getattr(self, "_buffer", None),
            model=model,                      # None = 按 agent 取默认模型
            get_state=(self._state.snapshot if getattr(self, "_state", None)
                       else None),
            get_mode=self._current_operating_mode,
            context_provider=self._chat_context_provider,
            registry=registry if registry is not None else self._registry,
            safety_limits=_chat_effective_safety_limits(
                getattr(self.config, "safety", None)),
            safety_recorder=getattr(self, "_safety_trace_recorder", None),
            turn_recorder=getattr(self, "_turn_trace_recorder", None),
            enable_hitl=enable_hitl,
            system_suffix=system_suffix or "",
            **limits)

    def v2_router_model(self):
        """v2 编排器路由用的 chat model（每个 runtime 一份，惰性）。

        ⚠️ **是 CoreRuntime 的方法，不是 API 层往 runtime 上挂的一个临时属性。**
        第一版把缓存写成 ``getattr(app, "_v2_router_model", None)``，
        ``test_api_layer_never_reaches_for_a_nonexistent_core_attribute`` 当场变红 ——
        那道闸门守的是「API 中继向核心要一个它没有的东西，会永远静默降级」，
        而缓存一个模型本来就是核心的事，不是中继的。

        建不起来返回 None：调用方据此退回旧路径，而不是拿一个半吊子路由去派发。
        """
        cached = getattr(self, "_v2_router_model_cache", None)
        if cached is not None:
            return cached
        try:
            from mast.agents._shared.models import make_chat_model

            model = make_chat_model("orchestrator")
        except Exception as exc:  # noqa: BLE001
            logger.warning("v2 router model unavailable: %s", exc)
            return None
        self._v2_router_model_cache = model
        return model

    def _v2_private_chat_enabled(self) -> bool:
        """``engine_v2_private_chat`` —— 退出 LangGraph 的第三个切换面。

        DEFAULT OFF / fail-safe OFF。**在 boot 时读一次**，与另外两个开关（每次调用
        live-read）刻意不同：私聊的历史是连续的，两个引擎交替写同一段会让「这一轮
        用哪个引擎读到什么」变成一个没人能回答的问题。boot 级选择把它整个排除，
        切换回另一引擎需要重启。
        """
        st = getattr(self, "_settings", None)
        try:
            return bool(st.get("engine_v2_private_chat")) if st is not None else False
        except Exception:  # noqa: BLE001 — 读不到开关 = 旧路径
            return False

    def _v2_background_enabled(self) -> bool:
        """``engine_v2_background`` —— 退出 LangGraph 的第二个切换面。

        DEFAULT OFF / live-read（每次 spawn 读一次）/ fail-safe OFF，与
        ``orchestrator_auto_background`` 同形状。翻回去在下一个后台 run 即刻生效。
        """
        st = getattr(self, "_settings", None)
        try:
            return bool(st.get("engine_v2_background")) if st is not None else False
        except Exception:  # noqa: BLE001 — 读不到开关 = 旧路径
            return False

    def _background_run_fn(self, instruction, thread_id, agents, emit, abort):
        """Stream one isolated background orchestrator run, forwarding each agent
        message to ``emit``. Runs on the manager's daemon thread — blocking here is
        fine (it is NOT an agents/**/graph.py node). Returns the final answer text."""
        if self._v2_background_enabled():
            try:
                return self._background_run_fn_v2(instruction, thread_id, agents,
                                                  emit, abort)
            except Exception as exc:  # noqa: BLE001
                # 回退**留痕**：静默兜底会让「新引擎一直在挂」和「工作正常」长得
                # 一模一样 —— 那正是这次迁移要根除的形状。
                logger.warning("v2 background orchestration failed (%s); falling "
                               "back to the LangGraph path", exc)
                try:
                    emit("_supervisor", "assistant",
                         f"（v2 编排失败已回退：{type(exc).__name__}: {exc}）")
                except Exception:  # noqa: BLE001
                    pass
        orch = self._build_background_orchestrator()
        if orch is None:
            raise RuntimeError("background orchestrator unavailable (no chat-model key?)")
        from langchain_core.messages import HumanMessage

        def _ns_agent(ns):
            if not ns:
                return None
            head = ns[0] if isinstance(ns, (tuple, list)) else str(ns)
            return str(head).split(":", 1)[0] or None

        def _text(m):
            c = getattr(m, "content", "") or ""
            if isinstance(c, list):
                c = " ".join(str(x) for x in c)
            return str(c)

        # Seeded product snapshot (2026-07-30). A background run is state-ISOLATED —
        # own thread_id, own InMemorySaver — which is what makes it safe to run
        # concurrently and also what made it blind: nothing the foreground produced
        # was visible to it. The snapshot is merged BY VALUE (never a shared channel:
        # LangGraph checkpoints are per-thread, so two runs on one thread corrupt each
        # other). Only pointers travel; the bodies stay on disk, which is shared and
        # authoritative — so load_document(doc_id) here reads the real latest text.
        seeded = dict(getattr(emit, "seed_artifacts", None) or {})
        snapshot_at = float(getattr(emit, "snapshot_at", 0.0) or 0.0)
        lead: list = []
        if seeded:
            from langchain_core.messages import SystemMessage

            lead.append(SystemMessage(content=self._seed_notice(seeded, snapshot_at)))
        state = {
            "messages": lead + [HumanMessage(content=instruction)],
            "executed_skills": [], "scan_paths": [], "scan_metadata": {},
            "error_log": [], "event_refs": [], "visit_count": {},
            "pending_approvals": {},
            **seeded,
        }
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 300}
        seen: set = set()
        final = ""
        for ns, chunk in orch.stream(state, config=cfg, stream_mode="updates",
                                     subgraphs=True):
            if abort.is_set():
                break
            if not isinstance(chunk, dict) or "__interrupt__" in chunk:
                continue
            owner = _ns_agent(ns)
            for node_name, node_state in chunk.items():
                resolved = owner or ("_supervisor" if node_name in (
                    "supervisor", "__start__", "__end__") else node_name)
                msgs = (node_state or {}).get("messages") or [] if isinstance(node_state, dict) else []
                # Context compaction: this run's history was
                # just replaced by a summary. It merges into the SAME group
                # transcript the operator reads, so it must leave a marker there
                # too instead of the summary being dropped as an "operator echo"
                # by the HumanMessage skip below.
                _comp = _background_compaction_event(msgs)
                if _comp is not None:
                    for m in msgs:
                        seen.add(getattr(m, "id", None) or id(m))
                    _mark = getattr(emit, "compaction", None)
                    if _mark is not None:
                        try:
                            _mark(resolved, _background_compaction_line(_comp))
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("background compaction marker failed: %s", exc)
                    continue
                for m in msgs:
                    mid = getattr(m, "id", None) or id(m)
                    if mid in seen:
                        continue
                    seen.add(mid)
                    cls = m.__class__.__name__
                    if cls == "HumanMessage":
                        continue
                    text = _text(m).strip()
                    # skip internal routing/handoff plumbing — keep real agent output
                    if not text or text.startswith(("[SUPERVISOR", "[HANDOFF")):
                        continue
                    role = ("agent" if cls == "AIMessage"
                            else "tool" if cls == "ToolMessage" else "agent")
                    emit(resolved, role, text[:2000])
                    if cls == "AIMessage":
                        final = text[:1000]
        # File this run's products back to the mainline (2026-07-30). Without this a
        # woken run's work exists on disk and is invisible to everyone — the same as
        # not having done it. It cannot write the mainline's checkpoint (per-thread;
        # two writers corrupt it), so it posts to the durable inbox and the
        # supervisor drains it at its next dispatch.
        self._post_background_products(orch, cfg, emit, agents)
        return final

    def _background_run_fn_v2(self, instruction, thread_id, agents, emit, abort):
        """同一件事，跑在 ``agentruntime.OrchestratorLoop`` 上。

        对调用方（``BackgroundRunManager``）完全同形：同样的
        ``emit(agent_id, role, text)`` 协议、同样返回最终文本，所以结果照旧流回同一个
        transcript，前端不知道换了引擎。

        与旧路径的三处**可见**差别（都是改善）：

        * 分支的**非正常结局会出现在 transcript 里**。旧路径下一次「限流停住」和一次
          「干完了」在记录上分不开——不交棒的分支什么都不说。
        * 路由解析不出来时**停下等人**，而不是猜一个目标。
        * 产物直接从分支返回值里来，不需要事后再去 checkpointer 里 ``carried_from``
          一遍（那一步在这条路上因此不存在——不是省略，是没有那个中间态）。
        """
        from mast.agentruntime.background import run_background_task

        router = getattr(self, "_bg_router_model", None)
        if router is None:
            from mast.agents._shared.models import make_chat_model

            router = make_chat_model("orchestrator")
            self._bg_router_model = router

        seeded = dict(getattr(emit, "seed_artifacts", None) or {})
        snapshot_at = float(getattr(emit, "snapshot_at", 0.0) or 0.0)
        notice = self._seed_notice(seeded, snapshot_at) if seeded else ""

        final = run_background_task(
            instruction=instruction, agents=list(agents), emit=emit, abort=abort,
            router_model=router, seed_artifacts=seeded, seed_notice=notice,
            run_id=str(getattr(emit, "run_id", "") or thread_id))
        return final

    def _post_background_products(self, orch, cfg, emit, agents) -> None:
        """Post a finished background run's artifact pointers to the return inbox.

        Reads the FINAL state from the run's own checkpointer rather than
        accumulating from the stream: the stream carries messages, while the products
        live in typed channels, and ``carried_from`` is the same function the handoff
        customs desk uses — so a product crosses this boundary exactly as it would
        cross an in-run one. One definition of "what counts as a product", not two.

        Entirely best-effort. A run that already did its work must not be reported as
        failed because the mailbox was unwritable.
        """
        try:
            from mast.agents._shared.artifact_channel import carried_from
            from mast.core.park_board import board

            snap = orch.get_state(cfg)
            products = carried_from(getattr(snap, "values", None) or {})
            if not products:
                return
            board().post_return(
                run_id=str(getattr(emit, "run_id", "") or ""),
                park_id=str(getattr(emit, "park_id", "") or ""),
                agent=(agents[0] if agents else ""),
                artifacts=products,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("background products not posted: %s", exc)

    # ── Agents-UI live snapshot hooks (wired onto ctx in api.bootstrap) ────
    # Read-only views of the in-process Agents-API state populated by the
    # orchestrator run-task stream (routes/orchestrator.py) + the hold/interject/
    # resolve endpoints (routes/agents_control.py). They give /agents/snapshot,
    # /agents/{id}/interrupts and /artifacts the LIVE overlay the old MASTApp
    # served; without them those endpoints degrade to a static registry view.
    def agents_snapshot(self) -> dict:
        """Live overlay for GET /api/agents/snapshot: active task, per-agent
        holds, handoff timeline, thread counts, capability flags. Never raises.

        Reads under the state lock so a concurrent run-task worker mutation
        (handoff append / active_agent_id / active flip) yields a consistent view."""
        import contextlib as _ctxlib
        st = getattr(self, "_agents_api_state", None) or {}
        _lock = st.get("lock") if isinstance(st, dict) else None
        with (_lock or _ctxlib.nullcontext()):
            task = dict(st.get("task") or {})
            holds = {k: bool(v) for k, v in (st.get("holds") or {}).items() if v}
            handoffs = list(task.get("handoffs") or [])
            threads = dict(task.get("threads") or {})
        handoff_events = [
            {"t": h.get("t"), "kind": h.get("kind", "handoff"),
             "text": h.get("text", ""),
             # Fan-out targets, so the topology can light up every agent the
             # supervisor started concurrently (empty on a serial handoff).
             "targets": list(h.get("targets") or [])}
            for h in handoffs[-50:] if isinstance(h, dict)
        ]
        threads_index = {k: len(v or []) for k, v in threads.items()}
        active_task = None
        if task:
            active_task = {
                "id": task.get("id"),
                "description": task.get("description", ""),
                "active": bool(task.get("active")),
                "final_text": task.get("final_text", ""),
                "error": task.get("error"),
            }
        return {
            "active_agent_id": task.get("active_agent_id"),
            # Every agent currently running. Parallel fan-out makes "the active
            # agent" plural; active_agent_id keeps the old single-slot contract
            # (it carries a joined label when several are live).
            "active_agents": list(task.get("active_agents") or []),
            "holds": holds,
            "handoff_events": handoff_events,
            "threads_index": threads_index,
            "active_task": active_task,
            "capabilities": {
                "interject": True,
                "hold": True,
                "abort": True,
                "model_switch_live": True,
                "tool_visibility": True,
                "buffer_summarizer_active": getattr(self, "_buffer", None) is not None,
            },
        }

    def agents_interrupts(self, agent_id: str = "__all__") -> list[dict]:
        """Live pending HITL interrupts (DANGEROUS skill gates + workflow-human
        nodes), optionally filtered to one agent.

        **``[]`` 是一句正面断言：「问过活闸门了,这会儿零条待批」。** 所以读不到
        的时候**抛**,不回 ``[]``。

        下游那条路早就写好了,只是一直走不到:
        ``routes/agents_topology._agents_interrupts`` 分「拿不到 ⇒ ``None``」与
        「拿到一个 list」,``get_agent_interrupts`` 再把 ``None`` 变成
        ``degraded=True``,并且把异常文本记进 warning —— 所以「为什么读不到」
        也不会丢。原来这里 ``except Exception: return []`` 递过去的永远是一个
        合法的空 list,于是路由发出 ``count=0, interrupt_gating=True,
        degraded=False``:「门禁开着,没有人在等」。而真相可以是一个 DANGEROUS
        技能正停在闸门上等批准 —— 用户面板上什么都没有,机器停着,没有任何
        东西说出了问题。**能停不能解。**(2026-08-15 普查 A1)

        没有 store 也抛,理由是同一条:``getattr(...) or {}`` 会让「这台机器没有
        闸门」和「闸门是空的」长得一模一样。同理 ``pending`` 缺席/不是 dict ⇒ 抛;
        只有一个**真的空** ``{}`` 才配回 ``[]``。
        """
        store = getattr(self, "_orch_interrupts", None)
        if not isinstance(store, dict):
            raise RuntimeError(
                "HITL 中断存储不可用(_orch_interrupts 缺失或不是 dict)—— "
                "这不是「零条待批」,是问不出来")
        raw = store.get("pending")
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"HITL 中断存储的 pending 表读不出来(是 {type(raw).__name__})—— "
                "这不是「零条待批」,是问不出来")
        # 刻意不吞异常:锁抛、迭代中被改写,都要让路由那条 degraded 路听见。
        lock = store.get("lock")
        if lock is not None:
            with lock:
                pending = list(raw.values())
        else:
            pending = list(raw.values())
        if agent_id and agent_id not in ("__all__", "_supervisor"):
            pending = [p for p in pending if isinstance(p, dict)
                       and p.get("agent_id") == agent_id]
        return [p for p in pending if isinstance(p, dict)]

    def agents_artifacts(self) -> dict:
        """Live workspace artifacts ``{produced, edits}`` (run-produced files +
        operator edits) for GET /api/artifacts. Never raises.

        ## 关于这里的 ``or {}`` 链(2026-08-15 普查时查过,**结论是不动**)

        它和上面 ``agents_interrupts`` 长得是同一个形状——「读不到」与「真的是
        空的」折成同一个值——但**它没有咬到任何人**,因为没有调用方读它的内容:

        * 唯一的消费方是 ``routes/agents_topology`` 的
          ``session_active=bool(_agents_artifacts(ctx))``。这个 dict 恒有两个键
          ⇒ 恒为真;唯一能让它变 False 的是 hook 缺席/抛出(那时 helper 回
          ``None``)。所以 ``session_active`` 量的是「有没有一个活 runtime 应答」,
          正是它的名字所指。
        * ``/api/artifacts`` 的清单来自磁盘那一次目录遍历,不是这个 hook。

        **要推翻这个结论需要出现什么**:任何一个开始读 ``produced`` / ``edits``
        **内容**的调用方——比如拿 ``not produced`` 判「这轮什么都没产出」。那一刻
        这里就必须像 ``agents_interrupts`` 一样,把「没有 state」与「state 里是空
        的」分开。在那之前多加一层三态,只是给一个没人问的问题增加答案。
        """
        st = getattr(self, "_agents_api_state", None) or {}
        task = st.get("task") or {}
        return {
            "produced": task.get("artifacts") or {},
            "edits": st.get("artifact_edits") or {},
        }

    @staticmethod
    def _has_real_env_sensors(sensors) -> bool:
        """True if any sensor is a real serial driver (not a placeholder).

        Delegates to environment.autodetect.has_real_sensors so this boot-time
        decision and the live /environment/sensors/rescan path can never drift
        apart about what counts as real hardware — they both gate whether the
        archive/alarm loop runs.
        """
        try:
            from mast.environment.autodetect import has_real_sensors
        except Exception:
            return False
        return has_real_sensors(sensors)


    def _latest_pressure_sample(self):
        """The newest vacuum reading, tagged with the class that produced it.

        Returns ``None`` rather than a zero when there is nothing to report —
        the interlock treats "no sample" as a refusal, and a fabricated 0.0 Pa
        would read as a perfect vacuum.

        The class name is half the value here. ``PlaceholderSensor`` answers with
        ``value=0.0, status="unavailable"`` on a machine with no gauge wired at
        all, so the number alone cannot distinguish "excellent vacuum" from "no
        instrument". Passing the class through lets the interlock refuse on
        identity, before it ever looks at the reading."""
        try:
            from mast.core.vacuum_interlock import PressureSample

            mon = getattr(self, "_monitor", None)
            if mon is None:
                return None
            latest = mon.get_latest() or {}
            sensors = getattr(mon, "_sensors", {}) or {}

            # 先按单位或类型选候选，再在同类候选中优先选择可读的真实传感器。
            # 不能让名称匹配的占位传感器遮住有效读数，也不能要求温度通道名含 temp。
            # 名称仅用于同类通道的偏好排序，不代替物理量类型判断。
            def _unit_of(reading) -> str:
                return str(getattr(reading, "unit", "") or "").strip().lower()

            names = [n for n, r in latest.items() if _unit_of(r) in ("pa", "mbar")]
            if not names:
                # 兜底：单位缺失或用了本表没列的写法（如 Torr）时，仍按名字找一次。
                names = [n for n in latest if "vacuum" in str(n).lower()]
            if not names:
                return None
            # 活的优先。**都不活时仍然返回一个** —— 互锁需要 `sensor_class` 才能给出
            # 那句「真空计是占位实现，读数恒为 0，这不是『完美真空』而是『没有数据』」
            # 的拒绝理由；返回 None 会把那句诊断连同拒绝的**原因**一起丢掉，
            # 用户只会看到一句「读不到」。
            names.sort(key=lambda n: str(getattr(latest[n], "status", "")) != "ok")
            name = names[0]
            reading = latest[name]
            sensor = sensors.get(name)
            return PressureSample(
                value=float(getattr(reading, "value", 0.0) or 0.0),
                unit=str(getattr(reading, "unit", "") or ""),
                status=str(getattr(reading, "status", "") or ""),
                timestamp=str(getattr(reading, "timestamp", "") or ""),
                sensor_name=str(name),
                sensor_class=type(sensor).__name__ if sensor is not None else "",
            )
        except Exception as exc:  # noqa: BLE001 — no sample is a safe answer
            logger.debug("latest pressure sample unavailable: %s", exc)
            return None

    def temperature_channels(self) -> "list":
        """这台机器上所有**温度型**通道（``list[TempChannel]``）。永不抛。

        枚举的是**传感器集合**，不只是「读到过的」：``monitor.get_latest()`` 只在
        归档循环跑起来之后才有内容，而开机时没有真硬件那条循环压根不启动。只看
        读数缓存就会把「配了温度计但这一拍还没读」说成「这台机器没有温度计」——
        两个问题，两个不同的真源：**传感器集合**回答「有没有」，**读数缓存**回答
        「现在多少度」。

        ``real`` 三态由 ``autodetect.is_real_sensor`` 判（与决定归档循环跑不跑的
        是同一个谓词），拿不到传感器对象时留 ``None``（不知道），**不当占位**。
        """
        from mast.core.temperature import TempChannel

        mon = getattr(self, "_monitor", None)
        if mon is None:
            return []
        try:
            latest = dict(mon.get_latest() or {})
        except Exception as exc:  # noqa: BLE001 — 读不到就当没有读数
            logger.debug("temperature channels: get_latest failed: %s", exc)
            latest = {}
        try:
            sensors = dict(getattr(mon, "_sensors", {}) or {})
        except Exception:  # noqa: BLE001 - defensive
            sensors = {}
        try:
            from mast.environment.autodetect import is_real_sensor
        except Exception:  # noqa: BLE001 — 分不出真假就一律「不知道」
            is_real_sensor = None  # type: ignore[assignment]

        out: list = []
        for name in dict.fromkeys(list(sensors) + list(latest)):
            reading = latest.get(name)
            sensor = sensors.get(name)
            inner = getattr(sensor, "_inner", sensor) if sensor is not None else None
            unit = str(getattr(reading, "unit", "") or "")
            # 还没读过时读数上没有单位 —— 回落到驱动自己声明的那个，
            # 否则「配了但没读过」的通道会因为单位为空而不被认成温度通道。
            declared_unit = str(getattr(inner, "_unit", "") or "")
            if not self._is_temperature_channel(inner, unit or declared_unit):
                continue
            real: "bool | None" = None
            if inner is not None and is_real_sensor is not None:
                try:
                    real = bool(is_real_sensor(inner))
                except Exception:  # noqa: BLE001
                    real = None
            out.append(TempChannel(
                name=str(name),
                value=getattr(reading, "value", None),
                unit=unit or declared_unit,
                status=str(getattr(reading, "status", "") or ""),
                timestamp=str(getattr(reading, "timestamp", "") or ""),
                driver=type(inner).__name__ if inner is not None else "",
                real=real,
            ))
        return out

    @staticmethod
    def _is_temperature_channel(inner, unit: str) -> bool:
        """这个通道量的是温度吗。单位优先，驱动类型兜底。

        两条判据都要：单位在**没读过**的通道上是空的，而驱动类型在只有一份读数
        快照（没有传感器对象）时是空的。任一成立即算。
        """
        from mast.core.temperature import is_temperature_unit

        if is_temperature_unit(unit):
            return True
        if inner is None:
            return False
        for module, cls_name in (
            ("mast.environment.lakeshore_temp", "LakeshoreTemperatureSensor"),
            ("mast.environment.placeholders", "TemperatureSensor"),
        ):
            try:
                mod = __import__(module, fromlist=[cls_name])
                if isinstance(inner, getattr(mod, cls_name)):
                    return True
            except Exception:  # noqa: BLE001 — 驱动不可用就只靠单位判
                continue
        return False

    def latest_temperature(self, channel: "str | None" = None):
        """**公共**温度只读口 —— ``TempReading``（值 + 年龄 + 出处 + 为什么没有值）。

        取代私有的 ``_latest_temperature_k``（它现在委托到这里）。差别不在多返回了
        几个字段，而在**「读不到」不再折叠成一个 None**：没装温度计（等下去永远等
        不到）、装了但此刻读不到（关掉占着 COM 口的程序就好）、值太旧（``age_s``
        随值返回，多旧算旧由调用方定）是三件要做不同事的事。

        只读、不取仪器令牌、永不抛。设计：``docs/v2/design/`` 的 P0 修复设计,
        修复项 节(通用层注释不写样品名前缀,见 ``stm_capability_vs_sample_layer.md``)。
        """
        from mast.core.temperature import NO_SOURCE, TempReading, read_temperature

        try:
            return read_temperature(self.temperature_channels(), channel=channel)
        except Exception as exc:  # noqa: BLE001 — 温度是注记，绝不能弄崩调用方
            logger.debug("latest_temperature failed: %s", exc)
            return TempReading(channel=str(channel or ""), reason=NO_SOURCE)

    def _latest_temperature_k(self) -> "float | None":
        """返回最新的开尔文温度，读不到时返回 None；永不抛异常。
        
        此兼容入口委托公共 latest_temperature；需要区分未安装、不可读与陈旧状态的
        调用方应使用公共结果。里程表仅作温度注记时仍可使用这个简化接口。
        
        温度通道按 unit 识别，不依赖名称含 temp；换算也依据 unit。
        多通道时优先样品台相关名称，其次使用任意有效开尔文读数。温度影响粗动数据的
        可比性，但这里仅记录注记，不能把磁体或其他部位的温度冒充样品台标定。"""
        _STAGE_HINTS = ("spm", "stage", "sample", "tip", "cryo")

        def _kelvin(reading) -> "float | None":
            if str(getattr(reading, "status", "")) != "ok":
                return None
            unit = str(getattr(reading, "unit", "") or "").strip().lower()
            try:
                val = float(getattr(reading, "value", None))
            except (TypeError, ValueError):
                return None
            if unit in ("k", "kelvin"):
                return val
            if unit in ("c", "°c", "degc", "celsius"):
                return val + 273.15
            return None

        try:
            mon = getattr(self, "_monitor", None)
            if mon is None:
                return None
            hits = []
            for name, reading in (mon.get_latest() or {}).items():
                k = _kelvin(reading)
                if k is not None:
                    hits.append((str(name), k))
            if not hits:
                return None
            for name, k in hits:
                if any(h in name.lower() for h in _STAGE_HINTS):
                    return k
            return hits[0][1]
        except Exception as exc:  # noqa: BLE001 — an annotation, never a blocker
            logger.debug("latest temperature unavailable: %s", exc)
        return None

    def _coarse_map_inputs(self):
        """``(marker rows, CoarseMapConfig)`` for the live scope, or None.

        Returning None (rather than an empty list) matters: the destination
        check must be able to tell "nothing recorded" from "cannot read the
        record", because the first means a fresh sample and the second means
        the "have we been here" answer is unknown."""
        try:
            storage = getattr(self, "_storage", None)
            if storage is None:
                return None
            el = getattr(self, "_experiment_log", None)
            exp_id = getattr(el, "current_experiment_id", None) if el else None
            sample_id = getattr(el, "current_sample_id", None) if el else None
            rows = storage.get_markers(exp_id, sample_id) or []
            return rows, self.build_coarse_map_config()
        except Exception as exc:  # noqa: BLE001
            logger.debug("coarse map inputs unavailable: %s", exc)
            return None

    def _log_vacuum_interlock_event(self, event: str, payload: dict) -> None:
        """Audit an attestation into the durable environment log.

        An operator signing "the pressure is safe" is a safety decision with a
        name on it; it belongs in the record next to the readings it stands in
        for, not only in a process-local variable that dies with the run."""
        try:
            if self._storage is None:
                return
            self._storage.log_environment(
                f"vacuum_interlock_{event}", 0.0, "",
                str(payload.get("reason") or event)[:120])
        except Exception as exc:  # noqa: BLE001 — best-effort audit
            logger.debug("vacuum interlock audit log failed: %s", exc)

    def _on_env_alarm(self, name: str, reading, prev_status: str) -> None:
        """EnvironmentMonitor callback on an over-limit / fault transition.

        Keeps a capped in-memory log (surfaced by GET /environment/alarms) AND —
        for a genuine hard ``alarm`` / ``error`` transition — actively cuts the
        experiment loss instead of just logging into the void (review
        2026-07-03): a vacuum failure or thermal runaway during an unattended
        overnight run used to reach only an unread list. Now a hard alarm:
          1. persists to the durable environment_log table,
          2. sets the orchestrator abort Event (stops the autonomous run),
          3. bridges an E_STOP into the vision buffer (UI banner + agent-path
             abort hook), and
          4. best-effort retracts the tip via the emergency port (protects it
             from a vacuum loss / thermal drift).
        A soft ``warning`` only records + surfaces; it does not stop the run.
        """
        try:
            import time as _t
            self._env_alarm_log.append({
                "t": _t.strftime("%H:%M:%S"),
                "sensor": name,
                "status": reading.status,
                "value": reading.value,
                "unit": reading.unit,
                "prev": prev_status,
            })
            # cap to last 50
            if len(self._env_alarm_log) > 50:
                self._env_alarm_log = self._env_alarm_log[-50:]
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("_on_env_alarm record failed: %s", exc)

        # Persist every transition (warning + alarm) to the durable log so the
        # overnight history survives a restart. Best-effort.
        try:
            if self._storage is not None:
                self._storage.log_environment(
                    name, float(reading.value), reading.unit or "", reading.status)
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("_on_env_alarm persist failed: %s", exc)

        # status=error 仅说明读取失败，不能据此断定物理环境越限。
        # alarm 表示有效读数确实越限，继续执行其停止、退针和闩锁策略；
        # error 只记录并报告读取问题，不在此路径写硬件。
        # 独立 SafetyWatchdog 直接读取电流，不能由环境镜像读取失败替代其判据。
        if reading.status == "error":
            logger.critical(
                "环境传感器读不到: %s (status=error, was %s) —— **不停运行、"
                "不退针**。读失败本身不携带任何关于世界的信息;只有真正越限"
                "(status=alarm)才触发停机。若这条反复出现,去查那个传感器的"
                "读取路径,不要去查真空/温度。",
                name, prev_status,
            )
            return
        # Only a HARD alarm triggers stop-loss. A soft warning is surfaced
        # (above) but does not abort a run or move hardware.
        if reading.status != "alarm":
            return
        _why = (f"环境告警:{name} = {reading.value} {reading.unit or ''} "
                f"越限(was {prev_status})")
        logger.critical(
            "ENVIRONMENT ALARM: sensor %s = %s %s (status=%s, was %s) — "
            "aborting autonomous run + retracting tip",
            name, reading.value, reading.unit or "", reading.status, prev_status,
        )
        # 2. Stop the autonomous run (shared abort Event; never replace, only set).
        try:
            from mast.core.execution_context import mark_abort
            ab = getattr(self, "_orch_abort", None)
            if ab is not None:
                # 留下**是谁停的**。不留的话,下游那句拒绝会写死成
                # 「用户已中止本次运行」—— 而用户根本没碰过它。
                mark_abort(ab, _why)
                # An environment alarm is an emergency: latch it so the next
                # run-task cannot clear it out from under a halted composite.
                self._orch_abort_emergency = True
                self._orch_abort_why = _why
            with self._orch_run_aborts_lock:
                for _rev in self._orch_run_aborts.values():
                    _rev.set()
            ex_ab = getattr(getattr(self, "_executor", None), "_abort_event", None)
            if ex_ab is not None:
                ex_ab.set()
        except Exception as exc:  # pragma: no cover
            logger.warning("env-alarm abort set failed: %s", exc)
        # 3. Bridge to the vision buffer as an E_STOP (UI banner + agent abort).
        try:
            from mast.buffer.active import get_active_buffer
            buf = get_active_buffer()
            if buf is not None:
                from mast.buffer.schemas import make_e_stop
                buf.emit_event(make_e_stop(
                    "environment",
                    f"{name} {reading.status}: {reading.value} {reading.unit or ''}",
                    seqno=buf.next_seq(),
                ))
        except Exception as exc:  # pragma: no cover
            logger.warning("env-alarm E_STOP emit failed: %s", exc)
        # 4. Best-effort emergency retract (protect the tip). Never raise.
        try:
            if self._pool is not None:
                rec = self._pool.safe_call("ZCtrl_Withdraw", 1, -1, role="emergency")
                if getattr(rec, "error", ""):
                    # emergency port may be down — try the main port.
                    self._pool.safe_call("ZCtrl_Withdraw", 1, -1, role="main")
        except Exception as exc:  # pragma: no cover
            logger.warning("env-alarm retract failed: %s", exc)


    def reload_overlay_skills(self, *, reason: str = "manual") -> dict:
        """重新加载技能覆盖层 —— UI 上那个「重新加载技能」按钮的后端。

        两件事必须一起做，缺一个就是「以为生效其实没有」：

        1. 重算注册表（``overlay.loader.reload_skills``）；
        2. 把三条消费者链都推一遍（``reload_wiring.refresh_after_skill_change``）
           —— 每个 agent 的工具表是**建图时**冻结的（全仓没有 ``bind_tools``），
           只换注册表的话模型手上还是旧的。

        任务运行中**两件都不做**，只排队：``ExecutionContext.run`` 每个子步骤都
        现查注册表，中途换会让跑到一半的 composite 后半段用新代码。
        """
        from mast.admin import reload_wiring as _rw
        from mast.skills.overlay.loader import reload_skills

        busy = self._task_is_busy()
        rep = reload_skills(self._registry, reason=reason, task_busy=busy)
        out: dict = {
            "status": rep.status,
            "summary": rep.describe(),
            "applied": [r.rel for r in rep.applied],
            "failed": [{"path": r.rel, "reason": r.reason} for r in rep.failed],
            "restored": rep.restored,
            "baseline_drift": rep.baseline_drift,
        }
        if busy:
            # 排队的是**整件事**（注册表 + 图），由 drain 一并做掉。
            with self._pending_rebuild_lock:
                self._pending_agent_rebuild = {
                    "reason": f"overlay reload: {reason}",
                    "at": _time_mod_time(),
                    "overlay": True,
                }
            out["refresh"] = {"queued": True}
            return out

        outcome = _rw.refresh_after_skill_change(f"overlay reload: {reason}")
        out["refresh"] = {
            "guard_reloaded": outcome.guard_reloaded,
            "chat_invalidated": outcome.chat_invalidated,
            "orchestrator": outcome.orchestrator,
            # 诚实：排队了 ≠ 已生效。
            "agent_path_pending": outcome.agent_path_pending,
            "described": outcome.describe(),
        }
        return out

    def _task_is_busy(self) -> bool:
        try:
            return bool((((getattr(self, "_agents_api_state", None) or {})
                          .get("task")) or {}).get("active"))
        except Exception:  # pragma: no cover — defensive
            return False

    def request_agent_rebuild(self, reason: str = "") -> tuple[str, str]:
        """Rebuild the agent tool table. Returns ``(code, ui_suffix)``.

        ``code`` is one of ``reload_wiring.ORCH_{REBUILDING, ABSENT,
        PENDING_QUEUED, FAILED}`` — a STRUCTURED answer, so callers stop having
        to sniff a Chinese sentence for a substring.

        Why a rebuild is needed at all: hot-(un)registration only updates the
        SkillRegistry, but every agent's tool list was frozen when its graph was
        built (there is no bind_tools anywhere) — without a rebuild a deleted
        composite stays callable, a rolled-back one keeps running the OLD
        version, and a widened envelope never reaches the model because the old
        pydantic schema rejects it first (admin/reload_wiring.py:25-30).

        While a task streams the rebuild is QUEUED, not dropped.
        """
        from mast.admin import reload_wiring as _rw

        if self._task_is_busy():
            with self._pending_rebuild_lock:
                self._pending_agent_rebuild = {
                    "reason": reason or "skill/override change",
                    "at": time.time(),
                }
            logger.info("agent rebuild QUEUED (task active): %s", reason)
            # NB: this suffix must keep containing the "task running" phrase --
            # reload_wiring's legacy branch still substring-matches it for
            # runtimes that predate this method (reload_wiring.py:141).
            return (_rw.ORCH_PENDING_QUEUED,
                    "（任务运行中，agent 工具表已排队——当前任务结束后自动生效）")

        if getattr(self, "_orchestrator", None) is None:
            return (_rw.ORCH_ABSENT, "")

        def _do():
            try:
                self._build_orchestrator()
                logger.info("Orchestrator rebuilt (%s)", reason or "unspecified")
            except Exception as exc:  # noqa: BLE001
                logger.warning("orchestrator rebuild failed (%s): %s", reason, exc)

        try:
            _threading_mod.Thread(target=_do, name="agent-tools-rebuild",
                                  daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not schedule orchestrator rebuild: %s", exc)
            return (_rw.ORCH_FAILED, "（agent 工具表重建调度失败——重启生效）")
        return (_rw.ORCH_REBUILDING, "（agent 工具表后台重建中，数秒后生效）")

    def pending_agent_rebuild(self) -> dict | None:
        """The queued rebuild, if any (read-only; for UI/status endpoints)."""
        with self._pending_rebuild_lock:
            return dict(self._pending_agent_rebuild or {}) or None

    def drain_pending_agent_rebuild(self, *, sync: bool = False) -> str | None:
        """Do a queued rebuild now. Returns its reason, or None if none queued.

        Compare-and-clear: the pending slot is TAKEN before the rebuild runs, so
        a request arriving mid-rebuild is not swallowed by this drain -- it
        re-queues and the next drain picks it up.

        ``sync=True`` blocks until the graph is rebuilt. Used on the run-task
        entry path, where the invariant is stronger than "eventually": **no task
        may ever start on a stale tool table.**
        """
        with self._pending_rebuild_lock:
            pending = self._pending_agent_rebuild
            if not pending:
                return None
            self._pending_agent_rebuild = None
        reason = str(pending.get("reason") or "queued rebuild")
        if pending.get("overlay"):
            # 覆盖层排队时**注册表也没动过**（整件事都挂起了）。所以补做要先重算
            # 注册表，再重建图 —— 顺序反了，重建出来的还是旧技能。
            try:
                from mast.skills.overlay.loader import reload_skills
                rep = reload_skills(self._registry, reason=reason)
                logger.info("补做覆盖层重载（%s）：%s", reason, rep.describe())
            except Exception as exc:  # noqa: BLE001
                logger.error("补做覆盖层重载失败：%s", exc, exc_info=True)
        if getattr(self, "_orchestrator", None) is None:
            logger.info("drained queued rebuild (%s) -- no orchestrator yet; "
                        "it will be built fresh anyway", reason)
            return reason
        if sync:
            try:
                self._build_orchestrator()
                logger.info("Orchestrator rebuilt from queue, sync (%s)", reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("queued orchestrator rebuild failed (%s): %s",
                               reason, exc)
            return reason

        def _do():
            try:
                self._build_orchestrator()
                logger.info("Orchestrator rebuilt from queue (%s)", reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("queued orchestrator rebuild failed (%s): %s",
                               reason, exc)

        try:
            _threading_mod.Thread(target=_do, name="agent-tools-rebuild-drain",
                                  daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not schedule queued rebuild: %s", exc)
        return reason

    def _request_composite_rebuild(self) -> str:
        """Back-compat shim: composite hot-(un)register asks for a rebuild.

        Signature unchanged (returns only the UI suffix) -- composite_panel and
        reload_wiring's legacy branch both call it this way.
        """
        return self.request_agent_rebuild("composite store change")[1]


    def _build_orchestrator(self, *, agent_model_overrides: dict | None = None) -> bool:
        """Serialize concurrent rebuilds behind _orch_build_lock — the buffer-loop
        _on_started callback, the per-agent override offload thread, and the
        anyio.to_thread _set_model route can all call this off the main thread;
        without the lock they race (one nulls self._orchestrator while another
        is mid-build)."""
        with self._orch_build_lock:
            return self._build_orchestrator_impl(
                agent_model_overrides=agent_model_overrides)


    def _build_orchestrator_impl(self, *, agent_model_overrides: dict | None = None) -> bool:
        """Build (or rebuild) the 6-agent orchestrator graph.

        Extracted from setup() so POST /agents/<id>/model can rebuild the
        graph with new per-agent models WITHOUT a full process restart — the
        next task then runs on the chosen models. Returns True if a graph was
        built. Safe to call repeatedly.

        ``agent_model_overrides`` maps agent name → ChatModel instance (NOT a
        model-id string); when omitted, persisted ConfigOverrideRegistry model
        ids are resolved to chat models so a UI model change takes effect on
        the next build.
        """
        self._orchestrator = None
        try:
            from mast.agents.orchestrator import graph as _orch_graph
            from mast.core.execution_context import ExecutionContext
            if not all(c is not None for c in [self._pool, self._state, self._registry]):
                logger.info("Orchestrator skipped — pool/state/registry not all initialised")
                return False

            def _orch_context():
                # 修复项: share the orchestrator abort Event with the context so
                # a running composite's check_abort() actually fires on GUI
                # abort / E_STOP (it used to poll a private dead Event). The
                # abort gates in ExecutionContext.run / .safe_call and in
                # skill_adapter all read THIS event.
                #
                # run_id scopes composite step-progress sidecars to the current
                # run, so a finished run's progress can never be resumed by the
                # next one (2026-07-10 fake-进针 root cause). Read live — each
                # run-task sets it before streaming.
                _rid = getattr(self, "_orch_run_id", "") or ""
                # UNION: the global emergency latch AND this run's own stop.
                # Two concurrent group runs used to share the single
                # ``_orch_abort`` slot, so aborting one stopped both and
                # starting one re-armed both (审计).
                _aborts = [self._orch_abort]
                _run_ev = self.run_abort_event(_rid) if _rid else None
                if _run_ev is not None:
                    _aborts.append(_run_ev)
                ctx = ExecutionContext(
                    pool=self._pool, state=self._state, registry=self._registry,
                    abort_event=_aborts,
                    run_id=_rid,
                    owner=f"群聊任务 {_rid or '(未命名)'}",
                )
                # the NON-LATCHING tip-quality stop a composite polls
                # between steps. Attached rather than passed to __init__ so the
                # signal stays out of the abort machinery it must not be
                # confused with; graph_executor reads it duck-typed, exactly
                # like check_abort / get_progress / emit_progress.
                _attach_halt_check(ctx, self, _rid)
                _attach_marker_sink(ctx, self)
                return ctx

            overrides = dict(agent_model_overrides or {})
            # Resolve persisted UI model overrides → chat-model instances so a
            # model change in the Agents tab actually re-routes the next run.
            if not overrides:
                overrides = self._resolve_override_models()
            # Supervisor override is honoured separately (build() takes it as a
            # dedicated arg, not via agent_model_overrides) — previously the
            # supervisor picker was persisted+displayed but silently ignored.
            sup_model = self._resolve_supervisor_model()

            # Shared persistent-memory tools attached to every agent so the
            # 6 agents can read/write cross-session memory. The provider is
            # re-evaluated on every tool call, so the namespace follows the
            # current experiment without rebuilding the graph. Tools come from
            # the shared module (agents._shared) — no agent imports a sibling.
            memory_tools = None
            if getattr(self, "_cognition", None) is not None:
                from mast.agents._shared.memory_tools import make_memory_tools

                def _agent_mem_provider():
                    eid = (self._experiment_log.current_experiment_id
                           if self._experiment_log else None)
                    ns = f"experiment:{eid}" if eid else "global"
                    return {"store": self._cognition.store, "namespace": ns,
                            "experiment_id": eid, "author": "agent"}

                try:
                    memory_tools = make_memory_tools(_agent_mem_provider)
                except Exception:
                    memory_tools = None

                # Cognition tools (brainstorm + dream) attached to every agent so
                # the orchestrator can reach them by routing to a tool-running
                # agent (the supervisor is a pure router). Same provider pattern;
                # an LLM is built once (best-effort) so brainstorm runs深入 rather
                # than the offline rule-based fallback.
                try:
                    from mast.agents._shared.cognition_tools import make_cognition_tools
                    _cog_llm = None
                    try:
                        from mast.agents._shared.models import make_chat_model
                        _cog_llm = make_chat_model("orchestrator", max_tokens=4096)
                    except Exception:
                        _cog_llm = None

                    def _agent_cog_provider():
                        eid = (self._experiment_log.current_experiment_id
                               if self._experiment_log else None)
                        return {"db_path": self.config.db_path,
                                "store": self._cognition.store,
                                "experiment_id": eid, "llm": _cog_llm}

                    _cog_tools = make_cognition_tools(_agent_cog_provider)
                    memory_tools = (memory_tools or []) + _cog_tools
                except Exception as _cog_exc:
                    logger.debug("cognition tools not attached: %s", _cog_exc)

            # Agent→user request tool (post a request to the 心愿单 board, e.g.
            # literature: "请上传全文"; instrument: "请操作硬件"). ASYNC, non-blocking.
            try:
                from mast.agents._shared.request_tools import make_request_tools

                def _agent_req_provider():
                    eid = (self._experiment_log.current_experiment_id
                           if self._experiment_log else None)
                    return {"agent_id": "agent", "experiment_id": eid}

                memory_tools = (memory_tools or []) + make_request_tools(_agent_req_provider)
            except Exception as _req_exc:
                logger.debug("request tool not attached: %s", _req_exc)

            # Read-only document access for EVERY agent (2026-07-30). The artifact
            # channel hands each consumer a doc_id and tells it to read the body;
            # for one day it named a `load_document` tool that existed nowhere in
            # the tree. Reading a document is not a per-agent privilege — the
            # channel points all six of them at documents by design. Attached
            # here, on the one list every agent receives, rather than in six
            # build_tools().
            try:
                from mast.agents._shared.document_tools import DOCUMENT_TOOLS

                memory_tools = (memory_tools or []) + list(DOCUMENT_TOOLS)
            except Exception as _doc_exc:
                logger.debug("document tools not attached: %s", _doc_exc)

            # Environment survey for EVERY agent (2026-07-30). The artifact channel
            # PUSHES each agent the fields its CONSUMES row lists; nothing could ask
            # the reverse question — "what is in this experiment right now?" — which
            # is the one a scheduling decision needs. The data existed
            # (artifacts.list_existing / class_status) with the topology REST
            # endpoint as its only reader.
            try:
                from mast.agents._shared.environment_tools import ENVIRONMENT_TOOLS

                memory_tools = (memory_tools or []) + list(ENVIRONMENT_TOOLS)
            except Exception as _env_exc:
                logger.debug("environment tools not attached: %s", _env_exc)

            # Ask the OPERATOR a structured question and BLOCK on the answer
            # (2026-08-01). request_tools (above) posts to the 心愿单 board and
            # continues; this one pauses the graph the way a DANGEROUS approval
            # does, for the decisions that are the operator's to make rather than
            # the agent's. Attached on the same shared list, so all six agents
            # get it from this one place.
            try:
                from mast.agents._shared.ask_tools import ASK_USER_TOOLS

                memory_tools = (memory_tools or []) + list(ASK_USER_TOOLS)
            except Exception as _ask_exc:
                logger.debug("ask_user tool not attached: %s", _ask_exc)

            # Step 1 (HITL): the orchestrator MUST be compiled with a
            # checkpointer so LangGraph interrupt()/HumanInTheLoopMiddleware can
            # pause the graph at a DANGEROUS-skill approval and a later
            # Command(resume=…) can continue it. A process-level InMemorySaver is
            # sufficient and SAFE for the "checkpoint must not hold tensors"
            # invariant: the only thing skills write into checkpointed state are
            # Command(update) dicts of file paths + short strings (no tensor /
            # ndarray / socket / Nanonis client ever lands in state). enable_hitl
            # is passed explicitly so the human-approval gate on EmergencyRetract
            # / BiasPulse / TipShape / MotorMove is live on the GUI path.
            from langgraph.checkpoint.memory import InMemorySaver
            if getattr(self, "_orch_checkpointer", None) is None:
                # P3-D: durable checkpointer — a process restart no longer
                # loses every paused HITL thread (InMemorySaver had been the
                # silent default; PostgresSaver was documented but never
                # wired). Falls back to InMemorySaver on any failure — a broken
                # checkpoint DB must never block instrument operation.
                #
                # WAL + busy_timeout (2026-07-28). The old comment here claimed
                # "the GUI worker threads serialize through langgraph's own
                # locking"; nothing in the tree backs that up, and the two other
                # SQLite users in this repo (chat/store.py:146-152,
                # logging/storage.py:25-49) both set these pragmas for exactly
                # the writers this file has: a group run, a private chat and the
                # background-run manager all writing checkpoints. Without them a
                # concurrent writer fails fast with "database is locked" —
                # inside the checkpointer, i.e. mid-run.
                try:
                    import sqlite3

                    from langgraph.checkpoint.sqlite import SqliteSaver
                    from mast._runtime_paths import project_root
                    ckpt_dir = project_root() / "experiments"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(
                        str(ckpt_dir / "orchestrator_checkpoints.sqlite"),
                        check_same_thread=False,
                    )
                    _tune_checkpoint_conn(conn)
                    self._orch_checkpointer = SqliteSaver(conn)
                    logger.info("Orchestrator checkpointer: SqliteSaver "
                                "(experiments/orchestrator_checkpoints.sqlite)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SqliteSaver unavailable (%s) — falling "
                                   "back to InMemorySaver", exc)
                    self._orch_checkpointer = InMemorySaver()
            # Tip-shape post-skill hook (producer-side glue, NOT the LLM tool):
            # after a successful TipShapeWithReadback it renders the z/current
            # figure, registers it in v2 records, and emits a vision-buffer
            # verdict. Lazy getters so it picks up the async buffer + v2 ids.
            try:
                from pathlib import Path as _P

                from mast.webui.tip_shape_records import make_tip_shape_post_hook
                _tip_hook = make_tip_shape_post_hook(
                    repos=getattr(self, "_v2_repos", None),
                    experiment_id_getter=lambda: getattr(self, "_v2_eid", None),
                    buffer_getter=lambda: getattr(self, "_buffer", None),
                    artifacts_dir=str(_P(self.config.db_path).parent / "tip_shapes"))
            except Exception as exc:
                logger.warning("tip_shape hook unavailable: %s", exc)
                _tip_hook = None
            # Thread the live-state + safety knobs into the orchestrator so the
            # instrument_control agent's SafetyGateMiddleware actually enforces
            # (a) state preconditions against the LIVE hardware (e.g.
            # z_controller_off → tip must be withdrawn before a coarse approach,
            # scan_not_running) and (b) admin-tightened SafetyLimits overrides —
            # both of which were silently inert on the multi-agent path because
            # neither get_state nor the admin override registry reached build_ic.
            # state.snapshot() returns the cached HardwareState instantly (the
            # 1 s background refresh keeps it live) so this stays non-blocking.
            _orch_get_state = self._state.snapshot if self._state is not None else None
            _orch_safety_limits = getattr(self.config, "safety", None)
            try:
                from mast.admin.override_store import ConfigOverrideRegistry
                _orch_override_registry = ConfigOverrideRegistry.get()
            except Exception as exc:
                logger.debug("override registry unavailable for orchestrator: %s", exc)
                _orch_override_registry = None
            # IC meta-tools (experiment/sample session mgmt + scan/knowledge/nav/
            # plan) so the 群聊 instrument_control has parity with its private chat
            # — lets it actually create/rename a MAST experiment/sample instead of
            # misreading the request as a Nanonis field.
            try:
                from mast.agents._shared.meta_tools import make_meta_tools
                _ic_meta_tools = make_meta_tools(self._meta_tool_context)
            except Exception as exc:  # noqa: BLE001
                logger.debug("orchestrator IC meta tools unavailable: %s", exc)
                _ic_meta_tools = None
            # experiment_design gets its own slice of the same meta-tool set:
            # experiment/sample lifecycle (so "新建实验/新建样品" routed to XD can
            # actually create the record instead of looping on its introspection
            # tools until the recursion cap — 2026-06-30), the ability to PERSIST
            # the plan it just wrote, read-only tip registry, and the knowledge
            # tools it needs to have an opinion at all.
            #
            # 名单本身是 meta_tools.DESIGN_TOOL_NAMES —— 每个名字进出的理由都写在
            # 那个常量上,而 artifacts.py 的数据流图派生的是**同一个常量**。这里
            # 不再重列名字:重列过一次,两份就分了叉(runtime 给了 create_plan,
            # 图上却没有 XD → experiment_plan 这条写边)。
            _xd_meta_tools = None
            if _ic_meta_tools:
                from mast.agents._shared.meta_tools import DESIGN_TOOL_NAMES
                _xd_names = set(DESIGN_TOOL_NAMES)
                _xd_meta_tools = [t for t in _ic_meta_tools
                                  if getattr(t, "name", "") in _xd_names]
            self._orchestrator = _orch_graph.build(
                buf=self._buffer,
                supervisor_model=sup_model,   # None → build() uses its default
                context_provider=_orch_context,
                control_provider=self._orch_control_provider,
                agent_model_overrides=overrides or None,
                # include_agents OMITTED on purpose (2026-08-21). ``build()``
                # already defaults to the full ``_AGENT_NAMES`` roster, and the
                # explicit copy that used to sit here was a second roster that
                # drifted the moment a seventh agent was added: the supervisor
                # validates every routing decision against THIS set, so an agent
                # missing from it is silently undispatchable — the router names
                # it, ``_coerce_targets`` drops it, and the run ends with a
                # perfectly ordinary "目标已完成". Nothing logs, nothing raises.
                # Spelling it out bought nothing (it was byte-identical to the
                # default) and cost exactly that.
                memory_tools=memory_tools,
                # 群聊 IC 用**活的**注册表(私聊早就传了)。不传的话 build_ic 回落
                # discover(),运行期注册的 spec / custom / overlay / agent 铸的技能
                # 在群聊里全部消失 —— 静默、无报错(2026-08-25)。
                instrument_registry=self._registry,
                instrument_extra_tools=_ic_meta_tools,
                experiment_design_extra_tools=_xd_meta_tools,
                checkpointer=self._orch_checkpointer,
                enable_hitl=True,
                instrument_post_hook=_tip_hook,
                get_state=_orch_get_state,
                get_mode=self._current_operating_mode,
                safety_limits=_orch_safety_limits,
                override_registry=_orch_override_registry,
                recorder=self._skill_trace_recorder,
                safety_recorder=self._safety_trace_recorder,
                turn_recorder=self._turn_trace_recorder,
                # Shared context-compaction + memory-recall middleware so 群聊
                # gets the same modern context management as the private chat.
                agent_extra_middleware=self._group_agent_middleware(),
                agent_call_limits=self._chat_call_limits(),
                # Conservative auto-background gate (default OFF): when ON, the
                # supervisor peels an independent literature survey off a fan-out
                # that also has instrument_control, so the instrument foreground
                # isn't barrier-blocked (the run-task bridge spawns the detached
                # run). Read live from settings so a flip needs no restart.
                background_gate=self._auto_background_enabled,
                # item ②: duration advisor — keeps a proven-FAST whitelisted agent
                # foreground (not worth the detach); slow/unknown → whitelist policy.
                is_slow_type=self._bg_is_slow_type,
                # Connects the supervisor's USD hard gate to the billing ledger.
                # The graph is built ONCE while the budget is per-RUN, so this is a
                # stable indirection that reads whichever meter is current — see
                # begin_run_budget().
                budget_probe=self._budget_remaining_usd,
                # Environment-driven activation gating (park an agent whose inputs
                # do not exist yet). DEFAULT OFF — see the gate's docstring.
                activation_gate=self._activation_gating_enabled,
            )
            logger.info("Orchestrator built (6 agents wired%s%s, supervisor routes; "
                        "interject+model-override live)",
                        f", {len(overrides)} agent override(s)" if overrides else "",
                        ", supervisor override" if sup_model is not None else "")
            return True
        except Exception as exc:
            logger.warning("Orchestrator build failed: %s", exc)
            self._orchestrator = None
            return False


    def _ensure_chat_checkpointer(self):
        """Reuse the orchestrator's SqliteSaver, or build one for the engine."""
        if getattr(self, "_orch_checkpointer", None) is not None:
            return self._orch_checkpointer
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
            from mast._runtime_paths import project_root
            ckpt_dir = project_root() / "experiments"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(ckpt_dir / "orchestrator_checkpoints.sqlite"),
                                   check_same_thread=False)
            _tune_checkpoint_conn(conn)
            self._orch_checkpointer = SqliteSaver(conn)
        except Exception as exc:  # noqa: BLE001
            from langgraph.checkpoint.memory import InMemorySaver
            logger.warning("chat checkpointer: SqliteSaver unavailable (%s) → InMemorySaver", exc)
            self._orch_checkpointer = InMemorySaver()
        return self._orch_checkpointer


    def _chat_agent_extra_tools(self, agent_id: str = "instrument_control") -> list:
        """Private-chat extras: memory + documents + environment + ask-the-operator
        for every agent, plus the instrument meta-tools for IC only."""
        tools: list = []
        cog = getattr(self, "_cognition", None)
        if cog is not None:
            try:
                # cog.memory_provider supplies writer/recaller → writes index +
                # search is semantic. experiment_id re-read per call by closure.
                from mast.agents._shared.memory_tools import make_memory_tools

                def _mem_provider():
                    return cog.memory_provider(
                        experiment_id=self._chat_experiment_id(), author="user")()

                tools += make_memory_tools(_mem_provider)
            except Exception as exc:  # noqa: BLE001
                logger.debug("chat memory tools unavailable: %s", exc)
        # Read-only document access, same as the group path gets: a private-chat
        # agent is shown the same artifact block (including its OWN last product's
        # doc_id, which is how a revision continues one document instead of
        # forking a second), so it needs the same reader.
        try:
            from mast.agents._shared.document_tools import DOCUMENT_TOOLS
            tools += list(DOCUMENT_TOOLS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat document tools unavailable: %s", exc)
        # conduct 工具:与群聊那条线同一族(orchestrator/graph.py 的 `_shared`)。
        # 两条入口给的工具面必须一样 —— 否则「能不能看多天实验跑到哪了」会取决于
        # 用户当时点开的是哪个页面,而本仓刚在 ask-the-operator 上栽过这个形状
        # (见下面那段注释:同一个能力群聊有、私聊没有,于是 agent 收得到回复却
        # 问不出问题)。把关在服务端,不在这里。
        try:
            from mast.agents._shared.conduct_tools import make_conduct_tools
            tools += make_conduct_tools(agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat conduct tools unavailable: %s", exc)
        # 技能市场工具:同上,两条入口必须给同一族。这里尤其要紧 —— 私聊是用户
        # 最常问「你能不能做 X」的地方,而 agent 要能回答「本机有这个技能,但它不在
        # 我手上,要不要装」就得先搜得到市场全集。它只能**推荐**,改订阅是人面上的
        # 动作(见 agents/_shared/market_tools.py)。
        try:
            from mast.agents._shared.market_tools import make_market_tools
            tools += make_market_tools(agent_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat market tools unavailable: %s", exc)
        # Environment survey, same as the group path: "what is already in this
        # experiment" is if anything MORE useful in 私聊, where there is no
        # supervisor to have looked first.
        try:
            from mast.agents._shared.environment_tools import ENVIRONMENT_TOOLS
            tools += list(ENVIRONMENT_TOOLS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat environment tools unavailable: %s", exc)
        # Ask-the-operator, same as the group path. This was group-only, which
        # made an agent's ability to ask for anything depend on WHICH ENTRY POINT
        # the operator happened to use: the middleware that reads answers back
        # runs on both paths, but the tools that create the request did not, so a
        # private chat could receive replies to questions it had no way to ask.
        # An instrument private chat needing "please change the sample" had to
        # say it in prose and hope somebody was reading.
        try:
            from mast.agents._shared.request_tools import make_request_tools

            def _req_provider(_aid=agent_id):
                return {"agent_id": _aid,
                        "experiment_id": self._chat_experiment_id()}

            tools += make_request_tools(_req_provider)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat request tools unavailable: %s", exc)
        # Ask the operator a question and BLOCK on the answer, same as the group
        # path. Only safe to attach here because routes/chat_stream now passes a
        # hitl_resolver — without one the interrupt would park the graph in its
        # checkpoint with no way to answer it.
        try:
            from mast.agents._shared.ask_tools import ASK_USER_TOOLS
            tools += list(ASK_USER_TOOLS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat ask_user tool unavailable: %s", exc)
        # instrument meta-tools (experiment/scan/knowledge/navigation/plan) are the
        # IC parity surface — only attach to the instrument_control private chat.
        if agent_id != "instrument_control":
            return tools
        try:
            from mast.agents._shared.meta_tools import make_meta_tools
            tools += make_meta_tools(self._meta_tool_context)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat meta tools unavailable: %s", exc)
        return tools

    def _resolve_session_dir(self) -> str | None:
        """Authoritative Nanonis .sxm output directory from Util_SessionPathGet.

        Cached for about 15 seconds. Return None without hardware so consumers
        can fall back to working-session directories. Normalize a file prefix
        to a directory because candidate discovery accepts directories only.
        """
        import time as _time
        now = _time.monotonic()
        cached = getattr(self, "_session_dir_cache", None)
        if cached is not None and (now - cached[1]) < 15.0:
            return cached[0]

        session_dir: str | None = None
        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                rec = pool.safe_call("Util_SessionPathGet", role="main")
                if rec is not None and not getattr(rec, "error", None):
                    parsed = getattr(rec, "return_value", None)
                    sp = ""
                    if isinstance(parsed, str):
                        sp = parsed
                    elif isinstance(parsed, (list, tuple)) and len(parsed) > 2:
                        vals = parsed[2]
                        if isinstance(vals, (list, tuple)):
                            for item in reversed(vals):  # path follows its size int
                                if isinstance(item, str):
                                    sp = item
                                    break
                        elif isinstance(vals, str):
                            sp = vals
                    sp = (sp or "").strip()
                    if sp:
                        from pathlib import Path as _P
                        p = _P(sp)
                        if p.is_dir():
                            session_dir = str(p)
                        elif p.parent and str(p.parent) not in (".", "") and p.parent.is_dir():
                            session_dir = str(p.parent)  # file-prefix → its folder
                        else:
                            session_dir = str(p)  # best-effort (may not exist yet)
            except Exception as exc:  # noqa: BLE001 — degrade, never break the caller
                logger.debug("session-path resolve failed: %s", exc)

        self._session_dir_cache = (session_dir, now)
        # Mirror onto the plain attrs the meta-tool context + records route read.
        self._session_path = session_dir
        self._session_dir = session_dir
        # Publish to the process-level registry so CONTEXT-LESS file searches
        # (data_processing's get_latest_scan_file runs with no pool) also see
        # the real Nanonis save dir.
        if session_dir:
            try:
                from mast.core.scan_registry import record_session_dir
                record_session_dir(session_dir)
            except Exception:  # noqa: BLE001 — registry is best-effort
                pass
        return session_dir

    def _meta_tool_context(self) -> dict:
        """Provider dict for make_meta_tools — the instrument meta-tool surface
        (experiment/sample SESSION management, scan/knowledge/navigation/plan).
        Shared by BOTH the IC private chat AND the 群聊 orchestrator's IC node so
        they are at parity (the group IC used to lack these, so it misread
        '新建实验/样品' as a Nanonis field instead of MAST's records session)."""
        # Best-effort refresh of the Nanonis save dir (cached ~15 s) so scan
        # getters resolve real .sxm; None when no hardware (callers degrade).
        try:
            _sess = self._resolve_session_dir()
        except Exception:  # noqa: BLE001
            _sess = None
        return {
            "experiment_log": getattr(self, "_experiment_log", None),
            "plan_store": getattr(self, "_plan_store", None),
            # Scan-map markers ARE the surface record: the map tools read them
            # and compute coverage / keep-out zones / the next position from
            # them, instead of from a parallel in-memory grid the agent had to
            # remember to update (2026-07-30).
            "storage": getattr(self, "_storage", None),
            "map_analysis_cfg": self.build_map_analysis_config,
            "coarse_map_cfg": self.build_coarse_map_config,
            "experiments_dir": str(getattr(self.config, "experiments_dir", "") or ""),
            "session_path": _sess,
            "session_dir": _sess,
            # True-parallel offload: lets the FOREGROUND agent detach a long/
            # independent sub-task (a literature survey) to a background run so it
            # can keep working the instrument without the super-step barrier
            # serialising them. Results merge back into the live group conversation.
            "spawn_background": self._tool_spawn_background,
        }

    def _tool_spawn_background(self, instruction: str,
                              agents: "tuple[str, ...] | list[str]" = ("literature",),
                              priority: str = "normal",
                              on_done: "dict | None" = None) -> dict:
        """Meta-tool backend: start a background run merged into the LIVE group
        conversation (if one is streaming). ``priority`` ('high'|'normal') sets the
        admission-queue priority; ``on_done`` is an EXPLICIT, caller-declared
        follow-up ({"next_agent", "note"}) surfaced to the foreground supervisor
        when the run completes (never auto-executed). Returns a small JSON-able
        dict — never raises (the tool surfaces the error text)."""
        mgr = self._ensure_background_manager()
        if mgr is None:
            return {"ok": False, "error": "background runs unavailable"}
        cid = ""
        st = getattr(self, "_agents_api_state", None)
        if isinstance(st, dict) and isinstance(st.get("task"), dict):
            t = st["task"]
            if t.get("active"):
                cid = str(t.get("conversation_id") or "")
        try:
            rec = mgr.spawn(instruction=instruction, conversation_id=cid,
                            agents=tuple(agents), priority=priority, on_done=on_done)
            return {"ok": True, "run_id": rec["run_id"], "agents": rec["agents"]}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool spawn_background failed: %s", exc)
            return {"ok": False, "error": f"{type(exc).__name__}"}


    def _chat_agent_middleware(self, agent_id: str | None = None) -> list:
        """Compaction + tool-refine + memory-recall middleware for chat / group.

        表本身在 :mod:`mast.agents._shared.shared_stack`（纯函数，2026-08-24 抽
        出去的）；这里只负责把 Runtime 才拿得到的东西喂进去：生效模型 id、
        summarizer、cognition、设置。

        抽出去的理由写在那个模块的 docstring 里，一句话版本：方法体里
        ``make_chat_model`` 没 key 就抛、被 except 吞掉，于是**单测里拿到的永远
        是缺了 compaction 与 tool_refine 的残表**，而断言照样绿 —— 闸门因此钉不
        住真实的挂载顺序。

        Compaction sizing, summariser, and per-turn tool refinement all follow the
        agent's ACTUAL model — ``resolve_effective_model_id``, which reads the
        persisted per-agent override, NOT ``get_model_id``, which only knows the
        code default. That distinction is the whole point: overriding an agent
        onto a 120k-window model used to leave compaction sized for a 250k window
        (trigger 175 500 tokens), so it could only fire after the provider had
        already 400'd the request. compaction_model / tool_refine_* settings
        override further.

        ``agent_id=None`` sizes by the orchestrator model. Group builds pass a
        REAL agent id per agent (see ``_group_agent_middleware``) — one shared
        instance sized for the orchestrator was wrong for the other five.
        """
        from mast.agents._shared.shared_stack import build_shared_middleware

        st = getattr(self, "_settings", None)
        cog = getattr(self, "_cognition", None)
        who = agent_id or "orchestrator"

        model_id = who
        summ = None
        try:
            from mast.agents._shared.models import (
                make_chat_model,
                resolve_effective_model_id,
            )
            model_id = resolve_effective_model_id(who)
            summ_alias = ((st.get("compaction_model") if st is not None else None) or "").strip()
            summ = (make_chat_model(model_id=summ_alias, max_tokens=4096) if summ_alias
                    else make_chat_model(who, max_tokens=4096))
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat summarizer model unavailable: %s", exc)

        def _sink(summary: str):
            if cog is None:
                return
            eid = self._chat_experiment_id()
            ns = f"experiment:{eid}" if eid else "global"
            cog.remember(ns, "summaries/chat-running.md", summary,
                         title="对话压缩摘要", kind="summary", experiment_id=eid)

        return build_shared_middleware(
            agent_id,
            model_id=model_id,
            summarizer=summ,
            cognition=cog,
            namespace_provider=self._chat_experiment_id,
            memory_sink=(_sink if cog is not None else None),
            settings=st,
        )

    def _group_agent_middleware(self):
        """A per-agent middleware factory for ``orchestrator.build()``.

        Returns a callable, not a list: each of the six agents needs its OWN
        compaction instance sized for the model IT runs (see
        ``_chat_agent_middleware``). Sharing one instance also meant sharing its
        per-turn caches across agents that run CONCURRENTLY under a fan-out.
        Results are memoised per build so a rebuild does not pay six times.
        """
        cache: dict[str, list] = {}

        def _for(agent_id: str) -> list:
            if agent_id not in cache:
                cache[agent_id] = self._chat_agent_middleware(agent_id)
            return cache[agent_id]

        return _for


    def _chat_call_limits(self) -> dict:
        # Per-agent call-limit kwargs for the DURABLE chat / group paths.
        # Reads the persisted SettingsStore live (same pattern as
        # _current_operating_mode). Cumulative THREAD caps default OFF so a
        # long-lived conversation is never bricked by a running total that
        # never resets (2026-07-07 fix); per-run caps stay as the in-turn
        # spin guard. Operators retune via the chat_*_calls_per_* keys.
        st = getattr(self, "_settings", None)

        from mast.agents._shared.call_limits import (
            DEFAULT_MODEL_CALLS_PER_RUN, DEFAULT_TOOL_CALLS_PER_RUN,
        )

        def _thread_cap(key: str) -> int:
            """累计上限。这里 ``0`` 一直就是「关掉」,因为出厂值本身就是 0。"""
            v = st.get(key) if st is not None else None
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        def _run_cap(key: str, default: int):
            """每轮上限原样交给 call_limits._norm_run。
            
            0 表示不限，其含义在统一入口处理并说明；不能在此处替换成另一个默认上限。"""
            v = st.get(key) if st is not None else None
            return default if v is None else v

        mt = _thread_cap("chat_model_calls_per_thread")   # 0 -> OFF
        tt = _thread_cap("chat_tool_calls_per_thread")    # 0 -> OFF
        return {
            "max_model_calls": mt if mt > 0 else None,
            "max_tool_calls": tt if tt > 0 else None,
            # 出厂值从 call_limits 派生 —— 这里曾经写着字面量 30,和那边的常量
            # 是两份副本。
            "max_model_calls_per_run": _run_cap(
                "chat_model_calls_per_run", DEFAULT_MODEL_CALLS_PER_RUN),
            "max_tool_calls_per_run": _run_cap(
                "chat_tool_calls_per_run", DEFAULT_TOOL_CALLS_PER_RUN),
        }

    def _chat_context_provider(self):
        """私聊回合的 ExecutionContext 工厂（两条引擎路径共用）。

        2026-08-27 从 ``_build_chat_agent_graph`` 里的 ``_ctx`` 闭包抽出来。
        抽的理由不是整洁，是它带着两条**不能有第二份**的逻辑：停止源的并集
        （E_STOP 必须到得了硬件闸门）与 run_id 的归属（防 2026-07-10「假进针」）。
        """
        # UNION of stop sources, not "session OR orchestrator" (P0,
        # 2026-07-11). The old `abort or self._orch_abort` let the chat
        # session's event WIN, so a composite running in the private chat
        # polled only that event — and E_STOP, which sets `_orch_abort`,
        # was invisible to it. The emergency stop literally could not stop
        # a composite running under the main chat. Passing both means a
        # context aborts when ANY of its stop sources fires: E-STOP always
        # reaches the hardware gates, while the chat's own Stop still stops
        # only that chat.
        eng = getattr(self, "_conv_engine", None)
        session_abort = eng.active_abort_event() if eng is not None else None
        aborts = [e for e in (session_abort, self._orch_abort) if e is not None]
        # OWN run id, not the orchestrator's (2026-07-28). This used to
        # read `self._orch_run_id`, which is written when a group run
        # starts and cleared nowhere — so a private-chat composite ran
        # under a group run's identity and the two shared one
        # step-progress sidecar. The resume guard rejects only TERMINAL
        # and STALE(30 min) progress; two runs in flight are neither, so
        # the later one resumed the earlier one's progress and skipped
        # steps it had never executed — the 2026-07-10「假进针」failure
        # mode, reopened across chains. See engine.active_run_id().
        _rid = ""
        if eng is not None:
            try:
                _rid = eng.active_run_id()
            except Exception:  # noqa: BLE001 — never break a chat turn
                _rid = ""
        if not _rid:
            # No live chat turn (a tool called outside stream_turn, or an
            # engine too old to carry one): a FRESH id is still correct —
            # an empty run_id would collide with every other empty one.
            import uuid as _uuid_mod
            _rid = f"chat-adhoc-{_uuid_mod.uuid4().hex[:8]}"
        from mast.core.execution_context import ExecutionContext

        ctx = ExecutionContext(
            pool=self._pool, state=self._state, registry=self._registry,
            abort_event=aborts,
            run_id=_rid,
            owner="主聊天/私聊")
        # same non-latching tip-quality stop as the group-chat
        # path — a composite run from the private chat is just as
        # capable of grinding through 5 regions on a broken tip.
        _attach_halt_check(ctx, self, _rid)
        _attach_marker_sink(ctx, self)
        return ctx

    def _build_chat_agent_graph(self, agent_id: str):
        """Factory for the ConversationEngine: a STANDALONE compiled agent graph.

        ``instrument_control`` (the main chat) gets the full instrument stack
        (live state / safety / meta-tools); the other 5 agents get a lighter
        standalone build. ALL get the shared checkpointer + compaction + memory
        recall + memory tools, with handoff tools suppressed (no parent graph).
        """
        from mast.core.execution_context import ExecutionContext
        ck = self._ensure_chat_checkpointer()
        extra_mw = self._chat_agent_middleware(agent_id)
        extra_tools = self._chat_agent_extra_tools(agent_id)

        if agent_id == "instrument_control":
            from mast.agents.instrument_control import graph as ic_graph

            _ctx = self._chat_context_provider

            return ic_graph.build(
                buf=getattr(self, "_buffer", None),
                context_provider=_ctx,
                registry=self._registry,
                checkpointer=ck,
                get_state=(self._state.snapshot if getattr(self, "_state", None) else None),
                get_mode=self._current_operating_mode,
                # 传入合并后的有效安全限值，使执行闸门与工具 schema 使用同一范围。
                # 不能只给私聊执行路径传出厂配置，而向模型公布用户覆盖后的配置。
                safety_limits=_chat_effective_safety_limits(
                    getattr(self.config, "safety", None)),
                recorder=getattr(self, "_skill_trace_recorder", None),
                safety_recorder=getattr(self, "_safety_trace_recorder", None),
                turn_recorder=getattr(self, "_turn_trace_recorder", None),
                enable_hitl=True,
                extra_tools=extra_tools,
                standalone=True,
                extra_middleware=extra_mw,
                **self._chat_call_limits(),
            )

        mod_path = self._CHAT_AGENT_MODULES.get(agent_id)
        if mod_path is None:
            raise ValueError(f"unknown agent for private chat: {agent_id}")
        import importlib
        mod = importlib.import_module(mod_path)
        return mod.build(
            getattr(self, "_buffer", None),
            checkpointer=ck,
            extra_tools=extra_tools,
            turn_recorder=getattr(self, "_turn_trace_recorder", None),
            standalone=True,
            extra_middleware=extra_mw,
            **self._chat_call_limits(),
        )


    def _build_chat_engine(self) -> bool:
        """Build the ConversationStore + ConversationEngine and bootstrap a default
        private conversation. Replaces the legacy MissionPlanner on the chat path."""
        try:
            from mast.chat.engine import ConversationEngine
            from mast.chat.store import ConversationStore
            if getattr(self, "_storage", None) is None:
                logger.info("Chat engine skipped — storage not ready")
                return False
            self._conv_store = ConversationStore.from_storage(self._storage)
            def _conv_record_sink(agent_id: str, user_text: str, assistant_text: str) -> None:
                """Mirror a finished chat turn into the experiment record so the
                conversation is part of that record (审查 — this was
                never wired, so chat never reached the records)."""
                storage = getattr(self, "_storage", None)
                if storage is None or not hasattr(storage, "log_conversation"):
                    return
                el = getattr(self, "_experiment_log", None)
                eid = getattr(el, "current_experiment_id", None) if el else None
                sid = getattr(el, "current_sample_id", None) if el else None
                if user_text:
                    storage.log_conversation("user", user_text, experiment_id=eid,
                                             sample_id=sid, agent=agent_id)
                if assistant_text:
                    storage.log_conversation("assistant", assistant_text,
                                             experiment_id=eid, sample_id=sid,
                                             agent=agent_id)

            # 结构化正文的第二个家 + 一次性历史导入（退出 LangGraph 的 strangler
            # 第 0 步）。**顺序是硬约束**：先导入、后双写。ConversationStore 的 seq
            # 是单调追加的，反过来会把数月的老历史排到今天的新消息后面。
            #
            # 导入本身幂等（per-thread 账本），失败不抛 —— 读路径此刻仍然是
            # checkpointer，新家少一段历史事后补得回来，而挡住服务启动不行。
            self._message_store = self._ensure_message_store()
            self._import_chat_history_once()

            if self._v2_private_chat_enabled() and self._message_store is not None:

                # 私聊引擎在启动时选择，避免两个引擎交替写同一段历史；改变选择需要重启。
                from mast.chat.engine_v2 import ConversationEngineV2

                self._conv_engine = ConversationEngineV2(
                    store=self._conv_store, message_store=self._message_store,
                    ic_loop_factory=self._build_instrument_loop_v2,
                    record_sink=_conv_record_sink)
                self._chat_abort_events = {}
                logger.info("Chat engine: v2 (agentruntime.AgentLoop) — "
                            "instrument_control 私聊在这条路上会被拒绝")
                self._bootstrap_active_conversation()
                self._register_fetch_resumer()
                return True

            self._conv_engine = ConversationEngine(
                graph_factory=self._build_chat_agent_graph,
                checkpointer=self._ensure_chat_checkpointer(),
                store=self._conv_store,
                message_store=self._message_store,
                # The engine sizes each agent's recursion_limit from the graph it
                # actually built, using the SAME model-call run cap the graph was
                # built with — so ModelCallLimitMiddleware ends a runaway turn
                # readably instead of GraphRecursionError ending it first.
                # (2026-08-04: the old literal 50 bought 3 tool calls, see
                # ConversationEngine._recursion_limit_for.)
                call_limits_provider=self._chat_call_limits,
                record_sink=_conv_record_sink)
            self._chat_abort_events = {}
            self._bootstrap_active_conversation()
            self._register_fetch_resumer()
            logger.info("Chat engine ready (active conversation %s)", self._active_conv_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat engine build failed: %s", exc)
            self._conv_engine = None
            self._conv_store = None
            return False

    def _bootstrap_active_conversation(self) -> None:
        """恢复（或新建）当前活动的私聊会话。

        两条引擎路径共用这一份：抄第二遍的话，两条路会在「新建会话用哪个 agent /
        带不带 experiment_id」上慢慢分叉，而那种分叉的症状是「换了个引擎之后
        新对话就没有实验归属了」——没人会想到去那里找。
        """
        existing = self._conv_store.list(kind="private", limit=1)
        if existing:
            self._active_conv_id = existing[0]["conversation_id"]
            return
        conv = self._conv_store.create(
            "instrument_control", kind="private", title="新对话",
            # provenance tag (current experiment, if any); the chat is never
            # scoped/filtered by it, so it still spans experiments.
            experiment_id=getattr(getattr(self, "_experiment_log", None),
                                  "current_experiment_id", None))
        self._active_conv_id = conv["conversation_id"]

    def _register_fetch_resumer(self) -> None:
        """把「取文到货时续跑」的入口登记上。

        在这里登记而不是 import 时：建图失败就不该留下 resumer，那样
        ``notify_fulfilled`` 会退化成「关掉板子、不叫醒任何人」——正是这个功能
        存在之前的行为。
        """
        try:
            from mast.core import fetch_resume
            fetch_resume.set_resumer(self._resume_after_fetch_fulfilled)
        except Exception as exc:  # noqa: BLE001 — never block the chat engine
            logger.info("fetch-resume registration skipped: %s", exc)

    # ── 对话正文的新家（退出 LangGraph 的 strangler 第 0 步） ─────────────

    def _ensure_message_store(self):
        """``chat_messages`` 表的句柄，建在 ConversationStore 的同一个 DB 文件里。

        建不起来返回 None —— 双写整个跳过，聊天照常工作（读路径此刻仍在
        checkpointer 上）。这不是可有可无的容错：一个还没有任何消费者的写路径，
        没有资格让服务起不来。
        """
        # 刻意**不缓存**：这个方法只在 ``_build_chat_engine`` 里、``_conv_store``
        # 刚建好之后调用一次。重建接线（reload_wiring）会换一个新的
        # ConversationStore，缓存住的旧句柄会指向上一个 DB 文件 —— 那是「陈旧默认
        # 值合理得让人看不出来」的经典形状。
        try:
            from mast.agentruntime.persist import MessageStore
            return MessageStore.from_conversation_store(self._conv_store)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_messages store unavailable: %s", exc)
            return None

    def _import_chat_history_once(self) -> None:
        """把 checkpoint 里的历史搬进 ``chat_messages``（幂等，每次启动都调）。

        ⚠️ **必须在双写开始之前跑**：ConversationStore 的 seq 单调追加，先双写再
        回填会把数月的老历史排到今天的新消息后面。

        账本（``chat_message_migration``）记录 per-thread 状态，既是幂等判据，也是
        「什么时候允许从 requirements 里删掉 langgraph」的客观依据 —— 反序列化那份
        数据需要它的序列化器在场，卸载必须晚于全部导入完成。
        """
        if getattr(self, "_message_store", None) is None:
            return
        try:
            from mast._runtime_paths import project_root
            from mast.agentruntime.migrate import import_all

            ckpt = project_root() / "experiments" / "orchestrator_checkpoints.sqlite"
            summary = import_all(checkpoint_db=ckpt,
                                 conversation_store=self._conv_store,
                                 message_store=self._message_store)
            if summary.get("done") or summary.get("failed"):
                logger.info("chat history import: %s", summary)
        except Exception as exc:  # noqa: BLE001 — 导入绝不许挡住服务启动
            logger.warning("chat history import skipped: %s", exc)

    # ── resuming a literature run once its paper arrives ──────────────────

    def _resume_after_fetch_fulfilled(
        self, work_id: str, requests: list, *, kind: str = "fetch",
        exclude_conversation_id: str = "",
    ) -> dict:
        """Carry on the conversations that were waiting on an operator.

        Called (via ``core.fetch_resume``) when either board gets an answer:

        * ``kind="fetch"`` — an upload or an agent fetch closed取文板 rows;
          ``requests`` is the pre-resolve snapshot, so each row still knows which
          conversation asked and why.
        * ``kind="request"`` — the operator answered 心愿单 rows (with a path, a
          note, or a dismissal).

        Both shapes carry ``origin_conversation_id``; only the instruction text
        differs, so the dispatch below is shared.

        Two shapes, because a private chat and a group run are driven differently:

        * **private** — the conversation's thread has exactly one driver, the
          ConversationEngine, and its history lives in a durable checkpointer. So
          we drive one more turn on that same thread: the agent picks up with its
          own context intact, even days later.
        * **group** — the group's thread belongs to the run-task state machine.
          Driving it from the side would race that machine for the checkpoint and
          bypass the operator's controls, so the follow-up goes through
          BackgroundRunManager instead and its transcript lands back in the group.

        Never raises; every skip is recorded so the caller can say what happened.
        """
        out: dict = {"resumed": 0, "skipped": []}

        def _skip(cid: str, why: str) -> None:
            out["skipped"].append({"conversation_id": cid, "reason": why})

        try:
            if not self._fetch_auto_resume_enabled():
                return {"resumed": 0, "skipped": [{"reason": "disabled by setting"}]}

            # One conversation asking for three papers gets ONE "carry on", not three.
            groups: dict[str, list] = {}
            for rec in requests or []:
                cid = str((rec or {}).get("origin_conversation_id") or "").strip()
                if not cid or cid == (exclude_conversation_id or ""):
                    continue
                groups.setdefault(cid, []).append(rec)
            if not groups:
                return {"resumed": 0, "skipped": [{"reason": "no resumable origin"}]}

            from mast.core import fetch_resume

            store = getattr(self, "_conv_store", None)
            for cid, recs in groups.items():
                try:
                    conv = store.get(cid) if store is not None else None
                    if not conv:
                        _skip(cid, "conversation gone")
                        continue
                    text = (fetch_resume.build_answer_instruction(recs)
                            if kind == "request"
                            else fetch_resume.build_resume_instruction(recs))
                    conv_kind = str(conv.get("kind") or "private")
                    if conv_kind == "group":
                        agents = self._resume_agents_for(kind, recs)
                        done = self._resume_group(cid, text, agents)
                        if not done:
                            _skip(cid, "background manager unavailable")
                    else:
                        done, why = self._resume_private(conv, text, kind=kind)
                        if not done:
                            _skip(cid, why)
                    if done:
                        out["resumed"] += 1
                        if kind == "fetch":
                            # Told. Anything NOT told here is swept up later by
                            # the readback middleware, which is how a request
                            # with no conversation to return to still arrives.
                            self._mark_fetch_announced(recs)
                except Exception as exc:  # noqa: BLE001 — one conversation's problem
                    logger.info("fetch-resume for %s failed: %s", cid, exc)
                    _skip(cid, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — never propagate into an ingest
            logger.warning("fetch-resume dispatch failed (%s): %s", work_id, exc)
        return out

    @staticmethod
    def _mark_fetch_announced(records: list) -> None:
        """Record that these arrivals have been handed to the agent."""
        try:
            from mast.knowledge import fetch_board as board_mod
            ids = [str((r or {}).get("request_id") or "") for r in (records or [])]
            board_mod.mark_announced([i for i in ids if i])
        except Exception as exc:  # noqa: BLE001 — bookkeeping is never fatal
            logger.debug("mark_announced failed: %s", exc)

    def _fetch_auto_resume_enabled(self) -> bool:
        """Default ON: an operator who uploads a paper wants it used."""
        try:
            store = getattr(self, "_settings", None)
            if store is None or not hasattr(store, "get"):
                return True
            val = store.get("literature_fetch_auto_resume")
            return True if val is None else bool(val)
        except Exception:  # noqa: BLE001
            return True

    #: Agents a fetch arrival may resume. Only literature posts fetch requests,
    #: so a paper landing must not interrupt somebody's instrument chat. Wishlist
    #: answers carry no such restriction: whoever asked is who gets continued.
    _FETCH_RESUMABLE_AGENTS = frozenset({"literature"})

    @staticmethod
    def _resume_agents_for(kind: str, recs: list) -> "tuple[str, ...]":
        """Which agent should run the follow-up in a GROUP conversation."""
        if kind == "fetch":
            return ("literature",)
        for r in recs or []:
            aid = str((r or {}).get("agent_id") or "").strip()
            if aid and aid != "agent":
                return (aid,)
        return ("literature",)

    def _resume_private(self, conv: dict, text: str, *,
                        kind: str = "fetch") -> "tuple[bool, str]":
        """Drive one more turn on a private chat, in a daemon thread."""
        agent_id = str(conv.get("agent_id") or "")
        if kind == "fetch" and agent_id not in self._FETCH_RESUMABLE_AGENTS:
            return False, f"agent {agent_id!r} is not resumable for a fetch arrival"
        eng = getattr(self, "_conv_engine", None)
        if eng is None:
            return False, "no conversation engine"
        thread_id = str(conv.get("thread_id") or "")
        cid = str(conv.get("conversation_id") or "")
        if thread_id and eng.is_active(thread_id):
            # Never interrupt a turn in flight. The engine's own guard is the
            # second latch; this pre-check keeps us from even starting.
            return False, "conversation is mid-turn"

        def _drain() -> None:
            try:
                for _ in eng.stream_turn(cid, text, abort=None):
                    pass
            except Exception as exc:  # noqa: BLE001 — the turn renders its own errors
                logger.info("fetch-resume turn for %s ended: %s", cid, exc)

        _threading.Thread(target=_drain, daemon=True,
                          name=f"fetch-resume-{cid[:8]}").start()
        logger.info("fetch-resume: continuing private conversation %s", cid)
        return True, ""

    def _resume_group(self, conversation_id: str, text: str,
                      agents: "tuple[str, ...]" = ("literature",)) -> bool:
        """Run the follow-up as a background run whose transcript joins the group."""
        try:
            mgr = self._ensure_background_manager()
        except Exception as exc:  # noqa: BLE001
            logger.info("fetch-resume: background manager unavailable: %s", exc)
            return False
        if mgr is None:
            return False
        try:
            mgr.spawn(instruction=text, agents=agents,
                      conversation_id=conversation_id,
                      title="用户已答复 · 自动续工", priority="normal")
        except Exception as exc:  # noqa: BLE001
            logger.info("fetch-resume: spawn failed for %s: %s", conversation_id, exc)
            return False
        logger.info("fetch-resume: background follow-up for group %s", conversation_id)
        return True


    def _resolve_override_models(self) -> dict:
        """Resolve persisted per-agent model overrides into chat-model objects.

        Reads ConfigOverrideRegistry agent overrides (model ids as the Agents
        UI sets them), maps UI ids back to full model ids, and builds chat
        models. Agents with no override are left out (build uses their default).
        Best-effort: a model that can't be built is skipped, not fatal.
        """
        out: dict = {}
        try:
            from mast.admin.override_store import ConfigOverrideRegistry
            from mast.agents._shared.models import make_chat_model
            ovr = ConfigOverrideRegistry.get().get_agent_overrides() or {}
        except Exception as exc:
            logger.debug("resolve override models: registry read failed: %s", exc)
            return out
        # UI choice id → full model id understood by make_chat_model
        ui_to_model = {
            "kimi-k2.6": "kimi-k2.6", "kimi-k2.7-code": "kimi-k2.7-code",
            "kimi-k3": "kimi-k3",
            "deepseek-v4-pro": "deepseek-v4-pro",
            "sonnet-4.6": SONNET_4_6, "haiku-4.5": HAIKU_4_5,
            "deepseek-r1.5": "deepseek-v4-pro",
            "minimax-m3": MINIMAX_M3,
            "glm-5.1": GLM_5_1, "glm-5.2": GLM_5_2,
        }
        for aid, entry in ovr.items():
            if not isinstance(entry, dict):
                continue
            ui_model = entry.get("model")
            # _supervisor handled by _resolve_supervisor_model(); buffer_summarizer
            # is a side-channel (not an orchestrator node) so its model is applied
            # in _run_buffer_summarizer, not here.
            if not ui_model or aid in ("_supervisor", "buffer_summarizer"):
                continue
            model_id = ui_to_model.get(ui_model, ui_model)
            # Thinking strength: per-agent override, else the high default. This
            # is what makes the Think segmented control real (was persisted but
            # ignored before 2.1.13). Effective only on Claude models; reasoning
            # models (Kimi/DeepSeek) think high intrinsically.
            level = entry.get("thinking") or self._AGENTS_DEFAULT_THINKING.get(aid, "high")
            try:
                out[aid] = make_chat_model(model_id=model_id, max_tokens=4096,
                                           temperature=0.2, thinking_level=level)
            except Exception as exc:
                logger.debug("resolve override models: %s=%s failed: %s", aid, model_id, exc)
        return out


    def _resolve_supervisor_model(self):
        """Build a chat model for a persisted _supervisor override, else None.

        Returning None lets orchestrator.build() use its own default
        (AGENT_MODEL['orchestrator']). Honouring this makes the supervisor
        model picker real instead of persisted-but-ignored.
        """
        try:
            from mast.admin.override_store import ConfigOverrideRegistry
            from mast.agents._shared.models import make_chat_model
            ovr = ConfigOverrideRegistry.get().get_agent_overrides() or {}
        except Exception:
            return None
        entry = ovr.get("_supervisor")
        if not isinstance(entry, dict) or not entry.get("model"):
            return None
        model_id = self._UI_TO_MODEL_ID.get(entry["model"], entry["model"])
        # No thinking_level here on purpose: this model drives
        # with_structured_output(Route), and Claude extended thinking is
        # incompatible with the forced tool call that uses (review 2.1.13 #4).
        try:
            return make_chat_model(model_id=model_id, max_tokens=2048,
                                   temperature=0.1)
        except Exception as exc:
            logger.debug("resolve supervisor model %s failed: %s", model_id, exc)
            return None


    @staticmethod
    def _estimated_coarse_displacement_m(steps) -> float | None:
        """How far a coarse move probably went, from the operator's step
        calibration — or None if they never measured one.

        Open loop with no position feedback whatsoever, and piezo-actuator step
        size varies with amplitude, frequency, temperature and load. This is a
        rough magnitude recorded so a human can later ask "roughly how far did we
        travel from the original spot"; nothing computes with it, and no marker
        is ever re-projected from it."""
        if steps is None:
            return None
        try:
            from mast.core import instrument_profile as _ip
            step_m = _ip.get_config("xy_motor_step_m", None)
            if not step_m:
                return None
            return abs(float(steps)) * float(step_m)
        except Exception:  # noqa: BLE001 — bookkeeping is best-effort
            return None

    def _record_map_marker(self, payload: dict) -> None:
        """Persist one positioned skill as a scan-map marker in the experiment
        record (requirement: 此地图就是实验记录的地图). Fire-and-forget + fail-safe:
        a logging failure can NEVER affect the skill result or the chat."""
        try:
            from mast.io.exp_map import (
                can_be_positioned,
                classify_skill,
                marker_from_skill,
            )
            skill = payload.get("skill") or ""
            # Cheap triage BEFORE taking a state snapshot — this runs after every
            # skill, and most skills are not positioned. The two explicit branches
            # are checked here as well because ``classify_skill`` deliberately
            # does not know them (their coordinate semantics differ from its
            # generic precedence).
            category = _registered_category(getattr(self, "_registry", None), skill)
            coarse = lateral_coarse_move_info(payload)
            # 撞了没有 / 撞在哪 —— **两个问题分开问**。一次撞针可能定不了位
            # (``CheckScanForCrash`` 不带坐标参数),而定不了位**不是**没撞。
            crashed = is_crash_report(payload)
            crash_xy = crash_point(payload)
            # 第四条准入:结果里带**逐点定位记录**的 composite。
            #
            # 名字规则(``_SKILL_KIND_RULES``)全靠子串匹配,一个新 composite 的名字
            # 里没有 "scan"/"sts"/"pulse" 就整个落不进来 —— 而它已经**逐点报了
            # 坐标**,那是比名字强得多的证据。此前 ``SearchDomainBoundary`` /
            # ``CrossPointTipCheck`` 就卡在这里:分诊在读 ``data`` 之前就 return,
            # 于是几十个采样点在地图上一个都不存在,零报错。
            #
            # ⚠️ 这条准入**绕过了** ``classify_skill``,所以它必须自己守住那条
            # 排除:只读动词(``Get*``/``Analyze*``/``Plot*``…)、``*SelfCheck``、
            # snake_case 的 agent 工具、光学台一族、ANALYSIS 类,一个都不能靠
            # 「我报了坐标」混进来 —— 它们报的坐标是从数据里读出来的,不是针尖
            # 去过的地方,进地图就是假足迹。``can_be_positioned`` 分得开「不可能」
            # 与「名字规则认不出」,而 ``classify_skill`` 把两者折叠成同一个 None。
            subrecords = (_marker_subrecords(payload.get("data"))
                          if can_be_positioned(skill, category) else [])
            if (classify_skill(skill, category) is None
                    and coarse is None and not crashed and not subrecords):
                return
            storage = getattr(self, "_storage", None)
            if storage is None:
                return
            # Post-skill state: the cached snapshot (≤1 s old via the 1 s
            # background refresh) reflects the frame/tip the skill just set. Do
            # NOT fall back to payload["state_before"] — that is the PRE-skill
            # position and would record a scan/move at its STARTING point, not
            # its result (). With no live snapshot we rely on the
            # skill's explicit x/y params; marker_from_skill returns None if it
            # can place nothing, so we simply skip rather than mis-place.
            state = None
            st = getattr(self, "_state", None)
            if st is not None:
                try:
                    state = st.snapshot()
                except Exception:  # noqa: BLE001
                    state = None
            params = payload.get("params") or {}
            status = "failed" if payload.get("success") is False else "done"
            el = getattr(self, "_experiment_log", None)
            exp_id = getattr(el, "current_experiment_id", None) if el else None
            sample_id = getattr(el, "current_sample_id", None) if el else None

            # ── Explicit branch: lateral coarse move = coordinate boundary ──
            # Position is the tip readback AT THIS MOMENT, which is the boundary
            # expressed in the OLD frame: the stage moved, the piezo readout did
            # not, so post-move readback equals pre-move position. A None here is
            # fine and is recorded as such — the generation count is derived from
            # the row's existence, never from its coordinate.
            if coarse is not None:
                tip_x = _finite(getattr(state, "x_pos_m", None))
                tip_y = _finite(getattr(state, "y_pos_m", None))
                meta = {"direction": coarse["direction"], "steps": coarse["steps"],
                        "pos_src": "tip_readback" if tip_x is not None else None}
                if coarse.get("partial"):
                    # The stage moved and the skill still failed (typically the
                    # re-approach afterwards). The move is REAL — that is why it
                    # is recorded — but the row must not read like a clean
                    # relocation, or the next reader inherits a success that
                    # never happened.
                    meta["partial"] = True
                    meta["error"] = str(payload.get("error") or "")[:200] or None
                est = self._estimated_coarse_displacement_m(coarse["steps"])
                if est is not None:
                    # Open-loop estimate from an operator-supplied step
                    # calibration. Recorded as a raw magnitude only: we do NOT
                    # claim a signed displacement vector, because the mapping
                    # from direction code to stage axis is rig-specific and this
                    # actuator has no position feedback at all.
                    meta["est_disp_m"] = est
                _log_one_marker(
                    storage, kind="coarse_move", x_m=tip_x, y_m=tip_y,
                    label=(f"粗动换区 {coarse['direction']}"
                           + (f" ×{coarse['steps']}" if coarse["steps"] else "")
                           + ("(横移完成,后续步骤失败)" if coarse.get("partial") else "")),
                    skill_name=skill, status=status,
                    exp_id=exp_id, sample_id=sample_id,
                    meta=meta, advance=False)
                # Everything that cached a piezo-frame position now points at the
                # wrong surface. The marker table is covered by coord_epoch; this
                # covers the rest (plan overlay, crash block-list).
                from mast.core.coarse_move_effects import on_coarse_move_recorded
                on_coarse_move_recorded(source=f"skill:{skill}")
                return

            # ── Explicit branch: crash point ──
            # Written IN ADDITION to the failed scan footprint below (two rows:
            # "a scan failed here" and "the tip crashed here"), because they
            # answer different questions and the avoidance model needs the point.
            if crashed:
                # 位点由 ``crash_position`` 定,**一级都不许编**(优先级 + 尺寸闸
                # 见那个函数)。避让半径从 ``AnalysisConfig`` 现取 —— 与选点器画圈
                # 用的是同一个数,这里不写第二个。
                try:
                    from mast.core.map_scope import analysis_config
                    _crash_r = analysis_config(state).radius_for("crash")
                except Exception:  # noqa: BLE001 — 取不到就不做尺寸闸
                    _crash_r = None
                crash_xy, pos_src, unlocated_why = crash_position(
                    payload, state, crash_r_m=_crash_r)
                if crash_xy is None:
                    # 地图上留一条**无坐标**的审计行(``has_xy`` 为假 ⇒ 不产生
                    # 避让圈,这是诚实的:我们画不出那个圈)。
                    _log_one_marker(
                        storage, kind="crash", x_m=None, y_m=None,
                        label="撞针(位置未知)", skill_name=skill, status="failed",
                        exp_id=exp_id, sample_id=sample_id,
                        meta={"pos_src": pos_src,
                              "unlocated_reason": unlocated_why,
                              "error": str(payload.get("error") or "")[:200] or None},
                        advance=False)
                    # 而**能说出来**的那一份在进程内记忆里:哨兵格让
                    # ``FindCleanSpot`` 的 ``crash_memory_unlocated`` 把
                    # 「撞针确实发生过但坐标读不到 ⇒ 圈画不出来 ⇒ 返回的点无法
                    # 保证不在其上」这句话说给调用方听。
                    #
                    # ⚠️ 只在**定不了位**时写 tracker。定得了位的那条路不写 ——
                    # ``full_scan.py`` 自己已经 ``record_crash`` 过了,这里再记一次
                    # 就是同一次撞针数两遍,而阈值是 2 ⇒ 一次撞针就会被判成
                    # 「同点连撞」,逼出一次不该发生的换区。
                    try:
                        from mast.core.tip_crash_tracker import (
                            get_tip_crash_tracker,
                        )
                        get_tip_crash_tracker().record_crash(None, None)
                    except Exception:  # noqa: BLE001 — 记账绝不影响技能结果
                        pass
                else:
                    _log_one_marker(
                        storage, kind="crash", x_m=crash_xy[0], y_m=crash_xy[1],
                        label="撞针", skill_name=skill, status="failed",
                        exp_id=exp_id, sample_id=sample_id,
                        meta={"pos_src": pos_src,
                              "error": str(payload.get("error") or "")[:200] or None},
                        advance=False)
                # Fall through: the scan's own footprint marker still gets
                # written by the generic path below (if the skill is positioned).

            # The generic path below assumes a positioned skill. Only the two
            # explicit branches above — and a result carrying per-item
            # positioned records — let an unpositioned one get this far.
            if classify_skill(skill, category) is None and not subrecords:
                return

            # MULTI-POSITION skills first (). A
            # composite that visits N places used to leave exactly ONE marker,
            # at the composite's FINAL position, with the composite's overall
            # status: 9 scans across two BatchRegionsScan batches drew 2 boxes,
            # and the 3 regions the safety gate had rejected outright were not
            # merely missing — the one box that WAS drawn said `done`. The
            # per-region breakdown existed all along (batch_regions_scan's
            # `region_records`), it just never reached the map. Keyed on the
            # SHAPE of the result (a list of positioned records), not on a skill
            # name, so any composite that reports per-item positions is covered.
            regions = subrecords
            if regions:
                for r in regions:
                    r_ok = r.get("success")
                    _log_one_marker(
                        storage,
                        # 子记录声明的 kind 优先(已按 KIND_STYLE 白名单过滤);
                        # 没声明才回退到按技能名分类。名字规则认不出新 composite
                        # 时,"scan" 这个兜底会把一批谱学点画成扫描框。
                        kind=(r.get("kind") or classify_skill(skill, category)
                              or "scan"),
                        x_m=r["x_m"], y_m=r["y_m"], w_m=r.get("w_m"),
                        h_m=r.get("h_m"), skill_name=skill,
                        status="done" if r_ok else "failed",
                        label=str(r.get("label") or ""),
                        exp_id=exp_id, sample_id=sample_id,
                        # 技能自己的 meta 排在**后面** —— 它是这条记录的主体
                        # (指纹 / verdict / 参照系版本),而上面四个是记录层的
                        # 记账。同名时以技能报的为准。
                        meta={"region": r.get("index"), "error": r.get("error"),
                              "artifact_path": r.get("artifact_path"),
                              "pos_src": "region_record",
                              **(r.get("meta") or {})},
                        advance=bool(r_ok))
                return

            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            m = marker_from_skill(skill, params, state, status=status, data=data,
                                  category=category)
            m, dat_meta = _refine_marker_from_saved_file(m, data)
            if m is None:
                return
            small = {k: v for k, v in params.items()
                     if isinstance(v, (int, float, str, bool))}
            # ``advance`` 显式传，别吃 ``_log_one_marker`` 的默认 True（S4 STS 设计
            # §1.4a 点名的就是这里：这条路径不传 advance，于是**任何**落到地图上的
            # 单标记都顺手消耗一步勘测路线）。判据与上面多标记路径的 ``bool(r_ok)``
            # 是同一条：路线是「已经走到了哪」，一次失败的扫描没有走到，把它记成走到
            # 就是这个仓库反复吃亏的「分不清『做了』和『说做了』」。
            _log_one_marker(
                storage, kind=m.kind, x_m=m.x_m, y_m=m.y_m, w_m=m.w_m,
                h_m=m.h_m, angle_deg=m.angle_deg, label=m.label,
                skill_name=skill, status=m.status, exp_id=exp_id,
                sample_id=sample_id,
                meta={"params": small, **_pos_provenance(m, params, state, data),
                      **dat_meta},
                advance=(m.status == "done"))
        except Exception:  # noqa: BLE001
            _log_swallowed("map_marker", "map marker record failed for %s",
                           payload.get("skill"))


    def _start_map_activity_watcher(self, interval_s: float = 1.5) -> None:
        """Daemon that turns MANUAL Nanonis activity into experiment-record + map
        markers — covering the full space of operations an operator can do by hand
        (requirement 3, "手动点击覆盖了所有 skill 的操作").

        Two complementary detectors, both fed by the cached snapshot / saved files
        so neither touches the GUI thread:
          • STATE DIFF — sustained changes in the polled HardwareState: bias /
            setpoint / Z-controller status / scan frame / tip / scan start. Param
            sweeps are debounced; a skill-driven change (incl. non-positioned
            SetBias) is suppressed via _map_last_skill_ts.
          • SPECTRUM FILES — new .dat/.3ds (point/grid STS) since last poll, placed
            at the exact xy from the file header. This is the only reliable way to
            see a manual spectrum (there is no pollable spectroscopy status).

        KNOWN BLIND SPOTS (documented; the same ops done THROUGH MAST are still
        captured as skill markers — only bare-Nanonis-GUI use is missed):
          • one-shot transients with no status method + sub-poll duration —
            manual *bias pulses* and *tip shaping*;
          • modules not polled in HardwareState — *lock-in* config and *coarse
            motor* moves (would need new Nanonis reads; see the detectability
            matrix / PROGRESS — deferred Tier-2/3);
          • a manual *approach/retract* is captured only as its Z-controller
            status edge (On/Withdrawing/SafeTip), not a distinct approach event.

        Idempotent; best-effort; never raises into the GUI."""
        import threading
        if getattr(self, "_map_watch_thread", None) is not None and \
                self._map_watch_thread.is_alive():
            return
        self._map_watch_stop = threading.Event()

        def _loop() -> None:
            import glob
            import os
            import time as _t
            from mast.io.exp_map import snapshot_track
            # Seed baseline=prev from the current state so startup isn't logged,
            # and pre-set the scope so the first tick isn't a spurious reset.
            try:
                seed = snapshot_track(self._state.snapshot()
                                      if getattr(self, "_state", None) else None)
            except Exception:  # noqa: BLE001
                seed = snapshot_track(None)
            baseline, prev = seed, dict(seed)
            el0 = getattr(self, "_experiment_log", None)
            self._map_watch_last_scope = (
                getattr(el0, "current_experiment_id", None) if el0 else None,
                getattr(el0, "current_sample_id", None) if el0 else None)
            # Seed the spectrum "seen" map {path: mtime} with pre-existing files so
            # we never log files that were already on disk before MAST started.
            # mtime-keyed (not path-keyed) so a file rewritten/recreated with the
            # same path IS re-detected (review).
            # Resolve the dirs to watch (session + working-sessions + experiments)
            # — previously never set, so manual-spectrum detection was dead.
            try:
                self._scan_search_dirs = self._current_scan_dirs()
            except Exception:  # noqa: BLE001
                self._scan_search_dirs = ()
            seen: dict[str, float] = {}
            try:
                for d in (getattr(self, "_scan_search_dirs", ()) or ()):
                    if not d:
                        continue
                    for ext in ("**/*.dat", "**/*.3ds"):
                        for p in glob.glob(os.path.join(str(d), ext), recursive=True):
                            try:
                                seen[p] = os.path.getmtime(p)
                            except OSError:
                                pass
            except Exception:  # noqa: BLE001
                pass
            tick_n = 0
            while not self._map_watch_stop.is_set():
                try:
                    baseline, prev = self._map_state_tick(baseline, prev)
                except Exception as exc:  # noqa: BLE001 — surface so a broken
                    # watcher is visible, not silently dead (review).
                    logger.warning("map state tick error (manual capture may be "
                                   "incomplete): %s", exc)
                # Spectrum scan is heavier (recursive glob + file reads) → run it
                # every ~4th tick (~6 s) rather than every 1.5 s.
                if tick_n % 4 == 0:
                    try:
                        self._map_spectrum_tick(seen)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("map spectrum tick error: %s", exc)
                    # 兜底收编：用户在 Nanonis 里手动存的文件也要进实验文件夹。
                    # 只做【发现】，复制交给 ingest sink —— 200 MB 的 .3ds 复制
                    # 在这个线程上会把手动 marker 检测卡住好几秒。
                    try:
                        self._data_ingest_tick(seen)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("data ingest tick error: %s", exc)
                # 对话导出（~60 s 一次）：一条走 idx_conv_updated 索引的查询，
                # 没有新消息就什么都不做。这是唯一能保住被 8000 行上限裁掉的
                # 转录历史的地方。
                if tick_n % 40 == 4:
                    try:
                        self._chat_export_tick()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("chat export tick error: %s", exc)
                tick_n += 1
                end = _t.monotonic() + max(0.5, float(interval_s))
                while _t.monotonic() < end and not self._map_watch_stop.is_set():
                    _t.sleep(0.2)

        self._map_watch_thread = threading.Thread(
            target=_loop, name="MapActivityWatcher", daemon=True)
        self._map_watch_thread.start()
        # Graceful stop on process exit (matches the env-monitor / TTS daemons):
        # set the stop event so the loop's `_t.sleep` slices exit promptly and
        # the SQLite connection isn't abandoned mid-write ().
        import atexit
        atexit.register(lambda: self._map_watch_stop.set())
        logger.info("Map activity watcher started (interval=%.1fs)", interval_s)


    def _map_state_tick(self, baseline: dict, prev: dict):
        """One state-diff poll → manual markers. Returns (new_baseline, new_prev).

        Suppression policy lives here (the detector is pure): skip when no
        experiment is active or a skill ran recently (advance baseline silently so
        the skill's change is never later mis-logged as manual), and reset on a
        scope (experiment/sample) change so the previous scope's window can't
        silence the new one."""
        import time as _t

        from mast.io.exp_map import (
            detect_manual_state_changes, snapshot_track, tip_xy_from_state,
        )
        st = getattr(self, "_state", None)
        storage = getattr(self, "_storage", None)
        if st is None or storage is None:
            return baseline, prev
        state = st.snapshot()
        if state is None:
            return baseline, prev
        cur = snapshot_track(state)
        el = getattr(self, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None

        # Scope (experiment/sample) change → fresh baseline, clear suppression,
        # skip this transition tick (covers GUI 开始样品/实验 + agent tools).
        scope = (exp_id, sample_id)
        if scope != getattr(self, "_map_watch_last_scope", "__unset__"):
            self._map_watch_last_scope = scope
            self._map_last_skill_ts = 0.0
            return cur, cur

        # Capture the skill stamp ONCE (single atomic read). 3 s covers ~2 poll
        # cycles of detection latency after a skill while keeping the window short
        # enough that a manual fix right after a skill isn't lost for long.
        def _skill_recent() -> bool:
            return (_t.monotonic()
                    - getattr(self, "_map_last_skill_ts", 0.0)) < 3.0
        if exp_id is None or _skill_recent():
            # Nothing to attach to, or a skill is responsible → advance baseline
            # silently so we never later log this change as manual.
            return cur, cur

        # A live scan-vision monitor means MAST is running THIS scan, so its
        # scan-start is system-owned — never mark it "手动开始扫描". The 3 s skill
        # window alone misses a composite scan whose Scan_Action fires long after
        # the skill's gate stamp. Fail-safe: any import/attr error
        # → don't suppress (falls back to the skill-window policy).
        try:
            from mast.vision.scan_monitor import is_monitor_running
            _scan_is_system = bool(is_monitor_running())
        except Exception:  # noqa: BLE001
            _scan_is_system = False
        markers, new_baseline = detect_manual_state_changes(
            baseline, cur, prev, tip_xy=tip_xy_from_state(state),
            suppress_scan_start=_scan_is_system)
        # Re-check AFTER the diff: if a skill fired while we were computing, those
        # changes are the skill's — drop them (TOCTOU mitigation, review).
        if markers and _skill_recent():
            return cur, cur
        for m in markers:
            try:
                storage.log_marker(
                    kind=m.kind, x_m=m.x_m, y_m=m.y_m, w_m=m.w_m, h_m=m.h_m,
                    angle_deg=m.angle_deg, label=m.label, skill_name="",
                    status="done", source="manual",
                    experiment_id=exp_id, sample_id=sample_id, meta=m.meta or {})
            except Exception as exc:  # noqa: BLE001
                logger.debug("manual state marker failed: %s", exc)
        return new_baseline, cur


    def _map_spectrum_tick(self, seen: dict) -> None:
        """One spectrum-file poll → manual STS markers for NEW/REWRITTEN .dat/.3ds.

        The saved file is the reliable signal of a manual spectrum (no pollable
        status); its header gives the exact xy. *seen* maps path→mtime: a file is
        processed when first seen OR when its mtime changes (so a path reused in a
        later experiment is re-detected). A file modified in the last few seconds
        is SKIPPED (it may still be mid-write by Nanonis) and retried next poll. A
        skill STS's file is suppressed via the recent-skill window. Recursive glob
        so files saved in session subdirectories are not missed. (review fixes.)"""
        import glob
        import os
        import time as _t

        from mast.io.exp_map import manual_marker_from_spectrum_file
        storage = getattr(self, "_storage", None)
        if storage is None:
            return
        el = getattr(self, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        recent_skill = (_t.monotonic()
                        - getattr(self, "_map_last_skill_ts", 0.0)) < 30.0
        now = _t.time()
        # Refresh the search dirs each tick — the Nanonis session dir can change
        # when the operator opens a new session / starts a new experiment.
        try:
            self._scan_search_dirs = self._current_scan_dirs()
        except Exception:  # noqa: BLE001
            pass
        for d in (getattr(self, "_scan_search_dirs", ()) or ()):
            if not d:
                continue
            for ext in ("**/*.dat", "**/*.3ds"):
                try:
                    paths = glob.glob(os.path.join(str(d), ext), recursive=True)
                except Exception:  # noqa: BLE001
                    continue
                for p in paths:
                    try:
                        mt = os.path.getmtime(p)
                    except OSError:
                        continue
                    if now - mt < 5.0:        # still being written → wait
                        continue
                    if seen.get(p) == mt:     # this version already processed
                        continue
                    seen[p] = mt              # mark this version processed
                    if exp_id is None or recent_skill:
                        continue  # no scope, or skill-owned → not a manual op
                    try:
                        m = manual_marker_from_spectrum_file(p)
                        if m is None:
                            continue
                        storage.log_marker(
                            kind=m.kind, x_m=m.x_m, y_m=m.y_m, w_m=m.w_m,
                            h_m=m.h_m, label=m.label, skill_name="",
                            status="done", source="manual",
                            experiment_id=exp_id, sample_id=sample_id,
                            meta=m.meta or {})
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("manual spectrum marker failed: %s", exc)
        # Bound memory: drop entries for files that no longer exist (cheap, only
        # when the map has grown large over a very long session).
        if len(seen) > 4000:
            for p in [q for q in seen if not os.path.exists(q)]:
                seen.pop(p, None)


    # ── 实验文件夹 (2026-07-28) ───────────────────────────────────
    # 设计文档 docs/v2/design/experiment_folder_persistence.md

    def _init_experiment_folders(self) -> None:
        """接线实验文件夹：设置 → 根目录、ingest sink、作用域订阅。

        全程 best-effort：归档是记账功能，它出问题绝不能影响仪器控制。
        """
        from mast.core import experiment_paths as ep

        # 设置 → holder（单向流，core 不 import webui）
        try:
            root = str(self._setting("experiment_root") or "").strip()
            if root:
                ep.set_experiment_root(root)
        except Exception as exc:  # noqa: BLE001
            logger.debug("experiment_root setting unreadable: %s", exc)

        if not self._ingest_enabled():
            logger.info("experiment folder ingest disabled by settings")
            return

        # 上次崩溃遗留的 .part 文件（原子落地的中间态）
        try:
            from mast.logging.v2.filestore import sweep_partials
            n = sweep_partials(ep.experiment_root())
            if n:
                logger.info("swept %d stale .part files", n)
        except Exception:  # noqa: BLE001
            pass

        try:
            from mast.logging.v2.ingest_sink import QueuedIngestSink
            self._ingest = QueuedIngestSink(self._v2_repos)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ingest sink init failed: %s", exc)
            self._ingest = None

        # 作用域切换 → 建/轮转实验与样品目录。挂在 scope.subscribe 上而不是塞进
        # ExperimentLog：切换是 logging 层的事，建目录是文件层的事，两者只通过
        # 这个回调相连。回调 post-commit、离线程、抛异常不影响切换本身。
        try:
            from mast.logging import scope as _scope
            self._scope_unsub = _scope.subscribe(self._on_scope_change)
        except Exception as exc:  # noqa: BLE001
            logger.debug("scope subscribe failed: %s", exc)

        # 启动时先把当前作用域的目录铺好，这样第一次扫描不用等切换事件。
        try:
            self._ensure_scope_dirs()
        except Exception as exc:  # noqa: BLE001
            logger.debug("initial scope dirs failed: %s", exc)
        # 同样，启动时就把 v2 记录对齐到恢复出来的作用域。
        try:
            self._link_v2_scope()
        except Exception as exc:  # noqa: BLE001
            logger.debug("initial v2 scope link failed: %s", exc)
        # 捡回上次会话的 scan_id → 路径映射。这张表原本纯内存，重启后
        # DP 的 load_scan(scan_id) 就解析不出上次那张图了。
        try:
            from mast.core.scan_registry import load_state
            n = load_state()
            if n:
                logger.info("restored %d scan registry records", n)
        except Exception:  # noqa: BLE001
            pass

    def _ingest_enabled(self) -> bool:
        try:
            v = self._setting("ingest_enabled")
            return True if v is None else bool(v)
        except Exception:  # noqa: BLE001
            return True

    def _on_scope_change(self, old, new) -> None:
        """作用域切换后的文件侧动作。已在 scope 的单槽线程池上，可以做 I/O。"""
        try:
            self._ensure_scope_dirs()
        except Exception as exc:  # noqa: BLE001
            logger.warning("scope dir creation failed: %r", exc)
        # v2 记录跟着作用域走。此前 _v2_eid 是开机常量，所有实验/样品的 v2
        # 记录都堆在同一行「实时会话」上。
        try:
            self._link_v2_scope()
        except Exception as exc:  # noqa: BLE001
            logger.debug("v2 scope link failed: %r", exc)
        # 换样品 → 换 CSV 句柄（下一条读数自然开新文件）。
        try:
            sink = getattr(self, "_env_csv", None)
            if sink is not None:
                sink.rotate()
        except Exception:  # noqa: BLE001
            pass
        # 切走之前把上一个作用域的对话导出一次。这只是优化 —— 60 秒的 tick
        # 本来也会导，正确性不依赖这里（INCREMENTAL-ONLY）。
        try:
            self._chat_export_tick()
        except Exception:  # noqa: BLE001
            pass
        # 换样品后 Nanonis 可能还指着上一个样品的原位目录。只告警，不动手 ——
        # 绝不背着用户改他的 Nanonis 配置。数据正确性已经由「归属真源是活跃
        # 样品」兜住了（副本落在当前样品下）。
        try:
            self._check_stale_session_path()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_scope_dirs(self) -> tuple[object, str] | None:
        """确保当前实验/样品的目录存在，返回 ``(exp_dir, sample_dir_name)``。

        **懒创建**：目录只在真的有当前作用域时才建。空实验不预建目录 —— 这既是
        「十年后重启」的正确行为，也让几十条历史空壳实验不污染顶层。
        """
        log = self._experiment_log
        st = self._storage
        if log is None or st is None:
            return None
        eid, sid = log.current_experiment_id, log.current_sample_id
        if not eid:
            return None

        from mast.core import experiment_paths as ep
        from mast.logging.v2 import manifest as mf

        exp = st.get_experiment(eid)
        if not exp:
            return None

        dir_name = (exp.get("dir_name") or "").strip()
        if not dir_name:
            dir_name = ep.experiment_dir_name(
                eid, exp.get("name") or "", str(exp.get("start_time") or ""))
            try:
                st.set_experiment_dir_name(eid, dir_name)
            except Exception:  # noqa: BLE001
                pass
        exp_dir = ep.experiment_dir(dir_name, create=True)

        sample_dir_name = ""
        if sid:
            smp = st.get_sample(sid)
            if smp:
                sample_dir_name = (smp.get("dir_name") or "").strip()
                if not sample_dir_name:
                    # sample_ordinal（不是 next_sample_index）：这一行已经在库里，
                    # 用 "下一个" 会把它自己也数进去，第一个样品就成了 S02。
                    idx = smp.get("sample_index") or st.sample_ordinal(sid)
                    sample_dir_name = ep.sample_dir_name(
                        int(idx), smp.get("name") or "", sid)
                    try:
                        st.set_sample_dir_name(sid, sample_dir_name, int(idx))
                    except Exception:  # noqa: BLE001
                        pass
                sp = ep.sample_dir(exp_dir, sample_dir_name, create=True)
                # 样品会话自带针尖信息:数据是用哪根针取的,一年后只有这里说得清。
                # 快照(不是引用):针尖行会退役、属性会被补记,而这份 sample.json
                # 要如实记住"当时装的是这一根、当时它是这样登记的"。
                extra = None
                try:
                    from mast.core.tip_state import current_tip_facts
                    tf = current_tip_facts()
                    if tf:
                        extra = {"current_tip": tf}
                except Exception:  # noqa: BLE001 — 快照失败不该拦住建目录
                    extra = None
                mf.write_sample_manifest(
                    sp, sample_id=sid, name=smp.get("name") or "",
                    experiment_id=eid, description=smp.get("description") or "",
                    sample_type=smp.get("sample_type") or "",
                    sample_subtype=smp.get("sample_subtype") or "",
                    created_at=str(smp.get("start_time") or ""),
                    dir_name=sample_dir_name,
                    index=int(smp.get("sample_index") or 0),
                    extra=extra)

        samples = []
        try:
            for s in st.get_samples(eid):
                samples.append({
                    "id": s.get("id"), "name": s.get("name"),
                    "index": s.get("sample_index"), "dir_name": s.get("dir_name"),
                    "sample_type": s.get("sample_type") or "",
                })
        except Exception:  # noqa: BLE001
            pass
        mf.write_experiment_manifest(
            exp_dir, experiment_id=eid, title=exp.get("name") or "",
            goal=exp.get("goal_text") or "",
            created_at=str(exp.get("start_time") or ""), dir_name=dir_name,
            samples=samples,
            provenance={"db_file": str(getattr(st, "_db_path", "")),
                        "v1_row_id": eid,
                        "v2_experiment_id": getattr(self, "_v2_eid", None)})
        mf.write_readme(exp_dir)
        return exp_dir, sample_dir_name

    def _submit_ingest(self, action_id, paths, *, source: str = "skill",
                       skill: str = "") -> None:
        """把本次 action 的产物交给 ingest sink。

        同步路径上只做：算 scope、put_nowait。任何异常都吞掉 —— 退化后的行为
        就是今天的行为（只登记不搬运），零回归。
        """
        sink = getattr(self, "_ingest", None)
        if sink is None or not paths:
            return
        try:
            scope = self._ensure_scope_dirs()
            if not scope:
                return          # 没有当前实验 → 不搬运（门控已经拦住产数据操作）
            exp_dir, sample_dir_name = scope
            if not sample_dir_name:
                return
            log = self._experiment_log
            from mast.core.experiment_paths import sample_dir as _sd
            sink.submit(
                paths, exp_dir=exp_dir, sample_dir_name=sample_dir_name,
                experiment_id=getattr(self, "_v2_eid", "") or "",
                sample_id=log.current_sample_id if log else None,
                action_id=action_id, source=source, skill=skill,
                copy_mode=self._ingest_copy_mode(),
                active_sample_dir=_sd(exp_dir, sample_dir_name),
                stray_sample_dir_name=sample_dir_name,
            )
        except Exception as exc:  # noqa: BLE001 — 归档失败绝不影响扫描
            logger.debug("ingest submit skipped: %r", exc)

    def _ingest_copy_mode(self) -> str:
        try:
            m = str(self._setting("ingest_copy_mode") or "copy").strip()
            return m if m in ("copy", "hardlink") else "copy"
        except Exception:  # noqa: BLE001
            return "copy"

    def _link_v2_scope(self) -> str | None:
        """把 ``_v2_eid`` 对齐到当前作用域（v1 实验 ↔ v2 campaign，样品 ↔ sample）。

        best-effort：失败就保留原来的 ``_v2_eid``（退化成今天的行为 —— 记录仍然
        写得进去，只是归属粗一点）。
        """
        repos = getattr(self, "_v2_repos", None)
        log = self._experiment_log
        st = self._storage
        if repos is None or log is None or st is None:
            return getattr(self, "_v2_eid", None)
        eid = log.current_experiment_id
        if not eid:
            return getattr(self, "_v2_eid", None)
        exp = st.get_experiment(eid)
        if not exp:
            return getattr(self, "_v2_eid", None)
        smp = st.get_sample(log.current_sample_id) if log.current_sample_id else None
        from mast.logging.v2.live import link_scope
        cid, v2_sid, v2_eid = link_scope(repos, v1_exp=exp, v1_sample=smp, storage=st)
        if v2_eid:
            self._v2_eid = v2_eid
        return getattr(self, "_v2_eid", None)

    def _current_env_scope(self) -> tuple[str | None, str | None]:
        """(experiment_id, sample_id) —— 给 environment_log 的归属列。"""
        log = self._experiment_log
        if log is None:
            return None, None
        return log.current_experiment_id, log.current_sample_id

    def _env_csv_scope(self):
        """(exp_dir, sample_dir_name, experiment_id, sample_id) 给 CSV sink。

        没有当前实验时返回 None → 读数只进 DB 不落文件（无处可落）。
        """
        log = self._experiment_log
        if log is None or not log.current_experiment_id:
            return None
        scope = self._ensure_scope_dirs()
        if not scope:
            return None
        exp_dir, sample_dir = scope
        return exp_dir, sample_dir, log.current_experiment_id, log.current_sample_id

    def _env_csv_enabled(self) -> bool:
        try:
            v = self._setting("env_csv_enabled")
            return True if v is None else bool(v)
        except Exception:  # noqa: BLE001
            return True

    # ── 环境历史（envhistory）接线 ────────────────────────────────────

    def _build_nanonis_env_sensors(self) -> list:
        """Nanonis 侧的环境传感器。**它们由 runtime 构造，不由 autodetect。**

        因为它们需要一个活的 ConnectionPool / InstrumentState，而 environment/
        层刻意不 import core/ —— 那条单向依赖是它能被单独测试的原因。

        两类：

        * ``tunnel_current`` —— 零 TCP，镜像 :class:`InstrumentState` 那个 1 Hz
          刷新已经读到的电流。这就是用户要的"针尖长时间恒流隧穿停留下的
          current"。**不另开轮询**：那个值每秒都已经在被读了，而它所在的
          ``monitor`` 角色早就满载。
        * ``environment_sensors.json`` 里 ``type="nanonis_signal"`` 的条目 ——
          接在控制器模拟输入上的表（磁场、液氦液面若是这种接法就走这里,
          零新代码）。

        返回空列表是完全正常的：没连 Nanonis、或者关掉了记录，都不该让环境
        监控少一条腿。

        ⚠️ 这里检查 ``eh_enabled`` 而不是只让 sink 去检查，是因为这些传感器**存在**
        本身就会往 ``environment_log`` 每 2 秒写一行。关掉记录 = 关掉保留期清扫,
        那时再往那张表加一条高频序列正是这个子系统要解决的问题。代价是：运行中
        把 ``eh_enabled`` 从关拨到开，镜像传感器要等下一次传感器重建才出现 ——
        「高级管理 → 重新扫描传感器」或重启。旋钮的其余部分都是下一拍生效。
        """
        out: list = []
        try:
            from mast.envhistory.thresholds import get_env_history_thresholds
            if not get_env_history_thresholds().enabled:
                return out
        except Exception:  # noqa: BLE001
            return out
        try:
            from mast.environment.nanonis_env import (
                InstrumentStateSensor,
                NanonisSignalSensor,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("nanonis env sensors unavailable: %r", exc)
            return out

        if getattr(self, "_state", None) is not None:
            try:
                out.append(InstrumentStateSensor(lambda: getattr(self, "_state", None)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("tunnel_current sensor init failed: %r", exc)
        try:
            from mast.environment.config import load_config
            for entry in (load_config().get("sensors") or []):
                if entry.get("type") != "nanonis_signal":
                    continue
                name = (entry.get("name") or entry.get("id") or "").strip()
                signal = (entry.get("signal_name") or entry.get("signal") or "").strip()
                idx = entry.get("signal_index")
                if not name or (not signal and idx is None):
                    logger.warning("nanonis_signal 条目缺 name 或 signal_name，已跳过：%r",
                                   entry)
                    continue
                out.append(NanonisSignalSensor(
                    lambda: getattr(self, "_pool", None),
                    name=name, signal_name=signal or None,
                    signal_index=int(idx) if idx is not None else None,
                    unit=str(entry.get("unit") or ""),
                    quiet_gated=bool(entry.get("quiet_gated", False)),
                ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("nanonis_signal sensors init failed: %r", exc)
        return out

    def _build_env_history_recorder(self, sensors: list):
        """建历史记录器并登记为进程单例（电流监控的段流钩子按名字找它）。"""
        try:
            from mast.envhistory.recorder import EnvHistoryRecorder, set_recorder
        except Exception as exc:  # noqa: BLE001
            logger.debug("env history recorder unavailable: %r", exc)
            return None
        try:
            rec = EnvHistoryRecorder(
                storage_getter=lambda: getattr(self, "_storage", None),
                pool_getter=lambda: getattr(self, "_pool", None),
                state_getter=lambda: getattr(self, "_state", None),
                scope_provider=self._env_csv_scope,
            )
            # 哪些序列只在仪器安静时才记 —— 由传感器自己声明（quiet_gated），
            # 而不是在 sink 里硬编码一张名字表，否则改个传感器名就静默失效。
            rec.sink.set_gated_sensors([
                s.name() for s in sensors
                if getattr(getattr(s, "_inner", s), "quiet_gated", False)
            ])
            set_recorder(rec)
            return rec
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create EnvHistoryRecorder: %s", exc)
            return None

    def _sample_dir_for(self, sample_id: str) -> str | None:
        """sample_id → 样品目录名（没有就现算一个并记下来）。"""
        st = self._storage
        if st is None or not sample_id:
            return None
        try:
            smp = st.get_sample(sample_id)
            if not smp:
                return None
            name = (smp.get("dir_name") or "").strip()
            if name:
                return name
            from mast.core.experiment_paths import sample_dir_name
            idx = smp.get("sample_index") or st.sample_ordinal(sample_id)
            name = sample_dir_name(int(idx), smp.get("name") or "", sample_id)
            st.set_sample_dir_name(sample_id, name, int(idx))
            return name
        except Exception:  # noqa: BLE001
            return None

    # ── watcher 兜底：收编手动保存的文件 ──────────────────────────

    def _data_ingest_tick(self, seen: dict) -> None:
        """扫 Nanonis 保存目录，把新出现的测量文件交给 ingest sink。

        **只做发现，绝不在这个线程上复制。** 与 ``_map_spectrum_tick`` 共用
        ``seen`` 字典（那边只用 mtime，这边用 (mtime, size) —— 同 mtime 重写也能
        抓到，所以值是元组时两边都能读）。

        ``seen`` 持久化在 ``.mast/ingest_state.json``：重启后不重扫已知文件，
        而 **MAST 关着的时候 Nanonis 写的文件因为不在 seen 里会被收进来** ——
        这正是我们要的兜底。
        """
        import glob
        import os

        sink = getattr(self, "_ingest", None)
        if sink is None or not self._ingest_watcher_enabled():
            return
        scope = self._ensure_scope_dirs()
        exp_dir, sample_dir = scope if scope else (None, None)

        from mast.core.experiment_paths import sample_dir as _sd
        from mast.logging.v2.filestore import Zone, classify

        active_sample_dir = _sd(exp_dir, sample_dir) if (exp_dir and sample_dir) else None
        log = self._experiment_log
        found: list[str] = []

        for d in (getattr(self, "_scan_search_dirs", ()) or ()):
            if not d:
                continue
            for ext in ("**/*.sxm", "**/*.dat", "**/*.3ds"):
                try:
                    paths = glob.glob(os.path.join(str(d), ext), recursive=True)
                except OSError:
                    continue
                for p in paths:
                    try:
                        stt = os.stat(p)
                    except OSError:
                        continue
                    key = (stt.st_mtime, stt.st_size)
                    prev = seen.get(p)
                    # 兼容 _map_spectrum_tick 写进来的裸 mtime。
                    prev_key = prev if isinstance(prev, tuple) else (prev, None)
                    if prev_key == key or (prev is not None and not isinstance(prev, tuple)
                                           and prev == stt.st_mtime):
                        continue
                    seen[p] = key
                    found.append(p)

        if not found:
            self._persist_ingest_state(exp_dir, seen)
            return

        for p in found[:64]:      # 一跳最多提交这么多，避免首次启动时雪崩
            zone = classify(p, active_sample_dir=active_sample_dir)
            if zone is Zone.MANAGED:
                continue          # 我们自己的副本 —— 递归闸门
            if exp_dir is None or not sample_dir:
                self._quarantine_file(p)
                continue
            aid = self._synth_manual_action(p)
            sink.submit(
                [p], exp_dir=exp_dir, sample_dir_name=sample_dir,
                experiment_id=getattr(self, "_v2_eid", "") or "",
                sample_id=log.current_sample_id if log else None,
                action_id=aid, source="manual", skill="",
                copy_mode=self._ingest_copy_mode(),
                zone=zone, active_sample_dir=active_sample_dir,
                stray_sample_dir_name=sample_dir,
            )
            if zone is Zone.INPLACE_FOREIGN:
                self._note_stale_session_path(p)

        self._persist_ingest_state(exp_dir, seen)

    def _ingest_watcher_enabled(self) -> bool:
        try:
            v = self._setting("ingest_watcher_enabled")
            return True if v is None else bool(v)
        except Exception:  # noqa: BLE001
            return True

    def _persist_ingest_state(self, exp_dir, seen: dict) -> None:
        """把 seen 落盘（每 ~20 次调用一次，避免频繁写小文件）。"""
        self._ingest_state_ticks = getattr(self, "_ingest_state_ticks", 0) + 1
        if exp_dir is None or self._ingest_state_ticks % 20 != 1:
            return
        try:
            import json
            p = exp_dir / ".mast" / "ingest_state.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            # 只留最近 4000 条，别让这个文件无限长
            items = list(seen.items())[-4000:]
            payload = {k: (list(v) if isinstance(v, tuple) else v) for k, v in items}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            import os
            os.replace(tmp, p)
        except Exception:  # noqa: BLE001
            pass

    def _synth_manual_action(self, path: str) -> str | None:
        """给手动保存的文件合成一条 action。

        ``scan_files.produced_by_action_id`` 是 ``NOT NULL REFERENCES actions(id)``，
        绕不过。合成的 action 让手动保存在动作时间线里成为一等公民 —— 与
        ``_map_spectrum_tick`` 已经在做的 ``source="manual"`` marker 完全平行。
        """
        repos = getattr(self, "_v2_repos", None)
        eid = getattr(self, "_v2_eid", None)
        if repos is None or not eid:
            return None
        try:
            aid = repos.actions.begin(
                experiment_id=eid, agent_id="operator",
                action_type="manual_save",
                params={"origin_path": str(path), "detected_by": "watcher"})
            repos.actions.succeed(aid)
            return aid
        except Exception:  # noqa: BLE001
            return None

    def _quarantine_file(self, path: str) -> None:
        """没有活跃实验/样品时收到的文件 → 隔离区。**绝不丢字节。**

        宁可让用户事后认领，也不能因为"当时没选样品"就把一次真实测量扔掉。
        """
        try:
            import json
            import shutil
            from datetime import datetime, timezone

            from mast.core.experiment_paths import quarantine_dir
            from mast.logging.v2.filestore import sha256_of

            src = Path(path)
            sha, size = sha256_of(src)
            # 按 sha 去重：seen 丢失后重新发现同一个文件，不该在隔离区里堆出
            # 好几份。索引是权威（隔离区的文件名可能因同名冲突而不同）。
            noted = getattr(self, "_quarantined_shas", None)
            if noted is None:
                noted = self._quarantined_shas = _load_quarantine_shas()
            if sha in noted:
                return
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            qd = quarantine_dir() / day
            qd.mkdir(parents=True, exist_ok=True)
            dest = qd / src.name
            if dest.exists():
                dest = qd / f"{src.stem}.{sha[:8]}{src.suffix}"
            shutil.copy2(src, dest)
            noted.add(sha)
            with open(quarantine_dir() / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sha256": sha, "size": size,
                    "rel_path": f"{day}/{src.name}",
                    "origin_path": str(src),
                    "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "reason": "no_active_sample",
                }, ensure_ascii=False) + "\n")
            self._emit_event("data.file.quarantined",
                             {"path": str(src), "reason": "no_active_sample"},
                             severity="warning")
        except Exception as exc:  # noqa: BLE001
            logger.debug("quarantine failed for %s: %r", path, exc)

    @staticmethod
    def _load_quarantine_shas() -> set:
        return _load_quarantine_shas()

    def _check_stale_session_path(self) -> None:
        """Nanonis 的保存目录是否还指着上一个样品。只发事件，不改 Nanonis。

        只在**原位模式已经被用户开启过**时才有意义 —— 如果 session path 根本
        不在实验文件夹里（默认情况），那是用户自己的目录，不该对它有任何意见。
        """
        scope = self._ensure_scope_dirs()
        if not scope:
            return
        exp_dir, sample_dir = scope
        if not sample_dir:
            return
        cur = getattr(self, "_session_dir", None) or getattr(self, "_session_path", None)
        if not cur:
            return
        from mast.core import nanonis_session as ns
        stale_dir = ns.session_path_sample(str(cur), exp_dir)
        if not stale_dir or stale_dir == sample_dir:
            return
        msg = ns.stale_warning(stale_dir, sample_dir)
        self._emit_event("data.session_path.stale", {
            "session_path": str(cur),
            "session_sample_dir": stale_dir,
            "current_sample_dir": sample_dir,
            "message": msg,
        }, severity="warning")
        logger.warning("%s", msg)

    def _note_stale_session_path(self, path: str) -> None:
        """Nanonis 还在往上一个样品的目录里写 —— 告警，但不动手。

        默认**绝不背着用户改 Nanonis 配置**。数据的正确性已经由"归属真源是
        活跃样品"兜住了（副本落在当前样品下），这里只是让用户知道该去改
        Nanonis 的保存目录。
        """
        last = getattr(self, "_stale_session_warned", 0.0)
        now = _t.monotonic()
        if now - last < 300:      # 5 分钟内只提醒一次
            return
        self._stale_session_warned = now
        log = self._experiment_log
        self._emit_event("data.session_path.stale", {
            "path": str(path),
            "current_sample": (log.current_sample_id if log else None),
            "suggested_action": "把 Nanonis 的保存目录重新指向当前样品，或忽略"
                                "（数据已按当前样品归档）",
        }, severity="warning")
        logger.warning("Nanonis session path 仍指向旧样品的目录：%s "
                       "—— 数据已按当前样品归档，但建议在 Nanonis 里改回来", path)

    def _emit_event(self, topic: str, payload: dict, *, severity: str = "info") -> None:
        repos = getattr(self, "_v2_repos", None)
        eid = getattr(self, "_v2_eid", None)
        if repos is None or not eid:
            return
        try:
            repos.events.publish(topic=topic, kind="event", payload=payload,
                                 severity=severity, experiment_id=eid,
                                 producer="runtime")
        except Exception:  # noqa: BLE001
            pass

    # ── 对话导出 ──────────────────────────────────────────────────

    def _chat_export_tick(self) -> None:
        """把新增的转录增量导出进实验文件夹。

        这是唯一能保住被 ``_MAX_TRANSCRIPT_ROWS``(8000) 裁掉的历史的地方：
        DB 会裁，按 seq 只追加的导出文件不会。
        """
        if not self._chat_export_enabled():
            return
        store = getattr(self, "_conv_store", None)
        log = self._experiment_log
        if store is None or log is None or not log.current_experiment_id:
            return
        scope = self._ensure_scope_dirs()
        if not scope:
            return
        exp_dir, _ = scope
        try:
            from mast.chat.export import export_conversations
            res = export_conversations(
                exp_dir, store, experiment_id=log.current_experiment_id,
                sample_dir_resolver=self._sample_dir_for)
            if res.get("entries"):
                logger.debug("chat export: %s", res)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chat export failed: %r", exc)

    def _chat_export_enabled(self) -> bool:
        try:
            v = self._setting("conv_export_enabled")
            return True if v is None else bool(v)
        except Exception:  # noqa: BLE001
            return True

    def _skill_trace_recorder(self, payload: dict) -> str | None:
        """Injected into wrap_skill; records one IC skill call as a tool_call step
        on the active trajectory. Fire-and-forget + fail-safe.

        Returns the v2 action id (or None) so wrap_skill can hand it to a
        post_hook — that is what keeps a rendered figure attached to the REAL
        action row instead of a second one invented for the same skill call."""
        # Mark skill activity (ANY skill) so the manual-activity watcher won't
        # misattribute this skill's state change (incl. non-positioned ones like
        # SetBias/SetSetpoint) as a manual operation. Fires at skill END; the
        # safety recorder also stamps it at the gate (≈ skill START) so the
        # window brackets a long skill's whole duration.
        import time as _t
        self._map_last_skill_ts = _t.monotonic()
        # Spatial map marker FIRST, independent of the training-trace sink: every
        # positioned skill (scan / STS / pulse / tip-shape / move) lands on the
        # experiment-record map regardless of whether trajectory recording is on.
        # This recorder is wired into BOTH the private-chat IC and the orchestrator
        # build, so it is the single point that covers every skill path.
        self._record_map_marker(payload)
        # Persist every skill outcome for the Records action timeline, including
        # failures. This store is independent of the optional training trace sink;
        # a recording error must not prevent skill execution.
        self._record_v1_action(payload)
        # Ground truth for the claim cross-check ().
        self._note_run_skill(payload)
        # V2 records action — the store the Records tab reads. Written HERE (the
        # recorder seam, which fires for every outcome) rather than from the
        # tip-shape post_hook, which wrap_skill only calls on success: that is
        # why the v2 store held 22 rows, all 'succeeded', for a session the v1
        # store recorded 27 actions and 5 failures for ().
        action_id = self._record_v2_action(payload)
        sink = getattr(self, "_trace_sink", None)
        traj = getattr(self, "_active_traj", None)
        if sink is None or traj is None:
            return action_id
        try:
            sink.record_step(
                trajectory_id=traj,
                step_type="tool_call",
                actor_kind="skill",
                agent_id="IC",
                tool_call_id=payload.get("tool_call_id") or None,
                # Point the step at the fact row it describes (RFC §1: steps hold
                # pointers, facts live in `actions`). The action is INSERTed
                # synchronously above and the sink's worker is FIFO, so the
                # referenced row always exists by the time this step is written.
                action_id=action_id,
                input={"params": payload.get("params"),
                       "state_before": self._state_to_dict(payload.get("state_before"))},
                output={"skill": payload.get("skill"),
                        "skill_version": payload.get("skill_version"),
                        "danger_level": payload.get("danger_level"),
                        "success": payload.get("success"),
                        "summary": payload.get("summary"),
                        "error": payload.get("error"),
                        "artifact_path": payload.get("artifact_path"),
                        "rolled_back": payload.get("rolled_back")},
                duration_ms=payload.get("duration_ms"),
            )
        except Exception:
            _log_swallowed("skill_trace_step", "skill trace step failed for %s",
                           payload.get("skill"))
        return action_id

    def _note_skill_activity(self) -> None:
        """打「刚有技能在动仪器」的戳 —— 手动操作监视器据此不把状态变化记成人手。"""
        import time as _t
        self._map_last_skill_ts = _t.monotonic()

    def _record_direct_skill_call(self, payload: dict) -> dict:
        """一次**直调**（技能直调 API / 外部 agent 网关）的入账 —— 只写事实表。

        为什么不复用 :meth:`_skill_trace_recorder`：直调的顶层技能走
        ``ExecutionContext.run``，地图标记已经由 ``run()`` 里的 ``marker_sink`` 记过
        了，再走那条 recorder 会**同一动作两条标记**；它还会把动作记进当前群聊
        run 的核对台账（按 ``_orch_run_id``）与训练轨迹（写死 IC）—— 对一个与群聊
        无关的外部调用，两者都是错的归属。这里只做三件事：打活动戳、v1 行、v2 行
        （v2 成功分支自带 scan_files 登记与实验文件夹归档）。

        返回 ``{"v1", "v2_action_id", "experiment_id", "sample_id"}``，让调用方能
        如实告诉外部 agent「这次动作记进去了没有、记在哪」—— 没有活动实验时 v1 的
        外键会让插入失败，那正是要说出来的事。
        """
        self._note_skill_activity()
        el = getattr(self, "_experiment_log", None)
        exp_id = getattr(el, "current_experiment_id", None) if el else None
        sample_id = getattr(el, "current_sample_id", None) if el else None
        v1_ok = bool(self._record_v1_action(payload))
        action_id = self._record_v2_action(payload)
        return {"v1": v1_ok, "v2_action_id": action_id,
                "experiment_id": exp_id, "sample_id": sample_id}


    def _record_v1_action(self, payload: dict) -> bool:
        """Persist one skill call as a V1 ActionRecord — the store the Records UI
        reads (get_experiment_detail → storage.get_actions). Records EVERY skill
        with its REAL outcome so failures (timeouts, crashes, missing-module) are
        visible in the experiment record, not just successes. Fire-and-forget +
        fail-safe: a logging failure must never affect the skill result.

        NOTE: precondition failures short-circuit in wrap_skill BEFORE the recorder
        fires, so they are not captured here; execution outcomes (the bulk — the
        Nanonis timeouts / CRASH_DETECTED / module-not-running of 2026-07-06) are.

        Returns True when the row was written (callers on the agent path ignore
        it; the direct-call path reports it back to the external caller).
        ``payload["context"]`` (optional) fills the ``actions.context`` column —
        who asked for this call when that is not the in-process agent.
        """
        storage = getattr(self, "_storage", None)
        if storage is None:
            return False
        try:
            import json as _json
            import uuid as _uuid
            from datetime import datetime as _dt

            from mast.core.types import ActionRecord, SkillResult

            el = getattr(self, "_experiment_log", None)
            exp_id = getattr(el, "current_experiment_id", None) if el else None
            sample_id = getattr(el, "current_sample_id", None) if el else None
            # action_to_dict json.dumps(parameters) WITHOUT a default handler, so a
            # stray non-serialisable arg would raise — pre-flatten to JSON-safe.
            try:
                params = _json.loads(_json.dumps(payload.get("params") or {}, default=str))
            except Exception:
                params = {}
            from mast.core.types import HardwareState

            # Preserve the result payload, including returned readings, state,
            # calls, duration, summary and artifact paths, when recording the action.
            duration_s = float(payload.get("duration_ms") or 0) / 1000.0
            elapsed = payload.get("elapsed_s")
            data = _v1_result_data(payload.get("data"))
            artifact = payload.get("artifact_path")
            if artifact and "artifact_path" not in data:
                data["artifact_path"] = str(artifact)
            calls = _v1_nanonis_calls(payload.get("nanonis_calls"))
            sb = payload.get("state_before")
            sa = payload.get("state_after")
            sb = sb if isinstance(sb, HardwareState) else None
            sa = sa if isinstance(sa, HardwareState) else None
            summary = str(payload.get("summary") or "") or None
            result = SkillResult(
                skill_name=str(payload.get("skill") or ""),
                success=bool(payload.get("success")),
                error=str(payload.get("error") or ""),
                data=data,
                elapsed_s=(float(elapsed) if isinstance(elapsed, (int, float))
                           else duration_s),
                state_before=sb,
                state_after=sa,
                nanonis_calls=calls,
                summary=summary,
            )
            rec = ActionRecord(
                id=_uuid.uuid4().hex,
                experiment_id=exp_id or "",
                sample_id=sample_id or "",
                timestamp=_dt.now().isoformat(),
                skill_name=str(payload.get("skill") or ""),
                skill_version=str(payload.get("skill_version") or ""),
                parameters=params,
                result=result,
                state_before=sb,
                state_after=sa,
                # The report / timeline / citations readers all look at the
                # ACTION-level list first (records_export.py, report.py,
                # citations/manager.py `if action.nanonis_calls:`), so it is
                # populated here as well as inside the result.
                nanonis_calls=calls,
                duration_s=duration_s,
                # WHO authorised this, not HOW DANGEROUS it is. This column used
                # to be handed the skill's danger level ("CONFIRM"/"AUTO"), which
                # made every consumer of the v1 vocabulary ("auto"/"llm"/"human")
                # read garbage. NOTE: it still cannot say a HUMAN approved a
                # CONFIRM-level skill — nothing on this path records the HITL
                # verdict, and the v2 `approvals` table is empty ( §2).
                approval_source=str(payload.get("approval_source") or "auto"),
                context=str(payload.get("context") or ""),
            )
            storage.log_action(rec)
            return True
        except Exception:  # logging must never break the skill
            _log_swallowed("v1_action_record", "V1 action record failed for %s",
                           payload.get("skill"))
            return False


    def _note_run_skill(self, payload: dict) -> None:
        """Accumulate executed skills and successfully written artifacts for this run.

        This bounded cross-check buffer lets claim auditing distinguish reported
        completion from recorded actions. It retains only the most recent runs."""
        run = str(getattr(self, "_orch_run_id", "") or "")
        led = getattr(self, "_run_ledger", None)
        if led is None:
            led = {}
            self._run_ledger = led
        entry = led.get(run)
        if entry is None:
            for k in list(led)[:-7]:      # keep the 8 most recent runs
                led.pop(k, None)
            entry = {"skills": [], "artifacts": []}
            led[run] = entry
        try:
            name = str(payload.get("skill") or "")
            if name:
                entry["skills"].append(name)
                del entry["skills"][:-500]
            if payload.get("success"):
                # Only a SUCCEEDED call may contribute an artifact: a failed
                # skill's would-be path is precisely the kind of thing that
                # must not corroborate a claim.
                entry["artifacts"].extend(_artifact_paths(payload))
                del entry["artifacts"][:-500]
        except Exception:  # noqa: BLE001 — the ledger must never break a skill
            _log_swallowed("run_ledger", "run ledger update failed for %s",
                           payload.get("skill"))


    def audit_run_claim(self, text: str, *, run_id: str | None = None) -> dict:
        """Cross-check an agent's claim against what the run's record shows.

        Returns ``mast.logging.v2.claim_audit.ClaimAudit.as_dict()``; ``ok`` is
        False only when the record PROVES a discrepancy (a cited file this run
        never produced and that isn't on disk, or a completed-measurement claim
        in a run with zero skill calls). Safe to call from anywhere — it never
        raises, and it declines to judge when there is no ledger to judge
        against (``_run_ledger`` absent ⇒ no skill has ever been recorded in
        this process, so "zero skills ran" would mean nothing)."""
        try:
            from mast.logging.v2.claim_audit import audit_claim
            led = getattr(self, "_run_ledger", None)
            if led is None:
                return audit_claim(text, executed_skills=None).as_dict()
            run = str(run_id if run_id is not None
                      else (getattr(self, "_orch_run_id", "") or ""))
            entry = led.get(run) or {"skills": [], "artifacts": []}
            return audit_claim(text, executed_skills=entry["skills"],
                               artifacts=entry["artifacts"]).as_dict()
        except Exception:  # noqa: BLE001
            _log_swallowed("audit_run_claim", "claim audit failed")
            return {"ok": True, "problems": [], "executed_skills": [],
                    "claimed_paths": [], "fabricated_paths": [],
                    "unsupported_completion": False}


    def _record_v2_action(self, payload: dict) -> str | None:
        """Persist each skill call as a V2 actions row and return its id.

        Record succeeded or failed outcome, error, actual parameters, duration and
        tool_call_id. The Records UI and audit queries share this store. Recording
        is best-effort and must not break the skill."""
        repos = getattr(self, "_v2_repos", None)
        eid = getattr(self, "_v2_eid", None)
        if repos is None or not eid:
            return None
        skill = str(payload.get("skill") or "")
        try:
            import json as _json
            try:
                params = _json.loads(_json.dumps(payload.get("params") or {},
                                                 default=str))
            except Exception:  # noqa: BLE001
                params = {}
            duration_ms = payload.get("duration_ms")
            duration_ms = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
            # agent_id / thread_id：payload 里带了就用（直调与外部网关 —— 它们与
            # 当前群聊线程无关，挂上去就是错的归属），没带就保持原来的缺省。
            aid = repos.actions.begin(
                experiment_id=eid,
                agent_id=str(payload.get("agent_id") or "instrument_control"),
                action_type=skill,
                params=params,
                thread_id=(payload["thread_id"] if "thread_id" in payload
                           else getattr(self, "_active_thread_id", None)),
                tool_call_id=payload.get("tool_call_id") or None,
            )
            # 本来会等人批准、现在直接跑掉的动作,补一行 approvals。
            # 语义 = **已执行并通知**(approver_kind=automated_policy)。
            # 判据在 skill_adapter 打的标(``auto_approved``),与中间件和 executor
            # 共用 ``core.auto_approval.would_have_asked`` —— 这里只消费,不重判。
            #
            # 写在 begin() 之后而不是之前:approvals.action_id 有外键指向 actions,
            # 而这条路上没有 DB 层的 requires_approval 策略行(policies 表默认不种,
            # 见 logging/v2/policy.py),所以不存在「先有鸡还是先有蛋」的触发器竞态。
            _auto_reason = payload.get("auto_approved")
            if aid and _auto_reason:
                from mast.core.auto_approval import record_auto_approval

                record_auto_approval(repos, aid, skill=skill,
                                     reason=str(_auto_reason), params=params)
            if payload.get("success"):
                repos.actions.succeed(aid, duration_ms=duration_ms)
                # File the artifacts this action really produced. The store held
                # 0 scan_files for a session that wrote 9 .sxm, so
                # mv_campaign_stats reported `scan_file_count: 0` and the only
                # surviving copy of those paths was a truncated chat transcript.
                arts = _artifact_paths(payload)
                _register_scan_files(repos, aid, arts)
                # …and copy them into the current sample's folder, so the
                # experiment folder is self-contained even though Nanonis saved
                # them wherever the operator configured it to (2026-07-28).
                # getattr: this function is also called unbound with a stub self
                # in tests, and archiving must never be what breaks the record.
                _submit = getattr(self, "_submit_ingest", None)
                if callable(_submit):
                    _submit(aid, arts, source="skill",
                            skill=str(payload.get("skill") or ""))
            else:
                err = (str(payload.get("error") or "")
                       or str(payload.get("summary") or "")
                       or "failed (no error text)")
                repos.actions.fail(aid, err[:2000], duration_ms=duration_ms)
            return aid
        except Exception:  # records must never break the skill
            _log_swallowed("v2_action_record", "V2 action record failed for %s", skill)
            return None


    def _turn_trace_recorder(self, payload: dict) -> None:
        """Injected into each sub-agent's RecorderMiddleware; records one
        agent_turn step (reasoning chain + tool_calls + token usage — the prime
        SFT signal) on the active trajectory. The step's tool_call_id links to the
        downstream tool_call/safety_gate/hitl steps of the FIRST tool call this
        turn emitted (the 贯通键). Fire-and-forget + fail-safe."""
        sink = getattr(self, "_trace_sink", None)
        traj = getattr(self, "_active_traj", None)
        if sink is None or traj is None:
            return
        try:
            tcs = payload.get("tool_calls") or []
            first_id = (tcs[0].get("id") if tcs and isinstance(tcs[0], dict) else None)
            sink.record_step(
                trajectory_id=traj, step_type="agent_turn",
                actor_kind="agent", agent_id=payload.get("agent_id"),
                model_id=payload.get("model_id"),
                tool_call_id=first_id,
                output={"reasoning": payload.get("reasoning"),
                        "content": payload.get("content"),
                        "tool_calls": tcs,
                        "usage": payload.get("usage"),
                        "finish_reason": payload.get("finish_reason")},
            )
        except Exception:
            logger.debug("turn trace recorder failed (swallowed)", exc_info=True)


    def _safety_trace_recorder(self, payload: dict) -> None:
        """Injected into SafetyGateMiddleware; records a safety_gate verdict
        (allow/block — positive/negative training samples) step on the active
        trajectory. Fire-and-forget + fail-safe."""
        # Stamp skill activity at the gate (≈ skill START) so the manual-activity
        # watcher suppresses attribution for the WHOLE duration of a long skill,
        # not just the moment it ends (the skill recorder stamps the end). ONLY
        # for ALLOWED skills — a BLOCKED skill never reaches hardware, so stamping
        # it would wrongly suppress the operator's genuine manual fixes in the
        # next few seconds (review: blocked skills must not gate manual capture).
        try:
            if payload.get("verdict") == "allow":
                import time as _t
                self._map_last_skill_ts = _t.monotonic()
        except Exception:  # noqa: BLE001
            pass
        sink = getattr(self, "_trace_sink", None)
        traj = getattr(self, "_active_traj", None)
        if sink is None or traj is None:
            return
        try:
            sink.record_step(
                trajectory_id=traj, step_type="safety_gate",
                actor_kind="safety", agent_id="IC",
                tool_call_id=payload.get("tool_call_id") or None,
                input={"skill": payload.get("skill"), "args": payload.get("args")},
                output={"verdict": payload.get("verdict"),
                        "reason": payload.get("reason")},
            )
        except Exception:
            logger.debug("safety trace recorder failed (swallowed)", exc_info=True)


