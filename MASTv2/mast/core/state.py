"""InstrumentState — port of v1 mast/core/state.py.

Port adjustment from v1:
  - Same ring buffer (20-sample) for sparklines kept for Phase 7 GUI re-use.
  - state.refresh() still publishes a HARDWARE_STATE event onto EventBus so
    the v2 GUI WebSocket subscriber keeps the Dashboard live until the
    BufferService producer wiring lands (see IC agent VisionProducer plan).
    Without this bridge the Dashboard freezes — a silent v1→v2 regression.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque

from mast.core.connection import ConnectionPool
from mast.core.types import HardwareState

logger = logging.getLogger(__name__)

_HISTORY_LEN = 20


def reply_scalar(return_value: Any, *, field: str = "") -> "float | None":
    """从一次 ``safe_call`` 的回包里取出**那一个数**;取不出返回 ``None``。

    回包是三段信封 ``(error_string, raw_bytes, body)``,数在 ``body`` 里。
    这个函数替掉散在各处的这一行:

        val = parsed[2][0] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else parsed

    它有**两个**毛病,而且五处一模一样地抄了五遍:

    1. ``body`` 是空表时 ``parsed[2][0]`` 抛 ``IndexError`` —— 断连瞬间的
       截断回包正是这个形状,于是技能以一句看不懂的 IndexError 失败;
    2. 形状判据不成立时走 ``else parsed``,把**整个回包**当成读数交出去。
       2026-08-13 那次锁机,这条兜底分支是最贴合的解释:抖动中收到一个两段
       的截断回包 ⇒ ``len(parsed) > 2`` 为假 ⇒ ``('', b'…')`` 被当成电流,
       一路裸写进状态缓存,再被环境传感器 ``float()`` 抛成「硬故障」。

    所以这里:body 取不到就是**取不到**,不编、不抛、不把信封当数。
    真正判断「这是不是一个数」的仍是 :func:`coerce_number` → ``scalar_float``。
    """
    body = return_value
    if isinstance(return_value, (list, tuple)) and len(return_value) > 2:
        body = return_value[2]
    if isinstance(body, (list, tuple)):
        if not body:
            logger.warning("%s 的回包 body 是空的 —— 没有读数可取(回包 %r)",
                           field or "读数", return_value)
            return None
        if len(body) == 1:
            body = body[0]
        # 多元素 body 原样交给 coerce_number 去拒 —— 在这里挑第 0 个,
        # 就是在双通道回包上悄悄选一路。
    return coerce_number(body, field=field)


def coerce_number(value: Any, *, field: str = "") -> "float | None":
    """状态缓存这一侧的「这能不能当成一个读数」—— 判断本身委托给规范实现。

    判据全在 :func:`mast.io.nanonis_files.scalar_float`:数(不含 ``bool``)→
    float;单元素序列递归解包;**多元素序列 / 字符串 / NaN / inf → None**。
    这里只加两样它不该管的东西:**字段名**和**一条 warning**。

    ## 为什么是委托而不是自己写一份

    第一版在这里自己实现了一遍判据 —— 那是这个函数的**第七份**拷贝
    (`scalar_float` 的 docstring 说它当初就是来替掉三份手写 ``_scalar`` 的)。
    而且那一版比规范实现少拒一样:``NaN``。``float(nan)`` 不抛,于是一个 NaN
    会被当成合法电流写进缓存,传感器报 ``status="ok"`` 值是 NaN ——
    又一次「不知道被当成一个数」。

    项目记忆里那句话是对的:**逐处打补丁只会制造第八份实现。**

    ## 返回 None 意味着什么

    「这个值我看不懂」。调用方该做的是**不写**(保留上一个真值),而不是写 0.0。
    """
    from mast.io.nanonis_files import scalar_float

    out = scalar_float(value)
    if out is None:
        logger.warning(
            "状态字段 %s 收到一个**不能当成读数**的值,已拒绝写入缓存: "
            "%r (type=%s)。Nanonis 的数值字段可能以 1-元组回来、也可能整个回包"
            "被兜底分支原样交出来 —— 值留在这里,便于认出是哪一种。",
            field or "?", value, type(value).__name__,
        )
    return out


class InstrumentState:
    """Cached snapshot of hardware state, refreshed via monitor connection."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._cache: HardwareState = HardwareState()
        self._history: dict[str, Deque[float]] = {
            "bias": deque(maxlen=_HISTORY_LEN),
            "current": deque(maxlen=_HISTORY_LEN),
            "z": deque(maxlen=_HISTORY_LEN),
        }

    def history(self, channel: str) -> list[float]:
        buf = self._history.get(channel)
        return list(buf) if buf is not None else []

    def refresh(self) -> HardwareState:
        """Read current state from Nanonis via monitor port. Returns updated HardwareState."""
        state = HardwareState()

        rec = self._pool.safe_call("Bias_Get", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                state.bias_v = float(parsed[0])

        _ZCTRL_STATUS = {1: "Off", 2: "On", 3: "Hold", 4: "SwitchingOff", 5: "SafeTip", 6: "Withdrawing"}
        rec = self._pool.safe_call("ZCtrl_StatusGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed and len(parsed) > 0:
                code = int(parsed[0])
                state.z_controller_on = (code == 2)
                state.z_controller_status = _ZCTRL_STATUS.get(code, f"Unknown({code})")

        # ACTIVE Z-controller identity. Multiple
        # Z controllers can be defined (e.g. "Current", "log Current", "df"); the
        # active one is the live feedback channel. Read it here so it rides in the
        # live-state block on every LLM call instead of only via GetZCtrlList.
        rec = self._pool.safe_call("ZCtrl_CtrlListGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            variables = self._extract_parsed(rec.return_value)
            names, active_idx = self._parse_ctrl_list(variables)
            if names:
                state.z_controller_names = names
                state.z_controller_index = active_idx
                if 0 <= active_idx < len(names):
                    state.z_controller_name = names[active_idx]

        rec = self._pool.safe_call("ZCtrl_SetpntGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                state.setpoint_a = float(parsed[0])

        rec = self._pool.safe_call("Current_Get", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                state.current_a = float(parsed[0])

        rec = self._pool.safe_call("ZCtrl_ZPosGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                state.z_pos_m = float(parsed[0])

        rec = self._pool.safe_call("FolMe_XYPosGet", 0, role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed and len(parsed) >= 2:
                state.x_pos_m = float(parsed[0])
                state.y_pos_m = float(parsed[1])

        z_high_limit = None
        rec = self._pool.safe_call("ZCtrl_LimitsGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed and len(parsed) >= 1:
                z_high_limit = float(parsed[0])

        if (state.z_controller_on is False
                and state.z_pos_m is not None
                and z_high_limit is not None):
            state.withdrawn = abs(state.z_pos_m - z_high_limit) < 1e-12
        else:
            state.withdrawn = False if state.z_controller_on else None

        rec = self._pool.safe_call("Scan_StatusGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                state.scan_running = bool(parsed[0]) if parsed else None

        # Lock-in modulation on/off. One extra round-trip per second, and it
        # buys the current monitor a context it cannot get any other way: the
        # operator switches modulation from the Nanonis panel, so tracking
        # MAST's own ConfigureLockIn calls would miss precisely the case that
        # matters. Left as None on a failed read — "we did not read it" must not
        # arrive downstream as "modulation is off", which is the reading that
        # would re-enable the very alert spam this exists to stop.
        rec = self._pool.safe_call("LockIn_ModOnOffGet", 1, role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed:
                try:
                    state.lockin_mod_on = bool(int(parsed[0]))
                except (TypeError, ValueError):
                    state.lockin_mod_on = None

        # Scan frame geometry — so the LLM never has to guess what unit
        # scale the instrument is in. Order: cx, cy, w, h, angle.
        rec = self._pool.safe_call("Scan_FrameGet", role="monitor")
        if not rec.error and rec.return_value is not None:
            parsed = self._extract_parsed(rec.return_value)
            if parsed and len(parsed) >= 5:
                try:
                    state.scan_center_x_m = float(parsed[0])
                    state.scan_center_y_m = float(parsed[1])
                    state.scan_width_m = float(parsed[2])
                    state.scan_height_m = float(parsed[3])
                    state.scan_angle_deg = float(parsed[4])
                except (TypeError, ValueError):
                    pass

        # A failed or ambiguous read is not new information. Preserve prior known
        # values for patchable fields when a refresh yields None; a definite
        # reading still supersedes the cache. This also preserves a successful
        # write-back until the next informative monitor refresh.
        prev = self._cache

        # Staleness: if NOT ONE of the core hardware reads landed, the monitor
        # link is down. The carry-forward below would then serve hours-old values
        # while HardwareState's default timestamp reads "now" — safety preconditions
        # and map records would treat stale data as live. Flag
        # it and preserve the previous good timestamp so the age stays honest.
        _CORE = ("bias_v", "z_controller_on", "current_a", "z_pos_m", "scan_running")
        fresh_reads = sum(getattr(state, f, None) is not None for f in _CORE)
        if fresh_reads == 0 and any(
                getattr(prev, f, None) is not None for f in _CORE):
            state.stale = True
            state.timestamp = getattr(prev, "timestamp", state.timestamp)
            logger.warning(
                "InstrumentState refresh read nothing (monitor link down?) — "
                "serving carried-forward values flagged stale")

        for _f in self._PATCHABLE_FIELDS:
            if getattr(state, _f, None) is None and getattr(prev, _f, None) is not None:
                setattr(state, _f, getattr(prev, _f))

        self._cache = state

        if state.bias_v is not None:
            self._history["bias"].append(state.bias_v)
        if state.current_a is not None:
            self._history["current"].append(state.current_a)
        if state.z_pos_m is not None:
            self._history["z"].append(state.z_pos_m)

        # Push real-time state update to UI via WebSocket (EventBus bridge —
        # replaced by BufferService once VisionProducer wiring lands).
        try:
            from mast.core.events import EventBus
            EventBus.get().publish_hardware_state(
                bias_v=state.bias_v,
                current_a=state.current_a,
                z_m=state.z_pos_m,
                z_controller_on=state.z_controller_on,
                z_controller_status=state.z_controller_status,
                setpoint_a=state.setpoint_a,
                scan_running=state.scan_running,
                withdrawn=state.withdrawn,
            )
        except Exception:
            pass

        return state

    def snapshot(self) -> HardwareState:
        return self._cache

    # State fields a skill's result data may carry — so a successful write-skill
    # can patch the cache the instant it returns, without waiting for the ~1s
    # background monitor refresh. This is what lets StartScan's precondition see
    # ``z_controller_on=True`` right after ``ZControllerOnOff(True)`` instead of a
    # stale unknown value between the write-back and the next monitor refresh.
    _PATCHABLE_FIELDS = frozenset({
        "bias_v", "current_a", "z_pos_m", "x_pos_m", "y_pos_m",
        "z_controller_on", "z_controller_status", "withdrawn", "scan_running",
        "setpoint_a", "scan_center_x_m", "scan_center_y_m", "scan_width_m",
        "scan_height_m", "scan_angle_deg",
        # Active Z-controller identity — carried forward across a failed refresh
        # so a momentary ZCtrl_CtrlListGet miss doesn't blank the active
        # controller out of the live-state block.
        "z_controller_name", "z_controller_index", "z_controller_names",
    })

    #: 这几个字段声明的类型是 ``float | None`` —— 写进去的必须是**一个数**。
    #:
    #: 为什么需要这道闸门:``apply_patch`` 从前是裸 ``setattr``,而喂它的是技能
    #: 结果里的 ``data`` 字典 —— 也就是说**每一个生产方都得记得自己强转**。
    #: 有一个没记得就够了:``GetCurrent`` 把 ``parsed[2][0]`` 原样放进 data,
    #: 而 Nanonis 的数值数组字段回来常是「一串 1-元组」,于是缓存里的
    #: ``current_a`` 变成 ``(1.2e-10,)``。
    #:
    #: 后果不是「读出来的数不好看」:环境传感器对它做 ``float()`` 抛 TypeError
    #: ⇒ 状态变 ``error`` ⇒ 2026-08-13 之前那条路会把它当硬故障 ⇒ **退针 + 挂上
    #: 一个解不开的急停闩**。一次装错类型的缓存写入,锁死了整台机器。
    #:
    #: 所以校验放在**收的这一侧**,而不是指望十几个生产方各自记得
    #: —— 会犯这个错的那一方,不能同时当校验方。
    _NUMERIC_FIELDS = frozenset({
        "bias_v", "current_a", "z_pos_m", "x_pos_m", "y_pos_m", "setpoint_a",
        "scan_center_x_m", "scan_center_y_m", "scan_width_m", "scan_height_m",
        "scan_angle_deg",
    })

    # 缓存通过后台刷新和 apply_patch 更新。
    # 工具调用与 composite 子步骤共用 patch_state_from_result，使下一步读取最新已确认状态。
    def apply_patch(self, **fields) -> None:
        """Patch known state fields onto the cached snapshot IN PLACE (e.g. after a
        write-skill returns the value it just set). Only whitelisted keys are
        applied; ``None`` values are skipped (don't clobber a known value with
        unknown), but ``False`` IS applied (e.g. scan_running=False after StopScan).
        Never reads hardware, never raises."""
        cache = self._cache
        for k, v in fields.items():
            if v is None or k not in self._PATCHABLE_FIELDS:
                continue
            if k in self._NUMERIC_FIELDS:
                v = coerce_number(v, field=k)
                if v is None:
                    continue  # 写不进去就不写:陈的真值好过一个假形状
            try:
                setattr(cache, k, v)
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------

    def start_background_refresh(self, interval_s: float = 1.0) -> None:
        """Spawn a daemon thread that calls ``refresh()`` every
        ``interval_s`` seconds. The thread fully owns the blocking
        ``safe_call`` invocations so callers in the GUI main thread can
        rely on ``snapshot()`` returning a recent (<= interval_s seconds
        old) value without any TCP round-trip.

        Idempotent: a second call is a no-op.
        """
        import threading
        if getattr(self, "_bg_thread", None) is not None and self._bg_thread.is_alive():
            return
        self._bg_stop = threading.Event()

        def _loop() -> None:
            import time as _t
            while not self._bg_stop.is_set():
                try:
                    self.refresh()
                except Exception as exc:
                    logger.debug("state background refresh error: %s", exc)
                # Sleep in small slices so stop() returns quickly.
                end = _t.monotonic() + max(0.1, float(interval_s))
                while _t.monotonic() < end and not self._bg_stop.is_set():
                    _t.sleep(0.1)
            logger.info("InstrumentState background refresh stopped")

        self._bg_thread = threading.Thread(
            target=_loop, name="InstrumentStateRefresh", daemon=True,
        )
        self._bg_thread.start()
        logger.info("InstrumentState background refresh started (interval=%.1fs)", interval_s)

    def stop_background_refresh(self) -> None:
        ev = getattr(self, "_bg_stop", None)
        if ev is not None:
            ev.set()

    def get_bias(self) -> float | None:
        return self._cache.bias_v

    def get_current(self) -> float | None:
        return self._cache.current_a

    def is_z_controller_on(self) -> bool | None:
        return self._cache.z_controller_on

    @staticmethod
    def _extract_parsed(ret) -> list | None:
        if isinstance(ret, (list, tuple)) and len(ret) >= 3:
            return ret[2]
        return None

    @staticmethod
    def _parse_ctrl_list(variables) -> tuple[list[str], int]:
        """Pull (controller_names, active_index) out of ZCtrl_CtrlListGet's
        Variables payload. ZCtrl.CtrlListGet ResponseTypes = ["i","i","*+c","i"]
        → Variables == [list_size, num_controllers, [name0, name1, ...],
        active_index]. Mirrors GetZCtrlList.execute()'s defensive scan: the
        names are the first list-of-strings; the active index is the int that
        appears AFTER the names (the two leading ints precede the names, so they
        are ignored). Returns ([], 0) on any shape we don't recognise."""
        names: list[str] = []
        active_index = 0
        if isinstance(variables, (list, tuple)):
            for item in variables:
                if isinstance(item, (list, tuple)) and item and all(
                    isinstance(s, str) for s in item
                ):
                    names = list(item)
                elif isinstance(item, int) and names:
                    active_index = item
        return names, active_index


# ── 写回:一个成功的写技能立刻更新缓存 ────────────────────────────────────────

def patch_state_from_result(state: Any, data: Any, *, what: str = "") -> None:
    """把成功技能的结果字段写回状态缓存，供工具边界和 composite 子步骤共用。
    
    子步骤不经过工具 post hook；若只在那里回写，后续步骤会读取陈旧状态，
    即使上一动作已成功也可能错误地拒绝。尽力而为，缓存更新失败不改变技能结果。"""
    if not isinstance(data, dict) or not data:
        return
    patch = getattr(state, "apply_patch", None)
    if not callable(patch):
        return
    try:
        patch(**data)
    except Exception:  # noqa: BLE001 — 缓存写回永远不许影响技能结果
        logger.debug("state apply_patch after %s failed (swallowed)", what or "skill")


#: 技能用来声明「这个硬件事实我**回读验证过**」的键。见 :func:`patch_verified_state`。
VERIFIED_STATE_KEY = "_verified_state"


def patch_verified_state(state: Any, data: Any, *, what: str = "") -> None:
    """把显式声明为已回读验证的事实写回缓存，成功与失败结果都适用。
    
    失败技能的普通 data 可能含目标值，不能整体写回；只消费 VERIFIED_STATE_KEY
    下的显式声明。预扫描早停即使返回失败，也能声明已经确认 scan_running=False，
    使后续前置条件读取正确事实。未声明的意图值不写回；更新失败不改变技能结果。"""
    if not isinstance(data, dict):
        return
    verified = data.get(VERIFIED_STATE_KEY)
    if not isinstance(verified, dict) or not verified:
        return
    patch = getattr(state, "apply_patch", None)
    if not callable(patch):
        return
    try:
        patch(**verified)
    except Exception:  # noqa: BLE001 — 同上,缓存一致性不许影响执行正确性
        logger.debug("verified-state patch after %s failed (swallowed)",
                     what or "skill")
