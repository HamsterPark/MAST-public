"""ExecutionContext — passed to v2 skill.execute() invocations.

v2 simplification of v1's ExecutionContext (mast/core/executor.py:260):
  - Drops `executor` reference (no SkillExecutor in v2; SafetyGateMiddleware
    handles safety wrapping at the LangChain agent layer instead).
  - Adds `state.snapshot()` access via `state` attribute (so skills'
    check_preconditions can read live HardwareState).
  - `run(skill_name, params)` recursively executes a sub-skill via the
    SkillRegistry, used by Composite skills (Phase 4 Session 4).

Construction is the responsibility of the agent's tool wrapper:
    ctx_provider = lambda: ExecutionContext(pool, state, registry)
    tools = [wrap_skill(skill, ctx_provider) for skill in registry.list_skills()]

For unit tests without hardware, use FakeCtx (see
tests/v2/unit/test_wrap_skill_minimal.py).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence  # noqa: F401 — used in a string annotation
from typing import TYPE_CHECKING, Any

from mast.core.types import NanonisCallRecord, SkillResult

if TYPE_CHECKING:
    from mast.core.connection import ConnectionPool
    from mast.core.registry import SkillRegistry
    from mast.core.state import InstrumentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Post-abort hardware policy (2026-07-11)
# ─────────────────────────────────────────────────────────────────────
# Once the operator aborts, the instrument must STOP — but a stop is itself a
# sequence of hardware commands, so the gate cannot simply block everything or
# the tip would be stranded mid-approach with feedback off. The rule:
#
#   read anything · stop anything · start nothing
#
# Explicit allow-list rather than a name heuristic: "Set" appears in both
# ``ZCtrl_Withdraw`` (safe) and ``Bias_Set`` (very much not), and getting this
# wrong in either direction is a hardware-safety bug. Any Nanonis verb not
# listed here is refused while an abort is latched.

# Reads are side-effect-free — and a stop sequence has to read status.
#
# Nanonis names every read with a ``Get`` suffix (or ``*Status`` / ``Read*``), so
# this substring test agrees EXACTLY with that convention over the real 684-verb
# API — 296 reads either way, no verb classified differently. It is nevertheless
# a substring test on an API we do not own: a future verb containing "get" as an
# accident of spelling (``…TargetSet`` — "tar-GET") would be waved through as a
# read and could then WRITE after an abort. test_abort_policy.py pins heuristic
# == convention against the live nanonis_spm API so such a verb fails a test
# instead of moving the tip.
def _is_read(method: str) -> bool:
    m = method.lower()
    return "get" in m or m.endswith("status") or "read" in m


# Writes that are STOPS. Several Nanonis verbs are OVERLOADED — the same method
# starts or stops depending on an argument — so an entry may pin the argument
# index and the set of values that mean "stop". ``None`` = unconditionally safe.
#
# Getting an index wrong here is a hardware-safety bug in the worst direction:
# Scan_Action(action, direction) takes action 0=START / 1=STOP / 2=PAUSE, so a
# rule that looked at the *direction* argument would have happily let a
# post-abort Scan_Action(0, …) START a fresh scan. Each rule is therefore keyed
# to the verb's own signature, and anything unlisted is refused.
#
# Every entry must be a REAL Nanonis verb that some MAST skill actually calls —
# pinned by test_abort_policy.py. The first draft of this table listed
# ``BiasSpectrMLS_Stop`` and ``GenSweep_Stop``, NEITHER of which exists (the MLS
# mode is stopped by BiasSpectr_Stop; the generic sweep verb is GenSwp_Stop), so
# two stop paths I believed were open were in fact closed — and the skills' OWN
# abort-cleanup handlers (pattern.py's _stop_pattern_experiment, folme's
# StopFolMe) were being refused by this very gate, leaving the grid experiment
# running on the controller after the operator aborted it.
_ABORT_SAFE_WRITES: dict[str, "tuple[int, frozenset[int]] | None"] = {
    # ── unconditional stops / retracts ──
    "ZCtrl_Withdraw": None,           # retract the tip — always the safe direction
    "Motor_StopMove": None,           # halt the coarse motor
    "FolMe_Stop": None,               # halt the follow-me XY tip motion
    "BiasSpectr_Stop": None,          # also stops an MLS (multi-line-segment) sweep
    "ZSpectr_Stop": None,
    "GenSwp_Stop": None,
    "PLLFreqSwp_Stop": None,          # (Modulator_index) — an index, not a mode flag
    "Pattern_ExpStop": None,          # the grid/pattern experiment
    # Waveform generators (2026-07-13). A generator left running keeps driving an
    # output line — the exact thing an operator pressing 中止 wants to end, and the
    # one MAST cannot see the far end of.
    "FunGen1Ch_Stop": None,
    "FunGen2Ch_Stop": None,
    # Loggers. Stopping a recording is never the dangerous direction.
    "DataLog_Stop": None,
    "TCPLog_Stop": None,
    # THE most important entry in this table (2026-07-13).
    #
    # A deployed Nanonis script runs on the REAL-TIME CONTROLLER. It issues no
    # safe_call — so this gate, the SafetyGate, the mode gate and HITL are all
    # blind to it, and the global bias/current/Z bounds do not apply inside it.
    # Every other entry here stops something that MAST itself started over TCP;
    # this one stops something MAST cannot otherwise touch AT ALL.
    #
    # Everywhere else in MAST, "we stopped sending commands" is close enough to
    # "the instrument stopped". Here it is not. If Script_Stop were refused while
    # an abort is latched, the operator would press 中止 and the script would keep
    # driving the tip — the abort doing the exact opposite of its job.
    "Script_Stop": None,
    # ── overloaded verbs: safe ONLY in their stop form ──
    # AutoApproach_OnOffSet(on_off): 0 = OFF (stop), 1 = ON (would RESTART).
    "AutoApproach_OnOffSet": (0, frozenset({0})),
    # Scan_Action(action, direction): action 0 = START, 1 = STOP, 2 = PAUSE.
    "Scan_Action": (0, frozenset({1, 2})),
    # Pattern_ExpPause(Pause_Resume): 1 = PAUSE, 0 = RESUME (pattern.py falls back
    # to this when Pattern_ExpStop is unavailable).
    "Pattern_ExpPause": (0, frozenset({1})),
    # AtomTrack_CtrlSet(AT_control, Status): Status 0 = off. Atom tracking actively
    # DRIVES the tip, so switching it off is a motion stop; Status 1 would (re)start
    # the tracker.
    "AtomTrack_CtrlSet": (1, frozenset({0})),

    # ── optional-hardware stops (设置 → 硬件模块; every module ships OFF) ──
    # These live here, BELOW the skill layer, on purpose. The module gate hides a
    # disabled module's skills from the agent — it does not and must not decide
    # what an abort may still stop. If the operator has a multi-probe system and
    # presses 中止, the probes must retract; a gate that refused the retract
    # because the abort flag is set would be the worst possible failure.
    "HSSwp_Stop": None,             # high-speed sweeper
    "APRFGen_SwpStop": None,        # RF freq/power/list sweep
    "PLLPhasSwp_Stop": None,        # PLL phase sweep
    "MProbeScanner_Stop": None,     # per-probe scanner motion
    # MProbeZCtrl_Withdraw(Scanner_Index): the multi-probe analogue of
    # ZCtrl_Withdraw — retract probe N. Same reasoning, same unconditional allow.
    "MProbeZCtrl_Withdraw": None,
    # APRFGen_RFOutOnOffSet(RF_Output): 0 = output OFF. An RF output is an output,
    # not a feedback loop — killing it is unambiguously a stop.
    "APRFGen_RFOutOnOffSet": (0, frozenset({0})),
    # Laser_OnOffSet(Status): 0 = laser OFF. Unlike a user output (whose wiring
    # MAST cannot know — a shutter wired normally-closed OPENS at 0), a laser's
    # "off" IS the low-energy state. Off is allowed after an abort; on never is.
    "Laser_OnOffSet": (0, frozenset({0})),
    # NB deliberately ABSENT: MProbeZCtrl_OnOffSet / KelvinCtrl_CtrlOnOffSet /
    # PICtrl_OnOffSet / Interf_CtrlOnOffSet. Switching a FEEDBACK LOOP off is not a
    # stop — it parks whatever the loop was holding with nothing holding it. That
    # is why the main ZCtrl_OnOffSet is not on this list either: the safe
    # post-abort action for a Z loop is Withdraw, not "off".
}


def _is_abort_safe(method: str, args: "tuple[Any, ...]") -> bool:
    """True if this Nanonis call may still run while an abort is latched."""
    if _is_read(method):
        return True
    if method not in _ABORT_SAFE_WRITES:
        return False
    rule = _ABORT_SAFE_WRITES[method]
    if rule is None:
        return True
    idx, stop_values = rule
    try:
        # An overloaded verb is safe ONLY in its stop form — the identical call
        # with the start value would resume the very motion we are aborting.
        return int(args[idx]) in stop_values
    except (IndexError, TypeError, ValueError):
        return False   # can't prove it's a stop → refuse


#: 挂在 abort ``threading.Event`` 上的「是谁停的」。
#:
#: 为什么挂在事件对象上而不是放进某个中央字典:停的来源分散在四个模块里
#: (急停按钮 / E_STOP 事件钩子 / 环境告警 / 每回合的会话停止),而**读**它的
#: 只有下游那一句拒绝语。让原因跟着事件走,加一个新来源就只用多写一行,不必
#: 记得再去登记表里补一笔 —— 「记得去另一处补登记」这件事本仓已经漏过多次。
ABORT_REASON_ATTR = "mast_abort_reason"


def mark_abort(event, reason: str) -> None:
    """``set()`` 一个 abort 事件,并**留下是谁停的**。

    直接 ``event.set()`` 也能停,但停下来之后没人知道为什么 —— 而下游那句
    拒绝语会替它编一个(从前一律说是用户干的)。用这个函数代替裸 ``set()``。

    原因写不进去不算失败:该停还是停。停不下来才是失败。
    """
    try:
        setattr(event, ABORT_REASON_ATTR, str(reason)[:200])
    except Exception:  # noqa: BLE001 — 记不下原因绝不能挡住「停」这件事
        pass
    event.set()


class ExecutionContext:
    """Provides safe_call + run + check_abort to v1-style skill.execute()."""

    def __init__(
        self,
        pool: "ConnectionPool",
        state: "InstrumentState",
        registry: "SkillRegistry",
        abort_event: "threading.Event | Sequence[threading.Event] | None" = None,
        approval_source: str = "auto",
        run_id: str = "",
        owner: str = "",
    ):
        # WHO is driving the instrument through this context — group-chat run,
        # private chat, signals API. Surfaced in the InstrumentBusy message so a
        # refused caller is told which of the three has the instrument, instead
        # of just "busy" (审计 致命一).
        self.owner = str(owner or "未知入口")
        self.pool = pool
        self.state = state
        self._registry = registry
        # Set by whoever already passed the sample gate (skill_adapter /
        # executor), so a composite's sub-steps are not re-gated mid-flight.
        # See the SAMPLE GATE block in run().
        self._scope_admitted: bool = False
        # ABORT is a UNION, not a single event (2026-07-11 P0).
        #
        # There is more than one legitimate "stop" in this system and they are
        # DIFFERENT threading.Events: the orchestrator/E-STOP event
        # (``_orch_abort``), and the per-conversation stop of a private chat /
        # voice session. The private-chat context used to be built with
        # ``abort_event=session_event or _orch_abort`` — i.e. the session event
        # WON — so a composite running under the main chat polled only the chat's
        # own event. E_STOP sets ``_orch_abort``, so **pressing the emergency stop
        # could not stop a composite running in the private chat**: check_abort()
        # returned False and every gate below it (skill_adapter / run / safe_call)
        # therefore let the hardware keep going.
        #
        # Taking the union fixes it in the right direction: a context aborts when
        # ANY of its stop sources fires. E-STOP is threaded into every context, so
        # it always wins; a chat's own Stop still only stops that chat.
        if abort_event is None:
            events: list[threading.Event] = []
        elif isinstance(abort_event, threading.Event):
            events = [abort_event]
        else:
            events = [e for e in abort_event if e is not None]
        self._aborts: list[threading.Event] = events or [threading.Event()]
        self._approval_source = approval_source
        # Identifies the CURRENT orchestrator run. Composite step-progress
        # sidecars are keyed by it, so a finished run's progress can never be
        # resumed by a later one (that cross-run reuse is what made a completed
        # AutoApproach short-circuit every subsequent 进针 into a fake success).
        self.run_id = str(run_id or "")

    def narrate(self, kind: str, **data) -> None:
        """向**用户**发一条旁白（"我们要打一发 10 V / 500 ms 的脉冲"）。

        转发到 ``mast.chat.narration.narrate``。放一个方法在这里纯粹是为了让技能
        作者手边就有（和 ``ctx.run`` / ``ctx.safe_call`` 一致的手感）——
        它**不是唯一入口**，也不该是：``GraphExecutor`` 和 ``ScanVisionMonitor``
        手里都不一定有 ctx，而那两个地方才是旁白的主要发点。

        会话 id 由 ``narrate`` 自己从 ``turn_context`` 取，不经过 ctx，所以这个
        方法不持有任何会话状态。永不抛、永不阻塞、没接上就是 no-op。

        ⚠️ 没有 ``text=`` 参数，这是**故意的**：句子在 ``chat/narration_templates``
        里拼，数字从技能真实的 params 里按 key 取。技能没有地方可以写一个
        编出来的数字。
        """
        try:
            from mast.chat import narration

            narration.narrate(kind, **data)
        except Exception:  # noqa: BLE001 — 一条旁白绝不许弄坏一次技能执行
            pass

    def safe_call(self, method_name: str, *args, role: str = "main",
                  allow_on_abort: bool = False,
                  recv_timeout_s: float | None = None) -> NanonisCallRecord:
        """Direct Nanonis call through the connection pool.

        ``recv_timeout_s`` 透传给 :meth:`ConnectionPool.safe_call`，只给
        **在仪器端阻塞很久才回话**的命令用（``BiasSpectr_Start`` 之类）。

        ABORT GATE (2026-07-11). This is the ONE choke point every hardware
        command flows through, and it is the deepest place a stop can bite. An
        abort used to be honoured only BETWEEN LangGraph super-steps, so a skill
        already executing kept driving the instrument — pressing 中止 during a
        scan/approach did nothing until the whole step returned, which is exactly
        why the operator experienced "中止后控制不了了".

        Post-abort policy:
          * READS always pass (a *Get* is side-effect-free, and a stop sequence
            needs to read status).
          * The STOP/RETRACT verbs always pass — they are HOW a run stops safely;
            blocking them would strand the tip.
          * Everything else (writes, motion, starts) is REFUSED with a clear
            error instead of touching the instrument.
          * ``allow_on_abort=True`` is the explicit escape hatch for cleanup code
            that must run *because* of the abort (e.g. AutoApproach stopping its
            own module).
        """
        if (not allow_on_abort) and self.check_abort() \
                and not _is_abort_safe(method_name, args):
            logger.warning("abort active — refusing Nanonis write %s%s",
                           method_name, args if args else "")
            try:
                from mast.core.diagnostics import record

                record("abort_block", method_name,
                       "用户已中止——仪器写命令被拒（只放行读取与停止/退针）",
                       args=list(args), run_id=self.run_id)
            except Exception:  # noqa: BLE001
                pass
            return NanonisCallRecord(
                method=method_name, args=args,
                error=("aborted by operator — instrument writes are blocked. "
                       "Only reads and stop/retract commands are allowed now."),
            )
        # **只在真要用的时候才往下传。** 无条件传 ``recv_timeout_s=None`` 会让
        # 每一个 duck-typed 的 pool（测试替身、别处的适配器）都必须跟着加这个
        # 关键字，哪怕那次调用根本不需要预算 —— 2026-09-09 就是这么一口气弄红
        # 了 5 个 abort-gate 测试的。默认路径保持与改动前逐字节相同；
        # 显式要预算的调用方若碰上不支持的 pool，则**应该**当场炸出来。
        if recv_timeout_s is None:
            return self.pool.safe_call(method_name, *args, role=role)
        return self.pool.safe_call(method_name, *args, role=role,
                                   recv_timeout_s=recv_timeout_s)

    def _safety_guard(self):
        """Lazily-built SafetyGuard (admin-merged limits), cached per context."""
        g = getattr(self, "_sg", None)
        if g is None:
            from mast.config import SafetyLimits
            from mast.core.safety import SafetyGuard
            g = SafetyGuard(SafetyLimits())
            self._sg = g
        return g

    def run(self, skill_name: str, params: dict,
            version: str | None = None) -> SkillResult:
        """Recursively execute a sub-skill. Used by composite skills.

        Runs the global numeric SafetyGate bounds (same caps the agent-layer
        middleware enforces) PLUS validate_params + check_preconditions before
        execute. The middleware only checks the COMPOSITE's top-level params, so
        without the bounds check here a composite's per-step params (e.g.
        GridSTS's per-point MoveToXY, or a spec composite's computed bias) could
        reach hardware out-of-range.

        Also enforces the approval gate at this leaf call point: the agent-layer
        SafetyGateMiddleware sees only the COMPOSITE's top-level tool call and
        SkillExecutor guards only the manual path, so a composite SUB-step
        reaches hardware through here alone .

        Scope, precisely: this gate promises only that HUMAN-level approval
        (DANGEROUS skills, and the open-loop coarse Z approach) never flows
        down implicitly. CONFIRM-level sub-steps are NOT re-gated here — on
        the v2 agent path CONFIRM ≡ auto-run (leaf CONFIRM skills run ungated
        there too), unlike executor.run which also rejects CONFIRM under an
        'auto' source. The coarse-approach + human-level check itself is kept
        aligned with safety_mw.py and executor.py.
        """
        # ── ABORT GATE (2026-07-11) ──────────────────────────────────────
        # A composite that is MID-FLIGHT when the operator aborts must not keep
        # firing its remaining sub-steps. GraphExecutor checks between plan
        # steps, but this is the choke point EVERY sub-skill call passes through
        # (spec-interpreter composites, hand-rolled run_composite overrides, and
        # any skill that calls ctx.run directly), so the guarantee belongs here.
        if self.check_abort():
            logger.info("abort active — refusing sub-skill '%s'", skill_name)
            return SkillResult(
                skill_name=skill_name, success=False,
                error=("aborted by operator — refusing to start sub-skill "
                       f"'{skill_name}'. Stop and report; do not retry."),
            )

        try:
            # P4: version=None 保持既有"取最新 semver"；钉住的版本不存在时
            # fail-closed（registry.get raises KeyError）。
            skill_cls = self._registry.get(skill_name, version)
        except KeyError as e:
            return SkillResult(skill_name=skill_name, success=False, error=str(e))

        skill = skill_cls()

        # ── APPROVAL GATE (fail-closed) ──────────────────────────────────
        from mast.core.safety import (
            is_calibration_change,
            is_coarse_drive_change,
            is_coarse_sample_approach,
            is_protection_disable,
            is_unguarded_lateral_coarse_move,
            mode_refusal,
        )
        try:
            # Admin-override aware, same metadata the executor path gates on.
            meta = self._registry._get_metadata(skill_cls)
        except Exception as exc:
            return SkillResult(
                skill_name=skill_name, success=False,
                error=f"metadata unreadable, refusing to execute sub-skill: {exc}",
            )
        # ── SAMPLE GATE (2026-07-28, defence in depth) ───────────────────
        # The top-level entry (skill_adapter / executor) has already gated the
        # composite itself. Re-gating every sub-step would only matter if the
        # scope changed MID-composite — and halting a composite halfway through
        # is worse than never starting it (the tip is already down, half the
        # grid is measured). So: if the caller was admitted, sub-steps inherit
        # that admission; the gate here only catches sub-steps of a composite
        # that itself never passed a gate (a direct ctx.run from a bare skill).
        if not self._scope_admitted:
            try:
                from mast.core.sample_gate import check_sample_scope
                from mast.logging.experiment_log import get_active_log
                gate_msg = check_sample_scope(meta, skill_name, get_active_log())
            except Exception:  # noqa: BLE001 — a broken gate must not block work
                gate_msg = None
            if gate_msg:
                logger.info("sample gate — refusing sub-skill '%s'", skill_name)
                return SkillResult(skill_name=skill_name, success=False, error=gate_msg)

        # ── 两种「要人」分开 ──────────────────────────────────────────────
        # 这一段原来把五个**参数条件硬闸**和一个**基线 safety_level**混成同一个
        # ``required = "human"``,然后一律拒绝并让人「去审批面板」。审批面板对
        # DANGEROUS 技能已经不再产生任何东西(整条确认框链路割掉了),所以那条拒绝
        # 语从这一刻起指向一扇不存在的门 —— 而 agent 会照着它反复重试。
        #
        # 分开之后:
        #   * **硬闸**(下面五条)照旧拒绝。它们不弹框、不等人,是拒绝型防护 ——
        #     这些参数边界保留独立拒绝语义；显式手动路径由其自己的授权入口处理。
        #     GUI / manual executor 使用 human approval_source。
        #   * **基线 DANGEROUS**(``safety_level`` 派生)转成「执行 + 留痕 + 通知」,
        #     与 agent 顶层路径(``AutoApprovalNoticeMiddleware``)口径一致 ——
        #     同一个技能不该因为「是不是被复合技能调用的」而有两种安全语义。
        # ── OPERATING-MODE GATE (2026-08-27) ─────────────────────────────
        # SAFE's contract ("the tip is fine — do NOT repair it",
        # :class:`mast.core.types.OperatingMode`) used to be enforced at the
        # AGENT's door only: ``safety_mw._mode_block`` sees the agent's own tool
        # calls, and ``skill_forge_tools`` re-implemented the same judgement for
        # ``run_composite``. Everything that reaches hardware through THIS
        # method walked straight past both — every composite sub-step, every
        # conduct step (``conduct/adapters.py`` → ``ctx.run``) and every
        # ``POST /api/skills/{name}/execute``. A conduct template containing
        # TipPulse really did pulse in SAFE, and it looked compliant doing it.
        #
        # ``!= "human"`` mirrors the hard gate just below: the manual GUI /
        # executor path IS the human authorisation, and the operating mode is
        # about what the system does **on its own**.
        #
        # Placed AFTER the metadata read (an unreadable metadata is already its
        # own refusal above — answering it here too would relabel an honest
        # error as a mode decision) and BEFORE the hard gates, which are
        # unconditional physical protections and must not be reachable only via
        # a mode-dependent branch.
        #
        # Unknown mode allows — see ``mode_refusal``. A process with no bound
        # mode source (tests, headless ``pipeline.main``, offline tools) behaves
        # byte-for-byte as it did before this block existed.
        if self._approval_source != "human":
            try:
                from mast.core.operating_mode import current_operating_mode

                _mode = current_operating_mode()
            except Exception:  # noqa: BLE001 — a broken mode read never blocks work
                logger.debug("operating-mode unreadable (allowing)", exc_info=True)
                _mode = None
            _refusal = mode_refusal(skill_name, meta, params, _mode)
            if _refusal:
                logger.info(
                    "Sub-skill '%s' refused by operating mode (approval_source=%s)",
                    skill_name, self._approval_source,
                )
                return SkillResult(
                    skill_name=skill_name, success=False,
                    error=f"[{skill_name}] {_refusal}",
                )

        hard_gate: str | None = None
        if is_coarse_sample_approach(skill_name, params):
            # 唯一一个物理上会毁东西的动作:开环粗动 Z 朝样品走(pan 型步进,
            # 没有反馈停止)。与技能自己的 safety_level 无关。
            hard_gate = "开环粗动 Z 向样品进针（没有反馈能让它停下来）"
        elif is_protection_disable(skill_name, params):
            # 关掉硬件保护(SafeTip / Z 软限位)= 拆掉撞针安全网,工作流不许悄悄干
            # (2026-07-03 review)。
            hard_gate = "关闭硬件保护（SafeTip / Z 软限位）"
        elif is_calibration_change(skill_name, params):
            # 改写 bias/current 标定系数会一次性废掉下游每一条电压/电流边界。
            hard_gate = "改写全局 bias/current 标定系数"
        elif is_coarse_drive_change(skill_name, params):
            # 粗动步进的驱动电压:设错不可恢复(有的控制器给 400 V,有的叠堆 300 V
            # 就烧),而没有任何读数能告诉你这是哪一台(2026-07-31)。
            hard_gate = "设置粗动马达驱动电压/频率"
        elif is_unguarded_lateral_coarse_move(skill_name, params):
            # 工作流直接去调裸的横向马达命令 = 绕开了有防护的换位路径。
            # RelocateCoarseXY 走 safe_call、不经过这里,所以只会逮到裸移动。
            hard_gate = "裸横向粗动（未走 RelocateCoarseXY 的防护路径）"

        if hard_gate is not None and self._approval_source != "human":
            logger.info(
                "Sub-skill '%s' refused inside composite: %s (approval_source=%s)",
                skill_name, hard_gate, self._approval_source,
            )
            return SkillResult(
                skill_name=skill_name, success=False,
                error=(
                    f"硬闸拒绝：'{skill_name}' 属于「{hard_gate}」，"
                    "自主路径上一律不执行——这不是等待审批，没有任何批准会到来，"
                    "换参数也绕不过去。STOP retrying。"
                    "确实需要时，请用户在 GUI 手动执行（手动路径本身就是人工授权）。"
                ),
            )

        if (self._safety_guard().requires_approval(meta) == "human"
                and self._approval_source != "human"):
            # 基线 DANGEROUS。此前在这里被拒;现在照跑,并留一行台账。
            try:
                from mast.core.auto_approval import notify, would_have_asked

                _reason = would_have_asked(meta, tool_name=skill_name, args=params)
                notify(skill_name,
                       f"{_reason or 'DANGEROUS 技能'} —— 复合技能子步骤，"
                       "已直接执行并通知，不再等待人工批准",
                       args=params, composite_substep=True,
                       approval_source=self._approval_source)
            except Exception:  # noqa: BLE001 — 通知坏了绝不能让子步骤失败
                logger.debug("auto_approval 通知失败(已忽略)", exc_info=True)

        # GLOBAL SAFETY BOUNDS — applies to every sub-skill call, using the
        # skill's ParameterSpec units + admin-merged SafetyLimits.
        try:
            violations = self._safety_guard().check_parameter_bounds(meta, params)
            if violations:
                return SkillResult(
                    skill_name=skill_name, success=False,
                    error="global safety bounds: " + "; ".join(violations),
                )
        except Exception as exc:  # pragma: no cover - bounds check is best-effort defence
            logger.debug("global bounds check skipped for %s: %s", skill_name, exc)
        # v1-style validate
        try:
            violations = skill.validate_params(params) if hasattr(skill, "validate_params") else []
        except Exception as e:
            return SkillResult(skill_name=skill_name, success=False, error=f"validate_params: {e}")
        if violations:
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error="; ".join(violations),
            )
        # Optional precondition check (against current cached state)
        try:
            current_state = self.state.snapshot() if self.state is not None else None
        except Exception:
            current_state = None
        if current_state is not None and hasattr(skill, "check_preconditions"):
            try:
                unmet = skill.check_preconditions(current_state)
                if unmet:
                    # 不满足就**先去问一次硬件再判一次**。
                    #
                    # ``core/executor.py``(手动/GUI 路径)从 v1 起就这么做,理由写在
                    # 那里:「Nanonis hardware state propagation can lag behind TCP
                    # replies」。**这条路径(agent + 所有 composite 子步骤)一直没有**,
                    # 而它恰恰是缓存最容易过期的那条 —— 子步骤刚把硬件改完,
                    # 1 Hz 的后台 refresh 还没轮到。
                    # 两条路径对同一件事一条做一条不做,本身就该有测试钉着。
                    unmet = self._recheck_after_refresh(skill, unmet)
                if unmet:
                    return SkillResult(
                        skill_name=skill_name,
                        success=False,
                        error="; ".join(unmet),
                    )
            except Exception as e:
                logger.debug("check_preconditions raised on %s: %s", skill_name, e)

        # ── INSTRUMENT ARBITRATION (2026-07-28) ──────────────────────────
        # One physical instrument, several entry points that never knew about
        # each other (the current inventory lives in instrument_lock's
        # docstring — a mechanism list, not a count: the count kept drifting). Re-entrant, so a composite holding the token does not
        # deadlock on its own sub-steps; skipped for reads and for the
        # stop/retract remedies (see instrument_lock.needs_token).
        from mast.core.instrument_lock import InstrumentBusy, hold_for_skill

        try:
            with hold_for_skill(meta, skill_name, self.owner):
                result = skill.execute(self, params)

            # 与工具边界共用缓存写回实现，使 composite 后续子步骤读取刚完成的硬件状态。
            if getattr(result, "success", False):
                from mast.core.state import patch_state_from_result

                patch_state_from_result(self.state, getattr(result, "data", None),
                                        what=skill_name)
            # composite 子步骤不经过 agent 工具边界，必须接入相同的扫描地图记录器。
            # 成功和失败均需记录：失败的脉冲也可能改变表面，不能继续把落点当成干净区域。
            _sink = getattr(self, "marker_sink", None)
            if callable(_sink):
                try:
                    _sink({"skill": skill_name, "params": params,
                           "success": bool(getattr(result, "success", False)),
                           "data": getattr(result, "data", None)})
                except Exception as exc:  # noqa: BLE001 — 记地图绝不影响执行
                    logger.debug("marker_sink failed for %s: %s", skill_name, exc)

            # 成功结果按普通字段写回；失败结果仅写回 VERIFIED_STATE_KEY 中显式回读验证的事实。
            # 预扫描早停虽失败，仍可确认扫描已停止；不能因 success=False 保留过期运行状态。
            # 未验证的目标值仍不得写入缓存。
            from mast.core.state import patch_verified_state

            patch_verified_state(self.state, getattr(result, "data", None),
                                 what=skill_name)
            return result
        except InstrumentBusy as busy:
            # 「被令牌拒了」与「这一步失败了」是两件事,而这里只能返回一个
            # 失败的 SkillResult(agent 那条路要的是一句能读的话)。所以再带一个
            # **结构化标记**:代码调用方(conduct Director)据此判 busy → 下一 tick
            # 重试,而不是吃掉一次重试预算 —— 一次正常的并发仲裁不该让一个从未
            # 动过仪器的步「失败」。
            #
            # 键名的单一真源是 ``mast.conduct.adapters.INSTRUMENT_BUSY_KEY``,
            # 那边有测试钉住这一对。**别改成按错误文案判**:文案是给人读的。
            return SkillResult(skill_name=skill_name, success=False,
                               error=busy.message(),
                               data={"instrument_busy": True,
                                     "holder": dict(busy.holder or {})})
        except Exception as e:
            # P2-G: LangGraph's GraphInterrupt is control flow (a composite's
            # human node pausing the run) — re-raise so the graph pauses;
            # converting it to a failed SkillResult would silently cancel HITL.
            try:
                from langgraph.errors import GraphInterrupt
                if isinstance(e, GraphInterrupt):
                    raise
            except ImportError:  # pragma: no cover
                pass
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    def _recheck_after_refresh(self, skill, unmet: list) -> list:
        """前置不满足时向硬件问一次,再判一次。返回仍然不满足的那些。

        与 ``core/executor.py`` 的手动路径同一个动作、同一个理由(硬件状态的传播
        落后于 TCP 回包)。读不到硬件就**保留原判**——刷新失败不该变成放行。
        """
        state = getattr(self, "state", None)
        refresh = getattr(state, "refresh", None)
        if not callable(refresh):
            return unmet
        try:
            fresh = refresh()
        except Exception as exc:  # noqa: BLE001 — 刷新失败保留原判,不放行
            logger.debug("precondition recheck refresh failed: %s", exc)
            return unmet
        try:
            return skill.check_preconditions(fresh) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("precondition recheck raised: %s", exc)
            return unmet

    def check_abort(self) -> bool:
        """True if ANY of this context's stop sources fired (see __init__).

        Composite skills call this between long steps; the three abort gates
        (skill_adapter / .run / .safe_call) all read it too, so this is the one
        predicate that decides whether the instrument may still be driven."""
        return any(e.is_set() for e in self._aborts)

    def abort_reason(self) -> str:
        """读取中止来源与原因；没有记录时返回空串，不猜测。
        
        中止可能来自用户急停、E_STOP 事件、环境告警等不同通道。
        调用方应展示实际记录；空串表示未知，不能统一归因于用户。"""
        for e in self._aborts:
            try:
                if e.is_set():
                    why = getattr(e, ABORT_REASON_ATTR, "")
                    if why:
                        return str(why)
            except Exception:  # noqa: BLE001 — 问原因永远不该把调用方搞崩
                continue
        return ""

    # Back-compat: a few call sites (and tests) reach for the raw event.
    @property
    def _abort(self) -> threading.Event:
        """The primary stop event. Prefer :meth:`check_abort` — it sees them all."""
        return self._aborts[0]


__all__ = ["ExecutionContext", "mark_abort", "ABORT_REASON_ATTR"]
