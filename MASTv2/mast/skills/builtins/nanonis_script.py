"""Nanonis Script module — real-time sequences, and the one hole in every MAST gate.

WHAT IT IS
==========
The Script module is Nanonis's real-time sequencer. A script is compiled and
**deployed onto the RT controller**, where it runs with hardware timing — no TCP
round-trips. It is the only way to do anything that needs microsecond determinism:
a pulse train, a bias sweep synchronised with the lock-in, a tip-conditioning
recipe, a **pump–probe delay scan**.

Its parameterisation channel is the **LUT** (look-up table): a float array the
script steps through at RT speed. That is how a delay scan works — load a table of
delay values, the script walks it. ``Script_LUTLoad`` accepts the array **directly
over TCP**, so the LUT is the one thing an agent can genuinely author here.

THE RISK, STATED PRECISELY
==========================
It is NOT that an LLM could upload arbitrary firmware. It cannot:
``Script_Load`` takes a **file path on the Nanonis machine**, not source code. The
agent can only run a file a human already put there.

The real thing is architectural, and it is worse:

    **A running script is invisible to, and untouchable by, every MAST safety
    layer.**

SafetyGate, the operating-mode gate, the ABORT gate, HITL — every one of them sits
on the ``safe_call`` TCP path. A deployed script issues no ``safe_call``s; it runs
on the controller. Therefore, inside a running script:

  * the global bounds (bias ±10 V, setpoint, Z, scan size) **do not apply**;
  * the abort gate **cannot stop it** — you press 中止 and the script keeps driving
    the tip;
  * the only thing that stops it is ``Script_Stop``.

So four things, and all four are load-bearing:

  1. **``Script_Stop`` is on the post-abort allow-list** (core/execution_context).
     If an abort could not kill a running script, 中止 would do the opposite of its
     job — and this is the one place where "MAST stopped issuing commands" is not
     the same as "the instrument stopped".
  2. **Every ``Run`` is written to the refusal ledger** (记录 → 诊断) with the slot
     and the LUT it ran with. MAST cannot see inside the script, so **that record
     is the only thing you will have afterwards.**
  3. **The skills say so, in their descriptions.** An agent that believes the
     safety gates protect it inside a script is more dangerous than one that knows
     they do not.
  4. **An operator-maintained allow-list of vetted slots**
     (``config/nanonis_scripts.json``). Empty by default → the agent can run
     nothing. This is the only honest form of "a human vetted this".

WHY ``Script_Load`` IS NOT A SKILL — and why that is not the same call as
``UserOut_LimitsSet``
======================================================================
For ``UserOut_LimitsSet`` the conclusion runs the other way: a Nanonis user
output is a small-signal line and **the physical layer already protects you** — the
wiring, the amplifier, the choice of what to connect. The limits are a
configuration convenience, not the last barrier.

Here there is **no physical layer**. A script can drive the bias, Z, and the coarse
motor to anything the hardware allows, and MAST sees none of it. The allow-list IS
the barrier — the only one. And ``Script_Load(slot, file)`` would let the agent put
a different file into a vetted slot, at which point "slot 1 is approved" means
nothing at all.

So slot contents stay with the human, in the Nanonis GUI. Want the agent to switch
recipes? Put each recipe in its own slot and vet each one. (``Script_Save`` and
``Script_LUTSave`` are also out — they write files on the Nanonis machine and buy
the agent nothing.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_CONFIG_NAME = "nanonis_scripts.json"


# ── the allow-list ───────────────────────────────────────────────────────────

def _allowlist_path() -> Path:
    from mast._runtime_paths import project_root

    return project_root() / "config" / _CONFIG_NAME


def load_allowlist() -> dict[int, dict]:
    """Vetted script slots → their declared properties. FAIL-CLOSED.

    A missing file, an unreadable file, a corrupt file, or an empty list all mean
    the SAME thing: **no slot is approved, the agent runs nothing**. There is no
    reading of this that opens the gate — an unknown vetting state is not a pass.
    Nothing in MAST writes this file; the operator owns it.
    """
    p = _allowlist_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 — a corrupt file is not an open gate
        logger.warning("script allow-list unreadable (%s): %s — 视为空清单", p, exc)
        return {}

    out: dict[int, dict] = {}
    for entry in (raw.get("allowed_slots") or []):
        if not isinstance(entry, dict):
            continue
        try:
            slot = int(entry["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        out[slot] = entry
    return out


def _refuse(name: str, slot: int, reason: str) -> SkillResult:
    """A refusal that says how to fix it — and records itself."""
    allowed = sorted(load_allowlist())
    msg = (
        f"{reason}\n"
        f"已审核的槽位：{allowed if allowed else '（空——没有任何脚本被批准）'}\n"
        f"MAST 看不见脚本内部：脚本跑在 RT 控制器上，安全门/模式门/中止门在它里面"
        f"全部不生效。所以只有你在 {_CONFIG_NAME} 里签过字的槽位才能跑。"
        f"审核步骤见该文件的 _how_to_vet。"
    )
    try:
        from mast.core.diagnostics import record

        record("safety_block", f"NanonisScript[slot {slot}]", reason,
               allowed_slots=allowed, skill=name)
    except Exception:  # noqa: BLE001
        pass
    return SkillResult(skill_name=name, success=False, error=msg, nanonis_calls=[])


def _gate(name: str, slot: int) -> "tuple[dict | None, SkillResult | None]":
    """(entry, refusal). Exactly one is non-None."""
    allow = load_allowlist()
    if not allow:
        return None, _refuse(
            name, slot,
            f"拒绝：脚本白名单为空，没有任何脚本槽位被批准供智能体运行。")
    entry = allow.get(slot)
    if entry is None:
        return None, _refuse(
            name, slot, f"拒绝：槽位 {slot} 不在已审核的白名单里。")
    return entry, None


def _values(record) -> list:
    rv = getattr(record, "return_value", None)
    if isinstance(rv, (list, tuple)) and len(rv) > 2 and isinstance(rv[2], (list, tuple)):
        return list(rv[2])
    return []


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


# ── reads ────────────────────────────────────────────────────────────────────

class ListNanonisScripts(BaseSkill):
    """Which script slots may I run, and what does each one do?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ListNanonisScripts",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "列出**用户**已经审过、允许自主使用的那些 Nanonis 脚本槽位，连同每一个做什么、"
                "接受什么 LUT 范围。想跑任何东西之前先调它：没过审的槽位会被拒绝，而这份名单通常很短。"
                "\n"
                "\n"
                "Nanonis 脚本跑在实时控制器上。MAST 的各道安全闸门在脚本**内部不**生效 —— 这正是只有过审槽位才允许运行的原因。"
            ),
            parameters=[],
            estimated_duration_s=0.1,
            composition_level=0,
            tags=["script", "read", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        allow = load_allowlist()
        slots = [
            {"slot": s,
             "name": e.get("name", ""),
             "description": e.get("description", ""),
             "allow_lut_write": bool(e.get("allow_lut_write", False)),
             "lut_min": e.get("lut_min"),
             "lut_max": e.get("lut_max"),
             "notes": e.get("notes", "")}
            for s, e in sorted(allow.items())
        ]
        return SkillResult(
            skill_name="ListNanonisScripts", success=True,
            data={"vetted_slots": slots, "count": len(slots),
                  "allowlist_file": str(_allowlist_path()),
                  "note": ("没有已审核的槽位——智能体不能运行任何脚本。请用户在"
                           f"{_CONFIG_NAME} 里审核并登记。" if not slots else
                           "只有上列槽位可运行；其余一律拒绝。")},
            nanonis_calls=[],
        )


class GetScriptData(BaseSkill):
    """Read what a script recorded into its Acquire Buffer."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScriptData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读一个 Nanonis 脚本录进 Acquire Buffer 的数据。脚本以实时速度把若干通道写进 buffer 1 或 2；"
                "每一次 'sweep' 是脚本里定义的一趟（sweep 从 0 起算）。返回一个由采集通道构成的 2-D 数组。"
            ),
            parameters=[
                ParameterSpec(
                    name="buffer", type="int",
                    description="Acquire Buffer 编号：1 或 2",
                    required=True, min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="sweep", type="int",
                    description="sweep 序号（0 起算）",
                    required=False, default=0, min_value=0, max_value=100000,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["script", "data", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        buf = int(params["buffer"])
        sweep = int(params.get("sweep", 0) or 0)
        rec = context.safe_call("Script_DataGet", buf, sweep)
        if rec.error:
            return _fail("GetScriptData", rec.error, [rec])
        return SkillResult(
            skill_name="GetScriptData", success=True,
            data={"buffer": buf, "sweep": sweep, "data": _values(rec)},
            nanonis_calls=[rec],
        )


class GetScriptChannels(BaseSkill):
    """Which channels is a script's Acquire Buffer recording?"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScriptChannels",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读某个脚本 Acquire Buffer 的通道列表。",
            parameters=[
                ParameterSpec(
                    name="buffer", type="int",
                    description="Acquire Buffer 编号：1 或 2",
                    required=True, min_value=1, max_value=2,
                ),
            ],
            estimated_duration_s=0.3,
            composition_level=0,
            tags=["script", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        buf = int(params["buffer"])
        rec = context.safe_call("Script_ChsGet", buf)
        if rec.error:
            return _fail("GetScriptChannels", rec.error, [rec])
        return SkillResult(skill_name="GetScriptChannels", success=True,
                           data={"buffer": buf, "channels": _values(rec)},
                           nanonis_calls=[rec])


# ── the one that matters ─────────────────────────────────────────────────────

class RunNanonisScript(BaseSkill):
    """Run a VETTED script on the real-time controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="RunNanonisScript",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在**实时控制器**上运行一个 Nanonis 脚本。只有用户审过的槽位（config/nanonis_scripts.json）"
                "才允许运行 —— 先调 ListNanonisScripts。\n"
                "\n"
                "**用它之前先读这一段。** 跑起来的脚本是在控制器上执行的，不走 TCP，所以 **MAST 的各层安全在它内部都不生效**："
                "全局的偏压／电流／Z 边界不被强制，运行模式闸门不被咨询，而且 **abort 闸门停不了它**。"
                "按 中止 只是让 MAST 不再发命令；它停不了一个已经在控制器上跑起来的脚本。唯一能停它的是 StopNanonisScript。"
                "\n"
                "\n"
                "请先把脚本部署上去（DeployNanonisScript）。如果它带参数，运行之前把参数载入它的 LUT（LoadScriptLUT）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="slot", type="int",
                    description="脚本槽位（1 起算）。必须在过审名单里。",
                    required=True, min_value=1, max_value=64,
                ),
                ParameterSpec(
                    name="wait_until_finished", type="bool",
                    description=(
                        "阻塞直到脚本跑完。对一段有界的序列，通常 True 才对；False 会让它继续跑着 —— 那样一来，"
                        "停下它就是你的责任。"
                    ),
                    required=False, default=True,
                ),
            ],
            estimated_duration_s=10.0,
            composition_level=0,
            tags=["script", "realtime", "write", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        slot = int(params["slot"])
        wait = 1 if params.get("wait_until_finished", True) else 0

        entry, refusal = _gate("RunNanonisScript", slot)
        if refusal is not None:
            return refusal

        rec = context.safe_call("Script_Run", slot, wait)
        if rec.error:
            return _fail("RunNanonisScript", rec.error, [rec])

        # MAST CANNOT SEE INSIDE THE SCRIPT. This line is the only account of what
        # ran that anyone will have afterwards — which slot, which recipe, blocking
        # or not, and (from the LUT record written by LoadScriptLUT) with what
        # numbers. Losing it means losing the ability to explain the run at all.
        try:
            from mast.core.diagnostics import record

            record("note", f"NanonisScript[slot {slot}]",
                   f"在 RT 控制器上运行脚本「{entry.get('name', '?')}」"
                   f"（{'阻塞至完成' if wait else '后台运行——需自行 Stop'}）。"
                   "MAST 的安全门/模式门/中止门在脚本内部均不生效。",
                   slot=slot, script_name=entry.get("name", ""),
                   description=entry.get("description", ""),
                   wait_until_finished=bool(wait))
        except Exception:  # noqa: BLE001
            pass

        return SkillResult(
            skill_name="RunNanonisScript", success=True,
            data={"slot": slot, "script_name": entry.get("name", ""),
                  "wait_until_finished": bool(wait),
                  "warning": ("脚本在 RT 控制器上运行——MAST 的安全门与中止门"
                              "对它不生效。要停它只能用 StopNanonisScript。")},
            nanonis_calls=[rec],
        )


class StopNanonisScript(BaseSkill):
    """Stop the running script. Always allowed — including after an abort."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopNanonisScript",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "停掉正在实时控制器上跑的脚本。这是**唯一**能停下一个正在跑的脚本的东西 —— abort 闸门停不了它，"
                "因为脚本并不在发 TCP 调用。因此它从不受闸门管辖、也从不被拒绝，包括在 abort 已经锁住的时候。"
            ),
            parameters=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["script", "stop", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rec = context.safe_call("Script_Stop")
        if rec.error:
            return _fail("StopNanonisScript", rec.error, [rec])
        return SkillResult(skill_name="StopNanonisScript", success=True,
                           data={"stopped": True}, nanonis_calls=[rec])


class DeployNanonisScript(BaseSkill):
    """Compile a vetted script down onto the RT controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DeployNanonisScript",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一个过审的脚本槽位部署到实时控制器上（编译 + 推送）。部署并不会运行它 —— 之后请调 RunNanonisScript。"
                "只有过审的槽位才允许被部署。"
            ),
            parameters=[
                ParameterSpec(
                    name="slot", type="int",
                    description="脚本槽位（1 起算）。必须已过审。",
                    required=True, min_value=1, max_value=64,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["script", "realtime", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        slot = int(params["slot"])
        _entry, refusal = _gate("DeployNanonisScript", slot)
        if refusal is not None:
            return refusal
        calls = []
        rec = context.safe_call("Script_Open")
        calls.append(rec)
        rec = context.safe_call("Script_Deploy", slot)
        calls.append(rec)
        if rec.error:
            return _fail("DeployNanonisScript", rec.error, calls)
        return SkillResult(skill_name="DeployNanonisScript", success=True,
                           data={"slot": slot, "deployed": True},
                           nanonis_calls=calls)


class UndeployNanonisScript(BaseSkill):
    """Take a script back off the RT controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="UndeployNanonisScript",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "把一个脚本槽位从实时控制器上撤下来。撤掉一个脚本从来不是有风险的那个方向，所以它从不受闸门管辖。"
            ),
            parameters=[
                ParameterSpec(
                    name="slot", type="int",
                    description="脚本槽位（1 起算）",
                    required=True, min_value=1, max_value=64,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["script", "realtime", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        slot = int(params["slot"])
        rec = context.safe_call("Script_Undeploy", slot)
        if rec.error:
            return _fail("UndeployNanonisScript", rec.error, [rec])
        return SkillResult(skill_name="UndeployNanonisScript", success=True,
                           data={"slot": slot, "deployed": False},
                           nanonis_calls=[rec])


# ── the LUT: the one thing the agent can really author ───────────────────────

class LoadScriptLUT(BaseSkill):
    """Load the values a script steps through — bounds-checked against the slot."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LoadScriptLUT",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一组数值载入某个脚本的查找表（Look-Up Table）—— 也就是脚本以实时速度逐项走过的那个数组。"
                "延时扫描就是这么工作的：把延时值载进去，脚本自己走完它们。\n"
                "\n"
                "LUT 是你在这里真正能撰写的**那一样**东西（脚本本身是用户的）。所以它的取值范围由过审条目框定："
                "一个声明了 lut_min/lut_max 的槽位，会拒绝范围之外的值。一个 LUT 以毫米为单位的延时线脚本，"
                "喂进去以微米计的数字，就会把台子开到硬限位上 —— 而 MAST 看不到脚本，也就无从知道它想要的是哪一种。"
            ),
            parameters=[
                ParameterSpec(
                    name="slot", type="int",
                    description="这些值是**给哪一个**过审脚本槽位用的",
                    required=True, min_value=1, max_value=64,
                ),
                ParameterSpec(
                    name="lut_index", type="int",
                    description="LUT 编号（1 起算）",
                    required=True, min_value=1, max_value=16,
                ),
                ParameterSpec(
                    name="values", type="str",
                    description="逗号分隔的数值（例如 '0, 0.5, 1.0, 1.5'）",
                    required=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["script", "lut", "write", "safety"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        slot = int(params["slot"])
        lut = int(params["lut_index"])

        entry, refusal = _gate("LoadScriptLUT", slot)
        if refusal is not None:
            return refusal
        if not entry.get("allow_lut_write", False):
            return _refuse("LoadScriptLUT", slot,
                           f"拒绝：槽位 {slot}（{entry.get('name', '?')}）"
                           "未被批准接受智能体写入 LUT（allow_lut_write=false）。")

        try:
            vals = [float(x) for x in str(params["values"]).replace(",", " ").split()]
        except ValueError:
            return _fail("LoadScriptLUT",
                         f"values 解析失败：{params['values']!r}（应为逗号分隔的数值）", [])
        if not vals:
            return _fail("LoadScriptLUT", "values 为空", [])

        lo, hi = entry.get("lut_min"), entry.get("lut_max")
        if lo is not None or hi is not None:
            lo = float(lo) if lo is not None else float("-inf")
            hi = float(hi) if hi is not None else float("inf")
            bad = [v for v in vals if not (lo <= v <= hi)]
            if bad:
                return _refuse(
                    "LoadScriptLUT", slot,
                    f"拒绝：LUT 值 {bad[:5]} 超出槽位 {slot} 声明的范围 "
                    f"[{lo:g}, {hi:g}]（该范围由用户在审核脚本时写定——"
                    "它知道脚本把这些数当什么单位用，MAST 不知道）。")

        calls = []
        rec = context.safe_call("Script_LUTOpen")
        calls.append(rec)
        # LUTLoad(LUT_index, file_path, LUT_values): an empty path means "use the
        # array I am sending" — that array is the whole point.
        rec = context.safe_call("Script_LUTLoad", lut, "", vals)
        calls.append(rec)
        if rec.error:
            return _fail("LoadScriptLUT", rec.error, calls)

        try:
            from mast.core.diagnostics import record

            record("note", f"NanonisScript[slot {slot}].LUT{lut}",
                   f"写入 {len(vals)} 个 LUT 值（范围 {min(vals):g}..{max(vals):g}）"
                   f"，供脚本「{entry.get('name', '?')}」使用",
                   slot=slot, lut_index=lut, n_values=len(vals),
                   min=min(vals), max=max(vals),
                   values=vals[:20])
        except Exception:  # noqa: BLE001
            pass

        return SkillResult(
            skill_name="LoadScriptLUT", success=True,
            data={"slot": slot, "lut_index": lut, "n_values": len(vals),
                  "min": min(vals), "max": max(vals),
                  "note": "需 DeployScriptLUT 才会推送到 RT 控制器。"},
            nanonis_calls=calls,
        )


class DeployScriptLUT(BaseSkill):
    """Push the LUT onto the RT controller."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="DeployScriptLUT",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "把一个 LUT 部署到实时控制器上，好让正在跑的脚本能逐项走它。请先载入数值（LoadScriptLUT）"
                "。"
            ),
            parameters=[
                ParameterSpec(
                    name="lut_index", type="int",
                    description="LUT 编号（1 起算）",
                    required=True, min_value=1, max_value=16,
                ),
                ParameterSpec(
                    name="wait_until_finished", type="bool",
                    description="阻塞直到部署完成",
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="timeout_ms", type="int",
                    description="部署超时（ms）；-1 = 一直等下去",
                    unit="ms", required=False, default=10000,
                    min_value=-1, max_value=600000,
                ),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["script", "lut", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        lut = int(params["lut_index"])
        wait = 1 if params.get("wait_until_finished", True) else 0
        timeout = int(params.get("timeout_ms", 10000) or 10000)
        rec = context.safe_call("Script_LUTDeploy", lut, wait, timeout)
        if rec.error:
            return _fail("DeployScriptLUT", rec.error, [rec])
        return SkillResult(skill_name="DeployScriptLUT", success=True,
                           data={"lut_index": lut, "deployed": True},
                           nanonis_calls=[rec])


# ── acquisition config ───────────────────────────────────────────────────────

class SetScriptChannels(BaseSkill):
    """Choose which channels a script's Acquire Buffer records."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetScriptChannels",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设定某个脚本的 Acquire Buffer 记录哪些信号通道。仅仅是配置 —— 它改变的是「测什么」"
                "，绝不改变仪器做什么。"
            ),
            parameters=[
                ParameterSpec(
                    name="buffer", type="int",
                    description="Acquire Buffer 编号：1 或 2",
                    required=True, min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="channels", type="str",
                    description="信号序号，逗号分隔（例如 '0,24'）",
                    required=True,
                ),
            ],
            estimated_duration_s=0.4,
            composition_level=0,
            tags=["script", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        buf = int(params["buffer"])
        try:
            chs = [int(x) for x in str(params["channels"]).replace(",", " ").split()]
        except ValueError:
            return _fail("SetScriptChannels",
                         f"channels 解析失败：{params['channels']!r}", [])
        if not chs:
            return _fail("SetScriptChannels", "channels 为空", [])
        rec = context.safe_call("Script_ChsSet", buf, chs)
        if rec.error:
            return _fail("SetScriptChannels", rec.error, [rec])
        return SkillResult(skill_name="SetScriptChannels", success=True,
                           data={"buffer": buf, "channels": chs},
                           nanonis_calls=[rec])


class SetScriptAutosave(BaseSkill):
    """Auto-save the Acquire Buffers to file after a script run."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetScriptAutosave",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "运行结束后把脚本 Acquire Buffer 里的数据自动存盘。任何长序列都建议开 —— 缓冲区是有限的，"
                "而没存下来的数据就等于没测过。"
            ),
            parameters=[
                ParameterSpec(
                    name="buffer", type="int",
                    description="Acquire Buffer 编号：1 或 2",
                    required=True, min_value=1, max_value=2,
                ),
                ParameterSpec(
                    name="sweep", type="int",
                    description="要保存的 sweep 编号；-1 = 全部 sweep",
                    required=False, default=-1, min_value=-1, max_value=100000,
                ),
                ParameterSpec(
                    name="all_sweeps_same_file", type="bool",
                    description="把每一趟 sweep 都放进同一个文件",
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="folder_path", type="str",
                    description="Nanonis 机器上的文件夹（留空 = 会话目录）",
                    required=False, default="",
                ),
                ParameterSpec(
                    name="basename", type="str",
                    description="文件基名",
                    required=False, default="mast_script",
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["script", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        buf = int(params["buffer"])
        sweep = int(params.get("sweep", -1) if params.get("sweep") is not None else -1)
        same = 1 if params.get("all_sweeps_same_file", True) else 0
        folder = str(params.get("folder_path", "") or "")
        base = str(params.get("basename", "mast_script") or "mast_script")
        rec = context.safe_call("Script_Autosave", buf, sweep, same, folder, base)
        if rec.error:
            return _fail("SetScriptAutosave", rec.error, [rec])
        return SkillResult(skill_name="SetScriptAutosave", success=True,
                           data={"buffer": buf, "sweep": sweep, "basename": base},
                           nanonis_calls=[rec])


# DELIBERATELY NOT SKILLS — see the module docstring:
#   Script_Load     — putting a DIFFERENT file into a vetted slot makes "slot 1 is
#                     approved" mean nothing. The allow-list is the ONLY barrier
#                     here (unlike UserOut, where the wiring protects you), so slot
#                     contents stay with the human, in the Nanonis GUI.
#   Script_Save     — writes a file on the Nanonis machine; buys the agent nothing.
#   Script_LUTSave  — likewise.

__all__ = [
    "ListNanonisScripts", "GetScriptData", "GetScriptChannels",
    "RunNanonisScript", "StopNanonisScript",
    "DeployNanonisScript", "UndeployNanonisScript",
    "LoadScriptLUT", "DeployScriptLUT",
    "SetScriptChannels", "SetScriptAutosave",
    "load_allowlist",
]
