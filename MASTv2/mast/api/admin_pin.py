"""The admin PIN — now actually checked.

WHAT WAS THERE BEFORE
=====================
``POST /api/admin/unlock-pin`` existed. It compared a SHA-256 of the entered PIN
against ``config/admin_pin.txt`` and minted a session token, with this comment:

    "The session-token issuance + auth wiring is finalised at integration; here it is
     a deterministic placeholder token."

Nothing ever finalised it. **No route checked the token, no code ever wrote
admin_pin.txt, and the frontend never called the endpoint.** The PIN gate was a
sign, not a lock — which is fine right up until someone says "only the advanced menu
can turn this on" and believes it.

WHAT IT GATES NOW
=================
Exactly two settings keys, and only these:

  * ``advanced_capabilities`` — powers that step around a protection (loading a
    script into a slot the allow-list vets, quitting Nanonis, loading a scan config
    MAST cannot read, holding the main TCP connection for a whole scan).
  * ``hardware_modules``      — because switching one ON hands the agent DANGEROUS
    skills: a laser, an RF amplifier, probes that can collide.

Everything else in 设置 (model, voice, appearance, knowledge, vision thresholds)
stays open. Getting those wrong costs you quality, not safety.

THE THREAT MODEL, STATED PLAINLY
================================
MAST's **in-process** agents have no path to this API — they have no HTTP tool and no
settings tool; their whole surface is Nanonis skills. An **external** agent (driving
MAST over HTTP, e.g. through the ``/api/ext/v1`` gateway and the Claude Code plugin)
holds the same credentials as the operator and CAN reach it, exactly as a person at
the keyboard can: the external-agent guide tells it not to, but nothing here enforces
that (delegated, scoped credentials were considered and deliberately not built).
So this is not a defence against a model. It is a defence against a human hand: a
mis-click, a second person at the bench, a browser tab left open on the admin page.
That is a real hazard and a PIN is the right size of answer to it. It is not, and does
not pretend to be, authentication.

FAIL-CLOSED, AND WHAT THAT MEANS THE FIRST TIME
===============================================
No PIN set ⇒ the guarded keys **cannot be written**. Not "written freely" — refused,
with an instruction to set one. Granting an agent the ability to overwrite a vetted
script slot should not be something that happens on a machine where nobody has yet
claimed to be the administrator.

Lost the PIN? Delete ``config/admin_pin.txt``. That is deliberate: this is a bench
instrument, not a bank, and locking the operator out of their own microscope would
be a worse failure than the one this guards against.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The two settings keys that may only be written with the PIN.
GUARDED_KEYS: frozenset[str] = frozenset({
    "advanced_capabilities",
    "hardware_modules",
    # The coarse stepper's per-rig drive ceiling. Unlike the other two — which
    # grant the AGENT a power — this one is a claim about the HARDWARE that
    # nothing can verify: the controller will happily output a voltage the piezo
    # stack does not survive, and no reading tells you which rig you are on. Get
    # it wrong and the stack is gone. It sits behind the PIN for the same reason
    # the others do (a guard against a stray human hand, not against the model —
    # the model has no path to this API at all), and it inherits the fail-closed
    # rule: with no PIN configured, it cannot be written, so an unattended rig
    # stays at "undeclared", which refuses every drive write (2026-07-31).
    "coarse_drive",
    # 修针方案表的用户覆写(2026-08-10 补上写入口)。同一条理由:它能**放宽硬件
    # 包络** —— ``max_abs_pulse_v`` / ``max_poke_depth_m`` 就在这张表里,而那两个数
    # 决定「多大的脉冲、多深的下压会被拒绝」。一次误操作直接放大物理动作的上限,
    # 与 ``hardware_modules`` 同类。
    #
    # ⚠️ 挂 PIN 是一个**安全取舍**,不是显然正确:它同时意味着「没设 PIN 的机器
    # 一律改不了方案表」(fail-closed)。根因是**根本没有写入口**,不是
    # PIN;但加了 PIN 之后,「没设 PIN」会变成新的卡点。
    # **这一条留给用户定夺**(见 docs/v2/design/forge_scan_working_point.md §八):
    # 要摘掉 PIN,把这一行删掉即可,判据是「误改包络的代价 vs 设 PIN 的麻烦」。
    "tip_conditioning_overrides",
})


def _pin_path() -> Path:
    from mast._runtime_paths import project_root
    return project_root() / "config" / "admin_pin.txt"


def _stored_hash() -> str | None:
    """The stored SHA-256 hex, '' if no PIN is set, None if we could not read.

    None and '' are DIFFERENT. '' means "nobody has set a PIN" (a state we can act
    on: tell them to set one). None means "we cannot tell" — and a guard that cannot
    tell must refuse, not wave through.
    """
    try:
        p = _pin_path()
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8").strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("admin PIN 读取失败: %s", exc)
        return None


def pin_is_set() -> bool:
    return bool(_stored_hash())


def verify_pin(raw: str | None) -> tuple[bool, str]:
    """(ok, reason). Reasons: '' | 'no_pin_set' | 'empty' | 'wrong' | 'unreadable'."""
    stored = _stored_hash()
    if stored is None:
        return False, "unreadable"
    if not stored:
        return False, "no_pin_set"
    entered = (raw or "").strip()
    if not entered:
        return False, "empty"
    digest = hashlib.sha256(entered.encode("utf-8")).hexdigest()
    return (True, "") if digest == stored else (False, "wrong")


def set_pin(new_pin: str, current_pin: str | None = None) -> tuple[bool, str]:
    """Set or change the admin PIN. Changing one requires the current one.

    (ok, reason). Reasons: '' | 'too_short' | 'wrong_current' | 'unreadable' | 'io_error'
    """
    new = (new_pin or "").strip()
    if len(new) < 4:
        return False, "too_short"

    stored = _stored_hash()
    if stored is None:
        return False, "unreadable"
    if stored:
        ok, _ = verify_pin(current_pin)
        if not ok:
            return False, "wrong_current"

    try:
        p = _pin_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Only the hash is ever written. The raw PIN does not touch the disk.
        p.write_text(hashlib.sha256(new.encode("utf-8")).hexdigest(), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("admin PIN 写入失败: %s", exc)
        return False, "io_error"

    logger.warning("admin PIN 已%s", "更改" if stored else "设置")
    return True, ""


_REASON_TEXT = {
    "no_pin_set": ("尚未设置管理 PIN。硬件模块与高级能力的开关必须先设置 PIN 才能改动——"
                   "把「覆盖已审脚本槽位」这类能力交给 agent，不该发生在还没有人认领管理员的机器上。"),
    "empty": "请输入管理 PIN。",
    "wrong": "管理 PIN 不正确。",
    "unreadable": "无法读取管理 PIN 文件——出于安全，拒绝改动。",
    "too_short": "PIN 至少 4 位。",
    "wrong_current": "当前 PIN 不正确。",
    "io_error": "PIN 写入失败（磁盘/权限）。",
}


def reason_text(reason: str) -> str:
    return _REASON_TEXT.get(reason, reason)


__all__ = ["GUARDED_KEYS", "pin_is_set", "verify_pin", "set_pin", "reason_text"]
