"""Injected text must never give a magnitude the model cannot use directly.

Every skill parameter is SI (metres, amperes, volts) — `unit="m"` appears on 74
of them, `unit="A"` on 6. So any number we inject into the system message in a
human unit is a number the model has to convert unaided, and a dropped exponent
there drives the tip into the sample.

2026-07-27 audit of every block that reaches a system message found two that
printed human units with NO SI counterpart anywhere in the text:

  experiment_prefs   "扫描尺寸(边长): 50 nm" / "电流设定点 setpoint: 100 pA"
                     under a heading that says **优先采用它们** — while the
                     skills take scan_width_m=5e-8 and setpoint_a=1e-10.
  _fmt_didv          stored didv_at_contact_v is VOLTS, rendered "3.500 µV";
                     the same block tells the agent to compare it against a live
                     lock-in R reading, which is in volts. Off by 1e6.

Both are worse than the block involved in the original coordinate incident:
live_state prints the SI value first (1.253e-06 m) and carries a MAGNITUDE CHECK
guard right after it. These two had neither.

The shape that is correct — and that safety_mw's correction text already used —
is: every human unit immediately followed by its SI equivalent.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_injection_si_units.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import re  # noqa: E402

import pytest  # noqa: E402


#: A number followed by a sub-SI unit prefix — the shape that needs an SI partner.
_HUMAN_UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(nm(?:/s)?|pA|nA|µV|uV|mV|µm|um)\b")
#: Scientific notation, i.e. a number the model can pass straight through.
#: 「同一行上有没有一个机器可用的值」。两种都算：
#:   * 指数写法 `5e-08`     —— 接近 1 的量（偏压、秒）仍然这么印
#:   * SI 前缀   `'50n'`     —— 2026-08-04 起，量程远小于 1 的参数**只接受**这种
#: 判的是「有没有给出可照抄的值」，不是「用哪种写法」；写法对不对由
#: tests/v2/unit/agents/test_injected_blocks_show_parseable_values.py 负责。
_SI_FORM = re.compile(r"\d(?:\.\d+)?e[+-]?\d+|'-?\d+(?:\.\d+)?[afpnuµmkMG]'", re.I)


def _human_units_all_have_si(text: str) -> list[str]:
    """Return the human-unit occurrences on lines carrying no SI value."""
    offenders = []
    for line in text.splitlines():
        if not _HUMAN_UNIT.search(line):
            continue
        if not _SI_FORM.search(line):
            offenders.append(line.strip())
    return offenders


# ════════════════════════════════════════════════════════════════════════════
# experiment_prefs
# ════════════════════════════════════════════════════════════════════════════

_PREFS = {"scan_size_nm": 50, "scan_speed_nm_s": 20, "setpoint_pa": 100,
          "bias_v": 0.5, "scan_lines": 256, "line_time_s": 0.5}


def test_prefs_block_gives_si_for_every_human_unit():
    from mast.agents._shared.experiment_prefs import format_prefs_block
    block = format_prefs_block(_PREFS)
    offenders = _human_units_all_have_si(block)
    assert not offenders, (
        "prefs block states a magnitude in human units with no SI value on the "
        f"same line — the model has nothing correct to copy:\n  " +
        "\n  ".join(offenders)
    )


def test_prefs_block_names_the_si_parameter():
    """Knowing the number is not enough — the model must know WHICH parameter."""
    from mast.agents._shared.experiment_prefs import format_prefs_block
    block = format_prefs_block(_PREFS)
    assert "scan_width_m" in block
    assert "setpoint_a" in block
    # 2026-08-04：实质不变（每个人类单位旁边都有一个可以照抄的机器值），变的是
    # 那个值的形式 —— 这个块明说「用括号里的 SI 值」，而 scan_width_m / setpoint_a
    # 都强制 SI 前缀，'5e-08' 抄过去会被 parse_si 拒掉。印出来的必须是能用的。
    assert "'50n'" in block, "50 nm must appear as '50n'"
    assert "'100p'" in block, "100 pA must appear as '100p'"


def test_prefs_block_tells_the_model_which_number_to_use():
    from mast.agents._shared.experiment_prefs import format_prefs_block
    block = format_prefs_block(_PREFS)
    assert "SI" in block and "括号" in block, (
        "the block must say explicitly to use the parenthesised SI value")


def test_prefs_block_empty_when_unset():
    """The shipped (unset) default must inject nothing at all."""
    from mast.agents._shared.experiment_prefs import format_prefs_block
    assert format_prefs_block({}) == ""
    assert format_prefs_block(None) == ""


# ════════════════════════════════════════════════════════════════════════════
# instrument_profile._fmt_didv
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("volts", [3.5e-6, 1.2e-3, 9.9e-4, 5.0e-3, 2.2e-5])
def test_didv_render_always_carries_volts(volts):
    """Whichever readable branch it takes, the stored VOLT value must be present
    — the agent compares this against a lock-in R reading that is in volts."""
    from mast.core.instrument_profile import _fmt_didv
    out = _fmt_didv(volts)
    assert _SI_FORM.search(out), f"no SI value in {out!r} (stored {volts} V)"
    # And the SI value must be the real one, not a rescaled copy.
    m = _SI_FORM.search(out)
    assert float(m.group(0)) == pytest.approx(volts, rel=1e-3), (
        f"{out!r} carries {m.group(0)} but the stored value is {volts}")


def test_didv_uncalibrated_is_not_a_number():
    from mast.core.instrument_profile import _fmt_didv
    assert _fmt_didv(None) == "尚未标定"


# ════════════════════════════════════════════════════════════════════════════
# experiment_design prompt — the guard used to exist only in instrument_control
# ════════════════════════════════════════════════════════════════════════════

def test_experiment_design_prompt_has_the_magnitude_guard():
    """XD's plan values flow into IC and drive the tip, but the mantissa guard
    lived only in IC's prompt. XD held the human-unit blocks AND no guard."""
    from mast.agents.experiment_design.prompts import SYSTEM_PROMPT as XD
    # 2026-08-04：实质三件事一件不少 —— 一个 nm 的换算例、一个 pA 的换算例、
    # 以及「只写尾数会怎样」的警告。变的只是正确形式：XD 产出的数流进 IC 去驱动
    # 针尖，而那些参数现在**强制 SI 前缀**，'5e-8' 会被直接拒绝。
    assert "'50n'" in XD, "no worked nm→SI example"
    assert "'100p'" in XD, "no worked pA→SI example"
    # 警告本身换了说法（「掉了指数仍是合法数字」比「只写尾数」更说明机理），
    # 但必须还在：这是整段的要害。
    assert "仍然是一个合法数字" in XD or "尾数" in XD, "no mantissa-alone warning"


def test_instrument_control_guard_still_present():
    """Pin the guard XD's was modelled on, so removing it fails loudly."""
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT as IC
    assert "1e-7" in IC or "1e-8" in IC
    assert "mantissa" in IC.lower()


def test_instrument_control_states_the_si_string_argument_format():
    """The prompt must describe the format the SCHEMA actually demands.

    Dimensioned arguments became strings on 2026-08-04 (a number-typed tool
    argument came back corrupted 12 times out of 12 on this provider). Until
    that day this prompt carried two 🔴 MANDATORY blocks teaching the model to
    emit ``width_m=1.5e-8`` — bare numbers, i.e. the one channel that does not
    work. A prompt that contradicts the schema is worse than a silent one: it
    produces confident, well-formed, rejected calls.

    This is not persuasion ; the
    argument format is a fact about the interface, and stating a fact is the
    prompt's job.
    """
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT as IC
    assert '"100n"' in IC, "no worked SI-string example"
    assert "case matters" in IC.lower() or "区分大小写" in IC, (
        "m=milli vs M=mega is unstated — a silent 1e9 error")
    # 钉的是**机理**，不是次数（2026-08-24）。「量过 12 次」对模型没有用 ——
    # 它既不能复现也不能查证；而「裸数字会丢指数，所以前缀是校验和」是它需要
    # 理解的那一半，没有这一半规则读起来就是武断的。次数与日期搬进了
    # prompts.py 的注释（溯源闸门要求），那里下一个改这个文件的人看得到。
    low = IC.lower()
    assert "丢指数" in IC or "指数会被拆开" in IC or "exponent" in low, (
        "没说清裸数字为什么活不下来，规则读起来就是武断的")
    assert "校验和" in IC or "checksum" in low, (
        "没说清前缀为什么是校验和 —— 那是它「坏了会响」的全部理由")


def test_instrument_control_still_teaches_the_setpoint_unit():
    """Format and UNIT are separate lessons; a format rewrite must not eat the
    unit one. It did, briefly, on 2026-08-04 — this pins it shut."""
    from mast.agents.instrument_control.prompts import SYSTEM_PROMPT as IC
    assert "AMPERES" in IC
    assert '"100p"' in IC, "no worked setpoint example in the demanded notation"


def test_design_bounds_are_stated_in_si():
    """The bounds block listed setpoint in [1 pA, 100 nA] — human units, no SI —
    right where the model reads what it may propose."""
    from mast.agents.experiment_design.prompts import SYSTEM_PROMPT as XD
    bounds = XD.split("Global parameter limits")[1][:400]
    assert "1e-12" in bounds and "1e-7" in bounds


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
