"""Parse SI-prefixed physical quantities with an explicit magnitude prefix.

A missing exponent can leave a syntactically valid number. For model-supplied
values spanning many decades, require a recognized SI prefix so a missing
prefix becomes a parse error instead of a plausible magnitude. The parser
converts the string before any binary instrument command is sent.

This convention is for model-supplied quantities, not values already held in
configuration or entered through a structured operator UI. A deliberate zero
uses a prefixed representation, such as 0p; a bare zero is refused because it
does not establish that magnitude information survived the input path.
"""

from __future__ import annotations

import re

#: Prefix → factor. CASE MATTERS: ``m`` is milli, ``M`` is mega — a case-folding
#: parser would turn a 3-milli into a 3-mega, which is the exact class of error
#: this module exists to prevent.
SI_PREFIXES: dict[str, float] = {
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,   # U+00B5 MICRO SIGN
    "μ": 1e-6,   # U+03BC GREEK SMALL LETTER MU — what most keyboards emit
    "m": 1e-3,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
}

#: The prefix group is NOT optional. That is the entire mechanism.
_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([afpnuµμmkMG])\s*$")


class SIParseError(ValueError):
    """A quantity that did not carry a prefix, or carried an unknown one."""


def parse_si(text: object, *, what: str = "值") -> float:
    """``"3p"`` → ``3e-12``. Raises :class:`SIParseError` on anything else.

    There is deliberately NO fallback to ``float(text)``. A fallback would restore
    exactly the failure being prevented: ``3p`` arriving as ``3`` would parse
    happily and be wrong by a factor of a trillion, silently. The whole value of
    this function is that it says no.

    Args:
        text: the quantity as written, e.g. ``"3p"``, ``"1.6667n"``, ``"150p"``.
        what: parameter name, used in the error message so a rejection is
            actionable rather than generic.
    """
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        raise SIParseError(
            f"{what} 必须写成带 SI 前缀的字符串(如 '3p'),不能是裸数字 {text!r}。"
            "裸数字一旦在传输中丢掉指数就变成另一个合法数字且无人察觉 —— "
            "前缀掉了则直接解析失败,这正是要求前缀的原因。"
        )
    s = str(text or "").strip()
    m = _PATTERN.match(s)
    if not m:
        raise SIParseError(
            f"{what} = {text!r} 无法解析。必须是「数字 + SI 前缀」,前缀不可省略,"
            f"例如 3p(=3e-12)、180n(=1.8e-07)、150p(=1.5e-10)。"
            f"可用前缀: a f p n u/µ m k M G(**区分大小写**: m=毫, M=兆)。"
        )
    mantissa, prefix = m.group(1), m.group(2)
    try:
        return float(mantissa) * SI_PREFIXES[prefix]
    except (TypeError, ValueError, KeyError) as exc:  # pragma: no cover
        raise SIParseError(f"{what} = {text!r} 无法解析") from exc


# ── Two tiers ───────────────────────────────────────────────────────────────
#
# Measured on the real path (kimi-k3, tool calling, tool_choice=auto,
# 2026-08-04): a NUMBER-typed tool argument returned the wrong value 12 times
# out of 12 — `3e-12` arrived as `3`, `1.8e-07` as `1.8`, `1.5e-10` as `1.5`.
# The same values through a STRING-typed argument came back byte-identical 12
# times out of 12. So the string is not a stylistic preference, it is the
# difference between 0% and 100% on the channel every hardware parameter uses.
#
# That gives two independent protections, and they are worth keeping distinct:
#
#   1. STRING-NESS — bypasses the broken constrained-decoding number grammar.
#      Applies to every dimensioned parameter, and costs nothing.
#   2. MANDATORY PREFIX — makes a lost magnitude a parse ERROR instead of a
#      plausible number. Only meaningful where unity is not a legitimate value:
#      for a Z gain, "3" is never right; for a bias in volts, "-2" is exactly
#      right, and demanding "-2000m" would be absurd.
#
# So the prefix is required only when the parameter's own range says a bare
# mantissa cannot be a legitimate value there.


def needs_strict_prefix(min_value: "float | None",
                        max_value: "float | None") -> bool:
    """Is a bare mantissa (0.1 … 1000) impossible for this parameter?

    True → the prefix is mandatory, because a dropped exponent would produce a
    number this parameter could never legitimately hold, and we would rather
    reject it than let the bounds be the only thing standing between a typo and
    the instrument.

    False → unity is a legitimate scale here (volts, seconds, degrees), so a
    plain ``"-2"`` is meaningful and the prefix stays optional. Those parameters
    still get the string treatment; they just cannot use the prefix as a
    magnitude checksum.

    An unknown bound answers False: "we do not know the range" is not evidence
    that a bare number is impossible.
    """
    try:
        hi = abs(float(max_value)) if max_value is not None else None
        lo = abs(float(min_value)) if min_value is not None else None
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return False
    if hi is not None and hi < 0.1:
        return True                       # whole range far BELOW unity
    if lo is not None and lo > 1000.0:
        return True                       # whole range far ABOVE unity
    return False


def parse_quantity(text: object, *, strict: bool, what: str = "值") -> float:
    """Parse a quantity written as a string. ``strict`` demands an SI prefix.

    Non-strict still accepts a prefix (``"5m"``), so an operator or a model may
    always write the panel form; it just additionally accepts ``"-2"``,
    ``"0.05"`` and ``"5e-3"``, which are the natural way to write a quantity
    whose scale is around one.

    A real float passes through untouched. That is not a loophole in the model
    path — the model cannot send a float, because the tool schema declares the
    field a string — it is for the internal callers (composite skills, the
    executor, tests) that legitimately hold numbers already.
    """
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if strict:
        return parse_si(text, what=what)
    s = str(text or "").strip()
    if not s:
        raise SIParseError(f"{what} 不能为空。")
    try:
        return parse_si(s, what=what)
    except SIParseError:
        pass
    try:
        return float(s)
    except (TypeError, ValueError) as exc:
        raise SIParseError(
            f"{what} = {text!r} 无法解析。写成数值字符串,可以带 SI 前缀 "
            f"(如 '5m' = 0.005)也可以是普通十进制/科学计数法(如 '0.05'、'5e-3')。"
        ) from exc


def format_si(value: float, *, digits: int = 6) -> str:
    """``3e-12`` → ``"3p"``. The inverse, for showing stored values back.

    Round-trips through :func:`parse_si`, so what the operator reads in the UI is
    what they could type back in. Six significant figures because five is not
    enough for the values actually in use — the approach time constant is
    16.667 µs, and ``%.4g`` renders that as ``16.67u``, quietly losing a digit on
    every display/edit cycle.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return "0p"
    for prefix in ("G", "M", "k", "", "m", "u", "n", "p", "f", "a"):
        factor = 1.0 if prefix == "" else SI_PREFIXES[prefix]
        scaled = v / factor
        if 1 <= abs(scaled) < 1000:
            mantissa = f"{scaled:.{digits}g}"
            # A magnitude with no prefix cannot be re-parsed, and this function's
            # output is meant to be typeable back in. Step down one decade.
            if prefix == "":
                return f"{float(mantissa) * 1000:.{digits}g}m"
            return f"{mantissa}{prefix}"
    return f"{v:.{digits}g}"


def format_si_readable(value: float, unit: str = "", *, digits: int = 4) -> str:
    """``3e-12, "m"`` → ``"3 pm"``；``0.5, "s"`` → ``"500 ms"``. **For reading.**

    The difference from :func:`format_si` is the goal, not the arithmetic:
    ``format_si`` is optimised to be *typed back in*, so a magnitude with no
    prefix steps down a decade (``format_si(10.0) == "10000m"``) — correct there,
    and it renders "a 10 V pulse" as "a 10000 m V pulse" here. Readability and
    round-tripping are two goals; this is the readable one. Both share the one
    prefix table, and ``test_narration_si_roundtrip`` pins that they name the
    same physical quantity for the same input.

    Lives in ``core`` rather than in the narration templates because it is not a
    narration concern: any operator-facing string with a number in it needs it.
    2026-08-18 the tilt loop was rendering ``第 2 轮后残余 Z 占用 6.4e-09 m`` into
    a sentence a human reads, because the only readable formatter in the repo
    lived one layer away in ``mast.chat``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == 0:
        return f"0 {unit}".strip()
    for prefix in ("G", "M", "k", "", "m", "u", "n", "p", "f", "a"):
        factor = 1.0 if prefix == "" else SI_PREFIXES[prefix]
        scaled = v / factor
        if 1 <= abs(scaled) < 1000:
            return f"{scaled:.{digits}g} {prefix}{unit}".strip()
    return f"{v:.{digits}g} {unit}".strip()


__all__ = [
    "SI_PREFIXES", "SIParseError",
    "format_si", "format_si_readable",
    "needs_strict_prefix", "parse_quantity", "parse_si",
]
