"""``AcquireDeltaFCurve``: reading a parsed Nanonis reply, and what happens when you don't.

A reply from the patched client is ``(error, raw_bytes, [values...])``. The numbers are one
level down and name lists are two. Every one of the three parsers in this skill originally
scanned only the outermost level, and every one of them failed the same way: it found no
number, reported the reading as absent, and the skill refused to run on an instrument that had
answered it perfectly. Nothing raised, nothing logged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.skills.builtins.deltaf_curve import AcquireDeltaFCurve  # noqa: E402

SKILL = AcquireDeltaFCurve()
#: what ``PLL_OutOnOffGet`` really answers with the output on
REPLY_ON = ("", b"\x00\x00\x00\x01", [1])
#: and what ``Signals_NamesGet`` answers: names two levels down, behind two header ints
NAMES_REPLY = ("", b"...", [44, 3, ["Current (A)", "OC M1 Freq. Shift (Hz)",
                                    "OC D1 Amplitude (m)"]])


def test_a_number_one_level_down_is_found():
    """This is the shape every reading arrives in. Reading only the top level sees an empty
    error string and a bytes blob, decides there is no number, and reports the PLL as off."""
    assert SKILL._first_number(REPLY_ON) == 1.0


def test_a_bare_number_and_a_flat_list_still_work():
    assert SKILL._first_number(25296.2) == pytest.approx(25296.2)
    assert SKILL._first_number(["", 34182.3]) == pytest.approx(34182.3)


def test_a_reply_with_no_number_gives_none_not_zero():
    """None means "could not read"; 0.0 would mean "the output is off", and the caller acts on
    the difference."""
    assert SKILL._first_number(("", b"x", [])) is None
    assert SKILL._first_number(("error text", b"", None)) is None


def test_the_signal_names_are_found_two_levels_down():
    got = SKILL._string_list(NAMES_REPLY)
    assert got == ["Current (A)", "OC M1 Freq. Shift (Hz)", "OC D1 Amplitude (m)"]


def test_a_reply_carrying_no_names_gives_none():
    assert SKILL._string_list(("", b"x", [44, 0, []])) is None
    assert SKILL._string_list("OC M1 Freq. Shift (Hz)") is None


def test_the_channels_come_back_in_the_order_the_sweep_needs():
    """Frequency shift first — it is the measurement; current and amplitude follow as the
    context that says whether the curve is trustworthy."""
    calls = []

    def call(name, *args):
        calls.append((name, args))

        class R:
            error = ""
            return_value = NAMES_REPLY
        return R()

    chans, err = SKILL._channels({}, call)
    assert err == ""
    names = NAMES_REPLY[2][2]
    assert names[chans[0]].lower().startswith("oc m1 freq")
    assert len(chans) == 3 and len(set(chans)) == 3


def test_explicit_channel_indexes_win_over_the_signal_table():
    chans, err = SKILL._channels({"channel_indexes": "17, 0, 16"}, None)
    assert chans == [17, 0, 16] and err == ""


def test_an_unreadable_signal_table_says_so_instead_of_guessing():
    class R:
        error = "timed out"
        return_value = None

    chans, err = SKILL._channels({}, lambda *a: R())
    assert chans is None and "channel_indexes" in err


# ── the sweep block ──
def _variables(n: int = 8):
    """A ZSpectr Variables block.

    ``[?, ?, channel names, rows, cols, data, ?, ?]`` with **rows = channels** and
    **cols = sweep points** — the row-major layout the official protocol uses. Transposing it
    hands out mixed traces that look like ordinary data."""
    z = np.linspace(0.0, -0.5e-9, n)
    df = -9.0 * np.exp(-((np.linspace(0, 1, n) - 0.6) ** 2) / 0.05)
    cur = np.full(n, 5e-11)
    data = np.vstack([z, df, cur]).astype(float)
    return [0, 0, ["Z rel (m)", "OC M1 Freq. Shift (Hz)", "Current (A)"],
            3, n, data, 0, []]


def test_the_sweep_is_parsed_out_of_the_variables_block_not_the_envelope():
    """Handing the whole ``(error, raw, variables)`` triple to the reshaper gets "3 items, not
    a spectrum" — which is true of the envelope and says nothing about the sweep."""
    data: dict = {}
    assert SKILL._parse(("", b"raw", _variables()), data) is True
    assert data["num_points"] == 8
    assert len(data["freq_shift_hz"]) == 8
    assert data["df_min_hz"] == pytest.approx(min(data["freq_shift_hz"]))
    assert data["z_at_df_min_m"] == pytest.approx(
        data["z_rel"][int(np.argmin(data["freq_shift_hz"]))])
    assert len(data["current_a"]) == 8


def test_an_unparsable_block_records_why():
    """"Could not parse" and "the sweep really had no points" look identical without it."""
    data: dict = {}
    assert SKILL._parse(("", b"raw", [1, 2, 3]), data) is False
    assert data.get("parse_error")
