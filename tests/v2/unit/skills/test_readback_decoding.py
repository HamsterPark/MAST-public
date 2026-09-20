"""Nanonis envelopes must be decoded before their values reach the model.

Independent synthetic replies keep the (error, raw bytes, values) wire shape.
Single values unwrap, multiple values retain order, controller states are
decoded from the value list, and no raw bytes leak into the result."""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
import struct
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

import pytest  # noqa: E402

from mast.skills.builtins.readback import (  # noqa: E402
    GetZControllerState,
    _decode_nanonis,
)


# ── independent synthetic replies with the protocol envelope ───────────────

_SYNTHETIC_RETURNS = {
    "ZCtrl_OnOffGet":          ("", b"\x00\x01\x00\x00", [1]),
    "ZCtrl_StatusGet":         ("", b"\x00\x01\x00\x00", [2]),      # 2 = On
    "ZCtrl_SetpntGet":         ("", struct.pack(">f", 3.25e-10), [3.25e-10]),
    "ZCtrl_GainGet":           ("", b"\x00", [1.25, 2.75, 0.002]),
    "ZCtrl_ZPosGet":           ("", b"\x00", [-6.5e-8]),
    "ZCtrl_LimitsGet":         ("", b"\x00", [-2e-6, 2e-6]),
    "ZCtrl_LimitsEnabledGet":  ("", b"\x00", [1]),
    "ZCtrl_TipLiftGet":        ("", b"\x00", [3e-9]),
    "ZCtrl_WithdrawRateGet":   ("", b"\x00", [2e-7]),
    "ZCtrl_SwitchOffDelayGet": ("", b"\x00", [0.08]),
    "ZCtrl_HomePropsGet":      ("", b"\x00", [1, 2e-8]),
}


class _Rec:
    def __init__(self, value):
        self.return_value = value
        self.error = ""


class _Ctx:
    def safe_call(self, verb, *a, **kw):
        return _Rec(_SYNTHETIC_RETURNS.get(verb, ("", b"", [0])))


# ════════════════════════════════════════════════════════════════════════════
# The decoder
# ════════════════════════════════════════════════════════════════════════════

def test_single_value_is_unwrapped():
    assert _decode_nanonis(("", b"\x00", [3.25e-10])) == 3.25e-10


def test_multi_value_stays_a_list():
    """Gains and limits are genuinely several numbers — unwrapping would lose them."""
    assert _decode_nanonis(("", b"\x00", [1.25, 2.75, 0.002])) == [1.25, 2.75, 0.002]


@pytest.mark.parametrize("plain", [42, 3.14, None, "already text", [1, 2]])
def test_non_wire_values_pass_through(plain):
    assert _decode_nanonis(plain) == plain


def test_bytes_never_survive_decoding():
    out = _decode_nanonis(("", b"\x00\x01\x00\x00", [1]))
    assert not isinstance(out, (bytes, tuple))


# ════════════════════════════════════════════════════════════════════════════
# The skill — the regression that mattered
# ════════════════════════════════════════════════════════════════════════════

def test_controller_on_is_no_longer_always_none():
    """A valid On flag inside the envelope must decode to True rather than unknown."""
    res = GetZControllerState().execute(_Ctx(), {})
    assert res.data["controller_on"] is True, (
        f"controller_on={res.data['controller_on']!r} — the RT truth is still "
        "unreadable, which reads to the agent as 'unknown'")


def test_no_raw_bytes_anywhere_in_the_result():
    """Nothing the model sees may contain undecoded wire data."""
    res = GetZControllerState().execute(_Ctx(), {})
    for key, val in res.data.items():
        assert not isinstance(val, bytes), f"{key} is raw bytes"
        if isinstance(val, tuple):
            assert not any(isinstance(x, bytes) for x in val), \
                f"{key} still carries the wire tuple: {val!r}"
    assert "b'" not in str(res.data), f"bytes leaked into the text: {res.data}"


def test_scalar_readings_are_numbers_the_agent_can_use():
    res = GetZControllerState().execute(_Ctx(), {})
    assert res.data["setpoint"] == pytest.approx(3.25e-10)
    assert res.data["z_m"] == pytest.approx(-6.5e-8)
    assert res.data["gains"] == [1.25, 2.75, 0.002]


def test_module_status_name_now_resolves():
    """The name lookup depended on finding that int too, so it was also dead."""
    res = GetZControllerState().execute(_Ctx(), {})
    assert res.data.get("module_status_name") == "On"


def test_agreement_means_no_warning():
    """RT says on, module says On — no disagreement banner."""
    res = GetZControllerState().execute(_Ctx(), {})
    assert "disagreement" not in res.data


def test_disagreement_is_still_detected(monkeypatch):
    """The whole point of reading both. RT on, module Off → warn."""
    import mast.skills.builtins.readback as rb
    patched = dict(_SYNTHETIC_RETURNS)
    patched["ZCtrl_StatusGet"] = ("", b"\x00", [1])      # 1 = Off

    class _Ctx2(_Ctx):
        def safe_call(self, verb, *a, **kw):
            return _Rec(patched.get(verb, ("", b"", [0])))

    res = rb.GetZControllerState().execute(_Ctx2(), {})
    assert res.data["controller_on"] is True
    assert res.data["module_status_name"] == "Off"
    assert "disagreement" in res.data, (
        "the RT-vs-module disagreement check could never fire while both sides "
        "decoded to None")


# ════════════════════════════════════════════════════════════════════════════
# GetTipShaperConfig — an 11-element array with no field names on the wire
# ════════════════════════════════════════════════════════════════════════════
#
# The protocol order distinguishes voltage, displacement, timing and GET flags.
# Independent values below exercise each position without copying a captured reply.

_SYNTHETIC_TIP_SHAPER = [0.08, 0, 1.75, -8e-10, 0.3, 0.4, 0.6, 9e-10, 0.7, 0.09, 1]


def _tip_shaper_ctx(props):
    class _C:
        def safe_call(self, verb, *a, **kw):
            return _Rec(("", b"\x00", list(props)))
    return _C()


def test_tip_shaper_props_come_back_named():
    from mast.skills.builtins.readback import GetTipShaperConfig

    res = GetTipShaperConfig().execute(_tip_shaper_ctx(_SYNTHETIC_TIP_SHAPER), {})
    named = res.data["props_named"]
    assert named["switch_off_delay_s"] == pytest.approx(0.08)
    assert named["change_bias"] is False           # PropsGet: 0=False, 1=True
    assert named["bias_v"] == pytest.approx(1.75)
    assert named["tip_lift_m"] == pytest.approx(-8e-10)
    assert named["lift_time_1_s"] == pytest.approx(0.3)
    assert named["bias_lift_v"] == pytest.approx(0.4)
    assert named["bias_settling_s"] == pytest.approx(0.6)
    assert named["lift_height_m"] == pytest.approx(9e-10)
    assert named["lift_time_2_s"] == pytest.approx(0.7)
    assert named["end_wait_s"] == pytest.approx(0.09)
    assert named["restore_feedback"] is True


def test_the_two_voltages_and_the_two_lifts_do_not_get_swapped():
    """Distinct synthetic sentinels prevent voltage, lift and timing index swaps from passing accidentally."""
    from mast.skills.builtins.readback import GetTipShaperConfig

    sentinel = [1.0, 1, 2.0, 3e-9, 4.0, 5.0, 6.0, 7e-9, 8.0, 9.0, 0]
    named = GetTipShaperConfig().execute(
        _tip_shaper_ctx(sentinel), {}).data["props_named"]
    assert named["bias_v"] == 2.0 and named["bias_lift_v"] == 5.0
    assert named["tip_lift_m"] == 3e-9 and named["lift_height_m"] == 7e-9
    assert named["restore_feedback"] is False


def test_raw_array_is_kept_alongside_the_names():
    """Naming is an interpretation; the array is the evidence for it."""
    from mast.skills.builtins.readback import GetTipShaperConfig

    res = GetTipShaperConfig().execute(_tip_shaper_ctx(_SYNTHETIC_TIP_SHAPER), {})
    assert res.data["props"] == _SYNTHETIC_TIP_SHAPER


def test_flag_codes_are_kept_next_to_the_bools():
    """``TipShape`` WRITES 2 for False (PropsSet: 0=no change/1=True/2=False)
    while PropsGet answers 0=False/1=True. The bool follows the GET convention;
    the raw code stays readable so the asymmetry is inspectable."""
    from mast.skills.builtins.readback import GetTipShaperConfig

    named = GetTipShaperConfig().execute(
        _tip_shaper_ctx(_SYNTHETIC_TIP_SHAPER), {}).data["props_named"]
    assert named["change_bias_raw"] == 0
    assert named["restore_feedback_raw"] == 1


def test_unexpected_length_names_nothing_and_says_why():
    """Conservative on a mismatch: naming the leading fields "as far as they go"
    looks authoritative while silently shifting everything after the divergence.
    An 11-field table applied to a 9-value answer is not 9 correct fields."""
    from mast.skills.builtins.readback import GetTipShaperConfig

    res = GetTipShaperConfig().execute(_tip_shaper_ctx([0.08, 0, 1.75]), {})
    assert res.data["props_named"] is None
    assert "3" in res.data["props_named_error"]
    assert "11" in res.data["props_named_error"]
    assert res.data["props"] == [0.08, 0, 1.75]


def test_names_match_the_parameters_tipshape_writes():
    """The read side and the write side must use ONE vocabulary.

    ``GetTipShaperConfig`` exists to be read BEFORE ``TipShape``; if the getter
    said ``bias`` where the setter says ``bias_v``, the agent would have to
    translate between them, which is the guessing this fix removes."""
    from mast.skills.builtins.readback import _TIP_SHAPER_PROPS
    from mast.skills.builtins.tip_shaper import TipShape

    written = {p.name for p in TipShape().metadata().parameters}
    read = {k for k, _unit in _TIP_SHAPER_PROPS}
    assert read <= written, sorted(read - written)


def test_protocol_field_order_is_recorded_in_the_source():
    """Pin the provenance. The order came from the protocol text, and the next
    person to touch this table must go back to the same place, not to memory."""
    import inspect

    import mast.skills.builtins.readback as rb

    src = inspect.getsource(rb)
    assert "tcp_protocol.txt" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
