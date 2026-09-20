"""The instrument profile is saved WHOLE — a key the form forgets is a key deleted.

``POST /api/settings`` replaces ``instrument_profile`` wholesale with whatever the
settings form sends. ``SettingsPage.tsx`` rebuilds that object from three hand-
maintained lists (``INSTRUMENT_NUM_FIELDS`` / ``INSTRUMENT_CHOICE_KEYS`` /
``INSTRUMENT_CALIB_KEYS``), so any profile key absent from all three is **erased
the next time the operator edits an unrelated field**.

The failure mode is silent and delayed: nothing errors, the value simply reads
back as ``None`` from then on, and whatever depended on it degrades to its
"cannot tell" branch. It has already happened to the qPlus free-oscillation
baseline — the denominator of the amplitude crash criterion — which sat in
``instrument_profile`` but in none of the three lists, so the detector reported
``no_baseline`` forever while looking perfectly healthy.

This is the same shape as ``SettingsStore.KNOWN_KEYS`` and
``override_store._ALL_FILES`` before it. The repo keeps re-learning it; this test
is how it stops.

Reads the TSX as text, like ``test_kind_style_frontend_parity``: there is no build
step here that could evaluate it, and comparing the two SOURCES is the point.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mast.core.instrument_profile import ALL_KEYS, EDITABLE_KEYS


def _settings_page_source() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        cand = p / "frontend" / "src" / "pages" / "SettingsPage.tsx"
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
        p = p.parent
    pytest.skip("frontend/ not present in this checkout")


def _num_field_keys(src: str) -> set[str]:
    body = src.split("const INSTRUMENT_NUM_FIELDS", 1)[1].split("];", 1)[0]
    return set(re.findall(r"\{\s*key:\s*\"([^\"]+)\"", body))


def _string_array(src: str, name: str) -> set[str]:
    body = src.split(f"const {name} = [", 1)[1].split("];", 1)[0]
    out = set(re.findall(r"\"([^\"]+)\"", body))
    # INSTRUMENT_CHOICE_KEYS holds *constants*, not literals — resolve them.
    for ident in re.findall(r"\b([A-Z][A-Z0-9_]*_KEY)\b", body):
        m = re.search(rf'const {ident} = "([^"]+)"', src)
        if m:
            out.add(m.group(1))
    return out


def _object_list_keys(src: str, name: str) -> set[str]:
    """``const NAME: {...}[] = [{ key: "...", ... }]`` 里的 key 字面量。"""
    body = src.split(f"const {name}", 1)[1].split("];", 1)[0]
    return set(re.findall(r"\{\s*key:\s*\"([^\"]+)\"", body))


def _form_keys(src: str) -> set[str]:
    return (
        _num_field_keys(src)
        | _object_list_keys(src, "INSTRUMENT_TEXT_FIELDS")
        | _string_array(src, "INSTRUMENT_CHOICE_KEYS")
        | _string_array(src, "INSTRUMENT_CALIB_KEYS")
    )


def test_every_profile_key_survives_a_save():
    """Any key the backend can hold must be echoed back by the form."""
    missing = sorted(set(ALL_KEYS) - _form_keys(_settings_page_source()))
    assert not missing, (
        "instrument_profile keys the settings form does not echo back: "
        f"{missing} — the store is whole-replace, so editing ANY other field "
        "deletes these. Add each to INSTRUMENT_NUM_FIELDS (with a UI row), "
        "INSTRUMENT_CHOICE_KEYS (with a UI block), or INSTRUMENT_CALIB_KEYS "
        "(runtime-written, shown read-only)."
    )


def test_the_qplus_baseline_specifically_survives():
    """Named explicitly because this is the one that actually broke.

    The baseline is re-earned only by a verified retract, so losing it costs a
    real physical operation — and until then every amplitude verdict is
    ``no_baseline``, i.e. "cannot tell", which callers must not read as "no crash"."""
    keys = _form_keys(_settings_page_source())
    for key in ("qplus_amplitude_baseline", "qplus_amplitude_signal_index"):
        assert key in keys, f"{key} would be wiped by any unrelated settings edit"


def test_the_form_has_no_keys_the_backend_would_drop():
    """A key the form sends but ``sanitize()`` does not know is silently discarded.

    Harmless at runtime, but it means a UI control that appears to persist and
    does not — which reads to the operator as "my setting did not take"."""
    extra = sorted(_form_keys(_settings_page_source()) - set(ALL_KEYS))
    assert not extra, (
        f"settings form sends profile keys instrument_profile.sanitize() drops: "
        f"{extra}"
    )


def test_clearable_calibration_is_a_strict_subset():
    """「清除标定」must not reach beyond the dI/dV calibration it names.

    Echoing a key back on save and offering to delete it are different questions.
    The qPlus baseline is echoed (so an unrelated edit cannot lose it) but is not
    clearable by that button, because it comes from a different mechanism."""
    src = _settings_page_source()
    clearable = _string_array(src, "INSTRUMENT_CLEARABLE_CALIB_KEYS")
    calib = _string_array(src, "INSTRUMENT_CALIB_KEYS")
    assert clearable, "INSTRUMENT_CLEARABLE_CALIB_KEYS not found in SettingsPage.tsx"
    assert clearable <= calib, (
        f"clearable keys outside the echoed calibration list: "
        f"{sorted(clearable - calib)}"
    )
    assert "qplus_amplitude_baseline" not in clearable, (
        "the dI/dV clear button must not throw away the qPlus baseline"
    )


def test_text_fields_are_not_routed_through_the_number_only_branches():
    """A free-text key echoed via the CALIB list is a key that silently vanishes.

    ``buildBase()`` copies calibration keys only when ``typeof === "number"``, and
    the backend coerces ``_CALIB_KEYS`` with ``float()``. Either path turns a model
    string into nothing — the qPlus baseline incident with a different key type.
    Text keys must live in their own list, with their own echo branch."""
    from mast.core.instrument_profile import _TEXT_SPEC

    src = _settings_page_source()
    calib = _string_array(src, "INSTRUMENT_CALIB_KEYS")
    nums = _num_field_keys(src)
    text_keys = set(_TEXT_SPEC)
    assert text_keys, "no free-text profile fields declared"
    assert not (text_keys & calib), (
        f"free-text keys sitting in INSTRUMENT_CALIB_KEYS: {sorted(text_keys & calib)}"
    )
    assert not (text_keys & nums), (
        f"free-text keys sitting in INSTRUMENT_NUM_FIELDS: {sorted(text_keys & nums)}"
    )
    assert text_keys <= _object_list_keys(src, "INSTRUMENT_TEXT_FIELDS"), (
        "every _TEXT_SPEC key needs a row in INSTRUMENT_TEXT_FIELDS"
    )


def test_bias_polarity_is_reachable_from_the_form():
    """Which side the bias sits on decides whether positive bias probes empty or
    filled states. Defaulting it silently is how a whole dataset gets read backwards."""
    assert "bias_applied_to" in _form_keys(_settings_page_source())


def test_editable_keys_are_all_operator_reachable():
    """Every CONFIG/CHOICE key is meant to be set by a human; if the form has no
    control for it, the default is the only value it will ever have."""
    keys = _form_keys(_settings_page_source())
    missing = sorted(set(EDITABLE_KEYS) - keys)
    assert not missing, f"operator-editable profile keys with no UI control: {missing}"


def test_every_declared_numeric_field_is_actually_rendered():
    """Declaring a field is not the same as putting it on screen.

    ``INSTRUMENT_NUM_FIELDS`` does two jobs: it drives ``buildBase()`` (so a key
    listed there survives a save) AND it supplies the spec for ``numRow(key)``.
    But the rows are laid out by hand-written key lists per section, so a field
    can be declared — passing every check above, since the key IS in the source —
    and still have no control anywhere in the form.

    The result reads exactly like a backend bug: the operator is told to fill
    something in, cannot find it, and the feature that needs it stays disabled.
    (Hit while adding the approach preset keys on 2026-08-03: all five were
    declared, none were rendered, and every existing parity test passed.)
    """
    src = _settings_page_source()
    declared = _num_field_keys(src)
    # Keys named inside any `[...].map(numRow)` list.
    rendered: set[str] = set()
    for block in re.findall(r"\[([^\]]*?)\]\.map\(numRow\)", src, re.S):
        rendered.update(re.findall(r"\"([^\"]+)\"", block))
    missing = sorted(declared - rendered)
    assert not missing, (
        f"declared in INSTRUMENT_NUM_FIELDS but never rendered: {missing} — "
        "add each to one of the `[...].map(numRow)` lists, or the operator has "
        "no way to set it."
    )
