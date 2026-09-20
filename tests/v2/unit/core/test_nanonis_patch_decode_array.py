"""v2 unit test for the array-unpacking fix in ``mast.core.nanonis_patch``.

nanonis_spm v1.0.9 appends ``struct.unpack``'s return value to the decoded list
without unwrapping it (``NanonisClass.py:186-196`` / ``:224-236``), and
``struct.unpack`` **always** returns a tuple. So every numeric array field came
back as ``[(0,), (30,)]`` instead of ``[0, 30]`` — not for one verb, for
``*i`` ``*I`` ``*f`` ``*d`` ``**f`` ``**I`` ``**i``, i.e. all of them
(KNOWN_ISSUES §2.20 / §2.21).

**These tests feed real wire bytes through the parser**, deliberately, because
that is the layer the defect lives at. Every skill test in the tree stubs
``return_value`` directly and therefore cannot see this class of bug at all —
that is exactly how it survived to the instrument twice.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_nanonis_patch_decode_array.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import struct

import pytest


# ── wire-body builders ───────────────────────────────────────────────────────
#
# A Nanonis response body is the concatenated fields followed by 8 trailing
# bytes (error status int32 + error-description size int32). ``parseError``
# decodes everything past those 8 bytes as the error string, so 8 zero bytes
# means "no error".

_NO_ERROR = b"\x00" * 8


def _i(v: int) -> bytes:
    return struct.pack(">i", v)


def _arr(fmt: str, values) -> bytes:
    return b"".join(struct.pack(">" + fmt, v) for v in values)


def _strings(items) -> bytes:
    """1D string array: each item is a 4-byte length followed by its bytes."""
    return b"".join(_i(len(s)) + s.encode() for s in items)


def _real_nanonis_class(*, patched: bool):
    """Load the GENUINE ``nanonis_spm.Nanonis`` from disk, once per call.

    ``tests/conftest.py`` replaces ``sys.modules["nanonis_spm"]`` with a
    MagicMock so the suite runs without hardware — which also means the class
    ``nanonis_patch.apply()`` installs onto is a mock, and a mock would happily
    "pass" any assertion made about decoded values. Same reasoning (and same
    loader) as ``test_nanonis_patch_send.py::_real_nanonis_class``.

    Loading a fresh module object per call is what lets one test hold a patched
    class and a pristine one side by side, which is how the "before" behaviour
    below is observed rather than asserted from memory.
    """
    import importlib.util
    import sysconfig

    # Resolve the INSTALLED copy first, not "whatever sys.path hits first".
    # There are many ``nanonis_spm/NanonisClass.py`` on this disk: the installed
    # one, plus one inside every frozen build under ``dist/*/_internal/`` — and
    # most of those are OTA increment baselines, so the set GROWS by one per
    # release. Excluding ``_internal`` is therefore structural (it is
    # PyInstaller's layout marker) rather than a list of directory names, which
    # would go stale next month.
    #
    # They are byte-identical today, so the old sys.path scan happened to be
    # right. It stops being right the first time someone applies the
    # after-every-pip-install parser patch to one copy and not the others —
    # and then path order silently decides which bytes this test measured.
    # Shape taken from fix-scan-semantics' ``53d2cbd``; kept identical on
    # purpose so the two loaders cannot drift apart.
    candidates = [Path(sysconfig.get_paths()["purelib"])]
    candidates += [Path(b) for b in sys.path if "_internal" not in b]
    candidates.append(Path(sys.executable).resolve().parents[1])
    for base in candidates:
        cand = base / "nanonis_spm" / "NanonisClass.py"
        if cand.exists():
            break
    else:
        pytest.skip("nanonis_spm package not installed on disk")

    spec = importlib.util.spec_from_file_location("_real_nanonis_class", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if patched:
        from mast.core import nanonis_patch

        mod.Nanonis.parseGeneralResponse = nanonis_patch._patched_parseGeneralResponse
        mod.Nanonis.decodeArray = nanonis_patch._patched_decodeArray
        mod.Nanonis.decodeArrayPrepended = nanonis_patch._patched_decodeArrayPrepended
    return mod.Nanonis


def _instance(cls):
    obj = cls.__new__(cls)          # the parser never touches the socket
    obj.displayInfo = 0
    return obj


@pytest.fixture()
def nano():
    """A patched ``Nanonis`` instance with no socket."""
    return _instance(_real_nanonis_class(patched=True))


# ── the prepended path: decodeArrayPrepended (``*i`` ``*f`` ``*d``) ──────────


def test_scan_buffer_get_int_array_is_bare_ints(nano):
    """``Scan.BufferGet`` — the real-machine failure of §2.20.

    ResponseTypes ``["i", "*i", "i", "i"]``. On the rig this returned
    ``[(0,), (30,)]`` and ``SetScanBuffer``'s ``[int(c) for c in raw_ch]`` died
    with "int() argument must be ... not 'tuple'" on the FIRST real call.
    """
    body = _i(2) + _arr("i", [0, 30]) + _i(256) + _i(256) + _NO_ERROR
    err, _raw, variables = nano.parseGeneralResponse(
        body, ["i", "*i", "i", "i"])

    assert err == ""
    assert variables[0] == 2
    assert variables[1] == [0, 30]          # NOT [(0,), (30,)]
    assert variables[2] == 256
    assert variables[3] == 256
    # The exact expression that crashed on the instrument.
    assert [int(c) for c in variables[1]] == [0, 30]


def test_genswp_acqchsget_full_reply(nano):
    """``GenSwp.AcqChsGet`` — §2.21's "crashes every single time" entry.

    ResponseTypes ``["i", "*i", "i", "i", "*+c"]``. ``sweep.py`` does
    ``[int(x) for x in idxs]``; before the fix that raised ``TypeError``, which
    ``ExecutionContext.run``'s catch-all turned into ``success=False`` — so the
    skill failed on every call and said nothing about why.
    """
    names = ["Current (A)", "Bias (V)"]
    names_blob = _strings(names)
    body = (
        _i(2)
        + _arr("i", [0, 24])
        + _i(len(names_blob))     # names size in bytes
        + _i(len(names))          # number of name strings
        + names_blob
        + _NO_ERROR
    )
    err, _raw, variables = nano.parseGeneralResponse(
        body, ["i", "*i", "i", "i", "*+c"])

    assert err == ""
    assert variables[1] == [0, 24]
    assert [int(x) for x in variables[1]] == [0, 24]
    # The string array must still decode — the fix must not shift the counter.
    assert variables[4] == names


def test_prepended_float_array_is_bare_floats(nano):
    """``Osci2T.TimebaseGet`` shape ``["i", "i", "*f"]`` (as patched by MAST).

    ``envhistory/zburst.py::_select_timebase`` does ``[float(x) for x in d[2]]``
    inside a ``try``; before the fix that raised, was swallowed, and the
    function returned ``0.0`` — so ``Osci2T_TimebaseSet`` was never reached and
    every z-burst ran on whatever timebase the operator last left behind.
    """
    timebases = [5e-05, 0.0001, 0.001]
    body = _i(0) + _i(len(timebases)) + _arr("f", timebases) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(body, ["i", "i", "*f"])

    got = variables[2]
    assert all(isinstance(v, float) for v in got)
    assert got == pytest.approx(timebases, rel=1e-6)
    # _select_timebase's own expression, and its argmin.
    values = [float(x) for x in got]
    assert min(range(len(values)), key=lambda i: values[i]) == 0


def test_prepended_double_array_keeps_eight_byte_stride(nano):
    """``*d`` — ``SpectrumAnlzr.DataGet`` as respecced by MAST's patch.

    float64 uses an 8-byte stride in both the helper and the parser's own byte
    counter; this pins that the unwrap did not disturb it.
    """
    ys = [1.5, -2.25, 3.125, 4.0]
    body = _i(0) + _i(0) + _i(len(ys)) + _arr("d", ys) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(
        body, ["i", "i", "i", "*d"])

    assert variables[3] == ys
    assert all(isinstance(v, float) for v in variables[3])


def test_prepended_unsigned_array_is_bare_ints(nano):
    """``*I`` takes the same path; pinned so the fix is not read as "*i only"."""
    body = _i(3) + _arr("I", [0, 7, 4294967295]) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(body, ["i", "*I"])
    assert variables[1] == [0, 7, 4294967295]


# ── the universal-length path: decodeArray (``**f`` ``**I`` ``**i``) ─────────


def test_pattern_cloud_get_float_arrays_are_bare_floats(nano):
    """``Pattern.CloudGet`` — ``["i", "**f", "**f"]``, §2.21's ``**f`` entry.

    ``**X`` takes its length from ``Variables[0]``, so both coordinate arrays
    are sized by the leading point count. Silently wrong rather than loud:
    ``pattern.py`` reported ``[(0.0,), (1e-09,)]`` as metres to the model.
    """
    xs = [0.0, 1e-09, 2e-09]
    ys = [0.0, -1e-09, -2e-09]
    body = _i(len(xs)) + _arr("f", xs) + _arr("f", ys) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(
        body, ["i", "**f", "**f"])

    assert variables[0] == 3
    assert variables[1] == pytest.approx(xs, rel=1e-6)
    assert variables[2] == pytest.approx(ys, rel=1e-6)
    assert all(isinstance(v, float) for v in variables[1])
    assert all(isinstance(v, float) for v in variables[2])


def test_universal_int_array_is_bare_ints(nano):
    """``**I`` — five verbs in the library use it."""
    body = _i(2) + _arr("I", [11, 22]) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(body, ["i", "**I"])
    assert variables[1] == [11, 22]


def test_decode_array_stride_follows_calcsize(nano):
    """``decodeArray`` derives its stride from the format, not a hardcoded 4.

    No ``**d`` spec ships in nanonis_spm today, so this is not reachable through
    ``parseGeneralResponse`` — but the parser **already** advances its counter
    by ``calcsize`` for this branch, so a hardcoded 4 here would have meant a
    correct counter and a ``struct.error`` one line earlier. Called directly.
    """
    values = [1.0, 2.5, -3.75]
    assert nano.decodeArray(_arr("d", values), 0, len(values), "d") == values


# ── the fields the fix must NOT touch ────────────────────────────────────────


def test_scalars_strings_and_2d_are_unchanged(nano):
    """Scalar / string / 2-D fields were already unwrapped; pin that they stay.

    ``2f`` goes through ``np.reshape``, which flattens the tuples on its own —
    that is why it was never part of this bug and must not become part of the
    fix.
    """
    import numpy as np

    body = (
        _i(2) + _i(2)                       # rows, cols
        + _arr("f", [1.0, 2.0, 3.0, 4.0])   # 2f
        + _i(5) + b"hello"                  # *-c: length then chars
        + _NO_ERROR
    )
    _err, _raw, variables = nano.parseGeneralResponse(
        body, ["i", "i", "2f", "i", "*-c"])

    assert variables[0] == 2 and variables[1] == 2
    assert isinstance(variables[2], np.ndarray)
    assert variables[2].shape == (2, 2)
    np.testing.assert_allclose(variables[2], [[1.0, 2.0], [3.0, 4.0]])
    assert variables[4] == "hello"


def test_plus_star_branch_still_unwraps(nano):
    """``+*i`` was already correct in MAST's parser patch — it must stay so.

    This branch is the reason the fix belongs here: the module already unwrapped
    in one of its two array paths and not the other.
    """
    body = _i(2) + _arr("i", [4, 9]) + _NO_ERROR
    _err, _raw, variables = nano.parseGeneralResponse(body, ["+*i"])
    assert variables[0] == [4, 9]


# ── the patch is a patch ─────────────────────────────────────────────────────


def test_unpatched_upstream_really_does_return_tuples():
    """The "before" state, observed rather than asserted from memory.

    If a future nanonis_spm fixes this upstream, this test goes red and tells
    the next reader that the patch has become redundant — the alternative is a
    patch nobody can prove is still doing anything.
    """
    pristine = _instance(_real_nanonis_class(patched=False))
    body = _i(2) + _arr("i", [0, 30]) + _i(256) + _i(256) + _NO_ERROR
    _err, _raw, variables = pristine.parseGeneralResponse(
        body, ["i", "*i", "i", "i"])

    assert variables[1] == [(0,), (30,)]
    with pytest.raises(TypeError):
        [int(c) for c in variables[1]]      # what the instrument hit


def test_apply_and_revert_swap_both_helpers():
    """``apply()`` / ``revert()`` must cover BOTH decode helpers.

    Identity check rather than behavioural: under ``tests/conftest.py`` the
    class being patched is a MagicMock, so only the binding is observable here.
    Behaviour is covered by the real-class tests above.
    """
    from mast.core import nanonis_patch
    from nanonis_spm import Nanonis

    nanonis_patch.apply()
    assert Nanonis.decodeArray is nanonis_patch._patched_decodeArray
    assert Nanonis.decodeArrayPrepended is nanonis_patch._patched_decodeArrayPrepended
    try:
        nanonis_patch.revert()
        assert Nanonis.decodeArray is nanonis_patch._original_decodeArray
        assert (Nanonis.decodeArrayPrepended
                is nanonis_patch._original_decodeArrayPrepended)
    finally:
        nanonis_patch.apply()
    assert Nanonis.decodeArray is nanonis_patch._patched_decodeArray
