"""v2 unit tests for the string-decoder fixes in ``mast.core.nanonis_patch``.

Found by probing ``ListScanMarkers`` on the instrument (2026-08-04, v6.1.1), which
returned ``IndexError: index out of range`` — i.e. a READ failure, NOT "there
are no markers on the map". The two are different answers and must not be
conflated.

``decodeStringPrepended`` (``NanonisClass.py:198-214``) has two independent
defects, both demonstrated below:

1. **The 4-byte length is not decoded big-endian at all.** It concatenates the
   DECIMAL representation of each byte and ``int()``s the result:
   ``int(str(b0) + str(b1) + str(b2) + str(b3))``. When the top three bytes are
   zero that is ``"0"+"0"+"0"+str(n) == str(n)``, so lengths ≤ 255 come out
   right **by coincidence**. At 256 it silently yields 10.
2. **``response[index + i]`` is a bare scalar index with no bounds check**, so a
   misaligned parse walks off the end and raises instead of letting
   ``parseError`` hand back the instrument's own message.

Why the crash is localised here even though the rig's exact bytes are unknown:
``IndexError: index out of range`` **with no prefix word** is what ``bytes``
scalar indexing raises (``list``/``tuple``/``str`` all prefix their message).
The only scalar ``response[...]`` accesses in the entire parser are
``NanonisClass.py`` lines 205, 208 and 220 — all three inside
``decodeStringPrepended`` / ``decodeSingularString``. Every other access is a
slice, and slices never raise.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_nanonis_patch_decode_string.py -x -v
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

_NO_ERROR = b"\x00" * 8


def _i(v: int) -> bytes:
    return struct.pack(">i", v)


def _arr(fmt: str, values) -> bytes:
    return b"".join(struct.pack(">" + fmt, v) for v in values)


def _strings(items) -> bytes:
    return b"".join(_i(len(s)) + s.encode() for s in items)


def _real_nanonis_class(*, patched: bool):
    """Load the genuine class from disk; see the twin helper in
    ``test_nanonis_patch_decode_array.py`` for why the conftest MagicMock makes
    this necessary."""
    import importlib.util
    import sysconfig

    # Installed copy first — see the twin loader in
    # ``test_nanonis_patch_decode_array.py`` for why sys.path order must not
    # decide this (many copies on disk, one per frozen build, growing per
    # release; ``_internal`` is excluded structurally, not by name).
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

    from mast.core import nanonis_patch

    # The parser patch itself predates this fix and ships in v6.1.1, so it goes
    # on in BOTH modes — otherwise "pristine" would also be missing the +*X
    # handling and the comparison would not isolate the string decoders.
    mod.Nanonis.parseGeneralResponse = nanonis_patch._patched_parseGeneralResponse
    if patched:
        for name in ("decodeArray", "decodeArrayPrepended",
                     "decodeStringPrepended", "decodeSingularString"):
            setattr(mod.Nanonis, name, getattr(nanonis_patch, "_patched_" + name))
    return mod.Nanonis


def _instance(cls):
    obj = cls.__new__(cls)
    obj.displayInfo = 0
    return obj


@pytest.fixture()
def nano():
    return _instance(_real_nanonis_class(patched=True))


@pytest.fixture()
def pristine():
    return _instance(_real_nanonis_class(patched=False))


# ── defect 1: the length decoder ─────────────────────────────────────────────


@pytest.mark.parametrize("n", [0, 1, 5, 100, 255, 256, 1000, 70000])
def test_string_length_is_decoded_big_endian(nano, n):
    """Every length round-trips, including the ones upstream gets wrong."""
    raw = _i(n) + b"x" * n
    got = nano.decodeStringPrepended(raw, 0, 1)
    assert len(got) == 1
    assert len(got[0]) == n


def test_upstream_length_decoder_is_wrong_at_256(pristine):
    """The "before" state, observed rather than asserted from memory.

    ≤ 255 works by coincidence; 256 silently decodes as 10 — a 246-byte
    under-read that shifts every subsequent field. This is the one that would
    have stayed invisible: no exception, just a short string.
    """
    assert len(pristine.decodeStringPrepended(_i(255) + b"x" * 255, 0, 1)[0]) == 255
    assert len(pristine.decodeStringPrepended(_i(256) + b"x" * 256, 0, 1)[0]) == 10
    with pytest.raises(IndexError):
        pristine.decodeStringPrepended(_i(1000) + b"x" * 1000, 0, 1)


# ── defect 2: no bounds check ────────────────────────────────────────────────


def test_truncated_body_does_not_raise_index_error(nano):
    """A declared length longer than the body truncates instead of raising.

    The point is not the truncated string — it is that ``parseError`` further
    down still gets to run, so the operator can be told what the INSTRUMENT
    said rather than being handed a Python traceback.
    """
    raw = _i(50) + b"only twelve\x00"      # declares 50 bytes, supplies 12
    got = nano.decodeStringPrepended(raw, 0, 1)
    assert got[0].startswith("only twelve")
    assert len(got[0]) == 12


def test_singular_string_is_bounds_checked(nano):
    """``*-c`` takes the same treatment — it is on the scan hot path
    (``Scan.FrameDataGrab``), where a misparse must not become an IndexError."""
    assert nano.decodeSingularString(b"abc", 0, 99) == "abc"
    assert nano.decodeSingularString(b"abc", 10, 4) == ""
    assert nano.decodeSingularString(b"abc", 0, 0) == ""


def test_bytes_index_error_message_is_unique_to_scalar_indexing():
    """Pins the deduction that localises the rig crash.

    ``IndexError: index out of range`` with NO prefix word comes only from
    ``bytes`` scalar indexing; list/tuple/str all prefix theirs. That is why the
    rig failure had to be inside the two string decoders — every other access in
    the parser is a slice, and slices never raise. If a future Python changes
    these messages, this test says so instead of the deduction quietly rotting.
    """
    with pytest.raises(IndexError, match=r"^index out of range$"):
        b"ab"[5]
    with pytest.raises(IndexError, match=r"list index out of range"):
        [1, 2][5]


# ── the fields the fix must NOT change ───────────────────────────────────────


def test_well_formed_string_arrays_decode_identically(nano, pristine):
    """Byte-for-byte parity with upstream on everything that already worked.

    ``channel_names`` decoding correctly on the instrument (2026-08-04) is the thing
    this must not break.
    """
    names = ["Current (A)", "Bias (V)", "Z (m)"]
    blob = _strings(names)
    body = (_i(3) + _arr("i", [0, 1, 2]) + _i(len(blob)) + _i(len(names))
            + blob + _NO_ERROR)
    spec = ["i", "*i", "i", "i", "*+c"]

    _e1, _r1, v_new = nano.parseGeneralResponse(body, spec)
    _e2, _r2, v_old = pristine.parseGeneralResponse(body, spec)
    assert v_new[4] == names
    assert v_old[4] == names          # upstream already handled this case
    assert v_new[4] == v_old[4]


def test_latin1_keeps_len_str_equal_len_bytes(nano):
    """Decoding must stay one-byte-one-char.

    Upstream builds strings with ``chr(byte)``, i.e. latin-1. Of the three
    string branches only ``**c`` feeds a DECODED length back into the byte
    counter (``counter += 4 + len(item)``; ``*+c`` and ``+*c`` both advance by a
    declared byte size). ``Marks.PointsGet`` is a ``**c``, so decoding as UTF-8
    would make ``len(str) < len(bytes)`` on any non-ASCII byte and shift every
    following field. Pinned because "just use utf-8" looks like a harmless
    modernisation — and because the ``+*c`` branch legitimately DOES use utf-8,
    which makes the inconsistency look like an oversight.
    """
    payload = b"\xe9\xe8\xff"          # 3 bytes, non-ASCII
    got = nano.decodeStringPrepended(_i(3) + payload, 0, 1)[0]
    assert len(got) == 3
    assert got == payload.decode("latin-1")


# ── the reported symptom ─────────────────────────────────────────────────────


def test_marks_pointsget_zero_markers_is_not_the_bug(nano, pristine):
    """Rules out the first hypothesis: zero-length arrays are fine either way.

    ``Marks.PointsGet`` is ``["i","**f","**f","i","**c","**I","**I"]``; with no
    markers every array is length 0. Both the patched and the unpatched decoders
    return empty lists without complaint, so "zero-length ``**f``/``**I``" is
    NOT what broke ``ListScanMarkers`` on the instrument.
    """
    spec = ["i", "**f", "**f", "i", "**c", "**I", "**I"]
    body = _i(0) + _i(0) + _NO_ERROR
    for who in (nano, pristine):
        err, _raw, v = who.parseGeneralResponse(body, spec)
        assert err == ""
        assert v == [0, [], [], 0, [], [], []]


def test_marks_pointsget_with_markers_round_trips(nano):
    """And a populated reply decodes fully — arrays bare, texts intact."""
    spec = ["i", "**f", "**f", "i", "**c", "**I", "**I"]
    texts = _strings(["A", "B"])
    body = (_i(2) + _arr("f", [1e-9, 2e-9]) + _arr("f", [3e-9, 4e-9])
            + _i(len(texts)) + texts
            + _arr("I", [255, 128]) + _arr("I", [1, 1]) + _NO_ERROR)
    err, _raw, v = nano.parseGeneralResponse(body, spec)
    assert err == ""
    assert v[0] == 2
    assert v[1] == pytest.approx([1e-9, 2e-9], rel=1e-6)
    assert v[4] == ["A", "B"]
    assert v[5] == [255, 128] and v[6] == [1, 1]


def test_error_reply_still_yields_the_instrument_message(nano):
    """An error reply (zeroed data fields + error section) must come back as
    TEXT, not as an exception — for every verb shape.

    This is the layout ``parseError``'s ``margin = 8`` implies (error status +
    size sit AFTER the data fields), and it is what the rig produced for
    ``GenSwp.AcqChsGet``: a clean "Cannot access ..." string.
    """
    msg = b"Cannot access the Marks module. Please make sure it is running."
    err_section = _i(1) + _i(len(msg)) + msg
    for spec, zeros in (
        (["i", "**f", "**f", "i", "**c", "**I", "**I"], _i(0) + _i(0)),
        (["i", "**f", "**f", "**f", "**f", "**I", "**I"], _i(0)),
        (["i", "*i", "i", "i", "*+c"], _i(0) + _i(0) + _i(0)),
    ):
        err, _raw, _v = nano.parseGeneralResponse(zeros + err_section, spec)
        assert "Cannot access the Marks module" in err
