"""A persisted scan frame must keep its image shape, or nothing can plot it.

Found by running the minimal experiment loop end to end on the real LLM
(2026-07-28). The run reached the report and produced a full, numerically
correct draft whose Results section opened with:

    **No figure is included in this report.** The saved frame was stored
    flattened as a 1-D array of shape (65536,) rather than a 2-D (256, 256)
    image, so the plotting routine (`plot_scan`) refused it and no image could
    be rendered.

The rows/cols were in the ``Scan_FrameDataGrab`` reply and were discarded on the
way to disk: ``parse_frame_grab`` ravels unless asked for ``shape_2d``, and
``GrabScanFrameData`` — the one skill whose entire job is to PERSIST the frame —
did not ask. Every number in that report was real; only the picture was missing,
which is the worst shape a data pipeline can fail in: it looks like it worked.

Pinned here:

* the persisted .npy is 2-D, with the dims the instrument reported;
* ``plot_scan`` renders it (the chain scan → figure is closed);
* crash detection still gets what IT needs (it only reads variance), so the
  fix does not change the tip-safety path.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_frame_persist_shape.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROWS, COLS = 64, 96          # deliberately NOT square: a raveled array cannot
                             # be recovered by guessing sqrt(n).


def _topo() -> np.ndarray:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:ROWS, 0:COLS]
    return ((np.sin(x * 0.5) * np.sin(y * 0.5)) * 3e-11
            + x * 2e-13 + rng.normal(0, 4e-12, (ROWS, COLS)))


class _Rec:
    """A REAL Scan_FrameDataGrab reply body:
    ``[name_len, name, rows, cols, data_2D, direction]``."""

    error = ""

    def __init__(self, arr):
        self.return_value = ("", b"", [1, "Z", ROWS, COLS,
                                       arr.astype(np.float64), 1])


class _Ctx:
    def __init__(self, arr):
        self._arr = arr
        self.calls: list = []

    def safe_call(self, verb, *a, **kw):
        self.calls.append((verb, a))
        return _Rec(self._arr)


# ════════════════════════════════════════════════════════════════════════════
# The persisted frame
# ════════════════════════════════════════════════════════════════════════════

def test_the_saved_npy_is_two_dimensional(tmp_path):
    from mast.skills.builtins.scan_frame import GrabScanFrameData

    out = tmp_path / "frame.npy"
    res = GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1,
                        "save_path": str(out)})
    assert res.success, res.error
    arr = np.load(out)
    assert arr.ndim == 2, f"存成了 {arr.ndim} 维 —— 下游画不出图"
    assert arr.shape == (ROWS, COLS)


def test_the_result_reports_the_real_shape(tmp_path):
    """The caller is told the shape, so a wrong one is visible in the trace."""
    from mast.skills.builtins.scan_frame import GrabScanFrameData

    out = tmp_path / "frame.npy"
    res = GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1,
                        "save_path": str(out)})
    assert res.data["shape"] == [ROWS, COLS]
    assert res.data["n_samples"] == ROWS * COLS


def test_the_values_survive_the_round_trip(tmp_path):
    from mast.skills.builtins.scan_frame import GrabScanFrameData

    topo = _topo()
    out = tmp_path / "frame.npy"
    GrabScanFrameData().execute(_Ctx(topo), {"channel_index": 0,
                                             "direction": 1,
                                             "save_path": str(out)})
    assert np.allclose(np.load(out), topo)


# ════════════════════════════════════════════════════════════════════════════
# The chain it unblocks
# ════════════════════════════════════════════════════════════════════════════

def test_plot_scan_can_render_the_persisted_frame(tmp_path, monkeypatch):
    """The whole point: scan → figure, closed."""
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    from mast.agents.data_processing.tools import plot_scan
    from mast.skills.builtins.scan_frame import GrabScanFrameData

    out = tmp_path / "frame.npy"
    GrabScanFrameData().execute(_Ctx(_topo()), {"channel_index": 0,
                                                "direction": 1,
                                                "save_path": str(out)})
    msg = str(plot_scan.invoke(
        {"args": {"path": str(out), "label": "topo"},
         "name": "plot_scan", "type": "tool_call",
         "id": "t1"}))
    assert "failed" not in msg, msg
    figs = list((tmp_path / "figures").glob("*.png"))
    assert figs, f"没出图: {msg}"
    assert figs[0].stat().st_size > 1000


def test_a_flat_frame_is_still_refused_not_guessed(tmp_path, monkeypatch):
    """Old 1-D files must NOT be reshaped by guessing — a wrong shape silently
    renders a wrong picture, which is worse than the honest refusal."""
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    from mast.agents.data_processing.tools import plot_scan

    flat = tmp_path / "flat.npy"
    np.save(flat, _topo().ravel())
    msg = str(plot_scan.invoke(
        {"args": {"path": str(flat), "label": "flat"},
         "name": "plot_scan", "type": "tool_call",
         "id": "t1"}))
    assert "failed" in msg
    assert "维" in msg or "dim" in msg.lower()


# ════════════════════════════════════════════════════════════════════════════
# The path that must NOT change: tip-crash detection
# ════════════════════════════════════════════════════════════════════════════

def test_two_grabs_do_not_share_one_path(tmp_path, monkeypatch):
    """A default-named frame is a shared mutable path, and this repo has been
    burned by one twice (composite sidecars, then the milestone PNGs).

    Observed in a real end-to-end run 2026-07-28: the default name was
    ``frame_ch0_dir1.npy``, so data_processing resolved that path BEFORE this
    run's grab landed, analysed a LEFTOVER from an earlier run, reported
    「实际为一维 8 元素全零数组」and bounced the whole task back for a re-scan —
    and when the grab did land it overwrote the earlier run's frame.
    """
    import mast.skills.builtins.scan_frame as SF

    monkeypatch.setattr(SF, "_frames_dir", lambda: tmp_path)
    first = SF.GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1})
    second = SF.GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1})
    assert first.success and second.success
    assert first.data["frame_path"] != second.data["frame_path"], (
        "两次抓帧写到了同一个文件 —— 后一次会毁掉前一次，前一次也可能被当成后一次读走")
    assert len(list(tmp_path.glob("*.npy"))) == 2
    # The channel/direction must still be readable from the name — that is what
    # makes a stray file identifiable months later.
    assert "ch0" in Path(first.data["frame_path"]).name
    assert "dir1" in Path(first.data["frame_path"]).name


def test_two_grabs_in_the_SAME_millisecond_do_not_share_one_path(tmp_path, monkeypatch):
    """The stamp alone is not enough, and the previous test only passed because
    saving the array happened to cross a millisecond boundary.

    Measured: 2000 consecutive reads of ``time.time()`` return two distinct
    values, so back-to-back grabs land in the same millisecond as a rule, not as
    an edge case. Freezing the clock makes the collision deterministic.
    """
    import mast.skills.builtins.scan_frame as SF

    monkeypatch.setattr(SF, "_frames_dir", lambda: tmp_path)
    monkeypatch.setattr(SF.time, "time", lambda: 1753800000.0)

    first = SF.GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1})
    second = SF.GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1})

    assert first.success and second.success
    assert first.data["frame_path"] != second.data["frame_path"]
    assert len(list(tmp_path.glob("*.npy"))) == 2
    assert "ch0" in Path(second.data["frame_path"]).name
    assert "dir1" in Path(second.data["frame_path"]).name


def test_an_explicit_save_path_is_still_honoured(tmp_path):
    """Run-uniqueness is the DEFAULT, not an override: a caller that names the
    file still gets exactly that file."""
    from mast.skills.builtins.scan_frame import GrabScanFrameData

    out = tmp_path / "explicit.npy"
    res = GrabScanFrameData().execute(
        _Ctx(_topo()), {"channel_index": 0, "direction": 1,
                        "save_path": str(out)})
    assert res.data["frame_path"] == str(out) and out.exists()


def test_crash_detection_still_sees_variance():
    """CheckScanForCrash only reads ptp/NaN — it works on either shape, and the
    fix must not have moved the tip-safety verdict."""
    from mast.skills.builtins.scan_frame import CheckScanForCrash

    good = CheckScanForCrash().execute(_Ctx(_topo()),
                                       {"channel_indices": "0"})
    assert good.success, good.error
    assert not good.data.get("crash_indicator"), "好数据被误判为 crash"
    assert good.data.get("status") == "ok"

    bad = CheckScanForCrash().execute(_Ctx(np.zeros((ROWS, COLS))),
                                      {"channel_indices": "0"})
    assert bad.data.get("crash_indicator") is True, (
        "零方差没有被判为 crash —— 针尖保护被这次改动动到了")
    assert bad.data.get("status") == "crash"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
