"""Data-Processing agent — fft_2d mirror-peak de-dup  + prompt/signature
consistency , plus a regression pin for the numpy sandbox reductions /
tuple-unpack (#41/#42, already fixed in the session-#6 hardening rewrite).

#45  fft_2d on a real-valued STM topograph used to report every physical Bragg
     vector TWICE — once at +k and once at its centro-symmetric mirror -k —
     inflating the apparent lattice-vector count. The fix restricts the peak
     search to a single half-plane so each ±k pair contributes exactly one peak.

#44  prompts.py advertised tool signatures (`fft_2d(sxm_path, channel)`,
     `load_scan(sxm_path)` returning "channels", run_numpy_snippet "stub only",
     detect_defects "mean diameter", …) that no longer matched the real tools.
     This pins the regenerated prompt against the live tool signatures so they
     can't silently drift apart again.

Runs with no LLM and no network — every input is a literal array written to a
temp .npy file.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/agents/data_processing/test_fft_and_prompt.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap (force the MASTv2 copy of `mast`, not v1 at repo-root) ─────
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import re

import numpy as np
import pytest

from mast.agents.data_processing.tools import fft_2d, run_numpy_snippet
from mast.agents.data_processing.prompts import SYSTEM_PROMPT


# Static public contract; never derived from the registry being tested.
EXPECTED_PUBLIC_DOMAIN_NAMES = {
    "load_scan", "record_analysis", "plot_scan", "plot_spectrum",
    "fft_2d", "plane_subtract", "detect_defects", "fit_sts_peaks",
    "run_numpy_snippet", "py_stage_data", "py_run", "mosaic_scans",
    "find_flat_region", "assess_cluster_roundness", "analyze_scan_image",
    "auto_process_scan_batch", "get_latest_scan_file", "list_scan_dir", "glob_scans",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(tmp_path, name, arr) -> str:
    p = tmp_path / name
    np.save(p, np.asarray(arr, dtype=np.float64))
    return str(p)


def _parse_peaks(out: str):
    """Pull (kx, ky, mag) tuples out of the fft_2d text report."""
    peaks = []
    for m in re.finditer(
        r"peak \d+: \(kx=([+\-][\d.]+), ky=([+\-][\d.]+)\) mag=([\d.eE+\-]+)", out
    ):
        peaks.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return peaks


# ════════════════════════════════════════════════════════════════════════════
# #45 — fft_2d mirror-peak de-duplication
# ════════════════════════════════════════════════════════════════════════════

def test_single_grating_reports_one_real_bragg_peak(tmp_path):
    """A 1D cosine grating has ONE physical Bragg vector. The old code returned
    it twice (+k and -k, identical magnitude). After the half-plane fix only one
    significant peak remains; the rest are numerically-zero noise."""
    N = 64
    _, x = np.indices((N, N))
    img = np.cos(2 * np.pi * 4 * x / N)
    out = fft_2d.invoke({"path": _save(tmp_path, "grating.npy", img), "top_n_peaks": 4})
    peaks = _parse_peaks(out)
    assert peaks, out

    # Exactly one peak should carry real spectral weight.
    strong = [p for p in peaks if p[2] > 1.0]
    assert len(strong) == 1, f"expected 1 strong (de-duplicated) peak, got {strong}"

    # No two reported peaks may be each other's ±k mirror (the old bug).
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            kxi, kyi, _ = peaks[i]
            kxj, kyj, _ = peaks[j]
            mirror = abs(kxi + kxj) < 1e-9 and abs(kyi + kyj) < 1e-9
            assert not mirror, f"mirror pair leaked: {peaks[i]} vs {peaks[j]}"


def test_peaks_all_lie_in_reported_half_plane(tmp_path):
    """Every reported peak must lie in the retained half-plane
    (ky > 0, or ky == 0 and kx >= 0) — the de-dup invariant."""
    N = 48
    y, x = np.indices((N, N))
    img = np.cos(2 * np.pi * 5 * x / N) + 0.7 * np.cos(2 * np.pi * 3 * y / N)
    out = fft_2d.invoke({"path": _save(tmp_path, "two.npy", img), "top_n_peaks": 4})
    for kx, ky, _ in _parse_peaks(out):
        assert ky > 1e-12 or (abs(ky) <= 1e-12 and kx >= -1e-12), (
            f"peak ({kx}, {ky}) is outside the retained half-plane"
        )


def test_square_lattice_reports_two_distinct_bragg_vectors(tmp_path):
    """Two orthogonal gratings (square lattice) → two distinct Bragg vectors.
    With mirror de-dup, the top-2 peaks must be the two real, non-mirror
    vectors (not +k/-k of a single one)."""
    N = 64
    y, x = np.indices((N, N))
    img = np.cos(2 * np.pi * 4 * x / N) + np.cos(2 * np.pi * 4 * y / N)
    out = fft_2d.invoke({"path": _save(tmp_path, "sq.npy", img), "top_n_peaks": 2})
    peaks = _parse_peaks(out)
    strong = [p for p in peaks if p[2] > 1.0]
    assert len(strong) == 2, f"expected 2 distinct Bragg vectors, got {strong}"
    (k1x, k1y, _), (k2x, k2y, _) = strong[0], strong[1]
    # The two strong vectors must NOT be mirrors of each other.
    assert not (abs(k1x + k2x) < 1e-9 and abs(k1y + k2y) < 1e-9)
    # They should be (roughly) orthogonal — one along x, one along y.
    assert {round(abs(k1x), 4), round(abs(k2x), 4)} == {0.0625, 0.0}


def test_flat_image_no_peaks_no_crash(tmp_path):
    """A featureless image has no non-DC peaks — must not crash or report any."""
    out = fft_2d.invoke({"path": _save(tmp_path, "flat.npy", np.full((32, 32), 3.0)),
                         "top_n_peaks": 4})
    assert "no non-dc peaks" in out.lower()
    assert _parse_peaks(out) == []


def test_top_n_larger_than_candidates_does_not_crash(tmp_path):
    """An absurd top_n_peaks must be capped to the candidate pool, never index
    past it (argpartition would raise on -k with k > len)."""
    N = 32
    _, x = np.indices((N, N))
    img = np.cos(2 * np.pi * 6 * x / N)
    out = fft_2d.invoke({"path": _save(tmp_path, "big.npy", img), "top_n_peaks": 100000})
    peaks = _parse_peaks(out)
    assert peaks, out
    # Still exactly one real peak.
    assert len([p for p in peaks if p[2] > 1.0]) == 1


def test_lattice_constant_uses_single_dedup_peak(tmp_path):
    """When nm_per_pixel metadata is present, the lattice constant is computed
    from the (single, de-duplicated) strongest peak. We feed metadata via the
    .sxm path; .npy carries none, so just assert the no-metadata branch is sane
    and the strong peak |k| matches the known grating frequency."""
    N = 64
    _, x = np.indices((N, N))
    img = np.cos(2 * np.pi * 8 * x / N)  # 8 cycles → kx = 8/64 = 0.125
    out = fft_2d.invoke({"path": _save(tmp_path, "lat.npy", img), "top_n_peaks": 4})
    strong = [p for p in _parse_peaks(out) if p[2] > 1.0]
    assert len(strong) == 1
    kx, ky, _ = strong[0]
    assert abs(kx - 0.125) < 1e-6 and abs(ky) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# #44 — prompt advertises the real tool signatures (no drift)
# ════════════════════════════════════════════════════════════════════════════

def test_prompt_no_phantom_channel_argument():
    # The old prompt had `fft_2d(sxm_path, channel)`; fft_2d has NO channel arg.
    assert "channel)" not in SYSTEM_PROMPT
    assert "top_n_peaks" in SYSTEM_PROMPT


def test_prompt_no_stale_stub_or_diameter_claims():
    # run_numpy_snippet is a real sandbox now, not a "stub only".
    assert "stub only" not in SYSTEM_PROMPT.lower()
    assert "stub" not in SYSTEM_PROMPT.lower()
    # detect_defects returns component SIZE (px area). The old prompt claimed it
    # returns "mean diameter" — that stale wording must be gone. (The new prompt
    # may say "area, not diameter" to clarify, so we forbid only the old claim.)
    assert "mean diameter" not in SYSTEM_PROMPT.lower()


def test_prompt_states_the_defaults_that_change_what_a_number_means():
    """The prompt keeps defaults that change numerical interpretation, while full tool signatures come from the bound schemas."""
    for fact in ("order=1", "min_size_px=5", "sigma_threshold=2.0",
                 "n_peaks=3", "top_n_peaks=4"):
        assert fact in SYSTEM_PROMPT, f"prompt no longer states {fact}"
    assert "面积" in SYSTEM_PROMPT, (
        "detect_defects 返回面积而不是直径 —— 不说清楚，那个数会被当成尺寸读")


def test_prompt_names_no_ghost_tools():
    """Cross-check public capability names, shared sources and every named prompt tool."""
    import re
    from mast.agents._shared.buffer_tools import make_buffer_tools
    from mast.agents.data_processing import tools as dp

    assert {t.name for t in dp.AGENT_TOOLS} == EXPECTED_PUBLIC_DOMAIN_NAMES
    expected_buffer = {"read_latest_tip_status", "get_scan_progress", "get_tip_history_since"}
    assert {t.name for t in make_buffer_tools(None)} == expected_buffer
    named = set(re.findall(r"`([a-z_][a-z0-9_]{3,})`", SYSTEM_PROMPT))
    named |= set(re.findall(r"\b([a-z_][a-z0-9_]{3,})\(", SYSTEM_PROMPT))
    allowed = EXPECTED_PUBLIC_DOMAIN_NAMES | expected_buffer | {
        "handoff_to_paper_writing", "handoff_to_supervisor", "ask_user",
        "save_result", "savefig", "npy_load", "curve_fit",
    }
    ghosts = sorted(named - allowed)
    assert not ghosts, f"Prompt advertises unavailable tools: {ghosts}"


def test_prompt_advertised_tools_match_actual_tool_names():
    """The independent public set must match the actual domain registration exactly."""
    from mast.agents.data_processing import tools as dp

    real_names = {t.name for t in dp.AGENT_TOOLS}
    assert real_names == EXPECTED_PUBLIC_DOMAIN_NAMES, (
        f"Public domain tools changed: {real_names ^ EXPECTED_PUBLIC_DOMAIN_NAMES}")
    unavailable = {
        "fit_gap_bcs", "fit_fano_kondo", "unmix_spectra", "level_lines",
        "subtract_plane_ransac", "subtract_poly2d", "destripe", "denoise_image",
        "deconvolve_tip", "measure_drift", "correct_lattice_distortion",
        "detect_atoms", "cluster_defects", "check_line_quality", "detect_atom_jump",
        "auto_crop_scan", "diff_scans", "compose_montage",
    }
    assert not {name for name in unavailable if name in SYSTEM_PROMPT}, (
        "The public prompt must not advertise excluded capabilities")


def test_prompt_mentions_every_real_tool():
    """Every available public tool needs a question-to-tool route; schemas alone do not explain when to use it."""
    from mast.agents.data_processing import tools as dp

    missing = sorted(t.name for t in dp.AGENT_TOOLS if t.name not in SYSTEM_PROMPT)
    assert not missing, (
        f"这些工具存在但提示词一次都没提：{missing}；"
        "在「问题 → 工具」表里给它们各加一行 —— 一句「什么时候用它」就够，"
        "不要把签名抄回来。")


# ════════════════════════════════════════════════════════════════════════════
# #41 / #42 — regression pin (fixed by session-#6 sandbox hardening)
# ════════════════════════════════════════════════════════════════════════════

def test_sandbox_reduction_methods_no_import_keyerror():
    """#41: numpy 2.x ndarray.sum/.mean/.max lazily import internals; the
    guarded __import__ must satisfy them instead of raising KeyError('__import__')."""
    out = run_numpy_snippet.invoke(
        {"code": "a = np.arange(12.0).reshape(3, 4)\n"
                 "r = (a.sum(), a.mean(), a.max())"}
    )
    assert "sandbox ok" in out.lower(), out
    assert "keyerror" not in out.lower()


def test_sandbox_tuple_unpacking_works():
    """#42: tuple unpacking must route through guarded_unpack_sequence bound to
    _unpack_sequence_ (not crash with a name/signature error)."""
    out = run_numpy_snippet.invoke(
        {"code": "a = np.array([1.0, 2.0, 3.0])\nx, y, z = a\nr = x + y + z"}
    )
    assert "sandbox ok" in out.lower(), out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
