"""The STS half of the loop: what the analysis tools say about a spectrum.

WHERE THIS CAME FROM
--------------------
End-to-end run of the spectroscopy chain, 2026-07-28 (real LLM, factory HITL,
fake instrument autosaving real Nanonis-format .dat files): 134 super-steps,
496 s, zero approvals, every artifact produced — 3 spectra, 3 figures, a draft
and an exported HTML. Every checklist item passed.

The artifacts were then read, and two things in them were wrong. Both are the
shape this project keeps hitting: **the run looks successful and the numbers on
the page are not real**.

1. ``load_scan`` on a 201×3 STS ``.dat`` reported::

       min: -1, max: 1, mean: 3.072e-10, std: 0.335

   A spectrum is a table of UNLIKE quantities — a bias column in volts beside
   currents in amps. Pooling them is meaningless, and every one of those four
   numbers is dominated by (or purely is) the bias axis: the currents are
   ~3e-10 A, and ``std 0.335`` is just the standard deviation of a -1..+1 V
   ramp diluted across three columns. The generated report published
   「三条谱 std 均约为 0.335」 as a repeatability statistic. The agent even
   hedged it — 「按文件三列汇总，仅作重复性粗查」 — and published it anyway,
   because the tool gave it no better number and no way to ask for one.

2. ``plot_spectrum`` labelled the y-axis ``I (A)`` on a figure whose second
   curve was the lock-in dI/dV, and legended the curves ``ch1`` / ``ch2``.

   The cause of (2) is a dead criterion. The function loaded the file through
   ``load_spectrum``, which stacks ``read_dat``'s ``columns`` dict through
   ``.values()`` and DROPS THE KEYS; ``plot_spectrum`` then invented
   ``ch1..chN`` — and then chose the y-label by pattern-matching those invented
   names for "di"/"lock"::

       names  = [f"ch{j + 1}" for j in range(arr.shape[0])]
       joined = " ".join(str(n).lower() for n in (names or []))
       ax.set_ylabel("dI/dV (a.u.)" if "di" in joined or "lock" in joined
                     else "I (A)")

   ``"di" in "ch1 ch2"`` is False for every spectrum that will ever exist, so
   the dI/dV branch was unreachable and EVERY spectrum figure MAST has produced
   is labelled ``I (A)``. The comment above it read "the loader tells us whether
   these are raw currents or a lock-in dI/dV" — the loader does tell us, and the
   line above threw the answer away.

Fixtures are the REAL Nanonis file in tests/v2/fixtures/nanonis/ wherever
possible: these bugs are about what real channel names do, so a synthetic file
with tidy names would not have caught them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
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

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402

_REAL_DAT = (Path(__file__).resolve().parents[2]
             / "fixtures" / "nanonis" / "bias_spectroscopy_200pt.dat")


@pytest.fixture(autouse=True)
def _isolate_scan_registry():
    """``_candidate_save_dirs`` searches ``scan_registry.known_scan_dirs()``,
    a MODULE-GLOBAL list, in addition to the caller's session path — and
    ``_attach_saved_dat`` registers into it on every success.

    So one test's session directory stays a search root for the next one, and
    ``find_latest_saved`` takes the newest match across ALL roots: the stale-file
    test below found the *previous* test's fresh .dat and reported a pass-looking
    failure. Worth writing down rather than just clearing — the same accumulation
    happens in a long-lived process, where a directory registered hours ago is
    still a candidate for "which file did this sweep just write".
    """
    from mast.core import scan_registry

    scan_registry.clear()
    yield
    scan_registry.clear()


def _lockin_dat(tmp_path: Path) -> Path:
    """An STS .dat carrying a lock-in channel — the ordinary configuration.

    ConfigureSTS turns the lock-in on, so Current + LIX in one file IS the
    normal STS output. The shipped fixture happens to hold Current + Current
    [bwd], which cannot distinguish "labelled I (A) correctly" from "labelled
    I (A) unconditionally"."""
    v = np.linspace(-1.0, 1.0, 51)
    cur = np.sign(v) * 3.0e-9 * np.clip(np.abs(v) - 0.4, 0, None) ** 2
    didv = np.gradient(cur, v)
    lines = ["Experiment\tbias spectroscopy", "Z-Ctrl hold\tTRUE", "",
             "[DATA]", "Bias calc (V)\tCurrent (A)\tLIX 1 omega (A)"]
    lines += [f"{v[i]:.6E}\t{cur[i]:.6E}\t{didv[i]:.6E}" for i in range(v.size)]
    p = tmp_path / "sts_lockin.dat"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════════════════
# The channel names must survive the loader
# ════════════════════════════════════════════════════════════════════════

class TestNamesSurviveLoading:
    def test_real_nanonis_dat_keeps_its_channel_names(self):
        from mast.data import load_spectrum_named

        arr, names = load_spectrum_named(_REAL_DAT)
        assert arr.ndim == 2 and arr.shape[1] == len(names)
        assert names[0].lower().startswith("bias")
        assert any("current" in n.lower() for n in names)

    def test_the_nameless_loader_is_what_dropped_them(self):
        """Documents the boundary: load_spectrum still returns a bare array (it
        has many callers). The named variant is the one to reach for when the
        names are going to be shown to a human or tested against."""
        from mast.data import load_spectrum, load_spectrum_named

        plain = load_spectrum(_REAL_DAT)
        named, names = load_spectrum_named(_REAL_DAT)
        assert np.allclose(plain, named)
        assert names and not hasattr(plain, "columns")

    def test_a_format_with_no_channel_names_says_so_rather_than_inventing(self, tmp_path):
        """Empty means UNKNOWN. Filling it with ch1/ch2 is what let a downstream
        test pattern-match its own placeholders and always miss."""
        from mast.data import load_spectrum_named

        p = tmp_path / "spec.npy"
        np.save(p, np.random.default_rng(0).normal(size=(20, 2)))
        arr, names = load_spectrum_named(p)
        assert arr.shape == (20, 2)
        assert names == []


# ════════════════════════════════════════════════════════════════════════
# …and reach the figure
# ════════════════════════════════════════════════════════════════════════

def _render(monkeypatch, tmp_path, src: Path) -> dict:
    """Run plot_spectrum and capture what actually landed on the Axes.

    Reads the rendered object rather than the PNG: the defect was a label, and a
    label is exactly what a "did it save a file?" assertion cannot see."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figs"))
    captured: dict = {}
    real_subplots = plt.subplots

    def _spy(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _spy)
    from mast.agents.data_processing.tools import plot_spectrum

    # 完整 ToolCall 形式：plot_spectrum 有 InjectedToolCallId（图像回路），
    # LangChain 因此要求这种调用形状 —— 而它也正是 graph 里的那一条路。
    out = str(plot_spectrum.invoke(
        {"args": {"path": str(src), "label": "t"},
         "name": "plot_spectrum", "type": "tool_call",
         "id": "t1"}))
    ax = captured.get("ax")
    assert ax is not None, f"plot_spectrum never drew anything: {out}"
    return {
        "out": out,
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "legend": [t.get_text() for t in (ax.get_legend().get_texts()
                                          if ax.get_legend() else [])],
    }


class TestFigureLabelsTellTheTruth:
    def test_lockin_spectrum_is_not_labelled_as_current(self, monkeypatch, tmp_path):
        """THE regression. Before: ylabel "I (A)" over a dI/dV curve, on a figure
        that goes straight into a manuscript."""
        r = _render(monkeypatch, tmp_path, _lockin_dat(tmp_path))
        assert r["ylabel"] != "I (A)", (
            "a Current + lock-in dI/dV plot is still labelled as pure current")
        assert "(A)" in r["ylabel"]          # the shared unit is still stated

    def test_legend_names_the_real_channels(self, monkeypatch, tmp_path):
        r = _render(monkeypatch, tmp_path, _lockin_dat(tmp_path))
        assert r["legend"] == ["Current (A)", "LIX 1 omega (A)"], r["legend"]
        assert not any(t.startswith("ch") for t in r["legend"])

    def test_x_axis_comes_from_the_file_not_from_an_assumption(self, monkeypatch,
                                                               tmp_path):
        """Hard-coding "Bias (V)" is wrong for a Z-spectroscopy .dat, which
        sweeps Z in metres."""
        r = _render(monkeypatch, tmp_path, _lockin_dat(tmp_path))
        assert r["xlabel"] == "Bias calc (V)"

    def test_a_pure_current_file_still_says_current(self, monkeypatch, tmp_path):
        """The real shipped fixture is Current + Current [bwd]. The fix must not
        turn a correct label into a vague one."""
        r = _render(monkeypatch, tmp_path, _REAL_DAT)
        assert r["ylabel"] == "I (A)", r["ylabel"]

    def test_the_return_describes_the_figure_that_was_drawn(self, monkeypatch,
                                                            tmp_path):
        """The writing agent captions from this string — it cannot see the PNG.
        The old return said only "2 curve(s), 201 points, x=Bias (V)", and in the
        end-to-end run the agent captioned all three single-panel figures
        「Top: I(V) curve; bottom: lock-in dI/dV」. A caption that does not match
        its figure is the same failure family as a number that is not real."""
        r = _render(monkeypatch, tmp_path, _lockin_dat(tmp_path))
        assert "单幅坐标系" in r["out"], r["out"]
        assert "Current (A)" in r["out"] and "LIX 1 omega (A)" in r["out"]
        assert r["ylabel"] in r["out"], "the return must state the y axis it drew"

    def test_axis_labels_render_with_the_cjk_font(self, monkeypatch, tmp_path):
        """Axis labels now carry text FROM THE FILE, so they can be non-ASCII.
        The CJK font used to be applied to the title only — which was invisible
        while every label was a hard-coded ASCII literal, and became tofu boxes
        the moment one was not (caught while fixing the label above)."""
        import inspect

        from mast.agents.data_processing import tools

        src = source_of(tools.plot_spectrum.func)
        for call in ("ax.set_xlabel(", "ax.set_ylabel("):
            seg = src[src.index(call):src.index(call) + 200]
            assert "fontproperties" in seg, f"{call} does not apply the CJK font"


# ════════════════════════════════════════════════════════════════════════
# load_scan must not pool unlike physical quantities
# ════════════════════════════════════════════════════════════════════════

class TestSpectrumStatisticsArePerColumn:
    def test_a_spectrum_reports_each_column_under_its_own_name(self):
        from mast.agents.data_processing.tools import load_scan

        out = load_scan.invoke({"path": str(_REAL_DAT)})
        assert "逐列统计" in out, out
        assert "Bias calc (V):" in out
        assert "Current (A):" in out

    def test_the_pooled_statistics_are_gone(self, tmp_path):
        """The exact numbers from the run: a file whose currents are ~1e-9 A
        reported mean 3.07e-10 / std 0.335, both of which are the bias ramp.
        Nothing in the output may still offer a whole-array std."""
        from mast.agents.data_processing.tools import load_scan
        from mast.data import load_spectrum

        src = _lockin_dat(tmp_path)
        pooled_std = float(np.nanstd(load_spectrum(src)))
        assert pooled_std > 0.1, "fixture no longer reproduces the pooling effect"

        out = load_scan.invoke({"path": str(src)})
        assert f"{pooled_std:.4g}" not in out, (
            f"the meaningless pooled std {pooled_std:.4g} is still being reported")

    def test_each_reported_number_matches_its_own_column(self, tmp_path):
        from mast.agents.data_processing.tools import load_scan
        from mast.data import load_spectrum_named

        src = _lockin_dat(tmp_path)
        arr, names = load_spectrum_named(src)
        out = load_scan.invoke({"path": str(src)})
        for j, nm in enumerate(names):
            line = next(ln for ln in out.splitlines() if ln.strip().startswith(nm + ":"))
            assert f"{np.nanstd(arr[:, j]):.4g}" in line, (
                f"{nm} reports a std that is not its own: {line}")

    def test_an_image_still_gets_whole_array_statistics(self, tmp_path):
        """A topograph IS one physical quantity — pooling is correct there, and
        this fix must not disturb the imaging chain that already runs green."""
        from mast.agents.data_processing.tools import load_scan

        p = tmp_path / "topo.npy"
        np.save(p, np.random.default_rng(1).normal(0, 1e-10, (64, 64)))
        out = load_scan.invoke({"path": str(p)})
        assert "逐列统计" not in out
        assert "std:" in out

    def test_a_headerless_dat_does_not_pretend_to_have_columns(self, tmp_path):
        """No [DATA] block → no names → fall back to the old whole-array report
        rather than labelling columns with guesses."""
        from mast.agents.data_processing.tools import load_scan

        p = tmp_path / "bare.dat"
        p.write_text("\n".join(f"{i * 0.1:.3f}\t{i * 1e-9:.3e}" for i in range(20)),
                     encoding="utf-8")
        out = load_scan.invoke({"path": str(p)})
        assert "failed" not in out.lower(), out


# ════════════════════════════════════════════════════════════════════════
# The joint the whole chain hangs on
# ════════════════════════════════════════════════════════════════════════

def test_acquire_sts_reports_where_nanonis_saved_the_dat(tmp_path, monkeypatch):
    """MAST never writes the spectrum — Nanonis autosaves it and AcquireSTS then
    looks for the newest .dat (``_attach_saved_dat`` → ``find_latest_saved``,
    120 s window). That path is the ONLY thing the analysis agent can be handed;
    without it the handoff carries nothing but the tool-return sidecar, which is
    exactly what shows data_processing crashing on.

    Pinned here because it is a joint held together by a filesystem timestamp:
    nothing else in the system fails if it silently stops working.
    """
    from mast.skills.builtins.spectroscopy import _attach_saved_dat

    session = tmp_path / "session"
    session.mkdir()
    dat = session / "BiasSpec00001.dat"
    dat.write_text("[DATA]\nBias calc (V)\tCurrent (A)\n0\t1e-9\n", encoding="utf-8")

    class _Ctx:
        session_path = str(session)

        def safe_call(self, *a, **kw):
            raise AssertionError("session_path should be used directly")

    data: dict = {}
    _attach_saved_dat(_Ctx(), data)
    assert data.get("path") == str(dat), (
        "AcquireSTS could not name the file Nanonis just saved — the analysis "
        "agent would be handed no measurement at all")


def test_a_stale_dat_is_not_claimed_as_this_acquisition(tmp_path):
    """The 120 s window is the only thing separating "the file we just made"
    from "someone else's data". A wrong path here points the analysis agent at
    the wrong measurement, which is worse than no path."""
    import os
    import time

    from mast.skills.builtins.spectroscopy import _attach_saved_dat

    session = tmp_path / "session"
    session.mkdir()
    old = session / "Yesterday00001.dat"
    old.write_text("[DATA]\nBias calc (V)\n0\n", encoding="utf-8")
    stale = time.time() - 3600
    os.utime(old, (stale, stale))

    class _Ctx:
        session_path = str(session)

    data: dict = {}
    _attach_saved_dat(_Ctx(), data)
    assert "path" not in data, f"claimed an hour-old file as this sweep: {data}"
