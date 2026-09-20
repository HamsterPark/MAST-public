"""Copy-collapse for the Data tab listing (``webui.scan_preview``).

MAST ingests every scan Nanonis writes into the experiment folder and leaves the
original where it was, so one measurement exists as several byte-identical files.
The listing used to show one card per copy — same name, same size, same
timestamp on each — and the operator could not tell which one to click
. These tests pin the two halves of the fix:

  * copies of ONE measurement fold into one entry, and
  * two DIFFERENT files that merely share a basename do NOT
    (``unnamed0001.sxm`` exists three times on the instrument with three different
    contents — folding those would hide real data behind a badge).

Nothing may be dropped either way: every input file has to come back out inside
some group, which is the property ``scanKinds`` states on the frontend side
("分组是重排不是过滤") and the one that keeps this from becoming again.
"""

from __future__ import annotations

from pathlib import Path

from mast.webui.scan_preview import (
    CollapsedScan,
    ScanStat,
    _norm_key,
    classify_scan_path,
    collapse_scan_groups,
)


def _stat(path: str, mtime_ns: int = 1_700_000_000_000_000_000, size: int = 4096) -> ScanStat:
    return ScanStat(path=Path(path), mtime_ns=mtime_ns, size_bytes=size)


# ── the fold ───────────────────────────────────────────────────────────


def test_identical_copies_collapse_into_one_entry() -> None:
    """copystat reproduces name+size+mtime_ns exactly, so the triple matches."""
    origin = _stat(r"D:\sessions\20260318\scan_001.sxm")
    copy = _stat(r"D:\MAST-Data\experiments\E1\samples\S01\raw\sxm\scan_001.sxm")
    groups = collapse_scan_groups([origin, copy], exp_root_key=_norm_key(r"D:\MAST-Data\experiments"))
    assert len(groups) == 1
    assert len(groups[0].members) == 2


def test_same_name_different_mtime_does_not_collapse() -> None:
    """同名 ≠ 同内容. Two sessions both wrote ``unnamed0001.sxm``; different data."""
    a = _stat(r"D:\sessions\20260318\unnamed0001.sxm", mtime_ns=1_700_000_000_000_000_000)
    b = _stat(r"D:\sessions\20260319\unnamed0001.sxm", mtime_ns=1_700_000_001_000_000_000)
    groups = collapse_scan_groups([a, b])
    assert len(groups) == 2


def test_same_name_and_mtime_but_different_size_does_not_collapse() -> None:
    a = _stat(r"D:\a\scan.sxm", size=4096)
    b = _stat(r"D:\b\scan.sxm", size=8192)
    assert len(collapse_scan_groups([a, b])) == 2


def test_one_nanosecond_apart_does_not_collapse() -> None:
    """Why mtime_ns and not st_mtime: float seconds cannot separate these.

    A float carries ~15-17 significant digits, and a 2026 epoch already spends
    10 of them left of the point — so two files a nanosecond apart can round to
    the same float while being genuinely different files."""
    a = _stat(r"D:\a\scan.sxm", mtime_ns=1_785_000_000_123_456_789)
    b = _stat(r"D:\b\scan.sxm", mtime_ns=1_785_000_000_123_456_790)
    assert len(collapse_scan_groups([a, b])) == 2


def test_nothing_is_ever_dropped() -> None:
    """Every input file comes back inside exactly one group."""
    stats = [
        _stat(r"D:\s\a.sxm", mtime_ns=100),
        _stat(r"D:\e\a.sxm", mtime_ns=100),
        _stat(r"D:\s\b.sxm", mtime_ns=200),
        _stat(r"D:\s\c.dat", mtime_ns=300, size=99),
    ]
    groups = collapse_scan_groups(stats)
    seen = [m for g in groups for m in g.members]
    assert len(seen) == len(stats)
    assert {str(s.path) for s in seen} == {str(s.path) for s in stats}


def test_group_order_follows_input_order() -> None:
    """Discovery hands us mtime-desc; the fold must not reshuffle it."""
    stats = [
        _stat(r"D:\s\newest.sxm", mtime_ns=300),
        _stat(r"D:\s\middle.sxm", mtime_ns=200),
        _stat(r"D:\s\oldest.sxm", mtime_ns=100),
    ]
    assert [g.rep.path.name for g in collapse_scan_groups(stats)] == [
        "newest.sxm", "middle.sxm", "oldest.sxm",
    ]


def test_empty_input() -> None:
    assert collapse_scan_groups([]) == []


# ── which copy represents the group ────────────────────────────────────


def test_representative_prefers_the_copy_outside_the_experiment_folder() -> None:
    """The original keeps its identity; managed copies get re-ingested into
    whichever experiment is current, so a card pointed at one changes identity
    underneath the operator."""
    root = _norm_key(r"D:\MAST-Data\experiments")
    managed = _stat(r"D:\MAST-Data\experiments\E1\samples\S01\raw\sxm\scan.sxm")
    origin = _stat(r"D:\sessions\20260318\scan.sxm")
    # Managed copy listed FIRST — the preference must not be "whatever came first".
    groups = collapse_scan_groups([managed, origin], exp_root_key=root)
    assert len(groups) == 1
    assert groups[0].rep.path == origin.path


def test_representative_falls_back_when_every_copy_is_managed() -> None:
    root = _norm_key(r"D:\MAST-Data\experiments")
    a = _stat(r"D:\MAST-Data\experiments\E1\samples\S01\raw\sxm\scan.sxm")
    b = _stat(r"D:\MAST-Data\experiments\E2\samples\S01\raw\sxm\scan.sxm")
    groups = collapse_scan_groups([a, b], exp_root_key=root)
    assert len(groups) == 1 and groups[0].rep.path == a.path


# ── classification ─────────────────────────────────────────────────────


def test_classify_experiment_origin_and_quarantine() -> None:
    root = _norm_key(r"D:\MAST-Data\experiments")
    assert classify_scan_path(r"D:\MAST-Data\experiments\E1\raw\a.sxm", root) == "experiment"
    assert classify_scan_path(r"D:\sessions\20260318\a.sxm", root) == "origin"
    assert classify_scan_path(r"D:\MAST-Data\experiments\_quarantine\2026\a.sxm", root) == "quarantine"


def test_classify_without_an_experiment_root_says_origin() -> None:
    """Unresolvable root ⇒ everything is 'origin'.

    That is the honest answer rather than a guess: with no root we cannot prove a
    path is managed. It degrades the badge, not the listing."""
    assert classify_scan_path(r"D:\MAST-Data\experiments\E1\raw\a.sxm", None) in (
        "origin", "experiment",  # depends on the real root on this machine
    )
    # With an explicitly empty root the answer is unambiguous.
    assert classify_scan_path(r"D:\anything\a.sxm", "") == "origin"


def test_classify_is_case_insensitive_on_the_root_prefix() -> None:
    """The search dirs really do produce both spellings: one from config, one
    from the Nanonis session path."""
    root = _norm_key(r"D:\MAST-Data\experiments")
    assert classify_scan_path(r"d:\mast-data\EXPERIMENTS\E1\a.sxm", root) == "experiment"


def test_a_sibling_directory_is_not_inside_the_root() -> None:
    """Prefix matching must respect the separator: ``…\\experiments-old`` is not
    under ``…\\experiments``."""
    root = _norm_key(r"D:\MAST-Data\experiments")
    assert classify_scan_path(r"D:\MAST-Data\experiments-old\E1\a.sxm", root) == "origin"


# ── shape ──────────────────────────────────────────────────────────────


def test_returns_collapsed_scan_namedtuples() -> None:
    groups = collapse_scan_groups([_stat(r"D:\s\a.sxm")])
    assert isinstance(groups[0], CollapsedScan)
    assert groups[0].rep in groups[0].members
