"""A skill's tool-return sidecar is not measurement data — and never was.

For example, a sidecar overflow file can read like this::

    load_scan(path=<tool_returns_dir>/AcquireSTS_example.txt)
    ValueError: zero-size array to reduction operation fmin
    which has no identity

``artifacts/tool_returns/*.txt`` is a SUMMARY OVERFLOW file — the repr/JSON of a
result dict that was too long to inline in a ToolMessage. Nothing is meant to
read it back as an array.

The guard exists in ``load_scan``. It had **no test**, which is how the sibling
defects in this repo got reintroduced all day: a fix with no regression pin is a
comment. This is that pin.

Why refusal and not best-effort parsing: a GridSTS summary can contain one
all-numeric line ("1400 1400 5"), which parses as a 1×3 "scan" and yields
plausible-looking statistics that are pure garbage. A wrong number the agent
believes is worse than an error it must handle — and the same run's report DID
publish a bias-axis standard deviation as a repeatability statistic.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_load_scan_rejects_tool_returns.py -q
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

from mast.agents.data_processing.tools import load_scan  # noqa: E402


@pytest.fixture()
def sidecar(tmp_path) -> Path:
    """The exact shape of the file the field call was handed."""
    d = tmp_path / "artifacts" / "tool_returns"
    d.mkdir(parents=True)
    f = d / "AcquireSTS_767f4756.txt"
    # A real summary: prose + a stray all-numeric line that WOULD parse.
    f.write_text(
        "{'skill': 'AcquireSTS', 'success': True, 'points': 200}\n"
        "1400 1400 5\n",
        encoding="utf-8")
    return f


def test_the_field_call_is_refused_not_crashed(sidecar):
    out = load_scan.invoke({"path": str(sidecar)})
    assert "failed" in out
    assert "zero-size" not in out, "还是那个 ValueError —— 守卫没生效"
    assert "Traceback" not in out


def test_the_refusal_says_what_to_read_instead(sidecar):
    """A refusal that does not name the alternative just gets retried."""
    out = load_scan.invoke({"path": str(sidecar)})
    assert ".dat" in out and ".sxm" in out
    assert "list_scan_dir" in out or "glob_scans" in out


def test_the_numeric_line_is_never_parsed_as_a_scan(sidecar):
    """The dangerous case: "1400 1400 5" is a valid 1×3 array. Statistics off
    it look completely plausible and are meaningless."""
    out = load_scan.invoke({"path": str(sidecar)})
    for leak in ("1400", "shape", "mean", "std"):
        assert leak not in out, f"守卫之后还是把它当数据解析了: {leak!r}"


def test_a_real_scan_next_to_it_still_loads(tmp_path):
    """The guard keys on the directory, so it must not shadow real data that
    merely lives nearby."""
    scans = tmp_path / "scans"
    scans.mkdir()
    p = scans / "real.npy"
    np.save(p, np.random.default_rng(0).normal(0, 1e-11, (64, 64)))
    out = load_scan.invoke({"path": str(p)})
    assert "failed" not in out, out
    assert "64" in out


def test_the_guard_matches_a_path_component_not_a_substring(tmp_path):
    """`tool_returns` must be a DIRECTORY on the path, not any occurrence in the
    filename — a scan legitimately called `tool_returns_test.npy` is data."""
    d = tmp_path / "scans"
    d.mkdir()
    p = d / "tool_returns_comparison.npy"
    np.save(p, np.random.default_rng(1).normal(0, 1e-11, (32, 32)))
    out = load_scan.invoke({"path": str(p)})
    assert "摘要溢出件" not in out, "按子串匹配了 —— 会误伤正常的扫描文件"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
