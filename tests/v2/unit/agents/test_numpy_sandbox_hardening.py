"""Numpy snippet sandbox hardening — security review MEDIUM #6.

`mast.agents.data_processing.tools.run_numpy_snippet` runs LLM-authored Python
in-process. The previous implementation used a *deny-list* over the full `np`
module (plus a whole `scipy.fft` module), which left a residual attack surface:
scipy exposed wholesale, `np.core`/`np.random`/`np.linalg` reachable as full
submodules, the guarded getattr non-recursive, and `safe_builtins` still
shipping `setattr`/`delattr`/`__build_class__`.

This file pins the *allow-list* model that replaced it:

  * Attacks — `import os` / bare `os` / `__import__('os')` / `open(...)` /
    `np.save` / `np.load` / `np.loadtxt` / `np.fromfile` / `np.memmap` /
    `np.genfromtxt` / `scipy_fft` / `getattr(np, '__loader__')` / dunder chains
    (`__class__`, `__globals__`, `__builtins__`) / `setattr` / class defs /
    `from numpy import save` / ndarray IO methods (`.tofile`, `.dump`, `.view`,
    `.ctypes`) — must ALL error or be undefined.
  * Legitimate numpy — basic arithmetic, array creation, `mean`/`std` (function
    and method form), specific fft functions (`np.fft.fft2`, `np.fft.fftshift`),
    linalg (`np.linalg.lstsq`, `np.linalg.norm`), tuple-unpacking
    (`yy, xx = np.indices(...)`), and the plane-subtract / fft_2d flows the DP
    agent really uses — must keep working.
  * `npy_load` — `.npy` only, `allow_pickle=False`, project data dirs only.

Runs with no LLM and no network: every input is a literal source string.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_numpy_sandbox_hardening.py -q -p no:randomly
"""
from __future__ import annotations

# ── path bootstrap (canonical block — force v2 MASTv2 copy of `mast`) ────────
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

import numpy as np
import pytest

# Module-level import so the cached `mast` resolves to the MASTv2 copy.
from mast.agents.data_processing import tools as dp_tools
from mast.agents.data_processing.tools import run_numpy_snippet
from mast._runtime_paths import project_root


# ── helpers ─────────────────────────────────────────────────────────────────

def _run(code: str) -> str:
    """Invoke the sandbox tool with a source string, return its text result."""
    return run_numpy_snippet.invoke({"code": code})


def _is_error(out: str) -> bool:
    """The sandbox reports denials as compile/runtime errors or NameErrors."""
    low = out.lower()
    return any(
        marker in low
        for marker in (
            "error",        # sandbox compile error / sandbox runtime error
            "denied",
            "not allowed",
            "not defined",
            "no allowed",
            "invalid",
            "not found",
        )
    )


def _is_ok(out: str) -> bool:
    return "sandbox ok" in out.lower()


# ════════════════════════════════════════════════════════════════════════════
# 1. Module / import escapes are dead
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    "import os",
    "import sys",
    "import subprocess",
    "import os.path",
    "r = os",                                  # bare name not in scope
    "r = sys",
    "r = __import__('os')",                    # RestrictedPython SyntaxError
    "from os import system\nr = 1",
    "from numpy import save\nr = save",        # would bind REAL np.save
    "from numpy import load\nr = load",
    "from scipy import fft\nr = fft",
    "import numpy as N\nr = N.save",           # real-module reach via alias
])
def test_imports_and_module_reach_are_blocked(code):
    out = _run(code)
    assert _is_error(out), f"expected blocked, got: {out!r}"


def test_scipy_fft_module_no_longer_exposed():
    # Old deny-list build injected `scipy_fft` as a whole module into scope.
    out = _run("r = scipy_fft")
    assert _is_error(out)
    assert "not defined" in out.lower()


# ════════════════════════════════════════════════════════════════════════════
# 2. File-IO primitives (eval/exec/open + numpy IO) are dead
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    "r = open('x', 'w')",
    "r = eval('1+1')",
    "exec('x = 1')\nr = 1",
    "r = compile('1', '<s>', 'eval')",
])
def test_exec_eval_open_compile_unavailable(code):
    assert _is_error(_run(code))


@pytest.mark.parametrize("attr", [
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt",
    "genfromtxt", "fromfile", "memmap", "fromregex",
    "lib", "core", "ctypes", "random", "testing", "f2py",
])
def test_numpy_io_and_submodule_attrs_denied(attr):
    out = _run(f"r = np.{attr}")
    assert _is_error(out), f"np.{attr} leaked: {out!r}"


@pytest.mark.parametrize("method", [
    "tofile", "dump", "dumps", "tobytes", "ctypes", "view", "setflags",
    "base", "data", "getfield", "byteswap",
])
def test_ndarray_io_and_escape_methods_denied(method):
    out = _run(f"a = np.zeros(4)\nr = a.{method}")
    assert _is_error(out), f"ndarray.{method} leaked: {out!r}"


# ════════════════════════════════════════════════════════════════════════════
# 3. Dunder / attribute-chain escapes are dead
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    "r = (1).__class__",
    "r = np.__class__",
    "a = np.zeros(3)\nr = a.__class__",
    "f = np.mean\nr = f.__globals__",
    "r = __builtins__",
    "r = (1).__class__.__bases__",
    "r = ().__class__.__bases__[0].__subclasses__()",
    "r = (lambda: 0).__globals__",
    "a = np.zeros(3)\nr = a.__reduce__()",
    "r = np.__loader__",
    "r = getattr(np, '__loader__')",           # getattr not in scope at all
    "r = getattr(np, 'save')",
])
def test_dunder_and_getattr_escapes_blocked(code):
    out = _run(code)
    assert _is_error(out), f"escape not blocked: {out!r}"


def test_getattr_name_is_not_defined():
    # getattr must not be reachable as a builtin at all (no name-based escape).
    out = _run("r = getattr")
    assert "not defined" in out.lower()


# ════════════════════════════════════════════════════════════════════════════
# 4. setattr / delattr / class building removed from builtins
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("code", [
    "setattr(np, 'x', 1)\nr = 1",
    "delattr(np, 'mean')\nr = 1",
    "class Foo:\n    pass\nr = 1",             # needs __build_class__
])
def test_write_and_classbuild_primitives_removed(code):
    out = _run(code)
    assert _is_error(out), f"write/classbuild primitive available: {out!r}"


# ════════════════════════════════════════════════════════════════════════════
# 5. Legitimate numpy usage still works
# ════════════════════════════════════════════════════════════════════════════

def test_basic_arithmetic():
    out = _run("x = 1 + 2")
    assert _is_ok(out)
    assert "x = 3" in out


def test_array_creation_and_reduction_methods():
    out = _run("a = np.arange(10)\ns = a.sum()\nm = a.mean()")
    assert _is_ok(out), out


def test_mean_std_function_form():
    out = _run("a = np.array([1.0, 2.0, 3.0, 4.0])\nmu = np.mean(a)\nsd = np.std(a)")
    assert _is_ok(out), out


def test_mean_std_method_form():
    out = _run("a = np.array([1.0, 2.0, 3.0, 4.0])\nr = a.std()")
    assert _is_ok(out), out


@pytest.mark.parametrize("expr", [
    "F = np.fft.fft2(np.zeros((8, 8)))\nm = np.abs(F).max()",
    "r = np.fft.fftshift(np.arange(8))",
    "r = np.fft.fftfreq(16)",
    "r = np.fft.rfft(np.ones(8))",
])
def test_fft_specific_functions_work(expr):
    out = _run(expr)
    assert _is_ok(out), out


@pytest.mark.parametrize("expr", [
    "A = np.eye(3)\nb = np.ones(3)\nx = np.linalg.lstsq(A, b, rcond=None)[0]",
    "r = np.linalg.norm(np.array([3.0, 4.0]))",
    "r = np.linalg.inv(np.eye(3) * 2.0)",
])
def test_linalg_specific_functions_work(expr):
    out = _run(expr)
    assert _is_ok(out), out


def test_tuple_unpacking_indices():
    # plane_subtract-style flow: yy, xx = np.indices(...) then column_stack/lstsq
    code = (
        "yy, xx = np.indices((4, 4))\n"
        "A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(16)])\n"
        "r = A.shape"
    )
    out = _run(code)
    assert _is_ok(out), out


def test_histogram_tuple_unpacking():
    out = _run("a = np.array([1.0, 2.0, 2.0, 3.0])\nh, e = np.histogram(a, bins=3)")
    assert _is_ok(out), out


def test_fft_2d_style_flow():
    # Mirrors the real fft_2d tool body (mean-subtract, fftshift, argpartition).
    code = (
        "a = np.arange(64.0).reshape(8, 8)\n"
        "a = a - a.mean()\n"
        "F = np.abs(np.fft.fftshift(np.fft.fft2(a)))\n"
        "flat = F.ravel()\n"
        "idx = np.argpartition(flat, -4)[-4:]\n"
        "r = idx.shape"
    )
    out = _run(code)
    assert _is_ok(out), out


def test_elementwise_math_and_builtins():
    out = _run(
        "x = np.linspace(0, 1, 5)\n"
        "y = np.exp(-x) + np.sin(x)\n"
        "r = max(1, 2, 3) + sum([i for i in range(4)])"
    )
    assert _is_ok(out), out


# ════════════════════════════════════════════════════════════════════════════
# 6. npy_load — .npy only, no pickle, project data dirs only
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def data_dir():
    d = project_root() / "data"
    d.mkdir(exist_ok=True)
    return d


def test_npy_load_inside_allowed_dir(data_dir):
    p = data_dir / "_sandbox_hardening_ok.npy"
    np.save(p, np.arange(12, dtype=np.float64).reshape(3, 4))
    try:
        out = _run(f"a = npy_load(r'{p}')\nr = a.mean()")
        assert _is_ok(out), out
    finally:
        p.unlink(missing_ok=True)


def test_npy_load_rejects_outside_dir(tmp_path):
    p = tmp_path / "evil.npy"
    np.save(p, np.ones(3))
    out = _run(f"a = npy_load(r'{p}')\nr = 1")
    assert _is_error(out)
    assert "outside the allowed" in out.lower()


def test_npy_load_rejects_non_npy_extension():
    out = _run("a = npy_load('/etc/passwd')\nr = 1")
    assert _is_error(out)
    assert "only loads .npy" in out.lower()


def test_npy_load_refuses_pickled_object_array(data_dir):
    p = data_dir / "_sandbox_hardening_pickle.npy"
    np.save(p, np.array([{"x": 1}], dtype=object))
    try:
        out = _run(f"a = npy_load(r'{p}')\nr = 1")
        # allow_pickle=False → numpy raises on object arrays.
        assert _is_error(out)
        assert "pickle" in out.lower() or "object array" in out.lower()
    finally:
        p.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 7. Direct-unit checks on the hardening helpers (no exec layer)
# ════════════════════════════════════════════════════════════════════════════

def test_sandbox_builtins_has_no_escape_primitives():
    b = dp_tools._build_sandbox_builtins()
    for bad in ("setattr", "delattr", "__build_class__", "open", "eval",
                "exec", "compile", "getattr"):
        assert bad not in b, f"{bad} must not be in sandbox builtins"
    # __import__ is present (numpy needs it) but is the guarded shim.
    assert b.get("__import__") is dp_tools._guarded_import


def test_guarded_import_blocks_dangerous_modules():
    for mod in ("os", "sys", "subprocess", "builtins", "socket", "shutil"):
        with pytest.raises(ImportError):
            dp_tools._guarded_import(mod)


def test_guarded_import_blocks_from_numpy_import_for_snippet():
    # Snippet frame (no numpy-prefixed __name__) cannot `from numpy import X`.
    with pytest.raises(ImportError):
        dp_tools._guarded_import("numpy", {"__name__": "<sandbox>"},
                                 None, ("save",), 0)


def test_numpy_proxy_exposes_only_whitelisted_names():
    proxy = dp_tools._NP_PROXY
    # whitelisted math is present
    assert hasattr(proxy, "mean")
    assert hasattr(proxy, "fft")
    assert hasattr(proxy, "linalg")
    # IO / module reach is absent
    for bad in ("save", "load", "loadtxt", "fromfile", "memmap", "lib",
                "core", "ctypes", "random"):
        with pytest.raises(AttributeError):
            getattr(proxy, bad)
    # fft / linalg sub-proxies are also whitelisted
    assert hasattr(proxy.fft, "fft2")
    assert hasattr(proxy.linalg, "lstsq")


def test_numpy_proxy_is_read_only():
    proxy = dp_tools._NP_PROXY
    with pytest.raises(AttributeError):
        setattr(proxy, "evil", 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
