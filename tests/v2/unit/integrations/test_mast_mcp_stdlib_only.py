"""The MAST MCP server imports the Python standard library and nothing else.

Why it matters: the server has to start on MAST's bundled embeddable Python
(no pip, isolated ``._pth`` path mode), on the development venv, and on any
Python >= 3.10 a user happens to have, without installing a package. One stray
third-party import breaks the first of those without breaking the other two.

``platform`` and ``uuid`` are banned outright: on some Windows Pythons
``platform`` queries WMI and has been seen to hang at import time.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (same bootstrap as test_wrap_skill_minimal) ──
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]
_SERVER_DIR = _REPO / "integrations" / "claude-code" / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import ast
import json
import os
import subprocess

import pytest

PACKAGE = "mast_mcp"
BANNED = {"platform", "uuid"}
EMBEDDED_PYTHON = _REPO / "MASTv2" / "pyruntime" / "python.exe"


def _server_files() -> list[Path]:
    files = sorted(_SERVER_DIR.rglob("*.py"))
    assert len(files) >= 7, f"expected run_server.py and the {PACKAGE} package, got {files}"
    return files


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "__import__":
            pytest.fail(f"{path.name}: dynamic __import__ hides imports from this check")
    return names


def test_server_imports_only_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {PACKAGE}
    bad = {f"{p.relative_to(_SERVER_DIR).as_posix()}: {name}"
           for p in _server_files() for name in _top_level_imports(p) if name not in allowed}
    assert not bad, "non-stdlib imports in the MCP server:\n  " + "\n  ".join(sorted(bad))


def test_server_never_imports_platform_or_uuid():
    bad = {f"{p.name}: {name}" for p in _server_files()
           for name in _top_level_imports(p) if name in BANNED}
    assert not bad, f"banned modules imported: {sorted(bad)}"


def test_platform_and_uuid_are_not_pulled_in_through_the_stdlib(tmp_path):
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import mast_mcp.stdio_rpc, mast_mcp.config, mast_mcp.client, mast_mcp.tools, "
        "mast_mcp.resources; "
        "print(sorted(m for m in ('platform', 'uuid') if m in sys.modules))")
    proc = subprocess.run([sys.executable, "-c", probe, str(_SERVER_DIR)], capture_output=True,
                          text=True, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_only_stdio_rpc_speaks_the_wire_protocol():
    """Swapping in an official MCP SDK must touch one file: keep the protocol in it."""
    pkg = _SERVER_DIR / PACKAGE
    for p in sorted(pkg.glob("*.py")):
        if p.name == "stdio_rpc.py":
            continue
        text = p.read_text(encoding="utf-8")
        for marker in ('"jsonrpc"', "sys.stdin", "sys.stdout"):
            assert marker not in text, f"{p.name} mentions {marker}; keep it in stdio_rpc.py"


@pytest.mark.skipif(not EMBEDDED_PYTHON.is_file(),
                    reason="MASTv2/pyruntime/python.exe not present (it is built by "
                           "MASTv2/scripts/build_pyruntime.py and not kept in git)")
def test_the_embedded_runtime_starts_the_server_and_lists_the_tools(tmp_path):
    """MAST's bundled Python runs in isolated ``._pth`` mode: the script directory is
    not on sys.path there, so ``run_server.py`` has to put it there itself."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    payload = "".join(json.dumps(m) + "\n" for m in messages).encode("utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MAST_")}
    env["MAST_URL"] = "http://127.0.0.1:9"
    proc = subprocess.run([str(EMBEDDED_PYTHON), str(_SERVER_DIR / "run_server.py")],
                          input=payload, capture_output=True, timeout=60, env=env,
                          cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    lines = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line]
    by_id = {m["id"]: m for m in lines}
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    assert len(by_id[2]["result"]["tools"]) == 20
