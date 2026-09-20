"""MCP stdio protocol of the MAST MCP server, driven as a real subprocess.

What an MCP client relies on: version negotiation, the tool list and its
schemas, JSON-RPC error codes, and a stdout that carries protocol lines and
nothing else. The server is started exactly the way Claude Code starts it
(``python run_server.py``), fed newline-delimited JSON on stdin, and read back
from stdout after stdin closes.
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

import json
import os
import subprocess

import pytest

RUN_SERVER = _SERVER_DIR / "run_server.py"

EXPECTED_TOOLS = {
    "mast_briefing", "mast_status", "mast_find_skills", "mast_skill_card", "mast_run",
    "mast_job", "mast_jobs", "mast_cancel", "mast_emergency_stop", "mast_scope",
    "mast_list_data", "mast_fetch", "mast_note_write", "mast_note_search",
    "mast_ask_operator", "mast_operator_reply", "mast_composite_draft",
    "mast_composite_save", "mast_propose_skill", "mast_handover",
}


def _env(**overrides: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MAST_")}
    # Port 9 on loopback: a valid, local address that nothing in these tests calls.
    env.update({"MAST_URL": "http://127.0.0.1:9", "MAST_MCP_LOG": "WARNING",
                "PYTHONIOENCODING": "cp1252"})   # a hostile text encoding must not matter
    env.update(overrides)
    return env


def _initialize(version: str = "2025-06-18", mid=1) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "method": "initialize",
            "params": {"protocolVersion": version, "capabilities": {},
                       "clientInfo": {"name": "pytest", "version": "0"}}}


def _exchange(messages: list, *, env: dict | None = None, python: str | None = None,
              cwd: Path | None = None, timeout: float = 60.0):
    """Run the server over ``messages``; return (parsed responses, raw stdout, stderr, code)."""
    payload = b"".join((m if isinstance(m, bytes) else json.dumps(m).encode("utf-8")) + b"\n"
                       for m in messages)
    proc = subprocess.run([python or sys.executable, str(RUN_SERVER)], input=payload,
                          capture_output=True, timeout=timeout, env=env or _env(),
                          cwd=str(cwd) if cwd else None)
    lines = [line for line in proc.stdout.split(b"\n") if line.strip()]
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, ValueError) as exc:  # pragma: no cover - the assertion
            pytest.fail(f"stdout line is not JSON ({exc}): {line[:200]!r}\n"
                        f"stderr: {proc.stderr.decode('utf-8', 'replace')[-2000:]}")
    return parsed, proc.stdout, proc.stderr, proc.returncode


def _by_id(responses: list) -> dict:
    return {r.get("id"): r for r in responses if isinstance(r, dict)}


@pytest.mark.parametrize("version", ["2025-06-18", "2025-03-26", "2024-11-05"])
def test_initialize_echoes_a_supported_version(version, tmp_path):
    out, _, _, code = _exchange([_initialize(version)], cwd=tmp_path)
    assert code == 0
    assert out[0]["result"]["protocolVersion"] == version


@pytest.mark.parametrize("version", ["1999-01-01", "2099-12-31", None])
def test_initialize_answers_the_latest_for_an_unknown_version(version, tmp_path):
    out, _, _, _ = _exchange([_initialize(version)], cwd=tmp_path)
    assert out[0]["result"]["protocolVersion"] == "2025-06-18"


def test_initialize_result_carries_capabilities_and_rules(tmp_path):
    out, _, _, _ = _exchange([_initialize()], cwd=tmp_path)
    result = out[0]["result"]
    assert result["serverInfo"]["name"] == "mast"
    assert result["serverInfo"]["version"]
    assert "tools" in result["capabilities"] and "resources" in result["capabilities"]
    rules = result["instructions"]
    for must in ("mast_briefing", "mast_scope", "mast_job", "refused_busy", "lost_on_restart",
                 "mast_ask_operator", "SAFE", "mast_handover"):
        assert must in rules, f"instructions lost the rule about {must!r}"


def test_tools_list_has_every_tool_with_a_schema(tmp_path):
    out, _, _, _ = _exchange([_initialize(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}],
                             cwd=tmp_path)
    tools = _by_id(out)[2]["result"]["tools"]
    names = [t["name"] for t in tools]
    assert set(names) == EXPECTED_TOOLS and len(names) == len(EXPECTED_TOOLS)
    for t in tools:
        schema = t["inputSchema"]
        assert schema["type"] == "object", t["name"]
        assert isinstance(schema.get("properties"), dict), t["name"]
        for req in schema.get("required", []):
            assert req in schema["properties"], f"{t['name']}: required {req!r} undeclared"
        assert len(t["description"]) > 40, t["name"]
        for prop, spec in schema["properties"].items():
            assert spec.get("description"), f"{t['name']}.{prop} has no description"


def test_unknown_method_is_method_not_found(tmp_path):
    out, _, _, _ = _exchange([_initialize(), {"jsonrpc": "2.0", "id": 7, "method": "foo/bar"}],
                             cwd=tmp_path)
    assert _by_id(out)[7]["error"]["code"] == -32601


def test_unknown_tool_is_invalid_params(tmp_path):
    out, _, _, _ = _exchange([_initialize(), {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                              "params": {"name": "mast_nope", "arguments": {}}}],
                             cwd=tmp_path)
    assert _by_id(out)[3]["error"]["code"] == -32602


def test_stdout_carries_protocol_lines_and_nothing_else(tmp_path):
    messages = [
        _initialize(),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        b"this is not json",
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled",
         "params": {"requestId": 99, "reason": "nothing is running"}},
        {"jsonrpc": "2.0", "id": "s-4", "method": "tools/call",
         "params": {"name": "mast_status", "arguments": {}}},     # port 9: connection refused
        {"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 6, "method": "no/such/method"},
    ]
    # DEBUG: every log line the server can emit is emitted, so a log handler that
    # writes to stdout (or a stray print) shows up here as a non-JSON line.
    out, raw, stderr, code = _exchange(messages, env=_env(MAST_MCP_LOG="DEBUG"), cwd=tmp_path)
    assert code == 0, stderr.decode("utf-8", "replace")
    assert b"mast-mcp" in stderr, "logging went nowhere; the purity check needs it on"
    assert raw.endswith(b"\n") and b"\r" not in raw
    for msg in out:
        assert msg.get("jsonrpc") == "2.0"
        assert "id" in msg and ("result" in msg) != ("error" in msg)
    ids = [m["id"] for m in out]
    # one answer per request with an id, one parse error, nothing for the notifications
    assert sorted(map(str, ids)) == sorted(map(str, [1, 2, None, 3, "s-4", 5, 6]))
    by_id = _by_id(out)
    assert by_id[None]["error"]["code"] == -32700
    assert by_id[2]["result"] == {}
    call = by_id["s-4"]["result"]
    assert call["isError"] is True and "127.0.0.1:9" in call["content"][0]["text"]


def test_remote_address_is_refused_without_allow_remote(tmp_path):
    env = _env(MAST_URL="http://192.0.2.10:7862")
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "mast_status", "arguments": {}}}
    out, _, _, _ = _exchange([_initialize(), call], env=env, cwd=tmp_path)
    result = _by_id(out)[2]["result"]
    text = result["content"][0]["text"]
    assert result["isError"] is True
    assert "not this machine" in text and "MAST_ALLOW_REMOTE" in text and "VPN" in text


def test_resources_list_is_empty_when_the_guide_is_missing(tmp_path):
    env = _env(MAST_GUIDE_DIR=str(tmp_path / "no-guide-here"))
    out, _, _, _ = _exchange([_initialize(),
                              {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
                              {"jsonrpc": "2.0", "id": 3, "method": "resources/templates/list"}],
                             env=env, cwd=tmp_path)
    by_id = _by_id(out)
    assert by_id[2]["result"] == {"resources": []}
    assert by_id[3]["result"]["resourceTemplates"][0]["uriTemplate"] == "mast://guide/{lang}/{file}"


def test_resources_serve_the_guide_in_both_languages_and_nothing_else(tmp_path):
    guide = tmp_path / "guide"
    (guide / "en").mkdir(parents=True)
    (guide / "zh").mkdir(parents=True)
    (guide / "en" / "README.md").write_text("# MAST guide\n\nHello.\n", encoding="utf-8")
    (guide / "zh" / "README.md").write_text("# MAST 指南\n\n扫描隧道显微镜。\n", encoding="utf-8")
    (tmp_path / "secret.md").write_text("outside the guide\n", encoding="utf-8")
    env = _env(MAST_GUIDE_DIR=str(guide))
    read = [{"jsonrpc": "2.0", "id": 10 + i, "method": "resources/read", "params": {"uri": uri}}
            for i, uri in enumerate(["mast://guide/zh/README.md", "mast://guide/en/README.md",
                                     "mast://guide/zh/../../secret.md", "mast://guide/fr/README.md",
                                     "file:///etc/passwd"])]
    out, _, _, _ = _exchange([_initialize(), {"jsonrpc": "2.0", "id": 2,
                                              "method": "resources/list"}] + read,
                             env=env, cwd=tmp_path)
    by_id = _by_id(out)
    uris = {r["uri"] for r in by_id[2]["result"]["resources"]}
    assert uris == {"mast://guide/en/README.md", "mast://guide/zh/README.md"}
    zh = by_id[10]["result"]["contents"][0]
    assert zh["text"] == "# MAST 指南\n\n扫描隧道显微镜。\n" and zh["mimeType"] == "text/markdown"
    assert by_id[11]["result"]["contents"][0]["text"].startswith("# MAST guide")
    for mid in (12, 13, 14):
        assert by_id[mid]["error"]["code"] == -32002


def test_a_batch_gets_one_array_answer(tmp_path):
    batch = json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                        {"jsonrpc": "2.0", "method": "notifications/initialized"},
                        {"jsonrpc": "2.0", "id": 2, "method": "foo"}]).encode()
    out, _, _, _ = _exchange([batch], cwd=tmp_path)
    assert len(out) == 1 and isinstance(out[0], list)
    answers = {a["id"]: a for a in out[0]}
    assert answers[1]["result"] == {} and answers[2]["error"]["code"] == -32601
