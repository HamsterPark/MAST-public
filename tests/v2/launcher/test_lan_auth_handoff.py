"""Tests for the secure LAN-auth launcher→child credential handoff.

Covers the v2.1.14 fix: the LAN password must never travel through a
child process's environment block. The launcher now writes the credentials to
a short-lived, owner-only temp file and passes only its path via
``LAN_ENV_AUTH_FILE``; the child reads it once and deletes it.

Also pins a fix: the LAN dialog / status text interpolates the real
``GUI_PORT`` / ``ADMIN_PORT`` constants instead of the wrong 7860/7861.

These tests touch only pure stdlib helpers in ``mast2_launcher`` — no Tk, no
LLM, no network — so they run under the v2 venv without a display.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# ── v2 import bootstrap (canonical block; keeps mast.* resolving to MASTv2/) ──
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest

# ── Load the launcher module by file path ────────────────────────────
# mast2_launcher.py lives at the repo root (not under MASTv2/mast/), so we load
# it directly rather than via `import`. Its module-level code is import-safe.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER_PATH = _REPO_ROOT / "mast2_launcher.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("mast2_launcher", _LAUNCHER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── #136: round-trip via the ephemeral file ──────────────────────────

def test_write_then_consume_roundtrip(launcher, tmp_path):
    p = launcher.write_lan_auth_handoff(tmp_path, "alice", "s3cr3t-pw")
    assert p.exists(), "handoff file should be written"
    user, pwd = launcher.consume_lan_auth_handoff(str(p))
    assert (user, pwd) == ("alice", "s3cr3t-pw")
    # Consuming MUST delete the file so the secret doesn't outlive boot.
    assert not p.exists(), "handoff file must be deleted after consume"


def test_password_with_spaces_and_unicode(launcher, tmp_path):
    # Password is read line-wise and .strip()'d; internal spaces survive,
    # leading/trailing whitespace is intentionally trimmed.
    p = launcher.write_lan_auth_handoff(tmp_path, "用户", "pa ss 密码 #1")
    user, pwd = launcher.consume_lan_auth_handoff(str(p))
    assert user == "用户"
    assert pwd == "pa ss 密码 #1"


def test_consume_missing_path_is_safe(launcher, tmp_path):
    assert launcher.consume_lan_auth_handoff("") == ("", "")
    assert launcher.consume_lan_auth_handoff(str(tmp_path / "nope.tmp")) == ("", "")


def test_unique_file_per_call(launcher, tmp_path):
    p1 = launcher.write_lan_auth_handoff(tmp_path, "u", "p1")
    p2 = launcher.write_lan_auth_handoff(tmp_path, "u", "p2")
    assert p1 != p2, "each handoff must be a distinct file (no main/admin contention)"
    # Both independently readable + self-deleting.
    assert launcher.consume_lan_auth_handoff(str(p1))[1] == "p1"
    assert launcher.consume_lan_auth_handoff(str(p2))[1] == "p2"


def test_handoff_file_lives_in_dedicated_subdir(launcher, tmp_path):
    p = launcher.write_lan_auth_handoff(tmp_path, "u", "p")
    assert p.parent.name == launcher.LAN_AUTH_HANDOFF_DIR
    p.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_posix_file_is_owner_only(launcher, tmp_path):
    p = launcher.write_lan_auth_handoff(tmp_path, "u", "p")
    mode = p.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    p.unlink(missing_ok=True)


# ── #136: the child resolver prefers the file, deletes it, falls back ─

def test_read_from_env_prefers_file(launcher, tmp_path, monkeypatch):
    p = launcher.write_lan_auth_handoff(tmp_path, "bob", "filepw")
    monkeypatch.setenv(launcher.LAN_ENV_AUTH_FILE, str(p))
    # Legacy env present too — file must win.
    monkeypatch.setenv(launcher.LAN_ENV_USER, "stale")
    monkeypatch.setenv(launcher.LAN_ENV_PASS, "stalepw")
    user, pwd = launcher._read_lan_auth_from_env()
    assert (user, pwd) == ("bob", "filepw")
    assert not p.exists(), "resolver must consume (delete) the handoff file"


def test_read_from_env_legacy_fallback(launcher, monkeypatch):
    # No file flag → fall back to legacy plaintext env vars (back-compat).
    monkeypatch.delenv(launcher.LAN_ENV_AUTH_FILE, raising=False)
    monkeypatch.setenv(launcher.LAN_ENV_USER, "carol")
    monkeypatch.setenv(launcher.LAN_ENV_PASS, "legacypw")
    assert launcher._read_lan_auth_from_env() == ("carol", "legacypw")


def test_read_from_env_falls_through_when_file_empty(launcher, tmp_path, monkeypatch):
    # File flag points at a non-existent file → fall through to legacy env.
    monkeypatch.setenv(launcher.LAN_ENV_AUTH_FILE, str(tmp_path / "gone.tmp"))
    monkeypatch.setenv(launcher.LAN_ENV_USER, "dave")
    monkeypatch.setenv(launcher.LAN_ENV_PASS, "fallpw")
    assert launcher._read_lan_auth_from_env() == ("dave", "fallpw")


def test_read_from_env_empty_when_nothing_set(launcher, monkeypatch):
    for var in (launcher.LAN_ENV_AUTH_FILE, launcher.LAN_ENV_USER,
                launcher.LAN_ENV_PASS):
        monkeypatch.delenv(var, raising=False)
    assert launcher._read_lan_auth_from_env() == ("", "")


# ── #136: _make_subprocess_env keeps the password OUT of the env block ─

class _FakeLauncher:
    """Minimal stand-in exposing just what _make_subprocess_env touches."""

    def __init__(self, root, user, pwd, enabled=True):
        self.root_dir = root
        self._lan_user = user
        self._lan_pass = pwd
        self._lan_enabled = enabled
        self.logged: list[str] = []

    def _log(self, msg):
        self.logged.append(msg)


def test_make_env_does_not_leak_password(launcher, tmp_path):
    fake = _FakeLauncher(tmp_path, "eve", "TOPSECRET", enabled=True)
    env = launcher.LauncherApp._make_subprocess_env(fake)

    # The password must NOT appear anywhere in the env block values.
    assert "TOPSECRET" not in env.values()
    assert all("TOPSECRET" not in v for v in env.values())
    assert launcher.LAN_ENV_PASS not in env, "legacy plaintext var must be absent"

    # LAN enabled + username (not secret) present; auth handed off via file.
    assert env[launcher.LAN_ENV_ENABLE] == "1"
    assert env[launcher.LAN_ENV_USER] == "eve"
    handoff = env[launcher.LAN_ENV_AUTH_FILE]
    assert Path(handoff).exists()

    # And the child resolver can recover the real password from that file.
    user, pwd = launcher.consume_lan_auth_handoff(handoff)
    assert (user, pwd) == ("eve", "TOPSECRET")


def test_make_env_strips_lan_vars_when_disabled(launcher, tmp_path, monkeypatch):
    # Parent env has stale LAN vars; with LAN off the child must inherit none.
    monkeypatch.setenv(launcher.LAN_ENV_ENABLE, "1")
    monkeypatch.setenv(launcher.LAN_ENV_USER, "stale")
    monkeypatch.setenv(launcher.LAN_ENV_PASS, "stalepw")
    monkeypatch.setenv(launcher.LAN_ENV_AUTH_FILE, str(tmp_path / "x.tmp"))
    fake = _FakeLauncher(tmp_path, "", "", enabled=False)
    env = launcher.LauncherApp._make_subprocess_env(fake)
    for var in (launcher.LAN_ENV_ENABLE, launcher.LAN_ENV_USER,
                launcher.LAN_ENV_PASS, launcher.LAN_ENV_AUTH_FILE):
        assert var not in env, f"{var} should be stripped when LAN disabled"


# ── #136: sweep clears leaked handoff files ──────────────────────────

def test_sweep_clears_leftover_files(launcher, tmp_path):
    p1 = launcher.write_lan_auth_handoff(tmp_path, "a", "1")
    p2 = launcher.write_lan_auth_handoff(tmp_path, "b", "2")
    assert p1.exists() and p2.exists()
    launcher._sweep_lan_auth_handoff(tmp_path)
    assert not p1.exists() and not p2.exists()


def test_sweep_no_dir_is_noop(launcher, tmp_path):
    # Must not raise when the handoff dir was never created.
    launcher._sweep_lan_auth_handoff(tmp_path / "does-not-exist")


# ── single-port (admin GUI removed 2026-06-09): GUI 7862 only ────────

def test_single_port_no_admin(launcher):
    # Main GUI port stays 7862; the admin GUI + its 7863 port are gone.
    assert launcher.GUI_PORT == 7862
    assert not hasattr(launcher, "ADMIN_PORT")
    # The push-update server port is independent and stays.
    assert launcher.PUSH_SERVER_PORT == 8766


def test_no_stray_v1_ports_or_admin_in_lan_user_text(launcher):
    """LAN-facing strings reference only GUI_PORT now (no v1 7860/7861, no admin
    7863). We assert against the launcher source directly."""
    src = _LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "7860 / 7861" not in src
    assert "主 7860 / 管理员 7861" not in src
    # Admin-port interpolations must be gone after single-port convergence.
    assert "{GUI_PORT} / {ADMIN_PORT}" not in src
    assert "主 {GUI_PORT} / 管理员 {ADMIN_PORT}" not in src
    assert "ADMIN_PORT" not in src
