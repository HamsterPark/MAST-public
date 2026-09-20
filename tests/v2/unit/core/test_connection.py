"""v2 unit tests for mast.core.connection.

Ported from tests/unit/test_connection.py 2026-05-19. Tests cover ConnectionPool
safe_call retry/reconnect logic and _reconnect_role success/failure paths.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_connection.py -x -v
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

import socket
from unittest.mock import MagicMock, patch

import pytest

from mast.core.types import NanonisCallRecord  # noqa: F401 — imported for parity with v1


# ── Helpers ──────────────────────────────────────────────────────────


class FakeConfig:
    """Minimal stand-in for mast.config.NanonisConfig.

    ConnectionPool only reads host / port_* / timeout_s; a plain attribute
    container is enough and avoids pulling pydantic into the test path.
    """
    host = "localhost"
    port_main = 6501
    port_monitor = 6502
    port_data = 6503
    port_emergency = 6504
    timeout_s = 5.0


class FakeNanonis:
    """Minimal stand-in for Nanonis."""

    def __init__(self, sock):
        self._sock = sock

    def close(self):
        pass

    def Util_VersionGet(self):
        return ("", b"", ["V5e"])


# ── Tests: safe_call reconnect behaviour ─────────────────────────────


class TestSafeCallReconnect:
    """Test that safe_call retries once on socket errors."""

    def _make_pool(self):
        from mast.core.connection import ConnectionPool
        pool = ConnectionPool(FakeConfig())
        # Manually inject a connection for "main"
        mock_sock = MagicMock(spec=socket.socket)
        nn = FakeNanonis(mock_sock)
        pool._connections["main"] = (mock_sock, nn)
        return pool, nn

    def test_success_no_reconnect(self):
        pool, nn = self._make_pool()
        record = pool.safe_call("Util_VersionGet")
        assert not record.error
        assert record.return_value is not None

    def test_method_not_found(self):
        pool, nn = self._make_pool()
        record = pool.safe_call("NonexistentMethod")
        assert "not found" in record.error

    def test_reconnect_on_socket_error(self):
        """First call raises socket.error → reconnect → second call succeeds."""
        pool, nn = self._make_pool()

        # Make the first call raise a socket error
        call_count = [0]

        def _failing_then_ok():
            call_count[0] += 1
            if call_count[0] == 1:
                raise socket.error("Connection reset")
            return ("", b"", ["V5e"])

        nn.Util_VersionGet = _failing_then_ok

        # Mock _reconnect_role to inject a new FakeNanonis
        new_nn = FakeNanonis(MagicMock())
        new_nn.Util_VersionGet = lambda: ("", b"", ["V5e"])

        def _fake_reconnect(role):
            pool._connections[role] = (MagicMock(), new_nn)
            return True

        with patch.object(pool, "_reconnect_role", side_effect=_fake_reconnect):
            record = pool.safe_call("Util_VersionGet")
        assert not record.error
        assert record.return_value == ("", b"", ["V5e"])

    def test_reconnect_fails(self):
        """Socket error + reconnect fails → error returned."""
        pool, nn = self._make_pool()

        def _always_fail():
            raise socket.error("Connection reset")

        nn.Util_VersionGet = _always_fail

        # _reconnect_role will fail since there's no real server
        with patch.object(pool, "_reconnect_role", return_value=False):
            record = pool.safe_call("Util_VersionGet")
        assert record.error
        assert "socket" in record.error.lower() or "Connection" in record.error

    def test_connection_lost_reconnect(self):
        """get() raises ConnectionError → reconnect → retry."""
        from mast.core.connection import ConnectionPool
        pool = ConnectionPool(FakeConfig())
        # No connections at all

        with patch.object(pool, "_reconnect_role", return_value=False):
            record = pool.safe_call("Util_VersionGet")
        assert "Connection lost" in record.error

    def test_nanonis_logic_error_no_reconnect(self):
        """Nanonis error string (not socket error) should NOT trigger reconnect."""
        pool, nn = self._make_pool()

        def _nanonis_error():
            return ("Error: invalid parameter", b"", [])

        nn.SomeMethod = _nanonis_error

        record = pool.safe_call("SomeMethod")
        # Should have error from Nanonis but not trigger reconnect
        assert record.error == "Error: invalid parameter"


# ── Tests: _reconnect_role ───────────────────────────────────────────


class TestReconnectRole:
    def test_reconnect_role_success(self):
        from mast.core.connection import ConnectionPool
        pool = ConnectionPool(FakeConfig())

        with patch("mast.core.connection.socket.socket") as mock_sock_cls, \
             patch("mast.core.connection.Nanonis") as mock_nn_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            mock_nn_cls.return_value = FakeNanonis(mock_sock)

            result = pool._reconnect_role("main")

        assert result is True
        assert "main" in pool._connections

    def test_reconnect_role_failure(self):
        from mast.core.connection import ConnectionPool
        pool = ConnectionPool(FakeConfig())

        with patch("mast.core.connection.socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = OSError("Connection refused")
            mock_sock_cls.return_value = mock_sock

            result = pool._reconnect_role("main")

        assert result is False

    def test_reconnect_role_invalid(self):
        from mast.core.connection import ConnectionPool
        pool = ConnectionPool(FakeConfig())
        assert pool._reconnect_role("invalid_role") is False
