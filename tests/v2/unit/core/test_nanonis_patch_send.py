"""Regression tests for the ``Nanonis.send`` hardening patch (dispatch audit
2026-07-28 致命三 "两颗炸弹").

nanonis_spm v1.0.9 ``Nanonis.send`` has two defects that make a MAST worker
thread unkillable:

* the body loop calls ``recv`` until the declared length is reached, but after
  the peer's EOF ``recv`` returns ``b''`` *without blocking* — the timeout
  never fires, the length never catches up, and the loop spins at 100 % CPU
  forever without returning or raising;
* every successful call ends with ``settimeout(1000)``, so the 5 s recv
  timeout only ever applies to the first command on a socket.

Both leave the per-role lock in ``ConnectionPool.safe_call`` held forever,
which is what turns one dead socket into a whole-service stall (and what makes
the emergency-retract fallback on ``role="main"`` queue behind the very
failure it is supposed to rescue).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/core/test_nanonis_patch_send.py -x -v
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

import struct
import threading
import time

import pytest


# ─────────────────────────────────────────────────────────────────────
# A fake socket that can be scripted with an EOF at any point
# ─────────────────────────────────────────────────────────────────────

def _frame(command: str, body: bytes) -> bytes:
    """Build a Nanonis response frame the way the RT controller does."""
    header = bytearray(str(command).ljust(32, "\0").encode())
    header += len(body).to_bytes(4, byteorder="big")
    header += b"\x00\x00\x00\x00"  # response flag + zero buffer
    assert len(header) == 40
    return bytes(header) + body


class _FakeSocket:
    """Scriptable socket. ``chunks`` are handed out in order; once exhausted
    every further ``recv`` returns ``b''`` — i.e. peer EOF, exactly the state
    that makes upstream spin."""

    def __init__(self, chunks: list[bytes], timeout: float | None = 5.0):
        self._pending = bytearray(b"".join(chunks))
        self._chunk_sizes = [len(c) for c in chunks]
        self._timeout = timeout
        self.sent: list[bytes] = []
        self.recv_calls = 0

    # -- socket API surface used by the patch --
    def sendall(self, data):
        self.sent.append(bytes(data))

    def send(self, data):  # pragma: no cover — sendall is preferred
        self.sent.append(bytes(data))
        return len(data)

    def recv(self, n):
        self.recv_calls += 1
        if not self._pending:
            return b""  # peer EOF — returns IMMEDIATELY, no timeout
        # Hand back at most the next scripted chunk, so fragmentation is real.
        take = min(n, self._chunk_sizes.pop(0) if self._chunk_sizes else n)
        out = bytes(self._pending[:take])
        del self._pending[:take]
        return out

    def gettimeout(self):
        return self._timeout

    def settimeout(self, value):
        self._timeout = value


def _real_nanonis_class():
    """Load the GENUINE nanonis_spm.Nanonis from disk.

    ``tests/conftest.py`` replaces ``sys.modules["nanonis_spm"]`` with a
    MagicMock so the suite runs without hardware — which also means the class
    the patch is installed on is a mock, and a mock would happily "pass" any
    framing assertion. These tests need the real ``handleString`` /
    ``handleArray`` / ``correctType`` helpers, so we import the module file
    directly and bind the patched ``send`` to that class.
    """
    import importlib.util

    for base in sys.path + [str(Path(sys.executable).resolve().parents[1])]:
        cand = Path(base) / "nanonis_spm" / "NanonisClass.py"
        if cand.exists():
            break
    else:
        cand = None
    if cand is None:
        # site-packages of the running interpreter
        import sysconfig

        cand = Path(sysconfig.get_paths()["purelib"]) / "nanonis_spm" / "NanonisClass.py"
    if not cand.exists():
        pytest.skip("nanonis_spm package not installed on disk")

    spec = importlib.util.spec_from_file_location("_real_nanonis_class", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from mast.core import nanonis_patch

    mod.Nanonis.send = nanonis_patch._patched_send
    return mod.Nanonis


def _nano(sock):
    return _real_nanonis_class()(sock)


# ─────────────────────────────────────────────────────────────────────
# The patch is installed
# ─────────────────────────────────────────────────────────────────────

def test_send_is_patched_on_the_class():
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    assert Nanonis.send is nanonis_patch._patched_send


def test_revert_and_reapply_round_trips_send():
    from nanonis_spm import Nanonis

    from mast.core import nanonis_patch

    nanonis_patch.revert()
    try:
        assert Nanonis.send is nanonis_patch._original_send
    finally:
        nanonis_patch.apply()
    assert Nanonis.send is nanonis_patch._patched_send


# ─────────────────────────────────────────────────────────────────────
# Bomb 2 — EOF must raise, not spin
# ─────────────────────────────────────────────────────────────────────

def test_eof_mid_body_raises_instead_of_spinning():
    """Header promises 16 bytes, peer delivers 4 then closes."""
    from mast.core import nanonis_patch  # noqa: F401 — ensures patch applied

    header = _frame("Util.VersionGet", b"x" * 16)[:40]
    sock = _FakeSocket([header, b"abcd"])
    nn = _nano(sock)

    t0 = time.perf_counter()
    with pytest.raises(ConnectionError):
        nn.send("Util.VersionGet", [], [])
    # Upstream would never get here at all; assert we also did not burn time.
    assert time.perf_counter() - t0 < 1.0
    # And it is a *ConnectionError* subclass so ConnectionPool.safe_call's
    # `except (socket.error, OSError, ConnectionError)` drives reconnect.
    assert issubclass(ConnectionAbortedError, ConnectionError)


def test_eof_on_header_raises():
    from mast.core import nanonis_patch  # noqa: F401

    sock = _FakeSocket([b"\x00" * 8])  # 8 of the 40 header bytes, then EOF
    nn = _nano(sock)

    with pytest.raises(ConnectionError):
        nn.send("Util.VersionGet", [], [])


def test_eof_immediately_raises_and_does_not_hang_the_thread():
    """The original defect, stated as a liveness property: a thread that
    calls send() on a dead peer must be finished a moment later."""
    from mast.core import nanonis_patch  # noqa: F401

    sock = _FakeSocket([])  # closed before anything arrived
    nn = _nano(sock)
    done = threading.Event()

    def _worker():
        try:
            nn.send("Util.VersionGet", [], [])
        except Exception:  # noqa: BLE001 — any raise is a pass, a hang is not
            pass
        finally:
            done.set()

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    assert done.wait(timeout=5.0), "send() did not return on a dead peer"


# ─────────────────────────────────────────────────────────────────────
# Bomb 1 — the 1000 s timeout leak
# ─────────────────────────────────────────────────────────────────────

def test_timeout_is_restored_after_a_successful_call():
    from mast.core import nanonis_patch  # noqa: F401

    body = struct.pack(">i", 7)
    sock = _FakeSocket([_frame("Util.VersionGet", body)], timeout=5.0)
    nn = _nano(sock)

    out = nn.send("Util.VersionGet", [], [])

    assert out == body
    # Upstream leaves 1000.0 here. That is the whole bug.
    assert sock.gettimeout() == 5.0


def test_timeout_is_restored_even_when_the_peer_dies():
    from mast.core import nanonis_patch  # noqa: F401

    header = _frame("Util.VersionGet", b"x" * 16)[:40]
    sock = _FakeSocket([header, b"ab"], timeout=5.0)
    nn = _nano(sock)

    with pytest.raises(ConnectionError):
        nn.send("Util.VersionGet", [], [])
    # A socket about to be reconnected must not carry 1000 s into its successor.
    assert sock.gettimeout() == 5.0


def test_poisoned_timeout_is_clamped_back_to_the_default():
    """A socket that already went through an UNPATCHED call carries 1000 s.
    The patch must not faithfully preserve that."""
    from mast.core import nanonis_patch

    body = struct.pack(">i", 1)
    sock = _FakeSocket([_frame("Util.VersionGet", body)], timeout=1000.0)
    nn = _nano(sock)

    nn.send("Util.VersionGet", [], [])

    assert sock.gettimeout() == nanonis_patch._default_recv_timeout_s
    assert sock.gettimeout() < 1000.0


def test_blocking_socket_gets_a_finite_timeout():
    from mast.core import nanonis_patch

    body = struct.pack(">i", 1)
    sock = _FakeSocket([_frame("Util.VersionGet", body)], timeout=None)
    nn = _nano(sock)

    nn.send("Util.VersionGet", [], [])

    assert sock.gettimeout() == nanonis_patch._default_recv_timeout_s


def test_set_default_recv_timeout_rejects_nonsense():
    from mast.core import nanonis_patch

    original = nanonis_patch._default_recv_timeout_s
    try:
        nanonis_patch.set_default_recv_timeout(2.5)
        assert nanonis_patch._default_recv_timeout_s == 2.5
        for bad in (0, -1, 1000.0, 99999, "abc", None):
            nanonis_patch.set_default_recv_timeout(bad)
            assert nanonis_patch._default_recv_timeout_s == 2.5
    finally:
        nanonis_patch.set_default_recv_timeout(original)


def test_connection_pool_publishes_its_configured_timeout():
    """connect_all() must hand its recv timeout to the patch, or the patch's
    own 5 s default would silently override the config."""
    from mast.config import NanonisConfig
    from mast.core import nanonis_patch
    from mast.core.connection import ConnectionPool

    original = nanonis_patch._default_recv_timeout_s
    try:
        cfg = NanonisConfig(host="127.0.0.1", timeout_s=3.5)
        pool = ConnectionPool(cfg)
        pool.connect_all()  # every port fails (nothing listening) — that's fine
        assert nanonis_patch._default_recv_timeout_s == 3.5
    finally:
        nanonis_patch.set_default_recv_timeout(original)


# ─────────────────────────────────────────────────────────────────────
# Framing parity with upstream
# ─────────────────────────────────────────────────────────────────────

def test_request_framing_matches_upstream():
    from mast.core import nanonis_patch  # noqa: F401

    body = struct.pack(">i", 0)
    sock = _FakeSocket([_frame("ZCtrl.OnOffSet", body)])
    nn = _nano(sock)

    nn.send("ZCtrl.OnOffSet", [1], ["I"])

    assert len(sock.sent) == 1
    msg = sock.sent[0]
    assert msg[:32] == b"ZCtrl.OnOffSet".ljust(32, b"\0")
    assert struct.unpack(">I", msg[32:36])[0] == 4       # body size
    assert struct.unpack(">H", msg[36:38])[0] == 1       # send-response-back
    assert msg[38:40] == b"\x00\x00"
    assert struct.unpack(">I", msg[40:])[0] == 1         # the argument


def test_fragmented_body_is_reassembled():
    from mast.core import nanonis_patch  # noqa: F401

    body = struct.pack(">4i", 1, 2, 3, 4)
    frame = _frame("Sig.ValsGet", body)
    # header split in two, body split in three
    sock = _FakeSocket([frame[:12], frame[12:40],
                        body[:5], body[5:9], body[9:]])
    nn = _nano(sock)

    out = nn.send("Sig.ValsGet", [], [])
    assert out == body


def test_zero_length_body_does_not_call_recv_a_thousand_times():
    """Upstream burns 1000 recv(0) syscalls on EVERY call because the loop
    condition is `... or counter < 1000`."""
    from mast.core import nanonis_patch  # noqa: F401

    sock = _FakeSocket([_frame("Scan.Action", b"")])
    nn = _nano(sock)

    out = nn.send("Scan.Action", [], [])
    assert out == b""
    assert sock.recv_calls <= 2


def test_wrong_command_still_returns_empty_list():
    """Parity: upstream prints and returns [] when the reply names another
    command. Behaviour preserved."""
    from mast.core import nanonis_patch  # noqa: F401

    sock = _FakeSocket([_frame("Some.Other", struct.pack(">i", 1))])
    nn = _nano(sock)

    assert nn.send("Util.VersionGet", [], []) == []


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
