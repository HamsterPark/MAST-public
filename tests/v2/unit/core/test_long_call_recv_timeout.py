"""长命令可在调用期间单独提高 socket 接收超时，并在结束或异常后恢复原值。
测试必须观察调用过程中的 socket 超时，不能只验证参数透传或没有报错。"""
from __future__ import annotations

import pytest

from mast.core.connection import _MAX_CALL_RECV_TIMEOUT_S, ConnectionPool


class _FakeSock:
    def __init__(self, timeout=5.0):
        self._to = timeout
        self.seen: list[float | None] = []

    def gettimeout(self):
        return self._to

    def settimeout(self, v):
        self._to = v
        self.seen.append(v)


class _FakeNanonis:
    """记录方法被调用**当时** socket 上的超时值。"""

    def __init__(self, sock):
        self._sock = sock
        self.timeout_during_call: float | None = None

    def SlowStart(self, *args):
        self.timeout_during_call = self._sock.gettimeout()
        return ("", b"", [])

    def QuickGet(self, *args):
        self.timeout_during_call = self._sock.gettimeout()
        return ("", b"", [1.0])


def _pool_with(sock, nn) -> ConnectionPool:
    pool = ConnectionPool.__new__(ConnectionPool)          # 不跑 __init__（会连真 TCP）
    import threading

    pool._connections = {"main": (sock, nn)}
    pool._struct_lock = threading.RLock()
    pool._locks = {"main": threading.RLock()}
    pool._closed = False

    class _AlwaysAllow:
        def allow(self):
            return True

    pool._breaker = _AlwaysAllow()
    pool.get = lambda role="main": nn                       # type: ignore[assignment]
    pool._role_lock = lambda role: pool._locks["main"]      # type: ignore[assignment]
    pool._on_comms_success = lambda *a, **k: None           # type: ignore[assignment]
    pool._on_comms_failure = lambda *a, **k: None           # type: ignore[assignment]
    return pool


def test_long_call_raises_timeout_during_the_call():
    sock = _FakeSock(timeout=5.0)
    nn = _FakeNanonis(sock)
    pool = _pool_with(sock, nn)

    rec = pool.safe_call("SlowStart", 1, "", recv_timeout_s=600.0)

    assert not rec.error, rec.error
    # 1) 抬高**落到了 socket 上**——不是只把参数传下去就算数
    assert nn.timeout_during_call == pytest.approx(600.0), (
        "调用期间 socket 超时还是 %r —— 抬高没落地，回包依旧会落在没人读的"
        "socket 上" % (nn.timeout_during_call,))
    # 2) 还原成**调用前那个值**
    assert sock.gettimeout() == pytest.approx(5.0)


def test_budget_is_clamped_below_the_library_bogus_value():
    """必须低于 ``nanonis_patch._LIB_BOGUS_TIMEOUT_S``(1000 s)。

    到了那个值，patch 会把它当成 nanonis_spm 泄漏出来的假超时而换回默认值，
    抬高就白做了 —— 而且这种失效是**静默**的。
    """
    from mast.core import nanonis_patch

    assert _MAX_CALL_RECV_TIMEOUT_S < nanonis_patch._LIB_BOGUS_TIMEOUT_S

    sock = _FakeSock(timeout=5.0)
    nn = _FakeNanonis(sock)
    pool = _pool_with(sock, nn)
    pool.safe_call("SlowStart", recv_timeout_s=99999.0)
    assert nn.timeout_during_call == pytest.approx(_MAX_CALL_RECV_TIMEOUT_S)
    assert nn.timeout_during_call < nanonis_patch._LIB_BOGUS_TIMEOUT_S


def test_default_path_touches_nothing():
    """不传 ``recv_timeout_s`` 时，socket 上一个字节都不该改。"""
    sock = _FakeSock(timeout=5.0)
    nn = _FakeNanonis(sock)
    pool = _pool_with(sock, nn)

    pool.safe_call("QuickGet")

    assert sock.seen == [], "默认路径动了 socket 超时：%r" % (sock.seen,)
    assert nn.timeout_during_call == pytest.approx(5.0)


def test_timeout_restored_even_when_the_call_raises():
    sock = _FakeSock(timeout=5.0)

    class _Boom(_FakeNanonis):
        def SlowStart(self, *args):
            self.timeout_during_call = self._sock.gettimeout()
            raise OSError("link died mid-sweep")

    nn = _Boom(sock)
    pool = _pool_with(sock, nn)
    pool._log_reconnect_attempt = lambda *a, **k: None      # type: ignore[assignment]
    pool._reconnect_role = lambda role: False               # type: ignore[assignment]

    rec = pool.safe_call("SlowStart", recv_timeout_s=600.0)

    assert rec.error and "OSError" in rec.error
    assert nn.timeout_during_call == pytest.approx(600.0)
    assert sock.gettimeout() == pytest.approx(5.0), "异常路径没还原超时"


# ── ExecutionContext 只在真要用时才透传 ──────────────────────────────────────

class _StrictPool:
    """只认**改动前**签名的 pool —— 多一个关键字就 TypeError。

    钉的是 2026-09-09 否掉的那版设计：`execution_context.safe_call` 曾经
    **无条件**把 `recv_timeout_s=None` 往下传，于是每一个 duck-typed 的 pool
    （测试替身、别处的适配器）都必须跟着加这个关键字，哪怕那次调用根本不用
    预算 —— 一口气弄红了 5 个 abort-gate 测试。默认路径必须与改动前逐字节相同。
    """

    def __init__(self):
        self.seen = []

    def safe_call(self, method_name, *args, role="main"):
        self.seen.append((method_name, args, role))
        from mast.core.types import NanonisCallRecord
        return NanonisCallRecord(method=method_name, args=args)


def _ctx_with(pool):
    from mast.core.execution_context import ExecutionContext
    ctx = ExecutionContext.__new__(ExecutionContext)
    ctx.pool = pool
    ctx.run_id = "t"
    ctx.check_abort = lambda: False
    return ctx


def test_context_default_path_does_not_forward_the_new_kwarg():
    pool = _StrictPool()
    rec = _ctx_with(pool).safe_call("Current_Get")
    assert not rec.error, rec.error
    assert pool.seen == [("Current_Get", (), "main")]


def test_context_forwards_it_only_when_asked():
    got = {}

    class _P(_StrictPool):
        def safe_call(self, method_name, *args, role="main", recv_timeout_s=None):
            got["v"] = recv_timeout_s
            from mast.core.types import NanonisCallRecord
            return NanonisCallRecord(method=method_name, args=args)

    _ctx_with(_P()).safe_call("BiasSpectr_Start", 1, "", recv_timeout_s=615.3)
    assert got["v"] == pytest.approx(615.3)
