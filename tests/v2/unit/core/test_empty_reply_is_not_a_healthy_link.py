"""空回包表示没有获得有效响应，不能被记作通信成功。

空元组不应清零失败计数或阻止重连；应用层错误文本则说明确实完成了通信往返。
测试区分两者，避免半开连接持续返回空值却被标作健康。"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.connection import ConnectionPool  # noqa: E402


class _Config:
    """与 test_connection.py 同一个替身形状 —— 不另造第二种。"""

    host = "localhost"
    port_main = 6501
    port_monitor = 6502
    port_data = 6503
    port_emergency = 6504
    timeout_s = 5.0


_GOOD = ("", b"\x00", [1.2e-10])
_EMPTY: tuple = ()


class _Nanonis:
    """按剧本逐次回包;超出剧本就一直回最后一个。"""

    def __init__(self, replies) -> None:
        self._replies = list(replies)
        self.calls = 0

    def close(self) -> None:
        pass

    def Current_Get(self):
        self.calls += 1
        return self._replies[min(self.calls - 1, len(self._replies) - 1)]


def _pool(replies):
    """真的 ConnectionPool,注入一个按剧本作答的假连接。"""
    pool = ConnectionPool(_Config())
    nn = _Nanonis(replies)
    pool._connections["main"] = (MagicMock(spec=socket.socket), nn)
    # 重连:换上一条**新剧本**的连接(空包之后回好包),并记下被调过。
    calls: list = []

    def _reconnect(role, _pool=pool, _calls=calls):
        _calls.append(role)
        fresh = _Nanonis(replies[1:] or [_GOOD])
        _pool._connections[role] = (MagicMock(spec=socket.socket), fresh)
        return True

    pool._reconnect_role = _reconnect          # type: ignore[assignment]
    return pool, nn, calls


def test_a_good_reply_is_a_success():
    """正常回包照旧 —— 不许被这次改动波及。"""
    pool, _nn, calls = _pool([_GOOD])
    rec = pool.safe_call("Current_Get")
    assert not rec.error
    assert rec.return_value == _GOOD
    assert calls == [], "正常回包触发了重连"


def test_an_empty_reply_is_not_reported_as_success():
    """空回包**不许**当成成功。这是整条链路卡死的第一步。"""
    pool, _nn, _calls = _pool([_EMPTY, _EMPTY])
    rec = pool.safe_call("Current_Get")
    assert rec.error, (
        "空回包被当成了成功 —— 于是熔断器被清零、重连永不触发,"
        "这个 socket 会持续返回空值，因此应进入重连流程。")
    assert "空包" in rec.error or "没有作答" in rec.error


def test_an_empty_reply_triggers_a_reconnect():
    """而且要**试着修**:退避重连必须被调到,不能只是报错了事。"""
    pool, _nn, calls = _pool([_EMPTY, _EMPTY])
    pool.safe_call("Current_Get")
    assert calls, "空回包没有触发任何重连尝试"


def test_a_reconnect_that_fixes_it_returns_the_good_reply():
    """重连之后对端答上来了 ⇒ 这一次调用就该成功(自愈,调用方无感)。"""
    pool, _nn, calls = _pool([_EMPTY, _GOOD])
    rec = pool.safe_call("Current_Get")
    assert calls, "没有尝试重连"
    assert not rec.error, f"重连成功了却仍然报错:{rec.error}"
    assert rec.return_value == _GOOD


def test_an_application_error_string_is_still_a_healthy_link():
    """反例钉子:Nanonis 回「参数不对」是一次**成功的往返**,链路是好的。

    没有这一条,上面几条会诱使人把「有 error 就当断链」—— 那会在每一个
    应用层错误上重连一次,把好端端的链路拆了。
    """
    pool, _nn, calls = _pool([("Error: bad parameter", b"", [])])
    rec = pool.safe_call("Current_Get")
    assert rec.error == "Error: bad parameter"
    assert calls == [], "应用层错误不该触发重连"


def test_the_error_text_says_it_is_a_link_problem_not_a_zero_reading():
    """持续空回包应被诊断为链路未应答，避免让调用者误查针尖或样品。"""
    pool, _nn, _calls = _pool([_EMPTY, _EMPTY])
    rec = pool.safe_call("Current_Get")
    assert "没有作答" in rec.error or "没在应答" in rec.error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 端口可以显式停用 ────────────────────────────────────────────────────

def test_a_port_set_to_zero_is_skipped_not_retried(monkeypatch):
    """端口为零表示显式停用该角色：不连接、不重试，也不生成链路故障日志。"""
    cfg = _Config()
    cfg.port_emergency = 0
    pool = ConnectionPool(cfg)
    results = {}
    try:
        results = pool.connect_all()
    except Exception:  # noqa: BLE001 — 连不上真端口是正常的,只看 emergency
        pass
    assert results.get("emergency") is False
    assert "emergency" not in pool._connections
    # 停用的角色不该进重连流程
    assert pool._reconnect_role("emergency") is False


def test_the_env_var_can_disable_a_role(monkeypatch):
    """环境变量是用户唯一的入口 —— 从前这四个端口只有代码里的默认值。"""
    from mast.config import NanonisConfig

    monkeypatch.setenv("MAST_NANONIS_PORT_EMERGENCY", "0")
    assert NanonisConfig().port_emergency == 0


def test_a_garbled_env_var_keeps_the_default(monkeypatch):
    """写错的端口号**忽略**,不夹紧、不启动失败。

    夹紧或猜一个值会让它静静连上**另一个端口** —— 那比启动失败危险得多。
    """
    from mast.config import NanonisConfig

    monkeypatch.setenv("MAST_NANONIS_PORT_MAIN", "not-a-port")
    assert NanonisConfig().port_main == 6501
    monkeypatch.setenv("MAST_NANONIS_PORT_MAIN", "99999")
    assert NanonisConfig().port_main == 6501
