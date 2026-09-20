"""Graceful shutdown must finish; a bounded daemon backstop records surviving threads and stacks before forcing exit.

Tests use isolated fake threads and a patched exit function to verify timing and diagnostics."""
from __future__ import annotations

import sys
import threading
import time
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

import pytest  # noqa: E402

from mast.api.routes import admin  # noqa: E402


def test_the_backstop_is_armed_and_is_itself_a_daemon():
    """兜底计时器自己绝不能成为「拖住进程」的那个线程。"""
    before = {t.name for t in threading.enumerate()}
    admin._arm_exit_backstop()
    try:
        t = next(t for t in threading.enumerate()
                 if t.name == "shutdown-exit-backstop" and t.name not in before)
        assert t.daemon is True, "兜底线程不是 daemon —— 它会变成新的僵壳原因"
    finally:
        for t in threading.enumerate():
            if t.name == "shutdown-exit-backstop":
                t.cancel()  # type: ignore[attr-defined]


def test_the_backstop_waits_rather_than_killing_immediately():
    """慢但正常的收尾不能被打断:uvicorn 要排空连接、池要关四条 socket、
    视觉可能正在推理中。兜底抓的是**卡住**,不是**慢**。"""
    assert admin._EXIT_BACKSTOP_S >= 10.0


def test_the_backstop_reports_surviving_threads_before_forcing(monkeypatch, caplog):
    """The backstop must log surviving threads before calling the patched process-exit function."""
    killed: list[int] = []
    monkeypatch.setattr(admin, "_EXIT_BACKSTOP_S", 0.02)
    monkeypatch.setattr(admin.os if hasattr(admin, "os") else __import__("os"),
                        "_exit", lambda code: killed.append(code), raising=False)

    import os as _os
    monkeypatch.setattr(_os, "_exit", lambda code: killed.append(code))

    stop = threading.Event()
    hog = threading.Thread(target=stop.wait, name="pretend-stuck-thread",
                           daemon=True)
    hog.start()
    try:
        with caplog.at_level("ERROR"):
            admin._arm_exit_backstop()
            deadline = time.monotonic() + 3.0
            while not killed and time.monotonic() < deadline:
                time.sleep(0.02)
        assert killed == [0], "兜底没有强制退出"
        text = caplog.text
        assert "仍未退出" in text
        assert "pretend-stuck-thread" in text, (
            "没有列出仍存活的线程 —— 那样下次真机再卡住还是查不出是谁")
    finally:
        stop.set()


def test_shutdown_still_closes_the_pool_before_any_forced_exit():
    """顺序不能反:强制退出跳过 atexit 与缓冲刷写,只有在**池已经优雅关闭之后**
    才可以接受 —— 否则 TCP 被硬断,Nanonis 端口会一直坏到重启 Nanonis。

    这里钉的是源码顺序:`app.shutdown()` 在 `_arm_exit_backstop()` 之前。
    """
    src = Path(admin.__file__).read_text(encoding="utf-8")
    body = src[src.index("def admin_shutdown("):src.index("_EXIT_BACKSTOP_S = ")]
    assert body.index("app_handle.shutdown()") < body.index("_arm_exit_backstop()"), (
        "兜底退出排到了池关闭前面 —— 那会硬断 TCP,把 Nanonis 端口弄坏")


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
