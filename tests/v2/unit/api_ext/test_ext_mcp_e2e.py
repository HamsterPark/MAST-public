"""端到端：Claude Code 插件的 MCP server（真子进程）→ 真 HTTP → 真外部面 → 真记录库。

MCP 客户端自己的测试对着一个手写的假 HTTP 服务器；外部面的测试用 TestClient。两边各自
都绿，契约照样可能对不上（字段名、状态码、请求体形状）—— 替身测不到替身与真货之间的缝。
这里把两个真东西接起来：uvicorn 在回环地址上起 ``create_ext_app``（挂在 ``/api/ext/v1``，
与生产同一前缀），MCP server 用 ``python run_server.py`` 起成子进程，按 Claude Code 的方式
逐条发 JSON-RPC。断言以**服务端的真实副作用**为准（记录库、记忆库、心愿单、文档库），
工具返回的文字只做辅助。
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from _ext_world import RAN, write_sxm

_REPO = Path(__file__).resolve().parents[4]
_RUN_SERVER = _REPO / "integrations" / "claude-code" / "server" / "run_server.py"
_JOB_ID = re.compile(r"j_[0-9a-f]{12}")


# ─────────────────────────────────────────────────────────────────────
# 真 HTTP：uvicorn 在回环地址的临时端口上起外部面
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def served(world):
    import uvicorn
    from fastapi import FastAPI

    parent = FastAPI()
    parent.mount("/api/ext/v1", world.app)
    server = uvicorn.Server(uvicorn.Config(parent, host="127.0.0.1", port=0,
                                           log_level="warning", lifespan="off"))
    t = threading.Thread(target=server.run, name="e2e-uvicorn", daemon=True)
    t.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn 没起来"
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(15)


# ─────────────────────────────────────────────────────────────────────
# 真 MCP：子进程，逐条请求 / 应答（tools/call 在 server 里各跑一条线程，一次灌入的
# 请求回包顺序不定，所以不用 subprocess.run(input=...)）
# ─────────────────────────────────────────────────────────────────────

class Mcp:
    def __init__(self, url: str, tmp: Path):
        env = {k: v for k, v in os.environ.items() if not k.startswith("MAST_")}
        env.update({"MAST_URL": url, "MAST_ACTOR": "e2e", "MAST_SESSION": "s-e2e",
                    "MAST_FETCH_DIR": str(tmp / "fetch"), "MAST_MCP_LOG": "WARNING",
                    "PYTHONIOENCODING": "cp1252"})
        self.p = subprocess.Popen([sys.executable, str(_RUN_SERVER)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                  cwd=str(tmp))
        self.q: queue.Queue = queue.Queue()
        self.err: list[bytes] = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=lambda: self.err.extend(iter(self.p.stderr.readline, b"")),
                         daemon=True).start()
        self._id = 0

    def _pump(self):
        for line in iter(self.p.stdout.readline, b""):
            if line.strip():
                self.q.put(json.loads(line.decode("utf-8")))

    def _send(self, msg: dict) -> None:
        self.p.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
        self.p.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 90.0) -> dict:
        self._id += 1
        mid = self._id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AssertionError(f"{method} 没有回包；stderr 尾部：\n"
                                     + b"".join(self.err[-30:]).decode("utf-8", "replace"))
            try:
                msg = self.q.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if msg.get("id") == mid:
                return msg

    def call(self, tool: str, args: dict | None = None, timeout: float = 90.0) -> tuple[str, bool]:
        r = self.request("tools/call", {"name": tool, "arguments": args or {}}, timeout)
        assert "result" in r, f"{tool} 是协议错误而不是工具结果：{r}"
        res = r["result"]
        return "".join(c.get("text", "") for c in res.get("content", [])), bool(res.get("isError"))

    def close(self) -> None:
        try:
            self.p.stdin.close()
            self.p.wait(30)
        finally:
            if self.p.poll() is None:  # pragma: no cover — 不该发生
                self.p.kill()


@pytest.fixture()
def mcp(served, world):
    m = Mcp(served, world.tmp)
    init = m.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "e2e", "version": "0"}})
    assert init["result"]["serverInfo"]["name"] == "mast"
    m._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield m
    m.close()


# ─────────────────────────────────────────────────────────────────────

def test_a_whole_session_through_the_real_gateway(mcp, world):
    text, err = mcp.call("mast_status")
    assert not err, text

    text, err = mcp.call("mast_briefing")
    assert not err and "外部网关测试" in text, text[:600]

    text, err = mcp.call("mast_find_skills", {"query": "ExtRead"})
    assert not err and "ExtRead" in text, text[:600]
    text, err = mcp.call("mast_skill_card", {"name": "ExtRead"})
    assert not err and "auto" in text, text[:600]

    # 一个作业：SI 字符串在服务端还原，动作带着调用方的名字进实验记录
    text, err = mcp.call("mast_run", {"skill": "ExtRead", "params": {"x_m": "5n"}, "wait_s": 15})
    assert not err and "succeeded" in text, text[:800]
    assert ("ExtRead", {"x_m": pytest.approx(5e-9)}) in [(n, p) for n, p in RAN]
    rows = world.st.recent_actions(world.eid, 5)
    assert rows and rows[0]["skill_name"] == "ExtRead"
    assert rows[0]["context"] == "ext:e2e/s-e2e", rows[0]

    # 笔记进记忆库（署名 ext:e2e），再搜得回来
    text, err = mcp.call("mast_note_write", {"title": "e2e 笔记", "content": "针尖条纹 3 pm"})
    assert not err, text
    notes = world.cog.store.list(f"experiment:{world.eid}", limit=20)
    assert any(n["author"] == "ext:e2e" and "条纹" in n["content"] for n in notes), notes
    text, err = mcp.call("mast_note_search", {"query": "条纹"})
    assert not err and "e2e 笔记" in text, text[:600]

    # 问操作员 → 操作员在界面上答 → 读得到答复（路径是一等字段）
    from mast.wishlist import list_agent_requests, resolve_agent_request

    text, err = mcp.call("mast_ask_operator", {"message": "那张图在哪？"})
    assert not err, text
    mine = [r for r in list_agent_requests() if r.get("agent_id") == "ext:e2e"]
    assert len(mine) == 1, mine
    resolve_agent_request(mine[0]["id"], "done", note="这里", path="D:/data/e2e")
    text, err = mcp.call("mast_operator_reply", {"request_id": mine[0]["id"]})
    assert not err and "D:/data/e2e" in text, text[:600]

    # 原始数据：列出 → 取帧（服务端统一朝向）→ 落在下载目录
    raw_fwd = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw_bwd = 100 + np.arange(12, dtype=np.float32).reshape(3, 4)
    sxm = write_sxm(world.tmp / "proj" / "working-sessions" / "e2e_up.sxm", raw_fwd, raw_bwd,
                    scan_dir="up")
    text, err = mcp.call("mast_list_data", {"n": 20})
    assert not err and "e2e_up.sxm" in text, text[:800]
    text, err = mcp.call("mast_fetch", {"path": str(sxm), "mode": "frame", "channel": "Z"})
    assert not err, text
    saved = sorted((world.tmp / "fetch").rglob("*.npz"))
    assert len(saved) == 1, saved
    z = np.load(saved[0], allow_pickle=False)
    np.testing.assert_array_equal(z["forward"], raw_fwd[::-1])
    np.testing.assert_array_equal(z["backward"], raw_bwd[:, ::-1][::-1])

    # 交接报告存进文档库，署名 ext:e2e
    text, err = mcp.call("mast_handover", {"summary": "e2e 交接", "next_steps": ["换区域"]})
    assert not err, text
    m = re.search(r'"doc_id"\s*:\s*"([^"]+)"', text)
    assert m, text[:800]
    from mast.documents.store import store

    entry = store().get(m.group(1))
    assert entry is not None and entry.versions[-1].created_by == "ext:e2e"
    body = entry.read_text()
    assert "e2e 交接" in body and "换区域" in body and "ExtRead" in body


def test_cancel_busy_and_emergency_stop_through_the_real_gateway(mcp, world):
    from mast.core.instrument_lock import instrument_lock

    # 取消：作业停在下一次检查处，原因写着是谁取消的
    text, err = mcp.call("mast_run", {"skill": "ExtSlow", "wait_s": 0})
    jid = _JOB_ID.search(text).group(0)
    time.sleep(0.3)
    text, err = mcp.call("mast_cancel", {"job_id": jid, "reason": "e2e 换方案"})
    assert not err, text
    text, err = mcp.call("mast_job", {"job_id": jid, "wait_s": 15})
    assert "cancelled" in text, text[:800]
    assert "e2e 换方案" in world.jm.get(jid).result["error"]

    # 撞锁：不排队，以 refused_busy 结束并说出谁在开车
    held, release = threading.Event(), threading.Event()

    def _holder():
        with instrument_lock().hold(owner="群聊任务", skill="LongScan"):
            held.set()
            release.wait(30)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        text, err = mcp.call("mast_run", {"skill": "ExtWrite", "wait_s": 20})
        assert err and "refused_busy" in text and "群聊任务" in text, text[:800]
        assert ("ExtWrite", {}) not in RAN
    finally:
        release.set()
        t.join(10)

    # 急停：闩上的原因写成这个外部 agent；在跑的外部作业被取消
    text, err = mcp.call("mast_run", {"skill": "ExtSlow", "wait_s": 0})
    jid2 = _JOB_ID.search(text).group(0)
    text, err = mcp.call("mast_emergency_stop", {"reason": "e2e 电流失控"})
    assert world.rt.estops and "外部 agent ext:e2e 触发急停" in world.rt.estops[-1], world.rt.estops
    assert "e2e 电流失控" in world.rt.estops[-1]
    text, err = mcp.call("mast_job", {"job_id": jid2, "wait_s": 15})
    assert world.jm.get(jid2).terminal, text[:600]
