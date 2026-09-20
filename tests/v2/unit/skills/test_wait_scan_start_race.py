"""扫描启动与运行之间允许宽限期；区分尚未启动、确实早停与正常完成。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.skills.builtins.scan_utils import WaitScanComplete  # noqa: E402


class _Ctx:
    """扫描状态与行数由替身独立控制。"""

    def __init__(self, status_script, lines_at=None, total=256):
        self.status_script = list(status_script)
        self.lines_at = lines_at or (lambda i: 0)
        self.total = total
        self.polls = 0
        self.stops = 0

    def safe_call(self, method, *args, **kw):
        if method == "Scan_StatusGet":
            i = self.polls
            self.polls += 1
            st = (self.status_script[i] if i < len(self.status_script)
                  else self.status_script[-1])
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [st]))
        if method == "Scan_Action":
            self.stops += 1
        return NanonisCallRecord(method=method, args=args)


def _run(ctx, *, timeout_ms=60000, monkeypatch=None):
    """跑完整个 WaitScanComplete,返回它的 data。"""
    sk = WaitScanComplete()
    # 行数测量:替身直接给,不去碰真的缓冲区解析(那部分另有测试)。
    def _measure(_real_ctx):
        done = int(ctx.lines_at(ctx.polls))
        return {"lines_done": done, "lines_total": ctx.total,
                "lines_verified": True,
                "stopped_early": bool(ctx.total > 0 and done < ctx.total)}
    sk._measure_lines = _measure  # type: ignore[assignment]
    res = sk.execute(ctx, {"timeout_ms": timeout_ms})
    return res.data or {}


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    """把轮询间隔和宽限期压到毫秒级 —— 这些测试量的是**逻辑**,不是墙钟。

    宽限期照比例缩(0.25 s),而不是设成 0:设成 0 会让第 3 条测试恒真,
    那就等于把被测的那个窗口测没了。
    """
    monkeypatch.setattr(WaitScanComplete, "_poll_interval_s", 0.01,
                        raising=False)
    monkeypatch.setattr(WaitScanComplete, "_START_GRACE_S", 0.25, raising=False)


# ── 1. 竞态本身 ────────────────────────────────────────────────────────────

def test_a_scan_that_takes_a_moment_to_arm_is_not_called_stopped():
    """头几次轮询未启动、之后进入运行的扫描应正常等待完成。"""
    # 前 3 次 0(还没起来)→ 跑 5 次 → 结束;结束时缓冲区是满的
    ctx = _Ctx([0, 0, 0, 1, 1, 1, 1, 1, 0], lines_at=lambda i: 256 if i > 3 else 0)
    d = _run(ctx)
    assert d["outcome"] == "completed", (
        f"起步慢的扫描被误判为 {d['outcome']}："
        "StartScan 成功,而第一次轮询在 0.31 秒读到 0")
    assert d["lines_done"] == 256
    assert not d.get("never_started")


def test_the_grace_only_applies_before_it_is_ever_seen_running():
    """跑起来之后再读到 0 ⇒ 立刻按「停了」处理,不再有任何宽限。

    没有这一条,宽限期会变成「每次扫描结束都多等 0.25 秒」,
    而那是把一个开扫的握手窗口误用成了收尾的缓冲。
    """
    ctx = _Ctx([1, 1, 0], lines_at=lambda i: 256)
    d = _run(ctx)
    assert d["outcome"] == "completed"
    assert ctx.polls == 3, f"多轮询了 {ctx.polls} 次 —— 宽限期漏到了收尾路径上"


# ── 2. 没把守卫关掉 ────────────────────────────────────────────────────────

def test_a_genuinely_truncated_scan_is_still_stopped_early():
    """真的跑了一半被停下,仍然要判 `stopped_early`。

    这条在防「修复顺手把判据关掉」—— v6.1.2 之前正是把每一种早停都当成完成,
    调用方拿着一帧从没采到的图去存盘 / 重配 / 重启。
    """
    ctx = _Ctx([1, 1, 1, 0], lines_at=lambda i: 61)
    d = _run(ctx)
    assert d["outcome"] == "stopped_early"
    assert d["stopped_early"] is True
    assert d.get("never_started") is False
    assert d["lines_done"] == 61


# ── 3. 「从没开始」是它自己的一句话 ────────────────────────────────────────

def test_never_starting_is_reported_as_its_own_outcome():
    """整个宽限期都没跑起来、缓冲区也空 ⇒ `never_started`,**不是** `stopped_early`。

    两者指向的下一步不同:前者查我们自己的发起时序,后者才去查用户 Stop /
    Nanonis 自停。混成一句话,读的人会去查一件没发生的事 —— 我就查了半天。
    """
    ctx = _Ctx([0], lines_at=lambda i: 0)
    d = _run(ctx)
    assert d["outcome"] == "never_started", d
    assert d["never_started"] is True
    assert d["stopped_early"] is False, (
        "同时报 stopped_early 会让上游继续说「去查是谁停的」")


def test_a_scan_that_finished_inside_the_grace_is_completed_not_never_started():
    """宽限期内它其实跑完了(快到没被任何一次轮询逮到) ⇒ 按行数判,不能说没开始。

    判据用的是**缓冲区行数**而不只是「见没见过 status≠0」,就是为了这一种。
    """
    ctx = _Ctx([0], lines_at=lambda i: 256)
    d = _run(ctx)
    assert d["outcome"] == "completed", d
    assert not d.get("never_started")


# ── 4. 正常路径逐字不变 ────────────────────────────────────────────────────

def test_an_already_running_scan_behaves_exactly_as_before():
    """一开始就在跑的扫描:轮询次数、结论都与修复前一致。"""
    ctx = _Ctx([1, 1, 1, 1, 0], lines_at=lambda i: 256)
    d = _run(ctx)
    assert d["outcome"] == "completed"
    assert d["timed_out"] is False
    assert ctx.polls == 5
    assert ctx.stops == 0, "正常结束不该发 Scan_Action"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
