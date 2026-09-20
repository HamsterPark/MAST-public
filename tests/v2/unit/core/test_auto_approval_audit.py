"""自动放行仍须写入 approvals 审计表，标记 automated_policy 与 auto_executed_notified，并记录策略依据；普通动作不应混入此表。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

import pytest

from mast.core.auto_approval import (
    AUTO_APPROVAL_METHOD,
    AUTO_APPROVER_KIND,
    would_have_asked,
)
from mast.core.runtime import CoreRuntime
from mast.core.types import SafetyLevel, SkillCategory, SkillMetadata


def _meta(name: str, level: SafetyLevel, caps=frozenset()) -> SkillMetadata:
    return SkillMetadata(name=name, version="1.0.0", category=SkillCategory.WRITE,
                         safety_level=level, description="d", parameters=[],
                         capabilities=caps)


@pytest.fixture()
def live_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.logging.v2.live import open_live_v2
    repos, eid = open_live_v2()
    assert repos is not None and eid
    return repos, eid


# ── 1. 判据两侧 ───────────────────────────────────────────────────────

def test_dangerous_would_have_asked():
    reason = would_have_asked(_meta("QuitNanonis", SafetyLevel.DANGEROUS),
                              mode="auto")
    assert reason and "DANGEROUS" in reason


@pytest.mark.parametrize("level", [SafetyLevel.AUTO, SafetyLevel.CONFIRM])
def test_non_dangerous_would_not(level):
    """反向面。只钉「该说是的时候说是」的话,一个恒真的判据也会绿。"""
    assert would_have_asked(_meta("GetBias", level), mode="auto") is None


def test_a_semi_mode_pulse_would_have_asked():
    """第二支判据:SEMI 模式的电脉冲。它原来归 ``ModeGatedPulseHITLMiddleware`` 管。"""
    from mast.core.safety import CAP_BIAS_PULSE

    meta = _meta("BiasPulse", SafetyLevel.CONFIRM, caps=frozenset({CAP_BIAS_PULSE}))
    assert would_have_asked(meta, tool_name="BiasPulse",
                            args={"bias_v": 3.0}, mode="semi") is not None
    # 同一发脉冲在 AUTO 下从来不需要人 —— 模式必须真的参与判断。
    assert would_have_asked(meta, tool_name="BiasPulse",
                            args={"bias_v": 3.0}, mode="auto") is None


def test_a_broken_meta_never_raises():
    """判据在工具调用热路径上。坏掉的失败模式是「没有通知」,不是「技能跑不了」。"""
    assert would_have_asked(None) is None
    assert would_have_asked(object()) is None


# ── 2. 审计行真的落地 ─────────────────────────────────────────────────

def _approvals(repos, action_id: str) -> list[dict]:
    with repos.actions.store.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM approvals WHERE action_id = ?", (action_id,)).fetchall()]


def test_an_auto_executed_dangerous_action_gets_an_approvals_row(live_v2):
    """**这份文件的主判据。** 动作落表之后,``approvals`` 里必须真的多一行。"""
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id="t-1")

    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "QuitNanonis", "params": {"confirm": True}, "success": True,
        "duration_ms": 12, "tool_call_id": "call_1",
        "auto_approved": "DANGEROUS 技能(QuitNanonis)",
    })
    assert aid

    rows = _approvals(repos, aid)
    assert len(rows) == 1, "自动放行的 DANGEROUS 动作没有留下审计行"
    row = rows[0]
    # 如实:没有人点过按钮,所以不能写成 human_operator。
    assert row["approver_kind"] == AUTO_APPROVER_KIND == "automated_policy"
    assert row["approval_method"] == AUTO_APPROVAL_METHOD
    # evidence 里要能读出「凭什么放的行」,而不只是「放行了」。
    assert "QuitNanonis" in row["approval_evidence"]
    assert "auto_executed" in row["approval_evidence"]


def test_an_ordinary_action_gets_no_approvals_row(live_v2):
    """反向面:普通动作不进这张表。

    审计的价值在于**稀**。把每一条 GetBias 都写进 approvals,等于把当年那 11 条
    真正要查的行埋进几千行噪声里 —— 那和空表一样查不出东西。"""
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id="t-2")

    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "GetBias", "params": {}, "success": True,
    })
    assert aid
    assert _approvals(repos, aid) == []


def test_a_failed_dangerous_action_is_still_audited(live_v2):
    """失败了也要有行。

    「跑过但失败了」和「根本没跑」是两回事,而审计要回答的是前者 —— 一个失败的
    DANGEROUS 动作照样碰过硬件。"""
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id="t-3")

    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "LockNanonisUI", "params": {}, "success": False,
        "error": "module not running",
        "auto_approved": "DANGEROUS 技能(LockNanonisUI)",
    })
    assert aid and len(_approvals(repos, aid)) == 1


def test_the_audit_write_never_breaks_the_action(live_v2, monkeypatch):
    """审计写失败绝不能反噬已经发生的动作。

    这条是 best-effort 的定义:动作已经在仪器上发生了,记不下来是记录的问题,
    把它变成一个异常只会让下游以为动作没发生。"""
    repos, eid = live_v2
    rt = SimpleNamespace(_v2_repos=repos, _v2_eid=eid, _active_thread_id="t-4")

    class _Boom:
        def issue(self, **kw):
            raise RuntimeError("disk full")

    monkeypatch.setattr(type(repos), "approvals", property(lambda self: _Boom()),
                        raising=False)
    aid = CoreRuntime._record_v2_action(rt, {
        "skill": "QuitNanonis", "params": {}, "success": True,
        "auto_approved": "DANGEROUS 技能(QuitNanonis)",
    })
    assert aid, "审计写失败把动作记录一起带走了"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
