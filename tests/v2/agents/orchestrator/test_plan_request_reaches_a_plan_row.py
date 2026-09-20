"""委托到方案的持久化链路。

验证 plan_request 写入委托记录、设计上下文能够读取、create_plan 写入正确的 PlanStore，
以及存储不可用时明确失败。不同版本中的同名表不能互相充当写入成功的证据。"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.agents._shared.meta_tools import (  # noqa: E402
    DESIGN_TOOL_NAMES,
    make_meta_tools,
)
from mast.planning.plan_store import PlanStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolcall import tool_call  # noqa: E402


@pytest.fixture()
def v2_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.agents._shared.data_paths import v2_experiment_db_path

    assert str(tmp_path) in str(v2_experiment_db_path()), "重定向没生效，不许继续"
    return tmp_path


def _design_tools(plan_store):
    """设计工具面：``make_meta_tools`` 按 ``DESIGN_TOOL_NAMES`` 过滤。

    分母从**注册表**取，不在这里手抄一份名单 —— 手抄的那份迟早与生产的岔开，
    而岔开时这条测试会绿着。
    """
    return [t for t in make_meta_tools(lambda: {"plan_store": plan_store})
            if str(getattr(t, "name", "")) in DESIGN_TOOL_NAMES]


def test_create_plan_is_actually_on_the_designers_tool_face():
    """先立前提：XD 手上真有 ``create_plan``。

    没有这一条，下面那些「plan 行出现了」可能只是因为我直接调了工具函数，
    而 agent 在生产里根本够不到它 —— 「树里有正确实现 ≠ agent 用得上」。
    """
    assert "create_plan" in DESIGN_TOOL_NAMES
    names = {str(getattr(t, "name", "")) for t in _design_tools(None)}
    assert "create_plan" in names


def test_a_commission_becomes_a_plan_row(tmp_path, v2_store):
    """RD 立纲领 → 委托 XD → XD 起草 → PlanStore 多一行。"""
    import json

    from mast.agents._shared.campaign_tools import make_campaign_tools

    rd = {str(getattr(t, "name", "")): t
          for t in make_campaign_tools("research_director")}
    created = json.loads(rd["campaign_create"].invoke({
        "title": "WO₂I₂ 条纹相", "hypothesis": "条纹周期与温度无关",
        "hypothesis_kind": "confirmatory"}))
    cid = created["campaign_id"]

    COMMISSION = "设计一个能区分 CDW 与表面重构的变温实验"
    ret = rd["campaign_request_plan"].invoke(tool_call(
        rd["campaign_request_plan"],
        {"campaign_id": cid, "plan_request": COMMISSION}))

    # (1) 委托落库了
    got = json.loads(rd["campaign_get"].invoke({"campaign_id": cid}))
    assert got["campaign"]["goal"]["plan_request"] == COMMISSION

    # (2) 委托在 XD 的**上下文里**看得见 —— 不是靠交接语复述
    from mast.agents._shared.artifact_channel import render_upstream_block

    ref = _campaign_ref_from(ret)
    assert ref is not None, "campaign_request_plan 没有把纲领登记为上游产物"
    block = render_upstream_block({"research_campaign": ref},
                                  "experiment_design", available_tools=set())
    assert COMMISSION in block, (
        "XD 在自己的上下文里看不到委托 —— 它只能从交接语里猜，"
        "而那句话会被压缩摘掉")

    # (3) XD 起草 ⇒ PlanStore 恰好一行
    store = PlanStore(db_path=str(tmp_path / "plans.db"))
    tools = {str(getattr(t, "name", "")): t for t in _design_tools(store)}
    before = len(store.list_plans())
    # ``create_plan`` 同样带 InjectedToolCallId —— 必须用完整 ToolCall 调。
    out = tools["create_plan"].invoke(tool_call(tools["create_plan"], {
        "name": "变温 CDW 判别", "goal": COMMISSION,
        "phases": [{"name": "低温成像", "steps": ["扫 20 nm"]}]}))
    assert "计划库不可用" not in str(out), out
    after = store.list_plans()
    assert len(after) == before + 1, f"没多出一行：{out}"
    # ``list_plans`` 回的是 dict，不是 ExperimentPlan —— 照它实际回的形状断言。
    row = after[-1]
    assert COMMISSION in str(row.get("goal") if isinstance(row, dict) else row), row


def test_without_a_plan_store_it_says_so_instead_of_succeeding_quietly(v2_store):
    """负例 —— 没有它，上面那条「多了一行」可能只是因为随便调什么都会成功。"""
    tools = {str(getattr(t, "name", "")): t for t in _design_tools(None)}
    out = str(tools["create_plan"].invoke(tool_call(tools["create_plan"], {
        "name": "x", "goal": "y", "phases": [{"name": "p", "steps": ["s"]}]})))
    assert "不可用" in out or "unavailable" in out.lower(), out


def test_the_two_plans_tables_are_not_the_same_truth(v2_store):
    """把当初那次误读钉下来：v1 的 PlanStore 与 v2 的 ``plans`` 表是两回事。

    v2 那张运行期**零写者**，所以「它是 0 行」永远成立，不能作为「XD 没起草」
    的证据。下次再有人拿它下结论时，这条测试是那句话的出处。
    """
    from mast.logging.v2 import schema

    assert "CREATE TABLE IF NOT EXISTS plans" in schema.DDL_PLANS
    root = Path(_MASTV2_ROOT, "mast")
    writers = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
               if "plans.create(" in p.read_text(encoding="utf-8", errors="ignore")]
    assert writers in ([], ["logging/v2/migrate.py"]), (
        f"v2 的 plans 表有了运行期写者：{writers} —— "
        "那这条注记要重写，两张表的关系也要重新裁决")


def _campaign_ref_from(ret):
    upd = getattr(ret, "update", None)
    if isinstance(upd, dict):
        return upd.get("research_campaign")
    if isinstance(upd, list):
        for item in upd:
            if isinstance(item, dict) and "research_campaign" in item:
                return item["research_campaign"]
    return None
