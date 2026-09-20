"""科研纲领的 ``done_when``：整体拒绝，读即求值，替换时说一声。

## 为什么是「整体拒绝」而不是「丢掉不合法的那几条」

丢掉 ``all`` 里一个合取项，等于把这条纲领的目标**悄悄改小** —— 于是它更早
「达成」，于是它下面还在等资料的 park 被错误地抑制唤醒。那是 fail-open 方向。
拒绝无害：没有 ``done_when`` 就是今天的行为。

同一条纪律在这个模块已经写着（``hypothesis_kind`` 不在闭集里直接拒、不替你改成
别的），这里只是把它用到判据上。
"""
from __future__ import annotations

import json
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

from mast.agents._shared.campaign_tools import make_campaign_tools  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolcall import tool_call  # noqa: E402


@pytest.fixture()
def v2_store(tmp_path, monkeypatch):
    """把 v2 记录库指到 tmp。

    **必须**设：不设的话这些测试会读写用户真实的
    ``experiments/mast_experiments_v2.db``，而本仓「测试污染真实数据」已犯过五次。
    """
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.agents._shared.data_paths import v2_experiment_db_path

    assert str(tmp_path) in str(v2_experiment_db_path()), "重定向没生效，不许继续"
    return tmp_path


@pytest.fixture()
def tools(v2_store):
    return {str(getattr(t, "name", "")): t
            for t in make_campaign_tools("research_director")}


def _call(tool, **kw) -> dict:
    return json.loads(tool.invoke(kw))


def _campaign_ref_from(ret):
    """从 ``ArtifactToolReturn`` 里把 CampaignRef 取出来。

    工具返回的是一个 ``Command``（update 里带产物）还是一个带 update 的对象，
    取决于 langchain 版本 —— 两种都认，认不出来就返回 None 让断言报出来，
    而不是在这里静默地把一次真失败读成「没有 ref」。
    """
    upd = getattr(ret, "update", None)
    if isinstance(upd, dict):
        return upd.get("research_campaign")
    if isinstance(upd, list):
        for item in upd:
            if isinstance(item, dict) and "research_campaign" in item:
                return item["research_campaign"]
    return None


def _create(tools, goal: dict | None = None, **kw) -> dict:
    body = {"title": "WO₂I₂ 条纹相的起源", "hypothesis": "条纹周期与温度无关",
            "hypothesis_kind": "confirmatory"}
    if goal is not None:
        body["goal_json"] = json.dumps(goal, ensure_ascii=False)
    body.update(kw)
    return _call(tools["campaign_create"], **body)


def _count(tools) -> int:
    return len(_call(tools["campaign_list"]).get("campaigns") or [])


# ── 合法 ────────────────────────────────────────────────────────────────

def test_a_valid_done_when_is_stored_normalised(tools):
    r = _create(tools, {"question": "是不是 CDW",
                        "done_when": [{"kind": "conduct_completed",
                                       "spec_id": "synthetic_sample_v1"}]})
    assert r["ok"] is True, r
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    dw = got["campaign"]["goal"]["done_when"]
    # 存进去的是**归一化后**的形状：读侧不必再猜写法（列表 = all）。
    assert dw == {"all": [{"kind": "conduct_completed", "spec_id": "synthetic_sample_v1",
                           "min_count": 1}]}


def test_success_criteria_is_kept_verbatim_and_never_evaluated(tools):
    """人读的那半原样留着 —— 但它不是判据，也不假装是。"""
    r = _create(tools, {"question": "q", "success_criteria": "看到就知道了",
                        "done_when": [{"kind": "claims_supported",
                                       "min_count": 2}]})
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    assert got["campaign"]["goal"]["success_criteria"] == "看到就知道了"
    # 判据只算 done_when 那一条
    assert got["campaign"]["goal_progress"]["total"] == 1


# ── 拒绝 ────────────────────────────────────────────────────────────────

def test_an_illegal_predicate_rejects_the_whole_thing(tools):
    before = _count(tools)
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "conduct_completed", "spec_id": "synthetic_sample_v1"},
        {"kind": "vibes"},
    ]})
    assert r["ok"] is False
    assert "vibes" in r["reason"]
    assert _count(tools) == before, (
        "非法判据竟然还是把纲领建出来了 —— 那份纲领的目标是被削弱过的")


def test_the_rejection_hands_back_the_closed_set(tools):
    """在**出错的地方**给闭集，而不是一句「请从目录里选」。

    「移除诱因，别说服模型」—— 本仓记过四次，没有一次靠劝说管用。
    """
    r = _create(tools, {"done_when": [{"kind": "nope"}]})
    kinds = {e["kind"] for e in r["done_when_catalog"]}
    assert {"artifact_present", "conduct_completed", "best_frame_settled"} <= kinds
    assert r["example"]
    assert len(r["problems"]) == 1


def test_a_missing_required_arg_is_named_with_its_position(tools):
    r = _create(tools, {"done_when": [
        {"kind": "artifact_present", "field": "analysis"},
        {"kind": "best_frame_settled"},           # 缺 tag
    ]})
    assert r["ok"] is False
    assert "[done_when[1]]" in r["problems"][0] and "tag" in r["problems"][0]


def test_update_with_an_illegal_done_when_changes_nothing(tools):
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 1}]})
    cid = r["campaign_id"]
    bad = _call(tools["campaign_update"], campaign_id=cid,
                goal_json=json.dumps({"done_when": [{"kind": "nope"}]}))
    assert bad["ok"] is False
    still = _call(tools["campaign_get"], campaign_id=cid)
    assert still["campaign"]["goal"]["done_when"] == {
        "all": [{"kind": "claims_supported", "min_count": 1}]}


# ── 整体替换的那个坑 ────────────────────────────────────────────────

def test_replacing_the_goal_without_done_when_says_so_out_loud(tools):
    """``goal`` 是**整体替换**。改 question 时顺手把判据弄丢，是一次静默的
    功能撤除 —— 库里从此没有机器可判的终止条件，而没有任何地方会说一声。"""
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 1}]})
    out = _call(tools["campaign_update"], campaign_id=r["campaign_id"],
                goal_json=json.dumps({"question": "换了个问法"}))
    assert out["ok"] is True, out
    assert "done_when" in (out.get("note") or ""), out


def test_no_warning_when_there_was_nothing_to_lose(tools):
    r = _create(tools, {"question": "q"})
    out = _call(tools["campaign_update"], campaign_id=r["campaign_id"],
                goal_json=json.dumps({"question": "换了个问法"}))
    assert "note" not in out or "done_when" not in out["note"]


def test_request_plan_keeps_the_done_when(tools):
    """``campaign_request_plan`` 是读-改-写。它不该顺手吃掉判据。"""
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 2}]})
    cid = r["campaign_id"]
    # ``campaign_request_plan`` 带 InjectedToolCallId —— 必须用完整 ToolCall 调。
    tools["campaign_request_plan"].invoke(tool_call(
        tools["campaign_request_plan"],
        {"campaign_id": cid, "plan_request": "设计一个能区分 CDW 与重构的实验"}))
    got = _call(tools["campaign_get"], campaign_id=cid)
    assert got["campaign"]["goal"]["done_when"] == {
        "all": [{"kind": "claims_supported", "min_count": 2}]}
    assert got["campaign"]["goal"]["plan_request"]


# ── 读即求值 ────────────────────────────────────────────────────────

def test_get_reports_progress_as_a_fact_not_a_question(tools):
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 2}]})
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    p = got["campaign"]["goal_progress"]
    assert p["verdict"] == "not_done" and p["satisfied"] == 0 and p["total"] == 1
    assert "论断" in p["reason"]


def test_a_campaign_without_done_when_is_unknown_not_done(tools):
    """「没人写过什么算答完」不是「答完了」。"""
    r = _create(tools, {"question": "q"})
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    assert got["campaign"]["goal_progress"]["verdict"] == "unknown"


def test_the_status_is_never_flipped_automatically(tools):
    """判据全真 ≠ 科学问题答完。

    自动翻状态的失败模式是 RD 见到 completed 就另起一份纲领（提示词里明令避免的
    重复纲领）；不自动翻的代价只是多一次记账，而那次记账有 created_by 可查。
    """
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 0 + 1}]})
    cid = r["campaign_id"]
    # 就算判据这一刻满足，状态也只能由人/RD 显式改。
    got = _call(tools["campaign_get"], campaign_id=cid)
    assert got["campaign"]["status"] == "draft"


# ── 交接：判据摘要随产物走 ──────────────────────────────────────────

def test_the_commission_carries_a_readable_criteria_summary(tools):
    from mast.agents._shared.artifact_channel import render_field

    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "conduct_completed", "spec_id": "synthetic_sample_v1"},
        {"kind": "claims_supported", "min_count": 2}]})
    cid = r["campaign_id"]
    ret = tools["campaign_request_plan"].invoke(tool_call(
        tools["campaign_request_plan"],
        {"campaign_id": cid, "plan_request": "设计一个实验"}))
    ref = _campaign_ref_from(ret)
    assert ref is not None and ref.done_when_brief
    line = render_field("research_campaign", ref, available_tools=set())
    assert "什么算答完（机器判）" in line
    # **带定义不带结论** —— 一次求值的结果只会过期。
    assert "满足" not in line


# ── 收尾留痕：同值合并，翻转落新行 ──────────────────────────────────

def test_publish_goal_check_dedups_identical_verdicts(tools, v2_store):
    """一条纲领跑十次 run 而判据没变 ⇒ 只留一行。

    没有去重，这张表会被「还是没满足」灌满 —— 而那种表没人会去读，于是留痕
    等于没留。
    """
    from mast.agents._shared.campaign_tools import _repos, publish_goal_check

    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 2}]})
    cid = r["campaign_id"]

    first = publish_goal_check(cid)
    assert first["ok"] is True and first["verdict"] == "not_done"
    publish_goal_check(cid)
    publish_goal_check(cid)

    repos = _repos()
    rows = repos.events.by_topic("campaign.goal_check", limit=200)
    assert len(rows) == 1, f"同一结论落了 {len(rows)} 行"


def test_an_empty_campaign_id_publishes_nothing(v2_store):
    """一次没有纲领的 run 就是没有纲领 —— **不猜**。

    替它挑一条「最近的」会把 A 的进度记到 B 头上，而那种错不会报错。
    """
    from mast.agents._shared.campaign_tools import publish_goal_check

    assert publish_goal_check("")["ok"] is False
    assert publish_goal_check("  ")["ok"] is False


def test_an_empty_done_when_also_counts_as_losing_it(tools):
    """``done_when: []`` 是**把判据清空**，不是「没提到它」。

    第一版的告警看的是「键在不在」；而空值经归一化会写回一个 ``None``——键在、
    值没了，于是这次静默清空一句话都不说。两种写法都要报。
    """
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 1}]})
    out = _call(tools["campaign_update"], campaign_id=r["campaign_id"],
                goal_json=json.dumps({"question": "q", "done_when": []}))
    assert out["ok"] is True, out
    assert "done_when" in (out.get("note") or ""), out
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    assert not got["campaign"]["goal"].get("done_when")


def test_a_written_done_when_gets_a_baseline(tools):
    """写下判据的那一刻要抓基线 —— 否则 ``artifact_present`` 在 campaign 侧
    永远没有参照点，任何一份历史产物都会让它假达成。"""
    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "artifact_present", "field": "analysis"}]})
    got = _call(tools["campaign_get"], campaign_id=r["campaign_id"])
    goal = got["campaign"]["goal"]
    # 抓得到就该在；抓不到（测试环境没有 artifacts 索引）则**不写**，
    # 下游会判 unknown 而不是假达成 —— 两种都可接受，唯独不许是一个空 dict。
    assert goal.get("baseline") is None or isinstance(goal["baseline"], dict)
    assert goal.get("baseline") != {}


def test_goal_history_has_a_reader(tools, v2_store):
    """`campaign.goal_check` 事件不是写了没人读 —— `campaign_get` 带回它。"""
    from mast.agents._shared.campaign_tools import publish_goal_check

    r = _create(tools, {"question": "q", "done_when": [
        {"kind": "claims_supported", "min_count": 2}]})
    cid = r["campaign_id"]
    publish_goal_check(cid)
    hist = _call(tools["campaign_get"], campaign_id=cid)["campaign"]["goal_history"]
    assert hist and hist[-1]["verdict"] == "not_done"
