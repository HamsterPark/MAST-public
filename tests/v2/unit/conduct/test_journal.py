"""实验文件夹里的 conduct 视图 —— ``spec_vNNN.md`` + ``progress.jsonl``。

设计 §4.8。这里钉四件事:

1. **每行自洽**:一行单独 ``json.loads`` 就能读懂(ts / conduct_id / kind /
   一句人话),不依赖上一行;
2. **内容变才发版**:同样的 spec 反复 sync 不生第二个文件;
3. **拿不到实验文件夹 ⇒ 如实报,不抛**:一份人读副本写不下去,绝不该让一个
   正在动仪器的流程停下来;
4. **「读不到」不折叠成 0**:目录在而文件还没有 = 0 行(答得上来);
   读不出行数 = ``None``(答不上来)。

全程 ``tmp_path`` + 注入 resolver:**绝不碰真实 experiments**
(测试污染真实数据本仓已五次,每次的入口都是一个善意的默认路径)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct.journal import (  # noqa: E402
    ConductJournal,
    PROGRESS_FILENAME,
    default_folder_resolver,
    render_spec_markdown,
)
from mast.conduct.spec import ConditionSpec, ParamSpec  # noqa: E402

from tests.v2.unit.conduct._harness import (  # noqa: E402
    rule_gate, spec as make_spec, stage, step, retract_step, wait_step,
)


@pytest.fixture()
def rig(tmp_path):
    """一个只往 tmp_path 里写的 journal。"""
    root = tmp_path / "exp-42"
    return ConductJournal("camp01", "exp-42", folder_resolver=lambda _e: root)


def _spec_with_wait():
    return make_spec([
        stage("A", [step("A.01", "ScanAt"),
                    retract_step("A.02"),
                    wait_step("A.03", kind="both",
                              condition=ConditionSpec(signal="temperature_k",
                                                      op="<=", value=5.0,
                                                      stale_after_s=600.0,
                                                      hold_s=1800.0,
                                                      desc="降到 5 K 并稳住"))],
              entry_gate=rule_gate("g_in")),
    ], params=(ParamSpec(name="bias_v", type="float", unit="V", default=0.5,
                         min_value=0.0, max_value=2.0, help="扫描偏压"),))


# ── 1. 快照 ────────────────────────────────────────────────────────

def test_spec_snapshot_shows_params_stages_gates_and_wait_condition():
    text = render_spec_markdown(_spec_with_wait(), {"bias_v": 1.25},
                                conduct_id="camp01", experiment_id="exp-42")
    assert "camp01" in text and "exp-42" in text
    assert "bias_v" in text and "1.25" in text and "| V |" in text
    assert "A.01" in text and "A.03" in text
    assert "g_in" in text                      # 闸门 id
    assert "temperature_k <= 5" in text        # 等待条件,连数带号
    assert "600" in text                       # stale_after_s:多久算读不到
    # 人读快照必须说清自己不是真源 —— 否则事后有人拿它当账本对。
    assert "不是真源" in text


def _spec_with_llm_gate():
    from mast.conduct.spec import GateOutcome, GateSpec

    node = {"id": "n", "responsibility": "判这一段还值不值得接着测",
            "routes": {"go": "继续", "hold": "停下来问人"},
            "route_descriptions": {"go": "证据支持继续",
                                   "hold": "证据不支持,或说不清"},
            "escape": "hold"}
    gate = GateSpec(gate_id="g_llm", kind="llm", llm_node=node,
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    return make_spec([stage("A", [step("A.01", "ScanAt")], exit_gate=gate)])


def test_the_snapshot_says_what_an_llm_gate_asks_the_model():
    """快照是用户 approve 之前唯一会读的东西,而一道 llm 闸门与一道 rule 闸门
    在上面那一行里只差括号里三个字母 —— 可这两者要批的是完全不同的东西。

    **它凭什么这么判**只写在 ``llm_node.responsibility`` 里;不渲染出来,那道能
    发出 ``detour``(半夜叫醒用户换样品)的闸门就是在没人读过题面的情况下
    被批准的。这与段内闸门那次快照缺口是同一个形状。
    """
    text = render_spec_markdown(_spec_with_llm_gate(), {}, conduct_id="c",
                                experiment_id="e")
    assert "g_llm" in text and "(llm)" in text
    assert "判这一段还值不值得接着测" in text          # 问模型的问题
    assert "证据不支持,或说不清" in text               # 选项含义
    assert "判不了" in text and "hold" in text          # 弃权走哪条
    assert "不填任何数值" in text                       # 席位没有数值权限


def test_the_snapshot_never_names_a_model_it_cannot_promise():
    """approve 时印一个模型名 = 承诺一件这里保证不了的事。

    真正答题的是调用那一刻 provider 回退链选中的那个(静默回退在本仓发生过)。
    哪个模型真的答了,逐条记在 ``progress.jsonl`` 上。
    """
    text = render_spec_markdown(_spec_with_llm_gate(), {}, conduct_id="c",
                                experiment_id="e")
    for name in ("kimi", "deepseek", "claude", "gpt", "qwen", "glm", "minimax"):
        assert name not in text.lower()


def test_an_llm_gate_without_a_responsibility_is_called_out():
    """没写题面的 llm 闸门要在快照上**显眼**,不是安静地少一行。"""
    from mast.conduct.spec import GateOutcome, GateSpec

    node = {"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}
    gate = GateSpec(gate_id="g_bare", kind="llm", llm_node=node,
                    routes={"go": GateOutcome("pass"),
                            "hold": GateOutcome("wait_operator")})
    s = make_spec([stage("A", [step("A.01", "ScanAt")], exit_gate=gate)])
    text = render_spec_markdown(s, {}, conduct_id="c", experiment_id="e")
    assert "没写 responsibility" in text


def test_snapshot_flags_params_the_template_never_declared():
    """库里存着模板没声明的键 ⇒ 快照上要写出来。

    一个被悄悄忽略的参数,填的人会一直以为它生效了。
    """
    text = render_spec_markdown(_spec_with_wait(), {"bias_v": 1.0, "typo_v": 9},
                                conduct_id="c", experiment_id="e")
    assert "typo_v" in text


def test_sync_spec_versions_only_when_content_changed(rig):
    s = _spec_with_wait()
    first = rig.sync_spec(s, {"bias_v": 1.0})
    assert first == "spec_v001.md"
    assert rig.sync_spec(s, {"bias_v": 1.0}) == "spec_v001.md"   # 一模一样 ⇒ 不发版
    second = rig.sync_spec(s, {"bias_v": 1.5})                   # 参数变了 ⇒ 发版
    assert second == "spec_v002.md"
    assert rig.status().spec_doc == "spec_v002.md"


def test_sync_spec_leaves_no_half_file(rig):
    """原子替换:目录里不该留下 ``.part``。"""
    rig.sync_spec(_spec_with_wait(), {"bias_v": 1.0})
    d = rig.dir()
    assert not list(d.glob("*.part"))


# ── 2. 进度流 ──────────────────────────────────────────────────────

def test_each_progress_line_stands_alone(rig):
    assert rig.append(kind="wait_entered", ts="2026-08-15T01:00:00",
                      status_after="waiting_operator", stage_id="A",
                      step_id="A.03", payload={"wait_id": "w1"})
    assert rig.append(kind="wait_ack", ts="2026-08-15T02:00:00",
                      status_after="waiting_operator", payload={"by": "operator"})
    raw = (rig.dir() / PROGRESS_FILENAME).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 2
    for r in rows:
        # 自洽 = 不看别的行也知道是谁、什么时候、发生了什么。
        assert r["conduct_id"] == "camp01"
        assert r["ts"] and r["kind"] and r["summary"]
    assert rows[0]["step_id"] == "A.03"
    assert rows[1]["payload"]["by"] == "operator"


def test_unknown_event_kind_is_still_written(rig):
    """摘要模板里没有的种类照样写 —— 「没写进 jsonl」比「摘要不好看」严重得多。"""
    assert rig.append(kind="something_new", ts="t0")
    row = json.loads((rig.dir() / PROGRESS_FILENAME).read_text(
        encoding="utf-8").splitlines()[0])
    assert row["kind"] == "something_new" and row["summary"] == "something_new"


def test_append_is_incremental_only(rig):
    """INCREMENTAL-ONLY:只追加,前面的行永不被改写。"""
    for i in range(5):
        rig.append(kind="step_finished", ts=f"t{i}", payload={"n": i})
    lines = (rig.dir() / PROGRESS_FILENAME).read_text(
        encoding="utf-8").splitlines()
    assert [json.loads(x)["payload"]["n"] for x in lines] == [0, 1, 2, 3, 4]
    # 没有 finalize / close / archive 这类会「封存」的动作。
    assert not any(hasattr(rig, name)
                   for name in ("finalize", "close", "archive", "seal"))


# ── 3. 拿不到文件夹:如实报,不抛 ────────────────────────────────

def test_no_experiment_folder_reports_reason_and_never_raises():
    j = ConductJournal("c1", "ghost", folder_resolver=lambda _e: None)
    assert j.append(kind="status_change", ts="t") is False
    assert j.sync_spec(_spec_with_wait(), {}) == ""
    st = j.status()
    assert st.wired is False
    assert st.reason                       # 说清是哪一种「没有」
    assert st.progress_lines is None       # 读不到 ≠ 0 行


def test_resolver_that_explodes_is_reported_not_propagated():
    def boom(_e):
        raise RuntimeError("库挂了")

    j = ConductJournal("c1", "e", folder_resolver=boom)
    assert j.append(kind="status_change", ts="t") is False
    assert "库挂了" in j.status().reason


# ── 4. 「读不到」与「0 行」是两件事 ─────────────────────────────

def test_zero_lines_and_unreadable_are_different_answers(rig):
    rig.sync_spec(_spec_with_wait(), {})          # 建出目录,但还没写进度
    assert rig.status().progress_lines == 0       # 答得上来 ⇒ 0
    rig.append(kind="adopted", ts="t")
    assert rig.status().progress_lines == 1


def test_default_resolver_is_a_function_not_a_baked_in_path():
    """默认解析器必须是**函数**,不是一个模块级路径常量。

    常量意味着 import 时就固化,``MAST2_PROJECT_ROOT`` 之类的重定向再也追不上
    —— 心愿单那份板子就是这么写进用户真实数据里的。
    """
    assert callable(default_folder_resolver)
    assert default_folder_resolver("") is None    # 空 id 不去猜一个目录
