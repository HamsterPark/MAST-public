"""The artifact data-flow graph must be DERIVED from reality, not asserted.

The old model was a hand-written {agent: artifact} table that drifted silently
from the code until it was pure fiction (it claimed paper_review reads .sxm scan
files, and 5 of its 6 artifact ids named MASTState fields that are dead). The
replacement derives the edges from the tools each agent actually holds — so the
failure mode to guard against is now different: the TOOL_ACCESS map itself could
name tools that do not exist.

These tests close that hole:
  * every tool named in TOOL_ACCESS must EXIST on some agent (no invented names);
  * the meta-tool name constants must match the tools the factory really builds;
  * the derived graph must reproduce the facts we know independently (the
    multi-writers, the read-only vision buffer, who may write a plan);
  * and — added 2026-08-14 — the derivation must read the SAME constant the
    runtime grants by. Deriving from a second, hand-kept copy is how this file
    came to pin a fact that had been false for a year.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from tests.v2.toolcall import tool_call
from mast.agents._shared.artifacts import (  # noqa: E402
    ARTIFACT_BY_ID,
    ARTIFACTS,
    PIPELINE,
    TOOL_ACCESS,
    agent_tool_names,
    derive_flow,
)


_MASTV2_ROOT_PATH = Path(_MASTV2_ROOT)


def _flow_by_id():
    return {f.artifact.id: f for f in derive_flow()}


def _xd_meta_grant() -> set[str]:
    """artifacts 派生出来的 experiment_design **meta 工具**面。

    与 META_TOOL_NAMES 取交是为了把 XD 自己的工具(describe_skills…)和每个 agent
    都有的 buffer/memory 工具摘出去 —— 这里要比的只是 runtime 授予的那一份。
    """
    from mast.agents._shared.meta_tools import META_TOOL_NAMES
    return agent_tool_names()["experiment_design"] & set(META_TOOL_NAMES)


def _design_meta_expected() -> set[str]:
    """真源:`DESIGN_TOOL_NAMES` 里真正存在于 meta 工具集里的那些名字。"""
    from mast.agents._shared.meta_tools import DESIGN_TOOL_NAMES, META_TOOL_NAMES
    return set(DESIGN_TOOL_NAMES) & set(META_TOOL_NAMES)


# ════════════════════════════════════════════════════════════════════════
# The map cannot rot: every name in it must be a REAL tool
# ════════════════════════════════════════════════════════════════════════

class TestMapIsGroundedInRealTools:
    def test_every_declared_tool_exists_on_some_agent(self):
        """If a tool named in TOOL_ACCESS exists nowhere, the edge it declares is
        a fiction — exactly the disease the old hand-written table died of."""
        held = agent_tool_names()
        everything: set[str] = set()
        for names in held.values():
            everything |= names
        missing = sorted(t for t in TOOL_ACCESS if t not in everything)
        assert not missing, (
            "TOOL_ACCESS names tools that no agent holds — these edges are "
            f"fiction: {missing}"
        )

    def test_every_declared_artifact_is_reachable(self):
        """An artifact nothing can read or write is dead weight in the graph."""
        flows = _flow_by_id()
        for art in ARTIFACTS:
            f = flows[art.id]
            assert f.writers or f.readers, (
                f"{art.id} has neither writers nor readers — no tool touches it"
            )

    def test_meta_tool_constants_match_the_real_factory(self):
        """META_TOOL_NAMES / LIFECYCLE_TOOL_NAMES are used to derive the graph
        WITHOUT building the tools (that needs a live provider). Pin them against
        what the factory actually produces, so they can never drift."""
        from mast.agents._shared.meta_tools import (
            LIFECYCLE_TOOL_NAMES,
            META_TOOL_NAMES,
            make_meta_tools,
        )
        real = {t.name for t in make_meta_tools(lambda: {})}
        assert set(META_TOOL_NAMES) == real, (
            "META_TOOL_NAMES drifted from make_meta_tools(): "
            f"missing={real - set(META_TOOL_NAMES)}, "
            f"stale={set(META_TOOL_NAMES) - real}"
        )
        assert set(LIFECYCLE_TOOL_NAMES) <= real

    def test_store_never_names_a_dead_state_field(self):
        """`store` exists so a claim can be CHECKED against the disk. The typed
        MASTState artifact slots (last_scan / experiment_plan / analysis / draft
        / review / tip_status / pending_scan) have ZERO reads and ZERO writes in
        the whole tree, so naming one is naming nothing."""
        for art in ARTIFACTS:
            assert "MASTState." not in art.store, (
                f"{art.id} claims a MASTState store — those slots are dead"
            )


# ════════════════════════════════════════════════════════════════════════
# The derived graph must reproduce facts we know independently
# ════════════════════════════════════════════════════════════════════════

class TestDerivedGraphMatchesReality:
    def test_multi_writers_fall_out_of_the_derivation(self):
        """The operator's point ('some documents can be written by several
        agents') is not something we assert — it EMERGES from who holds what."""
        flows = _flow_by_id()

        # figures: DP renders (mosaic/montage), PW embeds into the manuscript
        figures = flows["figures"]
        assert set(figures.writers) == {"data_processing", "paper_writing"}
        assert figures.multi_writer

        # the experiment DB: IC holds the full meta-tool set, XD the lifecycle
        # subset — both write it
        records = flows["experiment_records"]
        assert {"instrument_control", "experiment_design"} <= set(records.writers)
        assert records.multi_writer

        # memory: every agent carries the memory tools
        memory = flows["memory"]
        assert set(memory.writers) == set(PIPELINE)
        assert memory.multi_writer

    def test_the_design_agent_writes_the_design(self):
        """XD 持有 `create_plan`(2026-07-27 补的),所以它**是** experiment_plan 的
        写者 —— 图应当自己从工具清单里看出这一点。

        这条测试原来钉的是反面(`create_plan not in held["experiment_design"]`),
        而且是绿的:它比对的是 `artifacts.py` 自己的硬编码派生
        (`meta & LIFECYCLE`),不是 runtime 真正授予 XD 的集合。派生方定义了自己的
        输入 ⇒ 永远自洽,于是一条绿测试为一句假话背了一年书,并且会给下一个
        「照注释把 create_plan 从 XD 摘掉」的人发通行证。
        """
        held = agent_tool_names()
        assert "create_plan" in held["instrument_control"]
        assert "create_plan" in held["experiment_design"]

        plan = _flow_by_id()["experiment_plan"]
        assert {"instrument_control", "experiment_design"} <= set(plan.writers)
        # 2026-08-20:批准与执行三件**不再按角色裁**。从前这里断言 XD 不持有
        # 它们,理由是「批准不是设计 agent 的动作」;现在把关移到了服务端
        # (conduct 的自主度策略 + 参数包络),工具面对全部 agent 一样。
        # 详见 tests/v2/unit/agents/test_xd_design_tool_surface.py 里那条被
        # 翻过来的断言 —— 它同时钉住了「闸门搬去了哪」。
        for now_granted in ("approve_plan", "advance_plan", "pause_plan", "resume_plan"):
            assert now_granted in held["experiment_design"], (
                f"{now_granted} 又被按角色裁掉了 —— 如果这是有意的,"
                f"请先回答把关靠什么(「不给工具」防不住换个 agent 去调)")

    def test_xd_meta_grant_is_the_constant_runtime_filters_by(self):
        """**一个集合,两个消费者**:artifacts 派生 XD 边用的名单,和 runtime 建群聊
        时过滤 XD meta 工具用的名单,必须是同一个 `DESIGN_TOOL_NAMES`。

        比 `create_plan in ...` 更结实的地方在于:它钉的是**派生源**而不是某一个
        名字,所以下一次有人往 XD 加/减工具时,图会跟着动,而不是等谁想起来改注释。
        """
        assert _xd_meta_grant() == _design_meta_expected()

    def test_mutation_the_old_derivation_would_be_caught(self, monkeypatch):
        """先证明上面那条检查器会动手,再信它说「没问题」。

        变异 = 把派生源改回修复前的那份(`meta & LIFECYCLE`)。若断言仍然绿,
        说明它比对的两边根本是同一个东西,那它什么也没在校验。
        """
        from mast.agents._shared import meta_tools as mt

        # 真源快照必须在变异**之前**取:期望值和被测值都读同一个常量的话,
        # 变异会被两边同时吸收 —— 那正是这条测试要排除的自洽。
        expected = _design_meta_expected()
        monkeypatch.setattr(mt, "DESIGN_TOOL_NAMES", mt.LIFECYCLE_TOOL_NAMES)

        mutated = _xd_meta_grant()
        assert mutated != expected, (
            "把派生源换成 LIFECYCLE 之后 XD 的工具面没变 —— "
            "说明 agent_tool_names() 根本没在读 DESIGN_TOOL_NAMES")
        # 只断言「变了」还不够:一个恒返回空集的派生也会「变」。要正面证明它
        # **跟着常量走到了新值**,才算证明了这条线是活的。
        assert mutated == set(mt.LIFECYCLE_TOOL_NAMES) & set(mt.META_TOOL_NAMES)
        assert "create_plan" not in mutated  # 正是当年那条假事实

    def test_runtime_filters_xd_by_the_same_constant(self):
        """runtime 那一侧的钉子。**读源码**而不起 runtime:CoreRuntime 要真机连接,
        而这条要问的只是「它按什么名单过滤」。

        重列名字是这条断链的根因(runtime 一份、artifacts 一份、注释三份),所以
        重新出现**任何一个手写的工具名字面量**就判红。
        """
        src = (_MASTV2_ROOT_PATH / "mast" / "core" / "runtime.py").read_text(
            encoding="utf-8")
        head = src.index("_xd_meta_tools = None")
        tail = src.index("experiment_design_extra_tools=", head)
        block = src[head:tail]
        assert 0 < len(block) < 4000, "XD 过滤块没找对,这条断言在空转"
        assert "DESIGN_TOOL_NAMES" in block, (
            "runtime 不再按 DESIGN_TOOL_NAMES 过滤 XD —— 两个消费者又分叉了")
        for literal in ('"create_plan"', '"start_experiment"', '"get_current_tip"',
                        "'create_plan'", "'start_experiment'"):
            assert literal not in block, (
                f"XD 过滤块里又出现了手写的工具名 {literal}:"
                "名单要留在 meta_tools.DESIGN_TOOL_NAMES 一处")

    def test_vision_buffer_is_read_only_for_agents(self):
        """'Agents never write the buffer' is a standing invariant. The old
        one-writer-per-artifact table could not even express writers=[]."""
        vb = _flow_by_id()["vision_buffer"]
        assert vb.writers == []
        assert "instrument_control" in vb.readers
        assert ARTIFACT_BY_ID["vision_buffer"].system_written is True

    def test_review_loop_is_bidirectional(self):
        """PW writes the draft, PR reads it; PR writes the review, PW reads it
        back to revise. A linear pipeline picture cannot show this cycle."""
        flows = _flow_by_id()
        assert flows["draft"].writers == ["paper_writing"]
        assert "paper_review" in flows["draft"].readers
        assert flows["review"].writers == ["paper_review"]
        assert "paper_writing" in flows["review"].readers

    def test_no_agent_is_listed_as_both_writer_and_reader(self):
        """A writer implicitly reads its own artifact; listing it twice is what
        made the old UI show a meaningless 'R6' on every row."""
        for f in derive_flow():
            assert not (set(f.writers) & set(f.readers)), (
                f"{f.artifact.id}: {set(f.writers) & set(f.readers)} listed twice"
            )

    def test_paper_review_does_not_read_scan_files(self):
        """The single most obvious fiction of the old model."""
        assert "paper_review" not in _flow_by_id()["scan_files"].readers

    @pytest.mark.parametrize("agent", PIPELINE)
    def test_every_agent_has_at_least_one_edge(self, agent):
        flows = derive_flow()
        touched = any(agent in f.writers or agent in f.readers for f in flows)
        assert touched, f"{agent} touches no artifact at all — suspicious"


class TestExistingArtifacts:
    def test_lists_real_files_only(self, tmp_path, monkeypatch, documents_root):
        """Replaces task['artifacts'], which had NO producer anywhere in the tree
        and was therefore always {} — the 'produced' panel could only ever be
        empty, and a mock in the suite hid that. Every entry here is a real file."""
        from mast.agents._shared.artifacts import list_existing

        monkeypatch.setenv("MAST_DRAFTS_DIR", str(tmp_path / "drafts"))
        monkeypatch.setenv("MAST_REVIEWS_DIR", str(tmp_path / "reviews"))
        monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
        # ``documents_root`` redirects MAST_EXPERIMENT_ROOT (where documents live)
        # AND MAST_EXPERIMENT_DB (where plans live, beside PlanStore's default).
        # Both are needed: MAST2_PROJECT_ROOT redirects NEITHER, which is how a
        # subset run wrote 11 real document directories into the operator's data.
        #
        # This assertion is the tripwire for a newly-enumerated class that nobody
        # remembered to isolate: it fails the moment list_existing() starts
        # reading a real store the test did not redirect.
        assert list_existing() == []          # nothing produced yet — honestly empty

        from mast.agents.paper_writing.tools import save_draft
        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "# T\nbody"}))

        got = list_existing()
        assert len(got) == 1
        e = got[0]
        assert e["artifact_id"] == "draft"
        assert e["editable"] is True          # a file the operator CAN edit
        assert Path(e["path"]).is_file()      # and it really is on disk
        # The id is the document's own stable identity, not "<class>:<file stem>":
        # the old synthesised id changed whenever the title was reworded, so an id
        # handed to the editor could stop resolving.
        assert e["doc_id"] and ":" not in e["doc_id"]
        assert "T" in e["name"] and "v1" in e["name"]   # title + version, readable
