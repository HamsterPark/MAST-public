# -*- coding: utf-8 -*-
"""原子分辨结果旁白应包含判定、读数和分析图。

从真实发出方经 narrate 和 ConversationStore 核验落库记录，
避免模板路径与载荷嵌套不一致而静默回退到缺省文本。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.chat import narration as N  # noqa: E402
from mast.chat import narration_templates as T  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402

CID = "conv-atomic-narration"


@pytest.fixture()
def store(tmp_path):
    """真的 ``ConversationStore``,建在 tmp 上。

    **必须显式 set_store**:本仓「测试污染真实用户数据」已经发生五次,而旁白写的
    正是用户真实会话所在的那个库。
    """
    N.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "mast_conversations.db")
    N.set_store(st)
    yield st
    N.flush(2.0)
    N.reset_for_tests()


# 独立构造的 VerifyAtomicResolution 形状载荷。
# 字段名遵循实际生产方；可选诊断字段用于核验模板的兼容路径。
def _verify(**over) -> dict:
    """``VerifyAtomicResolution.aggregate`` **真的会回**的那些键。

    ⚠️ 2026-08-23 更正：第一版这里写的是 ``period_nm`` 与
    ``angular_concentration_backward`` —— 两个真源**从不产出**的名字。
    替身比真货多几个键，测试就是在对着一个虚构对象验；同一天在
    ``message_clock`` 上刚栽过一模一样的（替身自己造了 ``runtime.config``，
    而真的 ``Runtime`` 是 ``__slots__``，根本没有那个字段）。
    """
    d = {
        "verdict": "atomic_resolved",
        "angular_concentration": 600.5,
        "period_radial_nm": 0.300,          # ← 真名（不是 period_nm）
        "period_fast_axis_nm": 0.2600,
        "half_concentrations": [560.0, 640.0],
        # 反扫的角向集中度**不在这里** —— VerifyAtomicResolution 没有这个生产方。
        # 它由三联图渲染器交出来（那里本来就要量反扫来画数字块）。
    }
    d.update(over)
    return d


def _emit_and_read(store, verify: dict, rungs=(), path="D:/x/synthetic_final.sxm") -> dict:
    """跑**真的发出方**,把落库的那一行读回来。"""
    from mast.skills.composite.achieve_atomic import _narrate_atomic_result

    with turn_scope(conversation_id=CID, run_id="run-1"):
        _narrate_atomic_result(path, verify, list(rungs))
    assert N.flush(5.0), "旁白写线程没排空"
    rows = [r for r in store.messages_since(CID, 0)
            if r["kind"] == N.NARRATION_KIND]
    assert len(rows) == 1, f"应当正好落一行,实际 {len(rows)} 行"
    return rows[0]


# ── ① 接线:发出去的 payload 真能渲染出数字 ──────────────────────────────


def test_the_emitted_payload_renders_numbers(store, tmp_path, monkeypatch):
    """**这一条是这个文件存在的理由。** 发出方发的形状必须渲染得出读数。

    用一张**真的** .sxm：反扫的角向集中度全仓只有三联图渲染器一个生产方
    （``AssessAtomicPhase`` 只取正扫），假路径 ⇒ 图画不出来 ⇒ 反扫那一句
    永远缺席。给真帧才测得到整条链路。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    sxm = _lattice_sxm(tmp_path / "lattice.sxm")
    row = _emit_and_read(store, _verify(), path=str(sxm),
                         rungs=[{"rung": "r1"}, {"rung": "r3a"},
                                {"rung": "r3b:recheck"}])
    text = row["text"]
    assert text != T.TEMPLATES["atomic_resolution_achieved"].fallback, (
        f"发出方发的形状渲染不出结论 —— 说的还是兜底那句:{text}")
    assert "拿到原子分辨" in text, text
    assert "600.5" in text, f"角向集中度没进句子:{text}"
    assert "0.3" in text, f"周期没进句子:{text}"
    assert "反扫" in text, (
        f"反扫读数没进句子 —— 「只在一个方向上出现的周期是针尖产物」是验收"
        f"少不了的一条，缺了它这句话会**若无其事地少说一样**:{text}")
    assert "560.0" in text and "640.0" in text, f"帧内两半没进句子:{text}"
    assert "r3b:recheck" in text, f"爬到第几档没说:{text}"


@pytest.mark.parametrize("verdict", ["atomic_resolved", "atomic_absent",
                                     "undecidable"])
def test_every_verdict_survives_the_wire(store, verdict):
    """三态**逐个**走一遍真发出方 —— 漏掉一态就是一种处境永远看不见。"""
    row = _emit_and_read(store, _verify(verdict=verdict))
    assert row["text"] != T.TEMPLATES["atomic_resolution_achieved"].fallback, (
        f"{verdict} 走了兜底:{row['text']}")


def test_the_facts_carry_the_numbers(store):
    """meta.facts 应保存可追溯到输入的数字；不得在句子有内容时悄悄丢失证据字段。"""
    row = _emit_and_read(store, _verify(), rungs=[{"rung": "r1"}])
    meta = json.loads(row["meta"])
    assert meta.get("degraded") is not True, "标着 degraded —— 必需字段没读到"
    facts = meta.get("facts") or {}
    for key in ("verdict", "concentration", "period_nm", "rungs_used",
                "scan_path"):
        assert key in facts, f"facts 里没有 {key} —— 这句话的数对不回去:{facts}"
    assert facts["concentration"] == pytest.approx(600.5)


def test_the_template_paths_and_the_emitter_agree_on_nesting():
    """模板声明的路径与发出方发的键**必须同层**(静态孪生)。

    走 AST,不走字符串切分:源码里解释这个坑的注释本身就写着 ``result={...}``,
    按字符串找会先撞上那段注释。
    """
    from mast.skills.composite import achieve_atomic as A

    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "narrate"
             and n.args and isinstance(n.args[0], ast.Constant)
             and n.args[0].value == "atomic_resolution_achieved"]
    assert len(calls) == 1, f"发出方有 {len(calls)} 处,应当只有一处"
    sent = {kw.arg for kw in calls[0].keywords if kw.arg}
    tpl = T.TEMPLATES["atomic_resolution_achieved"]
    for path in tpl.requires + tpl.records:
        assert "." not in path, (
            f"模板路径 {path} 带了层级,而发出方是平铺发的 —— 两边对不上")
        assert path in sent, f"模板要 {path},而发出方发的键是 {sorted(sent)}"
    assert "result" not in sent, (
        "发出方又把载荷包进 result= 了 —— 模板读的是根级路径,"
        "嵌套会使模板路径无法解析并触发兜底")


# ── ② 三态纪律:判不了 ≠ 没有 ────────────────────────────────────────────


def test_the_three_verdicts_are_the_skills_own():
    """措辞表的三个键必须**逐字**等于裁决技能的 ``VERDICT_*``。

    ``mast.chat`` 不反向 import ``mast.skills``(层次),所以字符串是抄的;
    抄错的后果不是报错,是安静地退回「原子分辨判定「atomic_resolved」」——
    一句语法完全正确、看起来只是有点生硬的话。这条平价测试就是那份保险。
    """
    from mast.skills.composite.verify_atomic_resolution import (
        VERDICT_ABSENT, VERDICT_RESOLVED, VERDICT_UNDECIDABLE)

    assert set(T._ATOMIC_VERDICT_ZH) == {VERDICT_RESOLVED, VERDICT_ABSENT,
                                         VERDICT_UNDECIDABLE}
    assert T._ATOMIC_RESOLVED == VERDICT_RESOLVED
    assert T._ATOMIC_UNDECIDABLE == VERDICT_UNDECIDABLE


def test_undecidable_is_never_read_as_absent(store):
    """``undecidable`` 与 ``atomic_absent`` 指向完全不同的下一步。

    前者要换视野/像素或重扫(测量条件问题),后者要动针尖。把「判不了」念成
    「没有」,就是拿针尖去补一个成像参数问题 —— 与 ``poke_indent`` 那条
    「``insufficient_data`` 不许劝人加深」是同一条纪律。
    """
    undec = _emit_and_read(store, {"verdict": "undecidable"})["text"]
    assert "判不了" in undec
    assert "不等于" in undec, "必须明说它不等于「没有原子分辨」"
    assert "这一帧上没有原子分辨" not in undec


def test_an_unknown_verdict_does_not_invent_one(store):
    """不认识的判定要如实带出来,不许编一个。"""
    text = _emit_and_read(store, {"verdict": "something_new"})["text"]
    assert "something_new" in text
    assert "拿到原子分辨" not in text


def test_a_missing_verdict_says_so_and_says_nothing_else(store):
    """读不到结论 ⇒ 明说没记下来。**不猜,也不默认成「没有」。**"""
    text = _emit_and_read(store, {})["text"]
    assert "没记下来" in text
    assert "拿到原子分辨" not in text and "没有原子分辨" not in text


def test_the_tone_flags_the_two_ends(store):
    """配色不许和句子说两件事:没拿到 / 判不了都要显眼,拿到了是好消息。"""
    def tone(v):
        return json.loads(_emit_and_read(store, {"verdict": v})["meta"])["tone"]

    N.reset_for_tests()
    assert T.TEMPLATES["atomic_resolution_achieved"].tone_of(
        {"verdict": "atomic_resolved"}) == "good"
    for v in ("atomic_absent", "undecidable"):
        assert T.TEMPLATES["atomic_resolution_achieved"].tone_of(
            {"verdict": v}) == "warn"


# ── ③ 配图:真的落盘了,不是只 import 成功 ────────────────────────────────


def _lattice_sxm(path: Path, n: int = 64) -> Path:
    """一张**带真晶格**的最小 .sxm(正反扫两块,合成二维正弦)。

    形状照 ``tests/v2/unit/data/test_scan_format_parity.py::_make_sxm``,
    只是把递增序列换成有周期的图 —— 否则谱上没有峰,三联图无从画起。
    """
    header = (
        ":SCAN_PIXELS:\n" f"{n} {n}\n"
        ":SCAN_OFFSET:\n" "0.0 0.0\n"
        ":SCAN_RANGE:\n" "3E-9 3E-9\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.0\t0.0\n"
        "\n:SCANIT_END:\n"
    )
    y, x = np.mgrid[0:n, 0:n]
    img = (np.sin(x * 0.9) + np.sin(y * 0.9)) * 3e-11
    fwd = img.astype(">f4")
    bwd = img[:, ::-1].astype(">f4")
    path.write_bytes(header.encode() + b"\x1a\x04"
                     + fwd.tobytes() + bwd.tobytes())
    return path


def test_the_triptych_really_lands_on_disk(store, tmp_path, monkeypatch):
    """三联图要**真的落盘**,而且那一行旁白要挂得上它。

    发出方那段 import 里有一个容易写错的名字(``project_root`` 住在
    ``mast._runtime_paths``,不是 ``mast.core.paths``),而整段被 ``except`` 吞掉
    ⇒ 图**静默地永远画不出来**,技能照常报成功。只验 import 成功是验不到的。

    ``MAST2_PROJECT_ROOT`` 指到 tmp:产物绝不许落进用户真实的 artifacts。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    sxm = _lattice_sxm(tmp_path / "lattice.sxm")
    row = _emit_and_read(store, _verify(), rungs=[{"rung": "r1"}],
                         path=str(sxm))

    out_dir = tmp_path / "artifacts" / "atomic_reports"
    pngs = list(out_dir.glob("*.png")) if out_dir.is_dir() else []
    assert pngs, f"三联图没落盘 —— {out_dir} 里一个 png 都没有"
    assert pngs[0].stat().st_size > 0

    meta = json.loads(row["meta"])
    img = meta.get("image") or {}
    assert img.get("src") == str(pngs[0]), f"旁白挂的不是刚画的那张:{img}"
    assert img.get("origin") == "milestone_png", (
        "origin 必须是 milestone_png —— ``sxm`` 那一档没有生产方,"
        "取图端点连门都不开(挂着图却永远 404 比没有图更坏)")


def test_a_broken_frame_still_narrates(store, tmp_path, monkeypatch):
    """图画不出来时**话照说** —— 配图绝不许把旁白带走。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    row = _emit_and_read(store, _verify(), path=str(tmp_path / "nope.sxm"))
    assert "拿到原子分辨" in row["text"]
    assert not (json.loads(row["meta"]).get("image") or {}), "凭空挂了一张图"


# ── ④ 两条不该做的事,钉成测试 ───────────────────────────────────────────


def test_achieve_atomic_is_not_wired_into_result_kind_for_skill():
    """``AchieveAtomicResolution`` **不进** ``RESULT_KIND_FOR_SKILL``。三条理由:

    1. 那张表驱动的是 ``GraphExecutor._narrate_step_result`` —— 它只在这个技能
       **作为别人的子步骤**时触发。本技能在自己的 ``_verify`` 里已经发过一条,
       接上去就是同一件事说两遍。
    2. 那条路径发的是 ``result=data``(即 ``result.*`` 路径),而本模板是**根级**
       路径 —— 接上去等于 100% 走兜底,正是 ``poke_indent`` 那个坑。
    3. 它的 ``aggregate()`` 输出与这里的平铺载荷是两种形状,不是同一份数据。

    要推翻这条:先说明「同一条旁白发两遍」和「路径分层对不上」各自怎么解决。
    """
    assert "AchieveAtomicResolution" not in T.RESULT_KIND_FOR_SKILL
    assert "AchieveAtomicResolution" not in T.BEGIN_KIND_FOR_SKILL


def test_which_of_the_emitters_keys_the_verifier_returns_today():
    """发出方从 verify 读的每一个键，``VerifyAtomicResolution`` 都要真的回。

    **这条测试钉的是一个缺口,不是一个正确的现状。** 钉它的理由:另外三个数会
    永远是 ``None``,而句子会**若无其事地少说三样** —— 与 ``auto_tilt_result``
    读取不存在键时触发兜底同理，只是这里丢失数值而非整句。

    真源 ``verify_atomic_resolution.py::aggregate`` 的 ``out.update({...})``:

    * ``period_nm``                      → 它回的是 ``period_radial_nm`` /
                                           ``period_fast_axis_nm``,**名字不同**;
    * ``half_concentrations``            → 没透传(``partial["criterion"]`` 里**有**,
                                           一行 ``crit.get(...)`` 就能接上);
    * ``angular_concentration_backward`` → **全仓没有任何生产方**,要它得对反扫
                                           通道再跑一次判据,那是一个功能不是一行。

    接上任意一个之后这条会红 —— 那时把它从下面这张表里划掉,**这正是它的用处**。
    """
    from mast.skills.composite.graph_executor import CompositeProgress
    from mast.skills.composite.verify_atomic_resolution import (
        VerifyAtomicResolution)

    # ⚠️ **调真的 aggregate(),不 grep 源码。** 第一版是按字符串在
    # ``out.update({...})`` 那一段里找键名的,而 ``"period_nm"`` 在那段里确实
    # 出现了 —— 出现在 ``"period_radial_nm": crit.get("period_nm")`` 的**右边**,
    # 也就是它**读取**的键,不是它**产出**的键。一个按错位置去找的证据,
    # 回答的不是被问的那个问题。
    skill = VerifyAtomicResolution()
    # ``run_composite`` 开头就是这一行 —— 它不在 ``__init__`` 里,直接调
    # ``aggregate`` 会 AttributeError。照它做,不绕过它。
    skill._verdict_out = {}
    returned_keys = set(skill.aggregate(
        {}, CompositeProgress(composite_name="VerifyAtomicResolution")))

    from mast.skills.composite import achieve_atomic as A

    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_narrate_atomic_result")
    # 发出方从 **verify 这个回包** 上按名字取的那几个键。
    #
    # ⚠️ 必须绑定到 ``verify`` 这个名字,不能收「任意 ``.get("…")``」:后者会把
    # ``r.get("rung")``(rung 记录,不是 verify 回包)也算进来,而且会漏掉写在
    # ``narrate(...)`` 调用**外面**的 ``verify.get("half_concentrations")``。
    # 第一版正是这么写的,于是这条检查报出来的缺失集合是错的 ——
    # **证据回答的不是被问的那个问题。**
    read_keys = sorted({
        n.args[0].value
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "verify"
        and n.args and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)})
    assert read_keys, "解析不出发出方从 verify 读了哪些键 —— 写法变了,这条检查失效了"

    present = [k for k in read_keys if k in returned_keys]
    missing = [k for k in read_keys if k not in returned_keys]
    assert "verdict" in present and "angular_concentration" in present, (
        f"连这两个都不回了?present={present}")
    # 2026-08-23 缺口**已补齐**，这张表从三条变成空的：
    #   period_nm                      → 发出方改读真名 period_radial_nm
    #   half_concentrations            → aggregate 里一行 crit.get(...) 透传
    #   angular_concentration_backward → 不再向 verify 要；它由三联图渲染器
    #                                    交出来（那里本来就量反扫画数字块），
    #                                    全仓只有那一处生产方
    # 现在这条从「记录一个缺口」变成「**守住它别再裂开**」。
    assert missing == [], (
        f"发出方又开始读 verify 不产出的键了:{sorted(missing)} —— "
        f"那几个数会永远是 None，而句子会**若无其事地少说几样**")
