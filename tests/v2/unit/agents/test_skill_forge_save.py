"""保存 = 落版本 + 热注册 + 留痕。这一组钉住那三件事各自的**必要**部分。

最要紧的两条:

* **注册必须经 ``loader.register_spec``,不能裸调 ``registry.register``** ——
  撞名拒绝、缺失技能拒绝,以及**把 registry 传给 make_spec_skill**(安全级/能力
  标签的继承全靠那个参数)都长在 register_spec 里。绕过去就等于让声明说了算。
* **CAS 先于「内容没变」判** —— 反过来的话,一个拿着陈旧 base_version 的调用只要
  内容碰巧一样就会收到 ok=True,「你是最新的」会在它明明不是的时候被说出口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _forge_fixtures import (  # noqa: E402
    StubCtx,
    call,
    fresh_registry,
    spec_with,
    store,
    tools,
    two_step_spec,
)


def _kit(tmp_path):
    reg = fresh_registry()
    st = store(tmp_path)
    return reg, st, tools(reg=reg, ctx=StubCtx(), st=st)


def _save(ts, spec, **kw):
    return call(ts["save_composite"], spec_json=json.dumps(spec), **kw)


# ── 落版本 + 热注册 ─────────────────────────────────────────────────

def test_saving_registers_the_skill_so_it_is_runnable_immediately(tmp_path):
    reg, st, ts = _kit(tmp_path)
    out = _save(ts, two_step_spec())
    assert out["ok"] is True, out
    assert out["version"] == 1
    assert out["hot_registered"] is True
    assert reg.has("ScanThenCheck")
    assert st.exists("ScanThenCheck")


def test_the_saved_spec_is_stamped_with_the_agent_as_author(tmp_path):
    """``_author`` 是「这份是不是我造的」的唯一判据 —— 覆盖别人的作品靠它拦。"""
    _reg, st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    meta = st.load_meta("ScanThenCheck")
    assert meta["_author"] == "agent:instrument_control"
    assert meta["_origin"] == "agent_forge"


def test_registration_goes_through_register_spec_not_the_bare_registry(tmp_path, monkeypatch):
    """裸 ``registry.register`` 会跳过撞名拒绝、缺失技能拒绝和安全级继承。"""
    import mast.skills.composite.loader as loader
    seen: list[str] = []
    real = loader.register_spec

    def spy(registry, spec):
        seen.append(spec.name)
        return real(registry, spec)

    monkeypatch.setattr(loader, "register_spec", spy)
    _reg, _st, ts = _kit(tmp_path)
    assert _save(ts, two_step_spec())["ok"] is True
    assert seen == ["ScanThenCheck"]


def test_saving_pushes_the_refresh_chain_exactly_once(tmp_path, monkeypatch):
    """技能集合变了要让 UI 目录与 /agents/tools 失效,而且只推一次。"""
    import mast.agents._shared.skill_forge_tools as mod
    calls: list[str] = []
    monkeypatch.setattr(mod, "_refresh", lambda reason: calls.append(reason) or "ok")
    _reg, _st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    assert len(calls) == 1
    assert "ScanThenCheck" in calls[0]


def test_saving_leaves_a_diagnostics_breadcrumb(tmp_path, monkeypatch):
    """验收这条能力时该问的是「它今天造了什么」——那要可查,不能只在日志里。"""
    import mast.agents._shared.skill_forge_tools as mod
    rows: list[tuple] = []
    monkeypatch.setattr(mod, "_diag",
                        lambda kind, subj, reason, **f: rows.append((kind, subj, f)))
    _reg, _st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    assert [r[0] for r in rows] == ["skill_forged"]
    assert rows[0][1] == "ScanThenCheck"
    assert set(rows[0][2]["steps"]) == {"ScanAt", "AssessImageQuality"}


def test_skill_forged_is_a_real_diagnostics_kind(tmp_path):
    """`record()` 的 kind 是 Literal —— 用一个没声明的值只会静默混进 note 里。"""
    from mast.core import diagnostics
    assert "skill_forged" in diagnostics.Kind.__args__


# ── 乐观锁 / 版本膨胀 ──────────────────────────────────────────────

def test_overwriting_without_a_base_version_is_refused(tmp_path):
    _reg, _st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    out = _save(ts, two_step_spec())
    assert out["ok"] is False
    assert out["error"] == "base_version_required"
    assert out["stored_version"] == 1


def test_a_stale_base_version_is_a_conflict_even_when_the_content_matches(tmp_path):
    """CAS 必须在「内容没变」之前判。

    否则:调用方拿着 v1 的 base、库里已经是 v3、而它这次提交的内容碰巧和 v3 相同
    ⇒ 短路返回 ok=True。它会以为自己是最新的 —— 而 CAS 存在的全部理由就是回答
    「你是不是最新的」。
    """
    _reg, _st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    changed = two_step_spec()
    changed["description"] = "改过一次"
    assert _save(ts, changed, base_version=1)["ok"] is True     # → v2
    out = _save(ts, changed, base_version=1)                     # 陈旧 base,内容相同
    assert out["ok"] is False
    assert out["error"] == "version_conflict"
    assert out["stored_version"] == 2


def test_resaving_identical_content_does_not_grow_the_version_history(tmp_path):
    """每次 save 都落一份不可变快照;反复微调的循环能把版本刷到三位数。"""
    _reg, st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    out = _save(ts, two_step_spec(), base_version=1)
    assert out["ok"] is True
    assert out["unchanged"] is True
    assert out["version"] == 1
    assert len(st.list_versions("ScanThenCheck")) == 1


def test_a_real_edit_does_create_a_new_version(tmp_path):
    """短路要有边界:真改了就必须留下一版,否则历史会漏掉编辑。"""
    _reg, st, ts = _kit(tmp_path)
    _save(ts, two_step_spec())
    edited = two_step_spec()
    edited["description"] = "换了描述"
    out = _save(ts, edited, base_version=1)
    assert out["ok"] is True
    assert out.get("unchanged") is None
    assert out["version"] == 2
    assert len(st.list_versions("ScanThenCheck")) == 2


# ── 不许改用户的作品 ────────────────────────────────────────────

def test_an_operator_authored_composite_cannot_be_overwritten_by_the_agent(tmp_path):
    """他手上那份是他调过的 —— agent 要改就另存一份。"""
    reg, st, ts = _kit(tmp_path)
    from mast.skills.composite.spec import CompositeSpec
    st.save(CompositeSpec.from_dict(two_step_spec("OperatorMade")),
            extra_meta={"_author": "operator:alice"})
    out = _save(ts, two_step_spec("OperatorMade"), base_version=1)
    assert out["ok"] is False
    assert out["error"] == "not_yours"
    assert "换个名字" in "".join(out["problems"])


def test_a_composite_with_no_recorded_author_is_also_left_alone(tmp_path):
    """``load_meta`` 返回 {} 意思是「没记录」,不是「没人拥有」。

    把「读不到」折叠成一个具体的答案是本仓一天犯过五次的形状。这里保守的一侧是
    「别动」。
    """
    _reg, st, ts = _kit(tmp_path)
    from mast.skills.composite.spec import CompositeSpec
    st.save(CompositeSpec.from_dict(two_step_spec("Mystery")))
    out = _save(ts, two_step_spec("Mystery"), base_version=1)
    assert out["ok"] is False
    assert out["error"] == "not_yours"


# ── 安全级继承 ───────────────────────────────────────────────────

def _dangerous_leaf(reg) -> str:
    """一个 DANGEROUS、无必填参数、且**本机没被关掉**的技能名。

    关闭名单要一起过滤:被关掉的技能有它自己的拒绝理由,用它当素材的话这条测试
    会因为**另一个**原因红,而那种负例说明不了它想说明的事。
    """
    from mast.core.types import SafetyLevel
    from mast.agents._shared.skill_forge_tools import _disabled_names
    off = _disabled_names()
    return next(m.name for m in reg.list_skills()
                if m.safety_level is SafetyLevel.DANGEROUS
                and not m.parameters and m.name not in off)


def test_declaring_a_level_below_the_leaf_max_is_refused_at_save(tmp_path):
    """「给危险动作换个名字然后自称 auto」在保存期就被拒。

    这是整个工坊的安全底座第一层。它拒的不是**执行**而是**登记**:一份自称 auto
    的 spec 连进不了库,也就没有机会去骗下游任何一个读 safety_level 的闸门。
    """
    reg, _st, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": _dangerous_leaf(reg), "params": {}},
    ], name="QuietlyDangerous", safety_level="auto")
    out = _save(ts, spec)
    assert out["ok"] is False, out
    assert any("safety_level" in p for p in out["problems"]), out["problems"]
    assert not reg.has("QuietlyDangerous")


def test_a_composite_containing_a_dangerous_step_registers_as_dangerous(tmp_path):
    """第二层:登记之后**生效值**也是 dangerous,而且回执报的是生效值。

    第一层(声明不得低于叶子)是设计期检查,读的是 spec 文本;这一层读的是注册表
    里那个类真正的 metadata —— 下游所有闸门问的都是它。回执必须报生效值:报声明
    值等于在模型成功降级时告诉它「降级成功了」。
    """
    reg, _st, ts = _kit(tmp_path)
    from mast.core.types import SafetyLevel
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": _dangerous_leaf(reg), "params": {}},
    ], name="HonestlyDangerous", safety_level="dangerous")
    out = _save(ts, spec)
    assert out["ok"] is True, out
    assert out["effective_safety_level"] == "dangerous", out
    meta = reg._get_metadata(reg.get("HonestlyDangerous"))
    assert meta.safety_level is SafetyLevel.DANGEROUS


def test_a_composite_containing_a_pulse_inherits_the_capability_tag(tmp_path):
    """能力标签决定 SAFE/SEMI 拒不拒 —— 不继承的话 SAFE 对声明式工作流形同虚设。

    这一条和 safety_level 那两条不是一回事:``capabilities`` 没有「声明不得低于
    叶子」的设计期检查(spec 里根本没有这个字段可写),它**只**靠运行期继承。
    2026-08-01 审计发现的那个洞就是这里:``SpecComposite.metadata()`` 当时不填这
    个字段 ⇒ ``frozenset()`` ⇒ SAFE 对一切声明式工作流形同虚设。
    """
    reg, _st, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "b", "skill": "TipPulse",
         "params": {"pulse_v": 3.0, "duration_s": 0.1, "count": 1}},
    ], name="QuietPulser", safety_level="confirm")
    out = _save(ts, spec)
    assert out["ok"] is True, out
    meta = reg._get_metadata(reg.get("QuietPulser"))
    assert "bias_pulse" in (meta.capabilities or frozenset())


# ── 坏输入 ───────────────────────────────────────────────────────

def test_an_invalid_spec_says_which_step_is_wrong(tmp_path):
    """`ok=False` 配一个空 problems 是「不行但不说哪儿不行」——模型只能瞎猜。"""
    _reg, _st, ts = _kit(tmp_path)
    spec = spec_with([
        {"type": "step", "id": "a", "skill": "GetBias", "params": {}},
        {"type": "step", "id": "scan", "skill": "ScanAt", "params": {}},
    ], name="MissingParams")
    out = _save(ts, spec)
    assert out["ok"] is False
    assert out["error"] == "invalid_spec"
    assert out["problems"], "拒绝了却一条理由都没给"
    assert any("scan" in p for p in out["problems"])


def test_a_spec_without_a_name_is_refused(tmp_path):
    _reg, _st, ts = _kit(tmp_path)
    out = _save(ts, spec_with([], name=""))
    assert out["ok"] is False
    assert out["error"] == "bad_spec"
