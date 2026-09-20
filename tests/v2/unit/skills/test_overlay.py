"""Skill 覆盖层 —— 改 skill 不用发新版本。

这份测试分四组，每组盯一类「会静默出错」的事：

* **安全**：覆盖只能收紧。放松一次就等于把 HITL 门洗掉，而 UI 上看起来完全正常。
* **回滚**：停用之后必须回到**同一个内置类对象**，不是「一个同名的类」。
* **生效**：注册表换了 ≠ 模型手上换了。中间隔着一次图重建。
* **诊断**：「算不出这个技能」和「这个模块没有这个技能」必须分开说 —— 修法完全不同。

用动态造的假技能而不是真 skill：真 skill 的元数据会随实现变，一条测试因为别人
改了 SetBias 的包络而红，只会教人去改测试。
"""

from __future__ import annotations

import textwrap

import pytest

from mast.core.registry import SkillRegistry
from mast.core.types import (
    ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata,
)
from mast.skills.base import BaseSkill
from mast.skills.overlay import loader, manifest as M, paths as P

#: 覆盖层条目 → 它覆盖的模块。所有假技能都挂在这个模块名下。
REL = "builtins/_ovl_probe.py"
TARGET_MOD = "mast.skills.builtins._ovl_probe"


def _make_skill(name: str, *, level=SafetyLevel.DANGEROUS, caps=("bias_pulse",),
                lo=-2.0, hi=2.0, module=TARGET_MOD, version="1.0.0",
                rollback="Retract"):
    """造一个假技能类，``__module__`` 指到目标模块。"""
    meta = SkillMetadata(
        name=name, version=version, category=SkillCategory.WRITE,
        safety_level=level, description="probe",
        parameters=[ParameterSpec(name="bias_v", type="float", description="p",
                                  unit="V", required=True,
                                  min_value=lo, max_value=hi)],
        rollback_skill=rollback,
        capabilities=frozenset(caps),
    )
    cls = type(name, (BaseSkill,), {
        "metadata": lambda self, _m=meta: _m,
        "execute": lambda self, ctx, params: None,
    })
    cls.__module__ = module
    return cls


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    loader.reset_for_tests()
    yield
    loader.reset_for_tests()


@pytest.fixture
def reg():
    r = SkillRegistry()
    r.register(_make_skill("ProbeAlpha"))
    r.register(_make_skill("ProbeBeta", level=SafetyLevel.CONFIRM, caps=()))
    return r


def _write(source: str, *, enabled=True, allow_removals=None, rel=REL):
    d = P.overlay_dir(create=True)
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text(textwrap.dedent(source), encoding="utf-8")
    man = M.load()
    man.upsert(M.Entry(path=rel, enabled=enabled,
                       allow_removals=allow_removals or []))
    M.save(man)


#: 覆盖层文件的公共头 —— 造出与内置同名的技能。
HEAD = '''
    from mast.core.types import (ParameterSpec, SafetyLevel, SkillCategory,
                                 SkillMetadata)
    from mast.skills.base import BaseSkill


    def _meta(name, level, caps, lo, hi, rollback="Retract"):
        return SkillMetadata(
            name=name, version="1.0.0", category=SkillCategory.WRITE,
            safety_level=level, description="overlaid",
            parameters=[ParameterSpec(name="bias_v", type="float",
                                      description="p", unit="V", required=True,
                                      min_value=lo, max_value=hi)],
            rollback_skill=rollback,
            capabilities=frozenset(caps))
'''


def _skill_src(name, level="DANGEROUS", caps='("bias_pulse",)', lo=-2.0, hi=2.0,
               rollback='"Retract"'):
    return f'''
    class {name}(BaseSkill):
        def metadata(self):
            return _meta("{name}", SafetyLevel.{level}, {caps}, {lo}, {hi},
                         {rollback})

        def execute(self, ctx, params):
            return None
'''


# ── 安全：只能收紧 ──────────────────────────────────────────────────
def test_overlay_cannot_relax_safety_level(reg):
    """把 DANGEROUS 降成 AUTO 必须被拒 —— 整个模块，不是只跳过那一个技能。

    模块是用户编辑的单位；半应用状态正是本仓反复被咬的形状。
    """
    _write(HEAD + _skill_src("ProbeAlpha", level="AUTO")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    rep = loader.reload_skills(reg)
    assert rep.failed, "降级没被拒"
    reason = rep.failed[0].reason
    assert "safety_level" in reason and "dangerous" in reason and "auto" in reason
    assert reg.get("ProbeAlpha").__module__ == TARGET_MOD, "拒绝了却还是换了"


def test_overlay_cannot_drop_capabilities(reg):
    """能力标签只能加不能减 —— SAFE/SEMI 模式闸门依赖它。"""
    _write(HEAD + _skill_src("ProbeAlpha", caps="()")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    rep = loader.reload_skills(reg)
    assert rep.failed
    assert "bias_pulse" in rep.failed[0].reason


def test_overlay_cannot_widen_param_bounds(reg):
    _write(HEAD + _skill_src("ProbeAlpha", lo=-50.0, hi=50.0)
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    rep = loader.reload_skills(reg)
    assert rep.failed
    assert "上界" in rep.failed[0].reason or "下界" in rep.failed[0].reason


def test_overlay_cannot_drop_the_rollback_skill(reg):
    _write(HEAD + _skill_src("ProbeAlpha", rollback="None")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    rep = loader.reload_skills(reg)
    assert rep.failed
    assert "回滚" in rep.failed[0].reason


def test_tightening_is_allowed(reg):
    """收紧必须放行 —— 只测「拒绝」的话，一个恒拒的实现也能全绿。"""
    _write(HEAD + _skill_src("ProbeAlpha", lo=-1.0, hi=1.0,
                             caps='("bias_pulse", "tip_shaping")')
           + _skill_src("ProbeBeta", level="DANGEROUS", caps='("x",)'))
    rep = loader.reload_skills(reg)
    assert not rep.failed, rep.describe()
    assert "_overlay" in reg.get("ProbeAlpha").__module__


def test_relaxation_check_uses_raw_metadata(reg, tmp_path):
    """比对必须 raw ↔ raw，不能拿合并了管理员覆写的包络当基线。

    反例（这条测试的全部理由）：管理员合法地把 ProbeAlpha 降到 CONFIRM（有审计、
    有 PIN、有理由）→ 那个 CONFIRM 成了基线 → 一个同样声明 CONFIRM 的覆盖
    **合法通过** → 管理员事后撤销覆写，覆盖还在，这个技能永久停在 CONFIRM。
    两层各自看都讲得通，合起来把审批门洗掉了。
    """
    import json

    from mast.admin.override_store import SKILL_OVERRIDES, ConfigOverrideRegistry

    d = tmp_path / "config" / "overrides"
    d.mkdir(parents=True, exist_ok=True)
    (d / SKILL_OVERRIDES).write_text(
        json.dumps({"ProbeAlpha": {"safety_level": "confirm"}}), encoding="utf-8")
    ConfigOverrideRegistry.reset()
    ConfigOverrideRegistry.get(d)
    try:
        _write(HEAD + _skill_src("ProbeAlpha", level="CONFIRM")
               + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
        rep = loader.reload_skills(reg)
        assert rep.failed, "覆写把基线降下去了，于是放松被当成了合法"
        assert "safety_level" in rep.failed[0].reason
    finally:
        ConfigOverrideRegistry.reset()


# ── 三集合：静默删除 ────────────────────────────────────────────────
def test_a_module_that_drops_a_skill_is_refused(reg):
    """从模块里删掉一个技能，默认拒绝整个模块。

    99% 是误删或改名，而「静默保留内置版」正好是「以为生效其实没生效」：
    用户会看到那个技能还在，以为覆盖没起作用，然后去改别的地方。
    """
    _write(HEAD + _skill_src("ProbeAlpha"))     # 少了 ProbeBeta
    rep = loader.reload_skills(reg)
    assert rep.failed
    assert "ProbeBeta" in rep.failed[0].reason
    assert "allow_removals" in rep.failed[0].reason, "没给出逃生口"


def test_an_explicit_allow_removals_lets_it_through(reg):
    _write(HEAD + _skill_src("ProbeAlpha"), allow_removals=["ProbeBeta"])
    rep = loader.reload_skills(reg)
    assert not rep.failed, rep.describe()
    assert not reg.has("ProbeBeta"), "声明要移除，却还在"


# ── 回滚 ────────────────────────────────────────────────────────────
def test_disable_restores_the_very_same_builtin_class(reg):
    """停用后必须是**同一个类对象**，不是「一个同名的类」。

    ``is`` 而不是 ``==``：一个「重新 discover 一遍」的实现能让 ``==`` 通过，
    但那时别处持有的旧引用与注册表里的已经是两个对象了。
    """
    base = reg.get("ProbeAlpha")
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    assert not loader.reload_skills(reg).failed
    assert reg.get("ProbeAlpha") is not base

    man = M.load()
    man.upsert(M.Entry(path=REL, enabled=False))
    M.save(man)
    rep = loader.reload_skills(reg)

    assert reg.get("ProbeAlpha") is base, "没回到同一个类对象"
    assert reg.provenance("ProbeAlpha").origin == "builtin"
    assert not rep.baseline_drift


def test_disable_restores_every_version(reg):
    """内置有多版本时要整份放回去 —— ``_skills[name]`` 是 ``{version: class}``。"""
    v2 = _make_skill("ProbeAlpha", version="2.0.0")
    reg.register(v2)
    before = dict(reg._skills["ProbeAlpha"])
    assert len(before) == 2

    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    assert not loader.reload_skills(reg).failed
    man = M.load()
    man.upsert(M.Entry(path=REL, enabled=False))
    M.save(man)
    loader.reload_skills(reg)

    assert reg._skills["ProbeAlpha"] == before, "多版本没整份放回"


def test_disable_refuses_when_someone_else_took_the_name(reg, caplog):
    """第三方在我们之后注册了同名技能 → 拒绝回滚它，并大声说。

    盲目回填会把别人的注册覆盖掉，而那是一次**看不见的**破坏。
    """
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    assert not loader.reload_skills(reg).failed

    third = _make_skill("ProbeAlpha", module="some.third.party")
    reg.register(third)

    man = M.load()
    man.upsert(M.Entry(path=REL, enabled=False))
    M.save(man)
    with caplog.at_level("WARNING"):
        loader.reload_skills(reg)
    assert reg.get("ProbeAlpha") is third, "把第三方的注册覆盖掉了"
    assert any("不是我放的那个类" in r.getMessage() for r in caplog.records)


# ── 幂等 / 隔离 ─────────────────────────────────────────────────────
def test_reloading_an_unchanged_overlay_is_a_noop(reg):
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    assert loader.reload_skills(reg).status == loader.STATUS_OK
    assert loader.reload_skills(reg).status == loader.STATUS_NOCHANGE


def test_a_broken_file_does_not_take_down_the_others(reg):
    """单个模块失败只跳过它自己（同 custom_loader 的纪律）。"""
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    _write("def broken(:\n    pass\n", rel="builtins/_ovl_broken.py")
    rep = loader.reload_skills(reg)
    assert len(rep.applied) == 1 and len(rep.failed) == 1
    assert "_overlay" in reg.get("ProbeAlpha").__module__


def test_a_syntax_error_is_caught_before_any_exec(reg):
    """半保存的文件被读到是常事 —— 它必须变成一句话，不是半个加载。"""
    _write("class Half(BaseSkill:\n")
    rep = loader.reload_skills(reg)
    assert rep.failed and "语法错误" in rep.failed[0].reason
    assert reg.get("ProbeAlpha").__module__ == TARGET_MOD


def test_task_busy_queues_instead_of_swapping(reg):
    """任务运行中**整件事**挂起 —— 注册表一个类都不动。

    ``ExecutionContext.run`` 每个子步骤都现查注册表，中途换会让跑到一半的
    composite 后半段用新代码。那比「晚几分钟生效」危险得多，且不可复盘。
    """
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    rep = loader.reload_skills(reg, task_busy=True)
    assert rep.status == loader.STATUS_QUEUED
    assert reg.get("ProbeAlpha").__module__ == TARGET_MOD, "任务运行中却换了"


# ── 诊断：算不出 ≠ 没有 ─────────────────────────────────────────────
def test_a_metadata_that_raises_says_so(reg):
    """``metadata()`` 抛异常 ≠ 「这个模块没有这个技能」。

    静默跳过会让用户去找自己删了什么，而真正要修的是 metadata 里的 bug ——
    一次「读不到」被折叠成了一个具体的值。
    """
    _write(HEAD + '''
    class ProbeAlpha(BaseSkill):
        def metadata(self):
            raise RuntimeError("boom in metadata")

        def execute(self, ctx, params):
            return None
''')
    rep = loader.reload_skills(reg)
    assert rep.failed
    reason = rep.failed[0].reason
    assert "metadata()" in reason and "boom in metadata" in reason
    assert "不是「没有这个技能」" in reason


# ── provenance ──────────────────────────────────────────────────────
def test_provenance_records_what_was_displaced(reg):
    _write(HEAD + _skill_src("ProbeAlpha")
           + _skill_src("ProbeBeta", level="CONFIRM", caps="()"))
    assert not loader.reload_skills(reg).failed
    p = reg.provenance("ProbeAlpha")
    assert p.origin == "overlay"
    assert p.source_path == REL
    assert p.sha256 and p.displaced_module == TARGET_MOD
    assert p.signature == "local_unsigned"
    assert "顶掉" in p.describe()


def test_provenance_never_returns_none(reg):
    for name in ("ProbeAlpha", "ProbeBeta", "NoSuchSkillAtAll"):
        assert reg.provenance(name) is not None
    assert reg.provenance("NoSuchSkillAtAll").origin == "absent"


# ── 路径 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", ["../etc/passwd.py", "a/../../b.py",
                                 "/abs/x.py", "C:/x.py", "x.txt",
                                 "builtins/2bad/x.py"])
def test_bad_paths_are_refused(bad):
    ok, why = P.is_valid_rel(bad)
    assert not ok, f"{bad} 被当成了合法条目"
    assert why


def test_path_normalisation_does_not_eat_the_traversal():
    """``lstrip("./")`` 剥的是**字符集**不是前缀 —— 它会把 ``../x`` 吃成 ``x``，
    于是穿越检查还没跑，要查的那两个点已经没了。"""
    assert P.normalise_rel("../etc/x.py") == "../etc/x.py"
    assert P.normalise_rel("./builtins/x.py") == "builtins/x.py"
