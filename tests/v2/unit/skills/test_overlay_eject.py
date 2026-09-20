"""「导出到覆盖层」—— 取出源码字节，并说清楚**哪些入口它管不着**。

这份测试盯的四类事，每一类在实现过程中都真的错过一次：

* **反函数**：``rel_for_module`` 和 ``overlay_of`` 是同一条规则的两面。分开写就是
  「一侧改了另一侧没跟上」，而症状是导出的文件落在一个加载器找不到的路径上。
* **跨模块基类**：只看模块内继承，``make_special_tip`` 会被判成「一个技能类都没有」，
  于是报告会说出最重的那句「覆盖它一处都不会生效」——**而事实恰恰相反**。
  说反了比不说更糟。
* **三态**：基类解析不到时报「判断不了」，不是「不是技能类」。「读不到」被折叠成
  一个具体的值，是本仓记了一整页的那类事故。
* **drift**：覆盖**生效了**，但它基于三个版本前的代码，上游后来修的 bug 被原样盖了
  回去。没 sidecar 时报 ``None`` 而不是 ``False``。

真模块（``mast.skills.*``）只在「对账」那一组里用，而且断言的是**方向**
（registry 有的一个都不能漏），不是具体数字 —— 数字会随别人加技能而变，
一条因为别人加了技能而红的测试只会教人去改测试。
"""

from __future__ import annotations

import json

import pytest

from mast.skills.overlay import eject as E, paths as P


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """覆盖层目录指到 tmp —— 测试写进用户真实数据目录，本仓被咬过五次。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    yield


@pytest.fixture
def fake_src(tmp_path, monkeypatch):
    """一棵假的 ``mast/`` 源码树。

    不用真源码树，因为这一组测的是**行为**：真树里 ``bias.py`` 的导入方数量会随
    别人改代码而变。真树留给「对账」那一组。
    """
    root = tmp_path / "src"
    (root / "skills" / "builtins").mkdir(parents=True)
    (root / "skills" / "composite").mkdir(parents=True)
    (root / "agents").mkdir(parents=True)

    (root / "skills" / "base.py").write_text(
        "class BaseSkill:\n    pass\n", encoding="utf-8")
    # 被覆盖的目标：两个技能类 + 一个常量
    (root / "skills" / "builtins" / "probe.py").write_text(
        "from mast.skills.base import BaseSkill\n"
        "THRESHOLD_PA = 50.0\n"
        "class AlphaSkill(BaseSkill):\n    pass\n"
        "class BetaSkill(BaseSkill):\n    pass\n", encoding="utf-8")
    # 包 __init__ 显式 import 子模块（真实结构）
    (root / "skills" / "builtins" / "__init__.py").write_text(
        "from mast.skills.builtins.probe import AlphaSkill, BetaSkill\n",
        encoding="utf-8")
    # 直接绑定技能类的入口 —— 覆盖层管不着
    (root / "agents" / "tools.py").write_text(
        "from mast.skills.builtins.probe import AlphaSkill\n"
        "from mast.skills.builtins.probe import THRESHOLD_PA\n", encoding="utf-8")
    # 从**包**导入（node.module 不等于目标模块）
    (root / "agents" / "other.py").write_text(
        "from mast.skills.builtins import BetaSkill\n", encoding="utf-8")
    # 跨模块基类：中间基类在另一个模块里
    (root / "skills" / "composite" / "mid.py").write_text(
        "from mast.skills.base import BaseSkill\n"
        "class MidBase(BaseSkill):\n    pass\n", encoding="utf-8")
    (root / "skills" / "composite" / "leaf.py").write_text(
        "from mast.skills.composite.mid import MidBase\n"
        "class LeafSkill(MidBase):\n    pass\n", encoding="utf-8")
    # 基类解析不到 —— 应该报「判断不了」
    (root / "skills" / "builtins" / "murky.py").write_text(
        "from somewhere.unknown import Mystery\n"
        "class MurkySkill(Mystery):\n    pass\n", encoding="utf-8")

    from mast.pyexec import _srcfiles
    monkeypatch.setattr(_srcfiles, "source_root", lambda: root)
    return root


# --------------------------------------------------------------------------
# 反函数
# --------------------------------------------------------------------------

def test_rel_and_module_are_inverses():
    """``overlay_of`` 和 ``rel_for_module`` 必须互为反函数。

    两个方向分开写，就会出现「加载器按 A 规则找，导出按 B 规则写」——
    导出的文件落在加载器不看的路径上，而两边各自都「对」。
    """
    for rel in ["builtins/bias.py", "composite/auto_tilt.py", "paper/fit_bcs.py",
                "builtins/sub/deep.py"]:
        dotted = P.overlay_of(rel)
        assert dotted is not None
        assert P.rel_for_module(dotted) == rel


def test_rel_for_module_refuses_non_skill_modules():
    """架构代码不在覆盖层射程内 —— 那条线是用户明确画的。"""
    for dotted in ["mast.core.registry", "mast.agents.data_processing.tools",
                   "mast.api.routes.admin", "mast.skills"]:
        assert P.rel_for_module(dotted) is None


def test_rel_for_module_refuses_the_overlay_reserved_segments():
    """``_new`` / ``_packs`` / ``_overlay`` 是覆盖层自己的保留段，不是内置模块。"""
    for seg in (P.NEW_DIR, P.PACKS_DIR, "_overlay"):
        assert P.rel_for_module("mast.skills." + seg + ".thing") is None


# --------------------------------------------------------------------------
# 技能类识别 —— 三态
# --------------------------------------------------------------------------

def test_skill_classes_finds_direct_subclasses(fake_src):
    sk, unk = E._skill_classes(
        (fake_src / "skills" / "builtins" / "probe.py").read_bytes(),
        "mast.skills.builtins.probe", source_root=fake_src)
    assert sk == {"AlphaSkill", "BetaSkill"}
    assert not unk


def test_skill_classes_follows_a_base_imported_from_another_module(fake_src):
    """**变异测试**：去掉跨模块那一层闭包，这条会红。

    而它红的方式恰恰是最坏的那种 —— 报告会说「这个模块里没有技能类，覆盖它一处
    都不会生效」，用户据此放弃覆盖，**而覆盖本来是生效的**。
    """
    sk, unk = E._skill_classes(
        (fake_src / "skills" / "composite" / "leaf.py").read_bytes(),
        "mast.skills.composite.leaf", source_root=fake_src)
    assert sk == {"LeafSkill"}, "跨模块继承的技能类被漏判了 —— 报告会把结论说反"
    assert not unk


def test_unresolvable_base_reports_undecidable_not_absent(fake_src):
    """「判断不了」不能折成「不是技能类」。"""
    sk, unk = E._skill_classes(
        (fake_src / "skills" / "builtins" / "murky.py").read_bytes(),
        "mast.skills.builtins.murky", source_root=fake_src)
    assert sk == set()
    assert unk == {"MurkySkill"}
    w = E._warning("mast.skills.builtins.murky", sk, [], unk)
    assert "判断不了" in w
    # 断言的是那句**确定性判断**没被说出来。不能断言 "没有技能类" 不在里面 ——
    # 「判断不了这个模块里有没有技能类」恰好含这个子串（第一版就这么写，当场红了）。
    assert "一处都不" not in w


def test_a_module_with_no_classes_is_not_undecidable(fake_src):
    """一个类都没有的模块是**确定**没有技能类，不是「判断不了」。"""
    (fake_src / "skills" / "builtins" / "consts.py").write_text(
        "THRESHOLD = 1.0\n", encoding="utf-8")
    sk, unk = E._skill_classes(
        (fake_src / "skills" / "builtins" / "consts.py").read_bytes(),
        "mast.skills.builtins.consts", source_root=fake_src)
    assert sk == set() and unk == set()


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------

def test_scan_catches_import_from_the_package_not_just_the_module(fake_src):
    """``from mast.skills.builtins import BetaSkill`` 同样绕过注册表。

    它的 ``node.module`` 是包名不是模块名 —— 只按精确模块名匹配会漏掉这一整类，
    而漏报（说「没有别的入口」其实有）正是这个功能最坏的失败形态。
    """
    data = (fake_src / "skills" / "builtins" / "probe.py").read_bytes()
    sk, _ = E._skill_classes(data, "mast.skills.builtins.probe", source_root=fake_src)
    sites = E.scan_direct_importers(
        "mast.skills.builtins.probe", source_root=fake_src,
        exported=E._toplevel_names(data, "mast.skills.builtins.probe"),
        skill_names=sk)
    kinds = {s.kind for s in sites}
    assert "from_package" in kinds, "从包导入的绑定被漏掉了"
    assert any(s.file == "agents/other.py" and "BetaSkill" in s.names for s in sites)


def test_scan_skips_the_package_init_that_re_exports(fake_src):
    """包 ``__init__`` 的 import 是结构性的，不是绕过注册表的绑定。"""
    data = (fake_src / "skills" / "builtins" / "probe.py").read_bytes()
    sk, _ = E._skill_classes(data, "mast.skills.builtins.probe", source_root=fake_src)
    sites = E.scan_direct_importers("mast.skills.builtins.probe", source_root=fake_src,
                                    exported=E._toplevel_names(data, "x"), skill_names=sk)
    assert not any(s.file.endswith("builtins/__init__.py") for s in sites)


def test_binds_skill_separates_skill_classes_from_constants(fake_src):
    """绑技能类和绑常量要分开标 —— 决定不同。

    实测：``mast.skills.base`` 有 111 处 import、``composite.graph_executor`` 40 处，
    全是基础设施。混在一起报给用户就等于没报。
    """
    data = (fake_src / "skills" / "builtins" / "probe.py").read_bytes()
    sk, _ = E._skill_classes(data, "mast.skills.builtins.probe", source_root=fake_src)
    sites = E.scan_direct_importers(
        "mast.skills.builtins.probe", source_root=fake_src,
        exported=E._toplevel_names(data, "x"), skill_names=sk)
    by_name = {s.names[0]: s for s in sites if s.names}
    assert by_name["AlphaSkill"].binds_skill is True
    assert by_name["THRESHOLD_PA"].binds_skill is False


# --------------------------------------------------------------------------
# 三种 warning —— 区别是**决定不同**，不是措辞不同
# --------------------------------------------------------------------------

def test_warning_for_a_helper_module_says_it_wont_take_effect_at_all(fake_src):
    """没有技能类的模块（``_tip_policy.py`` 那种）覆盖了也不生效。

    这是「改了没反应」里最难自己想明白的一种，所以必须说得最重。
    """
    w = E._warning("mast.skills.builtins.helpers", set(),
                   [E.ImportSite("a.py", 1, "from_module", ("f",), "", False)], set())
    assert "一处都不" in w and "发版本" in w


def test_warning_for_a_clean_module_says_it_takes_effect(fake_src):
    w = E._warning("m", {"AlphaSkill"}, [], set())
    assert "全线生效" in w


def test_warning_names_the_files_that_keep_running_the_builtin(fake_src):
    sites = [E.ImportSite("agents/tools.py", 1, "from_module", ("A",), "", True)]
    w = E._warning("m", {"A"}, sites, set())
    assert "agents/tools.py" in w


# --------------------------------------------------------------------------
# eject 本体
# --------------------------------------------------------------------------

def test_eject_writes_the_bytes_verbatim(fake_src):
    """字节原样 —— 不加文件头注释。

    加了之后「我改过没有」就要靠「减去头部再算 sha」这种脆逻辑；旁挂 sidecar 之后
    比对是一次裸 sha256。
    """
    r = E.eject("mast.skills.builtins.probe")
    assert r.ok, r.reason
    src = (fake_src / "skills" / "builtins" / "probe.py").read_bytes()
    assert P.entry_path("builtins/probe.py").read_bytes() == src


def test_eject_does_not_enable_it(fake_src):
    """文件在目录里 ≠ 生效。启用是一次显式动作。"""
    from mast.skills.overlay import manifest as M
    assert E.eject("mast.skills.builtins.probe").ok
    assert [e for e in M.load().entries if e.enabled] == []


def test_eject_refuses_a_non_skill_module(fake_src):
    r = E.eject("mast.core.registry")
    assert not r.ok and "发版本" in r.reason


def test_eject_refuses_to_clobber_local_edits_and_says_which_case(fake_src):
    """已存在时要分清「和内置相同」和「里面有改动」—— 后者覆盖会丢东西。"""
    assert E.eject("mast.skills.builtins.probe").ok
    same = E.eject("mast.skills.builtins.probe")
    assert not same.ok and "逐字节相同" in same.reason

    p = P.entry_path("builtins/probe.py")
    p.write_bytes(p.read_bytes() + "\n# 用户的改动\n".encode("utf-8"))
    edited = E.eject("mast.skills.builtins.probe")
    assert not edited.ok
    assert "已经有改动" in edited.reason and "备份" in edited.reason
    assert p.read_bytes().endswith("# 用户的改动\n".encode("utf-8")), "拒绝之后不能动那个文件"


def test_eject_overwrite_replaces_it(fake_src):
    assert E.eject("mast.skills.builtins.probe").ok
    P.entry_path("builtins/probe.py").write_bytes("# 改过\n".encode("utf-8"))
    r = E.eject("mast.skills.builtins.probe", overwrite=True)
    assert r.ok
    assert P.entry_path("builtins/probe.py").read_bytes() != "# 改过\n".encode("utf-8")


def test_eject_writes_a_sidecar_with_the_origin_sha(fake_src):
    r = E.eject("mast.skills.builtins.probe")
    sc = json.loads(P.sidecar_path("builtins/probe.py").read_text(encoding="utf-8"))
    assert sc["overlay_of"] == "mast.skills.builtins.probe"
    assert sc["orig_sha256"] == r.sha256
    assert sc["app_version"], "版本号是空的 —— 兜底值合理得让人看不出兜底发生了"
    assert sc["skill_classes"] == ["AlphaSkill", "BetaSkill"]


def test_eject_result_carries_the_binds_count(fake_src):
    r = E.eject("mast.skills.builtins.probe")
    assert r.n_skill_binds >= 2      # AlphaSkill（模块）+ BetaSkill（包）
    assert r.warning and "内置版" in r.warning


# --------------------------------------------------------------------------
# drift —— 覆盖生效了，但它基于的是三个版本前的代码
# --------------------------------------------------------------------------

def test_drift_detects_the_builtin_changing_underneath(fake_src):
    assert E.eject("mast.skills.builtins.probe").ok
    d = E.drift("builtins/probe.py")
    assert d.known and d.drifted is False

    src = fake_src / "skills" / "builtins" / "probe.py"
    src.write_text(src.read_text(encoding="utf-8") + "\n# 上游修了个 bug\n",
                   encoding="utf-8")
    d2 = E.drift("builtins/probe.py")
    assert d2.drifted is True
    assert "盖回去" in d2.reason


def test_drift_without_a_sidecar_is_unknown_not_false(fake_src):
    """手写的覆盖也合法 —— 但那意味着**无从知道**它基于哪一版。

    报 ``False`` 就是把「不知道」答成了「没变」。
    """
    P.overlay_dir(create=True)
    p = P.entry_path("builtins/handwritten.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# 手写的\n", encoding="utf-8")
    d = E.drift("builtins/handwritten.py")
    assert d.known is False
    assert d.drifted is None, "「不知道」被答成了「没变」"


def test_drift_reports_unknown_when_the_builtin_is_gone(fake_src):
    assert E.eject("mast.skills.builtins.probe").ok
    (fake_src / "skills" / "builtins" / "probe.py").unlink()
    d = E.drift("builtins/probe.py")
    assert d.drifted is None and "已经没有" in d.reason


# --------------------------------------------------------------------------
# 真源码树上的对账 —— 断言方向，不断言数字
# --------------------------------------------------------------------------

def test_every_registered_skill_class_is_recognised_by_the_ast_judge():
    """AST 判据不能漏掉任何一个**真的注册了的**技能类。

    这是这条判据的有效性证明，而且方向是要紧的那个：漏判会让报告说出
    「覆盖它一处都不会生效」，而事实相反。反方向（AST 多认几个）无害 ——
    实测多出来的就是 ``CompositeSkillGraph`` / ``_TipComposite`` 这些抽象基类，
    而继承它们的技能确实覆盖不到，所以标出来反而更准。

    不断言具体数字：那会随别人加技能而红，只会教人去改测试。
    """
    from mast.core.registry import SkillRegistry
    from mast.pyexec import _srcfiles

    root = _srcfiles.source_root()
    seen: set[str] = set()
    for d in E.list_ejectable(source_root=root):
        src = _srcfiles.resolve(d["dotted"])
        if src is None:
            continue
        sk, _unk = E._skill_classes(src.read_bytes(), d["dotted"], source_root=root)
        seen |= sk

    reg = SkillRegistry()
    reg.discover()
    registered = {cls.__name__
                  for versions in reg._skills.values()
                  for cls in versions.values()}
    missed = registered - seen
    assert not missed, "AST 判据漏了这些已注册的技能类：" + ", ".join(sorted(missed)[:10])


def test_list_ejectable_covers_the_builtins_and_excludes_package_inits():
    from mast.pyexec import _srcfiles
    lst = E.list_ejectable(source_root=_srcfiles.source_root())
    dotted = {d["dotted"] for d in lst}
    assert "mast.skills.builtins.bias" in dotted
    assert not any(d["rel"].endswith("__init__.py") for d in lst)
    assert not any(d["dotted"].startswith("mast.skills._overlay") for d in lst)


# --------------------------------------------------------------------------
# 结构闸门：给 UI 的文案不能带 Markdown 强调标记
# --------------------------------------------------------------------------

def test_ui_facing_strings_carry_no_literal_markdown():
    """后端返回的文案会被界面**逐字**显示 —— 里面的 ``**`` 就是两个星号。

    这条是实机截图抓出来的：导出结果那张告警上明晃晃写着
    「覆盖这个模块**一处都不会生效**」。源码里用 ``**`` 强调读起来很顺，所以这个
    错误会一写再写 —— 加一道闸比记住可靠。

    日志字符串豁免：日志是给读源码的人看的，本仓既有风格就用 ``**``。
    """
    import ast
    from pathlib import Path

    from mast.skills.overlay import paths as _P

    root = Path(_P.__file__).parent
    mast_pkg = Path(_P.__file__).parents[2]          # …/mast
    files = sorted(root.glob("*.py")) + [
        mast_pkg / "api" / "routes" / "skill_overlay.py",
        mast_pkg / "update" / "skillpack.py",
        mast_pkg / "update" / "skillpack_client.py",
    ]

    # ⚠️ 文件不在 = **这条测试失败**，不是跳过。
    #
    # 第一版写的是 parents[2]（少数了一层，指到 MASTv2/api/routes/，不存在），
    # 配上一个 `if not f.is_file(): continue` —— 于是路由那个文件从来没被扫过，
    # 而测试一直是绿的。实测：往路由里写一句带 ** 的文案，它一声不吭地通过了。
    # 一个「看着在防护其实没有」的守卫，比没有守卫更坏，因为它让人不再去看。
    missing = [str(f) for f in files if not f.is_file()]
    assert not missing, (
        "扫描名单里的文件不存在（多半是路径写错或文件改名了）——"
        "这条测试因此什么都没检查：\n  " + "\n  ".join(missing))

    offenders: list[str] = []
    for f in files:
        tree = ast.parse(f.read_bytes(), filename=str(f))
        exempt: set[int] = set()
        for node in ast.walk(tree):
            # docstring
            body = getattr(node, "body", None)
            if (isinstance(body, list) and body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                exempt.add(id(body[0].value))
            # logger.xxx(...) 的参数
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "logger"):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        exempt.add(id(sub))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in exempt and "**" in node.value):
                offenders.append("%s:%d %s" % (f.name, node.lineno,
                                               node.value.replace("\n", " ")[:60]))
    assert not offenders, (
        "这些字符串会被界面逐字显示，里面的 ** 会原样出现在屏幕上：\n  "
        + "\n  ".join(offenders))
