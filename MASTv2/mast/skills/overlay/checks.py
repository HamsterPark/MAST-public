"""覆盖只能**收紧**，不能放松 —— 以及为什么这里没有 import 黑名单。

只能收紧
========
这条原则不是新发明的：声明式 composite 早就有
（``skills/composite/interpreter.py:74-117`` 的 ``_inherited_safety_level``、
``:120-158`` 的 ``_inherited_capabilities``）。那段 docstring 把理由写透了 ——
``safety_level`` 单一驱动 HITL，一个内含 DANGEROUS 步骤的东西如果能把自己声明成
``auto``，审批门就整个绕过去了，**而且是静默的**。

覆盖层是同一件事的另一个入口，所以用同一条规则、**直接 import 同一张 rank 表**
（抄第二份迟早漂）。

比对必须 raw ↔ raw
==================
两边都用 ``SkillRegistry._get_metadata_raw()``（作者的声明），不是合并了管理员
覆写之后的包络。否则：管理员合法地把内置 SetBias 降到 CONFIRM（有审计、有 PIN、
有理由）→ 那个 CONFIRM 成了基线 → 一个同样声明 CONFIRM 的 overlay **合法通过**
→ 管理员事后撤销覆写，overlay 还在，SetBias 永久停在 CONFIRM。
两层各自看都讲得通，合起来把审批门洗掉了。

为什么**不做** import 黑名单
===========================
威胁模型先说清楚。要防的是：

* **误操作** —— 一个半保存、语法错的文件被 exec；一个忘了删的实验文件悄悄生效；
* **未经授权的东西悄悄跑起来** —— 尤其是「从网上来的字节，没人点过头」。

**不**要防：恶意管理员（他对这台机器本来就有完全控制权，装个 .pth 就完事），
也**不**假装这是沙箱 —— ``llm/skill_author.py:87-94`` 自己已经写明 deny-list
可绕过、只是纵深防御。

现役 custom 轨复用的那份 deny-list 禁 ``os`` / ``subprocess`` / ``sys`` /
``shutil`` / ``pathlib`` / ``importlib``。真实的内置技能基本都要用 pathlib 和
numpy —— 那道门会把它们**全部挡死**，覆盖层根本用不起来。而放宽到能跑真技能之后，
它挡不住任何有意的东西，只剩「我们做了检查」的错觉。**那比没有更坏。**

所以这里只做**关于正确性、不是关于安全性**的检查（``ast.parse`` 成功、至少一个
BaseSkill 子类、三集合、只能收紧）。真正的门是三道人的显式动作或密码学事实：
显式启用（丢进目录不生效）、admin PIN、网络来的必须验签。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


def _rank():
    """安全级的严格程度序。**直接用 interpreter 的那一份。**"""
    from mast.skills.composite.interpreter import _SAFETY_RANK
    return _SAFETY_RANK


@dataclass
class RelaxationReport:
    """一次「只能收紧」比对的全部结果。**查完再报，不短路** ——
    用户一次看到全部问题，比修一个跑一次快得多。"""

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self, rel: str) -> str:
        lines = [f"覆盖层 {rel} 被拒绝（安全画像放松）："]
        lines += [f"  · {p}" for p in self.problems]
        lines.append(
            "需要放宽包络请走「技能管理」里的管理员覆写（单独审计、PIN 门控），"
            "不要走覆盖层。")
        return "\n".join(lines)


def check_not_relaxed(new_meta, base_meta) -> RelaxationReport:
    """新声明相对基线**只能收紧**。``base_meta`` 为 None（纯新增）时无条件通过。"""
    rep = RelaxationReport()
    if base_meta is None:
        return rep

    rank = _rank()
    new_lvl = getattr(new_meta, "safety_level", None)
    base_lvl = getattr(base_meta, "safety_level", None)
    if new_lvl is not None and base_lvl is not None:
        if rank.get(new_lvl, 1) < rank.get(base_lvl, 1):
            rep.problems.append(
                f"{new_meta.name}: safety_level {base_lvl.value} → {new_lvl.value}"
                "（覆盖只能收紧，不能放松）")

    lost = set(getattr(base_meta, "capabilities", ()) or ()) - \
        set(getattr(new_meta, "capabilities", ()) or ())
    if lost:
        rep.problems.append(
            f"{new_meta.name}: 丢失能力标签 {'、'.join(sorted(lost))}"
            "（SAFE/SEMI 模式闸门依赖它）")

    base_params = {p.name: p for p in (getattr(base_meta, "parameters", ()) or ())}
    new_params = {p.name: p for p in (getattr(new_meta, "parameters", ()) or ())}
    for pname, bp in base_params.items():
        np_ = new_params.get(pname)
        if np_ is None:
            continue                      # 删参数是接口变化，由三集合检查那边管
        if bp.min_value is not None and (
                np_.min_value is None or np_.min_value < bp.min_value):
            rep.problems.append(
                f"{new_meta.name}.{pname}: 下界 {bp.min_value} → "
                f"{np_.min_value}（超出内置包络）")
        if bp.max_value is not None and (
                np_.max_value is None or np_.max_value > bp.max_value):
            rep.problems.append(
                f"{new_meta.name}.{pname}: 上界 {bp.max_value} → "
                f"{np_.max_value}（超出内置包络）")
        if bp.allowed_values is not None:
            new_allowed = np_.allowed_values
            if new_allowed is None or not set(new_allowed) <= set(bp.allowed_values):
                rep.problems.append(
                    f"{new_meta.name}.{pname}: 取值集合超出内置的 "
                    f"{bp.allowed_values}")
        if bp.required and not np_.required:
            rep.problems.append(
                f"{new_meta.name}.{pname}: 必填 → 选填（放松）")

    if getattr(base_meta, "rollback_skill", None) and not getattr(
            new_meta, "rollback_skill", None):
        rep.problems.append(
            f"{new_meta.name}: 去掉了回滚技能 "
            f"{base_meta.rollback_skill}（去掉回滚是放松）")
    return rep


@dataclass
class SyntaxReport:
    ok: bool = True
    error: str = ""


def check_parses(source: str, rel: str) -> SyntaxReport:
    """能不能 ``ast.parse``。

    这条**真有用**，而且它防的不是攻击：它把「app 半加载了一个坏技能」变成
    「覆盖层被拒，第 42 行语法错误」。编辑器写盘不是原子的，一个正在保存的文件
    被读到是常事。
    """
    try:
        ast.parse(source, filename=rel)
    except SyntaxError as exc:
        return SyntaxReport(
            ok=False,
            error=f"{rel} 第 {exc.lineno} 行语法错误：{exc.msg}")
    except ValueError as exc:      # 源码里有 NUL 之类
        return SyntaxReport(ok=False, error=f"{rel} 读不成源码：{exc}")
    return SyntaxReport()


@dataclass
class SetReport:
    """一个覆盖模块对它所替换的内置模块做了什么。"""

    replaced: set = field(default_factory=set)
    added: set = field(default_factory=set)
    dropped: set = field(default_factory=set)

    def describe(self) -> str:
        bits = []
        if self.replaced:
            bits.append(f"覆盖 {len(self.replaced)} 个：{'、'.join(sorted(self.replaced))}")
        if self.added:
            bits.append(f"新增 {len(self.added)} 个：{'、'.join(sorted(self.added))}")
        if self.dropped:
            bits.append(f"移除 {len(self.dropped)} 个：{'、'.join(sorted(self.dropped))}")
        return "；".join(bits) or "没有技能"


def diff_sets(base_names: set[str], new_names: set[str]) -> SetReport:
    return SetReport(
        replaced=set(base_names) & set(new_names),
        added=set(new_names) - set(base_names),
        dropped=set(base_names) - set(new_names),
    )


def check_no_silent_removal(sets: SetReport, allow_removals) -> list[str]:
    """从一个模块里删掉技能，默认让**整个模块**加载失败。

    反例（这就是默认拒绝的理由）：用户从 ``bias.py`` 里删掉一个类，99% 是误删
    或改名。而「静默保留内置版」正好是「以为生效其实没生效」—— 他会看到那个技能
    还在，以为覆盖没起作用，然后去改别的地方。

    逃生口在清单里显式写 ``"allow_removals": ["SetBiasRange"]``。
    """
    allowed = set(allow_removals or ())
    unexpected = sorted(sets.dropped - allowed)
    if not unexpected:
        return []
    return [
        f"这个模块不再提供 {'、'.join(unexpected)} —— 内置版里有。"
        "误删还是改名？确实要移除的话，在 overlay.json 里写 "
        f'"allow_removals": {sorted(unexpected)}。'
    ]


__all__ = ["RelaxationReport", "SetReport", "SyntaxReport", "check_no_silent_removal",
           "check_not_relaxed", "check_parses", "diff_sets"]
