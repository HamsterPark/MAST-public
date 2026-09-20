"""实验模板注册表 —— **全仓唯一允许出现样品名的代码目录**。

分层见 ``docs/v2/design/stm_capability_vs_sample_layer.md`` §1:引擎在
``mast/conduct/`` 的其余模块里,一个样品名都不许有;某个具体实验的段落组合、
闸门路由、等待文案住在这里。

模板里也**只许有组合与参数引用,不许有判据逻辑** —— 由
:func:`mast.conduct.validator.lint_template_source` 用 AST 强制,而不是靠自觉。
理由:判据逻辑一进模板,第二个样品就要抄一份,而抄出来的第 N 份里通常只有一份
是对的。

本文件(注册表)属于引擎侧的接线,不受零逻辑 lint 约束 —— 但它也只做「名字 →
模板」这一件事。
"""

from mast.conduct.spec import ConductSpec
from mast.conduct.templates import _smoke, paper_frame

#: spec_id → 模板。加模板 = 在这里加一行 + 新建一个模块。
TEMPLATES: dict[str, ConductSpec] = {
    _smoke.SPEC.spec_id: _smoke.SPEC,
    paper_frame.SPEC.spec_id: paper_frame.SPEC,
}

#: spec_id → 定义它的模块(零逻辑 lint 要拿源码)。
TEMPLATE_MODULES = {
    _smoke.SPEC.spec_id: _smoke,
    paper_frame.SPEC.spec_id: paper_frame,
}


def get_template(spec_id: str) -> ConductSpec:
    """按 spec_id 取模板。取不到**抛异常并列出有哪些** —— 不返回 None。

    返回 None 的话,调用方一个 ``if not spec`` 就把「没这个模板」变成了
    「没有阶段」,然后 conduct 会以一份空 spec 的形态跑起来。
    """
    try:
        return TEMPLATES[spec_id]
    except KeyError:
        raise KeyError(
            f"没有模板 {spec_id!r};已注册的是 {sorted(TEMPLATES)}") from None


def list_templates() -> list[dict]:
    """给面板的下拉表。"""
    return [{"spec_id": s.spec_id, "spec_version": s.spec_version,
             "title": s.title, "stages": [st.stage_id for st in s.stages]}
            for s in TEMPLATES.values()]


__all__ = ["TEMPLATES", "TEMPLATE_MODULES", "get_template", "list_templates"]
