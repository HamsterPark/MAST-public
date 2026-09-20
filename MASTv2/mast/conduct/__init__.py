"""``mast.conduct`` —— 多天实验 conduct 的**通用引擎**。

设计:``docs/v2/design/campaign_director_design.md``;分层铁律:
``docs/v2/design/stm_capability_vs_sample_layer.md``。

## 分层(这条比任何一个字段都重要)

本包分两层,边界是**样品名**:

* **引擎层**(``spec`` / ``store`` / ``validator`` 以及后续的 Director/API)——
  状态机、持久化、校验。**一个样品名都不许出现**,由结构测试
  ``tests/v2/unit/test_no_sample_names_in_generic_layer.py`` 强制。
* **模板层**(``templates/<sample>.py``)——某个具体实验的段落组合、闸门路由、
  等待文案。**全仓唯一允许出现样品名的代码位置**,而且模板里也**只许有组合与
  参数引用,不许有判据逻辑**(判据逻辑一进模板,第二个样品就要抄一份)。
  这条由 :func:`mast.conduct.validator.lint_template_source` 用 AST 强制。

「换样品」的定义因此是**换数据,不改引擎代码**。若某个新样品逼得你改
``mast/conduct/`` 里 ``templates/`` 之外的任何一行,那是引擎漏了一个参数,
按缺陷处理。

## 本模块的范围(M1-a)

``spec``(定义)+ ``store``(进度真源)+ ``validator``(approve 前的 lint)。
**不含** Director tick 循环、REST 路由、实验文件夹渲染 —— 那些是 M1-b/c,
它们消费这里的类型,不该另起一套。
"""

from mast.conduct.spec import (
    ConductBudget,
    ConductSpec,
    ConditionSpec,
    DetourPolicy,
    EvidenceSpec,
    GateOutcome,
    GateSpec,
    ParamSpec,
    RuleLeaf,
    RuleTree,
    StageFailPolicy,
    StageSpec,
    StepSpec,
    WaitSpec,
)
from mast.conduct.store import ConductStore
from mast.conduct.validator import (
    Finding,
    ValidationReport,
    check_params,
    lint_template_source,
    validate_spec,
)

__all__ = [
    "ConductBudget", "ConductSpec", "ConditionSpec", "DetourPolicy",
    "EvidenceSpec", "GateOutcome", "GateSpec", "ParamSpec", "RuleLeaf",
    "RuleTree", "StageFailPolicy", "StageSpec", "StepSpec", "WaitSpec",
    "ConductStore",
    "Finding", "ValidationReport", "check_params", "lint_template_source",
    "validate_spec",
]
