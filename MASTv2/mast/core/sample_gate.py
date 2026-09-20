"""样品门控 —— 产数据的操作必须有一个归属的样品。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §11

用户的要求是硬的：**没有选定样品时，扫描/谱学这类会产生数据的操作应当被拦下
并要求先建/选样品。** 理由不是洁癖：一条不知道属于哪块样品的 .sxm，事后没有
任何人能定位它测的是什么，等于白测。

同时同样硬的另一条：**安全操作绝不能因为没选样品被挡住。** 急停、退针、
一切 Stop*、所有只读查询，在任何情况下都必须畅通。

★ 为什么这里 fail-open，而 instrument_lock.needs_token 是 fail-closed
--------------------------------------------------------------------

两者形状同构（都是"把哪些技能算数收敛到一处"），但**兜底方向刻意相反**：

* ``needs_token`` 分类不出来时**假定要锁**。它的失败模式是两条链路同时驱动
  同一台 Nanonis —— 那是**危险**，宁可多锁一次。
* 本门控分类不出来时**放行**。它的失败模式是一条记录没归属到样品 —— 那是
  **记账损失**，不是危险。而误拦一个安全操作可能让用户在针要撞上去的时候
  按不动按钮。

**歧义必须朝放行解。** 第 1–3 步是保险带，第 5 步是背带。

对话不在拦截范围内
------------------

用户明确要求只拦产数据的操作：跟 agent 讨论「这个实验该怎么做」发生在建样品
之前，拦掉它会造成鸡生蛋。没有样品时开的对话归实验级，选了样品之后开的归样品级
—— 这个分级在 ``api/routes/agents.py`` 里已经自动成立（``sample_id`` 为 NULL
即实验级），本模块不参与。
"""

from __future__ import annotations

from typing import Any

#: 无论如何都放行的技能名。这些是"补救"动作：它们存在的意义就是在出事时
#: 立刻能跑。⊇ instrument_lock.BYPASS_NAMES。
GATE_EXEMPT_NAMES: frozenset[str] = frozenset({
    "StopScan", "StopSTS", "StopMotor", "StopAutoApproach", "StopFolMe",
    "SafeRetract", "EmergencyRetract", "WithdrawTip",
    "AutoApproachClose", "StopAutoApproachAndWithdraw",
})

#: 带这些标签的技能一律放行（补救 / 安全 / 只读 / 状态查询）。
GATE_EXEMPT_TAGS: frozenset[str] = frozenset({
    "retract", "emergency", "withdraw", "safety", "stop", "read", "status",
    "diagnostic", "connection",
})

#: 带这些标签的技能会产生数据文件或数据记录 → 必须有样品。
DATA_TAGS: frozenset[str] = frozenset({
    "scan", "spectroscopy", "sweep", "grid", "datalog", "record",
    "manipulation", "lithography", "point_shoot", "pattern", "spectrum",
    "acquisition", "imaging",
})

#: 名字层面的兜底（有些技能标签不全，但它们确定会产数据）。
DATA_NAMES: frozenset[str] = frozenset({
    "StartScan", "SaveScan", "ScanOnce", "AcquireSTS", "BiasSpectroscopy",
    "GridSpectroscopy", "StartDataLog", "StartTcpLog", "TipPulse",
    "TipShape", "BiasPulse",
})


def requires_sample(meta: Any, skill_name: str = "") -> bool:
    """这个技能是否要求有活跃样品才能跑。

    判定顺序（前三步是保险带，最后一步是背带 —— 见模块 docstring）::

        1. 名字在豁免表           → False
        2. 标签命中豁免标签        → False
        3. category ∈ READ|ANALYSIS → False
        4. 标签/名字命中产数据集合  → True
        5. 其它                    → False（fail-open）
    """
    name = str(skill_name or getattr(meta, "name", "") or "")
    if name in GATE_EXEMPT_NAMES:
        return False

    tags = {str(t).lower() for t in (getattr(meta, "tags", None) or ())}
    if tags & GATE_EXEMPT_TAGS:
        return False

    category = getattr(getattr(meta, "category", None), "value", None)
    if category is None:
        category = getattr(meta, "category", None)
    if str(category).lower() in ("read", "analysis", "skillcategory.read",
                                 "skillcategory.analysis"):
        return False

    if tags & DATA_TAGS:
        return True
    if name in DATA_NAMES:
        return True

    capabilities = {str(c).lower() for c in (getattr(meta, "capabilities", None) or ())}
    if capabilities & {"tip_shaping", "bias_pulse", "produces_file"}:
        return True

    # 分类不出来 → 放行。见模块 docstring 关于 fail-open 方向的说明。
    return False


def sample_gate_message(skill_name: str, *, has_experiment: bool) -> str:
    """拒绝文案。**给人看的和给模型看的是同一句。**

    风格对齐 ``safety_mw`` 的 ``[safety_gate]`` 与 ``InstrumentBusy.message()``：
    说清发生了什么、为什么、有哪些出路，并明确告诉模型**不要原样重试** ——
    否则它会陷进重试循环而不是去问用户。
    """
    if not has_experiment:
        return (
            f"[sample_gate] no_active_experiment: 当前没有进行中的实验，"
            f"'{skill_name}' 未执行——扫描/谱学产生的数据必须归属到一个实验和样品，"
            f"否则以后没人能定位这条记录测的是什么。\n"
            f"出路：(1) 调用 start_experiment(\"实验名\") 开一个实验，再 "
            f"start_sample(\"样品名\")；(2) 请用户在右栏「实验 → 样品」里选择。\n"
            f"Do NOT retry this call unchanged — 在实验和样品选定前它会以完全"
            f"相同的方式失败。"
        )
    return (
        f"[sample_gate] no_active_sample: 当前没有选定样品，'{skill_name}' 未执行"
        f"——扫描/谱学产生的数据必须归属到一个样品，否则以后没人能定位这条记录"
        f"属于哪块样品。\n"
        f"出路：(1) 调用 start_sample(\"样品名\") 新建一个样品；"
        f"(2) 请用户在右栏「实验 → 样品」里选一个已有样品；"
        f"(3) 若不确定用哪个样品，先问用户——不要重试本次调用。\n"
        f"Do NOT retry this call unchanged — 在样品选定前它会以完全相同的方式失败。"
    )


def check_sample_scope(meta: Any, skill_name: str = "",
                       experiment_log: Any = None) -> str | None:
    """放行返回 ``None``，否则返回中文拒绝文案。

    ``experiment_log`` 为 None（无头 / 测试 / 记录系统未接线）时**一律放行** ——
    门控是记账约束，不该在记录系统本身缺席时把仪器锁死。
    """
    if experiment_log is None:
        return None
    if not requires_sample(meta, skill_name):
        return None
    try:
        eid = getattr(experiment_log, "current_experiment_id", None)
        sid = getattr(experiment_log, "current_sample_id", None)
    except Exception:  # noqa: BLE001 — 读指针失败不该把仪器锁死
        return None
    if sid:
        return None
    name = str(skill_name or getattr(meta, "name", "") or "(unknown)")
    return sample_gate_message(name, has_experiment=bool(eid))
