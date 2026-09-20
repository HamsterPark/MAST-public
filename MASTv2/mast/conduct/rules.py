"""闸门求值 —— **三态**判据树 + 路由映射。

设计:``campaign_director_design.md`` §4.4(GateSpec)、§6-4(闸门在步/阶段边界
消费)、§10-1(「读不到」不是一个值)、§10-5(fail-open 方向:默认保守)。

## 为什么是三态

一条判据 ``n_keep >= 1`` 有三种结果,不是两种:

* **成立** —— 证据在,比较为真;
* **不成立** —— 证据在,比较为假;
* **判不了** —— 证据**不在**,或者类型对不上(拿字符串和数比大小)。

把第三种折叠进第二种,就是把「没测出来」说成「测出来是零」。本仓一天出现过
五次这个形状。所以这里的求值器返回 :data:`TRUE` / :data:`FALSE` /
:data:`UNDECIDABLE`,而**「判不了」的去向由 spec 显式声明**
(``GateSpec.evidence_missing`` / ``unattended_escape``),不由求值器替它决定。

## 谁不在这里

* **证据收集**(按 source/selector/max_age_s/min_epoch 过滤)在 Director 里 ——
  它要读 store 和温度口,不是纯函数;
* **llm 判决**只在这里做**路由→裁决的映射**,``decide_route`` 本身注入
  (它自己已有测试,而且是 LLM 调用,不该在纯模块里)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from mast.conduct.spec import GateSpec, RuleLeaf, RuleTree

logger = logging.getLogger(__name__)

#: 判据树的三态。
TRUE = "true"
FALSE = "false"
UNDECIDABLE = "undecidable"

TRISTATE = (TRUE, FALSE, UNDECIDABLE)

#: 证据里取不到的标记。用一个哨兵对象而不是 ``None`` —— ``None`` 是一个**合法
#: 读数**(「这一项没有值」),和「这一项根本不在证据包里」是两件事。
_ABSENT = object()


@dataclass(frozen=True)
class GateResult:
    """一次闸门判定的完整答案。"""

    #: ``GATE_VERDICTS`` 之一。
    verdict: str
    #: 走的哪条路由(rule 闸门是 "pass"/"fail";证据缺席时为 "")。
    route: str = ""
    #: 人读理由,进事件日志。
    reason: str = ""
    #: 判据树的三态结果(llm 闸门为 None)。
    rule_state: "str | None" = None
    #: 是否走了 escape(判不了)。
    escaped: bool = False
    #: 是否因为证据缺席而没走到判据。
    evidence_missing: bool = False
    #: 唤过 LLM 没有(Director 据此记 llm_wakes)。
    llm_used: bool = False
    #: 这次判决的审计块(哪个模型答的、怎么解析出来的、逃没逃)。
    #:
    #: rule 闸门为 ``None``。**它要往下走到人眼前**:Director 把它放进
    #: ``gate_evaluated`` 的 payload,于是面板闸门史与 ``progress.jsonl`` 上
    #: 「这一条是模型判的、是哪个模型判的」是可读的,而不是与一条 rule 判定
    #: 长得一模一样。
    llm_audit: "dict | None" = None


def lookup(evidence: Mapping[str, Any], field: str) -> Any:
    """按点分路径取值;取不到返回哨兵(**不是 None**)。

    ``None`` 是一个合法读数(「这一项没有值」),和「这一项不在证据包里」是两件
    要做不同事的事 —— 前者可能是判据说不清,后者是证据根本没送到。
    """
    cur: Any = evidence
    for part in field.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return _ABSENT
    return cur


def _compare(op: str, left: Any, right: Any) -> str:
    if left is None and right is not None:
        # **``None`` 是产出方在说「这一项我没测出来」,不是一个可以比大小的值。**
        #
        # 走到这里意味着字段**在**证据包里(``lookup`` 没回哨兵),而它的值是
        # 「没有值」。三态字段是真实存在的形状 —— ``PreScanCheck.tip_ready``
        # 就是 ``True/False/None``,其中 ``None`` 逐字写着 inconclusive。
        # 折叠成 ``FALSE`` 的后果不是判错一次,是**把「这一帧没看清」说成
        # 「针坏了」**,而那句话会把用户送去换样品。
        #
        # ``right is None`` 时不拦:一条 ``field == None`` 的判据问的正是
        # 「它有没有值」,那是一个合法而且确定的问题。
        return UNDECIDABLE
    try:
        if op == "<=":
            return TRUE if left <= right else FALSE
        if op == ">=":
            return TRUE if left >= right else FALSE
        if op == "==":
            return TRUE if left == right else FALSE
        if op == "in":
            return TRUE if left in right else FALSE
    except TypeError:
        # 拿字符串和数比大小 —— 这不是「不成立」,是**判不了**。
        # 折叠成 False 会让一条坏掉的判据看起来像一次正常的否决。
        return UNDECIDABLE
    return UNDECIDABLE


def evaluate(rule, evidence: Mapping[str, Any]) -> str:
    """判据树 → :data:`TRUE` / :data:`FALSE` / :data:`UNDECIDABLE`。

    组合规则(短路方向是**保守**的):

    * ``all``:有一个假 ⇒ 假;否则有一个判不了 ⇒ 判不了;否则真。
    * ``any``:有一个真 ⇒ 真;否则有一个判不了 ⇒ 判不了;否则假。
    * ``not``:判不了 ⇒ 判不了(取反一个不知道还是不知道)。

    注意 ``all`` 的顺序:**先看假再看判不了**。一条确定为假的子判据足以否决
    整棵树,这时「另一条判不了」不改变结论 —— 而反过来把它报成判不了,会让一次
    本该干脆的否决变成一次要人来看的悬案。
    """
    if isinstance(rule, RuleLeaf):
        value = lookup(evidence, rule.field)
        if rule.op == "exists":
            # exists 是唯一一个「缺席」也能定论的算子 —— 那正是它存在的理由。
            return FALSE if value is _ABSENT else TRUE
        if value is _ABSENT:
            return UNDECIDABLE
        return _compare(rule.op, value, rule.value)

    if isinstance(rule, RuleTree):
        states = [evaluate(k, evidence) for k in rule.children]
        if rule.op == "not":
            inner = states[0]
            if inner == UNDECIDABLE:
                return UNDECIDABLE
            return FALSE if inner == TRUE else TRUE
        if rule.op == "all":
            if FALSE in states:
                return FALSE
            return UNDECIDABLE if UNDECIDABLE in states else TRUE
        if rule.op == "any":
            if TRUE in states:
                return TRUE
            return UNDECIDABLE if UNDECIDABLE in states else FALSE

    # 到不了这里(spec 的闭集拦在前面)。真到了就是判不了,不是通过。
    logger.warning("未知判据节点,按判不了处理: %r", rule)
    return UNDECIDABLE


def _uncertain_verdict(gate: GateSpec, attended: bool) -> tuple[str, str]:
    """判不了时的去向 + 理由。

    有人值守 ⇒ 请人来看(``wait_operator``);无人值守 ⇒ 按 spec 声明的
    ``unattended_escape``。**两条都不许是 pass**(spec 的 ``__post_init__``
    已经把 pass 从这两个位置上排除)。
    """
    if attended:
        return "wait_operator", "判不了,有人值守 ⇒ 请用户看一眼"
    return gate.unattended_escape, "判不了,无人值守 ⇒ 走 spec 声明的保守去向"


def evaluate_gate(gate: GateSpec, evidence: Mapping[str, Any], *,
                  attended: bool, missing: "tuple[str, ...]" = (),
                  decide_route: "Callable[[dict, dict], dict] | None" = None,
                  no_judge_reason: str = "llm 判决器未接入",
                  ) -> GateResult:
    """把一份证据判成一个裁决。

    ``missing`` 是 Director 收集证据时**收不到**的那几项(过期、跨代次、源不
    可用)。非空 ⇒ 直接走 ``gate.evidence_missing`` —— 结构过滤的意义就在这里:
    闸门根本看不到那些证据,所以不可能拿它们判。

    ``no_judge_reason`` 是 ``decide_route is None`` 那一支写进事件与面板的**那
    句话**。默认是「未接入」,而调用方常常知道一个更具体的原因(最典型:本阶段
    的 LLM 唤醒预算用完了)。去向完全一样,但说出来的话不一样 —— 一句「判决器
    未接入」印在一台明明接好了判决器的机器的面板上,会把用户送去查一根没有
    断的线。
    """
    if missing:
        return GateResult(
            verdict=gate.evidence_missing, route="",
            reason=f"证据缺席: {', '.join(missing)}",
            evidence_missing=True)

    if gate.kind == "rule":
        state = evaluate(gate.rule, evidence)
        if state == UNDECIDABLE:
            verdict, why = _uncertain_verdict(gate, attended)
            return GateResult(verdict=verdict, route="", reason=why,
                              rule_state=state, escaped=True)
        route = "pass" if state == TRUE else "fail"
        outcome = gate.routes[route]
        return GateResult(verdict=outcome.verdict, route=route,
                          reason=outcome.note or f"rule={state}",
                          rule_state=state)

    # ── llm 闸门 ────────────────────────────────────────────────────────
    if decide_route is None:
        # 判决器没接上(或本阶段唤醒预算用完)⇒ **判不了**,不是通过。
        verdict, _why = _uncertain_verdict(gate, attended)
        return GateResult(verdict=verdict, route="",
                          reason=f"{no_judge_reason} ⇒ 判不了", escaped=True)
    try:
        decision = decide_route(dict(gate.llm_node or {}), dict(evidence)) or {}
    except Exception as exc:  # noqa: BLE001 —— 判决失败绝不弄崩 conduct
        logger.warning("llm 闸门 %s 调用失败: %s", gate.gate_id, exc)
        verdict, _why = _uncertain_verdict(gate, attended)
        return GateResult(verdict=verdict, route="",
                          reason=f"llm 判决调用失败 ⇒ 判不了: {exc}",
                          escaped=True, llm_used=True,
                          llm_audit={"unavailable": str(exc)})

    route = str(decision.get("route") or "")
    escaped = bool(decision.get("escaped"))
    reason = str(decision.get("reason") or "")
    audit = _audit_of(decision)
    outcome = gate.routes.get(route)
    if outcome is None:
        # decide_route 保证返回值必是 node 的某条路由,而 spec 保证每条路由都有
        # 裁决 —— 走到这里说明两个保证之间被人改开了。判不了,不是通过。
        verdict, _why = _uncertain_verdict(gate, attended)
        return GateResult(verdict=verdict, route=route,
                          reason=f"路由 {route!r} 没有裁决映射 ⇒ 判不了",
                          escaped=True, llm_used=True, llm_audit=audit)
    if escaped and not attended:
        # 无人值守时 uncertain 不问人(问了也必超时),直接走保守分支。
        return GateResult(verdict=gate.unattended_escape, route=route,
                          reason=f"{reason}(无人值守 ⇒ 强制保守去向)",
                          escaped=True, llm_used=True, llm_audit=audit)
    return GateResult(verdict=outcome.verdict, route=route,
                      reason=reason or outcome.note, escaped=escaped,
                      llm_used=True, llm_audit=audit)


#: 一次判决里**给人看**的那几项。整份决策不进事件 payload —— 里面的 inputs
#: 快照可能很大,而事件流是要逐行读的。
_AUDIT_KEYS = ("model", "parse_path", "escaped", "escape_reason", "duration_ms")


def _audit_of(decision: Mapping[str, Any]) -> "dict | None":
    out = {k: decision[k] for k in _AUDIT_KEYS if k in decision}
    return out or None


__all__ = ["TRUE", "FALSE", "UNDECIDABLE", "TRISTATE", "GateResult",
           "lookup", "evaluate", "evaluate_gate"]
