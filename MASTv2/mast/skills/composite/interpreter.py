"""SpecComposite — generic interpreter that runs a declarative CompositeSpec.

One class executes ANY :class:`~mast.skills.composite.spec.CompositeSpec`. It is
a :class:`CompositeSkillGraph` subclass, so it reuses the whole existing engine:
the :class:`GraphExecutor` walks the (now data-driven) plan, emits
``CompositeProgress`` per step, checkpoints, and supports resume — no per-skill
Python.

Control flow:
  * ``plan_dynamic`` builds a variable context from the spec defaults + caller
    params, then walks the node tree, evaluating ``if`` conditions and ``loop``
    bounds with the safe evaluator and yielding a concrete :class:`CompositeStep`
    for each ``step`` node.
  * After each step runs, its ``result.data`` is bound back into the context
    under the node's ``id`` (if any) and under ``last``, so a later ``if`` /
    ``set`` / param expression can reference earlier measurements.

Step ids are made stable for resume by prefixing nested/looped steps with their
path + iteration index (e.g. ``scan_loop#3/measure``), so a re-invocation skips
already-completed steps deterministically.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeStep, GraphExecutor
from mast.skills.composite.spec import (
    DEFAULT_MAX_ITER,
    CompositeSpec,
    resolve_params,
    safe_eval,
)

logger = logging.getLogger(__name__)


class _Break(Exception):
    """P5 control signal: break out of the nearest enclosing loop."""


class _Continue(Exception):
    """P5 control signal: continue the nearest enclosing loop."""


class _Return(Exception):
    """P5 control signal: end the whole composite with an explicit verdict
    (a ``succeed`` / ``fail`` node). Caught in ``plan_dynamic``; the verdict
    drives ``_decide_outcome``."""

    def __init__(self, success: bool, reason: str = "") -> None:
        super().__init__(reason)
        self.success = success
        self.reason = reason


_SAFETY = {"auto": SafetyLevel.AUTO, "confirm": SafetyLevel.CONFIRM,
           "dangerous": SafetyLevel.DANGEROUS}

#: 安全级的严格程度序。用于「取子步骤中最严的那个」。
_SAFETY_RANK = {SafetyLevel.AUTO: 0, SafetyLevel.CONFIRM: 1,
                SafetyLevel.DANGEROUS: 2}


def _inherited_safety_level(spec: CompositeSpec, registry) -> SafetyLevel:
    """spec 的实际安全级 = max(声明值, 所有子步骤技能的安全级)。

    没有这条继承规则时,``safety_level`` 是纯手填的:一个内含 DANGEROUS 步骤的
    工作流可以把自己声明成 ``auto``,于是 HITL 门控(由 safety_level 单一驱动)
    整个绕过去 —— 而且是静默的,因为工作流构建器上那个下拉框看起来完全正常。

    **声明只能收紧,不能放松**:声明 dangerous 的工作流即便全是 AUTO 步骤也保持
    dangerous(用户认为它危险,那就是危险);声明 auto 但含 CONFIRM/DANGEROUS
    步骤的,按子步骤提级。

    查表是**惰性**的:spec 的加载可能早于它引用的技能注册完毕(loader 在
    registry.discover 之后跑,但模板 seeding 与自定义 spec 的顺序不保证),所以
    在 metadata() 首次被调用时才算,算不出来就保持声明值 —— 查不到技能时不能
    假装它是安全的,但也不能凭空拒绝整个 spec(那会让一次注册顺序问题变成技能
    消失)。这里选择保持声明值并记 warning:真正的兜底是 loader 的
    ``_missing_step_skills``,它会拒绝引用了不存在技能的 spec。
    """
    declared = _SAFETY.get(spec.safety_level, SafetyLevel.CONFIRM)
    if registry is None:
        return declared
    try:
        from mast.skills.composite.spec import collect_step_skills

        highest = declared
        for name in sorted(collect_step_skills(getattr(spec, "nodes", None))):
            try:
                sub_meta = registry._get_metadata(registry.get(name))
                sub_level = getattr(sub_meta, "safety_level", None)
            except Exception:      # 技能不在册 —— 交给 loader 的缺失检查处理
                continue
            if sub_level is None:
                continue
            if _SAFETY_RANK.get(sub_level, 1) > _SAFETY_RANK.get(highest, 1):
                highest = sub_level
        if highest is not declared:
            logger.info(
                "工作流 '%s' 声明为 %s,但含更高安全级的子步骤,按 %s 执行",
                spec.name, declared.value, highest.value,
            )
        return highest
    except Exception as exc:  # noqa: BLE001 - 推导失败不能让技能不可用
        logger.debug("safety 继承推导失败(保持声明值): %s", exc)
        return declared


def _inherited_capabilities(spec: CompositeSpec, registry) -> frozenset:
    """spec 的能力标签 = 所有子步骤技能能力标签的并集。

    与 :func:`_inherited_safety_level` 同一条道理，但**堵的是一个更硬的洞**：
    ``safety_level`` 决定要不要人工确认，``capabilities`` 决定 SAFE / SEMI 操作模式
    **拒不拒绝执行**。而 :func:`mast.core.safety.is_electrical_pulse` /
    :func:`~mast.core.safety.is_tip_shaping` 是**纯能力查表** —— 声明式 spec 此前
    完全不填这个字段，于是 ``caps = frozenset()`` ⇒ 一律 False ⇒ **SAFE 模式对
    声明式工作流形同虚设**：一个内含 ``TipPulse`` 的工作流（模板里就有
    ``ConditionTipUntilSharp``、``ScanThenConditionTip``，用户在技能构建器里也随时
    能拼一个）在 SAFE 下会被放行，真的打出脉冲。手写的修针技能全都老老实实声明了
    标签，唯独这条路是空的（2026-08-01 审计）。

    **只并集、不删减**：能力是「这个工作流会做什么」的事实陈述，子步骤会打脉冲，
    包着它的工作流就会打脉冲。spec 本身没有声明能力的字段，所以这里不存在
    「声明只能收紧」的问题。

    与安全级继承同样是**惰性 + 失败保持**：查不到的技能交给 loader 的
    ``_missing_step_skills`` 兜底，这里不能因为一次注册顺序问题就让技能不可用。
    """
    if registry is None:
        return frozenset()
    try:
        from mast.skills.composite.spec import collect_step_skills

        caps: set[str] = set()
        for name in sorted(collect_step_skills(getattr(spec, "nodes", None))):
            try:
                sub_meta = registry._get_metadata(registry.get(name))
            except Exception:      # 技能不在册 —— 交给 loader 的缺失检查处理
                continue
            caps |= set(getattr(sub_meta, "capabilities", None) or ())
        if caps:
            logger.info("工作流 '%s' 继承子步骤能力标签: %s",
                        spec.name, ", ".join(sorted(caps)))
        return frozenset(caps)
    except Exception as exc:  # noqa: BLE001 - 推导失败不能让技能不可用
        logger.debug("capabilities 继承推导失败(按无能力处理): %s", exc)
        return frozenset()


class _SafeMap(dict):
    """format_map 容错：缺键原样保留 {key}，模板错误不毁工作流。"""
    def __missing__(self, key):
        return "{" + str(key) + "}"


def _safe_format(template: str, values: dict) -> str:
    try:
        return template.format_map(_SafeMap(values))
    except Exception:  # pragma: no cover — 任意花括号畸形模板
        return template


def human_channel(payload: dict):
    """human 节点的 HITL 通道（P2-G，可在测试/其他宿主中整体替换）。

    默认实现 = LangGraph ``interrupt(payload)``：
      * 在 agent 路径的工具调用内：首跑抛 GraphInterrupt 冒泡暂停整图
        （必须任其传播——graph_executor/execution_context/skill_adapter 三处
        已加控制流豁免）；resume 重放时返回用户决议。
      * 在图运行时之外（GUI 手动路径/裸测试）：langgraph 抛非 interrupt
        异常 → 返回 None，调用方据此明确失败。"""
    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import interrupt
    except ImportError:  # pragma: no cover — langgraph pinned in v2
        return None
    try:
        return interrupt(payload)
    except GraphInterrupt:
        raise                      # 控制流，绝不吞
    except Exception as exc:  # noqa: BLE001 — 图运行时之外
        logger.debug("human_channel outside graph runtime: %s", exc)
        return None
_PTYPE = {"number": "float", "int": "int", "string": "str", "bool": "bool",
          "float": "float", "str": "str"}


class SpecComposite(CompositeSkillGraph):
    """Runs a :class:`CompositeSpec`. Instantiate with a spec instance."""

    #: 注册这个 spec 的 registry(由 ``make_spec_skill`` 绑到子类上)。安全级
    #: 继承要查子步骤技能的元数据,而 registry 没有全局单例。None = 独立构造
    #: (测试 / 预览),此时不做继承推导,保持声明值。
    _registry_ref = None
    #: 推导结果缓存 —— metadata() 在每次工具列表构建、每次 HITL map 推导、每次
    #: ctx.run 审批检查时都会被调用,不该每次重走一遍 registry。
    _safety_cache: "SafetyLevel | None" = None
    #: 同上，能力标签的推导缓存。``None`` = 尚未推导（``frozenset()`` 是合法结果，
    #: 所以不能用它当「未推导」的哨兵）。
    _caps_cache: "frozenset | None" = None

    def __init__(self, spec: CompositeSpec) -> None:
        super().__init__()
        self._spec = spec

    def _effective_safety_level(self) -> SafetyLevel:
        cls = type(self)
        cached = cls.__dict__.get("_safety_cache")
        if cached is not None:
            return cached
        level = _inherited_safety_level(self._spec, cls._registry_ref)
        cls._safety_cache = level
        return level

    def _effective_capabilities(self) -> frozenset:
        cls = type(self)
        cached = cls.__dict__.get("_caps_cache")
        if cached is not None:
            return cached
        caps = _inherited_capabilities(self._spec, cls._registry_ref)
        cls._caps_cache = caps
        return caps

    # -- metadata derived from the spec --
    def metadata(self) -> SkillMetadata:
        s = self._spec
        params = [
            ParameterSpec(
                name=p.name,
                type=_PTYPE.get(p.type, "str"),
                description=p.description,
                required=p.required,
                default=p.default,
            )
            for p in s.params
        ]
        return SkillMetadata(
            name=s.name,
            version=f"{s.version}.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=self._effective_safety_level(),
            description=s.description or f"Declarative composite '{s.name}'",
            parameters=params,
            estimated_duration_s=s.estimated_duration_s or 0.0,
            composition_level=3,
            tags=list(s.tags) + ["composite", "spec"],
            # Without this the SAFE/SEMI operating-mode gate cannot see a
            # declarative workflow's tip processing at all — see
            # _inherited_capabilities.
            capabilities=self._effective_capabilities(),
        )

    def aggregate(self, sub_results: dict, progress) -> dict:
        """P4: 在默认聚合之上，对声明式 outputs 签名求值（safe_eval against
        最终走查 ctx）。单个输出求值失败不毁结果——置 None 并记入
        ``output_errors``（resume 跳步导致引用缺绑定时尤其如此）。"""
        data = super().aggregate(sub_results, progress)
        outs = self._spec.outputs or []
        if outs:
            ctx = getattr(self, "_walk_ctx", None) or {}
            rendered: dict[str, Any] = {}
            errors: dict[str, str] = {}
            for o in outs:
                nm = str(o.get("name", ""))
                if not nm:
                    continue
                try:
                    rendered[nm] = safe_eval(str(o.get("expr", "")), ctx)
                except Exception as exc:  # noqa: BLE001
                    rendered[nm] = None
                    errors[nm] = str(exc)
            data["outputs"] = rendered
            if errors:
                data["output_errors"] = errors
        return data

    def _decide_outcome(self, all_good: bool, progress, data: dict):
        """P5 final ok/fail. Precedence: hard external abort → explicit
        succeed/fail verdict node → ``success_when`` expr → default (all steps
        ran without abort)."""
        if progress.aborted:
            return False, progress.aborted_reason or "composite aborted"
        ctx = getattr(self, "_walk_ctx", None) or {}
        verdict = getattr(self, "_verdict", None)
        if verdict is not None:
            ok, reason = verdict
            return (True, "") if ok else (False, reason or "composite failed")
        sw = (self._spec.success_when or "").strip()
        if sw:
            try:
                ok = bool(safe_eval(sw, ctx))
            except Exception as exc:  # noqa: BLE001 — bad expr ⇒ fail closed
                return False, f"success_when eval error: {exc}"
            if ok:
                return True, ""
            reason = "composite verdict: success_when is false"
            fm = (self._spec.fail_message or "").strip()
            if fm:
                try:
                    reason = str(safe_eval(fm, ctx))
                except Exception:  # noqa: BLE001
                    pass
            return False, reason
        if all_good:
            return True, ""
        return False, progress.aborted_reason or "composite aborted"

    # -- the plan is the spec tree, walked with conditionals/loops --
    def plan_dynamic(self, params: dict, executor: GraphExecutor) -> Iterator[CompositeStep]:
        ctx: dict[str, Any] = dict(self._spec.default_context())
        ctx.update(params or {})
        ctx["params"] = dict(ctx)   # also expose the whole param map as `params`
        # P4: aggregate() 在工作流结束后对 outputs 表达式求值 —— 捕获在走查中
        # 被持续就地更新的同一个 ctx dict。
        self._walk_ctx = ctx
        # P5 控制流：_opt_depth>0（在 try 区域内）时步骤强制 optional（不中止整图，
        # 保证 finally 必达）；succeed/fail 节点抛 _Return 在此接住、定最终裁决。
        self._opt_depth = 0
        self._verdict = None
        try:
            yield from self._walk(self._spec.nodes, ctx, executor, prefix="")
        except _Return as r:
            self._verdict = (bool(r.success), str(r.reason))
        except (_Break, _Continue):
            logger.warning("SpecComposite %s: break/continue outside a loop "
                           "— ignored", self._spec.name)

    def _walk(self, nodes: list[dict], ctx: dict, executor: GraphExecutor,
              prefix: str) -> Iterator[CompositeStep]:
        for i, node in enumerate(nodes):
            ntype = node.get("type")
            nid = node.get("id") or f"n{i}"
            if ntype == "step":
                step_id = f"{prefix}{nid}"
                resolved = resolve_params(node.get("params"), ctx)
                yield CompositeStep(
                    step_id=step_id,
                    skill_name=node["skill"],
                    params=resolved,
                    # P5: a step inside a try-body never aborts the composite
                    # (so the finally region always runs); its failure is
                    # surfaced via the `_failed` marker below for the spec to
                    # branch on.
                    optional=bool(node.get("optional", False)) or self._opt_depth > 0,
                    checkpoint_after=bool(node.get("checkpoint_after", True)),
                    tags=tuple(node.get("tags") or ()),
                    skill_version=node.get("skill_version") or None,
                )
                # Bind the sub-skill result back into the context so later
                # conditions/params can reference it. (On resume, a skipped
                # step has no fresh result — leave prior binding / unset.)
                res = executor.sub_results.get(step_id)
                if res is not None:
                    data = getattr(res, "data", {}) or {}
                    ctx["last"] = data
                    if node.get("id"):
                        ctx[node["id"]] = data
                elif step_id in executor.progress.failed_steps:
                    # P5: an optional / try-body step FAILED. Bind a marker so a
                    # later `if "_failed" in <id>` can react (DemoScanAndSTS:
                    # skip STS when the move failed; ShapeTip: retry on a failed
                    # plunge). safe_eval supports `in`, so this is checkable.
                    marker = {"_failed": True}
                    ctx["last"] = marker
                    if node.get("id"):
                        ctx[node["id"]] = marker

            elif ntype == "if":
                cond = bool(self._eval(node.get("cond"), ctx, f"{prefix}{nid}.cond"))
                branch = node.get("then") if cond else node.get("else")
                yield from self._walk(branch or [], ctx, executor,
                                      prefix=f"{prefix}{nid}/{'t' if cond else 'f'}/")

            elif ntype == "loop":
                yield from self._walk_loop(node, ctx, executor, prefix, nid)

            elif ntype == "set":
                ctx[node["var"]] = self._eval(node.get("value"), ctx, f"{prefix}{nid}.value")

            elif ntype == "llm":
                yield from self._walk_llm(node, ctx, executor, prefix, nid)

            elif ntype == "human":
                yield from self._walk_human(node, ctx, executor, prefix, nid)

            elif ntype == "agent":
                yield from self._walk_agent(node, ctx, executor, prefix, nid)

            elif ntype == "try":
                # P5: try/finally. The body runs non-aborting (steps forced
                # optional via _opt_depth); the finally region runs on normal
                # completion AND when the body exits early via break / continue /
                # succeed / fail (those propagate THROUGH the Python finally,
                # which legally yields because the generator is still running).
                # External HARD abort still stops the run before finally —
                # faithful to the hand-written composites (E-stop is a hard
                # stop; EmergencyRetract is the safety path).
                self._opt_depth += 1
                try:
                    _hard_abort = False
                    try:
                        yield from self._walk(node.get("body") or [], ctx,
                                              executor, prefix=f"{prefix}{nid}/try/")
                    except GeneratorExit:
                        # HARD external abort (E-STOP): the executor stopped
                        # consuming us, so the finally region CANNOT run (Python
                        # forbids yielding during GeneratorExit). The tip's safety
                        # is handled by the hardware E-STOP path (watchdog
                        # SafeRetract / EmergencyRetract), NOT this finally — make
                        # the skipped cleanup OBSERVABLE rather than silent
                        #.
                        _hard_abort = True
                        if node.get("finally"):
                            logger.warning(
                                "SpecComposite %s: try '%s' finally region SKIPPED "
                                "by hard abort — soft cleanup did not run; hardware "
                                "safety is handled by the E-STOP retract path.",
                                self._spec.name, nid)
                        raise
                    finally:
                        # Normal completion AND soft exits (break/continue/succeed/
                        # fail raise ordinary exceptions that pass through here and
                        # legally yield) run the finally region. Only a hard
                        # GeneratorExit skips it (can't yield during close()).
                        if not _hard_abort:
                            yield from self._walk(node.get("finally") or [], ctx,
                                                  executor, prefix=f"{prefix}{nid}/fin/")
                finally:
                    self._opt_depth -= 1

            elif ntype == "break":
                raise _Break()

            elif ntype == "continue":
                raise _Continue()

            elif ntype in ("succeed", "fail"):
                reason = ""
                if node.get("reason"):
                    reason = str(self._eval(node.get("reason"), ctx,
                                            f"{prefix}{nid}.reason"))
                raise _Return(success=(ntype == "succeed"), reason=reason)

            else:
                logger.warning("SpecComposite %s: unknown node type %r — skipped",
                               self._spec.name, ntype)

    def _walk_loop(self, node: dict, ctx: dict, executor: GraphExecutor,
                   prefix: str, nid: str) -> Iterator[CompositeStep]:
        mode = node.get("mode")
        var = node.get("var") or "i"
        body = node.get("body") or []
        max_iter = int(node.get("max_iter", DEFAULT_MAX_ITER))

        if mode == "repeat":
            count = int(self._eval(node.get("count"), ctx, f"{prefix}{nid}.count"))
            count = max(0, min(count, max_iter))
            for it in range(count):
                ctx[var] = it
                try:
                    yield from self._walk(body, ctx, executor, prefix=f"{prefix}{nid}#{it}/")
                except _Continue:
                    continue
                except _Break:
                    break

        elif mode == "foreach":
            iterable = self._eval(node.get("iterable"), ctx, f"{prefix}{nid}.iterable")
            try:
                items = list(iterable)
            except TypeError:
                logger.warning("SpecComposite %s: foreach iterable is not iterable",
                               self._spec.name)
                return
            for it, item in enumerate(items[:max_iter]):
                ctx[var] = item
                ctx[f"{var}_index"] = it
                try:
                    yield from self._walk(body, ctx, executor, prefix=f"{prefix}{nid}#{it}/")
                except _Continue:
                    continue
                except _Break:
                    break

        elif mode == "while":
            it = 0
            while it < max_iter and bool(self._eval(node.get("cond"), ctx,
                                                    f"{prefix}{nid}.cond[{it}]")):
                ctx[var] = it
                try:
                    yield from self._walk(body, ctx, executor, prefix=f"{prefix}{nid}#{it}/")
                except _Continue:
                    it += 1
                    continue
                except _Break:
                    break
                it += 1
            if it >= max_iter:
                logger.warning("SpecComposite %s: while loop hit max_iter=%d",
                               self._spec.name, max_iter)

    def _walk_llm(self, node: dict, ctx: dict, executor: GraphExecutor,
                  prefix: str, nid: str) -> Iterator[CompositeStep]:
        """Bounded LLM decision (P2-B). route 模式恰好点亮一个命名槽位；data
        模式把类型化结果绑进 ctx（失败走 on_error 槽）。

        RESUME 确定性：决策缓存进 ``progress.partial_data['_decisions']``
        （随 CompositeProgress 持久化）——恢复重放走到同一个决策点时复用
        原选择，绝不重新问 LLM（否则旧分支已完成的步骤会与新选择错乱）。
        循环内的决策点 key 含迭代序号前缀，每轮各自决策。"""
        from mast.skills.composite import llm_node as _lln

        mode = node.get("mode", "route")
        dkey = f"{prefix}{nid}"
        decisions = executor.progress.partial_data.setdefault("_decisions", {})
        cached = decisions.get(dkey)
        if cached is not None:
            decision = cached
        else:
            inputs = resolve_params(node.get("inputs"), ctx) or {}
            if mode == "route":
                decision = _lln.decide_route(node, inputs)
            else:
                decision = _lln.decide_data(node, inputs)
            decisions[dkey] = decision
            _lln.log_decision({
                "ts": time.time(),
                "workflow": self._spec.name,
                "workflow_version": self._spec.version,
                "node_id": node.get("id"),
                "decision_key": dkey,
                "mode": mode,
                "mechanism": "llm",
                "persona": node.get("persona"),
                "responsibility": node.get("responsibility"),
                "inputs": inputs,
                **decision,
            })

        if mode == "route":
            chosen = decision.get("route")
            bind = {"route": chosen, "reason": decision.get("reason", ""),
                    "escaped": bool(decision.get("escaped", False))}
            ctx["last"] = bind
            if node.get("id"):
                ctx[node["id"]] = bind
            branch = (node.get("routes") or {}).get(chosen) or []
            yield from self._walk(branch, ctx, executor,
                                  prefix=f"{prefix}{nid}/{chosen}/")
        else:
            if decision.get("ok"):
                data = dict(decision.get("data") or {})
                ctx["last"] = data
                if node.get("id"):
                    ctx[node["id"]] = data
            else:
                err = {"error": decision.get("escape_reason", "llm data failed"),
                       "escaped": True}
                ctx["last"] = err
                if node.get("id"):
                    ctx[node["id"]] = err
                yield from self._walk(node.get("on_error") or [], ctx, executor,
                                      prefix=f"{prefix}{nid}/err/")

    def _walk_human(self, node: dict, ctx: dict, executor: GraphExecutor,
                    prefix: str, nid: str) -> Iterator[CompositeStep]:
        """human 节点（P2-G）：暂停工作流等用户决策。

        agent 路径上 = LangGraph ``interrupt()``（spike 证明可从深层调用栈
        冒泡；resume 时本工具整体重放，靠 sidecar + 决策缓存幂等续跑）。
        无 HITL 通道的路径（手动/测试）→ 明确失败而不是悄悄跳过。
        决策缓存进 ``partial_data['_human']``：重放/恢复绝不重复打扰用户。"""
        dkey = f"{prefix}{nid}"
        store = executor.progress.partial_data.setdefault("_human", {})
        cached = store.get(dkey)
        if cached is None:
            inputs = resolve_params(node.get("inputs"), ctx) or {}
            routes = list((node.get("routes") or {}).keys())
            msg = _safe_format(str(node.get("message", "")), inputs)
            payload = {
                "kind": "workflow_human", "workflow": self._spec.name,
                "node_id": nid, "decision_key": dkey, "message": msg,
                "inputs": inputs, "routes": routes or ["resolved"],
            }
            # CRITICAL：interrupt 会中止整个工具调用——先把已完成步骤与
            # llm 决策缓存落 sidecar，否则 resume 重放会重复执行仪器动作。
            executor.flush_sidecar()
            decision = human_channel(payload)
            if decision is None:
                raise RuntimeError(
                    f"human 节点 {nid!r} 需要 HITL 通道（agent 路径的"
                    " interrupt/approval pane）——当前执行路径没有人工决策"
                    "通道，无法继续")
            route = (decision.get("route") if isinstance(decision, dict)
                     else str(decision))
            if routes and route not in routes:
                raise RuntimeError(
                    f"human 节点 {nid!r} 收到不在选项内的决策 {route!r}"
                    f"（可选：{routes}）")
            cached = {"route": route or "resolved",
                      "note": (decision.get("note", "")
                               if isinstance(decision, dict) else "")}
            store[dkey] = cached
            executor.flush_sidecar()
        bind = {"route": cached["route"], "note": cached.get("note", "")}
        ctx["last"] = bind
        if node.get("id"):
            ctx[node["id"]] = bind
        slot = (node.get("routes") or {}).get(cached["route"])
        if slot:
            yield from self._walk(slot, ctx, executor,
                                  prefix=f"{prefix}{nid}/{cached['route']}/")

    def _walk_agent(self, node: dict, ctx: dict, executor: GraphExecutor,
                    prefix: str, nid: str) -> Iterator[CompositeStep]:
        """agent 节点（P3-B）：受预算约束的域 agent 委托。结果文本绑 ctx[nid]
        （{"text","ok","note"}），失败/超时走 on_error 槽。委托结果缓存进
        partial_data['_agents']（resume 不重跑），并记决策日志。"""
        from mast.skills.composite import agent_node as _an
        from mast.skills.composite import llm_node as _lln

        dkey = f"{prefix}{nid}"
        store = executor.progress.partial_data.setdefault("_agents", {})
        cached = store.get(dkey)
        if cached is None:
            inputs = resolve_params(node.get("inputs"), ctx) or {}
            task = _safe_format(str(node.get("task", "")), inputs)
            executor.flush_sidecar()        # 委托可能数分钟，先落进度
            cached = _an.run_agent_task(node, task)
            store[dkey] = cached
            executor.flush_sidecar()
            _lln.log_decision({
                "ts": time.time(), "workflow": self._spec.name,
                "workflow_version": self._spec.version,
                "node_id": node.get("id"), "decision_key": dkey,
                "mode": "delegate", "mechanism": "agent",
                "agent": node.get("agent"), "task": task[:500],
                "inputs": inputs, **{k: cached.get(k) for k in
                                     ("ok", "note", "duration_ms")},
            })
        bind = {"text": cached.get("text", ""), "ok": bool(cached.get("ok")),
                "note": cached.get("note", "")}
        ctx["last"] = bind
        if node.get("id"):
            ctx[node["id"]] = bind
        if not bind["ok"]:
            yield from self._walk(node.get("on_error") or [], ctx, executor,
                                  prefix=f"{prefix}{nid}/err/")

    def _eval(self, expr: Any, ctx: dict, where: str) -> Any:
        """Evaluate an expression, raising a clear error tagged with its location."""
        from mast.skills.composite.spec import ExprError
        try:
            return safe_eval(expr, ctx)
        except ExprError as exc:
            raise ExprError(f"{self._spec.name} @ {where}: {exc}") from exc


def make_spec_skill(spec: CompositeSpec, registry=None):
    """Return a *class* bound to *spec* (so SkillRegistry, which instantiates
    skill classes with no args, can register a spec-based composite).

    ``registry`` 是安全级继承要用的:一个内含 DANGEROUS 步骤的工作流不能靠把
    自己声明成 ``auto`` 就绕过 HITL(门控完全由 safety_level 驱动)。传 None 时
    保持声明值 —— 注册路径(``register_spec``)一定会传,那是安全性真正依赖的
    那条路。
    """
    class _Bound(SpecComposite):
        _registry_ref = registry
        # Both caches must be reset PER BOUND CLASS: the lookups walk this
        # spec's own steps, and `cls.__dict__.get` would otherwise find the
        # base class's value through inheritance and hand one spec's verdict
        # to another.
        _safety_cache = None
        _caps_cache = None

        def __init__(self) -> None:
            super().__init__(spec)
    _Bound.__name__ = f"Spec_{spec.name}"
    _Bound.__qualname__ = _Bound.__name__
    return _Bound


__all__ = ["SpecComposite", "make_spec_skill"]
