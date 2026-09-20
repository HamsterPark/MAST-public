"""实验文件夹里的 conduct 视图 —— ``spec_vNNN.md`` + ``progress.jsonl``。

设计:``campaign_director_design.md`` §4.8。落点是**实验文件夹**
(``core/experiment_paths.py`` 那套),纪律是 **INCREMENTAL-ONLY**:只追加、
不 finalize、不归档、任意时刻拔电都自洽。

## 它不是真源

真源是 SQLite(``conduct_events`` + ``conducts`` 行)。这里写的是**人读视图**
与**灾难兜底**:

* **恢复不读它。** 重启清算走 SQLite;如果哪天有人让恢复流程读 jsonl,那这份
  文件就变成了第二个真源,而两个真源迟早会不一致。
* 每行**自洽** —— 带 ts / conduct_id / kind / 状态 / 一句人话。半行(崩在写
  中途)读侧丢弃;丢一行的代价是审计少一条,不是状态错一个。

## 谁读它(§10-10:产物要有消费者)

* **人**:晨检对账逐行走查(设计 §9 的 M4 验收就是这么做的);
* **面板**:``GET /api/conducts/{id}`` 回一个 ``folder`` 块 —— 路径、快照文件名、
  已写行数。这样「文件夹里到底有没有在写」是可问的,而不是要人去翻磁盘。

## 拿不到实验文件夹怎么办

**如实报,不静默**:``JournalStatus.reason`` 说清是哪一种(没有实验归属 /
实验行不存在 / 写失败),面板照原样显示。绝不因此让 Director 的状态机停下 ——
写一份人读副本失败,不该中断一个正在动仪器的流程。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

#: 实验文件夹里 conduct 产物的子目录名。
CONDUCT_DIRNAME = "conduct"
PROGRESS_FILENAME = "progress.jsonl"

#: 为什么 ``ConductBudget.usd_max`` 今天拦不住任何东西 —— **单一真源**,
#: 人读快照与面板都印这一句(两处各写一份,迟早只有一份跟着现实走)。
#:
#: 这不是「还没接」,是**计价单位对不上**:账本按 provider 的原生币种记账
#: (``billing/pricing.py``:中国 provider 记 CNY,Anthropic 记 USD),而**用哪家
#: 由调用那一刻的回退链决定** —— 所以一份 conduct 的花销天然可能跨币种。
#: 要和一个 USD 上限比大小就得有汇率,而 ``usd_to_cny_rate()`` 在没有覆盖文件时
#: 返回**写死的 7.2**,账本自己把折算总额标成
#: 「labelled 折算,never written to the ledger」。
#:
#: 一个 approve 一份要跑三天的流程的人,应该知道那条上限是不是真的。
#:
#: ⚠️ 这句话里**不许出现具体模型名** —— 与 llm 闸门那一行同一条理由:真正花钱
#: 的是回退链当时选中的那家,approve 时印一个名字就是承诺这里保证不了的事。
#: ``test_the_snapshot_never_names_a_model_it_cannot_promise`` 钉着(它逮到过
#: 这句话的第一版)。
USD_MAX_NOT_ENFORCEABLE = (
    "花销按 provider 的原生币种实测(中国 provider 计 CNY,Anthropic 计 USD,"
    "用哪家由调用时的回退链决定),而这条上限是 USD;合并两者需要汇率,"
    "而本仓不自造汇率 —— 所以没有任何东西会因为它停下来。要让它生效,"
    "得先让上限的计价单位与账本一致(见设计 §11-9)")

#: 事件种类 → 一句人话的模板。**闭集之外的种类照样写**(用 kind 本身当摘要),
#: 因为「没写进 jsonl」比「摘要不好看」严重得多。
_SUMMARY: dict[str, str] = {
    "created": "建了一份 conduct(草稿)",
    "approved": "用户批准,参数已冻结",
    "adopted": "指挥线程接手,开始执行",
    "status_change": "状态变化",
    "step_started": "开始执行一步",
    "step_finished": "一步结束",
    "step_interrupted": "进程死在这一步里 —— 该步产出不可信,不重放",
    "gate_evaluated": "闸门判定",
    "decision_overridden": "**用户放行了这一次闸门判定**(闸门的裁决原样留着)",
    "wait_entered": "进入等待",
    "wait_ack": "人确认了",
    "wait_waived": "condition 闸由人提供证据(waive,留痕)",
    "wait_condition_met": "物理条件达成",
    "wait_released": "等待解除,继续",
    "detour_entered": "进入绕道(证据代次 +1,旧证据作废)",
    "detour_returned": "绕道结束,回到原处",
    "estop_seen": "急停闩挂着 —— 停在原地",
    "estop_cleared": "急停闩已清 —— 闩清不等于针没事,先自检",
    "escalation_started": "升级:唤一次只读诊断",
    "escalation_verdict": "升级的建议回来了",
    "recovery_item": "恢复自检的一项",
    "op_consumed": "执行了一条用户意图",
    "op_rejected": "拒绝了一条意图(留痕,不静默 no-op)",
    "budget_tick": "花销记账",
    "heartbeat_stall": "心跳停滞告警",
    "aborted": "conduct 已中止",
    "completed": "conduct 全部完成",
}


#: 恢复自检的裁决 → 人话。三个值三句话:``unreadable`` **不是** ``fail`` 的
#: 委婉说法,它要人去做的事不一样。
_RECOVERY_VERDICT = {
    "pass": "过了", "fail": "**没过**", "unreadable": "**读不到(≠通过)**",
    "blocked": "**接不上**",
}


def _summary_for(kind: str, payload: "dict | None") -> str:
    """那一行的一句人话。

    ``progress.jsonl`` 的用法是**逐行走查**(设计 §9 的 M4 晨检)。一条恢复自检
    记录的 summary 若只说「恢复自检的一项」,那一行就得靠人去读 payload 才知道
    是过了还是没过 —— 而通宵跑下来这样的行有六条,逐行走查会变成逐行展开 JSON。
    答案在 payload 里就把它抬到这一行来。
    """
    base = _SUMMARY.get(kind, kind)
    p = dict(payload or {})
    if kind == "recovery_item" and p.get("item"):
        verdict = _RECOVERY_VERDICT.get(str(p.get("verdict") or ""),
                                        str(p.get("verdict") or "?"))
        return f"恢复自检 {p['item']}:{verdict}"
    return base


@dataclass(frozen=True)
class JournalStatus:
    """文件夹这一侧现在是什么情况 —— 面板照这个显示。"""

    #: 目录路径(拿不到就是空串)。
    path: str = ""
    #: 最新一版人读快照的文件名(还没渲染就是空串)。
    spec_doc: str = ""
    #: ``progress.jsonl`` 已写的行数;读不到是 ``None``(**不是 0**)。
    progress_lines: "int | None" = None
    #: 为什么没有路径 / 上一次写失败在哪。空串 = 一切正常。
    reason: str = ""

    @property
    def wired(self) -> bool:
        return bool(self.path)


def default_folder_resolver(experiment_id: str) -> "Path | None":
    """``experiment_id`` → 实验文件夹。实验行不存在返回 ``None``。

    走 ``documents.paths.exp_dir_for`` —— 与计划文档、报告、扫描文件**同一个**
    解析器。在这里另写一份「project_root / experiments / …」的算法,会立刻变成
    第二个真源,而路径分家那类 bug 本仓已经付过账。
    """
    eid = str(experiment_id or "").strip()
    if not eid:
        return None
    try:
        from mast.documents.paths import exp_dir_for

        return exp_dir_for(eid, create=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("conduct folder resolve failed for %r: %s", eid, exc)
        return None


class ConductJournal:
    """一份 conduct 在实验文件夹里的那两个文件。

    ``folder_resolver`` **必须可注入**:测试全部走 ``tmp_path``,绝不碰真实
    experiments(测试污染真实数据在本仓发生过五次,每一次的入口都是一个善意的
    默认路径)。默认值是解析器**函数**而不是一个路径常量,所以「不传就写到真实
    实验文件夹」这条逃生门只对生产接线开着。
    """

    def __init__(self, conduct_id: str, experiment_id: str, *,
                 folder_resolver: "Callable[[str], Path | None] | None" = None):
        self.conduct_id = str(conduct_id)
        self.experiment_id = str(experiment_id)
        self._resolve = folder_resolver or default_folder_resolver
        self._last_error = ""

    # ── 目录 ────────────────────────────────────────────────────────

    def dir(self, *, create: bool = False) -> "Path | None":
        base = None
        try:
            base = self._resolve(self.experiment_id)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"实验文件夹解析失败: {exc}"
            return None
        if base is None:
            self._last_error = (f"实验 {self.experiment_id!r} 没有文件夹"
                                f"(实验行不存在?)—— 人读副本没有落点")
            return None
        d = Path(base) / CONDUCT_DIRNAME / self.conduct_id
        if create:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self._last_error = f"建目录失败: {exc}"
                return None
        return d

    def status(self) -> JournalStatus:
        d = self.dir(create=False)
        if d is None:
            return JournalStatus(reason=self._last_error or "没有实验文件夹")
        docs = sorted(p.name for p in d.glob("spec_v*.md")) if d.is_dir() else []
        lines: "int | None" = None
        p = d / PROGRESS_FILENAME
        if p.is_file():
            try:
                lines = sum(1 for _ in p.open("r", encoding="utf-8",
                                              errors="replace"))
            except OSError as exc:
                # 读不到行数 ⇒ **None,不是 0**。0 会被读成「一行都没写」,
                # 而这两件事要做的处置不同(一个是查磁盘,一个是等)。
                self._last_error = f"progress.jsonl 读不到行数: {exc}"
        elif d.is_dir():
            # 目录在、文件还没有 ⇒ 真的是 0 行。这一支是**答得上来**的,
            # 所以给 0 而不是 None —— 把「答得上来」也报成「读不到」,
            # 会让真正的读不到淹没在噪声里。
            lines = 0
        return JournalStatus(path=str(d), spec_doc=docs[-1] if docs else "",
                             progress_lines=lines, reason=self._last_error)

    # ── 人读快照 ────────────────────────────────────────────────────

    def render_spec(self, spec, params: dict) -> str:
        """approve 时渲染的那份快照。**内容变才发版** —— 见 :meth:`sync_spec`。"""
        return render_spec_markdown(spec, params, conduct_id=self.conduct_id,
                                    experiment_id=self.experiment_id)

    def sync_spec(self, spec, params: dict) -> str:
        """写一版 ``spec_vNNN.md``,返回文件名(没写成就是空串)。

        内容与**最新一版**逐字相同就不发新版:一份三天的 conduct 会被 approve
        一次、但恢复自检可能走很多轮,每轮发一版就是版本爆炸(计划文档那边同一个
        取舍,同一个理由)。
        """
        d = self.dir(create=True)
        if d is None:
            return ""
        text = self.render_spec(spec, params)
        existing = sorted(d.glob("spec_v*.md"))
        if existing:
            try:
                if existing[-1].read_text(encoding="utf-8") == text:
                    return existing[-1].name
            except OSError:
                pass    # 读不出来就当它不一样,发新版 —— 少写不如多写
        nxt = len(existing) + 1
        name = f"spec_v{nxt:03d}.md"
        try:
            tmp = d / (name + ".part")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, d / name)      # 原子替换:崩在写中途不留半个文件
        except OSError as exc:
            self._last_error = f"写 {name} 失败: {exc}"
            logger.warning("conduct %s spec 快照写失败: %s", self.conduct_id, exc)
            return ""
        return name

    # ── 进度流 ──────────────────────────────────────────────────────

    def append(self, *, kind: str, ts: str, status_after: str = "",
               stage_id: str = "", step_id: str = "", run_id: str = "",
               payload: "dict | None" = None, summary: str = "") -> bool:
        """追加一行。**永不抛** —— 写人读副本失败不该停住状态机。

        每行自洽:不看上一行也能读懂这一行。
        """
        d = self.dir(create=True)
        if d is None:
            return False
        row = {
            "ts": ts,
            "conduct_id": self.conduct_id,
            "kind": kind,
            "summary": summary or _summary_for(kind, payload),
        }
        if stage_id:
            row["stage_id"] = stage_id
        if step_id:
            row["step_id"] = step_id
        if run_id:
            row["run_id"] = run_id
        if status_after:
            row["status_after"] = status_after
        if payload:
            row["payload"] = payload
        try:
            path = d / PROGRESS_FILENAME
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                fh.flush()
            return True
        except OSError as exc:
            self._last_error = f"追加 progress.jsonl 失败: {exc}"
            logger.warning("conduct %s progress 追加失败: %s", self.conduct_id, exc)
            return False


# ── 渲染(纯函数,好测)──────────────────────────────────────────────

def render_spec_markdown(spec, params: dict, *, conduct_id: str = "",
                         experiment_id: str = "") -> str:
    """spec + 冻结的 params → 人读 Markdown。

    给的是**用户批的时候看的那一份**:阶段、步骤、闸门去向、等待条件、
    以及每一个参数的值与单位。数字全部来自 spec/params,这里一个都不算、
    一个都不猜。
    """
    lines: list[str] = []
    lines.append(f"# {spec.title}")
    lines.append("")
    lines.append(f"- 模板:`{spec.spec_id}` v{spec.spec_version}")
    if conduct_id:
        lines.append(f"- conduct:`{conduct_id}`")
    if experiment_id:
        lines.append(f"- 实验:`{experiment_id}`")
    lines.append(f"- 值守模式初值:{'有人值守' if spec.attended_default else '无人值守'}")
    lines.append(f"- 自检全过后免确认续跑:{'是' if spec.auto_resume_after_recovery else '否'}")
    b = spec.budgets
    lines.append(f"- 预算:每阶段 LLM 唤醒 ≤ {b.llm_wakes_per_stage_max} 次"
                 f"(**这一条真的在拦**,用完即判不了);"
                 f"tick {b.tick_interval_s:g} s,等待期 {b.wait_tick_interval_s:g} s")
    lines.append(f"- ⚠️ 花销上限 ${b.usd_max:.2f} —— **今天拦不住任何东西**。"
                 f"{USD_MAX_NOT_ENFORCEABLE}")
    d = spec.detour
    if d.target_stage:
        lines.append(f"- 绕道:目标 {d.target_stage},最多 "
                     f"{d.max_detours_per_conduct} 次,返回方式 {d.on_return}")
    else:
        lines.append("- 绕道:**无**(本模板没有修针段,任何 detour 去向都会被校验器拒绝)")
    # 恢复自检的接触档:**它会动仪器**,所以人批的时候必须看得见它。
    # 没有这几行的话,一份 spec 快照会完整描述所有会动针的步骤 —— 除了重启之后
    # 那两步。它们不在任何阶段里,于是下面按 stage 走的枚举一个字都印不出来。
    # (§10c 那条教训的同一个形状:引擎多认一个位置,所有枚举都要走一遍。)
    rec = getattr(spec, "recovery", None)
    if rec is not None and rec.tip_check:
        names = " → ".join(f"{s.step_id}({s.skill or s.analysis_fn})"
                           for s in rec.tip_check)
        lines.append(f"- 重启后针尖复验(A3,**会动仪器**):{names};"
                     f"判据 `{rec.tip_rule}`;判不了重跑 {rec.tip_retries} 次")
        lines.append("  - 判**针坏**不会自动进修针段 —— 停下来问人"
                     "(理由见 director._tip_bad_reason)")
    else:
        lines.append("- 重启后针尖复验(A3):**没有声明** ⇒ 一旦重启,自检会报"
                     "「读不到」并停下来问人(读不到 ≠ 通过)")
    lines.append("")

    lines.append("## 参数(approve 时冻结)")
    lines.append("")
    if spec.params_schema:
        lines.append("| 名字 | 值 | 单位 | 说明 |")
        lines.append("|---|---|---|---|")
        for p in spec.params_schema:
            val = params.get(p.name, p.default)
            shown = "(未填,用默认)" if val is None else f"`{val}`"
            lines.append(f"| {p.name} | {shown} | {p.unit or '—'} | {p.help or ''} |")
    else:
        lines.append("(这个模板没有可填参数)")
    extra = [k for k in (params or {}) if k not in {p.name for p in spec.params_schema}]
    if extra:
        lines.append("")
        lines.append(f"> ⚠️ 库里还存着模板没有声明的键:{sorted(extra)} —— "
                     f"它们不会被任何绑定读到。")
    lines.append("")

    for stage in spec.stages:
        lines.append(f"## {stage.stage_id} · {stage.title}")
        lines.append("")
        flags = [f"必做:{'是' if stage.mandatory else '否'}",
                 f"失败策略:重试 {stage.on_fail.max_retries} 次后 {stage.on_fail.then}"]
        if stage.capabilities:
            flags.append(f"声明的 capability:{sorted(stage.capabilities)}")
        lines.append("- " + ";".join(flags))
        if stage.entry_gate is not None:
            lines.append(f"- 入口闸门:{_gate_line(stage.entry_gate)}")
        if stage.exit_gate is not None:
            lines.append(f"- 出口闸门:{_gate_line(stage.exit_gate)}")
        lines.append("")
        lines.append("| # | 步 | 形态 | 做什么 |")
        lines.append("|---|---|---|---|")
        for i, step in enumerate(stage.all_steps):
            lines.append(f"| {i} | {step.step_id} | {step.kind} | {_step_line(step)} |")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> 这份文件是**人读快照**,不是真源(真源在 SQLite)。"
                 "进度看同目录的 `progress.jsonl`;恢复流程**不读**这两个文件。")
    lines.append("")
    return "\n".join(lines)


def _gate_line(gate) -> str:
    routes = ", ".join(f"{k}→{v.verdict}" for k, v in sorted(gate.routes.items()))
    line = (f"`{gate.gate_id}`({gate.kind});路由 {routes};"
            f"无人值守时判不了 → {gate.unattended_escape};"
            f"证据缺席 → {gate.evidence_missing}")
    return line + _llm_gate_line(gate)


def _llm_gate_line(gate) -> str:
    """llm 闸门**多说四句**,附在 ``_gate_line`` 后面。

    快照是用户 approve 之前唯一会读的东西,而一道 llm 闸门与一道 rule 闸门在
    上面那一行里长得几乎一样 —— 差别只有括号里那三个字母。可这两者要批的是完全
    不同的东西:一道 llm 闸门能发出 ``detour``(= 半夜叫醒用户换样品,
    以小时计),而**它凭什么这么判**这件事,只写在 ``llm_node.responsibility``
    里,今天在快照上一个字都不出现。

    **不印模型名**:真正答题的是调用那一刻 provider 回退链选中的那个模型
    (静默回退在本仓发生过),approve 时印一个名字就是在承诺一件这里保证不了的
    事。哪个模型真的答了这一题,逐条记在 ``progress.jsonl`` 的
    ``gate_evaluated.payload.llm.model`` 上。
    """
    if getattr(gate, "kind", "") != "llm":
        return ""
    node = dict(getattr(gate, "llm_node", None) or {})
    node_routes = list((node.get("routes") or {}).keys())
    escape = node.get("escape") or (node_routes[-1] if node_routes else "")
    descs = node.get("route_descriptions") or {}
    bits = []
    resp = str(node.get("responsibility") or "").strip()
    bits.append(f"**问模型的问题**:{resp}" if resp
                else "⚠️ **这道闸没写 responsibility** —— 模型不知道自己在判什么")
    if descs:
        bits.append("选项含义:" + "、".join(
            f"{r}={descs.get(r, '(没写)')}" for r in node_routes))
    if escape:
        bits.append(f"模型说「判不了」时走 {escape} → "
                    f"{gate.routes[escape].verdict if escape in gate.routes else '?'}")
    bits.append("模型只选一个路由名,**不填任何数值**;每条路由的动作参数在模板里写死")
    return ";" + ";".join(bits)


def _step_line(step) -> str:
    """一行人读的步描述。

    **段内闸门在这里统一附加**,不在下面各分支里各加一次:``wait`` 与
    ``analysis`` 是提前 return 的,漏掉它们的话,一道长在等待步或分析步上的闸门
    会在快照上完全不存在 —— 而快照正是用户 approve 之前唯一会读的东西。
    """
    line = _step_line_body(step)
    gate = getattr(step, "gate", None)
    if gate is None:
        return line
    return f"{line};**跑完立刻判一次**:{_gate_line(gate)}"


def _step_line_body(step) -> str:
    if step.kind == "wait":
        w = step.wait
        if w is None:
            return "**等待步没有 WaitSpec**(模板有问题)"
        bits = [f"等待({w.kind}):{w.message}"]
        if w.ack_required:
            bits.append("要人确认")
        if w.condition is not None:
            c = w.condition
            refs = c.refs
            # 绑到参数上的那几个:写引用,**不写占位值** —— 快照上印一个
            # 「400 K」而实际等的是人填的 5 K,那份快照就是在撒谎。
            thr = refs.get("value") or f"{c.value:g}"
            stale = refs.get("stale_after_s") or f"{c.stale_after_s:g} s"
            bits.append(f"条件 {c.signal} {c.op} {thr}")
            hold = refs.get("hold_s") or (f"{c.hold_s:g} s" if c.hold_s else "")
            if hold:
                bits.append(f"保持 {hold}")
            bits.append(f"读数超过 {stale} 算读不到")
        return ";".join(bits)
    if step.kind == "analysis":
        return f"纯函数 `{step.analysis_fn}`" + (
            f";产出 {list(step.produces)}" if step.produces else "")
    bits = [f"技能 `{step.skill}`"]
    if step.params:
        bits.append("参数 " + ", ".join(f"{k}={v!r}" for k, v in
                                        sorted(step.params.items())))
    if step.bindings:
        bits.append("绑定 " + ", ".join(f"{k}←{v}" for k, v in
                                        sorted(step.bindings.items())))
    if step.retries:
        bits.append(f"重试 {step.retries} 次")
    return ";".join(bits)


__all__ = ["ConductJournal", "JournalStatus", "render_spec_markdown",
           "default_folder_resolver", "CONDUCT_DIRNAME", "PROGRESS_FILENAME",
           "USD_MAX_NOT_ENFORCEABLE"]
