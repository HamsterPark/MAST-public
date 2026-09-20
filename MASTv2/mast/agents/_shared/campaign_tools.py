"""科研纲领（campaign）工具族 —— Campaign 层的读写面。

## 这一层是什么

四层里最上面那一层，回答的是「**为什么做**」：

| 层 | 问题 | 载体 |
|---|---|---|
| **Campaign** | 为什么做 | ``logging/v2`` 的 ``campaigns`` 表（假设/目标/谱系，周~月） |
| Plan | 打算做什么 | ``planning/plan_store.py`` |
| Conduct | 正在怎么做 | ``mast/conduct/`` |
| Experiment/Sample/Action | 做出来了什么 | ``logging/v2`` 三层 |

一份 campaign 的寿命以周计，而一个 run、一个 thread、一份 checkpoint 都以小时计。
所以它必须落在库里 —— 只活在对话里的科研目标，压缩一次就没了，而下一个 run 会
从零开始重新发明一个听起来差不多的目标。

## 为什么在 ``_shared/`` 而不是在 ``agents/research_director/``

两个理由，第二个才是硬的：

1. 别的 agent 也要**读**它。实验设计要知道这次设计是为了区分哪两个假设，文献要
   知道这次检索是为哪条假设服务的 —— 而 agent_boundary 钩子（不随仓） 禁止
   ``agents/A/*`` import ``agents/B/*``，能共享的只有 ``agents._shared.*``。
2. 放这里之后，**把它发给谁是编排器的一行决定**（``orchestrator/graph.py`` 的
   ``_shared(agent_name)``），不是一次跨包重构。2026-08-20 的权限裁决说得很清楚：
   把关在服务端，不在「谁看得见哪个工具」。今天只有 research_director 拿到这一族，
   那是一次可以随时改的接线，不是一道安全边界。

## 这一族**不做**的三件事

* **不碰仪器。** 这里没有任何执行面：没有 ExecutionContext、没有技能注册表、
  没有 Nanonis 句柄。campaign 层的产出是一个**委托**，具体做什么由 plan 层写，
  怎么做由 conduct 层管，真正动手的只有 instrument_control。
* **不自己写 SQL。** 全部经 :mod:`mast.logging.v2.repos` 的 ``CampaignRepo`` /
  ``ExperimentRepo`` / ``ClaimGraphRepo``。表结构的 CHECK 约束、HLC 时钟、ULID
  都在那一层，绕过它就等于在旁边建第二套规则。
* **不发明数字。** campaign 层不写偏压、不写 setpoint、不写扫描尺寸。它写的是
  「要区分什么」和「什么算成功」；具体参数由实验设计从文献和历史里取，超包络的
  值由服务端**原样拒绝**（不夹紧 —— 夹紧会让「填错了」看起来像「填对了」）。
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool

from mast.agents._shared.artifact_channel import ArtifactToolReturn, campaign_ref

logger = logging.getLogger(__name__)

__all__ = ["make_campaign_tools", "CAMPAIGN_TOOL_NAMES"]

#: 这一族的工具名，与 :func:`make_campaign_tools` 的返回一一对应，有测试钉着。
#: （名单说有、图上没有，是本仓反复踩到的形状。）
CAMPAIGN_TOOL_NAMES: tuple[str, ...] = (
    "campaign_list",
    "campaign_get",
    "campaign_create",
    "campaign_update",
    "campaign_request_plan",
    "campaign_experiments",
    "campaign_claims",
)

#: 摘要里每条实验/主张的截断长度。state 里的东西按 pointer 走，这里给的是
#: 「要不要去读全文」的判据，不是全文。
_SNIP = 240

#: 客户端按状态筛选时的取数窗口。campaign 是周~月尺度的对象，几百条已经是很多年，
#: 所以这个窗口在实践中就是「全部」；窗口用满时 ``campaign_list`` 会说出来，
#: 而不是把「没扫到」答成「没有」。
_SCAN_WINDOW = 500


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _snip(text: Any, n: int = _SNIP) -> str:
    s = str(text or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _repos():
    """拿 v2 repos 句柄；库开不出来时回 None。**惰性 import**。

    与 conduct_tools._service 同一个形状、同一个理由：工具模块在**建图时**被导入，
    那时 runtime 可能还没起，而 SQLite 的 schema 初始化不该挂在 import 上。

    路径用 :func:`~mast.agents._shared.data_paths.v2_experiment_db_path` 显式解析
    后再交给 ``open_store``，**不**调用裸 ``open_store()`` —— 后者的兜底是
    ``Path(".")``，也就是「进程是从哪个目录起来的」。见那个函数的 docstring。
    """
    try:
        from mast.agents._shared.data_paths import v2_experiment_db_path
        from mast.logging.v2.repos import build_repos
        from mast.logging.v2.storage import open_store

        return build_repos(open_store(v2_experiment_db_path()))
    except Exception as exc:  # noqa: BLE001
        logger.debug("v2 records 库打不开: %s", exc)
        return None


def _check_done_when(goal: dict) -> "dict | None":
    """校验 ``goal["done_when"]``。合法回 ``None``，非法回一份**给模型看的**拒绝。

    **整体拒绝，不做部分丢弃。** 丢掉 ``all`` 里一个合法性存疑的合取项，等于把
    目标悄悄改小 —— 于是这条纲领更早「达成」，于是它下面的 park 被错误地抑制
    唤醒。那是 fail-open 方向。而拒绝无害：没有 ``done_when`` 就是今天的行为。

    同一条纪律在本模块已经写着（``hypothesis_kind`` 不在闭集里直接拒、不替你
    改成别的），这里只是把它用到判据上。

    报文带**目录与示例**：模型在它写错的那个地方拿到闭集，而不是被一句「请从
    目录里选」劝说 —— 后者本仓记过四次，没有一次管用。
    """
    raw = goal.get("done_when")
    if raw is None:
        return None
    if raw == [] or raw == {} or raw == "":
        # 空判据 = **把判据清空**，不是「没提到它」。归一化会把它变成 None 写回
        # 去，于是「键在、值是 None」——而告警看的是「键在不在」，就漏了。
        # 这里直接抹掉这个键，让下游的「原来有、现在没有」判得出来。
        goal.pop("done_when", None)
        return None
    try:
        from mast.goals import normalise_done_when
        from mast.goals.spec import EXAMPLE, catalog_json, done_when_to_json
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"目标判据模块不可用，没有写: {exc}"}
    spec, errs = normalise_done_when(raw)
    if errs:
        return {"ok": False,
                "reason": f"done_when 有 {len(errs)} 条不合法 —— **没有写**："
                          + "；".join(errs),
                "problems": errs,
                "done_when_catalog": catalog_json(),
                "example": EXAMPLE}
    # 归一化之后写回去：库里存的是解析过的形状，读侧不必再猜写法。
    goal["done_when"] = done_when_to_json(spec)
    # ── 基线：判据被**写下来的那一刻**，世界是什么样 ─────────────────
    #
    # 没有它，``artifact_present`` 这类相对判据在 campaign 侧永远没有参照点。
    # 第一版就是这样：``_cmp_versions`` 把「没有基线」读成「当时是空的」，于是
    # 实验文件夹里任何一份历史 analysis 都让判据 done —— 记录页显示「已达成
    # 1/1」，唤醒调度器把这条纲领下**每一份** park 都关掉，agent 一次都不醒。
    #
    # 抓不到就**不写**（下游读到「没有基线」会判 unknown，走安全方向），
    # 而不是写一个空 dict 假装那一刻世界是空的。
    try:
        from mast.goals.sources import snapshot_baseline

        base = snapshot_baseline()
        if base is not None:
            goal["baseline"] = base
    except Exception:  # noqa: BLE001 —— 抓基线失败不该让建纲领整个失败
        logger.debug("campaign done_when 基线抓不到（判据会判不了，不会假达成）")
    return None


def _goal_history(campaign_id: str, limit: int = 8) -> list:
    """这条纲领的判据走势（最近几条 ``campaign.goal_check`` 事件）。**永不抛。**

    事件是 ``publish_goal_check`` 在每次 run 收尾时落的，同值合并 —— 所以这串
    东西天然是「判据结论**变化**的历史」，而不是一堆重复行。
    """
    cid = str(campaign_id or "").strip()
    if not cid:
        return []
    repos = _repos()
    if repos is None:
        return []
    try:
        rows = repos.events.by_topic("campaign.goal_check", limit=200)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows:
        try:
            payload = r.get("payload") if isinstance(r.get("payload"), dict) else \
                json.loads(r.get("payload_json") or "{}")
        except Exception:  # noqa: BLE001
            continue
        if str((payload or {}).get("campaign_id") or "") != cid:
            continue
        out.append({"at": r.get("created_at") or r.get("hlc") or "",
                    "verdict": (payload or {}).get("verdict"),
                    "satisfied": (payload or {}).get("satisfied"),
                    "total": (payload or {}).get("total")})
    return out[-limit:]


def publish_goal_check(campaign_id: str) -> dict:
    """一次 run 收尾时，把这条纲领的判据现状落进 v2 ``events``。**永不抛。**

    为什么要落一条事件而不是每次现算：现算答的是「**此刻**怎么看」，而用户
    事后要问的常常是「它是**什么时候**变成满足的 / 那天为什么还没满足」——
    那要一条带时间戳的序列，而 ``campaigns`` 行上只有最新状态。

    **同值合并**靠 ``dedup_key``（纲领 + 结论 + x/y + 判据指纹）：一条纲领跑十次
    run 而判据没变，只留一行；判据翻转或被改写，才落新行。没有它，这张表会被
    「还是没满足」灌满，而那种表没人会去读。

    ``campaign_id`` 为空 ⇒ 什么都不做。**不猜**：一次没有纲领的 run 就是没有
    纲领，替它挑一条最近的会把 A 的进度记到 B 头上。
    """
    cid = str(campaign_id or "").strip()
    if not cid:
        return {"ok": False, "reason": "没有 campaign_id"}
    repos = _repos()
    if repos is None:
        return {"ok": False, "reason": "v2 记录库打不开"}
    try:
        row = repos.campaigns.get(cid)
        if not row:
            return {"ok": False, "reason": f"没有 {cid} 这一份纲领"}
        prog = _goal_progress(row)
        fp = ""
        try:
            from mast.goals import fingerprint, normalise_done_when

            goal = json.loads(row.get("goal_json") or "{}") or {}
            spec, _errs = normalise_done_when(
                goal.get("done_when") if isinstance(goal, dict) else None)
            fp = fingerprint(spec)
        except Exception:  # noqa: BLE001
            fp = ""
        repos.events.publish(
            topic="campaign.goal_check", kind="snapshot", producer="goals",
            dedup_key=(f"{cid}:{prog.get('verdict')}:"
                       f"{prog.get('satisfied')}/{prog.get('total')}:{fp}"),
            payload={"campaign_id": cid, **prog})
        return {"ok": True, **prog}
    except Exception as exc:  # noqa: BLE001 —— 留痕坏了绝不能让一次 run 报失败
        logger.debug("campaign goal_check 落不下去: %s", exc)
        return {"ok": False, "reason": str(exc)}


def _done_when_brief(row: dict) -> str:
    """一行人读的判据摘要，进 ``CampaignRef``。**带定义不带结论。**

    一次求值的结果只会过期（下游看到的是几跳之前的状态），而定义在这条纲领的
    一生里基本不变 —— 同一个理由，完整的 ``goal_json`` 也不进 state。
    """
    try:
        from mast.goals import describe_done_when, normalise_done_when

        goal = json.loads(row.get("goal_json") or "{}") or {}
        spec, errs = normalise_done_when(goal.get("done_when")
                                         if isinstance(goal, dict) else None)
        return "" if errs else describe_done_when(spec)
    except Exception:  # noqa: BLE001 —— 摘要坏了不许拦住产物交接
        return ""


def _goal_progress(row: dict) -> dict:
    """这份纲领的判据这一刻怎么看。**读即求值**，永不抛。

    RD 拿到的是一个**事实**（x/y 满足、还差什么），不是一个要它自己去判的问题。
    它据此可以 ``campaign_update(status="completed")`` —— 那是一次记账动作，
    有 ``created_by`` 和这一轮对话可查。

    **状态不自动翻。** campaign 是周~月尺度：「谓词全真」≠「科学问题答完」。
    自动翻的失败模式是 RD 见到 completed 就另起一份纲领（提示词里明令避免的
    重复纲领），而不自动翻的代价只是多一次记账。
    """
    try:
        from mast.goals import evaluate_done_when, normalise_done_when
        from mast.goals.sources import make_collector
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unknown", "reason": f"判据模块不可用: {exc}"}
    try:
        raw = row.get("goal_json")
        goal = json.loads(raw) if raw else {}
        if not isinstance(goal, dict):
            goal = {}
    except Exception:  # noqa: BLE001
        return {"verdict": "unknown", "reason": "goal_json 读不懂"}
    spec, errs = normalise_done_when(goal.get("done_when"))
    if errs:
        return {"verdict": "unknown", "reason": "done_when 读不懂：" + "；".join(errs)}
    if spec is None:
        return {"verdict": "unknown", "reason": "这份纲领没写 done_when（机器判不了）"}
    try:
        from mast.agents._shared.data_paths import (
            experiment_db_path,
            v2_experiment_db_path,
        )

        collect = make_collector(baseline=goal.get("baseline"),
                                 conduct_db=experiment_db_path(),
                                 v2_db=v2_experiment_db_path(),
                                 campaign_id=str(row.get("id") or ""),
                                 askable=False)
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unknown", "reason": f"证据来源不可用: {exc}"}
    v = evaluate_done_when(spec, collect)
    return {"verdict": v.verdict, "reason": v.reason,
            "satisfied": v.satisfied, "total": v.total,
            "per_predicate": [i.as_dict() for i in v.per_predicate]}


def _row_brief(row: dict) -> dict:
    """一行 campaign 的摘要视图。goal_json 解析成对象，解析不了就原样带上。

    「解析不了」和「是空的」必须分得开：前者是库里有东西但读不懂（该说出来），
    后者是真的还没写目标。
    """
    raw = row.get("goal_json")
    goal: Any
    try:
        goal = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001
        goal = {"_unparsed": _snip(raw, 400)}
    return {
        "campaign_id": row.get("id"),
        "title": row.get("title"),
        "hypothesis": row.get("hypothesis"),
        "hypothesis_kind": row.get("hypothesis_kind"),
        "status": row.get("status"),
        "goal": goal,
        "parent_campaign_id": row.get("parent_campaign_id"),
        "created_at": row.get("created_at"),
        "created_by": row.get("created_by"),
    }


def _resolve_campaign(repos, campaign_id: str) -> "tuple[dict | None, str]":
    """``(row, note)``。空 id = 最近的那一份（**并说清楚是替你挑的**）。

    「没有活跃纲领」不能被折叠成「读不到」，也不能被折叠成某一份具体的纲领 ——
    所以挑了哪一份、为什么挑它，都要跟着答案一起回去。
    """
    cid = (campaign_id or "").strip()
    if cid:
        row = repos.campaigns.get(cid)
        return row, ("" if row else f"没有 id={cid} 这一份纲领")
    rows = repos.campaigns.list(limit=1)
    if not rows:
        return None, "库里还没有任何科研纲领（不是读不到 —— 是真的没有）"
    return rows[0], f"没给 campaign_id，用的是最近建的那一份（{rows[0].get('id')}）"


def make_campaign_tools(agent_name: str = "") -> list:
    """建这一族工具。

    *agent_name* 只用于在 ``created_by`` 里署名（``agent:<name>``）—— **不用来
    决定给不给**。几个月后翻一份纲领时，「这条假设是谁提的」要答得上来。
    """
    who = f"agent:{agent_name or 'unknown'}"

    @tool("campaign_list")
    def campaign_list(status: str = "", limit: int = 20) -> str:
        """列出科研纲领（只读）。

        Args:
            status: 可选过滤 —— draft / running / paused / completed / aborted。
                    留空 = 全部。
            limit:  最多返回几条（按创建时间倒序）。

        每条附带该纲领下已有多少实验 / 动作 / 观测 —— 「这条线做过多少」是决定
        要不要继续做下去的第一个事实。
        """
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开（不是「没有纲领」）"})
        n = max(1, min(200, int(limit)))
        want = (status or "").strip()
        # 过滤是**客户端**做的（with_stats 没有 status 参数），所以取数窗口要大于
        # limit —— 否则「最近 20 条里没有 running」会被答成「没有 running」，正是
        # 本仓最常见的那种「读不到被折叠成一个值」。窗口用满时下面会如实说出来。
        window = _SCAN_WINDOW if want else n
        try:
            rows = repos.campaigns.with_stats(limit=window)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        truncated = want and len(rows) >= window
        if want:
            rows = [r for r in rows if str(r.get("status")) == want][:n]
        out = []
        for r in rows:
            brief = _row_brief(r)
            brief.pop("goal", None)   # 列表视图不带目标全文，用 campaign_get 读
            brief["experiment_count"] = r.get("experiment_count")
            brief["action_count"] = r.get("action_count")
            brief["last_activity"] = r.get("last_activity_hlc")
            out.append(brief)
        note = None
        if not out:
            note = ("库里没有符合条件的纲领" if want else "库里还没有任何科研纲领")
        if truncated:
            note = ((note + "；") if note else "") + (
                f"注意：只在最近 {_SCAN_WINDOW} 条里按状态筛的，更早的没扫到")
        return _j({"ok": True, "count": len(out), "campaigns": out, "note": note})

    @tool("campaign_get")
    def campaign_get(campaign_id: str = "") -> str:
        """读一份科研纲领的全文：假设、目标、谱系、以及它下面的实验计数（只读）。

        不给 campaign_id 就读**最近建的**那一份，并在返回里说明是替你挑的
        —— 「没有活跃纲领」和「读的是某一份」不能长得一样。
        """
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开（不是「没有纲领」）"})
        try:
            row, note = _resolve_campaign(repos, campaign_id)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        if row is None:
            return _j({"ok": True, "campaign": None, "note": note})
        brief = _row_brief(row)
        # 判据这一刻怎么看 —— **读即求值**。给 RD 的是一个事实（x/y 满足、
        # 还差什么），不是一个要它自己去判的问题。
        brief["goal_progress"] = _goal_progress(row)
        # 以及它**怎么走到这一步的**。现算只答得出「此刻」，而 RD 要判断的常常
        # 是「这条线还在动吗」——连着五次 run 都停在同一个 1/3，那是一个该改
        # 方案的信号，而单看一次快照完全看不出来。
        brief["goal_history"] = _goal_history(row.get("id") or "")
        cid = brief["campaign_id"]
        # 谱系：这一份是从哪来的、又生出了哪些后续。迭代要看得见，
        # 否则「改了假设」和「另起炉灶」在库里长得一模一样。
        try:
            children = [{"campaign_id": c.get("id"), "title": c.get("title"),
                         "status": c.get("status")}
                        for c in repos.campaigns.list(limit=200)
                        if c.get("parent_campaign_id") == cid]
        except Exception:  # noqa: BLE001
            children = []
        try:
            exps = repos.experiments.list(campaign_id=cid, limit=200)
        except Exception:  # noqa: BLE001
            exps = []
        finished = [e for e in exps if e.get("ended_at")]
        return _j({
            "ok": True,
            "campaign": brief,
            "children": children,
            "experiment_total": len(exps),
            "experiment_finished": len(finished),
            "experiment_open": len(exps) - len(finished),
            "note": note or None,
            "next": "看逐条实验结论用 campaign_experiments，看已成立的主张用 campaign_claims",
        })

    @tool("campaign_create")
    def campaign_create(title: str, hypothesis: str,
                        hypothesis_kind: str = "exploratory",
                        goal_json: str = "", parent_campaign_id: str = "") -> str:
        """新建一份科研纲领（DRAFT 态）。

        Args:
            title:            纲领名（一句话说清在追什么）。
            hypothesis:       **可证伪的**假设。「研究一下这块样品」不是假设，
                              「表面的条纹相是电荷序而非结构畸变」才是。
            hypothesis_kind:  exploratory（探索）/ confirmatory（验证）/
                              calibration（标定）/ methodology（方法学）四选一。
                              **不在这四个里会被直接拒绝**，不会被改成别的。
            goal_json:        目标的 JSON 对象。三个键：

                              * ``question`` —— 要回答什么（人读）。
                              * ``success_criteria`` —— 什么算答完了（人读的
                                注释；**没有任何代码求值它**）。
                              * ``done_when`` —— **机器判的**终止判据。只能从
                                下面的闭集里选；写不进这个闭集的判据就别写，
                                不要拿自由文本代替 —— 那样它不会被任何代码读。

                              ``done_when`` 的形状：一条谓词 ``{"kind": ...}``，
                              或 ``{"all"|"any"|"not": [...]}``，或一个列表
                              （= ``all``）。可选的 ``kind``：

                              * ``artifact_present{field}`` —— 产出了某份产物
                                （field ∈ literature_report / experiment_plan /
                                analysis / draft / review / last_scan /
                                research_campaign）
                              * ``artifact_count_at_least{field, n}``
                              * ``conduct_completed{spec_id | conduct_id}``
                                —— 某份执行模板跑完了
                              * ``best_frame_settled{tag}`` —— 那轮追猎连着
                                若干轮没更好（「差不多了」）
                              * ``claims_supported{min_count}`` —— 至少 N 条
                                论断被证据支持
                              * ``operator_confirmed{}`` —— 要人点头

                              例：``{"question": "…", "done_when":
                              {"all": [{"kind": "artifact_present",
                              "field": "analysis"},
                              {"kind": "claims_supported", "min_count": 2}]}}``
                              （``conduct_completed`` 的 ``spec_id`` 填你这条线
                              实际用的模板名 —— 那个名字属于模板层，不写在这里）

                              **不合法一律整体拒绝**（不会替你丢掉那几条）。
                              可留空。
            parent_campaign_id: 这一份是在迭代某份既有纲领时填 —— 谱系是
                              campaign 层的价值所在，别用「另起一份」代替迭代。

        建完是 draft：要推进用 campaign_update(status="running")。
        **这里不填任何仪器参数**（偏压 / setpoint / 尺寸都不属于这一层）。
        """
        if not (title or "").strip() or not (hypothesis or "").strip():
            return _j({"ok": False, "reason": "title 和 hypothesis 都必须写"})
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开，没有建"})
        try:
            from mast.logging.v2.repos import HYPOTHESIS_KINDS
        except Exception:  # noqa: BLE001
            HYPOTHESIS_KINDS = ("exploratory", "confirmatory",
                                "calibration", "methodology")
        kind = (hypothesis_kind or "").strip() or "exploratory"
        if kind not in HYPOTHESIS_KINDS:
            return _j({"ok": False, "reason":
                       f"hypothesis_kind 只能是 {list(HYPOTHESIS_KINDS)} 之一，"
                       f"你给的是 {kind!r} —— 没有建，也没有替你改成别的。"})
        goal: Any = {}
        if (goal_json or "").strip():
            try:
                goal = json.loads(goal_json)
            except Exception as exc:  # noqa: BLE001
                return _j({"ok": False, "reason": f"goal_json 解析不了: {exc}"})
            if not isinstance(goal, dict):
                return _j({"ok": False, "reason": "goal_json 要是一个 JSON 对象"})
            bad = _check_done_when(goal)
            if bad is not None:
                return _j(bad)
        parent = (parent_campaign_id or "").strip() or None
        if parent and not repos.campaigns.get(parent):
            return _j({"ok": False, "reason":
                       f"parent_campaign_id={parent} 不存在 —— 没有建。"
                       "谱系指向一份不存在的纲领，比没有谱系更糟。"})
        try:
            cid = repos.campaigns.create(
                title=title.strip(), hypothesis=hypothesis.strip(),
                hypothesis_kind=kind, goal=goal, created_by=who,
                parent_campaign_id=parent)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"建不出来: {exc}"})
        return _j({"ok": True, "campaign_id": cid, "status": "draft",
                   "next": "用 campaign_request_plan 把它委托给实验设计，"
                           "或用 campaign_update(status='running') 让它开始"})

    @tool("campaign_update")
    def campaign_update(campaign_id: str, hypothesis: str = "",
                        goal_json: str = "", title: str = "",
                        hypothesis_kind: str = "", status: str = "") -> str:
        """修订一份纲领：改假设 / 改目标 / 改状态。留空的字段**不动**。

        假设被数据推翻了就改假设，这正是 campaign 层存在的意义 —— 一个不能修订
        自己假设的科研纲领不叫科研纲领。改动会留下 ``created_by`` 之外的痕迹
        （这一步本身是一次 agent 轮次，在别处有记录）。

        Args:
            campaign_id:      要改哪一份（必填 —— 这里不替你猜）。
            hypothesis:       新的假设。
            goal_json:        新的目标 JSON 对象（**整体替换**，不是合并）。
            title:            新的名字。
            hypothesis_kind:  四选一，写错直接拒绝。
            status:           draft/running/paused/completed/aborted，写错直接拒绝。

        真的要另起一条线（而不是修订这一条）时：用
        ``campaign_create(parent_campaign_id=…)``，别把旧的改成新的 ——
        那样会把「我们改主意了」抹成「我们一直是这么想的」。
        """
        cid = (campaign_id or "").strip()
        if not cid:
            return _j({"ok": False, "reason":
                       "要改哪一份必须写明 campaign_id（改错一份纲领比不改更糟）"})
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开，没有改"})
        try:
            from mast.logging.v2.repos import CAMPAIGN_STATUSES, HYPOTHESIS_KINDS
        except Exception:  # noqa: BLE001
            HYPOTHESIS_KINDS = ("exploratory", "confirmatory",
                                "calibration", "methodology")
            CAMPAIGN_STATUSES = ("draft", "running", "paused",
                                 "completed", "aborted")
        if repos.campaigns.get(cid) is None:
            return _j({"ok": False, "reason": f"没有 id={cid} 这一份纲领"})

        kind = (hypothesis_kind or "").strip()
        if kind and kind not in HYPOTHESIS_KINDS:
            return _j({"ok": False, "reason":
                       f"hypothesis_kind 只能是 {list(HYPOTHESIS_KINDS)} 之一"})
        want_status = (status or "").strip()
        if want_status and want_status not in CAMPAIGN_STATUSES:
            return _j({"ok": False, "reason":
                       f"status 只能是 {list(CAMPAIGN_STATUSES)} 之一"})
        goal = None
        if (goal_json or "").strip():
            try:
                goal = json.loads(goal_json)
            except Exception as exc:  # noqa: BLE001
                return _j({"ok": False, "reason": f"goal_json 解析不了: {exc}"})
            if not isinstance(goal, dict):
                return _j({"ok": False, "reason": "goal_json 要是一个 JSON 对象"})
            bad = _check_done_when(goal)
            if bad is not None:
                return _j(bad)

        # 传了什么就报什么。**不要**用 `if v` 推断 —— goal_json="{}" 是一次真正的
        # 「把目标清空」，而空 dict 是 falsy：那样报出来的 changed 会漏掉一次
        # 确实发生了的写入，而漏报的写入正是事后最难查的一种。
        fields: dict[str, Any] = {}
        if title.strip():
            fields["title"] = title.strip()
        if hypothesis.strip():
            fields["hypothesis"] = hypothesis.strip()
        if kind:
            fields["hypothesis_kind"] = kind
        if goal is not None:
            fields["goal"] = goal
        # **在写之前**抓一次旧判据。写完再读，读到的已经是新值 —— 于是那句
        # 「你把 done_when 弄丢了」永远不会出现，而它存在的全部意义就是出现。
        had_done_when = False
        if goal is not None and not goal.get("done_when"):
            try:
                _prev = json.loads((repos.campaigns.get(cid) or {}).get("goal_json")
                                   or "{}")
                had_done_when = bool(isinstance(_prev, dict)
                                     and _prev.get("done_when"))
            except Exception:  # noqa: BLE001
                had_done_when = False

        changed: list[str] = []
        try:
            if fields and repos.campaigns.update(cid, **fields):
                changed += list(fields)
            if want_status:
                repos.campaigns.set_status(cid, want_status)
                changed.append("status")
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"改不了: {exc}"})
        if not changed:
            return _j({"ok": True, "campaign_id": cid, "changed": [],
                       "note": "所有字段都留空了 —— 什么都没改（这不是失败，"
                               "但也不是一次修订）"})
        out: dict[str, Any] = {"ok": True, "campaign_id": cid, "changed": changed}
        # ``goal`` 是**整体替换**。改 question 的时候顺手把 done_when 弄丢了，
        # 是一次静默的功能撤除 —— 库里从此没有机器可判的终止条件，而没有任何
        # 地方会说一声。所以这里说一声。
        if had_done_when:
            # 两种写法都算「去掉」：**没提到** done_when，和把它写成空
            # （``[]`` / ``{}`` / ``""`` —— 后者第一版漏了，因为它归一化之后
            # 键还在、只是值成了 None）。
            out["note"] = ("⚠ 本次替换把原有的 done_when 去掉了 —— "
                           "这份纲领从此没有机器可判的终止条件。"
                           "如果不是有意的，把它一起写回去。")
        return _j(out)

    @tool("campaign_request_plan")
    def campaign_request_plan(
        campaign_id: str, plan_request: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> "ArtifactToolReturn | str":
        """把「请为这条假设设计一个实验」的**委托**记到纲领上，并交给下游。

        这是 campaign 层的产出。它做两件事，缺一不可：

        1. 把委托写进纲领的 ``goal.plan_request``（落库 —— 几周后还查得到
           当初委托的是什么）；
        2. 把纲领登记为**本次协作的上游产物**，于是实验设计在它自己的上下文里
           直接看到假设与委托。**不要指望在交接语里复述** —— 那句话会被压缩摘掉，
           这个仓库已经为此付过账了。

        Args:
            campaign_id:  哪一份纲领的委托（必填）。
            plan_request: 委托正文。写「要区分什么 / 什么算答完了 / 有什么约束」，
                          **不要写偏压、setpoint、扫描尺寸** —— 那些由实验设计
                          从文献和历史里取，你在这里填一个数只会变成一个没有依据
                          却看起来像有依据的数。

        写完之后交给 experiment_design（handoff_to_experiment_design）。
        """
        cid = (campaign_id or "").strip()
        req = (plan_request or "").strip()
        if not cid or not req:
            return _j({"ok": False,
                       "reason": "campaign_id 和 plan_request 都必须写"})
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开，委托没有落库"})
        row = repos.campaigns.get(cid)
        if row is None:
            return _j({"ok": False, "reason": f"没有 id={cid} 这一份纲领"})
        try:
            goal = json.loads(row.get("goal_json") or "{}")
            if not isinstance(goal, dict):
                goal = {"_previous": goal}
        except Exception:  # noqa: BLE001
            goal = {}
        goal["plan_request"] = req
        try:
            repos.campaigns.update(cid, goal=goal)
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"委托落库失败: {exc}"})

        summary = _j({
            "ok": True, "campaign_id": cid,
            "plan_request": _snip(req, 400),
            "note": "委托已记到纲领上，并登记为上游产物 —— "
                    "实验设计会在自己的上下文里直接读到它",
            "next": "handoff_to_experiment_design",
        })
        ref = campaign_ref(
            campaign_id=cid, title=row.get("title") or "",
            hypothesis=row.get("hypothesis") or "",
            hypothesis_kind=row.get("hypothesis_kind") or "",
            status=row.get("status") or "",
            plan_request=req,
            parent_campaign_id=row.get("parent_campaign_id") or "",
            done_when_brief=_done_when_brief(row))
        return ArtifactToolReturn(summary, {"research_campaign": ref},
                                  tool_call_id=tool_call_id,
                                  name="campaign_request_plan")

    @tool("campaign_experiments")
    def campaign_experiments(campaign_id: str = "", limit: int = 20) -> str:
        """既往实验记录（只读）：做过什么、结果如何、结论是什么。

        Args:
            campaign_id: 只看某一份纲领下的实验。**留空 = 全库最近的实验**
                         （跨纲领），用来判断「这件事是不是已经有人做过了」。
            limit:       最多几条。

        ``exit_status`` 与 ``conclusion`` 为空**不代表实验失败**，只代表它还没
        结束或者没有人写结论 —— 这两者不能折叠成同一个答案。
        """
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开（不是「没有记录」）"})
        cid = (campaign_id or "").strip()
        n = max(1, min(200, int(limit)))
        try:
            rows = (repos.experiments.with_counts(campaign_id=cid, limit=n)
                    if cid else repos.experiments.with_counts(limit=n))
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        out = []
        for r in rows:
            ended = r.get("ended_at")
            out.append({
                "experiment_id": r.get("id"),
                "campaign_id": r.get("campaign_id"),
                "title": r.get("title"),
                "exp_type": r.get("exp_type"),
                "sample": r.get("sample_label"),
                "started_at": r.get("started_at"),
                "state": "已结束" if ended else "进行中",
                "exit_status": r.get("exit_status"),
                "conclusion": _snip(r.get("conclusion")) or None,
                "action_count": r.get("action_count"),
                "observation_count": r.get("observation_count"),
            })
        return _j({"ok": True, "scope": cid or "(全库最近)", "count": len(out),
                   "experiments": out,
                   "note": None if out else "这个范围里还没有实验记录"})

    @tool("campaign_claims")
    def campaign_claims(campaign_id: str = "", limit: int = 30) -> str:
        """已提出的主张 / 结论（只读）。

        claims 是「我们认为已经知道了什么」，与实验记录的「我们做了什么」分开存。
        ``status`` 三态：proposed（提出）/ supported（有证据支持）/ refuted（被推翻）
        —— **refuted 是最有价值的一条**，它是纲领该迭代的直接证据。

        campaign_id 留空 = 全库最近的主张。
        """
        repos = _repos()
        if repos is None:
            return _j({"ok": False, "reason": "v2 记录库打不开（不是「没有主张」）"})
        cid = (campaign_id or "").strip()
        try:
            rows = repos.claims.list_claims(
                campaign_id=cid or None, limit=max(1, min(200, int(limit))))
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"读库失败: {exc}"})
        out = [{
            "claim_id": r.get("id"),
            "statement": _snip(r.get("statement")),
            "status": r.get("status"),
            "confidence": r.get("confidence"),
            "experiment_id": r.get("experiment_id"),
            "campaign_id": r.get("campaign_id"),
            "created_at": r.get("created_at"),
        } for r in rows]
        return _j({"ok": True, "scope": cid or "(全库最近)", "count": len(out),
                   "claims": out,
                   "note": None if out else "这个范围里还没有任何主张"})

    return [campaign_list, campaign_get, campaign_create, campaign_update,
            campaign_request_plan, campaign_experiments, campaign_claims]
