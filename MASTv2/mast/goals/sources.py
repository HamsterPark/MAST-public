"""证据收集器 —— 每条谓词从哪读它那一个事实。

## 唯一的纪律

**读不到的字段不写进证据包。** 不是 ``None``，不是 ``0``，不是 ``False`` ——
让 ``lookup`` 回哨兵，让核回 UNDECIDABLE。写一个值进去等于把「读不到」变成
一个答案，而「读不到」与「否」驱动的下一步是相反的：前者要说出来（也许照旧
唤醒、也许请人看一眼），后者是「接着干」。

收集器可以捎带一个 ``_why``（人读的「为什么读不到」），它不进判据，只进
未满足清单的说明。

## 路径全部由调用方注入

模块内不解析 ``project_root()`` / ``MASTConfig()`` —— 那会让本包成为路径的
第二真源。默认值是**惰性**取的（函数体内 import），所以 import 本模块不会把
仪器栈、记录库、技能树一并拉起来。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


# ── 单条谓词的收集器 ──────────────────────────────────────────────────

def collect_conduct_completed(args: Mapping[str, Any], *,
                              db_path=None) -> "dict | None":
    """``conducts`` 表里跑完了几份。

    走只读通道：``cd_enabled=0``（出厂默认）只关掉**推进**，不关掉**看**。
    一条纲领的判据不该因为指挥线程没开就永远判不了。
    """
    try:
        from mast.conduct.store import ConductStore

        path = db_path
        if path is None:
            from mast.agents._shared.data_paths import experiment_db_path

            path = experiment_db_path()
        store = ConductStore(path)
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"conducts 库打不开：{exc}"}

    cid = str(args.get("conduct_id") or "").strip()
    try:
        if cid:
            row = store.get(cid)
            if row is None:
                # **查无此行 ≠ 没跑完。** 一个不存在的 id 多半是判据写错了，
                # 报成「还没跑完」会让这条纲领永远差一口气而没人知道为什么。
                return {"_why": f"conducts 里没有 {cid} 这一行（判据写错了？）"}
            return {"completed_count": 1 if row.get("status") == "completed" else 0}
        spec_id = str(args.get("spec_id") or "").strip()
        rows = store.list_conducts(status="completed", limit=200)
        return {"completed_count": sum(1 for r in rows
                                       if str(r.get("spec_id") or "") == spec_id)}
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"conducts 读失败：{exc}"}


def collect_best_frame_settled(args: Mapping[str, Any]) -> "dict | None":
    """「连着几轮没更好」—— 与 ``paper_frame`` 出口闸读同一份记录、同一个函数。

    文件不存在 ⇒ 追猎还没开始 ⇒ ``good_enough_to_stop=False``（**是** not_done，
    不是判不了：没开始就是没收手）。文件在但读不懂 ⇒ 判不了（**不许**当成
    没开始，那会让一次磁盘损坏读成「接着扫」）。
    """
    tag = str(args.get("tag") or "").strip()
    if not tag:
        return {"_why": "没给 tag"}
    try:
        from mast.skills.builtins.best_frame import (
            BestFrameStoreUnreadable,
            peek_store,
        )
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"best_frame 记录器不可用：{exc}"}
    try:
        out = peek_store(tag, dry_limit=args.get("dry_limit"))
    except BestFrameStoreUnreadable as exc:
        return {"_why": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"best_frames 读失败：{exc}"}
    return {"good_enough_to_stop": bool(out.get("good_enough_to_stop"))}


def collect_claims_supported(args: Mapping[str, Any], *, db_path=None,
                             campaign_id: str = "") -> "dict | None":
    """v2 记录库里被证据支持的论断条数。"""
    import sqlite3
    from contextlib import closing

    try:
        path = db_path
        if path is None:
            from mast.agents._shared.data_paths import v2_experiment_db_path

            path = v2_experiment_db_path()
        path = str(path)
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"v2 库路径不可用：{exc}"}
    sql = ("SELECT COUNT(*) FROM claims WHERE status IN ('supported','verified')")
    params: list = []
    if campaign_id:
        sql += " AND campaign_id=?"
        params.append(campaign_id)
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            n = conn.execute(sql, params).fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        return {"_why": f"claims 读失败：{exc}"}
    return {"count": int(n)}


def artifact_versions_from_disk() -> "dict | None":
    """每个可等产物类的「版本」——``(件数, 最新 mtime)``。

    与唤醒调度器读的是**同一个函数**：两处对「有没有新东西到」的看法必须一致，
    否则会出现「判据说到了、调度器说没到」这种互相矛盾又谁也不报错的形状。
    """
    try:
        from mast.core.wake_scheduler import _artifact_versions

        v = _artifact_versions()
        return v or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("goal: artifact versions unreadable (%s)", exc)
        return None


def _cmp_versions(now, base) -> "bool | None":
    """这一类产物相对基线有没有变化。**读不到 / 没有基线**都回 ``None``。

    ``base is None`` 曾经回 ``True``（「没有基线 = 目标设定时那一类是空的」）。
    那是把**「没记」**折叠成**「当时是空的」** —— 而这两件事在两条真实路径上
    经常发生：

    * campaign 侧从来不抓基线（判据是 RD 写进 ``goal_json`` 的，没有「设定那一刻
      的产物快照」这一步）；
    * run 侧第一跳抓基线时如果产物清单恰好读不到，也会留下一个没有基线的目标。

    折叠的后果是**目标被历史产物假满足**：实验文件夹里有上周任何一份 analysis，
    `artifact_present(analysis)` 就 done —— 于是 supervisor 第一跳就 END，或者
    唤醒调度器把这条纲领下每一份 park 都 `mark_done_by_goal` 关掉，一个 agent
    都不醒。

    现在回 ``None`` ⇒ 该项 UNDECIDABLE ⇒ 目标 unknown ⇒ 两条路都走**安全方向**
    （supervisor 按模型判断走并留痕；唤醒照旧唤醒）。要让这条谓词真正可用，
    就得**有人抓基线** —— 见 ``snapshot_baseline`` 的两个调用点。
    """
    if not now:
        return None
    count = 0
    try:
        count = int(now[0])
    except Exception:  # noqa: BLE001
        return None
    if count <= 0:
        return False          # 确定为空 —— 这一条是**读得到**的否定
    if base is None:
        return None
    try:
        return tuple(now) != tuple(base)
    except Exception:  # noqa: BLE001
        return None


def collect_artifact_present(args: Mapping[str, Any], *, baseline=None,
                             versions=None,
                             state_present: "Callable[[str], bool] | None" = None,
                             ) -> "dict | None":
    """产物到了没有 —— **相对基线**，除非判据明说不在乎。

    为什么默认要基线：一个续接的群聊线程里躺着上一个任务的 ``literature_report``，
    实验文件夹里躺着上周的草稿。不看基线的话，目标在第一跳就「达成」——那是
    「太早停」换了个方式复现。

    ``state_present`` 是 run 路径的快路：state 里带着 = 铁证（这一 run 里刚交
    过班）。state 里**没有**证明不了什么（新 run 的 state 本来就空），所以那一
    侧要落到磁盘。
    """
    field = str(args.get("field") or "")
    if not field:
        return {"_why": "没给 field"}
    if state_present is not None:
        # state 里带着 = **这一 run 里刚交过班**……只有当基线里它不在时才成立。
        #
        # 产物通道的字段是 ``last_wins`` LastValue，跨任务留在同一个 checkpoint
        # 线程里：续接一个群聊会话时，上一个任务产出的 ``analysis`` 就在 state
        # 里躺着。第一版这条快路直接 ``return True``，一眼都不看基线 ——
        # 于是 ``allow_preexisting=False`` 在这条路上完全失效，新任务第一跳就
        # 「达成」。
        try:
            if state_present(field) and field not in (baseline or {}).get(
                    "_state_present", ()):
                return {"new_since_baseline": True}
        except Exception:  # noqa: BLE001
            pass
    vers = versions if versions is not None else artifact_versions_from_disk()
    if not vers:
        return {"_why": "产物清单读不到"}
    if bool(args.get("allow_preexisting")):
        now = vers.get(field)
        try:
            return {"new_since_baseline": bool(now and int(now[0]) > 0)}
        except Exception:  # noqa: BLE001
            return {"_why": f"{field} 的计数读不懂"}
    got = _cmp_versions(vers.get(field), (baseline or {}).get(field))
    if got is None:
        return {"_why": f"{field} 的版本读不到"}
    return {"new_since_baseline": bool(got)}


def collect_artifact_count(args: Mapping[str, Any], *, baseline=None,
                           versions=None) -> "dict | None":
    field = str(args.get("field") or "")
    vers = versions if versions is not None else artifact_versions_from_disk()
    if not vers:
        return {"_why": "产物清单读不到"}
    now = vers.get(field)
    if not now:
        return {"_why": f"{field} 的版本读不到"}
    if baseline is None:
        # 同 ``_cmp_versions``：没有基线时把 n_base 当成 0，等于说「目标设定那一刻
        # 一份都没有」—— 于是历史产物全被算成这次的增量。
        return {"_why": f"{field} 没有基线，数不出「又产出了几份」"}
    try:
        n_now = int(now[0])
        base = baseline.get(field)
        n_base = int(base[0]) if base else 0
    except Exception:  # noqa: BLE001
        return {"_why": f"{field} 的计数读不懂"}
    return {"delta_count": max(0, n_now - n_base)}


def collect_operator_confirmed(args: Mapping[str, Any], *,
                               answer: "str | None" = None,
                               askable: bool = True) -> "dict | None":
    """用户确认了没有。

    ``askable=False``（没有 checkpointer，问不出去）⇒ 判不了，**不是**「没确认」：
    「问不了」与「问了说不」驱动的下一步不同。
    """
    if answer in ("yes", "no"):
        return {"answer": answer}
    if not askable:
        return {"_why": "这次运行没法向用户提问（没有 checkpointer）"}
    return {"answer": "no"}


# ── 组装 ──────────────────────────────────────────────────────────────

def make_collector(*, baseline=None, versions=None, state_present=None,
                   conduct_db=None, v2_db=None, campaign_id: str = "",
                   operator_answer: "str | None" = None,
                   askable: bool = True) -> "Callable[[str, dict], dict | None]":
    """把上面那些接成 ``collect(kind, args)``。

    每个来源都可注入，缺省惰性取真源。**一个来源缺席只让那一条谓词判不了**，
    不会让整份判据失效 —— 「读不到一半」和「什么都读不到」是两种不同的处境。
    """
    vers_cache: dict = {}

    def _vers():
        if "v" not in vers_cache:
            vers_cache["v"] = (versions if versions is not None
                               else artifact_versions_from_disk())
        return vers_cache["v"]

    def collect(kind: str, args: dict) -> "dict | None":
        if kind == "artifact_present":
            return collect_artifact_present(args, baseline=baseline,
                                            versions=_vers(),
                                            state_present=state_present)
        if kind == "artifact_count_at_least":
            return collect_artifact_count(args, baseline=baseline,
                                          versions=_vers())
        if kind == "conduct_completed":
            return collect_conduct_completed(args, db_path=conduct_db)
        if kind == "best_frame_settled":
            return collect_best_frame_settled(args)
        if kind == "claims_supported":
            return collect_claims_supported(args, db_path=v2_db,
                                            campaign_id=campaign_id)
        if kind == "operator_confirmed":
            return collect_operator_confirmed(args, answer=operator_answer,
                                              askable=askable)
        # 目录里有、这里没接 ⇒ 判不了。**不是** done。加谓词时会被
        # test_every_catalog_kind_has_a_collector 逮住。
        return {"_why": f"没有为 {kind} 接上收集器"}

    return collect


def snapshot_baseline(versions=None, *, state_present=None) -> "dict | None":
    """目标被设定那一刻的产物版本。之后的「新」都是相对它说的。

    **抓不到回 ``None``，不是 ``{}``。** 一个空 dict 是「那一刻什么都没有」——
    一个完全正当的基线；把读失败也写成它，后果是这个目标从此以为世界是空的，
    于是任何一份历史产物都算「新的」。调用方看到 ``None`` 要**下一跳再抓**，
    而不是把它当成一次成功的快照存下来。

    ``state_present`` 是一个 ``(field) -> bool``：基线要同时记下**当时 state 里
    已经有哪些产物**，否则续接会话时上一个任务留在通道里的产物会被读成「这一 run
    刚产出的」（见 :func:`collect_artifact_present` 的快路）。
    """
    v = versions if versions is not None else artifact_versions_from_disk()
    if not v:
        return None
    out = dict(v)
    if state_present is not None:
        present = []
        for field in v:
            try:
                if state_present(field):
                    present.append(field)
            except Exception:  # noqa: BLE001
                continue
        out["_state_present"] = tuple(present)
    return out


__all__ = ["make_collector", "snapshot_baseline", "artifact_versions_from_disk",
           "collect_artifact_present", "collect_artifact_count",
           "collect_conduct_completed", "collect_best_frame_settled",
           "collect_claims_supported", "collect_operator_confirmed"]
