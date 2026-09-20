"""跨天记住「目前最好的那一张」—— 以及「连着几轮没更好了」。

多天追猎一张图的典型流程：反复修整,直到觉得「差不多了」就存一张;此后若又
修出更好的就再存一张,直到连续多轮都没有更好的为止。

「差不多就放弃」翻成机器话不是一个阈值（你说不出「集中度 > X 就够了」），
而是 **loop-until-dry**：连着 N 轮都没能出比现有更好的，那就是差不多了。

## 为什么这是一个技能，而不是分析步、也不是 conduct 状态

* **分析步是纯函数** —— 不许碰全局可变状态，而且重跑必须给同样的答案。
  「目前最好的是哪张」天生带历史。
* **binding 只看得到每个 step_id 的最新值** —— 绕道回来之后同一步重跑，
  上一张的数据就从 flat 表里被覆盖了。
* **conduct 的账本本来就存的是「跑到哪儿」，不是「攒到了什么」。**

所以这里落成实验目录下的一个 JSON：**跨进程重启天然活着，而且人可以打开来看**。
一个只存在于内存里的「最好」，在几天的流程里等于没有。

## 判据在**图**上，不在针尖上

比较用的是每张图自己的成色（角向集中度），而不是当时针尖的读数 ——
产物是图。图会累积（每张都在磁盘上），针尖不会（只有一根、没有备份）。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 连着这么多张都没超过现有最好的 ⇒ 「差不多了」。
#: 这是认可的默认值。**不是标定出来的**，是一条约定 ——
#: 它的好处正在于不用假装知道「多好才算够」。
DEFAULT_DRY_LIMIT = 3

#: 新的一张要比现有最好的高出这个倍数才算「更好」。1.0 = 高一点点就算。
#: 出厂 1.05：判据本身有几个百分点的抖动，卡在 1.0 会把噪声记成进步，
#: 于是 dry_rounds 永远清零、永远不收手。
DEFAULT_BETTER_RATIO = 1.05

#: 「值不值得再花那一小时」的门槛基线（角向集中度）。与
#: ``composite.publication_frame.DEFAULT_MIN_CONCENTRATION`` 同一个数，
#: 从那里取，不在这里再写一遍。
#: 门槛 = ``max(基线, 现有最好 × DEFAULT_GATE_FRAC)`` ——
#: **越到后面越挑剔**：已经有一张 900 的图之后，再花一小时去扫一个 400 的针尖
#: 是亏的。人会这么做，而流程常常不会，所以把它写进机器。
DEFAULT_GATE_FRAC = 0.8


def _store_path(tag: str):
    from mast.config import MASTConfig

    d = MASTConfig().experiments_dir / "best_frames"
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tag))[:80]
    return d / f"{safe or 'default'}.json"


def _settled(dry_rounds, dry_limit) -> bool:
    """「差不多了」的**唯一**一处比较。

    这句话同时出现在三个地方：``peek``（判据侧读的那份）、``execute`` 的两条
    记录分支、以及 ``paper_frame`` 模板的出口闸。三处各写一遍 ``dry >= limit``
    今天算出来一样，但缺省值一岔（一处 ``dry_limit is None``、另一处
    ``params.get(...) or DEFAULT``）就会让模板和判据对同一份记录给出不同答案 ——
    而那种不一致不会报错。
    """
    return bool(int(dry_rounds or 0) >= int(dry_limit or 0))


class BestFrameStoreUnreadable(Exception):
    """记录文件在，但读不懂。

    **不是「还没开始」**。这两件事驱动完全相反的下一步:没开始 ⇒ 接着扫;
    读不懂 ⇒ 停下来让人看一眼(而且**不许覆盖**它)。折叠成同一个返回值,
    一次磁盘损坏就会被读成「这轮追猎刚起步」。
    """


def peek(tag: str, *, dry_limit: int | None = None,
         gate_floor_frac: float | None = None,
         gate_floor_base: float | None = None) -> dict:
    """只读现状:目前最好的是哪张、连着几轮没更好、这一轮的门槛是多少。

    2026-08-27 抽成模块级函数。在那之前这段只活在 ``TrackBestFrame.execute``
    的 ``peek`` 分支里,于是想问「这轮追猎收手了没」的第二个调用方
    (``mast.goals`` 的 ``best_frame_settled`` 谓词)只能自己再算一遍
    ``dry_rounds >= dry_limit``。**那个比较只许有一处** —— 它就是
    ``paper_frame`` 模板出口闸的那句话,两份实现意味着模板和判据可以对同一份
    记录给出不同的答案。

    ``good_enough_to_stop`` 的含义:没人说得出「角向集中度 > X 就够印了」,
    但人人说得出「连着三轮都没更好,那就是差不多了」。后者可写、可审、
    可解释,而且自动适应不同的样品与针尖。

    文件不存在 ⇒ 正常返回一个「零轮」的现状(追猎还没开始,不是异常)。
    文件在但读不懂 ⇒ 抛 :class:`BestFrameStoreUnreadable`。
    """
    limit = int(DEFAULT_DRY_LIMIT if dry_limit is None else dry_limit)
    p = _store_path(tag)
    if p.is_file():
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise BestFrameStoreUnreadable(f"记录 {p} 读不懂：{exc}") from exc
    else:
        rec = {}
    best = rec.get("best") or None
    entries = list(rec.get("entries") or [])
    dry = int(rec.get("dry_rounds") or 0)
    return {
        "tag": tag,
        "store": str(p),
        "best_path": (best or {}).get("frame_path"),
        "best_quality": (best or {}).get("quality"),
        "n_seen": len(entries),
        "dry_rounds": dry,
        "dry_limit": limit,
        "good_enough_to_stop": _settled(dry, limit),
        "gate_floor": _gate_floor_for(
            (best or {}).get("quality"),
            frac=gate_floor_frac, base=gate_floor_base),
        "recorded": False,
    }


#: :func:`peek` 的别名。``TrackBestFrame.execute`` 里有一个同名局部变量
#: (``peek = bool(params.get("peek"))``)会把模块函数遮蔽掉 —— 那个参数名是
#: 技能的对外契约,不能改,所以在这里给函数第二个名字。
peek_store = peek


def _gate_floor_for(best_q, *, frac: float | None = None,
                    base: float | None = None) -> float:
    """值不值得再花那一小时的门槛。**随 incumbent 上移。**"""
    if base is None:
        from mast.skills.composite.publication_frame import (
            DEFAULT_MIN_CONCENTRATION,
        )
        base = DEFAULT_MIN_CONCENTRATION
    f = float(DEFAULT_GATE_FRAC if frac is None else (frac or DEFAULT_GATE_FRAC))
    return max(float(base), float(best_q or 0.0) * f)


class TrackBestFrame(BaseSkill):
    """记下这一张，回「它是不是目前最好的」与「连着几轮没更好了」。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TrackBestFrame",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "记录一张候选帧和它的成色，然后回报它有没有**超过**当前最好的那张、"
                "以及连着多少轮没能超过。靠实验文件夹里的一个 JSON 落盘，所以为了"
                "一张能发表的图连着追好几天时，它扛得住重启 —— 一个只活在内存里的"
                "「目前最好」，跨几天等于没有。「够好了，可以收手」在这里表达成 "
                "loop-until-dry（连着 N 轮没有进步），不是一条质量阈值："
                "没有人说得出那条阈值，但每个人都说得出 N。"
            ),
            parameters=[
                ParameterSpec(
                    name="tag", type="str",
                    description=("这一轮追猎的标识（用 conduct_id 或实验 id）。"
                                 "不同的追猎各记各的，别互相污染。"),
                    required=True),
                ParameterSpec(
                    name="frame_path", type="str",
                    description="这一张图的路径。空 = 这一轮没出图（只推进 dry 计数）。",
                    required=False, default=""),
                ParameterSpec(
                    name="quality", type="float",
                    description=("这一张的成色（角向集中度）。**留空 = 判不了** —— "
                                 "那既不算更好也不算更差，不推进 dry 计数。"
                                 "把「判不了」记成「不够好」会让流程在一个"
                                 "读不到数的故障上收手。"),
                    required=False),
                ParameterSpec(
                    name="dry_limit", type="int",
                    description=f"连着几张没更好就算「差不多了」。出厂 {DEFAULT_DRY_LIMIT}。",
                    required=False, default=DEFAULT_DRY_LIMIT,
                    min_value=1, max_value=100),
                ParameterSpec(
                    name="better_ratio", type="float",
                    description=(f"要高出这个倍数才算更好。出厂 {DEFAULT_BETTER_RATIO}。"
                                 "卡在 1.0 会把判据自己的抖动记成进步，"
                                 "于是 dry 计数永远清零、永远不收手。"),
                    required=False, default=DEFAULT_BETTER_RATIO,
                    min_value=1.0, max_value=10.0),
                ParameterSpec(
                    name="gate_floor_base", type="float",
                    description=("「值不值得再花一小时」的门槛基线。留空 = 取"
                                 "ScanPublicationFrame 的出厂线（单一真源，"
                                 "不在这里再写一个数）。"),
                    required=False, min_value=0.0, max_value=100000.0),
                ParameterSpec(
                    name="gate_floor_frac", type="float",
                    description=(f"门槛 = max(基线, 现有最好 × 这个系数)。出厂 "
                                 f"{DEFAULT_GATE_FRAC}。**越到后面越挑剔** —— "
                                 "已经有一张 900 的图之后，再花一小时去扫一个 400 的"
                                 "针尖是亏的。"),
                    required=False, default=DEFAULT_GATE_FRAC,
                    min_value=0.0, max_value=10.0),
                ParameterSpec(
                    name="peek", type="bool",
                    description="只读现状，不记这一张（供闸门复查用）。",
                    required=False, default=False),
            ],
            estimated_duration_s=0.2,
            composition_level=0,
            tags=["record", "best", "publication", "loop-until-dry"],
        )


    @staticmethod
    def _gate_floor(params: dict, best_q) -> float:
        """值不值得再花那一小时的门槛。**随 incumbent 上移。**

        实现在模块级 :func:`_gate_floor_for` —— 与 :func:`peek` 共用一份,
        免得「门槛」在技能里和判据里各算各的。
        """
        return _gate_floor_for(best_q,
                               frac=params.get("gate_floor_frac"),
                               base=params.get("gate_floor_base"))

    def execute(self, context, params: dict) -> SkillResult:
        tag = str(params.get("tag") or "").strip()
        if not tag:
            return SkillResult(skill_name="TrackBestFrame", success=False,
                               error="没给 tag —— 不同的追猎必须各记各的，"
                                     "共用一个记录会把两次实验混在一起。")
        dry_limit = int(params.get("dry_limit") or DEFAULT_DRY_LIMIT)
        ratio = float(params.get("better_ratio") or DEFAULT_BETTER_RATIO)
        peek = bool(params.get("peek", False))
        path = str(params.get("frame_path") or "")
        q_raw = params.get("quality")

        p = _store_path(tag)
        try:
            rec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        except Exception as exc:  # noqa: BLE001
            return SkillResult(
                skill_name="TrackBestFrame", success=False,
                error=(f"记录 {p} 读不懂：{exc} —— **不覆盖它**。"
                       f"一份读不懂的历史比没有历史更该让人来看一眼。"))
        entries: list[dict] = list(rec.get("entries") or [])
        best = rec.get("best") or None

        out: dict[str, Any] = {"tag": tag, "store": str(p)}

        if peek:
            # 模块级 :func:`peek` 是这段的唯一实现 —— 判据侧
            # (``mast.goals`` 的 best_frame_settled)调的是同一个函数,
            # 于是「收手了没」在模板与判据里不可能给出两个答案。
            out.update(peek_store(
                tag, dry_limit=dry_limit,
                gate_floor_frac=params.get("gate_floor_frac"),
                gate_floor_base=params.get("gate_floor_base")))
            return SkillResult(skill_name="TrackBestFrame", success=True, data=out,
                               summary=self._say(out))

        # ── 「判不了」既不算更好也不算更差 ──────────────────────────
        if q_raw is None:
            dry = int(rec.get("dry_rounds") or 0)
            out.update({"is_best": None, "recorded": False,
                        "best_path": (best or {}).get("frame_path"),
                        "best_quality": (best or {}).get("quality"),
                        "n_seen": len(entries), "dry_rounds": dry,
                        "dry_limit": dry_limit,
                        "good_enough_to_stop": _settled(dry, dry_limit),
                        "gate_floor": self._gate_floor(
                            params, (best or {}).get("quality")),
                        "note": ("这一轮判不了成色 —— **不推进 dry 计数**。"
                                 "把「判不了」记成「不够好」会让流程在一个读不到数的"
                                 "故障上收手。")})
            return SkillResult(skill_name="TrackBestFrame", success=True, data=out,
                               summary=self._say(out))

        q = float(q_raw)
        prev_best_q = float((best or {}).get("quality") or 0.0)
        is_best = (best is None) or (q > prev_best_q * ratio)
        dry = 0 if is_best else int(rec.get("dry_rounds") or 0) + 1

        entries.append({"frame_path": path, "quality": q, "at": time.time(),
                        "is_best": bool(is_best)})
        if is_best:
            best = {"frame_path": path, "quality": q, "at": time.time(),
                    "index": len(entries) - 1}
        rec = {"tag": tag, "entries": entries, "best": best, "dry_rounds": dry,
               "dry_limit": dry_limit, "better_ratio": ratio,
               "updated_at": time.time()}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)

        out.update({"is_best": bool(is_best), "recorded": True, "quality": q,
                    "previous_best_quality": (prev_best_q or None),
                    "best_path": (best or {}).get("frame_path"),
                    "best_quality": (best or {}).get("quality"),
                    "n_seen": len(entries), "dry_rounds": dry,
                    "dry_limit": dry_limit,
                    "good_enough_to_stop": _settled(dry, dry_limit),
                    "gate_floor": self._gate_floor(
                        params, (best or {}).get("quality"))})
        return SkillResult(skill_name="TrackBestFrame", success=True, data=out,
                           summary=self._say(out))

    @staticmethod
    def _say(out: dict) -> str:
        if out.get("note"):
            return str(out["note"])
        n = out.get("n_seen") or 0
        bq = out.get("best_quality")
        if out.get("is_best"):
            prev = out.get("previous_best_quality")
            return ("这一张更好（%.1f%s），成为目前最好的；已看过 %d 张，dry 归零。"
                    % (float(out.get("quality") or 0),
                       ("，此前 %.1f" % prev) if prev else "，是第一张", n))
        return ("这一张没超过现有最好的（%.1f vs %s）；连着 %d 张没更好%s。"
                % (float(out.get("quality") or 0),
                   ("%.1f" % bq) if bq else "—",
                   out.get("dry_rounds") or 0,
                   ("，**到线了：差不多可以收手**" if out.get("good_enough_to_stop")
                    else "，还没到 %d" % (out.get("dry_limit") or 0))))


def make_tools(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return [wrap_skill(TrackBestFrame, context_provider)]


__all__ = ["TrackBestFrame", "DEFAULT_DRY_LIMIT", "DEFAULT_BETTER_RATIO"]
