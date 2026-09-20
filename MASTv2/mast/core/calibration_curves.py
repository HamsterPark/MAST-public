"""标定参考曲线:存、查、以及**在条件不匹配时拒绝翻译**。

## 归属:仪器域,不是实验域

一条「退针距离 ↔ lock-in 串扰」曲线描述的是**这台机器 + 这根针 + 这块样品**,不是
某一次实验。落 ``<project_root>/calibration/``,与实验记录分开(同一
原则)。曲线里带 ``experiment_id`` 只是**溯源**——「这条是在哪次实验里采的」——
不是归属:实验删了曲线还在,曲线不该跟着某次实验消失。

## 定位:参考曲线,不是权威默认

与知识库同一条纪律(``knowledge_not_ground_truth``):**这是某机某针某日的实测,
不是物理常数**。换针、换样品、换调制参数都要重采。所以:

* 按 ``(样品体系, 针尖类型)`` 分 profile,**只内建实测过的**(现在只有真机
  那一条,作「出厂示例」);没数据就不造 —— 造一条看起来合理的曲线,比没有曲线坏得多,
  因为它会被当成依据;
* 展示层必须明示采集条件与日期(``describe()`` 给的就是那句话);
* **消费侧引用前必须校验条件**(见 :func:`match_conditions`)。

## 最要紧的一条:条件对不上就不翻译

把串扰读数翻译成「距离量级」的前提是**曲线的采集条件与此刻一致**。调制幅度、频率、
针尖类型任何一个对不上,同一个读数对应的距离可以差一个量级。所以
:func:`translate` 在条件不匹配时**返回 None 并说明**,由调用方原样显示读数 ——
**不给一个静默的假距离**。这是本模块唯一真正危险的出口,其余都是读写。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from mast._runtime_paths import project_root

logger = logging.getLogger(__name__)

#: 已知的曲线种类。加新种类要同时想清楚它的 conditions 里哪些是**可比性条件**
#: (对不上就不能比),写进 :data:`MATCH_KEYS`。
KIND_RETRACT_LOCKIN = "retract_lockin_curve"

#: 每种曲线的**可比性条件**——两条曲线/一次读数要能比,这些必须一致。
#:
#: 不是「conditions 里所有的键」:温度和 setpoint 记下来是为了让人读懂,而
#: **调制幅度/频率/针尖类型**是物理上决定「同一个串扰读数对应多远」的那几个。
#: 把无关键放进来会让匹配永远失败(等于关掉功能);把关键键漏掉会让它静默给假距离。
MATCH_KEYS: "dict[str, tuple[str, ...]]" = {
    KIND_RETRACT_LOCKIN: ("mod_amp_v", "mod_freq_hz", "tip_type"),
}

#: 数值条件的相对容差。频率/幅度对不上 1% 以内算同一条件。
_REL_TOL = 0.01

_SAFE_NAME = re.compile(r"[^\w.-]+")


def calibration_dir(*, create: bool = False) -> Path:
    """曲线库目录。仪器域 —— 与实验记录分开。"""
    d = project_root() / "calibration"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(text: str) -> str:
    return _SAFE_NAME.sub("_", str(text or "")).strip("_") or "unnamed"


def save_curve(curve: "dict[str, Any]", *, overwrite: bool = False) -> Path:
    """把一条曲线落盘。返回路径。

    文件名带 kind + 样品 + 针型 + 日期,好让人在文件管理器里也认得出来;真正的
    身份仍然是文件内容(kind + conditions),不是文件名。
    """
    kind = str(curve.get("kind") or "").strip()
    if kind not in MATCH_KEYS:
        raise ValueError(
            f"未知曲线种类 {kind!r};已知:{sorted(MATCH_KEYS)}。"
            "加新种类时要同时决定它的可比性条件(MATCH_KEYS),否则消费侧无法判断"
            "「这条曲线现在能不能用」。")
    if not curve.get("points"):
        raise ValueError("曲线没有数据点 —— 空曲线不落盘(它只会被当成一条已有的曲线)。")

    cond = curve.get("conditions") or {}
    meta = curve.get("meta") or {}
    name = "_".join(_slug(x) for x in (
        kind, cond.get("sample", "sample"), cond.get("tip_type", "tip"),
        meta.get("date", "undated")))
    d = calibration_dir(create=True)
    path = d / f"{name}.json"
    if path.exists() and not overwrite:
        i = 2
        while (d / f"{name}_{i}.json").exists():
            i += 1
        path = d / f"{name}_{i}.json"
    path.write_text(json.dumps(curve, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    logger.info("标定曲线已保存:%s(%d 点)", path.name, len(curve["points"]))
    return path


def load_curves(kind: "str | None" = None) -> "list[dict[str, Any]]":
    """读出全部曲线(可按 kind 过滤)。坏文件跳过并记日志,不让一份坏 JSON 废掉整库。"""
    out: "list[dict[str, Any]]" = []
    d = calibration_dir()
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("标定曲线 %s 读不出来(跳过):%s", p.name, exc)
            continue
        if not isinstance(data, dict) or not data.get("points"):
            continue
        if kind and str(data.get("kind")) != kind:
            continue
        data["_path"] = str(p)
        out.append(data)
    return out


def describe(curve: "dict[str, Any]") -> str:
    """一句给人看的出处 —— **展示层必须带上它**。

    「这是某机某针某日的实测」这句话不写出来,曲线看起来就像一条物理常数。
    """
    c = curve.get("conditions") or {}
    m = curve.get("meta") or {}
    bits = [f"实测于 {m.get('date', '未记日期')}"]
    if c.get("sample"):
        bits.append(f"样品 {c['sample']}")
    if c.get("tip_type"):
        bits.append(f"针尖 {c['tip_type']}")
    if c.get("mod_amp_v") is not None and c.get("mod_freq_hz") is not None:
        bits.append(f"调制 {c['mod_amp_v']} V @ {c['mod_freq_hz']} Hz")
    if c.get("temperature_k") is not None:
        bits.append(f"{c['temperature_k']} K")
    return ("、".join(bits)
            + " —— **参考曲线,不是出厂常数**;换针/换样品/换调制参数请重采。")


def match_conditions(curve: "dict[str, Any]",
                     now: "dict[str, Any]") -> "tuple[bool, str]":
    """此刻的条件能不能用这条曲线。``(可用, 说明)``。

    **缺一不可,而且「读不到」算不匹配** —— 一个没读到的调制幅度不是「大概一样」,
    它是「不知道」,而不知道的时候翻译出来的距离没有任何依据。
    """
    kind = str(curve.get("kind") or "")
    keys = MATCH_KEYS.get(kind)
    if not keys:
        return False, f"未知曲线种类 {kind!r},无法判断可比性。"
    cond = curve.get("conditions") or {}
    bad: list[str] = []
    for k in keys:
        want, got = cond.get(k), now.get(k)
        if want is None or got is None:
            bad.append(f"{k}:曲线={want!r} 当前={got!r}(有一侧读不到)")
            continue
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            tol = max(abs(float(want)) * _REL_TOL, 1e-12)
            if abs(float(want) - float(got)) > tol:
                bad.append(f"{k}:曲线={want} 当前={got}")
        elif str(want).strip().lower() != str(got).strip().lower():
            bad.append(f"{k}:曲线={want!r} 当前={got!r}")
    if bad:
        return False, "条件不匹配(" + ";".join(bad) + ")"
    return True, "条件匹配"


def curve_keys(curve: "dict[str, Any]", *, value_key: "str | None" = None,
               distance_key: "str | None" = None) -> "tuple[str, str]":
    """这条曲线的 (值键, 距离键)。**优先用曲线自己声明的**。

    调用方写死键名 = 又一个跨文件的名字耦合,而这类耦合坏掉时的样子已经见过:
    查不到 ⇒ 一个看起来合理的空结果。曲线自己带着 ``value_key`` / ``distance_key``,
    换一种曲线就不必去改每个消费点。
    """
    v = value_key or str(curve.get("value_key") or "") or "y"
    d = distance_key or str(curve.get("distance_key") or "") or "cum_steps"
    return v, d


def curve_problems(curve: "dict[str, Any]") -> "list[str]":
    """这条曲线**永远**用不上的理由(空列表 = 没问题)。

    为什么要有这一条:``match_conditions`` 把「曲线里缺 tip_type」和「今天的针尖跟
    曲线对不上」报成同一句「条件不匹配」。前者是**数据缺陷**(这条曲线永远匹配不上,
    功能等于没开),后者是**正常的运行时状态**。两者报成一句话,一个永远空转的功能
    看起来就和一个正常工作、只是今天条件不符的功能一模一样。

    实例:2026-08-05 采的那条退针曲线,conditions 里写的是 ``mod_amplitude_v`` /
    ``mod_frequency_hz``,而 MATCH_KEYS 要的是 ``mod_amp_v`` / ``mod_freq_hz``,
    ``tip_type`` 则根本在 conditions 之外 —— 三处对不上,翻译会永远拒绝,而每一次
    拒绝的措辞都合情合理。
    """
    out: "list[str]" = []
    kind = str(curve.get("kind") or "")
    keys = MATCH_KEYS.get(kind)
    if not keys:
        return [f"未知曲线种类 {kind!r};已知:{sorted(MATCH_KEYS)}"]
    cond = curve.get("conditions") or {}
    missing = [k for k in keys if cond.get(k) is None]
    if missing:
        out.append(
            f"conditions 里缺可比性条件 {missing} —— 这条曲线**永远匹配不上**"
            f"(不是今天条件不符)。可比性条件:{list(keys)}")
    pts = curve.get("points") or []
    if len(pts) < 2:
        # 措辞保留「不足两个」:老钉子按这句话认这个拒绝,而这次改动只是把它从
        # translate 里挪到这儿,判据没变 —— 那就不该顺手换掉它的说法。
        out.append(f"可用点不足两个(只有 {len(pts)} 个),插不了值")
    else:
        vk, dk = curve_keys(curve)
        usable = sum(1 for p in pts
                     if isinstance(p.get(vk), (int, float))
                     and isinstance(p.get(dk), (int, float)))
        if usable < 2:
            out.append(
                f"点里没有可用的 {vk!r}/{dk!r} 两列(只有 {usable} 个点两列都在);"
                f"实际有的列:{sorted(pts[0]) if pts else []}")
    return out


def translate(curve: "dict[str, Any]", value: float, *,
              now: "dict[str, Any]", value_key: "str | None" = None,
              distance_key: "str | None" = None) -> "tuple[float | None, str]":
    """把一个读数翻译成曲线上的距离量级。``(距离 or None, 说明)``。

    ⚠️ **条件不匹配就返回 None** —— 调用方应当原样显示读数,并说明为什么没翻译。
    静默给一个假距离,比不给距离坏得多:它看起来是一条依据。

    线性插值,**不外推**:落在曲线两端之外时同样返回 None。曲线只描述它采过的那一段,
    「10000 步以外是什么样」这条曲线答不了。
    """
    broken = curve_problems(curve)
    if broken:
        # 数据缺陷要与「今天条件不符」分开报 —— 一个永远空转的功能不该长得像一个
        # 正常工作、只是今天条件不符的功能。
        return None, "这条曲线不可用(数据问题):" + ";".join(broken)
    ok, why = match_conditions(curve, now)
    if not ok:
        return None, why + " —— **不翻译**,请按原始读数使用。"
    value_key, distance_key = curve_keys(curve, value_key=value_key,
                                         distance_key=distance_key)
    pts = [(p.get(value_key), p.get(distance_key)) for p in curve.get("points", [])]
    pairs = sorted(((float(v), float(d)) for v, d in pts
                    if isinstance(v, (int, float)) and isinstance(d, (int, float))),
                   key=lambda t: t[0])
    if len(pairs) < 2:
        return None, "曲线上可用的点不足两个,插不了值。"
    lo, hi = pairs[0][0], pairs[-1][0]
    if not (lo <= value <= hi):
        return None, (f"读数 {value:.4g} 落在曲线覆盖范围 [{lo:.4g}, {hi:.4g}] 之外 —— "
                      "**不外推**。这条曲线只描述它采过的那一段。")
    for (v0, d0), (v1, d1) in zip(pairs, pairs[1:]):
        if v0 <= value <= v1:
            if v1 == v0:
                return d0, "曲线在此处为平段,取端点。"
            frac = (value - v0) / (v1 - v0)
            return d0 + frac * (d1 - d0), f"{describe(curve)}(线性插值)"
    return None, "插值失败(曲线点位异常)。"


__all__ = [
    "KIND_RETRACT_LOCKIN", "MATCH_KEYS", "calibration_dir", "save_curve",
    "load_curves", "describe", "match_conditions", "translate",
    "curve_keys", "curve_problems",
]
