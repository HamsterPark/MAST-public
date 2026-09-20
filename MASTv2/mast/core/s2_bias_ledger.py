"""逐偏压账 —— append-only JSONL，键是**偏压**，不是帧序号。

设计文档:``docs/v2/design/`` 的「S2 偏压序列」设计,D4/§3.2/§3.3
(文件名按「S2 偏压序列」检索;通用层注释不写样品名前缀 ——
``stm_capability_vs_sample_layer.md`` 拍板④)。

纯数据层:不碰硬件、不导入任何技能、不知道 conduct 是什么。上层把数递进来。

## 为什么键是偏压而不是序号

``scan_planner.order_series_monotonic`` 会**重排**序列(偏压来回跳会反复激起
针尖-样品结的回滞,所以单调走一遍)。重排之后「第 3 帧」和「第 3 个偏压」不是
一回事,拿序号当键会把两个偏压的证据串到一起。(设计陷阱 13)

## 为什么键要规范化

``expand_series`` 的 linear 展开做的是 ``start + step * i``,于是 -0.3 会以
``-0.30000000000000004`` 的形式出现。浮点相等不可靠 ⇒ 同一个偏压会长出两把
钥匙,两把钥匙各记各的账,而汇总时谁也不知道它们是同一个偏压。(设计陷阱 12)

键 = ``round(bias_v, 6)`` 的定宽字符串(µV 分辨,远细于任何真实设置),
**同时**原样保存 ``bias_v`` 浮点 —— 键是给对齐用的,值是给物理用的。

## 为什么不建在 scan_registry 上

``core/scan_registry`` 的 ``_MAX_KEEP = 50``,按 scan_id 淘汰最旧。一次多天的
序列必然把早期的帧挤出去,而那些帧正是「这一段扫的是同一片原子」的证据。
⇒ 账里存**绝对路径**,顺带存 scan_id 便于交叉查。(设计陷阱 11)

## append-only 的一个后果:``stale`` 不是存出来的

记录一旦落盘就不再改写,所以文件里的 ``stale`` 永远是**写入那一刻**的值(False)。
真正的答案由 :func:`read_ledger` 在读的时候用当前 evidence_epoch 对账算出来。
当前代次读不到时 ``stale`` 是 **None**(对不了账),**不是 False** —— 把「没对上账」
说成「不陈旧」,正是本仓反复出事的那一类。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "s2_bias_ledger/1"
LEDGER_FILENAME = "s2_bias_ledger.jsonl"

# ── 单个偏压的终态(闭集)────────────────────────────────────────────────
STATE_RESOLVED = "bias_resolved"                 # 任一次 atomic_resolved
STATE_ABSENT_CONFIRMED = "bias_absent_confirmed"  # 全 absent + 至少一次帧准入过
STATE_UNDECIDED = "bias_undecided"               # 证据不足。**不是「没有」**
STATE_BLOCKED = "bias_blocked"                   # 安全侧拒绝,不是物理结论
STATE_TIP_ABORTED = "bias_tip_aborted"           # 帧间守卫中止,等 S1 绕道
FINAL_STATES = (STATE_RESOLVED, STATE_ABSENT_CONFIRMED, STATE_UNDECIDED,
                STATE_BLOCKED, STATE_TIP_ABORTED)

#: 有信息量的终态 —— 「这个偏压我们得到了一个科学结论」。
#: ``absent_confirmed`` 在这里面:一块样品在某些偏压上本来就可以没有原子对比度
#: (态密度在那个能量上没有可成像的周期结构),那是一个结论,不是一次失败。
INFORMATIVE_STATES = (STATE_RESOLVED, STATE_ABSENT_CONFIRMED)

# ── 整个序列的出口(闭集)──────────────────────────────────────────────
EXIT_COMPLETE = "series_complete"
EXIT_PARTIAL = "series_partial"
EXIT_ABORTED_TIP = "series_aborted_tip"
EXIT_ABORTED_BUDGET = "series_aborted_budget"
EXIT_BLOCKED = "series_blocked"
SERIES_EXITS = (EXIT_COMPLETE, EXIT_PARTIAL, EXIT_ABORTED_TIP,
                EXIT_ABORTED_BUDGET, EXIT_BLOCKED)

# ── 单次尝试的裁决(转自 VerifyAtomicResolution,原样不翻译)──────────
VERDICT_RESOLVED = "atomic_resolved"
VERDICT_ABSENT = "atomic_absent"
VERDICT_UNDECIDABLE = "undecidable"

#: 一次尝试**没能产出裁决**时的出口。与三个 verdict 并列,不混用。
OUTCOME_BLOCKED = "blocked"        # 安全门/包络/BiasSettleChange 拒了
OUTCOME_TIP_ABORT = "tip_abort"    # 帧间守卫中止

ROLE_SERIES = "series"
ROLE_ANCHOR = "anchor"
ROLE_RESCAN = "rescan"


# ══════════════════════════════════════════════════════════════════════
# 键与显示
# ══════════════════════════════════════════════════════════════════════

def bias_key(bias_v: float) -> str:
    """规范化的偏压键。µV 分辨,定宽,零永远只有一种写法。

    ``-0.0`` 要特别处理:``round(-1e-9, 6)`` 得到 ``-0.0``,而
    ``f"{-0.0:.6f}"`` 是 ``"-0.000000"`` —— 与 ``"0.000000"`` 是两把钥匙,
    指着同一个偏压。加 ``0.0`` 把负零折平。
    """
    v = round(float(bias_v), 6) + 0.0
    return f"{v:.6f}"


def bias_human(bias_v: float) -> str:
    """给人读的偏压,如 ``"-300 mV"`` / ``"1.5 V"``。

    **刻意不走** :func:`mast.core.si_quantity.format_si`。那个函数的契约是
    「印出来的东西要能原样抄回参数里」,所以它把无前缀量级降一档:2 V 会印成
    ``2000m``。``agents/_shared/live_state_mw.py:41-48`` 已经为偏压做过同一个
    判断并写下理由(「偏压天然在 1 附近,普通小数就是它的自然写法」)。
    这里跟随那个既有判断,不另立第二套。

    权威的数是 ``bias_v``(SI 伏特),这一栏只是它的可读伴随。
    """
    v = float(bias_v)
    if v == 0:
        return "0 V"
    if abs(v) < 1.0:
        mv = v * 1e3
        return f"{mv:g} mV"
    return f"{v:g} V"


def ledger_path(base_dir: "str | os.PathLike") -> Path:
    """``<实验/样品/conduct 目录>/.mast/s2_bias_ledger.jsonl``。"""
    return Path(base_dir) / ".mast" / LEDGER_FILENAME


# ══════════════════════════════════════════════════════════════════════
# 写
# ══════════════════════════════════════════════════════════════════════

def make_record(
    *,
    bias_v: float,
    attempt: int,
    role: str = ROLE_SERIES,
    conduct_id: str = "",
    stage_id: str = "",
    frame_path: "str | None" = None,
    scan_id: "str | None" = None,
    nominal_center_m: "tuple[float, float] | list[float] | None" = None,
    actual_center_m: "tuple[float, float] | list[float] | None" = None,
    size_m: "float | None" = None,
    pixels: "int | None" = None,
    line_time_s: "float | None" = None,
    tier_name: str = "",
    resolver_warnings: "Iterable[str] | None" = None,
    verdict: "str | None" = None,
    outcome: "str | None" = None,
    remedy: "str | None" = None,
    metrics: "dict | None" = None,
    frame_admission_passed: "bool | None" = None,
    coord_epoch: "int | None" = None,
    evidence_epoch: "int | None" = None,
    tip_fingerprint: "dict | None" = None,
    profile_name: "str | None" = None,
    profile_provenance: "str | None" = None,
    drift_shift_px: "float | None" = None,
    drift_flag: "str | None" = None,
    xy_calibration_ref: "str | None" = None,
    z_calibration_note: "str | None" = None,
    note: str = "",
) -> dict:
    """一条**尝试**记录。纯构造,不落盘 —— 好让调用方与测试都能先看再写。

    ``verdict`` 与 ``outcome`` 是两栏:前者是判据说的话(三态之一),后者是
    「压根没走到判据」的出口(blocked / tip_abort)。**不合并** —— 合并之后
    「安全门拒了」和「判据说没有原子」在账里长得一模一样。
    """
    return {
        "schema": SCHEMA,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conduct_id": str(conduct_id or ""),
        "stage_id": str(stage_id or ""),
        "bias_key": bias_key(bias_v),
        "bias_v": float(bias_v),
        "bias_human": bias_human(bias_v),
        "attempt": int(attempt),
        "role": str(role),
        "frame_path": (str(frame_path) if frame_path else None),
        "scan_id": (str(scan_id) if scan_id else None),
        "nominal_center_m": _pair(nominal_center_m),
        "actual_center_m": _pair(actual_center_m),
        "size_m": _num_or_none(size_m),
        "pixels": (int(pixels) if pixels is not None else None),
        "line_time_s": _num_or_none(line_time_s),
        "tier_name": str(tier_name or ""),
        # resolver 的告警**原样收进来**。`purpose` 强制换档的那条 warning 是有
        # 信息的(设计陷阱 19):它说的是「你要的档位这个尺度上不成立」。
        "resolver_warnings": [str(w) for w in (resolver_warnings or ())],
        "verdict": (str(verdict) if verdict else None),
        "outcome": (str(outcome) if outcome else None),
        "remedy": (str(remedy) if remedy else None),
        "metrics": dict(metrics or {}),
        "frame_admission_passed": (None if frame_admission_passed is None
                                   else bool(frame_admission_passed)),
        "coord_epoch": _int_or_none(coord_epoch),
        "evidence_epoch": _int_or_none(evidence_epoch),
        # 写入那一刻它当然不陈旧。真正的答案是读的时候算的 —— 见模块说明。
        "stale": False,
        "tip_fingerprint": dict(tip_fingerprint or {}),
        "profile_name": (str(profile_name) if profile_name else None),
        "profile_provenance": (str(profile_provenance)
                               if profile_provenance else None),
        "drift_shift_px": _num_or_none(drift_shift_px),
        "drift_flag": (str(drift_flag) if drift_flag else None),
        "xy_calibration_ref": (str(xy_calibration_ref)
                               if xy_calibration_ref else None),
        "z_calibration_note": (str(z_calibration_note)
                               if z_calibration_note else None),
        "note": str(note or ""),
    }


def append(path: "str | os.PathLike", record: dict) -> Path:
    """把一条记录追加进 JSONL。**只追加,永不覆盖、永不重写已有行。**

    同一个偏压的第二次尝试是**第二条记录**,不是把第一条改掉 —— 覆盖会把
    「试了两次」抹成「试了一次」,而序列的可信度恰恰建在尝试史上。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=_jsonable)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return p


# ══════════════════════════════════════════════════════════════════════
# 读
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LedgerRead:
    """读账的结果。**坏行不丢也不猜** —— 单列一栏。"""

    rows: list[dict] = field(default_factory=list)
    unreadable: list[dict] = field(default_factory=list)
    path: str = ""
    exists: bool = False


def read_ledger(path: "str | os.PathLike", *,
                current_evidence_epoch: "int | None" = None) -> LedgerRead:
    """读全账,并按当前 evidence_epoch 对账算出每行的 ``stale``。

    ``current_evidence_epoch is None`` ⇒ 每行的 ``stale`` 是 **None**(对不了账),
    不是 False。闸门看到 None 必须当成「不知道」,不能当成「新鲜」。
    """
    p = Path(path)
    out = LedgerRead(path=str(p), exists=p.is_file())
    if not out.exists:
        return out
    try:
        raw = p.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        out.unreadable.append({"what": str(p), "why": f"读不开:{exc}"})
        return out
    for i, line in enumerate(raw, 1):
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except ValueError as exc:
            # 半截行(写到一半断电/断进程)不许静默丢:丢掉一条 resolved,
            # 汇总就会把那个偏压报成 undecided,而那是一句假话。
            out.unreadable.append(
                {"what": f"{p}:{i}", "why": f"这一行不是合法 JSON:{exc}"})
            continue
        if not isinstance(row, dict):
            out.unreadable.append(
                {"what": f"{p}:{i}", "why": f"这一行不是对象:{type(row).__name__}"})
            continue
        row["stale"] = _staleness(row, current_evidence_epoch)
        out.rows.append(row)
    return out


def _staleness(row: dict, current: "int | None") -> "bool | None":
    if current is None:
        return None                     # 对不了账 —— 不是「不陈旧」
    e = _int_or_none(row.get("evidence_epoch"))
    if e is None:
        return None                     # 这一行没带代次 —— 同样是对不了账
    return e != int(current)


# ══════════════════════════════════════════════════════════════════════
# 派生:每偏压终态
# ══════════════════════════════════════════════════════════════════════

def derive_final_state(attempts: "list[dict]") -> str:
    """一个偏压的终态。**优先级是这个函数的全部内容**,逐条写明理由。

    1. 任何一次 ``atomic_resolved`` ⇒ ``bias_resolved``。拿到过就是拿到过。
    2. 否则任何一次 ``tip_abort`` ⇒ ``bias_tip_aborted``。针尖出过事,这个偏压
       上的其余证据都得在 S1 绕道之后重新看 —— 不能先替它下结论。
    3. 否则完全没有裁决、且有过 ``blocked`` ⇒ ``bias_blocked``。安全侧拒绝是
       **拒绝**,不是「这里没有原子」。
    4. 否则所有裁决都是 ``atomic_absent`` **且**至少有一次帧准入通过
       ⇒ ``bias_absent_confirmed``。这是有信息量的科学结论。
    5. 其余一律 ``bias_undecided``。

    第 5 条是**兜底沉淀池**,这是刻意的。设计表里 ``absent_confirmed`` 要求
    「全部尝试都是 absent」,``undecided`` 要求「从未拿到过 absent」——
    于是 ``[absent, undecidable]`` 这种混合两条都不满足,闭集在那里有个洞。
    往哪边补是有代价差的:补成 absent_confirmed 是拿不足的证据去下
    「这里没有原子」的结论,补成 undecided 只是说「证据不够」。
    **不对称的代价决定方向** —— 洞往 undecided 补。
    """
    if not attempts:
        return STATE_UNDECIDED
    verdicts = [r.get("verdict") for r in attempts if r.get("verdict")]
    outcomes = [r.get("outcome") for r in attempts if r.get("outcome")]

    if any(v == VERDICT_RESOLVED for v in verdicts):
        return STATE_RESOLVED
    if any(o == OUTCOME_TIP_ABORT for o in outcomes):
        return STATE_TIP_ABORTED
    if not verdicts and any(o == OUTCOME_BLOCKED for o in outcomes):
        return STATE_BLOCKED
    if verdicts and all(v == VERDICT_ABSENT for v in verdicts) and any(
            r.get("frame_admission_passed") is True for r in attempts):
        return STATE_ABSENT_CONFIRMED
    return STATE_UNDECIDED


def by_bias(rows: "Iterable[dict]") -> "dict[str, list[dict]]":
    """按偏压键分组,组内按 attempt 升序。role=anchor 的行**也在里面**
    (它扫的就是起始偏压,是那个偏压的又一次尝试)。"""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("bias_key") or ""), []).append(r)
    for k in out:
        out[k].sort(key=lambda r: (_int_or_none(r.get("attempt")) or 0))
    return out


# ══════════════════════════════════════════════════════════════════════
# 派生:汇总 + 证据包
# ══════════════════════════════════════════════════════════════════════

def summarize(read: LedgerRead, *,
              planned_biases: "Iterable[float] | None" = None,
              fresh_only: bool = True) -> dict:
    """五栏分列的汇总。``undecided`` 与 ``absent`` **绝不合并**。

    ``planned_biases`` 给了就用它当分母:计划里有、账里一条记录都没有的偏压
    **不计进任何一栏**,而是进 ``unreadable``(「没跑到」也是一种读不到)。
    于是恒等式成立并被测试钉住::

        n_resolved + n_absent_confirmed + n_undecided + n_blocked
            + n_tip_aborted + len(unreadable_biases) == n_biases

    ``fresh_only`` 为真时只认 ``stale is False`` 的行。``stale is None``
    (对不了账)**既不算新鲜也不算陈旧** —— 它进 ``unreadable``。
    """
    unreadable: list[dict] = list(read.unreadable)
    usable: list[dict] = []
    for r in read.rows:
        st = r.get("stale")
        if not fresh_only:
            usable.append(r)
        elif st is False:
            usable.append(r)
        elif st is None:
            unreadable.append({
                "what": f"{r.get('bias_human')} attempt {r.get('attempt')}",
                "why": "这一行对不上 evidence_epoch(当前代次或行内代次读不到)"})
        # st is True ⇒ 陈旧,不删不报错,只是不参与汇总

    groups = by_bias(usable)
    per_bias: list[dict] = []
    counts = {s: 0 for s in FINAL_STATES}

    keys_planned: "list[str] | None" = None
    if planned_biases is not None:
        keys_planned = []
        for b in planned_biases:
            k = bias_key(b)
            if k not in keys_planned:
                keys_planned.append(k)

    for k in (keys_planned if keys_planned is not None else sorted(groups)):
        attempts = groups.get(k) or []
        if not attempts:
            unreadable.append({
                "what": bias_human(float(k)),
                "why": "计划里有这个偏压,账里一条记录都没有 —— 没跑到"})
            continue
        state = derive_final_state(attempts)
        counts[state] += 1
        resolved_at = next((r.get("ts") for r in attempts
                            if r.get("verdict") == VERDICT_RESOLVED), None)
        per_bias.append({
            "bias_key": k,
            "bias_human": attempts[-1].get("bias_human") or bias_human(float(k)),
            "final_state": state,
            "attempts_used": len(attempts),
            "last_frame_path": attempts[-1].get("frame_path"),
            "first_resolved_at": resolved_at,
            "evidence_epoch": attempts[-1].get("evidence_epoch"),
        })

    # 计划给了就以计划为分母(缺的那些进 unreadable);没给就只能以账里有的为准,
    # 此时「计划了多少」这件事本身是读不到的 —— 分母不许凭空补。
    n_biases = (len(keys_planned) if keys_planned is not None else len(per_bias))
    # 算**一次**,两个字段都由它派生 —— 嵌套详情与顶层裁决是同一份证据的两种读法,
    # 各算一遍就会有两个答案。
    ac = anchor_consistency(usable)
    return {
        "n_biases": n_biases,
        "n_resolved": counts[STATE_RESOLVED],
        "n_absent_confirmed": counts[STATE_ABSENT_CONFIRMED],
        "n_undecided": counts[STATE_UNDECIDED],
        "n_blocked": counts[STATE_BLOCKED],
        "n_tip_aborted": counts[STATE_TIP_ABORTED],
        "per_bias": per_bias,
        "anchor_consistency": ac,
        # 同一份证据的**顶层名字**。闸门的 rule 叶子读不了上面那个嵌套 dict ——
        # 在这个字段之前,S2 出口闸门只能放弃判「全部 absent_confirmed」那条路
        # (那是一条科学结论,不是失败),每一次都转人。
        "anchor_verdict": anchor_verdict(ac),
        "unreadable": unreadable,
    }


def anchor_consistency(rows: "Iterable[dict]") -> dict:
    """锚点帧的一致性 —— 「整段扫的是同一片原子」唯一的事后证据。

    没有锚点帧,这句话就是一个无从证明的断言,所以 ``n_anchors == 0`` 要能被
    上层看见(而不是表现成「一致」)。
    """
    anchors = [r for r in rows if r.get("role") == ROLE_ANCHOR]
    drifted = sorted({str(r.get("bias_human") or r.get("bias_key"))
                      for r in rows if r.get("drift_flag")})
    return {
        "n_anchors": len(anchors),
        "n_anchor_resolved": sum(1 for r in anchors
                                 if r.get("verdict") == VERDICT_RESOLVED),
        "drifted_segments": drifted,
    }


#: 「整段扫的是同一片原子」这句话的**闭集**裁决(2026-08-15 加)。
#:
#: 为什么是四个值而不是一个布尔:conduct 闸门的 rule 叶子读不了嵌套字段,
#: 所以这个结论必须有一个顶层名字。而扁平化最容易顺手做错的一步,是把它压成
#: ``bool`` —— 那样「一张锚点帧都没拍」与「拍了但一张都没判出」会一起变成
#: ``False``,和「锚点说这不是同一片」也变成同一个值。三件事下一步要做的完全不同:
#: 没拍是**补不回来**的(那一段已经扫完了),拍了没判出**可以补拍一张**,
#: 而漂移是一条**实打实的否定证据**。
ANCHOR_VERDICTS = (
    "corroborated",                 # 至少一张锚点帧判出原子分辨,且没有漂移标记
    "contradicted",                 # 检测到漂移 ⇒ 扫的不是同一片
    "unproven_no_anchor",           # 一张锚点帧都没有 ⇒ 无从证明,且补不回来
    "unproven_anchor_unresolved",   # 拍了锚点但一张都没判出 ⇒ 无从证明,可补拍
)


def anchor_verdict(anchors: "dict | None") -> str:
    """把 :func:`anchor_consistency` 压成一个闭集裁决(:data:`ANCHOR_VERDICTS`)。

    **漂移优先**:哪怕锚点帧自己判出了原子分辨,只要序列里有任何一段带漂移标记,
    「整段扫的是同一片」就已经被否证了 —— 锚点证明的是「回到起始偏压还能看见原子」,
    不是「中间那些帧没跑偏」。
    """
    a = anchors or {}
    if a.get("drifted_segments"):
        return "contradicted"
    if int(a.get("n_anchors") or 0) <= 0:
        return "unproven_no_anchor"
    if int(a.get("n_anchor_resolved") or 0) <= 0:
        return "unproven_anchor_unresolved"
    return "corroborated"


def series_exit(summary: dict, *,
                aborted_tip: bool = False,
                budget_exhausted: bool = False) -> str:
    """整个序列的出口(闭集 :data:`SERIES_EXITS`)。

    ``series_complete`` 要求**每一个计划中的偏压都拿到了有信息量的终态**
    (resolved 或 absent_confirmed)且没有读不到的。
    「部分成功 = 成功,但缺口必须写进 summary」是既有纪律,所以缺口存在时
    出口是 ``series_partial`` —— 它仍然不是失败,但它不许长得像 complete。
    """
    n = int(summary.get("n_biases") or 0)
    informative = (int(summary.get("n_resolved") or 0)
                   + int(summary.get("n_absent_confirmed") or 0))
    if aborted_tip:
        return EXIT_ABORTED_TIP
    if n and int(summary.get("n_blocked") or 0) == n:
        return EXIT_BLOCKED
    if budget_exhausted and informative < n:
        return EXIT_ABORTED_BUDGET
    if n and informative == n and not summary.get("unreadable"):
        return EXIT_COMPLETE
    return EXIT_PARTIAL


# ══════════════════════════════════════════════════════════════════════
def _pair(v) -> "list[float] | None":
    if v is None:
        return None
    try:
        return [float(v[0]), float(v[1])]
    except (TypeError, ValueError, IndexError):
        return None


def _num_or_none(v) -> "float | None":
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None        # NaN → None


def _int_or_none(v) -> "int | None":
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v: Any) -> Any:
    """落盘兜底:不认识的东西留 repr,**不让一条记录因为一个字段丢掉整行**。"""
    try:
        return float(v) if isinstance(v, (int, float)) else repr(v)
    except Exception:  # noqa: BLE001
        return repr(v)


__all__ = [
    "SCHEMA", "LEDGER_FILENAME",
    "FINAL_STATES", "INFORMATIVE_STATES", "SERIES_EXITS",
    "STATE_RESOLVED", "STATE_ABSENT_CONFIRMED", "STATE_UNDECIDED",
    "STATE_BLOCKED", "STATE_TIP_ABORTED",
    "EXIT_COMPLETE", "EXIT_PARTIAL", "EXIT_ABORTED_TIP",
    "EXIT_ABORTED_BUDGET", "EXIT_BLOCKED",
    "VERDICT_RESOLVED", "VERDICT_ABSENT", "VERDICT_UNDECIDABLE",
    "OUTCOME_BLOCKED", "OUTCOME_TIP_ABORT",
    "ROLE_SERIES", "ROLE_ANCHOR", "ROLE_RESCAN",
    "ANCHOR_VERDICTS", "anchor_verdict",
    "LedgerRead", "anchor_consistency", "append", "bias_human", "bias_key",
    "by_bias", "derive_final_state", "ledger_path", "make_record",
    "read_ledger", "series_exit", "summarize",
]
