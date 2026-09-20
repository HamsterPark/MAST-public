"""Commissioning report: turn real-instrument data into calibrated thresholds.

Every default in :mod:`mast.monitoring.thresholds` was set against SYNTHETIC
traces. They are physically shaped and the detectors were tuned so that pure
noise raises nothing, but the absolute levels — how many picoamps of noise a
healthy tip on THIS instrument actually shows, how much mains it picks up — are
guesses until the hardware says otherwise. Running against uncalibrated numbers
is how a monitor ends up either silent or crying wolf, and either way switched
off.

This reads what the daemon has already recorded and reports:

* the commissioning facts that only the instrument can answer (achieved sample
  rate, buffer depth, the timebase table, whether the pump is keeping up);
* the observed distribution of every feature, split by what the instrument was
  doing — a segment taken while scanning is not comparable with one taken on a
  parked tip, and mixing them produces a threshold that fits neither;
* for each alert threshold, whether it would fire on this data, with a suggested
  value derived from the observed spread.

It never writes settings. The suggestion is an input to a decision, not the
decision — an operator who knows the instrument should look at the numbers
before anything is applied.

Usage (locally, or over Tailscale against the instrument's API)::

    python -m mast.monitoring.commission                 # local store
    python -m mast.monitoring.commission --url http://<rig-host>:7870 \\
        --auth user:pass --hours 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Iterable, Optional

#: Features worth a suggested operating point, and the knob each one drives.
#: Only WARN-level knobs: the three CRITICAL rules are statements about the
#: instrument (railed, frozen, huge step) and are not calibrated from a healthy
#: baseline — a rail is a rail whatever the noise floor happens to be.
_CALIBRATABLE: tuple[tuple[str, str, str], ...] = (
    ("rms_detrended_a", "cm_rms_warn_a", "去趋势噪声 RMS"),
    ("spike_max_sigma", "cm_spike_sigma_warn", "尖峰(σ)"),
    ("line_ratio", "cm_line_ratio_warn", "工频比"),
    ("jump_rate_hz", "cm_jump_rate_warn_hz", "跳变率"),
    ("rtn_score", "cm_rtn_score_warn", "RTN 判分"),
)

#: Which WARN rule each calibratable feature feeds. Only needed to answer one
#: question — "is this feature judged while scanning?" — so it is a plain map
#: rather than a fourth element on ``_CALIBRATABLE`` (whose 3-tuple shape is
#: unpacked in several places, tests included).
_FEATURE_RULE: dict[str, str] = {
    "rms_detrended_a": "rms_high",
    "spike_max_sigma": "spike_warn",
    "line_ratio": "line_hum",
    "jump_rate_hz": "jump_burst",
    "rtn_score": "rtn_bistable",
}

#: Quantile the suggestion is anchored on, plus the headroom multiplier.
#: p99 of a healthy baseline, times a margin: a threshold at the observed
#: maximum fires on the next slightly worse segment; one at the median fires
#: constantly. The margin is deliberately generous because the cost of a false
#: WARN (an operator who stops reading them) exceeds a slightly late true one.
_ANCHOR_Q = 99.0
_HEADROOM = 1.5

#: `cm_sat_current_a` 的出厂默认。单独抄一份是为了能回答「这个值有没有被人改过」——
#: 直接 import thresholds 的默认会让本模块在冻结版里多一条 import，而这里只需要一个
#: 数。两者若漂开，test_commission 会红（钉在 test_sat_default_matches_thresholds）。
_SAT_SHIPPED_DEFAULT = 90e-9

# Z 与振幅使用各自量纲的阈值，基线只取不扫描、反馈开启且无技能占用的样本。
_AUX_CALIBRATABLE: tuple[tuple[str, str, str, bool], ...] = (
    # (aux 列名, knob, 标签, 取绝对值)
    ("z_drift_m_per_s", "cm_z_drift_warn_m_per_s", "Z 漂移率", True),
    ("z_step_m", "cm_z_step_warn_m", "Z 台阶", False),
)

#: aux 那两个阈值的出厂猜测值。与 ``_SAT_SHIPPED_DEFAULT`` 同一手法：报告要能回答
#: 「这个数有没有被本机标定过」。漂开会让 test_commission 变红。
_AUX_SHIPPED_DEFAULTS: dict[str, float] = {
    "cm_z_drift_warn_m_per_s": 50e-12,
    "cm_z_step_warn_m": 500e-12,
}

#: **只报分布，不给建议**的 aux 列。
#:
#: ``p99 × 1.5`` 这套自动建议对**上界**型阈值成立（「本底最大到这里，留 1.5 倍余量」）。
#: 振幅归零判据是个**下界** —— 它问「离零够不够近」，p99 在那个方向上什么都不说。
#: 拿同一套公式硬套会印出一个看起来标定过、实际毫无意义的数，比不印更糟。
#:
#: 所以这里只把两条分布摆出来，让人自己看 ``cm_amp_zero_frac`` 站得住站不住：
#: 未接触时振幅是本底的多少倍（应聚在 1.0 附近），以及相对散布有多大。
_AUX_REPORT_ONLY: tuple[tuple[str, str], ...] = (
    ("amp_frac_of_baseline", "振幅 / 未接触本底"),
    ("amp_rel_sd", "振幅相对散布"),
)


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or f in (float("inf"), float("-inf"))) else f


def _describe(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": min(vals),
        "p50": _pct(vals, 50),
        "p90": _pct(vals, 90),
        "p99": _pct(vals, 99),
        "max": max(vals),
    }


# ── data sources ────────────────────────────────────────────────────────────


def _tri_bool(v) -> Optional[bool]:
    """SQLite 的 0/1 转为 bool，None 保持未知，避免身份比较把有效样本排除。"""
    if v is None:
        return None
    return bool(v)


def _aux_rows_from_store(hours: float) -> list[dict]:
    """辅助通道的行，归一成与 ``_rows_from_api`` 相同的形状。"""
    import time

    from mast.monitoring.store import get_store

    rows, _total, _thinned = get_store().aux_query(
        since=time.time() - hours * 3600.0, limit=200_000)
    return [{
        "ts": r.get("ts"),
        "verdict": r.get("verdict") or "ok",
        "scanning": _tri_bool(r.get("ctx_scanning")),
        "skill": r.get("ctx_skill") or "",
        "z_ctrl_on": _tri_bool(r.get("ctx_zctrl_on")),
        "metrics": {k: v for k, v in r.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)},
    } for r in rows]


def _rows_from_store(hours: float) -> tuple[list[dict], dict]:
    import time

    from mast.monitoring.service import get_service
    from mast.monitoring.store import get_store

    store = get_store()
    since = time.time() - hours * 3600.0
    rows, _total, _thinned = store.features_query(since=since, limit=200_000)
    svc = get_service()
    status = svc.status() if svc is not None else {}
    # features_query returns raw columns; normalise to the API's row shape so
    # both sources feed the same analysis.
    out = []
    for r in rows:
        metrics = {k: v for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        out.append({
            "ts": r.get("t_start"), "verdict": r.get("alert_level") or "ok",
            "scanning": r.get("ctx_scanning"), "skill": r.get("ctx_skill") or "",
            "z_ctrl_on": r.get("ctx_zctrl_on"), "metrics": metrics,
        })
    return out, status


def _rows_from_api(url: str, hours: float, auth: Optional[str],
                   timeout: float = 30.0, insecure: bool = False
                   ) -> tuple[list[dict], dict]:
    """从远程 MAST 拉特征行 + 状态。

    ``insecure`` 关闭 TLS 证书校验，仅供显式选择时使用。采用自签名证书的
    部署需要配置受信任证书，或由调用方明确选择该开关。

    默认仍然校验。这是个安全开关,不能因为"反正在内网"就悄悄关掉:调用方必须显式
    写 ``--insecure``,而失败信息会告诉他这个开关存在。
    """
    import base64
    import ssl
    import time
    import urllib.request

    headers = {}
    if auth:
        token = base64.b64encode(auth.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"

    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    def _get(path: str) -> dict:
        req = urllib.request.Request(url.rstrip("/") + path, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as fh:   # noqa: S310
            return json.loads(fh.read().decode("utf-8"))

    since = time.time() - hours * 3600.0
    body = _get(f"/api/monitoring/features?since={since:.0f}&limit=200000")
    status = _get("/api/monitoring/status")

    # 辅助通道。**取不到不算失败** —— 老版本的 MAST 上根本没有这个端点，
    # 而电流那一段的标定不该因此整个跑不出来。
    aux_rows: list[dict] = []
    try:
        cols = ",".join(k for k, _knob, _lbl, _abs in _AUX_CALIBRATABLE)
        a = _get(f"/api/monitoring/aux/series?since={since:.0f}"
                 f"&max_points=5000&columns={cols}")
        aux_rows = _aux_rows_from_series(a)
    except Exception as exc:  # noqa: BLE001
        print(f"(辅助通道数据不可用,报告将跳过那一段:{exc})", file=sys.stderr)
    return (body.get("rows") or []), status, aux_rows


def _aux_rows_from_series(body: dict) -> list[dict]:
    """把列式的 ``/aux/series`` 回包转回按行的形状。

    列式在 wire 上省事,分析要的是行 —— 转换在这里做一次,而不是让 ``analyse_aux``
    同时懂两种形状。
    """
    t_s = body.get("t_s") or []
    series = body.get("series") or {}
    scanning = body.get("scanning") or []
    zctrl = body.get("z_ctrl_on") or []
    skills = body.get("skills") or []
    verdicts = body.get("verdicts") or []

    def at(seq, i):
        return seq[i] if i < len(seq) else None

    out = []
    for i, ts in enumerate(t_s):
        out.append({
            "ts": ts,
            "verdict": at(verdicts, i) or "ok",
            "scanning": _tri_bool(at(scanning, i)),
            "z_ctrl_on": _tri_bool(at(zctrl, i)),
            "skill": at(skills, i) or "",
            "metrics": {k: at(v, i) for k, v in series.items()},
        })
    return out


# ── analysis ────────────────────────────────────────────────────────────────


def _split(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Group by what the instrument was doing.

    A segment recorded while scanning, while a tip-shaping skill held the
    instrument, or on a parked tip are three different populations. Calibrating
    across all of them produces a threshold that fits none — and the 'quiet'
    group is the one a healthy baseline should come from.
    """
    groups: dict[str, list[dict]] = {"quiet": [], "scanning": [], "skill": [],
                                     "no_junction": []}
    for r in rows:
        if r.get("skill"):
            groups["skill"].append(r)
        elif r.get("z_ctrl_on") is False:
            groups["no_junction"].append(r)
        elif r.get("scanning"):
            groups["scanning"].append(r)
        else:
            groups["quiet"].append(r)
    return groups


def analyse(rows: list[dict], thresholds: dict) -> dict:
    """Distributions per group, plus a suggested operating point per knob."""
    groups = _split(rows)
    # Prefer the quiet population for calibration; fall back to scanning if the
    # instrument was never idle during the window. That fallback is NOT applied
    # to the rules that are suppressed while scanning — see below.
    baseline_name = "quiet" if len(groups["quiet"]) >= 30 else (
        "scanning" if len(groups["scanning"]) >= 30 else "")
    baseline = groups.get(baseline_name, [])

    dists: dict[str, dict] = {}
    for name, grp in groups.items():
        if not grp:
            continue
        dists[name] = {
            key: _describe([_finite(r.get("metrics", {}).get(key)) for r in grp])
            for key, _knob, _label in _CALIBRATABLE
        }

    # Do not calibrate scan-suppressed rules from scanning fallback samples: those features may describe topography rather than the quiet instrument noise.
    from mast.monitoring.alerts import SCAN_SUPPRESSED_RULES

    suggestions = []
    for key, knob, label in _CALIBRATABLE:
        current = _finite(thresholds.get(knob))
        if (baseline_name == "scanning"
                and _FEATURE_RULE.get(key) in SCAN_SUPPRESSED_RULES):
            suggestions.append({
                "feature": key, "knob": knob, "label": label,
                "current": current, "suggested": None, "n": 0,
                "note": "窗口内没有安静段,而这条判据在扫描期间本就不判 —— "
                        "拿扫描段标它会标出样品的台阶密度。请另取一段安静窗口。",
            })
            continue
        vals = [v for v in (_finite(r.get("metrics", {}).get(key)) for r in baseline)
                if v is not None]
        if not vals:
            suggestions.append({"feature": key, "knob": knob, "label": label,
                                "current": current, "suggested": None,
                                "n": 0, "note": "该窗口内无可用数据"})
            continue
        anchor = _pct(vals, _ANCHOR_Q) or 0.0
        would_fire = sum(1 for v in vals if current is not None and v > current)
        # A detector that never fires on a healthy baseline anchors at zero, and
        # zero × any headroom is still zero — a threshold of 0 makes EVERY later
        # segment an alert. That is the opposite of what the number is for, so
        # say "keep it" instead of emitting a value that is harmful if applied.
        if anchor <= 0.0:
            suggestions.append({
                "feature": key, "knob": knob, "label": label,
                "current": current, "suggested": None,
                "baseline_p99": anchor, "n": len(vals),
                "would_fire_now": would_fire,
                "would_fire_pct": 100.0 * would_fire / len(vals),
                "note": "基线上从未触发,无从标定 —— 保留当前值",
            })
            continue
        suggestions.append({
            "feature": key, "knob": knob, "label": label,
            "current": current, "suggested": anchor * _HEADROOM,
            "baseline_p99": anchor, "n": len(vals),
            "would_fire_now": would_fire,
            "would_fire_pct": 100.0 * would_fire / len(vals),
        })

    return {
        "n_rows": len(rows),
        "baseline_group": baseline_name,
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "distributions": dists,
        "suggestions": suggestions,
        # render() 需要看非可标定的 knob（饱和阈），而它只拿得到 report。
        "thresholds": dict(thresholds or {}),
        "timing": _timing(rows),
    }


def _timing(rows: Iterable[dict]) -> dict:
    """段落的真实落盘节奏 —— 墙钟覆盖率的分母。

    **为什么要算它**：覆盖率此前是手算的（「段内 1.024 s + 段间 0.31 s」），
    而它恰好是「新功能有没有拖慢电流那一路」唯一能回答问题的数。手算的数没法
    before/after 对比，也没法在远程跑。

    只用 ``ts``（每段的 t_start），所以本地库与远程 ``/api/monitoring/features``
    两条路都能算 —— 不新开端点，也不给 2 秒一次的 ``/status`` 加任何负担。
    """
    ts = sorted(v for v in (_finite(r.get("ts")) for r in rows) if v is not None)
    if len(ts) < 3:
        return {"n": len(ts)}
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    return {
        "n": len(ts), "first": ts[0], "last": ts[-1],
        "span_s": ts[-1] - ts[0],
        "cadence_p50_s": _pct(gaps, 50) if gaps else None,
        "cadence_p99_s": _pct(gaps, 99) if gaps else None,
    }


def analyse_aux(rows: list[dict], thresholds: dict) -> dict:
    """辅助通道标定只使用不扫描、Z 反馈开启且无技能占用的样本。扫描形貌与退针后的固定 Z 都不能代表隧穿本底；上下文未知的样本不进入基线。"""
    parked, scanning, busy, no_junction = [], [], [], []
    for r in rows:
        if r.get("skill"):
            busy.append(r)
        elif r.get("z_ctrl_on") is False:
            no_junction.append(r)
        elif r.get("scanning") is True:
            scanning.append(r)
        elif r.get("scanning") is False and r.get("z_ctrl_on") is True:
            parked.append(r)
        # 其余（上下文未知）两边都不算 —— 判据也不放行。

    dist: dict[str, dict] = {}
    suggestions: list[dict] = []
    for key, knob, label, absolute in _AUX_CALIBRATABLE:
        vals = [abs(v) if absolute else v
                for v in (_finite(r.get("metrics", {}).get(key)) for r in parked)
                if v is not None]
        dist[key] = _describe(vals)
        current = _finite(thresholds.get(knob))
        shipped = _AUX_SHIPPED_DEFAULTS.get(knob)
        uncalibrated = (current is not None and shipped is not None
                        and abs(current - shipped) < 1e-18)
        if not vals:
            suggestions.append({"feature": key, "knob": knob, "label": label,
                                "current": current, "suggested": None, "n": 0,
                                "uncalibrated": uncalibrated,
                                "note": "该窗口内没有「停在原地」的样本"})
            continue
        anchor = _pct(vals, _ANCHOR_Q) or 0.0
        would_fire = sum(1 for v in vals if current is not None and v > current)
        suggestions.append({
            "feature": key, "knob": knob, "label": label,
            "current": current,
            # 基线上从未触发（anchor=0）时给不出建议 —— 0 × 任何余量还是 0，
            # 而阈值 0 会让之后每一段都告警。和 analyse() 同一条判断。
            "suggested": (anchor * _HEADROOM) if anchor > 0 else None,
            "baseline_p99": anchor, "n": len(vals),
            "would_fire_now": would_fire,
            "would_fire_pct": 100.0 * would_fire / len(vals),
            "uncalibrated": uncalibrated,
            "note": None if anchor > 0 else "基线上恒为 0,无从标定 —— 保留当前值",
        })

    for key, _label in _AUX_REPORT_ONLY:
        dist[key] = _describe(
            [v for v in (_finite(r.get("metrics", {}).get(key)) for r in parked)
             if v is not None])

    return {
        "n_rows": len(rows),
        "group_sizes": {"parked": len(parked), "scanning": len(scanning),
                        "skill": len(busy), "no_junction": len(no_junction)},
        "distributions": dist,
        "suggestions": suggestions,
        "alerts_enabled": bool(_finite(thresholds.get("cm_aux_alerts_enabled"))),
    }


# ── reporting ───────────────────────────────────────────────────────────────


def _fmt(v: Optional[float], key: str) -> str:
    if v is None:
        return "—"
    if key.endswith("_a"):                 # amps → pA
        return f"{v * 1e12:.2f} pA"
    # 辅助通道是米与「米每秒」。用 pm 显示：Z 的一切有趣现象都在 pm～nm 量级，
    # 印成 5e-11 会让人每次都要心算。
    if key.endswith("_m_per_s"):
        return f"{v * 1e12:.2f} pm/s"
    if key.endswith("_m"):
        return f"{v * 1e12:.1f} pm"
    return f"{v:.3g}"


def render_aux(report: dict, status: dict) -> str:
    """报告里的【辅助通道】那一段。

    **没有数据时也照样打印。** 早先的版本在没有 aux 行时返回空串「不占版面」——
    但这份报告存在的目的之一就是「把辅助通道的阈值标出来」，那时候一片空白读起来
    就是「这个功能不存在」。silence looks like absence：宁可多两行，也要说清楚
    是没采到、还是没开、还是守护没跑。
    """
    aux_status = (status or {}).get("aux") or {}
    L: list[str] = ["═══ 辅助通道 · Z 位置 / qPlus 振幅 ═══\n"]

    if not report.get("n_rows"):
        L.append("【本窗口内没有辅助通道数据】")
        if not aux_status:
            L.append("  采集守护没在跑（或这个版本还没有辅助通道）——"
                     "这不是「本机没有 Z / qPlus」，是还没有人去看过。")
        elif not aux_status.get("enabled"):
            L.append("  记录开关 cm_aux_enabled = 0，所以什么都没记。")
        else:
            L.append(f"  记录开着但这段时间没有行落盘（detail: "
                     f"{aux_status.get('detail') or '—'}）。"
                     "让它跑一段「停在原地」的时间再回来看。")
        L.append("")
    chans = aux_status.get("channels") or []
    if chans:
        L.append("【通道】")
        for c in chans:
            avail = "✓" if c.get("available") else "✗ 本机没有"
            val = c.get("value")
            unit = c.get("unit") or ""
            shown = "—"
            if isinstance(val, (int, float)):
                shown = (f"{val * 1e12:.2f} pm" if unit == "m" else f"{val:.4g} {unit}")
            L.append(f"  {str(c.get('label_zh') or c.get('kind')):12s} {avail:10s}"
                     f" #{c.get('signal_index', -1):<4} 最新 {shown:>14s}"
                     f"  判级 {c.get('verdict', '?')}")
            if c.get("note"):
                L.append(f"      {c['note']}")
        tau = aux_status.get("amp_tau_s")
        interval = aux_status.get("interval_s") or 0.0
        if tau:
            verdict = "够" if interval <= tau else "**不够,采样比振幅还慢**"
            L.append(f"  振幅弛豫 τ = Q/(π f₀) = {tau * 1e3:.0f} ms；"
                     f"采样间隔 {interval:.2f} s —— {verdict}")
        else:
            L.append("  振幅弛豫 τ：未知(没扫过 PLL 共振)。"
                     "跑 AcquirePLLFreqSweep 之后这里会给出 1 Hz 够不够的**算得出来**的答案。")
        if aux_status.get("baseline_amp_m") is None:
            L.append("  ⚠ 没有未接触本底 → 归零判据静默(没有本底就不知道「不为零」"
                     "长什么样)。针尖确认未接触时调用"
                     " ReadTipOscillationAmplitude(set_baseline=True)。")
        L.append(f"  采样 {aux_status.get('sampled', 0)} 次，"
                 f"因角色忙跳过 {aux_status.get('skipped_busy', 0)} 次")
        L.append("")

    sizes = report.get("group_sizes") or {}
    if sizes:
        L.append("【分组样本数】(标定只用「停在原地」组:不扫图 + Z 反馈开 + 无技能占用)")
        for k, label in (("parked", "停在原地 ←用作基线"), ("scanning", "扫描中"),
                         ("skill", "技能占用"), ("no_junction", "退针/无结")):
            L.append(f"  {label:22s} {sizes.get(k, 0):6d}")
        L.append("  扫描中的 Z 在跟着形貌走 —— 拿它标阈值,标出来的是样品的台阶高度。")
        L.append("  退针态的 Z 是**冻住的**(漂移恒 0)——混进基线会把 p99 拉低,")
        L.append("  于是建议出一个过紧的阈值,装上去之后隧穿态每段都响。")
        L.append("  标定数据的仪器状态应与实际应用状态一致，")
        L.append("   避免把退针本底直接当作隧穿状态的阈值依据。")
        if not sizes.get("parked"):
            L.append("  ⚠ 基线组为空 —— 需要「针在隧穿态、Z 反馈开着、没在扫图」的时段。")
        L.append("")

    L.append("【阈值建议】(p99 × 1.5;绝不自动应用)")
    for s in report.get("suggestions") or []:
        key = s["feature"]
        cur = _fmt(s.get("current"), key)
        mark = " ★出厂猜测" if s.get("uncalibrated") else ""
        if not s.get("n") or s.get("suggested") is None:
            L.append(f"  {s['label']:16s} 当前 {cur:>14s}{mark}"
                     f"   {s.get('note') or '该窗口内无可用数据'}")
            continue
        pct = s.get("would_fire_pct", 0.0)
        flag = "  ⚠ 当前阈值会在基线上误报" if pct > 1.0 else ""
        L.append(f"  {s['label']:16s} 当前 {cur:>14s}{mark} → 建议 "
                 f"{_fmt(s.get('suggested'), key):>14s}"
                 f"   (基线上会触发 {pct:.1f}%){flag}")

    dists = report.get("distributions") or {}
    if any(dists.get(k, {}).get("n") for k, _ in _AUX_REPORT_ONLY):
        L.append("")
        L.append("【振幅本底分布】(只报不建议 —— 归零判据是个**下界**,p99 在那个方向上不说话)")
        for key, label in _AUX_REPORT_ONLY:
            d = dists.get(key) or {}
            if not d.get("n"):
                L.append(f"  {label:16s} —  该窗口内没有「停在原地」的样本")
                continue
            L.append(f"  {label:16s} n={d['n']:<6d} p50 {d['p50']:.3f}"
                     f"  p99 {d['p99']:.3f}  min {d['min']:.3f}  max {d['max']:.3f}")
        L.append("  「振幅 / 未接触本底」应当聚在 1.0 附近;它的 min 离 cm_amp_zero_frac")
        L.append("  (出厂 0.30)有多远,就是这条判据在本机还剩多少余量。")
        L.append("  ⚠ 这些样本是**没在进针**时录的,而判据只在进针期间判 ——")
        L.append("    它们能说明「不为零长什么样」,说明不了「撞上之后会掉到多少」。")

    if not report.get("alerts_enabled"):
        L.append("")
        L.append("  ⚠ **辅助通道的告警当前是关的**(cm_aux_alerts_enabled=0)。")
        L.append("    这是出厂状态,不是故障 —— 这两路的阈值从来没有在任何真机上标定过,")
        L.append("    而上面这份分布正是用来把它打开的。看过建议值、写进设置之后再开。")
    return "\n".join(L)


def render(report: dict, status: dict) -> str:
    L: list[str] = []
    L.append("═══ 电流监控 · 真机标定报告 ═══\n")

    L.append("【只有仪器能回答的事实】")
    fs = status.get("fs_hz") or 0.0
    rt = status.get("rt_freq_hz") or 0.0
    tbs = status.get("timebases_s") or []
    L.append(f"  采集状态      : {status.get('state', '?')}"
             f"  策略 {status.get('strategy') or '—'}")
    L.append(f"  实际采样率    : {fs:.0f} Hz"
             f"   (RT 环路 {rt:.0f} Hz)" if fs else "  实际采样率    : —")
    L.append(f"  通道          : {status.get('channel_name') or '—'}")
    L.append(f"  缓冲深度 n    : {status.get('n_buffer') or '—'}"
             "   ← 协议不提供,只能实测")
    if tbs:
        shown = ", ".join(f"{v * 1e6:.0f}µs" for v in tbs[:8])
        L.append(f"  可用时基({len(tbs)}) : {shown}"
                 f"{' …' if len(tbs) > 8 else ''}  当前 idx="
                 f"{status.get('timebase_index', -1)}")
    else:
        L.append("  可用时基      : —  (未读到;检查 nanonis_patch 是否生效)")
    # 时基表的自校验。放在表下面一行,因为它说的正是「上面那张表可不可信」。
    # 三态各说各的话:对上了 / 对不上 / 没法对 —— 最后一种既不是通过也不是故障。
    check = str(status.get("timebase_check") or "")
    if check.startswith("mismatch"):
        L.append(f"  ⚠ 时基表自校验 : {check}")
        L.append("    ↑ 仪器自报的档数与数组长度对不上。上面的采样率、以及"
                 "「还有没有更快的档」这两个结论都要先怀疑这一条。")
    elif check.startswith("unverified"):
        L.append(f"  时基表自校验  : {check}   ← 没法对账,不等于对上了")
    elif check:
        L.append(f"  时基表自校验  : {check}(档数、当前档位、数值范围三项均自洽)")

    st = status.get("pump_stats") or {}
    if st:
        fresh = st.get("fresh", 0)
        total_polls = fresh + st.get("duplicate", 0)
        dup_pct = (100.0 * st.get("duplicate", 0) / total_polls) if total_polls else 0.0
        L.append(f"  轮询          : 新帧 {fresh}  重复 {st.get('duplicate', 0)}"
                 f" ({dup_pct:.0f}%)  缺口 {st.get('gap', 0)}"
                 f"  忙 {st.get('busy', 0)}  不连续 {st.get('discontinuity', 0)}")
        L.append("    ↑ 重复占比高 = 轮询快于硬件填充(正常);"
                 "缺口多 = 没跟上,数据有洞")

        # 重复帧比例只有在仍持续出现新帧时才可解释为正常。
        # 新帧停滞且未形成有效段时必须独立报告，不能用重复率高掩盖采集停顿。
        if total_polls >= 100 and dup_pct >= 90.0:
            segs = status.get("segments_done", 0) or 0
            rate = fresh / total_polls if total_polls else 0.0
            L.append(f"    ⚠ 重复 {dup_pct:.0f}% 且新帧只占 {rate * 100:.1f}% —— 这不是"
                     f"「轮询快」，是**判新判不出来**。已落盘 {segs} 段。")
            L.append("      查:回包 t0 是否恒为同一个值(相对触发下它恒为 0),"
                     "以及判新是不是只看 t0。")
    L.append(f"  已采段数      : {status.get('segments_done', 0)}"
             f"  累计缺口 {status.get('gaps_total_s', 0):.1f} s")

    # 墙钟覆盖率为记录的总段时长除以首尾跨度，用于比较采集完整性。
    t = report.get("timing") or {}
    seg_s = _finite(status.get("segment_seconds")) or 0.0
    if t.get("n", 0) >= 3 and t.get("span_s", 0) > 0 and seg_s > 0:
        # 分母是**第一段开始到最后一段结束**，所以要给首尾跨度补上最后那一段自己的
        # 时长。不补的话 N 段只算了 N-1 段的墙钟，覆盖率被系统性高估（N=200 时
        # 77.3% vs 真值 76.9%），而这个数正是要拿去做 before/after 对比的。
        total_s = t["span_s"] + seg_s
        cov = 100.0 * t["n"] * seg_s / total_s
        L.append(f"  墙钟覆盖率    : {cov:.1f}%   "
                 f"({t['n']} 段 × {seg_s:.3g} s / 跨度 {total_s / 60:.1f} min)")
        L.append(f"  落盘节奏      : p50 {t['cadence_p50_s']:.3f} s"
                 f"  p99 {t['cadence_p99_s']:.3f} s"
                 "   ← 段间停顿=特征提取，p99 变大即有东西在抢 data 口")
    L.append("")

    L.append("【分组样本数】(标定只用「安静」组 —— 扫描/修针期间不可比)")
    for k, n in (report.get("group_sizes") or {}).items():
        mark = " ←用作基线" if k == report.get("baseline_group") else ""
        L.append(f"  {k:12s} {n:6d}{mark}")
    if not report.get("baseline_group"):
        L.append("  ⚠ 没有足够的基线样本(需要 ≥30 段),阈值建议不可用。")
    L.append("")

    base = report.get("baseline_group")
    dist = (report.get("distributions") or {}).get(base or "", {})
    if dist:
        L.append(f"【「{base}」组的特征分布】")
        L.append(f"  {'特征':22s} {'p50':>12s} {'p90':>12s} {'p99':>12s} {'max':>12s}")
        for key, _knob, label in _CALIBRATABLE:
            d = dist.get(key) or {}
            if not d.get("n"):
                continue
            L.append(f"  {label:22s} {_fmt(d.get('p50'), key):>12s}"
                     f" {_fmt(d.get('p90'), key):>12s}"
                     f" {_fmt(d.get('p99'), key):>12s}"
                     f" {_fmt(d.get('max'), key):>12s}")
        L.append("")

    L.append("【阈值建议】(p99 × 1.5;绝不自动应用)")
    for s in report.get("suggestions") or []:
        key = s["feature"]
        cur = _fmt(s.get("current"), key)
        sug = _fmt(s.get("suggested"), key)
        if not s.get("n") or s.get("suggested") is None:
            L.append(f"  {s['label']:22s} 当前 {cur:>12s}"
                     f"   {s.get('note', '该窗口内无可用数据')}")
            continue
        pct = s.get("would_fire_pct", 0.0)
        flag = "  ⚠ 当前阈值会在基线上误报" if pct > 1.0 else ""
        L.append(f"  {s['label']:22s} 当前 {cur:>12s} → 建议 {sug:>12s}"
                 f"   (基线上会触发 {pct:.1f}%){flag}")
    L.append("")
    L.append("说明:CRITICAL 三条(饱和/冻结/巨幅瞬变)不在此标定 —— 它们是关于仪器的")
    L.append("      陈述(贴轨就是贴轨),与健康基线的噪声水平无关。")
    L.append("      饱和阈 cm_sat_current_a 应设为前置放大器的真实量程。")
    # 饱和是三条 CRITICAL 之一,而它的出厂值是一个**猜测**("just under the usual
    # 100 nA preamp")。判据是 |I| >= cm_sat_current_a —— 阈值定得比真实量程高,
    # 前放贴轨时读数停在真实量程上,永远够不到阈值,**这条 CRITICAL 就被静默关掉了**。
    # 因此明确提示：饱和阈值必须按目标仪器的前放量程登记。
    _sat = (report.get("thresholds") or {}).get("cm_sat_current_a")
    if _sat is not None and abs(float(_sat) - _SAT_SHIPPED_DEFAULT) < 1e-18:
        L.append("")
        L.append(f"  ⚠ cm_sat_current_a 仍是**出厂猜测值** {_fmt(_sat, 'cm_sat_current_a')}"
                 "(注释里写的是 usual 100 nA preamp)。")
        L.append("    它没有被本机的前放量程校准过。**定得比真实量程高 = 饱和这条")
        L.append("    CRITICAL 被静默关掉**(贴轨时读数停在真实量程,永远够不到阈值)。")
        L.append("    问用户前放量程,然后写进设置 —— 这不是标定,是一次登记。")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mast.monitoring.commission",
        description="真机标定报告:观察分布 → 建议阈值(只读,不写设置)")
    ap.add_argument("--url", help="远程 MAST API,例如 http://<rig-host>:7870;"
                                  "省略则读本机 store")
    ap.add_argument("--auth", help="远程 basic auth,格式 user:pass")
    ap.add_argument("--hours", type=float, default=2.0, help="回看时长(小时)")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非报告")
    ap.add_argument("--insecure", action="store_true",
                    help="显式关闭 TLS 证书校验，仅用于需要此行为的连接。"
                         "默认始终校验证书，不自动放行。")
    args = ap.parse_args(argv)

    try:
        if args.url:
            rows, status, aux_rows = _rows_from_api(args.url, args.hours, args.auth,
                                                    insecure=args.insecure)
            thresholds = {}
            try:
                import base64
                import urllib.request
                headers = {}
                if args.auth:
                    tok = base64.b64encode(args.auth.encode()).decode("ascii")
                    headers["Authorization"] = f"Basic {tok}"
                req = urllib.request.Request(
                    args.url.rstrip("/") + "/api/monitoring/config", headers=headers)
                _ctx = None
                if args.insecure:
                    import ssl
                    _ctx = ssl.create_default_context()
                    _ctx.check_hostname = False
                    _ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, timeout=30, context=_ctx) as fh:   # noqa: S310
                    cfg = json.loads(fh.read().decode("utf-8"))
                thresholds = {k["key"]: k["value"] for k in (cfg.get("knobs") or [])}
            except Exception as exc:  # noqa: BLE001
                print(f"(读取远程阈值失败,建议值将缺少对比:{exc})", file=sys.stderr)
        else:
            rows, status = _rows_from_store(args.hours)
            aux_rows = _aux_rows_from_store(args.hours)
            from mast.monitoring.thresholds import get_monitor_thresholds
            thresholds = get_monitor_thresholds().to_mapping()
    except Exception as exc:  # noqa: BLE001
        print(f"取数失败:{exc}", file=sys.stderr)
        # 证书校验失败时给出明确指引，避免把证书问题误当作网络不可达。
        if "CERTIFICATE_VERIFY_FAILED" in str(exc) and not getattr(args, "insecure", False):
            print("提示:服务器证书未通过校验；请配置受信任证书，或显式使用 --insecure。", file=sys.stderr)
        return 1

    report = analyse(rows, thresholds)
    aux_report = analyse_aux(aux_rows, thresholds)
    if args.json:
        print(json.dumps({"status": status, "report": report, "aux": aux_report},
                         ensure_ascii=False, indent=2, default=float))
    else:
        print(render(report, status))
        print()
        print(render_aux(aux_report, status))
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())


__all__ = ["analyse", "analyse_aux", "render", "render_aux", "main"]
