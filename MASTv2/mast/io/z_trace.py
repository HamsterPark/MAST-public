"""事件前后「稳定值之差」—— z(t) 轨迹的跳变判定（纯函数）。

物理学家看一次电脉冲、或一次扎针尖时并不读整条曲线:看事件**之前**那段稳定的
z,和事件**之后**那段稳定的 z,两者差多少、往哪边。中间的瞬态峰(qPlus 音叉
起跳那种)不参与判断 —— 它属于过程,不属于结果。

本模块把那个判断写成纯函数:输入采样与时间戳,输出方向与幅度。无硬件、无存储、
无全局配置;每个阈值都由参数传入,测试才能陈述它要测的世界。

**方向是信号自己的方向**(z 变大 / 变小),不是物理解释。同一个 "up" 在扎针尖
时读作「针尖下面长了个 cluster」,在电脉冲时读作「针尖变短了几十 nm」,而 Z 增
大到底是远离还是靠近样品取决于接线 —— 见 ``instrument_profile.z_extend_sign``。
把解释留给调用方是有意的:判据只有一份,语义各归各家。

提取自 ``skills/builtins/tip_shaper_readback.py``(2026-08-01),原先是那个技能
的私有函数。电脉冲也要同一套判据,复制一份的下场是两边阈值各自漂移。
"""
from __future__ import annotations

from typing import Sequence

#: 跳变检测的默认稳健倍数。**8 是实测出来的,不是口味问题** —— 见
#: ``monitoring/features.py`` 的 ``FeatureParams``:纯高斯白噪声下,k=5 时一段
#: 干净记录能报出约 6 次/秒的假跳变,k=7 恰好归零而注入的真阶跃仍然全中,8 留出
#: 余量。**误报率随样本数变化**,所以窗口长度或采样率变动一个数量级时必须重标。
DEFAULT_JUMP_K = 8.0

#: 走 MAD-of-diffs 估噪声所需的最少样本。少于这个数中位数没有意义。
_MIN_SAMPLES_FOR_MAD = 4

#: 后窗兜底时至少要够到的样本数。**这个 2 不是新阈值** —— 它就是 ``step_verdict``
#: 本来声明的下限(那里的 ``len(post) < 2``)。取值依据与推翻它所需的观测,
#: 完整写在 ``step_verdict`` 的 docstring 里。
_MIN_WINDOW_SAMPLES = 2


#: 反馈重新接管之后，至少要有这么长的一段才敢下判断（秒）。
#:
#: 必须等电流回到 setpoint，并留下足够稳定样本；此默认时间需在目标仪器上复核。
_MIN_FEEDBACK_SEGMENT_S = 0.25

#: 判「电流回到 setpoint 了」的倍数。基线电流就是 setpoint（反馈闭合时按定义
#: 如此），所以这是个相对判据，不需要谁把 setpoint 传进来。
_CURRENT_BACK_K = 3.0

#: 判「针尖确实压进去了」的倍数。压 300 pm、kappa≈10/nm ⇒ 电流升约 e^6≈400 倍，
#: 所以 5 倍是个很松的门槛；它存在只是为了确认「有过一次压入」，从而不会在
#: 基线段上误取 t4。
_CURRENT_PRESSED_K = 5.0


def feedback_restored_t(current_s: Sequence[float], current_t: Sequence[float],
                        *, event_start_t: float) -> "float | None":
    """用电流定位反馈重新接管时刻，不能把计划时序当成实测边界。
    
    参数推算的 _stage_boundaries 可能与硬件执行时刻存在延迟。
    四段分别为：反馈基线；关闭反馈后压入、电流上升；Z 斜坡回到原位置但电流仍高；
    反馈重新开启、电流回到设定点并建立新平衡。
    第三段按命令回基线，不能说明操作后的针尖或表面状态；只有第四段可用于结果判断。
    
    返回 (t4, why)。no_press 表示未检测到压入，可继续使用普通尾窗；
    no_return 表示压入后未见电流返回，采集可能在反馈接管前结束，必须弃权。
    no_current / too_few 分别表示缺电流通道或样本不足，不合并成无意义的缺省状态。"""
    n = min(len(current_s), len(current_t))
    if n < 4:
        return None, "too_few"
    base = [abs(float(c)) for c, t in zip(current_s, current_t) if t < event_start_t]
    if len(base) < 2:
        return None, "too_few"
    i0 = _median(base)
    if not (i0 > 0):
        return None, "too_few"
    # 先确认「压进去过」—— 否则基线段本身就满足「电流等于 setpoint」，
    # t4 会落在扎针之前。
    pressed = None
    for k in range(n):
        if (current_t[k] >= event_start_t
                and abs(float(current_s[k])) > _CURRENT_PRESSED_K * i0):
            pressed = k
            break
    if pressed is None:
        return None, "no_press"
    # 压入之后，电流第一次回到 setpoint 附近
    for k in range(pressed, n):
        if abs(float(current_s[k])) <= _CURRENT_BACK_K * i0:
            return float(current_t[k]), "current"
    return None, "no_return"


def _max_gap(times: Sequence[float]) -> float:
    """全程最长的相邻样本间隔。**每一条返回路径都要带上它**,判不出来的那几条
    尤其要 —— 那正是最想知道「是不是有一次超长往返」的时刻。"""
    return max((b - a for a, b in zip(times, times[1:])), default=0.0)


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def _mad_diff_sigma(xs: Sequence[float]) -> float:
    """相邻差分的 MAD 换算成 σ。

    差分先把慢漂移消掉:一段基线若在缓慢爬升,直接取标准差会把漂移当噪声,容差
    随之撑大,真跳变反而被吞掉。``/√2`` 是因为做差把方差翻了一倍。与
    ``tilt_probe._estimate_noise`` 同一口径。"""
    n = len(xs)
    if n < 2:
        return 0.0
    diffs = [xs[i + 1] - xs[i] for i in range(n - 1)]
    med = _median(diffs)
    mad = _median([abs(d - med) for d in diffs])
    return mad * 1.4826 / (2.0 ** 0.5)


def baseline_sigma(xs: Sequence[float]) -> float:
    """估计基线噪声：优先 MAD-of-diffs，退化时回落标准差。
    
    样本过少或量化台阶可能使 MAD 精确为零，不能因此把比较容差降为零。"""
    n = len(xs)
    if n < 2:
        return 0.0
    if n >= _MIN_SAMPLES_FOR_MAD:
        s = _mad_diff_sigma(xs)
        if s > 0:
            return s
    return _std(xs)


def detect_jumps(samples: Sequence[float], times: Sequence[float],
                 k: float = DEFAULT_JUMP_K) -> dict:
    """一维轨迹上的稳健跳变检测。

    标记幅度超过 ``median(|Δ|) + k·σ_robust``(σ_robust = 1.4826·MAD)的一阶差分
    —— 抓突变,同时忽略 shaper 强加在 Z 上的平滑斜坡。返回事件表(时刻 + 带符号
    的差)、最大 |Δ| 与计数。"""
    if len(samples) < 3:
        return {"events": [], "max_abs_delta": 0.0, "count": 0}
    diffs = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]
    absd = sorted(abs(x) for x in diffs)
    med = absd[len(absd) // 2]
    mad = sorted(abs(a - med) for a in absd)[len(absd) // 2]
    sigma = 1.4826 * mad
    thr = med + k * (sigma if sigma > 0 else (med if med > 0 else 1e-30))
    events: list[dict] = []
    max_abs = 0.0
    for i, dv in enumerate(diffs):
        a = abs(dv)
        if a > max_abs:
            max_abs = a
        if a > thr:
            events.append({"t_s": times[i + 1], "delta": dv})
    return {"events": events, "max_abs_delta": max_abs, "count": len(events)}


def step_verdict(samples: Sequence[float], times: Sequence[float],
                 event_start_t: float | None, *, post_roll_s: float,
                 tol_k: float = 4.0, tol_abs_m: float = 0.0,
                 current_s: "Sequence[float] | None" = None,
                 current_t: "Sequence[float] | None" = None) -> dict:
    """事件前后稳定值之差 —— 跳没跳、往哪边、跳了多少。

    三个量:``z1`` = 事件前基线,``z_min`` = 全程最低点(只作记录,**不参与判定**
    —— 那是过程里的瞬态),``z3`` = 采集尾部安定下来的值。``delta = z3 - z1``。

      * ``|delta| <= tol`` → ``"none"``  没变
      * ``delta >  tol``   → ``"up"``    信号朝正方向跳
      * ``delta < -tol``   → ``"down"``  信号朝负方向跳

    ``tol = max(tol_abs_m, tol_k · baseline_sigma(z1 窗口))``。

    两个窗口都只取**后半段**:扎入的下降沿、回撤的斜坡都发生在窗口前半,让它们
    渗进 z1/z3 会把稳定值算歪。qPlus 音叉那种零均值振荡在这里自然被抵消(取的
    是中位数),而振荡撑大的 σ 又让容差自动变保守 —— 无需为它写特例。

    ═══════════════════════════════════════════════════════════════════════
    后窗按**点数**兜底,不只按时间(2026-08-05, KNOWN_ISSUES)
    ═══════════════════════════════════════════════════════════════════════

    ``post_roll_s`` 想表达的是「事件之后有足够多的**稳定态样本**」,而它用**时间**
    来表达这件事 —— 两者**只在采样均匀时等价**。采集循环是墙钟驱动、在调用返回
    之后才打时间戳的,所以一次异常长的 TCP 往返就会造出一个时间戳远在窗口之外的
    样本。``cap = times[-1]`` 于是被这一个离群值挟持,后窗往后平移,把**全部**真实
    样本挡在外面。

    一次长延迟可能令尾部窗口只剩一个离群样本，从而把有效采集误报为
    ``insufficient_data``。必须区分真正缺少样本与时间窗被延迟拉偏。

    所以:先按时间取窗;**若不足 ``_MIN_WINDOW_SAMPLES`` 个,就往回够到这个数为止**,
    并把这件事如实标出来(见 ``post_window_starved``)。

    ``_MIN_WINDOW_SAMPLES = 2`` 的依据(**不是随手挑的**):

    * 它就是这个函数**本来就声明的下限**(下面那句 ``len(post) < 2``),不是新引入的;
    * 统计上:容差是 ``tol_k·σ``(默认 4σ),而后窗只取后半段 —— 2 点的后半段是
      **1 点**,估计误差 1.0σ,仍比判据门限小 4 倍;
    * **而且 2 是使偏倚最小的选择**:往回够会伸向尚未稳定的瞬态,而「越靠后越稳」
      正是上一段那条设计前提。取后半段意味着实际用的是**最后一个点**,即最稳的
      那个;K 越大反而够回越靠近瞬态的地方(K=20 → 中位数取自 10 点,伸得更远)。

    **要推翻这个取值,需要观测到**:①一次「扩窗判定」与用户所见不符;
    或 ②采集结束时 Z 仍在动(最后那个点本身没稳下来),导致方向判反。
    **这两种至今都没有被观测到过。**

    ⚠️ **扩窗是对判据的修复,不是对卡顿的修复。** 那次异常长的往返仍然是一个
    该被上报的事实(本项目有 comms_health 子系统与成文的 Nanonis TCP 脆弱史),
    所以 ``max_gap_s`` **每次都算**,判定正常时也算 —— 只在退化时才算的量,
    又会是一个「只在不需要它的地方管用的守卫」。
    """
    if event_start_t is None or len(samples) < 4:
        return {"direction": "insufficient_data",
                "n_pre": 0, "n_post": 0, "post_window_starved": False,
                "post_window_s": 0.0, "max_gap_s": _max_gap(times)}
    pre = [z for z, t in zip(samples, times) if t < event_start_t]
    cap = times[-1]

    # 后窗必须位于反馈接管后的第四段。
    # 有电流数据时据此定位；没有才使用尾窗，不能把命令回基线的第三段当成最终平衡。
    t4, seg4_source = None, "no_current"
    if current_s is not None and current_t is not None:
        t4, seg4_source = feedback_restored_t(current_s, current_t,
                                              event_start_t=event_start_t)
    seg4_s = (cap - t4) if t4 is not None else None
    seg4_short = bool(t4 is not None and seg4_s < _MIN_FEEDBACK_SEGMENT_S)
    if seg4_short:
        seg4_source = "too_short"
    if t4 is not None and not seg4_short:
        post_idx = [i for i, t in enumerate(times) if t >= t4]
        win = seg4_s
    else:
        win = min(post_roll_s, cap * 0.4) if cap > 0 else post_roll_s
        post_idx = [i for i, t in enumerate(times) if t >= cap - win]
    # 后窗被一次超长往返饿死了 —— 往回够到下限为止。这是关于**这次采集**的事实,
    # 不是「判据自己搞定了」:标志与 max_gap_s 一起把它如实报出去。
    post_starved = (len(post_idx) < _MIN_WINDOW_SAMPLES
                    and len(samples) >= _MIN_WINDOW_SAMPLES)
    if post_starved:
        post_idx = list(range(len(samples) - _MIN_WINDOW_SAMPLES, len(samples)))
    post = [samples[i] for i in post_idx]
    # ⚠️ pre 侧**刻意不做同样的兜底**,有测试钉住(见 test_z_trace)。
    # pre 由 ``t < event_start_t`` 选出 —— 它不靠尾锚,没有被离群值挟持的失败模式。
    # 前窗点数少,就是基线**真的**不够,那时 insufficient_data 是实话。
    if len(pre) < 2 or len(post) < 2:
        # ⚠️ 判不出来的时候,**恰恰最需要**「采了几个点、最长往返多久」。
        # 这两条早退以前只回一个 direction,把诊断全丢了 —— 于是那些字段
        # 在不需要它们时幸存、在需要它们时消失。有测试钉住(test_z_trace /
        # test_tip_shaper_readback)。
        return {"direction": "insufficient_data",
                "n_pre": len(pre), "n_post": len(post),
                "post_window_starved": post_starved,
                "post_window_s": ((times[-1] - times[post_idx[0]])
                                  if post_idx else 0.0),
                "max_gap_s": _max_gap(times)}
    if seg4_source in ("too_short", "no_return"):
        # ═══════════════════════════════════════════════════════════════
        # 「采集在反馈接管之前/之中就结束了」的答案是**判不了**，不是「没扎上」
        # ═══════════════════════════════════════════════════════════════
        # 这两句话指向完全不同的下一步：一个是「把 post_roll_s 加长再扎」，
        # 一个是「加大扎入深度」。历史上这里输出的是后者，于是用户被送去
        # 一次次加深，而缺的其实是一秒钟的采集时间。
        #
        # ``no_return`` 尤其要命：电流升上去了、**再没回到 setpoint**，
        # 说明整条曲线只有第一到第三段。2026-08-20 的 168 条历史曲线里
        # 160 条是这一种 —— 也就是说那一整批数据**没有一条**记录过判定
        # 真正需要的那一段，而它们当时全都给出了确定的判定。
        why = ("反馈恢复之后只录到 %.3f s（下限 %.2f s）" % (seg4_s or 0.0,
                                                    _MIN_FEEDBACK_SEGMENT_S)
               if seg4_source == "too_short"
               else "电流压上去之后**再没回到 setpoint** —— 采集在反馈接管之前就结束了")
        return {"direction": "insufficient_data",
                "reason": ("feedback_segment_too_short" if seg4_source == "too_short"
                           else "feedback_segment_not_captured"),
                "n_pre": len(pre), "n_post": len(post),
                "post_window_starved": post_starved,
                "post_window_s": float(seg4_s or 0.0),
                "feedback_segment_s": (float(seg4_s) if seg4_s is not None else None),
                "feedback_segment_source": seg4_source,
                "feedback_restored_t": t4,
                "advice": ("%s —— 判不了。**不要据此加大扎入深度**，"
                           "先把 post_roll_s 调到 1.5 s 左右再扎一次。" % why),
                "max_gap_s": _max_gap(times)}
    pre_s = pre[len(pre) // 2:]
    post_s = post[len(post) // 2:]
    z1 = _median(pre_s)
    z3 = _median(post_s)
    sigma1 = baseline_sigma(pre_s)
    tol = max(float(tol_abs_m), tol_k * sigma1)
    delta = z3 - z1
    if abs(delta) <= tol:
        direction = "none"
    elif delta > 0:
        direction = "up"
    else:
        direction = "down"
    return {
        "direction": direction,
        "delta_m": delta,
        "z1_m": z1,
        "z3_m": z3,
        "z_min_m": min(samples),
        "z_max_m": max(samples),
        "baseline_sigma_m": sigma1,
        "tol_m": tol,
        "n_pre": len(pre),
        "n_post": len(post),
        # ── 这次采集本身的事实(与判定分开读)──────────────────────────
        # post_window_starved:预期的 post_roll_s 窗口里样本不够下限 ——
        #   即**这次采集期间有一次异常长的往返**,长到把整个后窗清空了。
        #   它说的是采集,不是补救;补救体现在 post_window_s 上。
        # max_gap_s:全程最长的相邻样本间隔。**每次都算**,判定正常时也算 ——
        #   comms_health 想看的是这个数,不是「判据有没有被迫扩窗」。
        "post_window_starved": post_starved,
        "post_window_s": (times[-1] - times[post_idx[0]]) if post_idx else 0.0,
        "max_gap_s": _max_gap(times),
        # ── 后窗是**怎么定出来的**（判定之外的事实）─────────────────
        # "current"    第四段由电流定位 —— 这是唯一可信的那一档
        # "no_press"   电流从没升上去 ⇒ 可能根本没压到表面；尾窗照判
        # "no_current" 没给电流通道 ⇒ 退回尾窗，可能骑在三/四段边界上
        # "too_few"    电流样本太少
        # "no_return" / "too_short" 走 insufficient_data，不会到这里
        "feedback_segment_source": seg4_source,
        "feedback_segment_s": (float(seg4_s) if seg4_s is not None else None),
        "feedback_restored_t": t4,
    }
