"""Standalone SQLite + .npy storage for the tunnelling-current monitor.

Same shape as :mod:`mast.billing.ledger`: its own DB file, one connection behind
one lock, WAL, and writes that swallow their own exceptions. A monitoring write
failing must never take down acquisition — losing a segment is an inconvenience,
crashing the daemon that was watching the tip is not.

**读不在这条豁免里。** 一次读失败折成空结果，说出来的是一句正面断言
（「查过了，零条告警」/「0 段、0 字节」），而调用方分不出它和真的零。
读查询因此抛 :class:`StoreQueryFailed` —— 哪些改了、哪些**刻意没改**，
逐条写在那个类的 docstring 里。

What lives where, and why:

* ``segments`` / ``features`` / ``alerts`` / ``labels`` rows are PERMANENT.
  They are small, and they are the corpus.
* Raw samples live in ``.npy`` files and are DISPOSABLE. At the real machine's
  20 kHz float32 that is ~288 MB/h, so they roll off by age or by total size.
* The ``envelope`` blob (min/max per 10 ms) is permanent and lives in the row.
  After the raw file is swept away the segment is still drawable and still
  carries its features — you lose the ability to re-analyse it, not the record
  that it happened.
* ``pinned`` rows are exempt from the sweep. Anything worth training on later
  (a tip-shaping window, an alert, a human label) gets pinned when it happens,
  not when someone remembers to.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start       REAL    NOT NULL,
    t_end         REAL    NOT NULL,
    osci_t0       REAL,
    fs_hz         REAL    NOT NULL,
    n_samples     INTEGER NOT NULL,
    n_runs        INTEGER NOT NULL DEFAULT 1,
    gap_s         REAL    NOT NULL DEFAULT 0,
    discontinuity INTEGER NOT NULL DEFAULT 0,
    npy_path      TEXT,
    npy_bytes     INTEGER NOT NULL DEFAULT 0,
    dtype         TEXT    NOT NULL DEFAULT 'float32',
    channel_name  TEXT    NOT NULL DEFAULT '',
    source        TEXT    NOT NULL DEFAULT 'osci1t',
    envelope      BLOB,
    envelope_dt_s REAL,
    pinned        INTEGER NOT NULL DEFAULT 0,
    pin_reason    TEXT,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seg_tstart ON segments(t_start);
CREATE INDEX IF NOT EXISTS idx_seg_pinned ON segments(pinned);
-- live_tail and the age sweep both filter on t_end. Without this they scan the
-- whole table, and the table only grows: rows are permanent by design, so at one
-- segment per second that is ~86k rows/day, each carrying an envelope BLOB. The
-- scan holds the same lock the acquisition thread needs, and the live chart
-- polls every 2 s.
CREATE INDEX IF NOT EXISTS idx_seg_tend ON segments(t_end);

CREATE TABLE IF NOT EXISTS features (
    segment_id       INTEGER PRIMARY KEY REFERENCES segments(id),
    t_start          REAL NOT NULL,
    fs_hz            REAL NOT NULL,
    mean_a           REAL, median_a REAL, min_a REAL, max_a REAL, ptp_a REAL,
    rms_a            REAL, rms_detrended_a REAL, slope_a_per_s REAL,
    kurtosis         REAL, skewness REAL,
    jump_count       INTEGER, jump_rate_hz REAL, max_step_a REAL,
    spike_count      INTEGER, spike_max_sigma REAL,
    band_0p1_1_a2    REAL, band_1_10_a2 REAL, band_10_45_a2 REAL,
    band_45_65_a2    REAL, band_65_200_a2 REAL, band_200_1k_a2 REAL,
    band_1k_5k_a2    REAL, band_5k_nyq_a2 REAL,
    line_power_a2    REAL, line_ratio REAL,
    inv_f_slope      REAL, inv_f_r2 REAL, white_floor_a2hz REAL,
    rtn_score        REAL, rtn_gap_a REAL, rtn_rate_hz REAL,
    rtn_dwell_hi_ms  REAL, rtn_dwell_lo_ms REAL, rtn_transitions INTEGER,
    sat_frac         REAL, railed_frac REAL, frozen INTEGER, unique_frac REAL,
    ctx_scanning     INTEGER, ctx_bias_v REAL, ctx_setpoint_a REAL,
    ctx_z_m          REAL, ctx_zctrl_on INTEGER, ctx_stale INTEGER,
    ctx_skill        TEXT NOT NULL DEFAULT '',
    alert_level      TEXT NOT NULL DEFAULT 'ok',
    extra_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_feat_tstart ON features(t_start);
CREATE INDEX IF NOT EXISTS idx_feat_level  ON features(alert_level);

-- ``acked`` 与 ``delivered_agent`` 是**两个主体**,不是一件事的两种说法:
--
--   acked           人在 UI 上把这条点掉了 —— 「我知道了,别再提醒我」。
--   delivered_agent 这条已经被放进过 agent 的上下文 —— 「它看见过了」。
--
-- 合成一个字段会让两个都问不出来:人点掉的那条会因此不再送给 agent(而人点掉
-- 的理由多半是「我不需要弹窗」,不是「agent 不必知道」);agent 看过的那条会
-- 在 UI 上显示成已确认,于是用户永远看不到有东西发生过。
-- User acknowledgement and agent delivery must each have a working writer.
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    level          TEXT NOT NULL,
    rule           TEXT NOT NULL,
    summary_zh     TEXT NOT NULL,
    segment_id     INTEGER REFERENCES segments(id),
    evidence_png   TEXT,
    features_json  TEXT,
    emitted_buffer INTEGER NOT NULL DEFAULT 0,
    acked          INTEGER NOT NULL DEFAULT 0,
    delivered_agent INTEGER NOT NULL DEFAULT 0,
    delivered_ts   REAL
);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(ts);
-- ⚠️ ``delivered_agent`` 上的索引**不在这里建**。这段脚本跑在补列之前,老库
-- (v7)那时还没有这一列,``CREATE INDEX`` 会当场 OperationalError 并让整个 store
-- 打不开 —— 也就是所有已装机器一升级就没有监控库。索引在 ``__init__`` 里补完列
-- 之后建,见那里。(这条是被 test_old_db_without_the_delivery_columns_is_migrated
-- 逮住的,不是想出来的。)

CREATE TABLE IF NOT EXISTS labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start    REAL NOT NULL,
    t_end      REAL NOT NULL,
    segment_id INTEGER REFERENCES segments(id),
    label      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'human',
    note       TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_label_t ON labels(t_start);

-- 辅助通道（Z 位置 / qPlus 振幅 / 频率偏移 / dI/dV lock-in），2026-08-03，
-- lock-in 那一路 2026-08-05 加入。
--
-- WIDE 而不是 (channel, value) 的窄表，两个理由：各路是同一次 Signals_ValsGet
-- 取回的，行数少几倍；而且**图要的是对齐的时间轴** —— 一次查询就拿到同一时刻的
-- 各路值，窄表得自己 pivot。
--
-- 加一路的代价因此是「加几列」，不是「多一张表」：``_aux_column_spec()`` 从写入
-- 方自己的列清单生成 ALTER，所以老库开机即补齐，不需要迁移脚本。
--
-- 窗口特征跟着原始值一起落盘，不是「反正能重算」：标定工具要的是分布，
-- 对七天的行每次重算 300 s 窗口是分钟级的活。features 表同理。
--
-- 没有 .npy，没有包络。这条路一秒几个浮点，整整一年也才几十 MB。
CREATE TABLE IF NOT EXISTS aux_samples (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   REAL NOT NULL,
    segment_id           INTEGER REFERENCES segments(id),
    z_m                  REAL, z_mean_m REAL, z_drift_m_per_s REAL,
    z_span_m             REAL, z_step_m REAL, z_step_count INTEGER,
    z_step_robust        INTEGER, z_headroom_frac REAL,
    z_drift_half1_m_per_s REAL, z_drift_half2_m_per_s REAL,
    z_drift_decaying     INTEGER, junction_age_s REAL,
    amp_m                REAL, amp_mean_m REAL, amp_rel_sd REAL,
    amp_frac_of_baseline REAL, amp_min_frac_of_baseline REAL,
    amp_drop_frac        REAL, amp_zero INTEGER, amp_zero_hold_s REAL,
    amp_ring_age_s       REAL, amp_off INTEGER,
    amp_gate_open        INTEGER, amp_blanked INTEGER,
    -- 音叉在不在被驱动。三态：1/0/**NULL**，NULL 是「没问过或没读到」。
    amp_excited          INTEGER,
    df_hz                REAL, df_drift_hz_per_s REAL, df_span_hz REAL,
    lockin_a             REAL, lockin_mean_a REAL, lockin_span_a REAL,
    lockin_mod_on        INTEGER,
    bias_v               REAL, bias_mean_v REAL, bias_span_v REAL,
    bias_step_v          REAL,
    -- 这个窗口里偏压改过没有。三态：1/0/**NULL**（NULL = 判不了）。
    bias_changed         INTEGER,
    ctx_scanning         INTEGER, ctx_zctrl_on INTEGER,
    ctx_skill            TEXT NOT NULL DEFAULT '',
    verdict              TEXT NOT NULL DEFAULT 'ok',
    rules                TEXT NOT NULL DEFAULT '',
    extra_json           TEXT
);
CREATE INDEX IF NOT EXISTS idx_aux_ts ON aux_samples(ts);

-- 噪声基线：一次刻意的多工况表征，判据拿它当**分母**（不是阈值 —— 阈值仍然
-- 只在 thresholds.py 一处）。设计见 docs/v2/design/current_noise_baseline.md。
--
-- 与 segments/features 的关系是**引用，不是复制**：每个工况点记下它用到的
-- seg_id 区间，统计量与那两张表同口径（baseline.width_stats 与
-- features.detrended_rms 逐位相同，有测试钉住）。
--
-- 这两张表**不参与滚动清理**：一次表征约 14 个点、每点 6 KB 的谱与直方图，
-- 与 288 MB/小时的原始段不是一个量级，纳入清理只带来风险不带来收益。
CREATE TABLE IF NOT EXISTS noise_baseline (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    label         TEXT    NOT NULL DEFAULT '',
    note          TEXT    NOT NULL DEFAULT '',
    -- 'complete' | 'aborted' | 'running'。aborted 的行**保留**：半份表征仍然
    -- 记录了它测到的那几个点，删掉等于销毁记录。但它不可被 activate。
    status        TEXT    NOT NULL DEFAULT 'running',
    active        INTEGER NOT NULL DEFAULT 0,
    n_points      INTEGER NOT NULL DEFAULT 0,
    t_start       REAL,
    t_end         REAL,
    fs_hz         REAL,
    -- 条件快照与派生模型都是 JSON：它们的字段会随认识变化，而把每个字段
    -- 提成一列会让每次认识更新都变成一次 schema 迁移。
    conditions    TEXT    NOT NULL DEFAULT '{}',
    sigma_model   TEXT,
    white_model   TEXT,
    lines         TEXT,
    repeatability TEXT,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_baseline_active ON noise_baseline(active);

CREATE TABLE IF NOT EXISTS noise_baseline_point (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_id   INTEGER NOT NULL REFERENCES noise_baseline(id),
    ordinal       INTEGER NOT NULL DEFAULT 0,
    tag           TEXT    NOT NULL DEFAULT '',
    ts            REAL    NOT NULL,
    bias_v        REAL, setpoint_a REAL, i_measured_a REAL, z_m REAL,
    seg_id_lo     INTEGER, seg_id_hi INTEGER, n_segments INTEGER NOT NULL DEFAULT 0,
    sigma_a       REAL, sigma_iqr_a REAL, sigma_mad_a REAL,
    fwhm_a        REAL, fwhm_over_sigma REAL, iqr_a REAL, p99_p1_a REAL, ptp_a REAL,
    kurtosis      REAL, skewness REAL, mean_a REAL,
    white_a2hz    REAL, line_ratio REAL,
    band_1_10_a2  REAL, band_10_45_a2 REAL, band_45_65_a2 REAL,
    band_65_200_a2 REAL, band_200_1k_a2 REAL,
    -- Z 通道（Osci2T 双通道 burst，与电流逐样本同步）。全部**只记录不判级**：
    -- 电流那边的判据是先对纯高斯白噪声验过误报率才敢上的，照搬到位移信号上是赌。
    z_sigma_m     REAL, z_step_rms_m REAL, z_white_m2hz REAL, z_ptp_m REAL,
    z_fwhm_m      REAL, z_mean_m REAL, z_n_runs INTEGER,
    -- Z 与电流的联合量。kappa 由 |Pzi|/Pzz 在高相干频段上求出 ——
    -- 从本来就要采的 burst 里免费得到一条 I-z 谱才能给的数。
    coh_fraction  REAL, coh_n_pairs INTEGER,
    kappa_per_m   REAL, apparent_barrier_ev REAL,
    psd_path      TEXT, hist_path TEXT, z_psd_path TEXT, coh_path TEXT,
    extra_json    TEXT,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bpoint_baseline ON noise_baseline_point(baseline_id);
"""

#: Schema versions add columns through generated ALTER statements.
#: 11: bias columns; 10: Z and current cross statistics; 9: noise baselines.
#: Baseline arrays are retained independently of rolling acquisition data.
#: 8: agent delivery is separate from user acknowledgement.
#: 7: excitation is nullable; 6: modulation state is nullable; 5: lock-in channels.
#: 4: amplitude zero and ring-age fields replace the retired collapse edge.
#: Historical columns remain readable; 3: junction age; 2: auxiliary samples.
#: Tables are created on open and missing columns are added without deleting old data.
_SCHEMA_VERSION = 11

#: Feature columns in the order the writer binds them.
_FEATURE_COLS: tuple[str, ...] = (
    "mean_a", "median_a", "min_a", "max_a", "ptp_a",
    "rms_a", "rms_detrended_a", "slope_a_per_s", "kurtosis", "skewness",
    "jump_count", "jump_rate_hz", "max_step_a",
    "spike_count", "spike_max_sigma",
    "band_0p1_1_a2", "band_1_10_a2", "band_10_45_a2", "band_45_65_a2",
    "band_65_200_a2", "band_200_1k_a2", "band_1k_5k_a2", "band_5k_nyq_a2",
    "line_power_a2", "line_ratio", "inv_f_slope", "inv_f_r2", "white_floor_a2hz",
    "rtn_score", "rtn_gap_a", "rtn_rate_hz",
    "rtn_dwell_hi_ms", "rtn_dwell_lo_ms", "rtn_transitions",
    "sat_frac", "railed_frac", "frozen", "unique_frac",
)

#: ``alerts`` 表上「这条送到 agent 了没有」的两列，及其 SQL 声明。
#:
#: 单独列出来而不是写死在迁移调用里，是为了让 :meth:`CurrentMonitorStore
#: .mark_alerts_delivered` 的写入列与迁移列**同源** —— 与 :func:`_aux_column_spec`
#: 同一条纪律（见那里的注释：手抄的迁移清单会跟着 schema 漂开，而漂开的表现是
#: 「更新静默失败、这一列悄悄永远是 0」）。
_ALERT_DELIVERY_COLS: tuple[tuple[str, str], ...] = (
    # ALTER TABLE ADD COLUMN 带 NOT NULL 必须给默认值。老库里已有的行因此都是
    # 「未送达」——**这是对的**：它们确实没被送给过任何 agent。
    ("delivered_agent", "INTEGER NOT NULL DEFAULT 0"),
    ("delivered_ts", "REAL"),
)


def _alert_column_spec() -> tuple[tuple[str, str], ...]:
    """``alerts`` 表在老库上需要补的列。见 :data:`_ALERT_DELIVERY_COLS`。"""
    return _ALERT_DELIVERY_COLS


#: ``noise_baseline_point`` 的测量列，顺序即写入顺序。同 ``_AUX_COLS`` 的纪律：
#: 迁移清单由 :func:`_baseline_point_column_spec` 从这一份生成，不手抄第二份。
_BPOINT_COLS: tuple[str, ...] = (
    "bias_v", "setpoint_a", "i_measured_a", "z_m",
    "seg_id_lo", "seg_id_hi", "n_segments",
    "sigma_a", "sigma_iqr_a", "sigma_mad_a",
    "fwhm_a", "fwhm_over_sigma", "iqr_a", "p99_p1_a", "ptp_a",
    "kurtosis", "skewness", "mean_a",
    "white_a2hz", "line_ratio",
    "band_1_10_a2", "band_10_45_a2", "band_45_65_a2",
    "band_65_200_a2", "band_200_1k_a2",
    # Z 通道与 Z-电流联合量（2026-08-18）。缺省 None：没采 Z 的工况点这些列
    # 就是空的，而空与 0 在这里是两件事 —— 0 m 的 Z 噪声是不可能的读数。
    "z_sigma_m", "z_step_rms_m", "z_white_m2hz", "z_ptp_m",
    "z_fwhm_m", "z_mean_m", "z_n_runs",
    "coh_fraction", "coh_n_pairs", "kappa_per_m", "apparent_barrier_ev",
)
_BPOINT_INT_COLS: frozenset[str] = frozenset({
    "seg_id_lo", "seg_id_hi", "n_segments", "ordinal",
    "z_n_runs", "coh_n_pairs",
})
_BPOINT_TEXT_COLS: frozenset[str] = frozenset({
    "tag", "psd_path", "hist_path", "z_psd_path", "coh_path", "extra_json",
})

#: ``noise_baseline`` 的可变列（id / ts / created_at 由建表语句拥有）。
#: `noise_baseline` 里存 JSON 的列。**读回时必须逐个 json.loads**，否则它们
#: 会以字符串身份撞上 API schema 的 dict 声明，让整个列表端点降级。
#: 与 `_BASELINE_COLS` 的关系由 test_baseline_json_cols_are_all_deserialised 钉住。
_BASELINE_JSON_COLS: tuple[str, ...] = (
    "conditions", "sigma_model", "white_model", "lines", "repeatability",
    "polarity", "bias_magnitude",
)

_BASELINE_COLS: tuple[str, ...] = (
    "label", "note", "status", "active", "n_points", "t_start", "t_end",
    "fs_hz", "conditions", "sigma_model", "white_model", "lines", "repeatability",
    # 保存正负偏压配对检查结果。
    "polarity",
    # 幅度依赖与极性依赖分别保存，便于独立解释。
    "bias_magnitude",
)
_BASELINE_INT_COLS: frozenset[str] = frozenset({"active", "n_points"})
_BASELINE_TEXT_COLS: frozenset[str] = frozenset({
    "label", "note", "status", "conditions", "sigma_model", "white_model",
    "lines", "repeatability",
})


def _baseline_column_spec() -> tuple[tuple[str, str], ...]:
    """``noise_baseline`` 写入方需要的列 —— 从写入方的清单生成，见 :func:`_aux_column_spec`。"""
    out: list[tuple[str, str]] = []
    for c in _BASELINE_COLS:
        if c in _BASELINE_TEXT_COLS:
            out.append((c, "TEXT"))
        elif c in _BASELINE_INT_COLS:
            out.append((c, "INTEGER"))
        else:
            out.append((c, "REAL"))
    return tuple(out)


def _baseline_point_column_spec() -> tuple[tuple[str, str], ...]:
    """``noise_baseline_point`` 同上。"""
    cols = ["ordinal", "tag", *_BPOINT_COLS, "psd_path", "hist_path",
            "z_psd_path", "coh_path", "extra_json"]
    out: list[tuple[str, str]] = []
    for c in cols:
        if c in _BPOINT_TEXT_COLS:
            out.append((c, "TEXT"))
        elif c in _BPOINT_INT_COLS:
            out.append((c, "INTEGER"))
        else:
            out.append((c, "REAL"))
    return tuple(out)


def _aux_column_spec() -> tuple[tuple[str, str], ...]:
    """``aux_samples`` 写入方需要的每一列 + 它的 SQL 声明，用于修补老库。

    **从写入方自己的列清单生成，不是手抄一份。** 手抄的迁移清单会跟着 schema
    漂开，而漂开的表现是「插入静默失败、表悄悄不再增长」—— 这个包已经为
    hand-maintained 清单（``SettingsStore.KNOWN_KEYS``、``override_store._ALL_FILES``）
    付过三次学费。这样生成的话，凡是 :meth:`CurrentMonitorStore.add_aux_sample`
    会写的列，迁移一定覆盖得到。
    """
    cols = ["segment_id", *_AUX_COLS, *_AUX_CTX_COLS, "verdict", "rules",
            "extra_json"]
    out: list[tuple[str, str]] = []
    for c in cols:
        if c in _AUX_TEXT_COLS:
            # ALTER TABLE ADD COLUMN 带 NOT NULL 必须给默认值。
            decl = "TEXT NOT NULL DEFAULT ''" if c != "extra_json" else "TEXT"
        elif c in _AUX_INT_COLS:
            decl = "INTEGER"
        else:
            decl = "REAL"
        out.append((c, decl))
    return tuple(out)


#: ``aux_samples`` 的上下文列。与 ``_CTX_COLS``（features 表用的）刻意分开：
#: aux 行不记 bias / setpoint / stale，那些在同一时刻的段落行上已经有了。
_AUX_CTX_COLS: tuple[str, ...] = ("ctx_scanning", "ctx_zctrl_on", "ctx_skill")


_CTX_COLS: tuple[str, ...] = (
    "ctx_scanning", "ctx_bias_v", "ctx_setpoint_a",
    "ctx_z_m", "ctx_zctrl_on", "ctx_stale", "ctx_skill",
)

#: ``aux_samples`` 的整数列与文本列（其余都是 REAL）。只在生成迁移用的列声明时
#: 用到 —— 表本身的类型由上面的 ``CREATE TABLE`` 决定，这里是为了让 ALTER 出来的
#: 列跟它一致。
_AUX_INT_COLS: frozenset[str] = frozenset({
    "segment_id", "z_step_count", "z_step_robust", "z_drift_decaying",
    "amp_zero", "amp_off", "amp_gate_open", "amp_blanked", "amp_excited",
    "lockin_mod_on", "bias_changed",
    "ctx_scanning", "ctx_zctrl_on",
})
_AUX_TEXT_COLS: frozenset[str] = frozenset({
    "ctx_skill", "verdict", "rules", "extra_json",
})

#: ``aux_samples`` 的测量列，顺序即写入顺序。与
#: ``monitoring.aux_channels.AUX_METRIC_COLUMNS`` 逐项相同 —— 有测试钉住（漂开就是静默丢列）。
_AUX_COLS: tuple[str, ...] = (
    "z_m", "z_mean_m", "z_drift_m_per_s", "z_span_m",
    "z_step_m", "z_step_count", "z_step_robust", "z_headroom_frac",
    "z_headroom_retract_m", "z_headroom_extend_m",
    "z_headroom_retract_frac", "z_headroom_extend_frac",
    "z_drift_half1_m_per_s", "z_drift_half2_m_per_s", "z_drift_decaying",
    "junction_age_s",
    "amp_m", "amp_mean_m", "amp_rel_sd",
    "amp_frac_of_baseline", "amp_min_frac_of_baseline", "amp_drop_frac",
    "amp_zero", "amp_zero_hold_s", "amp_ring_age_s", "amp_off",
    "amp_gate_open", "amp_blanked", "amp_excited",
    "df_hz", "df_drift_hz_per_s", "df_span_hz",
    "lockin_a", "lockin_mean_a", "lockin_span_a", "lockin_mod_on",
    # 偏压变化为三态：已改变、未改变或样本不足。未知不得折叠成未改变。
    "bias_v", "bias_mean_v", "bias_span_v", "bias_step_v", "bias_changed",
)


#: 段间空档超过「一整段的时长」= 采集**中断**，实时曲线在那里断开。
#:
#: **判定必须做在这一层，前端做不了。** 段**内部**的相邻点是包络步长（10 ms），
#: 段**之间**才是采集停顿，而拼成一维数组之后这两种间隔长得一模一样。前端按点距
#: 中位数判的话中位数就是 10 ms，于是**每一个段边界都会被划成断点** —— 一条本来
#: 好好的曲线画成上千道虚线。段边界只有这一层看得见。
#:
#: 1.0 有一句人话的意思：**「有整整一段时间我们什么都没测」**。默认配置下：
#:
#:     正常停顿   0.31 s ≈ 0.30 段时长   ← 设计如此（特征提取），不是中断
#:     丢一段     1.33 s ≈ 1.30 段时长   ← 有一整段的数据不存在
_TRACE_GAP_DURATIONS = 1.0

#: 但阈值还要**至少**是典型停顿的这个倍数。
#:
#: 只按段时长判会在一个**合法配置**上崩掉：``cm_segment_s`` 可以调到 0.2 s
#: （见 thresholds._RANGES），而特征提取那 0.31 s 的停顿不跟着变 —— 于是正常停顿
#: 比整段还长，每一个段边界都成了「中断」。这一项把阈值抬到实测停顿之上，
#: 让判据跟着这台机器的实际节奏走，而不是跟着一个假设的占空比走。
_TRACE_GAP_DEAD_FACTOR = 3.0

#: 少于这么多个停顿样本就**不启用**上面那一项。
#:
#: 一个样本的中位数就是它自己，于是「空档 vs 典型空档」恒成立不了 —— 窗口里只有
#: 两段、中间隔了六小时的时候，那六小时会被判成「典型」，断点一个都划不出来。
#: 这正是最需要划断点的场合。样本不够时退回纯段时长判据（它不需要跨段统计）。
_TRACE_GAP_MIN_DEAD_N = 5


def _median(xs: Sequence[float]) -> float:
    """中位数。空序列返回 0.0（调用方按「不知道」处理，不按 0 处理）。

    **不用均值**：均值会被它要找的那些空档本身拉长，空档越大越不像空档。
    同 ``aux_channels._nominal_dt``。
    """
    s = sorted(float(x) for x in xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def trace_gap_marks(spans: Sequence[tuple[float, float]]) -> list[float]:
    """采集中断的时刻列表。``spans`` = 每一段的 ``(起, 止)``，按时间升序。

    纯函数，没有 numpy、没有 store —— 判据本身可以直接喂几对数字来测，
    而这正是它值得单独存在的理由（同 ``aux_channels`` 把窗口特征从采样器里分出来）。

    返回的每个时刻是**空档的左边缘**（前一段结束的那一刻）：空洞就该从数据停下来
    的地方开始，而不是从空档正中开始。

    判据两条，取更严的那个当阈值（见 :data:`_TRACE_GAP_DURATIONS` /
    :data:`_TRACE_GAP_DEAD_FACTOR`）。判不出来时返回空表 —— **「不知道正常间隔是
    多少」不等于「没有中断」**，但在图上凭空划一道断点会让人去查一个没坏的东西，
    所以这一侧是刻意的：宁可少划，也不划错。
    """
    ok = [(float(a), float(b)) for a, b in spans if b > a]
    if len(ok) < 2:
        return []
    dur = _median([b - a for a, b in ok])
    if dur <= 0:
        return []
    dead = [nxt[0] - cur[1] for cur, nxt in zip(ok, ok[1:]) if nxt[0] > cur[1]]
    threshold = dur * _TRACE_GAP_DURATIONS
    if len(dead) >= _TRACE_GAP_MIN_DEAD_N:
        threshold = max(threshold, _median(dead) * _TRACE_GAP_DEAD_FACTOR)
    return [cur[1] for cur, nxt in zip(ok, ok[1:]) if nxt[0] - cur[1] > threshold]


def _splice_trace_gaps(
    ts: list[float], lo: list[float], hi: list[float], marks: Sequence[float],
) -> tuple[list[float], list[Optional[float]], list[Optional[float]]]:
    """在每个中断时刻插一行 ``None``。两条边缘都要插 —— band 才断得开。

    两个序列都已按时间升序，所以是一趟归并，不是每次插入都搬一次数组。

    ⚠️ **已知边界**：全局抽稀的桶是按位置切的，一个桶可能**跨过**空档，于是
    断点旁边那一个点的 min/max 里混进了空档另一侧的数据。影响范围就是那一个桶，
    而且 min/max 只会让它变宽、不会把真实越界收窄（同 live_tail 抽稀本身的口径）。
    不为它把抽稀改成分段做：那要把已经算好的 stride 拆开重算，换来的只是断点两侧
    各一个点的精度，而这张图的用处是「看得见中断」，不是「量断点两端的值」。
    """
    if not marks:
        return ts, list(lo), list(hi)
    out_t: list[float] = []
    out_lo: list[Optional[float]] = []
    out_hi: list[Optional[float]] = []
    it = iter(sorted(marks))
    pending = next(it, None)
    for t, a, b in zip(ts, lo, hi):
        while pending is not None and pending < t:
            # 时间戳必须严格递增，否则 uPlot 会画出一条往回走的线。断点落在
            # 前一段的末尾，而它按构造就在两个真实样本之间。
            if not out_t or pending > out_t[-1]:
                out_t.append(pending)
                out_lo.append(None)
                out_hi.append(None)
            pending = next(it, None)
        out_t.append(t)
        out_lo.append(a)
        out_hi.append(b)
    return out_t, out_lo, out_hi


def segment_npy_path(data_dir: Path, t_start: float, fs_hz: float) -> Path:
    """Per-segment file path, unique by construction.

    Named from the epoch millisecond plus a counter, and bucketed by day. A
    shared or reused name is the single mistake this project has already paid
    for twice — ``scan_frame`` documents frames overwriting each other across
    runs because the path was fixed.

    The collision loop assumes a SINGLE writer (the acquisition daemon). Two
    threads calling this in the same millisecond could both see the same name as
    free; nothing in this package does that, and the exporter only ever reads.
    """
    ms = int(round(t_start * 1000.0))
    day = time.strftime("%Y%m%d", time.localtime(t_start))
    stem = f"seg_{ms:013d}_{int(round(fs_hz))}hz"
    base = Path(data_dir) / "segments" / day
    path = base / f"{stem}.npy"
    n = 1
    while path.exists():                      # same-millisecond collision
        path = base / f"{stem}_{n:02d}.npy"
        n += 1
    return path


class StoreQueryFailed(RuntimeError):
    """一次**读查询**没做成 —— 与「查过了，没有」严格分开。

    ## 为什么写与读的降级方向相反

    模块 docstring 的那条豁免写的是**写**：「A monitoring **write** failing must
    never take down acquisition」。丢一段采集是不便，把守着针尖的守护线程带崩不是。
    这条对写成立，对读**不成立**：一次读失败被折成空结果，说出来的是一句正面断言。

    ``alerts_query`` 那个 ``([], 0)`` 是最贵的一例。``0`` 是一个**计数**，而它同时
    也是「我根本没查成」。conduct 的 L1 闸门(``conduct/adapters.py`` 的
    ``RuntimeMonitorEvents``)正拿它当证据：探针前面那两道检查(守护线程在跑 +
    store 存在)挡不住「store 在、查询炸了」，于是「监控库锁住了」会长成
    「这段时间很太平」，闸门放行一整夜的扫描。

    ## 抛，而不是回 ``None``

    判据是「调用方能不能把它和真的零条分开」。回 ``None`` 也分得开，但
    ``rows, total = store.alerts_query(...)`` 会在解包处炸成一句 ``TypeError``，
    读起来不像「读不到」；而**三个生产调用方本来就各有一条异常路径**：
    ``api/routes/monitoring.py`` 的 ``_guarded`` ⇒ ``degraded=True`` +
    ``detail``；``api/routes/vision.py`` 的 try/except ⇒ 没有证据缩略图；
    ``conduct/adapters.py`` 的 try/except ⇒ 探针回 ``None`` = 判不了。
    抛出去，三条路各自已经写好的话就都成立了。

    ## 哪些**不**改(逐个判过，不是一刀切)

    * 返回 ``None`` 的那些读(``latest_aux`` / ``latest_feature`` /
      ``feature_row`` / ``segment_meta`` / ``alert_evidence_png`` /
      ``read_segment_decimated``)：``None`` 不是正面断言，调用方一律
      ``or {}`` —— 本来就诚实的形状，动它只是制造噪声。
    * ``undelivered_alerts`` / ``critical_alerts_since``：两个调用方**各自写下过**
      「读不到就当没有」的理由(``forge_au_tip._critical_since`` 说得最清楚：
      它是**额外**加的一道网，不是唯一一道，贴轨/冻结照样落库、进面板、经
      ``AlertDeliveryMiddleware`` 到 agent 眼前)。那是一个做过的决定，不是一次疏忽。
    * 所有写(``add_*`` / ``set_*`` / ``ack_*`` / ``mark_*`` / ``clear_*`` /
      ``pin_*``)与两条保养扫(``sweep_aux`` / ``retention_sweep``)：模块 docstring
      那条豁免正是为它们写的，它们跑在采集线程上。
      (``retention_sweep`` 失败时回 ``removed: 0`` 确实同形 —— 盘会安静地涨。
      但让一次扫盘失败去掀掉采集线程是更贵的那一边，留在这里记着。)
    """


class CurrentMonitorStore:
    """Segment index, features, alerts and labels for the current monitor."""

    def __init__(self, db_path: Path | str, data_dir: Path | str | None = None):
        self._path = Path(db_path)
        self._data_dir = Path(data_dir) if data_dir else self._path.parent
        self._path.parent.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "segments").mkdir(parents=True, exist_ok=True)
        (self._data_dir / "evidence").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._over_budget = False          # edge-tracked; see retention_sweep
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._add_missing_columns("aux_samples", _aux_column_spec())
            self._add_missing_columns("alerts", _alert_column_spec())
            self._add_missing_columns("noise_baseline", _baseline_column_spec())
            self._add_missing_columns("noise_baseline_point",
                                      _baseline_point_column_spec())
            (self._data_dir / "baseline").mkdir(parents=True, exist_ok=True)
            # 投递查询是 ``WHERE delivered_agent=0 AND ts>=?``,每次 LLM 调用前跑
            # 一次。**必须在补列之后**建 —— 见 _SCHEMA 里 alerts 索引旁边那段。
            try:
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alert_undelivered"
                    " ON alerts(delivered_agent, ts)")
            except sqlite3.Error:
                # 索引建不出来只是慢,不是错;补列失败的库不该因此完全打不开。
                logger.debug("could not create idx_alert_undelivered",
                             exc_info=True)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            except sqlite3.Error:
                pass
            self._conn.commit()

    def _add_missing_columns(self, table: str,
                             columns: Sequence[tuple[str, str]]) -> None:
        """Add missing columns from the writer declarations. CREATE TABLE IF NOT EXISTS does not migrate existing tables; adding a field requires a corresponding ALTER, with errors surfaced instead of silently discarding new samples."""
        try:
            have = {r["name"] for r in self._conn.execute(
                f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            return
        if not have:
            return                      # table absent — the schema script owns it
        for name, decl in columns:
            if name in have:
                continue
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                logger.info("monitoring store: added %s.%s", table, name)
            except sqlite3.Error:
                logger.debug("could not add %s.%s", table, name, exc_info=True)

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def evidence_dir(self) -> Path:
        return self._data_dir / "evidence"

    # ── write (fail-safe: swallow + log, never raise) ────────────────────────
    def _rollback(self) -> None:
        """Undo a half-applied write before swallowing its exception.

        sqlite3 still opens an implicit transaction for DML, so an INSERT that
        succeeded followed by a commit that did not (a full disk is the realistic
        case, and §10 lists it as expected) leaves the connection holding an
        uncommitted transaction. Every later write then joins it, the WAL keeps
        growing, and a crash loses all of them at once — precisely the corpus
        this subsystem exists to accumulate.
        """
        try:
            self._conn.rollback()
        except Exception:  # noqa: BLE001 — best effort; the caller is already failing
            pass


    def add_segment(self, meta: dict, envelope_bytes: bytes | None = None,
                    envelope_dt_s: float | None = None) -> Optional[int]:
        """Insert one segment row. Returns its id, or None if the write failed."""
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO segments (t_start, t_end, osci_t0, fs_hz, n_samples,"
                    " n_runs, gap_s, discontinuity, npy_path, npy_bytes, dtype,"
                    " channel_name, source, envelope, envelope_dt_s, pinned,"
                    " pin_reason, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        float(meta.get("t_start", 0.0)), float(meta.get("t_end", 0.0)),
                        _f_or_none(meta.get("osci_t0")), float(meta.get("fs_hz", 0.0)),
                        int(meta.get("n_samples", 0)), int(meta.get("n_runs", 1)),
                        float(meta.get("gap_s", 0.0)),
                        1 if meta.get("discontinuity") else 0,
                        meta.get("npy_path") or None, int(meta.get("npy_bytes", 0)),
                        meta.get("dtype", "float32"), meta.get("channel_name", ""),
                        meta.get("source", "osci1t"),
                        envelope_bytes, _f_or_none(envelope_dt_s),
                        1 if meta.get("pinned") else 0, meta.get("pin_reason"),
                        time.time(),
                    ),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001 — storage must never break acquisition
            self._rollback()
            logger.debug("add_segment failed (swallowed)", exc_info=True)
            return None

    def add_features(self, segment_id: int, feats: dict, ctx: dict | None = None,
                     alert_level: str = "ok", extra: dict | None = None) -> None:
        ctx = ctx or {}
        cols = ["segment_id", "t_start", "fs_hz", *_FEATURE_COLS, *_CTX_COLS,
                "alert_level", "extra_json"]
        vals: list[Any] = [
            int(segment_id), float(feats.get("t_start", 0.0)),
            float(feats.get("fs_hz", 0.0)),
        ]
        vals.extend(_num_or_none(feats.get(c)) for c in _FEATURE_COLS)
        vals.extend([
            _bool_or_none(ctx.get("ctx_scanning")), _f_or_none(ctx.get("ctx_bias_v")),
            _f_or_none(ctx.get("ctx_setpoint_a")), _f_or_none(ctx.get("ctx_z_m")),
            _bool_or_none(ctx.get("ctx_zctrl_on")), _bool_or_none(ctx.get("ctx_stale")),
            str(ctx.get("ctx_skill") or ""),
        ])
        vals.extend([
            str(alert_level or "ok"),
            json.dumps(extra, ensure_ascii=False) if extra else None,
        ])
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO features ({','.join(cols)})"
                    f" VALUES ({','.join('?' * len(cols))})", vals,
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("add_features failed (swallowed)", exc_info=True)

    def add_aux_sample(self, ts: float, metrics: dict,
                       ctx: dict | None = None, *,
                       segment_id: int | None = None,
                       verdict: str = "ok",
                       rules: Sequence[str] | None = None,
                       extra: dict | None = None) -> Optional[int]:
        """One 1 Hz auxiliary sample (Z / qPlus amplitude / Δf) + its window stats.

        Same fail-safe contract as every other writer here: a failed write loses
        a sample, never the acquisition thread.
        """
        ctx = ctx or {}
        cols = ["ts", "segment_id", *_AUX_COLS, *_AUX_CTX_COLS,
                "verdict", "rules", "extra_json"]
        vals: list[Any] = [float(ts),
                           int(segment_id) if segment_id else None]
        vals.extend(_num_or_none(metrics.get(c)) for c in _AUX_COLS)
        vals.extend([
            _bool_or_none(ctx.get("ctx_scanning")),
            _bool_or_none(ctx.get("ctx_zctrl_on")),
            str(ctx.get("ctx_skill") or ""),
            str(verdict or "ok"),
            ",".join(str(r) for r in (rules or [])),
            json.dumps(extra, ensure_ascii=False) if extra else None,
        ])
        try:
            with self._lock:
                cur = self._conn.execute(
                    f"INSERT INTO aux_samples ({','.join(cols)})"
                    f" VALUES ({','.join('?' * len(cols))})", vals,
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("add_aux_sample failed (swallowed)", exc_info=True)
            return None

    def aux_query(self, since: float | None = None, until: float | None = None,
                  limit: int = 5000, offset: int = 0,
                  max_points: int | None = None,
                  newest: bool = False) -> tuple[list[dict], int, bool]:
        """Aux rows in a time range. Returns (rows, total, thinned).

        Thinning keeps the WORST verdict in each bucket, same rule as
        :meth:`features_query` — downsampling a chart must not be able to hide a
        row that raised something.

        ``newest=True`` 时 ``limit`` 从**最新**一端截,而不是最早那一端。

        ``ORDER BY ts ASC LIMIT ?`` 适用于配合 ``offset`` 的顺序分页。
        ``/monitoring/aux/series`` 查询最近时间窗且不分页，达到行数限制时
        必须保留最新一端，否则返回值会遗漏当前状态而时间戳仍显得自洽。

        ``offset`` 与 ``newest`` 互斥:分页的语义里「最新优先」没有意义,
        所以 ``offset`` 非零时忽略 ``newest``(而不是悄悄改变分页方向)。
        """
        try:
            with self._lock:
                where, args = _range_clause(since, until, "ts")
                total = int(self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM aux_samples{where}", args
                ).fetchone()["n"])
                if newest and not offset:
                    sql = (f"SELECT * FROM (SELECT * FROM aux_samples{where}"
                           f" ORDER BY ts DESC LIMIT ?) ORDER BY ts ASC")
                    rows = [dict(r) for r in self._conn.execute(
                        sql, args + [int(limit)]).fetchall()]
                else:
                    rows = [dict(r) for r in self._conn.execute(
                        f"SELECT * FROM aux_samples{where} ORDER BY ts ASC"
                        " LIMIT ? OFFSET ?", args + [int(limit), int(offset)],
                    ).fetchall()]
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed
            raise StoreQueryFailed(f"aux_query 失败: {exc}") from exc
        if not max_points or len(rows) <= max_points:
            return rows, total, False
        return _thin_by_severity(rows, int(max_points), key="verdict"), total, True

    def latest_aux(self) -> Optional[dict]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM aux_samples ORDER BY ts DESC LIMIT 1").fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            logger.debug("latest_aux failed (swallowed)", exc_info=True)
            return None

    def add_alert(self, *, ts: float, level: str, rule: str, summary_zh: str,
                  segment_id: int | None = None, evidence_png: str | None = None,
                  features: dict | None = None,
                  emitted_buffer: bool = False) -> Optional[int]:
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO alerts (ts, level, rule, summary_zh, segment_id,"
                    " evidence_png, features_json, emitted_buffer, acked)"
                    " VALUES (?,?,?,?,?,?,?,?,0)",
                    (float(ts), str(level), str(rule), str(summary_zh),
                     int(segment_id) if segment_id else None, evidence_png,
                     json.dumps(features, ensure_ascii=False, default=float)
                     if features else None,
                     1 if emitted_buffer else 0),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("add_alert failed (swallowed)", exc_info=True)
            return None

    def set_alert_evidence(self, alert_id: int, png_path: str) -> None:
        """Attach an evidence image after the fact.

        The alert row is written immediately and the picture lands when the
        renderer finishes — rendering takes up to a second, and doing it inline
        would stall acquisition exactly when something is going wrong.
        """
        try:
            with self._lock:
                self._conn.execute("UPDATE alerts SET evidence_png=? WHERE id=?",
                                   (str(png_path), int(alert_id)))
                self._conn.commit()
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("set_alert_evidence failed (swallowed)", exc_info=True)

    def ack_alert(self, alert_id: int) -> None:
        """人在 UI 上点掉一条告警。**与 agent 是否看过无关** —— 见建表语句上面
        那段「两个主体」。点掉不会让这条不再送给 agent,也不会替 agent 确认。"""
        try:
            with self._lock:
                self._conn.execute("UPDATE alerts SET acked=1 WHERE id=?", (int(alert_id),))
                self._conn.commit()
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("ack_alert failed (swallowed)", exc_info=True)

    # ── 送达 agent（2026-08-10）────────────────────────────────────────────

    def undelivered_alerts(self, since: float, limit: int = 50) -> list[dict]:
        """还没进过 agent 上下文的告警，新的在前。

        ``since`` 是**必填的**,没有默认值:一个 agent 刚接手时不该被半小时前
        早已处置完的历史刷屏,而「忘了传 since」若退化成「全表」,那正是它会
        发生的样子。调用方(投递中间件)从策略里取回看窗口。

        返回原始行 —— 折叠、静音、排序都是策略层的事(``alert_routing``),
        SQL 只负责「哪些还没送过」。
        """
        try:
            with self._lock:
                return [dict(r) for r in self._conn.execute(
                    "SELECT id, ts, level, rule, summary_zh, segment_id,"
                    " emitted_buffer, acked, delivered_agent"
                    " FROM alerts WHERE delivered_agent=0 AND ts>=?"
                    " ORDER BY ts DESC LIMIT ?",
                    (float(since), int(limit)),
                ).fetchall()]
        except Exception:  # noqa: BLE001
            logger.debug("undelivered_alerts failed (swallowed)", exc_info=True)
            return []

    def critical_alerts_since(self, since: float, limit: int = 10) -> list[dict]:
        """``since`` 之后的 CRITICAL 告警,新的在前。

        给**长跑的自治流程**用(``ForgeAuTip`` 的外环每轮开一次):它一轮一轮地
        扎、验、扫,中途针尖崩了没人告诉它,它会在废数据上把所有轮次跑完。

        刻意用**时间水位线**而不是 ``acked`` / ``delivered_agent``:
        * ``acked`` 说的是人点没点过,与「这条流程该不该继续」无关;
        * ``delivered_agent`` 说的是**某个 agent 的上下文**里出现过没有 —— 而
          composite 不是 agent,它没有上下文,消费那一列会把两个主体又搅在一起
          (而且会互相偷走对方的告警:agent 先看到就等于流程看不到了)。

        水位线是调用方自己持有的(「上次我看的时候是几点」),所以这里是纯查询,
        没有状态、没有副作用,两个消费者互不影响。
        """
        try:
            with self._lock:
                return [dict(r) for r in self._conn.execute(
                    "SELECT id, ts, level, rule, summary_zh, segment_id, acked"
                    " FROM alerts WHERE level='critical' AND ts>=?"
                    " ORDER BY ts DESC LIMIT ?",
                    (float(since), int(limit)),
                ).fetchall()]
        except Exception:  # noqa: BLE001
            logger.debug("critical_alerts_since failed (swallowed)", exc_info=True)
            return []

    def mark_alerts_delivered(self, alert_ids, ts: float | None = None) -> int:
        """把这些行记成「agent 已看过」。返回实际更新的行数。

        **只标真的被放进上下文的那些**,包括被折叠掉的同类(它们由计数代表了)。
        被静音的行**不标** —— agent 确实没看见,把它们标成已送达会让这一列不再
        回答它自己那个问题。代价是静音行每次都被重新扫到,而回看窗口 + LIMIT
        已经把这件事的成本封死了。

        返回值不是装饰:调用方拿它区分「没有东西要送」与「要送但一行都没写进去」。
        """
        ids = [int(i) for i in (alert_ids or []) if i is not None]
        if not ids:
            return 0
        try:
            import time as _time

            stamp = float(ts) if ts is not None else _time.time()
            marks = ",".join("?" for _ in ids)
            with self._lock:
                cur = self._conn.execute(
                    f"UPDATE alerts SET delivered_agent=1, delivered_ts=?"
                    f" WHERE id IN ({marks}) AND delivered_agent=0",
                    [stamp, *ids],
                )
                self._conn.commit()
                return int(cur.rowcount or 0)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("mark_alerts_delivered failed (swallowed)", exc_info=True)
            return 0

    def add_label(self, *, t_start: float, t_end: float, label: str,
                  segment_id: int | None = None, source: str = "human",
                  note: str | None = None) -> Optional[int]:
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO labels (t_start, t_end, segment_id, label, source,"
                    " note, created_at) VALUES (?,?,?,?,?,?,?)",
                    (float(t_start), float(t_end),
                     int(segment_id) if segment_id else None,
                     str(label), str(source), note, time.time()),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("add_label failed (swallowed)", exc_info=True)
            return None

    def clear_labels(self, segment_id: int, source: str = "human") -> None:
        try:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM labels WHERE segment_id=? AND source=?",
                    (int(segment_id), str(source)),
                )
                self._conn.commit()
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("clear_labels failed (swallowed)", exc_info=True)

    def set_pin(self, segment_id: int, pinned: bool = True,
                reason: str | None = None) -> bool:
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE segments SET pinned=?, pin_reason=? WHERE id=?",
                    (1 if pinned else 0, reason if pinned else None, int(segment_id)),
                )
                self._conn.commit()
            return True
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("set_pin failed (swallowed)", exc_info=True)
            return False

    # ── noise baseline ──────────────────────────────────────────────────
    #
    # 与本类其余写入方法同一条纪律：**吞异常 + 记日志，永不抛**。一次基线记账
    # 失败绝不能让正在跑的表征技能崩掉。区别只在返回值：这里的写入方法返回
    # id / bool，调用方据此知道有没有落地 —— 「静默失败」在这个子系统里已经
    # 出现过太多次，所以写入方至少要说得出「我没写成」。

    @property
    def baseline_dir(self) -> Path:
        return self._data_dir / "baseline"

    def create_baseline(self, *, label: str = "", note: str = "",
                        conditions: dict | None = None,
                        fs_hz: float | None = None) -> Optional[int]:
        """开一份新的基线（``status='running'``）。返回 id，失败返回 None。"""

        now = time.time()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO noise_baseline"
                    " (ts, label, note, status, active, n_points, t_start,"
                    "  fs_hz, conditions, created_at)"
                    " VALUES (?,?,?,'running',0,0,?,?,?,?)",
                    (now, str(label or ""), str(note or ""), now,
                     float(fs_hz) if fs_hz else None,
                     json.dumps(conditions or {}, ensure_ascii=False), now),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("create_baseline failed (swallowed)", exc_info=True)
            return None

    def add_baseline_point(self, baseline_id: int, *, tag: str = "",
                           ordinal: int = 0, metrics: dict | None = None,
                           psd: "tuple | None" = None,
                           hist: "tuple | None" = None,
                           z_psd: "tuple | None" = None,
                           coh: "tuple | None" = None,
                           extra: dict | None = None) -> Optional[int]:
        """记一个工况点。``psd`` = (freqs, values)，``hist`` = (edges, counts)。

        谱与直方图落 ``.npy`` 而不是 BLOB：它们要能被 numpy 直接读回来做离线
        分析，而这个子系统的既有约定（segments 的 npy）已经是这样。**存的是
        算好的谱，不是原始波形** —— 原始段 24 小时后会被清理，而基线要长期有效。
        """

        now = time.time()
        m = dict(metrics or {})
        psd_path = self._save_baseline_npy(baseline_id, ordinal, "psd", psd)
        hist_path = self._save_baseline_npy(baseline_id, ordinal, "hist", hist)
        z_psd_path = self._save_baseline_npy(baseline_id, ordinal, "zpsd", z_psd)
        coh_path = self._save_baseline_npy(baseline_id, ordinal, "coh", coh)
        cols = ["baseline_id", "ordinal", "tag", "ts", *_BPOINT_COLS,
                "psd_path", "hist_path", "z_psd_path", "coh_path",
                "extra_json", "created_at"]
        vals: list[Any] = [int(baseline_id), int(ordinal), str(tag or ""), now]
        for c in _BPOINT_COLS:
            v = m.get(c)
            if v is None:
                # n_segments 建表时是 NOT NULL DEFAULT 0，而显式绑 NULL 会撞
                # 约束（DEFAULT 只在列不出现在 INSERT 里时生效）。这一列的
                # 「没有」就是 0 段，不是未知。
                vals.append(0 if c == "n_segments" else None)
            elif c in _BPOINT_INT_COLS:
                try:
                    vals.append(int(v))
                except (TypeError, ValueError):
                    vals.append(None)
            else:
                try:
                    fv = float(v)
                    vals.append(fv if math.isfinite(fv) else None)
                except (TypeError, ValueError):
                    vals.append(None)
        vals += [psd_path, hist_path, z_psd_path, coh_path,
                 json.dumps(extra or {}, ensure_ascii=False), now]
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO noise_baseline_point (%s) VALUES (%s)"
                    % (", ".join(cols), ", ".join("?" * len(cols))), vals)
                self._conn.execute(
                    "UPDATE noise_baseline SET n_points=(SELECT COUNT(*) FROM"
                    " noise_baseline_point WHERE baseline_id=?) WHERE id=?",
                    (int(baseline_id), int(baseline_id)))
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("add_baseline_point failed (swallowed)", exc_info=True)
            # 曲线文件在 INSERT **之前**就落盘了。行没进去的话它们成了孤儿：
            # 没有任何一行指向它们，所以 delete_baseline 收不走，清理也不认识
            # 它们（基线目录按设计不参与滚动清理）。就地收掉。
            for p in (psd_path, hist_path, z_psd_path, coh_path):
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        logger.debug("孤儿曲线文件删不掉: %s", p, exc_info=True)
            return None

    def _save_baseline_npy(self, baseline_id: int, ordinal: int, kind: str,
                           payload) -> Optional[str]:
        if not payload:
            return None
        try:
            import numpy as _np
            a, b = payload
            d = self.baseline_dir
            d.mkdir(parents=True, exist_ok=True)
            p = d / ("b%04d_p%03d_%s.npz" % (int(baseline_id), int(ordinal), kind))
            _np.savez_compressed(p, x=_np.asarray(a, dtype=_np.float64),
                                 y=_np.asarray(b, dtype=_np.float64))
            return str(p)
        except Exception:  # noqa: BLE001
            logger.debug("baseline npy save failed (swallowed)", exc_info=True)
            return None

    def finish_baseline(self, baseline_id: int, *, status: str = "complete",
                        sigma_model: dict | None = None,
                        white_model: dict | None = None,
                        lines: Any = None, repeatability: Any = None,
                        polarity: Any = None,
                        bias_magnitude: Any = None,
                        t_end: float | None = None) -> bool:
        """收尾：写入跨点派生量并定 status。``aborted`` 的行照样保留它测到的点。"""


        def js(v):
            return None if v is None else json.dumps(v, ensure_ascii=False)

        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE noise_baseline SET status=?, sigma_model=?,"
                    " white_model=?, lines=?, repeatability=?, polarity=?,"
                    " bias_magnitude=?, t_end=? WHERE id=?",
                    (str(status), js(sigma_model), js(white_model), js(lines),
                     js(repeatability), js(polarity), js(bias_magnitude),
                     float(t_end or time.time()),
                     int(baseline_id)))
                self._conn.commit()
            return True
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("finish_baseline failed (swallowed)", exc_info=True)
            return False

    def activate_baseline(self, baseline_id: int | None) -> bool:
        """选定用于实时对比的那一份。``None`` = 全部停用（回到固定阈值行为）。

        只有 ``status='complete'`` 的行可被激活 —— 一份半截的表征没有跨点模型，
        激活它等于把判据的分母换成 ``None``，而那与「没有基线」是同一件事，
        却会在 UI 上显示成「有基线」。
        """
        try:
            with self._lock:
                if baseline_id is not None:
                    row = self._conn.execute(
                        "SELECT status FROM noise_baseline WHERE id=?",
                        (int(baseline_id),)).fetchone()
                    if row is None or str(row["status"]) != "complete":
                        logger.info("activate_baseline: %s 不是 complete，拒绝激活",
                                    baseline_id)
                        return False
                self._conn.execute("UPDATE noise_baseline SET active=0")
                if baseline_id is not None:
                    self._conn.execute(
                        "UPDATE noise_baseline SET active=1 WHERE id=?",
                        (int(baseline_id),))
                self._conn.commit()
            return True
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("activate_baseline failed (swallowed)", exc_info=True)
            return False

    def _baseline_row(self, row) -> dict:

        d = dict(row)
        # JSON 字段须同步建表、写入、反序列化与 API schema；读出的字符串须解析为 schema 声明的对象。
        for k in _BASELINE_JSON_COLS:
            raw = d.get(k)
            if raw:
                try:
                    d[k] = json.loads(raw)
                except (ValueError, TypeError):
                    d[k] = None
            else:
                d[k] = None
        d["active"] = bool(d.get("active"))
        return d

    def active_baseline(self) -> Optional[dict]:
        """当前激活的基线行，没有就是 None。判据每段都要问它一次。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM noise_baseline WHERE active=1"
                    " ORDER BY id DESC LIMIT 1").fetchone()
            return self._baseline_row(row) if row is not None else None
        except Exception:  # noqa: BLE001
            logger.debug("active_baseline failed (swallowed)", exc_info=True)
            return None

    def baselines(self, limit: int = 50) -> list[dict]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM noise_baseline ORDER BY id DESC LIMIT ?",
                    (int(limit),)).fetchall()
            return [self._baseline_row(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.debug("baselines failed (swallowed)", exc_info=True)
            return []

    def baseline(self, baseline_id: int,
                 with_points: bool = True) -> Optional[dict]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM noise_baseline WHERE id=?",
                    (int(baseline_id),)).fetchone()
                if row is None:
                    return None
                out = self._baseline_row(row)
                if with_points:
                    pts = self._conn.execute(
                        "SELECT * FROM noise_baseline_point WHERE baseline_id=?"
                        " ORDER BY ordinal, id", (int(baseline_id),)).fetchall()
                    out["points"] = [dict(p) for p in pts]
            return out
        except Exception:  # noqa: BLE001
            logger.debug("baseline read failed (swallowed)", exc_info=True)
            return None

    def baseline_point_curve(self, point_id: int, kind: str = "psd") -> Optional[dict]:
        """读回一个点的谱或直方图。文件没了就返回 None（不编一条曲线出来）。"""
        col = {"psd": "psd_path", "hist": "hist_path",
               "zpsd": "z_psd_path", "coh": "coh_path"}.get(str(kind), "psd_path")
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT %s AS p FROM noise_baseline_point WHERE id=?" % col,
                    (int(point_id),)).fetchone()
            if row is None or not row["p"]:
                return None
            import numpy as _np
            z = _np.load(str(row["p"]))
            return {"x": z["x"].tolist(), "y": z["y"].tolist(), "kind": kind}
        except Exception:  # noqa: BLE001
            logger.debug("baseline_point_curve failed (swallowed)", exc_info=True)
            return None

    def delete_baseline(self, baseline_id: int) -> bool:
        """删一份基线（连同它的 npy）。活跃的那份拒删 —— 删掉正在被判据用的
        分母会让判据在下一段静默换回固定阈值，而没有任何地方说过这件事。"""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT active FROM noise_baseline WHERE id=?",
                    (int(baseline_id),)).fetchone()
                if row is None:
                    return False
                if int(row["active"] or 0):
                    logger.info("delete_baseline: %s 正在被判据使用，拒绝删除",
                                baseline_id)
                    return False
                # 四条曲线路径全部收集 —— 漏一列的后果是文件永远留在盘上，
                # 而基线目录按设计不参与滚动清理，没有第二个人会来收。
                rows = self._conn.execute(
                    "SELECT psd_path, hist_path, z_psd_path, coh_path FROM"
                    " noise_baseline_point WHERE baseline_id=?",
                    (int(baseline_id),)).fetchall()
                paths = [v for r in rows for v in tuple(r) if v]
                self._conn.execute(
                    "DELETE FROM noise_baseline_point WHERE baseline_id=?",
                    (int(baseline_id),))
                self._conn.execute("DELETE FROM noise_baseline WHERE id=?",
                                   (int(baseline_id),))
                self._conn.commit()
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    logger.debug("could not remove %s", p, exc_info=True)
            return True
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("delete_baseline failed (swallowed)", exc_info=True)
            return False

    def pin_range(self, t0: float, t1: float, reason: str) -> int:
        """Pin every segment overlapping [t0, t1]. Returns rows pinned.

        Overlap, not containment: an event at the boundary of a segment must
        keep that segment, and a window shorter than one segment must still pin
        the one it lands in.
        """
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE segments SET pinned=1,"
                    " pin_reason=COALESCE(NULLIF(pin_reason,''), ?)"
                    " WHERE t_end >= ? AND t_start <= ?",
                    (str(reason), float(t0), float(t1)),
                )
                self._conn.commit()
                return int(cur.rowcount or 0)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("pin_range failed (swallowed)", exc_info=True)
            return 0

    # ── read ────────────────────────────────────────────────────────────────

    def latest_feature(self) -> Optional[dict]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM features ORDER BY t_start DESC LIMIT 1"
                ).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            logger.debug("latest_feature failed (swallowed)", exc_info=True)
            return None

    def feature_row(self, segment_id: int) -> Optional[dict]:
        """The stored features for one segment (used by the corpus exporter)."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM features WHERE segment_id=?", (int(segment_id),)
                ).fetchone()
            return dict(row) if row else None
        except Exception:  # noqa: BLE001
            logger.debug("feature_row failed (swallowed)", exc_info=True)
            return None

    def features_query(self, since: float | None = None, until: float | None = None,
                       limit: int = 2000, offset: int = 0,
                       max_points: int | None = None) -> tuple[list[dict], int, bool]:
        """Feature rows in a time range. Returns (rows, total, thinned).

        When ``max_points`` is set and the range holds more rows than that, rows
        are thinned by bucket — keeping the WORST ``alert_level`` in each bucket
        rather than an arbitrary member, so downsampling a chart can never hide
        an alert that actually fired.
        """
        try:
            with self._lock:
                where, args = _range_clause(since, until, "t_start")
                total = int(self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM features{where}", args
                ).fetchone()["n"])
                rows = [dict(r) for r in self._conn.execute(
                    f"SELECT * FROM features{where} ORDER BY t_start ASC"
                    " LIMIT ? OFFSET ?", args + [int(limit), int(offset)],
                ).fetchall()]
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed
            raise StoreQueryFailed(f"features_query 失败: {exc}") from exc

        if not max_points or len(rows) <= max_points:
            return rows, total, False
        return _thin_by_severity(rows, int(max_points)), total, True

    def segments_query(self, since: float | None = None, until: float | None = None,
                       pinned: bool | None = None, label: str | None = None,
                       limit: int = 100, offset: int = 0) -> dict:
        """Segment index with the joined label and verdict. ``label='unlabeled'``
        selects segments no human has judged yet."""
        try:
            with self._lock:
                where, args = _range_clause(since, until, "s.t_start")
                clauses = [where[7:]] if where else []
                if pinned is not None:
                    clauses.append("s.pinned = ?")
                    args.append(1 if pinned else 0)
                if label == "unlabeled":
                    clauses.append("l.label IS NULL")
                elif label:
                    clauses.append("l.label = ?")
                    args.append(str(label))
                w = (" WHERE " + " AND ".join(c for c in clauses if c)) if clauses else ""
                sql_from = (
                    " FROM segments s"
                    " LEFT JOIN (SELECT segment_id, label, note, created_at FROM labels"
                    "            WHERE source='human' GROUP BY segment_id) l"
                    "   ON l.segment_id = s.id"
                    " LEFT JOIN features f ON f.segment_id = s.id"
                )
                total = int(self._conn.execute(
                    f"SELECT COUNT(*) AS n{sql_from}{w}", args).fetchone()["n"])
                rows = [dict(r) for r in self._conn.execute(
                    "SELECT s.*, l.label AS label, l.note AS label_note,"
                    " l.created_at AS label_ts, f.alert_level AS alert_level"
                    f"{sql_from}{w} ORDER BY s.t_start DESC LIMIT ? OFFSET ?",
                    args + [int(limit), int(offset)],
                ).fetchall()]
                counts = self._conn.execute(
                    "SELECT (SELECT COUNT(*) FROM segments WHERE pinned=1) AS pinned,"
                    " (SELECT COUNT(DISTINCT segment_id) FROM labels"
                    "   WHERE source='human' AND segment_id IS NOT NULL) AS labeled"
                ).fetchone()
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed
            raise StoreQueryFailed(f"segments_query 失败: {exc}") from exc

        for r in rows:
            r.pop("envelope", None)             # blob never goes over the wire
            r["has_file"] = bool(r.get("npy_path"))
        return {"segments": rows, "total": total,
                "pinned_count": int(counts["pinned"] or 0),
                "labeled_count": int(counts["labeled"] or 0)}

    def segment_meta(self, segment_id: int, with_envelope: bool = False) -> Optional[dict]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT s.*, f.alert_level AS alert_level, l.label AS label,"
                    " l.note AS label_note, l.created_at AS label_ts"
                    " FROM segments s"
                    " LEFT JOIN features f ON f.segment_id = s.id"
                    " LEFT JOIN (SELECT segment_id, label, note, created_at FROM labels"
                    "            WHERE source='human' GROUP BY segment_id) l"
                    "   ON l.segment_id = s.id WHERE s.id = ?", (int(segment_id),),
                ).fetchone()
            if not row:
                return None
            d = dict(row)
            if not with_envelope:
                d.pop("envelope", None)
            d["has_file"] = bool(d.get("npy_path"))
            return d
        except Exception:  # noqa: BLE001
            logger.debug("segment_meta failed (swallowed)", exc_info=True)
            return None

    def read_segment_decimated(self, segment_id: int, max_points: int = 4000) -> Optional[dict]:
        """Samples for one segment, min/max-decimated for display.

        Falls back to the stored envelope when the raw ``.npy`` has been swept
        (``source='envelope'``), so the browser degrades to a coarser view
        instead of an empty one.
        """
        import numpy as np

        meta = self.segment_meta(segment_id, with_envelope=True)
        if not meta:
            return None
        fs = float(meta.get("fs_hz") or 0.0)
        t0 = float(meta.get("t_start") or 0.0)
        raw_path = meta.get("npy_path")
        arr = None
        if raw_path:
            try:
                arr = np.load(str(raw_path), mmap_mode="r")
                arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001 — file may have been swept mid-read
                logger.debug("segment npy unreadable: %s", raw_path, exc_info=True)
                arr = None

        if arr is not None and arr.size:
            t_rel, vals, decimated = _minmax_decimate(arr, fs, int(max_points))
            source = "raw"
            n_raw = int(arr.size)
        else:
            env = meta.get("envelope")
            if not env:
                return None
            e = np.frombuffer(env, dtype=np.float32).reshape(2, -1)
            dt = float(meta.get("envelope_dt_s") or 0.0)
            # Interleave min/max so the drawn band still shows the extremes.
            k = e.shape[1]
            t_rel = np.repeat(np.arange(k, dtype=np.float64) * dt, 2).tolist()
            vals = np.empty(2 * k, dtype=np.float64)
            vals[0::2] = e[0]
            vals[1::2] = e[1]
            vals = vals.tolist()
            source = "envelope"
            decimated = True
            n_raw = int(meta.get("n_samples") or 0)

        meta.pop("envelope", None)
        return {"seg_id": int(segment_id), "t0": t0, "fs_hz": fs,
                "n_samples_raw": n_raw, "t_s": t_rel, "i_a": vals,
                "source": source, "decimated": decimated, "meta": meta}

    def segment_psd(self, segment_id: int) -> Optional[dict]:
        """PSD of one segment, computed from the FULL-rate samples.

        Never from the decimated view — decimation aliases everything above the
        new Nyquist onto the bands the operator is trying to read.
        """
        import numpy as np

        meta = self.segment_meta(segment_id)
        if not meta or not meta.get("npy_path"):
            return None
        try:
            arr = np.asarray(np.load(str(meta["npy_path"])), dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            logger.debug("segment_psd load failed", exc_info=True)
            return None
        if arr.size < 16:
            return None
        from mast.io.signal_fft import compute_fft
        out = compute_fft(
            {"samples": arr.tolist(), "fs_hz": float(meta.get("fs_hz") or 0.0),
             "unit": "A", "channel_name": meta.get("channel_name") or "Current (A)"},
            window="hann", detrend=True, output="power",
        )
        return out or None

    def live_tail(self, window_s: float = 60.0, max_points: int = 1200) -> dict:
        """Recent envelopes stitched into one min/max band for the live chart.

        Reduction happens TWICE, and the first pass is why long windows are
        affordable at all. Inside each segment the envelope is reduced to a few
        times the final budget before anything becomes a Python float; only then
        are the segments concatenated and reduced once more onto ``max_points``.

        The naive order (expand everything, then decimate) is what this replaces:
        a request to make real-time current viewable for longer raised the ceiling from
        600 s to 6 h, and at ~1 segment/s with a 10 ms envelope step that is
        ~21k segments ≈ 2M points — materialised as Python floats to then discard
        99.9% of them, on an endpoint the page polls.

        Both passes reduce by min/max, never by mean or by subsampling. This band
        exists to show excursions, so the one operation that must not lose a
        spike is the one that shrinks the data.

        采集**中断**的地方，``i_min_a`` / ``i_max_a`` 里放一个 ``None``
        （需求：数据中断的时候不应强行连线）。前端把 null 画成空洞，
        线和中间那片带子一起断开。判据见 :func:`trace_gap_marks` —— 它必须在这一层
        判，因为「段内的包络步长」和「段之间的停顿」只有这里分得开。
        """
        import numpy as np

        now = time.time()
        since = now - max(1.0, float(window_s))
        empty = {"t_s": [], "i_min_a": [], "i_max_a": [], "last_ts": None,
                 "n_segments": 0, "n_gaps": 0}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT t_start, envelope, envelope_dt_s FROM segments"
                    " WHERE t_end >= ? AND envelope IS NOT NULL"
                    " ORDER BY t_start ASC", (since,),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed。**注意下一行**:
            # ``not rows`` 那条仍然回 ``empty`` —— 那是「查过了，这段窗口里没有段」，
            # 一个答得上来的答案。两条路径长得像，说的是相反的两句话。
            raise StoreQueryFailed(f"live_tail 失败: {exc}") from exc
        if not rows:
            return dict(empty)

        cap = int(max_points) if max_points and max_points > 0 else 0
        valid = [r for r in rows
                 if r["envelope"] is not None and float(r["envelope_dt_s"] or 0.0) > 0]
        # Point count straight off the blob length — two float32 rows, 8 B per
        # point — so the stride is known before a single byte is decoded.
        total = sum(len(r["envelope"]) // 8 for r in valid)
        seg_stride = 1
        if cap and total > 4 * cap:
            seg_stride = -(-total // (4 * cap))  # ceil

        t_parts: list = []
        lo_parts: list = []
        hi_parts: list = []
        #: 每一段真实覆盖到的 (起, 止)。用 ``base + n*dt`` 而不是最后一个画出来的点：
        #: 抽稀会丢掉段尾的时间戳，拿它当段末会把段间空档**虚报**成长一截，
        #: 于是正常停顿被判成中断。覆盖范围要按原始点数算。
        spans: list[tuple[float, float]] = []
        for r in valid:
            dt = float(r["envelope_dt_s"])
            e = np.frombuffer(r["envelope"], dtype=np.float32).reshape(2, -1)
            n = e.shape[1]
            if n == 0:
                continue
            base = float(r["t_start"])
            spans.append((base, base + n * dt))
            s = min(seg_stride, n)
            if s <= 1:
                t_parts.append(base + np.arange(n) * dt)
                lo_parts.append(e[0].astype(np.float64))
                hi_parts.append(e[1].astype(np.float64))
                continue
            k = n // s * s
            t_parts.append(base + np.arange(0, k, s) * dt)
            lo_parts.append(e[0, :k].astype(np.float64).reshape(-1, s).min(axis=1))
            hi_parts.append(e[1, :k].astype(np.float64).reshape(-1, s).max(axis=1))
            if k < n:
                # The ragged tail becomes its own bucket rather than being
                # dropped: a spike in the last 30 ms of a segment is exactly the
                # kind of thing this chart is watched for.
                t_parts.append(np.asarray([base + k * dt]))
                lo_parts.append(np.asarray([float(e[0, k:].min())]))
                hi_parts.append(np.asarray([float(e[1, k:].max())]))

        if not t_parts:
            return dict(empty, n_segments=len(rows))

        t_arr = np.concatenate(t_parts)
        lo_arr = np.concatenate(lo_parts)
        hi_arr = np.concatenate(hi_parts)

        if cap and t_arr.size > cap:
            step = -(-t_arr.size // cap)
            k = t_arr.size // step * step
            t_arr = t_arr[:k].reshape(-1, step)[:, 0]
            lo_arr = lo_arr[:k].reshape(-1, step).min(axis=1)
            hi_arr = hi_arr[:k].reshape(-1, step).max(axis=1)

        ts, lo, hi = _splice_trace_gaps(
            t_arr.tolist(), lo_arr.tolist(), hi_arr.tolist(),
            trace_gap_marks(spans))
        #: ``last_ts`` 取最后一个**真实**读数。断点只可能插在两段之间，末尾永远是
        #: 真数据，但这里不靠那个前提 —— 页面用它算「数据停更了多久」，
        #: 一个 None 会让「停更」这件事本身消失。
        last = next((t for t, v in zip(reversed(ts), reversed(lo))
                     if v is not None), None)
        return {"t_s": ts, "i_min_a": lo, "i_max_a": hi,
                "last_ts": last, "n_segments": len(rows),
                "n_gaps": sum(1 for v in lo if v is None)}

    def alerts_query(self, since: float | None = None, limit: int = 50,
                     level: str | None = None) -> tuple[list[dict], int]:
        """``(rows, total)``。查询失败 ⇒ :class:`StoreQueryFailed`，**不是 ``([], 0)``**。

        ``total`` 是一个计数：``0`` 说的是「查过了，这段时间零条告警」。让一次
        查询失败也说这句话，conduct 的 L1 闸门就会拿它当「很太平」放行一整夜。
        """
        try:
            with self._lock:
                where, args = _range_clause(since, None, "ts")
                clauses = [where[7:]] if where else []
                if level:
                    clauses.append("level = ?")
                    args.append(str(level))
                w = (" WHERE " + " AND ".join(c for c in clauses if c)) if clauses else ""
                total = int(self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM alerts{w}", args).fetchone()["n"])
                rows = [dict(r) for r in self._conn.execute(
                    "SELECT id, ts, level, rule, summary_zh, segment_id,"
                    " (evidence_png IS NOT NULL) AS evidence_available,"
                    " emitted_buffer, acked, delivered_agent"
                    f" FROM alerts{w} ORDER BY ts DESC LIMIT ?",
                    args + [int(limit)],
                ).fetchall()]
            return rows, total
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed：读不折叠
            raise StoreQueryFailed(f"alerts_query 失败: {exc}") from exc

    def alert_evidence_png(self, alert_id: int) -> Optional[bytes]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT evidence_png FROM alerts WHERE id=?", (int(alert_id),)
                ).fetchone()
            if not row or not row["evidence_png"]:
                return None
            p = Path(row["evidence_png"])
            return p.read_bytes() if p.is_file() else None
        except Exception:  # noqa: BLE001
            logger.debug("alert_evidence_png failed (swallowed)", exc_info=True)
            return None

    def labels_query(self, since: float | None = None, limit: int = 500) -> list[dict]:
        try:
            with self._lock:
                where, args = _range_clause(since, None, "t_start")
                return [dict(r) for r in self._conn.execute(
                    f"SELECT * FROM labels{where} ORDER BY t_start DESC LIMIT ?",
                    args + [int(limit)],
                ).fetchall()]
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed
            raise StoreQueryFailed(f"labels_query 失败: {exc}") from exc

    def storage_stats(self) -> dict:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS segments,"
                    " SUM(CASE WHEN npy_path IS NOT NULL THEN 1 ELSE 0 END) AS on_disk,"
                    " COALESCE(SUM(npy_bytes),0) AS bytes,"
                    " MIN(t_start) AS oldest, MAX(t_end) AS newest,"
                    " SUM(pinned) AS pinned FROM segments"
                ).fetchone()
            return {"segments": int(row["segments"] or 0),
                    "segments_on_disk": int(row["on_disk"] or 0),
                    "store_bytes": int(row["bytes"] or 0),
                    "oldest": row["oldest"], "newest": row["newest"],
                    "pinned": int(row["pinned"] or 0),
                    "db_path": str(self._path)}
        except Exception as exc:  # noqa: BLE001 — 见 StoreQueryFailed
            # 旧行为回的是一整套 0:「0 段、0 字节」是关于磁盘的**正面断言**,
            # 而它同时也是「库打不开」。面板照着这套 0 画出来的是一个空盘。
            raise StoreQueryFailed(f"storage_stats 失败: {exc}") from exc

    # ── retention ───────────────────────────────────────────────────────────

    def sweep_aux(self, keep_hours: float) -> int:
        """按 keep_hours 删除过期辅助采样，返回删除行数。辅助采样按年龄保留，不与波形文件共用容量淘汰预算；采样率提高会增加保留期内的存储量。"""
        try:
            cutoff = time.time() - max(0.0, float(keep_hours)) * 3600.0
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM aux_samples WHERE ts < ?", (cutoff,))
                self._conn.commit()
                return int(cur.rowcount or 0)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("sweep_aux failed (swallowed)", exc_info=True)
            return 0

    def retention_sweep(self, keep_hours: float, keep_gb: float) -> dict:
        """Drop raw ``.npy`` files by age and by total size. Rows always survive.

        Pinned segments are exempt from both rules — that is the whole point of
        pinning. Two passes because they answer different questions: "this is
        older than I care about" and "I am out of disk".

        Pinned bytes still COUNT towards the budget while being unreclaimable,
        so pinning more than ``keep_gb`` sweeps away every unpinned segment and
        still overshoots. That case is reported (``over_budget``) and logged
        rather than left to look like a successful sweep — the disk keeps
        growing and only a human can decide whether to raise the cap or unpin.
        """
        removed = 0
        freed = 0
        over_budget = False
        pinned_bytes = 0
        try:
            cutoff = time.time() - max(0.0, float(keep_hours)) * 3600.0
            with self._lock:
                stale = self._conn.execute(
                    "SELECT id, npy_path, npy_bytes FROM segments"
                    " WHERE pinned=0 AND npy_path IS NOT NULL AND t_end < ?",
                    (cutoff,),
                ).fetchall()
            for row in stale:
                freed += self._drop_npy(row)
                removed += 1

            budget = int(max(0.0, float(keep_gb)) * (1 << 30))
            with self._lock:
                sizes = self._conn.execute(
                    "SELECT COALESCE(SUM(npy_bytes),0) AS total,"
                    " COALESCE(SUM(CASE WHEN pinned=1 THEN npy_bytes ELSE 0 END),0)"
                    "   AS pinned FROM segments WHERE npy_path IS NOT NULL"
                ).fetchone()
            total = int(sizes["total"] or 0)
            pinned_bytes = int(sizes["pinned"] or 0)
            if total > budget:
                with self._lock:
                    candidates = self._conn.execute(
                        "SELECT id, npy_path, npy_bytes FROM segments"
                        " WHERE pinned=0 AND npy_path IS NOT NULL"
                        " ORDER BY t_start ASC").fetchall()
                for row in candidates:
                    if total <= budget:
                        break
                    n = self._drop_npy(row)
                    total -= n
                    freed += n
                    removed += 1
                if total > budget:
                    over_budget = True
                    # Log the EDGE, not the state. This condition persists until
                    # a human acts on it, and repeating the warning every sweep
                    # would bury whatever else is in the log — the same reason
                    # the service only publishes state events on transition.
                    if not self._over_budget:
                        logger.warning(
                            "current monitor: pinned segments hold %.2f GB, over "
                            "the %.2f GB retention budget — nothing left to "
                            "reclaim. Raise cm_keep_gb or unpin some segments.",
                            pinned_bytes / (1 << 30), keep_gb,
                        )
            if self._over_budget and not over_budget:
                logger.info("current monitor: retention is back within budget")
            self._over_budget = over_budget
        except Exception:  # noqa: BLE001
            logger.debug("retention_sweep failed (swallowed)", exc_info=True)
        return {"removed": removed, "freed_bytes": freed,
                "pinned_bytes": pinned_bytes, "over_budget": over_budget}

    def _drop_npy(self, row: sqlite3.Row) -> int:
        """Delete one segment's raw file and null its path. Returns bytes freed.

        Claim first, delete second — and re-check ``pinned`` inside the claim.
        A sweep can run for minutes over tens of thousands of rows, releasing the
        lock between each one, and the two things most likely to pin a segment
        during that window are exactly the two worth keeping: a tip-shaping skill
        finishing (SKILL_STEP → pin_range) and an operator pressing 「标记为好针尖」.
        Selecting ``pinned=0`` once at the start and trusting it for the rest of
        the sweep would delete the waveform behind a label that was just applied.

        The file is only removed after the row is claimed, and the claim is only
        kept if the removal succeeded: nulling the path for a file still on disk
        would orphan it forever (every later sweep filters on
        ``npy_path IS NOT NULL``) and permanently under-count the disk budget.
        On Windows this is not hypothetical — ``read_segment_decimated`` opens
        segments with ``mmap_mode="r"``, and a mapped file cannot be unlinked.
        """
        seg_id = int(row["id"])
        size = int(row["npy_bytes"] or 0)
        path = row["npy_path"]
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE segments SET npy_path=NULL, npy_bytes=0"
                    " WHERE id=? AND pinned=0 AND npy_path IS NOT NULL",
                    (seg_id,),
                )
                claimed = int(cur.rowcount or 0) == 1
                self._conn.commit()
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("could not claim segment %s for sweep", seg_id, exc_info=True)
            return 0
        if not claimed:
            return 0            # pinned (or already swept) since the snapshot

        if not path:
            return size
        p = Path(path)
        try:
            if p.is_file():
                size = size or p.stat().st_size
                os.remove(p)
        except OSError:
            # Put the row back so the file stays visible to a later sweep.
            logger.debug("could not remove %s — restoring the row", path, exc_info=True)
            try:
                with self._lock:
                    self._conn.execute(
                        "UPDATE segments SET npy_path=?, npy_bytes=? WHERE id=?",
                        (str(path), size, seg_id),
                    )
                    self._conn.commit()
            except Exception:  # noqa: BLE001
                self._rollback()
                logger.debug("could not restore npy_path for %s", seg_id, exc_info=True)
            return 0
        return size

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


# ── helpers ──────────────────────────────────────────────────────────────────


def _f_or_none(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _num_or_none(v) -> Optional[float]:
    """Convert numeric values while preserving None as unknown. False or zero is an observation, whereas None may indicate a failed read; storage must keep that distinction even when the alert verdict happens to be the same."""
    if v is None or isinstance(v, bool):
        return None if v is None else (1.0 if v else 0.0)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or f in (float("inf"), float("-inf"))) else f


def _bool_or_none(v) -> Optional[int]:
    if v is None:
        return None
    return 1 if v else 0


def _range_clause(since: float | None, until: float | None,
                  col: str) -> tuple[str, list]:
    clauses, args = [], []
    if since is not None:
        clauses.append(f"{col} >= ?")
        args.append(float(since))
    if until is not None:
        clauses.append(f"{col} < ?")
        args.append(float(until))
    return ((" WHERE " + " AND ".join(clauses)) if clauses else "", args)


_SEVERITY_ORDER = {"ok": 0, "suppressed": 0, "warn": 1, "critical": 2}


def _thin_by_severity(rows: list[dict], max_points: int,
                      key: str = "alert_level") -> list[dict]:
    """Bucket-thin a row list, keeping the worst-verdict row in each bucket.

    ``key`` is the verdict column: ``alert_level`` on ``features``, ``verdict``
    on ``aux_samples``. Passing the wrong one would silently make every row look
    like ``ok`` and thin arbitrarily — which is exactly the behaviour this
    function exists to prevent.
    """
    step = max(1, len(rows) // max_points)
    out: list[dict] = []
    for i in range(0, len(rows), step):
        chunk = rows[i:i + step]
        out.append(max(chunk, key=lambda r: _SEVERITY_ORDER.get(
            str(r.get(key) or "ok"), 0)))
    return out


def _minmax_decimate(arr, fs_hz: float, max_points: int):
    """Min/max decimation preserving spikes; returns (t_rel, values, decimated).

    Each bucket contributes its argmin and argmax IN INDEX ORDER, so the drawn
    line keeps both the extreme and the direction it happened in. Same algorithm
    as the frontend's chart decimator, so a segment looks the same whether the
    thinning happened on the server or in the browser.
    """
    import numpy as np

    n = int(arr.size)
    dt = 1.0 / fs_hz if fs_hz > 0 else 0.0
    if max_points <= 0 or n <= max_points:
        return (np.arange(n) * dt).tolist(), arr.tolist(), False
    buckets = max(1, max_points // 2)
    step = int(np.ceil(n / buckets))
    idx_out: list[int] = []
    for start in range(0, n, step):
        chunk = arr[start:start + step]
        if chunk.size == 0:
            continue
        lo = start + int(np.argmin(chunk))
        hi = start + int(np.argmax(chunk))
        idx_out.extend((lo, hi) if lo <= hi else (hi, lo))
    idx = np.asarray(idx_out, dtype=np.int64)
    return (idx * dt).tolist(), arr[idx].tolist(), True


# ── process-global singleton ─────────────────────────────────────────────────

_STORE: CurrentMonitorStore | None = None
_STORE_LOCK = threading.Lock()


def _default_dir() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "experiments" / "current_monitor"


def get_store() -> CurrentMonitorStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                base = _default_dir()
                _STORE = CurrentMonitorStore(base / "monitor.sqlite", base)
    return _STORE


def get_store_if_exists() -> "CurrentMonitorStore | None":
    """The store **only if something already made one**. Never creates.

    :func:`get_store` creates the DB (and its directories) on first call under
    ``project_root()/experiments/current_monitor/``. That is right for the
    acquisition daemon — it is the thing whose job is to have a store — and
    wrong for every **passive reader**:

    * a reader that creates an empty DB has, by construction, nothing to read,
      so the creation bought nothing;
    * and in tests it writes into the operator's REAL experiments directory.
      ``tests/v2/conftest.py`` has autouse guards for the wishlist and the
      literature registry precisely because that has happened before —
      **there is no such guard for this store**, so a passive reader calling
      ``get_store()`` is a pollution path with nothing standing behind it.

    Passive readers (the agent-facing alert delivery middleware, the
    ``ForgeAuTip`` outer-loop self-check) use THIS. If no monitor has ever run
    there are no alerts, and ``None`` is the honest answer.
    """
    return _STORE


def set_store_for_test(store: CurrentMonitorStore | None) -> None:
    """Swap the singleton (tests point it at a tmp dir)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store


__all__ = ["CurrentMonitorStore", "StoreQueryFailed", "get_store",
           "get_store_if_exists", "set_store_for_test", "segment_npy_path"]
