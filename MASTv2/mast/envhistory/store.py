"""环境历史的独立 SQLite 存储 —— 统计桶 + 噪声谱快照。

形制与 :mod:`mast.monitoring.store`（源头是 :mod:`mast.billing.ledger`）一致：
自己的 DB 文件、一条长连接配一把锁、WAL、写方法自己吞异常。**一次记录写入失败
绝不能把采集带下去** —— 丢一个桶是不便，把正在看着真空的监控线程搞崩不是。

什么永久、什么会滚掉
--------------------

* ``env_buckets`` / ``env_spectra`` 的行是**永久**的。满负荷 7 条序列 × 1 min
  桶 ≈ 0.44 GB/年，双通道谱 @30 min ≈ 70 MB/年 —— 这个量级不值得设计一套清理。
* 会滚掉的是**别人家**的表：``environment_log`` 里 2 秒一条的原始行，由
  :meth:`mast.logging.storage.ExperimentStorage.prune_environment_log` 删,
  本模块只负责在到期时喊一声。
* 实验文件夹里的 CSV 是权威原始记录，本子系统**永不**触碰。

为什么没有 1 小时粗桶
---------------------

因为 SQLite 对 50 万行做 ``GROUP BY CAST(bucket_ts/step AS INT)`` 是亚秒级的,
而且加权重算（``SUM(mean*n)/SUM(n)``、``MIN(min)``、``MAX(max)``）在数学上与
直接从原始数据聚合**完全相等**，不是近似。多一层粗桶省下的只是这点查询时间,
换来的却是第二条写路径和一次回填迁移 —— 不值。
"""
from __future__ import annotations

import array
import logging
import math
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

_SCHEMA = """
-- 统计桶,永久。主键即数据(WITHOUT ROWID):查询恒为 (sensor, 时间区间),
-- 主键序直接服务它,省掉一条独立的 ts 索引 —— 这张表只增不减,一年百万行,
-- 省下的那份索引是实打实的。
CREATE TABLE IF NOT EXISTS env_buckets (
    sensor        TEXT NOT NULL,
    bucket_ts     REAL NOT NULL,               -- epoch s, floor(ts/dt)*dt
    dt_s          REAL NOT NULL DEFAULT 60,    -- 桶宽;用户改了之后新行携新值
    n             INTEGER NOT NULL,            -- 进入统计的读数条数
    mean          REAL, min REAL, max REAL, std REAL,
    last          REAL,                        -- 桶内最后一条有效值
    unit          TEXT NOT NULL DEFAULT '',
    worst_status  TEXT NOT NULL DEFAULT 'ok',  -- 含被排除读数在内的最差状态
    n_excluded    INTEGER NOT NULL DEFAULT 0,  -- error/unavailable/被门拒的条数
    experiment_id TEXT, sample_id TEXT,
    PRIMARY KEY (sensor, bucket_ts, dt_s)
) WITHOUT ROWID;

-- 噪声谱快照,永久。freqs/psd 各是 float32 数组的裸字节,行 ~2 KB 且自包含。
-- 刻意不做"共享频率网格表":时基可变、fs 可变,省下的那半 KB 不值跨表引用
-- 带来的脆性(一条谱必须能单独读出来画)。
CREATE TABLE IF NOT EXISTS env_spectra (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,               -- 快照发射时刻
    channel       TEXT NOT NULL,               -- 'current' | 'z'
    span_s        REAL NOT NULL,               -- 本条覆盖的攒谱时间窗
    n_segments    INTEGER NOT NULL,            -- 参与 median 的段数
    fs_hz         REAL NOT NULL,
    f_lo_hz       REAL NOT NULL,
    f_hi_hz       REAL NOT NULL,
    n_points      INTEGER NOT NULL,
    freqs         BLOB NOT NULL,
    psd           BLOB NOT NULL,
    unit          TEXT NOT NULL DEFAULT 'A^2/Hz',
    quietness     TEXT NOT NULL DEFAULT 'quiet',
    ctx_bias_v    REAL, ctx_setpoint_a REAL, ctx_zctrl_on INTEGER,
    ctx_stable    INTEGER,                      -- 攒谱期间工作点变没变过
    experiment_id TEXT, sample_id TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spectra_chan_ts ON env_spectra(channel, ts);
"""

#: 2 = ``env_spectra.ctx_stable``（新增需求：噪声谱需要记录当时的
#:     状态）。**必须配 ALTER**：``CREATE TABLE IF NOT EXISTS`` 对已存在的表
#:     一个字都不改，而这个类的写方法全部吞异常 —— 真机上老库会变成
#:     「INSERT 报没有这一列 → 吞掉 → 谱记录悄悄停了」，而页面上看起来只是
#:     「最近没攒到谱」。
#: 1 = 初版。
_SCHEMA_VERSION = 2

#: 建表语句之后才加进来的列。``(表, 列, SQL 声明)``，开库时逐条补。
#:
#: 新列必须**可空**：ALTER TABLE ADD COLUMN 加 NOT NULL 必须给默认值，而给了默认值
#: 就分不出「老行没有这个信息」和「老行的值恰好等于默认值」。
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("env_spectra", "ctx_stable", "INTEGER"),
)

#: 与 ``mast.environment.alarm._SEVERITY_ORDER`` 同序。在 SQL 里 MAX(TEXT) 是
#: 字典序（'warning' > 'error' > 'alarm'），会把最糟的那个排到最前面之外去,
#: 所以再分桶时必须先映射成数字。
_SEVERITY_ORDER: dict[str, int] = {
    "ok": 0, "unavailable": 1, "warning": 2, "error": 3, "alarm": 4,
}
_SEVERITY_BY_RANK: dict[int, str] = {v: k for k, v in _SEVERITY_ORDER.items()}

_SEVERITY_CASE = (
    "MAX(CASE worst_status WHEN 'alarm' THEN 4 WHEN 'error' THEN 3"
    " WHEN 'warning' THEN 2 WHEN 'unavailable' THEN 1 ELSE 0 END)"
)

_BUCKET_COLS = ("sensor", "bucket_ts", "dt_s", "n", "mean", "min", "max",
                "std", "last", "unit", "worst_status", "n_excluded",
                "experiment_id", "sample_id")


def _pack_f32(values: Sequence[float]) -> bytes:
    """float 序列 → 小端 float32 裸字节。不依赖 numpy。"""
    a = array.array("f", (float(v) for v in values))
    if sys.byteorder != "little":  # pragma: no cover — 目标平台都是小端
        a.byteswap()
    return a.tobytes()


def _unpack_f32(blob: bytes | memoryview | None) -> list[float]:
    """小端 float32 裸字节 → float 列表。读端**刻意不用 numpy**：API 层要能在
    没有 numpy 的机器上把一条谱还给前端。"""
    if not blob:
        return []
    a = array.array("f")
    try:
        a.frombytes(bytes(blob))
    except ValueError:  # 长度不是 4 的倍数 —— 半条写坏的行，返回空而不是抛
        return []
    if sys.byteorder != "little":  # pragma: no cover
        a.byteswap()
    return [float(x) for x in a]


class EnvHistoryStore:
    """环境历史库：统计桶 + 噪声谱。所有写方法永不抛。"""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._add_missing_columns()
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            except sqlite3.Error:
                pass
            self._conn.commit()

    def _add_missing_columns(self) -> None:
        """把 :data:`_ADDED_COLUMNS` 补进老库。调用方已持锁。

        按 ``PRAGMA table_info`` 实测有没有，而不是按 ``user_version`` 推断：
        版本号是我们自己写的，一次写失败（磁盘满、库被占）就会让它和真实
        schema 说不同的话，而之后每次开库都会相信那句假话。列在不在是可以直接
        问出来的事实。
        """
        for table, column, decl in _ADDED_COLUMNS:
            try:
                have = {r["name"] for r in
                        self._conn.execute(f"PRAGMA table_info({table})")}
                if column in have:
                    continue
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                logger.info("env history: 补列 %s.%s", table, column)
            except sqlite3.Error:  # pragma: no cover
                # 补不上也要能开库：读旧数据比写新字段重要。
                logger.warning("env history: 无法补列 %s.%s", table, column,
                               exc_info=True)

    @property
    def path(self) -> Path:
        return self._path

    # ── 写（吞异常 + 先回滚，绝不抛） ────────────────────────────────

    def _rollback(self) -> None:
        """吞掉异常之前先撤销半成品写入。

        sqlite3 对 DML 仍然开隐式事务：INSERT 成功而 commit 失败（磁盘满是现实
        场景）会留下一个未提交事务，之后每一次写入都并进它，WAL 一路涨，然后
        一次崩溃把它们全丢掉 —— 恰恰是这个子系统要积累的东西。
        """
        try:
            self._conn.rollback()
        except Exception:  # noqa: BLE001 — 尽力而为；调用方已经在失败路径上
            pass

    def upsert_buckets(self, buckets: Iterable[Any]) -> int:
        """写入若干完成的桶。返回成功写入的行数（失败返回 0，不抛）。

        用 INSERT OR REPLACE：同一 (sensor, bucket_ts, dt_s) 重复写是幂等的。
        这在两种正常情况下会发生 —— 换实验时 flush 了一个未满的桶，之后又收到
        属于同一分钟的读数；以及测试里重放同一段数据。
        """
        rows = [b.to_row() if hasattr(b, "to_row") else tuple(b) for b in buckets]
        if not rows:
            return 0
        try:
            with self._lock:
                self._conn.executemany(
                    f"INSERT OR REPLACE INTO env_buckets ({','.join(_BUCKET_COLS)})"
                    f" VALUES ({','.join('?' * len(_BUCKET_COLS))})",
                    rows,
                )
                self._conn.commit()
            return len(rows)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("env history bucket write failed (swallowed)", exc_info=True)
            return 0

    def add_spectrum(self, *, ts: float, channel: str, span_s: float,
                     n_segments: int, fs_hz: float,
                     freqs: Sequence[float], psd: Sequence[float],
                     unit: str = "A^2/Hz", quietness: str = "quiet",
                     ctx: dict | None = None, ctx_stable: bool | None = None,
                     experiment_id: str | None = None,
                     sample_id: str | None = None,
                     created_at: float | None = None) -> Optional[int]:
        """写一条谱快照。返回行 id，失败返回 None。"""
        if not freqs or not psd or len(freqs) != len(psd):
            return None
        ctx = ctx or {}
        zctrl = ctx.get("ctx_zctrl_on")
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO env_spectra (ts, channel, span_s, n_segments, fs_hz,"
                    " f_lo_hz, f_hi_hz, n_points, freqs, psd, unit, quietness,"
                    " ctx_bias_v, ctx_setpoint_a, ctx_zctrl_on, ctx_stable,"
                    " experiment_id, sample_id, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        float(ts), str(channel), float(span_s), int(n_segments),
                        float(fs_hz), float(freqs[0]), float(freqs[-1]), len(freqs),
                        _pack_f32(freqs), _pack_f32(psd), str(unit), str(quietness),
                        _f_or_none(ctx.get("ctx_bias_v")),
                        _f_or_none(ctx.get("ctx_setpoint_a")),
                        None if zctrl is None else (1 if zctrl else 0),
                        None if ctx_stable is None else (1 if ctx_stable else 0),
                        experiment_id or None, sample_id or None,
                        float(created_at if created_at is not None else ts),
                    ),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001
            self._rollback()
            logger.debug("env history spectrum write failed (swallowed)", exc_info=True)
            return None

    # ── 读 ────────────────────────────────────────────────────────────

    def list_sensors(self) -> list[dict]:
        """桶里出现过的序列 + 时间范围 + 桶数。"""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT sensor, MAX(unit) AS unit, COUNT(*) AS n_buckets,"
                    " MIN(bucket_ts) AS first_ts, MAX(bucket_ts) AS last_ts"
                    " FROM env_buckets GROUP BY sensor ORDER BY sensor"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.debug("list_sensors failed", exc_info=True)
            return []

    def series(self, sensor: str, since: float | None = None,
               until: float | None = None,
               max_points: int | None = None) -> dict:
        """一条序列的桶。点数超限时在 **SQL 侧**再分桶，不在 Python 里抽稀。

        再分桶是加权重算而不是取样：``mean`` 用 ``Σ(mean·n)/Σn``、``min``/``max``
        取真正的极值、``std`` 用合并方差。所以放大看和缩小看是同一条曲线,
        而且**缩小视图不会藏掉一个越限点** —— 极值和最差状态都被保下来。
        """
        out: dict = {"sensor": sensor, "unit": "", "points": [],
                     "bucket_s_effective": 0.0, "thinned": False, "total": 0}
        where = ["sensor = ?"]
        params: list = [str(sensor)]
        if since is not None:
            where.append("bucket_ts >= ?")
            params.append(float(since))
        if until is not None:
            where.append("bucket_ts <= ?")
            params.append(float(until))
        clause = " AND ".join(where)
        try:
            with self._lock:
                head = self._conn.execute(
                    f"SELECT COUNT(*) AS n, MIN(bucket_ts) AS lo, MAX(bucket_ts) AS hi,"
                    f" MIN(dt_s) AS dt FROM env_buckets WHERE {clause}", params
                ).fetchone()
                total = int(head["n"] or 0)
                out["total"] = total
                if total == 0:
                    return out
                native_dt = float(head["dt"] or 60.0) or 60.0
                unit_row = self._conn.execute(
                    "SELECT unit FROM env_buckets WHERE sensor = ? AND unit <> ''"
                    " ORDER BY bucket_ts DESC LIMIT 1", [str(sensor)]
                ).fetchone()
                out["unit"] = str(unit_row["unit"]) if unit_row else ""

                mp = int(max_points) if max_points else 0
                if mp <= 0 or total <= mp:
                    out["bucket_s_effective"] = native_dt
                    rows = self._conn.execute(
                        f"SELECT bucket_ts AS ts, mean, min, max, std, n, n_excluded,"
                        f" worst_status FROM env_buckets WHERE {clause}"
                        f" ORDER BY bucket_ts", params
                    ).fetchall()
                    out["points"] = [
                        {"ts": float(r["ts"]), "mean": _f_or_none(r["mean"]),
                         "min": _f_or_none(r["min"]), "max": _f_or_none(r["max"]),
                         "std": _f_or_none(r["std"]), "n": int(r["n"] or 0),
                         "n_excluded": int(r["n_excluded"] or 0),
                         "worst_status": str(r["worst_status"] or "ok")}
                        for r in rows
                    ]
                    return out

                span = max(native_dt, float(head["hi"]) - float(head["lo"]) + native_dt)
                step = max(native_dt, span / float(mp))
                # 对齐到原生桶宽的整数倍，免得分组边界在两个原生桶之间来回切。
                step = math.ceil(step / native_dt) * native_dt
                out["bucket_s_effective"] = step
                out["thinned"] = True
                grouped = self._conn.execute(
                    f"SELECT CAST(bucket_ts / ? AS INTEGER) * ? AS ts,"
                    f" SUM(mean * n) / NULLIF(SUM(n), 0) AS mean,"
                    f" MIN(min) AS min, MAX(max) AS max,"
                    f" SUM(n) AS n, SUM(n_excluded) AS n_excluded,"
                    f" SUM(n * (std * std + mean * mean)) / NULLIF(SUM(n), 0) AS m2,"
                    f" {_SEVERITY_CASE} AS sev"
                    f" FROM env_buckets WHERE {clause}"
                    f" GROUP BY CAST(bucket_ts / ? AS INTEGER) ORDER BY ts",
                    [step, step, *params, step],
                ).fetchall()
            pts = []
            for r in grouped:
                mean = _f_or_none(r["mean"])
                m2 = _f_or_none(r["m2"])
                std = None
                if mean is not None and m2 is not None:
                    # 合并方差 = E[x²] - E[x]²。浮点抵消可能给出微小负数。
                    std = math.sqrt(max(0.0, m2 - mean * mean))
                pts.append({
                    "ts": float(r["ts"]), "mean": mean,
                    "min": _f_or_none(r["min"]), "max": _f_or_none(r["max"]),
                    "std": std, "n": int(r["n"] or 0),
                    "n_excluded": int(r["n_excluded"] or 0),
                    "worst_status": _SEVERITY_BY_RANK.get(int(r["sev"] or 0), "ok"),
                })
            out["points"] = pts
            return out
        except Exception:  # noqa: BLE001
            logger.debug("series query failed", exc_info=True)
            return out

    def spectra_query(self, channel: str | None = None,
                      since: float | None = None, until: float | None = None,
                      limit: int = 200) -> list[dict]:
        """谱快照的**元数据**列表 —— BLOB 不上线（一次 500 条就是 1 MB）。"""
        where: list[str] = []
        params: list = []
        if channel:
            where.append("channel = ?")
            params.append(str(channel))
        if since is not None:
            where.append("ts >= ?")
            params.append(float(since))
        if until is not None:
            where.append("ts <= ?")
            params.append(float(until))
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        try:
            with self._lock:
                rows = self._conn.execute(
                    # ctx_* 也在列表里：用户挑一条谱是按「哪一条是在 -1.2 V
                    # 隧穿下测的」，不是按行号。要它必须先看得见。
                    "SELECT id, ts, channel, span_s, n_segments, fs_hz, f_lo_hz,"
                    " f_hi_hz, n_points, unit, quietness,"
                    " ctx_bias_v, ctx_setpoint_a, ctx_zctrl_on, ctx_stable,"
                    " experiment_id, sample_id"
                    f" FROM env_spectra{clause} ORDER BY ts DESC LIMIT ?",
                    [*params, max(1, int(limit))],
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001
            logger.debug("spectra query failed", exc_info=True)
            return []

    def spectrum(self, spectrum_id: int) -> Optional[dict]:
        """一条完整的谱（含频率与功率数组）。"""
        try:
            with self._lock:
                r = self._conn.execute(
                    "SELECT * FROM env_spectra WHERE id = ?", [int(spectrum_id)]
                ).fetchone()
            if r is None:
                return None
            d = dict(r)
            d["freqs_hz"] = _unpack_f32(d.pop("freqs", None))
            d["psd"] = _unpack_f32(d.pop("psd", None))
            return d
        except Exception:  # noqa: BLE001
            logger.debug("spectrum read failed", exc_info=True)
            return None

    def storage_stats(self) -> dict:
        """库有多大、装了多少、在哪 —— 给 ``/status`` 与运维。

        ``path`` 是给远程验收的：一台只能通过 HTTP 够到的机器上，「这个数是从哪个
        文件来的」必须能问出来，否则排查只能靠猜安装布局。
        """
        out = {"path": str(self._path), "db_bytes": 0, "bucket_rows": 0,
               "spectra_rows": 0, "oldest_bucket_ts": None,
               "newest_bucket_ts": None}
        try:
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(self._path) + suffix)
                if p.exists():
                    out["db_bytes"] += int(p.stat().st_size)
            with self._lock:
                r = self._conn.execute(
                    "SELECT COUNT(*) AS n, MIN(bucket_ts) AS lo, MAX(bucket_ts) AS hi"
                    " FROM env_buckets"
                ).fetchone()
                out["bucket_rows"] = int(r["n"] or 0)
                out["oldest_bucket_ts"] = _f_or_none(r["lo"])
                out["newest_bucket_ts"] = _f_or_none(r["hi"])
                s = self._conn.execute("SELECT COUNT(*) AS n FROM env_spectra").fetchone()
                out["spectra_rows"] = int(s["n"] or 0)
        except Exception:  # noqa: BLE001
            logger.debug("storage_stats failed", exc_info=True)
        return out

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _f_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── 进程级单例 ───────────────────────────────────────────────────────

_STORE: EnvHistoryStore | None = None
_STORE_LOCK = threading.Lock()


def _default_path() -> Path:
    from mast._runtime_paths import project_root
    return Path(project_root()) / "experiments" / "env_history" / "env_history.sqlite"


def get_store() -> EnvHistoryStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = EnvHistoryStore(_default_path())
    return _STORE


def set_store_for_test(store: EnvHistoryStore | None) -> None:
    """换掉单例（测试把它指到 tmp 目录）。"""
    global _STORE
    with _STORE_LOCK:
        _STORE = store


__all__ = ["EnvHistoryStore", "get_store", "set_store_for_test"]
