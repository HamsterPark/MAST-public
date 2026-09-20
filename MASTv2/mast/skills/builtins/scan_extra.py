"""Additional scan skills: speed readback, buffer readback, save, latest-file lookup.

vendored from v1 mast/skills/builtins/scan_extra.py.
4 skills: GetScanSpeed, GetScanBuffer, SaveScan, GetLatestScanFile.
"""

from __future__ import annotations

from pathlib import Path

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import decode_reply, parse_buffer_get
from mast.skills.base import BaseSkill


class GetScanSpeed(BaseSkill):
    """Read back scan speed parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScanSpeed",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前的扫描速度参数。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "speed", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Scan_SpeedGet")
        if record.error:
            return SkillResult(
                skill_name="GetScanSpeed",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 6:
                data = {
                    "fwd_speed_m_s": float(vals[0]),
                    "bwd_speed_m_s": float(vals[1]),
                    "fwd_time_s": float(vals[2]),
                    "bwd_time_s": float(vals[3]),
                    "keep_constant": int(vals[4]),
                    "speed_ratio": float(vals[5]),
                }
        return SkillResult(
            skill_name="GetScanSpeed",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class GetScanBuffer(BaseSkill):
    """Read back scan buffer configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetScanBuffer",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="读取当前的 scan buffer（通道、像素数、行数）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["scan", "buffer", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("Scan_BufferGet")
        if record.error:
            return SkillResult(
                skill_name="GetScanBuffer",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        data: dict = {"raw": decode_reply(parsed)}
        # 通过 io 层解析通道列表，将单元素元组统一解成整数；不要把元组当通道号传出。
        buf = parse_buffer_get(parsed)
        if buf is not None:
            data = {
                "num_channels": buf["num_channels"],
                "channel_indexes": buf["channel_indexes"],
                "pixels": buf["pixels"],
                "lines": buf["lines"],
            }
            # 有字段没解出来时把原始回包一起带上 —— 否则现场只看到一个 None,
            # 而回包长什么样正是唯一能定位问题的东西。
            if any(v is None for v in data.values()):
                data["raw"] = str(parsed)
        return SkillResult(
            skill_name="GetScanBuffer",
            success=True,
            data=data,
            nanonis_calls=[record],
        )


class SaveScan(BaseSkill):
    """Save the current scan data to file."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SaveScan",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="把当前的扫描数据缓冲存成文件。",
            parameters=[
                ParameterSpec(
                    name="timeout_ms",
                    type="int",
                    description="保存的超时，单位 ms（-1 = 永远等待）",
                    unit="ms",
                    required=False,
                    default=-1,
                    min_value=-1,
                ),
            ],
            estimated_duration_s=5.0,
            composition_level=1,
            tags=["scan", "save", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        timeout_ms = params.get("timeout_ms", -1)
        # Scan_Save(Wait_until_saved, Timeout_ms)
        record = context.safe_call("Scan_Save", 1, timeout_ms)
        if record.error:
            return SkillResult(
                skill_name="SaveScan",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        parsed = record.return_value
        timed_out = False
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            vals = parsed[2]
            if isinstance(vals, (list, tuple)) and len(vals) >= 1:
                timed_out = bool(vals[0])
        # Surface the path of the freshly-saved .sxm so the LLM has something
        # concrete to tell the user (e.g. "saved to D:\…").
        latest = _find_latest_sxm(_candidate_save_dirs(context), max_age_s=120)
        if latest is not None:
            # Publish where scans actually land so context-less searches
            # (data_processing's file tools) can find them (2026-07-10 #92/#93).
            try:
                from mast.core.scan_registry import record_scan_path
                record_scan_path(latest)
            except Exception:  # noqa: BLE001 — registry is best-effort
                pass
        return SkillResult(
            skill_name="SaveScan",
            success=True,
            data={
                "timed_out": timed_out,
                "saved_path": str(latest) if latest else None,
            },
            nanonis_calls=[record],
        )


def _session_dir_from_caller(caller, *, role: str = "main") -> str | None:
    """Resolve Nanonis's current save dir via ``Util_SessionPathGet`` on any object
    exposing ``safe_call`` (an ExecutionContext or a ConnectionPool), normalised to
    a real directory. None on any failure / no hardware.

    NOTE: ``CoreRuntime._resolve_session_dir`` keeps a PARALLEL inline copy of this
    parse — core must not import skills (reverse-layer dep). Keep the two in sync."""
    sc = getattr(caller, "safe_call", None)
    if not callable(sc):
        return None
    try:
        try:
            rec = sc("Util_SessionPathGet", role=role)
        except TypeError:  # a safe_call without a role kwarg
            rec = sc("Util_SessionPathGet")
    except Exception:  # noqa: BLE001 — degrade, never raise into dir discovery
        return None
    if rec is None or getattr(rec, "error", None):
        return None
    parsed = getattr(rec, "return_value", None)
    sp = ""
    if isinstance(parsed, str):
        sp = parsed
    elif isinstance(parsed, (list, tuple)) and len(parsed) > 2:
        vals = parsed[2]  # ResponseTypes ["i","*-c"] → [size_int, path_str]
        if isinstance(vals, (list, tuple)):
            for item in reversed(vals):
                if isinstance(item, str):
                    sp = item
                    break
        elif isinstance(vals, str):
            sp = vals
    sp = (sp or "").strip()
    if not sp:
        return None
    p = Path(sp)
    try:
        if p.is_dir():
            return str(p)
        if p.parent and str(p.parent) not in (".", "") and p.parent.is_dir():
            return str(p.parent)  # SessionPathGet may return a file-prefix
    except OSError:
        pass
    return str(p)


def _candidate_save_dirs(context) -> list[Path]:
    """Plausible places Nanonis writes .sxm files. Order by likelihood."""
    cands: list[Path] = []
    # 1. The Nanonis session dir — most authoritative. Prefer a GUI-captured attr;
    #    else resolve it LIVE via the context's pool (Util_SessionPathGet). Was
    #    attr-only and NEVER populated → scans saved outside working-sessions were
    #    undiscoverable (GetLatestScanFile path:None, 2026-06-29).
    try:
        sp = getattr(context, "session_path", None) or getattr(context, "_session_path", None)
        if not sp and context is not None:
            sp = _session_dir_from_caller(context)
        if sp:
            cands.append(Path(sp))
            try:
                from mast.core.scan_registry import record_session_dir
                record_session_dir(sp)
            except Exception:
                pass
    except Exception:
        pass
    # 2. Directories scans are KNOWN to have landed in this session (recorded
    #    by SaveScan / session-path resolves). This is what lets a
    #    context-less caller — data_processing's get_latest_scan_file runs
    #    with context=None, no pool — find files in the real Nanonis session
    #    dir instead of only working-sessions.
    try:
        from mast.core.scan_registry import known_scan_dirs
        cands.extend(known_scan_dirs())
    except Exception:
        pass
    # 3. <data_root>/working-sessions/
    try:
        from mast._runtime_paths import project_root
        cands.append(project_root() / "working-sessions")
    except Exception:
        pass
    # 4. 当前活跃样品的 raw/nanonis/ —— 仅当用户显式把 Nanonis 的保存目录
    #    指向了实验文件夹（原位模式）。
    #
    #    刻意只加这一个目录，**绝不**加整个 experiment_root：find_latest_saved
    #    是 rglob，把实验根塞进来会让它每次翻遍所有历史副本（几千个文件），而且
    #    极可能把某个历史副本当成"刚存的那一个"。
    #
    #    （这里原本硬编码着一条开发机上的绝对路径，它跟着发布包进了每一台用户
    #    机器，而在那些机器上永远解析不到。2026-07-28 删除。）
    try:
        from mast.core.experiment_paths import nanonis_inplace_dir
        from mast.logging.experiment_log import get_active_log
        _log = get_active_log()
        _dir = getattr(_log, "current_sample_dir", None) if _log else None
        if callable(_dir):
            _sd = _dir()
            if _sd:
                cands.append(nanonis_inplace_dir(_sd))
    except Exception:
        pass
    # Dedupe while preserving order
    seen = set()
    out: list[Path] = []
    for c in cands:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out


def find_latest_saved(roots: list[Path], pattern: str = "*.sxm", *,
                      max_age_s: float = 86400) -> Path | None:
    """Scan *roots* recursively, return the file matching *pattern* with the
    newest mtime within max_age_s seconds; None if nothing recent.

    Shared with the spectroscopy skills, which need the same "what did Nanonis
    just autosave" answer for ``*.dat`` — Nanonis picks the directory and the
    index itself, so the only way to name the file we just created is to look
    for the newest one right after the call.
    """
    import time as _time
    now = _time.time()
    best: tuple[float, Path] | None = None
    for root in roots:
        try:
            for p in root.rglob(pattern):
                try:
                    age = now - p.stat().st_mtime
                except OSError:
                    continue
                if age > max_age_s:
                    continue
                if best is None or age < best[0]:
                    best = (age, p)
        except OSError:
            continue
    return best[1] if best else None


def _find_latest_sxm(roots: list[Path], *, max_age_s: float = 86400) -> Path | None:
    """The .sxm case — kept as the name the scan skills already call."""
    return find_latest_saved(roots, "*.sxm", max_age_s=max_age_s)


class GetLatestScanFile(BaseSkill):
    """Find the most recently written .sxm file across the candidate save dirs.

    Use after SaveScan when you need the actual file path on disk (e.g. to
    show the user, attach to a chat reply, or post-process). Falls back to
    scanning known candidate folders when Util_SessionPathGet doesn't give
    a usable path string.
    """

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetLatestScanFile",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "定位最近写入的那个 .sxm 扫描文件。先查 Nanonis 报出来的 "
                "session 路径，再查数据目录下的 working-sessions/，"
                "最后查历史遗留的开发目录。返回 "
                "{path: str, age_s: float}；若在 max_age_s 秒内一个都"
                "没找到，则返回 {path: null}。"
            ),
            parameters=[
                ParameterSpec(
                    name="max_age_s",
                    type="int",
                    description="只考虑最近 N 秒内被修改过的 .sxm 文件",
                    unit="s",
                    required=False,
                    default=300,
                    min_value=1,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=2,
            tags=["scan", "file", "read", "sxm"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        max_age = int(params.get("max_age_s", 300))
        cands = _candidate_save_dirs(context)
        latest = _find_latest_sxm(cands, max_age_s=max_age)
        if latest is None:
            return SkillResult(
                skill_name="GetLatestScanFile",
                success=True,
                data={
                    "path": None,
                    "searched_dirs": [str(c) for c in cands],
                    "max_age_s": max_age,
                },
            )
        import time as _time
        age = _time.time() - latest.stat().st_mtime
        return SkillResult(
            skill_name="GetLatestScanFile",
            success=True,
            data={
                "path": str(latest),
                "age_s": round(age, 1),
                "size_bytes": latest.stat().st_size,
                "searched_dirs": [str(c) for c in cands],
            },
        )
