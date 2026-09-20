"""Optical-stage acquisition skills — the atomic 'measure here' and the 1D/2D
raster scan (replaces ``2D stage.vi`` / ``Stage control.vi``).

- :class:`AcquireSignalPoint` (AUTO) — read + average Nanonis signals at the
  CURRENT optical position. The atomic building block: an agent or a composite
  can drive an experiment by hand as move → acquire → move → acquire, using the
  existing motion skills (OpticalStageMove / DelayLineMoveTo) + the trigger
  skill (PulseDigitalLine) + this.
- :class:`OpticalStageScan` (CONFIRM) — raster one or two optical stage axes
  over a grid, acquiring the same signals at every point → CSV + a 1D line /
  2D heat-map PNG. Covers the lab's ``2D stage.vi`` (PZTC XY map),
  ``Stage control.vi`` (Thorlabs 1D scan) and a two-stage grid in one skill.

Motion goes through the instruments registry; Nanonis through
``context.safe_call``. The per-point read/average/trigger machinery is shared
with PumpProbeScan via :mod:`mast.skills.builtins.optics_acquire`.
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

from mast._runtime_paths import project_root
from mast.core.types import (
    NanonisCallRecord,
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.instruments.base import InstrumentError, InstrumentUnavailable
from mast.instruments.registry import get_instrument_registry
from mast.skills.base import BaseSkill
from mast.skills.builtins.optics_acquire import PointAcquirer, parse_indices

logger = logging.getLogger(__name__)

__all__ = ["AcquireSignalPoint", "OpticalStageScan"]


def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n <= 1:
        return [start]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


# ── atomic: measure at the current point ──────────────────────────────────


class AcquireSignalPoint(BaseSkill):
    """Read + average Nanonis signals at the current optical position (atomic)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireSignalPoint",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "在**当前**光学位置上读取 Nanonis 信号并做软件平均：隧道电流和／或任意信号槽位（0-127）"
                "，每一路各采 `samples` 次。这是最小粒度的「就在这儿测一下」—— 把它和 OpticalStageMove / DelayLineMoveTo 以及 PulseDigitalLine 搭起来可以手工拼出一次扫描，"
                "或者直接用 OpticalStageScan 跑整幅光栅。纯读；不移动、也不触发。"
            ),
            parameters=[
                ParameterSpec(
                    name="read_current", type="bool",
                    description="采集隧道电流（Current_Get）",
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="signal_indices", type="str",
                    description=(
                        "要读取并平均的 Nanonis 信号槽位，逗号分隔（例如 '0,14'）；范围 0-127。"
                        "每加一路就多出 sig<idx>_mean/std。"
                    ),
                    required=False, default="",
                ),
                ParameterSpec(
                    name="samples", type="int",
                    description="每一路信号平均多少次读数",
                    required=False, default=10, min_value=1, max_value=10000,
                ),
                ParameterSpec(
                    name="sample_interval_s", type="float",
                    description="两次读数之间的间隔",
                    unit="s", required=False, default=0.01,
                    min_value=0.0, max_value=1.0,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["optics", "acquire", "signal", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        read_current = params.get("read_current", True)
        indices = parse_indices(params.get("signal_indices", ""))
        if not read_current and not indices:
            return SkillResult(
                skill_name="AcquireSignalPoint", success=False,
                error="nothing to acquire: enable read_current or add signal_indices",
            )
        acq = PointAcquirer(
            context, read_current=read_current, lockin_index=None,
            extra_indices=indices,
            samples=int(params.get("samples", 10)),
            interval_s=float(params.get("sample_interval_s", 0.01)),
        )
        nanonis_calls: list[NanonisCallRecord] = []
        try:
            row = acq.acquire(nanonis_calls)
        except (InstrumentError, ValueError) as exc:
            return SkillResult(
                skill_name="AcquireSignalPoint", success=False, error=str(exc),
                nanonis_calls=nanonis_calls,
            )
        row["n_samples"] = acq.samples
        bits = []
        if read_current and "current_a" in row:
            bits.append(f"I={row['current_a']:.4g} A")
        for xi in indices:
            bits.append(f"s{xi}={row.get(f'sig{xi}_mean'):.4g}")
        return SkillResult(
            skill_name="AcquireSignalPoint", success=True, data=row,
            nanonis_calls=nanonis_calls,
            summary="acquired " + (", ".join(bits) if bits else "point"),
        )


# ── complex: 1D / 2D raster scan ───────────────────────────────────────────


class OpticalStageScan(BaseSkill):
    """Raster 1 or 2 optical stage axes, acquiring Nanonis signals per point."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="OpticalStageScan",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "用光栅方式扫一台光学台，并把一路 Nanonis 信号在其上成图。快轴（device1/axis1）"
                "从 start1→stop1 走 points1 步；再给一条第二（慢）轴就得到一张 2D 图。"
                "每一点上它先稳定、可选地在触发线上打一个脉冲，然后对隧道电流和／或 signal_indices 取平均。"
                "CSV 加一张 1D 折线／2D 热图 PNG 写到 artifacts/optics_scan/。"
                "它涵盖 2D 台的 XY 成图、一维 Thorlabs 扫描，或者两台台子组成的网格。软行程限位由驱动强制执行。"
            ),
            parameters=[
                # fast (inner) axis — required
                ParameterSpec(name="device1", type="str", required=True,
                              description="快轴所在设备的 id（见 ListOpticalDevices）"),
                ParameterSpec(name="axis1", type="str", required=True,
                              description="该设备上快轴的轴名"),
                ParameterSpec(name="start1", type="float", required=True,
                              description="快轴起点（原生单位）"),
                ParameterSpec(name="stop1", type="float", required=True,
                              description="快轴终点（原生单位）"),
                ParameterSpec(name="points1", type="int", required=True,
                              min_value=2, max_value=10000,
                              description="快轴点数"),
                # slow (outer) axis — optional → 2D
                ParameterSpec(name="device2", type="str", required=False, default="",
                              description="慢轴所在设备的 id（'' = 做一维扫描）"),
                ParameterSpec(name="axis2", type="str", required=False, default="",
                              description="慢轴的轴名"),
                ParameterSpec(name="start2", type="float", required=False, default=0.0,
                              description="慢轴起点"),
                ParameterSpec(name="stop2", type="float", required=False, default=0.0,
                              description="慢轴终点"),
                ParameterSpec(name="points2", type="int", required=False, default=1,
                              min_value=1, max_value=10000,
                              description="慢轴点数（1 = 1D）"),
                # acquisition
                ParameterSpec(name="read_current", type="bool", required=False,
                              default=True, description="采集隧道电流"),
                ParameterSpec(name="signal_indices", type="str", required=False,
                              default="",
                              description="要读取并平均的信号槽位（例如 '0,14'）"),
                ParameterSpec(name="samples_per_point", type="int", required=False,
                              default=10, min_value=1, max_value=1000,
                              description="每点平均多少次读数"),
                ParameterSpec(name="sample_interval_s", type="float", required=False,
                              default=0.01, min_value=0.0, max_value=1.0, unit="s",
                              description="两次读数之间的间隔"),
                ParameterSpec(name="settle_extra_s", type="float", required=False,
                              default=0.05, min_value=0.0, max_value=10.0, unit="s",
                              description="到位之后额外驻留的时间"),
                ParameterSpec(name="move_timeout_s", type="float", required=False,
                              default=60.0, min_value=0.1, max_value=600.0, unit="s",
                              description="每次移动的时限"),
                # trigger
                ParameterSpec(name="trigger_enable", type="bool", required=False,
                              default=False,
                              description="每一点上在一条 Nanonis 数字线上打脉冲"),
                ParameterSpec(name="trigger_port", type="int", required=False,
                              default=0, min_value=0, max_value=8,
                              description="触发用的数字端口（0=A…）"),
                ParameterSpec(name="trigger_line", type="int", required=False,
                              default=1, min_value=1, max_value=8,
                              description="触发用的数字线 1-8"),
                ParameterSpec(name="trigger_width_s", type="float", required=False,
                              default=1e-4, min_value=1e-6, max_value=1.0, unit="s",
                              description="触发脉冲宽度"),
                # motion pattern
                ParameterSpec(name="serpentine", type="bool", required=False,
                              default=True,
                              description="让快轴蛇形走（每行反向），省掉长距离回扫"),
                ParameterSpec(name="return_to_start", type="bool", required=False,
                              default=True,
                              description="结束后把两条轴都移回起点"),
                ParameterSpec(name="dry_run", type="bool", required=False,
                              default=False,
                              description="**试跑**：把整个网格走一遍并稳定，但什么都不采集、什么都不存盘"),
                ParameterSpec(name="tag", type="str", required=False, default="",
                              description="附加在输出文件名上的标签"),
            ],
            estimated_duration_s=120.0,
            composition_level=0,
            tags=["optics", "stage", "scan", "map", "acquisition", "write"],
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _artifact_paths(tag: str) -> tuple[Path, Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in tag if c.isalnum() or c in "-_")[:40]
        base = f"optics_scan_{stamp}" + (f"_{safe}" if safe else "")
        out = project_root() / "artifacts" / "optics_scan"
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{base}.csv", out / f"{base}.png"

    @staticmethod
    def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _plot_1d(path: Path, rows: list[dict], series: list[tuple],
                 axis_label: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logger.warning("optics scan plot skipped: %s", exc)
            return False
        xs = [r["pos1"] for r in rows]
        fig, axes = plt.subplots(len(series), 1, sharex=True,
                                 figsize=(7, 3 * len(series)), squeeze=False)
        for i, (mcol, scol, label) in enumerate(series):
            ax = axes[i][0]
            ax.errorbar(xs, [r.get(mcol) for r in rows],
                        yerr=[r.get(scol) for r in rows] if scol else None,
                        fmt=".-", lw=0.8, ms=3, capsize=2)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[-1][0].set_xlabel(axis_label)
        fig.suptitle("Optical stage scan")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True

    @staticmethod
    def _plot_2d(path: Path, rows: list[dict], value_col: str, label: str,
                 p1: int, p2: int, extent: tuple, xl: str, yl: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            logger.warning("optics scan heatmap skipped: %s", exc)
            return False
        grid = [[float("nan")] * p1 for _ in range(p2)]
        for r in rows:
            rr, cc = int(r["row"]), int(r["col"])
            if 0 <= rr < p2 and 0 <= cc < p1:
                v = r.get(value_col)
                grid[rr][cc] = float(v) if isinstance(v, (int, float)) else float("nan")
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent,
                       interpolation="nearest")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Optical stage map · {label}")
        fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True

    # -- main ------------------------------------------------------------------

    def execute(self, context, params: dict) -> SkillResult:
        dry_run = bool(params.get("dry_run", False))
        read_current = params.get("read_current", True)
        indices = parse_indices(params.get("signal_indices", ""))
        if not dry_run and not read_current and not indices:
            return SkillResult(
                skill_name="OpticalStageScan", success=False,
                error=("nothing to acquire: enable read_current, add "
                       "signal_indices, or set dry_run=true to test motion"),
            )

        reg = get_instrument_registry()
        try:
            ax1 = reg.axis(params["device1"], params["axis1"])
        except (InstrumentError, InstrumentUnavailable) as exc:
            return SkillResult(skill_name="OpticalStageScan", success=False,
                               error=f"fast axis unavailable: {exc}")
        dev2 = str(params.get("device2", "") or "")
        axname2 = str(params.get("axis2", "") or "")
        two_d = bool(dev2 and axname2)
        ax2 = None
        if two_d:
            try:
                ax2 = reg.axis(dev2, axname2)
            except (InstrumentError, InstrumentUnavailable) as exc:
                return SkillResult(skill_name="OpticalStageScan", success=False,
                                   error=f"slow axis unavailable: {exc}")

        p1 = int(params["points1"])
        p2 = int(params.get("points2", 1)) if two_d else 1
        grid1 = _linspace(float(params["start1"]), float(params["stop1"]), p1)
        grid2 = _linspace(float(params.get("start2", 0.0)),
                          float(params.get("stop2", 0.0)), p2) if two_d else [None]
        move_timeout = float(params.get("move_timeout_s", 60.0))
        settle_extra = float(params.get("settle_extra_s", 0.05))
        serpentine = bool(params.get("serpentine", True))

        # per-point acquirer (skipped in dry run)
        trigger = None
        if bool(params.get("trigger_enable", False)) and not dry_run:
            trigger = (int(params.get("trigger_port", 0)),
                       int(params.get("trigger_line", 1)),
                       float(params.get("trigger_width_s", 1e-4)))
        acq = PointAcquirer(
            context, read_current=read_current, lockin_index=None,
            extra_indices=indices,
            samples=int(params.get("samples_per_point", 10)),
            interval_s=float(params.get("sample_interval_s", 0.01)),
            trigger=trigger,
        )
        nanonis_calls: list[NanonisCallRecord] = []
        if trigger:
            err = acq.configure_trigger(nanonis_calls)
            if err:
                return SkillResult(
                    skill_name="OpticalStageScan", success=False,
                    error=f"trigger line config failed: {err}",
                    nanonis_calls=nanonis_calls,
                )

        rows: list[dict] = []
        aborted = ""
        idx = 0
        t0 = time.monotonic()
        try:
            for oi, y in enumerate(grid2):
                if two_d and y is not None:
                    ax2.move_abs(y, wait=True, timeout=move_timeout)
                    if settle_extra > 0:
                        time.sleep(settle_extra)
                # snake the fast axis: reverse on odd rows to skip fly-back
                reverse = serpentine and (oi % 2 == 1)
                order = range(p1 - 1, -1, -1) if reverse else range(p1)
                for ii in order:
                    check_abort = getattr(context, "check_abort", None)
                    if callable(check_abort) and check_abort():
                        aborted = f"aborted by operator after {len(rows)} points"
                        raise _Abort()
                    st = ax1.move_abs(grid1[ii], wait=True, timeout=move_timeout)
                    if settle_extra > 0:
                        time.sleep(settle_extra)
                    row: dict = {
                        "index": idx, "row": oi, "col": ii,
                        "pos1": st.position,
                    }
                    if two_d:
                        row["pos2"] = y
                    if not dry_run:
                        row.update(acq.acquire(nanonis_calls))
                    rows.append(row)
                    idx += 1
        except _Abort:
            pass
        except Exception as exc:  # noqa: BLE001 — keep partial data
            aborted = f"{type(exc).__name__}: {exc}"
            logger.warning("optics scan interrupted at point %d: %s", len(rows), aborted)
        finally:
            if params.get("return_to_start", True):
                try:
                    ax1.move_abs(grid1[0], wait=False)
                    if two_d and grid2 and grid2[0] is not None:
                        ax2.move_abs(grid2[0], wait=False)
                except Exception:  # noqa: BLE001
                    pass

        if not rows:
            return SkillResult(skill_name="OpticalStageScan", success=False,
                               error=aborted or "scan produced no data",
                               nanonis_calls=nanonis_calls)

        elapsed = time.monotonic() - t0
        total = p1 * p2

        if dry_run:
            data = {
                "dry_run": True, "n_points": len(rows), "requested_points": total,
                "shape": [p2, p1] if two_d else [p1],
                "elapsed_s": round(elapsed, 3),
            }
            if aborted:
                data["interrupted_after_points"] = len(rows)
            summary = (f"[DRY RUN] optical scan stepped {len(rows)}/{total} points "
                       f"({'2D ' + str(p2) + 'x' + str(p1) if two_d else '1D ' + str(p1)}) "
                       f"in {elapsed:.1f}s — motion OK, no data saved")
            return SkillResult(skill_name="OpticalStageScan",
                               success=not aborted, data=data, error=aborted,
                               nanonis_calls=nanonis_calls, summary=summary)

        # artifacts
        fieldnames = list(rows[0].keys())
        csv_path, png_path = self._artifact_paths(str(params.get("tag", "")))
        self._write_csv(csv_path, rows, fieldnames)
        series = acq.plot_series
        plotted = False
        if series:
            if two_d:
                mcol, _s, label = series[0]
                extent = (float(params["start1"]), float(params["stop1"]),
                          float(params.get("start2", 0.0)), float(params.get("stop2", 0.0)))
                plotted = self._plot_2d(png_path, rows, mcol, label, p1, p2, extent,
                                        f"{params['axis1']} ({ax1.config.unit})",
                                        f"{axname2} ({ax2.config.unit})")
            else:
                plotted = self._plot_1d(png_path, rows, series,
                                        f"{params['axis1']} ({ax1.config.unit})")

        data = {
            "path": str(csv_path),
            "plot_path": str(png_path) if plotted else "",
            "n_points": len(rows), "requested_points": total,
            "shape": [p2, p1] if two_d else [p1],
            "dimensions": 2 if two_d else 1,
            "samples_per_point": acq.samples,
            "elapsed_s": round(elapsed, 3),
        }
        if indices:
            data["signal_indices"] = indices
        if trigger:
            data["triggered_line"] = {"port": trigger[0], "line": trigger[1]}
        if aborted:
            data["interrupted_after_points"] = len(rows)

        summary = (f"optical scan: {len(rows)}/{total} points "
                   f"({'2D ' + str(p2) + '×' + str(p1) if two_d else '1D ' + str(p1)}) "
                   f"in {elapsed:.1f}s → {csv_path.name}")
        if aborted:
            summary += f" [INTERRUPTED: {aborted}]"
        return SkillResult(skill_name="OpticalStageScan", success=not aborted,
                           data=data, error=aborted, nanonis_calls=nanonis_calls,
                           summary=summary)


class _Abort(Exception):
    """Internal: operator abort mid-scan (keeps partial data, no error text)."""
