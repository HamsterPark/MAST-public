"""畴参照系:哪几个畴、原型指纹长什么样、比对用什么容差 —— **全部来自运行时标定**。

:mod:`mast.vision.domain_phase` 只会算指纹和距离;「这是 A 相」这句话需要一个
**人确认过**的参照系。本模块负责把它从磁盘读上来,**只读**。

## 为什么永远没有内建参照系

``scan_prep_thresholds`` 有一个内建 profile,但它的注释写得很清楚:没有那个样品的
数据就造一个 profile 出来**等于伪造标定**。畴参照系比阈值更进一步 —— 它连
「有几个畴、叫什么名字」都是样品事实,而且**换样品即作废**。所以:

* 没有文件 ⇒ :func:`load_reference` 返回 ``None`` = 「没有参照系」;
  ``classify`` 一律 ``undetermined(no_reference)``,**这是正确行为,不是失败**。
* 文件读不动/schema 不对/原型是空的 ⇒ 同样是「没有参照系」,记一条日志,**不抛**。
  (照 ``scan_prep_thresholds._load_file_profiles`` 的既有形状:只读、按 mtime
  缓存、任何错误退化成「没有」。)
* ``match.*`` 留 ``0.0`` 表示**未标定** ⇒ :attr:`DomainReference.calibrated` 为假
  ⇒ 同样只出 ``no_reference``。**0 不是「零容差」**:``match_tol=0`` 当零容差用会
  让任何帧都判不出 label 而且不说原因;``ambiguity_margin=0`` 会把「两个原型长得
  一样」这件事静默放行;``mixed_coverage_min=0`` 会让每一帧都是 ``mixed``。

## 为什么不放进 instrument_profile,也不放进 scan_prep_thresholds

``instrument_profile`` 装的是**仪器硬事实**(退针方向、dI/dV 标定)——把样品事实
放进去,换样品时会静默沿用。``scan_prep_thresholds`` 装的是预处理阈值,语义不同,
而且它**有**内建档 —— 混进去会诱导有人造一个「出厂 A/B 相」。

## 落点与版本

    <project_root>/config/domain_references/<sample>-<YYYY-MM-DD>-vNNN.json

``vNNN`` **永不覆盖**(照实验文件夹 INCREMENTAL-ONLY 的纪律):改了容差就是新版本,
判定结果里记的 ``reference_version`` 才有意义。

``sample`` 留空且目录里有**多个样品**的参照系 ⇒ 返回 ``None`` 并记日志,**不猜**。
挑错样品的参照系会让判定看起来一切正常 —— 这正是「读不到不是一个值」那一族。

## schema(``schema: 1``)

    {
      "schema": 1, "version": "v001", "sample": "...", "created": "YYYY-MM-DD",
      "provenance": "哪天/哪个实验/哪几张帧/谁确认的 —— 必填一句人话",
      "confirmed_by": "<operator>",
      "symmetry_deg": 60.0,                      // 样品事实:六角 60、矩形 90
      "labels": ["A", "B"],                      // 用户起的名字,代码只保证闭集
      "prototypes": {"A": {"peaks": [[角度deg, 周期nm, 相对功率], ...],
                           "source_frames": ["<abs .sxm path>", ...]}, ...},
      "match": {"w_angle": 1.0, "w_period": 1.0, "match_tol": 0.0,
                "ambiguity_margin": 0.0, "mixed_coverage_min": 0.0},
      "measured_separation": {"intra_max": 0.0, "inter_min": 0.0, "ratio": 0.0,
                              "n_frames_per_cluster": 0}
    }

角度是**样品系**的(``domain_phase`` 的 ``k_angle_sample_deg``),不是帧系 ——
存帧系角度会让参照系只对当初那个扫描框角度有效。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: 参照系目录名(在 ``<project_root>/config/`` 下)。
DIRNAME = "domain_references"

#: 本模块认得的 schema 版本。别的版本一律当「没有参照系」——**不做静默兼容**。
SCHEMA = 1


@dataclass(frozen=True)
class DomainPrototype:
    """一个畴的原型指纹。``peaks`` 是 ``(样品系角度deg, 周期nm, 相对功率)``。"""

    label: str
    peaks: tuple[tuple[float, float, float], ...] = ()
    source_frames: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainReference:
    """一次标定的产物。``calibrated`` 为假时,判定层只许出 ``no_reference``。"""

    version: str
    sample: str
    symmetry_deg: float
    labels: tuple[str, ...]
    prototypes: dict[str, DomainPrototype]
    created: str = ""
    provenance: str = ""
    confirmed_by: str = ""
    w_angle: float = 1.0
    w_period: float = 1.0
    match_tol: float = 0.0
    ambiguity_margin: float = 0.0
    mixed_coverage_min: float = 0.0
    measured_separation: dict = field(default_factory=dict)
    source_path: str = ""

    @property
    def calibrated(self) -> bool:
        """比对参数齐不齐。**任何一个是 0 都算未标定**(见模块注释)。"""
        return bool(
            self.symmetry_deg > 0
            and self.labels
            and all(self.prototypes.get(lab) and self.prototypes[lab].peaks
                    for lab in self.labels)
            and self.match_tol > 0.0
            and self.ambiguity_margin > 0.0
            and self.mixed_coverage_min > 0.0
            and (self.w_angle + self.w_period) > 0.0
            and self.w_angle >= 0.0 and self.w_period >= 0.0
        )

    def peaks_of(self, label: str) -> tuple[tuple[float, float, float], ...]:
        proto = self.prototypes.get(str(label))
        return proto.peaks if proto else ()


def _peaks_from(raw) -> tuple[tuple[float, float, float], ...]:
    """``[[角度, 周期, 功率], ...]`` → 三元组;任何一条不合法就整份作废。"""
    out: list[tuple[float, float, float]] = []
    if not isinstance(raw, (list, tuple)):
        raise ValueError("peaks 不是数组")
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError(f"峰的形状不对: {item!r}")
        ang, per, rel = float(item[0]), float(item[1]), float(item[2])
        if not (per > 0):
            raise ValueError(f"周期必须为正: {per!r}")
        if not (rel >= 0):
            raise ValueError(f"相对功率不能为负: {rel!r}")
        out.append((ang % 180.0, per, rel))
    if not out:
        raise ValueError("原型里一个峰都没有")
    return tuple(out)


def reference_from_mapping(raw, *, source: str = "") -> DomainReference | None:
    """纯函数:一个 dict → :class:`DomainReference`,不合法返回 ``None``。

    测试与将来的写入方(人确认之后固化那一步)都用它做校验 —— 校验只写一遍。
    """
    try:
        if not isinstance(raw, dict):
            raise ValueError("顶层不是对象")
        schema = int(raw.get("schema", 0))
        if schema != SCHEMA:
            raise ValueError(f"schema {schema} 不认识(本模块只认 {SCHEMA})")
        labels = tuple(str(x) for x in (raw.get("labels") or ()))
        if not labels:
            raise ValueError("labels 是空的")
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels 有重名: {labels}")
        sym = float(raw.get("symmetry_deg") or 0.0)
        if not (0.0 < sym <= 360.0):
            raise ValueError(f"symmetry_deg 不合法: {sym!r}")
        protos_raw = raw.get("prototypes") or {}
        if not isinstance(protos_raw, dict):
            raise ValueError("prototypes 不是对象")
        protos: dict[str, DomainPrototype] = {}
        for lab in labels:
            body = protos_raw.get(lab)
            if not isinstance(body, dict):
                raise ValueError(f"label {lab!r} 没有原型")
            protos[lab] = DomainPrototype(
                label=lab,
                peaks=_peaks_from(body.get("peaks")),
                source_frames=tuple(str(p) for p in (body.get("source_frames") or ())),
            )
        match = raw.get("match") or {}
        if not isinstance(match, dict):
            raise ValueError("match 不是对象")

        def _num(key: str, default: float) -> float:
            v = match.get(key, default)
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"match.{key} 不是数: {v!r}") from None
            if f < 0:
                raise ValueError(f"match.{key} 不能为负: {f!r}")
            return f

        provenance = str(raw.get("provenance") or "").strip()
        if not provenance:
            # 一个数字从哪来,事后必须查得到 —— 照 ScanPrepThresholds 的强制。
            raise ValueError("provenance 是必填的(哪天/哪个实验/哪几张帧/谁确认的)")
        sep = raw.get("measured_separation") or {}
        return DomainReference(
            version=str(raw.get("version") or "").strip() or "v000",
            sample=str(raw.get("sample") or "").strip(),
            symmetry_deg=sym,
            labels=labels,
            prototypes=protos,
            created=str(raw.get("created") or "").strip(),
            provenance=provenance,
            confirmed_by=str(raw.get("confirmed_by") or "").strip(),
            w_angle=_num("w_angle", 1.0),
            w_period=_num("w_period", 1.0),
            match_tol=_num("match_tol", 0.0),
            ambiguity_margin=_num("ambiguity_margin", 0.0),
            mixed_coverage_min=_num("mixed_coverage_min", 0.0),
            measured_separation=dict(sep) if isinstance(sep, dict) else {},
            source_path=str(source),
        )
    except Exception as exc:  # noqa: BLE001 — 参照系坏了不该让判据崩掉
        logger.warning("畴参照系读不动,当作没有参照系:%s: %s (%s)",
                       type(exc).__name__, exc, source or "<mapping>")
        return None


def references_dir() -> Path:
    """参照系目录(跟随 ``MAST2_PROJECT_ROOT`` / 冻结版路径)。"""
    from mast._runtime_paths import project_root

    return project_root() / "config" / DIRNAME


_lock = threading.Lock()
_cache: tuple[tuple, tuple[DomainReference, ...]] | None = None


def _signature(d: Path) -> tuple:
    """目录里 ``*.json`` 的 (名, mtime, 大小) —— 缓存键。读不到就是空签名。"""
    try:
        return tuple(sorted(
            (p.name, p.stat().st_mtime, p.stat().st_size)
            for p in d.glob("*.json")))
    except OSError:
        return ()


def list_references() -> tuple[DomainReference, ...]:
    """目录里所有**合法**的参照系,按版本降序(最新在前)。按 mtime 缓存。

    坏文件被跳过并记一条日志 —— 一个坏文件不该让另一个好文件也用不上。
    """
    global _cache
    d = references_dir()
    sig = _signature(d)
    with _lock:
        cached = _cache
        if cached is not None and cached[0] == sig:
            return cached[1]
    out: list[DomainReference] = []
    for name, _mtime, _size in sig:
        p = d / name
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("畴参照系文件读不动,跳过:%s: %s (%s)",
                           type(exc).__name__, exc, p)
            continue
        ref = reference_from_mapping(raw, source=str(p))
        if ref is not None:
            out.append(ref)
    out.sort(key=_version_key, reverse=True)
    result = tuple(out)
    with _lock:
        _cache = (sig, result)
    return result


def _version_key(ref: DomainReference) -> tuple:
    """``v012`` → (12, 'v012');数字解析不出来就退到字典序(仍然确定)。"""
    v = ref.version.strip()
    digits = "".join(ch for ch in v if ch.isdigit())
    return (int(digits) if digits else -1, v)


def load_reference(version: str | None = None, *,
                   sample: str | None = None) -> DomainReference | None:
    """取一个参照系。**任何「取不到」都返回 ``None`` = 没有参照系**。

    * ``version`` 给了就精确匹配(找不到 ⇒ ``None``,不退到最新 —— 判定结果里
      记着版本号,悄悄换一版会让事后对账对不上)。
    * ``version`` 留空 ⇒ 取最新;但目录里**跨多个样品**时返回 ``None`` 并记日志,
      **不猜**:挑错样品的参照系会让判定看起来一切正常。
    """
    refs = list_references()
    if sample:
        refs = tuple(r for r in refs if r.sample == str(sample))
    if not refs:
        return None
    if version:
        for r in refs:
            if r.version == str(version):
                return r
        logger.info("没有版本为 %r 的畴参照系(有:%s)", version,
                    [r.version for r in refs])
        return None
    samples = {r.sample for r in refs}
    if len(samples) > 1:
        logger.warning(
            "config/%s/ 里有多个样品的畴参照系 %s —— 不猜用哪个,"
            "请显式传 sample=。", DIRNAME, sorted(samples))
        return None
    return refs[0]


def clear_cache() -> None:
    """丢掉 mtime 缓存(测试用;同一秒内改文件时 mtime 可能不变)。"""
    global _cache
    with _lock:
        _cache = None


__all__ = [
    "DIRNAME",
    "SCHEMA",
    "DomainPrototype",
    "DomainReference",
    "clear_cache",
    "list_references",
    "load_reference",
    "reference_from_mapping",
    "references_dir",
]
