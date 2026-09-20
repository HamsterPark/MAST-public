"""当前样品是什么衬底 —— 让 skill 层问得到，并把知识库的常数取出来。

针尖锻造的两个配方都要知道衬底:做 STS 的金属性针尖要拿**肖克利表面态**的 onset
位置当判据(Au(111) −0.49 V / Cu(111) −0.44 V / Ag(111) −0.065 V),做原子分辨针尖
要拿**原子行间距**当晶格常数的核对值。两个数都在知识库里躺着,但 skill 层此前既
读不到「现在装的是什么样品」,也没有从样品名到那些常数的通路。

形状照 :mod:`mast.core.map_scope`(同一层、同一条纪律):走
``get_active_log() → current_sample_id → storage.get_sample()``,任何一步失败都
fail-soft 返回「不知道」而不是抛。

## 「不知道」不等于「Au(111)」

``available=False`` 只表示读不到样品记录。**绝不能**在这种情况下默认按 Au(111)
处理:在 Ag(111) 上按 Au 的 −490 mV 找 onset 会一直找不到,配方于是反复扎针尖去
修一根其实没问题的针 —— 而真正的问题只是没人告诉系统台面上放的是什么。

这与 ``map_scope.load_markers`` 的 ``available`` 是同一条:读不到记录 ≠ 表面干净。

## 模糊匹配的陷阱

``knowledge.lookups.match_material`` 是给自由文本用的五趟匹配,对省略写法会给出
**别的材料**:实测 ``match_material("au111")`` 返回的是
``("2d_material", "MoS2_on_Au111")`` —— 一个长在 Au(111) 上的二硫化钼样品。拿它的
常数去当衬底判据是错的。所以这里:

* 先按 ``sample_subtype`` 精确查 ``clean_metal`` 表;
* 再退回模糊匹配,但**只接受 ``type_id == "clean_metal"`` 的结果**;
* 两条都不中就是「不知道」。

## 无表面态的衬底要说清楚

Pt(111) 在知识库里 ``surface_state_onset_ev`` 就是 ``None`` —— 那个面没有肖克利
表面态。这不是数据缺失,是物理事实,所以配方拿到 ``surface_state_onset_v is None``
时应当**拒绝执行**并说明,而不是换个数接着找。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: fcc(111) 面上，扫描看到的原子行间距 = 最近邻距离 × √3/2。
#:
#: 原子分辨图上量到的周期是**原子行**的间距而不是最近邻距离本身 —— 六角密堆
#: 层里相邻两排原子的垂直距离是 a·√3/2。拿最近邻距离直接当 expected_a_nm 会系统
#: 性地偏大 15%。
_ROW_SPACING_FACTOR = math.sqrt(3.0) / 2.0


@dataclass(frozen=True)
class SubstrateFacts:
    """当前衬底,以及从知识库取出的、判据要用的那几个常数。

    ``available=False`` 表示**不知道**衬底是什么 —— 不是「没有衬底」,更不是
    「可以按默认值处理」。
    """

    available: bool = False
    material: str | None = None          # "Au(111)"
    type_id: str | None = None           # "clean_metal"
    source: str = "unknown"              # explicit / sample_record / unknown
    surface_state_onset_v: float | None = None
    nearest_neighbor_nm: float | None = None
    row_spacing_nm: float | None = None  # 扫描图上量到的原子行间距
    step_height_nm: float | None = None
    #: 表面重构的条纹周期(Au(111) 的 22×√3 herringbone soliton 线对 ~6.3 nm)。
    #: **这是先验,不是阈值** —— ``AssessHerringbone`` 拿它当搜索窗的中心,不当
    #: 判定门槛。没有重构的面(Cu/Ag/Pt(111))这里就是 ``None``,和 Pt(111) 没有
    #: 肖克利表面态一样是**物理事实,不是数据缺失**。
    reconstruction_period_nm: float | None = None
    sample_id: str | None = None
    sample_name: str | None = None
    reason: str = ""

    @property
    def has_surface_state(self) -> bool:
        """这个面有没有肖克利表面态(Pt(111) 就没有)。"""
        return self.surface_state_onset_v is not None


def current_sample_facts() -> dict:
    """当前样品记录 —— ``{"available", "sample_id", "name", "sample_type",
    "sample_subtype"}``。

    读不到一律 ``available=False`` + 空字段,不抛。
    """
    out = {"available": False, "sample_id": None, "name": None,
           "sample_type": None, "sample_subtype": None}
    try:
        from mast.logging.experiment_log import get_active_log

        log = get_active_log()
        if log is None:
            return out
        sample_id = getattr(log, "current_sample_id", None)
        storage = getattr(log, "_storage", None)
        if not sample_id or storage is None:
            return out
        row = storage.get_sample(sample_id)
        if not row:
            return out
        out.update({
            "available": True,
            "sample_id": str(sample_id),
            "name": row.get("name"),
            "sample_type": row.get("sample_type"),
            "sample_subtype": row.get("sample_subtype"),
        })
    except Exception as exc:  # noqa: BLE001 — 读不到样品不该让流程失败
        logger.debug("读当前样品失败(按「不知道」处理): %s", exc)
    return out


def _lookup_clean_metal(name: str) -> "tuple[str, str] | None":
    """把一个材料名解析成 ``(type_id, material_name)``,只接受洁净金属面。

    先精确、后模糊,并且**丢弃非 clean_metal 的模糊命中** —— 见模块注释里
    ``match_material("au111") → MoS2_on_Au111`` 那个实测陷阱。
    """
    if not name or not str(name).strip():
        return None
    query = str(name).strip()
    try:
        from mast.knowledge.lookups import get_constants, match_material
    except Exception as exc:  # noqa: BLE001
        logger.debug("知识库不可用: %s", exc)
        return None

    # 精确:名字直接就是 clean_metal 表里的键。
    try:
        if get_constants("clean_metal", query):
            return "clean_metal", query
    except Exception:  # noqa: BLE001
        pass

    try:
        hit = match_material(query)
    except Exception:  # noqa: BLE001
        hit = None
    if not hit:
        return None
    type_id, material = hit
    if str(type_id) != "clean_metal":
        # 模糊匹配把 "au111" 匹到了长在 Au(111) 上的二维材料这类情况 —— 那是另一
        # 个样品,它的常数不能拿来当衬底判据。
        logger.debug("材料 %r 模糊匹配到 %s/%s，不是洁净金属面，不采用",
                     query, type_id, material)
        return None
    return "clean_metal", str(material)


def substrate_from_material(material: str, *, source: str = "explicit",
                            sample: dict | None = None) -> SubstrateFacts:
    """给定材料名，取出判据要用的常数。认不出返回 ``available=False``。"""
    hit = _lookup_clean_metal(material)
    sample = sample or {}
    if hit is None:
        return SubstrateFacts(
            available=False, source=source,
            sample_id=sample.get("sample_id"), sample_name=sample.get("name"),
            reason=(f"认不出衬底 {material!r}（知识库里没有对应的洁净金属面）。"
                    f"表面态与晶格常数都取不到，判据无从建立。"))
    type_id, name = hit
    try:
        from mast.knowledge.lookups import get_constants

        const = get_constants(type_id, name) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("取 %s 常数失败: %s", name, exc)
        const = {}

    def _f(key: str) -> float | None:
        val = const.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    onset = _f("surface_state_onset_ev")
    nn_ang = _f("nearest_neighbor_ang")
    nn_nm = nn_ang / 10.0 if nn_ang else None
    step_ang = _f("step_height_ang")
    return SubstrateFacts(
        available=True,
        material=name,
        type_id=type_id,
        source=source,
        # 表面态 onset 的能量(eV)与偏压(V)在 STS 上数值相同 —— 都是相对费米面的
        # 电子能量除以元电荷。
        surface_state_onset_v=onset,
        nearest_neighbor_nm=nn_nm,
        row_spacing_nm=(nn_nm * _ROW_SPACING_FACTOR) if nn_nm else None,
        step_height_nm=(step_ang / 10.0) if step_ang else None,
        reconstruction_period_nm=_f("herringbone_period_nm"),
        sample_id=sample.get("sample_id"),
        sample_name=sample.get("name"),
        reason=("" if onset is not None else
                f"{name} 没有肖克利表面态 —— 这个面本来就没有，不是数据缺失。"),
    )


def resolve_substrate(explicit: str | None = None) -> SubstrateFacts:
    """当前衬底:显式指定 > 从样品记录推断 > 不知道。

    显式值认不出时**不回落到推断** —— 用户给了显式值就照它办;认不出要让用户知道
    自己写错了,而不是悄悄换成别的衬底。
    """
    if explicit and str(explicit).strip():
        return substrate_from_material(str(explicit).strip(), source="explicit",
                                       sample=current_sample_facts())

    sample = current_sample_facts()
    if not sample["available"]:
        return SubstrateFacts(
            available=False, source="unknown",
            reason=("读不到当前样品记录，不知道台面上是什么衬底。"
                    "请显式指定衬底（例如 Au(111)），或先用 start_sample 登记样品。"
                    "注意：读不到 ≠ 可以按 Au(111) 处理。"))

    for key in ("sample_subtype", "sample_type"):
        val = sample.get(key)
        if not val:
            continue
        facts = substrate_from_material(str(val), source="sample_record",
                                        sample=sample)
        if facts.available:
            return facts

    return SubstrateFacts(
        available=False, source="unknown",
        sample_id=sample.get("sample_id"), sample_name=sample.get("name"),
        reason=(f"当前样品登记为 {sample.get('sample_subtype') or sample.get('sample_type')!r}，"
                f"知识库里没有对应的洁净金属面。请显式指定衬底。"))


__all__ = ["SubstrateFacts", "current_sample_facts", "resolve_substrate",
           "substrate_from_material"]
