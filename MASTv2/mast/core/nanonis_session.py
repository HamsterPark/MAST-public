"""把 Nanonis 的保存目录指向当前样品 —— **手动动作，默认永不自动执行**。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §7.7

现场明确保留了「在 Nanonis 里另设一个数据保存目录」这种用法 —— 那个设置
属于用户，MAST 不背着用户改。副本机制在「Nanonis 存在任意别处」的前提下独立
工作，这个模块只是额外提供一个**由用户手动触发**的入口。

本模块是**纯计算**：算路径、做预检、给出计划。真正下发 ``SetSessionPath`` 的是
调用方（API 路由 / meta-tool），因为 ``core/`` 不能 import ``skills/``。

为什么落到 ``raw/nanonis/`` 而不是 ``raw/sxm/``
-----------------------------------------------

Nanonis 会在 session 目录里写它自己的东西：``.ini``、settings 备份、
``unnamed####`` 半成品。给它一块专属地盘，``raw/{sxm,dat,3ds}/`` 才能保持
"只含完整、已哈希、已登记文件"的语义 —— 而那条语义正是 MANAGED 区（递归闸门）
的定义依据。

可逆性
------

改之前先把原路径存进 ``experiment.json.nanonis.previous_session_path``，UI 据此
提供常驻的「恢复原保存目录」。用户任何时候都能退回自己的目录。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def target_session_dir(exp_dir: Path, sample_dir_name: str,
                       *, create: bool = True) -> Path:
    """当前样品的 ``raw/nanonis/``。

    默认 ``create=True``：Nanonis 对不存在的路径行为不确定，先建出来。
    """
    from mast.core.experiment_paths import nanonis_inplace_dir, sample_dir
    sp = sample_dir(exp_dir, sample_dir_name)
    return nanonis_inplace_dir(sp, create=create)


def is_inplace_active(current_session_path: str | None,
                      exp_dir: Path | None, sample_dir_name: str | None) -> bool:
    """Nanonis 当前是不是正指着**当前样品**的原位目录。"""
    if not current_session_path or exp_dir is None or not sample_dir_name:
        return False
    from mast.core.experiment_paths import is_within
    try:
        return is_within(current_session_path,
                         target_session_dir(exp_dir, sample_dir_name, create=False))
    except (OSError, ValueError):
        return False


def session_path_sample(current_session_path: str | None,
                        exp_dir: Path | None) -> str | None:
    """如果 Nanonis 指着**某个**样品的原位目录，返回那个样品的目录名。

    用来检测「用户换了样品，但 Nanonis 还在往上一个样品的目录里写」——
    这是原位模式下最危险、且必然会发生的失效模式。
    """
    if not current_session_path or exp_dir is None:
        return None
    from mast.core.experiment_paths import is_within
    try:
        samples = Path(exp_dir) / "samples"
        if not is_within(current_session_path, samples):
            return None
        rel = Path(current_session_path).absolute().relative_to(
            Path(samples).absolute())
        return rel.parts[0] if rel.parts else None
    except (OSError, ValueError):
        return None


def plan_repoint(current_session_path: str | None, exp_dir: Path | None,
                 sample_dir_name: str | None,
                 *, experiment_title: str = "", sample_name: str = "") -> dict:
    """算出「把 Nanonis 指向当前样品」这一步该做什么。**纯计算，不下发。**

    返回::

        {ok, needs_change, target, previous_session_path, warnings[], reason}

    调用方拿到 ``ok and needs_change`` 才去调 ``SetSessionPath``。
    """
    out: dict = {
        "ok": False, "needs_change": False, "target": None,
        "previous_session_path": current_session_path or None,
        "warnings": [], "reason": "",
    }
    if exp_dir is None or not sample_dir_name:
        out["reason"] = "当前没有活跃样品——请先选择或新建一个样品。"
        return out

    try:
        target = target_session_dir(exp_dir, sample_dir_name, create=False)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"无法计算目标路径：{exc}"
        return out

    out["ok"] = True
    out["target"] = str(target)

    if is_inplace_active(current_session_path, exp_dir, sample_dir_name):
        out["reason"] = "Nanonis 已经指向当前样品的目录，无需改动。"
        return out
    out["needs_change"] = True

    # 路径预算：raw/nanonis/ 比 raw/sxm/ 多吃 4 个字符，加上 Nanonis 自己的
    # 文件名，长中文实验名 + 长样品名有可能顶到 Windows MAX_PATH。
    from mast.core.experiment_paths import experiment_root, fits_path_budget
    try:
        if not fits_path_budget(experiment_root(), Path(exp_dir).name,
                                sample_dir_name, inplace=True):
            out["warnings"].append(
                "实验名或样品名较长，原位保存的完整路径接近 Windows 260 字符上限；"
                "Nanonis 保存长文件名时可能失败。建议改用默认的复制模式。")
    except Exception:  # noqa: BLE001
        pass

    if current_session_path:
        out["warnings"].append(
            f"Nanonis 当前的保存目录是 {current_session_path} —— 改动后可随时"
            f"用「恢复原保存目录」退回。")
    else:
        out["warnings"].append(
            "读不到 Nanonis 当前的保存目录，因此无法提供一键恢复；"
            "请自行记下它现在的设置。")

    out["reason"] = (
        f"把 Nanonis 的保存目录指向"
        f"{('实验「' + experiment_title + '」的') if experiment_title else ''}"
        f"样品「{sample_name or sample_dir_name}」的 raw/nanonis/ 目录。"
        f"之后 Nanonis 直接写进实验文件夹，MAST 只登记不再复制。")
    return out


def stale_warning(session_sample_dir: str, current_sample_dir: str,
                  *, current_sample_name: str = "") -> str:
    """「Nanonis 还指着上一个样品」的告警文案。"""
    return (
        f"Nanonis 的保存目录仍指向样品目录 {session_sample_dir}，"
        f"而当前样品是 {current_sample_name or current_sample_dir}。\n"
        f"数据不会记错——归属以当前样品为准，副本已经落到当前样品下——但"
        f"Nanonis 那边的原始文件还堆在旧样品的文件夹里。"
        f"建议把 Nanonis 的保存目录重新指向当前样品，或改回你自己的目录。")
