"""Cross-check what an agent SAID it did against what the records say it did.

 the instrument_control agent reported

    [HANDOFF → data_processing] 5-point STS grid acquired (40 nm spacing, …).
    Per-point summary JSON: D:\\MAST-data\\artifacts\\tool_returns\\GridSTS_a0a17a99.txt

…and the run finished as 「完成」. In the 25 seconds that message covers, the
service log contains three HTTP calls to the model provider and nothing else:
no skill ran, no action row was written, no marker was placed, and the file it
named was never created. Nothing in the UI distinguishes "IC did it" from "IC said it did it".

This module is the comparison itself, kept pure so it can be tested and so it
can be called from wherever a run's final text is available:

    audit_claim(text, executed_skills=[...], artifacts=[...])

It reports only what it can PROVE from the run's own record:

  * a file path named in the text that this run did not produce and that does
    not exist on disk — the agent cited an artifact that isn't there;
  * a completion claim about a measurement in a run where the record shows
    ZERO skill calls — nothing was executed, so nothing could have completed.

Both rules are deliberately narrow. An audit that cries wolf gets switched off,
and the failure mode being defended against here is a *confident, specific*
report of work that never happened — which is exactly what these two rules see.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Absolute paths with a file extension: Windows (D:\a\b.txt, \\host\share\x.sxm)
# and POSIX (/a/b.dat). Bare filenames are NOT matched — too many false hits
# from ordinary prose.
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\s\\]+\\|/)"       # drive / UNC / root
    r"[^\s\"'<>|?*]{1,400}?"                      # body (lazy)
    r"\.[A-Za-z0-9]{1,8}"                         # extension
)

# Said of a measurement/observation, these mean "it has been done".
_DONE_WORDS = (
    "acquired", "collected", "measured", "recorded", "completed", "finished",
    "performed", "captured", "obtained", "scanned",
    "已完成", "已采集", "已测量", "已记录", "已扫描", "已获取", "采集完成",
    "测量完成", "扫描完成", "完成了",
)
# The claim has to be ABOUT instrument work, not about e.g. finishing a summary.
_WORK_WORDS = (
    "sts", "spectr", "spectrum", "spectra", "scan", "image", "topograph",
    "grid", "curve", "i-v", "iv ", "didv", "di/dv", "approach", "pulse",
    "谱", "扫图", "扫描", "图像", "曲线", "进针", "脉冲", "测量",
)


@dataclass
class ClaimAudit:
    """Verdict for one agent claim. ``ok`` is False iff something is provable."""
    ok: bool = True
    executed_skills: list[str] = field(default_factory=list)
    claimed_paths: list[str] = field(default_factory=list)
    fabricated_paths: list[str] = field(default_factory=list)
    unsupported_completion: bool = False
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "executed_skills": list(self.executed_skills),
            "claimed_paths": list(self.claimed_paths),
            "fabricated_paths": list(self.fabricated_paths),
            "unsupported_completion": self.unsupported_completion,
            "problems": list(self.problems),
        }

    def notice(self) -> str:
        """One operator-facing line, or '' when there is nothing to say."""
        if self.ok:
            return ""
        return "⚠️ 本次运行的记录与该消息不符：" + "；".join(self.problems)


def extract_claimed_paths(text: str) -> list[str]:
    """Absolute file paths mentioned in *text*, de-duplicated, order preserved."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _PATH_RE.finditer(text):
        p = m.group(0).rstrip(".,;:)]}）】。，、")
        key = p.lower().replace("/", "\\")
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def claims_completed_work(text: str) -> bool:
    """True when *text* asserts that instrument work has been carried out."""
    if not text:
        return False
    low = text.lower()
    return (any(w in low for w in _DONE_WORDS)
            and any(w in low for w in _WORK_WORDS))


def _same_path(a: str, b: str) -> bool:
    return (os.path.normcase(os.path.normpath(a))
            == os.path.normcase(os.path.normpath(b)))


def audit_claim(
    text: str,
    *,
    executed_skills: "list[str] | tuple[str, ...] | None" = None,
    artifacts: "list[str] | tuple[str, ...] | None" = None,
    path_exists=os.path.exists,
) -> ClaimAudit:
    """Compare an agent's claim against this run's record.

    Args:
      text:             what the agent said (a handoff note / final answer).
      executed_skills:  skills the RECORD says ran in this run. An empty list
                        means the record shows nothing ran — which is different
                        from ``None`` (no record available), where the
                        zero-skills rule is not applied at all.
      artifacts:        files this run really produced.
      path_exists:      injectable for tests.
    """
    skills = list(executed_skills) if executed_skills is not None else None
    arts = list(artifacts or [])
    a = ClaimAudit(executed_skills=list(skills or []))

    a.claimed_paths = extract_claimed_paths(text)
    for p in a.claimed_paths:
        if any(_same_path(p, q) for q in arts):
            continue
        try:
            if path_exists(p):
                continue
        except Exception:  # noqa: BLE001 — an unreadable path is not proof
            continue
        a.fabricated_paths.append(p)
    if a.fabricated_paths:
        a.problems.append(
            "消息里引用的文件本次运行没有产生、磁盘上也不存在：" +
            "、".join(a.fabricated_paths))

    if skills is not None and not skills and claims_completed_work(text):
        a.unsupported_completion = True
        a.problems.append("消息声称已完成测量，但本次运行没有任何技能被调用")

    a.ok = not a.problems
    return a


__all__ = ["ClaimAudit", "audit_claim", "claims_completed_work",
           "extract_claimed_paths"]
