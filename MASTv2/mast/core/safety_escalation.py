"""Process-local record of a refused tip-approach escalation.

ApproachTip first attempts controlled engagement and escalates only after its
checks justify coarse approach. A refusal must remain effective if another
entry point directly requests AutoApproach. This module guards that sequence
without changing AutoApproach metadata or the normal verified escalation path.

The record is bounded by TTL and is not persisted in a model checkpoint. A
subsequent ApproachTip attempt makes a fresh decision: engagement or permitted
escalation clears it, while another refusal updates its reason and timestamp.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


#: How long a refusal keeps blocking the side door. Long enough to cover the
#: incident's 133 s gap with room to spare, short enough that a refusal nobody
#: ever revisits does not outlive the hardware state it described. A retry of
#: ``ApproachTip`` supersedes it immediately, so this ceiling only matters when
#: the agent never goes back through the front door.
APPROACH_REFUSAL_TTL_S = 600.0

#: Skills that ARE the coarse approach escalation. ``MotorMove z-approach`` is
#: absent on purpose — it is already fail-closed in SafetyGate Layer 0 and
#: never reaches this check.
_ESCALATION_SKILLS = frozenset({"AutoApproach"})


@dataclass(frozen=True)
class ApproachRefusal:
    """A recorded refusal to escalate to the coarse approach."""

    reason: str
    source: str
    at_monotonic: float
    ttl_s: float
    #: WHICH chain recorded it (group run / private chat / signals API), so a
    #: different chain's success cannot clear it. See clear_approach_refusal.
    owner: str = ""

    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.at_monotonic)

    def expired(self) -> bool:
        return self.age_s() >= self.ttl_s


_lock = threading.Lock()
_refusal: ApproachRefusal | None = None


def is_approach_escalation(skill_name: str) -> bool:
    """True iff *skill_name* starts the coarse current-feedback approach."""
    return skill_name in _ESCALATION_SKILLS


def record_approach_refusal(
    reason: str,
    *,
    source: str = "ApproachTip",
    ttl_s: float = APPROACH_REFUSAL_TTL_S,
    owner: str = "",
) -> None:
    """Remember that the safe approach path declined to escalate.

    Overwrites any previous refusal — the newest verdict is the live one.
    ``owner`` identifies the chain that recorded it (group run / private chat /
    signals API); see :func:`clear_approach_refusal`.
    """
    global _refusal
    with _lock:
        _refusal = ApproachRefusal(
            reason=str(reason or "").strip() or "(no reason recorded)",
            source=source,
            at_monotonic=time.monotonic(),
            ttl_s=float(ttl_s),
            owner=str(owner or ""),
        )
    logger.warning(
        "approach escalation refused by %s (%s) — direct AutoApproach gated for "
        "%.0fs: %s", source, owner or "unscoped", ttl_s, reason,
    )


def clear_approach_refusal(why: str = "", *, owner: str | None = None) -> bool:
    """Drop the refusal — a fresh decision has superseded it. Idempotent.

    SCOPED (审计 致命一(c)). The refusal itself is
    deliberately process-global: there is one tip and one sample, so a refusal
    to escalate must gate every chain. Its CLEARING used to be global too, and
    that is not symmetric — a successful engage in the private chat wiped a
    refusal the group chat had recorded ten seconds earlier, reopening the very
    side door (refused ApproachTip → direct AutoApproach 133 s later) that this
    gate was added for on 2026-07-27.

    So: passing ``owner`` clears only a refusal recorded by that same chain. An
    unscoped refusal (owner "") can be cleared by anyone — legacy behaviour, and
    the right default for one. ``owner=None`` is the administrative override:
    unconditional, and logged as such.

    Returns True iff a refusal was actually dropped.
    """
    global _refusal
    with _lock:
        cur = _refusal
        if cur is None:
            return False
        if owner is not None and cur.owner and cur.owner != str(owner):
            logger.warning(
                "approach refusal NOT cleared: it was recorded by %s, and %s "
                "cannot clear another chain's refusal (%s)",
                cur.owner, owner, why or "no reason given")
            return False
        _refusal = None
    logger.info("approach refusal cleared by %s%s",
                owner if owner is not None else "administrative override",
                f": {why}" if why else "")
    return True


def active_approach_refusal() -> ApproachRefusal | None:
    """The live refusal, or None. Expired entries are dropped on read."""
    global _refusal
    with _lock:
        r = _refusal
        if r is not None and r.expired():
            _refusal = None
            r = None
    return r


__all__ = [
    "APPROACH_REFUSAL_TTL_S",
    "ApproachRefusal",
    "active_approach_refusal",
    "clear_approach_refusal",
    "is_approach_escalation",
    "record_approach_refusal",
]
