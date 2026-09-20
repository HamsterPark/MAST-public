"""Thread-safe event bus for real-time UI push via WebSocket.

Backend threads (SafetyWatchdog, InstrumentState, SkillExecutor) publish events.
The WebSocket handler subscribes and forwards to connected browser clients.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

# Ring buffer size — keeps the last N events so an HTTP-polling client can
# catch up after reconnecting (or when WebSocket is unavailable, e.g. across
# a corp firewall that strips the Upgrade header). 100 events ≈ several
# minutes of activity at typical mission rates.
_HISTORY_MAX = 100


class EventType(Enum):
    """Categories of real-time events pushed to the UI."""

    HARDWARE_STATE = "hardware_state"  # bias, current, Z, controller
    ANOMALY = "anomaly"                # safety watchdog trigger
    SCAN_COMPLETE = "scan_complete"    # new scan file detected
    SKILL_STEP = "skill_step"         # skill execution progress
    EXPERIMENT_UPDATE = "experiment"   # experiment / sample change
    CONNECTION = "connection"          # Nanonis connection change
    CURRENT_MONITOR = "current_monitor"  # tunnelling-current segment / alert / state
    # A versioned document landed on disk (2026-07-30). Until this existed, saving
    # a product emitted NOTHING — so "wake an agent once the thing it waits for
    # arrives" had no trigger to hook and could only be polled. Payload is
    # pointers + metadata only (doc_id / kind / version / path), never the body:
    # this bus replays the last 100 events, and report text would evict everything
    # else. See core/wake_scheduler.py for the consumer.
    ARTIFACT_SAVED = "artifact_saved"
    # One narration line landed in a conversation's transcript (2026-08-11) — the
    # side-channel that tells the OPERATOR what a long composite is doing while it
    # runs ("我们要打一发脉冲", "扫到 50%"). Pointers only ({conversation_id, seq, t}),
    # for the same reason as ARTIFACT_SAVED: a ForgeAuTip emits hundreds of these
    # and the narration text would evict everything else from the 100-event replay.
    # The client re-reads the rows it does not have via GET /api/chat/narration.
    # NEVER seen by an agent — see mast/chat/narration.py for why that is
    # structural (it is not on the message channel) rather than a filter.
    CHAT_NARRATION = "chat_narration"
    # ── campaign 指挥层的三种帧 (2026-08-15, campaign_director_design.md §7) ──
    #
    # 三个**分开的**类型而不是一个带 kind 的类型:面板对它们的反应不同 ——
    # status 触发一次 refetch,gate 追加一行闸门史,alert 亮一条告警条。合成一个
    # 类型的话每个客户端都要自己再分一次流,而「读端点必字面」那次教训正是
    # 分流写在两个地方、两边对不齐。
    #
    # ⚠️ 帧**只做触发,不做增量状态源**。面板收到任何一帧后 refetch
    # ``GET /api/conducts/{id}``(那是面板唯一的数据源)。理由:这条总线只重放
    # 最后 100 条,一个跑三天的 conduct 必然丢帧,而按帧累积状态的客户端会带着
    # 一个**错的**状态一直显示下去 —— 比没有实时推送糟得多。
    CONDUCT_STATUS = "conduct_status"
    CONDUCT_GATE = "conduct_gate"
    CONDUCT_ALERT = "conduct_alert"
    # 某个 agent 推荐用户订阅一个技能 (2026-08-26)。它**只是一次触发**：payload
    # 只有 {rec_id, skill, by_agent}，界面收到后 refetch
    # ``GET /api/skill-market/recommendations``（那才是唯一数据源）。理由同上面
    # 那段：这条总线只重放 100 条，按帧累积推荐列表的客户端迟早显示一份错的。
    #
    # agent 能到达的写路径**只有** pending 推荐 —— 它不能改自己的工具面。所以这
    # 一帧的意思是「有人想让你看一眼」，不是「工具面变了」。
    SKILL_RECOMMENDATION = "skill_recommendation"


@dataclass
class Event:
    """A single event to be broadcast."""

    type: EventType
    data: dict
    timestamp: float = field(default_factory=time.time)

    def to_json_dict(self) -> dict:
        """Serialise for WebSocket transmission."""
        return {
            "type": self.type.value,
            "data": self.data,
            "ts": self.timestamp,
        }


class EventBus:
    """Singleton publish/subscribe bus.

    * Thread-safe: any thread can call ``publish()``.
    * Subscribers are called synchronously inside ``publish()`` on the
      publishing thread.  The WebSocket handler uses an asyncio.Queue
      adapter so the actual ``send`` happens on the event loop.
    """

    _instance: EventBus | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()
        # Ring buffer of (monotonic_id, Event) so HTTP-polling clients can
        # ask "give me events with id > last_seen". Counter starts at 1 so
        # the polling client can request since_id=0 to get everything in
        # the buffer.
        self._history: deque[tuple[int, Event]] = deque(maxlen=_HISTORY_MAX)
        self._next_id: int = 1

    # ── Singleton accessor ────────────────────────────────────────
    @classmethod
    def get(cls) -> EventBus:
        """Return (or create) the global EventBus singleton."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Subscribe / unsubscribe ───────────────────────────────────
    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def subscribe_with_id(self, callback: "Callable[[int, Event], None]") -> Callable:
        """Subscribe and receive the event's SEQUENCE ID alongside it.

        The plain :meth:`subscribe` hands over an ``Event`` and nothing else, so
        a live subscriber cannot tell a client "you are at seq N" — and without
        that the reconnect contract (``?since=<seq>`` replays what you missed)
        has no way to produce the cursor in the first place. The ring buffer has
        always carried the ids; only the push path could not see them.

        Returns the *wrapper* actually registered, so the caller can pass it
        straight back to :meth:`unsubscribe`. Forgetting to unsubscribe is how a
        closed socket keeps receiving events forever.
        """
        def _wrapped(event: Event, _cb=callback) -> None:
            _cb(getattr(event, "_bus_seq", 0), event)

        with self._lock:
            self._subscribers.append(_wrapped)
        return _wrapped

    # ── Publish ───────────────────────────────────────────────────
    def publish(self, event: Event) -> None:
        """Broadcast *event* to all subscribers (synchronous, thread-safe).

        Also records the event in the ring buffer so HTTP-polling clients
        can fetch missed events via ``recent_events_since``.
        """
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._history.append((event_id, event))
            # Stamp the id ON the event so a push subscriber can report the
            # cursor a reconnecting client will send back as ``?since=``.
            # Set before fan-out and never mutated afterwards.
            try:
                object.__setattr__(event, "_bus_seq", event_id)
            except Exception:  # noqa: BLE001 — a frozen/exotic Event still fans out
                pass
            subs = tuple(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                logger.debug("EventBus subscriber error", exc_info=True)

    def recent_events_since(self, since_id: int) -> tuple[list[dict], int]:
        """Return events with id > since_id, plus the latest id seen.

        Used by the HTTP polling fallback (``/api/recent_events``) when the
        WebSocket transport is blocked. Each event is serialised via
        ``Event.to_json_dict`` plus a top-level ``id`` field so the client
        can update its ``since_id`` cursor.
        """
        with self._lock:
            snapshot = list(self._history)
            latest = self._next_id - 1
        out: list[dict] = []
        for ev_id, event in snapshot:
            if ev_id > since_id:
                payload = event.to_json_dict()
                payload["id"] = ev_id
                out.append(payload)
        return out, latest

    # ── Convenience helpers ───────────────────────────────────────
    def publish_hardware_state(self, **kwargs) -> None:
        """Shorthand: publish a HARDWARE_STATE event."""
        self.publish(Event(type=EventType.HARDWARE_STATE, data=kwargs))

    def publish_anomaly(self, **kwargs) -> None:
        """Shorthand: publish an ANOMALY event."""
        self.publish(Event(type=EventType.ANOMALY, data=kwargs))

    def publish_scan_complete(self, path: str) -> None:
        """Shorthand: publish a SCAN_COMPLETE event."""
        self.publish(Event(type=EventType.SCAN_COMPLETE, data={"path": path}))

    def publish_skill_step(self, skill: str, step: int, success: bool) -> None:
        """Shorthand: publish a SKILL_STEP event."""
        self.publish(Event(
            type=EventType.SKILL_STEP,
            data={"skill": skill, "step": step, "success": success},
        ))

    def publish_connection(self, connected: bool, detail: str = "") -> None:
        """Shorthand: publish a CONNECTION event."""
        self.publish(Event(
            type=EventType.CONNECTION,
            data={"connected": connected, "detail": detail},
        ))

    def publish_artifact_saved(self, **kwargs) -> None:
        """Shorthand: publish an ARTIFACT_SAVED event.

        Called by ``documents.store`` on every successful version write (new document
        AND new version — they share one success point). Pointers only: ``doc_id`` /
        ``kind`` / ``version`` / ``path`` / ``title`` / ``experiment_id``.

        ⚠️ Subscribers run SYNCHRONOUSLY on the publishing thread — here, that is the
        thread that just saved a document. A subscriber must therefore only ENQUEUE;
        anything slow (and certainly any LLM call) belongs on its own thread, or
        saving a report would block behind a scheduling decision.
        """
        self.publish(Event(type=EventType.ARTIFACT_SAVED, data=kwargs))

    def publish_chat_narration(self, conversation_id: str, seq: int,
                               t: float) -> None:
        """Shorthand: publish a CHAT_NARRATION event (pointer only).

        Called by the narration writer thread AFTER the row is committed, so a
        client that reacts by reading always finds it. Carries no text: see the
        EventType comment, and note that the same subscriber-runs-on-the-publisher
        rule as ``publish_artifact_saved`` applies — here the publishing thread is
        the narration writer, and a slow subscriber would stall the very queue
        that exists to keep narration off the instrument thread.
        """
        self.publish(Event(
            type=EventType.CHAT_NARRATION,
            data={"conversation_id": str(conversation_id or ""),
                  "seq": int(seq), "t": float(t)},
        ))

    def publish_conduct_status(self, conduct_id: str, **kwargs) -> None:
        """Shorthand: publish a CONDUCT_STATUS event (pointer only).

        Emitted by the ConductDirector on every state transition and on the
        60 s wait-state heartbeat frame. Carries ``conduct_id`` + the event
        kind + the status either side — never the panel payload, because the
        panel is rendered from ONE endpoint (``GET /api/conducts/{id}``) and a
        second, partial copy of that state on the wire is how the two drift.
        """
        self.publish(Event(type=EventType.CONDUCT_STATUS,
                           data={"conduct_id": str(conduct_id or ""), **kwargs}))

    def publish_conduct_gate(self, conduct_id: str, **kwargs) -> None:
        """Shorthand: publish a CONDUCT_GATE event (one gate verdict)."""
        self.publish(Event(type=EventType.CONDUCT_GATE,
                           data={"conduct_id": str(conduct_id or ""), **kwargs}))

    def publish_conduct_alert(self, conduct_id: str, **kwargs) -> None:
        """Shorthand: publish a CONDUCT_ALERT event.

        ``severity`` ∈ info | warn | crit and ``code`` says which condition
        (stale / stall / yielding / budget / estop / wait / …). This is the frame
        that has to reach a human who is NOT looking at the panel, so it is also
        the one the notifier pairs with a durable wishlist request.
        """
        self.publish(Event(type=EventType.CONDUCT_ALERT,
                           data={"conduct_id": str(conduct_id or ""), **kwargs}))

    def publish_current_monitor(self, **kwargs) -> None:
        """Shorthand: publish a CURRENT_MONITOR event.

        ``kind`` discriminates the payload: ``segment`` (one scalar summary per
        acquired segment, ~1 Hz), ``aux`` (the Z / qPlus-amplitude sample taken
        at the same segment boundary), ``alert``, or ``status`` (emitted only on
        a state transition). Scalars only — waveforms go to .npy and are fetched
        over REST, because this bus replays just the last 100 events and a
        stream of arrays would evict everything else.
        """
        self.publish(Event(type=EventType.CURRENT_MONITOR, data=kwargs))
