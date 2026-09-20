"""Unified conversation runtime for MAST.

A real per-agent **private chat** (私聊) onto the EXISTING agent engines (the
same compiled graphs the orchestrator routes to), plus durable multi-conversation
management shared by both the main Chat tab (私聊 instrument_control by default)
and the Agents page. Conversation identity lives entirely in the LangGraph
``thread_id``; the checkpointer is the source of truth for message history.

  - ``ConversationStore`` — SQLite index of conversations (id ↔ agent ↔ thread).
  - ``ConversationEngine`` — builds/caches standalone agent graphs and drives a
    turn (stream_turn) / reads history back (get_messages).
  - ``render`` — LangGraph message channel → chat-markdown for the GUI.
"""

from mast.chat.store import ConversationStore

__all__ = ["ConversationStore"]
