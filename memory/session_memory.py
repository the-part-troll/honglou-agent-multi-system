"""Session-scoped short-term memory for Streamlit and LangGraph runs."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import Config


class SessionMemoryManager:
    """Manage short-term chat history by session_id."""

    def __init__(self) -> None:
        self._messages: dict[str, list[BaseMessage]] = defaultdict(list)
        self._lock = RLock()

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return recent messages for a session."""

        with self._lock:
            return list(self._messages[session_id][-Config.MAX_HISTORY_MESSAGES :])

    def append_user_message(self, session_id: str, content: str) -> None:
        """Append a user message."""

        with self._lock:
            self._messages[session_id].append(HumanMessage(content=content))
            self._trim(session_id)

    def append_ai_message(self, session_id: str, content: str) -> None:
        """Append an assistant message."""

        with self._lock:
            self._messages[session_id].append(AIMessage(content=content))
            self._trim(session_id)

    def clear(self, session_id: str) -> None:
        """Clear a specific session."""

        with self._lock:
            self._messages.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        """Keep memory bounded."""

        max_messages = Config.MAX_HISTORY_MESSAGES
        if len(self._messages[session_id]) > max_messages:
            self._messages[session_id] = self._messages[session_id][-max_messages:]


session_memory = SessionMemoryManager()
