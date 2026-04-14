"""
memory.py — In-session memory for chat history and action log.

Keeps a rolling window of conversation turns and a full action history
so the UI can display "what has been done this session".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ActionRecord:
    timestamp: str
    transcription: str
    intents: list[str]
    actions: list[str]
    files_created: list[str]


class SessionMemory:
    """Lightweight in-process session store (resets on app restart)."""

    def __init__(self, max_chat_turns: int = 20):
        self.max_chat_turns = max_chat_turns
        self._chat_history: list[dict] = []   # [{role, content}, ...]
        self._action_log: list[ActionRecord] = []

    # ── Chat history ─────────────────────────────────────────────────────────

    def add_user_turn(self, text: str) -> None:
        self._chat_history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_turn(self, text: str) -> None:
        self._chat_history.append({"role": "assistant", "content": text})
        self._trim()

    def get_chat_history(self) -> list[dict]:
        return list(self._chat_history)

    def _trim(self) -> None:
        if len(self._chat_history) > self.max_chat_turns * 2:
            self._chat_history = self._chat_history[-(self.max_chat_turns * 2):]

    # ── Action log ───────────────────────────────────────────────────────────

    def log_action(
        self,
        transcription: str,
        intents: list[str],
        actions: list[str],
        files: list[str],
    ) -> None:
        self._action_log.append(
            ActionRecord(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                transcription=transcription,
                intents=intents,
                actions=actions,
                files_created=files,
            )
        )

    def get_action_log(self) -> list[ActionRecord]:
        return list(self._action_log)

    def clear(self) -> None:
        self._chat_history.clear()
        self._action_log.clear()

    # ── Summary for UI ────────────────────────────────────────────────────────

    def to_history_markdown(self) -> str:
        if not self._action_log:
            return "_No actions yet this session._"
        lines = []
        for rec in reversed(self._action_log):
            files_str = ", ".join(f"`{f}`" for f in rec.files_created) or "—"
            lines.append(
                f"**{rec.timestamp}** · _{', '.join(rec.intents)}_ \n"
                f"> {rec.transcription[:80]}{'…' if len(rec.transcription) > 80 else ''}\n"
                f"> Files: {files_str}"
            )
        return "\n\n".join(lines)
