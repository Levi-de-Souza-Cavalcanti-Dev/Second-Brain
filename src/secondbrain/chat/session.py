"""Conversational REPL with optional session persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from secondbrain.config import Settings
from secondbrain.constants import SESSIONS_DIRNAME
from secondbrain.models import AskRequest
from secondbrain.rag.pipeline import answer_question


@dataclass
class ChatSession:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def save(self, base_dir: Path) -> None:
        sessions_dir = base_dir / SESSIONS_DIRNAME
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{self.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for msg in self.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")


async def chat_turn(settings: Settings, session: ChatSession, user_text: str) -> str:
    req = AskRequest(query=user_text, top_k=8)
    session.append("user", user_text)
    out = await answer_question(settings, req)
    session.append("assistant", out.answer)
    return out.answer


def new_session() -> ChatSession:
    return ChatSession(session_id=str(uuid.uuid4()))
