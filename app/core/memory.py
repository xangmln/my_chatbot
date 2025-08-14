from __future__ import annotations
from typing import Dict
from langchain.memory import ConversationBufferMemory

from __future__ import annotations
from typing import Dict
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory # 추후 확장 대비

class SessionMemoryStore:
    """session_id 별로 ConversationBufferMemory를 관리하는 심플 스토어."""
    def __init__(self) -> None:
        self._store: Dict[str, ConversationBufferMemory] = {}

    def get(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self._store:
            self._store[session_id] = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
            )
        return self._store[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            del self._store[session_id]


session_memory = SessionMemoryStore()