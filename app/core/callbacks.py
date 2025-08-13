from __future__ import annotations
from typing import Any
from langchain.callbacks.base import BaseCallbackHandler

class StdoutStreamingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.total_chars = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True) # streaming: true이므로 flush: true 설정
        self.total_chars += len(token)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        print(f"\n[done] chars={self.total_chars}")