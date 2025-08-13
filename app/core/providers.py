from __future__ import annotations
from typing import Literal, Optional, Any, List

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

ProviderName = Literal["openai", "gemini"]

def get_chat_model(
        provider: ProviderName = "gemini",
        model: Optional[str] = None,
        temperature: float = 0.2, # 질의응답 용이므로 낮게 설정
        streaming: bool = True, # 실시간 응답 가능 (False 일때는 응답이 끝나면 한번에 출력)
        callbacks: Optional[List] = None,
        **kwargs: Any, # 각각의 추가 인자들을 모두 전달할 수 있도록
):
    callbacks = callbacks or []

    if provider == "openai":
        _model = model or "gpt-3.5-turbo"
        return ChatOpenAI(
            model=_model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            **kwargs,
        )
    elif provider == "gemini":
        _model = model or "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(
            model=_model,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")