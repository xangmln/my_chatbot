from __future__ import annotations
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from app.core.providers import get_chat_model, ProviderName
from app.core.utils import load_text

def build_chat_chain(
    *,
    provider: ProviderName = "openai",
    model: Optional[str] = None,
    temperature: float = 0.2,
    system_prompt_path: str = "app/core/prompts/system_ko.txt",
    streaming_callback=None,
) -> Runnable:
    system_prompt = load_text(system_prompt_path).strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    callbacks = [streaming_callback] if streaming_callback else None
    llm = get_chat_model(
        provider=provider,
        model=model,
        temperature=temperature,
        streaming=True if callbacks else False,
        callbacks=callbacks,
    )
    return prompt | llm
