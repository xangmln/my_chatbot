# main.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.core.chains import build_chat_chain
from app.config.settings import settings  # settings.MODEL_OPENAI / settings.MODEL_GEMINI 사용

st.set_page_config(page_title="Provider Switch Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Provider Switch Chatbot")

# --- 세션 상태 ---
if "history" not in st.session_state:
    st.session_state.history = []
if "provider" not in st.session_state:
    st.session_state.provider = "openai"   # 또는 "gemini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "chain" not in st.session_state:
    # settings에서 모델명 선택
    model = settings.MODEL_OPENAI if st.session_state.provider == "openai" else settings.MODEL_GEMINI
    st.session_state.chain = build_chat_chain(
        provider=st.session_state.provider,
        model=model,
        temperature=st.session_state.temperature,
        system_prompt_path="app/core/prompts/system_ko.txt",
        streaming_callback=None,
    )

# --- 사이드바: 제공자 & 온도만 ---
with st.sidebar:
    st.subheader("Settings")
    provider = st.selectbox(
        "Provider",
        ["openai", "gemini"],
        index=0 if st.session_state.provider == "openai" else 1,
        format_func=lambda x: "OpenAI (ChatGPT)" if x == "openai" else "Google (Gemini)",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)

    if provider != st.session_state.provider or temperature != st.session_state.temperature:
        st.session_state.provider = provider
        st.session_state.temperature = temperature
        # settings의 모델명을 사용해 체인 재생성
        model = settings.MODEL_OPENAI if provider == "openai" else settings.MODEL_GEMINI
        st.session_state.chain = build_chat_chain(
            provider=provider,
            model=model,
            temperature=temperature,
            system_prompt_path="app/core/prompts/system_ko.txt",
            streaming_callback=None,
        )
        st.toast(f"🔁 {provider}로 전환됨", icon="✅")

# --- 과거 대화 출력 ---
for m in st.session_state.history:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# --- 입력 처리 ---
user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    st.session_state.history.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    resp = st.session_state.chain.invoke({"history": st.session_state.history, "input": user_input})
    ai_text = getattr(resp, "content", str(resp))

    st.session_state.history.append(AIMessage(ai_text))
    with st.chat_message("assistant"):
        st.markdown(ai_text)
