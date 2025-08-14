
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.core.chains import build_chat_chain
from app.config.settings import settings  # settings.MODEL_OPENAI / settings.MODEL_GEMINI 사용

st.set_page_config(page_title="Practice Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Practice Chatbot")

# -------------------------
# 말풍선 스타일 추가 (CSS)
# -------------------------
st.markdown("""
<style>
.chat-container { max-width: 820px; margin: 0 auto; }
.chat-bubble {
  display: inline-block;
  max-width: 80%;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 18px;
  line-height: 1.65;
  font-size: 16px;
  word-break: break-word;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.user-bubble { 
  background: #2563eb; 
  color: #ffffff; 
  float: right; 
  border-bottom-right-radius: 6px;
}
.assistant-bubble { 
  background: #f5f5f7; 
  color: #111827; 
  float: left; 
  border-bottom-left-radius: 6px;
}
.clearfix { clear: both; }
</style>
""", unsafe_allow_html=True)
# --- 세션 상태 ---
if "history" not in st.session_state:
    st.session_state.history = []
if "provider" not in st.session_state:
    st.session_state.provider = "openai"   # 또는 "gemini"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2
if "chain" not in st.session_state:
    model = settings.MODEL_OPENAI if st.session_state.provider == "openai" else settings.MODEL_GEMINI
    st.session_state.chain = build_chat_chain(
        provider=st.session_state.provider,
        model=model,
        temperature=st.session_state.temperature,
        system_prompt_path="app/core/prompts/system_ko.txt",
        streaming_callback=None,
    )

# --- 사이드바: provider & temperature ---
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
        model = settings.MODEL_OPENAI if provider == "openai" else settings.MODEL_GEMINI
        st.session_state.chain = build_chat_chain(
            provider=provider,
            model=model,
            temperature=temperature,
            system_prompt_path="app/core/prompts/system_ko.txt",
            streaming_callback=None,
        )
        st.toast(f"🔁 {provider}로 전환됨", icon="✅")

# -------------------------
# 말풍선 렌더 함수
# -------------------------
def render_bubble(role: str, text: str):
    """role: 'user' or 'assistant'"""
    cls = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(
        f'<div class="chat-container"><div class="chat-bubble {cls}">{text}</div><div class="clearfix"></div></div>',
        unsafe_allow_html=True
    )

# 과거 대화 출력
for m in st.session_state.history:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    render_bubble(role, m.content)

# 입력 처리
user_input = st.chat_input("메시지를 입력하세요…")
if user_input:
    st.session_state.history.append(HumanMessage(user_input))
    render_bubble("user", user_input)

    resp = st.session_state.chain.invoke({"history": st.session_state.history, "input": user_input})
    ai_text = getattr(resp, "content", str(resp))

    st.session_state.history.append(AIMessage(ai_text))
    render_bubble("assistant", ai_text)