# main.py
import base64
import hashlib
import streamlit as st
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage

from app.core.chains import build_chat_chain
from app.config.settings import settings
from app.service.tts import tts_bytes

# =========================
# 페이지 & 말풍선 스타일
# =========================
st.set_page_config(page_title="TTS Chatbot", page_icon="🔊", layout="centered")
st.title("🔊 TTS Chatbot")

st.markdown("""
<style>
.chat-container { max-width: 820px; margin: 0 auto; }
.chat-bubble {
  display: inline-block;
  max-width: 80%;
  padding: 12px 16px;
  margin: 8px 6px;
  border-radius: 18px;
  line-height: 1.65;
  font-size: 16px;
  word-break: break-word;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.user-bubble { background: #2563eb; color: #ffffff; float: right; border-bottom-right-radius: 6px; }
.assistant-bubble { background: #f5f5f7; color: #111827; float: left; border-bottom-left-radius: 6px; }
.clearfix { clear: both; }
.replay-row { margin: -2px 6px 8px; }
.replay-row small { color: #6b7280; }
</style>
""", unsafe_allow_html=True)

def render_bubble(role: str, text: str):
    cls = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(
        f'<div class="chat-container"><div class="chat-bubble {cls}">{text}</div><div class="clearfix"></div></div>',
        unsafe_allow_html=True
    )

def autoplay_audio(audio_bytes: bytes, mime: str = "audio/mpeg"):
    """사용자 상호작용 후 자동 재생"""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay><source src="data:{mime};base64,{b64}"></audio>',
        unsafe_allow_html=True
    )

def tts_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# =========================
# 세션 상태
# =========================
if "history" not in st.session_state:
    st.session_state.history: List[HumanMessage | AIMessage] = []

if "provider" not in st.session_state:
    st.session_state.provider = "openai"  # "openai" | "gemini"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

if "chain" not in st.session_state:
    model = settings.MODEL_OPENAI if st.session_state.provider == "openai" else settings.MODEL_GEMINI
    st.session_state.chain = build_chat_chain(
        provider=st.session_state.provider,
        model=model,
        temperature=st.session_state.temperature,
        system_prompt_path="app/core/prompts/system_ko.txt",
        streaming_callback=None,  # 완성본만 렌더
    )

# TTS 설정/캐시
st.session_state.setdefault("tts_enabled", True)     # 자동 낭독 기본 ON
st.session_state.setdefault("tts_cache", {})         # type: Dict[str, bytes]

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.subheader("Settings")
    provider = st.selectbox(
        "Provider",
        ["openai", "gemini"],
        index=0 if st.session_state.provider == "openai" else 1,
        format_func=lambda x: "OpenAI (ChatGPT)" if x == "openai" else "Google (Gemini)",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
    st.divider()
    st.session_state.tts_enabled = st.toggle("🔊 AI 응답 자동 재생 (TTS)", value=st.session_state.tts_enabled)

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

# =========================
# 과거 대화 렌더 + 각 AI 말풍선 아래 “다시 듣기”
# =========================
for i, m in enumerate(st.session_state.history):
    if isinstance(m, HumanMessage):
        render_bubble("user", m.content)
    else:
        render_bubble("assistant", m.content)
        # 말풍선 바로 아래에 '응답 다시 듣기' 버튼 (개별 키 필수)
        with st.container():
            c1, c2 = st.columns([1, 6])
            with c1:
                if st.button("🔁 다시 듣기", key=f"replay_{i}", use_container_width=True):
                    key = tts_cache_key(m.content)
                    audio = st.session_state.tts_cache.get(key)
                    if audio is None:
                        try:
                            audio = tts_bytes(m.content) or b""
                            st.session_state.tts_cache[key] = audio
                        except Exception as e:
                            st.warning(f"TTS 오류: {e}")
                            audio = b""
                    autoplay_audio(audio)
            with c2:
                st.markdown('<div class="replay-row"><small>AI 응답 오디오 재생</small></div>', unsafe_allow_html=True)

# =========================
# 하단 고정 입력: st.chat_input (자동 하단 고정 + 우측 전송 버튼 + Enter 전송)
# =========================
user_text = st.chat_input("메시지를 입력하세요…")  # ← Streamlit 기본 하단 고정

if user_text and user_text.strip():
    user_text = user_text.strip()

    # 1) 유저 메시지
    st.session_state.history.append(HumanMessage(user_text))
    render_bubble("user", user_text)

    # 2) 모델 응답
    resp = st.session_state.chain.invoke({"history": st.session_state.history, "input": user_text})
    ai_text = getattr(resp, "content", str(resp))
    st.session_state.history.append(AIMessage(ai_text))
    render_bubble("assistant", ai_text)

    # 3) 자동 음성 재생 + 캐시 저장
    if st.session_state.tts_enabled:
        try:
            key = tts_cache_key(ai_text)
            audio = st.session_state.tts_cache.get(key)
            if audio is None:
                audio = tts_bytes(ai_text) or b""
                st.session_state.tts_cache[key] = audio
            autoplay_audio(audio)
        except Exception as e:
            st.warning(f"TTS 오류: {e}")
