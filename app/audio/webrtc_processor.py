# app/audio/webrtc_processor.py
import time
import queue
import numpy as np
import av
from collections import deque
from streamlit_webrtc import AudioProcessorBase 
from app.config.settings import settings

class MicAudioProcessor(AudioProcessorBase):
    
    def __init__(self) -> None:
        self.sample_rate = 48000
        self.buffer = deque()
        self.last_push = time.time()
        self.segment_sec = settings.REAL_TIME_SEGMENT_SEC
        import streamlit as st
        st.session_state.setdefault("stt_queue", queue.Queue())

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray(format="flt")
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=0)
        pcm = np.clip(pcm, -1.0, 1.0).astype("float32")
        self.buffer.append(pcm)

        now = time.time()
        if (now - self.last_push) >= self.segment_sec:
            self.last_push = now
            if self.buffer:
                chunk = np.concatenate(list(self.buffer))
                self.buffer.clear()
                import streamlit as st
                st.session_state["stt_queue"].put((chunk, self.sample_rate))

        return frame
