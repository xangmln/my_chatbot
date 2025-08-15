# app/audio/webrtc_processor.py
import time
import queue
import numpy as np
import av
from collections import deque
from streamlit_webrtc import AudioProcessorBase 
from app.config.settings import settings

RMS_GATE = 1e-4  # 이 값보다 작으면 무음 취급

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.sample_rate = 48000
        self.buffer = deque()
        self.last_push = time.time()
        self.segment_sec = settings.REAL_TIME_SEGMENT_SEC
        import streamlit as st
        st.session_state.setdefault("stt_queue", queue.Queue())
        st.session_state.setdefault("vu_meter", 0.0)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # 샘플레이트를 프레임에서 신뢰
        try:
            if frame.sample_rate:
                self.sample_rate = int(frame.sample_rate)
        except Exception:
            pass

        # float32 [-1,1]로 변환
        # planar/packed 모두 대응
        pcm = frame.to_ndarray()  # dtype/shape은 입력에 따라 다름
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

        # 채널 정규화 (C x N 또는 N x C 모두 커버)
        if pcm.ndim == 2:
            # (channels, samples) 또는 (samples, channels)
            if pcm.shape[0] in (1, 2) and pcm.shape[1] > 2:
                # (C, N) 가정
                pcm = pcm.mean(axis=0)
            elif pcm.shape[1] in (1, 2) and pcm.shape[0] > 2:
                # (N, C) 가정
                pcm = pcm.mean(axis=1)
            else:
                # 애매하면 1D로 평탄화 후 사용
                pcm = pcm.reshape(-1)
        else:
            pcm = pcm.reshape(-1)

        # 안전 클리핑
        pcm = np.clip(pcm, -1.0, 1.0)

        # 볼륨(RMS) 계산 → UI 표시용
        rms = float(np.sqrt(np.mean(pcm**2)) if pcm.size else 0.0)
        try:
            import streamlit as st
            st.session_state["vu_meter"] = rms
        except Exception:
            pass

        # 버퍼링
        self.buffer.append(pcm)

        # 세그먼트마다 큐로 전송 (무음은 제외)
        now = time.time()
        if (now - self.last_push) >= self.segment_sec:
            self.last_push = now
            if self.buffer:
                chunk = np.concatenate(list(self.buffer))
                self.buffer.clear()
                if float(np.sqrt(np.mean(chunk**2))) >= RMS_GATE:
                    import streamlit as st
                    st.session_state["stt_queue"].put((chunk, self.sample_rate))

        return frame

