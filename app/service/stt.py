from __future__ import annotations
from typing import Optional, Tuple, Iterable
import tempfile
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


# 크기: "tiny", "base", "small", "medium", "large-v3"
_MODEL: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(
            "base",
            device="auto",
            compute_type="int8",   # 품질↑ 원하면 "float16"/"int8_float16" 등 시도
        )
    return _MODEL

# ---- 48kHz float32 mono → 16kHz로 리샘플 (권장) ----
def _to_16k(pcm_f32_mono: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if sr == 16000:
        return pcm_f32_mono, 16000
    # 정확한 리샘플은 scipy의 resample_poly 사용
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(16000, sr)
    up = 16000 // g
    down = sr // g
    x = resample_poly(pcm_f32_mono, up, down).astype("float32")
    return x, 16000

def stt_from_pcm(
    pcm_f32_mono: np.ndarray,
    sample_rate: int,
    language_code: Optional[str] = "ko",
    vad_filter: bool = True,
) -> str:
    """
    실시간 세그먼트(2~4초)용: PCM(float32 mono) → 텍스트
    """
    model = _get_model()
    wav16, sr16 = _to_16k(pcm_f32_mono, sample_rate)

    # faster-whisper는 넘파이/파일 모두 가능하지만,
    # 간헐적 edge-case 방지를 위해 임시 wav 파일 경유 (안정적)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, wav16, sr16, format="WAV", subtype="PCM_16")
        segments, info = model.transcribe(
            tmp.name,
            language=language_code,      # "ko" 권장
            vad_filter=vad_filter,       # 간단 VAD (원하면 True로)
            beam_size=5,                 # 품질 튜닝 여지
            best_of=5,
            condition_on_previous_text=False,  # 세그먼트 독립 처리(실시간에 적합)
        )

        # segments 는 generator. 짧은 세그먼트라 1~2개일 가능성 큼
        text_parts: list[str] = []
        for seg in segments:
            if seg.text:
                text_parts.append(seg.text.strip())

        return " ".join(t for t in text_parts if t).strip()

def stt_from_file(
    file_path: str, language_code: Optional[str] = "ko"
) -> str:
    """
    업로드 파일(전체 파일) → 텍스트
    """
    model = _get_model()
    segments, info = model.transcribe(
        file_path,
        language=language_code,
        vad_filter=False,
        beam_size=5,
        best_of=5,
    )
    return " ".join((s.text or "").strip() for s in segments if s.text).strip()
