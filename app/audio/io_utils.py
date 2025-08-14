import io
import numpy as np
import soundfile as sf

def pcm_to_wav(pcm: np.ndarray, sample_rate: int = 48000) -> bytes:
    """float32 mono [-1,1] → WAV(PCM_16) bytes"""
    buf = io.BytesIO()
    sf.write(buf, pcm, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()