# app/services/tts_service.py
from typing import Optional
from app.client.eleven_client import get_eleven_client
from app.config.settings import settings

def tts_bytes(text: str,
              voice_id: Optional[str] = None,
              model_id: Optional[str] = None,
              output_format: Optional[str] = None) -> bytes:
    client = get_eleven_client()
    voice = voice_id or settings.ELEVENLABS_VOICE_ID
    model = model_id or settings.ELEVEN_TTS_MODEL
    fmt = output_format or settings.ELEVEN_OUTPUT_FORMAT
    stream = client.text_to_speech.convert(
        voice_id=voice,
        model_id=model,
        text=text,
        output_format=fmt,
    )
    return b"".join(stream)
