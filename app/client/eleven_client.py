from functools import lru_cache
from elevenlabs.client import ElevenLabs
from app.config.settings import settings

@lru_cache(maxsize=1)
def get_eleven_client() -> ElevenLabs:
    if not settings.ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVEN_API_KEY is not set")
    return ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
