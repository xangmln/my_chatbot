from elevenlabs.client import ElevenLabs
from app.config.settings import settings

client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
print(hasattr(client, "speech_to_text"))  # True면 실제 있음
print(client.speech_to_text)               # <elevenlabs.SpeechToText ...> 이런 식으로 나와야 정상
