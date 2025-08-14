from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None

    MODEL_OPENAI: str = "gpt-3.5-turbo"
    MODEL_GEMINI: str = "gemini-2.0-flash"
    ELEVENLABS_VOICE_ID: str | None = None
    ELEVEN_STT_MODEL: str = "scribe_v1"
    ELEVEN_TTS_MODEL: str = "eleven_multilingual_v2"
    ELEVEN_OUTPUT_FORMAT: str = "mp3_44100_128"
    REAL_TIME_SEGMENT_SEC: float = 3.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
settings = Settings()
