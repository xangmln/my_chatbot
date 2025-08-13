from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None

    MODEL_OPENAI: str = "gpt-3.5-turbo"
    MODEL_GEMINI: str = "gemini-2.0-flash"
    ELEVENLABS_VOICE_ID: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
