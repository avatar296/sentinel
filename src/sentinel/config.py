from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel"
    MODEL_PATH: str = "models/fraud_model.joblib"
    FRAUD_THRESHOLD: float = 0.7
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
