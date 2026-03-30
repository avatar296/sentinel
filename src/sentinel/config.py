from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel"
    MODEL_PATH: str = "models/fraud_model.joblib"
    FRAUD_THRESHOLD: float = 0.8
    REVIEW_THRESHOLD: float = 0.4
    RULES_THRESHOLD: float = 0.5
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    SCORING_MODE: str = "champion"
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
