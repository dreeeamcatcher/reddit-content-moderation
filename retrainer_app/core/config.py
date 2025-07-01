from pydantic import PostgresDsn, field_validator, ValidationInfo
from pydantic_settings import BaseSettings
from typing import List, Optional, Any
import praw
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # --- Database Settings ---
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = ''
    DATABASE_URL: Optional[PostgresDsn] = None

    @field_validator("DATABASE_URL", mode='before')
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info: ValidationInfo) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=info.data.get("POSTGRES_USER"),
            password=info.data.get("POSTGRES_PASSWORD"),
            host=info.data.get("POSTGRES_HOST"),
            port=info.data.get("POSTGRES_PORT"),
            path=f"{info.data.get('POSTGRES_DB')}",
        )


    # --- MLflow Settings ---
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"


    # --- Inference Service Settings ---
    MLFLOW_MODEL_CHAMPION_URI: str = ""
    MLFLOW_MODEL_NAME: str = ""
    MLFLOW_CHAMPION_ALIAS: str = ""

    # --- Monitor Service Settings ---
    MONITOR_LOW_CONFIDENCE_THRESHOLD: float = 0.7
    MONITOR_TRIGGER_THRESHOLD: float = 0.1

    # --- Gemma3 API Settings ---
    LLM_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()
