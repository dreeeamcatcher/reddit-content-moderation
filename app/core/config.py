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

    # --- Reddit API Settings ---
    REDDIT_CLIENT_ID: str = ""
    REDDIT_CLIENT_SECRET: str = ""
    REDDIT_USER_AGENT: str = "my-reddit-app/0.1" 

    # --- MLflow Settings ---
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # --- Application Behavior Settings ---
    SUBREDDITS_TO_FETCH: List[str] = ["politics", "worldnews", "changemyview", 'unpopularopinion', 'Debate', 'TrueUnpopularOpinion', 'PoliticalDiscussion']
    POST_FETCH_LIMIT: int = 10

    # --- Inference Service Settings ---
    MLFLOW_MODEL_NAME: str = ""
    MLFLOW_CHAMPION_ALIAS: str = ""

    # --- Gemma3 API Settings ---
    LLM_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()

def get_reddit_client() -> praw.Reddit:
    if not all([settings.REDDIT_CLIENT_ID, settings.REDDIT_CLIENT_SECRET, settings.REDDIT_USER_AGENT]):
        raise ValueError("Reddit API credentials (client_id, client_secret, user_agent) are not fully configured.")
    return praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID,
        client_secret=settings.REDDIT_CLIENT_SECRET,
        user_agent=settings.REDDIT_USER_AGENT,
        check_for_async=False 
    )
