from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.db import engine, Base  
from app.data_fetcher.api.data_fetcher_api import router as data_fetcher_router
from app.inference.api.inference_api import router as inference_router
from app.core.config import settings
import mlflow
import mlflow.transformers


# Create database tables on startup
def create_tables():
    Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    print("Database tables created (if they didn't exist).")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    
    # Construct the model URI dynamically from the model name and alias
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}@{settings.MLFLOW_CHAMPION_ALIAS}"
    print(f"Loading model from URI: {model_uri}")

    # Load model components
    model_components = mlflow.transformers.load_model(
        model_uri=model_uri,
        return_type="components"
    )
    print("Model and tokenizer loaded from MLflow.")

    # Get model version
    client = mlflow.MlflowClient()
    champion_version_obj = client.get_model_version_by_alias(
        name=settings.MLFLOW_MODEL_NAME, 
        alias=settings.MLFLOW_CHAMPION_ALIAS
    )
    model_version_str = f"v{champion_version_obj.version}"
    print(f"Loaded model version: {model_version_str}")

    # Store all components in app state
    app.state.model_components = {
        "model": model_components["model"],
        "tokenizer": model_components["tokenizer"],
        "version": model_version_str
    }
    
    yield


app = FastAPI(title="Automated Reddit Content Moderation System", lifespan=lifespan)


app.include_router(data_fetcher_router, prefix="/fetcher", tags=["Data Fetcher"])
app.include_router(inference_router, prefix="/inference", tags=["Inference"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Automated Reddit Content Moderation System"}
