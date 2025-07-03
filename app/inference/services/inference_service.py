from datetime import datetime, timezone
import sys
from zoneinfo import ZoneInfo
import httpx
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging
import mlflow
from fastapi import Request

import torch
import torch.nn.functional as F

from app.data_fetcher.repositories.reddit_post import RedditPostRepository
from app.data_fetcher.services.reddit_service import RedditService
from app.data_fetcher.schemas.reddit_post import RedditPost as RedditPostSchema
from app.data_fetcher.api.data_fetcher_api import get_reddit_service
from app.inference.repositories.prediction_repository import PredictionRepository
from app.inference.schemas.prediction import PredictionCreate, Prediction as PredictionSchema
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, db: Session, request: Request, reddit_service: RedditService):
        self.db = db
        self.app_state = request.app.state
        self.unprocessed_post_repo = RedditPostRepository(db)
        self.prediction_repo = PredictionRepository(db)
        self.reddit_service = reddit_service
        self.kyiv_tz = ZoneInfo("Europe/Kyiv")
        self._load_model_components_from_state()

    def _load_model_components_from_state(self):
        """Loads or re-loads the model components from the app state."""
        self.classifier = self.app_state.model_components['model']
        self.tokenizer = self.app_state.model_components['tokenizer']
        self.model_version = self.app_state.model_components['version']
        logger.info(f"InferenceService loaded with model version: {self.model_version}")

    async def _check_and_update_model_if_needed(self):
        """Checks MLflow for a new champion model and updates the app state if necessary."""
        try:
            client = mlflow.MlflowClient()
            champion_alias = settings.MLFLOW_CHAMPION_ALIAS
            model_name = settings.MLFLOW_MODEL_NAME
            
            logger.info(f"Checking for new champion model for '{model_name}' with alias '{champion_alias}'")
            
            # Get the version associated with the 'champion' alias
            champion_version_obj = client.get_model_version_by_alias(model_name, champion_alias)
            champion_version_str = f"v{champion_version_obj.version}"

            if self.model_version != champion_version_str:
                logger.info(f"New champion model found! Current: {self.model_version}, New: {champion_version_str}. Updating...")
                
                # Load the new model from MLflow
                model_uri = f"models:/{model_name}@{champion_alias}"
                # mlflow.transformers.load_model returns a pipeline-like object
                new_model_pipeline = mlflow.transformers.load_model(model_uri)
                
                # Update the application state with the new model components
                self.app_state.model_components = {
                    "model": new_model_pipeline.model,
                    "tokenizer": new_model_pipeline.tokenizer,
                    "version": champion_version_str # Use the version string
                }
                
                # Reload the components in the current service instance
                self._load_model_components_from_state()
                logger.info(f"Successfully updated and loaded new model version {champion_version_str}")
            else:
                logger.info("Current model is up-to-date with the champion version.")

        except Exception as e:
            logger.error(f"Failed to check or update model from MLflow: {e}", exc_info=True)
            # Continue with the existing model if the check fails
            pass


    def _classify_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"label": "neutral", "confidence_score": 1.0, "error": "Empty input text"}
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            output = self.classifier(**inputs)
            logits = output.logits.detach()
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            label_map = {0: "neutral", 1: "hate_speech"}
            pred_idx = int(torch.argmax(logits, dim=1).item())
            return {
                "label": label_map.get(pred_idx, str(pred_idx)),
                "confidence_score": probs[pred_idx]
            }
        except Exception as e:
            logger.error(f"Error during text classification for text '{text[:100]}...': {e}", exc_info=True)
            return {"label": "error", "confidence_score": 0.0, "error": str(e)}


    async def process_unprocessed_posts(self) -> List[PredictionSchema]:
        logger.info(f"Starting to process unprocessed posts")
        
        # Check for model updates before processing
        await self._check_and_update_model_if_needed()
        
        unprocessed_posts: List[RedditPostSchema] = self.unprocessed_post_repo.get_unprocessed_posts()
        
        created_predictions_db: List[PredictionSchema] = []

        if not unprocessed_posts:
            logger.info("No unprocessed posts found to process with the current limit.")
            return created_predictions_db

        for post_schema in unprocessed_posts:
            logger.info(f"Processing post ID: {post_schema.post_id}")
            
            # Classify post text
            if post_schema.text:
                classification_result = self._classify_text(post_schema.text)
                if "error" not in classification_result:
                    prediction_data = PredictionCreate(
                        post_id=post_schema.post_id,
                        comment_id=None, # For the main post text
                        text_type='post',
                        original_text=post_schema.text,
                        label=classification_result['label'],
                        confidence_score=classification_result['confidence_score'],
                        model_version=self.model_version,
                        
                        prediction_timestamp=datetime.now(self.kyiv_tz)
                    )
                    db_prediction = self.prediction_repo.create_prediction(prediction_data)
                    created_predictions_db.append(PredictionSchema.model_validate(db_prediction))
            
            # Classify comments if they exist
            if post_schema.comments:
                for i, comment_text in enumerate(post_schema.comments): 
                    comment_id_for_db = f"{post_schema.post_id}_comment_{i}"

                    if comment_text: # Ensure comment is not empty
                        comment_classification_result = self._classify_text(comment_text)
                        if "error" not in comment_classification_result:
                            prediction_data = PredictionCreate(
                                post_id=post_schema.post_id,
                                comment_id=comment_id_for_db,
                                text_type='comment',
                                original_text=comment_text,
                                label=comment_classification_result['label'],
                                confidence_score=comment_classification_result['confidence_score'],
                                model_version=self.model_version,
                                prediction_timestamp=datetime.now(self.kyiv_tz)
                            )
                            db_prediction = self.prediction_repo.create_prediction(prediction_data)
                            created_predictions_db.append(PredictionSchema.model_validate(db_prediction))
            
            # Mark the original post as processed in the data_fetcher's database
            self.reddit_service.mark_post_as_processed(post_schema.post_id)

        logger.info(f"Finished processing batch. Created {len(created_predictions_db)} predictions.")
        return created_predictions_db

    def get_predictions_for_post(self, post_id: str) -> List[PredictionSchema]:
        db_predictions = self.prediction_repo.get_predictions_by_post_id(post_id)
        return [PredictionSchema.model_validate(p) for p in db_predictions]
    
    def get_filtered_predictions(self, label, confidence_min, confidence_max, start_date, end_date):
        logger.info(
        f"Fetching predictions with filters: label='{label}', "
        f"confidence_min={confidence_min}, confidence_max={confidence_max}, "
        f"start_date={start_date}, end_date={end_date}"
    )
        confidence_min = float(confidence_min) if confidence_min is not None else None
        confidence_max = float(confidence_max) if confidence_max is not None else None
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        return self.prediction_repo.get_filtered_predictions(label, confidence_min, confidence_max, start_date_obj, end_date_obj)
