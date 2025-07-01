from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging

from app.data_fetcher.api.data_fetcher_api import get_reddit_service
from app.data_fetcher.services.reddit_service import RedditService
from app.inference.services.inference_service import InferenceService
from app.inference.schemas.prediction import Prediction as PredictionSchema
from app.inference.repositories.prediction_repository import PredictionRepository
from app.core.db import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="app/inference/templates")

def get_inference_service(request: Request, db: Session = Depends(get_db), reddit_service: RedditService = Depends(get_reddit_service)) -> InferenceService:
    return InferenceService(db=db, request=request, reddit_service=reddit_service)

def get_prediction_repository(db: Session = Depends(get_db)) -> PredictionRepository:
    return PredictionRepository(db)

@router.post("/process-posts/", response_model=List[PredictionSchema])
async def process_posts_and_predict(
    service: InferenceService = Depends(get_inference_service)
):
    """
    Triggers the processing of unprocessed posts.
    """
    logger.info(f"Received request to process posts and predict")
    try:
        predictions = await service.process_unprocessed_posts()
        logger.info(f"Successfully processed posts. Returning {len(predictions)} predictions.")
        return predictions
    except Exception as e:
        logger.error(f"Error during post processing and prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{post_id}", response_model=List[PredictionSchema])
def get_predictions_for_post_route(
    post_id: str,
    service: InferenceService = Depends(get_inference_service)
):
    """
    Retrieves all predictions associated with a specific post_id.
    """
    logger.info(f"Received request to get predictions for post_id: {post_id}")
    try:
        predictions = service.get_predictions_for_post(post_id)
        if not predictions:
            pass # return an empty list if no predictions found
        return predictions
    except Exception as e:
        logger.error(f"Error retrieving predictions for post_id {post_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-predictions/", response_class=HTMLResponse)
async def view_predictions_ui(
    request: Request,
    label_filter: Optional[str] = Query(None),
    confidence_min: Optional[float] = Query(None),
    confidence_max: Optional[float] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    repo: PredictionRepository = Depends(get_prediction_repository)
):
    """
    UI endpoint to view predictions with filtering.
    """
    logger.info(
        f"UI request to view predictions with filters: label='{label_filter}', "
        f"confidence_min={confidence_min}, confidence_max={confidence_max}, "
        f"start_date={start_date}, end_date={end_date}"
    )
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        predictions = repo.get_predictions_with_filters(
            label=label_filter,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
            start_date=start_date_obj,
            end_date=end_date_obj
        )
        return templates.TemplateResponse(
            "predictions_view.html",
            {
                "request": request,
                "predictions": predictions,
                "label_filter": label_filter,
                "confidence_min": confidence_min,
                "confidence_max": confidence_max,
                "start_date": start_date,
                "end_date": end_date
            }
        )
    except Exception as e:
        logger.error(f"Error generating predictions view: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating predictions view: {str(e)}")

