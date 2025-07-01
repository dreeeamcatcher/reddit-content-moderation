from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from app.inference.models.prediction import Prediction
from app.inference.schemas.prediction import PredictionCreate 

class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_prediction(self, prediction: PredictionCreate) -> Prediction:
        db_prediction = Prediction(**prediction.model_dump())
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction

    def get_predictions_by_post_id(self, post_id: str) -> List[Prediction]:
        return self.db.query(Prediction).filter(Prediction.post_id == post_id).all()

    def get_predictions_with_filters(
        self,
        label: Optional[str] = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Prediction]:
        query = self.db.query(Prediction)
        if label:
            query = query.filter(Prediction.label.ilike(f"%{label}%")) # Case-insensitive partial match, could be better with exact match from dictionary
        if confidence_min is not None:
            query = query.filter(Prediction.confidence_score >= confidence_min)
        if confidence_max is not None:
            query = query.filter(Prediction.confidence_score <= confidence_max)
        if start_date:
            query = query.filter(Prediction.prediction_timestamp >= start_date)
        if end_date:
            query = query.filter(Prediction.prediction_timestamp < end_date + timedelta(days=1))
        return query.order_by(Prediction.prediction_timestamp.desc()).all()
