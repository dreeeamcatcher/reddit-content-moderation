from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import date, datetime, time, timedelta

from retrainer_app.monitor.models.prediction import Prediction

class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_predictions_for_n_days(self, start_date: date, n_days: int = 1) -> List[Prediction]:
        period_start = datetime.combine(start_date, time.min)
        period_end = datetime.combine(start_date, time.max) + timedelta(days=n_days - 1)
        
        return self.db.query(Prediction).filter(
            Prediction.prediction_timestamp >= period_start,
            Prediction.prediction_timestamp <= period_end
        ).all()
