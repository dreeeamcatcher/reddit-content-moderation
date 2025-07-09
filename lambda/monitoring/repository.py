import os
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, timezone
from models import Prediction

class PredictionRepository:
    def __init__(self):
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set.")
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_predictions_last_24_hours(self) -> List[Prediction]:
        """
        Fetches all predictions from the database that were created in the last 24 hours.
        """
        db = self.SessionLocal()
        try:
            time_24_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            
            predictions = db.query(Prediction).filter(Prediction.prediction_timestamp >= time_24_hours_ago).all()
            return predictions
        finally:
            db.close()
