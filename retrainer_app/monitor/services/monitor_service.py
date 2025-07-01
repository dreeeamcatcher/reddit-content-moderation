from sqlalchemy.orm import Session
from datetime import date
import logging

from retrainer_app.core.config import settings
from retrainer_app.monitor.repositories.prediction_repository import PredictionRepository
from retrainer_app.monitor.schemas.monitor import MonitorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitorService:
    def __init__(self, db: Session):
        self.db = db
        self.prediction_repository = PredictionRepository(db)

    def check_predictions_and_trigger_retraining(self) -> MonitorResponse:
        today = date.today()
        logger.info(f"Checking predictions for {today}")

        hate_speech_predictions = self.prediction_repository.get_hate_speech_predictions_for_n_days(start_date=today, n_days=1)

        total_hate_speech_predictions = len(hate_speech_predictions)
        if total_hate_speech_predictions == 0:
            return MonitorResponse(
                message="No hate speech predictions found for today.",
                low_confidence_count=0,
                total_hate_speech_predictions=0,
                low_confidence_percentage=0.0,
                retraining_triggered=False
            )

        low_confidence_count = sum(
            1 for p in hate_speech_predictions if p.confidence_score < settings.MONITOR_LOW_CONFIDENCE_THRESHOLD
        )

        low_confidence_percentage = (low_confidence_count / total_hate_speech_predictions)

        retraining_triggered = low_confidence_percentage > settings.MONITOR_TRIGGER_THRESHOLD

        if retraining_triggered:
            message = f"Retraining triggered: Low confidence predictions ({low_confidence_percentage:.2%}) exceeded threshold ({settings.MONITOR_TRIGGER_THRESHOLD:.2%})."
            logger.warning(message)
            # Retraining wiil be triggered through Airflow based on the 'retraining_triggered' flag
        else:
            message = "Monitoring check complete. Retraining not required."
            logger.info(message)

        return MonitorResponse(
            message=message,
            low_confidence_count=low_confidence_count,
            total_hate_speech_predictions=total_hate_speech_predictions,
            low_confidence_percentage=low_confidence_percentage,
            retraining_triggered=retraining_triggered
        )
