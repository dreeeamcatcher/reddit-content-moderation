from pydantic import BaseModel

class MonitorResponse(BaseModel):
    message: str
    low_confidence_count: int
    total_hate_speech_predictions: int
    low_confidence_percentage: float
    retraining_triggered: bool
