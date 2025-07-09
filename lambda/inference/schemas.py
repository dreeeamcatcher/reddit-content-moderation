from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionCreate(BaseModel):
    """
    Pydantic schema for creating a new Prediction record, matching the full original schema.
    """
    post_id: str
    comment_id: Optional[str] = None
    text_type: str
    original_text: str
    label: str
    confidence_score: float
    model_version: str
    prediction_timestamp: Optional[datetime] = None
