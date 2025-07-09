from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Prediction(BaseModel):
    """
    Pydantic schema for a Prediction record.
    """
    id: int
    post_id: str
    comment_id: Optional[str] = None
    text_type: str
    original_text: str
    label: str
    confidence_score: float
    model_version: str
    prediction_timestamp: datetime

    class Config:
        from_attributes = True
