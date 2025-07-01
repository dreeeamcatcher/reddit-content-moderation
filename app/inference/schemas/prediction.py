from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictionBase(BaseModel):
    post_id: str
    comment_id: Optional[str] = None
    text_type: str
    original_text: str
    label: str
    confidence_score: float
    model_version: str
    prediction_timestamp: Optional[datetime] = None

class PredictionCreate(PredictionBase):
    pass

class Prediction(PredictionBase):
    id: int
    prediction_timestamp: datetime

    class Config:
        from_attributes = True
