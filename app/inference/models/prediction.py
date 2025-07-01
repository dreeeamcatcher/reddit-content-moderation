from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.core.db import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, index=True, nullable=False) 
    comment_id = Column(String, index=True, nullable=True)
    
    text_type = Column(String, nullable=False)
    original_text = Column(Text, nullable=False)
    label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
