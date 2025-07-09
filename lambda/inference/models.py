from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Boolean, Float, func
from sqlalchemy.ext.declarative import declarative_base

# Define the Base here so all models can inherit from it.
Base = declarative_base()

class RedditPost(Base):
    """
    Represents a raw post fetched from Reddit, stored in the 'raw_posts' table.
    """
    __tablename__ = "raw_posts"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, unique=True, index=True)
    subreddit = Column(String)
    title = Column(String)
    text = Column(Text)
    comments = Column(JSON)
    created_utc = Column(DateTime)
    is_processed = Column(Boolean, default=False, nullable=False)

class Prediction(Base):
    """
    Represents a prediction made by the model, matching the original, full schema.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, index=True, nullable=False)
    comment_id = Column(String, index=True, nullable=True) # Using index as comment id
    
    text_type = Column(String, nullable=False) # 'post' or 'comment'
    original_text = Column(Text, nullable=False)
    label = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
