from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

# Define the Base here so the model can inherit from it.
# This makes the Lambda package self-contained.
Base = declarative_base()

class RedditPost(Base):
    __tablename__ = "raw_posts"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, unique=True, index=True)
    subreddit = Column(String)
    title = Column(String)
    text = Column(Text)
    comments = Column(JSON)
    created_utc = Column(DateTime)
    is_processed = Column(Boolean, default=False, nullable=False)
