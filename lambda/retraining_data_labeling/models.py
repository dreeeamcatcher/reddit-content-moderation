from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Boolean, func
from sqlalchemy.ext.declarative import declarative_base

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

class LabelledPostContent(Base):
    """
    Represents a post that has been labeled by the LLM.
    """
    __tablename__ = "labelled_post_contents"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, index=True)
    comment_id = Column(String, index=True, nullable=True)
    text = Column(Text)
    label = Column(Integer)
    text_type = Column(String)
    created_utc = Column(DateTime(timezone=True), server_default=func.now())
