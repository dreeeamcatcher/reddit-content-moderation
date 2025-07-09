from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class RedditPost(BaseModel):
    """
    Pydantic schema for a RedditPost record.
    """
    id: int
    post_id: str
    subreddit: str
    title: str
    text: str
    comments: Optional[List[str]] = None
    created_utc: datetime
    is_processed: bool

    class Config:
        from_attributes = True

class LabelledPostContentCreate(BaseModel):
    """
    Pydantic schema for creating a new LabelledPostContent record.
    """
    post_id: str
    comment_id: Optional[str] = None
    text: str
    label: int
    text_type: str 
    created_utc: Optional[datetime] = None

class LabelledPostContent(LabelledPostContentCreate):
    """
    Pydantic schema for a LabelledPostContent record.
    """
    id: int

    class Config:
        from_attributes = True
