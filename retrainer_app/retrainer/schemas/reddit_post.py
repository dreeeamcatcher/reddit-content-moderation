from pydantic import BaseModel
from datetime import datetime
from typing import List

class RedditPostBase(BaseModel):
    post_id: str
    subreddit: str
    title: str
    text: str
    comments: List[str] = []
    created_utc: datetime
    is_processed: bool = False

class RedditPostCreate(RedditPostBase):
    pass

class RedditPost(RedditPostBase):
    id: int
    is_processed: bool

    class Config:
        from_attributes = True
