from pydantic import BaseModel
from datetime import datetime
from typing import List

class RedditPostCreate(BaseModel):
    """
    Pydantic schema for creating a new RedditPost record in the database.
    """
    post_id: str
    subreddit: str
    title: str
    text: str
    comments: List[str] = []
    created_utc: datetime
    is_processed: bool = False
