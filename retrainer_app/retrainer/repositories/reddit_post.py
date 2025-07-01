from datetime import datetime, timedelta
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List
from retrainer_app.retrainer.models.reddit_post import RedditPost
from retrainer_app.retrainer.schemas.reddit_post import RedditPostCreate

class RedditPostRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_posts(self) -> List[RedditPost]:
        return self.db.query(RedditPost).all()
    
    def get_posts_for_n_days(self, start_date: datetime, n_days: int = 1) -> List[RedditPost]:
        period_start = datetime.combine(start_date, datetime.min.time())
        period_end = datetime.combine(start_date, datetime.max.time()) + timedelta(days=n_days - 1)
        return self.db.query(RedditPost).filter(
            RedditPost.created_utc >= period_start,
            RedditPost.created_utc <= period_end
        ).all()
