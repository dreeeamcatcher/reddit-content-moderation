from datetime import datetime, timedelta
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
from app.data_fetcher.models.reddit_post import RedditPost
from app.data_fetcher.schemas.reddit_post import RedditPostCreate

class RedditPostRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_post(self, post: RedditPostCreate) -> RedditPost:
        db_post_data = post.model_dump() 
        db_post = RedditPost(**db_post_data)
        self.db.add(db_post)
        self.db.commit()
        self.db.refresh(db_post)
        return db_post

    def get_post_by_id(self, post_id: str) -> RedditPost | None:
        return self.db.query(RedditPost).filter(RedditPost.post_id == post_id).first()

    def get_all_posts(self) -> List[RedditPost]:
        return self.db.query(RedditPost).all()

    def get_unprocessed_posts(self) -> List[RedditPost]:
        return self.db.query(RedditPost).filter(RedditPost.is_processed == False).all()

    def batch_create_posts(self, posts: List[RedditPostCreate]) -> List[RedditPost]:
        db_posts = [RedditPost(**post.model_dump()) for post in posts]
        self.db.add_all(db_posts)
        self.db.commit()
        for db_post in db_posts:
            self.db.refresh(db_post)
        return db_posts

    def mark_post_as_processed(self, post_id: str) -> RedditPost | None:
        db_post = self.get_post_by_id(post_id)
        if db_post:
            db_post.is_processed = True
            self.db.commit()
            self.db.refresh(db_post)
        return db_post

    def get_posts_with_filter(self, processed_status: str = "all", start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[RedditPost]:
        query = self.db.query(RedditPost)
        if processed_status == "processed":
            query = query.filter(RedditPost.is_processed == True)
        elif processed_status == "unprocessed":
            query = query.filter(RedditPost.is_processed == False)
        
        if start_date:
            query = query.filter(RedditPost.created_utc >= start_date)
        if end_date:
            # Add 1 day to the end date to include the whole day
            query = query.filter(RedditPost.created_utc < end_date + timedelta(days=1))

        return query.order_by(RedditPost.created_utc.desc()).all()
    
    # for development purposes, to reset the processed status of all posts
    def mark_all_as_unprocessed(self) -> int:
        updated = self.db.query(RedditPost).filter(RedditPost.is_processed == True).update(
            {RedditPost.is_processed: False}, synchronize_session=False
        )
        self.db.commit()
        return updated
