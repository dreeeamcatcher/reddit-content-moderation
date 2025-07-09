import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta, timezone
from models import RedditPost, LabelledPostContent
from schemas import LabelledPostContentCreate

class PostRepository:
    def __init__(self):
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set.")
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_posts_last_24_hours(self):
        """
        Fetches all posts from the database that were created in the last 24 hours.
        """
        db = self.SessionLocal()
        try:
            time_24_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            
            posts = db.query(RedditPost).filter(RedditPost.created_utc >= time_24_hours_ago).all()
            return posts
        finally:
            db.close()

    def save_labelled_post(self, post: LabelledPostContentCreate):
        """
        Saves a single labeled post to the database.
        """
        db = self.SessionLocal()
        try:
            db_post = LabelledPostContent(**post.model_dump())
            db.add(db_post)
            db.commit()
            db.refresh(db_post)
            return db_post
        finally:
            db.close()
