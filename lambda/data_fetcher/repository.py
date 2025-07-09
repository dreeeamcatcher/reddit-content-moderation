from sqlalchemy.orm import Session
from typing import List, Optional

# Models and schemas will be in the same directory for the Lambda package
from models import RedditPost
from schemas import RedditPostCreate

class RedditPostRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_post_by_id(self, post_id: str) -> Optional[RedditPost]:
        """
        Retrieves a post from the database by its unique Reddit post ID.
        """
        return self.db.query(RedditPost).filter(RedditPost.post_id == post_id).first()

    def batch_create_posts(self, posts: List[RedditPostCreate]) -> List[RedditPost]:
        """
        Creates multiple post records in the database from a list of Pydantic schemas.
        """
        db_posts = [RedditPost(**post.model_dump()) for post in posts]
        self.db.add_all(db_posts)
        self.db.commit()
        # After committing, the db_posts objects are expired. We need to refresh them
        # to get the database-generated values like the primary key.
        for db_post in db_posts:
            self.db.refresh(db_post)
        return db_posts
