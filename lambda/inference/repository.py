from sqlalchemy.orm import Session
from typing import List, Optional

# Models and schemas will be in the same directory for the Lambda package
from models import RedditPost, Prediction
from schemas import PredictionCreate

class InferenceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_unprocessed_posts(self, limit: int = 100) -> List[RedditPost]:
        """
        Gets a batch of posts that have not yet been processed.
        """
        return self.db.query(RedditPost).filter(RedditPost.is_processed == False).limit(limit).all()

    def create_predictions(self, predictions: List[PredictionCreate]) -> List[Prediction]:
        """
        Creates multiple prediction records in the database in a single batch.
        """
        db_predictions = [Prediction(**p.model_dump()) for p in predictions]
        self.db.add_all(db_predictions)
        self.db.commit()
        for p in db_predictions:
            self.db.refresh(p)
        return db_predictions

    def mark_posts_as_processed(self, post_ids: List[str]) -> int:
        """
        Marks a list of posts as processed given their Reddit post IDs.
        """
        if not post_ids:
            return 0
            
        updated_count = self.db.query(RedditPost).filter(RedditPost.post_id.in_(post_ids)).update(
            {RedditPost.is_processed: True}, synchronize_session=False
        )
        self.db.commit()
        return updated_count
