from sqlalchemy.orm import Session
from retrainer_app.retrainer.models.labelled_post_content import LabelledPostContent
from retrainer_app.retrainer.schemas.labelled_post_content import LabelledPostContentCreate
from sqlalchemy import func
from datetime import date, datetime, time, timedelta
from typing import Optional


class LabelledPostContentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, labelled_post_content: LabelledPostContentCreate) -> LabelledPostContent:
        db_labelled_post_content = LabelledPostContent(**labelled_post_content.dict())
        self.db.add(db_labelled_post_content)
        self.db.commit()
        self.db.refresh(db_labelled_post_content)
        return db_labelled_post_content

    def get_all(self):
        return self.db.query(LabelledPostContent).all()
    
    def get_labelled_posts_for_n_days(self, start_date: date, n_days: int = 1):
        period_start = datetime.combine(start_date, time.min)
        period_end = datetime.combine(start_date, time.max) + timedelta(days=n_days - 1)
        return self.db.query(LabelledPostContent).filter(
            LabelledPostContent.created_utc >= period_start,
            LabelledPostContent.created_utc <= period_end
        ).all()

    def get_labelled_posts_by_date_range(self, start_date: Optional[datetime], end_date: Optional[datetime]):
        query = self.db.query(LabelledPostContent)
        if start_date:
            query = query.filter(LabelledPostContent.created_utc >= start_date)
        if end_date:
            query = query.filter(LabelledPostContent.created_utc < end_date + timedelta(days=1))
        return query.order_by(LabelledPostContent.created_utc.desc()).all()
