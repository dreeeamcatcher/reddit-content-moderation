from sqlalchemy import Column, Integer, String, Text, DateTime, func
from retrainer_app.core.db import Base

class LabelledPostContent(Base):
    __tablename__ = "labelled_post_contents"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, index=True)
    comment_id = Column(String, index=True, nullable=True)
    text = Column(Text)
    label = Column(Integer)
    text_type = Column(String)
    created_utc = Column(DateTime(timezone=True), server_default=func.now())
