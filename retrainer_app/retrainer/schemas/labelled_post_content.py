from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class LabelledPostContentCreate(BaseModel):
    post_id: str
    comment_id: Optional[str] = None
    text: str
    label: int
    text_type: str 
    created_utc: Optional[datetime] = None

class LabelledPostContent(LabelledPostContentCreate):
    id: int

    class Config:
        from_attributes  = True
