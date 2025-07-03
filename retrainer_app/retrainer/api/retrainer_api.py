from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from retrainer_app.core.db import get_db
from retrainer_app.retrainer.repositories.reddit_post import RedditPostRepository
from retrainer_app.retrainer.repositories.labelled_post_content_repository import LabelledPostContentRepository
from retrainer_app.retrainer.schemas.labelled_post_content import LabelledPostContent
from retrainer_app.retrainer.services.retrainer_service import RetrainerService

router = APIRouter()
templates = Jinja2Templates(directory="retrainer_app/retrainer/templates")

def get_retrainer_service(db: Session = Depends(get_db)):
    fetcher_repo = RedditPostRepository(db)
    labelled_repo = LabelledPostContentRepository(db)
    return RetrainerService(fetcher_repo, labelled_repo)

@router.post("/label-posts", response_model=List[LabelledPostContent])
def label_today_posts(service: RetrainerService = Depends(get_retrainer_service)):
    posts = service.get_current_date_original_posts()
    if not posts:
        raise HTTPException(status_code=404, detail="No posts found for today.")
    labelled_posts = service.label_posts(posts)
    return labelled_posts

@router.get("/labelled-posts", response_class=HTMLResponse)
def view_labelled_posts(
    request: Request, 
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    service: RetrainerService = Depends(get_retrainer_service)
):

    posts = service.get_labelled_posts(start_date=start_date, end_date=end_date)
    return templates.TemplateResponse(
        "labelled_posts_view.html", 
        {
            "request": request, 
            "posts": posts,
            "start_date": start_date,
            "end_date": end_date
        }
    )

@router.post("/retrain-and-evaluate", response_model=dict)
def retrain_and_evaluate_model(service: RetrainerService = Depends(get_retrainer_service)):
    try:
        service.retrain_and_evaluate()
        return {"message": "Model retraining and evaluation completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
