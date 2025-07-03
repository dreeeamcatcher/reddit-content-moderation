from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
from sqlalchemy.orm import Session
from app.data_fetcher.services.reddit_service import RedditService
from app.data_fetcher.schemas.reddit_post import RedditPost
from app.data_fetcher.repositories.reddit_post import RedditPostRepository
from app.core.db import get_db
from app.core.config import get_reddit_client
import praw
from app.core.config import settings
from datetime import datetime


router = APIRouter()

templates = Jinja2Templates(directory="app/data_fetcher/templates")


def get_reddit_service(db: Session = Depends(get_db), reddit_client: praw.Reddit = Depends(get_reddit_client)) -> RedditService:
    repository = RedditPostRepository(db)
    return RedditService(repository, reddit_client)


@router.post("/fetch/{subreddit}", response_model=List[RedditPost])
async def fetch_posts(
    subreddit: str,
    limit: int = settings.POST_FETCH_LIMIT,
    service: RedditService = Depends(get_reddit_service)
):
    try:
        posts = await service.fetch_subreddit_posts(subreddit, limit)
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch", response_model=List[RedditPost])
async def fetch_posts_from_predefined_subreddits(
    service: RedditService = Depends(get_reddit_service)
):
    """Fetches posts from the predefined list of subreddits."""
    try:
        posts = await service.fetch_predefined_subreddits_posts()
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/posts", response_model=List[RedditPost])
def get_posts(service: RedditService = Depends(get_reddit_service)):
    return service.get_all_posts()

@router.get("/posts/unprocessed", response_model=List[RedditPost])
def get_unprocessed_posts(service: RedditService = Depends(get_reddit_service)):
    return service.get_unprocessed_posts()

@router.put("/posts/{post_id}/mark-processed", response_model=RedditPost)
def mark_post_as_processed(post_id: str, service: RedditService = Depends(get_reddit_service)):
    try:
        post = service.mark_post_as_processed(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-posts/", response_class=HTMLResponse)
async def view_posts_ui(
    request: Request,
    processed_status: str = Query("all", enum=["all", "processed", "unprocessed"]),
    start_date: str = Query(None),
    end_date: str = Query(None),
    service: RedditService = Depends(get_reddit_service),
):
    '''
    UI endpoint to view posts with filtering.
    '''

    posts = service.get_filtered_posts(
        processed_status=processed_status, 
        start_date=start_date, 
        end_date=end_date
    )
    return templates.TemplateResponse(
        "posts_view.html",
        {
            "request": request, 
            "posts": posts, 
            "processed_status": processed_status,
            "start_date": start_date,
            "end_date": end_date
        }
    )
