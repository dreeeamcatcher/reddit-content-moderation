import logging
import sys
import praw
from typing import List
from datetime import datetime
from app.data_fetcher.repositories.reddit_post import RedditPostRepository
from app.data_fetcher.schemas.reddit_post import RedditPostCreate
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RedditService:
    def __init__(self, repository: RedditPostRepository, reddit_client: praw.Reddit):
        self.repository = repository
        self.reddit = reddit_client

    async def fetch_subreddit_posts(self, subreddit_name: str, limit: int = 10) -> List[RedditPostCreate]:
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_to_store = []
        
        for submission in subreddit.new(limit=limit):
            # Check if post already exists before fetching comments to save API calls
            if self.repository.get_post_by_id(submission.id):
                continue # Skip if post already exists

            submission.comments.replace_more(limit=None) # Fetch all comments
            comments_body = [comment.body for comment in submission.comments.list()]
            
            post_data = RedditPostCreate(
                post_id=submission.id,
                subreddit=subreddit_name,
                title=submission.title,
                text=submission.selftext,
                comments=comments_body, 
                created_utc=datetime.fromtimestamp(submission.created_utc),
                is_processed=False
            )
            
            # Storing the Pydantic model
            posts_to_store.append(post_data)

        # Batch create posts to reduce database calls
        # if posts_to_store:
        #     return self.repository.batch_create_posts(posts_to_store)
        
        # return []

        # Loop create posts
        created_db_posts = []
        for post_to_create in posts_to_store:
            created_db_post = self.repository.create_post(post_to_create)
            if created_db_post: # Ensure post was actually created
                 created_db_posts.append(created_db_post) # Append the DB model instance
        return created_db_posts # Return list of DB models

    async def fetch_predefined_subreddits_posts(self) -> List[RedditPostCreate]:
        """Fetches posts from a predefined list of subreddits specified in config."""
        logger.info(f"Fetching posts from predefined subreddits")
        all_fetched_posts = []
        for subreddit_name in settings.SUBREDDITS_TO_FETCH:
            fetched_posts = await self.fetch_subreddit_posts(subreddit_name, settings.POST_FETCH_LIMIT)
            all_fetched_posts.extend(fetched_posts)
        logger.info(f"Fetched {len(all_fetched_posts)} posts from predefined subreddits")
        return all_fetched_posts

    def get_all_posts(self):
        return self.repository.get_all_posts()

    def get_unprocessed_posts(self):
        return self.repository.get_unprocessed_posts()

    def mark_post_as_processed(self, post_id: str):
        return self.repository.mark_post_as_processed(post_id)

    def get_filtered_posts(self, processed_status, start_date=None, end_date=None) -> List[RedditPostCreate]:
        logger.info(f"Fetching posts with processed status: {processed_status}, start date: {start_date}, end date: {end_date}")
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        return self.repository.get_filtered_posts(processed_status, start_date=start_date_obj, end_date=end_date_obj)
