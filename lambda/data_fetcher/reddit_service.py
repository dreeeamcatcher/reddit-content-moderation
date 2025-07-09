import logging
import sys
import praw
from typing import List
from datetime import datetime
from sqlalchemy.orm import Session

# We will copy the repository and schema into the same directory for the Lambda package
from repository import RedditPostRepository
from schemas import RedditPostCreate

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class RedditService:
    def __init__(self, db: Session, client_id: str, client_secret: str, user_agent: str):
        """
        Initializes the RedditService.
        - db: An SQLAlchemy Session object.
        - client_id, client_secret, user_agent: Reddit API credentials.
        """
        self.repository = RedditPostRepository(db)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def fetch_and_save_new_posts(self, subreddits: List[str], limit: int = 10) -> int:
        """
        Fetches new posts from a list of subreddits and saves them to the database.
        Returns the number of new posts saved.
        """
        total_posts_saved = 0
        for subreddit_name in subreddits:
            logger.info(f"Fetching posts from r/{subreddit_name}...")
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_to_store = []

            for submission in subreddit.new(limit=limit):
                # Check if post already exists to avoid duplicates
                if self.repository.get_post_by_id(submission.id):
                    continue

                # Fetch all comments
                submission.comments.replace_more(limit=0)  # A limit of 0 fetches all comments
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
                posts_to_store.append(post_data)

            if posts_to_store:
                created_posts = self.repository.batch_create_posts(posts_to_store)
                total_posts_saved += len(created_posts)
                logger.info(f"Saved {len(created_posts)} new posts from r/{subreddit_name}.")

        return total_posts_saved
