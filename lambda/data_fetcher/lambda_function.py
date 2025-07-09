import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Import from the local files within the Lambda package
from models import Base, RedditPost
from reddit_service import RedditService

def get_db_session():
    """
    Creates and returns a new database session.
    Reads the database URL from environment variables.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")
        
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def lambda_handler(event, context):
    """
    The main entry point for the AWS Lambda function.
    """
    load_dotenv() # Useful for local testing, does nothing in Lambda if no .env
    
    print("Data fetcher Lambda function invoked.")
    
    try:
        # --- Get Reddit Credentials from Environment Variables ---
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "Reddit Content Moderation Bot v1.0")

        if not all([reddit_client_id, reddit_client_secret]):
            raise ValueError("Reddit API credentials not found in environment variables.")

        # --- Database and Service Setup ---
        db_session = get_db_session()
        reddit_service = RedditService(
            db=db_session,
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # --- Fetch and Save Data ---
        # For a real-world scenario, you might pass the subreddit list in the event
        subreddits = ["politics", "worldnews", "changemyview", 'unpopularopinion', 'Debate', 'TrueUnpopularOpinion', 'PoliticalDiscussion']
        print(f"Fetching posts from subreddits: {subreddits}")
        
        new_posts_count = reddit_service.fetch_and_save_new_posts(subreddits=subreddits, limit=10)
        
        print(f"Successfully saved {new_posts_count} new posts.")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully fetched and saved {new_posts_count} new posts.')
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return an error response
        return {
            'statusCode': 500,
            'body': json.dumps(f'An error occurred: {str(e)}')
        }
    finally:
        if 'db_session' in locals() and db_session:
            db_session.close()
            print("Database session closed.")
