import os
import json
import boto3
from google import genai
from repository import PostRepository
from schemas import LabelledPostContentCreate
from datetime import datetime, timezone

# Configuration
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize clients
s3_client = boto3.client('s3')
repository = PostRepository()

def get_label_from_gemini(text):
    """Calls the Gemini API to classify text as spoiler or not."""
    if not text or not text.strip():
        return 0 # Cannot process empty text

    # Configure the Gemini client on each invocation
    # This is a good practice for Lambda to avoid issues with shared resources
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Classify the following text as either hate/offensive speech or neutral.
    If the text contains hate or offensive speech, respond with 1.
    If the text is neutral, respond with 0.
    Only reply with a single number (1 or 0).
    Text: "{text}"
    Label:
    """
    try:
        response = client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=prompt,
            )
        label = response.text.strip()
        if label in ['0', '1']:
            return int(label)
        else:
            return 0 # Default to neutral if response is invalid
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return 0 # Default to neutral on error

def lambda_handler(event, context):
    """
    Fetches posts from the last 24 hours, labels them using Gemini,
    and saves the labeled data to the database and to S3.
    """
    print("Data labeling function execution started.")

    if not S3_BUCKET_NAME or not GEMINI_API_KEY:
        print("Error: Environment variables S3_BUCKET_NAME or GEMINI_API_KEY not set.")
        raise ValueError("S3_BUCKET_NAME and GEMINI_API_KEY must be set.")

    posts_db = repository.get_posts_last_24_hours()
    print(f"Found {len(posts_db)} posts in the last 24 hours to label.")

    if not posts_db:
        print("No new posts found in the last 24 hours to label.")
        return {'statusCode': 200, 'body': json.dumps({'message': 'No posts to process.'})}

    labelled_posts_for_s3 = []
    for post in posts_db:
        # Label post main text
        main_text = f"{post.title or ''} -- {post.text or ''}"
        main_text_label = get_label_from_gemini(main_text)
        labelled_main_text = LabelledPostContentCreate(
            post_id=post.post_id,
            text=main_text,
            label=main_text_label,
            text_type='post',
            created_utc=datetime.now(timezone.utc)
        )
        repository.save_labelled_post(labelled_main_text)
        labelled_posts_for_s3.append(labelled_main_text.model_dump(mode='json'))

        # Label post body
        if post.comments:
            for i, comment_text in enumerate(post.comments):
                text_label = get_label_from_gemini(comment_text)
                labelled_text = LabelledPostContentCreate(
                    post_id=post.post_id,
                    comment_id=f"{post.post_id}_comment_{i}",
                    text=comment_text,
                    label=text_label,
                    text_type='comment',
                    created_utc=datetime.now(timezone.utc)
                )
                repository.save_labelled_post(labelled_text)
                labelled_posts_for_s3.append(labelled_text.model_dump(mode='json'))

    print(f"Successfully labelled and saved {len(labelled_posts_for_s3)} pieces of content to the database.")

    # Save the labeled data to S3 for the training job
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M')
    file_name = f"labeled-data-{timestamp}.json"
    s3_key = f"retraining/labeled-data/{file_name}"

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key,
        Body=json.dumps(labelled_posts_for_s3).encode('utf-8')
    )

    print(f"Successfully saved labeled data to s3://{S3_BUCKET_NAME}/{s3_key}")
    print("Data labeling function execution finished.")

    # Pass the S3 path to the next step in the Step Function
    return {
        'statusCode': 200,
        'labeled_data_path': f"s3://{S3_BUCKET_NAME}/{s3_key}"
    }
