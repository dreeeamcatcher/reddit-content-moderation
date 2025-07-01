from datetime import datetime, timezone
from typing import List, Optional
import mlflow
import logging
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from retrainer_app.core.config import settings
from retrainer_app.retrainer.repositories.reddit_post import RedditPostRepository
from retrainer_app.retrainer.repositories.labelled_post_content_repository import LabelledPostContentRepository
from retrainer_app.retrainer.schemas.labelled_post_content import LabelledPostContent, LabelledPostContentCreate
from retrainer_app.retrainer.schemas.reddit_post import RedditPostCreate
from google import genai
from retrainer_app.retrainer.utils.reddit_post_dataset import RedditPostDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RetrainerService:
    def __init__(
        self, 
        fetcher_repository: RedditPostRepository, 
        labelled_post_content_repository: LabelledPostContentRepository
    ):
        self.fetcher_repository = fetcher_repository
        self.labelled_post_content_repository = labelled_post_content_repository
        self.llm_client = genai.Client(api_key=settings.LLM_API_KEY)


    def get_current_date_original_posts(self):
        today = datetime.now().date()
        return self.fetcher_repository.get_posts_for_n_days(start_date=today, n_days=1)

    def get_current_date_labelled_posts(self):
        today = datetime.now().date()
        return self.labelled_post_content_repository.get_labelled_posts_for_n_days(start_date=today, n_days=1)

    def get_labelled_posts(self, start_date: Optional[datetime], end_date: Optional[datetime]) -> List[LabelledPostContent]:
        return self.labelled_post_content_repository.get_labelled_posts_by_date_range(start_date, end_date)

    def get_all_labelled_posts(self) -> List[LabelledPostContent]:
        return self.labelled_post_content_repository.get_all()

    def label_posts(self, posts: List[RedditPostCreate]) -> List[LabelledPostContent]:
        labelled_contents = []
        # Get existing labelled content IDs to avoid re-labelling
        existing_labelled_posts = self.labelled_post_content_repository.get_all()
        existing_post_ids = {lp.post_id for lp in existing_labelled_posts if lp.text_type == 'post'}
        existing_comment_ids = {lp.comment_id for lp in existing_labelled_posts if lp.text_type == 'comment' and lp.comment_id}
        logger.info(f"Found {len(existing_post_ids) + len(existing_comment_ids)} existing labelled posts and comments.")

        for post in posts:
            # Label post text if not already labelled
            if post.post_id not in existing_post_ids:
                post_label = self.call_llm(post.text)
                labelled_post_to_create = LabelledPostContentCreate(
                    post_id=post.post_id,
                    text=post.text,
                    label=post_label,
                    text_type='post',
                    created_utc=datetime.now(timezone.utc)
                )
                created_post = self.labelled_post_content_repository.create(labelled_post_to_create)
                labelled_contents.append(created_post)

            # Label comments if not already labelled
            for i, comment_text in enumerate(post.comments):
                comment_id_for_db = f"{post.post_id}_comment_{i}"
                if comment_id_for_db not in existing_comment_ids:
                    comment_label = self.call_llm(comment_text)
                    labelled_comment_to_create = LabelledPostContentCreate(
                        post_id=post.post_id,
                        comment_id=comment_id_for_db,
                        text=comment_text,
                        label=comment_label,
                        text_type='comment',
                        created_utc=datetime.now(timezone.utc)
                    )
                    created_comment = self.labelled_post_content_repository.create(labelled_comment_to_create)
                    labelled_contents.append(created_comment)
        logger.info(f"Labelled {len(labelled_contents)} posts and comments.")
        return labelled_contents


    def call_llm(self, text: str):
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=f"Classify the following text as either hate/offensive speech or neutral. If the text contains hate or offensive speech, respond with 1. If the text is neutral, respond with 0. Only reply with a single number (1 or 0). Post: {text}"
            )
            response_text = response.text.strip()
            if response_text in ['0', '1']:
                return int(response_text)
            else:
                return 0 # Default to neutral if response is invalid
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return 0 # Default to neutral on error


    def predict(self, model, dataset):
        trainer = Trainer(model=model)
        predictions = trainer.predict(dataset)
        return np.argmax(predictions.predictions, axis=-1)


    def retrain_and_evaluate(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # 1. Fetch newly labeled data
        today = datetime.now().date()
        labelled_contents = self.labelled_post_content_repository.get_labelled_posts_for_n_days(start_date=today, n_days=1)
        if len(labelled_contents) < 100: # Minimum data for training
            logger.info("Not enough new data to retrain.")
            return

        texts = [content.text for content in labelled_contents]
        labels = [content.label for content in labelled_contents]

        # 2. Load current champion model and tokenizer using alias
        client = mlflow.tracking.MlflowClient()
        champion_model_uri = None
        champion_version = None
        
        try:
            champion_version = client.get_model_version_by_alias(name=settings.MLFLOW_MODEL_NAME, alias=settings.MLFLOW_CHAMPION_ALIAS)
            champion_model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{champion_version.version}"
            
            loaded_components = mlflow.transformers.load_model(champion_model_uri, return_type="components")
            tokenizer = loaded_components['tokenizer']
            model = loaded_components['model']
            
            logger.info(f"Found production model version {champion_version.version}. Proceeding with retraining.")
        except mlflow.exceptions.MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info("No production model found. Using initial model for the first training run.")
                initial_model_path = settings.MLFLOW_MODEL_LOCAL_ARTIFACTS
                try:
                    tokenizer = AutoTokenizer.from_pretrained(initial_model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(initial_model_path)
                except OSError:
                    logger.error(f"Error: Initial model not found at {initial_model_path}. Please ensure the initial model is available.")
                    return
            else:
                raise e

        # 3. Create train and test datasets
        logger.info("Creating train and test datasets...")
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        train_dataset = RedditPostDataset(X_train, y_train, tokenizer)
        test_dataset = RedditPostDataset(X_test, y_test, tokenizer)
        logger.info("Train and test datasets created successfully.")
        logger.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

        # 4. Fine-tune the model (create a challenger)
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=5,
            weight_decay=0.01,
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        logger.info("Starting model training...")

        trainer.train()
        
        logger.info("Model training completed successfully.")

        # 5. Save challenger model to MLflow and register a new version
        challenger_version = None
        with mlflow.start_run() as run:
            logger.info("Logging and registering challenger model...")
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                task='text-classification',
                registered_model_name=settings.MLFLOW_MODEL_NAME,
            )
            challenger_version = model_info.registered_model_version
            challenger_model_uri = model_info.model_uri
            logger.info(f"Challenger model registered as version {challenger_version} at {challenger_model_uri}")

        # If no champion, promote challenger directly to production
        if not champion_model_uri:
            logger.info("No champion model to compare against. Promoting new model to Production.")
            client.set_registered_model_alias(
                name=settings.MLFLOW_MODEL_NAME,
                alias=settings.MLFLOW_CHAMPION_ALIAS,
                version=challenger_version
            )
            return
        logger.info(f"Challenger model saved with version {challenger_version} at {challenger_model_uri}")

        # 6. Evaluate champion and challenger
        champion_components = mlflow.transformers.load_model(champion_model_uri, return_type="components")
        champion_model = champion_components['model']
        
        challenger_components = mlflow.transformers.load_model(challenger_model_uri, return_type="components")
        challenger_model = challenger_components['model']

        champion_preds = self.predict(champion_model, test_dataset)
        challenger_preds = self.predict(challenger_model, test_dataset)

        true_labels = [item['labels'].item() for item in test_dataset]

        champion_f1 = f1_score(true_labels, champion_preds)
        challenger_f1 = f1_score(true_labels, challenger_preds)

        logger.info(f"Champion F1 score: {champion_f1}")
        logger.info(f"Challenger F1 score: {challenger_f1}")

        # 7. Promote the winner
        if challenger_f1 > champion_f1:
            logger.info(f"Promoting challenger to Production. Old champion version: {champion_version.version}")
            # Set the new champion
            client.set_registered_model_alias(
                name=settings.MLFLOW_MODEL_NAME,
                alias=settings.MLFLOW_CHAMPION_ALIAS,
                version=challenger_version
            )
            # Archive the old champion
            client.set_registered_model_alias(
                name=settings.MLFLOW_MODEL_NAME,
                alias=f"archived_champion_v{champion_version.version}",
                version=champion_version.version
            )
        else:
            logger.info("Champion remains in Production.")
