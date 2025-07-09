from datetime import datetime, timezone
import os
import json
import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import local modules that will be packaged with this Lambda
from repository import InferenceRepository
from schemas import PredictionCreate

def get_db_session():
    """Creates and returns a new database session."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set.")
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def get_latest_approved_model(sagemaker_client, model_package_group_name):
    """Finds the latest approved model package in a SageMaker Model Registry group."""
    try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if not response['ModelPackageSummaryList']:
            raise ValueError(f"No approved models found in group '{model_package_group_name}'.")
        
        latest_model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
        print(f"Found latest approved model package: {latest_model_package_arn}")
        
        model_package_desc = sagemaker_client.describe_model_package(ModelPackageName=latest_model_package_arn)
        
        return {
            'model_data_url': model_package_desc['InferenceSpecification']['Containers'][0]['ModelDataUrl'],
            'image_uri': model_package_desc['InferenceSpecification']['Containers'][0]['Image'],
            'model_version': model_package_desc.get('ModelPackageVersion', 'N/A')
        }
    except Exception as e:
        print(f"Error getting latest approved model: {e}")
        raise

def deploy_or_update_endpoint(sagemaker_client, endpoint_name, model_info, role_arn):
    """Deploys a new SageMaker endpoint or updates an existing one if the model version is different."""
    try:
        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_desc = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_desc['EndpointConfigName'])
        production_variant = endpoint_config_desc['ProductionVariants'][0]
        model_desc = sagemaker_client.describe_model(ModelName=production_variant['ModelName'])
        
        # A simple way to check if the model is the same is by comparing ModelDataUrl.
        # For a more robust check, you could use tags or model version numbers.
        if model_desc['PrimaryContainer']['ModelDataUrl'] == model_info['model_data_url']:
            print(f"Endpoint '{endpoint_name}' is already using the latest approved model. No update needed.")
            return
    except sagemaker_client.exceptions.ClientError:
        # Endpoint does not exist, so we will create it.
        pass

    model_name = f"{endpoint_name}-model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create SageMaker Model
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': model_info['image_uri'],
            'ModelDataUrl': model_info['model_data_url'],
        },
        ExecutionRoleArn=role_arn
    )
    
    # Create or Update Endpoint Configuration
    endpoint_config_name = f"{endpoint_name}-config-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium'
        }]
    )
    
    # Check if endpoint exists to decide whether to create or update
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating existing endpoint: {endpoint_name}")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    except sagemaker_client.exceptions.ClientError:
        print(f"Creating new endpoint: {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    # Wait for the endpoint to be in service
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint '{endpoint_name}' is in service.")


def invoke_sagemaker_endpoint(sagemaker_runtime, endpoint_name: str, text: str) -> Dict[str, Any]:
    """Invokes the SageMaker endpoint and returns the prediction result."""
    if not text or not text.strip():
        return None
        
    payload = json.dumps({"text": text})
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )
    return json.loads(response['Body'].read().decode())

def lambda_handler(event, context):
    """
    Main entry point for the Inference Lambda function.
    - Fetches unprocessed posts from the database.
    - Invokes the SageMaker endpoint for the post text and each comment.
    - Saves the predictions back to the database.
    """
    load_dotenv()
    print("Inference Lambda function invoked.")
    
    db_session = None
    try:
        # --- Get Configuration from Environment Variables ---
        sagemaker_endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME")
        model_package_group_name = os.getenv("MODEL_PACKAGE_GROUP_NAME")
        role_arn = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")

        if not sagemaker_endpoint_name or not model_package_group_name or not role_arn:
            raise ValueError("SAGEMAKER_ENDPOINT_NAME, MODEL_PACKAGE_GROUP_NAME, and SAGEMAKER_EXECUTION_ROLE_ARN must be set.")

        # --- Service and Repository Setup ---
        db_session = get_db_session()
        repo = InferenceRepository(db_session)
        sagemaker_client = boto3.client("sagemaker")
        sagemaker_runtime = boto3.client("sagemaker-runtime")

        # --- Get and Deploy Latest Model ---
        latest_model_info = get_latest_approved_model(sagemaker_client, model_package_group_name)
        deploy_or_update_endpoint(sagemaker_client, sagemaker_endpoint_name, latest_model_info, role_arn)
        model_version = latest_model_info['model_version']

        # --- Get Unprocessed Posts ---
        unprocessed_posts = repo.get_unprocessed_posts(limit=25)
        if not unprocessed_posts:
            print("No unprocessed posts found. Exiting.")
            return {'statusCode': 200, 'body': json.dumps('No unprocessed posts to process.')}
        
        print(f"Found {len(unprocessed_posts)} unprocessed posts to predict with model version {model_version}.")

        # --- Process Posts and Comments ---
        predictions_to_create = []
        for post in unprocessed_posts:
            main_text = f"{post.title or ''} -- {post.text or ''}"
            result = invoke_sagemaker_endpoint(sagemaker_runtime, sagemaker_endpoint_name, main_text)
            if result:
                predictions_to_create.append(PredictionCreate(
                    post_id=post.post_id,
                    text_type='post',
                    original_text=main_text,
                    label=result.get('label'),
                    confidence_score=result.get('confidence'),
                    model_version=model_version,
                    prediction_timestamp=datetime.now(timezone.utc)
                ))

            if post.comments:
                for i, comment_text in enumerate(post.comments):
                    result = invoke_sagemaker_endpoint(sagemaker_runtime, sagemaker_endpoint_name, comment_text)
                    if result:
                        predictions_to_create.append(PredictionCreate(
                            post_id=post.post_id,
                            comment_id=str(i),
                            text_type='comment',
                            original_text=comment_text,
                            label=result.get('label'),
                            confidence_score=result.get('confidence'),
                            model_version=model_version,
                            prediction_timestamp=datetime.now(timezone.utc)
                        ))

        # --- Save Predictions and Mark Posts as Processed ---
        if predictions_to_create:
            repo.create_predictions(predictions_to_create)
            print(f"Successfully created {len(predictions_to_create)} predictions.")

        processed_post_ids = [post.post_id for post in unprocessed_posts]
        repo.mark_posts_as_processed(processed_post_ids)
        print(f"Marked {len(processed_post_ids)} posts as processed.")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {len(unprocessed_posts)} posts, generating {len(predictions_to_create)} predictions.')
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'An error occurred: {str(e)}')
        }
    finally:
        if db_session:
            db_session.close()
            print("Database session closed.")
