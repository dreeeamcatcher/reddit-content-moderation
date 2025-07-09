import os
import json
import boto3
from repository import PredictionRepository
from schemas import Prediction as PredictionSchema

# Configuration
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
RETRAIN_TRIGGER_THRESHOLD = float(os.environ.get("RETRAIN_TRIGGER_THRESHOLD", "0.25"))
STEP_FUNCTION_ARN = os.environ.get("STEP_FUNCTION_ARN")

sfn_client = boto3.client('stepfunctions')
repository = PredictionRepository()

def lambda_handler(event, context):
    """
    The main entry point for the monitoring Lambda function.
    """
    print("Monitoring function execution started.")

    predictions_db = repository.get_predictions_last_24_hours()

    if not predictions_db:
        print("No predictions found in the last 24 hours.")
        return {'statusCode': 200, 'body': 'No predictions to process.'}

    predictions = [PredictionSchema.model_validate(p) for p in predictions_db]

    low_confidence_count = 0
    for pred in predictions:
        if pred.confidence_score < CONFIDENCE_THRESHOLD:
            low_confidence_count += 1
    
    total_predictions = len(predictions)
    low_confidence_proportion = low_confidence_count / total_predictions

    print(f"Total predictions: {total_predictions}")
    print(f"Low confidence predictions: {low_confidence_count}")
    print(f"Low confidence proportion: {low_confidence_proportion:.2f}")

    if low_confidence_proportion > RETRAIN_TRIGGER_THRESHOLD:
        print(f"Proportion {low_confidence_proportion:.2f} exceeds threshold {RETRAIN_TRIGGER_THRESHOLD}. Triggering retraining pipeline.")
        
        if not STEP_FUNCTION_ARN:
            print("Error: STEP_FUNCTION_ARN environment variable not set. Cannot trigger pipeline.")
            return {'statusCode': 500, 'body': 'STEP_FUNCTION_ARN not configured.'}

        sfn_client.start_execution(
            stateMachineArn=STEP_FUNCTION_ARN,
            input=json.dumps({"status": "Retraining triggered by monitoring"})
        )
        print("Successfully triggered Step Functions pipeline.")
    else:
        print(f"Proportion {low_confidence_proportion:.2f} is within the threshold. No action needed.")

    print("Monitoring function execution finished.")
    return {
        'statusCode': 200,
        'body': json.dumps({
            'total_predictions': total_predictions,
            'low_confidence_count': low_confidence_count,
            'low_confidence_proportion': low_confidence_proportion
        })
    }
