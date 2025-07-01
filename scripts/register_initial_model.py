import mlflow
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "reddit-content-moderator")
if not MLFLOW_MODEL_NAME:
    raise ValueError("MLFLOW_MODEL_NAME environment variable not set.")

MLFLOW_CHAMPION_ALIAS = os.getenv("MLFLOW_CHAMPION_ALIAS", "champion")
if not MLFLOW_CHAMPION_ALIAS:
    raise ValueError("MLFLOW_CHAMPION_ALIAS environment variable not set.")

MLFLOW_MODEL_LOCAL_ARTIFACTS = os.getenv("MLFLOW_MODEL_LOCAL_ARTIFACTS")
if not MLFLOW_MODEL_LOCAL_ARTIFACTS:
    raise ValueError("MLFLOW_MODEL_LOCAL_ARTIFACTS environment variable not set.")

def register_model():
    if not os.path.exists(MLFLOW_MODEL_LOCAL_ARTIFACTS):
        print(f"Error: Model path {MLFLOW_MODEL_LOCAL_ARTIFACTS} does not exist.")
        return
    if not os.listdir(MLFLOW_MODEL_LOCAL_ARTIFACTS):
        print(f"Error: Model path {MLFLOW_MODEL_LOCAL_ARTIFACTS} is empty.")
        return

    try:
        print(f"Starting model registration for '{MLFLOW_MODEL_NAME}' from path '{MLFLOW_MODEL_LOCAL_ARTIFACTS}'")
        client = mlflow.MlflowClient()

        # Check if the registered model exists
        try:
            client.get_registered_model(MLFLOW_MODEL_NAME)
            print(f"Registered model '{MLFLOW_MODEL_NAME}' already exists.")
        except mlflow.exceptions.RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                print(f"Registered model '{MLFLOW_MODEL_NAME}' does not exist. Creating it.")
                client.create_registered_model(MLFLOW_MODEL_NAME)
            else:
                raise

        # Check if a model with the champion alias already exists
        try:
            champion_version = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS)
            if champion_version:
                print(f"Model '{MLFLOW_MODEL_NAME}' with alias '{MLFLOW_CHAMPION_ALIAS}' already registered.")
                print("Skipping registration.")
                return
        except mlflow.exceptions.RestException as e:
            # If the alias does not exist, MLflow might return either of these error codes
            if e.error_code in ["RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"]:
                print(f"No model with alias '{MLFLOW_CHAMPION_ALIAS}' found. Proceeding with registration.")
            else:
                raise # Re-raise other exceptions

        # Start an MLflow run to log the model
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")

            print(f"Loading Hugging Face model and tokenizer from {MLFLOW_MODEL_LOCAL_ARTIFACTS}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MLFLOW_MODEL_LOCAL_ARTIFACTS)
                hf_model = AutoModelForSequenceClassification.from_pretrained(MLFLOW_MODEL_LOCAL_ARTIFACTS)
                print("Hugging Face model and tokenizer loaded successfully.")
            except Exception as e:
                print(f"Error loading model/tokenizer from {MLFLOW_MODEL_LOCAL_ARTIFACTS}: {e}")
                raise

            print("Logging Hugging Face model with mlflow.transformers.log_model...")
            # Log the Hugging Face model and tokenizer directly
            model_info = mlflow.transformers.log_model(
                transformers_model={
                    "model": hf_model,
                    "tokenizer": tokenizer,
                },
                task="text-classification",
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME
            )
            print(f"Model artifacts from '{MLFLOW_MODEL_LOCAL_ARTIFACTS}' logged successfully using mlflow.transformers.log_model.")
            print(f"Model URI: {model_info.model_uri}")

            # Fetch the latest model version
            latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME)
            if latest_versions:
                registered_model_version = latest_versions[0].version
                print(f"Model registered as version: {registered_model_version}")
                # Set the "champion" alias for this initial model version
                client.set_registered_model_alias(
                    name=MLFLOW_MODEL_NAME,
                    alias=MLFLOW_CHAMPION_ALIAS,
                    version=registered_model_version
                )
                print(f"Alias '{MLFLOW_CHAMPION_ALIAS}' set for model '{MLFLOW_MODEL_NAME}' version '{registered_model_version}'.")
            else:
                print("Model was logged but not registered with a version. Cannot set alias.")


        print(f"Model '{MLFLOW_MODEL_NAME}' registered successfully with run ID '{run_id}'.")
        print(f"Check the MLflow UI at {MLFLOW_TRACKING_URI} for the registered model.")

    except Exception as e:
        print(f"An error occurred during model registration: {e}")

if __name__ == "__main__":
    register_model()
