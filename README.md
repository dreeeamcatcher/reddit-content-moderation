# Automated Reddit Content Moderation System

This project is a full-cycle MLOps implementation for an Automated Reddit Content Moderation System. It's designed as a university project to practice and demonstrate a complete MLOps pipeline, from data ingestion and model training to deployment, monitoring, and automated retraining.

## Project Pipeline

The system follows a continuous machine learning pipeline:

1.  **Data Fetcher Service**: Periodically queries the Reddit API for new posts in specified subreddits. This is orchestrated by an **Airflow DAG**.
2.  **Data Storage (Raw Posts)**: Stores raw post data (ID, text, timestamp) in a PostgreSQL database.
3.  **Inference Service**: Loads the current production ML model from the Model Registry (MLFlow). It fetches raw data, performs classification, and outputs predictions with confidence scores.
4.  **Log Storage (Predictions)**: Stores prediction results in the PostgreSQL database.
5.  **Monitoring Service**: Periodically queries the prediction logs to analyze prediction distributions and confidence levels, comparing them against predefined baselines or thresholds. This is orchestrated by an **Airflow DAG**.
6.  **Automated Retraining Trigger**: If model drift or performance degradation is detected, the Monitoring Service notifies the Retraining Orchestrator.
7.  **Retraining Orchestrator**:
    *   Fetches a recent batch of raw data.
    *   (Concept) Sends data to an LLM Labeling Service for annotation.
    *   Stores newly labeled data in the database.
    *   Triggers the Model Training Service with the new data.
8.  **Model Training Service**:
    *   Fine-tunes the current "champion" model from the MLFlow production stage.
    *   If no champion model is available, it downloads the initial base model for fine-tuning.
9.  **Model Registry**: Stores the newly trained model artifact in MLFlow, versioning it for production use.
10. **Model Deployment**: The Retraining Orchestrator updates the Inference Service to use the new model from the Model Registry.

## Architecture and Technologies

The system is built with a microservices architecture, containerized using Docker.

-   **Backend Services**: **FastAPI** is used to build all API services, following a 3-layer architecture (API/Routes, Service/Business Logic, Repository/Data Access).
-   **ML Platform**: **MLFlow** is used for the Model Registry and experiment tracking.
-   **Database**: **PostgreSQL** serves as the central data store for raw posts, predictions, and MLFlow metadata.
-   **Orchestration**: **Apache Airflow** is used for orchestrating the data fetching and monitoring/retraining pipelines.
-   **Containerization**: **Docker** and **Docker Compose** are used to build, run, and manage the services.

## Reddit API Integration

The Data Fetcher service uses the **PRAW (Python Reddit API Wrapper)** library to connect to the Reddit API and retrieve posts from specified subreddits.

To use this service, you must register your own "script" application on Reddit:

1.  Go to [https://www.reddit.com/prefs/apps/](https://www.reddit.com/prefs/apps/).
2.  Click "are you a developer? create an app...".
3.  Fill out the form, selecting "script" as the application type.
4.  Once created, you will get a **client ID** (under your app name) and a **client secret**.

These credentials must be added to your `.env` file as `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.

## Initial Model

The initial classification model is a fine-tuned version of `bert-lite`. It was trained on the `ucberkeley-dlab/measuring-hate-speech` dataset. The training process is documented in the `scripts/train-bert-lite.ipynb` notebook. The resulting model artifacts are stored in the `data/initial-model/` directory.

## Project Structure

```
.
├── app/                # Main FastAPI app (Data Fetcher, Inference)
├── retrainer_app/      # FastAPI app for Monitoring and Retraining
├── dags/               # Airflow DAGs for orchestration
├── data/               # Holds initial data and models
├── docker/             # Dockerfiles for different services
├── logs/               # Stores Airflow logs (mounted as a volume)
├── mlartifacts/        # MLFlow artifacts (mounted volume)
├── scripts/            # Utility scripts (e.g., initial model registration)
├── docker-compose.yml  # Main Docker Compose file
└── requirements.txt    # Python dependencies
```

**Note**: The `logs` and `mlartifacts` directories will be created automatically on your host machine when you run the services for the first time.

### Note on Bind Mounts

This project uses Docker's `develop` mode with `watch` actions (a modern replacement for bind mounts) in the `docker-compose.yml` file. This is configured for live-reloading during development (e.g., syncing `./app` to `/app/app` in the container). While this is set up, the core logic of the MLOps pipeline (like model exchange via MLFlow) does not depend on these development-time mounts for its operation.

## How to Run

### Prerequisites

-   Docker
-   Docker Compose

### 1. Environment Setup

Create a `.env` file in the root directory by copying the example:

```bash
cp .env.example .env
```

Fill in the `.env` file with your credentials and settings. The following variables are required by `docker-compose.yml`:

```
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=reddit_db
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
```

### 2. Build and Run the System

Run the following command to build and start all services in detached mode:

```bash
docker-compose up --build -d
```

### 3. Initial Model Registration

The initial model is automatically registered in the MLFlow Model Registry when the MLFlow service starts up. The `docker/mlflow/start.sh` script runs the `scripts/register_initial_model.py` script after the MLFlow server is running. This ensures that the inference service has a model to use from the very beginning.

### 4. Accessing Services

Once the containers are running, you can access the different parts of the system:

-   **Main Application (Data Fetcher & Inference)**: `http://localhost:8000/docs`
-   **Retrainer Application (Monitoring & Retraining)**: `http://localhost:8001/docs`
-   **MLFlow UI**: `http://localhost:5001`
-   **Airflow UI**: `http://localhost:8080`
-   **PostgreSQL Database**: Accessible on port `5432`
