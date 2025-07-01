#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Start the MLflow server in the background
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB} &

# Wait for the MLflow server to be ready
echo "Waiting for MLflow server to start..."
until curl -s http://localhost:5000/health > /dev/null; do
    sleep 1
done
echo "MLflow server is up and running."

# Run the model registration script
echo "Running initial model registration script..."
python /app/register_initial_model.py

wait
