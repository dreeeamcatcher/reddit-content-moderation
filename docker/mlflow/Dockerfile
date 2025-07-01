FROM ghcr.io/mlflow/mlflow:v3.1.1

# Install psycopg2 for PostgreSQL backend store and other dependencies
USER root
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY docker/mlflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model registration script
COPY scripts/register_initial_model.py /app/register_initial_model.py

# Copy the startup script and make it executable
COPY docker/mlflow/start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
