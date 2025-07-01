FROM python:3.12-slim

WORKDIR /retrainer_app

# Install system dependencies if any
# RUN apt-get update && apt-get install -y libpq-dev gcc

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY retrainer_app ./retrainer_app

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "retrainer_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
