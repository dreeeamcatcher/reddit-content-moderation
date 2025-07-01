FROM apache/airflow:slim-latest-python3.12
USER root
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

USER airflow
COPY docker/airflow/requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
