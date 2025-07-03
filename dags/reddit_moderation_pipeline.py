from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.operators.empty import EmptyOperator

def check_retraining_trigger(ti):
    retraining_data = ti.xcom_pull(task_ids='monitor_task')
    if retraining_data and retraining_data.get('retraining_triggered'):
        return 'label_posts_task'
    return 'stop_pipeline'

with DAG(
    dag_id="reddit_content_moderation_pipeline",
    start_date=pendulum.datetime(2025, 6, 29, tz="Europe/Kyiv"),
    catchup=False,
    schedule="0 */12 * * *",
    tags=["reddit", "moderation"],
) as dag:
    fetch_posts_task = HttpOperator(
        task_id="fetch_posts_task",
        http_conn_id="app",
        endpoint="/fetcher/fetch",
        method="POST",
        headers={"Content-Type": "application/json"},
        log_response=True
    )

    predict_task = HttpOperator(
        task_id="predict_task",
        http_conn_id="app",
        endpoint="/inference/process-posts",
        method="POST",
        log_response=True
    )

    monitor_task = HttpOperator(
        task_id="monitor_task",
        http_conn_id="retrainer",
        endpoint="/monitor/run",
        method="POST",
        log_response=True,
        response_filter=lambda response: response.json()
    )

    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=check_retraining_trigger,
    )

    label_posts_task = HttpOperator(
        task_id="label_posts_task",
        http_conn_id="retrainer",
        endpoint="/retrainer/label-posts",
        method="POST",
        log_response=True,
        extra_options={'timeout': 1800}, # Increased timeout for labeling posts
    )

    retrain_task = HttpOperator(
        task_id="retrain_task",
        http_conn_id="retrainer",
        endpoint="/retrainer/retrain-and-evaluate",
        method="POST",
        log_response=True,
        extra_options={'timeout': 1800}, # Increased timeout for retraining
    )

    stop_pipeline = EmptyOperator(task_id='stop_pipeline')

    fetch_posts_task >> predict_task >> monitor_task >> branch_task
    branch_task >> label_posts_task >> retrain_task
    branch_task >> stop_pipeline

