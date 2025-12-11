# ruff: noqa: F401
from a4s_eval.celery_app import celery_app
from a4s_eval.celery_tasks import poll_and_run_evaluation
from a4s_eval.tasks.data_metric_tasks import dataset_evaluation_task
from a4s_eval.tasks.datashape_tasks import auto_discover_datashape
from a4s_eval.tasks.prediction_metric_tasks import model_evaluation_task
from a4s_eval.utils.logging import get_logger

get_logger().info("Starting worker...")
