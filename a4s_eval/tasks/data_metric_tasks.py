import traceback
import uuid
from multiprocessing.util import get_logger

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric_registry
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    get_project_datashape,
    post_measures,
)
from a4s_eval.utils.dates import DateIterator


@celery_app.task
def dataset_evaluation_task(evaluation_pid: uuid.UUID) -> None:
    get_logger().info(f"Starting evaluation task for {evaluation_pid}.")

    # Check if any evaluators are registered
    evaluator_list = list(data_metric_registry)
    get_logger().info(f"Registered evaluators ({len(evaluator_list)}):")
    for name, _ in evaluator_list:
        get_logger().info(f"  - {name}")

    try:
        evaluation = get_evaluation(evaluation_pid)
        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        evaluation.model.dataset.data = get_dataset_data(evaluation.model.dataset.pid)

        metrics: list[Measure] = []

        x_test = evaluation.dataset.data

        iteration_count = 0
        datashape = get_project_datashape(evaluation.project.pid)

        try:
            if not datashape.date:
                raise ValueError(
                    "Datashape is missing a date feature, which is required for time-based evaluation."
                )
            date_iterator = DateIterator(
                date_round="1 D",
                window=evaluation.project.window_size,
                freq=evaluation.project.frequency,
                df=evaluation.dataset.data,
                date_feature=datashape.date.name,
            )

            for i, (date_val, x_curr) in enumerate(date_iterator):
                iteration_count += 1
                get_logger().info(
                    f"Iteration {i}, date: {date_val}, data shape: {x_curr.shape}"
                )
                evaluation.dataset.data = x_curr

                for name, evaluator in data_metric_registry:
                    get_logger().info(f"Running evaluator: {name}")
                    new_metrics = evaluator(
                        datashape, evaluation.model.dataset, evaluation.dataset
                    )
                    metrics.extend(new_metrics)

        except Exception as e:
            get_logger().error(f"Error in DateIterator: {e}")
            traceback.print_exc()

        get_logger().info(f"Total iterations: {iteration_count}")
        get_logger().info(f"Total metrics generated: {len(metrics)}")

        evaluation.dataset.data = x_test

        get_logger().debug(f"Posting {len(metrics)} metrics to API...")
        try:
            response = post_measures(evaluation_pid, metrics)
            get_logger().info(
                f"Metrics posted successfully, status: {response.status_code}."
            )
        except Exception as e:
            get_logger().error(f"Error posting metrics: {e}")
            raise

        get_logger().info(f"Tasked complete for {evaluation_pid}")

    except Exception as e:
        get_logger().error(f"Error in evaluation task: {e}")
        raise
