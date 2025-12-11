import uuid

import numpy as np
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metrics.data_metrics.drift_metric import (
    data_drift_metric,
)


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


def test_data_drift_metric_generates_metrics(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    """
    # This function tests the data drift metric to ensure it generates some metrics.
    """

    metrics = data_drift_metric(data_shape, ref_dataset, test_dataset)
    assert len(metrics) == len(ref_dataset.shape.features)


def test_data_drift_metric_metrics_not_nan(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    metrics = data_drift_metric(data_shape, ref_dataset, test_dataset)
    assert all(not np.isnan(metric.score) for metric in metrics)
