import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


def robust_roc_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """Calculate ROC AUC score with handling for binary classification probabilities.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities (can be 2D for binary classification)

    Returns:
        np.ndarray: ROC AUC score
    """
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[
            :, 1
        ]  # Use probability of positive class for binary classification
    return roc_auc_score(y_true, y_pred_proba)


@prediction_metric(name="Empty model pred proba metric")
def empty_model_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    return []


@prediction_metric(name="Classification Performance metric: Accuracy")
def classification_accuracy_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Measure(
        name="Accuracy",
        score=accuracy_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@prediction_metric(name="Classification Performance metric: F1 Score")
def classification_f1_score_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Measure(
        name="F1",
        score=f1_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@prediction_metric(name="Classification Performance metric: Precision")
def classification_precision_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Measure(
        name="Precision",
        score=precision_score(y_true, y_pred, zero_division=0.0),
        time=date,
    )

    return [metric]


@prediction_metric(name="Classification Performance metric: Recall")
def classification_recall_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Measure(
        name="Recall",
        score=recall_score(y_true, y_pred),
        time=date,
    )

    return [metric]


@prediction_metric(
    name="Classification Performance metric: Matthews Correlation Coefficient"
)
def classification_matthews_corrcoef_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()
    y_pred = np.argmax(y_pred_proba, axis=1)

    metric = Measure(
        name="MCC",
        score=matthews_corrcoef(y_true, y_pred),
        time=date,
    )

    return [metric]


@prediction_metric(name="Classification Performance metric: RROCAUC")
def classification_roc_auc_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred_proba: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max()
    date = date.to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    metric = Measure(
        name="ROCAUC",
        score=robust_roc_auc_score(y_true, y_pred_proba),
        time=date,
    )

    return [metric]
