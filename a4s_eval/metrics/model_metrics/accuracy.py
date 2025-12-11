from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Any

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


def _custom_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the accuracy: (correct predictions / total predictions)."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    # Ensure comparison is robust (e.g., handles different data types that might arise)
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)

    return correct_predictions / total_predictions


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    
    # 1. Prepare Data (Features X and Target Y)
    
    # Identify feature columns (X)
    feature_names = [f.name for f in datashape.features]
    X_df = dataset.data[feature_names] # Input features as a DataFrame

    # Identify target column (Y) and extract true labels
    target_name = datashape.target.name
    y_true = dataset.data[target_name].to_numpy() # Target for comparison
    
    # Limit dataset if it's too large (as per guideline)
    if len(X_df) > 10000:
        X_df = X_df.iloc[:10000]
        y_true = y_true[:10000]
        print(f"Dataset limited to first {len(X_df)} examples for performance testing.")
    
    X = X_df.to_numpy(dtype=np.float32)
    
    try:
        y_pred = functional_model.predict_class(X)
    except Exception as e:
        # Re-raise with informative error about the input type mismatch
        raise RuntimeError(f"Functional model prediction failed (Input type: {type(X)}): {e}") from e

    # Ensure predictions are a numpy array for consistent comparison
    if not isinstance(y_pred, np.ndarray):
         y_pred = np.array(y_pred)
    
    # 3. Calculate Accuracy
    accuracy_value = _custom_accuracy_score(y_true, y_pred)

    # 4. Return the result as a Measure
    current_time = datetime.now()
    return [
        Measure(
            name="accuracy",
            score=accuracy_value,
            unit="ratio",
            time=current_time,
            metadata={"sample_count": len(X)},
        )
    ]