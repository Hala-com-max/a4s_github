import pandas as pd
from typing import Sequence, Any, List
import datetime
import time

from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric
from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure


def get_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates the character-level edit distance (ED_C)."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    rows, cols = len(s1) + 1, len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s1[row - 1] == s2[col - 1] else 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,
                dist[row][col - 1] + 1,
                dist[row - 1][col - 1] + cost,
            )

    return dist[row][col]


@prediction_metric(name="char_perturbation_rate")
def char_perturbation_rate(
    datashape: DataShape, ref_container: Any, reference: Dataset, evaluated: Dataset
) -> list[Measure]:
    """
    Calculates the Character-Level Perturbation Rate (ER_C) for adversarial attacks.
    """
    measures: list[Measure] = []

    if not datashape.input_columns:
        raise ValueError("DataShape must define at least one input column.")

    text_column = datashape.input_columns[0]

    X_ref = reference.data[text_column].astype(str)
    X_eval = evaluated.data[text_column].astype(str)

    total_edit_distance = 0
    total_original_chars = 0
    start_time = time.time()

    for original_text, perturbed_text in zip(X_ref, X_eval):
        edit_distance = get_levenshtein_distance(original_text, perturbed_text)
        total_edit_distance += edit_distance
        total_original_chars += len(original_text)
    end_time = time.time()  # <--- NEW LINE
    duration = end_time - start_time # <--- NEW LINE
    if total_original_chars > 0:
        perturbation_rate = total_edit_distance / total_original_chars
    else:
        perturbation_rate = 0.0

    measures.append(Measure(
        name="char_perturbation_rate",
        score=perturbation_rate,
        unit="ratio",
        time=duration,
        index=len(reference.data) 
    ))

    return measures