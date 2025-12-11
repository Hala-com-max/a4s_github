import uuid
import uuid
import pandas as pd
import pytest

from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.data_model.evaluation import (
    DataShape,
    Dataset,
    Model,
    Feature,
    FeatureType,
)

from tests.save_measures_utils import save_measures


class DummyFunctionalModel:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self._df = df
        self._target = target_col

    def predict_class(self, X):
        # Return the true labels from the underlying dataframe (perfect predictor)
        return self._df[self._target].values[: len(X)]


def test_accuracy_metric_batches():
    # find the registered accuracy evaluator
    funcs = model_metric_registry.get_functions()
    assert "accuracy" in funcs, "accuracy metric is not registered"
    evaluator = funcs["accuracy"]

    # Load a small sample dataset and expand it to >10000 rows
    df = pd.read_csv("tests/data/lcld_v2.csv", low_memory=False)
    if df.empty:
        pytest.skip("no data available")

    repeat = max(1, (10000 // len(df)) + 1)
    big_df = pd.concat([df] * repeat, ignore_index=True)
    assert len(big_df) > 10000

    # Build minimal DataShape: features are all columns except the target
    target_col = "charged_off"
    feature_names = [c for c in big_df.columns if c != target_col]
    features = [
        Feature(
            pid=uuid.uuid4(),
            name=name,
            feature_type=FeatureType.FLOAT,
            min_value=0,
            max_value=1,
        )
        for name in feature_names
    ]

    data_shape = DataShape(
        features=features,
        target=Feature(
            pid=uuid.uuid4(),
            name=target_col,
            feature_type=FeatureType.CATEGORICAL,
            min_value=0,
            max_value=1,
        ),
    )

    dataset = Dataset(pid=uuid.uuid4(), shape=data_shape, data=big_df)
    model = Model(pid=uuid.uuid4(), model=None, dataset=dataset)

    functional_model = DummyFunctionalModel(big_df, target_col)

    measures = evaluator(data_shape, model, dataset, functional_model)
    save_measures("accuracy", measures)

    assert len(measures) > 0
