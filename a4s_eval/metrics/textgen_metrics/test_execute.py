import uuid

import pandas as pd
from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric_registry
from a4s_eval.metric_registries.textgen_metric_registry import TextgenMetric
from a4s_eval.service.functional_model import TextGenerationModel
from a4s_eval.service.model_factory import load_model
import pytest

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)

from tests.save_measures_utils import save_measures


@pytest.fixture
def textgen_dataset() -> pd.DataFrame:
    return pd.read_parquet("./tests/data/squad_val.parquet")

@pytest.fixture
def data_shape() -> DataShape:
    """Fixture defining the DataShape for a QA/TextGen task."""
    def text_feat(feature_name: str) -> Feature:
        return Feature(
            pid=uuid.uuid4(),
            name=feature_name,
            feature_type=FeatureType.TEXT,
            min_value=0,
            max_value=0,
        )

    # FEATURES: Context and Question are the inputs to the TextGen model.
    features = [text_feat(e) for e in ["context", "question"]]
    
    # TARGET: 'answers' is the ground truth/reference text for TextGen metrics.
    target = text_feat("answers") 
    
    # DATE: Assuming a date field is needed for compliance.
    date = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value=pd.to_datetime("2024-01-01"),
        max_value=pd.to_datetime("2024-12-31"),
    )

    data_shape = {"features": features, "target": target, "date": date}

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(textgen_dataset: pd.DataFrame, data_shape: DataShape) -> Dataset:
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=textgen_dataset)


@pytest.fixture
def ref_dataset(textgen_dataset, data_shape: DataShape) -> Dataset:
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=textgen_dataset,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def functional_model() -> TextGenerationModel:
    model_config = ModelConfig(
        path="llama3.2:1b",
        framework=ModelFramework.OLLAMA,
        task=ModelTask.TEXT_GEN,
    )

    model = load_model(model_config)
    if not isinstance(model, TextGenerationModel):
        raise TypeError(f"Model type error {type(model)} !=TextGenerationModel.")
    return model


def test_non_empty_registry():
    assert len(textgen_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", textgen_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, TextgenMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TextGenerationModel,
):
    measures = evaluator_function[1](
        data_shape, ref_model, test_dataset, functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0
