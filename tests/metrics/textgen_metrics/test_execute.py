import uuid
import pytest
from pathlib import Path
import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.data_model.load_hf_dataset import load_hf_dataset
from a4s_eval.service.functional_model import TextGenerationModel
from a4s_eval.service.model_factory import load_model
from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric_registry, TextgenMetric

from tests.save_measures_utils import save_measures


@pytest.fixture
def test_dataset() -> Dataset:
    # Load full Hugging Face dataset
    return load_hf_dataset(name="ag_news", split="test")


@pytest.fixture
def ref_dataset(test_dataset: Dataset) -> Dataset:
    return test_dataset  # Use same as reference for zero-shot


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> TextGenerationModel:
    # Example: reference model wrapper (no actual model)
    from a4s_eval.data_model.evaluation import Model
    return Model(pid=uuid.uuid4(), model=None, dataset=ref_dataset)


@pytest.fixture
def functional_model() -> TextGenerationModel:
    from a4s_eval.data_model.evaluation import ModelConfig, ModelFramework, ModelTask
    model_config = ModelConfig(
        path="llama3.2:1b",
        framework=ModelFramework.OLLAMA,
        task=ModelTask.TEXT_GEN,
    )

    model = load_model(model_config)
    if not isinstance(model, TextGenerationModel):
        raise TypeError(f"Model type error {type(model)} != TextGenerationModel.")
    return model


def test_non_empty_registry():
    assert len(textgen_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", textgen_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, TextgenMetric],
    test_dataset: Dataset,
    ref_dataset: Dataset,
    ref_model: TextGenerationModel,
    functional_model: TextGenerationModel,
):
    measures = evaluator_function[1](
        test_dataset.shape,
        ref_model,
        test_dataset,
        functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0
