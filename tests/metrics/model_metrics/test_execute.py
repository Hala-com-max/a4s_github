import uuid
import numpy as np 
import pandas as pd
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model
import pytest
from pathlib import Path

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry, ModelMetric
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model

from tests.save_measures_utils import save_measures

# --- FIXTURES ---
@pytest.fixture
def data_shape() -> DataShape:
    """Fixture defining the DataShape for a QA/TextGen task."""
    # Correct CSV path
    metadata_path = Path(__file__).parents[2] / "data" / "lcld_v2_metadata_api.csv"
    
    # Read CSV
    metadata = pd.read_csv(metadata_path).to_dict(orient="records")

    # Assign unique IDs
    for record in metadata:
        record["pid"] = uuid.uuid4()

    # Extract target and date fields
    target_record = next(rec for rec in metadata if rec["name"] == "charged_off")
    date_record = next(rec for rec in metadata if rec["name"] == "issue_d")

    # Build features list
    features = [item for item in metadata if item["name"] not in ["charged_off", "issue_d"]]

    # Construct DataShape
    data_shape_dict = {
        "features": features,
        "target": target_record,
        "date": date_record,
    }

    return DataShape.model_validate(data_shape_dict)
@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data, data_shape: DataShape) -> Dataset:
    data = tab_class_train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )

@pytest.fixture
def functional_model() -> TabularClassificationModel:
    model_path = Path(__file__).parents[2] / "data" / "lcld_v2_tabtransformer.pt"

    model_config = ModelConfig(
        path=str(model_path),
        framework=ModelFramework.TORCH,
        task=ModelTask.CLASSIFICATION,
    )

    model = load_model(model_config)
    return model

def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TabularClassificationModel,
):
    measures = evaluator_function[1](
        data_shape, ref_model, test_dataset, functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0

@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_accuracy_metric_with_batching(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TabularClassificationModel,
):
    metric_name, metric_func = evaluator_function
    
    # 1. Only run this complex test for the 'accuracy' metric
    if metric_name != "accuracy":
        pytest.skip(f"Skipping batch test for metric: {metric_name}")

    # 2. Define Batch Parameters
    REQUESTED_BATCH_SIZE = 10000 
    full_df: pd.DataFrame = test_dataset.data
    total_samples = len(full_df)
    
    # --- FIX: Define the effective batch size for the test loop ---
    if total_samples > 1 and total_samples <= REQUESTED_BATCH_SIZE:
        # Use 500 to guarantee 2 batches for the 1000-row fixture
        effective_batch_size = 500
        print(f"Test running with effective batch size {effective_batch_size} due to small fixture data.")
    else:
        # Use the requested size for larger fixtures
        effective_batch_size = REQUESTED_BATCH_SIZE
        
    if total_samples <= 1:
         pytest.skip(f"Dataset size ({total_samples}) is too small to demonstrate batching.")

    all_measures = []
    
    # 3. Iterate through the dataset in batches using the effective size
    for start_index in range(0, total_samples, effective_batch_size):
        end_index = min(start_index + effective_batch_size, total_samples)
        
        # Create the batch DataFrame by slicing the full dataset
        batch_df = full_df.iloc[start_index:end_index]
        
        # 4. Override the dataset.data for the metric call
        batched_dataset = test_dataset.model_copy(deep=True)
        batched_dataset.data = batch_df 
        
        print(f"Processing accuracy batch: indices {start_index} to {end_index}")

        # 5. Call the metric function with the batched data
        measures = metric_func(
            data_shape, 
            ref_model, 
            batched_dataset, 
            functional_model
        )
        
        all_measures.extend(measures)
        
        # Save measures for each batch to ensure all are recorded in the CSV
        save_measures(evaluator_function[0], measures)

    # 6. Assertion to ensure multiple batches were run
    # Calculate expected batches based on the size actually used in the loop
    expected_batches = np.ceil(total_samples / effective_batch_size)
    
    assert len(all_measures) > 0, "No measures were returned."
    
    # If the total data is actually larger than 10,000, we expect multiple batches.
    if total_samples > REQUESTED_BATCH_SIZE:
        assert len(all_measures) > 1, "Expected more than one measure when data size > 10,000."
    
    # The final assertion must match the number of times the loop ran.
    assert len(all_measures) == int(expected_batches), f"Expected {int(expected_batches)} measures, got {len(all_measures)}."
    
    print(f"\nSuccessfully processed {len(all_measures)} accuracy measures from {int(expected_batches)} batches.")