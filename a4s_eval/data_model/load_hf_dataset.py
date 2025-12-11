from datasets import load_dataset
import pandas as pd
import uuid
from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType


def load_hf_dataset(
    name: str = "ag_news",
    split: str = "test",
    text_column: str = "text",
    label_column: str = "label"
) -> Dataset:
    """
    Load a Hugging Face dataset and convert it to an A4S Dataset.
    """
    hf_data = load_dataset(name, split=split)

    # Create a DataFrame compatible with A4S metrics
    df = pd.DataFrame({
        "question": hf_data[text_column],
        "label": hf_data[label_column]
    })

    # Create a minimal DataShape for text-based evaluation
    text_feature = Feature(
        pid=uuid.uuid4(),
        name="question",
        feature_type=FeatureType.TEXT,
        min_value=0,
        max_value=0
    )
    target_feature = Feature(
        pid=uuid.uuid4(),
        name="label",
        feature_type=FeatureType.CATEGORICAL,
        min_value=0,
        max_value=0
    )

    shape = DataShape(features=[text_feature], target=target_feature)

    return Dataset(
        pid=uuid.uuid4(),
        shape=shape,
        data=df
    )
