import enum
import uuid
from typing import Any

import pandas as pd
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from pydantic import BaseModel, ConfigDict, field_serializer


class FeatureType(str, enum.Enum):
    """Enumeration of supported feature data types.

    Attributes:
        INTEGER: For integer numerical features
        FLOAT: For floating-point numerical features
        CATEGORICAL: For categorical/nominal features
        DATE: For date features
    """

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    DATE = "date"
    TEXT = "text"


class Feature(BaseModel):
    """Represents a single feature (column) in a dataset.

    This class defines the properties of individual features,
    including their data type and valid value ranges.

    Attributes:
        id (int): Primary key for the feature.
        name (str): Name of the feature.
        feature_type (FeatureType): Data type of the feature.
        min_value (Optional[float]): Minimum allowed value.
        max_value (Optional[float]): Maximum allowed value.
    """

    # Feature attributes
    pid: uuid.UUID
    name: str
    feature_type: FeatureType
    min_value: float | Any
    max_value: float | Any

    @field_serializer("feature_type")
    def serialize_feature_type(self, feature_type: FeatureType) -> str:
        return feature_type.value

    @field_serializer("pid")
    def serialize_pid(self, pid: uuid.UUID | None) -> str | None:
        return str(pid) if pid is not None else None


class DataShape(BaseModel):
    features: list[Feature]
    target: Feature | None = None
    date: Feature | None = None


class Dataset(BaseModel):
    pid: uuid.UUID
    shape: DataShape
    data: pd.DataFrame | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Model(BaseModel):
    pid: uuid.UUID
    model: InferenceSession | None = None

    dataset: Dataset

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Project(BaseModel):
    pid: uuid.UUID
    name: str
    frequency: str
    window_size: str


class Evaluation(BaseModel):
    pid: uuid.UUID
    dataset: Dataset
    model: Model
    project: Project


class ModelFramework(str, enum.Enum):
    ONNX = "onnx"
    TORCH = "torch"
    GGUF = "gguf"
    OLLAMA = "ollama"


class ModelTask(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TEXT_GEN = "text_generation"


class ModelConfig(BaseModel):
    framework: ModelFramework
    task: ModelTask
    path: str
