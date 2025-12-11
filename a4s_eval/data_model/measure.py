"""Data model for representing evaluation metrics and their associated metadata."""

import uuid
from datetime import datetime

from pydantic import BaseModel, field_serializer


class Measure(BaseModel):
    """Represents a single evaluation metric with its value and associated metadata.

    This class is used to store various types of metrics including model performance metrics,
    data drift metrics, and feature-specific metrics. Each metric is timestamped and can be
    associated with a model, feature, or dataset through their respective IDs.
    """

    name: str  # Name of the metric (e.g., 'accuracy', 'f1_score', 'drift')
    score: float  # Numerical value of the metric
    time: datetime  # Timestamp when the metric was computed

    feature_pid: uuid.UUID | None = None

    @field_serializer("time")
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()

    @field_serializer("feature_pid")
    def serialize_pid(self, pid: uuid.UUID | None) -> str | None:
        return str(pid) if pid is not None else None
