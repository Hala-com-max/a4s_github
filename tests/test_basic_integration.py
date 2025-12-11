"""Basic integration tests for A4S Evaluation service.

These tests verify core functionality without requiring external services.
They are designed to run in CI environments reliably.
"""

import uuid
from datetime import datetime

import pytest

from a4s_eval.data_model.evaluation import Feature, FeatureType
from a4s_eval.data_model.measure import Measure


def test_celery_tasks_import():
    """Test that Celery tasks can be imported without errors."""
    try:
        from a4s_eval import celery_tasks

        # Just verify the module loads, don't try to connect to Redis
        assert hasattr(celery_tasks, "celery_app")
    except ImportError as e:
        pytest.fail(f"Failed to import Celery tasks: {e}")


def test_evaluation_data_model():
    """Test that evaluation data models work correctly."""

    # Test metric creation
    metric = Measure(name="test_metric", score=0.95, time=datetime.now())
    assert metric.name == "test_metric"
    assert metric.score == 0.95

    # Test feature creation
    feature = Feature(
        pid=uuid.uuid4(),
        name="test_feature",
        feature_type=FeatureType.FLOAT,
        min_value=0.0,
        max_value=1.0,
    )
    assert feature.name == "test_feature"
    assert feature.feature_type == FeatureType.FLOAT


def test_drift_metric_import():
    """Test that drift evaluation functions can be imported."""
    try:
        from a4s_eval.metrics.data_metrics.drift_metric import (
            categorical_drift_test,
            data_drift_metric,
            numerical_drift_test,
        )

        # Just verify they can be imported
        assert callable(data_drift_metric)
        assert callable(numerical_drift_test)
        assert callable(categorical_drift_test)
    except ImportError as e:
        pytest.fail(f"Failed to import drift evaluation functions: {e}")
