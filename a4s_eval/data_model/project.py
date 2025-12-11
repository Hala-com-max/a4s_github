"""Data model for representing projects in the A4S evaluation framework.

This module defines the Project class which serves as a container for configurations
and associations between datasets, models, and their evaluation settings.
"""

from pydantic import BaseModel

from a4s_eval.data_model.evaluation import Dataset


class Project(BaseModel):
    """Represents a machine learning project with its evaluation configuration.

    This class defines the settings for how a dataset should be evaluated over time,
    including the frequency of evaluations and the size of the time window to analyze.
    It also maintains a reference to the associated dataset.
    """

    name: str  # Name of the project
    frequency: str  # Frequency of evaluation (e.g., '1D' for daily)
    window_size: str  # Size of the rolling window for analysis (e.g., '7D' for 7 days)
    dataset: Dataset  # The dataset being analyzed in this project
