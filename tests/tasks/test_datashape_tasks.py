import uuid
from unittest.mock import MagicMock, patch

import pandas as pd

from a4s_eval.data_model.evaluation import DataShape
from a4s_eval.tasks.datashape_tasks import auto_discover_datashape


def patch_datashape(datashape_pid: uuid, datashape: DataShape) -> None:
    for f in datashape.features:
        assert f.min_value <= f.max_value


def patch_datashape_status(datashape_pid: uuid.UUID, status: str) -> None:
    assert status == "auto"


def test_auto_discover_datashape() -> None:
    mock_datashape = MagicMock()
    mock_datashape.pid = uuid.uuid4()

    mock_data = pd.read_csv("./tests/data/lcld_v2.csv")

    with patch(
        "a4s_eval.tasks.datashape_tasks.get_datashape_request",
        return_value=mock_datashape,
    ), patch(
        "a4s_eval.tasks.datashape_tasks.get_dataset_data", return_value=mock_data
    ), patch(
        "a4s_eval.tasks.datashape_tasks.patch_datashape", new=patch_datashape
    ), patch(
        "a4s_eval.tasks.datashape_tasks.patch_datashape_status",
        new=patch_datashape_status,
    ):
        auto_discover_datashape(mock_datashape.pid)
