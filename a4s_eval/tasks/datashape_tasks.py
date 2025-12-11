import uuid

from a4s_eval.celery_app import celery_app
from a4s_eval.data_model.evaluation import DataShape, Feature, FeatureType
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_datashape_request,
    patch_datashape,
    patch_datashape_status,
)
from a4s_eval.utils.logging import get_logger

type_mapping = {
    "int64": FeatureType.INTEGER,
    "float64": FeatureType.FLOAT,
    # object type mostly comes from textual data
    "object": FeatureType.CATEGORICAL,
    "datetime64[ns]": FeatureType.DATE,
}


@celery_app.task
def auto_discover_datashape(datashape_pid: uuid.UUID) -> None:
    try:
        data = get_datashape_request(datashape_pid)
        dataset_pid = data["dataset_pid"]
        df = get_dataset_data(dataset_pid)

        date = None
        features = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            col_type = type_mapping[col_type]

            _feature = Feature(
                pid=uuid.uuid4(),
                name=col,
                feature_type=col_type,
                min_value=df[col].min(),
                max_value=df[col].max(),
            )

            # Manual setting categorical features min/max to 0
            # as Feature min max are float
            if col_type in [FeatureType.CATEGORICAL, FeatureType.DATE]:
                _feature.min_value = 0
                _feature.max_value = 0

            features.append(_feature)

        datashape = DataShape(features=features, date=date, target=None)

        get_logger().debug(datashape.model_dump_json())
        patch_datashape(dataset_pid, datashape)
        patch_datashape_status(datashape_pid, "auto")
    except Exception as e:
        get_logger().error(
            f"Error during auto-discovery of datashape {datashape_pid}: {e}"
        )
        patch_datashape_status(datashape_pid, "failed")
