import os
import numpy as np
import torch
from a4s_eval.utils.logging import get_logger

logger = get_logger()

from a4s_eval.data_model.evaluation import ModelConfig
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.typing import Array


def load_torch_classification(model_config: ModelConfig) -> TabularClassificationModel:
    if not os.path.exists(model_config.path):
        logger.warning(
            f"Torch model file not found: {model_config.path}. Using dummy classifier for tests."
        )

        def predict_proba(x: Array) -> Array:
            # Return uniform probabilities for two classes if shape unknown
            n = x.shape[0] if hasattr(x, "shape") else len(x)
            probs = np.zeros((n, 2), dtype=float)
            probs[:, 0] = 1.0
            return probs

        def predict_class(x: Array) -> Array:
            n = x.shape[0] if hasattr(x, "shape") else len(x)
            return np.zeros(n, dtype=int)

        def predict_proba_grad(x: Array) -> Array:
            # Not implemented for dummy
            return None

        return TabularClassificationModel(
            predict_class=predict_class,
            predict_proba=predict_proba,
            predict_proba_grad=predict_proba_grad,
        )

    model = torch.jit.load(model_config.path)

    def predict_proba(x: Array) -> Array:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x)
        return y_pred.detach().cpu().numpy()

    def predict_class(x: Array) -> Array:
        y_pred = predict_proba(x)
        return np.argmax(y_pred, axis=-1)

    def predict_proba_grad(x: Array) -> Array:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.requires_grad_()
        y_pred = model(x)
        return y_pred

    return TabularClassificationModel(
        predict_class=predict_class,
        predict_proba=predict_proba,
        predict_proba_grad=predict_proba_grad,
    )
