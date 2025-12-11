from typing import Any
from a4s_eval.data_model.evaluation import ModelConfig, ModelFramework, ModelTask
from a4s_eval.service.ollama_models import load_ollama_text_model
from a4s_eval.service.torch_models import load_torch_classification


def load_model(model_config: ModelConfig) -> Any:
    if model_config.task == ModelTask.CLASSIFICATION:
        if model_config.framework == ModelFramework.TORCH:
            return load_torch_classification(model_config)

    if model_config.task == ModelTask.TEXT_GEN:
        if model_config.framework == ModelFramework.OLLAMA:
            return load_ollama_text_model(model_config)
    raise NotImplementedError
