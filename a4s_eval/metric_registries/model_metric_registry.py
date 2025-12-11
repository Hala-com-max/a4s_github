from typing import Callable, Iterator, Protocol

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.utils.logging import get_logger


logger = get_logger()


class ModelMetric(Protocol):
    def __call__(
        self,
        datashape: DataShape,
        model: Model,
        dataset: Dataset,
        functional_model: TabularClassificationModel,
    ) -> list[Measure]:
        raise NotImplementedError


class ModelMetricRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, ModelMetric] = {}
        logger.debug("ModelMetricRegistry initialized")

    def register(self, name: str, func: ModelMetric) -> None:
        logger.debug(f"Registering metric evaluator: {name}")
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, ModelMetric]]:
        logger.debug(f"Iterating over {len(self._functions)} registered evaluators")
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, ModelMetric]:
        return self._functions


model_metric_registry = ModelMetricRegistry()


def model_metric(name: str) -> Callable[[ModelMetric], ModelMetric]:
    """Decorator to register a function as a metric evaluator for A4S.
        name: The name to register the evaluator under.

    Returns:
        Callable[[ModelMetric], ModelMetric]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """
    logger.debug(f"Creating metric evaluator decorator for: {name}")

    def func_decorator(func: ModelMetric) -> ModelMetric:
        model_metric_registry.register(name, func)
        return func

    return func_decorator
