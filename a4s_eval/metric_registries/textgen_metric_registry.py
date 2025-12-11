from typing import Callable, Iterator, Protocol

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.functional_model import (
    TextGenerationModel,
)
from a4s_eval.utils.logging import get_logger


logger = get_logger()


class TextgenMetric(Protocol):
    def __call__(
        self,
        datashape: DataShape,
        model: Model,
        dataset: Dataset,
        functional_model: TextGenerationModel,
    ) -> list[Measure]:
        raise NotImplementedError


class TextgenMetricRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, TextgenMetric] = {}
        logger.debug("TextgenMetricRegistry initialized")

    def register(self, name: str, func: TextgenMetric) -> None:
        logger.debug(f"Registering metric evaluator: {name}")
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, TextgenMetric]]:
        logger.debug(f"Iterating over {len(self._functions)} registered evaluators")
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, TextgenMetric]:
        return self._functions


textgen_metric_registry = TextgenMetricRegistry()


def textgen_metric(name: str) -> Callable[[TextgenMetric], TextgenMetric]:
    """Decorator to register a function as a metric evaluator for A4S.
        name: The name to register the evaluator under.

    Returns:
        Callable[[TextgenMetric], TextgenMetric]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """
    logger.debug(f"Creating metric evaluator decorator for: {name}")

    def func_decorator(func: TextgenMetric) -> TextgenMetric:
        textgen_metric_registry.register(name, func)
        return func

    return func_decorator
