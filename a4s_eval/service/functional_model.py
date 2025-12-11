from dataclasses import dataclass
from typing import Any, Protocol

from a4s_eval.typing import Array, TextInput, TextOutput


class PredictClassFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictProbaFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictProbaGradFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictValueFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictValueGradFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class GenerateTextFn(Protocol):
    """Generate text from prompt input."""

    def __call__(self, text_input: TextInput, **kwargs: Any) -> TextOutput: ...


class GenerateLogitsFn(Protocol):
    """Return raw logits for next-token prediction."""

    def __call__(self, text_input: TextInput) -> Array: ...


@dataclass(frozen=True)
class TabularClassificationModel:
    predict_class: PredictClassFn
    predict_proba: PredictProbaFn | None
    predict_proba_grad: PredictProbaGradFn | None = None


@dataclass
class TabularRegressionModel:
    predict_value: PredictValueFn
    predict_value_grad: PredictProbaGradFn | None = None


@dataclass(frozen=True)
class TextGenerationModel:
    generate_text: GenerateTextFn
    generate_logits: GenerateLogitsFn | None = None
