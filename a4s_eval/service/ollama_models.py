import ollama
from typing import Any
from a4s_eval.data_model.evaluation import ModelConfig
from a4s_eval.service.functional_model import TextGenerationModel
from a4s_eval.typing import TextInput, TextOutput


def load_ollama_text_model(model_config: ModelConfig) -> TextGenerationModel:
    """
    Load a text generation model backed by Ollama and return a TextGenerationModel.
    model_config.path is used as the Ollama model name (e.g. "gpt-4o" or local model name).
    """

    model_name = model_config.path

    def generate_text(text_input: TextInput, **kwargs: Any) -> TextOutput:
        """
        Generate text from a prompt or list of prompts using the ollama python API.
        This function relies solely on the ollama package (no CLI subprocess fallback).
        """
        if isinstance(text_input, list):
            text_input = " ".join(text_input)

        if not hasattr(ollama, "generate"):
            raise RuntimeError(
                "ollama package does not expose a 'generate' function. Install a compatible version."
            )

        try:
            resp = ollama.generate(model=model_name, prompt=text_input, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"ollama.generate failed: {exc}") from exc

        # Normalize common response shapes
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            for key in ("text", "output", "response", "content"):
                if key in resp:
                    return resp[key]
            return str(resp)
        if isinstance(resp, (list, tuple)):
            return " ".join(map(str, resp))
        # Some versions may return an object with attributes
        for attr in ("text", "content", "output", "response"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if isinstance(val, (list, tuple)):
                    return " ".join(map(str, val))
                return str(val)
        return str(resp)

    return TextGenerationModel(generate_text=generate_text)
