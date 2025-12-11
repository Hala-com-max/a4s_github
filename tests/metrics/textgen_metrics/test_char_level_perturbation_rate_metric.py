import numpy as np
import pandas as pd
import time
from Levenshtein import distance as levenshtein_distance

from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.service.functional_model import TextGenerationModel

from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import ModelWrapper
from textattack import Attacker, Attack, AttackArgs
from textattack.datasets import Dataset as TextAttackDataset
from textattack.attack_results import SuccessfulAttackResult
from textattack.transformations import (
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapNeighboringCharacterSwap,
    CompositeTransformation,
)
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification


class A4SClassificationWrapper(ModelWrapper):
    """Wrap A4S TextGenerationModel as zero-shot classification for TextAttack."""

    def __init__(self, functional_model: TextGenerationModel, label_map: dict):
        self.model = functional_model
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.label_index_map = {v: k for k, v in label_map.items()}

    def _create_prompt(self, text: str) -> str:
        allowed_labels = ", ".join(self.label_map.values())
        return f"""
SYSTEM INSTRUCTION: Classify the text into one of: {allowed_labels}.
Respond with ONLY the label.

[TEXT]
{text}
[/TEXT]

LABEL:
"""

    def __call__(self, text_list: list[str]) -> np.ndarray:
        outputs = []
        for text in text_list:
            prompt = self._create_prompt(text)
            scores = np.zeros(self.num_classes)
            try:
                raw = self.model.generate_text(prompt)
                predicted = raw.strip().upper()
                if predicted in self.label_index_map:
                    idx = self.label_index_map[predicted]
                    scores[:] = 0.01 / (self.num_classes - 1)
                    scores[idx] = 0.99
                else:
                    scores[:] = 1.0 / self.num_classes
            except:
                scores[:] = 1.0 / self.num_classes
            outputs.append(scores)
        return np.array(outputs)


@textgen_metric(name="StrongDeepWordBug")
def strong_deepwordbug_metric(
    datashape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TextGenerationModel,
) -> list[Measure]:
    if test_dataset.data is None:
        raise ValueError("Dataset is missing data for evaluation.")

    sample_size = min(100, len(test_dataset.data))
    sample_df = test_dataset.data[["question", "label"]].head(sample_size).copy()

    unique_labels = sorted(sample_df["label"].unique())
    label_map = {i: str(l).upper() for i, l in enumerate(unique_labels)}
    wrapper = A4SClassificationWrapper(functional_model, label_map)

    reverse_label = {v: k for k, v in label_map.items()}
    ta_data = TextAttackDataset([
        (row["question"], reverse_label[str(row["label"]).upper()])
        for _, row in sample_df.iterrows()
    ])

    goal = UntargetedClassification(wrapper)

    transformation = CompositeTransformation([
        WordSwapRandomCharacterDeletion(random_one=True),
        WordSwapRandomCharacterInsertion(random_one=True),
        WordSwapRandomCharacterSubstitution(random_one=True),
        WordSwapNeighboringCharacterSwap(random_one=True)
    ])

    constraints = [
        RepeatModification(),
        #StopwordModification(),
        LevenshteinEditDistance(max_edit_distance=1000)
    ]

    search = GreedyWordSwapWIR()
    attack = Attack(goal, constraints, transformation, search)

    attack_args = AttackArgs(
        num_examples=sample_size,
        disable_stdout=True,
        query_budget=50000,
    )

    attacker = Attacker(attack, ta_data, attack_args)

    start = time.time()
    results = attacker.attack_dataset()
    elapsed = time.time() - start

    perturb_rates = []
    success_count = 0
    for r in results:
        if isinstance(r, SuccessfulAttackResult):
            success_count += 1
            orig = r.original_text()
            pert = r.perturbed_text()
            dist = levenshtein_distance(orig, pert)
            orig_len = len(orig.replace(" ", ""))
            if orig_len > 0:
                perturb_rates.append(dist / orig_len)

    avg_perturb = sum(perturb_rates) / len(perturb_rates) if perturb_rates else 0.0
    success_rate = success_count / len(results)

    return [
        Measure(
            name="characterlevelperturbationrate",
            score=avg_perturb,
            unit="ratio",
            time=elapsed,
            meta={
                "attack": "DeepWordBug (CompositeTransform + HighBudget)",
                "success_rate": success_rate,
                "evaluated_examples": len(results)
            }
        )
    ]
