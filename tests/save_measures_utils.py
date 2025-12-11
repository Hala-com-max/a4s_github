import pandas as pd
from a4s_eval.data_model.measure import Measure
import os

OUTPUT_FOLDER = "./tests/data/measures/"


def save_measures(name: str, measures: list[Measure]) -> None:
    df = pd.DataFrame([m.model_dump() for m in measures])
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df.to_csv(OUTPUT_FOLDER + name.lower().replace(' ', '_') + ".csv", index=False)
