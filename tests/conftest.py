import numpy as np
import pandas as pd
import pytest

# Using the explicit direct import to resolve the registry loading issue:
import a4s_eval.metrics.textgen_metrics.char_level_perturbation_rate_metric 


DATE_FEATURE = "issue_d"
N_SAMPLES: int | None = 100


def sample(df: pd.DataFrame) -> pd.DataFrame:
    if N_SAMPLES:
        out: pd.DataFrame = df.iloc[:N_SAMPLES]
        return out

    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.to_datetime(df[DATE_FEATURE])
    i_train = np.where(
        (pd.to_datetime("2013-01-01") <= t) & (t <= pd.to_datetime("2015-12-31"))
    )[0]
    i_test = np.where(
        (pd.to_datetime("2016-01-01") <= t) & (t <= pd.to_datetime("2017-12-31"))
    )[0]
    out: tuple[pd.DataFrame, pd.DataFrame] = df.iloc[i_train], df.iloc[i_test]
    return out


# 🛠️ FIX: Use autouse=True with the direct import above to guarantee metric loading
@pytest.fixture(scope="session", autouse=True)
def load_all_metrics() -> None:
    pass

@pytest.fixture(scope="session")
def tab_class_dataset() -> pd.DataFrame:
    return pd.read_csv("./tests/data/lcld_v2.csv", low_memory=False)

@pytest.fixture(scope="session")
def tab_class_train_data(tab_class_dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(tab_class_dataset)[0])


@pytest.fixture(scope="session")
def tab_class_test_data(tab_class_dataset: pd.DataFrame) -> pd.DataFrame:
    return sample(get_splits(tab_class_dataset)[1])