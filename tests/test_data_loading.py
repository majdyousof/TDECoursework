from pathlib import Path

from src.data_loading import load_dataset


def test_dataset_loads_and_has_expected_columns():
    dataset_path = Path("data/raw/Dataset 1 2024.csv")
    df = load_dataset(dataset_path)

    expected_columns = {
        "HHID",
        "TT_WALK",
        "TT_PT",
        "TT_CAR",
        "INCOME",
        "GENDER",
        "AGE",
        "CAR_OWNERSHIP",
        "CHOSEN_MODE",
    }

    assert set(expected_columns).issubset(df.columns)
    assert len(df) > 0
