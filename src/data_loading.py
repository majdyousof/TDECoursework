from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the TDE coursework dataset and clean the basic schema."""
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "ID"})

    if "CAR OWNERSHIP" in df.columns:
        df = df.rename(columns={"CAR OWNERSHIP": "CAR_OWNERSHIP"})

    return df
