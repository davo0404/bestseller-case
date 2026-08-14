from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "fashion_boutique_dataset.json"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "weekly_sku_features.json"


def load_sales_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw boutique sales data and normalize the key fields."""
    data_path = Path(path) if path is not None else RAW_DATA_PATH
    df = pd.read_json(data_path)

    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["size"] = df["size"].fillna("OS")
    df["sku_id"] = (
        df["category"].astype(str)
        + "_"
        + df["brand"].astype(str)
        + "_"
        + df["color"].astype(str)
        + "_"
        + df["size"].astype(str)
    )
    df["week_start"] = df["purchase_date"].dt.to_period("W").dt.start_time

    return df


def save_processed_data(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist processed weekly data to the processed-data folder."""
    output_path = Path(path) if path is not None else PROCESSED_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="index", indent=2)
    return output_path
