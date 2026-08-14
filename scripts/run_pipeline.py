from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bestseller_forecast.data import load_sales_data, save_processed_data
from src.bestseller_forecast.features import build_feature_dataset
from src.bestseller_forecast.models import train_forecast_model
from src.bestseller_forecast.evaluation import print_evaluation_summary


def main() -> None:
    raw_df = load_sales_data()
    feature_df = build_feature_dataset(raw_df)
    save_processed_data(feature_df)

    model_info = train_forecast_model(feature_df)
    print_evaluation_summary(model_info)


if __name__ == "__main__":
    main()
