from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

CATEGORICAL_COLUMNS = ["category", "brand", "color", "size"]
FEATURE_COLUMNS = [
    "category",
    "brand",
    "color",
    "size",
    "avg_price",
    "avg_markdown",
    "avg_rating",
    "week_of_year",
    "month",
    "lag_1_sales",
    "lag_2_sales",
    "rolling_4w_avg",
]


def train_forecast_model(model_df: pd.DataFrame) -> dict:
    """Train a gradient boosting regressor and return model + predictions."""
    for column in CATEGORICAL_COLUMNS:
        model_df[column] = model_df[column].astype("category")

    X = model_df[FEATURE_COLUMNS]
    y = model_df["weekly_sales"]

    split_date = model_df["week_start"].max() - pd.Timedelta(weeks=8)
    train_mask = model_df["week_start"] < split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = HistGradientBoostingRegressor(categorical_features=CATEGORICAL_COLUMNS, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
    }
