from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE for forecast performance."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {"mae": mae, "rmse": rmse}


def print_evaluation_summary(model_info: dict) -> None:
    """Render a concise evaluation summary to stdout."""
    scores = evaluate_predictions(model_info["y_test"], model_info["predictions"])

    print("--- Model Results ---")
    print(f"Train set size: {len(model_info['X_train'])} samples")
    print(f"Test set size:  {len(model_info['X_test'])} samples")
    print(f"Mean Absolute Error (MAE): {scores['mae']:.4f} units")
    print(f"Root Mean Squared Error (RMSE): {scores['rmse']:.4f} units")
