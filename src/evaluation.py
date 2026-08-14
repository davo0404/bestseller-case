import prediction as ps
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(ps.y_test, ps.predictions)
rmse = np.sqrt(mean_squared_error(ps.y_test, ps.predictions))

print(f"--- Model Results ---")
print(f"Train set size: {len(ps.X_train)} samples")
print(f"Test set size:  {len(ps.X_test)} samples")
print(f"Mean Absolute Error (MAE): {mae:.4f} units")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} units")

"""
--- Model Results ---
Train set size: 43 samples
Test set size:  198 samples
Mean Absolute Error (MAE): 0.8076 units
Root Mean Squared Error (RMSE): 1.5950 units

Understanding the Output
MAE = 0.81 units: On average, our model's weekly forecast for any individual SKU is off by less than 1 item.
RMSE = 1.59 units: The square root error is slightly higher, which accounts for occasional larger spikes in demand.

"""
