import prediction as ps
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(ps.y_test, ps.predictions)
rmse = np.sqrt(mean_squared_error(ps.y_test, ps.predictions))

print(f"--- Model Results ---")
print(f"Split Date:  {ps.split_date}")
print(f"Max Week Start Date:  {ps.pr.model_df['week_start'].max()}") #Crazy, I know. Just quick debugging.
print(f"Train set size: {len(ps.X_train)} samples")
print(f"Test set size:  {len(ps.X_test)} samples")
print(f"Mean Absolute Error (MAE): {mae:.4f} units")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} units")

"""
Initial results for 4 weeks ahead including rating
--- Model Results ---
Train set size: 43 samples
Test set size:  198 samples
Mean Absolute Error (MAE): 0.8076 units
Root Mean Squared Error (RMSE): 1.5950 units


Results for 4 weeks ahead w/o rating. Results improved after not the change to not drop the rating even though NA.
Attributed to more samples.
--- Model Results ---
Split Date:  2025-07-07 00:00:00
Max Week Start Date:  2025-08-04 00:00:00
Train set size: 56 samples
Test set size:  221 samples
Mean Absolute Error (MAE): 0.7291 units
Root Mean Squared Error (RMSE): 1.5187 units

1 week ahead, but most of the data is in the last week.
Interestingly, the reuslts are worse even though the train data is more abundant.
Likely sensitive to spikes
--- Model Results ---
Split Date:  2025-07-28 00:00:00
Max Week Start Date:  2025-08-04 00:00:00
Train set size: 64 samples
Test set size:  213 samples
Mean Absolute Error (MAE): 0.7597 units
Root Mean Squared Error (RMSE): 1.5480 units
"""
