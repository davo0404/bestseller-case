## Notes

**Hi all,**

I know the purpose of this case was not to give a fully detailed solution, but since I was studying about it I thought I might as well simulate the situation described in the case to gain some first hand experience.

To start with, I've found a data set on Kaggel that might resemble the described in the case. [More details about the data set](https://www.kaggle.com/datasets/pratyushpuri/retail-fashion-boutique-data-sales-analytics-2025)

Since I don't have a paid AI subscription on my private, I used Gemini to draft me the structure of the project.

Afterwards, I created this repo and its structure and started splitting, debugging and adjusting the draft produced by Gemini.

The structure of this repo is not professional, I just used it to explore and learn. Not meant for prod.

## Step-by-step walkthrough guide ##

### 1. src/processing.py

- The original copy of the data set is in /data/raw/. I preffer to use JSON over CSV because columnar data is usually faster, but with such a small data set, it didn't matter.
- At this stage the data is enhanced a bit by adding 'one size' to accessories.
- An SKU ID is created by collating columns to create a unique id (example: Dresses_Zara_Beige_S).
- The individual sales are aggregated to obtain weekly_sales per SKU.
- purchase_date is refined to week_start and week_of_year, to get the flow of time.
- Afterwards, introduced the lags.
- The proccessed data is in data/processed/

### 2. src/prediction.py

- Here, the feature and target are set as follows:

```python
feature_cols = [
    'category', 'brand', 'color', 'size', 
    'avg_price', 'avg_markdown', 'avg_rating', 
    'week_of_year', 'month', 'lag_1_sales', 'lag_2_sales', 'rolling_4w_avg'
]

X = pr.model_df[feature_cols]
y = pr.model_df['weekly_sales']
```
- Afterwards, the data is split in train and test sets. The last 4 weeks are for testing.
```python
split_date = pr.model_df['week_start'].max() - pd.Timedelta(weeks=4)
train_mask = pr.model_df['week_start'] < split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
```

- This is the model that Gemini recommended me. This bit is new to me so for now I can't judge if it is the right choice. I hope you will agree.
```Python
from sklearn.ensemble import HistGradientBoostingRegressor

# Initialize and train the model
model = HistGradientBoostingRegressor(categorical_features=cat_cols, random_state=42)
model.fit(X_train, y_train)

# Generate predictions on the unseen test set
predictions = model.predict(X_test)
```

### 3. src/evaluation.py

- Mean Absolute Error and Root Mean Squared Error. I read that rmse is very similar to mae, but it penalizes large forecasting mistakes much more heavily

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(ps.y_test, ps.predictions)
rmse = np.sqrt(mean_squared_error(ps.y_test, ps.predictions))
```

### 4. Results and adjustments

I made a couple of adjustments and got 3 different results.

#### First itteration
```python
Initial results for 4 weeks ahead including rating
--- Model Results ---
Train set size: 43 samples
Test set size:  198 samples
Mean Absolute Error (MAE): 0.8076 units
Root Mean Squared Error (RMSE): 1.5950 units
```
- Out of the box, here are the results.
- The test size seemed unreazonably larger than the train size.

#### Second itteration
```python
Attributed to more samples.
--- Model Results ---
Train set size: 56 samples
Test set size:  221 samples
Mean Absolute Error (MAE): 0.7291 units
Root Mean Squared Error (RMSE): 1.5187 units
```
- I've noticed that in the cleanup, rows that did not have a rating lag were removed. It seemed to me that the rating would not play a major role in such a short-horizon/small data set, so I removed the rating altogether from the evaluation and kept the rows that had ratings as NaN.
- This improved the model slightly and increased the train size.

#### Third itteration
 - At this point, I was confused by the large difference between train and test sets so I've done some digging.

 - Here is my first mistake. I made the wrong assumption that the data is evenly distributed, but one week's volume massively outweigh all the other week volumes combined.
```python
print(
    weekly_df.groupby('week_start')[['weekly_sales']]
    .count()
    .sort_values(by='weekly_sales', ascending=False)
)

# 2025-08-04          1167
# 2025-07-21            17
# 2025-04-21            15
```

- Another mistake, I wrongly assumed that the data set is from 1st of Jan to 31st of Dec, which led me to believe that the abundant week is in the middle of the year. Here are, in fact, the min and max:
```python
print("Min:", df['week_start'].min()) # 2024-08-05 00:00:00
print("Max:", df['week_start'].max()) # 2025-08-04 00:00:00
```

- In the third itteration, I've split the data to have only the last abundant week for testing which would still mean that the two data sets are very unbalanced, but it would increase the train data a bit, hopefully leading to better accuracy.

```python
--- Model Results ---
Split Date:  2025-07-28 00:00:00
Max Week Start Date:  2025-08-04 00:00:00
Train set size: 64 samples
Test set size:  213 samples
Mean Absolute Error (MAE): 0.7597 units
Root Mean Squared Error (RMSE): 1.5480 units
```

- Interestingly, the accuracy worsened! Evaluating on a broader 4-week window provides a more stable evaluation benchmark than evaluating on a narrow 1 week test horizon, where weekly demand spikes or stockout anomalies can artificially inflate test error.

### Conlcusion

It was very fun to play with this. I hope this will serve as a good foundation for conversation.

## Additional Notes

For token optimisation, I've used [RTK](https://github.com/rtk-ai/rtk).

Skills: Grilling by Matt Pocock

AI tools: Gemini and GitHub Copilot auto.

