import processing as pr
import pandas as pd

# Set categorical columns as pandas category types
cat_cols = ['category', 'brand', 'color', 'size']
for col in cat_cols:
    pr.model_df[col] = pr.model_df[col].astype('category')

# Select features (X) and target (y)
feature_cols = [
    'category', 'brand', 'color', 'size', 
    'avg_price', 'avg_markdown', 'avg_rating', 
    'week_of_year', 'month', 'lag_1_sales', 'lag_2_sales', 'rolling_4w_avg'
]

X = pr.model_df[feature_cols]
y = pr.model_df['weekly_sales']

# Split date: hold out the last 8 weeks for evaluation
split_date = pr.model_df['week_start'].max() - pd.Timedelta(weeks=8)
train_mask = pr.model_df['week_start'] < split_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]


from sklearn.ensemble import HistGradientBoostingRegressor

# Initialize and train the model
model = HistGradientBoostingRegressor(categorical_features=cat_cols, random_state=42)
model.fit(X_train, y_train)

# Generate predictions on the unseen test set
predictions = model.predict(X_test)
