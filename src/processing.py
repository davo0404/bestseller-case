import pandas as pd

# 1. Load data and convert dates
df = pd.read_json('data/raw/fashion_boutique_dataset.json')
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Fill missing sizes (e.g., accessories that don't have sizes)
df['size'] = df['size'].fillna('OS') # OS = One Size

# Define unique SKU ID
df['sku_id'] = df['category'] + "_" + df['brand'] + "_" + df['color'] + "_" + df['size']

# Get the start date of each week
df['week_start'] = df['purchase_date'].dt.to_period('W').dt.start_time

# Aggregate daily transactions -> Weekly SKU totals
weekly_df = df.groupby(['sku_id', 'category', 'brand', 'color', 'size', 'week_start']).agg(
    weekly_sales=('product_id', 'count'),
    avg_price=('current_price', 'mean'),
    avg_markdown=('markdown_percentage', 'mean'),
    avg_rating=('customer_rating', 'mean')
).reset_index()

# Sort chronologically by SKU and week
weekly_df = weekly_df.sort_values(['sku_id', 'week_start']).reset_index(drop=True)

# print(weekly_df.head())
#                             sku_id     category       brand  color size week_start  weekly_sales  avg_price  avg_markdown  avg_rating
# 0  Accessories_Ann Taylor_Beige_OS  Accessories  Ann Taylor  Beige   OS 2025-07-14             1      42.23      0.000000         NaN
# 1  Accessories_Ann Taylor_Beige_OS  Accessories  Ann Taylor  Beige   OS 2025-08-04             6      40.94     15.433333       2.175
# 2  Accessories_Ann Taylor_Black_OS  Accessories  Ann Taylor  Black   OS 2024-09-09             1      38.04      6.200000       3.700
# 3  Accessories_Ann Taylor_Black_OS  Accessories  Ann Taylor  Black   OS 2024-10-07             1      77.48      0.000000       3.300
# 4  Accessories_Ann Taylor_Black_OS  Accessories  Ann Taylor  Black   OS 2025-02-03             1      33.49     45.400000         NaN


# 1. Temporal / Calendar features
weekly_df['week_of_year'] = weekly_df['week_start'].dt.isocalendar().week.astype(int)
weekly_df['month'] = weekly_df['week_start'].dt.month


# 2. Historical Lag & Rolling features per SKU
weekly_df['lag_1_sales'] = weekly_df.groupby('sku_id')['weekly_sales'].shift(1)
weekly_df['lag_2_sales'] = weekly_df.groupby('sku_id')['weekly_sales'].shift(2)
weekly_df['rolling_4w_avg'] = weekly_df.groupby('sku_id')['weekly_sales'].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
)

# Clean up initial rows where lag features are empty (NaN)
model_df = weekly_df.dropna(subset=['lag_1_sales', 'avg_rating']).copy()

model_df.to_json('data/processed/weekly_sku_features.json', orient='index', indent=2 )
