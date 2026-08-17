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


### Here I was just having a look at the table
# print(weekly_df[["sku_id", "weekly_sales", "week_of_year"]].sort_values(by="week_of_year"))
"""
                           sku_id  weekly_sales  week_of_year
211    Bottoms_Ann Taylor_White_S             1             1
739          Dresses_Zara_Beige_S             1             1
1166    Shoes_Forever21_Purple_XL             1             1
134   Accessories_Uniqlo_Green_OS             1             1
1516            Tops_Gap_Green_XS             1             1
...                           ...           ...           ...
1219             Shoes_H&M_Blue_M             1            52
1385     Tops_Ann Taylor_Beige_XS             1            52
354          Bottoms_H&M_Purple_M             1            52
1607            Tops_Mango_Red_XL             1            52
1156    Shoes_Forever21_Brown_XXL             1            52

[1699 rows x 3 columns]
"""


### Here I had the wrong assumption that my min date is 1st of January and max date is 31st of Dec. Very wrong.
# print(weekly_df.groupby('week_of_year')[['weekly_sales']].count())

"""
              weekly_sales
week_of_year              
1                        5
2                        9
3                       10
4                        8
5                       12
6                       11
7                       14
8                        7
9                        4
10                      10
11                       8
12                       9
13                      13
14                      14
15                      11
16                      12
17                      15
18                      11
19                      13
20                       9
21                      11
22                      11
23                       6
24                      13
25                       6
26                       6
27                       7
28                      11
29                      11
30                      17
31                      12
32                    1172
33                      12
34                       8
35                      11
36                      14
37                      15
38                      11
39                      15
40                      15
41                       9
42                       9
43                      10
44                       9
45                       9
46                      11
47                       8
48                       6
49                       9
50                       3
51                      13
52                      14
"""


### I should've checked this sooner. Using the week of the year was misleading
#print(weekly_df.groupby('week_start')[['weekly_sales']].count())

"""
week_start              
2024-08-05             5
2024-08-12            12
2024-08-19             8
2024-08-26            11
2024-09-02            14
2024-09-09            15
2024-09-16            11
2024-09-23            15
2024-09-30            15
2024-10-07             9
2024-10-14             9
2024-10-21            10
2024-10-28             9
2024-11-04             9
2024-11-11            11
2024-11-18             8
2024-11-25             6
2024-12-02             9
2024-12-09             3
2024-12-16            13
2024-12-23            14
2024-12-30             5
2025-01-06             9
2025-01-13            10
2025-01-20             8
2025-01-27            12
2025-02-03            11
2025-02-10            14
2025-02-17             7
2025-02-24             4
2025-03-03            10
2025-03-10             8
2025-03-17             9
2025-03-24            13
2025-03-31            14
2025-04-07            11
2025-04-14            12
2025-04-21            15
2025-04-28            11
2025-05-05            13
2025-05-12             9
2025-05-19            11
2025-05-26            11
2025-06-02             6
2025-06-09            13
2025-06-16             6
2025-06-23             6
2025-06-30             7
2025-07-07            11
2025-07-14            11
2025-07-21            17
2025-07-28            12
2025-08-04          1167
"""


# 2. Historical Lag & Rolling features per SKU
weekly_df['lag_1_sales'] = weekly_df.groupby('sku_id')['weekly_sales'].shift(1)
weekly_df['lag_2_sales'] = weekly_df.groupby('sku_id')['weekly_sales'].shift(2)
weekly_df['rolling_4w_avg'] = weekly_df.groupby('sku_id')['weekly_sales'].transform(
    lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
)

# # Clean up initial rows where lag features are empty (NaN)
model_df = weekly_df.dropna(subset=['lag_1_sales']).copy()


model_df.to_json('data/processed/weekly_sku_features.json', orient='index', indent=2 )


# print(len(weekly_df))  #1699
# print(len(model_df))   #241 #277 after dropping the rating

print("Min:", df['week_start'].min())
print("Max:", df['week_start'].max())