from __future__ import annotations

import pandas as pd


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily transactions into weekly SKU-level features."""
    weekly_df = df.groupby(
        ["sku_id", "category", "brand", "color", "size", "week_start"],
        as_index=False,
    ).agg(
        weekly_sales=("product_id", "count"),
        avg_price=("current_price", "mean"),
        avg_markdown=("markdown_percentage", "mean"),
        avg_rating=("customer_rating", "mean"),
    )

    weekly_df = weekly_df.sort_values(["sku_id", "week_start"]).reset_index(drop=True)

    weekly_df["week_of_year"] = weekly_df["week_start"].dt.isocalendar().week.astype(int)
    weekly_df["month"] = weekly_df["week_start"].dt.month

    weekly_df["lag_1_sales"] = weekly_df.groupby("sku_id")["weekly_sales"].shift(1)
    weekly_df["lag_2_sales"] = weekly_df.groupby("sku_id")["weekly_sales"].shift(2)
    weekly_df["rolling_4w_avg"] = weekly_df.groupby("sku_id")["weekly_sales"].transform(
        lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
    )

    model_df = weekly_df.dropna(subset=["lag_1_sales", "avg_rating"]).copy()
    return model_df
