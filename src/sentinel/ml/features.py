import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "gas_station",
    "restaurant",
    "online_retail",
    "travel",
    "entertainment",
    "healthcare",
    "utilities",
    "other",
]

HIGH_RISK_COUNTRIES = {"NG", "RU", "CN", "BR", "IN", "PH", "VN"}

FEATURE_COLUMNS = [
    "amount_log",
    "hour",
    "day_of_week",
    "is_online_flag",
    "merchant_cat_code",
    "is_high_risk_country",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    features["amount_log"] = np.log1p(df["amount"].astype(float))
    ts = pd.to_datetime(df["transaction_time"], format="ISO8601")
    features["hour"] = ts.dt.hour
    features["day_of_week"] = ts.dt.dayofweek
    features["is_online_flag"] = df["is_online"].astype(int)

    cat_map = {cat: i for i, cat in enumerate(MERCHANT_CATEGORIES)}
    features["merchant_cat_code"] = (
        df["merchant_category"].str.lower().map(cat_map).fillna(len(cat_map) - 1).astype(int)
    )

    features["is_high_risk_country"] = (
        df["location_country"].str.upper().isin(HIGH_RISK_COUNTRIES).astype(int)
    )

    return features[FEATURE_COLUMNS]
