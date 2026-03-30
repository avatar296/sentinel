"""Generate synthetic transaction data for training."""

import random
from datetime import datetime, timedelta

import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "gas_station", "restaurant",
    "online_retail", "travel", "entertainment", "healthcare", "utilities",
]
MERCHANT_NAMES = {
    "grocery": ["Whole Foods", "Kroger", "Trader Joes", "Aldi"],
    "electronics": ["Best Buy", "Micro Center", "Newegg", "B&H Photo"],
    "gas_station": ["Shell", "BP", "Chevron", "ExxonMobil"],
    "restaurant": ["Chipotle", "Olive Garden", "McDonalds", "Subway"],
    "online_retail": ["Amazon", "eBay", "Walmart.com", "Target.com"],
    "travel": ["Delta Airlines", "Marriott", "Expedia", "Hertz"],
    "entertainment": ["Netflix", "AMC Theaters", "Spotify", "Steam"],
    "healthcare": ["CVS Pharmacy", "Walgreens", "Quest Diagnostics", "Zocdoc"],
    "utilities": ["ConEdison", "Comcast", "Verizon", "AT&T"],
}
CARD_TYPES = ["visa", "mastercard", "amex", "discover"]
COUNTRIES_NORMAL = ["US", "US", "US", "US", "US", "CA", "GB", "DE", "FR", "JP"]
COUNTRIES_HIGH_RISK = ["NG", "RU", "CN", "BR", "PH"]


def generate_normal_transaction(base_time: datetime) -> dict:
    category = random.choice(MERCHANT_CATEGORIES)
    amount_ranges = {
        "grocery": (5, 200), "electronics": (20, 1500), "gas_station": (10, 80),
        "restaurant": (8, 120), "online_retail": (5, 500), "travel": (50, 2000),
        "entertainment": (5, 60), "healthcare": (10, 300), "utilities": (30, 250),
    }
    lo, hi = amount_ranges[category]
    return {
        "amount": round(random.uniform(lo, hi), 2),
        "currency": "USD",
        "merchant_category": category,
        "merchant_name": random.choice(MERCHANT_NAMES[category]),
        "card_last_four": f"{random.randint(0, 9999):04d}",
        "card_type": random.choice(CARD_TYPES),
        "transaction_time": (base_time + timedelta(
            hours=random.gauss(14, 4), minutes=random.randint(0, 59)
        )).isoformat(),
        "location_country": random.choice(COUNTRIES_NORMAL),
        "is_online": category in ("online_retail", "entertainment") or random.random() < 0.1,
        "is_fraud": 0,
    }


def generate_fraudulent_transaction(base_time: datetime) -> dict:
    category = random.choice(["electronics", "online_retail", "travel"])
    return {
        "amount": round(random.uniform(500, 15000), 2),
        "currency": "USD",
        "merchant_category": category,
        "merchant_name": random.choice(MERCHANT_NAMES[category]),
        "card_last_four": f"{random.randint(0, 9999):04d}",
        "card_type": random.choice(CARD_TYPES),
        "transaction_time": (base_time + timedelta(
            hours=random.choice([2, 3, 4, 23, 0, 1]), minutes=random.randint(0, 59)
        )).isoformat(),
        "location_country": random.choice(COUNTRIES_HIGH_RISK + ["US"]),
        "is_online": random.random() < 0.8,
        "is_fraud": 1,
    }


def generate_dataset(n_total: int = 10000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    n_fraud = int(n_total * fraud_rate)
    n_normal = n_total - n_fraud

    base = datetime(2025, 1, 1)
    rows = []

    for i in range(n_normal):
        day_offset = random.randint(0, 180)
        rows.append(generate_normal_transaction(base + timedelta(days=day_offset)))

    for i in range(n_fraud):
        day_offset = random.randint(0, 180)
        rows.append(generate_fraudulent_transaction(base + timedelta(days=day_offset)))

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    output = "data/sample_transactions.csv"
    df.to_csv(output, index=False)
    fraud_count = df["is_fraud"].sum()
    print(f"Generated {len(df)} transactions ({fraud_count} fraudulent, {fraud_count/len(df)*100:.1f}%) -> {output}")
