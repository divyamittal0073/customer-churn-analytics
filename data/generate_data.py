"""
generate_data.py
-----------------
Generates a realistic synthetic e-commerce dataset:
  - customers.csv     : demographic + acquisition info
  - transactions.csv  : purchase history over a 24-month window
  - engagement.csv     : support / app engagement signals

The data is generated with built-in behavioral patterns (not pure noise) so that
RFM segmentation and churn prediction produce meaningful, explainable results.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_CUSTOMERS = 2000
SNAPSHOT_DATE = datetime(2025, 12, 31)          # "today" for RFM purposes
WINDOW_START = SNAPSHOT_DATE - timedelta(days=730)  # 24-month lookback

REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Organic Search", "Paid Ads", "Referral", "Social Media", "Email Campaign"]
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books"]

# Behavioral archetypes -> drives realistic churn patterns
ARCHETYPES = {
    "loyal":       {"weight": 0.18, "freq_lambda": 1.4, "churn_bias": -2.2, "aov_mult": 1.3},
    "regular":     {"weight": 0.32, "freq_lambda": 0.7, "churn_bias": -0.5, "aov_mult": 1.0},
    "occasional":  {"weight": 0.30, "freq_lambda": 0.35, "churn_bias": 0.6, "aov_mult": 0.85},
    "one_and_done":{"weight": 0.13, "freq_lambda": 0.08, "churn_bias": 1.8, "aov_mult": 0.9},
    "new":         {"weight": 0.07, "freq_lambda": 0.9, "churn_bias": -0.2, "aov_mult": 1.1},
}


def build_customers():
    archetype_names = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[a]["weight"] for a in archetype_names]

    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        archetype = np.random.choice(archetype_names, p=weights)
        if archetype == "new":
            signup_date = SNAPSHOT_DATE - timedelta(days=int(np.random.uniform(1, 60)))
        else:
            signup_date = WINDOW_START + timedelta(days=int(np.random.uniform(0, 700)))

        rows.append({
            "customer_id": f"CUST{i:05d}",
            "signup_date": signup_date.date().isoformat(),
            "age": int(np.clip(np.random.normal(38, 12), 18, 75)),
            "gender": random.choice(["Male", "Female", "Other"]),
            "region": random.choice(REGIONS),
            "acquisition_channel": random.choice(CHANNELS),
            "_archetype": archetype,
        })
    return pd.DataFrame(rows)


def build_transactions(customers):
    all_txns = []
    txn_id = 1
    for _, cust in customers.iterrows():
        archetype = ARCHETYPES[cust["_archetype"]]
        signup = datetime.fromisoformat(cust["signup_date"])
        active_days = max((SNAPSHOT_DATE - signup).days, 1)

        # expected number of purchases scales with tenure and archetype frequency
        expected_orders = max(0, np.random.poisson(archetype["freq_lambda"] * (active_days / 30)))

        for _ in range(expected_orders):
            offset = np.random.uniform(0, active_days)
            txn_date = signup + timedelta(days=offset)
            if txn_date > SNAPSHOT_DATE:
                continue
            base_amount = np.random.gamma(shape=3.0, scale=18) * archetype["aov_mult"]
            all_txns.append({
                "transaction_id": f"TXN{txn_id:07d}",
                "customer_id": cust["customer_id"],
                "transaction_date": txn_date.date().isoformat(),
                "amount": round(float(base_amount), 2),
                "category": random.choice(CATEGORIES),
                "discount_used": np.random.choice([0, 1], p=[0.7, 0.3]),
            })
            txn_id += 1
    return pd.DataFrame(all_txns)


def build_engagement(customers):
    rows = []
    for _, cust in customers.iterrows():
        archetype = ARCHETYPES[cust["_archetype"]]
        # more engaged archetypes have more logins, fewer complaint tickets
        engagement_level = -archetype["churn_bias"]
        rows.append({
            "customer_id": cust["customer_id"],
            "support_tickets_last_90d": max(0, int(np.random.poisson(max(0.05, 0.6 - 0.15 * engagement_level)))),
            "app_logins_last_30d": max(0, int(np.random.poisson(max(0.1, 4 + 3 * engagement_level)))),
            "email_open_rate": round(float(np.clip(np.random.normal(0.35 + 0.08 * engagement_level, 0.15), 0, 1)), 2),
            "avg_support_csat": round(float(np.clip(np.random.normal(3.9 + 0.3 * engagement_level, 0.7), 1, 5)), 1),
        })
    return pd.DataFrame(rows)


def main():
    customers = build_customers()
    transactions = build_transactions(customers)
    engagement = build_engagement(customers)

    # keep archetype hidden from downstream files (it's the "ground truth" generator,
    # not something a real business would have) -- but stash it for validation use only
    customers_public = customers.drop(columns=["_archetype"])

    customers_public.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    engagement.to_csv("data/engagement.csv", index=False)
    customers[["customer_id", "_archetype"]].to_csv("data/_ground_truth_archetype.csv", index=False)

    print(f"customers.csv      -> {len(customers_public)} rows")
    print(f"transactions.csv   -> {len(transactions)} rows")
    print(f"engagement.csv     -> {len(engagement)} rows")


if __name__ == "__main__":
    main()
