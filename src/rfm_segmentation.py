"""
rfm_segmentation.py
--------------------
Computes Recency, Frequency, Monetary (RFM) metrics per customer from raw
transaction data, scores each dimension 1-5 using quantile binning, and
assigns each customer to a human-readable business segment.

Output: outputs/rfm_segments.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime

SNAPSHOT_DATE = datetime(2025, 12, 31)


def compute_rfm(transactions: pd.DataFrame, snapshot_date: datetime = SNAPSHOT_DATE) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    rfm = transactions.groupby("customer_id").agg(
        recency_days=("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("transaction_id", "count"),
        monetary=("amount", "sum"),
    ).reset_index()

    rfm["avg_order_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    # Recency: lower is better -> reverse scoring (5 = most recent)
    rfm["R_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency & Monetary: higher is better. Use rank-based qcut to handle ties/duplicates.
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)
    return rfm


def assign_segment(row) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r >= 3 and f <= 2 and m <= 2:
        return "Promising"
    if r == 3 and f == 3:
        return "Needs Attention"
    if r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    if r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    if r <= 2 and f <= 2 and m <= 2:
        return "Hibernating"
    if r == 1:
        return "Lost"
    return "Others"


def run_rfm_pipeline(transactions_path="data/transactions.csv", output_path="outputs/rfm_segments.csv") -> pd.DataFrame:
    transactions = pd.read_csv(transactions_path)
    rfm = compute_rfm(transactions)
    rfm = score_rfm(rfm)
    rfm["segment"] = rfm.apply(assign_segment, axis=1)
    rfm.to_csv(output_path, index=False)
    return rfm


if __name__ == "__main__":
    rfm = run_rfm_pipeline()
    print(rfm["segment"].value_counts())
    print(f"\nSaved -> outputs/rfm_segments.csv ({len(rfm)} customers)")
