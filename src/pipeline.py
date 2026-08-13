"""
pipeline.py
-----------
End-to-end orchestrator:
  1. Generates synthetic data (if not already present)
  2. Runs RFM segmentation
  3. Trains/evaluates churn models and scores every customer
  4. Joins everything into a single, denormalized, Power-BI-ready table

Run from the project root:
    python src/pipeline.py
"""

import os
import subprocess
import sys
import pandas as pd

import churn_model


def ensure_data():
    if not os.path.exists("data/customers.csv"):
        print("No raw data found -- generating synthetic dataset...")
        subprocess.run([sys.executable, "data/generate_data.py"], check=True)
    else:
        print("Raw data already present, skipping generation.")


def build_powerbi_dataset():
    customers = pd.read_csv("data/customers.csv")
    engagement = pd.read_csv("data/engagement.csv")
    preds = pd.read_csv("outputs/churn_predictions.csv")

    final = (
        preds.merge(customers, on="customer_id", how="left")
             .merge(engagement, on="customer_id", how="left")
    )

    final["signup_date"] = pd.to_datetime(final["signup_date"])
    final["tenure_months"] = (final["tenure_days"] / 30).round(1)
    final["customer_value_tier"] = pd.qcut(
        final["monetary"].rank(method="first"), 4,
        labels=["Bronze", "Silver", "Gold", "Platinum"]
    )

    col_order = [
        "customer_id", "signup_date", "age", "gender", "region", "acquisition_channel",
        "tenure_days", "tenure_months",
        "recency_days", "frequency", "monetary", "avg_order_value", "RFM_total", "segment",
        "customer_value_tier",
        "support_tickets_last_90d", "app_logins_last_30d", "email_open_rate", "avg_support_csat",
        "churn_probability", "risk_tier", "recommended_action", "churned",
    ]
    final = final[col_order]
    final.to_csv("outputs/powerbi_dataset.csv", index=False)

    # A compact segment-level summary table, handy for a Power BI KPI card / summary page
    summary = final.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_monetary=("monetary", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        high_risk_count=("risk_tier", lambda x: (x == "High").sum()),
    ).reset_index().round(2)
    summary.to_csv("outputs/segment_summary.csv", index=False)

    print(f"powerbi_dataset.csv -> {len(final)} rows, {len(final.columns)} columns")
    print(f"segment_summary.csv -> {len(summary)} segments")
    return final, summary


def main():
    ensure_data()
    print("\n=== Step 1/2: Churn model training (includes RFM segmentation) ===")
    churn_model.main()
    print("\n=== Step 2/2: Building Power BI export ===")
    build_powerbi_dataset()
    print("\nPipeline complete. See /outputs for all generated files.")


if __name__ == "__main__":
    main()
