"""
churn_model.py
--------------
Builds the churn feature set (RFM + engagement + demographics), engineers
the churn label, trains a Decision Tree and an XGBoost classifier, compares
them, and persists the winning model plus evaluation artifacts.

Churn label definition:
    A customer is labeled "churned" (1) if they have made no purchase in the
    90 days leading up to the snapshot date AND their account is older than
    90 days (so brand-new customers aren't unfairly penalized).

Output:
    outputs/model_comparison.json
    outputs/feature_importance.csv
    outputs/confusion_matrix.png
    outputs/roc_curve.png
    outputs/churn_predictions.csv
    outputs/churn_model.pkl
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from xgboost import XGBClassifier

from rfm_segmentation import run_rfm_pipeline

SNAPSHOT_DATE = datetime(2025, 12, 31)
CHURN_WINDOW_DAYS = 90
RANDOM_STATE = 42


def build_feature_set():
    customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
    engagement = pd.read_csv("data/engagement.csv")
    rfm = run_rfm_pipeline()

    df = customers.merge(rfm, on="customer_id", how="left").merge(engagement, on="customer_id", how="left")

    # customers with zero transactions never made it into rfm -> treat as long-recency, zero freq/monetary
    df["recency_days"] = df["recency_days"].fillna((SNAPSHOT_DATE - df["signup_date"]).dt.days)
    df["frequency"] = df["frequency"].fillna(0)
    df["monetary"] = df["monetary"].fillna(0)
    df["avg_order_value"] = df["avg_order_value"].fillna(0)
    for c in ["R_score", "F_score", "M_score", "RFM_total"]:
        df[c] = df[c].fillna(1)
    df["segment"] = df["segment"].fillna("Hibernating")

    df["tenure_days"] = (SNAPSHOT_DATE - df["signup_date"]).dt.days

    # ---- Churn label ----
    df["churned"] = np.where(
        (df["recency_days"] > CHURN_WINDOW_DAYS) & (df["tenure_days"] > CHURN_WINDOW_DAYS),
        1, 0
    )

    return df


# NOTE: recency_days / R_score / RFM_total are deliberately EXCLUDED as model features.
# The churn label is derived directly from recency (>90 days since last purchase), so
# including it would leak the label into the features and produce a trivial, useless
# model. Instead the model learns to anticipate churn from *leading* behavioral signals
# (purchase frequency/value trend, engagement, support friction) rather than restating
# the definition of churn itself -- this is what makes the prediction actionable.
FEATURES_NUMERIC = [
    "age", "tenure_days", "frequency", "monetary", "avg_order_value",
    "F_score", "M_score",
    "support_tickets_last_90d", "app_logins_last_30d", "email_open_rate", "avg_support_csat",
]
FEATURES_CATEGORICAL = ["gender", "region", "acquisition_channel"]


def encode_features(df):
    df = df.copy()
    encoders = {}
    for col in FEATURES_CATEGORICAL:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    feature_cols = FEATURES_NUMERIC + [c + "_enc" for c in FEATURES_CATEGORICAL]
    return df, feature_cols, encoders


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1_score": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }, proba, pred


def main():
    df = build_feature_set()
    df_enc, feature_cols, encoders = encode_features(df)

    X = df_enc[feature_cols]
    y = df_enc["churned"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_enc.index, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE, class_weight="balanced")
    dt.fit(X_train, y_train)
    dt_metrics, dt_proba, dt_pred = evaluate(dt, X_test, y_test)

    # ---- XGBoost ----
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, xgb_proba, xgb_pred = evaluate(xgb, X_test, y_test)

    comparison = {"decision_tree": dt_metrics, "xgboost": xgb_metrics}
    with open("outputs/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps(comparison, indent=2))

    # pick winner by ROC-AUC
    winner_name = "xgboost" if xgb_metrics["roc_auc"] >= dt_metrics["roc_auc"] else "decision_tree"
    winner_model = xgb if winner_name == "xgboost" else dt
    winner_proba_test = xgb_proba if winner_name == "xgboost" else dt_proba
    winner_pred_test = xgb_pred if winner_name == "xgboost" else dt_pred
    print(f"\nBest model: {winner_name} (ROC-AUC = {comparison[winner_name]['roc_auc']})")

    # ---- Feature importance (winner) ----
    importances = winner_model.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi.to_csv("outputs/feature_importance.csv", index=False)

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, winner_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Retained", "Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({winner_name})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    # ---- ROC curve plot (both models) ----
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={dt_metrics['roc_auc']})")
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={xgb_metrics['roc_auc']})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison"); plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()

    # ---- Score ALL customers with the winning model (full dataset) ----
    full_proba = winner_model.predict_proba(X)[:, 1]
    df_out = df_enc[["customer_id", "segment", "recency_days", "frequency", "monetary",
                      "avg_order_value", "RFM_total", "tenure_days", "churned"]].copy()
    df_out["churn_probability"] = np.round(full_proba, 4)

    def risk_tier(p):
        if p >= 0.66:
            return "High"
        elif p >= 0.33:
            return "Medium"
        return "Low"

    df_out["risk_tier"] = df_out["churn_probability"].apply(risk_tier)

    def recommend(row):
        if row["risk_tier"] == "High" and row["segment"] in ("Champions", "Loyal Customers", "Cannot Lose Them"):
            return "Priority win-back: personal outreach + loyalty offer"
        if row["risk_tier"] == "High":
            return "Send targeted discount / re-engagement email"
        if row["risk_tier"] == "Medium":
            return "Monitor + include in nurture campaign"
        if row["segment"] == "Champions":
            return "Upsell / referral program invite"
        return "Standard engagement / newsletter"

    df_out["recommended_action"] = df_out.apply(recommend, axis=1)
    df_out.to_csv("outputs/churn_predictions.csv", index=False)

    with open("outputs/churn_model.pkl", "wb") as f:
        pickle.dump({"model": winner_model, "encoders": encoders, "feature_cols": feature_cols,
                     "winner_name": winner_name}, f)

    print(f"\nSaved: outputs/model_comparison.json, feature_importance.csv, confusion_matrix.png,")
    print(f"       roc_curve.png, churn_predictions.csv, churn_model.pkl")
    return df, df_out, comparison


if __name__ == "__main__":
    main()
