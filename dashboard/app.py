"""Sentinel Fraud Detection Dashboard — Streamlit app.

Run with: streamlit run dashboard/app.py
"""

import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Sentinel Dashboard", layout="wide")

DB_URL = "postgresql://sentinel:sentinel@localhost:5432/sentinel"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_transactions():
    """Load transactions from PostgreSQL. Falls back to sample data if DB is unavailable."""
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(DB_URL)
        df = pd.read_sql("SELECT * FROM transactions ORDER BY created_at DESC LIMIT 5000", engine)
        engine.dispose()
        return df, "database"
    except Exception:
        # Fall back to sample CSV with simulated scores
        try:
            df = pd.read_csv("data/sample_transactions.csv")
            # Simulate scores for demo purposes
            rng = np.random.default_rng(42)
            df["fraud_score"] = rng.beta(2, 8, size=len(df))
            df.loc[df["is_fraud"] == 1, "fraud_score"] = rng.beta(6, 3, size=(df["is_fraud"] == 1).sum())
            df["rules_score"] = rng.beta(1.5, 6, size=len(df))
            df.loc[df["is_fraud"] == 1, "rules_score"] = rng.beta(4, 3, size=(df["is_fraud"] == 1).sum())

            # Simulate decisions
            conditions = [
                df["fraud_score"] >= 0.8,
                (df["fraud_score"] >= 0.4) & (df["rules_score"] >= 0.5),
                (df["fraud_score"] >= 0.4) & (df["rules_score"] < 0.5),
                (df["fraud_score"] < 0.4) & (df["rules_score"] >= 0.5),
            ]
            choices = ["FLAG", "FLAG", "REVIEW", "REVIEW"]
            df["decision"] = np.select(conditions, choices, default="APPROVE")
            df["is_flagged"] = df["decision"] == "FLAG"

            if "created_at" not in df.columns:
                base = datetime.now() - timedelta(days=30)
                df["created_at"] = [base + timedelta(hours=i) for i in range(len(df))]

            return df, "sample_csv"
        except FileNotFoundError:
            return pd.DataFrame(), "none"


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

st.title("Sentinel Fraud Detection Dashboard")
st.caption("Real-time overview of transaction scoring, rules engine, and escalation routing")

df, source = load_transactions()

if df.empty:
    st.error("No data available. Start the API and create some transactions, or place sample_transactions.csv in data/.")
    st.stop()

st.sidebar.markdown(f"**Data source:** `{source}`")
st.sidebar.markdown(f"**Transactions:** {len(df):,}")

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------

st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

has_decision = "decision" in df.columns and df["decision"].notna().any()

with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    flagged = int(df["is_flagged"].sum()) if "is_flagged" in df.columns else 0
    st.metric("Flagged (FLAG)", flagged, delta=None)
with col3:
    review = int((df["decision"] == "REVIEW").sum()) if has_decision else 0
    st.metric("In Review", review)
with col4:
    approved = int((df["decision"] == "APPROVE").sum()) if has_decision else len(df) - flagged - review
    st.metric("Approved", approved)

# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

st.header("Score Distributions")
col_left, col_right = st.columns(2)

with col_left:
    if "fraud_score" in df.columns and df["fraud_score"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        scores = df["fraud_score"].dropna()
        ax.hist(scores, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(0.4, color="orange", linestyle="--", label="Review threshold")
        ax.axvline(0.8, color="red", linestyle="--", label="Flag threshold")
        ax.set_title("ML Fraud Score Distribution")
        ax.set_xlabel("Score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

with col_right:
    if "rules_score" in df.columns and df["rules_score"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        scores = df["rules_score"].dropna()
        ax.hist(scores, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.axvline(0.5, color="orange", linestyle="--", label="Rules threshold")
        ax.set_title("Rules Engine Score Distribution")
        ax.set_xlabel("Score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

# ---------------------------------------------------------------------------
# Layer performance (if ground truth available)
# ---------------------------------------------------------------------------

truth_col = None
if "is_fraud" in df.columns:
    truth_col = "is_fraud"
elif "isFraud" in df.columns:
    truth_col = "isFraud"

if truth_col and "fraud_score" in df.columns:
    st.header("Layer Performance (Precision / Recall / F1)")

    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = df[truth_col].astype(int)
    layer_data = {}

    if "rules_score" in df.columns and df["rules_score"].notna().any():
        rules_pred = (df["rules_score"] >= 0.5).astype(int)
        layer_data["Rules Only"] = {
            "Precision": precision_score(y_true, rules_pred, zero_division=0),
            "Recall": recall_score(y_true, rules_pred),
            "F1": f1_score(y_true, rules_pred, zero_division=0),
        }

    if df["fraud_score"].notna().any():
        ml_pred = (df["fraud_score"] >= 0.5).astype(int)
        layer_data["ML Only"] = {
            "Precision": precision_score(y_true, ml_pred, zero_division=0),
            "Recall": recall_score(y_true, ml_pred),
            "F1": f1_score(y_true, ml_pred, zero_division=0),
        }

    if "Rules Only" in layer_data and "ML Only" in layer_data:
        combined_or = ((ml_pred == 1) | (rules_pred == 1)).astype(int)
        combined_and = ((ml_pred == 1) & (rules_pred == 1)).astype(int)
        layer_data["Combined OR"] = {
            "Precision": precision_score(y_true, combined_or, zero_division=0),
            "Recall": recall_score(y_true, combined_or),
            "F1": f1_score(y_true, combined_or, zero_division=0),
        }
        layer_data["Combined AND"] = {
            "Precision": precision_score(y_true, combined_and, zero_division=0),
            "Recall": recall_score(y_true, combined_and),
            "F1": f1_score(y_true, combined_and, zero_division=0),
        }

    if layer_data:
        perf_df = pd.DataFrame(layer_data).T
        st.dataframe(perf_df.style.format("{:.4f}"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(perf_df))
        w = 0.25
        ax.bar(x - w, perf_df["Precision"], w, label="Precision", color="#3498db")
        ax.bar(x, perf_df["Recall"], w, label="Recall", color="#e74c3c")
        ax.bar(x + w, perf_df["F1"], w, label="F1", color="#2ecc71")
        ax.set_xticks(x)
        ax.set_xticklabels(perf_df.index, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Layer Performance Comparison")
        st.pyplot(fig)
        plt.close()

# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

st.header("Review Queue")

if has_decision:
    review_df = df[df["decision"] == "REVIEW"].head(50)
    if review_df.empty:
        st.info("No transactions currently in the review queue.")
    else:
        display_cols = ["amount", "merchant_category", "location_country",
                        "fraud_score", "rules_score", "decision"]
        available = [c for c in display_cols if c in review_df.columns]
        st.dataframe(review_df[available], use_container_width=True)
else:
    st.info("Decision data not available. Run transactions through the API to populate the review queue.")

# ---------------------------------------------------------------------------
# Decision breakdown
# ---------------------------------------------------------------------------

if has_decision:
    st.header("Decision Breakdown")
    decision_counts = df["decision"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors_map = {"APPROVE": "#2ecc71", "REVIEW": "#f39c12", "FLAG": "#e74c3c"}
    bar_colors = [colors_map.get(d, "#95a5a6") for d in decision_counts.index]
    decision_counts.plot.bar(ax=ax, color=bar_colors)
    ax.set_title("Transactions by Decision")
    ax.set_ylabel("Count")
    ax.set_xticklabels(decision_counts.index, rotation=0)
    st.pyplot(fig)
    plt.close()
