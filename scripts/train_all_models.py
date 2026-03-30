"""Train 4 models on Kaggle Credit Card Fraud dataset, log to MLflow, save to BentoML.

Usage: .venv/bin/python scripts/train_all_models.py
"""

import time
from pathlib import Path

import bentoml
import bentoml.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/kaggle/creditcard.csv")
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("SENTINEL MODEL TRAINING PIPELINE")
print("=" * 70)

print(f"\n[1/7] Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Fraud rate: {df['Class'].mean():.4%} ({int(df['Class'].sum())} / {len(df)})")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("\n[2/7] Engineering features...")
X = df.drop(columns=["Class"]).copy()
y = df["Class"].astype(int)

# Log-transform Amount and normalize Time
X["Amount_log"] = np.log1p(X["Amount"])
X["Time_hour"] = (X["Time"] / 3600) % 24
X.drop(columns=["Amount", "Time"], inplace=True)

print(f"  Features: {X.shape[1]} ({list(X.columns[:5])} ... {list(X.columns[-3:])})")

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
print("\n[3/7] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"  Train: {len(X_train):,} ({y_train.mean():.4%} fraud)")
print(f"  Test:  {len(X_test):,} ({y_test.mean():.4%} fraud)")
print(f"  scale_pos_weight: {fraud_ratio:.1f}")

# ---------------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------------
model_configs = {
    "logreg": {
        "display": "Logistic Regression",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42, C=0.1,
            )),
        ]),
    },
    "random-forest": {
        "display": "Random Forest",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
    },
    "xgboost": {
        "display": "XGBoost",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                scale_pos_weight=fraud_ratio, eval_metric="aucpr",
                random_state=42,
            )),
        ]),
    },
    "gradient-boosting": {
        "display": "Gradient Boosting",
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42,
            )),
        ]),
    },
}

# ---------------------------------------------------------------------------
# 5. Train, evaluate, log to MLflow, save to BentoML
# ---------------------------------------------------------------------------
print("\n[4/7] Training models with MLflow tracking...")

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("sentinel-model-comparison")

results = {}

for short_name, cfg in model_configs.items():
    display = cfg["display"]
    pipeline = cfg["pipeline"]

    print(f"\n  --- {display} ---")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "train_time_s": round(train_time, 2),
    }

    results[short_name] = {
        "display": display,
        "metrics": metrics,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "pipeline": pipeline,
    }

    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"    AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    Train time: {train_time:.1f}s")

    # MLflow
    with mlflow.start_run(run_name=short_name):
        mlflow.log_param("model_type", short_name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("fraud_ratio", round(fraud_ratio, 1))
        clf = pipeline.named_steps["clf"]
        for k, v in clf.get_params().items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)

    # BentoML
    tag = bentoml.sklearn.save_model(
        "sentinel-fraud",
        pipeline,
        labels={"framework": short_name, "name": short_name, "dataset": "kaggle-creditcard"},
        metadata={k: float(v) for k, v in metrics.items()},
    )
    print(f"    BentoML tag: {tag}")

# ---------------------------------------------------------------------------
# 6. Comparison summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[5/7] MODEL COMPARISON SUMMARY")
print("=" * 70)

summary_data = {}
for name, r in results.items():
    summary_data[r["display"]] = r["metrics"]

summary_df = pd.DataFrame(summary_data).T
summary_df = summary_df.round(4)
print(f"\n{summary_df.to_string()}")

best_by_auc_pr = summary_df["auc_pr"].idxmax()
best_by_f1 = summary_df["f1"].idxmax()
print(f"\n  Best by AUC-PR: {best_by_auc_pr} ({summary_df.loc[best_by_auc_pr, 'auc_pr']:.4f})")
print(f"  Best by F1:     {best_by_f1} ({summary_df.loc[best_by_f1, 'f1']:.4f})")

# Classification reports for each model
print("\n" + "-" * 70)
print("DETAILED CLASSIFICATION REPORTS")
print("-" * 70)
for name, r in results.items():
    print(f"\n  {r['display']}:")
    report = classification_report(y_test, r["y_pred"], target_names=["Legit", "Fraud"], digits=4)
    for line in report.split("\n"):
        print(f"    {line}")

# ---------------------------------------------------------------------------
# 7. Generate figures
# ---------------------------------------------------------------------------
print(f"\n[6/7] Generating figures to {FIGURES_DIR}/...")

sns.set_theme(style="whitegrid", palette="colorblind")
colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

# ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, color=color, label=f"{r['display']} ({r['metrics']['auc_roc']:.3f})")
    prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
    axes[1].plot(rec, prec, color=color, label=f"{r['display']} ({r['metrics']['auc_pr']:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[1].set_title("Precision-Recall Curves")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
print(f"  Saved {FIGURES_DIR / 'roc_pr_curves.png'}")

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title(r["display"], fontsize=10)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
print(f"  Saved {FIGURES_DIR / 'confusion_matrices.png'}")

# Threshold analysis for best model
best_short = [k for k, v in results.items() if v["display"] == best_by_auc_pr][0]
best_probs = results[best_short]["y_prob"]

thresholds = np.arange(0.05, 0.95, 0.05)
tm = []
for t in thresholds:
    preds = (best_probs >= t).astype(int)
    if preds.sum() == 0:
        continue
    tm.append({
        "threshold": t,
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds, zero_division=0),
    })
tm_df = pd.DataFrame(tm)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tm_df["threshold"], tm_df["precision"], "b-o", markersize=4, label="Precision")
ax.plot(tm_df["threshold"], tm_df["recall"], "r-o", markersize=4, label="Recall")
ax.plot(tm_df["threshold"], tm_df["f1"], "g-o", markersize=4, label="F1")
ax.axvspan(0.0, 0.4, alpha=0.08, color="green", label="APPROVE zone")
ax.axvspan(0.4, 0.8, alpha=0.08, color="orange", label="REVIEW zone")
ax.axvspan(0.8, 1.0, alpha=0.08, color="red", label="FLAG zone")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title(f"Threshold Analysis — {best_by_auc_pr}")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "threshold_analysis.png", dpi=150, bbox_inches="tight")
print(f"  Saved {FIGURES_DIR / 'threshold_analysis.png'}")

# Metrics comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(summary_df))
w = 0.18
for i, (metric, color) in enumerate(zip(["auc_roc", "auc_pr", "f1", "precision", "recall"], colors + ["#f39c12"])):
    ax.bar(x + i * w, summary_df[metric], w, label=metric.upper().replace("_", "-"), alpha=0.85)
ax.set_xticks(x + 2 * w)
ax.set_xticklabels(summary_df.index, rotation=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Model Comparison — All Metrics")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
print(f"  Saved {FIGURES_DIR / 'model_comparison.png'}")

# ---------------------------------------------------------------------------
# 8. BentoML store listing
# ---------------------------------------------------------------------------
print("\n[7/7] BentoML model store:")
print(f"  {'Tag':<50} {'AUC-PR':<10} {'F1':<10} {'Recall':<10}")
print("  " + "-" * 80)
for model in bentoml.models.list("sentinel-fraud"):
    meta = model.info.metadata
    print(f"  {str(model.tag):<50} {meta.get('auc_pr', 0):<10.4f} {meta.get('f1', 0):<10.4f} {meta.get('recall', 0):<10.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print(f"  MLflow UI: cd {Path.cwd()} && .venv/bin/mlflow ui --backend-store-uri mlruns")
print(f"  BentoML models: .venv/bin/bentoml models list sentinel-fraud")
print("=" * 70)
