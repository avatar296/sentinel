import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sentinel.ml.evaluate import evaluate_model
from sentinel.ml.features import build_features

try:
    import mlflow
except ImportError:
    mlflow = None


def train(data_path: str = "data/sample_transactions.csv", output_path: str = "models/fraud_model.joblib"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions ({df['is_fraud'].sum()} fraudulent)")

    X = build_features(df)
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": fraud_ratio,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(**params)),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test.values, y_prob)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")

    # Log to MLflow if available
    if mlflow is not None:
        mlflow.set_experiment("sentinel-training")
        with mlflow.start_run(run_name="xgboost-train"):
            mlflow.log_params(params)
            mlflow.log_params({"data_path": data_path, "n_rows": len(df), "fraud_rate": float(y.mean())})
            mlflow.log_metrics({"auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"]})
            mlflow.log_artifact(output_path)
        print("Logged run to MLflow")

    # Save to BentoML model store if available
    try:
        import bentoml
        tag = bentoml.sklearn.save_model(
            "sentinel-fraud",
            pipeline,
            labels={"framework": "xgboost", "name": "xgboost", "source": "train-script"},
            metadata={"auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"]},
        )
        print(f"Saved to BentoML model store: {tag}")
    except ImportError:
        pass


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    output = sys.argv[2] if len(sys.argv) > 2 else "models/fraud_model.joblib"
    train(data, output)
