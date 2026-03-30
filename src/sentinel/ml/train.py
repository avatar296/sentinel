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

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=fraud_ratio,
            eval_metric="aucpr",
            random_state=42,
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    evaluate_model(y_test.values, y_prob)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    output = sys.argv[2] if len(sys.argv) > 2 else "models/fraud_model.joblib"
    train(data, output)
