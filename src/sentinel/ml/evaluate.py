import numpy as np
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)

    auc_pr = average_precision_score(y_true, y_prob)

    report = classification_report(y_true, y_pred)

    print("=== Classification Report ===")
    print(report)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")

    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "classification_report": report}
