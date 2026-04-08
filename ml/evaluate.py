"""
ml/evaluate.py
Load the saved model + test data and print / plot a full evaluation report.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, MODEL_PATH

TEST_DATA_PATH = os.path.join(os.path.dirname(MODEL_PATH), "test_data.pkl")
PLOTS_DIR = os.path.join(os.path.dirname(MODEL_PATH), "plots")


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train first.")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {TEST_DATA_PATH}. Train first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TEST_DATA_PATH, "rb") as f:
        X_test, y_test = pickle.load(f)

    return model, X_test, y_test


def evaluate(threshold: float = 0.5):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model, X_test, y_test = load_artifacts()

    log("Running predictions on test set…", "INFO")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Classification report ─────────────────────────────────────────────────
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    log(f"ROC-AUC  : {roc_auc:.4f}", "INFO")
    log(f"Avg Prec : {avg_prec:.4f}", "INFO")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    log(f"Saved → {cm_path}", "SUCCESS")
    plt.close()

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    log(f"Saved → {roc_path}", "SUCCESS")
    plt.close()

    # ── Precision-Recall Curve ────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="green", lw=2, label=f"AP = {avg_prec:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path)
    log(f"Saved → {pr_path}", "SUCCESS")
    plt.close()

    # ── Feature Importance ────────────────────────────────────────────────────
    try:
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_
        from utils.helpers import FEATURE_COLS
        feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
        names, vals = zip(*feat_imp)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names[::-1], vals[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances (Random Forest)")
        plt.tight_layout()
        fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(fi_path)
        log(f"Saved → {fi_path}", "SUCCESS")
        plt.close()
    except Exception as e:
        log(f"Could not plot feature importance: {e}", "WARNING")

    log("Evaluation complete.", "SUCCESS")
    return {"roc_auc": roc_auc, "avg_precision": avg_prec}


if __name__ == "__main__":
    evaluate()
