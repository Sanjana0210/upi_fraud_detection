"""
ml/train_model.py
Train and compare three fraud-detection models on data/processed/cleaned.parquet.
Saves the best model as ml/model.pkl and the label encoder as ml/encoder.pkl.
"""

import json
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, BASE_DIR

# ─── Paths ────────────────────────────────────────────────────────────────────
PARQUET_PATH   = os.path.join(BASE_DIR, "data", "processed", "cleaned.parquet")
MODEL_OUT      = os.path.join(BASE_DIR, "ml", "model.pkl")
ENCODER_OUT    = os.path.join(BASE_DIR, "ml", "encoder.pkl")
CM_PLOT_PATH   = os.path.join(BASE_DIR, "ml", "confusion_matrix.png")
FI_PLOT_PATH   = os.path.join(BASE_DIR, "ml", "feature_importance.png")
COMPARE_OUT    = os.path.join(BASE_DIR, "ml", "model_comparison.json")

RANDOM_STATE   = 42
TEST_SIZE      = 0.20

FEATURES = [
    "type_encoded", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_balance_ratio", "is_round_amount", "zero_orig_balance",
]
TARGET = "isFraud"


# ─── 1. Load ──────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Processed parquet not found at {PARQUET_PATH}.\n"
            "Run  spark/process_data.py  first to generate it."
        )
    log(f"Loading parquet → {PARQUET_PATH}", "INFO")
    df = pd.read_parquet(PARQUET_PATH)
    log(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}", "INFO")
    return df


# ─── 2. Feature engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log("Engineering new features…", "INFO")
    df = df.copy()
    df["balance_diff_orig"]        = df["oldbalanceOrg"]  - df["newbalanceOrig"]
    df["balance_diff_dest"]        = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"]  = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_round_amount"]          = (df["amount"] % 1000 == 0).astype(int)
    df["zero_orig_balance"]        = (df["oldbalanceOrg"] == 0).astype(int)
    log("New features: balance_diff_orig, balance_diff_dest, amount_to_balance_ratio, "
        "is_round_amount, zero_orig_balance", "SUCCESS")
    return df


# ─── 3. Encode + split ────────────────────────────────────────────────────────
def prepare(df: pd.DataFrame):
    # type_encoded is already numeric from Spark; create a dummy encoder
    # so predict.py (which handles raw string types) stays consistent
    le = LabelEncoder()
    le.classes_ = np.array(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    df = df.copy()

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    log(f"Class distribution — Legit: {(y==0).sum():,}  Fraud: {(y==1).sum():,}", "INFO")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}", "INFO")
    return X_train, X_test, y_train, y_test, le


# ─── 4. SMOTE ─────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    log("Applying SMOTE to balance training set…", "INFO")
    sm = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log(f"After SMOTE — Legit: {(y_res==0).sum():,}  Fraud: {(y_res==1).sum():,}", "SUCCESS")
    return X_res, y_res


# ─── 5. Model definitions ─────────────────────────────────────────────────────
def build_models():
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )

    if _XGBOOST_AVAILABLE:
        log("XGBoost found — using XGBClassifier as second model.", "INFO")
        boost = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=99, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE,
            tree_method="hist", n_jobs=-1,
        )
        boost_name = "XGBoost"
    else:
        log("XGBoost not found — falling back to GradientBoostingClassifier.", "WARNING")
        boost = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
        boost_name = "GradientBoosting"

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_STATE,
    )

    return [
        ("Random Forest",   rf),
        (boost_name,        boost),
        ("Logistic Regression", lr),
    ]


# ─── 6. Evaluate one model ────────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test) -> dict:
    from sklearn.metrics import roc_curve, confusion_matrix as cm_func

    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # Confusion matrix
    cm = cm_func(y_test, y_pred).tolist()

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        fi = np.abs(model.coef_[0]).tolist()
    else:
        fi = []

    log(f"\n{'─'*55}", "INFO")
    log(f"  {name}", "INFO")
    log(f"{'─'*55}", "INFO")
    log(f"  Accuracy  : {acc:.4f}", "INFO")
    log(f"  Precision : {prec:.4f}", "INFO")
    log(f"  Recall    : {rec:.4f}", "INFO")
    log(f"  F1 Score  : {f1:.4f}", "INFO")
    log(f"  ROC-AUC   : {auc:.4f}", "INFO")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}", "INFO")

    return {"name": name, "model": model,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_prob": y_prob,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "confusion_matrix": cm, "feature_importance": fi}


# ─── 7. Save confusion matrices ───────────────────────────────────────────────
def save_confusion_matrices(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"],
            display_labels=["Legit", "Fraud"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(res["name"], fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrices — UPI Fraud Detection", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(CM_PLOT_PATH), exist_ok=True)
    fig.savefig(CM_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Confusion matrix plot saved → {CM_PLOT_PATH}", "SUCCESS")


# ─── 8. Save feature importance ───────────────────────────────────────────────
def save_feature_importance(best_result):
    model = best_result["model"]
    name  = best_result["name"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        log("Best model has no feature importances — skipping plot.", "WARNING")
        return

    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURES[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_features[::-1], sorted_importance[::-1],
                   color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {name}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FI_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Feature importance plot saved → {FI_PLOT_PATH}", "SUCCESS")


# ─── 9. Pick best model ───────────────────────────────────────────────────────
def pick_best(results) -> dict:
    best = max(results, key=lambda r: r["auc"])
    log(f"\n{'='*55}", "SUCCESS")
    log(f"  BEST MODEL: {best['name']}", "SUCCESS")
    log(f"  Chosen because it achieved the highest ROC-AUC: {best['auc']:.4f}", "SUCCESS")
    log(f"  (F1={best['f1']:.4f}  |  Recall={best['rec']:.4f}  |  Precision={best['prec']:.4f})", "SUCCESS")
    log(f"{'='*55}\n", "SUCCESS")
    return best


# ─── 10. Save comparison JSON for dashboard ───────────────────────────────────
def save_comparison_json(results, best):
    """Save model comparison data as JSON so the dashboard can render charts."""
    data = {
        "best_model": best["name"],
        "best_reason": f"Highest ROC-AUC ({best['auc']:.4f})",
        "features": FEATURES,
        "models": [],
    }
    for r in results:
        data["models"].append({
            "name":               r["name"],
            "accuracy":           round(r["acc"],  4),
            "precision":          round(r["prec"], 4),
            "recall":             round(r["rec"],  4),
            "f1_score":           round(r["f1"],   4),
            "roc_auc":            round(r["auc"],  4),
            "fpr":                r["fpr"],
            "tpr":                r["tpr"],
            "confusion_matrix":   r["confusion_matrix"],
            "feature_importance": r["feature_importance"],
        })

    os.makedirs(os.path.dirname(COMPARE_OUT), exist_ok=True)
    with open(COMPARE_OUT, "w") as f:
        json.dump(data, f, indent=2)
    log(f"Comparison JSON saved → {COMPARE_OUT}", "SUCCESS")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data()

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Prepare / encode / split
    X_train, X_test, y_train, y_test, le = prepare(df)

    # 4. SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 5. Scale (Logistic Regression needs it; apply to all for fairness)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    # 6. Train + evaluate
    models  = build_models()
    results = []
    for name, model in models:
        log(f"Training {name}…", "INFO")
        model.fit(X_train_scaled, y_train_res)
        res = evaluate_model(name, model, X_test_scaled, y_test)
        results.append(res)

    # 7. Plots + comparison data
    save_confusion_matrices(results, y_test)
    best = pick_best(results)
    save_feature_importance(best)
    save_comparison_json(results, best)

    # 8. Save best model + scaler + encoder
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    artifact = {"model": best["model"], "scaler": scaler, "features": FEATURES}
    joblib.dump(artifact, MODEL_OUT)
    joblib.dump(le, ENCODER_OUT)
    log(f"Model artifact saved → {MODEL_OUT}", "SUCCESS")
    log(f"Label encoder saved  → {ENCODER_OUT}", "SUCCESS")


if __name__ == "__main__":
    main()
