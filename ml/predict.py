"""
ml/predict.py
Fraud prediction helpers for the UPI Fraud Detection project.

Public API:
    predict_transaction(transaction: dict) -> dict
    batch_predict(df: pd.DataFrame)        -> pd.DataFrame
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import List

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, BASE_DIR

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, "ml", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "ml", "encoder.pkl")

FEATURES = [
    "type_encoded", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_balance_ratio", "is_round_amount", "zero_orig_balance",
]

# ─── Risk thresholds ──────────────────────────────────────────────────────────
_LOW_THRESHOLD    = 0.30
_MEDIUM_THRESHOLD = 0.60


# ─── Cached model loading ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_artifact():
    """Load model artifact dict and encoder once; cache for all subsequent calls."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run  python ml/train_model.py  first."
        )
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Encoder not found at {ENCODER_PATH}.\n"
            "Run  python ml/train_model.py  first."
        )
    artifact = joblib.load(MODEL_PATH)
    encoder  = joblib.load(ENCODER_PATH)
    log("Model and encoder loaded (cached).", "SUCCESS")
    return artifact, encoder


# ─── Feature engineering (mirrors train_model.py exactly) ────────────────────
def _engineer(df: pd.DataFrame, encoder) -> pd.DataFrame:
    df = df.copy()

    # Encode string type -> type_encoded integer
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    if df["type"].dtype == object:
        df["type_encoded"] = df["type"].map(type_map).fillna(-1).astype(int)
    else:
        df["type_encoded"] = df["type"].astype(int)

    df["balance_diff_orig"]       = df["oldbalanceOrg"]  - df["newbalanceOrig"]
    df["balance_diff_dest"]       = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_round_amount"]         = (df["amount"] % 1000 == 0).astype(int)
    df["zero_orig_balance"]       = (df["oldbalanceOrg"] == 0).astype(int)
    return df


# ─── Fraud flag generator ─────────────────────────────────────────────────────
def _generate_flags(row: pd.Series) -> List[str]:
    flags = []
    if row["oldbalanceOrg"] > 0 and row["balance_diff_orig"] >= row["oldbalanceOrg"]:
        flags.append("Entire balance drained")
    if row["amount"] > 400_000:
        flags.append("Unusually large amount")
    if row["is_round_amount"] == 1:
        flags.append("Round number amount")
    if row["oldbalanceOrg"] == 0:
        flags.append("Zero starting balance")
    return flags


# ─── Risk level ───────────────────────────────────────────────────────────────
def _risk_level(confidence: float) -> str:
    if confidence < _LOW_THRESHOLD:
        return "LOW"
    if confidence < _MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


# ─── Public API ───────────────────────────────────────────────────────────────
def predict_transaction(transaction: dict) -> dict:
    """
    Predict fraud for a single transaction.

    Args:
        transaction: dict with keys —
            type, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest

    Returns:
        dict with prediction, label, confidence, risk_level, flags
    """
    required = {"type", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"}
    missing = required - set(transaction.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    artifact, encoder = _load_artifact()
    model   = artifact["model"]
    scaler  = artifact["scaler"]
    feature_cols = artifact["features"]

    df = pd.DataFrame([transaction])
    df = _engineer(df, encoder)
    df = df[feature_cols].astype(float)

    X_scaled   = scaler.transform(df)
    prediction = int(model.predict(X_scaled)[0])
    confidence = float(model.predict_proba(X_scaled)[0][1])

    row   = _engineer(pd.DataFrame([transaction]), encoder).iloc[0]
    flags = _generate_flags(row)

    return {
        "prediction": prediction,
        "label":      "FRAUD" if prediction == 1 else "LEGITIMATE",
        "confidence": round(confidence, 4),
        "risk_level": _risk_level(confidence),
        "flags":      flags,
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud prediction on an entire DataFrame.

    Adds columns: prediction, confidence, risk_level.
    Input DataFrame must contain the 6 raw transaction fields.
    """
    artifact, encoder = _load_artifact()
    model  = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = artifact["features"]

    engineered = _engineer(df.copy(), encoder)
    X_scaled   = scaler.transform(engineered[feature_cols].astype(float))

    preds       = model.predict(X_scaled)
    confidences = model.predict_proba(X_scaled)[:, 1]

    result = df.copy()
    result["prediction"]  = preds.astype(int)
    result["confidence"]  = confidences.round(4)
    result["risk_level"]  = [_risk_level(c) for c in confidences]
    return result


# ─── Main: demo with 3 example transactions ───────────────────────────────────
if __name__ == "__main__":
    examples = [
        {
            # Obvious fraud — entire balance drained via TRANSFER
            "name": "Obvious Fraud",
            "type": "TRANSFER",
            "amount": 750_000.00,
            "oldbalanceOrg":  750_000.00,
            "newbalanceOrig": 0.00,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 750_000.00,
        },
        {
            # Obvious legitimate — small grocery-style PAYMENT
            "name": "Obvious Legitimate",
            "type": "PAYMENT",
            "amount": 1_250.50,
            "oldbalanceOrg":  45_000.00,
            "newbalanceOrig": 43_749.50,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 0.00,
        },
        {
            # Borderline — round large amount, zero dest balance before
            "name": "Borderline",
            "type": "CASH_OUT",
            "amount": 200_000.00,
            "oldbalanceOrg":  210_000.00,
            "newbalanceOrig": 10_000.00,
            "oldbalanceDest": 0.00,
            "newbalanceDest": 200_000.00,
        },
    ]

    log("=" * 58, "INFO")
    log("  UPI Fraud Detection — predict.py demo", "INFO")
    log("=" * 58, "INFO")

    for txn in examples:
        name = txn.pop("name")
        try:
            result = predict_transaction(txn)
            log(f"\n[{name}]", "INFO")
            log(f"  Type       : {txn['type']}", "INFO")
            log(f"  Amount     : ₹{txn['amount']:,.2f}", "INFO")
            log(f"  Prediction : {result['label']}  (confidence={result['confidence']:.4f})", 
                "SUCCESS" if result["prediction"] == 0 else "ERROR")
            log(f"  Risk Level : {result['risk_level']}", "INFO")
            if result["flags"]:
                log(f"  Flags      : {', '.join(result['flags'])}", "WARNING")
            else:
                log("  Flags      : None", "INFO")
        except FileNotFoundError as e:
            log(str(e), "ERROR")
            log("Skipping demo — train the model first.", "WARNING")
            break
