"""
tests/test_predict.py
Unit tests for ml/predict.py — pure logic functions that don't need a trained model.
"""

import pytest
import pandas as pd

from ml.predict import _engineer, _risk_level, _generate_flags, FEATURES


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fraud_transaction():
    """A transaction that looks like obvious fraud."""
    return pd.DataFrame([{
        "type":           "TRANSFER",
        "amount":         750_000.0,
        "oldbalanceOrg":  750_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 750_000.0,
    }])


@pytest.fixture
def legit_transaction():
    """A normal-looking small payment."""
    return pd.DataFrame([{
        "type":           "PAYMENT",
        "amount":         1_250.50,
        "oldbalanceOrg":  45_000.0,
        "newbalanceOrig": 43_749.50,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }])


# ─── _engineer() ──────────────────────────────────────────────────────────────

def test_engineer_adds_all_features(fraud_transaction):
    """_engineer should produce all columns listed in FEATURES."""
    result = _engineer(fraud_transaction, encoder=None)
    for feat in FEATURES:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_engineer_type_encoding(fraud_transaction):
    """TRANSFER should be encoded as 4."""
    result = _engineer(fraud_transaction, encoder=None)
    assert result["type_encoded"].iloc[0] == 4


def test_engineer_balance_diff(legit_transaction):
    """balance_diff_orig should be oldbalanceOrg - newbalanceOrig."""
    result = _engineer(legit_transaction, encoder=None)
    expected = 45_000.0 - 43_749.50
    assert abs(result["balance_diff_orig"].iloc[0] - expected) < 0.01


def test_engineer_round_amount():
    """is_round_amount should be 1 for multiples of 1000."""
    df = pd.DataFrame([{
        "type": "CASH_OUT", "amount": 5000.0,
        "oldbalanceOrg": 10000.0, "newbalanceOrig": 5000.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 5000.0,
    }])
    result = _engineer(df, encoder=None)
    assert result["is_round_amount"].iloc[0] == 1


def test_engineer_non_round_amount(legit_transaction):
    """is_round_amount should be 0 for non-round amounts."""
    result = _engineer(legit_transaction, encoder=None)
    assert result["is_round_amount"].iloc[0] == 0


def test_engineer_zero_balance():
    """zero_orig_balance should be 1 when oldbalanceOrg is 0."""
    df = pd.DataFrame([{
        "type": "CASH_IN", "amount": 1000.0,
        "oldbalanceOrg": 0.0, "newbalanceOrig": 1000.0,
        "oldbalanceDest": 50000.0, "newbalanceDest": 49000.0,
    }])
    result = _engineer(df, encoder=None)
    assert result["zero_orig_balance"].iloc[0] == 1


# ─── _risk_level() ────────────────────────────────────────────────────────────

def test_risk_level_low():
    assert _risk_level(0.10) == "LOW"
    assert _risk_level(0.29) == "LOW"


def test_risk_level_medium():
    assert _risk_level(0.30) == "MEDIUM"
    assert _risk_level(0.59) == "MEDIUM"


def test_risk_level_high():
    assert _risk_level(0.60) == "HIGH"
    assert _risk_level(0.99) == "HIGH"


def test_risk_level_boundaries():
    """Boundary values should map correctly."""
    assert _risk_level(0.0) == "LOW"
    assert _risk_level(1.0) == "HIGH"


# ─── _generate_flags() ────────────────────────────────────────────────────────

def test_flags_drained_balance(fraud_transaction):
    """Should flag 'Entire balance drained' for a full-drain transfer."""
    row = _engineer(fraud_transaction, encoder=None).iloc[0]
    flags = _generate_flags(row)
    assert "Entire balance drained" in flags


def test_flags_large_amount(fraud_transaction):
    """Should flag 'Unusually large amount' for amounts > 400k."""
    row = _engineer(fraud_transaction, encoder=None).iloc[0]
    flags = _generate_flags(row)
    assert "Unusually large amount" in flags


def test_flags_round_number():
    """Should flag 'Round number amount' for multiples of 1000."""
    df = pd.DataFrame([{
        "type": "TRANSFER", "amount": 500_000.0,
        "oldbalanceOrg": 500_000.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 500_000.0,
    }])
    row = _engineer(df, encoder=None).iloc[0]
    flags = _generate_flags(row)
    assert "Round number amount" in flags


def test_flags_legit_no_flags(legit_transaction):
    """A normal transaction should have no suspicious flags."""
    row = _engineer(legit_transaction, encoder=None).iloc[0]
    flags = _generate_flags(row)
    assert len(flags) == 0
