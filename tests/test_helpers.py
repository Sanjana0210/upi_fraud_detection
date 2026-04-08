"""
tests/test_helpers.py
Unit tests for utils/helpers.py — pure functions that don't need external services.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from utils.helpers import (
    log,
    load_config,
    timer,
    encode_transaction_type,
    feature_engineer,
    timestamp_to_str,
    FEATURE_COLS,
    TARGET_COL,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A minimal PaySim-like DataFrame for testing feature engineering."""
    return pd.DataFrame({
        "type":           ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"],
        "amount":         [10000.0,    250.0,     50000.0,    3000.0,    1500.0],
        "oldbalanceOrg":  [10000.0,    50000.0,   60000.0,    0.0,       8000.0],
        "newbalanceOrig": [0.0,        49750.0,   10000.0,    3000.0,    6500.0],
        "oldbalanceDest": [0.0,        0.0,       100000.0,   50000.0,   0.0],
        "newbalanceDest": [10000.0,    0.0,       150000.0,   47000.0,   0.0],
        "isFraud":        [1,          0,         0,          0,         0],
    })


# ─── log() ────────────────────────────────────────────────────────────────────

def test_log_does_not_raise(capsys):
    """log() should print to stdout without raising."""
    log("test message", "INFO")
    log("warning message", "WARNING")
    log("error message", "ERROR")
    captured = capsys.readouterr()
    assert "test message" in captured.out
    assert "warning message" in captured.out
    assert "error message" in captured.out


def test_log_unknown_level(capsys):
    """log() should handle unknown levels gracefully."""
    log("custom level", "CUSTOM")
    captured = capsys.readouterr()
    assert "custom level" in captured.out


# ─── load_config() ────────────────────────────────────────────────────────────

def test_load_config_returns_dict():
    """load_config() should return a dict with expected keys."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    expected_keys = {"MONGO_URI", "MONGO_DB", "KAFKA_BROKER", "KAFKA_TOPIC",
                     "KAFKA_GROUP_ID", "PRODUCER_DELAY"}
    assert expected_keys.issubset(set(cfg.keys()))


def test_load_config_defaults():
    """Default values should be sensible localhost addresses."""
    cfg = load_config()
    assert "localhost" in cfg["MONGO_URI"] or "mongo" in cfg["MONGO_URI"]
    assert cfg["KAFKA_TOPIC"] == "upi_transactions"


# ─── timer() ──────────────────────────────────────────────────────────────────

def test_timer_returns_result():
    """Decorated function should still return its result."""
    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_timer_preserves_name():
    """timer() should preserve the original function name via functools.wraps."""
    @timer
    def my_function():
        pass

    assert my_function.__name__ == "my_function"


# ─── encode_transaction_type() ─────────────────────────────────────────────────

def test_encode_transaction_type(sample_df):
    """Should add type_encoded column with correct integer mapping."""
    result = encode_transaction_type(sample_df)
    assert "type_encoded" in result.columns
    mapping = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    for idx, row in result.iterrows():
        assert row["type_encoded"] == mapping[row["type"]]


def test_encode_does_not_mutate(sample_df):
    """encode_transaction_type() should not modify the input DataFrame."""
    original = sample_df.copy()
    encode_transaction_type(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


# ─── feature_engineer() ───────────────────────────────────────────────────────

def test_feature_engineer_adds_columns(sample_df):
    """feature_engineer() should add all expected derived columns."""
    result = feature_engineer(sample_df)
    for col in ["amount_log", "balance_diff_orig", "balance_diff_dest",
                "error_balance_orig", "error_balance_dest", "type_encoded"]:
        assert col in result.columns, f"Missing column: {col}"


def test_feature_engineer_balance_diff(sample_df):
    """balance_diff_orig should be oldbalanceOrg - newbalanceOrig."""
    result = feature_engineer(sample_df)
    expected = sample_df["oldbalanceOrg"] - sample_df["newbalanceOrig"]
    np.testing.assert_array_almost_equal(result["balance_diff_orig"].values,
                                         expected.values)


def test_feature_engineer_amount_log(sample_df):
    """amount_log should equal np.log1p(amount)."""
    result = feature_engineer(sample_df)
    expected = np.log1p(sample_df["amount"])
    np.testing.assert_array_almost_equal(result["amount_log"].values,
                                         expected.values)


# ─── timestamp_to_str() ───────────────────────────────────────────────────────

def test_timestamp_to_str_format():
    """Should return YYYYMMDD_HHMMSS format."""
    ts = datetime(2026, 4, 8, 15, 30, 45)
    assert timestamp_to_str(ts) == "20260408_153045"


def test_timestamp_to_str_default():
    """Calling with no argument should return a string of length 15."""
    result = timestamp_to_str()
    assert len(result) == 15  # YYYYMMDD_HHMMSS
    assert result[8] == "_"


# ─── Constants ─────────────────────────────────────────────────────────────────

def test_feature_cols_is_list():
    """FEATURE_COLS should be a non-empty list of strings."""
    assert isinstance(FEATURE_COLS, list)
    assert len(FEATURE_COLS) > 0
    assert all(isinstance(c, str) for c in FEATURE_COLS)


def test_target_col():
    """TARGET_COL should be 'isFraud'."""
    assert TARGET_COL == "isFraud"
