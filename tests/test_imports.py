"""
tests/test_imports.py
Smoke tests — verify all project modules import without errors.
"""

import pytest


def test_import_utils_helpers():
    """utils.helpers should import successfully."""
    from utils.helpers import log, load_config, timer, BASE_DIR
    assert callable(log)
    assert callable(load_config)
    assert callable(timer)
    assert isinstance(BASE_DIR, str)


def test_import_ml_train():
    """ml.train_model should import successfully."""
    from ml.train_model import load_data, engineer_features, prepare, FEATURES, TARGET
    assert callable(load_data)
    assert callable(engineer_features)
    assert callable(prepare)
    assert isinstance(FEATURES, list)
    assert isinstance(TARGET, str)


def test_import_ml_predict():
    """ml.predict should import successfully."""
    from ml.predict import predict_transaction, batch_predict, FEATURES
    assert callable(predict_transaction)
    assert callable(batch_predict)
    assert isinstance(FEATURES, list)


def test_import_ml_evaluate():
    """ml.evaluate should import successfully."""
    from ml.evaluate import evaluate
    assert callable(evaluate)


def test_import_spark_process():
    """spark.process_data should import successfully."""
    import spark.process_data
    assert spark.process_data is not None
