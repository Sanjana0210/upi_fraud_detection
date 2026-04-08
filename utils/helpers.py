"""
utils/helpers.py
Shared utility functions used across the UPI Fraud Detection project.

Exported:
    log()               - timestamped, colour-coded console logger
    load_config()       - loads .env file and returns a config dict
    get_mongo_client()  - returns a verified MongoClient (localhost:27017)
    get_db()            - returns the "upi_fraud_db" database handle
    timer               - decorator that prints how long a function took
    + feature-engineering helpers consumed by spark/, ml/, kafka/, dashboard/
"""

import os
import time
import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw", "paysim.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "paysim_processed.csv")
MODEL_PATH          = os.path.join(BASE_DIR, "ml", "fraud_model.pkl")
ENV_PATH            = os.path.join(BASE_DIR, ".env")

# ---------------------------------------------------------------------------
# 1. log()
# ---------------------------------------------------------------------------

_COLOURS: Dict[str, str] = {
    "INFO":    "\033[94m",   # blue
    "SUCCESS": "\033[92m",   # green
    "WARNING": "\033[93m",   # yellow
    "ERROR":   "\033[91m",   # red
    "DEBUG":   "\033[90m",   # grey
}
_RESET = "\033[0m"


def log(message: str, level: str = "INFO") -> None:
    """
    Print a timestamped, colour-coded log message to stdout.

    Args:
        message (str): Text to display.
        level   (str): Severity — INFO | SUCCESS | WARNING | ERROR | DEBUG.

    Example:
        >>> log("Model saved", "SUCCESS")
        [2026-03-07 12:00:00] SUCCESS  - Model saved
    """
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colour = _COLOURS.get(level.upper(), "")
    label  = level.upper().ljust(7)
    print(f"{colour}[{ts}] {label} - {message}{_RESET}")


# ---------------------------------------------------------------------------
# 2. load_config()
# ---------------------------------------------------------------------------

def load_config(env_path: str = ENV_PATH) -> Dict[str, str]:
    """
    Load environment variables from a .env file and return them as a dict.

    Falls back to the current process environment if .env is absent — no
    exception is raised in that case.

    Args:
        env_path (str): Absolute path to the .env file.

    Returns:
        dict[str, str]: Resolved configuration values.

    Raises:
        Exception: Re-raises any unexpected error during dotenv loading.

    Example:
        >>> cfg = load_config()
        >>> cfg["MONGO_URI"]
        'mongodb://localhost:27017/'
    """
    try:
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=False)
            log(f".env loaded from: {env_path}", "INFO")
        else:
            log(f".env not found at {env_path} — using process environment.", "WARNING")

        return {
            "MONGO_URI":      os.getenv("MONGO_URI",      "mongodb://localhost:27017/"),
            "MONGO_DB":       os.getenv("MONGO_DB",       "upi_fraud_db"),
            "KAFKA_BROKER":   os.getenv("KAFKA_BROKER",   "localhost:9092"),
            "KAFKA_TOPIC":    os.getenv("KAFKA_TOPIC",    "upi_transactions"),
            "KAFKA_GROUP_ID": os.getenv("KAFKA_GROUP_ID", "fraud_detection_group"),
            "PRODUCER_DELAY": os.getenv("PRODUCER_DELAY", "0.05"),
        }
    except Exception as exc:
        log(f"load_config() failed: {exc}", "ERROR")
        raise


# ---------------------------------------------------------------------------
# 3. get_mongo_client()
# ---------------------------------------------------------------------------

def get_mongo_client(uri: Optional[str] = None, timeout_ms: int = 3_000) -> MongoClient:
    """
    Create and return a MongoDB client connected to the given URI.

    Sends a lightweight ``ping`` to verify the connection before returning.
    Prints a success message on a successful connection.

    Args:
        uri        (str, optional): MongoDB connection URI.  Defaults to the
                                    ``MONGO_URI`` env-var or localhost:27017.
        timeout_ms (int):           Server-selection timeout in ms (default 3000).

    Returns:
        MongoClient: A live, verified MongoDB client.

    Raises:
        ConnectionFailure:            If the server is unreachable.
        ServerSelectionTimeoutError:  If the server does not respond in time.

    Example:
        >>> client = get_mongo_client()
        >>> client.list_database_names()
    """
    try:
        config = load_config()
        uri    = uri or config["MONGO_URI"]
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")           # raises if unreachable
        log(f"✅ MongoDB connected successfully → {uri}", "SUCCESS")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        log(f"MongoDB connection FAILED ({uri}): {exc}", "ERROR")
        raise
    except Exception as exc:
        log(f"Unexpected error in get_mongo_client(): {exc}", "ERROR")
        raise


# ---------------------------------------------------------------------------
# 4. get_db()
# ---------------------------------------------------------------------------

def get_db(db_name: Optional[str] = None) -> Database:
    """
    Return the ``upi_fraud_db`` MongoDB database handle.

    Calls ``get_mongo_client()`` internally to ensure the connection is live.

    Args:
        db_name (str, optional): Database name.  Defaults to ``"upi_fraud_db"``
                                 or the ``MONGO_DB`` environment variable.

    Returns:
        Database: A pymongo Database object.

    Raises:
        ConnectionFailure: Propagated from get_mongo_client() if unreachable.

    Example:
        >>> db = get_db()
        >>> db["transactions"].count_documents({})
    """
    try:
        config  = load_config()
        db_name = db_name or config.get("MONGO_DB", "upi_fraud_db")
        client  = get_mongo_client()
        db      = client[db_name]
        log(f"Using database: '{db_name}'", "INFO")
        return db
    except Exception as exc:
        log(f"get_db() failed for db='{db_name}': {exc}", "ERROR")
        raise


# ---------------------------------------------------------------------------
# 5. timer  (decorator)
# ---------------------------------------------------------------------------

def timer(func: Callable) -> Callable:
    """
    Decorator that measures and prints the wall-clock execution time of any
    wrapped function.

    On success  → logs at SUCCESS level: ``func() finished in X.XXXs``
    On exception → logs at ERROR level then re-raises the original exception.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: Wrapped function with identical signature.

    Example:
        >>> @timer
        ... def train():
        ...     time.sleep(2)
        >>> train()
        [2026-03-07 12:00:02] SUCCESS  - train() finished in 2.001s
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result  = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log(f"{func.__name__}() finished in {elapsed:.3f}s", "SUCCESS")
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log(f"{func.__name__}() FAILED after {elapsed:.3f}s — {exc}", "ERROR")
            raise
    return wrapper


# ---------------------------------------------------------------------------
# Feature-engineering helpers  (shared by spark/, ml/, kafka/, dashboard/)
# ---------------------------------------------------------------------------

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw PaySim CSV dataset from disk.

    Args:
        path (str): Path to paysim.csv.

    Returns:
        pd.DataFrame: All original columns.

    Raises:
        FileNotFoundError: If paysim.csv is not found at *path*.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Download paysim.csv from Kaggle and place it in data/raw/"
            )
        log(f"Loading raw data from: {path}", "INFO")
        df = pd.read_csv(path)
        log(f"Loaded {len(df):,} rows × {df.shape[1]} columns.", "SUCCESS")
        return df
    except FileNotFoundError:
        raise
    except Exception as exc:
        log(f"load_raw_data() failed: {exc}", "ERROR")
        raise


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Load the feature-engineered dataset produced by spark/process_data.py.

    Args:
        path (str): Path to the processed CSV.

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: If the processed file has not been generated yet.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Processed data not found: {path}\n"
                "Run spark/process_data.py first."
            )
        log(f"Loading processed data from: {path}", "INFO")
        df = pd.read_csv(path)
        log(f"Loaded {len(df):,} rows.", "SUCCESS")
        return df
    except FileNotFoundError:
        raise
    except Exception as exc:
        log(f"load_processed_data() failed: {exc}", "ERROR")
        raise


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode the ``type`` column.

    Mapping: CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4.

    Args:
        df (pd.DataFrame): Must contain a ``type`` column.

    Returns:
        pd.DataFrame: Copy of *df* with an added ``type_encoded`` column.
    """
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    df = df.copy()
    df["type_encoded"] = df["type"].map(type_map).fillna(-1).astype(int)
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engineered features used by the ML model.

    Added columns:
        - ``amount_log``         : log1p of transaction amount
        - ``balance_diff_orig``  : oldbalanceOrg  − newbalanceOrig
        - ``balance_diff_dest``  : newbalanceDest − oldbalanceDest
        - ``error_balance_orig`` : expected vs actual balance change (sender)
        - ``error_balance_dest`` : expected vs actual balance change (receiver)

    Args:
        df (pd.DataFrame): Raw PaySim DataFrame.

    Returns:
        pd.DataFrame: Copy of *df* with all derived columns appended.

    Raises:
        KeyError: If a required source column is absent.
    """
    try:
        df = encode_transaction_type(df)
        df = df.copy()
        df["amount_log"]         = np.log1p(df["amount"])
        df["balance_diff_orig"]  = df["oldbalanceOrg"]  - df["newbalanceOrig"]
        df["balance_diff_dest"]  = df["newbalanceDest"] - df["oldbalanceDest"]
        df["error_balance_orig"] = df["oldbalanceOrg"]  - df["amount"] - df["newbalanceOrig"]
        df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        return df
    except KeyError as exc:
        log(f"feature_engineer() — missing column: {exc}", "ERROR")
        raise


FEATURE_COLS: list = [
    "type_encoded",
    "amount",
    "amount_log",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_diff_orig",
    "balance_diff_dest",
    "error_balance_orig",
    "error_balance_dest",
]
TARGET_COL: str = "isFraud"


def get_features_and_target(df: pd.DataFrame):
    """
    Apply feature engineering and split into X / y.

    Args:
        df (pd.DataFrame): Raw PaySim DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y) feature matrix and target.
    """
    df = feature_engineer(df)
    return df[FEATURE_COLS], df[TARGET_COL]


def timestamp_to_str(ts: Optional[datetime] = None) -> str:
    """
    Return a compact timestamp string for use in file-name suffixes.

    Args:
        ts (datetime, optional): Defaults to ``datetime.now()``.

    Returns:
        str: Formatted as ``YYYYMMDD_HHMMSS``.
    """
    ts = ts or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Self-test  →  python utils/helpers.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. log — exercise all levels
    log("Testing INFO level",    "INFO")
    log("Testing WARNING level", "WARNING")
    log("Testing ERROR level",   "ERROR")
    log("Testing DEBUG level",   "DEBUG")

    # 2. load_config
    cfg = load_config()
    log(f"Config: {cfg}", "INFO")

    # 3. timer decorator
    @timer
    def dummy_work(n: int) -> int:
        """Simulate CPU work."""
        return sum(range(n))

    log(f"dummy_work result: {dummy_work(1_000_000)}", "INFO")

    # 4. MongoDB  (only works when mongod is running)
    try:
        client = get_mongo_client()
        db     = get_db()
        log(f"Collections in '{db.name}': {db.list_collection_names()}", "INFO")
        client.close()
    except Exception as e:
        log(f"MongoDB not running — skipping connection test: {e}", "WARNING")

    log("✅ helpers.py self-test complete.", "SUCCESS")
