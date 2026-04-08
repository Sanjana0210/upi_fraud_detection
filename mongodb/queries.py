"""
mongodb/queries.py
Reusable analytical queries against the MongoDB transactions collection.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_mongo_client, log

DB_NAME = "upi_fraud_db"
COLLECTION_NAME = "transactions"


def get_collection():
    client = get_mongo_client()
    col = client[DB_NAME][COLLECTION_NAME]
    return col, client


# ─── Basic Counts ─────────────────────────────────────────────────────────────

def count_total():
    col, client = get_collection()
    result = col.count_documents({})
    client.close()
    log(f"Total transactions: {result:,}", "INFO")
    return result


def count_fraud():
    col, client = get_collection()
    result = col.count_documents({"isFraud": 1})
    client.close()
    log(f"Fraud transactions: {result:,}", "INFO")
    return result


def fraud_by_type() -> pd.DataFrame:
    """Return fraud count grouped by transaction type."""
    col, client = get_collection()
    pipeline = [
        {"$group": {
            "_id": "$type",
            "total": {"$sum": 1},
            "fraud_count": {"$sum": "$isFraud"},
        }},
        {"$sort": {"fraud_count": -1}},
    ]
    result = list(col.aggregate(pipeline))
    client.close()
    df = pd.DataFrame(result).rename(columns={"_id": "type"})
    df["fraud_rate_%"] = (df["fraud_count"] / df["total"] * 100).round(2)
    return df


def avg_transaction_amount_by_fraud() -> pd.DataFrame:
    """Return average transaction amount split by fraud/non-fraud."""
    col, client = get_collection()
    pipeline = [
        {"$group": {
            "_id": "$isFraud",
            "avg_amount": {"$avg": "$amount"},
            "count": {"$sum": 1},
        }},
    ]
    result = list(col.aggregate(pipeline))
    client.close()
    df = pd.DataFrame(result).rename(columns={"_id": "isFraud"})
    df["isFraud"] = df["isFraud"].map({0: "Legit", 1: "Fraud"})
    return df


def fraud_over_time() -> pd.DataFrame:
    """Fraud count per simulation step (time unit)."""
    col, client = get_collection()
    pipeline = [
        {"$match": {"isFraud": 1}},
        {"$group": {"_id": "$step", "fraud_count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    result = list(col.aggregate(pipeline))
    client.close()
    df = pd.DataFrame(result).rename(columns={"_id": "step"})
    return df


def top_fraud_senders(n: int = 10) -> pd.DataFrame:
    """Top N accounts that sent the most fraudulent transactions."""
    col, client = get_collection()
    pipeline = [
        {"$match": {"isFraud": 1}},
        {"$group": {"_id": "$nameOrig", "fraud_count": {"$sum": 1}, "total_amount": {"$sum": "$amount"}}},
        {"$sort": {"fraud_count": -1}},
        {"$limit": n},
    ]
    result = list(col.aggregate(pipeline))
    client.close()
    df = pd.DataFrame(result).rename(columns={"_id": "nameOrig"})
    return df


def sample_transactions(limit: int = 100, only_fraud: bool = False) -> pd.DataFrame:
    """Fetch a sample of transactions as a DataFrame."""
    col, client = get_collection()
    query = {"isFraud": 1} if only_fraud else {}
    cursor = col.find(query, {"_id": 0}).limit(limit)
    df = pd.DataFrame(list(cursor))
    client.close()
    return df


if __name__ == "__main__":
    print("Total:", count_total())
    print("Fraud:", count_fraud())
    print("\nFraud by type:\n", fraud_by_type())
    print("\nAvg amount by fraud:\n", avg_transaction_amount_by_fraud())
