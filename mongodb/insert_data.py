"""
mongodb/insert_data.py

Reads data/raw/paysim.csv in 50 000-row chunks and inserts into three
MongoDB collections inside "upi_fraud_db":

    transactions        – every row (all columns)
    fraud_cases         – only rows where isFraud == 1
    transaction_summary – aggregated stats per transaction type

Indexes are created before any data is written.
If a collection already contains documents, insertion is skipped.
A text progress bar and a final summary are printed to stdout.
"""

import os
import sys
import time
from typing import Optional

import pandas as pd
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError

# ---------------------------------------------------------------------------
# Project root on sys.path so "utils" is importable from anywhere
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_mongo_client, log, RAW_DATA_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_NAME    = "upi_fraud_db"
CHUNK_SIZE = 50_000

PAYSIM_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_progress(chunk_num: int, total_chunks: int) -> None:
    """Print a plain-text progress bar without tqdm."""
    pct      = int(chunk_num / total_chunks * 100)
    filled   = pct // 5                       # 20 blocks = 100 %
    bar      = "#" * filled + "-" * (20 - filled)
    print(
        f"  Inserting chunk {chunk_num:>{len(str(total_chunks))}}/{total_chunks}"
        f"  [{bar}]  {pct:3d}% complete",
        flush=True,
    )


def _count_csv_rows(path: str) -> int:
    """Count total data rows in the CSV (fast line count minus header)."""
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1          # subtract header


def _collection_has_data(db, name: str) -> bool:
    """Return True if the collection exists and contains at least one doc."""
    return name in db.list_collection_names() and db[name].count_documents({}) > 0


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

def create_indexes(db) -> None:
    """
    Create indexes on transactions and fraud_cases collections.

    transactions : nameOrig, nameDest, isFraud, type, step  (all ASCENDING)
    fraud_cases  : amount (DESCENDING)
    """
    txn = db["transactions"]
    txn.create_index([("nameOrig",  ASCENDING)], background=True)
    txn.create_index([("nameDest",  ASCENDING)], background=True)
    txn.create_index([("isFraud",   ASCENDING)], background=True)
    txn.create_index([("type",      ASCENDING)], background=True)
    txn.create_index([("step",      ASCENDING)], background=True)
    log("Indexes created on 'transactions'.", "SUCCESS")

    fraud = db["fraud_cases"]
    fraud.create_index([("amount", DESCENDING)], background=True)
    log("Index created on 'fraud_cases'.", "SUCCESS")


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def build_transaction_summary(db) -> None:
    """
    Aggregate per-type stats from the transactions collection and write
    them into transaction_summary.

    Each document: { type, total_count, fraud_count, fraud_rate, avg_amount, max_amount }
    """
    log("Building transaction_summary...", "INFO")
    pipeline = [
        {
            "$group": {
                "_id":         "$type",
                "total_count": {"$sum": 1},
                "fraud_count": {"$sum": "$isFraud"},
                "avg_amount":  {"$avg": "$amount"},
                "max_amount":  {"$max": "$amount"},
            }
        },
        {
            "$project": {
                "_id":         0,
                "type":        "$_id",
                "total_count": 1,
                "fraud_count": 1,
                "fraud_rate":  {
                    "$round": [
                        {"$multiply": [
                            {"$divide": ["$fraud_count", "$total_count"]},
                            100
                        ]},
                        4
                    ]
                },
                "avg_amount":  {"$round": ["$avg_amount", 2]},
                "max_amount":  1,
            }
        },
        {"$sort": {"type": 1}},
    ]

    summary_col = db["transaction_summary"]
    summary_col.drop()                         # always rebuild from live data
    results = list(db["transactions"].aggregate(pipeline, allowDiskUse=True))
    if results:
        summary_col.insert_many(results)
        log(f"transaction_summary: {len(results)} type-level documents written.", "SUCCESS")
    else:
        log("No data in transactions to summarise.", "WARNING")


# ---------------------------------------------------------------------------
# Main insertion routine
# ---------------------------------------------------------------------------

def insert_data(csv_path: str = RAW_DATA_PATH) -> None:
    """
    Full pipeline:
        1. Connect to MongoDB "upi_fraud_db"
        2. Skip already-populated collections
        3. Create indexes
        4. Stream-insert transactions + fraud_cases in 50 000-row chunks
        5. Build transaction_summary via aggregation
        6. Print final summary

    Args:
        csv_path (str): Path to paysim.csv. Defaults to data/raw/paysim.csv.

    Raises:
        FileNotFoundError : If paysim.csv does not exist at csv_path.
        Exception         : Re-raised on any unrecoverable MongoDB error.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Download paysim.csv from Kaggle and place it in data/raw/"
        )

    # ── Connect ──────────────────────────────────────────────────────────────
    client = get_mongo_client()
    db     = client[DB_NAME]

    # ── Skip check ───────────────────────────────────────────────────────────
    txn_exists   = _collection_has_data(db, "transactions")
    fraud_exists = _collection_has_data(db, "fraud_cases")

    if txn_exists and fraud_exists:
        print("Data already exists, skipping...")
        log("Both 'transactions' and 'fraud_cases' already populated. Skipping.", "WARNING")
        client.close()
        return

    # ── Indexes (create before inserting for efficiency) ─────────────────────
    create_indexes(db)

    # ── Count total rows for progress reporting ───────────────────────────────
    log(f"Counting rows in {csv_path} ...", "INFO")
    total_rows   = _count_csv_rows(csv_path)
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    log(f"Total rows: {total_rows:,}  |  Chunks: {total_chunks}", "INFO")

    # ── Stream-insert ─────────────────────────────────────────────────────────
    txn_col   = db["transactions"]
    fraud_col = db["fraud_cases"]

    total_inserted = 0
    total_fraud    = 0
    chunk_num      = 0
    start_time     = time.perf_counter()

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            usecols=PAYSIM_COLUMNS,
            dtype={
                "step":            int,
                "amount":          float,
                "oldbalanceOrg":   float,
                "newbalanceOrig":  float,
                "oldbalanceDest":  float,
                "newbalanceDest":  float,
                "isFraud":         int,
                "isFlaggedFraud":  int,
            },
        )

        for chunk in reader:
            chunk_num += 1
            _print_progress(chunk_num, total_chunks)

            # Replace NaN with None for MongoDB compatibility
            chunk = chunk.where(pd.notnull(chunk), None)
            records = chunk.to_dict(orient="records")

            # ── transactions ─────────────────────────────────────────────────
            if not txn_exists:
                try:
                    txn_col.insert_many(records, ordered=False)
                    total_inserted += len(records)
                except BulkWriteError as bwe:
                    inserted_ok = bwe.details.get("nInserted", 0)
                    total_inserted += inserted_ok
                    log(f"Chunk {chunk_num}: partial write ({inserted_ok} ok) — {bwe.details['writeErrors'][0]['errmsg']}", "WARNING")

            # ── fraud_cases ──────────────────────────────────────────────────
            if not fraud_exists:
                fraud_records = [r for r in records if r.get("isFraud") == 1]
                if fraud_records:
                    try:
                        fraud_col.insert_many(fraud_records, ordered=False)
                        total_fraud += len(fraud_records)
                    except BulkWriteError as bwe:
                        inserted_ok = bwe.details.get("nInserted", 0)
                        total_fraud += inserted_ok
                        log(f"Chunk {chunk_num} fraud partial write: {inserted_ok} ok", "WARNING")

    except Exception as exc:
        log(f"Fatal error during insertion at chunk {chunk_num}: {exc}", "ERROR")
        client.close()
        raise

    # ── transaction_summary ───────────────────────────────────────────────────
    build_transaction_summary(db)

    elapsed = time.perf_counter() - start_time

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print(f"  Total documents inserted : {total_inserted:,}")
    print(f"  Total fraud cases        : {total_fraud:,}")
    print(f"  Insertion time           : {elapsed:.2f} seconds")
    print("=" * 50)
    log("insert_data() complete.", "SUCCESS")

    client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    insert_data()
