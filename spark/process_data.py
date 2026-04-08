"""
spark/process_data.py

PySpark pipeline that:
  1. Reads the "transactions" collection from MongoDB (upi_fraud_db)
  2. Runs 5 full analyses and writes each result back to MongoDB
  3. Saves a cleaned, label-encoded Parquet file for ML training

---------------------------------------------------------------------------
Run via spark-submit (recommended):
    spark-submit \\
        --packages org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 \\
        spark/process_data.py

Or plain Python (PYSPARK_SUBMIT_ARGS set automatically below):
    python spark/process_data.py
---------------------------------------------------------------------------
"""

import os
import sys

# ---------------------------------------------------------------------------
# Set MongoDB Spark Connector package BEFORE PySpark is imported so that
# both `python script.py` and `spark-submit` work without extra flags.
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--packages org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 pyspark-shell",
)

from pyspark.sql import SparkSession          # noqa: E402
from pyspark.sql import functions as F        # noqa: E402
from pyspark.sql.types import IntegerType     # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, BASE_DIR       # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONGO_URI   = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME     = "upi_fraud_db"
PARQUET_OUT = os.path.join(BASE_DIR, "data", "processed", "cleaned.parquet")

_MONGO_READ_URI  = f"{MONGO_URI.rstrip('/')}/{DB_NAME}.transactions"
_MONGO_WRITE_URI = f"{MONGO_URI.rstrip('/')}/{DB_NAME}"


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

def build_spark_session() -> SparkSession:
    """
    Initialise SparkSession with MongoDB connector config.

    Config:
        - App name         : UPI_Fraud_Detection
        - MongoDB read URI : localhost:27017/upi_fraud_db.transactions
        - driver memory    : 4g
        - executor memory  : 2g
        - shuffle partitions: 8  (suitable for a single-node dev machine)
    """
    spark = (
        SparkSession.builder
        .appName("UPI_Fraud_Detection")
        # MongoDB Spark Connector 10.x read / write URIs
        .config("spark.mongodb.read.connection.uri",  _MONGO_READ_URI)
        .config("spark.mongodb.write.connection.uri", _MONGO_WRITE_URI)
        # Memory
        .config("spark.driver.memory",  "4g")
        .config("spark.executor.memory", "2g")
        # Tuning
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    log("SparkSession initialised.", "SUCCESS")
    return spark


# ---------------------------------------------------------------------------
# MongoDB I/O helpers
# ---------------------------------------------------------------------------

def read_mongo(spark: SparkSession, collection: str):
    """Read a MongoDB collection into a Spark DataFrame."""
    df = (
        spark.read.format("mongodb")
        .option("uri",        MONGO_URI)
        .option("database",   DB_NAME)
        .option("collection", collection)
        .load()
    )
    # Drop the ObjectId _id column — not serialisable to Parquet / other sinks
    if "_id" in df.columns:
        df = df.drop("_id")
    return df


def write_mongo(df, collection: str) -> None:
    """Overwrite a MongoDB collection with a Spark DataFrame."""
    (
        df.write.format("mongodb")
        .mode("overwrite")
        .option("uri",        MONGO_URI)
        .option("database",   DB_NAME)
        .option("collection", collection)
        .save()
    )
    log(f"Written to MongoDB collection '{collection}'.", "SUCCESS")


# ---------------------------------------------------------------------------
# Analysis 1 — Transaction Type Stats
# ---------------------------------------------------------------------------

def analysis_type_stats(df):
    """
    Per transaction type:
        count, sum/avg/min/max amount, fraud_count, fraud_percentage

    Saved to: spark_type_stats
    """
    log("Analysis 1 — Transaction Type Stats ...", "INFO")

    result = df.groupBy("type").agg(
        F.count("*")               .alias("count"),
        F.round(F.sum("amount"), 2).alias("total_amount"),
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.min("amount"), 2).alias("min_amount"),
        F.round(F.max("amount"), 2).alias("max_amount"),
        F.sum("isFraud")           .alias("fraud_count"),
    ).withColumn(
        "fraud_percentage",
        F.round(F.col("fraud_count") / F.col("count") * 100, 4),
    ).orderBy("type")

    result.show(10, truncate=False)
    write_mongo(result, "spark_type_stats")
    return result


# ---------------------------------------------------------------------------
# Analysis 2 — Hourly (step) Pattern
# ---------------------------------------------------------------------------

def analysis_hourly_pattern(df):
    """
    Per simulation step (hour):
        total_transactions, fraud_count, fraud_rate, is_peak_fraud_hour flag
        (peak = fraud_rate in top 10 % of all steps)

    Saved to: spark_hourly_stats
    """
    log("Analysis 2 — Hourly Pattern ...", "INFO")

    hourly = df.groupBy("step").agg(
        F.count("*")  .alias("total_transactions"),
        F.sum("isFraud").alias("fraud_count"),
    ).withColumn(
        "fraud_rate",
        F.round(F.col("fraud_count") / F.col("total_transactions") * 100, 4),
    ).orderBy("step")

    # Mark peak fraud hours: fraud_rate >= 90th-percentile fraud_rate
    threshold = hourly.approxQuantile("fraud_rate", [0.90], 0.01)[0]
    result = hourly.withColumn(
        "is_peak_fraud_hour",
        F.when(F.col("fraud_rate") >= threshold, True).otherwise(False),
    )

    result.show(10, truncate=False)
    write_mongo(result, "spark_hourly_stats")
    return result


# ---------------------------------------------------------------------------
# Analysis 3 — High-Risk Senders
# ---------------------------------------------------------------------------

def analysis_high_risk_senders(df):
    """
    Top 50 nameOrig accounts by fraud transaction count, with total fraud amount.

    Saved to: spark_high_risk_senders
    """
    log("Analysis 3 — High-Risk Senders (top 50) ...", "INFO")

    result = (
        df.filter(F.col("isFraud") == 1)
        .groupBy("nameOrig").agg(
            F.count("*")               .alias("fraud_count"),
            F.round(F.sum("amount"), 2).alias("total_fraud_amount"),
            F.round(F.avg("amount"), 2).alias("avg_fraud_amount"),
        )
        .orderBy(F.desc("fraud_count"))
        .limit(50)
    )

    result.show(10, truncate=False)
    write_mongo(result, "spark_high_risk_senders")
    return result


# ---------------------------------------------------------------------------
# Analysis 4 — Amount Distribution (bucketed)
# ---------------------------------------------------------------------------

def analysis_amount_distribution(df):
    """
    Bucket transaction amounts into ranges, count fraud vs legitimate per bucket.

    Buckets: 0-1K | 1K-10K | 10K-100K | 100K-500K | 500K+

    Saved to: spark_amount_distribution
    """
    log("Analysis 4 — Amount Distribution ...", "INFO")

    bucketed = df.withColumn(
        "amount_bucket",
        F.when(F.col("amount") <   1_000, "0-1K")
         .when(F.col("amount") <  10_000, "1K-10K")
         .when(F.col("amount") < 100_000, "10K-100K")
         .when(F.col("amount") < 500_000, "100K-500K")
         .otherwise("500K+"),
    )

    result = (
        bucketed.groupBy("amount_bucket", "isFraud").agg(
            F.count("*").alias("count"),
        )
        .withColumn(
            "label",
            F.when(F.col("isFraud") == 1, "Fraud").otherwise("Legitimate"),
        )
        .drop("isFraud")
        .orderBy("amount_bucket", "label")
    )

    result.show(20, truncate=False)
    write_mongo(result, "spark_amount_distribution")
    return result


# ---------------------------------------------------------------------------
# Analysis 5 — Balance Anomaly / Drain Transactions
# ---------------------------------------------------------------------------

def analysis_drain_transactions(df):
    """
    Transactions where the sender's balance is fully drained (newbalanceOrig == 0)
    AND the transaction is fraudulent — strong signal of account takeover.

    Saved to: spark_drain_transactions
    """
    log("Analysis 5 — Balance Anomaly / Drain Transactions ...", "INFO")

    result = df.filter(
        (F.col("newbalanceOrig") == 0) & (F.col("isFraud") == 1)
    ).select(
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFlaggedFraud",
    )

    drain_count = result.count()
    log(f"Drain transactions found: {drain_count:,}", "INFO")
    result.show(10, truncate=False)
    write_mongo(result, "spark_drain_transactions")
    return result


# ---------------------------------------------------------------------------
# Parquet Export — cleaned + label-encoded
# ---------------------------------------------------------------------------

def save_cleaned_parquet(df) -> None:
    """
    Drop PII columns (nameOrig, nameDest), label-encode 'type', save Parquet.

    type encoding: CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4

    Saved to: data/processed/cleaned.parquet
    """
    log("Saving cleaned Parquet ...", "INFO")

    os.makedirs(os.path.dirname(PARQUET_OUT), exist_ok=True)

    cleaned = (
        df.drop("nameOrig", "nameDest")
        .withColumn(
            "type_encoded",
            F.when(F.col("type") == "CASH_IN",  0)
             .when(F.col("type") == "CASH_OUT",  1)
             .when(F.col("type") == "DEBIT",     2)
             .when(F.col("type") == "PAYMENT",   3)
             .when(F.col("type") == "TRANSFER",  4)
             .otherwise(-1)
             .cast(IntegerType()),
        )
        .drop("type")
    )

    cleaned.write.mode("overwrite").parquet(PARQUET_OUT)
    log(f"Cleaned Parquet saved → {PARQUET_OUT}", "SUCCESS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    """Run all analyses end-to-end."""
    spark = build_spark_session()

    log("Reading 'transactions' from MongoDB ...", "INFO")
    df = read_mongo(spark, "transactions")

    total = df.count()
    log(f"Loaded {total:,} rows from MongoDB.", "SUCCESS")

    # Cache — re-used by all 5 analyses
    df.cache()

    analysis_type_stats(df)
    analysis_hourly_pattern(df)
    analysis_high_risk_senders(df)
    analysis_amount_distribution(df)
    analysis_drain_transactions(df)
    save_cleaned_parquet(df)

    df.unpersist()
    spark.stop()
    log("All analyses complete. Spark stopped.", "SUCCESS")


if __name__ == "__main__":
    run()
