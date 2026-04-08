"""
kafka/producer.py
Simulate a real-time UPI transaction stream by reading
data/processed/cleaned.parquet and publishing each row as a JSON
message to the Kafka topic "upi-transactions".
"""

import os
import sys
import json
import time
import uuid
import sysconfig
from datetime import datetime, timezone

import pandas as pd

# ── The project has a local kafka/ folder that shadows the installed
#    kafka-python package. Fix: put site-packages at the front of sys.path
#    so Python finds the real package first.
_sp = sysconfig.get_path("purelib")
if _sp and _sp not in sys.path:
    sys.path.insert(0, _sp)

from kafka import KafkaProducer, KafkaAdminClient          # noqa: E402
from kafka.admin import NewTopic                           # noqa: E402
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, BASE_DIR

# ─── Config ───────────────────────────────────────────────────────────────────
KAFKA_BROKER  = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC         = os.getenv("KAFKA_TOPIC",  "upi-transactions")
DELAY_SECONDS = float(os.getenv("PRODUCER_DELAY", "0.05"))   # 20 msg/s — easy on CPU
MAX_MESSAGES  = int(os.getenv("MAX_MESSAGES", "100000"))      # stop after 100K (enough for demo)
PARQUET_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned.parquet")

# type_encoded -> readable name (mirrors Spark encoding)
_TYPE_NAMES = {0: "CASH_IN", 1: "CASH_OUT", 2: "DEBIT", 3: "PAYMENT", 4: "TRANSFER"}
_MAX_RETRIES = 3


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ensure_topic(broker: str, topic: str) -> None:
    """Create the Kafka topic if it does not already exist."""
    try:
        admin = KafkaAdminClient(bootstrap_servers=[broker], request_timeout_ms=5000)
        admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
        log(f"Topic '{topic}' created.", "SUCCESS")
        admin.close()
    except TopicAlreadyExistsError:
        log(f"Topic '{topic}' already exists — skipping creation.", "INFO")
    except Exception as e:
        log(f"Could not verify/create topic: {e}", "WARNING")


def _make_producer(broker: str) -> KafkaProducer:
    """Connect to Kafka with up to _MAX_RETRIES attempts."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=3,
                request_timeout_ms=10_000,
            )
            log(f"Connected to Kafka broker at {broker}.", "SUCCESS")
            return producer
        except NoBrokersAvailable:
            log(f"Attempt {attempt}/{_MAX_RETRIES} — broker not reachable at {broker}.", "WARNING")
            if attempt < _MAX_RETRIES:
                log("Retrying in 3 seconds…", "INFO")
                time.sleep(3)

    log(f"Could not connect to Kafka after {_MAX_RETRIES} attempts. Exiting.", "ERROR")
    sys.exit(1)


# ─── Main streaming loop ──────────────────────────────────────────────────────
def stream(broker: str = KAFKA_BROKER, topic: str = TOPIC, delay: float = DELAY_SECONDS,
           max_messages: int = MAX_MESSAGES):
    # 1. Load parquet
    if not os.path.exists(PARQUET_PATH):
        log(f"Parquet not found at {PARQUET_PATH}. Run spark/process_data.py first.", "ERROR")
        sys.exit(1)

    log(f"Loading parquet → {PARQUET_PATH}", "INFO")
    df = pd.read_parquet(PARQUET_PATH)

    # 2. Shuffle so fraud appears randomly throughout the stream
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    log(f"Loaded {len(df):,} rows (shuffled). Fraud rows: {df['isFraud'].sum():,}", "INFO")

    # 3. Connect + ensure topic exists
    _ensure_topic(broker, topic)
    producer = _make_producer(broker)

    # 4. Stream
    total_sent = fraud_count = legit_count = 0

    log(f"Starting stream → topic='{topic}'  delay={delay}s per message", "INFO")
    log("Press Ctrl+C to stop.\n", "INFO")

    try:
        for _, row in df.iterrows():
            txn_id   = str(uuid.uuid4())
            is_fraud = int(row["isFraud"])
            type_name = _TYPE_NAMES.get(int(row.get("type_encoded", -1)), "UNKNOWN")

            message = {
                "transaction_id": txn_id,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "type":           type_name,
                "amount":         round(float(row["amount"]), 2),
                "oldbalanceOrg":  round(float(row["oldbalanceOrg"]), 2),
                "newbalanceOrig": round(float(row["newbalanceOrig"]), 2),
                "oldbalanceDest": round(float(row["oldbalanceDest"]), 2),
                "newbalanceDest": round(float(row["newbalanceDest"]), 2),
                "isFraud":        is_fraud,
                "isFlaggedFraud": int(row.get("isFlaggedFraud", 0)),
            }

            producer.send(topic, value=message)
            total_sent += 1

            if total_sent >= max_messages:
                log(f"Reached MAX_MESSAGES limit ({max_messages:,}). Stopping.", "SUCCESS")
                break

            if is_fraud:
                fraud_count += 1
                status = "🚨 FRAUD"
            else:
                legit_count += 1
                status = "✅ LEGIT"

            print(
                f"📤 Sent transaction {txn_id[:8]}… | "
                f"Type: {type_name:<8} | "
                f"Amount: ₹{message['amount']:>12,.2f} | "
                f"Actual: {status}"
            )

            # Summary every 100 messages
            if total_sent % 100 == 0:
                print(
                    f"\n📊 Sent: {total_sent:,} | "
                    f"Fraud sent: {fraud_count:,} | "
                    f"Legit sent: {legit_count:,}\n"
                )

            time.sleep(delay)

    except KeyboardInterrupt:
        log("\nStream interrupted by user.", "WARNING")

    finally:
        producer.flush()
        producer.close()
        log(
            f"Stream ended — Total: {total_sent:,} | "
            f"Fraud: {fraud_count:,} | Legit: {legit_count:,}",
            "SUCCESS",
        )


if __name__ == "__main__":
    stream()
