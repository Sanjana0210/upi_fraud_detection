"""
kafka/consumer.py
Consumes UPI transaction messages from Kafka topic "upi-transactions",
runs the fraud model, prints results, saves fraud alerts to MongoDB,
and emits a live stats summary every 60 seconds.
"""

import os
import sys
import json
import time
import sysconfig
from datetime import datetime, timezone

# Limit sklearn/joblib parallel threads — prevents maxing out all CPU cores
os.environ.setdefault("LOKY_MAX_CPU_THREADS", "2")

# ── site-packages fix: prevent local kafka/ folder from shadowing kafka-python
_sp = sysconfig.get_path("purelib")
if _sp and _sp not in sys.path:
    sys.path.insert(0, _sp)

from kafka import KafkaConsumer                                      # noqa: E402
from kafka.errors import NoBrokersAvailable, KafkaError              # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import log, get_db, BASE_DIR                      # noqa: E402

# Append ml/ so predict.py can be imported directly
sys.path.append(os.path.join(BASE_DIR, "ml"))
from predict import predict_transaction                               # noqa: E402

# ─── Config ───────────────────────────────────────────────────────────────────
KAFKA_BROKER   = os.getenv("KAFKA_BROKER",    "localhost:9092")
TOPIC          = os.getenv("KAFKA_TOPIC",     "upi-transactions")
GROUP_ID       = os.getenv("KAFKA_GROUP_ID",  "fraud_detection_group")
STATS_INTERVAL = int(os.getenv("STATS_INTERVAL_SEC", "60"))   # seconds
MAX_RETRIES    = 5
RETRY_DELAY    = 5   # seconds between reconnect attempts

# ─── MongoDB helpers ──────────────────────────────────────────────────────────
def _get_collections():
    """Return (alerts_col, stats_col) from upi_fraud_db."""
    db = get_db()
    return db["live_fraud_alerts"], db["live_stats"]


def _save_alert(col, txn: dict, result: dict) -> None:
    doc = {
        "transaction_id": txn.get("transaction_id"),
        "type":           txn.get("type"),
        "amount":         txn.get("amount"),
        "confidence":     result["confidence"],
        "risk_level":     result["risk_level"],
        "flags":          result["flags"],
        "timestamp":      txn.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "nameOrig":       txn.get("nameOrig"),
        "nameDest":       txn.get("nameDest"),
        "saved_at":       datetime.now(timezone.utc).isoformat(),
    }
    try:
        col.insert_one(doc)
    except Exception as e:
        log(f"MongoDB insert failed: {e}", "ERROR")


def _save_stats(col, stats: dict) -> None:
    doc = {**stats, "recorded_at": datetime.now(timezone.utc).isoformat()}
    try:
        col.insert_one(doc)
    except Exception as e:
        log(f"MongoDB stats insert failed: {e}", "ERROR")


# ─── Consumer factory ─────────────────────────────────────────────────────────
def _make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,          # allows periodic stats check
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
    )


# ─── Main loop ────────────────────────────────────────────────────────────────
def run():
    log(f"Starting consumer → broker={KAFKA_BROKER}  topic='{TOPIC}'", "INFO")

    alerts_col, stats_col = _get_collections()

    # Rolling counters
    total_processed = 0
    total_fraud     = 0
    conf_sum        = 0.0
    last_stats_ts   = time.time()

    consumer      = None
    retry_count   = 0

    while True:
        # ── (Re)connect ───────────────────────────────────────────────────────
        try:
            if consumer is None:
                consumer = _make_consumer()
                log("Connected to Kafka. Waiting for messages… (Ctrl+C to stop)", "SUCCESS")
                retry_count = 0
        except NoBrokersAvailable:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                log("Max retries exceeded. Exiting.", "ERROR")
                sys.exit(1)
            log(f"Broker not available — retry {retry_count}/{MAX_RETRIES} in {RETRY_DELAY}s…", "WARNING")
            time.sleep(RETRY_DELAY)
            continue

        # ── Consume messages ──────────────────────────────────────────────────
        try:
            for msg in consumer:
                txn = msg.value

                # Run model
                try:
                    # predict_transaction expects raw string type
                    # producer sends readable names, so pass through directly
                    result = predict_transaction(txn)
                except Exception as e:
                    log(f"Prediction error: {e}", "ERROR")
                    continue

                total_processed += 1
                txn_id  = txn.get("transaction_id", "N/A")[:8]
                amount  = txn.get("amount", 0)

                if result["prediction"] == 1:
                    total_fraud += 1
                    conf_sum    += result["confidence"]
                    flags_str    = ", ".join(result["flags"]) if result["flags"] else "—"
                    print(
                        f"🚨 FRAUD DETECTED | ID: {txn_id}… | "
                        f"Amount: ₹{amount:>12,.2f} | "
                        f"Confidence: {result['confidence']*100:.1f}% | "
                        f"Flags: {flags_str}"
                    )
                    _save_alert(alerts_col, txn, result)
                else:
                    print(
                        f"✅ LEGITIMATE      | ID: {txn_id}… | "
                        f"Amount: ₹{amount:>12,.2f}"
                    )

                # ── Periodic stats ────────────────────────────────────────────
                now = time.time()
                if now - last_stats_ts >= STATS_INTERVAL:
                    fraud_rate  = (total_fraud / total_processed * 100) if total_processed else 0
                    avg_conf    = (conf_sum / total_fraud * 100) if total_fraud else 0
                    print(
                        f"\n📊 LIVE STATS | Processed: {total_processed:,} | "
                        f"Frauds caught: {total_fraud:,} | "
                        f"Fraud rate: {fraud_rate:.2f}% | "
                        f"Avg confidence: {avg_conf:.1f}%\n"
                    )
                    _save_stats(stats_col, {
                        "processed":   total_processed,
                        "fraud_count": total_fraud,
                        "fraud_rate":  round(fraud_rate, 4),
                        "avg_confidence": round(avg_conf, 4),
                    })
                    last_stats_ts = now

        except KafkaError as e:
            log(f"Kafka error: {e} — reconnecting…", "WARNING")
            try:
                consumer.close()
            except Exception:
                pass
            consumer = None
            time.sleep(RETRY_DELAY)

        except KeyboardInterrupt:
            log("\nConsumer stopped by user.", "WARNING")
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if consumer:
        consumer.close()

    fraud_rate = (total_fraud / total_processed * 100) if total_processed else 0
    avg_conf   = (conf_sum / total_fraud * 100) if total_fraud else 0
    log(
        f"Final stats — Processed: {total_processed:,} | "
        f"Frauds: {total_fraud:,} | "
        f"Fraud rate: {fraud_rate:.2f}% | "
        f"Avg confidence: {avg_conf:.1f}%",
        "SUCCESS",
    )


if __name__ == "__main__":
    run()
