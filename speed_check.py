from pymongo import MongoClient
from datetime import datetime

db = MongoClient("mongodb://localhost:27017/")["upi_fraud_db"]

first  = db.live_stats.find_one(sort=[("recorded_at", 1)])
latest = db.live_stats.find_one(sort=[("recorded_at", -1)])

t1 = datetime.fromisoformat(first["recorded_at"])
t2 = datetime.fromisoformat(latest["recorded_at"])
elapsed_sec = (t2 - t1).total_seconds()

processed   = latest["processed"]
rate        = processed / elapsed_sec if elapsed_sec else 0
remaining   = 2_234_638
eta_hours   = remaining / rate / 3600 if rate else 999

print(f"Speed             : {rate:.1f} msg/sec")
print(f"Processed (session): {processed:,}")
print(f"Elapsed            : {elapsed_sec/60:.1f} minutes")
print(f"Remaining in Kafka : ~2,234,638")
print(f"ETA to finish      : ~{eta_hours:.0f} hours")
print()
print("--- For demo you DON'T need to finish ---")
print(f"Fraud alerts in DB : {db.live_fraud_alerts.count_documents({})}")
