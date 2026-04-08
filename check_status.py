from pymongo import MongoClient

db = MongoClient("mongodb://localhost:27017/")["upi_fraud_db"]

total_txns        = db.transactions.count_documents({})
total_fraud_cases = db.fraud_cases.count_documents({})
total_alerts      = db.live_fraud_alerts.count_documents({})
total_stats_docs  = db.live_stats.count_documents({})
latest            = db.live_stats.find_one(sort=[("recorded_at", -1)])

amt_agg = list(db.live_fraud_alerts.aggregate([
    {"$group": {
        "_id": None,
        "total_amount": {"$sum": "$amount"},
        "avg_conf":     {"$avg": "$confidence"},
        "max_conf":     {"$max": "$confidence"},
        "min_conf":     {"$min": "$confidence"},
    }}
]))

risk_agg = list(db.live_fraud_alerts.aggregate([
    {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
]))

type_agg = list(db.live_fraud_alerts.aggregate([
    {"$group": {"_id": "$type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]))

print("=" * 52)
print("   FULL MONGODB STATUS REPORT")
print("=" * 52)
print()
print("--- STATIC DATA (insert_data.py + spark) ---")
print(f"  Total transactions    : {total_txns:,}")
print(f"  Known fraud cases     : {total_fraud_cases:,}")
print(f"  Dataset fraud rate    : {total_fraud_cases/total_txns*100:.4f}%")
print()
print("--- LIVE CONSUMER DATA (Kafka pipeline) ---")
print(f"  Fraud alerts in DB    : {total_alerts:,}")
print(f"  Stat checkpoints      : {total_stats_docs:,}")
if latest:
    print(f"  Last session processed: {latest['processed']:,}")
    print(f"  Last session frauds   : {latest['fraud_count']:,}")
    print(f"  Last session rate     : {latest['fraud_rate']}%")
    print(f"  Last checkpoint at    : {latest['recorded_at']}")
print()
if amt_agg:
    a = amt_agg[0]
    print(f"  Total fraud amt caught: Rs {a['total_amount']:,.2f}")
    print(f"  Avg model confidence  : {a['avg_conf']*100:.2f}%")
    print(f"  Confidence range      : {a['min_conf']*100:.1f}% – {a['max_conf']*100:.1f}%")
print()
print("  Risk level breakdown:")
for r in sorted(risk_agg, key=lambda x: x["count"], reverse=True):
    print(f"    {r['_id']:<10}: {r['count']:,}")
print()
print("  Fraud alerts by transaction type:")
for t in type_agg:
    print(f"    {t['_id']:<12}: {t['count']:,}")
print()
print("--- SPARK COLLECTIONS ---")
for name in ["spark_type_stats", "spark_hourly_stats",
             "spark_amount_distribution", "spark_high_risk_senders"]:
    n = db[name].count_documents({})
    print(f"  {name:<38}: {n:,} docs")
