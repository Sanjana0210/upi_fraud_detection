#!/bin/bash
# Run producer safely - sends 200K more messages at 20/sec (takes ~3 hours)
# Logs to producer.log so it doesn't flood your screen
pkill -f "producer.py" 2>/dev/null
sleep 1
cd /Users/hriday/Documents/upi_fraud_detection/kafka
PRODUCER_DELAY=0.05 MAX_MESSAGES=200000 nohup /Users/hriday/Documents/upi_fraud_detection/.venv/bin/python producer.py > producer.log 2>&1 &
echo "Producer started as PID: $!"
echo "Sending 200K messages at 20/sec (~2.75 hours)"
echo "Monitor: tail -f /Users/hriday/Documents/upi_fraud_detection/kafka/producer.log"
