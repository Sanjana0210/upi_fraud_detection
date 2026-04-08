#!/bin/bash
pkill -f "consumer.py" 2>/dev/null
sleep 2
cd /Users/hriday/Documents/upi_fraud_detection/kafka
nohup /Users/hriday/Documents/upi_fraud_detection/.venv/bin/python consumer.py > /dev/null 2>&1 &
echo "Consumer started as PID: $!"
