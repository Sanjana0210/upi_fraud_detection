# UPI Fraud Detection System 🔐
### Big Data Analytics — College Capstone Project

A end-to-end fraud detection pipeline built on the **PaySim** dataset (6.3 million transactions), combining real-time streaming, distributed processing, NoSQL storage, machine learning, and an interactive dashboard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Big Data | PySpark 3.5 |
| Streaming | Apache Kafka + kafka-python |
| Database | MongoDB (pymongo) |
| ML | Scikit-learn + imbalanced-learn (SMOTE) |
| Dashboard | Streamlit + Plotly |
| Dataset | [PaySim — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |

---

## Project Structure

```
upi_fraud_detection/
├── data/
│   ├── raw/paysim.csv          ← Download from Kaggle
│   └── processed/              ← Auto-generated
├── mongodb/
│   ├── insert_data.py          ← Bulk-insert into MongoDB
│   └── queries.py              ← Analytical queries
├── spark/
│   └── process_data.py         ← PySpark ETL + feature engineering
├── kafka/
│   ├── producer.py             ← Stream transactions to Kafka
│   └── consumer.py             ← Consume + predict + alert
├── ml/
│   ├── train_model.py          ← Train Random Forest w/ SMOTE
│   ├── predict.py              ← Batch & single prediction
│   └── evaluate.py             ← Metrics + plots
├── dashboard/
│   └── app.py                  ← Streamlit dashboard
├── utils/
│   └── helpers.py              ← Shared utilities
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download **paysim.csv** from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at:
```
data/raw/paysim.csv
```

### 3. Process data with PySpark
```bash
python spark/process_data.py
```

### 4. Load data into MongoDB
```bash
# Make sure MongoDB is running on localhost:27017
python mongodb/insert_data.py
```

### 5. Train the ML model
```bash
python ml/train_model.py
```

### 6. Evaluate the model
```bash
python ml/evaluate.py
# Plots saved to ml/plots/
```

### 7. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 8. Real-time streaming (optional)
```bash
# Terminal 1 — Start Kafka (ZooKeeper + Broker must be running)
python kafka/producer.py

# Terminal 2
python kafka/consumer.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection string |
| `KAFKA_BROKER` | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC` | `upi_transactions` | Kafka topic name |
| `PRODUCER_DELAY` | `0.05` | Seconds between messages |

---

## ML Pipeline

```
Raw CSV
  └─► Feature Engineering (PySpark / pandas)
        └─► Train/Test Split (80/20, stratified)
              └─► SMOTE oversampling (minority class)
                    └─► StandardScaler
                          └─► Random Forest Classifier
                                └─► Evaluate (ROC-AUC, PR-AUC, F1)
```

**Key features engineered:**
- `amount_log` — log1p of transaction amount
- `balance_diff_orig` — sender's balance drop
- `balance_diff_dest` — receiver's balance gain
- `error_balance_orig / dest` — discrepancy between expected and actual balance changes (strong fraud indicator)

---

## Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPI cards, fraud rate, type distribution |
| 🔍 Fraud Analysis | Fraud by type, time trends, balance patterns |
| 🤖 Live Predictor | Enter transaction details → instant prediction |
| 📈 Model Metrics | Confusion matrix, ROC, PR curve, feature importance |
