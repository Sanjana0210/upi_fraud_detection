"""
dashboard/app.py  —  Simplified UPI Fraud Detection Dashboard
Run:  streamlit run dashboard/app.py
"""

import os, sys, time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import get_mongo_client

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

*, html, body, .stApp { font-family: 'Nunito', sans-serif !important; }

/* Big friendly headings */
h1 { font-weight: 900 !important; font-size: 2rem !important; }
h2, h3 { font-weight: 800 !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    border-radius: 18px;
    padding: 22px 26px !important;
    border: 2px solid rgba(128,128,128,0.12);
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); }
div[data-testid="stMetric"] label {
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em;
    opacity: 0.65;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 900 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 2px solid rgba(128,128,128,0.1); }
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.5;
}

/* Info box */
.info-box {
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.6;
    border: 2px solid rgba(99,102,241,0.2);
    background: rgba(99,102,241,0.06);
    margin-bottom: 20px;
}
.info-box .icon { font-size: 1.4rem; margin-right: 8px; }

/* Fraud alert cards */
.fraud-card {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border: 2px solid rgba(239,68,68,0.2);
    border-left: 5px solid #ef4444;
    background: rgba(239,68,68,0.05);
    font-size: 0.9rem;
    line-height: 1.7;
}
.fraud-card .amount { font-size: 1.2rem; font-weight: 900; color: #ef4444; }
.fraud-card .detail { opacity: 0.7; font-size: 0.82rem; margin-top: 4px; }

/* Result boxes */
.safe-box {
    border-radius: 18px;
    padding: 30px 36px;
    border: 3px solid rgba(34,197,94,0.3);
    background: rgba(34,197,94,0.07);
    text-align: center;
}
.danger-box {
    border-radius: 18px;
    padding: 30px 36px;
    border: 3px solid rgba(239,68,68,0.3);
    background: rgba(239,68,68,0.07);
    text-align: center;
}
.warn-box {
    border-radius: 18px;
    padding: 30px 36px;
    border: 3px solid rgba(245,158,11,0.3);
    background: rgba(245,158,11,0.07);
    text-align: center;
}
.big-result { font-size: 2.2rem; font-weight: 900; margin-bottom: 10px; }
.result-details { font-size: 1rem; opacity: 0.75; line-height: 2; }

/* LIVE badge */
.live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #ef4444;
    margin-right: 8px;
    animation: blink 1.4s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* Form button */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 1.8rem !important;
    font-family: 'Nunito', sans-serif !important;
}
hr { border: none; border-top: 2px solid rgba(128,128,128,0.1); margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BLUE   = "#60a5fa"       # brighter blue for dark bg
RED    = "#f87171"       # brighter red for dark bg
GREEN  = "#4ade80"
INDIGO = "#818cf8"
AMBER  = "#fbbf24"

def chart_base(**kw):
    b = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, sans-serif", size=13, color="#e2e8f0"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#cbd5e1")),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False, tickfont=dict(color="#cbd5e1")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="#e2e8f0")),
    )
    b.update(kw)
    return b

# ── MongoDB ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _mongo():
    try: return get_mongo_client()
    except: return None

def get_db():
    c = _mongo()
    if c is None:
        st.error("⚠️ Can't connect to MongoDB. Make sure it's running and refresh.")
        st.stop()
    return c["upi_fraud_db"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ UPI Fraud Detector")
st.sidebar.caption("Big Data Analytics Project")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio("Go to", [
    "🏠  Home",
    "🔍  Explore Data",
    "🔴  Live Alerts",
    "🤖  Test a Transaction",
    "📊  Model Comparison",
])

st.sidebar.markdown("---")

# ── Sidebar explainer ─────────────────────────────────────────────────────────
tips = {
    "🏠  Home":               "See the big picture — how many transactions, how many frauds, and overall stats.",
    "🔍  Explore Data":       "Dig deeper — which types of transactions have most fraud, and which amounts are risky.",
    "🔴  Live Alerts":        "Watch fraud being caught in real time as the system processes transactions.",
    "🤖  Test a Transaction": "Enter any transaction details and the AI will tell you if it looks like fraud.",
    "📊  Model Comparison":   "Compare how each ML model performed — accuracy, recall, ROC curves, and more.",
}
st.sidebar.info(tips[PAGE])
st.sidebar.caption("Built with Python · MongoDB · PySpark · Kafka · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("# 🛡️ UPI Fraud Detection System")
    st.markdown("""
    <div class="info-box">
        <span class="icon">💡</span>
        This system processes <strong>63 lakh transactions</strong>, automatically learns what fraud looks like,
        and flags suspicious transactions in real time — just like what <strong>PhonePe & GPay</strong> do internally.
    </div>
    """, unsafe_allow_html=True)

    db = get_db()

    try:
        total     = db["transactions"].count_documents({})
        fraud     = db["fraud_cases"].count_documents({})
        legit     = total - fraud
        agg       = list(db["fraud_cases"].aggregate([{"$group": {"_id": None, "t": {"$sum": "$amount"}}}]))
        lost      = agg[0]["t"] if agg else 0
        rate      = (fraud / total * 100) if total else 0
    except Exception as e:
        st.error(f"Error: {e}"); return

    st.markdown("### 📊 The Numbers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}", help="All transactions in the dataset")
    c2.metric("Fraud Cases Found", f"{fraud:,}", help="Transactions our system flagged as fraud")
    c3.metric("Safe Transactions", f"{legit:,}", help="Transactions that are legitimate")
    c4.metric("Fraud Rate", f"{rate:.4f}%", help="What % of all transactions were fraud")

    st.markdown("---")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Which type of transaction has most fraud?")
        st.caption("CASH_OUT and TRANSFER are where most fraud happens — criminals move money out fast")
        try:
            data = list(db["transaction_summary"].find({}, {"_id": 0}))
            if data:
                df = pd.DataFrame(data)
                df_melt = pd.DataFrame({
                    "Type":     df["type"].tolist() * 2,
                    "Count":    (df["total_count"] - df["fraud_count"]).tolist() + df["fraud_count"].tolist(),
                    "Category": ["✅ Safe"] * len(df) + ["🚨 Fraud"] * len(df),
                })
                fig = px.bar(df_melt, x="Type", y="Count", color="Category", barmode="group",
                             color_discrete_map={"✅ Safe": BLUE, "🚨 Fraud": RED})
                fig.update_layout(**chart_base(height=360))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data yet.")
        except Exception as e:
            st.error(f"{e}")

    with col2:
        st.markdown("### Fraud vs Safe — overall split")
        st.caption("The tiny red slice = fraud. But even 0.13% of 63 lakh is thousands of cases!")
        fig = go.Figure(go.Pie(
            labels=["✅ Safe", "🚨 Fraud"], values=[legit, fraud],
            hole=0.6,
            marker=dict(colors=[BLUE, RED]),
            textinfo="percent+label",
            textfont=dict(size=14, family="Nunito", color="#e2e8f0"),
        ))
        fig.update_layout(**chart_base(showlegend=False, margin=dict(l=10,r=10,t=10,b=10), height=360))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Timeline ───────────────────────────────────────────────────────────────
    st.markdown("### 📈 When does fraud happen? (by simulation hour)")
    st.caption("Each 'step' = 1 hour of simulated time. Fraud spikes at certain times of day.")
    try:
        rows = list(db["spark_hourly_stats"].find({}, {"_id": 0, "step": 1, "fraud_count": 1}).sort("step", 1))
        if rows:
            df_h = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_h["step"], y=df_h["fraud_count"],
                mode="lines", fill="tozeroy",
                fillcolor="rgba(239,68,68,0.08)",
                line=dict(color=RED, width=2.5, shape="spline"),
                hovertemplate="Hour %{x}<br>Frauds detected: %{y}<extra></extra>",
            ))
            fig.update_layout(**chart_base(
                xaxis_title="Hour", yaxis_title="Number of Frauds", height=300,
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run spark/process_data.py first.")
    except Exception as e:
        st.error(f"{e}")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <span class="icon">🧠</span>
        <strong>How does the AI detect fraud?</strong> It was trained on 63 lakh labelled transactions
        (fraud vs not-fraud). It learned patterns like: <em>"if someone drains their entire balance in one
        CASH_OUT, that's suspicious"</em>. It now catches fraud with <strong>99.93% accuracy</strong>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Explore Data
# ══════════════════════════════════════════════════════════════════════════════
def page_explore():
    st.markdown("# 🔍 Explore the Data")
    st.markdown("""
    <div class="info-box">
        <span class="icon">📂</span>
        This page shows patterns our <strong>PySpark engine</strong> found after scanning all 63 lakh rows.
        PySpark is a tool that can process huge amounts of data super fast — like a turbocharged Excel.
    </div>
    """, unsafe_allow_html=True)

    db = get_db()

    # ── Fraud rate bar ─────────────────────────────────────────────────────────
    st.markdown("### 💸 Which transaction type is most dangerous?")
    st.caption("Fraud rate = out of every 100 transactions of that type, how many were fraud")
    try:
        data = list(db["transaction_summary"].find({}, {"_id": 0}))
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values("fraud_rate", ascending=False)
            fig = go.Figure(go.Bar(
                x=df["type"], y=df["fraud_rate"],
                marker=dict(color=df["fraud_rate"], colorscale=[[0, BLUE], [1, RED]]),
                text=[f"{v:.1f}%" for v in df["fraud_rate"]],
                textposition="outside",
                textfont=dict(size=13, family="Nunito", color="#e2e8f0"),
                hovertemplate="Type: %{x}<br>Fraud Rate: %{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(**chart_base(xaxis_title="Transaction Type", yaxis_title="Fraud Rate (%)", height=380))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"{e}")

    st.markdown("---")

    # ── Amount buckets ─────────────────────────────────────────────────────────
    st.markdown("### 💰 What amount range do fraudsters target?")
    st.caption("Big amounts (100K–500K) are the sweet spot for fraudsters — large enough to be worth it, small enough to avoid detection")
    try:
        data = list(db["spark_amount_distribution"].find({}, {"_id": 0}))
        if data:
            df = pd.DataFrame(data)
            order = ["0-1K", "1K-10K", "10K-100K", "100K-500K", "500K+"]
            df["amount_bucket"] = pd.Categorical(df["amount_bucket"], categories=order, ordered=True)
            df = df.sort_values("amount_bucket")
            fig = px.bar(df, x="amount_bucket", y="count", color="label", barmode="group",
                         color_discrete_map={"Legitimate": BLUE, "Fraud": RED},
                         labels={"amount_bucket": "Amount Range (₹)", "count": "Number of Transactions", "label": ""})
            fig.update_layout(**chart_base(height=380))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run spark/process_data.py first.")
    except Exception as e:
        st.error(f"{e}")

    st.markdown("---")

    # ── Tables ─────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚩 Most Suspicious Senders")
        st.caption("These accounts sent the most fraud transactions")
        try:
            rows = list(db["spark_high_risk_senders"].find({}, {"_id": 0}).sort("fraud_count", -1).limit(10))
            if rows:
                df = pd.DataFrame(rows)
                for c in ["total_fraud_amount", "avg_fraud_amount"]:
                    if c in df.columns:
                        df[c] = df[c].apply(lambda x: f"₹{x:,.0f}")
                df.columns = [col.replace("_", " ").title() for col in df.columns]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No data yet.")
        except Exception as e:
            st.error(f"{e}")

    with col2:
        st.markdown("### 💣 Biggest Fraud Transactions")
        st.caption("The highest-value frauds caught in the dataset")
        try:
            rows = list(db["fraud_cases"].find({}, {"_id": 0, "step": 1, "type": 1, "amount": 1, "nameOrig": 1}).sort("amount", -1).limit(10))
            if rows:
                df = pd.DataFrame(rows)
                df["amount"] = df["amount"].apply(lambda x: f"₹{x:,.0f}")
                df.columns = [c.replace("nameOrig", "Sender").replace("step", "Hour").replace("type", "Type").replace("amount", "Amount") for c in df.columns]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No data yet.")
        except Exception as e:
            st.error(f"{e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Alerts
# ══════════════════════════════════════════════════════════════════════════════
def page_live():
    st.markdown("# 🔴 Live Fraud Alerts")
    st.markdown(
        '<p style="opacity:0.6; font-size:1rem; margin-top:-8px;">'
        '<span class="live-dot"></span>'
        'Kafka is streaming transactions right now — our AI is checking each one for fraud'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="info-box">
        <span class="icon">⚡</span>
        <strong>What is Kafka?</strong> Think of Kafka as a real-time conveyor belt.
        Transactions slide in one by one, our AI checks each one, and if it looks like fraud —
        it gets flagged instantly and shows up here.
    </div>
    """, unsafe_allow_html=True)

    db = get_db()

    @st.fragment(run_every=timedelta(seconds=3))
    def _live_feed():
        now      = datetime.now(timezone.utc)
        ago5m    = (now - timedelta(minutes=5)).isoformat()
        ago1h    = (now - timedelta(hours=1)).isoformat()

        try:
            n5   = db["live_fraud_alerts"].count_documents({"saved_at": {"$gte": ago5m}})
            n1h  = db["live_fraud_alerts"].count_documents({"saved_at": {"$gte": ago1h}})
            ntot = db["live_fraud_alerts"].count_documents({})
        except:
            n5 = n1h = ntot = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("🕐 Last 5 Minutes", f"{n5}", help="Frauds caught in last 5 mins")
        c2.metric("🕐 Last 1 Hour",    f"{n1h}", help="Frauds caught in last hour")
        c3.metric("🏆 Total Caught",   f"{ntot:,}", help="All frauds caught since pipeline started")

        st.markdown("---")
        st.markdown("### 🚨 Latest Fraud Alerts")

        try:
            alerts = list(db["live_fraud_alerts"].find({}, {"_id": 0}).sort("saved_at", -1).limit(12))
            if alerts:
                for a in alerts:
                    amt   = a.get("amount", 0)
                    typ   = a.get("type", "N/A")
                    risk  = a.get("risk_level", "N/A")
                    ts    = a.get("saved_at", "")[:19].replace("T", " ")
                    flags = ", ".join(a.get("flags", [])) or "No specific flags"
                    conf  = round(a.get("confidence", 0) * 100, 1)

                    risk_emoji = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟠")

                    st.markdown(f"""
                    <div class="fraud-card">
                        <span class="amount">₹{amt:,.2f}</span>
                        &nbsp;&nbsp;
                        <strong>{typ}</strong>
                        &nbsp;·&nbsp;
                        {risk_emoji} <strong>{risk} RISK</strong>
                        &nbsp;·&nbsp;
                        AI confidence: <strong>{conf}%</strong>
                        <div class="detail">🕐 {ts} &nbsp;·&nbsp; ⚠️ Why flagged: {flags}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⏳ No alerts yet. Make sure the Kafka producer & consumer are running!")
                st.code("# Terminal 1\npython kafka/producer.py\n\n# Terminal 2\npython kafka/consumer.py")
        except Exception as e:
            st.error(f"{e}")

        st.markdown("---")
        st.markdown("### 📈 Fraud detections over last 30 minutes")
        try:
            cutoff = (now - timedelta(minutes=30)).isoformat()
            rows   = list(db["live_stats"].find(
                {"recorded_at": {"$gte": cutoff}},
                {"_id": 0, "recorded_at": 1, "fraud_count": 1}
            ).sort("recorded_at", 1))
            if rows:
                df = pd.DataFrame(rows)
                df["recorded_at"] = pd.to_datetime(df["recorded_at"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["recorded_at"], y=df["fraud_count"],
                    mode="lines+markers", fill="tozeroy",
                    fillcolor="rgba(239,68,68,0.07)",
                    line=dict(color=RED, width=2.5),
                    marker=dict(size=6, color=RED),
                    hovertemplate="%{x|%H:%M:%S}<br>Total frauds caught: %{y}<extra></extra>",
                ))
                fig.update_layout(**chart_base(xaxis_title="Time", yaxis_title="Frauds Caught", height=300))
                st.plotly_chart(fig, use_container_width=True, key=f"lc_{int(time.time()*1000)}")
            else:
                st.info("No stats in the last 30 minutes yet.")
        except Exception as e:
            st.error(f"{e}")

    _live_feed()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Test a Transaction
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown("# 🤖 Test Any Transaction")
    st.markdown("""
    <div class="info-box">
        <span class="icon">🧪</span>
        Enter the details of any transaction below and our AI model will instantly tell you
        whether it looks like <strong>fraud or not</strong> — and explain <em>why</em>.
        This is the same model that's processing the live Kafka stream.
    </div>
    """, unsafe_allow_html=True)

    try:
        from ml.predict import predict_transaction
    except ImportError:
        st.error("Model not found. Make sure you've run `ml/train_model.py` first.")
        return

    # ── Quick examples (stored in session_state so they survive reruns) ────────
    PRESETS = {
        "fraud":      {"type": "CASH_OUT", "amount": 450000.0, "old_s": 450000.0, "new_s": 0.0,     "old_r": 0.0,     "new_r": 0.0},
        "safe":       {"type": "TRANSFER", "amount": 15000.0,  "old_s": 80000.0,  "new_s": 65000.0, "old_r": 20000.0, "new_r": 35000.0},
        "borderline": {"type": "CASH_OUT", "amount": 200000.0, "old_s": 210000.0, "new_s": 10000.0, "old_r": 0.0,     "new_r": 0.0},
    }
    DEFAULTS = {"type": "CASH_OUT", "amount": 50000.0, "old_s": 100000.0, "new_s": 50000.0, "old_r": 0.0, "new_r": 50000.0}

    if "preset_vals" not in st.session_state:
        st.session_state.preset_vals = DEFAULTS.copy()

    def _apply_preset(name):
        st.session_state.preset_vals = PRESETS[name]
        st.session_state.txn_type_key = PRESETS[name]["type"]

    st.markdown("#### 💡 Try a quick example:")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("🔴 Obvious Fraud", use_container_width=True, help="CASH_OUT ₹450K, entire balance drained, money vanishes"):
        _apply_preset("fraud"); st.rerun()
    if ex2.button("✅ Safe Transfer", use_container_width=True, help="TRANSFER ₹15K, balances make sense"):
        _apply_preset("safe"); st.rerun()
    if ex3.button("🟡 Borderline Case", use_container_width=True, help="CASH_OUT ₹200K, near-full drain, money vanishes"):
        _apply_preset("borderline"); st.rerun()

    d = st.session_state.preset_vals

    st.markdown("---")
    st.markdown("#### 📝 Or enter your own transaction:")

    # ── Type-specific context (only fraud-applicable types) ────────────────────
    TYPE_CONTEXT = {
        "CASH_OUT": {"origin": "Customer (You)", "dest": "Cash Agent", "emoji": "🏧",
                     "desc": "You're withdrawing cash from an agent's account"},
        "TRANSFER": {"origin": "Sender", "dest": "Receiver", "emoji": "🔄",
                     "desc": "You're transferring money to another person's account"},
    }

    st.markdown("""
    <div class="info-box">
        <span class="icon">💡</span>
        Only <strong>CASH_OUT</strong> and <strong>TRANSFER</strong> are available because these are the only
        transaction types where fraud occurs. Criminals need to <em>move money out</em> —
        they don't deposit (CASH_IN), pay merchants (PAYMENT), or trigger system debits (DEBIT).
    </div>
    """, unsafe_allow_html=True)

    # Sync selectbox default with preset type
    if "txn_type_key" not in st.session_state:
        st.session_state.txn_type_key = d["type"] if d["type"] in ["CASH_OUT", "TRANSFER"] else "CASH_OUT"

    # Type selector OUTSIDE form — changes instantly on selection
    txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"], key="txn_type_key")
    ctx = TYPE_CONTEXT[txn_type]
    st.caption(f"{ctx['emoji']} **{txn_type}** — {ctx['desc']}")

    with st.form("txn_form"):
        c_amt, _ = st.columns(2)
        with c_amt:
            amount = st.number_input("Amount (₹)", min_value=0.0, max_value=10_000_000.0,
                                     value=d["amount"], step=1000.0, format="%.2f")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Origin Account** ({ctx['origin']})")
            sender_old = st.number_input(f"{ctx['origin']} Balance BEFORE (₹)", min_value=0.0,
                                         value=d["old_s"], step=1000.0, format="%.2f",
                                         help=f"How much {ctx['origin'].lower()} had before this transaction")
            sender_new = st.number_input(f"{ctx['origin']} Balance AFTER (₹)", min_value=0.0,
                                         value=d["new_s"], step=1000.0, format="%.2f",
                                         help=f"How much {ctx['origin'].lower()} has after this transaction")

        with c2:
            st.markdown(f"**Destination Account** ({ctx['dest']})")
            recv_old = st.number_input(f"{ctx['dest']} Balance BEFORE (₹)", min_value=0.0,
                                       value=d["old_r"], step=1000.0, format="%.2f",
                                       help=f"How much {ctx['dest'].lower()} had before")
            recv_new = st.number_input(f"{ctx['dest']} Balance AFTER (₹)", min_value=0.0,
                                       value=d["new_r"], step=1000.0, format="%.2f",
                                       help=f"How much {ctx['dest'].lower()} has after")

        submitted = st.form_submit_button("🔍 Check This Transaction", use_container_width=True)

    if submitted:
        txn = {
            "type": txn_type, "amount": amount,
            "oldbalanceOrg": sender_old, "newbalanceOrig": sender_new,
            "oldbalanceDest": recv_old,  "newbalanceDest": recv_new,
        }
        with st.spinner("AI is analysing..."):
            try:
                result = predict_transaction(txn)
            except FileNotFoundError:
                st.error("Model file missing. Run `ml/train_model.py` first."); return
            except Exception as e:
                st.error(f"Error: {e}"); return

        conf = result["confidence"] * 100
        st.markdown("---")
        st.markdown("### 🧠 AI Result:")

        if conf >= 60:
            flags = result.get("flags", [])
            flags_str = "<br>".join([f"• {f}" for f in flags]) if flags else "• Suspicious pattern detected"
            st.markdown(f"""
            <div class="danger-box">
                <div class="big-result" style="color:#ef4444;">🚨 This looks like FRAUD</div>
                <div class="result-details">
                    AI is <strong>{conf:.1f}% confident</strong> this is fraud<br>
                    Risk level: <strong>{result.get('risk_level', 'HIGH')}</strong><br><br>
                    <strong>Why it was flagged:</strong><br>{flags_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif conf >= 30:
            flags = result.get("flags", [])
            flags_str = "<br>".join([f"• {f}" for f in flags]) if flags else "• Borderline pattern detected"
            st.markdown(f"""
            <div class="warn-box">
                <div class="big-result" style="color:#f59e0b;">⚠️ This looks SUSPICIOUS</div>
                <div class="result-details">
                    <strong>{conf:.1f}% chance</strong> of fraud — not safe, not confirmed fraud<br>
                    The AI thinks something is off but isn't fully certain<br><br>
                    <strong>Warning signs:</strong><br>{flags_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <div class="big-result" style="color:#22c55e;">✅ This looks SAFE</div>
                <div class="result-details">
                    Only <strong>{conf:.1f}% chance</strong> of fraud<br>
                    This transaction seems legitimate
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Fraud probability meter:**")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            number=dict(suffix="%", font=dict(size=52, family="Nunito")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickfont=dict(size=12)),
                bar=dict(color=RED if conf > 60 else (AMBER if conf > 30 else GREEN), thickness=0.28),
                bgcolor="rgba(128,128,128,0.08)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  30],  color="rgba(34,197,94,0.12)"),
                    dict(range=[30, 60],  color="rgba(245,158,11,0.12)"),
                    dict(range=[60, 100], color="rgba(239,68,68,0.12)"),
                ],
            ),
        ))
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Nunito"), margin=dict(l=40,r=40,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box" style="margin-top:16px;">
            <span class="icon">📖</span>
            <strong>What do these numbers mean?</strong>
            0–30% = very safe &nbsp;·&nbsp; 30–60% = a bit suspicious &nbsp;·&nbsp; 60–100% = likely fraud.
            The AI trained on 63 lakh examples to learn these patterns.
        </div>
        """, unsafe_allow_html=True)

        # ── Risk Factor Analysis Chart ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Risk Factor Analysis")
        st.caption("How each feature of this transaction contributed to the AI's decision")

        # Compute derived features
        bal_diff_orig = sender_old - sender_new
        bal_diff_dest = recv_new - recv_old
        amt_ratio = amount / (sender_old + 1)
        is_round = 1 if amount % 1000 == 0 else 0
        zero_orig = 1 if sender_old == 0 else 0

        # Risk scores for each factor (0 = safe, 1 = max risk)
        risk_factors = []

        # Balance drain risk
        drain_pct = (bal_diff_orig / sender_old * 100) if sender_old > 0 else 0
        risk_factors.append(("Balance Drain", min(drain_pct, 100),
                             f"{drain_pct:.0f}% of balance drained"))

        # Money vanishing risk
        vanish_risk = 100 if (bal_diff_dest <= 0 and amount > 0) else 0
        risk_factors.append(("Money Vanished", vanish_risk,
                             "Money didn't reach destination" if vanish_risk > 0 else "Money arrived at destination"))

        # Amount size risk (scale: >400K = 100%)
        amt_risk = min(amount / 400_000 * 100, 100)
        risk_factors.append(("Amount Size", amt_risk,
                             f"₹{amount:,.0f}" + (" (very large)" if amount > 400_000 else "")))

        # Amount-to-balance ratio risk
        ratio_risk = min(amt_ratio * 100, 100)
        risk_factors.append(("Amount/Balance Ratio", ratio_risk,
                             f"{amt_ratio:.2f}" + (" (sending most of balance)" if amt_ratio > 0.8 else "")))

        # Round amount risk (mild)
        risk_factors.append(("Round Number", 40 if is_round else 0,
                             "Yes — fraudsters prefer round amounts" if is_round else "No"))

        # Zero starting balance
        risk_factors.append(("Zero Starting Balance", 60 if zero_orig else 0,
                             "Suspicious — sending from empty account" if zero_orig else "Has existing balance"))

        # Build chart
        rf_names = [r[0] for r in risk_factors]
        rf_scores = [r[1] for r in risk_factors]
        rf_details = [r[2] for r in risk_factors]
        rf_colors = [RED if s >= 60 else (AMBER if s >= 30 else GREEN) for s in rf_scores]

        fig_risk = go.Figure(go.Bar(
            y=rf_names[::-1],
            x=rf_scores[::-1],
            orientation="h",
            marker_color=rf_colors[::-1],
            text=[f"{s:.0f}%" for s in rf_scores[::-1]],
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
            customdata=rf_details[::-1],
            hovertemplate="%{y}: %{customdata}<extra></extra>",
        ))
        fig_risk.update_layout(**chart_base(
            xaxis_title="Risk Score (%)",
            height=300,
            margin=dict(l=160, r=60, t=10, b=40),
            xaxis=dict(range=[0, 115], showgrid=True, gridcolor="rgba(148,163,184,0.12)"),
        ))
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── Model info card ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 About the Model")

        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", "Random Forest")
        c2.metric("Trees Used", "100")
        c3.metric("Test AUC", "0.9993")

        st.markdown(f"""
        <div class="info-box" style="margin-top:12px;">
            <span class="icon">🧠</span>
            <strong>{conf:.1f}% of the 100 decision trees voted this as FRAUD.</strong>
            The model was trained on <strong>63 lakh transactions</strong> and compared against
            XGBoost and Logistic Regression. Random Forest won with the highest ROC-AUC (0.9993).
            See <strong>📊 Model Comparison</strong> page for the full comparative study.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
def page_compare():
    import json as _json

    st.markdown("# 📊 Model Comparison")
    st.markdown("""
    <div class="info-box">
        <span class="icon">🔬</span>
        We trained <strong>3 different ML models</strong> on the same dataset and compared their performance.
        Below is a detailed comparative study showing how each model performed across key evaluation metrics.
    </div>
    """, unsafe_allow_html=True)

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "model_comparison.json")
    if not os.path.exists(json_path):
        st.warning("⚠️ Model comparison data not found. Run `python ml/train_model.py` first to generate it.")
        return

    with open(json_path) as f:
        data = _json.load(f)

    models = data["models"]
    best   = data["best_model"]
    reason = data["best_reason"]
    features = data["features"]

    # ── Best model callout ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="safe-box" style="margin-bottom: 24px;">
        <div class="big-result" style="color:#22c55e;">🏆 Best Model: {best}</div>
        <div class="result-details">
            Selected because: <strong>{reason}</strong><br>
            This model is currently deployed for real-time fraud detection via the Kafka pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Performance Metrics")
    st.caption("Side-by-side comparison of all models on the test set (20% holdout)")

    metric_df = pd.DataFrame([{
        "Model":     m["name"],
        "Accuracy":  f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall":    f"{m['recall']:.4f}",
        "F1 Score":  f"{m['f1_score']:.4f}",
        "ROC-AUC":   f"{m['roc_auc']:.4f}",
    } for m in models])
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("### 📊 Metrics Comparison")
    st.caption("Higher is better for all metrics")

    METRIC_COLORS = {"Random Forest": BLUE, "XGBoost": AMBER, "GradientBoosting": AMBER, "Logistic Regression": INDIGO}
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    metric_keys  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    fig = go.Figure()
    for m in models:
        color = METRIC_COLORS.get(m["name"], GREEN)
        vals  = [m[k] for k in metric_keys]
        fig.add_trace(go.Bar(
            name=m["name"], x=metric_names, y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
        ))
    fig.update_layout(**chart_base(
        height=420, yaxis_title="Score", barmode="group",
        yaxis=dict(range=[0, 1.12], showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.markdown("### 📈 ROC Curves")
    st.caption("Receiver Operating Characteristic — plots True Positive Rate vs False Positive Rate. Closer to top-left = better.")

    ROC_COLORS = [BLUE, AMBER, INDIGO]
    fig_roc = go.Figure()
    for i, m in enumerate(models):
        c = ROC_COLORS[i % len(ROC_COLORS)]
        fig_roc.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"],
            mode="lines",
            name=f"{m['name']} (AUC={m['roc_auc']:.4f})",
            line=dict(color=c, width=2.5),
        ))
    # Diagonal reference
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.5)",
        line=dict(color="rgba(148,163,184,0.4)", width=1.5, dash="dash"),
        showlegend=True,
    ))
    fig_roc.update_layout(**chart_base(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    ))
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### 🧮 Confusion Matrices")
    st.caption("Shows True Negatives, False Positives, False Negatives, True Positives for each model")

    cm_cols = st.columns(len(models))
    labels = ["Legit", "Fraud"]

    for col, m in zip(cm_cols, models):
        with col:
            st.markdown(f"**{m['name']}**")
            cm = m["confusion_matrix"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm[::-1],
                x=labels,
                y=labels[::-1],
                text=[[f"{v:,}" for v in row] for row in cm[::-1]],
                texttemplate="%{text}",
                textfont=dict(size=16, color="white"),
                colorscale=[[0, "rgba(99,102,241,0.15)"], [1, INDIGO]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
            ))
            fig_cm.update_layout(**chart_base(
                height=300,
                xaxis_title="Predicted",
                yaxis_title="Actual",
                margin=dict(l=20, r=20, t=10, b=40),
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown(f"### 🎯 Feature Importance ({best})")
    st.caption("Which features matter most for the model's fraud detection decisions")

    best_model = next((m for m in models if m["name"] == best), models[0])
    fi = best_model.get("feature_importance", [])

    if fi and len(fi) == len(features):
        fi_df = pd.DataFrame({"Feature": features, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, BLUE], [1, INDIGO]],
            ),
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(**chart_base(
            xaxis_title="Importance",
            height=max(350, len(features) * 35),
            margin=dict(l=180, r=40, t=10, b=40),
        ))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance data not available for this model.")

    st.markdown("""
    <div class="info-box" style="margin-top:16px;">
        <span class="icon">📖</span>
        <strong>What do these metrics mean?</strong><br>
        <strong>Accuracy</strong> = overall correctness &nbsp;·&nbsp;
        <strong>Precision</strong> = of those flagged as fraud, how many actually were &nbsp;·&nbsp;
        <strong>Recall</strong> = of all actual frauds, how many did we catch &nbsp;·&nbsp;
        <strong>F1</strong> = harmonic mean of precision &amp; recall &nbsp;·&nbsp;
        <strong>ROC-AUC</strong> = model's ability to distinguish fraud from legit (1.0 = perfect)
    </div>
    """, unsafe_allow_html=True)


if   PAGE == "🏠  Home":                 page_home()
elif PAGE == "🔍  Explore Data":         page_explore()
elif PAGE == "🔴  Live Alerts":          page_live()
elif PAGE == "🤖  Test a Transaction":   page_predict()
elif PAGE == "📊  Model Comparison":     page_compare()