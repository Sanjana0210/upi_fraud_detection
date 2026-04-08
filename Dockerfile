# ─── UPI Fraud Detection — Docker Image ──────────────────────────────────────
# Runs the Streamlit dashboard with all ML + data dependencies pre-installed.

FROM python:3.10-slim AS base

# System deps: Java for PySpark, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed ml/plots

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
