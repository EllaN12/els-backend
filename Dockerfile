# ── Backend: FastAPI on Google Cloud Run ─────────────────────────────────────
FROM python:3.9-slim

WORKDIR /app

# libgomp1 (OpenMP) is required by lightgbm/xgboost; absent from -slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API entrypoint, els package, CRM database, and models
COPY main.py .
COPY email_lead_scoring/ ./email_lead_scoring/
COPY database/ ./database/
COPY models/ ./models/

# Cloud Run injects $PORT (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
