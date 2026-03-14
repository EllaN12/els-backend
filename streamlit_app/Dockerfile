# ── Frontend: Streamlit on Google Cloud Run ──────────────────────────────────
FROM python:3.9-slim

WORKDIR /app

# Streamlit + requests + plotly only — keeps the image small
COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy Streamlit source files
COPY app.py .
COPY helpers.py .
COPY constants.py .

# Copy default leads data
COPY data/ ./data/

# Cloud Run injects $PORT; Streamlit must listen on the same port.
ENV PORT=8501
EXPOSE 8501

# Pass the backend API endpoint as an env var (override at deploy time).
# The default uses the existing Cloud Run backend URL.
ENV BACKEND_ENDPOINT=https://backend-els-720685387106.us-central1.run.app

CMD ["sh", "-c", \
     "streamlit run app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --server.enableCORS=false \
      --server.enableXsrfProtection=false"]
