# Email Lead Scoring — Deployment Protocol
## Google Cloud Run (FastAPI Backend + Streamlit Frontend)

**Format:** 7 chunks of ~15 minutes each. Chunks 1–6 are required (~90 min total); Chunk 7 is optional CI/CD. Do them in order — each builds on the last.

| Chunk | Time | Task | Done |
|-------|------|------|------|
| 1 | 15 min | Backend Dockerfile + local build test | ☐ |
| 2 | 15 min | GCP project setup | ☐ |
| 3 | 15 min | Build + push backend image | ☐ |
| 4 | 15 min | Deploy + verify backend | ☐ |
| 5 | 15 min | Build + deploy frontend | ☐ |
| 6 | 15 min | End-to-end test + cleanup | ☐ |
| 7 | 15 min | (Optional) CI/CD trigger + rollback check | ☐ |

---

## Architecture Overview

```
                        ┌─────────────────────────────┐
User ──► frontend-els   │  Streamlit on Cloud Run     │
         (port 8501)    │  BACKEND_ENDPOINT env var ──┼──► backend-els (FastAPI, port 8080)
                        └─────────────────────────────┘         │
                                                                ├─ models/*.pkl  (XGBoost, sklearn pipeline)
                                                                └─ database/crm_database.sqlite
```

| Layer | GCP Service | Purpose |
|-------|-------------|---------|
| App hosting | Cloud Run (managed) | Serverless containers for backend + frontend |
| Container registry | Container Registry (`gcr.io`) | Docker images for both services |
| CI/CD | Cloud Build | Build → push → deploy on demand or on git push |
| Logging | Cloud Logging | Build and runtime logs |

Both services are stateless: model artifacts and the SQLite database are baked into the backend image.

---

## Chunk 1 — Backend Dockerfile + Local Build Test (15 min)

The frontend Dockerfile already exists (`streamlit_app/Dockerfile`). The backend needs one at the repo root:

```dockerfile
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
```

Note: the deployed entrypoint is root `main.py` (`uvicorn main:app`). All four `COPY` lines are required — `main.py` is the app, `email_lead_scoring/` is the scoring package, `database/` holds the CRM SQLite (`database/crm_database.sqlite`), and `models/` holds `blended_models_final`, the model used by `/predict` and `/calculate_lead_strategy` (`main.py` lines 119 & 160).

A root `.dockerignore` keeps the build context small — it excludes `venv/`, `.git`, `streamlit_app/`, and docs.

Sanity-check the build locally:

```bash
cd els-backend
docker build -t backend-els .
docker run -p 8081:8080 backend-els   # then open http://localhost:8081/docs
```

`-p 8081:8080` maps host port **8081** → container port **8080** (the container always
listens on 8080). The host port is arbitrary — if 8080 is already in use locally, pick any
free port (here 8081) and open the matching `http://localhost:<host-port>/docs`. This only
affects local testing; Cloud Run still injects `$PORT` and routes to container port 8080.

✅ **Done when:** image builds and Swagger UI loads locally.

---

## Chunk 2 — GCP Project Setup (15 min)

```bash
gcloud auth login

# Set your real project id ONCE, then reuse $PROJECT_ID in every command below.
# Do NOT paste the literal "YOUR_PROJECT_ID" — that produces a gcr.io path that
# doesn't exist and the deploy fails with "Image ... not found".
export PROJECT_ID=your-real-project-id          # e.g. phrasal-aegis-499119-k0
gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com

gcloud config set run/region us-central1
```

✅ **Done when:** `gcloud services list --enabled` shows all three APIs.

---

## Chunk 3 — Build + Push Backend Image (15 min)

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/backend-els .
```

Heavy dependencies (pycaret, xgboost, catboost) make this the longest build — start it, let it run, watch progress in the Cloud Build console.

✅ **Done when:** build reports SUCCESS and the image appears in `gcr.io`.

---

## Chunk 4 — Deploy + Verify Backend (15 min)

```bash
# 2Gi memory required — pycaret + xgboost load at startup
gcloud run deploy backend-els \
  --image gcr.io/$PROJECT_ID/backend-els \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300

# Capture the URL — needed in Chunk 5
BACKEND_URL=$(gcloud run services describe backend-els \
  --region us-central1 --format 'value(status.url)')

# Smoke tests
curl -s ${BACKEND_URL}/get_email_subscribers | head -c 200
open ${BACKEND_URL}/docs
```

✅ **Done when:** `/get_email_subscribers` returns data and Swagger UI loads.

---

## Chunk 5 — Build + Deploy Frontend (15 min)

The frontend reads the backend URL from the `BACKEND_ENDPOINT` env var (`streamlit_app/constants.py`).

```bash
cd streamlit_app

gcloud builds submit --tag gcr.io/$PROJECT_ID/frontend-els .

gcloud run deploy frontend-els \
  --image gcr.io/$PROJECT_ID/frontend-els \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8501 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 120 \
  --set-env-vars BACKEND_ENDPOINT=${BACKEND_URL}
```

✅ **Done when:** deploy prints a frontend URL.

---

## Chunk 6 — End-to-End Test + Cleanup (15 min)

1. Open the frontend URL; upload `streamlit_app/data/leads.csv`.
2. Confirm the lead strategy table and expected-value chart render.
3. Update the live-app URLs in `README.md` if they changed.
4. Commit `Dockerfile` and `DEPLOY.md`:

```bash
git add Dockerfile DEPLOY.md
git commit -m "Add backend Dockerfile and Cloud Run deployment protocol"
git push
```

✅ **Done when:** a fresh browser session can score leads end-to-end.

---

## Chunk 7 — Optional: CI/CD Trigger + Rollback Check (15 min)

`streamlit_app/cloudbuild.yaml` automates build → push → deploy for the frontend.

```bash
# Manual trigger
gcloud builds submit --config streamlit_app/cloudbuild.yaml .

# GitHub push trigger
gcloud builds triggers create github \
  --repo-name=els-backend \
  --repo-owner=EllaN12 \
  --branch-pattern=^main$ \
  --build-config=streamlit_app/cloudbuild.yaml \
  --substitutions=_REGION=us-central1,_SERVICE=frontend-els,_BACKEND_ENDPOINT=${BACKEND_URL}

# Rollback drill
gcloud run revisions list --service backend-els --region us-central1
gcloud run services update-traffic backend-els \
  --region us-central1 --to-revisions REVISION_NAME=100
```

| Substitution | Example | Purpose |
|--------------|---------|---------|
| `_REGION` | `us-central1` | Cloud Run region |
| `_SERVICE` | `frontend-els` | Cloud Run service name |
| `_BACKEND_ENDPOINT` | `https://backend-els-….run.app` | Injected into the frontend container |

✅ **Done when:** a push to `main` rebuilds the frontend automatically.

---

## Configuration Reference

| Setting | backend-els | frontend-els |
|---------|-------------|--------------|
| Port | 8080 | 8501 |
| Memory | 2Gi | 512Mi |
| CPU | 2 | 1 |
| Instances | 0–3 | 0–3 |
| Timeout | 300s | 120s |
| Env vars | `PORT` | `PORT`, `BACKEND_ENDPOINT` |
| Auth | Public (`--allow-unauthenticated`) | Public |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `gcloud run deploy` fails: `Image 'gcr.io/YOUR_PROJECT_ID/backend-els' not found` | The literal `YOUR_PROJECT_ID` placeholder was pasted instead of the real id, and/or the image was never built | `export PROJECT_ID=<real-id>`, run `gcloud builds submit --tag gcr.io/$PROJECT_ID/backend-els .` first, then deploy with `--image gcr.io/$PROJECT_ID/backend-els` |
| `gcloud builds submit` fails with 403 `storage.objects.get` on the `_cloudbuild` bucket | Newer projects run Cloud Build as the Compute Engine default SA, which lacks storage access | `gcloud projects add-iam-policy-binding PROJECT_ID --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com --role=roles/cloudbuild.builds.builder`, wait ~60s, retry |
| Backend container fails to start | OOM loading pycaret/xgboost models | Confirm `--memory 2Gi`; check Cloud Logging for `Memory limit exceeded` |
| 500 with `OSError: libgomp.so.1: cannot open shared object file` | lightgbm/xgboost need OpenMP, missing from `python:3.9-slim` | Add `RUN apt-get update && apt-get install -y libgomp1` to the Dockerfile, rebuild |
| 500 with `FileNotFoundError: …model_h2o_stacked_ensemble.pkl` | Endpoint pointed a PyCaret loader at an H2O artifact | Use a PyCaret `.pkl` model path (e.g. `models/blended_models_final`) |
| Frontend shows connection errors | `BACKEND_ENDPOINT` stale or missing | Redeploy frontend with `--set-env-vars BACKEND_ENDPOINT=<current backend URL>` |
| 503 on first request | Cold start with min-instances 0 | Expected; set `--min-instances 1` to eliminate |
| Build fails on dependency resolution | Pinned versions in `requirements.txt` conflict | Build locally with `docker build .` to reproduce, then adjust pins |
| Streamlit blank page | CORS/XSRF defaults | Dockerfile already passes `--server.enableCORS=false --server.enableXsrfProtection=false`; confirm CMD unchanged |

---

## Cost Notes

Both services scale to zero (`--min-instances 0`), so idle cost is ~$0. Main cost drivers are backend cold starts (2Gi × 2 CPU) and Cloud Build minutes. For a demo/portfolio workload this stays comfortably within the Cloud Run free tier.
