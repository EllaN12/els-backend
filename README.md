# Email Lead Scoring (ELS)

[![Deployed on Google Cloud Run](https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=google-cloud&logoColor=white)](https://frontend-els-741818216556.us-central1.run.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Overview

Business Science Inc. is an online educational school specialising in data science for business training. The Marketing Department's email campaigns are a key revenue channel, but excessive send frequency is causing subscriber unsubscribes — each of which represents a potential lost customer. Marketing Management is exploring a data-driven strategy to balance email frequency and maximise customer conversion.

This project builds a full end-to-end **email lead scoring** system that:

1. Quantifies the financial cost of mass email blasting (cost simulation).
2. Trains a machine learning model to score each subscriber on their purchase probability.
3. Segments the email list into **Hot Leads** (target with sales emails) and **Cold Leads** (withhold sales emails, nurture instead).
4. Optimises the scoring threshold to maximise expected business value.
5. Exposes the full pipeline via a **FastAPI REST API** and a **Streamlit self-service UI**, both deployed on **Google Cloud Run**.

**Key result:** At an 85% monthly sales safeguard level, the strategy yields **$180,000 in expected value**, generates **$33,000 in savings**, and maintains **$214,000 in monthly sales**.

---

## Live Application

| Service | URL |
|---------|-----|
| Streamlit Frontend | https://frontend-els-741818216556.us-central1.run.app/ |
| FastAPI Backend (Swagger docs) | `<backend-url>/docs` |

---

## Data Sources

**SQLite CRM database** (`app/database/crm_database.sqlite`) — 3 interconnected tables:

| Table | Records | Key Attributes |
|-------|---------|---------------|
| `Subscribers` | 19,919 | `mailchimp_id`, `user_email`, `member_rating`, `optin_time`, `country_code` |
| `Tags` | ~26,000 | `mailchimp_id`, `tag` — maps subscribers to course/event interactions |
| `Transactions` | ~4,600 | `user_email`, purchase records — defines the `made_purchase` target |

---

## Methods and Tools

### CRISP-DM Methodology

**1. Business Understanding — Cost Simulation (`cost_calculations.py`)**

Before any modelling, the financial risk of mass emailing was quantified:

```
Monthly Cost = List Size × Unsub Rate × Emails/Month × Conversion Rate × Avg Customer Value
```

A parameter grid simulation produces a heatmap of unsubscription costs across different growth rates (1%–3.5%) and conversion rates (4%–6%), with projected annual costs reaching **$3M–$4M+** under high-growth scenarios.

**2. Data Preparation (`database.py`)**

`db_read_and_process_els_data()` automates the full ETL pipeline:

- Joins `Subscribers ⟕ Tags` (aggregating tag counts per subscriber).
- Extracts `email_provider` from user email addresses.
- Calculates `optin_days` (subscriber recency) and `tag_count_by_optin_day` (engagement rate).
- Pivots tags into wide binary feature columns (`tag_*`).
- Filters low-sale countries and labels the rest as `Other`.
- Creates the binary target `made_purchase` from the Transactions join.

**3. Exploratory Data Analysis (`exploratory.py`)**

- `explore_sales_by_category()` — groups by categorical features (country, member rating) to identify top-converting segments.
- `explore_sales_by_numeric()` — quantile analysis of numeric features split by purchase status; `tag_count` emerged as the strongest signal.

**4. Feature Engineering**

PyCaret's preprocessing pipeline handles:

| Step | Method |
|------|--------|
| Numeric imputation | Mean |
| Categorical imputation | Mode (most frequent) |
| Ordinal encoding | `member_rating` (OrdinalEncoder) |
| One-hot encoding | `country_code` (OneHotEncoder) |
| Train/test split | 80/20 stratified (15,935 / 3,984 rows) |
| Cross-validation | StratifiedKFold, 5 folds |

The dataset expanded from **52 raw features to 73 engineered features** after preprocessing.

---

### Machine Learning (`modeling.py`)

**Experiment tracking:** All runs logged to MLflow (experiment: `email_lead_scoring_0`). Best runs retrieved via `mlflow_get_best_run()` and scored with `mlflow_score_leads()`.

#### All-Models Benchmark (PyCaret `compare_models`)

14 classifiers were benchmarked. Top results:

| Rank | Model | AUC | Accuracy |
|------|-------|-----|----------|
| 1 | Gradient Boosting Classifier | **0.7837** | 0.9516 |
| 2 | AdaBoost Classifier | 0.7706 | 0.9522 |
| 3 | LightGBM | 0.7660 | 0.9502 |
| 4 | Ridge Classifier | 0.7513 | 0.9531 |

#### Models Trained

| Model | Artifact | Notes |
|-------|----------|-------|
| Gradient Boosting (base) | `blended_models_final.pkl` | Best from PyCaret `compare_models` |
| XGBoost (tuned) | `xgb_model_tuned.pkl` | Hyperparameter-tuned via PyCaret |
| XGBoost Pipeline | `pipeline_xgb.pkl` | Production sklearn pipeline with full preprocessing |
| AdaBoost (tuned) | `ada_model_tuned.pkl` | Adaptive boosting |
| CatBoost (tuned) | `catboost_model_tuned.pkl` | Categorical-native boosting |
| H2O Stacked Ensemble | `model_h2o_stacked_ensemble/` | AutoML stacking via H2O |
| Blended Ensemble | `blended_models_final.pkl` | Weighted blend of top PyCaret models — **selected for prediction** |

#### Cross-Validation Results (Tuned Models, 5-fold)

| Model | Mean AUC | Mean Accuracy | Std AUC |
|-------|----------|---------------|---------|
| XGBoost (tuned) | 0.7877 | 0.9511 | ±0.0177 |
| **Blended Ensemble** | **0.7914** | **0.9515** | ±0.0161 |

#### Best Model Performance — GradientBoostingClassifier (full dataset refit)

| Metric | Class 0 (no purchase) | Class 1 (purchase) |
|--------|-----------------------|--------------------|
| Precision | 0.958 | **0.848** |
| Recall | 0.999 | 0.146 |
| F1 | 0.978 | 0.249 |
| Support | 3,792 | 192 |
| **AUC** | — | **0.84** |

**Confusion matrix:** TN 3,787 · FP 5 · FN 164 · TP 28

The model achieves very high precision on predicted buyers (84.8%) — extremely few false alarms — making it well-suited for lead scoring where targeting the wrong people is costly. The low recall on Class 1 is expected given the heavily imbalanced dataset (~5% purchase rate), and is addressed by the threshold optimisation step.

**Top features by importance:**

| Feature | Importance |
|---------|-----------|
| `tag_count` | ~50% |
| `optin_days` | ~14% |
| `tag_count_by_optin_day` | ~5% |
| `country_code_us` | ~3% |

---

### Lead Strategy & Threshold Optimisation (`lead_strategy.py`)

Scored leads are ranked by `lead_score` and categorised using a threshold on cumulative gain:

- **Hot Lead** — `gain ≤ threshold` → receive sales emails
- **Cold Lead** — `gain > threshold` → withheld from sales emails (routed to nurture)

`lead_score_strategy_optimization()` sweeps 100 threshold values (0→1) and selects the one that **maximises expected value** subject to a configurable monthly sales safeguard (default: 90% of maximum).

**Expected value formula:**

```
EV = Sales Retained (Hot Leads) + Unsub Savings (Cold Leads) − Missed Sales (Cold Leads)
```

**Result at 85% safeguard:**

| Metric | Value |
|--------|-------|
| Expected Value | **$180,000** |
| Expected Savings | **$33,000** |
| Monthly Sales | **$214,000** |

---

## Project Structure

```
els-backend/
│
├── app/
│   ├── main.py                          # FastAPI application (6 REST endpoints)
│   ├── database/
│   │   └── crm_database.sqlite          # SQLite CRM database
│   ├── models/                          # Trained model artifacts
│   │   ├── pipeline_xgb.pkl             # Production XGBoost sklearn pipeline
│   │   ├── xgb_model_tuned.pkl          # Tuned XGBoost (PyCaret)
│   │   ├── blended_models_final.pkl     # Blended ensemble — used for strategy
│   │   ├── model_h2o_stacked_ensemble/  # H2O AutoML model
│   │   ├── ada_model_tuned.pkl
│   │   └── catboost_model_tuned.pkl
│   └── email_lead_scoring/              # Core analytics module
│       ├── __init__.py
│       ├── cost_calculations.py         # Cost simulation functions
│       ├── database.py                  # ETL pipeline (db_read_and_process_els_data)
│       ├── exploratory.py               # EDA helper functions
│       ├── modeling.py                  # Model scoring + MLflow utilities
│       └── lead_strategy.py             # Threshold optimisation + expected value
│
├── venv/
│   └── streamlit_app/                   # Streamlit frontend
│       ├── app.py                       # Main UI (file upload, analysis, download)
│       ├── helpers.py                   # Lead strategy + plotting helpers
│       └── constants.py                 # Backend endpoint config (env var)
│
├── requirements.txt                     # Python dependencies
├── constants.py                         # Shared constants
├── DEPLOY.md                            # Google Cloud Run deployment guide
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Animated welcome page with link to Swagger docs |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/get_email_subscribers` | Return all subscribers from the database |
| `POST` | `/data` | Data passthrough utility |
| `POST` | `/predict` | Score leads using `xgb_model_tuned.pkl` |
| `POST` | `/predict_xgb` | Score leads using `pipeline_xgb.pkl` (sklearn pipeline) |
| `POST` | `/calculate_lead_strategy` | Full pipeline: score → optimise → return strategy, expected value, and threshold table |

---

## Installation and Setup

### Prerequisites

- Python 3.9+
- Node.js (optional, for local development tooling)
- Docker (for containerised deployment)
- Google Cloud SDK (for Cloud Run deployment — included in repo)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/EllaN12/Email-Lead-Scoring-Frontend.git
cd els-backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 5. Run the Streamlit frontend (separate terminal)
cd venv/streamlit_app
streamlit run app.py
```

The API will be available at `http://localhost:8080` (Swagger UI at `/docs`) and the Streamlit app at `http://localhost:8501`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Port for the FastAPI server |
| `BACKEND_ENDPOINT` | *(see constants.py)* | Backend URL used by the Streamlit app |

---

## Deployment (Google Cloud Run)

Both services are containerised and deployed to Google Cloud Run. See **[DEPLOY.md](DEPLOY.md)** for the full step-by-step guide including:

- Building and pushing Docker images via `gcloud builds submit`
- Deploying the FastAPI backend (`backend-els`) with 2Gi memory
- Deploying the Streamlit frontend (`frontend-els`) with the backend URL injected as an environment variable
- CI/CD via `cloudbuild.yaml`

---

## Key Dependencies

| Category | Library | Version |
|----------|---------|---------|
| Data | pandas, numpy | 2.1.4 / 1.26.4 |
| Data utilities | pyjanitor, pandas-flavor | 0.27.0 / 0.6.0 |
| ML — AutoML | pycaret | 3.3.1 |
| ML — Models | xgboost, catboost, scikit-learn | 2.0.3 / 1.2.5 / 1.4.2 |
| ML — Imbalanced | imbalanced-learn | 0.12.2 |
| Experiment tracking | mlflow | 2.12.1 |
| Visualisation | plotly, plotly-resampler | 5.21.0 / 0.10.0 |
| API | fastapi, uvicorn, pydantic | 0.110.2 / 0.29.0 / 2.7.0 |
| Database | SQLAlchemy, sqlparse | 2.0.29 / 0.5.0 |

---

## Acknowledgements

- **PyCaret** — AutoML pipeline management and model comparison
- **H2O.ai** — Stacked ensemble AutoML
- **MLflow** — Experiment tracking and model registry
- **FastAPI** — High-performance REST API framework
- **Streamlit** — Interactive frontend for the marketing team
- **Google Cloud Run** — Serverless container deployment
- **pandas / NumPy** — Data manipulation and analysis
- **Plotly** — Interactive data visualisation
