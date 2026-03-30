# Sentinel

Multi-layer transaction fraud detection system with rules engine, ML model ensemble, escalation routing, and drift monitoring.

**FastAPI** | **XGBoost** | **PostgreSQL** | **BentoML** | **MLflow** | **Streamlit**

---

## Architecture

Sentinel uses a tiered detection approach — transactions flow through independent layers that each contribute a signal, then an escalation router combines them into a final decision.

```
Transaction → Rules Engine → ML Model → Escalation Router → Decision
                  │               │              │
              5 weighted      Champion or     APPROVE / REVIEW / FLAG
              rules            Ensemble
```

**Why layers?** Rules catch known patterns (high amounts, velocity spikes) with high precision. ML catches subtle patterns across many features. The escalation router handles disagreements — when rules fire but ML is confident, or vice versa — by routing to human review instead of making a bad automated call.

Full Mermaid diagrams: [docs/architecture.md](docs/architecture.md)

## Detection Layers

### Rules Engine
Five independent rules, each with a weight. Scores aggregate to a `rules_score` (0.0–1.0):

| Rule | Weight | Trigger |
|------|--------|---------|
| High Amount | 0.4 / 0.7 | > $5k / > $10k |
| Velocity | 0.6 | 3+ txns from same card in 10 min |
| Geo Anomaly | 0.3 / 0.5 | High-risk country / country change |
| Time Anomaly | 0.2 | 1am – 5am |
| Merchant Risk | 0.3 | Risky category + amount > $2k |

### ML Models
Four models trained on the [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284k transactions, 0.17% fraud rate). Managed via BentoML with hot-swap support:

| Model | AUC-PR | F1 | Precision | Recall |
|-------|--------|-----|-----------|--------|
| **XGBoost** (champion) | **0.882** | **0.853** | 0.880 | 0.827 |
| Random Forest | 0.818 | 0.829 | 0.842 | 0.816 |
| Logistic Regression | 0.710 | 0.104 | 0.055 | 0.908 |
| Gradient Boosting | 0.686 | 0.763 | 0.807 | 0.725 |

### Escalation Routing

| ML Score | Rules Flagged | Decision |
|----------|--------------|----------|
| >= 0.8 | any | FLAG |
| 0.4 – 0.8 | yes | FLAG |
| 0.4 – 0.8 | no | REVIEW |
| < 0.4 | yes | REVIEW |
| < 0.4 | no | APPROVE |

### Drift Detection
Rolling-window PSI (Population Stability Index) monitors score distributions. The first 1,000 predictions build a baseline; subsequent predictions are compared against it. PSI > 0.2 triggers a drift alert via `GET /api/v1/drift`.

## Model Management

BentoML model store enables runtime model management without restarts:

```bash
# List loaded models
curl http://localhost:8000/api/v1/models

# Compare all models on a transaction
curl -X POST http://localhost:8000/api/v1/models/compare \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"amount": 500, "merchant_category": "electronics", ...}}'

# Promote a different model to champion
curl -X PUT http://localhost:8000/api/v1/models/random-forest/champion

# Remove an underperforming model
curl -X DELETE http://localhost:8000/api/v1/models/gradient-boosting

# Switch to ensemble mode (average all models)
curl -X PUT http://localhost:8000/api/v1/models/mode \
  -H "Content-Type: application/json" -d '{"mode": "ensemble"}'
```

## ML Pipeline

```
Kaggle Data → Feature Engineering → Train 4 Models → MLflow Tracking → BentoML Store → API
```

Train all models with experiment tracking:

```bash
.venv/bin/python scripts/train_all_models.py
```

This trains all four models, logs params/metrics to MLflow, saves each to BentoML's model store, and generates comparison figures in `docs/figures/`.

Browse experiments:
```bash
.venv/bin/mlflow ui --backend-store-uri mlruns
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Start PostgreSQL + MLflow
docker compose up -d db mlflow

# Run migrations
alembic upgrade head

# Download data and train models
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/kaggle/
python -c "import zipfile; zipfile.ZipFile('data/kaggle/creditcardfraud.zip').extractall('data/kaggle/')"
.venv/bin/python scripts/train_all_models.py

# Start the API
uvicorn sentinel.main:app --reload

# Run tests
pytest
```

API docs: `http://localhost:8000/docs`

## Project Structure

```
src/sentinel/
├── main.py                         # FastAPI app, lifespan init
├── config.py                       # Settings (thresholds, URIs)
├── database.py                     # Async SQLAlchemy
├── api/
│   ├── health.py                   # Health check + drift endpoint
│   ├── transactions.py             # Transaction CRUD + review queue
│   └── models.py                   # Model management (load/remove/promote/compare)
├── schemas/
│   ├── transaction.py              # Request/response schemas
│   └── model.py                    # Model API schemas
├── models/
│   └── transaction.py              # ORM model
└── services/
    ├── fraud_scorer.py             # ML scoring (champion/ensemble/heuristic)
    ├── model_registry.py           # BentoML-backed multi-model registry
    ├── rules_engine.py             # 5-rule engine with weighted scoring
    ├── escalation.py               # APPROVE/REVIEW/FLAG decision matrix
    ├── velocity_tracker.py         # In-memory card velocity tracking
    └── drift_detector.py           # Rolling-window PSI drift detection

scripts/
└── train_all_models.py             # Train 4 models → MLflow + BentoML

dashboard/
└── app.py                          # Streamlit dashboard

tests/                              # 48 tests
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI + Uvicorn | Async REST API |
| ML | XGBoost, scikit-learn | Model training + inference |
| Model Store | BentoML | Versioned model management, hot-swap |
| Experiment Tracking | MLflow | Params, metrics, artifacts |
| Database | PostgreSQL + async SQLAlchemy | Transaction persistence |
| Migrations | Alembic | Schema versioning |
| Dashboard | Streamlit | Score distributions, layer metrics |
| Containers | Docker Compose | PostgreSQL, API, MLflow |
