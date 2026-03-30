# Sentinel

Transaction fraud detection MVP built with FastAPI, XGBoost, and PostgreSQL.

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Start PostgreSQL

```bash
docker compose up -d db
```

### 3. Run migrations

```bash
alembic upgrade head
```

### 4. Generate training data and train the model

```bash
python scripts/generate_data.py
python -m sentinel.ml.train
```

### 5. Start the API

```bash
uvicorn sentinel.main:app --reload
```

### 6. Test it

```bash
curl -X POST http://localhost:8000/api/v1/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 125.50,
    "merchant_category": "grocery",
    "merchant_name": "Whole Foods",
    "card_last_four": "4242",
    "card_type": "visa",
    "transaction_time": "2025-06-15T14:30:00Z",
    "location_country": "US",
    "is_online": false
  }'
```

API docs available at `http://localhost:8000/docs`.

## Running with Docker

```bash
docker compose up
```

## Running Tests

```bash
pytest
```

## Project Structure

```
src/sentinel/
├── main.py          # FastAPI app
├── config.py        # Settings
├── database.py      # Async SQLAlchemy
├── api/             # HTTP endpoints
├── models/          # ORM models
├── schemas/         # Pydantic schemas
├── services/        # Business logic + ML inference
└── ml/              # Training pipeline
```

## Tech Stack

- **API**: FastAPI + Uvicorn
- **ML**: XGBoost + scikit-learn
- **Database**: PostgreSQL + SQLAlchemy (async)
- **Migrations**: Alembic
