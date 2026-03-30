# Sentinel Architecture

## System Overview

```mermaid
graph TB
    subgraph External["External"]
        Client["Client / API Consumer"]
        Kaggle["Kaggle Dataset"]
    end

    subgraph Training["Training Pipeline"]
        Script["train_all_models.py"]
    end

    subgraph Serving["Sentinel API  —  FastAPI :8000"]
        Router["API Router /api/v1"]

        subgraph Endpoints["Endpoints"]
            TxnAPI["POST /transactions"]
            ReviewAPI["GET /transactions/review-queue"]
            ModelsAPI["/models/*<br/>list · load · remove<br/>promote · compare · mode"]
            DriftAPI["GET /drift"]
        end

        subgraph Core["Scoring Pipeline"]
            Scorer["FraudScorer<br/>(champion | ensemble)"]
            Rules["Rules Engine<br/>5 rules"]
            Escalation["Escalation Router<br/>APPROVE · REVIEW · FLAG"]
        end

        subgraph State["App State"]
            Registry["ModelRegistry<br/>(thread-safe)"]
            Velocity["VelocityTracker<br/>(in-memory)"]
            Drift["DriftDetector<br/>(rolling-window PSI)"]
        end
    end

    subgraph Storage["Storage Layer"]
        BentoML["BentoML Model Store<br/>logreg · random-forest<br/>xgboost · gradient-boosting"]
        Postgres["PostgreSQL<br/>transactions table"]
        MLflow["MLflow :5000<br/>Experiment Tracking"]
    end

    subgraph Dashboard["Dashboard"]
        Streamlit["Streamlit App<br/>Score distributions<br/>Layer performance<br/>Review queue"]
    end

    Client -->|"POST transaction"| Router
    Router --> TxnAPI
    Router --> ReviewAPI
    Router --> ModelsAPI

    TxnAPI --> Scorer
    TxnAPI --> Rules
    Scorer -->|"ML score"| Escalation
    Rules -->|"Rules verdict"| Escalation
    Escalation -->|"decision + reasons"| Postgres

    Scorer --> Registry
    Rules --> Velocity
    Scorer -->|"record(score)"| Drift
    DriftAPI --> Drift
    ModelsAPI --> Registry

    Registry <-->|"load / list"| BentoML

    Script -->|"train + evaluate"| MLflow
    Script -->|"save_model()"| BentoML
    Script -->|"read"| Kaggle

    Streamlit -->|"read"| Postgres

    style Scorer fill:#3498db,color:#fff
    style Rules fill:#e74c3c,color:#fff
    style Escalation fill:#f39c12,color:#fff
    style Registry fill:#9b59b6,color:#fff
    style BentoML fill:#2ecc71,color:#fff
    style MLflow fill:#1abc9c,color:#fff
    style Drift fill:#e67e22,color:#fff
```

## Transaction Scoring Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant S as FraudScorer
    participant R as Rules Engine
    participant V as VelocityTracker
    participant E as Escalation Router
    participant D as DriftDetector
    participant DB as PostgreSQL

    C->>API: POST /api/v1/transactions
    API->>S: score(transaction)

    alt Champion Mode
        S->>S: registry.score_champion(features)
    else Ensemble Mode
        S->>S: registry.score_ensemble(features)
    end

    S-->>API: fraud_score (0.0 - 1.0)

    API->>R: evaluate_rules(transaction, tracker)
    R->>V: count_recent(card)
    R->>V: get_last_country(card)
    R->>R: high_amount · velocity · geo_anomaly · time · merchant_risk

    R-->>API: RulesVerdict (rules_score, triggered_rules)

    API->>E: route(ml_score, rules_verdict)

    Note over E: ML ≥ 0.8 → FLAG<br/>ML 0.4-0.8 + rules → FLAG<br/>ML 0.4-0.8 no rules → REVIEW<br/>ML < 0.4 + rules → REVIEW<br/>ML < 0.4 no rules → APPROVE

    E-->>API: decision + reasons

    API->>DB: INSERT transaction<br/>(fraud_score, rules_score, decision, model_used)
    API->>V: record(card, country)
    API->>D: record(fraud_score)
    API-->>C: 201 TransactionResponse
```

## Model Training & Promotion

```mermaid
flowchart LR
    subgraph Train["1 — Train"]
        Data["Kaggle Credit Card<br/>284k transactions"]
        FE["Feature Engineering<br/>30 features (V1-V28 + Amount_log + Time_hour)"]
        Split["Stratified Split<br/>80/20"]
        Models["Train 4 Models<br/>LogReg · RF · XGBoost · GradBoost"]
    end

    subgraph Track["2 — Track"]
        MLflow["MLflow<br/>params · metrics · artifacts"]
    end

    subgraph Store["3 — Store"]
        BentoML["BentoML Store<br/>sentinel-fraud:*<br/>versioned + metadata"]
    end

    subgraph Serve["4 — Serve"]
        Registry["ModelRegistry<br/>auto-loads from store<br/>best AUC-PR → champion"]
        Scorer["FraudScorer<br/>champion or ensemble"]
    end

    subgraph Manage["5 — Manage"]
        API["Model API<br/>GET /models<br/>PUT /models/{name}/champion<br/>DELETE /models/{name}<br/>POST /models/compare"]
    end

    Data --> FE --> Split --> Models
    Models -->|"log_params + log_metrics"| MLflow
    Models -->|"bentoml.sklearn.save_model()"| BentoML
    BentoML -->|"load_from_store()"| Registry
    Registry --> Scorer
    API -->|"hot-swap at runtime"| Registry

    style Models fill:#3498db,color:#fff
    style MLflow fill:#1abc9c,color:#fff
    style BentoML fill:#2ecc71,color:#fff
    style Registry fill:#9b59b6,color:#fff
    style API fill:#f39c12,color:#fff
```

## Model Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAIN            STORE             SERVE            MANAGE     │
│  ─────            ─────             ─────            ──────     │
│                                                                 │
│  train_all_  ──►  BentoML     ──►  ModelRegistry  ◄──  API     │
│  models.py        Store            (in memory)        Endpoints │
│                   (on disk)                                     │
│       │                │                │                │      │
│       │                │                │                │      │
│       ▼                ▼                ▼                ▼      │
│                                                                 │
│  MLflow         sentinel-fraud:    Champion mode    Load new    │
│  Experiment     · logreg           Ensemble mode    Remove bad  │
│  Tracking       · random-forest    Score all        Promote     │
│                 · xgboost          Compare          Switch mode │
│                 · grad-boosting                                 │
│                                                                 │
│  Metrics:       Each tagged with:  Auto-selects     Hot-swap    │
│  AUC-PR         · framework        best AUC-PR     without     │
│  AUC-ROC        · metrics           as champion     restart    │
│  F1             · dataset                                       │
│  Precision                                                      │
│  Recall                                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Matrix

```
                    ┌─────────────────────────────────────┐
                    │         ESCALATION ROUTING           │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │  ML Score     Rules      Decision   │
                    │  ─────────    ─────      ────────   │
                    │  ≥ 0.8        any    →   🔴 FLAG    │
                    │  0.4 – 0.8    yes    →   🔴 FLAG    │
                    │  0.4 – 0.8    no     →   🟡 REVIEW  │
                    │  < 0.4        yes    →   🟡 REVIEW  │
                    │  < 0.4        no     →   🟢 APPROVE │
                    │                                     │
                    └─────────────────────────────────────┘

    Rules Engine (5 independent rules):
    ┌──────────────────┬────────┬──────────────────────────┐
    │ Rule             │ Weight │ Trigger                  │
    ├──────────────────┼────────┼──────────────────────────┤
    │ High Amount      │ 0.4/0.7│ > $5k / > $10k          │
    │ Velocity         │ 0.6    │ 3+ txns in 10 min       │
    │ Geo Anomaly      │ 0.3/0.5│ High-risk country /     │
    │                  │        │ country change           │
    │ Time Anomaly     │ 0.2    │ 1am – 5am               │
    │ Merchant Risk    │ 0.3    │ Risky category + > $2k   │
    └──────────────────┴────────┴──────────────────────────┘
    Aggregate: rules_score = min(1.0, Σ triggered weights)
```

## Infrastructure

```
┌──────────────────────────────────────────────────────────┐
│                    docker-compose                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ PostgreSQL   │  │ Sentinel API│  │ MLflow Server   │  │
│  │ :5432        │  │ :8000       │  │ :5000           │  │
│  │              │  │             │  │                 │  │
│  │ transactions │  │ FastAPI     │  │ Experiments     │  │
│  │ table        │◄─┤ + BentoML  │  │ Params/Metrics  │  │
│  │              │  │   Store     │  │ Artifacts       │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│                          │                               │
│                    ┌─────┴──────┐                        │
│                    │ Streamlit  │                        │
│                    │ Dashboard  │                        │
│                    └────────────┘                        │
└──────────────────────────────────────────────────────────┘
```
