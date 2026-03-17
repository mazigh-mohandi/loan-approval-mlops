# Loan Approval MLOps Pipeline

> MLOps Course Final Project — Pôle Léonard de Vinci, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Definition & Data](#2-problem-definition--data)
3. [System Architecture](#3-system-architecture)
4. [MLOps Practices](#4-mlops-practices)
5. [Monitoring & Reliability](#5-monitoring--reliability)
6. [Team Collaboration](#6-team-collaboration)
7. [Limitations & Future Work](#7-limitations--future-work)

---

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for **loan approval prediction**. Given a loan application, the system predicts whether it should be **Approved** or **Rejected** using a machine learning model served through a REST API.

The pipeline covers the full lifecycle: data ingestion, preprocessing, feature engineering, model training with experiment tracking, containerised model serving, and CI/CD automation.

### Quick Start

```bash
# Install dependencies
pip install uv
uv sync

# Train the model
uv run python -m src.train

# Run the API locally
uv run uvicorn src.api.main:app --reload

# Run with Docker
docker build -t loan-approval .
docker run -p 8000:8000 loan-approval
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/metrics` | Request & prediction counters |
| `POST` | `/predict` | Predict loan approval |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000
  }'
```

**Example response:**

```json
{"prediction": 1, "label": "Approved"}
```

---

## 2. Problem Definition & Data

### Task

**Binary classification** — predict whether a loan application will be approved or rejected.

### Dataset

- **Source:** [Kaggle — Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Size:** 4,269 samples, 13 columns
- **Target:** `loan_status` (`Approved` / `Rejected`)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `no_of_dependents` | int | Number of dependents |
| `education` | categorical | Graduate / Not Graduate |
| `self_employed` | categorical | Yes / No |
| `income_annum` | float | Annual income (INR) |
| `loan_amount` | float | Loan amount requested |
| `loan_term` | int | Loan term (months) |
| `cibil_score` | int | Credit score (300-900) |
| `residential_assets_value` | float | Value of residential assets |
| `commercial_assets_value` | float | Value of commercial assets |
| `luxury_assets_value` | float | Value of luxury assets |
| `bank_asset_value` | float | Value of bank assets |

### Evaluation Metric

**AUC-ROC** — chosen because it measures the model's ability to rank positive over negative samples regardless of threshold, suitable for imbalanced classes.

The trained model achieves **AUC = 1.0** on the test set, reflecting the highly separable nature of this dataset (CIBIL score is a near-perfect predictor).

---

## 3. System Architecture

```
data/raw/  ->  preprocessing.py  ->  build_features.py
                                            |
                                        train.py
                                            |
                                       MLflow runs
                                            |
                                    src/api/main.py
                                            |
                                       FastAPI app
                                            |
                          +-----------------+-----------------+
                          |                                   |
                   Docker build                    GitHub Actions CI
                   (Dockerfile)                (.github/workflows/ci.yml)
```

### Key Components

| Component | Technology | Role |
|-----------|-----------|------|
| Environment | `uv` + `pyproject.toml` | Reproducible Python deps |
| Preprocessing | `pandas` | CSV loading, whitespace cleaning |
| Feature Engineering | custom | Binary encoding, derived ratios |
| Model | `RandomForestClassifier` (n=200) | Binary classification |
| Experiment Tracking | `MLflow` | Params, metrics, model artifacts |
| Model Serving | `FastAPI` + `uvicorn` | REST API inference |
| Containerisation | `Docker` | Portable deployment |
| CI/CD | GitHub Actions | Automated tests on push/PR |

---

## 4. MLOps Practices

### 4.1 Environment Management — UV

All dependencies are declared in `pyproject.toml` and locked in `uv.lock`, guaranteeing reproducible installs across machines:

```bash
uv sync        # install all deps
uv run pytest  # run tests in the managed env
```

### 4.2 Version Control — Git & GitHub

- Feature branches + Pull Requests for every change
- Each PR reviewed by at least one other team member before merge
- Meaningful commit messages describing intent, not just changes

### 4.3 Code Quality — Pre-commit & Ruff

Pre-commit hooks run automatically on every `git commit`:

```bash
pip install pre-commit
pre-commit install   # one-time setup
```

Hooks configured (`.pre-commit-config.yaml`):
- **ruff** — linting + auto-fix
- **ruff-format** — consistent code formatting
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-merge-conflict**

### 4.4 Testing — pytest

Unit tests cover the three core modules:

```bash
uv run pytest -v
```

| Test file | What is tested |
|-----------|---------------|
| `tests/test_preprocessing.py` | `load_data`: whitespace stripping, shape |
| `tests/test_features.py` | `engineer_features`: binary encoding, derived features |
| `tests/test_train.py` | `train()`: full run with dummy data + MLflow mocking |

### 4.5 Experiment Tracking — MLflow

Every training run logs:
- **Parameters:** `n_estimators`
- **Metrics:** `auc` (AUC-ROC on test set)
- **Artifacts:** serialised model (`model.pkl`)

```bash
uv run mlflow ui   # view experiments at http://localhost:5000
```

The model is saved locally under `src/mlruns/` and loaded at API startup via the `MLFLOW_RUN_ID` environment variable (defaults to the committed run).

### 4.6 CI/CD — GitHub Actions

`.github/workflows/ci.yml` runs on every push and PR to `main`:
- Python 3.11 and 3.12 matrix
- Installs dependencies
- Runs the full pytest suite

---

## 5. Monitoring & Reliability

### Request Logging

Every HTTP request is logged with method, path, status code, and response time:

```
2026-03-17 14:23:01  INFO  POST /predict  status=200  duration=12.4ms
```

### Health Check

```bash
curl http://localhost:8000/
# {"status": "ok", "model": "RandomForestClassifier", "run_id": "bc4622..."}
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
# {"requests_total": 42, "predictions_total": 38, "errors_total": 0}
```

The `/metrics` endpoint exposes:
- `requests_total` — total HTTP requests received
- `predictions_total` — successful predictions served
- `errors_total` — prediction failures

### Potential Future Monitoring

- **Data drift detection** using Evidently AI on incoming feature distributions
- **Model performance degradation** alerts when AUC drops below threshold
- **Prometheus + Grafana** dashboards for production-grade observability

---

## 6. Team Collaboration

| Member | Role | Main Contributions |
|--------|------|--------------------|
| Mazigh MOHAMMEDI | Project Lead | Repository setup, UV environment, initial project structure |
| JeanBrice | Data Engineer | Dataset integration, project architecture, derived data pipeline |
| Marc NGANGNANG | ML Engineer | Training pipeline, MLflow integration, unit tests, CI/CD, FastAPI |
| *(4th member)* | — | — |

All work was tracked through GitHub Pull Requests. Each PR required at least one peer review before merging to `main`.

---

## 7. Limitations & Future Work

### Current Limitations

- **AUC = 1.0** — The dataset is highly separable (CIBIL score alone nearly perfectly predicts outcome), so the metric does not reflect real-world performance on noisier data.
- **No scaler persistence** — The `StandardScaler` is fitted at training time but not saved alongside the model. The API uses raw features, which is only safe because `RandomForest` does not require feature scaling.
- **In-memory metrics** — The `/metrics` counters reset on container restart. A production system would use Prometheus or a time-series database.
- **MLflow runs committed to repo** — The `src/mlruns/` directory is tracked in Git for portability. In production, MLflow would point to a remote tracking server.

### Future Work

- Save the scaler as part of a `sklearn.Pipeline` for end-to-end reproducibility
- Register models in the MLflow Model Registry and load by stage (`Production`)
- Add a retraining trigger based on data drift detection (Evidently AI)
- Deploy to a cloud provider (AWS ECS, GCP Cloud Run, or Azure Container Apps)
- Add integration tests that spin up the full Docker container
- Implement structured JSON logging for log aggregation pipelines (ELK, Datadog)
