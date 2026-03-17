import logging
import os
import time

import mlflow.sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.build_features import engineer_features
from src.preprocessing import load_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_RUN_ID = os.getenv("MLFLOW_RUN_ID", "bc4622917dbc4bf5aa081bcab6f91dbd")
_MODEL_PATH = os.path.join(_BASE_DIR, "src", "mlruns", "0", _RUN_ID, "artifacts", "model")

logger.info("Loading model from %s", _MODEL_PATH)
model = mlflow.sklearn.load_model(_MODEL_PATH)
logger.info("Model loaded successfully")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Loan Approval API", version="1.0.0")

# Simple in-memory counters for /metrics
_stats = {"requests_total": 0, "predictions_total": 0, "errors_total": 0}


# ---------------------------------------------------------------------------
# Middleware – request logging + timing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    _stats["requests_total"] += 1
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  status=%d  duration=%.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str          # "Graduate" | "Not Graduate"
    self_employed: str      # "Yes" | "No"
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


class PredictionResponse(BaseModel):
    prediction: int         # 1 = Approved, 0 = Rejected
    label: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "run_id": _RUN_ID}


@app.get("/metrics")
def metrics():
    return _stats


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    import pandas as pd

    try:
        raw = application.model_dump()
        raw["loan_id"] = 0
        raw["loan_status"] = "Approved"  # placeholder, dropped in engineer_features
        df = pd.DataFrame([raw])
        df = engineer_features(df)
        features = df.drop(columns=["loan_status", "loan_id"])
        pred = int(model.predict(features)[0])
        _stats["predictions_total"] += 1
        label = "Approved" if pred == 1 else "Rejected"
        logger.info("Prediction: %s (CIBIL=%d)", label, application.cibil_score)
        return PredictionResponse(prediction=pred, label=label)
    except Exception as exc:
        _stats["errors_total"] += 1
        logger.error("Prediction error: %s", exc)
        raise
