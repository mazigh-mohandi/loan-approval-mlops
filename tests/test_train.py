import pytest
import pandas as pd
import numpy as np
from src import train
import mlflow

def test_train_runs(tmp_path, monkeypatch):
    """
    Test que train() s'exécute correctement sur un petit dataset dummy.
    """

    # --- 1. Crée un DataFrame dummy minimal ---
    n = 10  # nombre d'exemples
    df = pd.DataFrame({
        "loan_id": range(1, n + 1),
        "loan_status": ["Approved", "Rejected"] * (n // 2),
        "education": ["Graduate", "Not Graduate"] * (n // 2),
        "self_employed": ["Yes", "No"] * (n // 2),
        "residential_assets_value": np.random.randint(1000, 5000, size=n),
        "commercial_assets_value": np.random.randint(500, 2000, size=n),
        "luxury_assets_value": np.zeros(n),
        "bank_asset_value": np.random.randint(100, 500, size=n),
        "income_annum": np.random.randint(30000, 100000, size=n),
        "loan_amount": np.random.randint(5000, 20000, size=n)
    })

    # --- 2. Sauvegarde le CSV dans tmp_path ---
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)

    # --- 3. Monkeypatch load_data pour retourner le DataFrame dummy ---
    monkeypatch.setattr(train, "load_data", lambda _: df)

    # --- 4. Monkeypatch mlflow pour éviter d'écrire sur disque réel ---
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: DummyRunContext())

    # --- 5. Exécute train() et vérifie qu'aucune exception n'est levée ---
    train.train()


# --- Contexte factice pour mlflow.start_run() ---
class DummyRunContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass