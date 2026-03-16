import pytest
from src.train import train
import os

# tests/test_train.py
def test_train_runs(tmp_path, monkeypatch):
    
    import pandas as pd
    import numpy as np

    # Crée un CSV dummy minimal
    n = 10  # nombre d'exemples
    df = pd.DataFrame({
        "loan_id": range(1, n+1),
        "loan_status": ["Approved", "Rejected"] * (n//2),
        "education": ["Graduate", "Not Graduate"] * (n//2),
        "self_employed": ["Yes", "No"] * (n//2),
        "residential_assets_value": np.random.randint(1000, 5000, size=n),
        "commercial_assets_value": np.random.randint(500, 2000, size=n),
        "luxury_assets_value": np.zeros(n),
        "bank_asset_value": np.random.randint(100, 500, size=n),
        "income_annum": np.random.randint(30000, 100000, size=n),
        "loan_amount": np.random.randint(5000, 20000, size=n)
    })

    # Sauvegarde le CSV
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)

    # Monkeypatch load_data pour retourner le DataFrame
    from src import train
    monkeypatch.setattr(train, "load_data", lambda _: df)

    # Teste que train() s'exécute sans planter
    train.train()