import pytest
import pandas as pd

@pytest.fixture
def sample_df():
    data = {
        "loan_id": [1, 2, 3],
        "loan_status": ["Approved", "Rejected", "Approved"],
        "education": ["Graduate", "Not Graduate", "Graduate"],
        "self_employed": ["Yes", "No", "Yes"],
        "residential_assets_value": [10000, 5000, 20000],
        "commercial_assets_value": [5000, 2000, 10000],
        "luxury_assets_value": [2000, 0, 5000],
        "bank_asset_value": [3000, 1000, 4000],
        "income_annum": [50000, 60000, 70000],
        "loan_amount": [20000, 25000, 30000]
    }
    return pd.DataFrame(data)