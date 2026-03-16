import numpy as np

def engineer_features(df):

    df = df.copy()

    df["loan_status"] = (df["loan_status"] == "Approved").astype(int)
    df["education"] = (df["education"] == "Graduate").astype(int)
    df["self_employed"] = (df["self_employed"] == "Yes").astype(int)

    asset_cols = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]

    df["total_assets"] = df[asset_cols].sum(axis=1)

    df["debt_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)

    df["assets_to_loan"] = df["total_assets"] / (df["loan_amount"] + 1)

    return df