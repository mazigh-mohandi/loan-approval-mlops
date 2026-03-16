from src.build_features import engineer_features

def test_engineer_features(sample_df):
    df_feat = engineer_features(sample_df)

    # Vérifie les colonnes binaires
    assert set(df_feat["loan_status"].unique()) <= {0, 1}
    assert set(df_feat["education"].unique()) <= {0, 1}
    assert set(df_feat["self_employed"].unique()) <= {0, 1}

    # Vérifie les nouvelles colonnes calculées
    assert "total_assets" in df_feat.columns
    assert "debt_to_income" in df_feat.columns
    assert "assets_to_loan" in df_feat.columns

    # Vérifie que total_assets = somme des assets
    expected_total = sample_df[
        ["residential_assets_value", "commercial_assets_value",
         "luxury_assets_value", "bank_asset_value"]
    ].sum(axis=1).tolist()
    assert df_feat["total_assets"].tolist() == expected_total