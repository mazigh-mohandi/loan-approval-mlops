import pandas as pd
from src.preprocessing import load_data

def test_load_data(tmp_path):
    # Création d’un fichier CSV temporaire
    df = pd.DataFrame({
        "col1": [" a", "b ", " c "],
        "col2": ["x", "y", " z"]
    })
    file = tmp_path / "data.csv"
    df.to_csv(file, index=False)

    df_loaded = load_data(file)
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (3, 2)
    # Vérifie strip sur les colonnes string
    assert df_loaded["col1"].tolist() == ["a", "b", "c"]
    assert df_loaded["col2"].tolist() == ["x", "y", "z"]

