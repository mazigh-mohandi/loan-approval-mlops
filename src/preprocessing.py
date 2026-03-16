# chemin = '../data/raw/loan_approval_dataset.csv'
import pandas as pd

def load_data(path: str):

    # Chargement & nettoyage initial
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Strip explicite des colonnes string (évite les espaces invisibles)
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ('string','str')]
    for col in str_cols:
        df[col] = df[col].str.strip()

    # print(f'Shape : {df.shape}  |  Valeurs manquantes : {df.isnull().sum().sum()}')
    
    return df
