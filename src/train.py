import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import load_data
from .build_features import engineer_features

def train():

    df = load_data('../data/raw/loan_approval_dataset.csv')

    df = engineer_features(df)

    X = df.drop(columns=["loan_status", "loan_id"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
   )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200)

    with mlflow.start_run():

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds)

        mlflow.log_metric("auc", auc)
        mlflow.log_param("n_estimators", 200)

        mlflow.sklearn.log_model(model, "model")

        print("AUC:", auc)


if __name__ == "__main__":
    train()