from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI()

model = mlflow.sklearn.load_model("models:/LoanModel/Production")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")

def predict(data: dict):

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    return {"prediction": int(pred)}