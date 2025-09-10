from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model, scaler, and feature names
model = joblib.load("mlp_classifier.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Define input schema
class InputData(BaseModel):
    data: list[dict]

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)

        # Reindex columns to match training order
        df = df.reindex(columns=feature_names)

        # Scale features
        X_scaled = scaler.transform(df)

        # Predict probabilities
        probs = model.predict_proba(X_scaled)

        # Compute expected value (continuous output)
        expected_values = (probs * np.array([0, 1, 2, 3])).sum(axis=1)

        # Return predictions
        return {"predictions": expected_values.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
