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

        # Load fixed machine types
        fixed_machine_types = joblib.load("fixed_machine_types.pkl")

        # Map unknown machine types to 'Other'
        df["Machine type"] = df["Machine type"].apply(
            lambda x: x if x in fixed_machine_types else "Other"
        )

        # One-hot encode 'Machine type'
        df = pd.get_dummies(df, columns=["Machine type"], drop_first=False)

        # Add missing columns and reorder
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(df)

        # Predict probabilities
        probs = model.predict_proba(X_scaled)

        # Compute expected value
        expected_values = (probs * np.array([0, 1, 2, 3])).sum(axis=1)

        return {"predictions": expected_values.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
