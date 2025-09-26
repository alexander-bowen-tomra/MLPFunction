from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import joblib
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model components
model = joblib.load("mlp_classifier.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
fixed_machine_types = joblib.load("fixed_machine_types.pkl")

# Define input schema
class InputData(BaseModel):
    data: list[dict]

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)
        logger.info("🔍 Raw Input DataFrame:\n%s", df.head())

        # Map unknown machine types to 'Other'
        if "Machine type" in df.columns:
            df["Machine type"] = df["Machine type"].apply(
                lambda x: x if x in fixed_machine_types else "Other"
            )

            # One-hot encode 'Machine type'
            df = pd.get_dummies(df, columns=["Machine type"], drop_first=False)
            logger.info("🔍 One-Hot Encoded Columns:\n%s", df.columns.tolist())

        # Add missing columns and reorder to match training features
        missing_cols = [col for col in feature_names if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df = df[feature_names]
        logger.info("🔍 Final Feature Matrix Columns:\n%s", df.columns.tolist())
        logger.info("🔍 Missing Columns Added:\n%s", missing_cols)

        # Scale features
        X_scaled = scaler.transform(df)
        #logger.info("🔍 Scaled Feature Sample:\n%s", X_scaled[:5])

        # Predict probabilities
        probs = model.predict_proba(X_scaled)
        #logger.info("🔍 Predicted Probabilities:\n%s", probs[:5])

        # Compute expected value (continuous output)
        expected_values = (probs * np.arange(11)).sum(axis=1)
        #logger.info("🔍 Expected Values:\n%s", expected_values[:5])

        return {"predictions": expected_values.tolist()}

    except Exception as e:
        logger.error("❌ Prediction Error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
