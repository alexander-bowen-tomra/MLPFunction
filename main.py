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

# Define input schema
class InputData(BaseModel):
    data: list[dict]

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)
        logger.info("🔍 Raw Input DataFrame:\n%s", df.head())

        # Define machine type categories
        backroom_machines = ["T9", "T8", "T100"]
        revolution_machines = ["R1", "R2", "R3", "Bulk Collection"]
        basic_machines = ["B1", "B3", "B5", "B7", "B9"]
        flow_standalone_machines = ["T70", "T70 Dual", "S1", "S2", "M1", "T70 TriSort", "T70 Single", "T70 Dual Maxi", "T90"]

        # Define machine type to category mapping
        machine_type_map = {
            **{m: "backroom" for m in backroom_machines},
            **{m: "revolution" for m in revolution_machines},
            **{m: "basic" for m in basic_machines},
            **{m: "flow_standalone" for m in flow_standalone_machines}
        }

        # Map machine types to categories, defaulting to 'legacy'
        if "Machine type" in df.columns:
            df["Machine type"] = df["Machine type"].map(machine_type_map).fillna("legacy")

        # One-hot encode the simplified categories
        df = pd.get_dummies(df, columns=["Machine type"], drop_first=False)


        # Check for missing features
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            logger.warning("⚠️ Missing Features in Input: %s", missing_cols)

        # Add missing columns with default value 0
        for col in missing_cols:
            df[col] = 0

        # Reorder columns to match training features
        df = df[feature_names]


        # Scale features
        X_scaled = scaler.transform(df)
        #logger.info("🔍 Scaled Feature Sample:\n%s", X_scaled[:5])

        # Predict probabilities
        probs = model.predict_proba(X_scaled)
        #logger.info("🔍 Predicted Probabilities:\n%s", probs[:5])

        # Compute expected value (continuous output)
        expected_values = (probs * np.arange(4)).sum(axis=1)
        logger.info("🔍 Expected Values:\n%s", expected_values[:5])

        return {"predictions": expected_values.tolist()}

    except Exception as e:
        logger.error("❌ Prediction Error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
