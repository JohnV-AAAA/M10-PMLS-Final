# FINAL EXAM - BANK LOAN DEFAULT PREDICTION
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------
# Load the trained Random Forest model
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.joblib"

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Bank Loan Default Prediction API",
    description=(
        "Random Forest API for estimating the probability "
        "of customer loan default."
    ),
    version="1.0"
)


# ---------------------------------------------------------
# Define the input data structure
# ---------------------------------------------------------

class CustomerData(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Bank Loan Default Prediction API",
        "status": "running",
        "model": "Random Forest",
        "number_of_trees": 500
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    # Convert the submitted customer data to a DataFrame
    input_data = pd.DataFrame(
        [{
            "AGE": customer.AGE,
            "EMPLOY": customer.EMPLOY,
            "ADDRESS": customer.ADDRESS,
            "DEBTINC": customer.DEBTINC,
            "CREDDEBT": customer.CREDDEBT,
            "OTHDEBT": customer.OTHDEBT
        }]
    )

    # Generate the predicted class
    predicted_default = int(model.predict(input_data)[0])

    # Estimate the probability of class 1 (Defaulter)
    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    # Return the prediction to the client
    return {
        "predicted_default": predicted_default,
        "probability_of_default": probability,
        "probability_percent": round(probability * 100, 2)
    }