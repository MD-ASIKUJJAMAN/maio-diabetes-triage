
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Diabetes Triage ML Service")

# Load the model trained in Step 4
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "model.joblib"
)

model = joblib.load(MODEL_PATH)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

# Define the ten required patient features
class PatientData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(patient: PatientData):
    features = pd.DataFrame([patient.model_dump()])

    prediction = model.predict(features)[0]

    return {"prediction": float(prediction)}
