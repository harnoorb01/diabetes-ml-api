# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware   # <-- add this
from pydantic import BaseModel
import numpy as np, joblib

# load your saved files
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# create FastAPI app
app = FastAPI(title="Diabetes Predictor")

# --- CORS middleware (needed if UI is on another site) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define input schema
class InputRow(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(x: InputRow):
    arr = np.array([[x.Pregnancies, x.Glucose, x.BloodPressure,
                     x.SkinThickness, x.Insulin, x.BMI,
                     x.DiabetesPedigreeFunction, x.Age]])
    arr
