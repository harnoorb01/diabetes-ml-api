from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, joblib

# load files (as you already had)
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Diabetes Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class InputRow(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "Diabetes ML API is running.", "try": ["/health", "/docs", "POST /predict"]}

@app.get("/health")
def health():
    return {"ok": True}

# NEW: quick sanity check
@app.get("/debug")
def debug():
    return {
        "model_loaded": hasattr(model, "predict"),
        "scaler_loaded": hasattr(scaler, "transform")
    }

# FIXED: robust predict with explicit return and error surfacing
@app.post("/predict")
def predict(x: InputRow):
    try:
        arr = np.array([[x.Pregnancies, x.Glucose, x.BloodPressure,
                         x.SkinThickness, x.Insulin, x.BMI,
                         x.DiabetesPedigreeFunction, x.Age]])
        arr_scaled = scaler.transform(arr)
        y = int(model.predict(arr_scaled)[0])   # 0 or 1
        return {"label": y}
    except Exception as e:
        # If anything goes wrong, you’ll see the message in the response/logs
        raise HTTPException(status_code=500, detail=str(e))
