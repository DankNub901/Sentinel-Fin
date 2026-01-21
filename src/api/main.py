from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd
from contextlib import asynccontextmanager

from src.database.connection import engine, get_db
from src.database import models
from src.ml.loader import get_model

# 1. Setup the Model Container
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    ml_models["fraud_detector"] = get_model()
    yield
    ml_models.clear()

# 2. Schema for incoming requests
class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    type_encoded: int

# 3. Initialize App
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Sentinel-Fin API", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "active", "system": "Sentinel-Fin Fraud Engine", "database": "connected"}

@app.post("/predict")
async def predict_fraud(data: Transaction, db: Session = Depends(get_db)):
    if not ml_models.get("fraud_detector"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature Engineering
    error_balance = data.newbalanceOrig + data.amount - data.oldbalanceOrg
    input_data = pd.DataFrame([{
        "amount": data.amount, 
        "oldbalanceOrg": data.oldbalanceOrg,
        "newbalanceOrig": data.newbalanceOrig, 
        "errorBalanceOrig": error_balance,
        "type_encoded": data.type_encoded
    }])

    # AI Prediction
    prediction = ml_models["fraud_detector"].predict(input_data)[0]
    probability = ml_models["fraud_detector"].predict_proba(input_data)[0][1]
    verdict = "FLAGGED" if prediction else "APPROVED"

    # Save to Database
    new_log = models.PredictionLog(
        amount=data.amount,
        old_balance=data.oldbalanceOrg,
        new_balance=data.newbalanceOrig,
        verdict=verdict,
        probability=float(probability),
        is_fraud=bool(prediction)
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log) # This gets the ID back from the DB

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "verdict": verdict,
        "log_id": new_log.id
    }