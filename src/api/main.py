from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func  # Added for analytics
from pydantic import BaseModel
import pandas as pd
import shap
from contextlib import asynccontextmanager

from src.database.connection import engine, get_db
from src.database import models
from src.engine.loader import get_model

# 1. Setup the Model & Explainer Container
ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    model = get_model()
    ml_components["fraud_detector"] = model
    # Pre-initialize the SHAP explainer (TreeExplainer is fastest for XGBoost)
    ml_components["explainer"] = shap.TreeExplainer(model)
    yield
    ml_components.clear()

# 2. Schema
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
    return {"status": "active", "system": "Sentinel-Fin Fraud Engine"}

@app.post("/predict")
async def predict_fraud(data: Transaction, db: Session = Depends(get_db)):
    if not ml_components.get("fraud_detector"):
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
    model = ml_components["fraud_detector"]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    verdict = "FLAGGED" if prediction else "APPROVED"

    # NEW: SHAP Explainability (Why did it flag?)
    reasoning = []
    if probability > 0.1:  # Explain anything with even a small risk
        explainer = ml_components["explainer"]
        shap_values = explainer.shap_values(input_data)
        
        # Get feature impacts and sort them
        feature_names = input_data.columns
        # For binary classification, shap_values[0] is the vector of impacts
        impacts = dict(zip(feature_names, shap_values[0]))
        
        # Sort by absolute impact (top 3)
        top_features = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for feat, val in top_features:
            direction = "increased" if val > 0 else "decreased"
            reasoning.append(f"{feat} {direction} risk score")

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
    db.refresh(new_log)

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "verdict": verdict,
        "reasoning": reasoning,
        "log_id": new_log.id
    }

@app.get("/api/v1/analytics")
def get_analytics(db: Session = Depends(get_db)):
    total_processed = db.query(models.PredictionLog).count()
    total_flagged = db.query(models.PredictionLog).filter(models.PredictionLog.verdict == "FLAGGED").count()
    avg_prob = db.query(func.avg(models.PredictionLog.probability)).scalar() or 0
    
    # Get the 5 most recent threats
    threats = db.query(models.PredictionLog)\
        .filter(models.PredictionLog.verdict == "FLAGGED")\
        .order_by(models.PredictionLog.timestamp.desc())\
        .limit(5).all()

    # Manual conversion to dict to ensure Streamlit can read it
    recent_threats_list = []
    for t in threats:
        recent_threats_list.append({
            "id": t.id,
            "amount": t.amount,
            "probability": f"{t.probability:.2%}",
            "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M") 
        })

    return {
        "metrics": {
            "total_processed": total_processed,
            "total_flagged": total_flagged,
            "fraud_rate": f"{(total_flagged / total_processed * 100):.2f}%" if total_processed > 0 else "0%",
            "avg_confidence": f"{avg_prob:.2%}"
        },
        "recent_threats": recent_threats_list
    }