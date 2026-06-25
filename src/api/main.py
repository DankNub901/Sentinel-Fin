from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func  # Added for analytics
from pydantic import BaseModel
import pandas as pd
import numpy as np
import shap
import json
from contextlib import asynccontextmanager
from typing import List, Optional

from src.database.connection import engine, get_db
from src.database import models
from src.engine.loader import get_calibrated_model

from src.constants import (
    FEATURES, 
    TRANSACTION_TYPES, 
    API_TITLE, 
    SYSTEM_NAME,
    HEURISTIC_AMOUNT_LIMIT,
    HEURISTIC_DRAIN_RATIO
)

# 1. Setup the Model & Explainer Container
ml_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    model, calibrator, features = get_calibrated_model()
    
    if not model:
        print("CRITICAL: Failed to load calibrated fraud brain.")
    
    ml_components["fraud_detector"] = model
    # Pre-initialize the SHAP explainer (TreeExplainer is fastest for XGBoost)
    ml_components["explainer"] = shap.TreeExplainer(model)
    yield
    ml_components.clear()

# 2. Schema
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str
    is_simulated: bool = False
    session_id: Optional[str] = None
    
    # Allow optional injection of behavioral features (passed by simulation script)
    channel_risk: Optional[float] = None
    dest_mule_heat: Optional[float] = None
    sender_recent_velocity: Optional[float] = None
    amt_acceleration: Optional[float] = None
    sender_volatility: Optional[float] = None
    is_new_dest_pair: Optional[int] = None
    personal_amt_z_score: Optional[float] = None
    late_night_flag: Optional[int] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    global_step_velocity: Optional[float] = None
    is_layering_attempt: Optional[int] = None
    sender_fan_out: Optional[int] = None
    account_activity_density: Optional[float] = None
    time_since_last_tx: Optional[float] = None



class TransactionBatch(BaseModel):
    transactions: List[Transaction]

# 3. Initialize App
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title=API_TITLE, lifespan=lifespan)

def build_behavioral_features(tx_dict: dict, db: Session) -> dict:
    """
    Enriches raw transaction data with behavioral metrics.
    If features are not already provided (e.g., Manual Investigator Entry),
    this function falls back to baseline values or live DB lookups.
    """
    # 1. Direct conversions from raw inputs
    if tx_dict.get("channel_risk") is None:
        tx_dict["channel_risk"] = 1.0 if tx_dict["type"] in ["TRANSFER", "CASH_OUT"] else 0.0
        
    if tx_dict.get("late_night_flag") is None:
        hour = tx_dict["step"] % 24
        tx_dict["late_night_flag"] = 1 if hour <= 4 else 0
        tx_dict["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        tx_dict["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        
    if tx_dict.get("is_layering_attempt") is None:
        tx_dict["is_layering_attempt"] = 1 if tx_dict["nameOrig"] == tx_dict["nameDest"] else 0

    # 2. Historical Fallbacks for Manual Entry (Can be replaced with raw SQL/Redis window functions)
    defaults = {
        "dest_mule_heat": 1.0,
        "sender_recent_velocity": 1.0,
        "amt_acceleration": 1.0,
        "sender_volatility": 0.0,
        "is_new_dest_pair": 1,
        "personal_amt_z_score": 0.0,
        "global_step_velocity": 10.0,
        "sender_fan_out": 1,
        "account_activity_density": 0.5,
        "time_since_last_tx": 0.0
    }
    
    for feat, fallback in defaults.items():
        if tx_dict.get(feat) is None:
            tx_dict[feat] = fallback
            
    return tx_dict

@app.get("/")
def health_check():
    return {"status": "active", "system": SYSTEM_NAME}

@app.post("/predict")
async def predict_fraud(data: Transaction, db: Session = Depends(get_db)):
    if not ml_components.get("fraud_detector"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tx_data = build_behavioral_features(data.model_dump(), db)
    input_data = pd.DataFrame([tx_data])[FEATURES]

    dmatrix_input = xgb.DMatrix(input_data)

    # 2. AI Prediction (Base Model)
    model = ml_components["fraud_detector"]
    prediction = 1 if probability >= 0.5 else 0
    probability = float(model.predict(dmatrix_input)[0])

    # 3. NEW: AML & HEURISTIC GUARDRAILS
    # These catch "Common Sense" fraud that the ML model might miss
    reasoning = []
    drain_ratio = data.amount / data.oldbalanceOrg if data.oldbalanceOrg > 0 else 0
    
    # Account Drain Check
    if data.amount > HEURISTIC_AMOUNT_LIMIT and drain_ratio > HEURISTIC_DRAIN_RATIO:
        prediction = 1  # Force the flag
        probability = max(probability, 0.95)  # Ensure high confidence for the auditor
        reasoning.append(f"Heuristic Alert: Account Drain Detected ({drain_ratio:.2%} depletion)")

        # TRANSFER: Flag reason (Layering)
        if data.type_encoded == TRANSACTION_TYPES["TRANSFER"]:
            reasoning.append("AML Warning: Possible 'Layering' activity. Rapid fund shifting to external account detected.")
            reasoning.append("High risk of 'Pass-through' behavior; source of funds may be obscured.")
        
        #CASH_OUT: Flag reason (Integration)
        elif data.type_encoded == TRANSACTION_TYPES["CASH_OUT"]:
            reasoning.append("AML Warning: Possible 'Integration' phase. High-value liquidation of funds to untraceable cash.")

    # 4. SHAP Explainability (Enhanced to store impacts)
    shap_data = {} # To store for DB
    if probability > 0.1:
        explainer = ml_components["explainer"]
        shap_values = explainer.shap_values(input_data)
        feature_names = input_data.columns
        impacts = dict(zip(feature_names, shap_values[0]))
        
        # Convert floats to strings/standard floats for JSON compatibility
        shap_data = {k: float(v) for k, v in impacts.items()}
        
        top_features = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feat, val in top_features:
            direction = "increased" if val > 0 else "decreased"
            reasoning.append(f"AI Factor: {feat} {direction} risk score")

    # 5. Final Verdict Determination
    verdict = "FLAGGED" if prediction else "APPROVED"

    # 6. Save to Database
    new_log = models.PredictionLog(
        amount=data.amount,
        old_balance=data.oldbalanceOrg,
        new_balance=data.newbalanceOrig,
        expected_new_balance=expected_new, # Track the math fix
        type_code=data.type_encoded,        # Track the raw position
        name_orig=data.nameOrig,
        name_dest=data.nameDest,
        is_simulated=data.is_simulated,     # Track if it's a bot
        session_id=data.session_id,         # Group the simulation
        verdict="FLAGGED" if prediction else "APPROVED",
        probability=probability,
        is_fraud=bool(prediction),
        shap_summary=shap_data,
        status="PENDING"
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(probability, 4),
        "verdict": verdict,
        "reasoning": reasoning,
        "log_id": new_log.id
    }

# --- 1. Update Schema ---
class TransactionBatch(BaseModel):
    transactions: List[Transaction]

# --- 2. Add the Batch Endpoint ---
@app.post("/predict/batch")
async def predict_batch(batch: TransactionBatch, db: Session = Depends(get_db)):
    if not ml_components.get("fraud_detector"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # A. Efficiently convert Pydantic list to a single DataFrame
    df_batch = pd.DataFrame([t.model_dump() for t in batch.transactions])
    
    # B. Vectorized Feature Engineering
    df_batch['expected_new'] = df_batch['oldbalanceOrg'] - df_batch['amount']
    df_batch['errorBalanceOrig'] = df_batch['newbalanceOrig'] - df_batch['expected_new']
    
    # Prepare data for model (keep only the columns the model was trained on)
    input_features = df_batch[FEATURES]

    dmatrix_batch = xgb.DMatrix(input_features)

    # C. Batch AI Prediction
    model = ml_components["fraud_detector"]
    batch_preds = (batch_probs >= 0.5).astype(int)
    batch_probs = model.predict(dmatrix_batch)

    # D. Batch SHAP Explanations
    explainer = ml_components["explainer"]
    # We calculate SHAP for the whole batch at once
    shap_values_batch = explainer.shap_values(input_features)
    feature_names = input_features.columns

    # E. Process Results and Save to DB
    new_logs = []
    for i in range(len(df_batch)):
        # Calculate Heuristic/AML logic per row
        row = df_batch.iloc[i]
        current_prob = float(batch_probs[i])
        current_pred = int(batch_preds[i])
        
        # AML Guardrail (Manual override logic)
        drain_ratio = row['amount'] / row['oldbalanceOrg'] if row['oldbalanceOrg'] > 0 else 0
        if row['amount'] > HEURISTIC_AMOUNT_LIMIT and drain_ratio > HEURISTIC_DRAIN_RATIO:
            current_pred = 1
            current_prob = max(current_prob, 0.95)

        # Extract SHAP for this specific row
        impacts = dict(zip(feature_names, shap_values_batch[i]))
        shap_json = {k: float(v) for k, v in impacts.items()}

        # Create Log Object
        new_logs.append(models.PredictionLog(
            amount=float(row['amount']),
            old_balance=float(row['oldbalanceOrg']),
            new_balance=float(row['newbalanceOrig']),
            expected_new_balance=float(row['expected_new']),
            type_code=int(row['type_encoded']),
            name_orig=row.get('nameOrig', "Unknown"),
            name_dest=row.get('nameDest', "Unknown"),
            is_simulated=bool(row.get('is_simulated', True)), # Assume True for batch
            session_id=row.get('session_id'),
            verdict="FLAGGED" if current_pred else "APPROVED",
            probability=current_prob,
            is_fraud=bool(current_pred),
            status = "PENDING",
            shap_summary=shap_json
        ))

    # F. Bulk Save to Postgres (The real speed boost)
    db.add_all(new_logs)
    db.commit()

    return {
        "processed": len(new_logs), 
        "flags": sum(1 for l in new_logs if l.is_fraud)
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
            "sender": t.name_orig,
            "receiver": t.name_dest,
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

# Temporary placeholder for Llama logic
def call_llama_logic(log_entry):
    return f"AI Audit for Log #{log_entry.id}: Analysis pending integration with Ollama."

@app.post("/api/v1/audit/{log_id}")
async def generate_llm_audit(log_id: int, db: Session = Depends(get_db)):
    # 1. Fetch the log from the DB
    log_entry = db.query(models.PredictionLog).filter(models.PredictionLog.id == log_id).first()
    
    # 2. Feed the log data + SHAP summary into Llama 3
    # (We will write this prompt tomorrow to make it sound like a forensic accountant)
    audit_report = call_llama_logic(log_entry) 
    
    # 3. Update the DB with the report
    log_entry.reviewer_notes = audit_report
    db.commit()
    
    return {"report": audit_report}