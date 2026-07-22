from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, text  # Added for analytics
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import json
import time
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from src.database.connection import engine, get_db
from src.database import models
from src.engine.loader import get_calibrated_model
from src.database.redis_client import get_redis, check_redis_health, close_redis

from src.constants import (
    FEATURES, 
    TRANSACTION_TYPES, 
    API_TITLE, 
    SYSTEM_NAME,
    HEURISTIC_AMOUNT_LIMIT,
    HEURISTIC_DRAIN_RATIO
)

SHAP_BUFFER_LOGS = []
SHAP_BUFFER_FEATURES = []
OLDEST_FRAUD_TIMESTAMP = None
SHAP_LOCK = asyncio.Lock()

# 1. Component Lifespan
ml_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load model once on startup
    model, calibrator, features = get_calibrated_model()
    if not model:
        print("CRITICAL: Failed to load calibrated fraud brain.")
    
    ml_components["fraud_detector"] = model
    ml_components["explainer"] = shap.TreeExplainer(model)
    
    # 2. Connect to Redis Feature Store
    print("🔌 Connecting to Redis Feature Store...")
    try:
        await get_redis()
        if await check_redis_health():
            print("✅ Redis Feature Store is Online!")
        else:
            print("⚠️ Redis Startup Warning: Ping failed.")
    except Exception as e:
        print(f"⚠️ Redis Connection Error: {e}")

    yield
    ml_components.clear()
    print("🧹 Closing Redis connection pool...")
    await close_redis()

# 2. Schema
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    nameDest: str

    oldbalanceOrg: float = 0.0      
    newbalanceOrig: float = 0.0     
    type_encoded: int = 4

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

models.Base.metadata.create_all(bind=engine)
app = FastAPI(title=API_TITLE, lifespan=lifespan)

# 3. Core Logic Helpers
async def build_behavioral_features(tx_dict: dict, db: Session, redis_conn=None) -> dict:
    if tx_dict.get("channel_risk") is None:
        tx_dict["channel_risk"] = 1.0 if tx_dict["type"] in ["TRANSFER", "CASH_OUT"] else 0.0
        
    if tx_dict.get("late_night_flag") is None:
        hour = tx_dict["step"] % 24
        tx_dict["late_night_flag"] = 1 if hour <= 4 else 0
        tx_dict["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        tx_dict["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
        
    if tx_dict.get("is_layering_attempt") is None:
        tx_dict["is_layering_attempt"] = 1 if tx_dict["nameOrig"] == tx_dict["nameDest"] else 0

    # a. fallbacks for manual entry (Can be replaced with raw SQL/Redis window functions)
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

async def process_inference_pipeline(raw_transactions: List[dict], db: Session, default_status: str = "PENDING") -> tuple[List[models.PredictionLog], pd.DataFrame]:
    """The single source of truth enginer for all prediction requests"""

    redis_conn = await get_redis()

    processed_rows = [
        await build_behavioral_features(tx, db, redis_conn=redis_conn) 
        for tx in raw_transactions
        ]
    df_features = pd.DataFrame(processed_rows)[FEATURES]

    model = ml_components["fraud_detector"]
    batch_probs = model.predict(xgb.DMatrix(df_features))
    batch_preds = (batch_probs >= 0.5).astype(int)

    new_logs = []
    for i, row in enumerate(processed_rows):
        prob = float(batch_probs[i])
        pred = int(batch_preds[i])
        
        # Heuristic rules engine
        drain_ratio = row['amount'] / row['oldbalanceOrg'] if row['oldbalanceOrg'] > 0 else 0
        if row['amount'] > HEURISTIC_AMOUNT_LIMIT and drain_ratio > HEURISTIC_DRAIN_RATIO:
            pred = 1
            prob = max(prob, 0.95)

        new_logs.append(models.PredictionLog(
            amount=float(row["amount"]),
            old_balance=float(row["oldbalanceOrg"]),
            new_balance=float(row['newbalanceOrig']),
            expected_new_balance=float(row["oldbalanceOrg"] - row["amount"]),
            type_code=int(row.get('type_encoded', TRANSACTION_TYPES.get("TRANSFER", 4))),
            name_orig=row.get('nameOrig', "Unknown"),
            name_dest=row.get('nameDest', "Unknown"),
            is_simulated=bool(row.get('is_simulated', True)),
            session_id=row.get('session_id'),
            verdict="FLAGGED" if pred else "APPROVED",
            probability=prob,
            is_fraud=bool(pred),
            status = default_status,
            shap_summary={}
        ))
    return new_logs, df_features

async def check_and_flush_shap_bucket(db: Session):
    global SHAP_BUFFER_LOGS, SHAP_BUFFER_FEATURES, OLDEST_FRAUD_TIMESTAMP

    if not SHAP_BUFFER_LOGS:
        return {"current_pending_fraud":0, "oldest_fraud_age_sec":0.0}

    oldest_age_seconds = time.time() - OLDEST_FRAUD_TIMESTAMP

    if len(SHAP_BUFFER_LOGS) >= 5 or oldest_age_seconds > 3.0:
        df_fraud_features = pd.concat(SHAP_BUFFER_FEATURES, axis=0).reset_index(drop=True)

        shap_values_subset = ml_components["explainer"].shap_values(xgb.DMatrix(df_fraud_features))
        feature_names = df_fraud_features.columns

        for idx, log_entry in enumerate(SHAP_BUFFER_LOGS):
            
            active_log = db.merge(log_entry)
            impacts = dict(zip(feature_names, shap_values_subset[idx]))
            active_log.shap_summary = {k: float(v) for k, v in impacts.items()}
            active_log.status = "PROCESSED"

        db.commit()

        SHAP_BUFFER_LOGS.clear()
        SHAP_BUFFER_FEATURES.clear()
        OLDEST_FRAUD_TIMESTAMP = None

        return {"current_pending_fraud":0, "oldest_fraud_age_sec":0.0}

    return{
        "current_pending_fraud": len(SHAP_BUFFER_LOGS),
        "oldest_fraud_age_sec": round(oldest_age_seconds, 2)
    }


# 4. API Endpoints
@app.get("/")
def health_check():
    return {"status": "active", "system": SYSTEM_NAME}

@app.post("/predict")
async def predict_fraud(data: Transaction, db: Session = Depends(get_db)):
    if not ml_components.get("fraud_detector"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logs, df_features = await process_inference_pipeline([data.model_dump()], db, default_status="PROCESSED")
    target_log = logs[0] 

    
    # a2. AML & HEURISTIC GUARDRAILS (manual entry)
    reasoning = []
    drain_ratio = data.amount / data.oldbalanceOrg if data.oldbalanceOrg > 0 else 0
    
    # manual heuristic strings
    if data.amount > HEURISTIC_AMOUNT_LIMIT and drain_ratio > HEURISTIC_DRAIN_RATIO:
        reasoning.append(f"Heuristic Alert: Account Drain Detected ({drain_ratio:.2%} depletion)")
        # TRANSFER: Flag reason (Layering)
        if data.type_encoded == TRANSACTION_TYPES.get("TRANSFER", 4):
            reasoning.append("AML Warning: Possible 'Layering' activity. Rapid fund shifting detected.")
        #CASH_OUT: Flag reason (Integration)
        elif data.type_encoded == TRANSACTION_TYPES.get("CASH_OUT", 1):
            reasoning.append("AML Warning: Possible 'Integration' phase. High-value liquidation.")

    if target_log.is_fraud:

        # calculate SHAP values instantly
        shap_values = ml_components["explainer"].shap_values(xgb.DMatrix(df_features))
        feature_names = df_features.columns

        impacts = dict(zip(feature_names, shap_values[0]))
        target_log.shap_summary = {k: float(v) for k, v in impacts.items()}

        # Map to text strings for the investigator
        top_features = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for feat, val in top_features:
            direction = "increased" if val > 0 else "decreased"
            reasoning.append(f"AI Factor: {feat} {direction} risk score")

    db.add(target_log)
    db.commit()
    db.refresh(target_log)

    return {
        "is_fraud": target_log.is_fraud,
        "fraud_probability": round(target_log.probability, 4),
        "verdict": target_log.verdict,
        "reasoning": reasoning,
        "log_id": target_log.id
    }


# --- Add the Batch Endpoint ---
@app.post("/predict/batch")
async def predict_batch(batch: TransactionBatch, db: Session = Depends(get_db)):
    global SHAP_BUFFER_LOGS, SHAP_BUFFER_FEATURES, OLDEST_FRAUD_TIMESTAMP

    if not ml_components.get("fraud_detector"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_tx_list = [t.model_dump() for t in batch.transactions]
    logs, df_features = await process_inference_pipeline(raw_tx_list, db, default_status="PENDING")
    
    db.add_all(logs)
    db.commit()

    async with SHAP_LOCK:
        for i, log in enumerate(logs):
            if log.is_fraud:
                if not SHAP_BUFFER_LOGS:
                    OLDEST_FRAUD_TIMESTAMP = time.time()
                    
                SHAP_BUFFER_LOGS.append(log)
                SHAP_BUFFER_FEATURES.append(df_features.iloc[[i]])

        stats = await check_and_flush_shap_bucket(db)

    return {
        "processed": len(logs), 
        "flags": sum(1 for l in logs if l.is_fraud),
        "bucket_status": stats
    }

@app.get("/api/v1/analytics")
def get_analytics(db: Session = Depends(get_db)):
    analytics = db.query(
        func.count(models.PredictionLog.id).label("total"),
        func.count(func.nullif(models.PredictionLog.verdict != "FLAGGED", True)).label("flagged"),
        func.avg(models.PredictionLog.probability).label("avg_prob")
    ).first()

    total_processed = analytics.total or 0
    total_flagged = analytics.flagged or 0
    avg_prob = analytics.avg_prob or 0
    
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