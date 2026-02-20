from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from datetime import datetime
from sqlalchemy.sql import func
from src.database.connection import Base

class TransactionAudit(Base):
    __tablename__ = "transaction_audit"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # ML Results
    fraud_score = Column(Float)  # Probability (0.0 to 1.0)
    is_flagged = Column(Integer)  # 0 (Safe) or 1 (Fraud)
    
    # The "Why" (SHAP Values)
    audit_reasoning = Column(JSON) 
    
    # Context
    customer_id = Column(String, index=True)
    type = Column(String)  # TRANSFER, CASH_OUT, etc.

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # --- Transaction Data ---
    amount = Column(Float)
    old_balance = Column(Float)
    new_balance = Column(Float)
    # New Field: To store what the balance SHOULD have been (Fix #2 in your checklist)
    expected_new_balance = Column(Float) 

    name_orig = Column(String)
    name_dest = Column(String)
    type_code = Column(Integer) # For the position-based XGBoost requirement

    # --- Simulation Tracking (Objective Fix) ---
    is_simulated = Column(Boolean, default=False, index=True)
    session_id = Column(String, nullable=True, index=True) # UUID or "GAN_TEST_01"

    # --- ML Results ---
    verdict = Column(String)
    probability = Column(Float)
    is_fraud = Column(Boolean)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # --- Metadata & Logic ---
    status = Column(String, default="PENDING")
    shap_summary = Column(JSON)
    reviewer_notes = Column(String)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())