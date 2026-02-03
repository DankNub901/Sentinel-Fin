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
    
    amount = Column(Float)
    old_balance = Column(Float)
    new_balance = Column(Float)

    name_orig = Column(String)  # Sender (e.g., C12345)
    name_dest = Column(String)  # Receiver (e.g., M67890)

    verdict = Column(String)  # "FLAGGED" or "APPROVED"
    probability = Column(Float)
    is_fraud = Column(Boolean)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    status = Column(String, default="PENDING")  # To track "RESOLVED", "DISMISSED", etc.
    shap_summary = Column(JSON)                 # To store the SHAP math data
    reviewer_notes = Column(String)             # To store Llama 3 reports or human notes
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())