from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

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