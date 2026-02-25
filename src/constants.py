# src/constants.py
# 1. Feature Order: MUST match the XGBoost training column order exactly
FEATURES = [
    "amount", 
    "oldbalanceOrg", 
    "newbalanceOrig", 
    "errorBalanceOrig", 
    "type_encoded"
]

# 2. Categorical Mapping
TRANSACTION_TYPES = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# 3. Application Metadata
API_TITLE = "Sentinel-Fin API"
SYSTEM_NAME = "Sentinel-Fin Fraud Engine"

# 4. Heuristic Thresholds (Easier to tune)
HEURISTIC_AMOUNT_LIMIT = 1000
HEURISTIC_DRAIN_RATIO = 0.90