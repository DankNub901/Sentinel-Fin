# src/constants.py
# 1. Feature Order: MUST match the XGBoost training column order exactly
FEATURES = [
    "amount", 
    "channel_risk",
    "dest_mule_heat", 
    "sender_recent_velocity", 
    "amt_acceleration",
    "sender_volatility", 
    "is_new_dest_pair",
    "personal_amt_z_score", 
    "late_night_flag", 
    "hour_sin", 
    "hour_cos",
    "global_step_velocity",
    "is_layering_attempt",
    "sender_fan_out",
    "account_activity_density",
    "time_since_last_tx"
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