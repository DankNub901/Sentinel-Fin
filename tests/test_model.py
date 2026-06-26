import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from src.engine.loader import get_calibrated_model
from src.constants import FEATURES

@pytest.fixture(scope="module")
def fraud_model():
    """Loads the raw XGBoost booster asset once for the model testing layer"""
    model, calibrator, features = get_calibrated_model()
    if not model:
        pytest.skip("Hugging Face model asset could not be retrieved. Skipping model tests.")
    return model

@pytest.fixture
def baseline_features():
    """Generates a valid single-row DataFrame matching the exact trained FEATURES schema"""
    # Initialize all 16 features with realistic baseline 'safe' values
    base_data = {feat: 0.0 for feat in FEATURES}
    
    # Set explicit non-zero baseline values for features requiring it
    base_data["amount"] = 50.0
    base_data["global_step_velocity"] = 10.0
    base_data["account_activity_density"] = 0.1
    
    return pd.DataFrame([base_data])[FEATURES]


def test_model_probability_bounds(fraud_model, baseline_features):
    """Statistically verify that raw booster predictions reside strictly in [0.0, 1.0]"""
    dmatrix = xgb.DMatrix(baseline_features)
    prob = float(fraud_model.predict(dmatrix)[0])
    
    assert 0.0 <= prob <= 1.0, f"Model predicted out-of-bounds probability: {prob}"


def test_inference_determinism(fraud_model, baseline_features):
    """Ensure the model is perfectly deterministic given immutable feature matrices"""
    dmatrix_first = xgb.DMatrix(baseline_features)
    dmatrix_second = xgb.DMatrix(baseline_features.copy())
    
    prob_first = float(fraud_model.predict(dmatrix_first)[0])
    prob_second = float(fraud_model.predict(dmatrix_second)[0])
    
    assert prob_first == prob_second, "Inference engine returned non-deterministic probabilities!"


def test_directional_sensitivity_to_risk(fraud_model, baseline_features):
    """
    Verify that the model is directionally sensitive to known risk variables.
    Spiking risk metrics should mathematically force an increase in fraud probability.
    """
    # 1. Get baseline safe score
    dmatrix_safe = xgb.DMatrix(baseline_features)
    prob_safe = float(fraud_model.predict(dmatrix_safe)[0])
    
    # 2. Forge a highly anomalous, risky transaction profile
    risky_features = baseline_features.copy()
    risky_features["amount"] = 850000.0
    # Activate core behavioral indicators found in your Polars pipeline
    risky_features["channel_risk"] = 1.0          # TRANSFER / CASH_OUT channel
    risky_features["dest_mule_heat"] = 45.0        # Recipient is a high-volume hub
    risky_features["is_new_dest_pair"] = 1         # Never interacted before
    risky_features["personal_amt_z_score"] = 6.5   # Massive standard deviation spike
    risky_features["late_night_flag"] = 1          # Occurred during high-risk hours
    
    dmatrix_risk = xgb.DMatrix(risky_features)
    prob_risk = float(fraud_model.predict(dmatrix_risk)[0])
    
    # The statistical brain must recognize this vector shift
    assert prob_risk > prob_safe, (
        f"Model failed risk sensitivity test. Safe Prob: {prob_safe} | Risk Prob: {prob_risk}"
    )