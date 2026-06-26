import os
import joblib
from huggingface_hub import hf_hub_download

# Use your verified repo and filename
REPO_ID = "xNub/xgboost-fraud-detection-calibrated"
FILENAME = "calibrated_fraud_model.joblib"

def get_calibrated_model():
    """
    Downloads the model from Hugging Face (if not already cached)
    and loads it into memory.
    """
    print(f"--- Sentinel-Fin: Loading Intelligence from {REPO_ID} ---")
    
    try:
        # This checks your local cache first, then downloads if needed
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=FILENAME,
            token=os.getenv("HF_TOKEN")
        )
        
        artifacts = joblib.load(model_path)
        model = artifacts["model"]
        calibrator = artifacts["calibrator"]
        features = artifacts["features"]

        print("--- Success: Calibrated Fraud Brain is Online ---")
        return model, calibrator, features
        
    except Exception as e:
        print(f"--- Error: Could not load calibrated pipeline. {e} ---")
        return None, None, None

if __name__ == "__main__":
    # Test the loader independently
    model, calibrator, features = get_calibrated_model()
    if model:
        print(f"Verified features expected by the model: {features}")