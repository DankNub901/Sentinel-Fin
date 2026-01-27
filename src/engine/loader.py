import os
import joblib
from huggingface_hub import hf_hub_download

# Use your verified repo and filename
REPO_ID = "xNub/Sentinel-XGBoost"
FILENAME = "fraud_model.pkl"

def get_model():
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
        
        model = joblib.load(model_path)
        print("--- Success: Fraud Detection Brain is Online ---")
        return model
        
    except Exception as e:
        print(f"--- Error: Could not load model. {e} ---")
        return None

if __name__ == "__main__":
    # Test the loader independently
    get_model()