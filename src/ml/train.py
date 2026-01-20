import os
import joblib
import xgboost as xgb
from huggingface_hub import HfApi, login
from src.ml.preprocessor import clean_data
from sklearn.model_selection import train_test_split

def run_pipeline():
    # 1. Login to HF
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("CRITICAL: HF_TOKEN not found in environment!")
        return
    login(token=hf_token)

    # 2. Train the Model
    print("--- Training Fraud Model ---")
    df, encoder = clean_data("/app/data/raw/PaySim.csv")
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrig', 'type_encoded']
    X = df[features]
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scale_pos_weight=99 handles the rare fraud cases
    model = xgb.XGBClassifier(scale_pos_weight=99, n_estimators=100)
    model.fit(X_train, y_train)

    # 3. Save locally inside container
    model_path = "fraud_model.pkl"
    joblib.dump(model, model_path)

    # 4. Create Repo & Upload to Hugging Face
    api = HfApi()
    repo_id = "xNub/Sentinel-XGBoost"
    
    print(f"--- Creating/Uploading to HF: {repo_id} ---")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="fraud_model.pkl",
        repo_id=repo_id,
        repo_type="model",
    )
    print("--- Success! Model is versioned on Hugging Face ---")

if __name__ == "__main__":
    run_pipeline()