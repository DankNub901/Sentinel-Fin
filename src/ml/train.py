import os
import joblib
import xgboost as xgb
from huggingface_hub import HfApi, login
from src.ml.preprocessor import clean_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import confusion_matrix
from src.constants import FEATURES

def run_pipeline():
    # 1. Login to HF
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("CRITICAL: HF_TOKEN not found in environment!")
        return
    login(token=hf_token)

    # 2. Train the Model
    print("--- Training Fraud Model ---")
    df = clean_data("/app/data/raw/PaySim.csv")
    
    X = df[FEATURES]
    y = df['isFraud']
    
    # TRIPLE SPLIT: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    model = xgb.XGBClassifier(
    n_estimators=200,         # High enough for variety, low enough for speed
    learning_rate=0.07,       # Balanced with 200 trees
    max_depth=7,              # Deep enough for the 5-feature interactions
    scale_pos_weight=25,      # The "Anti-Paranoia" setting
    subsample=0.8,            # Forces trees to be diverse
    colsample_bytree=0.9,     # Ensures features aren't over-relied upon
    min_child_weight=3,       # one "must-add" to prevent overfitting to outliers
    random_state=42,
    tree_method="hist",
    n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Evaluate Performance (The "Final Exam")
    print("\n --- MODEL PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))

    print("\n --- CONFUSION MATRIX (Raw Counts) ---")
    print(confusion_matrix(y_test, y_pred))
    
    auprc = average_precision_score(y_test, y_probs)
    print(f" Area Under PR Curve (AUPRC): {auprc:.4f}")

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