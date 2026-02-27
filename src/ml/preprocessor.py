import pandas as pd
from src.constants import TRANSACTION_TYPES

def clean_data(file_path):
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    df = df[df['type'].isin(TRANSACTION_TYPES.keys())]
    
    # Feature Engineering: Fixing balance discrepancies (consistency fix)
    df['expected_new'] = df['oldbalanceOrg'] - df['amount']
    df['errorBalanceOrig'] = df['newbalanceOrig'] - df['expected_new']
    
    # mapping types from constants.py
    df['type_encoded'] = df['type'].map(TRANSACTION_TYPES)
    
    return df