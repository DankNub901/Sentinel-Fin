import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(file_path):
    print(f"Reading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter for high-risk transaction types
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    
    # Feature Engineering: Catching balance discrepancies
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Encode 'type' as numbers (TRANSFER=0, CASH_OUT=1, etc.)
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    return df, le