import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Sentinel-Fin Dashboard", layout="wide")

st.title("🛡️ Sentinel-Fin: Fraud Command Center")

# Sidebar for manual prediction
st.sidebar.header("Test New Transaction")
amount = st.sidebar.number_input("Amount", min_value=0.0)
old_bal = st.sidebar.number_input("Old Balance", min_value=0.0)
new_bal = st.sidebar.number_input("New Balance", min_value=0.0)
t_type = st.sidebar.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])

if st.sidebar.button("Scan Transaction"):
    # Map type to your encoded values
    type_map = {"TRANSFER": 4, "CASH_OUT": 1, "PAYMENT": 3, "DEBIT": 2, "CASH_IN": 0}
    payload = {
        "amount": amount,
        "oldbalanceOrg": old_bal,
        "newbalanceOrig": new_bal,
        "type_encoded": type_map[t_type]
    }
    res = requests.post("http://api:8000/predict", json=payload).json()
    
    if res["is_fraud"]:
        st.sidebar.error(f"🚨 ALERT: {res['verdict']}")
        st.sidebar.write(f"**Fraud Probability:** {res['fraud_probability']}")
        
        # This is the part that shows the SHAP reasons
        if "reasoning" in res and res["reasoning"]:
            st.sidebar.write("---")
            st.sidebar.write("**AI Analysis (Why?):**")
            for reason in res["reasoning"]:
                st.sidebar.info(f"📍 {reason}")
    else:
        st.sidebar.success(f"✅ APPROVED (Prob: {res['fraud_probability']})")

# Main Dashboard
try:
    data = requests.get("http://api:8000/api/v1/analytics").json()
    m = data["metrics"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scanned", m["total_processed"])
    col2.metric("Total Flagged", m["total_flagged"], delta_color="inverse")
    col3.metric("Avg. Confidence", m["avg_confidence"])

    st.subheader("Recent Flagged Threats")
    if data["recent_threats"]:
        df = pd.DataFrame(data["recent_threats"])
        st.table(df[["id", "amount", "probability", "timestamp"]])
    else:
        st.info("No threats detected yet.")
except Exception as e:
    st.error("Could not connect to API. Is the backend running?")