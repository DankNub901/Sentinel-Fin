import streamlit as st
import requests
import pandas as pd
import os

# Use 'api' for internal Docker networking
BACKEND_URL = os.environ.get("BACKEND_URL", "http://api:8000")

st.set_page_config(page_title="Sentinel-Fin Dashboard", layout="wide")

st.title("🛡️ Sentinel-Fin: Fraud Command Center")

# Sidebar for manual prediction
st.sidebar.header("Test New Transaction")
name_orig = st.sidebar.text_input("Sender Name", value="C_USER_1")
name_dest = st.sidebar.text_input("Recipient Name", value="M_MERCH_1")
amount = st.sidebar.number_input("Amount", min_value=0.0)
old_bal = st.sidebar.number_input("Old Balance", min_value=0.0)
new_bal = st.sidebar.number_input("New Balance", min_value=0.0)
t_type = st.sidebar.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])

# --- SIDEBAR LOGIC ---
if st.sidebar.button("Scan Transaction"):
    type_map = {"TRANSFER": 4, "CASH_OUT": 1, "PAYMENT": 3, "DEBIT": 2, "CASH_IN": 0}
    payload = {
        "amount": amount,
        "oldbalanceOrg": old_bal,
        "newbalanceOrig": new_bal,
        "type_encoded": type_map[t_type],
        "nameOrig": name_orig,
        "nameDest": name_dest 
    }
    try:
        res = requests.post(f"{BACKEND_URL}/predict", json=payload).json()
        # Save result to session state so it survives the next button click
        st.session_state['last_result'] = res
    except Exception as e:
        st.sidebar.error(f"Scan failed: {e}")

# Display results if they exist in session state
if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    
    if res.get("is_fraud"):
        st.sidebar.error(f"🚨 ALERT: {res.get('verdict')}")
        st.sidebar.write(f"**Fraud Probability:** {res.get('fraud_probability')}")
        
        if "reasoning" in res and res["reasoning"]:
            st.sidebar.write("---")
            st.sidebar.write("**AI Analysis (Technical):**")
            for reason in res["reasoning"]:
                st.sidebar.info(f"📍 {reason}")
            
            st.sidebar.write("---")
            # Moved OUTSIDE the Scan button block so it stays active
            if st.sidebar.button("📄 Generate Audit Report"):
                with st.sidebar.spinner("Ollama is analyzing regulations..."):
                    try:
                        from src.compliance_rag.reporter import ComplianceReporter
                        reporter = ComplianceReporter()
                        
                        audit_report = reporter.generate_report(res["reasoning"])
                        
                        st.sidebar.markdown("### 🖋️ Official Justification")
                        st.sidebar.success(audit_report)
                        
                        st.sidebar.download_button(
                            label="Download Report",
                            data=audit_report,
                            file_name=f"audit_log_{res.get('log_id', 'new')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.sidebar.error(f"AI Service busy: Ensure 'llama3' is pulled in Ollama.")
    else:
        st.sidebar.success("✅ Transaction Approved")

# --- MAIN DASHBOARD ---
st.subheader("Live System Metrics")

try:
    # Use BACKEND_URL consistently
    response = requests.get(f"{BACKEND_URL}/api/v1/analytics", timeout=5)
    if response.status_code == 200:
        data = response.json()
        m = data.get("metrics", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scanned", m.get("total_processed", 0))
        col2.metric("Total Flagged", m.get("total_flagged", 0), delta_color="inverse")
        # FIXED: Removed formatting that caused the 'f' error
        col3.metric("Avg. Confidence", m.get("avg_confidence", "0%"))

        st.subheader("Recent Flagged Threats")
        if data.get("recent_threats"):
            df = pd.DataFrame(data["recent_threats"])
            st.dataframe(df[["id", "sender", "receiver", "amount", "probability", "timestamp"]], use_container_width=True)
        else:
            st.info("No threats detected yet. System is clear.")
    else:
        st.warning("⚠️ API is booting up. Please refresh in a few seconds...")

except Exception as e:
    st.error("📡 Connecting to Sentinel Engine...")
    st.info("Check logs: `docker logs -f sentinel_api`")
    st.write(f"Connection Details: {e}")