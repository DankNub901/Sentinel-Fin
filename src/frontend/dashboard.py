import streamlit as st
import requests
import pandas as pd
import os
import time
import subprocess
import signal

# --- CONFIG ---
BACKEND_URL = os.environ.get("BACKEND_URL", "http://api:8000")

st.set_page_config(page_title="Sentinel-Fin Dashboard", layout="wide")

# --- SIMULATION STATE MANAGEMENT ---
if "sim_process_pid" not in st.session_state:
    st.session_state.sim_process_pid = None

def start_simulation(rate):
    # Runs generator.py as a background process within the container
    proc = subprocess.Popen(
        ["python3", "src/engine/generator.py"],
        env={**os.environ, "VELOCITY": str(rate)}
    )
    st.session_state.sim_process_pid = proc.pid

def stop_simulation():
    if st.session_state.sim_process_pid:
        try:
            os.kill(st.session_state.sim_process_pid, signal.SIGTERM)
            st.session_state.sim_process_pid = None
        except:
            st.session_state.sim_process_pid = None

# --- UI HEADER ---
st.title("🛡️ Sentinel-Fin: Fraud Command Center")

# --- SIDEBAR: MANUAL SCAN & AUDIT (Your Original Logic) ---
st.sidebar.header("🏦 Manual Investigator Tool")
with st.sidebar.expander("Single Transaction Scan", expanded=True):
    name_orig = st.text_input("Sender Name", value="C_USER_1")
    name_dest = st.text_input("Recipient Name", value="M_MERCH_1")
    amount = st.number_input("Amount", min_value=0.0)
    old_bal = st.number_input("Old Balance", min_value=0.0)
    new_bal = st.number_input("New Balance", min_value=0.0)
    t_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])

    if st.button("🔍 Scan Now", use_container_width=True):
        type_map = {"TRANSFER": 4, "CASH_OUT": 1, "PAYMENT": 3, "DEBIT": 2, "CASH_IN": 0}
        payload = {
            "amount": amount, "oldbalanceOrg": old_bal, "newbalanceOrig": new_bal,
            "type_encoded": type_map[t_type], "nameOrig": name_orig, "nameDest": name_dest 
        }
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload).json()
            st.session_state['last_result'] = res
        except Exception as e:
            st.sidebar.error(f"Connection Error: {e}")

# --- DISPLAY MANUAL RESULTS (SHAP & REASONING) ---
if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    if res.get("is_fraud"):
        st.sidebar.error(f"🚨 ALERT: {res.get('verdict')}")
        st.sidebar.write(f"**Fraud Probability:** {res.get('fraud_probability')}")
        
        if "reasoning" in res and res["reasoning"]:
            st.sidebar.write("---")
            st.sidebar.write("**AI Analysis (SHAP + Heuristics):**")
            for reason in res["reasoning"]:
                st.sidebar.info(f"📍 {reason}")
            
            st.sidebar.write("---")
            if st.sidebar.button("📄 Generate Audit Report"):
                with st.sidebar.spinner("Llama 3 generating compliance report..."):
                    try:
                        from src.compliance_rag.reporter import ComplianceReporter
                        reporter = ComplianceReporter()
                        audit_report = reporter.generate_report(res["reasoning"])
                        st.sidebar.success("Audit Complete")
                        st.sidebar.markdown(audit_report)
                    except:
                        st.sidebar.error("Ollama Service Offline")
    else:
        st.sidebar.success("✅ Transaction Approved")

# --- MAIN DASHBOARD: SIMULATION & ANALYTICS ---
col_sim, col_metrics = st.columns([1, 2])

with col_sim:
    st.subheader("🚀 Simulation Engine")
    st.write("Control the GAN-generated data stream.")
    sim_rate = st.slider("Delay between batches (sec)", 0.1, 5.0, 1.0)
    
    if st.session_state.sim_process_pid is None:
        if st.button("▶️ Start Simulation", use_container_width=True):
            start_simulation(sim_rate)
            st.rerun()
    else:
        if st.button("🛑 Stop Simulation", use_container_width=True):
            stop_simulation()
            st.rerun()
        st.info("🔥 Simulation Active")

with col_metrics:
    st.subheader("📊 Live System Metrics")
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/analytics", timeout=2)
        if response.status_code == 200:
            data = response.json()
            m = data.get("metrics", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Scanned", m.get("total_processed", 0))
            c2.metric("Flagged", m.get("total_flagged", 0))
            c3.metric("Fraud Rate", m.get("fraud_rate", "0%"))
        else:
            st.warning("API Booting...")
    except:
        st.error("API Connection Lost")

st.write("---")
st.subheader("⚠️ High-Risk Activity Log (Real-Time)")
try:
    response = requests.get(f"{BACKEND_URL}/api/v1/analytics", timeout=2)
    if response.status_code == 200:
        threats = response.json().get("recent_threats", [])
        if threats:
            df = pd.DataFrame(threats)
            st.dataframe(df[["id", "sender", "receiver", "amount", "probability", "timestamp"]], use_container_width=True)
        else:
            st.info("Monitoring stream... no threats detected.")
except:
    pass

# --- AUTO-REFRESH ---
# This ensures the Scanned count and Table update while the simulation runs
time.sleep(3)
st.rerun()