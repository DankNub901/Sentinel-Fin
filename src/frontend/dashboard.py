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
        except Exception:
            st.session_state.sim_process_pid = None

# --- UI HEADER ---
st.title("🛡️ Sentinel-Fin: Real-Time Fraud Engine")
st.caption("High-throughput risk scoring powered by XGBoost, Polars, and FastAPI")

# --- SIDEBAR: MANUAL INVESTIGATOR TOOL ---
st.sidebar.header("🏦 Manual Investigator Tool")
with st.sidebar.expander("Single Transaction Scan", expanded=True):
    step = st.number_input("Step (Hour)", min_value=1, value=1, step=1)
    name_orig = st.text_input("Sender Name", value="C_USER_1")
    name_dest = st.text_input("Recipient Name", value="M_MERCH_1")
    amount = st.number_input("Amount ($)", min_value=0.0, value=150000.0)
    old_bal = st.number_input("Old Balance ($)", min_value=0.0, value=200000.0)
    new_bal = st.number_input("New Balance ($)", min_value=0.0, value=50000.0)
    t_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])

    # --- ADVANCED BEHAVIORAL OVERRIDES (OPTIONAL) ---
    with st.sidebar.expander("🛠️ Advanced Behavioral Signals (Optional)"):
        dest_mule_heat = st.number_input("Mule Heat", value=None, placeholder="Default: 1.0")
        sender_recent_velocity = st.number_input("Recent Velocity", value=None, placeholder="Default: 1.0")
        amt_acceleration = st.number_input("Amount Acceleration", value=None, placeholder="Default: 1.0")
        sender_volatility = st.number_input("Sender Volatility", value=None, placeholder="Default: 0.0")
        personal_amt_z_score = st.number_input("Personal Z-Score", value=None, placeholder="Default: 0.0")
        global_step_velocity = st.number_input("Global Step Velocity", value=None, placeholder="Default: 10.0")
        time_since_last_tx = st.number_input("Time Since Last Tx", value=None, placeholder="Default: 0.0")
        account_activity_density = st.number_input("Activity Density", value=None, placeholder="Default: 0.5")
        sender_fan_out = st.number_input("Sender Fan Out", value=None, placeholder="Default: 1")
        is_new_dest_pair = st.selectbox("Is New Dest Pair?", [None, 1, 0])
        is_layering_attempt = st.selectbox("Is Layering Attempt?", [None, 1, 0])

    if st.sidebar.button("🔍 Scan Now", use_container_width=True):
        type_map = {"TRANSFER": 4, "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "DEBIT": 5}
        raw_payload = {
            "step": int(step),
            "type": t_type,
            "amount": float(amount),
            "oldbalanceOrg": float(old_bal),
            "newbalanceOrig": float(new_bal),
            "type_encoded": type_map.get(t_type, 4),
            "nameOrig": name_orig,
            "nameDest": name_dest,
            # Behavioral Overrides
            "dest_mule_heat": dest_mule_heat,
            "sender_recent_velocity": sender_recent_velocity,
            "amt_acceleration": amt_acceleration,
            "sender_volatility": sender_volatility,
            "personal_amt_z_score": personal_amt_z_score,
            "global_step_velocity": global_step_velocity,
            "time_since_last_tx": time_since_last_tx,
            "account_activity_density": account_activity_density,
            "sender_fan_out": sender_fan_out,
            "is_new_dest_pair": is_new_dest_pair,
            "is_layering_attempt": is_layering_attempt,
        }
        # Filter out None values
        payload = {k: v for k, v in raw_payload.items() if v is not None}
        
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload).json()
            st.session_state['last_result'] = res
        except Exception as e:
            st.sidebar.error(f"Connection Error: {e}")

# --- DISPLAY MANUAL SCAN RESULTS ---
if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    if res.get("is_fraud"):
        st.sidebar.error(f"🚨 ALERT: {res.get('verdict')}")
        st.sidebar.write(f"**Fraud Probability:** {res.get('fraud_probability'):.2%}")
        
        if "reasoning" in res and res["reasoning"]:
            st.sidebar.write("---")
            st.sidebar.write("**AI Risk Drivers (SHAP + Heuristics):**")
            for reason in res["reasoning"]:
                st.sidebar.info(f"📍 {reason}")
            
            st.sidebar.write("---")
            if st.sidebar.button("📄 Generate Audit Report", use_container_width=True):
                with st.sidebar.spinner("Llama generating compliance report..."):
                    try:
                        from src.compliance_rag.reporter import ComplianceReporter
                        reporter = ComplianceReporter()
                        audit_report = reporter.generate_report(res["reasoning"])
                        st.sidebar.success("Audit Complete")
                        st.sidebar.markdown(audit_report)
                    except Exception:
                        st.sidebar.error("Ollama / Compliance Service Offline")
    else:
        st.sidebar.success(f"✅ Transaction Approved ({res.get('fraud_probability', 0):.2%})")

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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scanned", m.get("total_processed", 0))
            c2.metric("Flagged", m.get("total_flagged", 0))
            c3.metric("Fraud Rate", m.get("fraud_rate", "0%"))
            c4.metric("Avg Confidence", m.get("avg_confidence", "0%"))
        else:
            st.warning("API Booting...")
    except Exception:
        st.error("API Connection Lost")

st.divider()

# --- OPTION B: TOP THREAT CARD + FULL THREAT STREAM TABLE ---
st.subheader("⚠️ High-Risk Activity Stream")
try:
    response = requests.get(f"{BACKEND_URL}/api/v1/analytics", timeout=2)
    if response.status_code == 200:
        threats = response.json().get("recent_threats", [])
        if threats:
            # Highlight top threat (Option B)
            top_threat = threats[0]
            with st.container(border=True):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.markdown(f"**Top Threat:** Log #{top_threat['id']}")
                tc2.metric("Risk Score", top_threat['probability'])
                tc3.metric("Amount", f"${top_threat['amount']:,.2f}")
                tc4.caption(f"**Timestamp:** {top_threat['timestamp']}")
                st.caption(f"Sender: `{top_threat['sender']}` ➡️ Receiver: `{top_threat['receiver']}`")

            # Table of all threats
            df = pd.DataFrame(threats)
            st.dataframe(
                df[["id", "sender", "receiver", "amount", "probability", "timestamp"]], 
                use_container_width=True
            )
        else:
            st.info("Monitoring stream... no threats detected.")
except Exception:
    pass

# --- AUTO-REFRESH ---
time.sleep(3)
st.rerun()