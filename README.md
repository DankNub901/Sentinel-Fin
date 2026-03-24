
# Sentinel-Fin: High-Throughput Fraud Detection & Automated Auditing

**Sentinel-AI** is a production-grade fraud detection pipeline designed for high-frequency financial environments. It integrates Gradient Boosted Trees for real-time inference with an LLM-driven "Regulatory Agent" to automate compliance reporting and explainability.

## Key Innovation: Resource-Aware Engineering
Designed to run on consumer-grade hardware (Intel Iris Xe), the system utilizes **Vectorized Batching** and **Asynchronous I/O** to maintain a throughput of 50 transactions/second while simultaneously running a local Llama 3 model for explainable AI (XAI).



---

## Tech Stack & Rationale

| Component | Technology | Engineering Rationale |
| :--- | :--- | :--- |
| **ML Engine** | XGBoost | Optimized for imbalanced tabular data common in financial fraud. |
| **API Layer** | FastAPI + Uvicorn | Asynchronous handling of batch ingestion to prevent blocking I/O. |
| **Explanation** | SHAP (TreeExplainer) | Deployed selectively on flagged events to optimize CPU cycles. |
| **Audit Agent** | Llama 3 (Ollama) | Localized LLM for privacy-first, on-demand regulatory reporting. |
| **Orchestration** | LangChain | Manages RAG (Context-Injected Inference) across compliance documentation. |
| **Storage** | PostgreSQL | Relational storage for audit logs with indexed temporal lookups. |
| **DevOps** | Docker Compose | Ensures environment parity and isolated resource allocation. |

---

## Architectural Deep Dive

### 1. The Dual-Engine Detection Logic
Sentinel-AI does not rely solely on probabilistic models. It employs a **Hybrid Detection Strategy**:
- **ML Layer:** Captures non-linear, hidden patterns in transaction behavior.
- **Heuristic Guardrails:** Implements a "Safety Layer" for high-risk anomalies (e.g., Rule-101: 90%+ account depletion) that overrides the model to ensure zero-day protection against obvious theft.

### 2. High-Throughput Batch Pipeline
To circumvent the overhead of atomic API calls, the system implements a `predict/batch` endpoint:
- **Vectorization:** Transaction data is converted into a single NumPy/Pandas matrix for simultaneous inference.
- **Bulk Persistence:** Uses SQLAlchemy's `add_all()` to minimize database round-trips, significantly reducing I/O wait times on integrated hardware.

### 3. Explainability & Context-Injected RAG
Most fraud systems are "black boxes." Sentinel-AI provides transparency:
- **Ground Truth Injection:** The SHAP attribution values (the "why") are fed directly into the Llama 3 prompt.
- **Regulatory Alignment:** The model references a local knowledge base (`regulations.txt`) to cite specific Rule IDs (e.g., Rule-202) in the final audit report, mimicking a human compliance officer.

---

## Evolution of Architecture: The "Lean" Refactor

The current iteration of Sentinel-AI is the result of a rigorous optimization phase to maintain high throughput on 16GB RAM systems.

### 1. Service Decommissioning (The "Why")
| Decommissioned Service | Replaced By | Engineering Rationale |
| :--- | :--- | :--- |
| **Apache Kafka** | `asyncio` + Batching | Kafka’s JVM overhead (2GB+ RAM) was too heavy for local dev. Replaced with internal Python async buffers. |
| **Locust / Mimesis** | Pre-computed SDV Stream | Real-time synthetic generation competed with XGBoost for CPU cycles. Moved to "Streamed Replay" of pre-generated data. |
| **Heavy Vector DB** | Context-Injected Prompting | For a targeted compliance rulebook, a Vector DB added unnecessary search latency. Rules are now cached in-memory for O(1) retrieval. |

### 2. Revamped Services
- **Ollama Persistence:** Dedicated container with volume-mapped storage to prevent model re-loads.
- **Postgres Optimization:** Switched from atomic inserts to **Bulk persistence**, reducing DB write-latency by ~85% during 50tx/s bursts.

---

## MLOps: Rigorous Training & Governance

### 1. Temporal Data Splitting (Anti-Cheating)
Unlike standard stratified shuffles, Sentinel-AI implements a **Temporal Split (70/15/15)** based on the transaction `step`.
- **Logic:** The model trains on past data to predict future fraud, simulating real-world bank deployment and preventing "Future-Lookahead" leakage.

### 2. AUPRC-Driven Optimization
In highly imbalanced datasets (Fraud < 0.1%), **Accuracy is a vanity metric.** - **Metric:** The pipeline is tuned for **AUPRC (Area Under Precision-Recall Curve)**.
- **Dynamic Balancing:** The pipeline calculates a **Dynamic `scale_pos_weight`** using the ratio of `negative / positive` classes to penalize missed fraud cases.

### 3. Versioned Model Registry
Model weights are hosted on the **Hugging Face Hub**, allowing the API to pull specific artifacts (`fraud_model.pkl`) during container build.

---

## 📈 Current Roadmap & Performance Tuning

### **Phase 1: Performance Optimization (Completed)**
* [x] Implemented Async simulation streaming 50tx/sec.
* [x] Integrated SHAP for feature-level transparency.
* [x] Containerized the full stack for "one-command" deployment.

### **Phase 2: Behavioral Feature Engineering (In Progress)**
Transitioning from **State-Based** (static balances) to **Behavior-Based** features:
* [ ] **Velocity Metrics:** 1h and 24h transaction frequency per user via Postgres Windows.
* [ ] **Relative Magnitude:** Current `amount` vs. 30-day moving average per account.
* [ ] **Redis Integration:** Caching layer for millisecond-latency behavioral lookups.
