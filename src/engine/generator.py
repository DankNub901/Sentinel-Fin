import pandas as pd
import httpx
import asyncio
import os
import uuid
from src.constants import TRANSACTION_TYPES, API_TITLE

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000/predict/batch")
CSV_FILE = "/app/data/simulation/simulation_ready_for_redis.csv"
BATCH_SIZE = 50
VELOCITY = float(os.getenv("VELOCITY", 1.0))

async def send_batch(client, batch_data, batch_num):
    try:
        response = await client.post(API_URL, json=batch_data, timeout=30.0)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch {batch_num} | Flags: {result.get('flags', 0)}")
        else:
            print(f"❌ API Error: {response.status_code} | {response.text}")
    except Exception as e:
        print(f"⚠️ Connection Error: {e}")

async def run_simulation():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    # Create a unique ID for this simulation run
    sim_session_id = f"sim_{uuid.uuid4().hex[:8]}"

    df = pd.read_csv(CSV_FILE)
    df['type_encoded'] = df['type'].map(TRANSACTION_TYPES)
    
    # Adding Metadata so the API/DB can track this correctly
    df['is_simulated'] = True
    df['session_id'] = sim_session_id


    print(f"🚀 Sentinel-Fin: Starting Async Simulation...")

    async with httpx.AsyncClient() as client:
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i : i + BATCH_SIZE]

            records = chunk.where(pd.notnull(chunk), None).to_dict(orient='records')

            batch_data = {"transactions": records}
            
            # Fire and wait (respecting the velocity/delay)
            await send_batch(client, batch_data, i//BATCH_SIZE + 1)
            await asyncio.sleep(VELOCITY)

if __name__ == "__main__":
    asyncio.run(run_simulation())