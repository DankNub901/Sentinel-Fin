import pandas as pd
import httpx
import asyncio
import os

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000/predict/batch")
CSV_FILE = "/app/data/simulation/simulation_ready_for_redis.csv"
BATCH_SIZE = 50
VELOCITY = float(os.getenv("VELOCITY", 1.0))

async def send_batch(client, batch_data, batch_num):
    try:
        response = await client.post(API_URL, json=batch_data, timeout=10.0)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch {batch_num} | Flags: {result['flags_detected']}")
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Connection Error: {e}")

async def run_simulation():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE)
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    df['type_encoded'] = df['type'].map(type_map)

    print(f"🚀 Sentinel-Fin: Starting Async Simulation...")

    async with httpx.AsyncClient() as client:
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i : i + BATCH_SIZE]
            batch_data = {"transactions": chunk.to_dict(orient='records')}
            
            # Fire and wait (respecting the velocity/delay)
            await send_batch(client, batch_data, i//BATCH_SIZE + 1)
            await asyncio.sleep(VELOCITY)

if __name__ == "__main__":
    asyncio.run(run_simulation())