import requests
import time
import random
from mimesis import Finance, Person

# Configuration
# Note: Use 'api' if running inside Docker, or 'localhost' if running the script directly on WSL
API_URL = "http://api:8000/predict" 
finance = Finance()
person = Person()

def generate_transaction():
    """Generates a mix of normal and suspicious financial behavior."""
    
    # 15% chance to generate a 'Targeted Attack' profile
    is_attack = random.random() < 0.15
    
    # Randomly pick a transaction type (CASH_OUT=1, TRANSFER=4)
    tx_type = random.choice([1, 4])
    
    # Generate realistic balances
    old_balance = float(finance.price(minimum=500, maximum=50000))
    
    if is_attack:
        # ATTACK: High-velocity drain (90% of account balance)
        amount = old_balance * random.uniform(0.90, 0.99)
    else:
        # NORMAL: Casual spending (1% to 20% of account balance)
        amount = old_balance * random.uniform(0.01, 0.20)

    payload = {
        "step": random.randint(1, 744), # Simulation hour
        "type_encoded": tx_type,
        "amount": round(amount, 2),
        "oldbalanceOrg": round(old_balance, 2),
        "newbalanceOrig": round(old_balance - amount, 2)
    }
    
    return payload

def start_stress_test(velocity=2.0):
    """
    velocity: Seconds to wait between transactions.
    """
    print(f"🔥 Starting Sentinel Stress Test Engine...")
    print(f"📡 Target API: {API_URL}")
    print(f"⏱️ Velocity: {velocity}s per transaction\n")

    try:
        while True:
            tx_data = generate_transaction()
            
            try:
                response = requests.post(API_URL, json=tx_data, timeout=5)
                result = response.json()
                
                status_icon = "🚨" if result.get("verdict") == "FLAGGED" else "✅"
                print(f"{status_icon} [Sent] ${tx_data['amount']:>8} | [Result] {result.get('verdict'):<8} | [Prob] {result.get('probability'):.2f}")
                
            except requests.exceptions.RequestException as e:
                print(f"❌ API Connection Error: {e}")
            
            time.sleep(velocity)
            
    except KeyboardInterrupt:
        print("\n🛑 Stress test stopped by user.")

if __name__ == "__main__":
    # You can change the velocity here (e.g., 0.1 for high-speed stress testing)
    start_stress_test(velocity=1.5)