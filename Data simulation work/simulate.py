import requests   # to send HTTP requests to the backend
import random     # to generate random flow values
import time       # to add delay between readings

# The backend API URL (your teammate's endpoint)
API_URL = "http://127.0.0.1:5000/predict"

def generate_flow():
    """Returns a flow value — normal most of the time, spike 10% of the time."""
    
    is_leak = random.random() < 0.10  # 10% chance of simulating a leak
    
    if is_leak:
        flow = random.uniform(40, 80)   # abnormal spike = leak
    else:
        flow = random.uniform(5, 15)    # normal water usage
    
    return round(flow, 2)  # round to 2 decimal places

def send_to_backend(flow_value):
    """Sends the flow reading to the backend API and prints the response."""
    
    payload = {"flow": flow_value}  # format the backend expects
    
    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        result = response.json()
        print(f"Flow: {flow_value} L/min  →  Status: {result['status']}")
    
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend not running. Retrying in 3s...")
    
    except Exception as e:
        print(f"Error: {e}")

def run_simulation():
    """Main loop — generates and sends data every 2-3 seconds."""
    
    print("Simulation started. Sending data to backend...")
    print("-" * 45)
    
    while True:
        flow = generate_flow()
        send_to_backend(flow)
        time.sleep(random.uniform(2, 3))  # wait 2–3 seconds before next reading

# Entry point
if __name__ == "__main__":
    run_simulation()