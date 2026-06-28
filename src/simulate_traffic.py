import requests
import random
import time

API_URL = "http://127.0.0.1:8000/predict"

def generate_normal_request():
    """Requests similar to training data distribution."""
    return {
        "distance_km": round(random.uniform(0.5, 10.0), 1),
        "items_count": random.randint(1, 6),
        "is_peak_hour": random.choice([0, 1]),
        "traffic_level": random.choice([1, 2, 3]),
    }

def generate_drifted_request():
    """Requests outside training distribution — simulates drift."""
    return {
        "distance_km": round(random.uniform(15.0, 30.0), 1),   # much farther
        "items_count": random.randint(8, 15),                    # much larger orders
        "is_peak_hour": 1,                                       # always peak
        "traffic_level": 3,                                      # always high traffic
    }

def main():
    print("=== Simulating QuickFoods API Traffic ===")
    print("Sending 30 normal requests, then 20 drifted requests...\n")

    # Phase 1: Normal traffic
    for i in range(30):
        payload = generate_normal_request()
        resp = requests.post(API_URL, json=payload)
        data = resp.json()
        print(f"[Normal  {i+1:02d}] dist={payload['distance_km']:5.1f} items={payload['items_count']} → {data['delivery_time_min']:.1f} min")
        time.sleep(0.1)

    print("\n--- Drift begins ---\n")

    # Phase 2: Drifted traffic
    for i in range(20):
        payload = generate_drifted_request()
        resp = requests.post(API_URL, json=payload)
        data = resp.json()
        print(f"[Drifted {i+1:02d}] dist={payload['distance_km']:5.1f} items={payload['items_count']} → {data['delivery_time_min']:.1f} min")
        time.sleep(0.1)

    print(f"\nDone. 50 predictions logged to logs/predictions.jsonl")


if __name__ == "__main__":
    main()