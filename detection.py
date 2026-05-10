# ============================================================
# Smart Water Leakage Detection Module
# Model: Isolation Forest (scikit-learn)
# ============================================================

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from typing import TypedDict

# Type definition for prediction result
class PredictionResult(TypedDict):
    prediction: str
    alert_triggered: bool
    confidence: float
    anomaly_score: float
    reason: str


# -------------------------------
# Model Config
# -------------------------------
MODEL_PATH = "model.joblib"
model = None


# -------------------------------
# 1. Train Model (only if needed)
# -------------------------------
def train_model():
    """
    Train the Isolation Forest model on normal water flow data (8–15 L/min)
    and save it to disk.
    """
    global model

    np.random.seed(42)

    # Generate normal flow data
    normal_flow = np.random.uniform(8.0, 15.0, 1000)
    X_train = normal_flow.reshape(-1, 1)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train)

    # Save model
    joblib.dump(model, MODEL_PATH)


# -------------------------------
# 2. Load Model (used by backend)
# -------------------------------
def load_model():
    """
    Load model from disk. If not found, train a new one.
    """
    global model

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        train_model()


# -------------------------------
# 3. Prediction Function (MAIN FUNCTION)
# -------------------------------
def predict_leak(flow: float) -> PredictionResult:
    """
    Predict if a water flow value indicates leakage.

    Parameters:
        flow (float): Water flow in L/min

    Returns:
        dict: {
            prediction: str,
            alert_triggered: bool,
            confidence: float,
            anomaly_score: float,
            reason: str
        }
    """
    global model

    if model is None:
        raise Exception("Model not loaded. Call load_model() first.")

    # Prepare input
    reading = np.array([[flow]])

    # Predict
    prediction = model.predict(reading)[0]   # 1 = normal, -1 = anomaly
    anomaly_score = float(model.decision_function(reading)[0])

    is_anomaly = (prediction == -1)

    # -------------------------------
    # Reason (Explainability)
    # -------------------------------
    if flow > 30:
        reason = "Unusually high water flow"
    elif flow < 1:
        reason = "Very low flow (possible slow leak or blockage)"
    elif is_anomaly:
        reason = "Anomalous pattern detected"
    else:
        reason = "Normal usage pattern"

    # -------------------------------
    # Confidence (simple scaling)
    # -------------------------------
    confidence = float(min(1.0, abs(anomaly_score)))

    # -------------------------------
    # Final Output
    # -------------------------------
    return {
        "prediction": "Leak Detected 🚨" if is_anomaly else "Normal ✅",
        "alert_triggered": bool(is_anomaly),
        "confidence": confidence,
        "anomaly_score": anomaly_score,
        "reason": reason
    }


# -------------------------------
# 4. Optional Test (for you only)
# -------------------------------
if __name__ == "__main__":
    load_model()

    test_values = [10.5, 12.0, 30.0, 45.0, 0.5]

    print("=== Test Results ===")
    for value in test_values:
        result = predict_leak(value)
        print(f"{value} → {result['prediction']} ({result['reason']})")
