# app.py
# ---------------------------------------------------------------
# Flask API — AI Smart Water Leakage Detection
# Compatible with ML teammate's model.py (Isolation Forest)
# Endpoint: POST /predict
# ---------------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS          # pip install flask-cors
from model import load_model, predict_leak

app = Flask(__name__)
CORS(app)   # allows the frontend (index.html) to call this API

# ── Load ML model once when server starts ───────────────────────
load_model()
print("✅ Model loaded and ready.")


# ── POST /predict ────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "flow": <number> }

    Returns JSON:
    {
        "flow":            <number>,
        "status":          "Leak Detected 🚨" | "Normal ✅",
        "alert_triggered": true | false,
        "confidence":      <float>,
        "anomaly_score":   <float>,
        "reason":          <string>
    }
    """
    data = request.get_json()

    # 1. Must be valid JSON body
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    # 2. 'flow' key must exist
    if "flow" not in data:
        return jsonify({"error": "Missing required field: 'flow'."}), 400

    flow_value = data["flow"]

    # 3. 'flow' must be a number
    if not isinstance(flow_value, (int, float)):
        return jsonify({"error": "'flow' must be a numeric value."}), 400

    # 4. Run ML prediction
    result = predict_leak(float(flow_value))

    # 5. Return full result + echo the flow value
    return jsonify({
        "flow":            flow_value,
        "status":          result["prediction"],       # "Leak Detected 🚨" or "Normal ✅"
        "alert_triggered": result["alert_triggered"],
        "confidence":      round(result["confidence"], 4),
        "anomaly_score":   round(result["anomaly_score"], 4),
        "reason":          result["reason"]
    }), 200


# ── GET /health ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running ✅"}), 200


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
