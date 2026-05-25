"""
server_iot.py
─────────────
Flask + Flask-SocketIO server

- Starts the MQTT inference pipeline in a background thread
- Pushes `new_reading` events to all WebSocket clients in real-time
- Serves REST endpoints: /api/latest, /api/history, /api/status
- Serves the web dashboard at GET /
"""

import os
import sys
import threading

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

# ── ensure project root is on path ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mqtt_subscriber import MQTTInferencePipeline, BROKER_HOST, BROKER_PORT, TOPIC_SENSOR

# ── Webapp directory (same folder as this script) ─────────────────────────────
WEBAPP_DIR = os.path.join(BASE_DIR, "webapp")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=WEBAPP_DIR, static_url_path="")
app.config["SECRET_KEY"] = "smart-env-monitor-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",   # works without eventlet/gevent
)

# ── Global pipeline instance ───────────────────────────────────────────────────
pipeline: MQTTInferencePipeline | None = None


def _new_reading_handler(data: dict):
    """Called by mqtt_subscriber on every new inference result → push to browser."""
    socketio.emit("new_reading", data)
    
    # Emit specific prediction events for real-time updates
    if "insights" in data:
        insights = data["insights"]
        if "som" in insights:
            socketio.emit("som_prediction", {
                "timestamp": data.get("timestamp"),
                "som": insights["som"],
                "sensor": data.get("sensor", {})
            })
        if "rbf" in insights:
            socketio.emit("rbf_prediction", {
                "timestamp": data.get("timestamp"),
                "rbf": insights["rbf"],
                "sensor": data.get("sensor", {})
            })
        # Emit all predictions together
        socketio.emit("all_predictions", {
            "timestamp": data.get("timestamp"),
            "predictions": insights,
            "action": data.get("action", {}),
            "sensor": data.get("sensor", {})
        })


def start_pipeline():
    global pipeline
    pipeline = MQTTInferencePipeline(
        data_dir=os.path.join(BASE_DIR, "data"),
        model_dir=os.path.join(BASE_DIR, "models"),
        buffer_size=100,
    )
    pipeline.on_new_reading = _new_reading_handler
    pipeline.start()
    print("[✓] MQTT pipeline running in background thread")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "mqtt_connected": pipeline.connected if pipeline else False,
        "message_count":  pipeline.message_count if pipeline else 0,
        "broker":         f"{BROKER_HOST}:{BROKER_PORT}",
        "topic_data":     TOPIC_SENSOR,
    })


@app.route("/api/latest")
def api_latest():
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    reading = pipeline.get_latest()
    if reading is None:
        return jsonify({"error": "no data yet — is the ESP32 running?"}), 404
    return jsonify(reading)


@app.route("/api/history")
def api_history():
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    n       = int(request.args.get("n", 50))
    history = pipeline.get_history(n)
    return jsonify(history)


@app.route("/api/predictions/som")
def api_som_predictions():
    """Returns SOM (Self-Organizing Map) predictions for the latest reading."""
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    reading = pipeline.get_latest()
    if reading is None or "insights" not in reading:
        return jsonify({"error": "no data yet — is the ESP32 running?"}), 404
    som_data = reading["insights"].get("som", {})
    return jsonify({
        "timestamp": reading.get("timestamp"),
        "som": som_data,
        "sensor_data": reading.get("sensor", {})
    })


@app.route("/api/predictions/rbf")
def api_rbf_predictions():
    """Returns RBF (Radial Basis Function) predictions for the latest reading."""
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    reading = pipeline.get_latest()
    if reading is None or "insights" not in reading:
        return jsonify({"error": "no data yet — is the ESP32 running?"}), 404
    rbf_data = reading["insights"].get("rbf", {})
    return jsonify({
        "timestamp": reading.get("timestamp"),
        "rbf": rbf_data,
        "sensor_data": reading.get("sensor", {})
    })


@app.route("/api/predictions/all")
def api_all_predictions():
    """Returns all AI predictions (RBF, SOM, ART, Fuzzy) for the latest reading."""
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    reading = pipeline.get_latest()
    if reading is None:
        return jsonify({"error": "no data yet — is the ESP32 running?"}), 404
    return jsonify({
        "timestamp": reading.get("timestamp"),
        "predictions": reading.get("insights", {}),
        "sensor_data": reading.get("sensor", {}),
        "action": reading.get("action", {})
    })


@app.route("/api/history/som")
def api_som_history():
    """Returns historical SOM predictions."""
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    n = int(request.args.get("n", 50))
    history = pipeline.get_history(n)
    som_history = [
        {
            "timestamp": reading.get("timestamp"),
            "som": reading.get("insights", {}).get("som", {})
        }
        for reading in history if "insights" in reading
    ]
    return jsonify(som_history)


@app.route("/api/history/rbf")
def api_rbf_history():
    """Returns historical RBF predictions."""
    if pipeline is None:
        return jsonify({"error": "pipeline not started"}), 503
    n = int(request.args.get("n", 50))
    history = pipeline.get_history(n)
    rbf_history = [
        {
            "timestamp": reading.get("timestamp"),
            "rbf": reading.get("insights", {}).get("rbf", {})
        }
        for reading in history if "insights" in reading
    ]
    return jsonify(rbf_history)


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_client_connect(auth=None):
    print("[WS] Client connected")
    # Send the last reading immediately so the UI isn't blank on load
    if pipeline:
        reading = pipeline.get_latest()
        if reading:
            from flask_socketio import emit as ws_emit
            ws_emit("new_reading", reading)


@socketio.on("disconnect")
def on_client_disconnect():
    print("[WS] Client disconnected")


@socketio.on("request_predictions")
def on_request_predictions(request_type=None):
    """Client can request specific prediction types: 'som', 'rbf', or 'all'."""
    if pipeline is None:
        return {"error": "pipeline not started"}
    
    reading = pipeline.get_latest()
    if reading is None or "insights" not in reading:
        return {"error": "no predictions yet"}
    
    insights = reading["insights"]
    
    if request_type == "som":
        return {
            "timestamp": reading.get("timestamp"),
            "som": insights.get("som", {}),
            "sensor": reading.get("sensor", {})
        }
    elif request_type == "rbf":
        return {
            "timestamp": reading.get("timestamp"),
            "rbf": insights.get("rbf", {}),
            "sensor": reading.get("sensor", {})
        }
    else:  # "all" or default
        return {
            "timestamp": reading.get("timestamp"),
            "predictions": insights,
            "action": reading.get("action", {}),
            "sensor": reading.get("sensor", {})
        }



# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("  Smart Environment Monitor — Web Server")
    print("═" * 70)
    print(f"  Dashboard          →  http://localhost:5000")
    print(f"\n  REST API:")
    print(f"    Latest Data    →  http://localhost:5000/api/latest")
    print(f"    SOM Prediction →  http://localhost:5000/api/predictions/som")
    print(f"    RBF Prediction →  http://localhost:5000/api/predictions/rbf")
    print(f"    All Predictions →  http://localhost:5000/api/predictions/all")
    print(f"    History        →  http://localhost:5000/api/history?n=50")
    print(f"    SOM History    →  http://localhost:5000/api/history/som?n=50")
    print(f"    RBF History    →  http://localhost:5000/api/history/rbf?n=50")
    print(f"    Status         →  http://localhost:5000/api/status")
    print(f"\n  WebSocket Events:")
    print(f"    new_reading       - Full sensor + action + predictions")
    print(f"    som_prediction    - SOM cluster + state predictions")
    print(f"    rbf_prediction    - System state + temperature trend + gas danger")
    print(f"    all_predictions   - All AI model predictions together")
    print(f"\n  MQTT →  {BROKER_HOST}:{BROKER_PORT}")
    print(f"  Topic →  {TOPIC_SENSOR}")
    print("═" * 70 + "\n")

    # Start MQTT pipeline in a daemon thread before serving
    t = threading.Thread(target=start_pipeline, daemon=True)
    t.start()

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
