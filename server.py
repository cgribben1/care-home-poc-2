"""Care-home event server — receives telemetry + alerts from ESP32, serves dashboard."""

from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "server_ui"))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/telemetry", methods=["POST"])
def receive_telemetry():
    data = request.get_json(force=True) or {}
    data["server_ts"] = time.time()
    socketio.emit("telemetry", data)
    return jsonify({"ok": True})


@app.route("/event", methods=["POST"])
@app.route("/api/alert", methods=["POST"])
def receive_event():
    data = request.get_json(force=True) or {}
    data["server_ts"] = time.time()
    label = data.get("event_type") or data.get("class", "?")
    print(f"[ALERT] {label}  conf={data.get('confidence', 0):.2f}  device={data.get('device_id','?')}")
    socketio.emit("alert", data)
    return jsonify({"ok": True})


@app.route("/heartbeat", methods=["POST"])
@app.route("/api/heartbeat", methods=["POST"])
def receive_heartbeat():
    data = request.get_json(force=True) or {}
    data["server_ts"] = time.time()
    print(f"[HB] device={data.get('device_id','?')}  rssi={data.get('rssi_dbm','?')} dBm")
    socketio.emit("heartbeat", data)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Care-home server starting on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
