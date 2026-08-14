"""Care-home event server — receives telemetry + alerts from ESP32, serves dashboard."""

from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "server_ui"))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

CALLMEBOT_PHONE  = os.environ.get("CALLMEBOT_PHONE", "")
CALLMEBOT_APIKEY = os.environ.get("CALLMEBOT_APIKEY", "")


def whatsapp_alert(message: str) -> None:
    if not CALLMEBOT_PHONE or not CALLMEBOT_APIKEY:
        return
    try:
        params = urllib.parse.urlencode({
            "phone":  CALLMEBOT_PHONE,
            "text":   message,
            "apikey": CALLMEBOT_APIKEY,
        })
        urllib.request.urlopen(
            f"https://api.callmebot.com/whatsapp.php?{params}", timeout=5
        )
    except Exception as e:
        print(f"[WhatsApp] send failed: {e}")


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
    conf  = data.get("confidence", 0)
    device = data.get("device_id", "?")
    print(f"[ALERT] {label}  conf={conf:.2f}  device={device}")
    socketio.emit("alert", data)
    whatsapp_alert(f"🚨 Care-home alert: {label} detected (confidence {conf:.0%}) — device {device}")
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
    if CALLMEBOT_PHONE:
        print(f"WhatsApp alerts → {CALLMEBOT_PHONE}")
    else:
        print("WhatsApp alerts disabled — set CALLMEBOT_PHONE and CALLMEBOT_APIKEY")
    print("Care-home server starting on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
