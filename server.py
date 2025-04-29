from flask import Flask, Response, send_file, jsonify, request
from flask_cors import CORS
from websocket_server import WebsocketServer
import threading
import cv2
import main
import json
import time
import csv
import os

app = Flask(__name__)
CORS(app)

# Initialize main.py
main.init_camera()
main.init_log()

# WebSocket server
ws_server = WebsocketServer(host='127.0.0.1', port=8090)
def on_client_connect(client, server):
    print(f"WebSocket client connected: {client['address']}")
ws_server.set_fn_new_client(on_client_connect)

def send_status():
    while True:
        if main.trackers:
            active_trackers = {
                str(pid): {
                    'status': data['status'],
                    'idle_time': int(time.time() - data['last_active']) if data['status'] == 'IDLE' else 0,
                    'confidence': float(data['confidence'])
                }
                for pid, data in main.trackers.items()
                if time.time() - data['last_seen'] <= 5
            }
            print(f"Sending WebSocket data: {active_trackers}")
            try:
                ws_server.send_message_to_all(json.dumps(active_trackers))
            except Exception as e:
                print(f"WebSocket send error: {e}")
        else:
            print("No active trackers")
        time.sleep(0.1)

threading.Thread(target=send_status, daemon=True).start()
threading.Thread(target=ws_server.run_forever, daemon=True).start()

# Video streaming
def generate_frames():
    while main.cap.isOpened():
        ret, frame = main.cap.read()
        if not ret:
            break
        try:
            frame, _ = main.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Frame processing error: {e}")
            continue
    main.release_camera()

# Routes
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_log')
def download_log():
    return send_file('worker_log.csv', as_attachment=True)

@app.route('/clear_trackers', methods=['POST'])
def clear_trackers():
    main.clear_trackers()
    return {"message": "Trackers cleared"}, 200

@app.route('/status')
def get_status():
    active_trackers = {
        str(pid): {
            'status': data['status'],
            'idle_time': int(time.time() - data['last_active']) if data['status'] == 'IDLE' else 0,
            'confidence': float(data['confidence'])
        }
        for pid, data in main.trackers.items()
        if time.time() - data['last_seen'] <= 5
    }
    return jsonify(active_trackers)

@app.route('/movement_threshold', methods=['POST'])
def set_movement_threshold():
    data = request.json
    threshold = data.get('threshold')
    if isinstance(threshold, (int, float)) and threshold > 0:
        main.set_movement_threshold(threshold)
        return {"message": f"Movement threshold set to {threshold}"}, 200
    return {"error": "Invalid threshold"}, 400

@app.route('/toggle_debug', methods=['POST'])
def toggle_debug():
    debug_mode = main.toggle_debug()
    return {"message": f"Debug mode {'ON' if debug_mode else 'OFF'}"}, 200

@app.route('/log_data', methods=['GET'])
def get_log_data():
    try:
        with open('worker_log.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Read header row
            print(f"CSV headers: {headers}")
            data = []
            for row in reader:
                # Ensure row has at least 3 columns
                if len(row) < 3:
                    print(f"Skipping invalid row: {row}")
                    continue
                # Use confidence if available, else default to 0.0
                confidence = float(row[3]) if len(row) > 3 and row[3].strip() else 0.0
                data.append({
                    'timestamp': row[0],
                    'person_id': row[1],
                    'status': row[2],
                    'confidence': confidence
                })
            print(f"Sending log data: {data}")
            return jsonify(data)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return jsonify([])
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        main.release_camera()