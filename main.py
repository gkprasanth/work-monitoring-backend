import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque
import logging

# Suppress TensorFlow Lite warnings
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.INFO)

# Setup MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6, model_selection=1)

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Setup HOG-based people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Global state
cap = None
trackers = {}
next_person_id = 1
last_cleanup_time = time.time()
debug_mode = False

# Constants
IDLE_THRESHOLD = 3
MOVEMENT_THRESHOLD = 0.01
MAX_DISTANCE_THRESHOLD = 0.3
BUFFER_SIZE = 15
CLEANUP_INTERVAL = 2.0
CONFIDENCE_THRESHOLD = 0.65

def init_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

def release_camera():
    global cap
    if cap:
        cap.release()

def init_log():
    with open("worker_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if os.path.getsize("worker_log.csv") == 0:
            writer.writerow(["Timestamp", "PersonID", "Status", "ConfidenceScore"])

def log_status(person_id, status, confidence=1.0):
    with open("worker_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), person_id, status, confidence])

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_movement(current_keypoints, previous_keypoints):
    if len(current_keypoints) == 0 or len(previous_keypoints) == 0:
        return 0.0
    if len(current_keypoints) <= 7:
        valid_movements = []
        weights = [1.0] * len(current_keypoints)
        for i in range(min(len(current_keypoints), len(previous_keypoints))):
            movement = np.linalg.norm(current_keypoints[i] - previous_keypoints[i])
            valid_movements.append(movement * weights[i])
        return np.mean(valid_movements) if valid_movements else 0.0
    important_indices = [0, 11, 12, 13, 14, 15, 16]
    importance_weights = {0: 1.5, 11: 1.2, 12: 1.2, 13: 1.0, 14: 1.0, 15: 1.8, 16: 1.8}
    movements = []
    for i in important_indices:
        if i < len(current_keypoints) and i < len(previous_keypoints):
            if current_keypoints[i][0] > 0 and previous_keypoints[i][0] > 0:
                weight = importance_weights.get(i, 1.0)
                movements.append(np.linalg.norm(current_keypoints[i] - previous_keypoints[i]) * weight)
    if movements:
        return np.mean(movements)
    valid_movements = []
    for i in range(min(len(current_keypoints), len(previous_keypoints))):
        if current_keypoints[i][0] > 0 and previous_keypoints[i][0] > 0:
            valid_movements.append(np.linalg.norm(current_keypoints[i] - previous_keypoints[i]))
    return np.mean(valid_movements) if valid_movements else 0.0

def detect_multiple_people(frame):
    h, w, _ = frame.shape
    people_keypoints = []
    confidence_scores = []
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(img_rgb)
    face_found = False
    if face_results.detections:
        for detection in face_results.detections:
            if detection.score[0] >= CONFIDENCE_THRESHOLD:
                face_found = True
                bbox = detection.location_data.relative_bounding_box
                x, y = bbox.xmin, bbox.ymin
                width, height = bbox.width, bbox.height
                body_width = width * 2.0
                body_height = height * 3.0
                body_x = max(0, x - (body_width - width)/2)
                body_y = y
                face_center_x = x + width/2
                face_center_y = y + height/2
                keypoints = np.array([
                    [face_center_x, face_center_y],
                    [face_center_x - width*0.3, face_center_y],
                    [face_center_x + width*0.3, face_center_y],
                    [face_center_x - width*0.4, face_center_y + height*0.8],
                    [face_center_x + width*0.4, face_center_y + height*0.8],
                    [face_center_x - width*0.5, face_center_y + height*1.5],
                    [face_center_x + width*0.5, face_center_y + height*1.5],
                ])
                people_keypoints.append(keypoints)
                confidence_scores.append(float(detection.score[0]))
    if not face_found:
        pose_results = pose.process(img_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y] for lm in landmarks])
            people_keypoints.append(keypoints)
            confidence_scores.append(0.85)
    if not people_keypoints:
        frame_resized = cv2.resize(frame, (min(400, w), min(400, h*w//400)))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05, finalThreshold=2)
        scale_x = w / frame_resized.shape[1]
        scale_y = h / frame_resized.shape[0]
        for i, (x, y, w_box, h_box) in enumerate(boxes):
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w_box * scale_x)
            h_orig = int(h_box * scale_y)
            head_y = y_orig + h_orig * 0.2
            center_x = x_orig + w_orig / 2
            keypoints = np.array([
                [center_x / w, head_y / h],
                [(center_x - w_orig*0.25) / w, (y_orig + h_orig*0.35) / h],
                [(center_x + w_orig*0.25) / w, (y_orig + h_orig*0.35) / h],
                [(center_x - w_orig*0.3) / w, (y_orig + h_orig*0.5) / h],
                [(center_x + w_orig*0.3) / w, (y_orig + h_orig*0.5) / h],
            ])
            people_keypoints.append(keypoints)
            confidence_scores.append(min(1.0, weights[i] * 0.1))
    if not people_keypoints:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        for (x, y, w_face, h_face) in faces:
            face_center_x = (x + w_face/2) / w
            face_center_y = (y + h_face/2) / h
            keypoints = np.array([
                [face_center_x, face_center_y],
                [face_center_x - 0.05, face_center_y + 0.1],
                [face_center_x + 0.05, face_center_y + 0.1],
                [face_center_x - 0.08, face_center_y + 0.2],
                [face_center_x + 0.08, face_center_y + 0.2],
            ])
            people_keypoints.append(keypoints)
            confidence_scores.append(0.7)
    return people_keypoints, confidence_scores

def find_matching_person(people_keypoints, confidence_scores):
    global next_person_id
    assigned_ids = []
    for person_idx, (keypoints, confidence) in enumerate(zip(people_keypoints, confidence_scores)):
        if len(keypoints) == 0:
            continue
        centroid = np.mean(keypoints, axis=0)
        best_match = None
        best_distance = float('inf')
        for pid, data in trackers.items():
            if pid in assigned_ids:
                continue
            prev_keypoints = data['keypoints']
            if len(prev_keypoints) == 0:
                continue
            prev_centroid = np.mean(prev_keypoints, axis=0)
            dist = distance(centroid, prev_centroid)
            if dist < MAX_DISTANCE_THRESHOLD and dist < best_distance:
                best_match = pid
                best_distance = dist
        if best_match is not None:
            update_tracker(best_match, keypoints, confidence)
            assigned_ids.append(best_match)
        else:
            new_id = next_person_id
            trackers[new_id] = {
                'keypoints': keypoints,
                'last_active': time.time(),
                'last_seen': time.time(),
                'status': 'WORKING',
                'movement_buffer': deque([0.0] * 3, maxlen=BUFFER_SIZE),
                'confidence': confidence
            }
            log_status(new_id, "WORKING", confidence)
            assigned_ids.append(new_id)
            next_person_id += 1
    return assigned_ids

def update_tracker(person_id, keypoints, confidence):
    prev_kps = trackers[person_id]['keypoints']
    movement = calculate_movement(keypoints, prev_kps)
    trackers[person_id]['movement_buffer'].append(movement)
    trackers[person_id]['keypoints'] = keypoints
    trackers[person_id]['last_seen'] = time.time()
    trackers[person_id]['confidence'] = confidence
    smoothed_movement = np.mean(trackers[person_id]['movement_buffer'])
    if debug_mode:
        print(f"Person {person_id} | Movement: {movement:.5f} | Smoothed: {smoothed_movement:.5f} | Conf: {confidence:.2f}")
    current_status = trackers[person_id]['status']
    if smoothed_movement > MOVEMENT_THRESHOLD:
        if current_status != "WORKING":
            trackers[person_id]['status'] = "WORKING"
            log_status(person_id, "WORKING", confidence)
        trackers[person_id]['last_active'] = time.time()
    else:
        idle_time = time.time() - trackers[person_id]['last_active']
        if idle_time > IDLE_THRESHOLD and current_status != "IDLE":
            trackers[person_id]['status'] = "IDLE"
            log_status(person_id, "IDLE", confidence)
            play_alert()

def cleanup_trackers():
    global trackers
    current_time = time.time()
    trackers_to_remove = []
    for pid in trackers:
        if current_time - trackers[pid]['last_seen'] > 5:
            trackers_to_remove.append(pid)
    for pid in trackers_to_remove:
        if debug_mode:
            print(f"Removing tracker {pid} - not seen recently")
        del trackers[pid]

def get_id_color(person_id):
    np.random.seed(person_id * 137)
    color = tuple(map(int, np.random.randint(50, 230, size=3).tolist()))
    return color

def process_frame(frame):
    global last_cleanup_time
    current_time = time.time()
    h, w, _ = frame.shape
    if current_time - last_cleanup_time > CLEANUP_INTERVAL:
        cleanup_trackers()
        last_cleanup_time = current_time
    people_keypoints, confidence_scores = detect_multiple_people(frame)
    active_ids = find_matching_person(people_keypoints, confidence_scores)
    active_trackers = {}
    for pid in active_ids:
        data = trackers[pid]
        active_trackers[pid] = {
            'status': data['status'],
            'idle_time': int(current_time - data['last_active']) if data['status'] == 'IDLE' else 0,
            'confidence': data['confidence']
        }
        keypoints = data['keypoints']
        valid_keypoints = [kp for kp in keypoints if kp[0] > 0]
        if valid_keypoints:
            centroid = np.mean(valid_keypoints, axis=0)
            x, y = int(centroid[0] * w), int(centroid[1] * h)
            status = data['status']
            confidence = data.get('confidence', 0.5)
            id_color = get_id_color(pid)
            status_color = (0, 255, 0) if status == "WORKING" else (0, 0, 255)
            status_text = f"ID {pid}: {status}"
            if status == "IDLE":
                idle_time = int(current_time - data['last_active'])
                status_text += f" ({idle_time}s)"
            cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            circle_radius = int(10 * min(1.5, max(0.7, confidence)))
            cv2.circle(frame, (x, y), circle_radius, id_color, -1)
            cv2.circle(frame, (x, y), circle_radius + 2, (255, 255, 255), 1)
            for i, kp in enumerate(keypoints):
                kp_x, kp_y = int(kp[0] * w), int(kp[1] * h)
                kp_color = id_color
                if i == 0:
                    kp_color = (255, 200, 0)
                cv2.circle(frame, (kp_x, kp_y), 3, kp_color, -1)
                if i > 0 and i < len(keypoints):
                    if i % 2 == 1 and i+1 < len(keypoints):
                        next_kp_x, next_kp_y = int(keypoints[i+1][0] * w), int(keypoints[i+1][1] * h)
                        cv2.line(frame, (kp_x, kp_y), (next_kp_x, next_kp_y), id_color, 1)
                    if i <= 2:
                        head_x, head_y = int(keypoints[0][0] * w), int(keypoints[0][1] * h)
                        cv2.line(frame, (kp_x, kp_y), (head_x, head_y), id_color, 1)
    cv2.putText(frame, f"People tracked: {len(active_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Movement threshold: {MOVEMENT_THRESHOLD:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return frame, active_trackers

def clear_trackers():
    global trackers, next_person_id
    trackers = {}
    next_person_id = 1

def set_movement_threshold(value):
    global MOVEMENT_THRESHOLD
    MOVEMENT_THRESHOLD = max(0.001, value)

def toggle_debug():
    global debug_mode
    debug_mode = not debug_mode
    return debug_mode