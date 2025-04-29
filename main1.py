import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
import winsound
from collections import deque

# Setup MediaPipe Face Detection with improved settings
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6, 
                                                 model_selection=1)  # Use model 1 for better distance detection

# Setup MediaPipe Pose with better settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, 
                     min_tracking_confidence=0.7, 
                     model_complexity=1)  # Balanced model for better accuracy
mp_drawing = mp.solutions.drawing_utils

# Add HOG-based people detector as a backup
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

# Improved Constants
IDLE_THRESHOLD = 3  # seconds
MOVEMENT_THRESHOLD = 0.01  # Lower threshold for better sensitivity
MAX_DISTANCE_THRESHOLD = 0.3  # Higher for better tracking continuity
BUFFER_SIZE = 15  # Larger buffer for smoother movement tracking
CLEANUP_INTERVAL = 2.0  # More frequent cleanup
CONFIDENCE_THRESHOLD = 0.65  # For face detection

# Trackers
next_person_id = 1
trackers = {}
last_cleanup_time = time.time()
debug_mode = False

# Initialize log file
def init_log():
    with open("worker_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if os.path.getsize("worker_log.csv") == 0:
            writer.writerow(["Timestamp", "PersonID", "Status", "ConfidenceScore"])

# Enhanced logging with confidence score
def log_status(person_id, status, confidence=1.0):
    with open("worker_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), person_id, status, confidence])

def play_alert():
    # Play two beeps for better notification
    winsound.Beep(1000, 400)
    time.sleep(0.1)
    winsound.Beep(1200, 400)

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_movement(current_keypoints, previous_keypoints):
    """Enhanced movement calculation with weighted importance"""
    if len(current_keypoints) == 0 or len(previous_keypoints) == 0:
        return 0.0
        
    # For face-only detection (5-7 points)
    if len(current_keypoints) <= 7:
        valid_movements = []
        weights = [1.0] * len(current_keypoints)  # Equal weights for face points
        
        for i in range(min(len(current_keypoints), len(previous_keypoints))):
            movement = np.linalg.norm(current_keypoints[i] - previous_keypoints[i])
            valid_movements.append(movement * weights[i])
        
        if valid_movements:
            return np.mean(valid_movements)
        return 0.0
    
    # For full pose detection - weighted by importance
    # Head, shoulders, elbows, wrists have higher weights
    important_indices = [0, 11, 12, 13, 14, 15, 16]
    importance_weights = {
        0: 1.5,    # Head (most important)
        11: 1.2,   # Left shoulder
        12: 1.2,   # Right shoulder
        13: 1.0,   # Left elbow
        14: 1.0,   # Right elbow
        15: 1.8,   # Left wrist (most indicative of working)
        16: 1.8    # Right wrist (most indicative of working)
    }
    
    movements = []
    for i in important_indices:
        if (i < len(current_keypoints) and i < len(previous_keypoints)):
            if current_keypoints[i][0] > 0 and previous_keypoints[i][0] > 0:
                weight = importance_weights.get(i, 1.0)
                movements.append(np.linalg.norm(current_keypoints[i] - previous_keypoints[i]) * weight)
    
    if movements:
        return np.mean(movements)
    
    # Fallback to all valid points
    valid_movements = []
    for i in range(min(len(current_keypoints), len(previous_keypoints))):
        if current_keypoints[i][0] > 0 and previous_keypoints[i][0] > 0:
            valid_movements.append(np.linalg.norm(current_keypoints[i] - previous_keypoints[i]))
    
    if valid_movements:
        return np.mean(valid_movements)
    return 0.0

def detect_multiple_people(frame):
    """Multi-method approach for improved person detection"""
    h, w, _ = frame.shape
    people_keypoints = []
    confidence_scores = []
    
    # Method 1: MediaPipe Face Detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(img_rgb)
    
    face_found = False
    if face_results.detections:
        for detection in face_results.detections:
            # Only use high confidence detections
            if detection.score[0] >= CONFIDENCE_THRESHOLD:
                face_found = True
                
                # Get face bounding box
                bbox = detection.location_data.relative_bounding_box
                x, y = bbox.xmin, bbox.ymin
                width, height = bbox.width, bbox.height
                
                # Scale up bbox slightly to estimate upper body
                body_width = width * 2.0
                body_height = height * 3.0
                body_x = max(0, x - (body_width - width)/2)
                body_y = y
                
                # Create improved keypoints for face and upper body
                face_center_x = x + width/2
                face_center_y = y + height/2
                
                # More precise keypoint estimates
                keypoints = np.array([
                    [face_center_x, face_center_y],                          # Face center
                    [face_center_x - width*0.3, face_center_y],              # Left eye estimate
                    [face_center_x + width*0.3, face_center_y],              # Right eye estimate
                    [face_center_x - width*0.4, face_center_y + height*0.8], # Left shoulder
                    [face_center_x + width*0.4, face_center_y + height*0.8], # Right shoulder
                    [face_center_x - width*0.5, face_center_y + height*1.5], # Left elbow
                    [face_center_x + width*0.5, face_center_y + height*1.5], # Right elbow
                ])
                people_keypoints.append(keypoints)
                confidence_scores.append(float(detection.score[0]))
    
    # Method 2: Try MediaPipe Pose for better body tracking
    if not face_found:
        pose_results = pose.process(img_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            # Extract only the keypoints we need (33 points from MediaPipe Pose)
            keypoints = np.array([[lm.x, lm.y] for lm in landmarks])
            people_keypoints.append(keypoints)
            confidence_scores.append(0.85)  # Assumed confidence for pose detection
    
    # Method 3: HOG People Detector as last resort
    if not people_keypoints:
        # Resize for faster detection
        frame_resized = cv2.resize(frame, (min(400, w), min(400, h*w//400)))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Detect people
        boxes, weights = hog.detectMultiScale(
            gray, 
            winStride=(8, 8),
            padding=(8, 8), 
            scale=1.05,
            finalThreshold=2
        )
        
        # Scale boxes back to original size
        scale_x = w / frame_resized.shape[1]
        scale_y = h / frame_resized.shape[0]
        
        for i, (x, y, w_box, h_box) in enumerate(boxes):
            # Scale box coordinates
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w_box * scale_x)
            h_orig = int(h_box * scale_y)
            
            # Create simple keypoints for HOG detection
            head_y = y_orig + h_orig * 0.2
            center_x = x_orig + w_orig / 2
            
            keypoints = np.array([
                [center_x / w, head_y / h],  # Head estimate
                [(center_x - w_orig*0.25) / w, (y_orig + h_orig*0.35) / h],  # Left shoulder
                [(center_x + w_orig*0.25) / w, (y_orig + h_orig*0.35) / h],  # Right shoulder
                [(center_x - w_orig*0.3) / w, (y_orig + h_orig*0.5) / h],   # Left elbow
                [(center_x + w_orig*0.3) / w, (y_orig + h_orig*0.5) / h],   # Right elbow
            ])
            people_keypoints.append(keypoints)
            confidence_scores.append(min(1.0, weights[i] * 0.1))  # Scale weight to reasonable confidence
    
    # Method 4: OpenCV's Haar Cascade as a final backup
    if not people_keypoints:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        for (x, y, w_face, h_face) in faces:
            face_center_x = (x + w_face/2) / w
            face_center_y = (y + h_face/2) / h
            
            keypoints = np.array([
                [face_center_x, face_center_y],  # Face center
                [face_center_x - 0.05, face_center_y + 0.1],  # Left shoulder
                [face_center_x + 0.05, face_center_y + 0.1],  # Right shoulder
                [face_center_x - 0.08, face_center_y + 0.2],  # Left elbow
                [face_center_x + 0.08, face_center_y + 0.2],  # Right elbow
            ])
            people_keypoints.append(keypoints)
            confidence_scores.append(0.7)  # Default confidence for Haar
    
    return people_keypoints, confidence_scores

def find_matching_person(people_keypoints, confidence_scores):
    """Match each detected person with existing trackers or create new ones"""
    global next_person_id
    assigned_ids = []
    
    # For each person detected in the current frame
    for person_idx, (keypoints, confidence) in enumerate(zip(people_keypoints, confidence_scores)):
        if len(keypoints) == 0:
            continue
            
        centroid = np.mean(keypoints, axis=0)
        best_match = None
        best_distance = float('inf')
        
        # Find the closest existing tracker
        for pid, data in trackers.items():
            if pid in assigned_ids:  # Skip already assigned IDs
                continue
                
            prev_keypoints = data['keypoints']
            if len(prev_keypoints) == 0:
                continue
                
            prev_centroid = np.mean(prev_keypoints, axis=0)
            dist = distance(centroid, prev_centroid)
            
            if dist < MAX_DISTANCE_THRESHOLD and dist < best_distance:
                best_match = pid
                best_distance = dist
        
        # If match found, update the tracker
        if best_match is not None:
            update_tracker(best_match, keypoints, confidence)
            assigned_ids.append(best_match)
        else:
            # Create new tracker
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
    """Update the person's tracker with new keypoints"""
    # Calculate movement between previous and current keypoints
    prev_kps = trackers[person_id]['keypoints']
    movement = calculate_movement(keypoints, prev_kps)
    
    # Add movement to buffer
    trackers[person_id]['movement_buffer'].append(movement)
    
    # Update keypoints, timestamp, and confidence
    trackers[person_id]['keypoints'] = keypoints
    trackers[person_id]['last_seen'] = time.time()
    trackers[person_id]['confidence'] = confidence
    
    # Calculate smoothed movement
    smoothed_movement = np.mean(trackers[person_id]['movement_buffer'])
    
    if debug_mode:
        print(f"Person {person_id} | Movement: {movement:.5f} | Smoothed: {smoothed_movement:.5f} | Conf: {confidence:.2f}")
    
    # Update status based on movement with hysteresis (prevents rapid switching)
    current_status = trackers[person_id]['status']
    
    if smoothed_movement > MOVEMENT_THRESHOLD:
        if current_status != "WORKING":
            trackers[person_id]['status'] = "WORKING"
            log_status(person_id, "WORKING", confidence)
        trackers[person_id]['last_active'] = time.time()
    else:
        # Only change to IDLE if movement has been below threshold for some time
        idle_time = time.time() - trackers[person_id]['last_active']
        if idle_time > IDLE_THRESHOLD and current_status != "IDLE":
            trackers[person_id]['status'] = "IDLE"
            log_status(person_id, "IDLE", confidence)
            play_alert()

def cleanup_trackers():
    """Clean up old trackers"""
    global trackers
    current_time = time.time()
    
    # Remove trackers that haven't been seen recently
    trackers_to_remove = []
    for pid in trackers:
        if current_time - trackers[pid]['last_seen'] > 5:  # 5 seconds timeout
            trackers_to_remove.append(pid)
    
    for pid in trackers_to_remove:
        if debug_mode:
            print(f"Removing tracker {pid} - not seen recently")
        del trackers[pid]

def get_id_color(person_id):
    """Generate a consistent color for each person ID"""
    np.random.seed(person_id * 137)
    color = tuple(map(int, np.random.randint(50, 230, size=3).tolist()))
    return color

def process_frame(frame):
    """Process a frame and update trackers"""
    global last_cleanup_time
    current_time = time.time()
    h, w, _ = frame.shape
    
    # Periodically clean up trackers
    if current_time - last_cleanup_time > CLEANUP_INTERVAL:
        cleanup_trackers()
        last_cleanup_time = current_time
    
    # Detect people in the frame with confidence scores
    people_keypoints, confidence_scores = detect_multiple_people(frame)
    
    # Match detected people with trackers
    active_ids = find_matching_person(people_keypoints, confidence_scores)
    
    # Draw results
    for pid in active_ids:
        data = trackers[pid]
        keypoints = data['keypoints']
        
        # Calculate centroid
        valid_keypoints = [kp for kp in keypoints if kp[0] > 0]
        if valid_keypoints:
            centroid = np.mean(valid_keypoints, axis=0)
            x, y = int(centroid[0] * w), int(centroid[1] * h)
            
            # Get status and color
            status = data['status']
            confidence = data.get('confidence', 0.5)
            id_color = get_id_color(pid)
            status_color = (0, 255, 0) if status == "WORKING" else (0, 0, 255)
            
            # Display status with confidence
            status_text = f"ID {pid}: {status}"
            if status == "IDLE":
                idle_time = int(current_time - data['last_active'])
                status_text += f" ({idle_time}s)"
            
            cv2.putText(frame, status_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add confidence indicator
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw identification circle - size based on confidence
            circle_radius = int(10 * min(1.5, max(0.7, confidence)))
            cv2.circle(frame, (x, y), circle_radius, id_color, -1)
            cv2.circle(frame, (x, y), circle_radius + 2, (255, 255, 255), 1)
            
            # Draw keypoints and connections for better visualization
            for i, kp in enumerate(keypoints):
                kp_x, kp_y = int(kp[0] * w), int(kp[1] * h)
                # Different colors for different keypoint types
                kp_color = id_color
                if i == 0:  # Head
                    kp_color = (255, 200, 0)
                cv2.circle(frame, (kp_x, kp_y), 3, kp_color, -1)
                
                # Draw connections between keypoints
                if i > 0 and i < len(keypoints):
                    if i % 2 == 1 and i+1 < len(keypoints):  # Connect pairs (shoulders, elbows)
                        next_kp_x, next_kp_y = int(keypoints[i+1][0] * w), int(keypoints[i+1][1] * h)
                        cv2.line(frame, (kp_x, kp_y), (next_kp_x, next_kp_y), id_color, 1)
                    if i <= 2:  # Connect head to shoulders
                        head_x, head_y = int(keypoints[0][0] * w), int(keypoints[0][1] * h)
                        cv2.line(frame, (kp_x, kp_y), (head_x, head_y), id_color, 1)
    
    # Show active tracker count and movement threshold
    cv2.putText(frame, f"People tracked: {len(active_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Movement threshold: {MOVEMENT_THRESHOLD:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return frame

# Initialize log file
if not os.path.exists("worker_log.csv"):
    with open("worker_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "PersonID", "Status", "ConfidenceScore"])
else:
    init_log()

print("\n--- Enhanced Worker Monitor with Multi-Person Support ---")
print("Controls:")
print("  q - Quit application")
print("  c - Clear all trackers")
print("  d - Toggle debug mode")
print("  + - Increase movement threshold")
print("  - - Decrease movement threshold")
print("  t - Test alert sound")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    try:
        frame = process_frame(frame)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        continue

    # Show the frame
    cv2.imshow("Worker Monitor", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear all trackers on 'c' press
        trackers = {}
        next_person_id = 1
        print("Cleared all trackers")
    elif key == ord('d'):
        # Toggle debug mode
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    elif key == ord('+') or key == ord('='):
        # Increase movement threshold
        MOVEMENT_THRESHOLD += 0.002
        print(f"Movement threshold increased to: {MOVEMENT_THRESHOLD:.5f}")
    elif key == ord('-') or key == ord('_'):
        # Decrease movement threshold
        MOVEMENT_THRESHOLD = max(0.001, MOVEMENT_THRESHOLD - 0.002)
        print(f"Movement threshold decreased to: {MOVEMENT_THRESHOLD:.5f}")
    elif key == ord('t'):
        # Test alert sound
        play_alert()
        print("Testing alert sound")

cap.release()
cv2.destroyAllWindows()