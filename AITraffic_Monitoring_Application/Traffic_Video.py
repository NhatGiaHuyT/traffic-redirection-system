import streamlit as st
import cv2
import torch
import numpy as np
import warnings
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt  # For interactive charts
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------- TRACKING & SPEED CALCULATION CODE -------------------------
import os
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################################################
#                             USER CONFIGURABLES                               #
################################################################################

# Provide a default YOLO model path
DEFAULT_MODEL_PATH = r"model_epoch_5.pt"

# Default scaling factor for speed calculation (meters per pixel)
SCALE_FACTOR = 0.05

# Class ID to class name mapping for your custom model (if needed)
class_names = [
    "Motorbike", "Car", "Bus", "Truck", 
    "Motorbike(night)", "Car(night)", "Bus(night)", "Truck(night)"
]

# Initialize a dictionary for track -> class name
track_class_map = {}

# Initialize a dictionary to store per-track info if needed (like direction, ROI checks, etc.)
track_info = {}

################################################################################
#                           ROI / POLYGON CONFIGURATION                         #
################################################################################
# For demonstration, let's define a simple rectangular ROI or polygon
# so we can count vehicles only inside this region. You can let the user
# define these coordinates from the Streamlit UI if you like.

# Example polygon in (x, y) format. Adjust to your video’s resolution:
ROI_POLYGON = np.array([
    [200, 200],
    [1100, 200],
    [1100, 600],
    [200, 600]
], dtype=np.int32)

def is_center_in_roi(x1, y1, x2, y2, polygon):
    """
    Check if the bounding box center is within a polygon ROI.
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Use cv2.pointPolygonTest to check if center is inside ROI.
    # pointPolygonTest returns +1 if inside, 0 if on edge, -1 if outside
    result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)
    return result >= 0  # True if inside or on edge

################################################################################
#                         VEHICLE DIRECTION DETECTION                          #
################################################################################
def detect_direction(track, x1, y1):
    """
    Determine direction of travel for a track based on
    the difference in (x, y) from a previously stored position.
    """
    if hasattr(track, 'prev_position_dir'):
        prev_x, prev_y = track.prev_position_dir
        dx = x1 - prev_x
        dy = y1 - prev_y
        
        # A simple classification:
        #   dx > 0 => moving right;  dx < 0 => moving left
        #   dy > 0 => moving down;   dy < 0 => moving up
        direction = ""
        if abs(dx) > abs(dy):
            if dx > 0:
                direction = "Right"
            else:
                direction = "Left"
        else:
            if dy > 0:
                direction = "Down"
            else:
                direction = "Up"
        track.prev_position_dir = [x1, y1]
        return direction
    else:
        track.prev_position_dir = [x1, y1]
        return "Unknown"

################################################################################
#                        DEEPSORT TRACKER & YOLO MODEL INIT                    #
################################################################################
@st.cache_resource  # Cache the model and tracker across reruns
def init_models(model_path):
    """
    Initialize the YOLO model and DeepSort tracker once.
    """
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    return model, tracker

################################################################################
#                               DETECTION & TRACKING                           #
################################################################################
def calculate_iou(boxA, boxB):
    """
    Calculate IOU between two bounding boxes: boxA (ltrb) and boxB (xywh).
    """
    # Convert track bbox (ltrb) to xywh
    boxA = [boxA[0], boxA[1], boxA[2] - boxA[0], boxA[3] - boxA[1]]
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    # Compute the intersection
    x1 = max(xA, xB)
    y1 = max(yA, yB)
    x2 = min(xA + wA, xB + wB)
    y2 = min(yA + hA, yB + hB)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    interArea = (x2 - x1) * (y2 - y1)
    boxAArea = wA * hA
    boxBArea = wB * hB
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_frame(frame, model, tracker, conf_threshold=0.5):
    """
    Run the YOLO model on the frame, parse detections, and update DeepSort tracks.
    """
    results = model(frame)
    if isinstance(results, list):
        results = results[0]
    detections = []
    
    # Each row in results.boxes.data: [x1, y1, x2, y2, confidence, class]
    if results.boxes is not None and results.boxes.data is not None:
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = model.names[int(cls)]
            detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), int(cls), class_name])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Associate track IDs with class names based on best IoU
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_bbox = track.to_ltrb()  # (left, top, right, bottom)
        best_match = None
        best_iou = 0
        for detection in detections:
            detection_bbox = detection[0]
            iou = calculate_iou(track_bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = detection
        if best_match and best_iou > 0.3:
            track_class_map[track.track_id] = best_match[3]
    return tracks

def draw_tracks(frame, tracks, scale_factor, 
                color_by_speed=False,
                direction_detection=False):
    """
    Draw bounding boxes, class names, and speed info on the frame.
    Optionally color boxes by speed, and detect direction of movement.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track_class_map.get(track_id, 'Unknown')
        
        # Speed Calculation
        if hasattr(track, 'prev_position'):
            prev_position = track.prev_position
            distance_pixels = np.linalg.norm(np.array([x1, y1]) - np.array(prev_position))
            speed_mps = distance_pixels * scale_factor
            speed_text = f"Speed: {speed_mps:.2f} m/s"
            track.prev_position = [x1, y1]
        else:
            speed_text = "Speed: N/A"
            track.prev_position = [x1, y1]

        # Direction Detection
        direction_text = ""
        if direction_detection:
            direction_text = detect_direction(track, x1, y1)
        
        # Speed-based color coding
        box_color = (0, 255, 0)  # Default: green
        if color_by_speed and "Speed: N/A" not in speed_text:
            # Parse speed from text
            speed_val = float(speed_text.split(":")[1].replace("m/s", "").strip())
            if speed_val < 5:
                box_color = (0, 255, 0)      # Green
            elif 5 <= speed_val < 10:
                box_color = (0, 165, 255)   # Orange
            else:
                box_color = (0, 0, 255)     # Red
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # ID text
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Class text
        cv2.putText(frame, class_name, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Speed text
        cv2.putText(frame, speed_text, (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Direction text (optional)
        if direction_text:
            cv2.putText(frame, f"Dir: {direction_text}", (x1, y1 - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

################################################################################
#                    VEHICLE COUNTS, ROI FILTER, PER-CLASS AGG                 #
################################################################################
def get_vehicle_counts_in_roi(tracks, polygon):
    """
    Count how many vehicles are in the ROI polygon, and also aggregate by class.
    """
    total_in_roi = 0
    class_counts = defaultdict(int)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        # Check if center is inside ROI
        if is_center_in_roi(x1, y1, x2, y2, polygon):
            total_in_roi += 1
            cls_name = track_class_map.get(track.track_id, 'Unknown')
            class_counts[cls_name] += 1
    return total_in_roi, dict(class_counts)

################################################################################
#                    ADAPTIVE CONGESTION THRESHOLDS & ALERTS                   #
################################################################################
def get_congestion_level(total_vehicle_count, threshold_medium, threshold_heavy):
    """
    Return 'Light', 'Medium', or 'Heavy' based on user-defined thresholds.
    """
    if total_vehicle_count >= threshold_heavy:
        return 'Heavy'
    elif total_vehicle_count >= threshold_medium:
        return 'Medium'
    else:
        return 'Light'

def display_congestion_notification(congestion_class, placeholder):
    colors = {
        'Heavy': ('rgba(255, 0, 0, 0.5)', 'red'),
        'Medium': ('rgba(255, 165, 0, 0.5)', 'orange'),
        'Light': ('rgba(0, 255, 0, 0.5)', 'green'),
    }
    bg_color, border_color = colors.get(congestion_class, ('rgba(0, 0, 0, 0.5)', 'black'))
    placeholder.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 10px; text-align: center;">
            <strong style="color: black; font-size: 20px;">{congestion_class} Congestion Detected!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

def trigger_alert(congestion_class, alert_mode="none"):
    """
    Optionally trigger an alert if 'Heavy' congestion is detected.
    Could be an audio beep or a webhook (Slack, Twilio, etc.).
    """
    if congestion_class == "Heavy":
        if alert_mode == "audio":
            # Play a beep sound or any local audio file
            # (Streamlit doesn't directly support playing local audio,
            #  but you can embed an HTML audio tag or a web-based sound.)
            pass
        elif alert_mode == "webhook":
            # Example: Send a Slack or Twilio message
            # requests.post("https://hooks.slack.com/services/...", json={"text": "Heavy Congestion Detected!"})
            pass
        # Add more alert methods if needed

################################################################################
#                         HISTORICAL DATA LOGGING / CSV                        #
################################################################################
def log_data_to_csv(log_df, csv_file="traffic_log.csv"):
    """
    Append the current frame's data to a CSV for offline analysis.
    """
    # If file doesn't exist, write with header; otherwise append
    import os
    if not os.path.exists(csv_file):
        log_df.to_csv(csv_file, index=False)
    else:
        log_df.to_csv(csv_file, mode='a', header=False, index=False)

################################################################################
#                             STREAMLIT APPLICATION                            #
################################################################################

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: black; white-space: nowrap;'>Enhanced Traffic Feed with Multiple Features</h1>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Configuration & Controls")

# 1. Model path
model_path = st.sidebar.text_input("YOLO Model Path", value=DEFAULT_MODEL_PATH)
model, tracker = init_models(model_path)

# 2. Video or Live Stream
video_options = {
    "Camera System 1 (Local Video)": r"./videos/Traffic.mp4",
    "Camera System 2": r"./videos/Traffic2.mp4",
    "Camera System 3": r"./videos/Traffic3.mp4",
    "Camera System 4": r"./videos/vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1806644.mov",
    "Live Webcam (ID 0)": "webcam0",  # We'll handle "webcam0" as a special case
    # You can also add RTSP links here if you have IP cameras
}

selected_video = st.sidebar.selectbox("Select Video Source:", list(video_options.keys()))

# 3. Adaptive Threshold Sliders
threshold_medium = st.sidebar.slider("Medium Congestion Threshold", min_value=1, max_value=50, value=10)
threshold_heavy = st.sidebar.slider("Heavy Congestion Threshold", min_value=5, max_value=100, value=20)

# 4. Speed-based Color Coding
color_by_speed = st.sidebar.checkbox("Color bounding boxes by speed", value=True)

# 5. Direction Detection
direction_detection = st.sidebar.checkbox("Enable direction detection", value=True)

# 6. Alert Mode
alert_mode = st.sidebar.selectbox("Alert Mode", ["none", "audio", "webhook"], index=0)

# 7. ROI-based Counting
use_roi = st.sidebar.checkbox("Use ROI-based Counting", value=True)

# 8. Show real-time charts
show_charts = st.sidebar.checkbox("Show Real-Time Charts (Time-Series)", value=True)

# 9. Historical Logging
enable_logging = st.sidebar.checkbox("Enable CSV Logging", value=False)

# 10. Playback controls
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("Back", key='back_button'):
    st.session_state.page = 'home'

# Background image
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    encoded_image = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# You can change the background if you like
set_background("1.jpg")

# Video capture
def init_video_capture(video_source):
    if video_source == "webcam0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)
    return cap

cap = init_video_capture(video_options[selected_video])
desired_width, desired_height = 1280, 720

# Placeholders for Streamlit UI
frame_placeholder = st.empty()
controls_placeholder = st.empty()
congestion_placeholder = st.empty()
vehicle_count_placeholder = st.empty()
chart_placeholder = st.empty()
fps_placeholder = st.empty()

# Playback controls
playback_status = st.session_state.get("playback_status", "play")
current_frame = st.session_state.get("current_frame", 0)

with controls_placeholder.container():
    col0, col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1, 2])
    with col1:
        if st.button("⏪", key='backward'):
            current_frame -= 30
            playback_status = "pause"
    with col2:
        if st.button("▶️", key='play'):
            playback_status = "play"
    with col3:
        if st.button("⏸", key='pause'):
            playback_status = "pause"
    with col4:
        if st.button("⏩", key='forward'):
            current_frame += 30
            playback_status = "pause"

st.session_state["playback_status"] = playback_status
st.session_state["current_frame"] = current_frame

# Lists for real-time charting
time_series_data = []
time_series_timestamps = []

prev_time = time.time()

# ------------------------- FRAME PROCESSING LOOP -------------------------
while cap.isOpened():
    start_loop_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    if playback_status == "play":
        current_frame += 1
    
    # If reading from a file, set position
    if selected_video != "Live Webcam (ID 0)":
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    frame = cv2.resize(frame, (desired_width, desired_height))

    # 1. Process tracking
    tracks = process_frame(frame, model, tracker, conf_threshold=0.5)

    # 2. Draw bounding boxes, speeds, etc.
    frame = draw_tracks(frame, tracks, 
                        scale_factor=SCALE_FACTOR,
                        color_by_speed=color_by_speed,
                        direction_detection=direction_detection)

    # 3. ROI-based counting
    if use_roi:
        total_in_roi, class_counts_roi = get_vehicle_counts_in_roi(tracks, ROI_POLYGON)
        # Optionally draw ROI
        cv2.polylines(frame, [ROI_POLYGON], True, (255, 255, 0), 2)
    else:
        # If ROI is disabled, count all confirmed tracks
        total_in_roi = sum([1 for t in tracks if t.is_confirmed()])
        class_counts_roi = {}

    # 4. Congestion level
    congestion_class = get_congestion_level(total_in_roi, threshold_medium, threshold_heavy)

    # 5. Overlay info on the frame
    cv2.rectangle(frame, (0, 0), (320, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Vehicles: {total_in_roi}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Congestion: {congestion_class}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 6. FPS calculation
    end_loop_time = time.time()
    fps = 1.0 / (end_loop_time - start_loop_time + 1e-8)  # to avoid division by zero
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 7. Show frame
    frame_placeholder.image(frame, channels='BGR')

    # 8. Congestion notification + optional alert
    display_congestion_notification(congestion_class, congestion_placeholder)
    trigger_alert(congestion_class, alert_mode)

    # 9. Vehicle count bar chart
    with vehicle_count_placeholder.container():
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # If using ROI-based class counts, let's show them
        if use_roi:
            if not class_counts_roi:
                # If no vehicles, just show empty
                ax.bar(["None"], [0], color='gray')
                ax.set_title("Vehicles in ROI")
            else:
                keys = list(class_counts_roi.keys())
                vals = list(class_counts_roi.values())
                ax.bar(keys, vals, color='orange', alpha=0.7)
                ax.set_title("Vehicle Count in ROI by Class")
        else:
            # If ROI disabled, just show total_in_roi
            ax.bar(["All Vehicles"], [total_in_roi], color='green', alpha=0.7)
            ax.set_title("Total Vehicles (No ROI)")

        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, color='yellow', linestyle=':', linewidth=0.7)
        st.pyplot(fig)

    # 10. Real-time charting (time-series of total vehicles in ROI)
    if show_charts:
        timestamp = datetime.now()
        time_series_timestamps.append(timestamp)
        time_series_data.append(total_in_roi)

        # Build a DataFrame
        chart_df = pd.DataFrame({
            "timestamp": time_series_timestamps,
            "vehicle_count": time_series_data
        })

        # Use Altair to create a line chart
        line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x='timestamp:T',
            y='vehicle_count:Q'
        ).properties(title="Real-Time Vehicle Count in ROI")
        chart_placeholder.altair_chart(line_chart, use_container_width=True)

    # 11. Historical Logging
    if enable_logging:
        # Prepare a row of data
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "video_source": selected_video,
            "vehicles_in_roi": total_in_roi,
            "congestion_level": congestion_class,
            "fps": fps,
        }
        log_df = pd.DataFrame([log_data])
        log_data_to_csv(log_df, csv_file="traffic_log.csv")

    # If playback is paused, break out of the loop
    if playback_status == "pause":
        time.sleep(0.1)  # Sleep briefly so we don't max CPU
        continue

cap.release()