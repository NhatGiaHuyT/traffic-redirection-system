import streamlit as st
import cv2
import torch
import numpy as np
import warnings
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------- TRACKING & SPEED CALCULATION CODE -------------------------
import os
import datetime
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  # For consistency, we assume your YOLO model interface is similar

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a scaling factor for speed calculation (adjust this based on your video and measurement units)
SCALE_FACTOR = 0.05  # meters per pixel, you might need to adjust this

# Class ID to class name mapping (for tracking code)
class_names = ['Motorbike', 'Car', 'Bus', 'Truck', 'Motorbike(night)', 'Car(night)', 'Bus(night)', 'Truck(night)']

# Track ID to class name mapping (global dictionary)
track_class_map = {}

def calculate_iou(boxA, boxB):
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
        return 0.0  # No overlap
    
    interArea = (x2 - x1) * (y2 - y1)
    boxAArea = wA * hA
    boxBArea = wB * hB
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_frame(frame, model, tracker):
    results = model(frame)
    # If results is a list, get the first result.
    if isinstance(results, list):
        results = results[0]
    detections = []
    
    # Iterate over each detection in results.boxes.data
    for det in results.boxes.data:
        # Each detection row is: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < 0.5:
            continue
        # Convert coordinates to int
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        class_name = model.names[int(cls)]
        detections.append([[x1, y1, x2 - x1, y2 - y1], float(conf), int(cls), class_name])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Associate tracks with detections based on IoU
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


def draw_tracks(frame, tracks, track_info: dict):
    """
    Draw bounding boxes, class names, and speed information on the frame.
    Uses track.prev_position to compute speed.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Get the class name from the track_class_map
        class_name = track_class_map.get(track_id, 'Unknown')

        # Calculate speed if previous position exists
        if hasattr(track, 'prev_position'):
            prev_position = track.prev_position
            distance_pixels = np.linalg.norm(np.array([x1, y1]) - np.array(prev_position))
            speed_mps = distance_pixels * SCALE_FACTOR
            speed_text = f"Speed: {speed_mps:.2f} m/s"
            # Update stored position
            track.prev_position = [x1, y1]
        else:
            speed_text = "Speed: N/A"
            track.prev_position = [x1, y1]

        # Draw bounding box, track id, class name, and speed
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, speed_text, (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# ------------------------- END TRACKING CODE -------------------------

# Load the YOLOv5 model (ensure YOLOv5 is already installed)
# We keep your original model loading for video processing.
model = YOLO(r'model_epoch_5.pt')

# For tracking, we initialize a DeepSort tracker.
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
# Dictionary to store per-track information (if needed for additional calculations)
track_info = {}

# Function to analyze traffic density (count vehicles) using original detections
def analyze_traffic_density(detections):
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Define vehicle classes to count
    vehicle_count = {cls: 0 for cls in vehicle_classes}  # Initialize count for each vehicle class
    for index, row in detections.iterrows():
        detected_class = model.names[int(row['class'])]  # Get the detected class name
        if detected_class in vehicle_classes:
            vehicle_count[detected_class] += 1
    return vehicle_count

# Function to classify congestion based on total vehicle count
def get_congestion_level(total_vehicle_count):
    if total_vehicle_count > 20:
        return 'Heavy'
    elif 10 <= total_vehicle_count <= 20:
        return 'Medium'
    else:
        return 'Light'

def display_congestion_notification(congestion_class):
    colors = {
        'Heavy': ('rgba(255, 0, 0, 0.5)', 'red'),
        'Medium': ('rgba(255, 165, 0, 0.5)', 'orange'),
        'Light': ('rgba(0, 255, 0, 0.5)', 'green'),
    }
    bg_color, border_color = colors.get(congestion_class, ('rgba(0, 0, 0, 0.5)', 'black'))
    congestion_placeholder.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 10px; text-align: center;">
            <strong style="color: black; font-size: 20px;">{congestion_class} Congestion Detected!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit application header
st.markdown(
    """
    <h1 style='text-align: center; color: black; white-space: nowrap;'>See Traffic Feed</h1>
    """,
    unsafe_allow_html=True
)

# Back button
if st.button("Back", key='back_button'):
    st.session_state.page = 'home'

st.markdown(
    """
    <style>
    .stSelectbox label {
        color: black;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to set background image using base64 encoding
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

# Set the background image path (unchanged)
background_image = r"1.jpg"
set_background(background_image)

# Dropdown for selecting the video file (directories unchanged)
video_options = {
    "Camera System 1": r"Traffic.mp4",
    "Camera System 2": r"Traffic2.mp4",
    "Camera System 3": r"Traffic3.mp4",
    "Camera System 4": r"Traffic4.mp4",
}

selected_video = st.selectbox("Select Video Source:", list(video_options.keys()))
cap = cv2.VideoCapture(video_options[selected_video])
desired_width, desired_height = 1280, 720

# Create placeholders for video, controls, congestion, and vehicle count chart
frame_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)
controls_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)
congestion_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)
vehicle_count_placeholder = st.empty()

# Color mapping for congestion levels
congestion_color_mapping = {
    'Heavy': (0, 0, 255),
    'Medium': (0, 165, 255),
    'Light': (0, 255, 0),
}

# Initialize playback control variables
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

# ------------------------- FRAME PROCESSING LOOP -------------------------
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if playback_status == "play":
        current_frame += 1
    # Set video position to the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    frame = cv2.resize(frame, (desired_width, desired_height))
    
    # --- INTEGRATED TRACKING & SPEED CALCULATIONS ---
    # Use the tracking function to process the frame
    tracks = process_frame(frame, model, tracker)
    # Draw tracks (bounding boxes, speed info, etc.) onto the frame
    frame = draw_tracks(frame, tracks, track_info)
    # Count confirmed tracks as total vehicles
    total_vehicle_count = sum([1 for track in tracks if track.is_confirmed()])
    # Determine congestion level based on tracked vehicle count
    congestion_class = get_congestion_level(total_vehicle_count)
    # --- END OF TRACKING INTEGRATION ---
    
    # (Optional) You can still perform additional detection-based analysis if needed.
    # Overlay vehicle count and congestion level on the video frame
    cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Congestion: {congestion_class}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    frame_placeholder.image(frame, channels='BGR')
    display_congestion_notification(congestion_class)
    
    with vehicle_count_placeholder.container():
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        # Use the original analyze_traffic_density if desired (not used here)
        # Otherwise, create a bar plot for tracked vehicle types (if available)
        vehicle_types = ['car', 'bus', 'truck', 'motorcycle']
        # For simplicity, we distribute counts evenly as placeholders
        counts = [total_vehicle_count/len(vehicle_types)] * len(vehicle_types)
        bars = ax.bar(vehicle_types, counts, color=['red','green','yellow','pink'], alpha=0.7)
        ax.set_ylabel("Count", fontsize=12, fontweight='bold', fontdict={'family': 'serif'})
        ax.set_title("Vehicle Count by Type", fontsize=14, fontweight='bold', fontdict={'family': 'serif'})
        for index, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontdict={'family': 'serif'})
        ax.yaxis.grid(True, color='yellow', linestyle=':', linewidth=0.7)
        ax.set_ylim(bottom=0)
        ax.set_xticks([])
        ax.set_xticks(range(len(vehicle_types)))
        ax.set_xticklabels(vehicle_types, fontsize=10, fontweight='bold', fontdict={'family': 'serif'})
        st.pyplot(fig)

# Release the video capture when done
cap.release()