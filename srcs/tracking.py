import os
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define input video path and output video path
input_video_path = r"D:\Traffic Image Object Detection\vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1806644.mov"
output_video_path = r"D:\Traffic Image Object Detection\output_video.mp4"  # Adjust output path as needed
confidence_threshold = 0.5

# Define a scaling factor for speed calculation (adjust this based on your video and measurement units)
SCALE_FACTOR = 0.05  # meters per pixel, you might need to adjust this

def initialize_video_capture(video_input):
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")
    return cap

def initialize_model():
    model = YOLO(r'D:\Traffic Image Object Detection\model\model_epoch_5.pt')  # Adjust to your YOLO model path

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

import numpy as np
from collections import defaultdict

# Class ID to class name mapping
class_names = ['Motorbike', 'Car', 'Bus', 'Truck', 'Motorbike(night)', 'Car(night)', 'Bus(night)', 'Truck(night)']

# Track ID to class name mapping
track_class_map = {}

def process_frame(frame, model, tracker):
    results = model(frame, verbose=False)[0]
    detections = []
    
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        
        if confidence < confidence_threshold:
            continue
        
        class_name = class_names[int(label)]  # Get class name from label index
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, int(label), class_name])  # Append bbox and class name

    tracks = tracker.update_tracks(detections, frame=frame)

    # Match tracks with detections based on bounding box overlap
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_bbox = track.to_ltrb()  # Get the track bounding box (left, top, right, bottom)
        
        # Find the detection that best matches this track based on IOU (Intersection over Union)
        best_match = None
        best_iou = 0
        for detection in detections:
            detection_bbox = detection[0]  # Get detection bounding box
            iou = calculate_iou(track_bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = detection

        # If a match is found, associate the track with the corresponding class
        if best_match and best_iou > 0.3:  # Adjust IOU threshold as needed
            track_class_map[track.track_id] = best_match[3]  # Store class name for track ID
    
    return tracks

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

def draw_tracks(frame, tracks, unnamed: dict = None):
    """
    Draw bounding boxes, class names, and speed information on the frame.

    Args:
        frame (np.array): Input frame
        tracks (list): List of Track objects
        unnamed (dict): Dictionary to store track information
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Get the class name from the track_class_map
        class_name = track_class_map.get(track_id, 'Unknown')

        # Calculate speed (assuming the previous position is saved)
        if hasattr(track, 'prev_position'):
            prev_position = track.prev_position
            distance_pixels = np.linalg.norm(np.array([x1, y1]) - np.array(prev_position))
            speed_mps = distance_pixels * SCALE_FACTOR
            speed_text = f"Speed: {speed_mps:.2f} m/s"
            if unnamed is not None:
                unnamed[track_id]["last_position"] = [x1, y1]
        else:
            speed_text = "Speed: N/A"
            unnamed[track_id] = {"class": class_name, "first_position": [x1, y1]}

        
        # Draw bounding box, class name, and speed
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, speed_text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Store the current position for the next frame
        track.prev_position = [x1, y1]

    return frame
