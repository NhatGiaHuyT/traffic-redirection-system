import cv2
import torch
import logging
import numpy as np
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from tracking import process_frame, draw_tracks, initialize_video_capture, initialize_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Congestion parameters
DENSITY_THRESHOLD = 10  # Number of vehicles per region for congestion
SPEED_THRESHOLD = 2.0   # Speed in m/s below which the road is considered congested
FRAME_ANALYSIS_WINDOW = 30  # Analyze congestion over the last N frames
SCALE_FACTOR = 0.05  # meters per pixel, you might need to adjust this


# Define input and output paths
input_video_path = r"D:\Traffic Image Object Detection\vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1806644.mov"
output_video_path = r"D:\Traffic Image Object Detection\output_congestion_video.mp4"

# Congestion prediction function
def predict_congestion(frame_count, vehicle_count, avg_speed):
    if vehicle_count > DENSITY_THRESHOLD and avg_speed < SPEED_THRESHOLD:
        logger.info(f"Congestion detected at frame {frame_count}: Vehicles={vehicle_count}, Avg Speed={avg_speed:.2f} m/s")
        return True
    return False

# Function to calculate average speed of all tracked vehicles
def calculate_avg_speed(tracks):
    total_speed = 0.0
    valid_track_count = 0
    
    for track in tracks:
        if hasattr(track, 'prev_position') and track.is_confirmed():
            curr_position = np.array([track.to_ltrb()[0], track.to_ltrb()[1]])
            prev_position = np.array(track.prev_position)
            distance_pixels = np.linalg.norm(curr_position - prev_position)
            speed_mps = distance_pixels * SCALE_FACTOR
            total_speed += speed_mps
            valid_track_count += 1
    
    if valid_track_count == 0:
        return 0.0
    return total_speed / valid_track_count
