import cv2
import numpy as np
from sklearn.cluster import KMeans
from tracking import initialize_model, initialize_tracker, track_objects

# Paths
input_video_path = r"D:\Traffic Image Object Detection\vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1806644.mov"
output_video_path = r"D:\Traffic Image Object Detection\output_video.mp4"
yolo_model_path = r"D:\Traffic Image Object Detection\model\model_epoch_5.pt"  # YOLO model path

# Parameters
lane_cluster_count = 3  # Adjust based on the number of lanes
confidence_threshold = 0.5
SCALE_FACTOR = 0.05  # meters per pixel (adjust to your use case)

# Initialize YOLO and DeepSort
model = initialize_model(yolo_model_path)
tracker = initialize_tracker()

# Process frame for lane clustering
def cluster_lanes(tracked_objects, frame_width):
    centroids = []
    for obj in tracked_objects:
        bbox = obj['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2  # Get centroid x-coordinate
        centroids.append([cx])

    # Perform clustering
    kmeans = KMeans(n_clusters=lane_cluster_count, random_state=42)
    cluster_labels = kmeans.fit_predict(centroids)

    # Assign each object to a lane
    lane_assignments = {}
    for idx, obj in enumerate(tracked_objects):
        lane_assignments[obj['id']] = cluster_labels[idx]

    return lane_assignments, kmeans.cluster_centers_

# Detect congestion
def detect_congestion(lane_assignments):
    lane_counts = {}
    for lane in lane_assignments.values():
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

    congested_lanes = [lane for lane, count in lane_counts.items() if count > 5]  # Threshold for congestion
    return congested_lanes

# Process video
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track objects
        tracked_objects = track_objects(frame, model, tracker, confidence_threshold)

        # Cluster lanes
        lane_assignments, lane_centers = cluster_lanes(tracked_objects, frame_width)

        # Detect congestion
        congested_lanes = detect_congestion(lane_assignments)

        # Annotate frame
        for obj in tracked_objects:
            track_id = obj['id']
            bbox = obj['bbox']
            lane = lane_assignments[track_id]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} Lane: {lane}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for idx, center in enumerate(lane_centers):
            cv2.line(frame, (int(center[0]), 0), (int(center[0]), frame_height), (255, 0, 0), 2)
            cv2.putText(frame, f"Lane {idx}", (int(center[0]), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        for lane in congested_lanes:
            cv2.putText(frame, f"Congested Lane: {lane}", (10, 50 + lane * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Video processing complete. Output saved to:", output_path)

if __name__ == "__main__":
    process_video(input_video_path, output_video_path)
