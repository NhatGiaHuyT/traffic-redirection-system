import cv2
import numpy as np
from tracking import ObjectTracker
from cluster import KMeansCluster

# Initialize the tracker
tracker = ObjectTracker(model_path="models/model_epoch_5.pt")

# Parameters for clustering
cluster = KMeansCluster(n_clusters=6)
tracked_data = {}

# Input and output video paths
input_video_path = "input_video.mp4"
output_video_path = "output_video.avi"

# Open the video file
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track objects in the current frame
    detections = tracker.process_frame(frame)

    for det in detections:
        x1, y1, x2, y2, obj_id = det

        # Update tracked data
        if obj_id not in tracked_data:
            tracked_data[obj_id] = []
        tracked_data[obj_id].append(((x1 + x2) // 2, (y1 + y2) // 2))

        # Draw bounding boxes and object IDs on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Perform clustering on tracked data
print("Performing clustering...")
cluster.tracked_data_ = tracked_data
cluster.tracked_paths_2_paths_n_lines()

# Use minmax range of paths as features for clustering
features = cluster.minmax_range_paths_
cluster.fit(features)

# Visualize clustered paths
clustered_paths, _ = cluster.fit(features, is_return=True)
cluster.plot_grouped_paths(grouped_paths=clustered_paths)

print(f"Processed video saved to {output_video_path}.")
