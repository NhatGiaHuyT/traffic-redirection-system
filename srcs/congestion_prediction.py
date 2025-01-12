import cv2
import numpy as np
from tracking import ObjectTracker
from path_clusterer import PathClusterer

# Initialize the tracker
tracker = ObjectTracker(model_path="models/model_epoch_5.pt")

# Initialize PathClusterer
cluster = PathClusterer(n_clusters=6)
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

# Prepare data for clustering
paths = [np.array(path) for path in tracked_data.values()]
print("Number of paths:", len(paths))

# Get directions from paths
directions = cluster.get_line_directions(paths)

# Align directions for consistency
aligned_directions = cluster.align_directions(directions)

# Perform clustering
print("Clustering paths...")
labels = cluster.fit_and_predict(aligned_directions)

# Visualize clustered paths
print("Visualizing clustered paths...")
cluster.plot_grouped_paths(paths, labels, img_shape=(frame_height, frame_width))

print(f"Processed video saved to {output_video_path}.")
