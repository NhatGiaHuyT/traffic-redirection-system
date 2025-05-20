# traffic-redirection-system

# How to run

## 1. Install dependencies

- Create a virtual environment to isolate project dependencies and avoid conflicts with other projects.

```bash
python -m venv venv
```

- Activate the virtual environment.
  - On Windows:

```bash
./venv/Scripts/activate
```

    + On Linux or macOS:

```bash
source venv/bin/activate
```

- Install the required packages using pip with the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## 2. Run the program

- To run the program, execute the following command in your terminal:

```bash
(venv) > streamlit run .\src\HomePage.py
```

# CHANGELOG

## 1. Phu 1/9/2025

To test repository permissions, I made some modifications to the `tracking.py` file. I added an optional dictionary parameter, `unnamed`, to the draw function to store tracking object data, including the object's `class`, `first_position`, and `last_position` that the model had detected. Additionally, I created a `cluster` file to hold the unfinished `KMeansCluster` class.

## 2. Huy 1/9/2025

# cluster.py

This module provides a `KMeansCluster` class for clustering and visualizing tracking data using the KMeans algorithm:

- Load and process tracking data.
- Cluster paths and lines based on feature vectors.
- Visualize grouped paths using Matplotlib.
- Save and load KMeans model states.

# congestion_prediction.py

This module provides tools for predicting and analyzing traffic congestion using real-time object detection, tracking, and speed estimation:

- Detect and track vehicles in video frames using YOLO and DeepSORT.
- Calculate average vehicle speed and density.
- Predict and log congestion based on customizable thresholds.
- Save processed video with visualized tracking and congestion indicators.

# tracking.py

This module is responsible for vehicle detection, tracking, and class identification in video frames. It integrates YOLO for object detection and DeepSORT for tracking, providing essential functionalities for traffic monitoring and analysis:

- Detect vehicles and classify them into predefined categories.
- Track vehicle movements across frames using DeepSORT.
- Compute Intersection over Union (IoU) for detection and track association.
- Annotate video frames with bounding boxes, class names, and estimated speeds.
