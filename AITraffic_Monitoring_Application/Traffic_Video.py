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

# Load the YOLOv5 model (ensure YOLOv5 is already installed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# Function to analyze traffic density (count vehicles)
def analyze_traffic_density(detections):
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Define vehicle classes to count
    vehicle_count = {cls: 0 for cls in vehicle_classes}  # Initialize count for each vehicle class
    
    # Loop through detections and count vehicles
    for index, row in detections.iterrows():
        detected_class = model.names[int(row['class'])]  # Get the detected class name
        if detected_class in vehicle_classes:
            vehicle_count[detected_class] += 1  # Count each specific vehicle type
    
    return vehicle_count

# Function to classify congestion based on total vehicle count
def get_congestion_level(total_vehicle_count):
    if total_vehicle_count > 20:  # Example threshold for heavy congestion
        return 'Heavy'
    elif 10 <= total_vehicle_count <= 20:  # Example threshold for medium congestion
        return 'Medium'
    else:
        return 'Light'

def display_congestion_notification(congestion_class):
    # Define the colors based on congestion level
    colors = {
        'Heavy': ('rgba(255, 0, 0, 0.5)', 'red'),     # Semi-transparent red
        'Medium': ('rgba(255, 165, 0, 0.5)', 'orange'),  # Semi-transparent orange
        'Light': ('rgba(0, 255, 0, 0.5)', 'green'),     # Semi-transparent green
    }

    # Get the background and border color based on the congestion class
    bg_color, border_color = colors.get(congestion_class, ('rgba(0, 0, 0, 0.5)', 'black'))  # Default to black

    congestion_placeholder.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 10px; text-align: center;">
            <strong style="color: black; font-size: 20px;">{congestion_class} Congestion Detected!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit application
st.markdown(
    """
    <h1 style='text-align: center; color: black; white-space: nowrap;'>See Traffic Feed</h1>
    """,
    unsafe_allow_html=True
)

# Back button in the top right corner
if st.button("Back", key='back_button'):
    st.session_state.page = 'home'  # Change session state to go back to home

st.markdown(
    """
    <style>
    .stSelectbox label {
        color: black;      /* Change the color of the selectbox label */
        font-weight: bold; /* Make the text bold */
        font-size: 18px;   /* Increase the font size */
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

# Set the background image path
background_image = r"1.jpg" # Update with your image path
set_background(background_image)

# Dropdown for selecting the video file
video_options = {
    "Camera System 1": r"vecteezy_ho-chi-minh-city-traffic-at-intersection-vietnam_1806644.mov",
    "Camera System 2": r"Traffic2.mp4",
    "Camera System 3": r"Traffic3.mp4",
    "Camera System 4": r"Traffic4.mp4",
}

selected_video = st.selectbox("Select Video Source:", list(video_options.keys()))

# Initialize video capture for the selected video
cap = cv2.VideoCapture(video_options[selected_video])

# Set the desired window size for displaying video
desired_width, desired_height = 1280, 720  # Adjusted for larger video

# Create placeholders for the video, vehicle count chart, and congestion notification
frame_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)  # Add a small gap between components
controls_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)  # Add a small gap between components
congestion_placeholder = st.empty()  # Container for congestion label
st.markdown("<br>", unsafe_allow_html=True)  # Add a small gap between components
vehicle_count_placeholder = st.empty()

# Define color mapping for congestion levels
congestion_color_mapping = {
    'Heavy': (0, 0, 255),     # Red for heavy congestion
    'Medium': (0, 165, 255),  # Orange for medium congestion
    'Light': (0, 255, 0),     # Green for light congestion
}

# Initialize playback control variables
playback_status = st.session_state.get("playback_status", "play")
current_frame = st.session_state.get("current_frame", 0)

# Define buttons for play, pause, forward, and backward using Unicode symbols and center them
with controls_placeholder.container():
    # Create empty spaces on either side of the buttons to center them
    col0, col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1, 2])

    with col1:
        if st.button("⏪", key='backward'):
            current_frame -= 30  # Move back 30 frames (1 second for 30 FPS)
            playback_status = "pause"
    with col2:
        if st.button("▶️", key='play'):
            playback_status = "play"
    with col3:
        if st.button("⏸", key='pause'):
            playback_status = "pause"
    with col4:
        if st.button("⏩", key='forward'):
            current_frame += 30  # Move forward 30 frames
            playback_status = "pause"

# Save the updated playback state
st.session_state["playback_status"] = playback_status
st.session_state["current_frame"] = current_frame

# Frame processing loop
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Increment or adjust frame index
    if playback_status == "play":
        current_frame += 1
    elif playback_status == "pause":
        pass  # Freeze at current frame

    # Set video position to the current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Resize frame to desired resolution
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Perform detection using YOLOv5
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get bounding box results

    # Analyze traffic density (count vehicles)
    vehicle_count = analyze_traffic_density(detections)
    total_vehicle_count = sum(vehicle_count.values())  # Total count for congestion level

    # Automatically determine congestion level based on total vehicle count
    congestion_class = get_congestion_level(total_vehicle_count)

    # Loop through each detection and apply highlight to the vehicle area based on congestion level
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  # Bounding box coordinates
        detected_class = model.names[int(row['class'])]  # Get the detected class name

        # Only apply highlight if the detected object is a vehicle
        if detected_class in congestion_color_mapping:
            # Get the overlay color based on the congestion level
            overlay_color = congestion_color_mapping[congestion_class]  # Red for Heavy, Orange for Medium, Green for Light

            # Extract the region of interest (ROI) for the detected vehicle
            roi = frame[y1:y2, x1:x2]
            
            # Create an overlay for the specific vehicle area
            overlay = np.full(roi.shape, overlay_color, dtype=np.uint8)  # Create an overlay for the vehicle area
            alpha = 0.3  # Transparency factor
            
            # Blend the overlay with the original vehicle area
            highlighted_roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
            
            # Update the frame with the highlighted vehicle area
            frame[y1:y2, x1:x2] = highlighted_roi

    # Add bounding boxes and labels to the frame
    results.render()

    # Overlay vehicle count and congestion level on the video frame
    cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1)  # Semi-transparent black background
    cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Congestion: {congestion_class}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the updated frame (video) along with the bounding boxes and highlights
    frame_placeholder.image(frame, channels='BGR')

    # Update congestion notification based on congestion level
    display_congestion_notification(congestion_class)

    with vehicle_count_placeholder.container():
        # Create a bar plot using Matplotlib
        fig, ax = plt.subplots()

        # Set transparency and style
        fig.patch.set_alpha(0)  # Make the background of the figure transparent
        ax.patch.set_alpha(0)  # Make the axes background transparent as well

        # Plot the bar chart
        bars = ax.bar(vehicle_count.keys(), vehicle_count.values(), color=['red','green','yellow','pink'], alpha=0.7)

        # Set labels, font styles, and bold font
        # ax.set_xlabel("Vehicle Type", fontsize=12, fontweight='bold', fontdict={'family': 'serif'})  # Removed x-axis label
        ax.set_ylabel("Count", fontsize=12, fontweight='bold', fontdict={'family': 'serif'})
        ax.set_title("Vehicle Count by Type", fontsize=14, fontweight='bold', fontdict={'family': 'serif'})

        # Display vehicle count above each bar
        for index, bar in enumerate(bars):
            height = bar.get_height()
            # Display vehicle count above the bar
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontdict={'family': 'serif'})

        # Add dotted yellow grid lines
        ax.yaxis.grid(True, color='yellow', linestyle=':', linewidth=0.7)

        # Set the y-axis limits to show only positive counts
        ax.set_ylim(bottom=0)  # Start from 0 for a cleaner look

        # Adjust the y-ticks to show relevant values
        ax.set_yticks(range(0, max(vehicle_count.values()) + 2))  # Set y-ticks appropriately

        # Remove existing x-ticks to avoid duplicates
        ax.set_xticks([])  # Clear existing x-ticks

        # Set the x-axis labels directly
        vehicle_types = ['car', 'bus', 'truck', 'motorcycle']
        ax.set_xticks(range(len(vehicle_types)))  # Set x-ticks based on the number of vehicle types
        ax.set_xticklabels(vehicle_types, fontsize=10, fontweight='bold', fontdict={'family': 'serif'})  # Set the x-tick labels

        # Display the plot in the Streamlit app
        st.pyplot(fig)



    
# Release the video capture when done
cap.release()
