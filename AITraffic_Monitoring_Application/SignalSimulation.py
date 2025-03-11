import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import base64

# Custom CSS for enhanced visuals
st.markdown(
    """
    <style>
        /* Add background image */
        .stApp {
            background-image: url(https://images.unsplash.com/photo-1520594923568-1b5d82587f86?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fHJvYWRzfGVufDB8fDB8fHww);  /* Replace with your own image URL */
            background-size: cover;
            background-position: center;
        }

        /* Center the header */
        .title h1 {
            font-size: 3em;
            color: #fff;
            text-align: center;
            margin-bottom: 0.5em;
        }

        /* Style the traffic cards */
        .traffic-card {
            background-color: #f0f0f0;
            border-radius: 30px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }

    """, unsafe_allow_html=True
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


# Header with enhanced styling
# Header with enhanced styling
st.markdown(
    """
    <div class='title'>
        <h1>🚦 Traffic Signal Optimisation Simulation 🚦</h1>
    </div>
    """, unsafe_allow_html=True
)

# Load and display the local image below the title
image_path = r"SS.png"  # Replace with your image path
st.image(image_path, use_column_width=True)

# Define your directions (Road A, B, C, D)
directions = ["Road A", "Road B", "Road C", "Road D"]

def display_traffic_chart(traffic_volumes, directions):
    """ Displays a bar chart of traffic volumes for the given roads with enhanced visuals. """
    # Color-code the chart based on traffic volume severity
    colors = ['green' if volume < 10 else 'orange' if volume < 20 else 'red' for volume in traffic_volumes]

    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    plt.bar(directions, traffic_volumes, color=colors)
    
    # Set title and labels with bold font
    plt.title('Traffic Volumes on Roads', fontsize=18, color='#333', fontweight='bold')
    plt.ylabel('Traffic Volume', fontsize=14, fontweight='bold')
    plt.xlabel('Roads', fontsize=14, fontweight='bold')
    
    # Keep the x-axis labels horizontal
    plt.xticks(rotation=0, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(plt)


def adjust_for_weather(traffic_volumes, weather_condition):
    """ Adjusts traffic volume based on weather conditions. """
    if weather_condition == "rainy":
        return [volume * 2 for volume in traffic_volumes]  # Double the traffic for rain
    elif weather_condition == "foggy":
        return [volume * 1.5 for volume in traffic_volumes]  # 1.5x the traffic for fog
    else:
        return traffic_volumes

def calculate_timings(traffic_volumes, scenario=None, closed_road=None, congested_road=None, priority_road=None):
    """ Calculates timings based on traffic volume and given scenario. """
    base_green_time = 10  # Base green light time in seconds
    base_yellow_time = 3   # Base yellow light duration
    base_red_time = 5      # Base red light duration

    timings = []

    for i, volume in enumerate(traffic_volumes):
        green_time = base_green_time + (volume // 10)  # Increase green time based on traffic volume

        # Adjust yellow light duration based on weather and traffic
        yellow_time = base_yellow_time + (volume // 15)  # More volume means longer yellow
        if weather_condition == "rainy":
            yellow_time += 2  # Increase yellow light during rain

        # Adjust red light duration based on traffic volume
        red_time = base_red_time + (volume // 20)  # More volume means longer red
        if road_condition == "Emergency Response":
            red_time *= 0.5  # Shorten red time for emergency response
        elif weather_condition == "foggy":
            red_time += 2  # Increase red time in foggy weather

        # Apply scenarios
        if scenario == "Road Closure" and directions[i] == closed_road:
            continue  # Skip closed road

        if scenario == "Traffic Incident" and congested_road == directions[i]:
            green_time *= 1.5  # More time for congested road

        if scenario == "Construction Zone" and congested_road == directions[i]:
            green_time *= 1.2  # More time for construction congestion

        if scenario == "Emergency Response" and priority_road == directions[i]:
            green_time *= 1.7  # Prioritize the emergency road
        else:
            green_time *= 0.8  # Reduce green time for other roads during emergency

        # You can add unique adjustments based on combinations here
        if scenario == "Traffic Incident" and weather_condition == "rainy":
            yellow_time += 1  # Increase yellow during rainy incidents
        
        if scenario == "Construction Zone" and weather_condition == "foggy":
            green_time *= 0.9  # Reduce green time during construction in fog

        timings.append((green_time, yellow_time, red_time))

    return timings


def display_result_cards(traffic_volumes, timings, directions, closed_road=None):
    """Displays traffic data in a card layout, sorted by green light duration."""
    # Combine the data and sort by green light duration
    sorted_data = sorted(zip(directions, traffic_volumes, timings), key=lambda x: x[2][0], reverse=True)

    for direction, volume, timing in sorted_data:
        # Skip the closed road
        if direction == closed_road:
            continue
        
        st.markdown(f"""
            <div class="traffic-card" style="background: linear-gradient(to right, rgba(255, 255, 255, 0.95), rgba(240, 240, 240, 0.9)); 
                                                  border-radius: 15px; 
                                                  padding: 20px; 
                                                  box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.2); 
                                                  margin: 10px 0; 
                                                  transition: transform 0.2s, box-shadow 0.2s; 
                                                  border: 1px solid #ccc;">
                <h2 style="color: #333; font-weight: bold; font-size: 2em; margin-bottom: 10px; text-align: center;">{direction}</h2>
                <p style="color: black; font-size: 1.1em; text-align: center;">Traffic Volume: <strong>{volume}</strong></p>
                <p style="color: darkgreen; font-size: 1.1em; text-align: center;">Green Light: <strong>{timing[0]} seconds</strong></p>
                <p style="color: orange; font-size: 1.1em; text-align: center;">Yellow Light: <strong>{timing[1]} seconds</strong></p>
                <p style="color: red; font-size: 1.1em; text-align: center;">Red Light: <strong>{timing[2]} seconds</strong></p>
            </div>
        """, unsafe_allow_html=True)



# Define your directions (Road A, B, C, D)
directions = ["Road A", "Road B", "Road C", "Road D"]

# Streamlit UI
st.sidebar.header("User Inputs")
weather_condition = st.sidebar.selectbox("Select Weather Condition", ["clear", "rainy", "foggy"])
road_condition = st.sidebar.selectbox("Select Road Condition", ["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"])

traffic_volumes = []
closed_road = None  # Initialize closed_road variable

if road_condition == "Normal":
    # Sidebar input for traffic volumes
    for direction in directions:
        traffic_volume = st.sidebar.number_input(f"Enter traffic volume for {direction}:", min_value=0, value=10)
        traffic_volumes.append(traffic_volume)

elif road_condition == "Road Closure":
    # Ask for the affected road when road closure is selected
    closed_road = st.sidebar.selectbox("Select the affected road", directions)
    
    # Set traffic volume for the closed road to 0
    traffic_volumes = [0 if direction == closed_road else st.sidebar.number_input(f"Enter traffic volume for {direction}:", min_value=0, value=10) for direction in directions]

else:
    # Ask for the affected road when any condition except "Normal" is selected
    affected_road = st.sidebar.selectbox("Select the affected road", directions)
    
    # Sidebar input for traffic volumes
    for direction in directions:
        if direction == affected_road and road_condition == "Road Closure":
            traffic_volume = 0  # Traffic volume is 0 for closed road
        else:
            traffic_volume = st.sidebar.number_input(f"Enter traffic volume for {direction}:", min_value=0, value=10)
        
        # Apply specific logic based on road condition
        if road_condition == "Traffic Incident" and direction == affected_road:
            traffic_volume *= 1.5  # Traffic incident causes congestion
        elif road_condition == "Construction Zone" and direction == affected_road:
            traffic_volume *= 1.3  # Construction causes less congestion than incidents
        
        traffic_volumes.append(traffic_volume)

# Adjust traffic volumes based on weather condition
traffic_volumes = adjust_for_weather(traffic_volumes, weather_condition)

# Display traffic volumes chart
display_traffic_chart(traffic_volumes, directions)

# Calculate timings based on the scenario
timings = calculate_timings(traffic_volumes)

# Display results, passing closed_road to the function
display_result_cards(traffic_volumes, timings, directions, closed_road=closed_road)

