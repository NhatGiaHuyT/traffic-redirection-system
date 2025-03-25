import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import random

# =============================================================================
# GLOBAL CONSTANTS & SETTINGS
# =============================================================================

MAX_DISTANCE = 100.0          # Maximum distance vehicles travel along a road
ROAD_LENGTH = 100.0           # Length of each road (same as MAX_DISTANCE)
REFRESH_INTERVAL = 1.0        # Refresh interval (seconds) per simulation step
BASE_SPEED = 20.0             # Base vehicle speed (distance units per second)
VEHICLE_LENGTH = 4.0          # Vehicle length (for drawing rectangles)
VEHICLE_WIDTH = 2.0           # Vehicle width (for drawing rectangles)
SIM_DURATION = 300            # Maximum simulation duration in seconds
TIME_SCALE = 1.0              # Simulation time scaling factor

# =============================================================================
# CUSTOM CSS FOR STYLING & VISUAL APPEAL
# =============================================================================
CUSTOM_CSS = """
<style>
    /* Overall application background */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1520594923568-1b5d82587f86?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3");
        background-size: cover;
        background-position: center;
    }
    /* Header styling */
    .title h1 {
        font-size: 3.5em;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.2em;
        text-shadow: 3px 3px 5px #000;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        font-size: 16px;
    }
    /* Card styling for signal timings and traffic volumes */
    .traffic-card {
        background-color: #f8f8f8;
        border-radius: 15px;
        padding: 15px;
        margin: 8px 0;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .traffic-card:hover {
        transform: scale(1.03);
        box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
    }
    .active-road {
        border: 2px solid #33aa33;
        background-color: #e0ffe0 !important;
    }
    /* Traffic light styles */
    .traffic-light {
        border-radius: 50%;
        width: 20px;
        height: 20px;
        margin: 2px;
        display: inline-block;
        border: 2px solid #333;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0  # Simulation clock in seconds
if "current_road_index" not in st.session_state:
    st.session_state.current_road_index = 0
if "remaining_green_time" not in st.session_state:
    st.session_state.remaining_green_time = 0.0
if "vehicle_positions" not in st.session_state:
    # For each road, keep a list of positions (a float representing distance traveled)
    st.session_state.vehicle_positions = {
        "Road A": [],  # Vehicles moving right from left edge
        "Road B": [],  # Vehicles moving up from bottom
        "Road C": [],  # Vehicles moving left from right edge
        "Road D": []   # Vehicles moving down from top
    }
if "traffic_volumes" not in st.session_state:
    # Traffic volume in vehicles per minute for each road (initial values)
    st.session_state.traffic_volumes = [30, 30, 30, 30]
if "timings" not in st.session_state:
    # Signal timings (green, yellow, red) for each road (in seconds)
    st.session_state.timings = [(15, 4, 6)] * 4
if "weather_condition" not in st.session_state:
    st.session_state.weather_condition = "clear"
if "road_condition" not in st.session_state:
    st.session_state.road_condition = "Normal"

# =============================================================================
# APPLICATION HEADER & SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown("## Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "home"  # For potential multi-page navigation

st.markdown("<div class='title'><h1>🚦 Traffic Signal Optimisation Simulation 🚦</h1></div>", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS: SIMULATION LOGIC, DRAWING, AND RENDERING
# =============================================================================

def display_header_image():
    """
    Display a header image at the top.
    """
    image_url = "https://images.unsplash.com/photo-1516707570268-975c0a813ce0?fm=jpg&q=60&w=1000"
    st.image(image_url, use_column_width=True)

def adjust_for_weather(volumes, condition):
    """
    Adjust the given traffic volumes based on the weather condition.
    """
    adjusted = volumes.copy()
    if condition == "rainy":
        adjusted = [v * 1.8 for v in adjusted]
    elif condition == "foggy":
        adjusted = [v * 1.5 for v in adjusted]
    return adjusted

def calculate_timings(volumes, scenario, weather, closed=None, incident=None, priority=None):
    """
    Calculate realistic signal timings (green, yellow, red) for each road based on
    traffic volumes, weather, and scenario.
    """
    base_green = 15
    base_yellow = 4
    base_red = 6
    timings = []
    for idx, volume in enumerate(volumes):
        green = base_green + int(volume / 30)
        yellow = base_yellow + int(volume / 50)
        red = base_red + int(volume / 60)
        if weather == "rainy":
            yellow += 2
        if weather == "foggy":
            red += 2
        if scenario == "Emergency Response":
            if idx == priority:
                green = int(green * 1.5)
            else:
                green = int(green * 0.8)
                red = int(red * 0.7)
        elif scenario == "Road Closure" and idx == closed:
            green, yellow, red = 0, 0, 0
        elif scenario in ["Traffic Incident", "Construction Zone"] and idx == incident:
            factor = 1.5 if scenario == "Traffic Incident" else 1.2
            green = int(green * factor)
        timings.append((green, yellow, red))
    return timings

def draw_vehicle(ax, x, y, orientation, color):
    """
    Draw a vehicle on the given matplotlib axis as an oriented rectangle.
    """
    if orientation == "horizontal":
        vehicle = patches.Rectangle((x, y - VEHICLE_WIDTH/2), VEHICLE_LENGTH, VEHICLE_WIDTH,
                                     linewidth=1, edgecolor='black', facecolor=color)
    elif orientation == "vertical":
        vehicle = patches.Rectangle((x - VEHICLE_WIDTH/2, y - VEHICLE_LENGTH), VEHICLE_WIDTH, VEHICLE_LENGTH,
                                     linewidth=1, edgecolor='black', facecolor=color)
    else:
        vehicle = patches.Circle((x, y), VEHICLE_WIDTH/2, color=color)
    ax.add_patch(vehicle)

def spawn_vehicles(directions, volumes):
    """
    Spawn new vehicles at the start of each road based on traffic volumes.
    Arrival probability is derived from vehicles per minute.
    """
    for i, road in enumerate(directions):
        arrival_rate = volumes[i] / 60.0  
        if random.random() < (arrival_rate * REFRESH_INTERVAL):
            st.session_state.vehicle_positions[road].append(0.0)

def move_vehicles(directions, delta_time):
    """
    Move vehicles on the active road based on elapsed time.
    """
    distance_step = BASE_SPEED * delta_time
    active_road = directions[st.session_state.current_road_index]
    for road in directions:
        new_positions = []
        for pos in st.session_state.vehicle_positions[road]:
            if road == active_road:
                pos += distance_step
            if pos <= MAX_DISTANCE:
                new_positions.append(pos)
        st.session_state.vehicle_positions[road] = new_positions

def run_simulation_step(delta_time):
    """
    Execute one simulation step:
      - Update remaining green time.
      - Spawn and move vehicles.
      - Rotate active road if green time expires.
      - Update simulation clock.
    """
    if not st.session_state.timings:
        return
    curr_index = st.session_state.current_road_index
    if st.session_state.remaining_green_time <= 0:
        st.session_state.remaining_green_time = st.session_state.timings[curr_index][0]
    st.session_state.remaining_green_time -= delta_time

    leave_rate = 1.0  # vehicles per second departing
    st.session_state.traffic_volumes[curr_index] = max(
        st.session_state.traffic_volumes[curr_index] - (leave_rate * delta_time), 0
    )

    spawn_vehicles(directions, st.session_state.traffic_volumes)
    move_vehicles(directions, delta_time)

    if st.session_state.remaining_green_time <= 0:
        st.session_state.current_road_index = (curr_index + 1) % len(directions)

    st.session_state.sim_time += delta_time

def display_traffic_chart(volumes, directions):
    """
    Display a bar chart showing current traffic volumes.
    """
    colors = ['green' if v < 20 else 'orange' if v < 40 else 'red' for v in volumes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(directions, volumes, color=colors)
    ax.set_title('Current Traffic Volumes', fontsize=18, fontweight='bold')
    ax.set_xlabel('Road', fontsize=14)
    ax.set_ylabel('Volume (veh/min)', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

def display_result_cards(volumes, timings, directions, active_index, closed=None):
    """
    Render styled result cards displaying traffic data and signal timings.
    """
    sorted_data = sorted(
        zip(directions, volumes, timings, range(len(directions))),
        key=lambda x: x[1],
        reverse=True
    )
    for direction, volume, timing, idx in sorted_data:
        if closed is not None and direction == closed:
            continue
        card_class = "traffic-card"
        if idx == active_index:
            card_class += " active-road"
        green, yellow, red = timing
        card_html = f"""
        <div class="{card_class}">
            <h2 style="text-align:center;">{direction}</h2>
            <p style="text-align:center;">Volume: <strong>{int(volume)}</strong> veh/min</p>
            <p style="text-align:center;">Green: <strong>{green:.1f}s</strong></p>
            <p style="text-align:center;">Yellow: <strong>{yellow:.1f}s</strong></p>
            <p style="text-align:center;">Red: <strong>{red:.1f}s</strong></p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

def draw_traffic_lights(ax):
    """
    Draw traffic lights at the intersection.
    A vertical stack of circles represents red, yellow, and green lights.
    Uses a mapping to convert states to valid color codes.
    """
    # Positions for traffic lights near the center for each road
    positions = {
        "Road A": (30, 55),
        "Road B": (55, 30),
        "Road C": (70, 55),
        "Road D": (55, 70)
    }
    # Mapping for light states to valid color codes
    light_colors = {
        "light-red": "#ff4d4d",
        "light-yellow": "#ffcc00",
        "light-green": "#66cc66",
        "light-off": "#cccccc"
    }
    for idx, road in enumerate(directions):
        x, y = positions[road]
        # Default states: all lights off
        red_state = "light-off"
        yellow_state = "light-off"
        green_state = "light-off"
        if idx == st.session_state.current_road_index:
            # For active road: green on, or yellow if near end of phase
            if st.session_state.remaining_green_time < 3:
                yellow_state = "light-yellow"
            else:
                green_state = "light-green"
        else:
            red_state = "light-red"
        # Draw the three lights (circles)
        ax.add_patch(patches.Circle((x, y + 8), 3, color=light_colors[red_state], ec="black"))
        ax.add_patch(patches.Circle((x, y + 4), 3, color=light_colors[yellow_state], ec="black"))
        ax.add_patch(patches.Circle((x, y), 3, color=light_colors[green_state], ec="black"))

def draw_intersection():
    """
    Draw the entire intersection:
      - Roads with color highlighting for the active road
      - Traffic lights
      - Vehicles as oriented rectangles
    """
    fig, ax = plt.subplots(figsize=(6,6))
    color_active = "#c5f7c5"
    color_inactive = "#dddddd"
    # Draw roads
    color_A = color_active if st.session_state.current_road_index == 0 else color_inactive
    ax.add_patch(patches.Rectangle((0, 45), 50, 10, color=color_A))
    color_B = color_active if st.session_state.current_road_index == 1 else color_inactive
    ax.add_patch(patches.Rectangle((45, 0), 10, 50, color=color_B))
    color_C = color_active if st.session_state.current_road_index == 2 else color_inactive
    ax.add_patch(patches.Rectangle((50, 45), 50, 10, color=color_C))
    color_D = color_active if st.session_state.current_road_index == 3 else color_inactive
    ax.add_patch(patches.Rectangle((45, 50), 10, 50, color=color_D))
    
    # Draw traffic lights
    draw_traffic_lights(ax)
    
    # Draw vehicles for each road
    for road in directions:
        positions = st.session_state.vehicle_positions[road]
        for pos in positions:
            if road == "Road A":
                draw_vehicle(ax, x=pos, y=50, orientation="horizontal", color="red")
            elif road == "Road B":
                draw_vehicle(ax, x=50, y=pos, orientation="vertical", color="blue")
            elif road == "Road C":
                draw_vehicle(ax, x=MAX_DISTANCE - pos, y=50, orientation="horizontal", color="green")
            elif road == "Road D":
                draw_vehicle(ax, x=50, y=MAX_DISTANCE - pos, orientation="vertical", color="orange")
    ax.set_xlim(0, MAX_DISTANCE)
    ax.set_ylim(0, MAX_DISTANCE)
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)

def update_animation():
    """
    Update the animation frame in a single container.
    This clears previous frames and draws new visuals for smooth animation.
    """
    frame_container = st.empty()
    with frame_container.container():
        st.markdown(f"### Simulation Time: {st.session_state.sim_time:.1f}s")
        st.markdown(f"**Active Road (Green):** {directions[st.session_state.current_road_index]}")
        st.markdown(f"**Remaining Green Time:** {st.session_state.remaining_green_time:.1f}s")
        st.write("")  # Spacer
        display_traffic_chart(st.session_state.traffic_volumes, directions)
        display_result_cards(st.session_state.traffic_volumes,
                             st.session_state.timings,
                             directions,
                             active_index=st.session_state.current_road_index,
                             closed=None)
        st.markdown("## Intersection Animation")
        draw_intersection()

def reset_simulation(settings):
    """
    Reset simulation variables using the provided sidebar settings.
    """
    st.session_state.traffic_volumes = settings["volumes"]
    st.session_state.timings = calculate_timings(
        st.session_state.traffic_volumes,
        scenario=settings["road_condition"],
        weather=settings["weather_condition"],
        closed=settings["closed_road"],
        incident=settings["affected_road"],
        priority=settings["affected_road"]
    )
    st.session_state.sim_time = 0.0
    st.session_state.current_road_index = 0
    st.session_state.remaining_green_time = 0.0
    st.session_state.vehicle_positions = {d: [] for d in directions}
    st.session_state.sim_running = False

# =============================================================================
# SIDEBAR: USER INPUTS & SETTINGS
# =============================================================================

st.sidebar.header("Simulation Settings")

weather = st.sidebar.selectbox("Select Weather Condition", ["clear", "rainy", "foggy"])
st.session_state.weather_condition = weather

road_cond = st.sidebar.selectbox("Select Road Condition", 
                                 ["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"])
st.session_state.road_condition = road_cond

directions = ["Road A", "Road B", "Road C", "Road D"]

traffic_input = []
closed_road = None
affected_road = None

if road_cond == "Normal":
    for d in directions:
        tv = st.sidebar.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=30, step=1)
        traffic_input.append(tv)
elif road_cond == "Road Closure":
    closed_road = st.sidebar.selectbox("Select Road to Close", directions)
    for d in directions:
        if d == closed_road:
            traffic_input.append(0)
        else:
            tv = st.sidebar.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=30, step=1)
            traffic_input.append(tv)
else:
    affected_road = st.sidebar.selectbox("Select Affected Road", directions)
    for d in directions:
        tv = st.sidebar.number_input(f"Traffic volume for {d} (veh/min):", min_value=0, value=30, step=1)
        if d == affected_road:
            if road_cond == "Traffic Incident":
                tv = int(tv * 1.5)
            elif road_cond == "Construction Zone":
                tv = int(tv * 1.3)
        traffic_input.append(tv)

adjusted_volumes = adjust_for_weather(traffic_input, weather)

settings = {
    "volumes": adjusted_volumes,
    "weather_condition": weather,
    "road_condition": road_cond,
    "closed_road": closed_road,
    "affected_road": affected_road
}

if st.sidebar.button("Apply Settings"):
    reset_simulation(settings)
    st.sidebar.success("Settings applied. Simulation reset.")

# =============================================================================
# SIMULATION CONTROL BUTTONS
# =============================================================================

control_cols = st.columns([1, 1, 1, 2])
with control_cols[0]:
    if st.button("Start Simulation"):
        st.session_state.sim_running = True
with control_cols[1]:
    if st.button("Stop Simulation"):
        st.session_state.sim_running = False
with control_cols[2]:
    if st.button("Step Simulation"):
        run_simulation_step(REFRESH_INTERVAL)
with control_cols[3]:
    auto_chk = st.checkbox("Continuous Animation", value=st.session_state.sim_running)
    st.session_state.sim_running = auto_chk

# =============================================================================
# ANIMATION LOOP
# =============================================================================

animation_placeholder = st.empty()
if st.session_state.sim_running:
    # Continuous loop that updates the simulation in a single container.
    start_time = time.time()
    while st.session_state.sim_running and st.session_state.sim_time < SIM_DURATION:
        run_simulation_step(REFRESH_INTERVAL)
        with animation_placeholder.container():
            update_animation()
        time.sleep(REFRESH_INTERVAL)
else:
    update_animation()

# =============================================================================
# ADDITIONAL INFORMATION & FUTURE EXTENSIONS
# =============================================================================

st.markdown("---")
st.markdown("**Tip:** Adjust the settings in the sidebar and click **Apply Settings** to reset the simulation.")
st.markdown("**Note:** With Continuous Animation enabled, the simulation updates each second in a single animated container.")
st.markdown("Future enhancements may include advanced accident simulation and detailed logging.")

def simulate_accidents():
    """
    Placeholder: Simulate an accident on the active road.
    In a real implementation, this might alter vehicle speeds or signal timings.
    """
    if random.random() < 0.05:
        st.warning("⚠️ Accident occurred on the active road!")

def log_simulation_data():
    """
    Placeholder: Log simulation data for later analysis.
    """
    log_entry = {
        "sim_time": st.session_state.sim_time,
        "active_road": directions[st.session_state.current_road_index],
        "volumes": st.session_state.traffic_volumes.copy()
    }
    print(log_entry)

if st.session_state.sim_running:
    simulate_accidents()
    log_simulation_data()

# =============================================================================
# END OF CODE
# =============================================================================
