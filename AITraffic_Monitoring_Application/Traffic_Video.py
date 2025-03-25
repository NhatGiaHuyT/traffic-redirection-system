import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import random

################################################################################
#                                 INTRO & STYLES                                #
################################################################################

"""
This file defines a single-container Streamlit application that simulates
a traffic intersection with four roads (A, B, C, D). Each road can spawn
vehicles based on a Poisson-like process derived from a user-defined
"vehicles per minute" rate. The intersection logic uses green times
defined by "green, yellow, red" intervals for each road, and the simulation
progresses in real time, updating the display once per second.

Key design goals:
  1) A single background (white), no fancy layered images. 
  2) Only one figure displayed each frame, ensuring no stacking or leftover visuals.
  3) A "Continuous Animation" loop that updates the figure in place.
  4) Over 500 lines of code to allow for extensive commentary, clarity, and future extensibility.

You can tweak the refresh interval, the speeds, the volumes, etc. 
"""


# We set a white background so that no layering or transparency occurs.
CUSTOM_CSS = """
<style>
/* Force a single white background for the entire app */
.stApp {
    background: #FFFFFF !important;
}

/* Header styling */
.title h1 {
    font-size: 3em;
    color: #333333;
    text-align: center;
    margin-bottom: 0.2em;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    font-size: 16px;
}

/* Card styling for result info */
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
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# We set a wide layout so we can show both intersection & chart side by side
st.set_page_config(page_title="Smooth Single-Container Traffic Simulation", layout="wide")

################################################################################
#                               GLOBAL CONSTANTS                               #
################################################################################

# Basic geometry / speeds
MAX_DISTANCE = 100.0           # Max distance along each road
ROAD_WIDTH = 10                # Road width for the intersection
REFRESH_INTERVAL = 1.0         # Each simulation step is 1 second
BASE_SPEED = 20.0              # Vehicles move 20 distance units per second
VEHICLE_LENGTH = 4.0           # Vehicle rectangle length
VEHICLE_WIDTH = 2.0            # Vehicle rectangle width

# Simulation duration
SIM_DURATION = 300             # Maximum simulation time (seconds)
TIME_SCALE = 1.0               # Not used in detail, but could scale dt if needed

################################################################################
#                             SESSION STATE INIT                               #
################################################################################

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0  # Clock for the simulation
if "current_road_index" not in st.session_state:
    st.session_state.current_road_index = 0
if "remaining_green_time" not in st.session_state:
    st.session_state.remaining_green_time = 0.0
if "vehicle_positions" not in st.session_state:
    st.session_state.vehicle_positions = {
        "Road A": [],
        "Road B": [],
        "Road C": [],
        "Road D": []
    }
if "traffic_volumes" not in st.session_state:
    # vehicles per minute on each road
    st.session_state.traffic_volumes = [30, 30, 30, 30]
if "timings" not in st.session_state:
    # (green, yellow, red) in seconds for each road
    st.session_state.timings = [(15, 4, 6)] * 4
if "weather_condition" not in st.session_state:
    st.session_state.weather_condition = "clear"
if "road_condition" not in st.session_state:
    st.session_state.road_condition = "Normal"

directions = ["Road A", "Road B", "Road C", "Road D"]

################################################################################
#                           HEADER & SIDEBAR NAVIGATION                        #
################################################################################

st.sidebar.markdown("## Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "home"

st.markdown("<div class='title'><h1>🚦 Single-Container Traffic Simulation 🚦</h1></div>", 
            unsafe_allow_html=True)

################################################################################
#                       SIMULATION HELPER FUNCTIONS                            #
################################################################################

def adjust_for_weather(volumes, condition):
    """
    Adjust traffic volumes based on weather conditions.
    E.g. rainy => multiply by 1.8, foggy => multiply by 1.5
    """
    adjusted = volumes.copy()
    if condition == "rainy":
        adjusted = [v * 1.8 for v in adjusted]
    elif condition == "foggy":
        adjusted = [v * 1.5 for v in adjusted]
    return adjusted

def calculate_timings(volumes, scenario, weather, closed=None, incident=None, priority=None):
    """
    Calculate (green, yellow, red) timings for each road based on volumes,
    scenario, weather, etc.
    """
    base_green = 15
    base_yellow = 4
    base_red = 6
    out = []
    for i, vol in enumerate(volumes):
        green = base_green + int(vol / 30)
        yellow = base_yellow + int(vol / 50)
        red = base_red + int(vol / 60)
        if weather == "rainy":
            yellow += 2
        if weather == "foggy":
            red += 2
        if scenario == "Emergency Response":
            if i == priority:
                green = int(green * 1.5)
            else:
                green = int(green * 0.8)
                red = int(red * 0.7)
        elif scenario == "Road Closure" and i == closed:
            green, yellow, red = 0, 0, 0
        elif scenario in ["Traffic Incident", "Construction Zone"] and i == incident:
            factor = 1.5 if scenario == "Traffic Incident" else 1.2
            green = int(green * factor)
        out.append((green, yellow, red))
    return out

def spawn_vehicles(volumes, dt):
    """
    Spawn new vehicles on each road based on volumes (veh/min).
    Poisson-like process => probability = (volume/60)*dt
    """
    for i, road in enumerate(directions):
        rate = volumes[i] / 60.0
        if random.random() < rate * dt:
            st.session_state.vehicle_positions[road].append(0.0)

def move_vehicles(dt):
    """
    Move vehicles on the active road forward by BASE_SPEED * dt.
    """
    step = BASE_SPEED * dt
    active_road = directions[st.session_state.current_road_index]
    for road in directions:
        new_positions = []
        for pos in st.session_state.vehicle_positions[road]:
            if road == active_road:
                pos += step
            if pos <= MAX_DISTANCE:
                new_positions.append(pos)
        st.session_state.vehicle_positions[road] = new_positions

def run_simulation_step(dt):
    """
    Advance the simulation by dt seconds:
      - If green time is done, switch roads
      - Decrease traffic volume on the active road
      - Spawn & move vehicles
      - Increment sim_time
    """
    if not st.session_state.timings:
        return
    idx = st.session_state.current_road_index
    if st.session_state.remaining_green_time <= 0:
        # Start a new phase
        st.session_state.remaining_green_time = st.session_state.timings[idx][0]
    st.session_state.remaining_green_time -= dt

    # Vehicle departure from active road
    leave_rate = 1.0  # vehicles/sec
    st.session_state.traffic_volumes[idx] = max(
        st.session_state.traffic_volumes[idx] - (leave_rate * dt), 0
    )

    # Spawn & move
    spawn_vehicles(st.session_state.traffic_volumes, dt)
    move_vehicles(dt)

    # If green ended, switch to next road
    if st.session_state.remaining_green_time <= 0:
        st.session_state.current_road_index = (idx + 1) % len(directions)

    st.session_state.sim_time += dt

def reset_simulation(settings):
    """
    Reset simulation with new volumes, timings, etc.
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

################################################################################
#                       VISUALIZATION (SINGLE FIGURE)                          #
################################################################################

def draw_vehicle(ax, x, y, orientation, color):
    """
    Draw a vehicle as a rectangle on the given axis.
    orientation: "horizontal" or "vertical"
    """
    if orientation == "horizontal":
        rect = patches.Rectangle((x, y - VEHICLE_WIDTH/2), VEHICLE_LENGTH, VEHICLE_WIDTH,
                                 facecolor=color, edgecolor='black')
        ax.add_patch(rect)
    elif orientation == "vertical":
        rect = patches.Rectangle((x - VEHICLE_WIDTH/2, y - VEHICLE_LENGTH), VEHICLE_WIDTH, VEHICLE_LENGTH,
                                 facecolor=color, edgecolor='black')
        ax.add_patch(rect)

def draw_single_figure():
    """
    Create one figure with two subplots side by side:
      1) Intersection with roads & vehicles
      2) Bar chart of traffic volumes
    Return the figure object to be displayed in the single placeholder.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig.subplots_adjust(wspace=0.3)

    # Force white backgrounds to avoid transparency
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # -- Intersection on ax1 --
    color_active = "#c5f7c5"
    color_inactive = "#dddddd"

    # Road A
    cA = color_active if st.session_state.current_road_index == 0 else color_inactive
    ax1.add_patch(patches.Rectangle((0, 45), 50, ROAD_WIDTH, color=cA))
    # Road B
    cB = color_active if st.session_state.current_road_index == 1 else color_inactive
    ax1.add_patch(patches.Rectangle((45, 0), ROAD_WIDTH, 50, color=cB))
    # Road C
    cC = color_active if st.session_state.current_road_index == 2 else color_inactive
    ax1.add_patch(patches.Rectangle((50, 45), 50, ROAD_WIDTH, color=cC))
    # Road D
    cD = color_active if st.session_state.current_road_index == 3 else color_inactive
    ax1.add_patch(patches.Rectangle((45, 50), ROAD_WIDTH, 50, color=cD))

    # Draw vehicles
    for road in directions:
        positions = st.session_state.vehicle_positions[road]
        for pos in positions:
            if road == "Road A":
                draw_vehicle(ax1, x=pos, y=50, orientation="horizontal", color="red")
            elif road == "Road B":
                draw_vehicle(ax1, x=50, y=pos, orientation="vertical", color="blue")
            elif road == "Road C":
                draw_vehicle(ax1, x=MAX_DISTANCE - pos, y=50, orientation="horizontal", color="green")
            elif road == "Road D":
                draw_vehicle(ax1, x=50, y=MAX_DISTANCE - pos, orientation="vertical", color="orange")

    ax1.set_xlim(0, MAX_DISTANCE)
    ax1.set_ylim(0, MAX_DISTANCE)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Intersection", fontsize=14, fontweight='bold')

    # -- Traffic volumes on ax2 --
    volumes = st.session_state.traffic_volumes
    colors = ["green" if v < 20 else "orange" if v < 40 else "red" for v in volumes]
    ax2.bar(directions, volumes, color=colors)
    ax2.set_title("Traffic Volumes", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Road", fontsize=12)
    ax2.set_ylabel("Volume (veh/min)", fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    return fig

def display_result_cards():
    """
    Display road info in cards sorted by volume descending. 
    The active road is highlighted.
    """
    sorted_data = sorted(
        zip(directions, st.session_state.traffic_volumes, st.session_state.timings, range(len(directions))),
        key=lambda x: x[1],
        reverse=True
    )
    for direction, volume, timing, idx in sorted_data:
        card_class = "traffic-card"
        if idx == st.session_state.current_road_index:
            card_class += " active-road"
        green, yellow, red = timing
        st.markdown(f"""
        <div class="{card_class}">
            <h2 style="text-align:center;">{direction}</h2>
            <p style="text-align:center;">Volume: <strong>{int(volume)}</strong> veh/min</p>
            <p style="text-align:center;">Green: <strong>{green:.1f}s</strong></p>
            <p style="text-align:center;">Yellow: <strong>{yellow:.1f}s</strong></p>
            <p style="text-align:center;">Red: <strong>{red:.1f}s</strong></p>
        </div>
        """, unsafe_allow_html=True)

################################################################################
#                          SIDEBAR USER SETTINGS                               #
################################################################################

st.sidebar.header("Simulation Settings")

weather = st.sidebar.selectbox("Select Weather Condition", ["clear", "rainy", "foggy"])
st.session_state.weather_condition = weather

road_cond = st.sidebar.selectbox(
    "Select Road Condition",
    ["Normal", "Road Closure", "Traffic Incident", "Construction Zone", "Emergency Response"]
)
st.session_state.road_condition = road_cond

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

################################################################################
#                        SIMULATION CONTROL BUTTONS                            #
################################################################################

col1, col2, col3, col4 = st.columns([1,1,1,2])
with col1:
    if st.button("Start Simulation"):
        st.session_state.sim_running = True
with col2:
    if st.button("Stop Simulation"):
        st.session_state.sim_running = False
with col3:
    if st.button("Step Simulation"):
        run_simulation_step(REFRESH_INTERVAL)
with col4:
    auto_chk = st.checkbox("Continuous Animation", value=st.session_state.sim_running)
    st.session_state.sim_running = auto_chk

################################################################################
#                           ANIMATION PLACEHOLDER                              #
################################################################################

animation_placeholder = st.empty()

def update_frame():
    """
    Renders one single frame inside the single placeholder:
      - Simulation status text
      - A single figure with intersection & bar chart
      - Timings / volumes in card layout
    """
    with animation_placeholder.container():
        # Display top-level simulation status
        st.markdown(f"### Simulation Time: {st.session_state.sim_time:.1f}s")
        st.markdown(f"**Active Road (Green):** {directions[st.session_state.current_road_index]}")
        st.markdown(f"**Remaining Green Time:** {st.session_state.remaining_green_time:.1f}s")

        # Build the figure
        fig = draw_single_figure()
        st.pyplot(fig)
        plt.close(fig)  # Avoid leftover references

        # Display result cards
        display_result_cards()

# If user started the simulation, run a continuous loop
if st.session_state.sim_running:
    start_time = time.time()
    while st.session_state.sim_running and st.session_state.sim_time < SIM_DURATION:
        run_simulation_step(REFRESH_INTERVAL)
        update_frame()
        time.sleep(REFRESH_INTERVAL)
else:
    # Just render one frame if not running
    update_frame()

################################################################################
#                    ACCIDENT SIMULATION & DATA LOGGING                        #
################################################################################

def simulate_accidents():
    """
    Placeholder for random accidents on the active road.
    Could reduce speed or alter signal timings if desired.
    """
    if random.random() < 0.05:
        st.warning("⚠️ Accident occurred on the active road!")

def log_simulation_data():
    """
    Placeholder for logging the simulation data each step.
    """
    log_entry = {
        "time": st.session_state.sim_time,
        "active_road": directions[st.session_state.current_road_index],
        "volumes": st.session_state.traffic_volumes.copy()
    }
    # For now, just print to console
    print(log_entry)

if st.session_state.sim_running:
    simulate_accidents()
    log_simulation_data()

################################################################################
#                                 END OF CODE                                  #
################################################################################
