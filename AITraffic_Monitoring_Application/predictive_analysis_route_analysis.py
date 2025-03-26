import streamlit as st
from streamlit_lottie import st_lottie


import pickle
import folium
from streamlit_folium import folium_static
import pandas as pd
import heapq
import requests
import warnings
import random
from datetime import datetime, timedelta
import altair as alt
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

################################################################################
#                               GLOBAL FUNCTIONS                               #
################################################################################

st.markdown(
    """
    <style>
    /* Overall background color for the page */
    body {
        background-color: #F8F9FA; /* a light grey */
    }

    /* Default text color */
    .stText, .stTitle, .stHeader, .stSubheader, .stMarkdown, .stCaption,
    .stRadio, .stSelectbox, .stButton, .stCheckbox, .stTabs, .stDataFrame,
    .stMetric, .stAlert, .css-1v0mbdj, .css-1cpxqw2 {
        color: #212529 !important; /* a dark grey/black */
    }

    /* Make headers more visible */
    h1, h2, h3, h4, h5, h6 {
        color: #212529 !important;
    }

    /* Adjust the top margin to reduce whitespace */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Example: darken the main header bar at the top */
    header, .css-18e3th9 {
        background-color: #343A40;
    }

    /* If you have a sidebar, you can change its background too */
    .css-1d391kg {
        background-color: #FFFFFF !important;
    }

    /* Buttons: override primary color if desired */
    .stButton>button {
        background-color: #1E88E5 !important;
        color: #FFFFFF !important;
        border-radius: 5px;
        border: none;
    }

    .stElementContainer, .st-emotion-cache-ah6jdd, p {
        color: #212529 !important; /* a dark grey/black */
    }

    /* Lottie animation container styling if needed */
    .lottie-container {
        margin-top: 0px;
        margin-bottom: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function: load_graph_model
def load_graph_model(pickle_file):
    """
    Load a pre-trained graph model from a pickle file.
    """
    with open(pickle_file, 'rb') as f:
        graph = pickle.load(f)
    return graph

# Function: load_csv_dataset
def load_csv_dataset(csv_file):
    """
    Load a CSV dataset containing route and traffic data.
    Expected columns: Date, Time, Day of the Week, Origin, Destination, Route, Weather Conditions,
                      Accident Reports, Traffic Intensity, Distance.
    """
    data = pd.read_csv(csv_file)
    return data

# Function: modify_graph_with_data
def modify_graph_with_data(graph, data):
    """
    Modify the graph weights dynamically based on dataset information.
    Weight calculation uses distance, accident reports, and traffic intensity.
    """
    for index, row in data.iterrows():
        origin = row['Origin']
        destination = row['Destination']

        # Get required columns with defaults
        distance = row.get('Distance', 0)
        accidents = row.get('Accident Reports', 0)
        traffic = row.get('Traffic Intensity', 0)
        
        # Calculate weight: distance + (accidents factor) + (traffic factor)
        weight = distance + (accidents * 100) + (traffic * 10)
        
        # Ensure graph has nodes for origin and destination
        if origin not in graph:
            graph[origin] = {}
        if destination not in graph:
            graph[destination] = {}

        # Update the graph in both directions (undirected graph assumption)
        graph[origin][destination] = (weight, row.get('Route', 'N/A'))
        graph[destination][origin] = (weight, row.get('Route', 'N/A'))
        
    return graph

# Function: dijkstra
def dijkstra(graph, start, end):
    """
    Find the shortest path between start and end nodes using Dijkstra's algorithm.
    Returns total cost, path as list, and route details.
    """
    priority_queue = [(0, start, [], "")]
    visited = set()
    
    while priority_queue:
        total_cost, current_node, path, route_taken = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        
        path = path + [current_node]
        visited.add(current_node)
        
        if current_node == end:
            return total_cost, path, route_taken
        
        for neighbor, (weight, route) in graph[current_node].items():
            if neighbor not in visited:
                new_route_taken = route_taken + f"{current_node} -> {neighbor} (Route: {route})\n"
                heapq.heappush(priority_queue, (total_cost + weight, neighbor, path, new_route_taken))
    
    return float("inf"), [], ""

# Function: load_lottieurl
def load_lottieurl(url: str):
    """
    Load a Lottie animation from a URL.
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

# Function: predict_congestion_and_time
def predict_congestion_and_time(route_data, time_adjustment=0):
    """
    Predict congestion level and travel time based on the Traffic Intensity.
    Optionally, adjust traffic intensity based on time of day.
    """
    traffic_intensity = int(route_data['Traffic Intensity']) + time_adjustment
    
    if traffic_intensity < 30:
        congestion = 'Low'
        predicted_time = 15
    elif 30 <= traffic_intensity < 60:
        congestion = 'Medium'
        predicted_time = 30
    else:
        congestion = 'High'
        predicted_time = 60
    return congestion, predicted_time

################################################################################
#                               APP CONFIGURATION                              #
################################################################################


# Back button in the top right corner
if st.button("Back", key='back_button'):
    st.session_state.page = 'home'  # Change session state to go back to home

# Custom CSS for select boxes and other components
st.markdown(
    """
    <style>
    .stSelectbox label {
        color: #333;
        font-weight: bold;
        font-size: 18px;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

################################################################################
#                             SIDEBAR CONTROLS                                 #
################################################################################

st.sidebar.header("Navigation & Controls")

# Sidebar: Data file selection (for extensibility)
data_file = st.sidebar.text_input("Data CSV File Path", "district5_complex_routes_with_datetime.csv")
graph_file = st.sidebar.text_input("Graph Model File Path", "route_optimization_graph.pkl")

# Sidebar: Time of Day selection for prediction adjustments
time_option = st.sidebar.selectbox(
    "Select Time of Day",
    ["Peak Hours (8-10 AM, 5-7 PM)", "Off-Peak Hours (All other times)"]
)

# Sidebar: Animation toggle
animation_toggle = st.sidebar.checkbox("Show Animations", value=True)

################################################################################
#                              LOAD DATA & MODELS                              #
################################################################################

# Load traffic dataset
try:
    traffic_data = load_csv_dataset(data_file)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Extract unique route entries (drop duplicates)
unique_routes = traffic_data[['Origin', 'Destination', 'Route', 'Weather Conditions', 'Accident Reports', 'Traffic Intensity', 'Distance']].drop_duplicates()

# Load graph model
try:
    graph = load_graph_model(graph_file)
except Exception as e:
    st.error(f"Error loading graph model: {e}")
    st.stop()

# Modify graph with real-time dataset information
graph = modify_graph_with_data(graph, unique_routes)

################################################################################
#                               TITLE & HEADER                                 #
################################################################################

st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>
        Traffic Congestion and Route Optimization Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

################################################################################
#                             ANIMATION DISPLAY                                #
################################################################################

if animation_toggle:
    # Load traffic animation
    traffic_animation = load_lottieurl("https://lottie.host/ac9b87c1-302c-4eae-89af-1ac6d54c163b/8gYZJC5OpB.json")
    if traffic_animation:
        st_lottie(traffic_animation, height=200, key="traffic")

################################################################################
#                           WEATHER ANIMATION & INFO                           #
################################################################################

def get_weather_animation():
    """
    Randomly select a weather animation (sunny or cloudy) based on the current day.
    """
    current_day = datetime.now().day
    random.seed(current_day)
    weather_choice = random.choice(['clear', 'cloudy'])
    if weather_choice == 'clear':
        return load_lottieurl("https://lottie.host/d8ca6425-9e61-49b9-9bf7-8247dac3a459/tJrOUe25Kc.json"), "Sunny"
    else:
        return load_lottieurl("https://lottie.host/a6de97d2-f147-4699-9a80-e1f6aba6bb6d/FZJoVUa2Np.json"), "Cloudy"

weather_animation, weather_description = get_weather_animation()

################################################################################
#                          USER INPUT FOR ROUTE SELECTION                      #
################################################################################

st.subheader("Select Your Route")

# Dropdown: Select Origin
origin = st.selectbox("Select Origin", unique_routes['Origin'].unique())

# Dropdown: Select Destination based on Origin
def get_destinations(origin):
    return list(unique_routes[unique_routes['Origin'] == origin]['Destination'].unique())

destination = st.selectbox("Select Destination", get_destinations(origin))

# Display selected date/time info from dataset sample (if desired)
st.write("### Selected Route Details from Historical Data")
st.write(f"Origin: **{origin}**")
st.write(f"Destination: **{destination}**")

################################################################################
#                        ROUTE PREDICTION & CONGESTION INFO                    #
################################################################################

# Get selected route data (if multiple, choose first)
try:
    selected_route = unique_routes[(unique_routes['Origin'] == origin) & 
                                   (unique_routes['Destination'] == destination)].iloc[0]
except IndexError:
    st.error("No data available for the selected route. Please try another combination.")
    st.stop()

# Adjust traffic intensity based on time selection
time_adjustment = 20 if time_option == "Peak Hours (8-10 AM, 5-7 PM)" else 0

# Predict congestion and travel time
congestion, predicted_time = predict_congestion_and_time(selected_route, time_adjustment)

# Display prediction details
st.markdown("#### Route Prediction")
st.write(f"**Congestion Level:** {congestion}")
st.write(f"**Predicted Travel Time:** {predicted_time} minutes")
st.write(f"**Current Weather:** {weather_description}")
if animation_toggle and weather_animation:
    st_lottie(weather_animation, height=200, key="weather")

################################################################################
#                         MAP VISUALIZATION SECTION                            #
################################################################################

# Coordinates dictionary for map visualization
# Note: For District 5, update these with real coordinates of your landmarks
location_coords = {
    "Chợ Bình Tây": (10.749827770462066, 106.65103751151939),
    "Đền Bà Thiên Hậu": (10.75336599156594, 106.66131566993364),
    "Chợ Bình Đông": (10.740305157329562, 106.64346801433803),
    "Bưu điện Chợ Lớn": (10.75015281650651, 106.65920920671381),
    "Trung tâm Văn hóa Chợ Lớn": (10.752567211813698, 106.66837345199895),
    "Bệnh viện Quận 5": (10.754172214677302, 106.6658424815772),
    "Trường Tiểu học Chợ Lớn": 10.75146329767459, 106.6653704125339),
    "Trung tâm Thương Mại Chợ Lớn": (10.754380689267403, 106.66551845199916),
    "Đền Vua": (10.753735421388246, 106.68254348391028),
    "Trường Trung học Chợ Lớn": (10.752469890339768, 106.66679354438318),
    "Nhà thờ Chợ Lớn": (10.752450409355223, 106.65462434250949),
    "Công viên District 5": 10.765354741619404, 106.68074468391033),
    "Trung tâm Y tế Quận 5": (10.75973350653036, 106.669339853849),
    "Trung tâm Văn hóa Dân gian": (10.752621903536784, 106.66833300315976),
    "Sân vận động Quận 5": (10.76126477470413, 106.66344058113613),
    "Bảo tàng Quận 5": (10.776855424333268, 106.69992342910145),
    "Ngã tư Lê Văn Sỹ": (10.79663069590259, 106.66568831496896),
    "Trạm xe buýt Quận 5": (10.762648542663369, 106.67774630409893),
    "Cửa hàng Điện máy X": (10.755582614035308, 106.68065572575534),
    "Siêu thị Co-op Mart": (10.761366423779116, 106.67140083683259)
}

# Retrieve lat/lon for origin and destination
lat_o, lon_o = location_coords.get(origin, (0, 0))
lat_d, lon_d = location_coords.get(destination, (0, 0))

if lat_o == 0 or lon_o == 0 or lat_d == 0 or lon_d == 0:
    st.error("One or more location coordinates not found. Please verify the location names.")
else:
    # Set polyline color based on congestion level
    polyline_color = "green" if congestion == "Low" else "yellow" if congestion == "Medium" else "red"

    # Create folium map centered on origin
    m = folium.Map(location=[lat_o, lon_o], zoom_start=13)
    
    # Add polyline for route
    folium.PolyLine(
        locations=[(lat_o, lon_o), (lat_d, lon_d)],
        color=polyline_color,
        weight=6,
        tooltip=f"{origin} to {destination}: {congestion} congestion"
    ).add_to(m)
    
    # Add markers for origin and destination
    folium.CircleMarker(
        location=[lat_o, lon_o],
        radius=8,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.7,
        popup=f"Start: {origin}"
    ).add_to(m)
    
    folium.CircleMarker(
        location=[lat_d, lon_d],
        radius=8,
        color="green",
        fill=True,
        fill_color="green",
        fill_opacity=0.7,
        popup=f"End: {destination}"
    ).add_to(m)
    
    # Display map in app
    st.subheader("Route Map")
    folium_static(m)

################################################################################
#                          BEST ROUTE OPTIMIZATION                             #
################################################################################

st.markdown("## Best Route Optimization Using Dijkstra's Algorithm")
if origin and destination:
    total_cost, path, route_taken = dijkstra(graph, origin, destination)
    if path:
        st.write(f"**The best route from {origin} to {destination} is:**")
        st.write(f"**Distance (Cost):** {total_cost:.2f} units")
        st.text("Detailed Route Path:")
        st.text(route_taken)
    else:
        st.write("No path found between the selected origin and destination.")

################################################################################
#                          DATA EXPLORATION & VISUALIZATION                    #
################################################################################

st.markdown("## Data Exploration & Analysis")

# Create tabs for various exploratory views
tabs = st.tabs(["Overview", "Traffic Analysis", "Accidents & Weather", "Time Trends"])

# --------------------- TAB 1: OVERVIEW ---------------------
with tabs[0]:
    st.subheader("Dataset Overview")
    st.write("Below is a sample of the traffic congestion dataset:")
    st.dataframe(traffic_data.head(20), height=300)

    st.write("### Basic Statistics")
    st.write(traffic_data.describe())

# --------------------- TAB 2: TRAFFIC ANALYSIS ---------------------
with tabs[1]:
    st.subheader("Traffic Intensity Distribution")
    fig_traffic = px.histogram(traffic_data, x="Traffic Intensity", nbins=50,
                               title="Distribution of Traffic Intensity")
    st.plotly_chart(fig_traffic, use_container_width=True)
    
    st.subheader("Traffic Intensity by Route")
    route_traffic = traffic_data.groupby("Route")["Traffic Intensity"].mean().reset_index()
    fig_route_traffic = px.bar(route_traffic, x="Route", y="Traffic Intensity",
                               title="Average Traffic Intensity by Route",
                               labels={"Traffic Intensity": "Avg Traffic Intensity"})
    st.plotly_chart(fig_route_traffic, use_container_width=True)

# --------------------- TAB 3: ACCIDENTS & WEATHER ---------------------
with tabs[2]:
    st.subheader("Accident Reports Analysis")
    fig_accidents = alt.Chart(traffic_data).mark_bar().encode(
        x=alt.X("Accident Reports:Q", bin=alt.Bin(maxbins=20)),
        y='count()',
        tooltip=['count()']
    ).properties(width=700, height=400, title="Distribution of Accident Reports")
    st.altair_chart(fig_accidents, use_container_width=True)

    st.subheader("Weather Conditions Overview")
    weather_counts = traffic_data["Weather Conditions"].value_counts().reset_index()
    weather_counts.columns = ["Weather Conditions", "Count"]
    fig_weather = px.pie(weather_counts, names="Weather Conditions", values="Count",
                         title="Weather Conditions Proportion")
    st.plotly_chart(fig_weather, use_container_width=True)

# --------------------- TAB 4: TIME TRENDS ---------------------
with tabs[3]:
    st.subheader("Traffic Trends by Day of the Week")
    day_traffic = traffic_data.groupby("Day of the Week")["Traffic Intensity"].mean().reset_index()
    fig_day_traffic = px.line(day_traffic, x="Day of the Week", y="Traffic Intensity",
                              title="Average Traffic Intensity by Day of the Week",
                              markers=True)
    st.plotly_chart(fig_day_traffic, use_container_width=True)

    st.subheader("Accidents Over Time")
    # Convert Date column to datetime
    traffic_data["Date"] = pd.to_datetime(traffic_data["Date"])
    accidents_over_time = traffic_data.groupby("Date")["Accident Reports"].sum().reset_index()
    fig_accidents_time = px.area(accidents_over_time, x="Date", y="Accident Reports",
                                 title="Daily Total Accident Reports")
    st.plotly_chart(fig_accidents_time, use_container_width=True)

################################################################################
#                          CUSTOM INTERACTIVE FILTERS                          #
################################################################################

st.markdown("## Interactive Data Filters")

# Sidebar filter for date range selection
st.sidebar.markdown("### Filter Data by Date")
min_date = traffic_data["Date"].min()
max_date = traffic_data["Date"].max()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date])

# Filter dataset based on selected date range
if len(date_range) == 2:
    filtered_data = traffic_data[(traffic_data["Date"] >= pd.to_datetime(date_range[0])) &
                                 (traffic_data["Date"] <= pd.to_datetime(date_range[1]))]
else:
    filtered_data = traffic_data.copy()

st.write("### Data Sample After Date Filter")
st.dataframe(filtered_data.head(10), height=250)

################################################################################
#                          ADVANCED VISUALIZATION                              #
################################################################################

st.markdown("## Advanced Visualizations")

# Advanced Chart: Interactive Scatter Plot for Distance vs Traffic Intensity
st.subheader("Distance vs. Traffic Intensity Scatter Plot")
scatter_fig = px.scatter(filtered_data, x="Distance", y="Traffic Intensity",
                         color="Weather Conditions",
                         hover_data=["Origin", "Destination", "Accident Reports"],
                         title="Relationship between Distance and Traffic Intensity")
st.plotly_chart(scatter_fig, use_container_width=True)

# Advanced Chart: Heatmap for Average Traffic Intensity by Day & Weather
st.subheader("Heatmap: Day of the Week vs Weather Conditions")
heatmap_data = filtered_data.groupby(["Day of the Week", "Weather Conditions"])["Traffic Intensity"].mean().reset_index()
heatmap_fig = px.density_heatmap(heatmap_data, x="Day of the Week", y="Weather Conditions",
                                 z="Traffic Intensity", color_continuous_scale="Viridis",
                                 title="Avg Traffic Intensity by Day and Weather")
st.plotly_chart(heatmap_fig, use_container_width=True)

################################################################################
#                          FUTURE IMPROVEMENTS & NOTES                         #
################################################################################

st.markdown("## Future Improvements & Developer Notes")
st.info("""
- **Real-time Data Integration:** Consider integrating live traffic data feeds to update predictions dynamically.
- **Machine Learning Models:** The current prediction is rule-based. Integrating an ML model trained on historical data could improve accuracy.
- **User Customization:** Expand sidebar filters to include route selection, weather conditions, and time-of-day adjustments.
- **Mapping Enhancements:** Use more detailed geospatial data and markers, and consider clustering for dense areas.
- **Performance Optimization:** As the dataset grows, optimize data loading and caching to maintain performance.
""")

################################################################################
#                          FOOTER & ADDITIONAL INFORMATION                     #
################################################################################

st.markdown("""
---
**Traffic Congestion and Route Optimization Dashboard**  
Developed using Streamlit, Folium, Altair, and Plotly.  
For more details, contact the development team.
""")

################################################################################
#                              ADDITIONAL UTILITY CODE                         #
################################################################################
# The following code is reserved for future functionality such as exporting data,
# advanced routing analysis, and integration with external APIs.

def export_filtered_data(dataframe, filename="filtered_data.csv"):
    """
    Export the filtered dataframe to a CSV file.
    """
    dataframe.to_csv(filename, index=False)
    st.success(f"Filtered data exported to {filename}")

if st.sidebar.button("Export Filtered Data"):
    export_filtered_data(filtered_data)

def show_data_summary(df):
    """
    Display a detailed summary of the dataset.
    """
    st.markdown("### Data Summary")
    st.write("Number of records:", df.shape[0])
    st.write("Columns:", list(df.columns))
    st.write(df.describe())

if st.sidebar.checkbox("Show Data Summary"):
    show_data_summary(filtered_data)