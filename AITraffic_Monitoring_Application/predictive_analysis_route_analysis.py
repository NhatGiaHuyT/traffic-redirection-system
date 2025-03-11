import streamlit as st
import pickle
import folium
from streamlit_folium import folium_static
import pandas as pd
import heapq
import requests
import warnings
import random
from streamlit_lottie import st_lottie
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

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

# Load the saved graph model (pickle file)
def load_graph_model(pickle_file):
    with open(pickle_file, 'rb') as f:
        graph = pickle.load(f)
    return graph

# Load dataset from CSV
def load_dataset():
    data = pd.read_csv(r"Routes_New.csv")
    return data

# Modify graph weights dynamically based on dataset information
def modify_graph_with_data(graph, data):
    for index, row in data.iterrows():
        origin = row['Origin']
        destination = row['Destination']
        
        # Get required columns
        distance = row.get('Distance', 0)  # Default to 0 if the column doesn't exist
        accidents = row.get('Accident_Reports', 0)  # Default to 0
        traffic = row.get('Traffic_Intensity', 0)  # Default to 0
        
        # Adjust the weight based on real-time factors (accidents, traffic)
        weight = distance + (accidents * 100) + (traffic * 10)

        # Ensure the graph has both origin and destination
        if origin not in graph:
            graph[origin] = {}
        if destination not in graph:
            graph[destination] = {}

        # Update the existing weight with new data
        graph[origin][destination] = (weight, row.get('Route', 'N/A'))  # Use the route from the dataset
        graph[destination][origin] = (weight, row.get('Route', 'N/A'))  # Assuming the graph is undirected

    return graph

# Dijkstra's Algorithm to find the shortest path
def dijkstra(graph, start, end):
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

# Function to load Lottie animations from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the saved traffic prediction model
model_path = r"traffic_prediction_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Title for the app
st.markdown(
    """
    <h1 style='text-align: center;'>
        Traffic Congestion and Route Optimization
    </h1>
    """, 
    unsafe_allow_html=True
)

# Load traffic animation (add your traffic animation URL)
traffic_animation = load_lottieurl("https://lottie.host/ac9b87c1-302c-4eae-89af-1ac6d54c163b/8gYZJC5OpB.json")  # Replace with your traffic animation URL

# Display traffic animation under the title
st_lottie(traffic_animation, height=200, key="traffic")

# Load weather animations
clear_animation = load_lottieurl("https://lottie.host/d8ca6425-9e61-49b9-9bf7-8247dac3a459/tJrOUe25Kc.json")  # Example sunny weather animation
cloudy_animation = load_lottieurl("https://lottie.host/a6de97d2-f147-4699-9a80-e1f6aba6bb6d/FZJoVUa2Np.json")  # Example cloudy weather animation

# Randomize weather based on day
def get_weather_animation():
    current_day = datetime.now().day  # Get current day
    random.seed(current_day)  # Set seed based on the day, so it's consistent within the day
    weather_choice = random.choice(['clear', 'cloudy'])  # Randomly choose clear or cloudy
    
    if weather_choice == 'clear':
        return clear_animation, "Sunny"
    else:
        return cloudy_animation, "Cloudy"

# Load the dataset
file_path = r"Traffic_Congestion_dataset.csv"
traffic_data = pd.read_csv(file_path)

# Extract unique routes data once
unique_routes = traffic_data[['Origin', 'Destination', 'Route', 'Weather Conditions', 'Accident Reports', 'Traffic Intensity', 'Distance']].drop_duplicates()

# Function to get available destinations for the selected origin
def get_destinations(origin):
    return list(unique_routes[unique_routes['Origin'] == origin]['Destination'].unique())

# Function to predict congestion level and travel time based on dataset
def predict_congestion_and_time(route_data):
    traffic_intensity = int(route_data['Traffic Intensity'])
    
    # Determine congestion level
    if traffic_intensity < 30:
        congestion = 'Low'
        predicted_time = 15  # Minutes for low congestion
    elif 30 <= traffic_intensity < 60:
        congestion = 'Medium'
        predicted_time = 30  # Minutes for medium congestion
    else:
        congestion = 'High'
        predicted_time = 60  # Minutes for high congestion

    return congestion, predicted_time

# Unified Dropdown Pair for Origin and Destination
origin = st.selectbox('Select Origin', unique_routes['Origin'].unique())
destination = st.selectbox('Select Destination', get_destinations(origin))



# Get the weather animation and description
weather_animation, weather_description = get_weather_animation()


# Get the selected route data for prediction
selected_route = unique_routes[(unique_routes['Origin'] == origin) & 
                               (unique_routes['Destination'] == destination)].iloc[0]


# Get predicted congestion level and travel time for the selected route
congestion, predicted_time = predict_congestion_and_time(selected_route)


# Map part - (Using lat/lon from a dictionary)
location_coords = {
    "Connaught Place": (28.6315, 77.2167),
    "India Gate": (28.6129, 77.2295),
    "Dhaula Kuan": (28.5921, 77.1734),
    "Gurugram": (28.4595, 77.0266),
    "IFFCO Chowk, Gurugram":(28.4722,77.0724),
    "Dilli Haat":(28.5733,77.2075),
    "Janakpuri Metro Station":(28.6331,77.0867),
    "Dwarka Sector 21":(28.5522,77.0583),
    "Khan Market":(28.6002,77.2270),
    "Safdarjung Enclave":(28.5647,77.1949),
    "lajpat Nagar":(28.5649,77.2403),
    "Greater Kailash":(28.5482,77.2380),
    "Rohini":(28.7383, 77.0822),
    "Shiv Vihar": (28.7253, 77.2793)
    # Add more locations as needed...
}

# Approximate lat/lon for these locations (for visualization)
lat_o, lon_o = location_coords.get(origin, (0, 0))  # Default to (0,0) if location is not found
lat_d, lon_d = location_coords.get(destination, (0, 0))

# Check if both lat/lon are valid before rendering the map
if lat_o == 0 or lon_o == 0 or lat_d == 0 or lon_d == 0:
    st.error("Selected location coordinates not found. Please check the location names.")
else:
    # Determine polyline color based on congestion level
    if congestion == 'Low':
        polyline_color = 'green'
    elif congestion == 'Medium':
        polyline_color = 'yellow'
    else:
        polyline_color = 'red'

    # Create a folium map centered at the origin location
    m = folium.Map(location=[lat_o, lon_o], zoom_start=12)

    # Add the route to the map
    folium.PolyLine(
        locations=[(lat_o, lon_o), (lat_d, lon_d)],
        color=polyline_color,  # Change color based on congestion
        weight=6,
        tooltip=f"{origin} to {destination}: {congestion} congestion"
    ).add_to(m)

    # Add circle markers for the origin and destination
    folium.CircleMarker(
        location=[lat_o, lon_o],
        radius=10,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"Start: {origin}"
    ).add_to(m)

    folium.CircleMarker(
        location=[lat_d, lon_d],
        radius=10,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.6,
        popup=f"End: {destination}"
    ).add_to(m)

    # Show the map in Streamlit
    folium_static(m)
    
# Get today's date
today = datetime.now().date()

# Calculate the next day
next_day = today + timedelta(days=1)

# Display the next day for which the prediction is being made
st.write(f"**Date (Next Day):** {next_day}")

st.write(f"**Origin:** {origin}")
st.write(f"**Destination:** {destination}")    

st.write(f"**Congestion Level:** {congestion}")
st.write(f"**Predicted Travel Time:** {predicted_time} minutes")
# Display weather animation under congestion prediction section


# Display the current weather condition in congestion details
st.write(f"**Today's Weather:** {weather_description}")
st_lottie(weather_animation, height=200, key="weather")




# Load graph model
graph_model_path = r"route_optimization_graph.pkl"
graph = load_graph_model(graph_model_path)

# Modify the graph with real-time data
graph = modify_graph_with_data(graph, unique_routes)

# Find the best route using Dijkstra's algorithm
if origin and destination:
    total_cost, path, route_taken = dijkstra(graph, origin, destination)

    # Display the results
    if path:
        st.write(f"The best route from **{origin}** to **{destination}** is:")
        st.write(f"Distance: **{total_cost} units**.")
        
        # Show the route taken with detailed information
        st.write("**Route Details:**")
        st.write(route_taken)  # Displaying the detailed route taken
    else:
        st.write("No path found between the selected origin and destination.")
        
# Add a time selector for peak/off-peak hours
time_option = st.selectbox(
    'Select Time of Day',
    ['Peak Hours (8-10 AM, 5-7 PM)', 'Off-Peak Hours (All other times)']
)

# Modify the congestion prediction function to account for time selection
def predict_congestion_and_time(route_data, time_option):
    traffic_intensity = int(route_data['Traffic Intensity'])

    # Adjust traffic intensity based on time selection
    if time_option == 'Peak Hours (8-10 AM, 5-7 PM)':
        traffic_intensity += 20  # Increase traffic intensity during peak hours

    # Determine congestion level
    if traffic_intensity < 30:
        congestion = 'Low'
        predicted_time = 15  # Minutes for low congestion
    elif 30 <= traffic_intensity < 60:
        congestion = 'Medium'
        predicted_time = 30  # Minutes for medium congestion
    else:
        congestion = 'High'
        predicted_time = 60  # Minutes for high congestion

    return congestion, predicted_time

# Get predicted congestion level and travel time for the selected route
congestion, predicted_time = predict_congestion_and_time(selected_route, time_option)

# Display congestion details
st.write(f"**Congestion Level:** {congestion}")
st.write(f"**Predicted Travel Time:** {predicted_time} minutes")

