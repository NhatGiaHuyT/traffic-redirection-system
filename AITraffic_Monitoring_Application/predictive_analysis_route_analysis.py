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
#                             LANGUAGE CONTROLS                                #
################################################################################
lang_mapping = {"vi":{
    "page_title": "Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình",
    # sidebar nav
    "side_nav_title": "Điều hướng & Điều khiển",
    "side_nav_data_file": "Đường dẫn tệp CSV dữ liệu",
    "side_nav_graph_file": "Đường dẫn tệp mô hình đồ thị",
    "side_nav_time_option": "Chọn thời gian trong ngày",
    "side_nav_time_option_values": ["Giờ cao điểm (8-10 AM, 5-7 PM)", "Giờ thấp điểm (Tất cả các thời gian khác)"],
    "side_nav_animation": "Hiện hoạt hình",
    # sidebar filtering
    "side_fil_title": "Lọc dữ liệu theo ngày",
    "side_fil_date_range": "Chọn khoảng thời gian",
    "side_fil_button": "XUẤT DỮ LIỆU ĐÃ LỌC",
    "side_fil_summary": "Hiện tóm tắt dữ liệu",
    # route prediction
    "page_title": "Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình",
    "select_route": "Chọn lộ trình của bạn",
    "select_origin_label": "Chọn điểm xuất phát",
    "select_destination_label": "Chọn điểm đến",
    "select_route_details": "Chi tiết lộ trình đã chọn từ dữ liệu lịch sử",
    "select_route_details_origin": "Điểm xuất phát",
    "select_route_details_destination": "Điểm đến",
    "route_prediction": "Dự đoán lộ trình",
    "route_prediction_congestion": "Mức độ tắc nghẽn",
    "route_prediction_time": "Thời gian di chuyển dự đoán",
    "route_prediction_weather": "Thời tiết hiện tại",
    "route_map": "Bản đồ lộ trình",
    "best_route": "Tối ưu hóa lộ trình tốt nhất bằng thuật toán Dijkstra",
    "best_route_details": "Lộ trình tốt nhất từ {} đến {} là:",
    "best_route_distance": f"**Khoảng cách (Chi phí):** {{:.2f}} đơn vị",
    "best_route_path": "Chi tiết lộ trình",
    # data exploration
    "data_exploration": "Khám phá và phân tích dữ liệu",
    "tabs_values": ["Tổng quan", "Phân tích giao thông", "Tai nạn & Thời tiết", "Xu hướng theo thời gian"],
    "tabs_overview": "Tổng quan",
    "tabs_traffic_analysis": "Phân tích giao thông",
    "tabs_accidents_weather": "Tình trạng tai nạn và thời tiết",
    "tabs_time_trends": "Xu hướng theo thời gian",
    # overview tab
    "overview_title": "Tổng quan về tập dữ liệu",
    "overview_sample": "Dưới đây là một mẫu của tập dữ liệu tắc nghẽn giao thông:",
    "overview_statistics": "Thống kê cơ bản",
    # traffic analysis tab
    "traffic_analysis_title": "Phân phối cường độ giao thông",
    "traffic_analysis_route": "Cường độ giao thông trung bình theo lộ trình",
    "traffic_analysis_distribution_title": "Phân phối cường độ giao thông",
    "traffic_analysis_distribution_x": "Cường độ giao thông",
    "traffic_analysis_distribution_y": "Số lượng",
    "traffic_analysis_x_label": "Lộ trình",
    "traffic_analysis_y_label": "Cường độ giao thông trung bình",
    # accidents & weather tab
    "accidents_weather_title": "Phân tích báo cáo tai nạn",
    "accidents_weather_weather": "Tổng quan về điều kiện thời tiết",
    "accidents_weather_distribution_title": "Phân phối báo cáo tai nạn",
    "accidents_weather_distribution_x": "Báo cáo tai nạn (Phân loại)",
    "accidents_weather_distribution_y": "Số lượng bản ghi",
    "accidents_weather_weather_conditions_title": "Tỷ lệ điều kiện thời tiết",
    "accidents_weather_weather_conditions": "Điều kiện thời tiết",
    "accidents_weather_weather_conditions_count": "Số lượng",
    # time trends tab
    "time_trends_title": "Xu hướng giao thông theo ngày trong tuần",
    "time_trends_by_day_graph_title": "Cường độ giao thông trung bình theo ngày trong tuần",
    "time_trends_by_day_graph_x": "Ngày trong tuần",
    "time_trends_accidents": "Tai nạn theo thời gian",
    "time_trends_accidents_graph_title": "Tổng số báo cáo tai nạn hàng ngày",
    "time_trends_accidents_graph_x": "Ngày",
    "time_trends_accidents_graph_y": "Báo cáo tai nạn",
    # interactive filters
    "interactive_filters": "Bộ lọc dữ liệu tương tác",
    "interactive_filters_sample": "Mẫu dữ liệu sau bộ lọc ngày",
    # advanced visualizations
    "advanced_visualizations": "Hình ảnh nâng cao",
    "advanced_scatter": "Biểu đồ phân tán khoảng cách so với cường độ giao thông",
    "advanced_scatter_x": "Khoảng cách",
    "advanced_scatter_y": "Cường độ giao thông",
    "advanced_heatmap": "Bản đồ nhiệt: Ngày trong tuần so với điều kiện thời tiết",
    "advanced_heatmap_title": "Bản đồ nhiệt cường độ giao thông trung bình theo ngày trong tuần và điều kiện thời tiết",
    "advanced_heatmap_x": "Ngày trong tuần",
    "advanced_heatmap_y": "Điều kiện thời tiết",
    "advanced_heatmap_z": "Tổng cường độ giao thông",
    # future improvements
    "future_improvements": "Cải tiến trong tương lai & Ghi chú của nhà phát triển",
    "future_improvements_note": """
- **Tích hợp dữ liệu thời gian thực:** Cân nhắc tích hợp nguồn dữ liệu giao thông trực tiếp để cập nhật dự đoán một cách động.
- **Mô hình học máy:** Dự đoán hiện tại dựa trên quy tắc. Tích hợp mô hình học máy được đào tạo trên dữ liệu lịch sử có thể cải thiện độ chính xác.
- **Tùy chỉnh của người dùng:** Mở rộng bộ lọc thanh bên để bao gồm lựa chọn lộ trình, điều kiện thời tiết và điều chỉnh thời gian trong ngày.
- **Cải tiến bản đồ:** Sử dụng dữ liệu địa lý chi tiết hơn và các điểm đánh dấu, và xem xét việc phân nhóm cho các khu vực đông đúc.
- **Tối ưu hóa hiệu suất:** Khi tập dữ liệu lớn lên, tối ưu hóa việc tải và lưu trữ dữ liệu để duy trì hiệu suất.
""",
    # footer
    "footer": """
---
**Bảng điều khiển tắc nghẽn giao thông và tối ưu hóa lộ trình**
Được phát triển bằng Streamlit, Folium, Altair và Plotly.
Liên hệ với nhóm phát triển để biết thêm chi tiết.
""",

    # additional utility code
    "data_summary": "Hiện tóm tắt dữ liệu",
    "data_summary_title": "Tóm tắt dữ liệu",
    "data_summary_records": "Số lượng bản ghi",
    "data_summary_columns": "Cột",
    
}, 
"en":{
    "page_title": "Traffic Congestion and Route Optimization Dashboard",
    # sidebar nav
    "side_nav_title": "Navigation & Controls",
    "side_nav_data_file": "Data CSV File Path",
    "side_nav_graph_file": "Graph Model File Path",
    "side_nav_time_option": "Select Time of Day",
    "side_nav_time_option_values": ["Peak Hours (8-10 AM, 5-7 PM)", "Off-Peak Hours (All other times)"],
    "side_nav_animation": "Show Animations",
    # sidebar filtering
    "side_fil_title": "Filter Data by Date",    
    "side_fil_date_range": "Select date range",
    "side_fil_button": "EXPORT FILTERED DATA",
    "side_fil_summary": "Show Data Summary",
    # route prediction
    "page_title": "Traffic Congestion and Route Optimization Dashboard",
    "select_route": "Select Your Route",
    "select_origin_label": "Select Origin",
    "select_destination_label": "Select Destination",
    "select_route_details": "Selected Route Details from Historical Data",
    "select_route_details_origin": "Origin",
    "select_route_details_destination": "Destination",
    "route_prediction": "Route Prediction",
    "route_prediction_congestion": "Congestion Level",
    "route_prediction_time": "Predicted Travel Time",
    "route_prediction_weather": "Current Weather",
    "route_map": "Route Map",
    "best_route": "Best Route Optimization Using Dijkstra's Algorithm",
    "best_route_details": "The best route from {} to {} is:",
    "best_route_distance": f"**Distance (Cost): {{:.2f}} units**",
    "best_route_path": "Detailed Route Path",
    # data exploration
    "data_exploration": "Data Exploration & Analysis",
    "tabs_values": ["Overview", "Traffic Analysis", "Accidents & Weather", "Time Trends"],
    "tabs_overview": "Overview",
    "tabs_traffic_analysis": "Traffic Analysis",
    "tabs_accidents_weather": "Accidents & Weather",
    "tabs_time_trends": "Time Trends",
    # overview tab
    "overview_title": "Dataset Overview",
    "overview_sample": "Below is a sample of the traffic congestion dataset:",
    "overview_statistics": "Basic Statistics",
    # traffic analysis tab
    "traffic_analysis_title": "Traffic Intensity Distribution",
    "traffic_analysis_route": "Average Traffic Intensity by Route",
    "traffic_analysis_distribution_title": "Distribution of Traffic Intensity",
    "traffic_analysis_distribution_x": "Traffic Intensity",
    "traffic_analysis_distribution_y": "Count",
    "traffic_analysis_x_label": "Route",
    "traffic_analysis_y_label": "Avg Traffic Intensity",
    # accidents & weather tab
    "accidents_weather_title": "Accident Reports Analysis",
    "accidents_weather_weather": "Weather Conditions Overview",
    "accidents_weather_distribution_title": "Distribution of Accident Reports",
    "accidents_weather_distribution_x": "Accident Reports (Binned)",
    "accidents_weather_distribution_y": "Count of Records",
    "accidents_weather_weather_conditions_title": "Weather Conditions Proportion",
    "accidents_weather_weather_conditions": "Weather Conditions",
    "accidents_weather_weather_conditions_count": "Count",
    # time trends tab
    "time_trends_title": "Traffic Trends by Day of the Week",
    "time_trends_by_day_graph_title": "Average Traffic Intensity by Day of the Week",
    "time_trends_by_day_graph_x": "Day of the Week",
    "time_trends_by_day_graph_y": "Traffic Intensity",
    "time_trends_accidents": "Accidents Over Time",
    "time_trends_accidents_graph_title": "Daily Total Accident Reports",
    "time_trends_accidents_graph_x": "Date",
    "time_trends_accidents_graph_y": "Accident Reports",
    # interactive filters
    "interactive_filters": "Interactive Data Filters",
    "interactive_filters_sample": "Data Sample After Date Filter",
    # advanced visualizations
    "advanced_visualizations": "Advanced Visualizations",
    "advanced_scatter": "Distance vs. Traffic Intensity Scatter Plot",
    "advanced_scatter_x": "Distance",
    "advanced_scatter_y": "Traffic Intensity",
    "advanced_heatmap": "Heatmap: Day of the Week vs Weather Conditions",
    "advanced_heatmap_title": "Average Traffic Intensity Heatmap by Day of the Week and Weather Conditions",
    "advanced_heatmap_x": "Day of the Week",
    "advanced_heatmap_y": "Weather Conditions",
    "advanced_heatmap_z": "Sum of Traffic Intensity",
    # future improvements
    "future_improvements": "Future Improvements & Developer Notes",
    "future_improvements_note": """
- **Real-time Data Integration:** Consider integrating live traffic data feeds to update predictions dynamically.
- **Machine Learning Models:** The current prediction is rule-based. Integrating an ML model trained on historical data could improve accuracy.
- **User Customization:** Expand sidebar filters to include route selection, weather conditions, and time-of-day adjustments.
- **Mapping Enhancements:** Use more detailed geospatial data and markers, and consider clustering for dense areas.
- **Performance Optimization:** As the dataset grows, optimize data loading and caching to maintain performance.
""",
    # footer
    "footer": """
---
**Traffic Congestion and Route Optimization Dashboard**
Developed using Streamlit, Folium, Altair, and Plotly.
For more details, contact the development team.
""",
    # additional utility code
    "data_summary_title": "Data Summary",
    "data_summary_records": "Number of records",
    "data_summary_columns": "Columns",
}}

if "lang" not in st.session_state:
    st.session_state.lang = "vi"  # Default to Vietnamese

lang = st.session_state.lang
################################################################################
#                             SIDEBAR CONTROLS                                 #
################################################################################

st.sidebar.header(lang_mapping[lang]["side_nav_title"])

# Sidebar: Data file selection (for extensibility)
data_file = st.sidebar.text_input(lang_mapping[lang]["side_nav_data_file"], "district5_complex_routes_with_datetime.csv")
graph_file = st.sidebar.text_input(lang_mapping[lang]["side_nav_graph_file"], "route_optimization_graph.pkl")

# Sidebar: Time of Day selection for prediction adjustments
time_option = st.sidebar.selectbox(
    lang_mapping[lang]["side_nav_time_option"],
    lang_mapping[lang]["side_nav_time_option_values"],
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
        {page_title}
    </h1>
    """.format(page_title=lang_mapping[lang]["page_title"]),
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

st.subheader(lang_mapping[lang]["select_route"])

# Dropdown: Select Origin
origin = st.selectbox(lang_mapping[lang]["select_origin_label"], unique_routes['Origin'].unique())

# Dropdown: Select Destination based on Origin
def get_destinations(origin):
    return list(unique_routes[unique_routes['Origin'] == origin]['Destination'].unique())

destination = st.selectbox(lang_mapping[lang]["select_destination_label"]
    , get_destinations(origin))

# Display selected date/time info from dataset sample (if desired)
st.write(f"### {lang_mapping[lang]['select_route_details']}")
st.write(f"**{lang_mapping[lang]['select_route_details_origin']}:** {origin}")
st.write(f"**{lang_mapping[lang]['select_route_details_destination']}:** {destination}")

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
time_adjustment = 20 if time_option == (
    lang == "en" and "Peak Hours (8-10 AM, 5-7 PM)" or "Giờ cao điểm (8-10 AM, 5-7 PM)"
) else 0

# Predict congestion and travel time
congestion, predicted_time = predict_congestion_and_time(selected_route, time_adjustment)

# Display prediction details
st.markdown(f"#### {lang_mapping[lang]['route_prediction']}")
st.write(f"**{lang_mapping[lang]['route_prediction_congestion']}:** {congestion}")
st.write(f"**{lang_mapping[lang]['route_prediction_time']}:** {predicted_time} minutes")
st.write(f"**{lang_mapping[lang]['route_prediction_weather']}:** {weather_description}")
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
    "Trường Tiểu học Chợ Lớn": (10.75146329767459, 106.6653704125339),
    "Trung tâm Thương Mại Chợ Lớn": (10.754380689267403, 106.66551845199916),
    "Đền Vua": (10.753735421388246, 106.68254348391028),
    "Trường Trung học Chợ Lớn": (10.752469890339768, 106.66679354438318),
    "Nhà thờ Chợ Lớn": (10.752450409355223, 106.65462434250949),
    "Công viên District 5": (10.765354741619404, 106.68074468391033),
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
    st.subheader(lang_mapping[lang]["route_map"])
    folium.Marker(location=[lat_o, lon_o], popup=origin).add_to(m)
    folium.Marker(location=[lat_d, lon_d], popup=destination, icon=folium.Icon(color="green")).add_to(m)
    folium_static(m)

################################################################################
#                          BEST ROUTE OPTIMIZATION                             #
################################################################################

st.markdown(f"## {lang_mapping[lang]['best_route']}")
if origin and destination:
    total_cost, path, route_taken = dijkstra(graph, origin, destination)
    if path:
        st.write(f"**{lang_mapping[lang]['best_route_details'].format(origin, destination)}**")
        st.write(f"**{lang_mapping[lang]['best_route_distance'].format(total_cost)}**")
        st.text(f"{lang_mapping[lang]['best_route_path']}")
        st.text(route_taken)
    else:
        if lang == "en":
            st.write("No path found between the selected origin and destination.")
        else:
            st.write("Không tìm thấy lộ trình giữa điểm xuất phát và điểm đến đã chọn.")

################################################################################
#                          DATA EXPLORATION & VISUALIZATION                    #
################################################################################

st.markdown(f"## {lang_mapping[lang]['data_exploration']}")

# Create tabs for various exploratory views
tabs = st.tabs(lang_mapping[lang]["tabs_values"])

# --------------------- TAB 1: OVERVIEW ---------------------
with tabs[0]:
    st.subheader(lang_mapping[lang]["overview_title"])
    st.write(lang_mapping[lang]["overview_sample"])
    st.dataframe(traffic_data.head(20), height=300)

    st.write(f"### {lang_mapping[lang]['overview_statistics']}")
    st.write(traffic_data.describe())

# --------------------- TAB 2: TRAFFIC ANALYSIS ---------------------
with tabs[1]:
    st.subheader(lang_mapping[lang]["traffic_analysis_title"])
    fig_traffic = px.histogram(traffic_data, x="Traffic Intensity", nbins=50,
                               title=lang_mapping[lang]["traffic_analysis_distribution_title"],
                                 labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_distribution_x"],
                                            "count": lang_mapping[lang]["traffic_analysis_distribution_y"]})
    st.plotly_chart(fig_traffic, use_container_width=True)
    
    st.subheader(lang_mapping[lang]["traffic_analysis_route"])
    route_traffic = traffic_data.groupby("Route")["Traffic Intensity"].mean().reset_index()
    fig_route_traffic = px.bar(route_traffic, x="Route", y="Traffic Intensity",
                               title=lang_mapping[lang]["traffic_analysis_route"],
                               labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_y_label"],
                                       "Route": lang_mapping[lang]["traffic_analysis_x_label"]})
    st.plotly_chart(fig_route_traffic, use_container_width=True)

# --------------------- TAB 3: ACCIDENTS & WEATHER ---------------------
with tabs[2]:
    st.subheader(lang_mapping[lang]["accidents_weather_title"])
    fig_accidents = alt.Chart(traffic_data).mark_bar().encode(
        x=alt.X("Accident Reports:Q", bin=alt.Bin(maxbins=20)),
        y='count()',
        tooltip=['count()']
    ).properties(width=700, height=400, title=lang_mapping[lang]["accidents_weather_distribution_title"])
    st.altair_chart(fig_accidents, use_container_width=True)

    st.subheader(lang_mapping[lang]["accidents_weather_weather"])
    weather_counts = traffic_data["Weather Conditions"].value_counts().reset_index()
    weather_counts.columns = [lang_mapping[lang]["accidents_weather_weather_conditions"],
                                lang_mapping[lang]["accidents_weather_weather_conditions_count"]]
    fig_weather = px.pie(weather_counts, names=lang_mapping[lang]["accidents_weather_weather_conditions"],
                            values=lang_mapping[lang]["accidents_weather_weather_conditions_count"],
                         title=lang_mapping[lang]["accidents_weather_weather_conditions_title"])
    st.plotly_chart(fig_weather, use_container_width=True)

# --------------------- TAB 4: TIME TRENDS ---------------------
with tabs[3]:
    st.subheader(lang_mapping[lang]["time_trends_title"])
    day_traffic = traffic_data.groupby("Day of the Week")["Traffic Intensity"].mean().reset_index()
    fig_day_traffic = px.line(day_traffic, x="Day of the Week", y="Traffic Intensity",
                              title=lang_mapping[lang]["time_trends_by_day_graph_title"],
                                labels={"Traffic Intensity": lang_mapping[lang]["traffic_analysis_y_label"],
                                        "Day of the Week": lang_mapping[lang]["time_trends_by_day_graph_x"]},
                              markers=True)
    st.plotly_chart(fig_day_traffic, use_container_width=True)

    st.subheader(lang_mapping[lang]["time_trends_accidents"])
    # Convert Date column to datetime
    traffic_data["Date"] = pd.to_datetime(traffic_data["Date"])
    accidents_over_time = traffic_data.groupby("Date")["Accident Reports"].sum().reset_index()
    fig_accidents_time = px.area(accidents_over_time, x="Date", y="Accident Reports",
                                 title="Daily Total Accident Reports",
                                    labels={"Accident Reports": lang_mapping[lang]["time_trends_accidents_graph_y"],
                                            "Date": lang_mapping[lang]["time_trends_accidents_graph_x"]})
    st.plotly_chart(fig_accidents_time, use_container_width=True)

################################################################################
#                          CUSTOM INTERACTIVE FILTERS                          #
################################################################################

st.markdown(f"## {lang_mapping[lang]['interactive_filters']}")

# Sidebar filter for date range selection
st.sidebar.markdown(f"### {lang_mapping[lang]['side_fil_title']}")
min_date = traffic_data["Date"].min()
max_date = traffic_data["Date"].max()
date_range = st.sidebar.date_input(lang_mapping[lang]["side_fil_date_range"], [min_date, max_date])

# Filter dataset based on selected date range
if len(date_range) == 2:
    filtered_data = traffic_data[(traffic_data["Date"] >= pd.to_datetime(date_range[0])) &
                                 (traffic_data["Date"] <= pd.to_datetime(date_range[1]))]
else:
    filtered_data = traffic_data.copy()

st.write(f"### {lang_mapping[lang]['interactive_filters_sample']}")
st.dataframe(filtered_data.head(10), height=250)

################################################################################
#                          ADVANCED VISUALIZATION                              #
################################################################################

st.markdown(f"## {lang_mapping[lang]['advanced_visualizations']}")

# Advanced Chart: Interactive Scatter Plot for Distance vs Traffic Intensity
st.subheader(lang_mapping[lang]["advanced_scatter"])
scatter_fig = px.scatter(filtered_data, x="Distance", y="Traffic Intensity",
                         color="Weather Conditions",
                         hover_data=["Origin", "Destination", "Accident Reports"],
                         title=lang_mapping[lang]["advanced_scatter"],
                            labels={"Distance": lang_mapping[lang]["advanced_scatter_x"],
                                    "Traffic Intensity": lang_mapping[lang]["advanced_scatter_y"]})
st.plotly_chart(scatter_fig, use_container_width=True)

# Advanced Chart: Heatmap for Average Traffic Intensity by Day & Weather
st.subheader(lang_mapping[lang]["advanced_heatmap"])
heatmap_data = filtered_data.groupby(["Day of the Week", "Weather Conditions"])["Traffic Intensity"].mean().reset_index()
heatmap_fig = px.density_heatmap(heatmap_data, x="Day of the Week", y="Weather Conditions",
                                 z="Traffic Intensity", color_continuous_scale="Viridis",
                                 title=lang_mapping[lang]["advanced_heatmap_title"],
                                    labels={"Traffic Intensity": lang_mapping[lang]["advanced_heatmap_z"],
                                            "Day of the Week": lang_mapping[lang]["advanced_heatmap_x"],
                                            "Weather Conditions": lang_mapping[lang]["advanced_heatmap_y"]})
st.plotly_chart(heatmap_fig, use_container_width=True)

################################################################################
#                          FUTURE IMPROVEMENTS & NOTES                         #
################################################################################

st.markdown(f"## {lang_mapping[lang]['future_improvements']}")
st.info(lang_mapping[lang]["future_improvements_note"])

################################################################################
#                          FOOTER & ADDITIONAL INFORMATION                     #
################################################################################

st.markdown(f"### {lang_mapping[lang]['footer']}")

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

if st.sidebar.button(lang_mapping[lang]["side_fil_button"]):
    export_filtered_data(filtered_data)

def show_data_summary(df):
    """
    Display a detailed summary of the dataset.
    """
    st.markdown("### Data Summary")
    st.write("Number of records:", df.shape[0])
    st.write("Columns:", list(df.columns))
    st.write(df.describe())

if st.sidebar.checkbox(lang_mapping[lang]["data_summary"]):
    show_data_summary(filtered_data)