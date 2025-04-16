import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="RouteVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a visually pleasing traffic-themed interface
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Montserrat', sans-serif;
        color: #333;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        color: mistyrose;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><path d="M95,50 L85,35 L85,20 L75,20 L75,10 L25,10 L25,20 L15,20 L15,35 L5,50 L15,65 L15,80 L25,80 L25,90 L75,90 L75,80 L85,80 L85,65 Z" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="2"/></svg>');
        background-repeat: repeat;
        opacity: 0.15;
    }
    
    .header-title {
        color: white; 
        font-size: 42px; 
        font-weight: 700; 
        text-align: center; 
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 10;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9); 
        font-size: 18px; 
        text-align: center; 
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 10;
    }
    
    /* Card Styles */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 28px;
        height: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
        border-top: 5px solid #2a5298;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 48px;
        text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #1e3c72, #4286f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    .feature-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 12px;
        color: #1e3c72;
        text-align: center;
    }
    
    .feature-description {
        font-size: 15px;
        color: #666;
        margin-bottom: 20px;
        text-align: center;
        line-height: 1.5;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        margin-top: auto;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 12px rgba(46, 82, 152, 0.3);
        height: auto;
    }
    
    .stButton > button:hover {
        background: linear-gradient(to right, #2a5298, #1e3c72);
        box-shadow: 0 6px 15px rgba(46, 82, 152, 0.4);
        transform: translateY(-2px);
    }
    
    /* Traffic theme elements */
    .traffic-line {
        height: 3px;
        background: repeating-linear-gradient(90deg, #ffcc00, #ffcc00 20px, transparent 20px, transparent 40px);
        margin: 30px 0;
        animation: moveLine 10s linear infinite;
    }
    
    @keyframes moveLine {
        0% { background-position: 0 0; }
        100% { background-position: 100px 0; }
    }
    
    /* Footer Styles */
    .footer {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin-top: 40px;
        text-align: center;
        box-shadow: 0 -5px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Home Button */
    .home-button {
        position: fixed;
        top: 62px;
        left: 20px;
        z-index: 999;
    }

    .lang-button {
        position: fixed;
        top: 62px;
        left: 20px;
        z-index: 999;
    }

    .home-button button {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 22px;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .home-button button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
    }

    
    
    /* Road Animation */
    .road-container {
        height: 30px;
        overflow: hidden;
        position: relative;
        margin: 10px 0 30px 0;
    }
    
    .road {
        height: 100%;
        background-color: #333;
        position: relative;
    }
    
    .road-line {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        height: 4px;
        background: #ffcc00;
        width: 30px;
        animation: roadLine 5s linear infinite;
    }
    
    @keyframes roadLine {
        0% { transform: translateX(-100px) translateY(-50%); }
        100% { transform: translateX(calc(100vw + 100px)) translateY(-50%); }
    }
    
    .road-line:nth-child(1) { animation-delay: 0s; }
    .road-line:nth-child(2) { animation-delay: 1s; }
    .road-line:nth-child(3) { animation-delay: 2s; }
    .road-line:nth-child(4) { animation-delay: 3s; }
    .road-line:nth-child(5) { animation-delay: 4s; }
    
    /* Card container to ensure equal height */
    .feature-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    
    /* Fix for button display inside cards */
    .button-wrapper {
        margin-top: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'lang' not in st.session_state:
    st.session_state.lang = 'vi'
selected_lang = st.toggle("🇻🇳", value="vi", key="lang_selector") and "vi" or "en"
# Add language button
lang_mapping = {
    "vi": {
        "home_title": "Chào mừng đến với RouteVision AI!",
        "home_subtitle": "Nền tảng Giám sát & Tối ưu hóa Giao thông Tiên tiến",
        "home_traffic_feed": "Thông Tin Giao Thông",
        "home_route_optimizer": "Lộ Trình Tối Ưu",
        "home_signal_simulation": "Mô Phỏng Tín Hiệu",
        "home_traffic_feed_desc": "Xem camera giao thông thời gian thực với phân tích AI. Giám sát ùn tắc, sự cố và lưu lượng giao thông trên toàn mạng lưới của bạn.",
        "home_route_optimizer_desc": "Tìm lộ trình nhanh nhất với phân tích dự đoán AI. Tránh ùn tắc và giảm thời gian di chuyển với thông tin giao thông thời gian thực.",
        "home_signal_simulation_desc": "Kiểm soát và mô phỏng tín hiệu giao thông thông minh. Kiểm tra chiến lược thời gian tín hiệu và tối ưu hóa lưu lượng giao thông với hệ thống AI của chúng tôi.",
        "home_feature_desc": "Tính năng nổi bật",
        "home_feature_traffic_feed": "KHÁM PHÁ THÔNG TIN GIAO THÔNG",
        "home_feature_route_optimizer": "TỐI ƯU LỘ TRÌNH",
        "home_feature_signal_simulation": "THỰC HIỆN MÔ PHỎNG",
        "home_dashboard_preview": "Làm cho các thành phố thông minh hơn",
        "home_dashboard_desc": "RouteVision AI giúp các nhà quản lý giao thông, nhà quy hoạch đô thị và người đi lại đưa ra quyết định tốt hơn thông qua phân tích dữ liệu thời gian thực và mô hình dự đoán. Nền tảng của chúng tôi giảm ùn tắc, khí thải và thời gian di chuyển.",
        "home_footer": "© 2025 RouteVision AI - Chuyển Đổi Giao Thông Đô Thị Thông Qua Trí Tuệ Nhân Tạo",
        "home_button": "🏠 Trang Chủ",
    },
    "en": {
        "home_title": "Welcome to RouteVision AI!",
        "home_subtitle": "Advanced Traffic Monitoring & Optimization Platform",
        "home_traffic_feed": "Traffic Feed",
        "home_route_optimizer": "Route Optimizer",
        "home_signal_simulation": "Signal Simulation",
        "home_traffic_feed_desc": "View real-time traffic cameras with AI-powered analytics. Monitor congestion, incidents, and traffic flow across your network.",
        "home_route_optimizer_desc": "Find the fastest routes with AI predictive analysis. Avoid congestion and reduce travel time with real-time traffic insights.",
        "home_signal_simulation_desc": "Smart traffic signal control and simulation. Test signal timing strategies and optimize traffic flow with our AI-powered system.",
        "home_feature_desc": "Featured",
        "home_feature_traffic_feed": "EXPLORE TRAFFIC FEED",
        "home_feature_route_optimizer": "OPTIMIZE ROUTES",
        "home_feature_signal_simulation": "RUN SIMULATION",
        "home_dashboard_preview": "Making Cities Smarter",
        "home_dashboard_desc": "RouteVision AI helps traffic managers, city planners, and commuters make better decisions through real-time data analysis and predictive modeling. Our platform reduces congestion, emissions, and travel times.",
        "home_footer": "© 2025 RouteVision AI - Transforming Urban Mobility Through Artificial Intelligence",
        "home_button": "🏠 Home",
    }

}
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang

# Add home button when not on home page
if st.session_state.page != 'home':
    # Hidden button to trigger the action
    if st.button(lang_mapping[st.session_state.lang]["home_button"]
        , key="home-button-hidden", help="Return to homepage"):
        st.session_state.page = 'home'
    

# Logic to navigate to different Python files
if st.session_state.page == 'home':
    # Animated road at the top
    st.markdown(
        """
        <div class="road-container">
            <div class="road">
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
                <div class="road-line"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Header with traffic-themed design
    st.markdown(
        """
        <div class="main-header">
            <h1 class="header-title">
                {home_title}
            </h1>
            <h3 class="header-subtitle">
                {home_subtitle}
            </h3>
        </div>
        """.format(
            home_title=lang_mapping[st.session_state.lang]["home_title"],
            home_subtitle=lang_mapping[st.session_state.lang]["home_subtitle"]
        ),
        unsafe_allow_html=True
    )
    
    # Create horizontal layout with enhanced cards and WORKING BUTTONS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🎥</div>
                    <div class="feature-title">{home_traffic_feed}</div>
                    <div class="feature-description">{home_traffic_feed_desc}</div>
                </div>
            </div>
            """
            .format(
                home_traffic_feed=lang_mapping[st.session_state.lang]["home_traffic_feed"],
                home_traffic_feed_desc=lang_mapping[st.session_state.lang]["home_traffic_feed_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button(lang_mapping[st.session_state.lang]["home_traffic_feed"], key="traffic_btn"):
            st.session_state.page = 'traffic_video'
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🗺️</div>
                    <div class="feature-title">{home_route_optimizer}</div>
                    <div class="feature-description">{home_route_optimizer_desc}</div>
                </div>
            </div>
            """
            .format(
                home_route_optimizer=lang_mapping[st.session_state.lang]["home_route_optimizer"],
                home_route_optimizer_desc=lang_mapping[st.session_state.lang]["home_route_optimizer_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button(lang_mapping[st.session_state.lang]["home_route_optimizer"]
            , key="route_btn"):
            st.session_state.page = 'route_optimize_predictor'
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-container">
                    <div class="feature-icon">🚦</div>
                    <div class="feature-title">{home_signal_simulation}</div>
                    <div class="feature-description">{home_signal_simulation_desc}</div>
                </div>
            </div>
            """.format(
                home_signal_simulation=lang_mapping[st.session_state.lang]["home_signal_simulation"],
                home_signal_simulation_desc=lang_mapping[st.session_state.lang]["home_signal_simulation_desc"]
            ),
            unsafe_allow_html=True
        )
        if st.button("RUN SIMULATION", key="signal_btn"):
            st.session_state.page = 'smart_signal'
    
    # Another road animation
    st.markdown(
        """
        <div class="traffic-line"></div>
        """,
        unsafe_allow_html=True
    )
    
    # Dashboard preview section (optional)
    st.markdown(
        """
        <div style="text-align: center; margin: 40px 0 20px 0;">
            <h2 style="color: #1e3c72; font-size: 28px; margin-bottom: 15px;">{home_dashboard_preview}</h2>
            <p style="color: #555; max-width: 800px; margin: 0 auto; font-size: 16px; line-height: 1.6;">
                {home_dashboard_desc}
            </p>
        </div>
        """.format(
            home_dashboard_preview=lang_mapping[st.session_state.lang]["home_dashboard_preview"],
            home_dashboard_desc=lang_mapping[st.session_state.lang]["home_dashboard_desc"]
        ),
        unsafe_allow_html=True
    )
    
    # Footer section
    st.markdown(
        """
        <div class="footer">
            <p style="margin: 0; color: #666; font-size: 13px;">{home_footer}</p>
        </div>
        """.format(
            home_footer=lang_mapping[st.session_state.lang]["home_footer"]
        )
        ,
        unsafe_allow_html=True
    )
    
else:
    # Simulate the redirection by running the corresponding Python file
    try:
        if st.session_state.page == 'traffic_video':
            exec(open(r"Traffic_Video.py", encoding="utf-8").read())
        elif st.session_state.page == 'route_optimize_predictor':
            exec(open(r"predictive_analysis_route_analysis.py", encoding="utf-8").read())
        elif st.session_state.page == 'smart_signal':
            exec(open(r"SignalSimulation.py", encoding="utf-8").read())
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.button("Return to Home", on_click=lambda: setattr(st.session_state, 'page', 'home'))
