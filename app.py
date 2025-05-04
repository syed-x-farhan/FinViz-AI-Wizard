import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score
import io
import base64
import json

# Set page configuration
st.set_page_config(
    page_title="FinViz AI Wizard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light mode CSS with improved aesthetics
st.markdown("""
    <style>
    /* ===== GLOBAL STYLES ===== */
    html, body, .stApp, .main .block-container {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* ===== HEADER ===== */
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e0e0e0 !important;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0 !important;
    }
    
    /* ===== MAIN CONTENT ===== */
    /* Text elements */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* ===== INPUT WIDGETS ===== */
    /* Text input */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Selectbox */
    .stSelectbox select {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #ced4da !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #ffffff !important;
        border: 1px dashed #adb5bd !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3498db !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .stButton button:disabled {
        background-color: #bdc3c7 !important;
        color: #7f8c8d !important;
    }
    
    /* ===== DATA DISPLAY ===== */
    /* Dataframes */
    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* ===== EXPANDERS ===== */
    .stExpander {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e0e0e0 !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background-color: #3498db !important;
    }
    
    /* ===== RADIO BUTTONS & CHECKBOXES ===== */
    .stRadio label, .stCheckbox label {
        color: #333333 !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* ===== CUSTOM STYLES ===== */
    .dataset-option {
        margin: 16px 0 !important;
    }
    
    .step-button {
        background-color: #ffffff;
        color: #3498db;
        border: 2px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .step-button.active {
        background-color: #3498db;
        color: #ffffff;
    }
    .step-label {
        color: #333333;
        margin-left: 10px;
    }
    .step-label.active {
        color: #3498db;
        font-weight: 600;
    }
    
    .main-title {
        color: #2c3e50 !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }
    
    .dataset-container {
        margin: 20px 0;
    }
    
    .next-step-button {
        margin-top: 30px;
        text-align: center;
    }

    .metric-card {
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3498db;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }

    .feature-card {
        margin: 10px 0;
    }

    .model-card {
        margin: 15px 0;
    }
    
    .model-card:hover {
        border-color: #3498db;
    }

    .model-card.selected {
        border: 2px solid #3498db;
    }

    .result-container {
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0  # Show welcome screen on first load
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data_name' not in st.session_state:
    st.session_state.data_name = None

def create_sidebar():
    st.sidebar.markdown("""
        <style>
        .sidebar-title {
            color: #2c3e50 !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
        }
        </style>
        <div class="sidebar-title">Pipeline Steps</div>
    """, unsafe_allow_html=True)
    
    steps = [
        "Data Collection",
        "Data Preprocessing",
        "Feature Engineering",
        "Model Selection",
        "Training & Evaluation",
        "Results Visualization"
    ]
    
    for i, step in enumerate(steps, 1):
        is_active = i == st.session_state.current_step
        st.sidebar.markdown(f"""
            <div style="display: flex; align-items: center; margin: 12px 0;">
                <div class="step-button {'active' if is_active else ''}">{i}</div>
                <div class="step-label {'active' if is_active else ''}">{step}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Add reset button at the bottom of sidebar
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    if st.sidebar.button("Reset Application", key="reset_app"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

def fetch_yfinance_data(ticker, period="1y", interval="1d"):
    try:
        with st.spinner(f"Fetching data for {ticker}..."):
            time.sleep(1)  # Respect rate limits
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                st.error(f"No data found for ticker {ticker}")
                return None
            data.reset_index(inplace=True)  # Convert index to column
            return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def display_dataset_and_next_button(data, source_type):
    """Display dataset preview and next step button"""
    st.markdown("<div class='dataset-container'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #2c3e50;'>Dataset Preview ({source_type})</h3>", unsafe_allow_html=True)
    
    # Show first 10 rows of the dataset
    st.dataframe(data.head(10), use_container_width=True)
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<p><strong>Rows:</strong> {data.shape[0]}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p><strong>Columns:</strong> {data.shape[1]}</p>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<p><strong>Data Range:</strong> {data.index.min().strftime('%Y-%m-%d') if hasattr(data.index.min(), 'strftime') else 'N/A'} to {data.index.max().strftime('%Y-%m-%d') if hasattr(data.index.max(), 'strftime') else 'N/A'}</p>", unsafe_allow_html=True)
    
    # Basic data visualization
    if source_type.startswith("Yahoo Finance"):
        try:
            # Plot closing prices
            st.subheader("Closing Price")
            fig = px.line(data, x='Date', y='Close', title=f'{source_type.split(":")[1].strip()} Stock Price')
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot volume
            st.subheader("Trading Volume")
            fig = px.bar(data, x='Date', y='Volume', title=f'{source_type.split(":")[1].strip()} Trading Volume')
            fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate visualization: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add next step button
    st.markdown("<div class='next-step-button'>", unsafe_allow_html=True)
    if st.button("Proceed to Data Preprocessing →", key="next_step_button"):
        st.session_state.current_step = 2
        st.markdown("""
            <div style='text-align: center; margin-top: 20px;'>
                <img src='https://media.giphy.com/media/3o7buirY0g0g0g0g0g/giphy.gif' style='width: 300px; border-radius: 10px;'/>
                <p style='color: #7f8c8d;'>Preparing your data for analysis...</p>
            </div>
        """, unsafe_allow_html=True)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def dataset_selection():
    st.empty()  # Clear previous content
    st.markdown("<h2 class='main-title'>Choose Your Data Source</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='dataset-option'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #2c3e50;'>Yahoo Finance</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d;'>Fetch real-time financial data from Yahoo Finance</p>", unsafe_allow_html=True)
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", key="ticker_input")
        period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], key="period_select")
        interval = st.selectbox("Select Interval", ["1d", "5d", "1wk", "1mo"], key="interval_select")
        if st.button("Fetch Data", key="fetch_yfinance"):
            if ticker:
                with st.spinner("Fetching data from Yahoo Finance..."):
                    data = fetch_yfinance_data(ticker, period=period, interval=interval)
                    if data is not None:
                        st.session_state.dataset = data
                        st.session_state.data_source = "yfinance"
                        st.session_state.data_name = f"{ticker}"
                        st.success(f"Data for {ticker} fetched successfully!")
                        display_dataset_and_next_button(data, f"Yahoo Finance: {ticker}")
            else:
                st.warning("Please enter a ticker symbol")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='dataset-option'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #2c3e50;'>Upload Your Dataset</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d;'>Upload your own financial dataset (Kragle or custom) in CSV or Excel format</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.dataset = data
                st.session_state.data_source = "upload"
                st.session_state.data_name = uploaded_file.name
                st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
                display_dataset_and_next_button(data, f"Uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

def welcome_screen():
    st.markdown("""
        <style>
        .welcome-container {
            text-align: center;
            padding-top: 40px;
        }
        .welcome-title {
            color: #2c3e50;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .welcome-subtitle {
            color: #7f8c8d;
            font-size: 1.2rem;
            margin-bottom: 1.5rem;
        }
        .start-button {
            background: #3498db !important;
            color: white !important;
            font-size: 1.2rem;
            padding: 12px 24px;
            border-radius: 8px;
            margin-top: 2rem;
            transition: all 0.3s ease;
        }
        .start-button:hover {
            background: #2980b9 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
        <div class='welcome-container'>
            <img src='https://img.icons8.com/color/96/000000/combo-chart--v1.png' width='80'/>
            <div class='welcome-title'>Welcome to FinViz AI Wizard</div>
            <div class='welcome-subtitle'>Explore financial data through machine learning with this interactive pipeline tool.</div>
            <div class='welcome-subtitle'>Upload datasets, analyze trends, and build predictive models - all in one place.</div>
            <img src='https://media.giphy.com/media/L1R1tvI9svkIWwpVYr/giphy.gif' style='width:400px; border-radius:16px; margin:1.5rem 0;'/>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started →", key="start_button", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

def data_preprocessing():
    st.empty()  # Clear previous content
    st.markdown("<h2 class='main-title'>Data Preprocessing</h2>", unsafe_allow_html=True)
    
    if st.session_state.dataset is not None:
        data = st.session_state.dataset.copy()
        
        st.subheader("Original Dataset Preview")
        st.dataframe(data.head(5), use_container_width=True)
        
        # Display missing values
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Missing Values")
            if total_missing > 0:
                st.markdown(f"<p>Total missing values: <strong>{total_missing}</strong></p>", unsafe_allow_html=True)
                st.dataframe(missing_values[missing_values > 0], use_container_width=True)
                
                # Create pie chart for missing values
                missing_data = pd.DataFrame({
                    'Status': ['Missing', 'Present'],
                    'Count': [total_missing, data.size - total_missing]
                })
                fig = px.pie(missing_data, values='Count', names='Status', 
                            title='Missing vs Present Values', 
                            color_discrete_sequence=['#ff6b6b', '#48dbfb'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in the dataset!")
        
        with col2:
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame({'Data Type': data.dtypes}), use_container_width=True)
            
            # Display basic statistics
            st.subheader("Summary Statistics")
            numeric_cols = data.select_dtypes(include=[np.number])
            if not numeric_cols.empty:
                st.dataframe(numeric_cols.describe(), use_container_width=True)
            else:
                st.warning("No numeric columns found for statistics.")
        
        # Preprocessing options
        st.subheader("Preprocessing Options")
        
        # Get numeric columns for preprocessing
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns found in the dataset. Preprocessing options are limited.")
            include_columns = []
        else:
            include_columns = st.multiselect(
                "Select columns to include (leave empty to include all numeric columns)",
                options=numeric_columns,
                default=numeric_columns
            )
        
        col1, col2 = st.columns(2)
        with col1:
            handle_missing = st.radio(
                "Handle Missing Values",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with zero", "None"],
                key="missing_option"
            )
        
        with col2:
            scaling_method = st.radio(
                "Data Scaling",
                ["StandardScaler", "MinMaxScaler", "None"],
                key="scaling_option"
            )
        
        # Process data button
        if st.button("Apply Preprocessing", key="process_data_button"):
            with st.spinner("Preprocessing data..."):
                try:
                    # If no columns specified, use all numeric columns
                    if not include_columns and numeric_columns:
                        processed_data = data[numeric_columns]
                        col_list = numeric_columns
                    elif include_columns:
                        processed_data = data[include_columns]
                        col_list = include_columns
                    else:
                        st.error("No numeric columns selected for preprocessing.")
                        return
                    
                    # Handle missing values
                    if handle_missing == "Drop rows":
                        processed_data = processed_data.dropna()
                        st.info(f"Dropped {len(data) - len(processed_data)} rows with missing values.")
                    elif handle_missing == "Fill with mean":
                        for col in col_list:
                            if processed_data[col].dtype in [np.float64, np.int64]:
                                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
                        st.info("Filled missing values with column means.")
                    elif handle_missing == "Fill with median":
                        for col in col_list:
                            if processed_data[col].dtype in [np.float64, np.int64]:
                                processed_data[col].fillna(processed_data[col].median(), inplace=True)
                        st.info("Filled missing values with column medians.")
                    elif handle_missing == "Fill with zero":
                        processed_data.fillna(0, inplace=True)
                        st.info("Filled missing values with zeros.")
                    
                    # Apply scaling
                    if scaling_method != "None" and not processed_data.empty:
                        try:
                            scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
                            scaled_data = scaler.fit_transform(processed_data)
                            processed_data = pd.DataFrame(scaled_data, columns=processed_data.columns)
                            st.info(f"Applied {scaling_method} to the data.")
                        except Exception as e:
                            st.error(f"Error during scaling: {str(e)}")
                            return
                    
                    # Store processed data
                    st.session_state.processed_data = processed_data
                    
                    # Display processed data
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_data.head(5), use_container_width=True)
                    
                    # Compare original vs processed data
                    if scaling_method != "None" and not processed_data.empty:
                        try:
                            # Check if we have at least two columns to plot
                            if len(processed_data.columns) >= 2:
                                # Plot first two columns before and after preprocessing
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Before Preprocessing")
                                    orig_data = data[processed_data.columns[:2]]
                                    fig = px.scatter(orig_data, x=orig_data.columns[0], y=orig_data.columns[1], 
                                                    title="Original Data (First 2 Features)")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.subheader("After Preprocessing")
                                    fig = px.scatter(processed_data, x=processed_data.columns[0], y=processed_data.columns[1], 
                                                    title="Processed Data (First 2 Features)")
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                # If only one column, plot single feature
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Before Preprocessing")
                                    orig_data = data[processed_data.columns[0]]
                                    fig = px.histogram(orig_data, title="Original Data Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.subheader("After Preprocessing")
                                    fig = px.histogram(processed_data[processed_data.columns[0]], 
                                                    title="Processed Data Distribution")
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate comparison plots: {str(e)}")
                    
                    st.success("Preprocessing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
                    return
        
        # Next step button (only show if data has been processed)
        if st.session_state.processed_data is not None:
            if st.button("Proceed to Feature Engineering →", key="next_step_preprocessing"):
                st.session_state.current_step = 3
                st.markdown("""
                    <div style='text-align: center; margin-top: 20px;'>
                        <img src='https://media.giphy.com/media/3o7buirY0g0g0g0g0g/giphy.gif' style='width: 300px; border-radius: 10px;'/>
                        <p style='color: #7f8c8d;'>Engineering features for better insights...</p>
                    </div>
                """, unsafe_allow_html=True)
                st.rerun()
    else:
        st.warning("No dataset loaded. Please go back to Data Collection.")
        if st.button("← Go to Data Collection", key="back_to_data"):
            st.session_state.current_step = 1
            st.rerun()

def feature_engineering():
    st.empty()  # Clear previous content
    st.markdown("<h2 class='main-title'>Feature Engineering</h2>", unsafe_allow_html=True)
    
    if st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data.copy()
        
        st.subheader("Processed Data")
        st.dataframe(processed_data.head(5), use_container_width=True)
        
        # Features and target selection
        st.subheader("Select Features and Target")
        
        all_columns = processed_data.columns.tolist()
        
        if len(all_columns) < 2:
            st.error("Not enough columns for modeling. Please go back and include more features.")
            if st.button("← Go back to Preprocessing", key="back_to_preprocessing"):
                st.session_state.current_step = 2
                st.rerun()
            return
        
        # Target selection
        target_col = st.selectbox(
            "Select Target Variable",
            options=all_columns,
            index=0,
            key="target_selection"
        )
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Feature Variables",
            options=[col for col in all_columns if col != target_col],
            default=[col for col in all_columns if col != target_col],
            key="feature_selection"
        )
        
        if len(feature_cols) == 0:
            st.warning("Please select at least one feature.")
        else:
            # Feature engineering options
            st.subheader("Feature Engineering Options")
            
            st.markdown("### Add New Features")
            
            # Option to create lag features for time series data
            if st.session_state.data_source == "yfinance":
                create_lag = st.checkbox("Create lag features (for time series prediction)", key="create_lag")
                if create_lag:
                    lag_periods = st.slider("Number of lag periods", min_value=1, max_value=10, value=3, key="lag_periods")
            
            # Option to create polynomial features
            create_poly = st.checkbox("Create polynomial features", key="create_poly")
            if create_poly:
                poly_degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=2, key="poly_degree")
            
            # Option to create technical indicators for financial data
            if st.session_state.data_source == "yfinance":
                create_tech = st.checkbox("Create technical indicators (MA, RSI, etc.)", key="create_tech")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Apply Feature Engineering button
            if st.button("Apply Feature Engineering", key="apply_fe_button"):
                if len(feature_cols) > 0:
                    with st.spinner("Applying feature engineering..."):
                        X = processed_data[feature_cols].copy()
                        y = processed_data[target_col].copy()
                        
                        # Store original features
                        original_cols = X.columns.tolist()
                        
                        # Apply lag features if selected
                        if st.session_state.data_source == "yfinance" and create_lag:
                            for col in original_cols:
                                for i in range(1, lag_periods + 1):
                                    X[f"{col}_lag_{i}"] = X[col].shift(i)
                            # Drop rows with NaN from lag creation
                            X.dropna(inplace=True)
                            y = y.iloc[lag_periods:].reset_index(drop=True)
                            st.info(f"Created {lag_periods} lag features for each original feature.")
                        
                        # Apply polynomial features if selected
                        if create_poly:
                            from sklearn.preprocessing import PolynomialFeatures
                            if len(X) > 0:  # Check if X is not empty after lag feature creation
                                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                                poly_features = poly.fit_transform(X[original_cols])
                                poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
                                poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
                                
                                # Join polynomial features with original dataframe
                                X = X.reset_index(drop=True)
                                X = pd.concat([X, poly_df], axis=1)
                                st.info(f"Created {poly_features.shape[1]} polynomial features.")
                        
                        # Apply technical indicators if selected
                        if st.session_state.data_source == "yfinance" and create_tech:
                            # Get original data to calculate indicators
                            orig_data = st.session_state.dataset.copy()
                            
                            # Simple Moving Average (SMA)
                            if 'Close' in orig_data.columns:
                                # 20-day SMA
                                orig_data['SMA_20'] = orig_data['Close'].rolling(window=20).mean()
                                # 50-day SMA
                                orig_data['SMA_50'] = orig_data['Close'].rolling(window=50).mean()
                                
                                # Add SMA features to X
                                if len(orig_data) > 50:  # Need at least 50 days for 50-day SMA
                                    # Align indices
                                    if len(X) <= len(orig_data[['SMA_20', 'SMA_50']].dropna()):
                                        sma_df = orig_data[['SMA_20', 'SMA_50']].iloc[-len(X):].reset_index(drop=True)
                                        X = pd.concat([X, sma_df], axis=1)
                                        st.info("Added SMA technical indicators.")
                                    else:
                                        st.warning("Could not add SMA indicators due to data length mismatch.")
                        
                        # Store features and target in session state
                        st.session_state.features = X
                        st.session_state.target = y
                        
                        # Display engineered features
                        st.subheader("Engineered Features Preview")
                        st.dataframe(X.head(5), use_container_width=True)
                        
                        # Feature information
                        st.metric("Total Features", X.shape[1])
                        
                        # Visualize feature correlations
                        st.subheader("Feature Correlation Heatmap")
                        corr = X.corr()
                        fig = px.imshow(corr, 
                                      color_continuous_scale='RdBu_r', 
                                      title="Feature Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualize feature importance if not too many features
                        if X.shape[1] < 20:
                            st.subheader("Feature Distribution")
                            selected_feat = st.selectbox("Select feature to visualize:", X.columns)
                            fig = px.histogram(X, x=selected_feat, 
                                             title=f"Distribution of {selected_feat}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("Feature engineering applied successfully!")
                        
                        # Show relationship with target
                        st.subheader("Feature vs Target")
                        if X.shape[1] < 20:
                            selected_feat = st.selectbox("Select feature:", X.columns, key="feat_target_viz")
                            fig = px.scatter(x=X[selected_feat], y=y, 
                                           title=f"{selected_feat} vs {target_col}",
                                           labels={"x": selected_feat, "y": target_col})
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select at least one feature.")
            
            # Next step button (only show if features have been created)
            if st.session_state.features is not None:
                st.markdown("<div class='next-step-button'>", unsafe_allow_html=True)
                if st.button("Proceed to Model Selection →", key="next_step_fe"):
                    st.session_state.current_step = 4
                    st.markdown("""
                        <div style='text-align: center; margin-top: 20px;'>
                            <img src='https://media.giphy.com/media/3o7buirY0g0g0g0g0g/giphy.gif' style='width: 300px; border-radius: 10px;'/>
                            <p style='color: #7f8c8d;'>Selecting the best model for your data...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        
    else:
        st.warning("No processed data available. Please complete the Data Preprocessing step first.")
        if st.button("← Go to Data Preprocessing", key="back_to_preprocessing"):
            st.session_state.current_step = 2
            st.rerun()

def model_selection():
    st.empty()  # Clear previous content
    st.markdown("<h2 class='main-title'>Model Selection & Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.features is not None and st.session_state.target is not None:
        X = st.session_state.features
        y = st.session_state.target
        
        st.metric("Features Shape", f"{X.shape[0]} rows, {X.shape[1]} columns")
        st.metric("Target Shape", f"{len(y)} values")
        
        # Test-train split parameters
        st.subheader("Train-Test Split")
        test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, key="test_size") / 100
        
        random_state = st.checkbox("Use Random State (for reproducibility)", value=True, key="random_state_checkbox")
        if random_state:
            random_seed = st.number_input("Random Seed", min_value=0, max_value=1000, value=42, key="random_seed")
        else:
            random_seed = None
        
        # Split button
        if st.button("Split Data", key="split_button"):
            with st.spinner("Splitting data into training and testing sets..."):
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed
                )
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Display split information
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{X_train.shape[0]}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Training Samples</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{X_test.shape[0]}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Testing Samples</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualize the split
                split_df = pd.DataFrame({
                    'Set': ['Training', 'Testing'],
                    'Count': [X_train.shape[0], X_test.shape[0]]
                })
                
                fig = px.pie(split_df, values='Count', names='Set', 
                           title='Train-Test Split',
                           color_discrete_sequence=['#3498db', '#f39c12'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Data split successfully!")
        
        # Only show model selection if data has been split
        if st.session_state.X_train is not None:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Choose a Machine Learning Model")
            
            # Determine if regression or classification based on target values
            unique_values = len(np.unique(st.session_state.target))
            is_binary = unique_values == 2
            
            if unique_values <= 10:  # Likely classification
                st.markdown(f"Target appears to be a **classification** problem with {unique_values} unique classes.")
                problem_type = "classification"
            else:  # Likely regression
                st.markdown("Target appears to be a **regression** problem.")
                problem_type = "regression"
            
            # Model selection
            model_options = []
            if problem_type == "regression":
                model_options.extend(["Linear Regression"])
            if problem_type == "classification" and is_binary:
                model_options.extend(["Logistic Regression"])
            model_options.extend(["K-Means Clustering"])  # Clustering doesn't need target
            
            # Model cards
            st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 20px;'>", unsafe_allow_html=True)
            
            selected_model = None
            
            # Linear Regression Card
            if "Linear Regression" in model_options:
                st.markdown(f"""
                <div class="model-card {'selected' if st.session_state.model_type == 'Linear Regression' else ''}">
                    <h3>Linear Regression</h3>
                    <p>A linear approach to modeling the relationship between a dependent variable and one or more independent variables.</p>
                    <p><strong>Good for:</strong> Predicting numerical values, finding relationships between variables.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select Linear Regression", key="select_linear"):
                    st.session_state.model_type = "Linear Regression"
                    selected_model = "Linear Regression"
                    st.rerun()
            
            # Logistic Regression Card
            if "Logistic Regression" in model_options:
                st.markdown(f"""
                <div class="model-card {'selected' if st.session_state.model_type == 'Logistic Regression' else ''}">
                    <h3>Logistic Regression</h3>
                    <p>A statistical model that uses a logistic function to model a binary dependent variable.</p>
                    <p><strong>Good for:</strong> Binary classification, probability estimation.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select Logistic Regression", key="select_logistic"):
                    st.session_state.model_type = "Logistic Regression"
                    selected_model = "Logistic Regression"
                    st.rerun()
            
            # K-Means Card
            st.markdown(f"""
            <div class="model-card {'selected' if st.session_state.model_type == 'K-Means Clustering' else ''}">
                <h3>K-Means Clustering</h3>
                <p>An unsupervised learning algorithm that groups similar data points into clusters.</p>
                <p><strong>Good for:</strong> Identifying patterns, segmentation, anomaly detection.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select K-Means Clustering", key="select_kmeans"):
                st.session_state.model_type = "K-Means Clustering"
                selected_model = "K-Means Clustering"
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show hyperparameters for selected model
            if st.session_state.model_type is not None:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader(f"Configure {st.session_state.model_type}")
                
                if st.session_state.model_type == "Linear Regression":
                    fit_intercept = st.checkbox("Fit Intercept", value=True, key="lr_fit_intercept")
                    normalize = st.checkbox("Normalize", value=False, key="lr_normalize")
                    
                    # Train model button
                    if st.button("Train Model", key="train_lr_button"):
                        with st.spinner("Training Linear Regression model..."):
                            # Initialize model
                            model = LinearRegression(fit_intercept=fit_intercept)
                            
                            # Train model
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            
                            # Make predictions
                            y_pred = model.predict(st.session_state.X_test)
                            
                            # Calculate metrics
                            mse = mean_squared_error(st.session_state.y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(st.session_state.y_test, y_pred)
                            
                            # Store model and metrics
                            st.session_state.model = model
                            st.session_state.predictions = y_pred
                            st.session_state.evaluation_metrics = {
                                "MSE": mse,
                                "RMSE": rmse,
                                "R²": r2
                            }
                            
                            st.success("Linear Regression model trained successfully!")
                
                elif st.session_state.model_type == "Logistic Regression":
                    C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, key="lr_C")
                    max_iter = st.slider("Maximum iterations", 100, 1000, 100, key="lr_max_iter")
                    
                    # Train model button
                    if st.button("Train Model", key="train_logistic_button"):
                        with st.spinner("Training Logistic Regression model..."):
                            # Initialize model
                            model = LogisticRegression(C=C, max_iter=max_iter)
                            
                            # Train model
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            
                            # Make predictions
                            y_pred = model.predict(st.session_state.X_test)
                            y_prob = model.predict_proba(st.session_state.X_test)[:, 1]
                            
                            # Calculate metrics
                            accuracy = accuracy_score(st.session_state.y_test, y_pred)
                            conf_matrix = confusion_matrix(st.session_state.y_test, y_pred)
                            
                            # Store model and metrics
                            st.session_state.model = model
                            st.session_state.predictions = y_pred
                            st.session_state.evaluation_metrics = {
                                "Accuracy": accuracy,
                                "Confusion Matrix": conf_matrix
                            }
                            
                            st.success("Logistic Regression model trained successfully!")
                
                elif st.session_state.model_type == "K-Means Clustering":
                    n_clusters = st.slider("Number of clusters", 2, 10, 3, key="kmeans_n_clusters")
                    init_method = st.selectbox("Initialization method", ["k-means++", "random"], key="kmeans_init")
                    n_init = st.slider("Number of initializations", 1, 20, 10, key="kmeans_n_init")
                    
                    # Train model button
                    if st.button("Train Model", key="train_kmeans_button"):
                        with st.spinner("Training K-Means Clustering model..."):
                            # Initialize model
                            model = KMeans(n_clusters=n_clusters, init=init_method, n_init=n_init, random_state=random_seed)
                            
                            # Use only the features, no target for clustering
                            X_combined = pd.concat([st.session_state.X_train, st.session_state.X_test])
                            
                            # Train model
                            model.fit(X_combined)
                            
                            # Get cluster labels
                            labels = model.labels_
                            
                            # Calculate silhouette score
                            silhouette = silhouette_score(X_combined, labels) if len(np.unique(labels)) > 1 else 0
                            
                            # Store model and metrics
                            st.session_state.model = model
                            st.session_state.predictions = labels
                            st.session_state.evaluation_metrics = {
                                "Silhouette Score": silhouette,
                                "Inertia": model.inertia_
                            }
                            
                            st.success("K-Means Clustering model trained successfully!")
            
            # Next step button (only show if model has been trained)
            if st.session_state.model is not None:
                st.markdown("<div class='next-step-button'>", unsafe_allow_html=True)
                if st.button("Proceed to Evaluation & Results →", key="next_step_model"):
                    st.session_state.current_step = 5
                    st.markdown("""
                        <div style='text-align: center; margin-top: 20px;'>
                            <img src='https://media.giphy.com/media/3o7buirY0g0g0g0g0g/giphy.gif' style='width: 300px; border-radius: 10px;'/>
                            <p style='color: #7f8c8d;'>Analyzing results and generating insights...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.warning("Feature engineering has not been completed. Please return to the previous step.")
        if st.button("← Go to Feature Engineering", key="back_to_fe"):
            st.session_state.current_step = 3
            st.rerun()

def results_visualization():
    st.empty()  # Clear previous content
    st.markdown("<h2 class='main-title'>Model Evaluation & Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is not None and st.session_state.evaluation_metrics is not None:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # Display evaluation metrics
        st.subheader("Model Performance Metrics")
        
        metrics = st.session_state.evaluation_metrics
        
        # Create a visually appealing metrics display
        if st.session_state.model_type in ["Linear Regression"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.4f}")
            with col3:
                st.metric("R-squared", f"{metrics['R²']:.4f}")
            
            # Visualize predictions vs actual
            st.subheader("Predictions vs Actual Values")
            
            # Get actual and predicted values
            y_test = st.session_state.y_test
            y_pred = st.session_state.predictions
            
            # Create DataFrame for visualization
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            # Scatter plot
            fig = px.scatter(results_df, x='Actual', y='Predicted', 
                           title='Actual vs Predicted Values',
                           labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
            
            # Add perfect prediction line
            min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
            max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            results_df['Residuals'] = results_df['Actual'] - results_df['Predicted']
            
            fig = px.scatter(results_df, x='Predicted', y='Residuals',
                           title='Residuals Plot',
                           labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'})
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Get model coefficients
            model = st.session_state.model
            feature_names = st.session_state.X_train.columns
            coefficients = model.coef_
            
            # Create coefficient DataFrame
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            # Bar chart of coefficients
            fig = px.bar(coef_df.head(15), x='Feature', y='Coefficient',
                       title='Top 15 Feature Coefficients',
                       color='Coefficient',
                       color_continuous_scale='RdBu_r',
                       labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'})
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif st.session_state.model_type == "Logistic Regression":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = metrics['Confusion Matrix']
            
            # Create labels
            if conf_matrix.shape[0] == 2:
                labels = ['Negative', 'Positive']
            else:
                labels = [f'Class {i}' for i in range(conf_matrix.shape[0])]
            
            # Plot confusion matrix
            fig = px.imshow(conf_matrix,
                          x=labels,
                          y=labels,
                          text_auto=True,
                          color_continuous_scale='Blues',
                          title='Confusion Matrix',
                          labels=dict(x="Predicted Label", y="True Label", color="Count"))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Get model coefficients
            model = st.session_state.model
            feature_names = st.session_state.X_train.columns
            coefficients = model.coef_[0]  # For binary classification
            
            # Create coefficient DataFrame
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            # Bar chart of coefficients
            fig = px.bar(coef_df.head(15), x='Feature', y='Coefficient',
                       title='Top 15 Feature Coefficients',
                       color='Coefficient',
                       color_continuous_scale='RdBu_r',
                       labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'})
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif st.session_state.model_type == "K-Means Clustering":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
            with col2:
                st.metric("Inertia", f"{metrics['Inertia']:.4f}")
            
            # Visualize clusters with PCA if dimensions > 2
            st.subheader("Cluster Visualization")
            
            X_combined = pd.concat([st.session_state.X_train, st.session_state.X_test])
            labels = st.session_state.predictions
            
            if X_combined.shape[1] > 2:
                st.markdown("Using PCA to reduce dimensions for visualization")
                from sklearn.decomposition import PCA
                
                # Apply PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_combined)
                
                # Create a DataFrame with the first two principal components
                pca_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'Cluster': labels
                })
                
                # Scatter plot of clusters
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                               title='Cluster Visualization (PCA)',
                               labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                               color_continuous_scale='viridis')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explained variance
                st.markdown(f"**Explained Variance Ratio:** {pca.explained_variance_ratio_[0]:.4f}, {pca.explained_variance_ratio_[1]:.4f}")
                st.markdown(f"**Total Explained Variance:** {sum(pca.explained_variance_ratio_):.4f}")
            else:
                # If only 2 dimensions, visualize directly
                plot_df = X_combined.copy()
                plot_df['Cluster'] = labels
                
                fig = px.scatter(plot_df, x=plot_df.columns[0], y=plot_df.columns[1], color='Cluster',
                               title='Cluster Visualization',
                               labels={plot_df.columns[0]: plot_df.columns[0], plot_df.columns[1]: plot_df.columns[1]},
                               color_continuous_scale='viridis')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster centers
            st.subheader("Cluster Centers")
            
            # Get cluster centers
            centers = st.session_state.model.cluster_centers_
            
            # Create DataFrame for centers
            centers_df = pd.DataFrame(centers, columns=X_combined.columns)
            
            # Display cluster centers
            st.dataframe(centers_df, use_container_width=True)
            
            # Visualize cluster centers
            if X_combined.shape[1] > 2:
                # Use PCA for visualization
                centers_pca = pca.transform(centers)
                centers_pca_df = pd.DataFrame(centers_pca, columns=['PC1', 'PC2'])
                
                # Add centers to the scatter plot
                fig.add_trace(go.Scatter(
                    x=centers_pca_df['PC1'],
                    y=centers_pca_df['PC2'],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='x'),
                    name='Cluster Centers'
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            # Add cluster labels to the original data
            X_combined['Cluster'] = labels
            
            # Calculate statistics for each cluster
            cluster_stats = X_combined.groupby('Cluster').agg(['mean', 'std']).round(4)
            st.dataframe(cluster_stats, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download results button
        if st.button("Download Results", key="download_results"):
            # Create a dictionary to store all results
            results = {
                'model_type': st.session_state.model_type,
                'metrics': {k: float(v) if isinstance(v, (np.float64, np.int64)) else v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in st.session_state.evaluation_metrics.items()},
                'predictions': st.session_state.predictions.tolist() if isinstance(st.session_state.predictions, np.ndarray) else st.session_state.predictions,
                'features': st.session_state.features.columns.tolist(),
                'target': st.session_state.target.name if hasattr(st.session_state.target, 'name') else 'target'
            }
            
            # Convert to JSON
            results_json = json.dumps(results, indent=4)
            
            # Create download button
            st.download_button(
                label="Download Results as JSON",
                data=results_json,
                file_name="model_results.json",
                mime="application/json"
            )
        
        # Reset button
        if st.button("Start New Analysis", key="reset_analysis"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.markdown("""
                <div style='text-align: center; margin-top: 20px;'>
                    <img src='https://media.giphy.com/media/3o7buirY0g0g0g0g0g/giphy.gif' style='width: 300px; border-radius: 10px;'/>
                    <p style='color: #7f8c8d;'>Starting fresh analysis...</p>
                </div>
            """, unsafe_allow_html=True)
            st.rerun()
    else:
        st.warning("No model has been trained yet. Please complete the previous steps.")
        if st.button("← Go to Model Selection", key="back_to_model"):
            st.session_state.current_step = 4
            st.rerun()

def main():
    """Main application flow"""
    # Create sidebar
    create_sidebar()
    
    # Welcome screen
    if st.session_state.current_step == 0:
        welcome_screen()
    
    # Data collection
    elif st.session_state.current_step == 1:
        dataset_selection()
    
    # Data preprocessing
    elif st.session_state.current_step == 2:
        data_preprocessing()
    
    # Feature engineering
    elif st.session_state.current_step == 3:
        feature_engineering()
    
    # Model selection and training
    elif st.session_state.current_step == 4:
        model_selection()
    
    # Results visualization
    elif st.session_state.current_step == 5:
        results_visualization()

if __name__ == "__main__":
    main()