# ================ 0. REQUIRED LIBRARIES ================

import streamlit as st
import pandas as pd
import talib # Replace pandas_ta with talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
import pytz
import feedparser
from email.utils import mktime_tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
import time as a_time
import re
import yfinance as yf
import pyotp
import qrcode
from PIL import Image
import base64
import io
import requests
import hashlib
import random
import logging
import traceback

# ================ 0.5 ERROR HANDLING SETUP ================

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_error(error, context="", user_message=None):
    """
    Centralized error handling function.
    
    Args:
        error: The exception object
        context: Where the error occurred
        user_message: Friendly message to show user
    """
    # Log the error
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(traceback.format_exc())
    
    # Show user-friendly message
    if user_message:
        st.error(user_message)
    else:
        # Generic error messages based on context
        if "broker" in context.lower():
            st.error("⚠️ Broker connection issue. Please check your connection and try again.")
        elif "data" in context.lower():
            st.error("📊 Data loading failed. The service might be temporarily unavailable.")
        elif "order" in context.lower():
            st.error("❌ Order placement failed. Please check the details and try again.")
        elif "calculation" in context.lower():
            st.error("🧮 Calculation error. Please try with different parameters.")
        else:
            st.error("🔧 An unexpected error occurred. Please try again.")
    
    # Show detailed error in expander for debugging
    with st.expander("Technical Details (for support)"):
        st.code(f"Context: {context}\nError: {str(error)}\n\n{traceback.format_exc()}")

def safe_execute(func, context="", default_return=None, user_message=None):
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        context: Description of what the function does
        default_return: What to return if function fails
        user_message: Custom user message on error
    """
    try:
        return func()
    except Exception as e:
        handle_error(e, context, user_message)
        return default_return

# ================ 1. STYLING AND CONFIGURATION ===============

st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def apply_custom_styling():
    """Applies a comprehensive CSS stylesheet for professional theming with proper light mode support."""
    try:
        theme_css = """
        <style>
            :root {
                --dark-bg: #0E1117;
                --dark-secondary-bg: #161B22;
                --dark-widget-bg: #21262D;
                --dark-border: #30363D;
                --dark-text: #c9d1d9;
                --dark-text-light: #8b949e;
                --dark-green: #28a745;
                --dark-red: #da3633;
                --dark-orange: #d29922;
                --dark-blue: #58a6ff;
                --dark-purple: #bc8cff;

                --light-bg: #FFFFFF;
                --light-secondary-bg: #F8F9FA;
                --light-widget-bg: #F0F2F6;
                --light-border: #dee2e6;
                --light-text: #212529;
                --light-text-light: #6c757d;
                --light-green: #198754;
                --light-red: #dc3545;
                --light-orange: #fd7e14;
                --light-blue: #0d6efd;
                --light-purple: #6f42c1;
            }

            body.dark-theme {
                --primary-bg: var(--dark-bg);
                --secondary-bg: var(--dark-secondary-bg);
                --widget-bg: var(--dark-widget-bg);
                --border-color: var(--dark-border);
                --text-color: var(--dark-text);
                --text-light: var(--dark-text-light);
                --green: var(--dark-green);
                --red: var(--dark-red);
                --orange: var(--dark-orange);
                --blue: var(--dark-blue);
                --purple: var(--dark-purple);
            }

            body.light-theme {
                --primary-bg: var(--light-bg);
                --secondary-bg: var(--light-secondary-bg);
                --widget-bg: var(--light-widget-bg);
                --border-color: var(--light-border);
                --text-color: var(--light-text);
                --text-light: var(--light-text-light);
                --green: var(--light-green);
                --red: var(--light-red);
                --orange: var(--light-orange);
                --blue: var(--light-blue);
                --purple: var(--light-purple);
            }

            body {
                background-color: var(--primary-bg);
                color: var(--text-color);
                transition: all 0.3s ease;
            }
            
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                background-color: var(--primary-bg);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-color) !important;
            }
            
            p, div, span {
                color: var(--text-color);
            }
            
            hr {
                background: var(--border-color);
                margin: 1rem 0;
            }

            /* Streamlit component styling */
            .stButton>button {
                border: 1px solid var(--border-color);
                background-color: var(--widget-bg);
                color: var(--text-color);
                transition: all 0.2s ease;
            }
            .stButton>button:hover {
                border-color: var(--green);
                color: var(--green);
                background-color: var(--secondary-bg);
            }
            
            .stTextInput>div>div>input, 
            .stNumberInput>div>div>input, 
            .stSelectbox>div>div>select,
            .stTextArea>div>div>textarea {
                background-color: var(--widget-bg);
                border: 1px solid var(--border-color);
                color: var(--text-color);
                border-radius: 4px;
            }
            
            .stTextInput>div>div>input:focus, 
            .stNumberInput>div>div>input:focus, 
            .stSelectbox>div>div>select:focus {
                border-color: var(--blue);
                box-shadow: 0 0 0 1px var(--blue);
            }
            
            .stRadio>div {
                background-color: var(--widget-bg);
                border: 1px solid var(--border-color);
                padding: 8px;
                border-radius: 8px;
            }
            
            .stCheckbox>div {
                color: var(--text-color);
            }
            
            .stCheckbox>div>label>div:first-child {
                background-color: var(--widget-bg);
                border: 1px solid var(--border-color);
            }
            
            .stSlider>div>div>div {
                background-color: var(--border-color);
            }
            
            .stSlider>div>div>div>div {
                background-color: var(--green);
            }
            
            /* Dataframe styling */
            .dataframe {
                background-color: var(--widget-bg) !important;
                color: var(--text-color) !important;
            }
            
            .dataframe th {
                background-color: var(--secondary-bg) !important;
                color: var(--text-color) !important;
                border-color: var(--border-color) !important;
            }
            
            .dataframe td {
                background-color: var(--widget-bg) !important;
                color: var(--text-color) !important;
                border-color: var(--border-color) !important;
            }
            
            /* Metric cards */
            .metric-card {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                padding: 1.5rem;
                border-radius: 10px;
                border-left-width: 5px;
                color: var(--text-color);
            }
            
            .trade-card {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                padding: 1.5rem;
                border-radius: 10px;
                border-left-width: 5px;
                color: var(--text-color);
            }

            /* Notification bar */
            .notification-bar {
                position: sticky;
                top: 0;
                width: 100%;
                background-color: var(--secondary-bg);
                color: var(--text-color);
                padding: 8px 12px;
                z-index: 999;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 0.9rem;
                border-bottom: 1px solid var(--border-color);
                margin-left: -20px;
                margin-right: -20px;
                width: calc(100% + 40px);
            }
            .notification-bar span {
                margin: 0 15px;
                white-space: nowrap;
            }
            
            /* HFT Terminal */
            .hft-depth-bid {
                background: linear-gradient(to left, rgba(0, 128, 0, 0.3), rgba(0, 128, 0, 0.05));
                padding: 2px 5px;
                border-radius: 3px;
                margin: 2px 0;
            }
            .hft-depth-ask {
                background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05));
                padding: 2px 5px;
                border-radius: 3px;
                margin: 2px 0;
            }
            .tick-up {
                color: var(--green);
                animation: flash-green 0.5s;
            }
            .tick-down {
                color: var(--red);
                animation: flash-red 0.5s;
            }
            @keyframes flash-green {
                0% { background-color: rgba(40, 167, 69, 0.3); }
                100% { background-color: transparent; }
            }
            @keyframes flash-red {
                0% { background-color: rgba(218, 54, 51, 0.3); }
                100% { background-color: transparent; }
            }
            
            /* Plotly chart containers */
            .js-plotly-plot .plotly, 
            .js-plotly-plot .plotly div {
                background-color: var(--widget-bg) !important;
            }
            
            /* Streamlit expander */
            .streamlit-expanderHeader {
                background-color: var(--secondary-bg);
                color: var(--text-color);
                border: 1px solid var(--border-color);
            }
            
            .streamlit-expanderContent {
                background-color: var(--widget-bg);
                border: 1px solid var(--border-color);
                border-top: none;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: var(--secondary-bg);
                border-bottom: 1px solid var(--border-color);
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: var(--secondary-bg);
                color: var(--text-color);
                border: 1px solid var(--border-color);
                border-bottom: none;
                border-radius: 4px 4px 0 0;
            }
            
            .stTabs [aria-selected="true"] {
                background-color: var(--widget-bg) !important;
                color: var(--text-color) !important;
                border-bottom: 2px solid var(--green) !important;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: var(--secondary-bg) !important;
                border-right: 1px solid var(--border-color);
            }
            
            [data-testid="stSidebar"] .css-1d391kg {
                background-color: var(--secondary-bg);
            }
            
            /* Alerts and status messages */
            .stAlert {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                color: var(--text-color);
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--secondary-bg);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border-color);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--text-light);
            }
            
            /* Code blocks */
            .stCodeBlock {
                background-color: var(--widget-bg);
                border: 1px solid var(--border-color);
                border-radius: 4px;
            }
            
            /* Data editor */
            [data-testid="stDataFrame"] {
                background-color: var(--widget-bg);
            }
            
            /* Chat messages */
            .stChatMessage {
                background-color: var(--secondary-bg);
                border: 1px solid var(--border-color);
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            
        </style>
        """
        st.markdown(theme_css, unsafe_allow_html=True)
        
        # Apply theme class to body
        js_theme = f"""
        <script>
            // Remove existing theme classes
            document.body.classList.remove('light-theme', 'dark-theme');
            // Add current theme class
            document.body.classList.add('{st.session_state.theme.lower()}-theme');
            
            // Also set Plotly chart theme
            const plotlyThemes = document.querySelectorAll('.js-plotly-plot');
            plotlyThemes.forEach(plot => {{
                plot.style.backgroundColor = getComputedStyle(document.body).getPropertyValue('--widget-bg');
            }});
        </script>
        """
        st.components.v1.html(js_theme, height=0)
    except Exception as e:
        handle_error(e, "apply_custom_styling", "Failed to apply custom styling")

# Centralized data source configuration
ML_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NSE"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NFO"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NFO"
    },
    "GOLD": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv",
        "tradingsymbol": "GOLDM",
        "exchange": "MCX"
    },
    "USDINR": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv",
        "tradingsymbol": "USDINR",
        "exchange": "CDS"
    },
    "SENSEX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv",
        "tradingsymbol": "SENSEX",
        "exchange": "BSE"
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
        "tradingsymbol": "^GSPC",
        "exchange": "yfinance"
    }
}

# ================ 1.5 INITIALIZATION ========================

def initialize_session_state():
    """Initializes all necessary session state variables with proper default watchlist."""
    try:
        if 'broker' not in st.session_state: st.session_state.broker = None
        if 'kite' not in st.session_state: st.session_state.kite = None
        if 'profile' not in st.session_state: st.session_state.profile = None
        if 'login_animation_complete' not in st.session_state: st.session_state.login_animation_complete = False
        if 'authenticated' not in st.session_state: st.session_state.authenticated = False
        if 'two_factor_setup_complete' not in st.session_state: st.session_state.two_factor_setup_complete = False
        if 'pyotp_secret' not in st.session_state: st.session_state.pyotp_secret = None
        if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
        
        # Improved default watchlists with popular stocks (no bonds)
        if 'watchlists' not in st.session_state:
            st.session_state.watchlists = {
                "Watchlist 1": [
                    {'symbol': 'RELIANCE', 'exchange': 'NSE'},
                    {'symbol': 'TCS', 'exchange': 'NSE'},
                    {'symbol': 'HDFCBANK', 'exchange': 'NSE'},
                    {'symbol': 'INFY', 'exchange': 'NSE'},
                    {'symbol': 'HINDUNILVR', 'exchange': 'NSE'}
                ],
                "Watchlist 2": [
                    {'symbol': 'ICICIBANK', 'exchange': 'NSE'},
                    {'symbol': 'SBIN', 'exchange': 'NSE'},
                    {'symbol': 'BAJFINANCE', 'exchange': 'NSE'},
                    {'symbol': 'KOTAKBANK', 'exchange': 'NSE'},
                    {'symbol': 'ITC', 'exchange': 'NSE'}
                ],
                "Watchlist 3": [
                    {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
                    {'symbol': 'BANKNIFTY', 'exchange': 'NSE'},
                    {'symbol': 'SENSEX', 'exchange': 'BSE'}
                ]
            }
        
        if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Watchlist 1"
        if 'order_history' not in st.session_state: st.session_state.order_history = []
        if 'basket' not in st.session_state: st.session_state.basket = []
        if 'last_order_details' not in st.session_state: st.session_state.last_order_details = {}
        if 'underlying_pcr' not in st.session_state: st.session_state.underlying_pcr = "NIFTY"
        if 'strategy_legs' not in st.session_state: st.session_state.strategy_legs = []
        if 'calculated_greeks' not in st.session_state: st.session_state.calculated_greeks = None
        if 'messages' not in st.session_state: st.session_state.messages = []
        if 'ml_forecast_df' not in st.session_state: st.session_state.ml_forecast_df = None
        if 'ml_instrument_name' not in st.session_state: st.session_state.ml_instrument_name = None
        if 'backtest_results' not in st.session_state: st.session_state.backtest_results = None
        if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
        if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []
        if 'last_bot_result' not in st.session_state: st.session_state.last_bot_result = None
        
        # Add new dialog state variables
        if 'show_quick_trade' not in st.session_state: st.session_state.show_quick_trade = False
        if 'show_most_active' not in st.session_state: st.session_state.show_most_active = False
        if 'show_2fa_dialog' not in st.session_state: st.session_state.show_2fa_dialog = False
        if 'show_qr_dialog' not in st.session_state: st.session_state.show_qr_dialog = False
        
        # Add automated mode variables
        if 'automated_mode' not in st.session_state:
            st.session_state.automated_mode = {
                'enabled': False,
                'running': False,
                'bots_active': {},
                'total_capital': 10000,
                'risk_per_trade': 2,
                'max_open_trades': 5,
                'trade_history': [],
                'performance_metrics': {},
                'last_signal_check': None
            }
    except Exception as e:
        handle_error(e, "initialize_session_state", "Failed to initialize application state")

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    try:
        if st.session_state.get('broker') == "Zerodha":
            return st.session_state.get('kite')
        return None
    except Exception as e:
        handle_error(e, "get_broker_client", "Failed to get broker client")
        return None

def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    try:
        if 'show_quick_trade' not in st.session_state:
            st.session_state.show_quick_trade = False
        
        # Button to open the dialog
        if st.sidebar.button("Quick Trade", use_container_width=True) or st.session_state.show_quick_trade:
            st.session_state.show_quick_trade = True
            
            # Create the dialog content
            st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
            
            if symbol is None:
                symbol = st.text_input("Symbol").upper()
            
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
            quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
            price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

            col1, col2 = st.columns(2)
            if col1.button("Submit Order", use_container_width=True):
                if symbol and quantity > 0:
                    place_order(get_instrument_df(), symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
                    st.session_state.show_quick_trade = False
                    st.rerun()
                else:
                    st.warning("Please fill in all fields.")
            
            if col2.button("Cancel", use_container_width=True):
                st.session_state.show_quick_trade = False
                st.rerun()
    except Exception as e:
        handle_error(e, "quick_trade_dialog", "Failed to open quick trade dialog")

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    try:
        holidays_by_year = {
            2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
            2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
            2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
        }
        return holidays_by_year.get(year, [])
    except Exception as e:
        handle_error(e, "get_market_holidays", "Failed to get market holidays")
        return []

def get_market_status():
    """Checks if the Indian stock market is open."""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        holidays = get_market_holidays(now.year)
        market_open_time, market_close_time = time(9, 15), time(15, 30)
        
        if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
            return {"status": "CLOSED", "color": "#FF4B4B"}
        if market_open_time <= now.time() <= market_close_time:
            return {"status": "OPEN", "color": "#28a745"}
        return {"status": "CLOSED", "color": "#FF4B4B"}
    except Exception as e:
        handle_error(e, "get_market_status", "Failed to get market status")
        return {"status": "UNKNOWN", "color": "#FF4B4B"}

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    try:
        status_info = get_market_status()
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist).strftime("%H:%M:%S IST")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style="text-align: right;">
                    <h5 style="margin: 0;">{current_time}</h5>
                    <h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            b_col1, b_col2 = st.columns(2)
            if b_col1.button("Buy", use_container_width=True, key="header_buy"):
                st.session_state.show_quick_trade = True
                st.rerun()
            if b_col2.button("Sell", use_container_width=True, key="header_sell"):
                st.session_state.show_quick_trade = True
                st.rerun()

        st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    except Exception as e:
        handle_error(e, "display_header", "Failed to display header")

def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
    try:
        overnight_tickers = {"GIFT NIFTY": "NIFTY_F1", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
        data = get_global_indices_data(overnight_tickers)
        
        if not data.empty:
            bar_html = "<div class='notification-bar'>"
            for name, ticker in overnight_tickers.items():
                row = data[data['Ticker'] == name]
                if not row.empty:
                    price = row.iloc[0]['Price']
                    change = row.iloc[0]['% Change']
                    if not np.isnan(price):
                        color = 'var(--green)' if change > 0 else 'var(--red)'
                        bar_html += f"<span>{name}: {price:,.2f} <span style='color:{color};'>({change:+.2f}%)</span></span>"
            bar_html += "</div>"
            st.markdown(bar_html, unsafe_allow_html=True)
    except Exception as e:
        handle_error(e, "display_overnight_changes_bar", "Failed to display overnight changes")

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def get_theme_colors():
    """Returns color scheme based on current theme."""
    try:
        is_dark = st.session_state.get('theme') == 'Dark'
        return {
            'primary': '#28a745' if is_dark else '#198754',
            'secondary': '#161B22' if is_dark else '#F8F9FA',
            'accent': '#58a6ff' if is_dark else '#0d6efd',
            'text': '#c9d1d9' if is_dark else '#212529',
            'text_light': '#8b949e' if is_dark else '#6c757d',
            'border': '#30363D' if is_dark else '#dee2e6',
            'success': '#28a745' if is_dark else '#198754',
            'danger': '#da3633' if is_dark else '#dc3545',
            'warning': '#d29922' if is_dark else '#fd7e14',
            'info': '#58a6ff' if is_dark else '#0dcaf0'
        }
    except Exception as e:
        handle_error(e, "get_theme_colors", "Failed to get theme colors")
        return {}

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    """Generates a Plotly chart with proper theme adaptation."""
    try:
        fig = go.Figure()
        if df.empty: return fig
        chart_df = df.copy()
        
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.droplevel(0)
            
        chart_df.columns = [str(col).lower() for col in chart_df.columns]
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in chart_df.columns for col in required_cols):
            st.error(f"Charting error for {ticker}: Dataframe is missing required columns (open, high, low, close).")
            return go.Figure()

        # Get theme colors
        is_dark = st.session_state.get('theme') == 'Dark'
        colors = get_theme_colors()
        text_color = colors['text']
        grid_color = 'rgba(128, 128, 128, 0.2)' if is_dark else 'rgba(128, 128, 128, 0.1)'
        
        if chart_type == 'Heikin-Ashi':
            ha_close = (chart_df['open'] + chart_df['high'] + chart_df['low'] + chart_df['close']) / 4
            ha_open = (chart_df['open'].shift(1) + chart_df['close'].shift(1)) / 2
            ha_open.iloc[0] = (chart_df['open'].iloc[0] + chart_df['close'].iloc[0]) / 2
            ha_high = chart_df[['high', 'open', 'close']].max(axis=1)
            ha_low = chart_df[['low', 'open', 'close']].min(axis=1)
            
            fig.add_trace(go.Candlestick(
                x=chart_df.index, 
                open=ha_open, 
                high=ha_high, 
                low=ha_low, 
                close=ha_close, 
                name='Heikin-Ashi'
            ))
        elif chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=chart_df.index, 
                y=chart_df['close'], 
                mode='lines', 
                name='Line',
                line=dict(color='#1f77b4')
            ))
        elif chart_type == 'Bar':
            fig.add_trace(go.Ohlc(
                x=chart_df.index, 
                open=chart_df['open'], 
                high=chart_df['high'], 
                low=chart_df['low'], 
                close=chart_df['close'], 
                name='Bar'
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=chart_df.index, 
                open=chart_df['open'], 
                high=chart_df['high'], 
                low=chart_df['low'], 
                close=chart_df['close'], 
                name='Candlestick'
            ))
            
        # Bollinger Bands using TA-Lib
        if 'close' in chart_df.columns:
            upperband, middleband, lowerband = talib.BBANDS(chart_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            fig.add_trace(go.Scatter(
                x=chart_df.index, 
                y=lowerband, 
                line=dict(color='rgba(135,206,250,0.5)', width=1), 
                name='Lower Band'
            ))
            fig.add_trace(go.Scatter(
                x=chart_df.index, 
                y=upperband, 
                line=dict(color='rgba(135,206,250,0.5)', width=1), 
                fill='tonexty', 
                fillcolor='rgba(135,206,250,0.1)', 
                name='Upper Band'
            ))
            
        if forecast_df is not None:
            fig.add_trace(go.Scatter(
                x=forecast_df.index, 
                y=forecast_df['Predicted'], 
                mode='lines', 
                line=dict(color='orange', dash='dash'), 
                name='Forecast'
            ))
            if conf_int_df is not None:
                fig.add_trace(go.Scatter(
                    x=conf_int_df.index, 
                    y=conf_int_df['lower'], 
                    line=dict(color='rgba(255,165,0,0.2)', width=1), 
                    name='Lower CI', 
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=conf_int_df.index, 
                    y=conf_int_df['upper'], 
                    line=dict(color='rgba(255,165,0,0.2)', width=1), 
                    fill='tonexty', 
                    fillcolor='rgba(255,165,0,0.2)', 
                    name='Confidence Interval'
                ))
        
        # Use proper template based on theme
        template = 'plotly_dark' if is_dark else 'plotly_white'
        
        fig.update_layout(
            title=f'{ticker} Price Chart ({chart_type})',
            yaxis_title='Price (INR)',
            xaxis_rangeslider_visible=False,
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'  # Transparent background
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color),
            xaxis=dict(gridcolor=grid_color),
            yaxis=dict(gridcolor=grid_color)
        )
        return fig
    except Exception as e:
        handle_error(e, "create_chart", f"Failed to create chart for {ticker}")
        return go.Figure()
    
@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    try:
        client = get_broker_client()
        if not client: return pd.DataFrame()
        if st.session_state.broker == "Zerodha":
            df = pd.DataFrame(client.instruments())
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'])
            return df
        else:
            st.warning(f"Instrument list for {st.session_state.broker} not implemented.")
            return pd.DataFrame()
    except Exception as e:
        handle_error(e, "get_instrument_df", "Failed to fetch instrument data")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    try:
        if instrument_df.empty: return None
        match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
        return match.iloc[0]['instrument_token'] if not match.empty else None
    except Exception as e:
        handle_error(e, "get_instrument_token", f"Failed to get instrument token for {symbol}")
        return None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API."""
    try:
        client = get_broker_client()
        if not client or not instrument_token: return pd.DataFrame()
        if st.session_state.broker == "Zerodha":
            if not to_date: to_date = datetime.now().date()
            if not from_date:
                days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
                from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
            
            if from_date > to_date:
                from_date = to_date - timedelta(days=1)
                
            try:
                records = client.historical_data(instrument_token, from_date, to_date, interval)
                df = pd.DataFrame(records)
                if df.empty: return df
                df.set_index('date', inplace=True)
                df.index = pd.to_datetime(df.index)
                return df

            except kite_exceptions.KiteException as e:
                st.error(f"Kite API Error (Historical): {e}")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"An unexpected error occurred fetching historical data: {e}")
                return pd.DataFrame()
        else:
            st.warning(f"Historical data for {st.session_state.broker} not implemented.")
            return pd.DataFrame()
    except Exception as e:
        handle_error(e, "get_historical_data", "Failed to fetch historical data")
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
    try:
        client = get_broker_client()
        if not client or not symbols_with_exchange: return pd.DataFrame()
        if st.session_state.broker == "Zerodha":
            instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
            try:
                quotes = client.quote(instrument_names)
                watchlist = []
                for item in symbols_with_exchange:
                    instrument = f"{item['exchange']}:{item['symbol']}"
                    if instrument in quotes:
                        quote = quotes[instrument]
                        last_price = quote['last_price']
                        prev_close = quote['ohlc']['close']
                        change = last_price - prev_close
                        pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                        watchlist.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': last_price, 'Change': change, '% Change': pct_change})
                return pd.DataFrame(watchlist)
            except Exception as e:
                st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
                return pd.DataFrame()
        else:
            st.warning(f"Watchlist for {st.session_state.broker} not implemented.")
            return pd.DataFrame()
    except Exception as e:
        handle_error(e, "get_watchlist_data", "Failed to fetch watchlist data")
        return pd.DataFrame()

@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth (order book) for a given instrument."""
    try:
        client = get_broker_client()
        if not client or not instrument_token:
            return None
        try:
            depth = client.depth(instrument_token)
            return depth.get(str(instrument_token))
        except Exception as e:
            st.toast(f"Error fetching market depth: {e}", icon="⚠️")
            return None
    except Exception as e:
        handle_error(e, "get_market_depth", "Failed to fetch market depth")
        return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for a given underlying."""
    try:
        client = get_broker_client()
        if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
        if st.session_state.broker == "Zerodha":
            exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "NATURALGAS": "MCX", "USDINR": "CDS"}
            exchange = exchange_map.get(underlying, 'NFO')
            ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}.get(underlying, underlying)
            ltp_exchange = "NSE" if exchange == "NFO" else exchange
            underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"
            try:
                underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
            except Exception:
                underlying_ltp = 0.0
            
            options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
            if options.empty: return pd.DataFrame(), None, underlying_ltp, []
            
            expiries = sorted(options['expiry'].dt.date.unique())
            three_months_later = datetime.now().date() + timedelta(days=90)
            available_expiries = [e for e in expiries if datetime.now().date() <= e <= three_months_later]
            if not available_expiries: return pd.DataFrame(), None, underlying_ltp, []
            
            if not expiry_date: 
                expiry_date = available_expiries[0]
            else: 
                expiry_date = pd.to_datetime(expiry_date).date()
            
            chain_df = options[options['expiry'].dt.date == expiry_date].sort_values(by='strike')
            ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
            pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
            instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
            if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
            
            try:
                quotes = client.quote(instruments_to_fetch)
                ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
                pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
                ce_df['oi'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
                pe_df['oi'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
                
                final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                       pe_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                       on='strike', suffixes=('_CE', '_PE')).rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'oi_CE': 'CALL OI', 'oi_PE': 'PUT OI', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
                
                return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries
            except Exception as e:
                st.error(f"Failed to fetch real-time OI data: {e}")
                return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        else:
            st.warning(f"Options chain for {st.session_state.broker} not implemented.")
            return pd.DataFrame(), None, 0.0, []
    except Exception as e:
        handle_error(e, "get_options_chain", f"Failed to fetch options chain for {underlying}")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio positions and holdings from the broker."""
    try:
        client = get_broker_client()
        if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
        if st.session_state.broker == "Zerodha":
            try:
                positions = client.positions().get('net', [])
                holdings = client.holdings()
                positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
                total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
                holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
                total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
                return positions_df, holdings_df, total_pnl, total_investment
            except Exception as e:
                st.error(f"Kite API Error (Portfolio): {e}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
        else:
            st.warning(f"Portfolio for {st.session_state.broker} not implemented.")
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    except Exception as e:
        handle_error(e, "get_portfolio", "Failed to fetch portfolio data")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order through the broker's API."""
    try:
        client = get_broker_client()
        if not client:
            st.error("Broker not connected.")
            return
        if st.session_state.broker == "Zerodha":
            try:
                is_option = any(char.isdigit() for char in symbol)
                if is_option:
                    exchange = 'NFO'
                else:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
                    if instrument.empty:
                        st.error(f"Symbol '{symbol}' not found.")
                        return
                    exchange = instrument.iloc[0]['exchange']

                order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
                st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
                st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
            except Exception as e:
                st.toast(f"❌ Order failed: {e}", icon="🔥")
                st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})
        else:
            st.warning(f"Order placement for {st.session_state.broker} not implemented.")
    except Exception as e:
        handle_error(e, "place_order", f"Failed to place order for {symbol}")

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
    try:
        analyzer = SentimentIntensityAnalyzer()
        
        news_sources = {
            "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
            "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
            "Livemint": "https://www.livemint.com/rss/markets",
            "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
            "Reuters World": "http://feeds.reuters.com/Reuters/worldNews",
            "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml"
        }
        all_news = []
        for source, url in news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    published_date_tuple = entry.published_parsed if hasattr(entry, 'published_parsed') else entry.updated_parsed
                    published_date = datetime.fromtimestamp(mktime_tz(published_date_tuple)) if published_date_tuple else datetime.now()
                    if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                        all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
            except Exception:
                continue
        return pd.DataFrame(all_news)
    except Exception as e:
        handle_error(e, "fetch_and_analyze_news", "Failed to fetch and analyze news")
        return pd.DataFrame()

def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE function to remove sklearn dependency."""
    try:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except Exception as e:
        handle_error(e, "mean_absolute_percentage_error", "Failed to calculate MAPE")
        return float('inf')

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data, forecast_steps=30):
    """Trains a Seasonal ARIMA model for time series forecasting."""
    try:
        if _data.empty or len(_data) < 100:
            return None, None, None

        df = _data.copy()
        df.index = pd.to_datetime(df.index)
        
        try:
            decomposed = seasonal_decompose(df['close'], model='additive', period=7)
            seasonally_adjusted = df['close'] - decomposed.seasonal

            model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
            
            # Backtesting
            fitted_values = model.fittedvalues + decomposed.seasonal
            backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values}).dropna()
            
            # Forecasting
            forecast_result = model.get_forecast(steps=forecast_steps)
            forecast_adjusted = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            last_season_cycle = decomposed.seasonal.iloc[-7:]
            future_seasonal_values = np.tile(last_season_cycle.values, forecast_steps // 7 + 1)[:forecast_steps]
            
            future_forecast = forecast_adjusted + future_seasonal_values
            
            future_dates = pd.to_datetime(pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_steps))
            forecast_df = pd.DataFrame({'Predicted': future_forecast.values}, index=future_dates)
            
            # Add seasonality back to confidence intervals
            conf_int_df = pd.DataFrame({
                'lower': conf_int.iloc[:, 0] + future_seasonal_values,
                'upper': conf_int.iloc[:, 1] + future_seasonal_values
            }, index=future_dates)
            
            return forecast_df, backtest_df, conf_int_df

        except Exception as e:
            st.error(f"Seasonal ARIMA model training failed: {e}")
            return None, None, None
    except Exception as e:
        handle_error(e, "train_seasonal_arima_model", "Failed to train ARIMA model")
        return None, None, None

@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads and combines historical data from a static CSV with live data from the broker."""
    try:
        source_info = ML_DATA_SOURCES.get(instrument_name)
        if not source_info:
            st.error(f"No data source configured for {instrument_name}")
            return pd.DataFrame()
        try:
            response = requests.get(source_info['github_url'])
            response.raise_for_status()
            hist_df = pd.read_csv(io.StringIO(response.text))
            hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True).dt.tz_localize(None)
            hist_df.set_index('Date', inplace=True)
            hist_df.columns = [col.lower() for col in hist_df.columns]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in hist_df.columns:
                    hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
            hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        except Exception as e:
            st.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()
            
        live_df = pd.DataFrame()
        if get_broker_client() and source_info.get('tradingsymbol') and source_info.get('exchange') != 'yfinance':
            instrument_df = get_instrument_df()
            token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
            if token:
                from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
                live_df = get_historical_data(token, 'day', from_date=from_date)
                if not live_df.empty: 
                    live_df.index = live_df.index.tz_convert(None)
                    live_df.columns = [col.lower() for col in live_df.columns]
        elif source_info.get('exchange') == 'yfinance':
            try:
                live_df = yf.download(source_info['tradingsymbol'], period="max")
                if not live_df.empty: 
                    live_df.index = live_df.index.tz_localize(None)
                    live_df.columns = [col.lower() for col in live_df.columns]
            except Exception as e:
                st.error(f"Failed to load yfinance data: {e}")
                live_df = pd.DataFrame()
                
        if not live_df.empty:
            hist_df.index = hist_df.index.tz_localize(None) if hist_df.index.tz is not None else hist_df.index
            live_df.index = live_df.index.tz_localize(None) if live_df.index.tz is not None else live_df.index
            
            combined_df = pd.concat([hist_df, live_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            return combined_df
        else:
            hist_df.sort_index(inplace=True)
            return hist_df
    except Exception as e:
        handle_error(e, "load_and_combine_data", f"Failed to load data for {instrument_name}")
        return pd.DataFrame()

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price and Greeks."""
    try:
        if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
        return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}
    except Exception as e:
        handle_error(e, "black_scholes", "Failed to calculate Black-Scholes values")
        return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    try:
        if T <= 0 or market_price <= 0: return np.nan
        equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
        try:
            return newton(equation, 0.5, tol=1e-5, maxiter=100)
        except (RuntimeError, TypeError):
            return np.nan
    except Exception as e:
        handle_error(e, "implied_volatility", "Failed to calculate implied volatility")
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    try:
        if df.empty: return {}
        latest = df.iloc[-1].copy()
        latest.index = latest.index.str.lower()
        interpretation = {}
        
        # RSI using TA-Lib
        if 'close' in df.columns:
            rsi = talib.RSI(df['close'], timeperiod=14)
            if not np.isnan(rsi.iloc[-1]):
                rsi_val = rsi.iloc[-1]
                interpretation['RSI (14)'] = "Overbought (Bearish)" if rsi_val > 70 else "Oversold (Bullish)" if rsi_val < 30 else "Neutral"
        
        # Stochastic using TA-Lib
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        if not np.isnan(slowk.iloc[-1]):
            stoch_k = slowk.iloc[-1]
            interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
        
        # MACD using TA-Lib
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        if not np.isnan(macd.iloc[-1]) and not np.isnan(macdsignal.iloc[-1]):
            interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd.iloc[-1] > macdsignal.iloc[-1] else "Bearish Crossover"
        
        # ADX using TA-Lib
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        if not np.isnan(adx.iloc[-1]):
            adx_val = adx.iloc[-1]
            interpretation['ADX (14)'] = f"Strong Trend ({adx_val:.1f})" if adx_val > 25 else f"Weak/No Trend ({adx_val:.1f})"
        
        return interpretation
    except Exception as e:
        handle_error(e, "interpret_indicators", "Failed to interpret technical indicators")
        return {}

# ================ 4. HNI & PRO TRADER FEATURES ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
    try:
        client = get_broker_client()
        if not client:
            st.error("Broker not connected.")
            return
        
        if st.session_state.broker == "Zerodha":
            orders_to_place = []
            for item in basket_items:
                instrument = instrument_df[instrument_df['tradingsymbol'] == item['symbol']]
                if instrument.empty:
                    st.toast(f"❌ Could not find symbol {item['symbol']} in instrument list. Skipping.", icon="🔥")
                    continue
                exchange = instrument.iloc[0]['exchange']

                order = {
                    "tradingsymbol": item['symbol'],
                    "exchange": exchange,
                    "transaction_type": client.TRANSACTION_TYPE_BUY if item['transaction_type'] == 'BUY' else client.TRANSACTION_TYPE_SELL,
                    "quantity": int(item['quantity']),
                    "product": client.PRODUCT_MIS if item['product'] == 'MIS' else client.PRODUCT_CNC,
                    "order_type": client.ORDER_TYPE_MARKET if item['order_type'] == 'MARKET' else client.ORDER_TYPE_LIMIT,
                }
                if order['order_type'] == client.ORDER_TYPE_LIMIT:
                    order['price'] = item['price']
                orders_to_place.append(order)
            
            if not orders_to_place:
                st.warning("No valid orders to place in the basket.")
                return

            try:
                client.place_order(variety=client.VARIETY_REGULAR, orders=orders_to_place)
                st.toast("✅ Basket order placed successfully!", icon="🎉")
                st.session_state.basket = []
                st.rerun()
            except Exception as e:
                st.toast(f"❌ Basket order failed: {e}", icon="🔥")
    except Exception as e:
        handle_error(e, "execute_basket_order", "Failed to execute basket order")

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("'sensex_sectors.csv' not found. Sector allocation will be unavailable.")
        return None
    except Exception as e:
        handle_error(e, "get_sector_data", "Failed to load sector data")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to highlight ITM/OTM in the options chain."""
    try:
        if df.empty or 'STRIKE' not in df.columns or ltp == 0:
            return df.style

        def highlight_itm(row):
            styles = [''] * len(row)
            if row['STRIKE'] < ltp:
                styles[df.columns.get_loc('CALL LTP')] = 'background-color: #2E4053'
                styles[df.columns.get_loc('CALL OI')] = 'background-color: #2E4053'
            if row['STRIKE'] > ltp:
                styles[df.columns.get_loc('PUT LTP')] = 'background-color: #2E4053'
                styles[df.columns.get_loc('PUT OI')] = 'background-color: #2E4053'
            return styles

        return df.style.apply(highlight_itm, axis=1)
    except Exception as e:
        handle_error(e, "style_option_chain", "Failed to style option chain")
        return df.style

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    try:
        if 'show_most_active' not in st.session_state:
            st.session_state.show_most_active = False
        
        if st.button("Most Active Options", use_container_width=True) or st.session_state.show_most_active:
            st.session_state.show_most_active = True
            
            st.subheader(f"Most Active {underlying} Options (By Volume)")
            with st.spinner("Fetching data..."):

                client = get_broker_client()
                if not client:
                    st.toast("Broker not connected.", icon="⚠️")
                    return pd.DataFrame()
                
                try:
                    chain_df, expiry, _, _ = get_options_chain(underlying, instrument_df)
                    if chain_df.empty or expiry is None:
                        return pd.DataFrame()
                    ce_symbols = chain_df['CALL'].dropna().tolist()
                    pe_symbols = chain_df['PUT'].dropna().tolist()
                    all_symbols = [f"NFO:{s}" for s in ce_symbols + pe_symbols]

                    if not all_symbols:
                        return pd.DataFrame()

                    quotes = client.quote(all_symbols)
                    
                    active_options = []
                    for symbol, data in quotes.items():
                        prev_close = data.get('ohlc', {}).get('close', 0)
                        last_price = data.get('last_price', 0)
                        change = last_price - prev_close
                        pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                        
                        active_options.append({
                            'Symbol': data.get('tradingsymbol'),
                            'LTP': last_price,
                            'Change %': pct_change,
                            'Volume': data.get('volume', 0),
                            'OI': data.get('oi', 0)
                        })
                    
                    df = pd.DataFrame(active_options)
                    df_sorted = df.sort_values(by='Volume', ascending=False)
                    st.dataframe(df_sorted.head(10), use_container_width=True, hide_index=True)

                    if st.button("Close", use_container_width=True):
                        st.session_state.show_most_active = False
                        st.rerun()

                except Exception as e:
                    st.error(f"Could not fetch most active options: {e}")
    except Exception as e:
        handle_error(e, "show_most_active_dialog", "Failed to show most active options dialog")

@st.cache_data(ttl=60)
def get_global_indices_data_enhanced(tickers):
    """Enhanced version of global indices data fetcher with better error handling."""
    try:
        if not tickers:
            return pd.DataFrame()
        
        try:
            data_yf = yf.download(list(tickers.values()), period="5d", progress=False)
            if data_yf.empty:
                return pd.DataFrame()

            data = []
            for ticker_name, yf_ticker_name in tickers.items():
                if len(tickers) > 1:
                    # For multiple tickers, extract data for each ticker
                    if yf_ticker_name in data_yf['Close'].columns:
                        hist = data_yf['Close'][yf_ticker_name]
                    else:
                        continue
                else:
                    hist = data_yf['Close']

                if len(hist) >= 2:
                    last_price = float(hist.iloc[-1])
                    prev_close = float(hist.iloc[-2])
                    change = last_price - prev_close
                    pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                    data.append({'Ticker': ticker_name, 'Price': last_price, 'Change': change, '% Change': pct_change})
                else:
                    data.append({'Ticker': ticker_name, 'Price': np.nan, 'Change': np.nan, '% Change': np.nan})

            return pd.DataFrame(data)

        except Exception as e:
            st.error(f"Failed to fetch data from yfinance: {e}")
            return pd.DataFrame()
    except Exception as e:
        handle_error(e, "get_global_indices_data_enhanced", "Failed to fetch global indices data")
        return pd.DataFrame()

# ================ ALGO TRADING BOTS SECTION ================

def momentum_trader_bot(instrument_df, symbol, capital=100):
    """Momentum trading bot that buys on upward momentum and sells on downward momentum."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, '5minute', period='1d')
            if data.empty or len(data) < 20:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate indicators with TA-Lib
            data['RSI'] = talib.RSI(data['close'], timeperiod=14)
            data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
            data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            signals = []
            
            # Momentum signals
            if (latest['EMA_20'] > latest['EMA_50'] and 
                prev['EMA_20'] <= prev['EMA_50']):
                signals.append("EMA crossover - BULLISH")
            
            rsi_val = latest.get('RSI', 50)
            if rsi_val < 30:
                signals.append("RSI oversold - BULLISH")
            elif rsi_val > 70:
                signals.append("RSI overbought - BEARISH")
            
            # Price momentum
            if len(data) >= 6:
                price_change_5min = ((latest['close'] - data.iloc[-6]['close']) / data.iloc[-6]['close']) * 100
                if price_change_5min > 0.5:
                    signals.append(f"Strong upward momentum: +{price_change_5min:.2f}%")
            
            # Calculate position size
            current_price = latest['close']
            quantity = max(1, int((capital * 0.8) / current_price))
            
            action = "HOLD"
            if len([s for s in signals if "BULLISH" in s]) >= 2:
                action = "BUY"
            elif len([s for s in signals if "BEARISH" in s]) >= 2:
                action = "SELL"
            
            return {
                "bot_name": "Momentum Trader",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "Medium"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def mean_reversion_bot(instrument_df, symbol, capital=100):
    """Mean reversion bot that trades on price returning to mean levels."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, '15minute', period='5d')
            if data.empty or len(data) < 50:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate Bollinger Bands using TA-Lib
            upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            data['BB_Upper'] = upperband
            data['BB_Middle'] = middleband
            data['BB_Lower'] = lowerband
            
            latest = data.iloc[-1]
            
            signals = []
            current_price = latest['close']
            bb_lower = latest.get('BB_Lower', current_price)
            bb_upper = latest.get('BB_Upper', current_price)
            bb_middle = latest.get('BB_Middle', current_price)
            
            # Mean reversion signals
            if current_price <= bb_lower * 1.02:
                signals.append("Near lower Bollinger Band - BULLISH")
            
            if current_price >= bb_upper * 0.98:
                signals.append("Near upper Bollinger Band - BEARISH")
            
            # Distance from mean
            distance_from_mean = ((current_price - bb_middle) / bb_middle) * 100
            if abs(distance_from_mean) > 3:
                signals.append(f"Price {abs(distance_from_mean):.1f}% from mean")
            
            # RSI for confirmation
            data['RSI'] = talib.RSI(data['close'], timeperiod=14)
            rsi = data['RSI'].iloc[-1]
            if rsi < 35:
                signals.append("RSI supporting oversold condition")
            elif rsi > 65:
                signals.append("RSI supporting overbought condition")
            
            # Calculate position size
            quantity = max(1, int((capital * 0.6) / current_price))
            
            action = "HOLD"
            if any("BULLISH" in s for s in signals) and rsi < 40:
                action = "BUY"
            elif any("BEARISH" in s for s in signals) and rsi > 60:
                action = "SELL"
            
            return {
                "bot_name": "Mean Reversion",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "Low"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def volatility_breakout_bot(instrument_df, symbol, capital=100):
    """Volatility breakout bot that trades on breakouts from consolidation."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, '30minute', period='5d')
            if data.empty or len(data) < 30:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate ATR using TA-Lib
            data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
            data['Range'] = data['high'] - data['low']
            avg_range = data['Range'].rolling(window=20).mean()
            
            latest = data.iloc[-1]
            current_price = latest['close']
            current_atr = latest['ATR']
            current_range = latest['Range']
            
            signals = []
            
            # Volatility signals
            if current_range > avg_range.iloc[-1] * 1.5:
                signals.append("High volatility - potential breakout")
            
            # Price action signals
            prev_high = data['high'].iloc[-2]
            prev_low = data['low'].iloc[-2]
            
            if current_price > prev_high + current_atr * 0.5:
                signals.append("Breakout above resistance - BULLISH")
            
            if current_price < prev_low - current_atr * 0.5:
                signals.append("Breakdown below support - BEARISH")
            
            # Volume confirmation (if available)
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                if data['volume'].iloc[-1] > avg_volume * 1.2:
                    signals.append("High volume confirmation")
            
            # Calculate position size based on ATR
            atr_percentage = (current_atr / current_price) * 100
            risk_per_trade = min(20, max(5, atr_percentage * 2))
            quantity = max(1, int((capital * (risk_per_trade / 100)) / current_price))
            
            action = "HOLD"
            if any("BULLISH" in s for s in signals):
                action = "BUY"
            elif any("BEARISH" in s for s in signals):
                action = "SELL"
            
            return {
                "bot_name": "Volatility Breakout",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "High"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def value_investor_bot(instrument_df, symbol, capital=100):
    """Value investor bot focusing on longer-term value signals."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, 'day', period='1y')
            if data.empty or len(data) < 100:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate moving averages using TA-Lib
            data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
            data['SMA_200'] = talib.SMA(data['close'], timeperiod=200)
            data['EMA_21'] = talib.EMA(data['close'], timeperiod=21)
            
            latest = data.iloc[-1]
            current_price = latest['close']
            
            signals = []
            
            # Trend analysis
            if latest['SMA_50'] > latest['SMA_200']:
                signals.append("Bullish trend (50 > 200 SMA)")
            else:
                signals.append("Bearish trend (50 < 200 SMA)")
            
            # Support and resistance levels
            support_20 = data['low'].rolling(window=20).min().iloc[-1]
            resistance_20 = data['high'].rolling(window=20).max().iloc[-1]
            
            distance_to_support = ((current_price - support_20) / current_price) * 100
            distance_to_resistance = ((resistance_20 - current_price) / current_price) * 100
            
            if distance_to_support < 5:
                signals.append("Near strong support - BULLISH")
            
            if distance_to_resistance < 5:
                signals.append("Near strong resistance - BEARISH")
            
            # Monthly performance
            monthly_return = ((current_price - data['close'].iloc[-21]) / data['close'].iloc[-21]) * 100
            if monthly_return < -10:
                signals.append("Oversold on monthly basis - BULLISH")
            elif monthly_return > 15:
                signals.append("Overbought on monthly basis - BEARISH")
            
            # Calculate position size for longer term
            quantity = max(1, int((capital * 0.5) / current_price))
            
            action = "HOLD"
            bullish_signals = len([s for s in signals if "BULLISH" in s])
            bearish_signals = len([s for s in signals if "BEARISH" in s])
            
            if bullish_signals >= 2 and bearish_signals == 0:
                action = "BUY"
            elif bearish_signals >= 2 and bullish_signals == 0:
                action = "SELL"
            
            return {
                "bot_name": "Value Investor",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "Low"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def scalper_bot(instrument_df, symbol, capital=100):
    """High-frequency scalping bot for quick, small profits."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, 'minute', period='1d')
            if data.empty or len(data) < 100:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate scalping indicators using TA-Lib
            data['RSI_9'] = talib.RSI(data['close'], timeperiod=9)
            data['EMA_8'] = talib.EMA(data['close'], timeperiod=8)
            data['EMA_21'] = talib.EMA(data['close'], timeperiod=21)
            
            latest = data.iloc[-1]
            current_price = latest['close']
            
            signals = []
            
            # Scalping signals
            if latest['EMA_8'] > latest['EMA_21']:
                signals.append("Fast EMA above slow EMA - BULLISH")
            else:
                signals.append("Fast EMA below slow EMA - BEARISH")
            
            rsi_9 = latest['RSI_9']
            if rsi_9 < 25:
                signals.append("Extremely oversold - BULLISH")
            elif rsi_9 > 75:
                signals.append("Extremely overbought - BEARISH")
            
            # Price momentum for scalping
            price_change_3min = ((current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]) * 100
            if abs(price_change_3min) > 0.3:
                signals.append(f"Strong short-term momentum: {price_change_3min:+.2f}%")
            
            # Calculate small position size for scalping
            quantity = max(1, int((capital * 0.3) / current_price))
            
            action = "HOLD"
            if (any("BULLISH" in s for s in signals) and 
                "BEARISH" not in str(signals) and
                rsi_9 < 70):
                action = "BUY"
            elif (any("BEARISH" in s for s in signals) and 
                  "BULLISH" not in str(signals) and
                  rsi_9 > 30):
                action = "SELL"
            
            return {
                "bot_name": "Scalper Pro",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "Very High"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def trend_follower_bot(instrument_df, symbol, capital=100):
    """Trend following bot that rides established trends."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, 'hour', period='1mo')
            if data.empty or len(data) < 100:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate trend indicators using TA-Lib
            data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
            data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
            data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
            
            # Simple trend detection without SuperTrend
            data['Trend'] = np.where(data['EMA_20'] > data['EMA_50'], 1, -1)
            
            latest = data.iloc[-1]
            current_price = latest['close']
            
            signals = []
            
            # Trend strength
            adx = latest.get('ADX', 0)
            if adx > 25:
                signals.append(f"Strong trend (ADX: {adx:.1f})")
            else:
                signals.append(f"Weak trend (ADX: {adx:.1f})")
            
            # Trend direction
            if latest['EMA_20'] > latest['EMA_50']:
                signals.append("Uptrend confirmed - BULLISH")
            else:
                signals.append("Downtrend confirmed - BEARISH")
            
            # Price relative to trend
            if current_price > latest['EMA_20']:
                signals.append("Price above short-term trend - BULLISH")
            else:
                signals.append("Price below short-term trend - BEARISH")
            
            # Pullback opportunities
            if (latest['EMA_20'] > latest['EMA_50'] and 
                current_price < latest['EMA_20'] and 
                current_price > latest['EMA_50']):
                signals.append("Pullback in uptrend - BULLISH")
            
            elif (latest['EMA_20'] < latest['EMA_50'] and 
                  current_price > latest['EMA_20'] and 
                  current_price < latest['EMA_50']):
                signals.append("Pullback in downtrend - BEARISH")
            
            # Calculate position size
            quantity = max(1, int((capital * 0.7) / current_price))
            
            action = "HOLD"
            bullish_count = len([s for s in signals if "BULLISH" in s])
            bearish_count = len([s for s in signals if "BEARISH" in s])
            
            if bullish_count >= 2 and adx > 20:
                action = "BUY"
            elif bearish_count >= 2 and adx > 20:
                action = "SELL"
            
            return {
                "bot_name": "Trend Follower",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "current_price": current_price,
                "signals": signals,
                "capital_required": quantity * current_price,
                "risk_level": "Medium"
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

# Dictionary of all available bots
ALGO_BOTS = {
    "Momentum Trader": momentum_trader_bot,
    "Mean Reversion": mean_reversion_bot,
    "Volatility Breakout": volatility_breakout_bot,
    "Value Investor": value_investor_bot,
    "Scalper Pro": scalper_bot,
    "Trend Follower": trend_follower_bot
}

def execute_bot_trade(instrument_df, bot_result):
    """Displays bot recommendations WITHOUT automatic execution - requires manual confirmation."""
    try:
        if bot_result.get("error"):
            st.error(bot_result["error"])
            return
        
        if bot_result["action"] == "HOLD":
            st.info(f"🤖 {bot_result['bot_name']} recommends HOLDING {bot_result['symbol']}")
            return
        
        action = bot_result["action"]
        symbol = bot_result["symbol"]
        quantity = bot_result["quantity"]
        current_price = bot_result["current_price"]
        required_capital = bot_result["capital_required"]
        
        st.success(f"""
        🚀 **{bot_result['bot_name']} Recommendation:**
        - **Action:** {action} {quantity} shares of {symbol}
        - **Current Price:** ₹{current_price:.2f}
        - **Required Capital:** ₹{required_capital:.2f}
        - **Risk Level:** {bot_result['risk_level']}
        """)
        
        # Display signals first
        st.subheader("📊 Analysis Signals")
        for signal in bot_result["signals"]:
            if "BULLISH" in signal:
                st.success(f"✅ {signal}")
            elif "BEARISH" in signal:
                st.error(f"❌ {signal}")
            else:
                st.info(f"📈 {signal}")
        
        # Manual execution section - clearly separated
        st.markdown("---")
        st.subheader("🚀 Manual Trade Execution")
        st.warning("**Manual Confirmation Required:** Click the button below ONLY if you want to execute this trade.")
        
        col1, col2 = st.columns(2)
        
        if col1.button(f"Execute {action} Order", key=f"execute_{symbol}", use_container_width=True, type="primary"):
            # Only execute when user explicitly clicks
            place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
            st.toast(f"✅ {action} order for {symbol} placed successfully!", icon="🎉")
            st.rerun()
        
        if col2.button("Ignore Recommendation", key=f"ignore_{symbol}", use_container_width=True):
            st.info("Trade execution cancelled.")
            st.rerun()
    except Exception as e:
        handle_error(e, "execute_bot_trade", "Failed to execute bot trade")

def page_algo_bots():
    """Main algo bots page where users can run different trading bots."""
    try:
        display_header()
        st.title("🤖 Algo Trading Bots")
        st.info("Run automated trading bots with minimum capital of ₹100. Each bot uses different strategies and risk profiles.", icon="🤖")
        
        instrument_df = get_instrument_df()
        if instrument_df.empty:
            st.info("Please connect to a broker to use algo bots.")
            return
        
        # Bot selection and configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_bot = st.selectbox(
                "Select Trading Bot",
                list(ALGO_BOTS.keys()),
                help="Choose a trading bot based on your risk appetite and trading style"
            )
            
            # Bot descriptions
            bot_descriptions = {
                "Momentum Trader": "Trades on strong price momentum and trend continuations. Medium risk.",
                "Mean Reversion": "Buys low and sells high based on statistical mean reversion. Low risk.",
                "Volatility Breakout": "Captures breakouts from low volatility periods. High risk.",
                "Value Investor": "Focuses on longer-term value and fundamental trends. Low risk.",
                "Scalper Pro": "High-frequency trading for quick, small profits. Very high risk.",
                "Trend Follower": "Identifies and rides established market trends using ADX and EMAs. Medium risk."
            }
            
            st.markdown(f"**Description:** {bot_descriptions.get(selected_bot, 'N/A')}")
        
        with col2:
            trading_capital = st.number_input(
                "Trading Capital (₹)",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Minimum ₹100 required"
            )
        
        st.markdown("---")
        
        # Symbol selection and bot execution
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("Stock Selection")
            all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
            selected_symbol = st.selectbox(
                "Select Stock",
                sorted(all_symbols),
                index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0
            )
            
            # Show current price
            quote_data = get_watchlist_data([{'symbol': selected_symbol, 'exchange': 'NSE'}])
            if not quote_data.empty:
                current_price = quote_data.iloc[0]['Price']
                st.metric("Current Price", f"₹{current_price:.2f}")
        
        with col4:
            st.subheader("Bot Execution")
            st.write(f"**Selected Bot:** {selected_bot}")
            st.write(f"**Available Capital:** ₹{trading_capital:,}")
            
            if st.button("🚀 Run Trading Bot", use_container_width=True, type="primary"):
                with st.spinner(f"Running {selected_bot} analysis..."):
                    bot_function = ALGO_BOTS[selected_bot]
                    bot_result = bot_function(instrument_df, selected_symbol, trading_capital)
                    
                    if bot_result and not bot_result.get("error"):
                        st.session_state.last_bot_result = bot_result
                        st.rerun()
        
        # Display bot results
        if 'last_bot_result' in st.session_state and st.session_state.last_bot_result:
            bot_result = st.session_state.last_bot_result
            
            if bot_result.get("error"):
                st.error(bot_result["error"])
            else:
                st.markdown("---")
                st.subheader("🤖 Bot Analysis Results")
                
                # Create metrics cards
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    action_color = "green" if bot_result["action"] == "BUY" else "red" if bot_result["action"] == "SELL" else "orange"
                    st.markdown(f'<div class="metric-card" style="border-color: {action_color};">'
                                f'<h3 style="color: {action_color};">{bot_result["action"]}</h3>'
                                f'<p>Recommended Action</p></div>', unsafe_allow_html=True)
                
                with col6:
                    st.metric("Quantity", bot_result["quantity"])
                
                with col7:
                    st.metric("Capital Required", f"₹{bot_result['capital_required']:.2f}")
                
                with col8:
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
                    st.markdown(f'<div class="metric-card" style="border-color: {risk_color.get(bot_result["risk_level"], "gray")};">'
                                f'<h3 style="color: {risk_color.get(bot_result["risk_level"], "gray")};">{bot_result["risk_level"]}</h3>'
                                f'<p>Risk Level</p></div>', unsafe_allow_html=True)
                
                # Display signals and execute trade
                execute_bot_trade(instrument_df, bot_result)

        # Bot performance history
        st.markdown("---")
        st.subheader("📈 Bot Performance Tips")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **Best Practices:**
            - Start with minimum capital (₹100)
            - Use 'Value Investor' for beginners
            - 'Scalper Pro' requires constant monitoring
            - Always check signals before executing
            - Combine multiple bot recommendations
            """)
        
        with tips_col2:
            st.markdown("""
            **Risk Management:**
            - Never risk more than 2% per trade
            - Use stop losses with every trade
            - Diversify across different bots
            - Monitor performance regularly
            - Adjust capital based on experience
            """)
        
        # Quick bot comparison
        with st.expander("🤖 Bot Comparison Guide"):
            comparison_data = {
                "Bot": list(ALGO_BOTS.keys()),
                "Risk Level": ["Medium", "Low", "High", "Low", "Very High", "Medium"],
                "Holding Period": ["Hours", "Days", "Minutes", "Weeks", "Minutes", "Days"],
                "Capital Recommended": ["₹1,000+", "₹500+", "₹2,000+", "₹2,000+", "₹5,000+", "₹1,500+"],
                "Best For": ["Trend riding", "Safe returns", "Quick profits", "Long term", "Experienced", "Trend following"]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    except Exception as e:
        handle_error(e, "page_algo_bots", "Failed to load algo bots page")

# ... (continuing with the rest of the code with similar error handling)

# ================ AUTOMATED BOT FUNCTIONS ================

def automated_momentum_trader(instrument_df, symbol):
    """Enhanced momentum trader for automated mode."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, '5minute', period='1d')
            if data.empty or len(data) < 50:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate multiple indicators
            data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
            data['RSI_21'] = talib.RSI(data['close'], timeperiod=21)
            data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
            data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
            data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['close'])
            
            latest = data.iloc[-1]
            prev = data.iloc[-2]
            
            signals = []
            score = 0
            
            # EMA Crossover (30 points)
            if (latest['EMA_20'] > latest['EMA_50'] and 
                prev['EMA_20'] <= prev['EMA_50']):
                signals.append("EMA Bullish Crossover")
                score += 30
            
            # RSI Signals (25 points)
            rsi_14 = latest['RSI_14']
            if 30 < rsi_14 < 70:  # Avoid extremes
                if rsi_14 > 50:
                    signals.append("RSI Bullish")
                    score += 15
                if rsi_14 > latest['RSI_21']:
                    signals.append("RSI Positive Divergence")
                    score += 10
            
            # MACD Signals (20 points)
            if (latest['MACD'] > latest['MACD_Signal'] and 
                prev['MACD'] <= prev['MACD_Signal']):
                signals.append("MACD Bullish Crossover")
                score += 20
            
            # Volume confirmation (15 points)
            if 'volume' in data.columns and len(data) > 20:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                if data['volume'].iloc[-1] > avg_volume * 1.2:
                    signals.append("High Volume Confirmation")
                    score += 15
            
            # Price momentum (10 points)
            if len(data) >= 10:
                price_change_30min = ((latest['close'] - data['close'].iloc[-6]) / data['close'].iloc[-6]) * 100
                if price_change_30min > 1:
                    signals.append("Strong Short-term Momentum")
                    score += 10
            
            current_price = latest['close']
            risk_level = "High" if score >= 60 else "Medium" if score >= 40 else "Low"
            
            action = "HOLD"
            if score >= 50:  # Minimum threshold for trade
                action = "BUY"
            elif score <= 20:  # Very weak momentum could signal SELL
                # Check if we have an existing position to sell
                open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                              if t.get('symbol') == symbol and t.get('status') == 'OPEN']
                if open_trades and open_trades[0]['action'] == 'BUY':
                    action = "SELL"
            
            return {
                "bot_name": "Auto Momentum Trader",
                "symbol": symbol,
                "action": action,
                "quantity": 1,  # Will be calculated based on risk
                "current_price": current_price,
                "signals": signals,
                "score": score,
                "risk_level": risk_level,
                "capital_required": current_price
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

def automated_mean_reversion(instrument_df, symbol):
    """Enhanced mean reversion bot for automated mode."""
    try:
        exchange = 'NSE'
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            return {"error": f"Could not find instrument for {symbol}"}
        
        try:
            data = get_historical_data(token, '15minute', period='5d')
            if data.empty or len(data) < 100:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate Bollinger Bands and other indicators
            upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            data['BB_Upper'] = upperband
            data['BB_Middle'] = middleband
            data['BB_Lower'] = lowerband
            data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
            
            latest = data.iloc[-1]
            current_price = latest['close']
            
            signals = []
            score = 0
            
            # Bollinger Band position (40 points)
            bb_position = (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            
            if bb_position < 0.1:  # Near lower band
                signals.append("Near Lower Bollinger Band")
                score += 40
            elif bb_position > 0.9:  # Near upper band
                signals.append("Near Upper Bollinger Band")
                score += 40
            
            # RSI confirmation (30 points)
            rsi = latest['RSI_14']
            if bb_position < 0.1 and rsi < 35:  # Oversold confirmation
                signals.append("RSI Oversold Confirmation")
                score += 30
            elif bb_position > 0.9 and rsi > 65:  # Overbought confirmation
                signals.append("RSI Overbought Confirmation")
                score += 30
            
            # Distance from mean (20 points)
            distance_from_mean = abs((current_price - latest['BB_Middle']) / latest['BB_Middle']) * 100
            if distance_from_mean > 3:
                signals.append(f"Price {distance_from_mean:.1f}% from mean")
                score += 20
            
            # Volume confirmation (10 points)
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                if data['volume'].iloc[-1] > avg_volume * 1.5:
                    signals.append("High Volume Reversion Signal")
                    score += 10
            
            risk_level = "Low" if score >= 60 else "Medium" if score >= 30 else "High"
            
            action = "HOLD"
            if score >= 50 and bb_position < 0.1:  # Buy near lower band
                action = "BUY"
            elif score >= 50 and bb_position > 0.9:  # Sell near upper band
                # Check for existing position
                open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                              if t.get('symbol') == symbol and t.get('status') == 'OPEN']
                if open_trades and open_trades[0]['action'] == 'BUY':
                    action = "SELL"
            
            return {
                "bot_name": "Auto Mean Reversion",
                "symbol": symbol,
                "action": action,
                "quantity": 1,
                "current_price": current_price,
                "signals": signals,
                "score": score,
                "risk_level": risk_level,
                "capital_required": current_price
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Bot execution failed: {str(e)}"}

# Dictionary of automated bots
AUTOMATED_BOTS = {
    "Auto Momentum Trader": automated_momentum_trader,
    "Auto Mean Reversion": automated_mean_reversion
}

# ================ ALGO BOTS PAGE FUNCTIONS ================

def page_algo_bots():
    """Main algo bots page with both semi-automated and fully automated modes."""
    try:
        display_header()
        st.title("🤖 Algo Trading Bots")
        
        # Initialize automated mode
        initialize_automated_mode()
        
        instrument_df = get_instrument_df()
        if instrument_df.empty:
            st.info("Please connect to a broker to use algo bots.")
            return
        
        # Mode selection tabs
        tab1, tab2 = st.tabs(["🚀 Semi-Automated Bots", "⚡ Fully Automated Bots"])
        
        with tab1:
            page_semi_automated_bots(instrument_df)
        
        with tab2:
            page_fully_automated_bots(instrument_df)
    except Exception as e:
        handle_error(e, "page_algo_bots", "Failed to load algo bots page")

def page_semi_automated_bots(instrument_df):
    """Semi-automated bots page with comprehensive symbol selection including all stocks and commodities."""
    try:
        st.info("Run automated analysis and get trading signals. Manual confirmation required for execution.", icon="🚀")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_bot = st.selectbox(
                "Select Trading Bot",
                list(ALGO_BOTS.keys()),
                help="Choose a trading bot based on your risk appetite and trading style",
                key="semi_bot_select"
            )
            
            # Bot descriptions
            bot_descriptions = {
                "Momentum Trader": "Trades on strong price momentum and trend continuations. Medium risk.",
                "Mean Reversion": "Buys low and sells high based on statistical mean reversion. Low risk.",
                "Volatility Breakout": "Captures breakouts from low volatility periods. High risk.",
                "Value Investor": "Focuses on longer-term value and fundamental trends. Low risk.",
                "Scalper Pro": "High-frequency trading for quick, small profits. Very high risk.",
                "Trend Follower": "Rides established trends with multiple confirmation signals. Medium risk."
            }
            
            st.markdown(f"**Description:** {bot_descriptions[selected_bot]}")
        
        with col2:
            trading_capital = st.number_input(
                "Trading Capital (₹)",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100,
                help="Minimum ₹100 required",
                key="semi_capital"
            )
        
        st.markdown("---")
        
        # Symbol selection and bot execution - ENHANCED WITH ALL SYMBOLS
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("Stock & Commodity Selection")
            
            # Get all available symbols from instrument_df (stocks, commodities, indices)
            if not instrument_df.empty:
                # Filter for common segments
                available_instruments = instrument_df[
                    instrument_df['exchange'].isin(['NSE', 'BSE', 'MCX', 'NFO'])
                ].copy()
                
                # Create display names with exchange info
                available_instruments['display_name'] = available_instruments.apply(
                    lambda x: f"{x['tradingsymbol']} ({x['exchange']})", 
                    axis=1
                )
                
                # Sort by trading symbol
                available_instruments = available_instruments.sort_values('tradingsymbol')
                
                # Get unique symbols (avoid duplicates)
                unique_symbols = available_instruments.drop_duplicates('tradingsymbol')
                
                # Create selection options
                symbol_options = unique_symbols['tradingsymbol'].tolist()
                display_options = unique_symbols['display_name'].tolist()
                
                # Create mapping for display
                symbol_to_display = dict(zip(symbol_options, display_options))
                display_to_symbol = {v: k for k, v in symbol_to_display.items()}
                
                # Add popular stocks at the top for easy access
                popular_symbols = [
                    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 
                    'ICICIBANK', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC',
                    'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'TITAN', 'DMART'
                ]
                
                # Reorder options: popular first, then alphabetical
                ordered_symbols = []
                ordered_displays = []
                
                # Add popular symbols first
                for symbol in popular_symbols:
                    if symbol in symbol_to_display:
                        ordered_symbols.append(symbol)
                        ordered_displays.append(symbol_to_display[symbol])
                
                # Add remaining symbols alphabetically
                remaining_symbols = [s for s in symbol_options if s not in popular_symbols]
                for symbol in sorted(remaining_symbols):
                    ordered_symbols.append(symbol)
                    ordered_displays.append(symbol_to_display[symbol])
                
                # Commodities section - add specific commodities
                commodities = ['GOLDM', 'SILVERM', 'CRUDEOIL', 'NATURALGAS', 'COPPER', 'ZINC', 'ALUMINIUM']
                for commodity in commodities:
                    if commodity in symbol_to_display and commodity not in ordered_symbols:
                        ordered_symbols.append(commodity)
                        ordered_displays.append(symbol_to_display[commodity])
                
                # Use selectbox with search functionality
                selected_display = st.selectbox(
                    "Select Stock/Commodity",
                    options=ordered_displays,
                    index=0,  # Default to first popular symbol (RELIANCE)
                    help="Search and select from all available stocks and commodities",
                    key="semi_symbol_select"
                )
                
                # Extract symbol from display name
                selected_symbol = display_to_symbol.get(selected_display, ordered_symbols[0])
                
                # Get exchange for the selected symbol
                selected_instrument = available_instruments[
                    available_instruments['tradingsymbol'] == selected_symbol
                ].iloc[0]
                selected_exchange = selected_instrument['exchange']
                
                # Show current price
                quote_data = get_watchlist_data([{'symbol': selected_symbol, 'exchange': selected_exchange}])
                if not quote_data.empty:
                    current_price = quote_data.iloc[0]['Price']
                    change = quote_data.iloc[0]['Change']
                    change_pct = quote_data.iloc[0]['% Change']
                    
                    st.metric(
                        "Current Price", 
                        f"₹{current_price:.2f}",
                        delta=f"{change:.2f} ({change_pct:.2f}%)",
                        delta_color="normal"
                    )
                    
                    # Show additional info for commodities
                    if selected_exchange == 'MCX':
                        st.caption(f"Commodity - {selected_exchange}")
                    else:
                        st.caption(f"Equity - {selected_exchange}")
                else:
                    st.info("Price data loading...")
                    
            else:
                st.error("Instrument data not available. Please check broker connection.")
                selected_symbol = None
        
        with col4:
            st.subheader("Bot Execution")
            st.write(f"**Selected Bot:** {selected_bot}")
            st.write(f"**Available Capital:** ₹{trading_capital:,}")
            if selected_symbol:
                st.write(f"**Selected Symbol:** {selected_symbol}")
            
            if st.button("🚀 Run Trading Bot", use_container_width=True, type="primary", key="semi_run", disabled=not selected_symbol):
                with st.spinner(f"Running {selected_bot} analysis on {selected_symbol}..."):
                    bot_function = ALGO_BOTS[selected_bot]
                    bot_result = bot_function(instrument_df, selected_symbol, trading_capital)
                    
                    if bot_result and not bot_result.get("error"):
                        st.session_state.last_bot_result = bot_result
                        st.rerun()
        
        # Display bot results
        if 'last_bot_result' in st.session_state and st.session_state.last_bot_result:
            bot_result = st.session_state.last_bot_result
            
            if bot_result.get("error"):
                st.error(bot_result["error"])
            else:
                st.markdown("---")
                st.subheader("🤖 Bot Analysis Results")
                
                # Create metrics cards
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    action_color = "green" if bot_result["action"] == "BUY" else "red" if bot_result["action"] == "SELL" else "orange"
                    st.markdown(f'<div class="metric-card" style="border-color: {action_color};">'
                              f'<h3 style="color: {action_color};">{bot_result["action"]}</h3>'
                              f'<p>Recommended Action</p></div>', unsafe_allow_html=True)
                
                with col6:
                    st.metric("Quantity", bot_result["quantity"])
                
                with col7:
                    st.metric("Capital Required", f"₹{bot_result['capital_required']:.2f}")
                
                with col8:
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
                    st.markdown(f'<div class="metric-card" style="border-color: {risk_color.get(bot_result["risk_level"], "gray")};">'
                              f'<h3 style="color: {risk_color.get(bot_result["risk_level"], "gray")};">{bot_result["risk_level"]}</h3>'
                              f'<p>Risk Level</p></div>', unsafe_allow_html=True)
                
                # Display signals and execute trade
                execute_bot_trade(instrument_df, bot_result)

        # Bot performance tips
        st.markdown("---")
        st.subheader("📈 Bot Performance Tips")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("""
            **Best Practices:**
            - Start with minimum capital (₹100)
            - Use 'Value Investor' for beginners
            - 'Scalper Pro' requires constant monitoring
            - Always check signals before executing
            - Combine multiple bot recommendations
            - Test on paper trading first
            """)
        
        with tips_col2:
            st.markdown("""
            **Symbol Selection Guide:**
            - **Large Caps**: RELIANCE, TCS, HDFCBANK (Stable, lower risk)
            - **Mid Caps**: TATACONSUM, ADANIPORTS (Higher volatility)
            - **Commodities**: GOLDM, SILVERM (Different risk profile)
            - **Banking**: HDFCBANK, ICICIBANK, SBIN (Sector-specific)
            - **IT**: TCS, INFY, HCLTECH (Tech sector exposure)
            """)
    except Exception as e:
        handle_error(e, "page_semi_automated_bots", "Failed to load semi-automated bots page")

# ... (continuing with the rest of the functions with similar error handling pattern)

# ================ MAIN APP LOGIC AND AUTHENTICATION ============

def main_app():
    """The main application interface after successful login with proper theme handling."""
    try:
        apply_custom_styling()
        display_overnight_changes_bar()
        
        # Show dialogs if needed
        if st.session_state.get('show_quick_trade', False):
            quick_trade_dialog()
        
        # 2FA Check
        if st.session_state.get('profile'):
            if not st.session_state.get('two_factor_setup_complete'):
                qr_code_dialog()
                return
            if not st.session_state.get('authenticated', False):
                two_factor_dialog()
                return

        st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
        st.sidebar.caption(f"Connected via {st.session_state.broker}")
        st.sidebar.divider()
        
        st.sidebar.header("Terminal Controls")
        
        # Theme selector with better styling
        current_theme = st.session_state.theme
        new_theme = st.sidebar.radio(
            "Theme", 
            ["Dark", "Light"], 
            horizontal=True,
            index=0 if current_theme == "Dark" else 1,
            key="theme_selector"
        )
        
        if new_theme != current_theme:
            st.session_state.theme = new_theme
            st.rerun()
        
        st.session_state.terminal_mode = st.sidebar.radio(
            "Terminal Mode", 
            ["Cash", "Futures", "Options", "HFT"], 
            horizontal=True
        )
        st.sidebar.divider()
        
        # Dynamic refresh interval based on mode
        if st.session_state.terminal_mode == "HFT":
            refresh_interval = 2
            auto_refresh = True
            st.sidebar.header("HFT Mode Active")
            st.sidebar.caption(f"Refresh Interval: {refresh_interval}s")
        else:
            st.sidebar.header("Live Data")
            auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
            refresh_interval = st.sidebar.number_input(
                "Interval (s)", 
                min_value=5, 
                max_value=60, 
                value=10, 
                disabled=not auto_refresh
            )
        
        st.sidebar.divider()
        
        st.sidebar.header("Navigation")
        pages = {
            "Cash": {
                "Dashboard": page_dashboard,
                "Algo Trading Bots": page_algo_bots,
                "Premarket Pulse": page_premarket_pulse,
                "Advanced Charting": page_advanced_charting,
                "Market Scanners": page_momentum_and_trend_finder,
                "Portfolio & Risk": page_portfolio_and_risk,
                "Fundamental Analytics": page_fundamental_analytics,
                "Basket Orders": page_basket_orders,
                "Forecasting (ML)": page_forecasting_ml,
                "Algo Strategy Hub": page_algo_strategy_maker,
                "AI Discovery": page_ai_discovery,
                "AI Assistant": page_ai_assistant,
                "Economic Calendar": page_economic_calendar,
            },
            "Options": {
                "F&O Analytics": page_fo_analytics,
                "Options Strategy Builder": page_option_strategy_builder,
                "Greeks Calculator": page_greeks_calculator,
                "Portfolio & Risk": page_portfolio_and_risk,
                "AI Assistant": page_ai_assistant,
            },
            "Futures": {
                "Futures Terminal": page_futures_terminal,
                "Advanced Charting": page_advanced_charting,
                "Algo Strategy Hub": page_algo_strategy_maker,
                "Portfolio & Risk": page_portfolio_and_risk,
                "AI Assistant": page_ai_assistant,
            },
            "HFT": {
                "HFT Terminal": page_hft_terminal,
                "Portfolio & Risk": page_portfolio_and_risk,
            }
        }
        
        selection = st.sidebar.radio(
            "Go to", 
            list(pages[st.session_state.terminal_mode].keys()), 
            key='nav_selector'
        )
        
        st.sidebar.divider()
        
        # Theme preview in sidebar
        colors = get_theme_colors()
        st.sidebar.caption("Theme Preview")
        col1, col2, col3 = st.sidebar.columns(3)
        col1.markdown(f'<div style="background-color:{colors["primary"]}; width:20px; height:20px; border-radius:3px;"></div>', unsafe_allow_html=True)
        col2.markdown(f'<div style="background-color:{colors["success"]}; width:20px; height:20px; border-radius:3px;"></div>', unsafe_allow_html=True)
        col3.markdown(f'<div style="background-color:{colors["danger"]}; width:20px; height:20px; border-radius:3px;"></div>', unsafe_allow_html=True)
        
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        no_refresh_pages = ["Forecasting (ML)", "AI Assistant", "AI Discovery", "Algo Strategy Hub", "Algo Trading Bots"]
        if auto_refresh and selection not in no_refresh_pages:
            st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
        
        pages[st.session_state.terminal_mode][selection]()
    except Exception as e:
        handle_error(e, "main_app", "Failed to load main application")

# --- Application Entry Point ---
if __name__ == "__main__":
    try:
        initialize_session_state()
        
        if 'profile' in st.session_state and st.session_state.profile:
            if st.session_state.get('login_animation_complete', False):
                main_app()
            else:
                show_login_animation()
        else:
            login_page()
    except Exception as e:
        handle_error(e, "main", "Application failed to start")
