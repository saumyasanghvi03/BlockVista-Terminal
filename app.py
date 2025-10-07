# ================ 0. REQUIRED LIBRARIES ================
import streamlit as st
import pandas as pd
import pandas_ta as ta
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

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def apply_custom_styling():
    """Applies a comprehensive CSS stylesheet for professional theming."""
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

            --light-bg: #FFFFFF;
            --light-secondary-bg: #F0F2F6;
            --light-widget-bg: #F8F9FA;
            --light-border: #dee2e6;
            --light-text: #212529;
            --light-text-light: #6c757d;
            --light-green: #198754;
            --light-red: #dc3545;
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
        }

        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        h1, h2, h3, h4, h5 {
            color: var(--text-color) !important;
        }
        
        hr {
            background: var(--border-color);
        }

        .stButton>button {
            border-color: var(--border-color);
            background-color: var(--widget-bg);
            color: var(--text-color);
        }
        .stButton>button:hover {
            border-color: var(--green);
            color: var(--green);
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
            background-color: var(--widget-bg);
            border-color: var(--border-color);
            color: var(--text-color);
        }
        .stRadio>div {
            background-color: var(--widget-bg);
            border: 1px solid var(--border-color);
            padding: 8px;
            border-radius: 8px;
        }
        
        .metric-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }
        
        .trade-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }

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
        
        .market-notification {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--widget-bg);
            border: 2px solid;
            border-radius: 15px;
            padding: 2rem;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
            min-width: 300px;
            max-width: 500px;
        }
        .market-notification.open {
            border-color: var(--green);
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(40, 167, 69, 0.1) 100%);
        }
        .market-notification.warning {
            border-color: #ffc107;
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(255, 193, 7, 0.1) 100%);
        }
        .market-notification.closed {
            border-color: var(--red);
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(220, 53, 69, 0.1) 100%);
        }
        .market-notification.info {
            border-color: #17a2b8;
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(23, 162, 184, 0.1) 100%);
        }
        
        .hft-depth-bid {
            background: linear-gradient(to left, rgba(0, 128, 0, 0.3), rgba(0, 128, 0, 0.05));
            padding: 2px 5px;
        }
        .hft-depth-ask {
            background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05));
            padding: 2px 5px;
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
            0% { background-color: rgba(40, 167, 69, 0.5); }
            100% { background-color: transparent; }
        }
        @keyframes flash-red {
            0% { background-color: rgba(218, 54, 51, 0.5); }
            100% { background-color: transparent; }
        }
        
        .algo-bot-card {
            background: var(--widget-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .algo-bot-card:hover {
            border-color: var(--green);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .algo-bot-card.running {
            border-left: 5px solid var(--green);
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(40, 167, 69, 0.05) 100%);
        }
        .algo-bot-card.stopped {
            border-left: 5px solid var(--red);
        }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    js_theme = f"""
    <script>
        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add('{st.session_state.theme.lower()}-theme');
    </script>
    """
    st.components.v1.html(js_theme, height=0)

# ================ ENHANCED DATA COLLECTION MODULE ================

# Enhanced data sources with multiple fallbacks
ENHANCED_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "yfinance_ticker": "^NSEI",
        "tradingsymbol": "NIFTY 50", 
        "exchange": "NSE",
        "fallback": "yfinance"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "yfinance_ticker": "^NSEBANK", 
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NFO",
        "fallback": "yfinance"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "yfinance_ticker": "FINNIFTY.NS",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NFO",
        "fallback": "yfinance"
    },
    "GOLD": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv",
        "yfinance_ticker": "GC=F",
        "tradingsymbol": "GOLDM",
        "exchange": "MCX", 
        "fallback": "yfinance"
    },
    "USDINR": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv",
        "yfinance_ticker": "INR=X",
        "tradingsymbol": "USDINR",
        "exchange": "CDS",
        "fallback": "yfinance"
    },
    "SENSEX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv",
        "yfinance_ticker": "^BSESN",
        "tradingsymbol": "SENSEX",
        "exchange": "BSE",
        "fallback": "yfinance"
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
        "yfinance_ticker": "^GSPC",
        "tradingsymbol": "^GSPC",
        "exchange": "yfinance",
        "fallback": "yfinance"
    },
    "NIFTY MIDCAP 100": {
        "yfinance_ticker": "^CNXMD",
        "tradingsymbol": "NIFTYMID100",
        "exchange": "NSE", 
        "fallback": "yfinance"
    }
}

def download_hourly_data_yfinance(ticker, period="2y"):
    """Download hourly data using yfinance with error handling."""
    try:
        data = yf.download(ticker, period=period, interval="1h")
        if data.empty:
            return pd.DataFrame()
        
        # Reset index and rename columns
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure standard column names
        column_mapping = {
            'date': 'datetime',
            'open': 'open', 
            'high': 'high',
            'low': 'low', 
            'close': 'close',
            'volume': 'volume'
        }
        
        data = data.rename(columns=column_mapping)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        
        return data
        
    except Exception as e:
        st.error(f"Error downloading {ticker} from yfinance: {e}")
        return pd.DataFrame()

def download_from_github(url):
    """Download CSV data from GitHub with error handling."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Handle different date column names
        date_columns = ['date', 'datetime', 'time', 'timestamp']
        date_col = next((col for col in date_columns if col in data.columns), None)
        
        if date_col:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data.set_index(date_col, inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return data
        
    except Exception as e:
        st.error(f"Error downloading from GitHub {url}: {e}")
        return pd.DataFrame()

def get_enhanced_historical_data(instrument_name, data_type="hourly"):
    """
    Enhanced historical data fetcher with multiple fallback sources.
    
    Parameters:
    - instrument_name: Name of the instrument from ENHANCED_DATA_SOURCES
    - data_type: "hourly" or "daily"
    """
    source_info = ENHANCED_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()
    
    data = pd.DataFrame()
    
    # Try GitHub first if available
    if "github_url" in source_info:
        data = download_from_github(source_info["github_url"])
    
    # If GitHub fails or we need hourly data, try yfinance
    if data.empty or data_type == "hourly":
        yf_ticker = source_info.get("yfinance_ticker")
        if yf_ticker:
            period = "2y" if data_type == "hourly" else "max"
            interval = "1h" if data_type == "hourly" else "1d"
            
            try:
                yf_data = yf.download(yf_ticker, period=period, interval=interval)
                if not yf_data.empty:
                    yf_data = yf_data.reset_index()
                    yf_data.columns = [col.lower() for col in yf_data.columns]
                    
                    # Handle different date column names in yfinance
                    if 'date' in yf_data.columns:
                        yf_data.rename(columns={'date': 'datetime'}, inplace=True)
                    elif 'index' in yf_data.columns:
                        yf_data.rename(columns={'index': 'datetime'}, inplace=True)
                    
                    yf_data['datetime'] = pd.to_datetime(yf_data['datetime'])
                    yf_data.set_index('datetime', inplace=True)
                    
                    # If we have existing data, merge them
                    if not data.empty:
                        # Keep yfinance data for periods not in GitHub data
                        combined_data = pd.concat([data, yf_data])
                        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                        data = combined_data.sort_index()
                    else:
                        data = yf_data
            except Exception as e:
                st.error(f"Error downloading {instrument_name} from yfinance: {e}")
    
    # Add technical indicators if we have data
    if not data.empty:
        try:
            # Basic technical indicators
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # RSI
                data['rsi_14'] = ta.rsi(data['close'], length=14)
                
                # Moving averages
                data['sma_20'] = ta.sma(data['close'], length=20)
                data['ema_12'] = ta.ema(data['close'], length=12)
                data['ema_26'] = ta.ema(data['close'], length=26)
                
                # MACD
                macd = ta.macd(data['close'])
                if macd is not None:
                    data = pd.concat([data, macd], axis=1)
                
                # Bollinger Bands
                bb = ta.bbands(data['close'])
                if bb is not None:
                    data = pd.concat([data, bb], axis=1)
                    
        except Exception as e:
            st.warning(f"Could not add technical indicators for {instrument_name}: {e}")
    
    return data

# Replace the existing ML_DATA_SOURCES with enhanced version
ML_DATA_SOURCES = ENHANCED_DATA_SOURCES

# ================ 1.5 INITIALIZATION ========================
def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'broker' not in st.session_state: st.session_state.broker = None
    if 'kite' not in st.session_state: st.session_state.kite = None
    if 'profile' not in st.session_state: st.session_state.profile = None
    if 'login_animation_complete' not in st.session_state: st.session_state.login_animation_complete = False
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'two_factor_setup_complete' not in st.session_state: st.session_state.two_factor_setup_complete = False
    if 'pyotp_secret' not in st.session_state: st.session_state.pyotp_secret = None
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'watchlists' not in st.session_state:
        st.session_state.watchlists = {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
            "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
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
    if 'fundamental_companies' not in st.session_state: st.session_state.fundamental_companies = []
    
    # HFT Terminal specific state variables
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []
    if 'market_notifications_shown' not in st.session_state: st.session_state.market_notifications_shown = {}
    if 'show_2fa_dialog' not in st.session_state: st.session_state.show_2fa_dialog = False
    if 'show_qr_dialog' not in st.session_state: st.session_state.show_qr_dialog = False
    
    # Algo Bot specific state variables
    if 'algo_bots_running' not in st.session_state: st.session_state.algo_bots_running = {}
    if 'algo_bot_capital' not in st.session_state: st.session_state.algo_bot_capital = 100000
    if 'algo_bot_positions' not in st.session_state: st.session_state.algo_bot_positions = {}

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    if 'show_quick_trade' not in st.session_state:
        st.session_state.show_quick_trade = False
    
    if symbol or st.button("Quick Trade", key="quick_trade_btn"):
        st.session_state.show_quick_trade = True
    
    if st.session_state.show_quick_trade:
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
                instrument_df = get_instrument_df()
                place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
                st.session_state.show_quick_trade = False
                st.rerun()
            else:
                st.warning("Please fill in all fields.")
        
        if col2.button("Cancel", use_container_width=True):
            st.session_state.show_quick_trade = False
            st.rerun()

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    holidays_by_year = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
        2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    }
    return holidays_by_year.get(year, [])

def get_market_status():
    """Checks if the Indian stock market is open."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if market_open_time <= now.time() <= market_close_time:
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def check_market_timing_notifications():
    """Checks for market timing events and shows notifications."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    today_str = now.strftime('%Y-%m-%d')
    
    # Initialize notification tracking for today
    if 'market_notifications_shown' not in st.session_state:
        st.session_state.market_notifications_shown = {}
    if today_str not in st.session_state.market_notifications_shown:
        st.session_state.market_notifications_shown[today_str] = {}
    
    notifications_shown_today = st.session_state.market_notifications_shown[today_str]
    
    # Market timing events
    market_events = [
        {"time": time(9, 15), "type": "open", "title": "Market Open", "message": "Indian Stock Market is now open for trading", "class": "open", "duration": 10},
        {"time": time(9, 45), "type": "ipo_preopen", "title": "IPO Pre-Opening Window", "message": "IPO pre-opening session has started. Orders can be placed but will be executed after 10:00 AM", "class": "info", "duration": 10},
        {"time": time(10, 0), "type": "ipo_open", "title": "IPO Trading Starts", "message": "IPO orders are now being executed in the market", "class": "info", "duration": 10},
        {"time": time(15, 15), "type": "closing_warning", "title": "Market Closing Soon", "message": "Market will close in 15 minutes. Place your final orders", "class": "warning", "duration": 10},
        {"time": time(15, 30), "type": "closed", "title": "Market Closed", "message": "Indian Stock Market is now closed for the day", "class": "closed", "duration": 10}
    ]
    
    # Check if we should show any notifications
    for event in market_events:
        event_key = f"{today_str}_{event['type']}"
        
        # Check if it's time for this event and we haven't shown it yet
        if (current_time >= event["time"] and 
            current_time <= (datetime.combine(now.date(), event["time"]) + timedelta(seconds=event["duration"])).time() and
            event_key not in notifications_shown_today):
            
            # Show the notification
            show_market_notification(event["title"], event["message"], event["class"], event["duration"])
            
            # Mark as shown
            notifications_shown_today[event_key] = True
            break

def show_market_notification(title, message, notification_class, duration=10):
    """Displays a market timing notification."""
    notification_html = f"""
    <div class="market-notification {notification_class}">
        <h3>{title}</h3>
        <p>{message}</p>
        <small>This notification will auto-close in {duration} seconds</small>
    </div>
    """
    st.markdown(notification_html, unsafe_allow_html=True)
    st.session_state.market_notification_time = datetime.now()
    st.session_state.market_notification_duration = duration

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
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
            quick_trade_dialog()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            quick_trade_dialog()

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    
def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
    overnight_tickers = {"GIFT NIFTY": "IN=F", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
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

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    """Generates a Plotly chart with various chart types and overlays."""
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

    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Bar'))
    else:
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
        
    bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
    bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
    if bbl_col and bbu_col:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbl_col], line=dict(color='rgba(135,206,250,0.5)', width=1), name='Lower Band'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbu_col], line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='Upper Band'))
        
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        if conf_int_df is not None:
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
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

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API."""
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
            
            # Add technical indicators
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True); df.ta.coppock(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True); df.ta.macd(append=True); df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
            except Exception:
                pass 
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

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
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

@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth (order book) for a given instrument."""
    client = get_broker_client()
    if not client or not instrument_token:
        return None
    try:
        # Use quote method to get market depth data instead of depth
        quote_data = client.quote([str(instrument_token)])
        instrument_key = str(instrument_token)
        if instrument_key in quote_data:
            return quote_data[instrument_key].get('depth')
        return None
    except Exception as e:
        st.toast(f"Error fetching market depth: {e}", icon="⚠️")
        return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for a given underlying."""
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

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio positions and holdings from the broker."""
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

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order through the broker's API."""
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

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
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

def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE function to remove sklearn dependency."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data, forecast_steps=30):
    """Trains a Seasonal ARIMA model for time series forecasting."""
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

@st.cache_data
def load_and_combine_data(instrument_name):
    """Enhanced version using the new data fetcher."""
    return get_enhanced_historical_data(instrument_name, "daily")

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price and Greeks."""
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError):
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty: return {}
    latest = df.iloc[-1].copy()
    latest.index = latest.index.str.lower()
    interpretation = {}
    
    rsi_col = next((col for col in latest.index if 'rsi' in col), None)
    stoch_k_col = next((col for col in latest.index if 'stok' in col), None)
    macd_col = next((col for col in latest.index if 'macd' in col and 'macds' not in col and 'macdh' not in col), None)
    signal_col = next((col for col in latest.index if 'macds' in col), None)
    adx_col = next((col for col in latest.index if 'adx' in col), None)

    r = latest.get(rsi_col)
    if r is not None:
        interpretation['RSI (14)'] = "Overbought (Bearish)" if r > 70 else "Oversold (Bullish)" if r < 30 else "Neutral"
    
    stoch_k = latest.get(stoch_k_col)
    if stoch_k is not None:
        interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
    
    macd = latest.get(macd_col)
    signal = latest.get(signal_col)
    if macd is not None and signal is not None:
        interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    
    adx = latest.get(adx_col)
    if adx is not None:
        interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
    
    return interpretation

# ================ 4. HNI & PRO TRADER FEATURES ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
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

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("'sensex_sectors.csv' not found. Sector allocation will be unavailable.")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to highlight ITM/OTM in the options chain."""
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

def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    if 'show_most_active' not in st.session_state:
        st.session_state.show_most_active = False
    
    if st.button("Most Active Options", key="most_active_btn"):
        st.session_state.show_most_active = True
    
    if st.session_state.show_most_active:
        st.subheader(f"Most Active {underlying} Options (By Volume)")
        with st.spinner("Fetching data..."):
            active_df = get_most_active_options(underlying, instrument_df)
            if not active_df.empty:
                st.dataframe(active_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not retrieve data for most active options.")
        
        if st.button("Close", key="close_most_active"):
            st.session_state.show_most_active = False
            st.rerun()

def get_most_active_options(underlying, instrument_df):
    """Fetches the most active options by volume for a given underlying."""
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
        return df_sorted.head(10)

    except Exception as e:
        st.error(f"Could not fetch most active options: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        data_yf = yf.download(list(tickers.values()), period="5d")
        if data_yf.empty:
            return pd.DataFrame()

        data = []
        for ticker_name, yf_ticker_name in tickers.items():
            if len(tickers) > 1:
                hist = data_yf.loc[:, (slice(None), yf_ticker_name)]
                hist.columns = hist.columns.droplevel(1)
            else:
                hist = data_yf

            if len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                data.append({'Ticker': ticker_name, 'Price': last_price, 'Change': change, '% Change': pct_change})
            else:
                data.append({'Ticker': ticker_name, 'Price': np.nan, 'Change': np.nan, '% Change': np.nan})

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS ============

# --- Bharatiya Market Pulse (BMP) Functions ---
def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns the score and a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30:
        return 50, "Calculating...", "#cccccc"
    
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    vix_min, vix_max = lookback_df['vix_value'].min(), lookback_df['vix_value'].max()
    vix_norm = 100 - (((vix_value - vix_min) / (vix_max - vix_min)) * 100) if (vix_max - vix_min) > 0 else 50

    bmp_score = (0.40 * nifty_norm) + (0.40 * sensex_norm) + (0.20 * vix_norm)
    bmp_score = min(100, max(0, bmp_score))

    if bmp_score >= 80:
        label, color = "Bharat Udaan (Very Bullish)", "#00b300"
    elif bmp_score >= 60:
        label, color = "Bharat Pragati (Bullish)", "#33cc33"
    elif bmp_score >= 40:
        label, color = "Bharat Santulan (Neutral)", "#ffcc00"
    elif bmp_score >= 20:
        label, color = "Bharat Sanket (Bearish)", "#ff6600"
    else:
        label, color = "Bharat Mandhi (Very Bearish)", "#ff0000"

    return bmp_score, label, color

@st.cache_data(ttl=300)
def get_nifty50_constituents(instrument_df):
    """Fetches the list of NIFTY 50 stocks by filtering the Kite API instrument list."""
    if instrument_df.empty:
        return pd.DataFrame()
    
    nifty50_symbols = [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'HINDUNILVR', 'ITC', 
        'LT', 'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT', 
        'AXISBANK', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'M&M', 'NESTLEIND',
        'ADANIENT', 'TATASTEEL', 'INDUSINDBK', 'TECHM', 'NTPC', 'MARUTI', 
        'BAJAJ-AUTO', 'POWERGRID', 'HCLTECH', 'ADANIPORTS', 'BPCL', 'COALINDIA', 
        'EICHERMOT', 'GRASIM', 'JSWSTEEL', 'SHREECEM', 'HEROMOTOCO', 'HINDALCO',
        'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'SBILIFE',
        'TATAMOTORS', 'BRITANNIA', 'DIVISLAB', 'BAJAJFINSV', 'SUNPHARMA', 'HDFCLIFE'
    ]
    
    nifty_constituents = instrument_df[
        (instrument_df['tradingsymbol'].isin(nifty50_symbols)) & 
        (instrument_df['segment'] == 'NSE')
    ].copy()

    constituents_df = pd.DataFrame({
        'Symbol': nifty_constituents['tradingsymbol'],
        'Name': nifty_constituents['tradingsymbol']
    })
    
    return constituents_df.drop_duplicates(subset='Symbol').head(15)

def create_nifty_heatmap(instrument_df):
    """Generates a Plotly Treemap for NIFTY 50 stocks."""
    constituents_df = get_nifty50_constituents(instrument_df)
    if constituents_df.empty:
        return go.Figure()
    
    symbols_with_exchange = [{'symbol': s, 'exchange': 'NSE'} for s in constituents_df['Symbol'].tolist()]
    live_data = get_watchlist_data(symbols_with_exchange)
    
    if live_data.empty:
        return go.Figure()
        
    full_data = pd.merge(live_data, constituents_df, left_on='Ticker', right_on='Symbol', how='left')
    full_data['size'] = full_data['Price'].astype(float) * 1000
    
    fig = go.Figure(go.Treemap(
        labels=full_data['Ticker'],
        parents=[''] * len(full_data),
        values=full_data['size'],
        marker=dict(
            colorscale='RdYlGn',
            colors=full_data['% Change'],
            colorbar=dict(title="% Change"),
        ),
        text=full_data['Ticker'],
        textinfo="label",
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{customdata[1]:.2f}%<extra></extra>',
        customdata=np.column_stack([full_data['Price'], full_data['% Change']])
    ))

    fig.update_layout(title="NIFTY 50 Heatmap (Live)")
    return fig

@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using a more reliable yfinance ticker."""
    try:
        data = yf.download("IN=F", period="1d", interval="1m")
        if not data.empty:
            return data
    except Exception:
        pass
    return pd.DataFrame()

def page_dashboard():
    """A completely redesigned 'Trader UI' Dashboard."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    # BMP Calculation and Display
    bmp_col, heatmap_col = st.columns([1, 1], gap="large")
    with bmp_col:
        st.subheader("Bharatiya Market Pulse (BMP)")
        if not index_data.empty:
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
            
            nifty_hist = get_historical_data(get_instrument_token('NIFTY 50', instrument_df, 'NSE'), 'day', period='1y')
            sensex_hist = get_historical_data(get_instrument_token('SENSEX', instrument_df, 'BSE'), 'day', period='1y')
            vix_hist = get_historical_data(get_instrument_token('INDIA VIX', instrument_df, 'NSE'), 'day', period='1y')
            
            if not nifty_hist.empty and not sensex_hist.empty and not vix_hist.empty:
                lookback_data = pd.DataFrame({
                    'nifty_change': nifty_hist['close'].pct_change() * 100,
                    'sensex_change': sensex_hist['close'].pct_change() * 100,
                    'vix_value': vix_hist['close']
                }).dropna()
                
                bmp_score, bmp_label, bmp_color = get_bmp_score_and_label(nifty_row['% Change'], sensex_row['% Change'], vix_row['Price'], lookback_data)
                
                st.markdown(f'<div class="metric-card" style="border-color:{bmp_color};"><h3>{bmp_score:.2f}</h3><p style="color:{bmp_color}; font-weight:bold;">{bmp_label}</p><small>Proprietary score from NIFTY, SENSEX, and India VIX.</small></div>', unsafe_allow_html=True)
                with st.expander("What do the BMP scores mean?"):
                    st.markdown("""
                    - **80-100 (Bharat Udaan):** Very Strong Bullish Momentum.
                    - **60-80 (Bharat Pragati):** Moderately Bullish Sentiment.
                    - **40-60 (Bharat Santulan):** Neutral or Sideways Market.
                    - **20-40 (Bharat Sanket):** Moderately Bearish Sentiment.
                    - **0-20 (Bharat Mandhi):** Very Strong Bearish Momentum.
                    """)
            else:
                st.info("BMP data is loading...")
        else:
            st.info("BMP data is loading...")
    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        st.plotly_chart(create_nifty_heatmap(instrument_df), use_container_width=True)

    st.markdown("---")
    
    # --- Middle Row: Main Content Area ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            st.session_state.active_watchlist = st.radio(
                "Select Watchlist",
                options=st.session_state.watchlists.keys(),
                horizontal=True,
                label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 15:
                            st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange})
                            st.rerun()
                        else:
                            st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            # Display watchlist
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", unsafe_allow_html=True)
                    w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                    
                    quantity = w_cols[2].number_input("Qty", min_value=1, step=1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                    
                    if w_cols[3].button("B", key=f"buy_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                    if w_cols[4].button("S", key=f"sell_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
                    if w_cols[5].button("🗑️", key=f"del_{row['Ticker']}", use_container_width=True):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != row['Ticker']]
                        st.rerun()
                st.markdown("---")

        with tab2:
            st.subheader("My Portfolio")
            _, holdings_df, total_pnl, total_investment = get_portfolio()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
            st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            with st.expander("View Holdings"):
                if not holdings_df.empty:
                    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings found.")
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
    
    # --- Bottom Row: Live Ticker Tape ---
    ticker_symbols = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist'), [])
    
    if ticker_symbols:
        ticker_data = get_watchlist_data(ticker_symbols)
        
        if not ticker_data.empty:
            ticker_html = "".join([
                f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {'#28a745' if item['Change'] > 0 else '#FF4B4B'};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>"
                for _, item in ticker_data.iterrows()
            ])
            
            st.markdown(f"""
            <style>
                @keyframes marquee {{
                    0%   {{ transform: translate(100%, 0); }}
                    100% {{ transform: translate(-100%, 0); }}
                }}
                .marquee-container {{
                    width: 100%;
                    overflow: hidden;
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    background-color: #1a1a1a;
                    border-top: 1px solid #333;
                    padding: 5px 0;
                    white-space: nowrap;
                }}
                .marquee-content {{
                    display: inline-block;
                    padding-left: 100%;
                    animation: marquee 35s linear infinite;
                }}
            </style>
            <div class="marquee-container">
                <div class="marquee-content">
                    {ticker_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    st.subheader("Chart Layout")
    layout_option = st.radio("Select Layout", ["Single Chart", "2 Charts", "4 Charts", "6 Charts"], horizontal=True)
    
    chart_counts = {"Single Chart": 1, "2 Charts": 2, "4 Charts": 4, "6 Charts": 6}
    num_charts = chart_counts[layout_option]
    
    st.markdown("---")
    
    if num_charts == 1:
        render_chart_controls(0, instrument_df)
    elif num_charts == 2:
        cols = st.columns(2)
        for i, col in enumerate(cols):
            with col:
                render_chart_controls(i, instrument_df)
    elif num_charts == 4:
        for i in range(2):
            cols = st.columns(2)
            with cols[0]:
                render_chart_controls(i * 2, instrument_df)
            with cols[1]:
                render_chart_controls(i * 2 + 1, instrument_df)
    elif num_charts == 6:
        for i in range(2):
            cols = st.columns(3)
            with cols[0]:
                render_chart_controls(i * 3, instrument_df)
            with cols[1]:
                render_chart_controls(i * 3 + 1, instrument_df)
            with cols[2]:
                render_chart_controls(i * 3 + 2, instrument_df)

def render_chart_controls(i, instrument_df):
    """Helper function to render controls for a single chart."""
    with st.container(border=True):
        st.subheader(f"Chart {i+1}")
        
        chart_cols = st.columns(4)
        ticker = chart_cols[0].text_input("Symbol", "NIFTY 50", key=f"ticker_{i}").upper()
        period = chart_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key=f"period_{i}")
        interval = chart_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key=f"interval_{i}")
        chart_type = chart_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{i}")

        token = get_instrument_token(ticker, instrument_df)
        data = get_historical_data(token, interval, period=period)

        if data.empty:
            st.warning(f"No data to display for {ticker} with selected parameters.")
        else:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True, key=f"chart_{i}")

            order_cols = st.columns([2,1,1,1])
            order_cols[0].markdown("**Quick Order**")
            quantity = order_cols[1].number_input("Qty", min_value=1, step=1, key=f"qty_{i}", label_visibility="collapsed")
            
            if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'BUY', 'MIS')
            if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'SELL', 'MIS')

def page_premarket_pulse():
    """Global market overview and premarket indicators with a trader-focused UI."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Premarket & Global Cues")
    st.markdown("---")

    st.subheader("Global Market Snapshot")
    global_tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "Hang Seng": "^HSI"}
    global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_tickers))
        for i, (name, ticker_symbol) in enumerate(global_tickers.items()):
            data_row = global_data[global_data['Ticker'] == name]
            if not data_row.empty:
                price = data_row.iloc[0]['Price']
                change = data_row.iloc[0]['% Change']
                if not np.isnan(price):
                    cols[i].metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                else:
                    cols[i].metric(label=name, value="N/A", delta="--")
    else:
        st.info("Loading global market data...")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY 50 Futures (Live Proxy)")
        gift_data = get_gift_nifty_data()
        if not gift_data.empty:
            st.plotly_chart(create_chart(gift_data, "NIFTY 50 Futures (Proxy)"), use_container_width=True)
        else:
            st.warning("Could not load NIFTY 50 Futures chart data.")
            
    with col2:
        st.subheader("Key Asian Markets")
        asian_tickers = {"Nikkei 225": "^N225", "Hang Seng": "^HSI"}
        asian_data = get_global_indices_data(asian_tickers)
        if not asian_data.empty:
            for name, ticker_symbol in asian_tickers.items():
                data_row = asian_data[asian_data['Ticker'] == name]
                if not data_row.empty:
                    price = data_row.iloc[0]['Price']
                    change = data_row.iloc[0]['% Change']
                    if not np.isnan(price):
                        st.metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                    else:
                        st.metric(label=name, value="N/A", delta="--")
        else:
            st.info("Loading Asian market data...")

    st.markdown("---")

    st.subheader("Latest Market News")
    news_df = fetch_and_analyze_news()
    if not news_df.empty:
        for _, news in news_df.head(10).iterrows():
            sentiment_score = news['sentiment']
            if sentiment_score > 0.2:
                icon = "🔼"
            elif sentiment_score < -0.2:
                icon = "🔽"
            else:
                icon = "▶️"
            st.markdown(f"**{icon} [{news['title']}]({news['link']})** - *{news['source']}*")
    else:
        st.info("News data is loading...")

def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("F&O Analytics Hub")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access F&O Analytics.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "PCR Analysis", "Volatility & OI Analysis"])
    
    with tab1:
        st.subheader("Live Options Chain")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            st.session_state.underlying_pcr = underlying 
        
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)

        if not chain_df.empty:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")
            with col3:
                if st.button("Most Active Options", use_container_width=True):
                    show_most_active_dialog(underlying, instrument_df)

            st.dataframe(
                style_option_chain(chain_df, underlying_ltp).format({
                    'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}',
                    'STRIKE': '₹{:.0f}',
                    'CALL OI': '{:,.0f}', 'PUT OI': '{:,.0f}'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Could not load options chain data.")
    
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        
        chain_df, _, _, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)
        if not chain_df.empty and 'CALL OI' in chain_df.columns:
            total_ce_oi = chain_df['CALL OI'].sum()
            total_pe_oi = chain_df['PUT OI'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total CE OI", f"{total_ce_oi:,.0f}")
            col2.metric("Total PE OI", f"{total_pe_oi:,.0f}")
            col3.metric("PCR", f"{pcr:.2f}")
            
            if pcr > 1.3:
                st.success("High PCR suggests potential bearish sentiment (more Puts bought for hedging/speculation).")
            elif pcr < 0.7:
                st.error("Low PCR suggests potential bullish sentiment (more Calls bought).")
            else:
                st.info("PCR indicates neutral sentiment.")
        else:
            st.info("PCR data is loading... Select an underlying in the 'Options Chain' tab first.")
    
    with tab3:
        st.subheader("Volatility & Open Interest Surface")
        st.info("Real-time implied volatility and OI analysis for options contracts.")

        chain_df, expiry, underlying_ltp, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)

        if not chain_df.empty and expiry and underlying_ltp > 0:
            T = (expiry - datetime.now().date()).days / 365.0
            r = 0.07

            with st.spinner("Calculating Implied Volatility..."):
                chain_df['IV_CE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['CALL LTP'], 'call') * 100,
                    axis=1
                )
                chain_df['IV_PE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['PUT LTP'], 'put') * 100,
                    axis=1
                )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_CE'], mode='lines+markers', name='Call IV', line=dict(color='cyan')), secondary_y=False)
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_PE'], mode='lines+markers', name='Put IV', line=dict(color='magenta')), secondary_y=False)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['CALL OI'], name='Call OI', marker_color='rgba(0, 255, 255, 0.4)'), secondary_y=True)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['PUT OI'], name='Put OI', marker_color='rgba(255, 0, 255, 0.4)'), secondary_y=True)

            fig.update_layout(
                title_text=f"{st.session_state.get('underlying_pcr', 'NIFTY')} IV & OI Profile for {expiry.strftime('%d %b %Y')}",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select an underlying and expiry in the 'Options Chain' tab to view the volatility surface.")

def page_forecasting_ml():
    """A page for advanced ML forecasting with an improved UI and corrected formulas."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Advanced ML Forecasting")
    st.info("Train a Seasonal ARIMA model to forecast future prices. This is for educational purposes and not financial advice.", icon="🧠")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument")
        
        forecast_durations = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
        duration_key = st.radio("Forecast Duration", list(forecast_durations.keys()), horizontal=True, key="ml_duration")
        forecast_steps = forecast_durations[duration_key]

        if st.button("Train Model & Forecast", use_container_width=True, type="primary"):
            with st.spinner(f"Loading data and training model for {instrument_name}..."):
                data = load_and_combine_data(instrument_name)
                if data.empty or len(data) < 100:
                    st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
                else:
                    forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                    st.session_state.update({
                        'ml_forecast_df': forecast_df,
                        'ml_backtest_df': backtest_df,
                        'ml_conf_int_df': conf_int_df,
                        'ml_instrument_name': instrument_name,
                        'ml_historical_data': data,
                        'ml_duration_key': duration_key
                    })
                    st.success("Model trained successfully!")

    with col2:
        if 'ml_instrument_name' in st.session_state:
            instrument_name = st.session_state.ml_instrument_name
            st.subheader(f"Forecast Results for {instrument_name}")

            forecast_df = st.session_state.get('ml_forecast_df')
            backtest_df = st.session_state.get('ml_backtest_df')
            conf_int_df = st.session_state.get('ml_conf_int_df')
            data = st.session_state.get('ml_historical_data')
            duration_key = st.session_state.get('ml_duration_key')

            if forecast_df is not None and backtest_df is not None and data is not None and conf_int_df is not None:
                fig = create_chart(data.tail(252), instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Backtest Prediction', line=dict(color='orange', dash='dot')))
                fig.update_layout(title=f"{instrument_name} Forecast vs. Historical Data")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Model Performance (Backtest)")
                
                backtest_durations = {"Full History": len(backtest_df), "Last Year": 252, "6 Months": 126, "3 Months": 63}
                backtest_duration_key = st.selectbox("Select Backtest Period", list(backtest_durations.keys()))
                backtest_period = backtest_durations[backtest_duration_key]
                
                display_df = backtest_df.tail(backtest_period)

                mape = mean_absolute_percentage_error(display_df['Actual'], display_df['Predicted'])
                
                metric_cols = st.columns(2)
                metric_cols[0].metric(f"Accuracy ({backtest_duration_key})", f"{100 - mape:.2f}%")
                metric_cols[1].metric(f"MAPE ({backtest_duration_key})", f"{mape:.2f}%")

                with st.expander(f"View {duration_key} Forecast Data"):
                    display_df_forecast = forecast_df.join(conf_int_df)
                    st.dataframe(display_df_forecast.style.format("₹{:.2f}"), use_container_width=True)
            else:
                st.info("Train a model to see the forecast results.")
        else:
            st.info("Select an instrument and run the forecast to see results.")

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty and positions_df.empty:
        st.info("No holdings or positions found to analyze.")
        return

    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Live Order Book"])
    
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        else:
            st.info("No open positions for the day.")
    
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            st.markdown("---")

            st.subheader("Portfolio Allocation")
            
            sector_df = get_sector_data()
            
            holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
            
            if not holdings_df.empty and sector_df is not None:
                holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left')
                if 'Sector' not in holdings_df.columns:
                    holdings_df['Sector'] = 'Uncategorized'
                holdings_df['Sector'].fillna('Uncategorized', inplace=True)
            else:
                holdings_df['Sector'] = 'Uncategorized'
            
            col1_alloc, col2_alloc = st.columns(2)
            
            with col1_alloc:
                st.subheader("Stock-wise Allocation")
                fig_stock = go.Figure(data=[go.Pie(
                    labels=holdings_df['tradingsymbol'],
                    values=holdings_df['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == "Dark" else 'plotly_white')
                st.plotly_chart(fig_stock, use_container_width=True)
                
            if 'Sector' in holdings_df.columns:
                with col2_alloc:
                    st.subheader("Sector-wise Allocation")
                    sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                    fig_sector = go.Figure(data=[go.Pie(
                        labels=sector_allocation['Sector'],
                        values=sector_allocation['current_value'],
                        hole=.3,
                        textinfo='label+percent'
                    )])
                    fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == "Dark" else 'plotly_white')
                    st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No holdings found.")

    with tab3:
        st.subheader("Live Order Book")
        if client:
            try:
                orders = client.orders()
                if orders:
                    orders_df = pd.DataFrame(orders)
                    st.dataframe(orders_df[[
                        'order_timestamp', 'tradingsymbol', 'transaction_type',
                        'order_type', 'quantity', 'average_price', 'status'
                    ]], use_container_width=True, hide_index=True)
                else:
                    st.info("No orders placed today.")
            except Exception as e:
                st.error(f"Failed to fetch order book: {e}")
        else:
            st.info("Broker not connected.")

def page_ai_assistant():
    """An AI-powered assistant for portfolio management and market queries."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Portfolio-Aware Assistant")
    instrument_df = get_instrument_df()

    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your portfolio or the markets today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_lower = prompt.lower()
                response = "I can help with your portfolio, orders, and live market data. For example, try asking 'What are my positions?' or 'Show me the option chain for BANKNIFTY'."
                client = get_broker_client()
                if not client:
                    response = "I am not connected to your broker. Please log in first."
                
                elif any(word in prompt_lower for word in ["holdings", "investments"]):
                    _, holdings_df, _, _ = get_portfolio()
                    response = f"Here are your current holdings:\n```\n{tabulate(holdings_df, headers='keys', tablefmt='psql')}\n```" if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    response = f"Your total P&L is ₹{total_pnl:,.2f}. Here are your positions:\n```\n{tabulate(positions_df, headers='keys', tablefmt='psql')}\n```" if not positions_df.empty else "You have no open positions."
                elif "orders" in prompt_lower:
                    orders = client.orders()
                    response = f"Here are today's orders:\n```\n{tabulate(pd.DataFrame(orders), headers='keys', tablefmt='psql')}\n```" if orders else "You have no orders for the day."
                elif any(word in prompt_lower for word in ["funds", "margin", "balance"]):
                    funds = client.margins()
                    response = f"Available Funds:\n- Equity: ₹{funds['equity']['available']['live_balance']:,.2f}\n- Commodity: ₹{funds['commodity']['available']['live_balance']:,.2f}"
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    try:
                        ticker = prompt.split(" of ")[-1].strip().upper()
                        instrument = instrument_df[instrument_df['tradingsymbol'] == ticker]
                        if not instrument.empty:
                            exchange = instrument.iloc[0]['exchange']
                            ltp_data = get_watchlist_data([{'symbol': ticker, 'exchange': exchange}])
                            price = ltp_data.iloc[0]['Price'] if not ltp_data.empty else "N/A"
                            response = f"The current price of {ticker} is {price}."
                        else:
                            response = f"I could not find the ticker '{ticker}'. Please check the symbol."
                    except Exception:
                        response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                
                elif "buy" in prompt_lower or "sell" in prompt_lower:
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9\-\&_]+)', prompt_lower)
                    if match:
                        trans_type = match.group(1).upper()
                        quantity = int(match.group(2))
                        symbol = match.group(3).upper()

                        st.session_state.last_order_details = {
                            "symbol": symbol,
                            "quantity": quantity,
                            "transaction_type": trans_type,
                            "confirmed": False
                        }
                        
                        response = f"I can place a {trans_type} order for {quantity} shares of {symbol}. Please confirm by typing 'confirm'."
                    else:
                        response = "I couldn't understand the order. Please use a format like 'Buy 100 shares of RELIANCE'."
                elif prompt_lower == "confirm" and "last_order_details" in st.session_state and not st.session_state.last_order_details["confirmed"]:
                    order_details = st.session_state.last_order_details
                    place_order(instrument_df, order_details['symbol'], order_details['quantity'], 'MARKET', order_details['transaction_type'], 'MIS')
                    order_details['confirmed'] = True
                    response = f"Confirmed and placed {order_details['transaction_type']} order for {order_details['quantity']} shares of {order_details['symbol']}."

                elif "technical analysis for" in prompt_lower:
                    ticker = prompt.split("for")[-1].strip().upper()
                    token = get_instrument_token(ticker, instrument_df)
                    if token:
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            analysis = interpret_indicators(data)
                            response = f"**Technical Analysis for {ticker}:**\n\n" + "\n".join([f"- **{k}:** {v}" for k, v in analysis.items()])
                        else:
                            response = f"Could not retrieve enough data for {ticker} to perform analysis."
                    else:
                        response = f"Could not find the ticker '{ticker}'."
                
                elif "news for" in prompt_lower:
                    query = prompt.split("for")[-1].strip()
                    news_df = fetch_and_analyze_news(query)
                    if not news_df.empty:
                        response = f"**Top 3 news headlines for {query}:**\n\n" + "\n".join([f"1. [{row['title']}]({row['link']}) - _{row['source']}_" for _, row in news_df.head(3).iterrows()])
                    else:
                        response = f"No recent news found for '{query}'."
                
                elif "greeks for" in prompt_lower or "iv for" in prompt_lower:
                    try:
                        option_symbol_match = re.search(r'([A-Z]+)(\d{2}[A-Z]{3}\d+[CEPE]{2})', prompt.upper())
                        if option_symbol_match:
                            option_symbol = option_symbol_match.group(0)
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry - datetime.now().date()).days, 0) / 365.0
                            iv = implied_volatility(underlying_ltp, option_details['strike'], T, 0.07, ltp, option_details['instrument_type'].lower())
                            
                            if not np.isnan(iv):
                                greeks = black_scholes(underlying_ltp, option_details['strike'], T, 0.07, iv, option_details['instrument_type'].lower())
                                response = f"Calculated Greeks for **{option_symbol}**:\n- **Implied Volatility (IV):** {iv*100:.2f}%\n- **Delta:** {greeks['delta']:.4f}\n- **Gamma:** {greeks['gamma']:.4f}\n- **Vega:** {greeks['vega']:.4f}\n- **Theta:** {greeks['theta']:.4f}\n- **Rho:** {greeks['rho']:.4f}"
                            else:
                                response = f"Could not calculate IV or Greeks for {option_symbol}. The LTP might be zero or the option might be illiquid."
                        else:
                            response = "Please specify a valid option symbol (e.g., NIFTY24SEPWK123000CE)."
                    except (AttributeError, IndexError):
                        response = "I couldn't find a valid option symbol in your query. Please use the full symbol (e.g., BANKNIFTY24OCT60000CE)."
                    except Exception as e:
                        response = f"An error occurred: {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def page_basket_orders():
    """A page for creating, managing, and executing basket orders."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Basket Orders")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the basket order feature.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add Order to Basket")
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0
            
            if st.form_submit_button("Add to Basket"):
                if symbol and quantity > 0:
                    st.session_state.basket.append({
                        'symbol': symbol,
                        'transaction_type': transaction_type,
                        'quantity': quantity,
                        'product': product,
                        'order_type': order_type,
                        'price': price if price > 0 else None
                    })
                    st.success(f"Added {symbol} to basket!")
                    st.rerun()

    with col2:
        st.subheader("Current Basket")
        if st.session_state.basket:
            for i, order in enumerate(st.session_state.basket):
                with st.expander(f"{order['transaction_type']} {order['quantity']} {order['symbol']}"):
                    st.write(f"**Product:** {order['product']}")
                    st.write(f"**Order Type:** {order['order_type']}")
                    if order['price']:
                        st.write(f"**Price:** ₹{order['price']}")
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.basket.pop(i)
                        st.rerun()
            
            st.markdown("---")
            if st.button("Execute Basket Order", type="primary", use_container_width=True):
                execute_basket_order(st.session_state.basket, instrument_df)
        else:
            st.info("Your basket is empty. Add orders using the form on the left.")

def run_backtest(strategy_func, data, **params):
    """Runs a backtest for a given strategy function."""
    df = data.copy()
    signals = strategy_func(df, **params)
    
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    portfolio_value = []
    
    for i in range(len(df)):
        if signals[i] == 'BUY' and position == 0:
            position = capital / df['close'][i]
            capital = 0
        elif signals[i] == 'SELL' and position > 0:
            capital = position * df['close'][i]
            position = 0
        
        current_value = capital + (position * df['close'][i])
        portfolio_value.append(current_value)
        
    pnl = (portfolio_value[-1] - initial_capital) / initial_capital * 100
    
    return pnl, pd.Series(portfolio_value, index=df.index)

def rsi_strategy(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """Simple RSI Crossover Strategy"""
    rsi = ta.rsi(df['close'], length=rsi_period)
    signals = [''] * len(df)
    for i in range(1, len(df)):
        if rsi[i-1] < rsi_oversold and rsi[i] > rsi_oversold:
            signals[i] = 'BUY'
        elif rsi[i-1] > rsi_overbought and rsi[i] < rsi_overbought:
            signals[i] = 'SELL'
    return signals

def macd_strategy(df, fast=12, slow=26, signal=9):
    """MACD Crossover Strategy"""
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
    signals = [''] * len(df)
    for i in range(1, len(df)):
        if macd[f'MACD_{fast}_{slow}_{signal}'][i-1] < macd[f'MACDs_{fast}_{slow}_{signal}'][i-1] and macd[f'MACD_{fast}_{slow}_{signal}'][i] > macd[f'MACDs_{fast}_{slow}_{signal}'][i]:
            signals[i] = 'BUY'
        elif macd[f'MACD_{fast}_{slow}_{signal}'][i-1] > macd[f'MACDs_{fast}_{slow}_{signal}'][i-1] and macd[f'MACD_{fast}_{slow}_{signal}'][i] < macd[f'MACDs_{fast}_{slow}_{signal}'][i]:
            signals[i] = 'SELL'
    return signals

def supertrend_strategy(df, period=7, multiplier=3):
    """Supertrend Strategy"""
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=period, multiplier=multiplier)
    signals = [''] * len(df)
    st_col = next((col for col in supertrend.columns if 'SUPERT' in col), None)
    if not st_col: return signals 
    for i in range(1, len(df)):
        if df['close'][i] > supertrend[st_col][i-1] and df['close'][i-1] <= supertrend[st_col][i-1]:
            signals[i] = 'BUY'
        elif df['close'][i] < supertrend[st_col][i-1] and df['close'][i-1] >= supertrend[st_col][i-1]:
            signals[i] = 'SELL'
    return signals

def page_algo_strategy_maker():
    """Algo Strategy Maker page with pre-built, backtestable, and executable strategies."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("Algo Strategy Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to use the Algo Strategy Hub.")
        return

    st.info("Select a pre-built strategy, configure its parameters, and run a backtest on historical data. You can then place trades based on the latest signal.", icon="🤖")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Strategy Configuration")
        
        strategy_options = {
            "RSI Crossover": rsi_strategy,
            "MACD Crossover": macd_strategy,
            "Supertrend Follower": supertrend_strategy,
        }
        selected_strategy_name = st.selectbox("Select a Strategy", list(strategy_options.keys()))
        
        st.markdown("**Instrument**")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'NFO', 'MCX', 'CDS'])]['tradingsymbol'].unique()
        symbol = st.selectbox("Select Symbol", all_symbols, index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0)
        
        st.markdown("**Parameters**")
        params = {}
        if selected_strategy_name == "RSI Crossover":
            params['rsi_period'] = st.slider("RSI Period", 5, 30, 14)
            params['rsi_overbought'] = st.slider("RSI Overbought", 60, 90, 70)
            params['rsi_oversold'] = st.slider("RSI Oversold", 10, 40, 30)
        elif selected_strategy_name == "MACD Crossover":
            params['fast'] = st.slider("Fast Period", 5, 20, 12)
            params['slow'] = st.slider("Slow Period", 20, 50, 26)
            params['signal'] = st.slider("Signal Period", 5, 20, 9)
        elif selected_strategy_name == "Supertrend Follower":
            params['period'] = st.slider("ATR Period", 5, 20, 7)
            params['multiplier'] = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.5)

        st.markdown("**Trade Execution**")
        quantity = st.number_input("Trade Quantity", min_value=1, value=1)
        
        run_button = st.button("Run Backtest & Get Signal", use_container_width=True, type="primary")

    with col2:
        if run_button:
            with st.spinner(f"Running backtest for {selected_strategy_name} on {symbol}..."):
                exchange = instrument_df[instrument_df['tradingsymbol'] == symbol].iloc[0]['exchange']
                token = get_instrument_token(symbol, instrument_df, exchange=exchange)
                data = get_historical_data(token, 'day', period='1y')
                
                if not data.empty and len(data) > 50: 
                    pnl, portfolio_curve = run_backtest(strategy_options[selected_strategy_name], data, **params)
                    latest_signal = strategy_options[selected_strategy_name](data, **params)[-1]

                    st.session_state['backtest_results'] = {
                        'pnl': pnl,
                        'curve': portfolio_curve,
                        'signal': latest_signal,
                        'symbol': symbol,
                        'quantity': quantity
                    }
                else:
                    st.error("Could not fetch enough historical data to run the backtest.")
                    if 'backtest_results' in st.session_state:
                        st.session_state['backtest_results'] = None

        if st.session_state.get('backtest_results') is not None:
            results = st.session_state['backtest_results']
            st.subheader("Backtest Results")
            st.metric("Total P&L (1 Year)", f"{results['pnl']:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['curve'].index, y=results['curve'], mode='lines', name='Portfolio Value'))
            fig.update_layout(title="Portfolio Growth Over 1 Year", yaxis_title="Portfolio Value (₹)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Live Signal & Trading")
            signal = results['signal']
            color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
            st.markdown(f"### Latest Signal: <span style='color:{color};'>{signal if signal else 'HOLD'}</span>", unsafe_allow_html=True)

            if signal in ["BUY", "SELL"]:
                if st.button(f"Place {signal} Order for {results['quantity']} of {results['symbol']}", use_container_width=True):
                    place_order(instrument_df, results['symbol'], results['quantity'], "MARKET", signal, "MIS")

# ============ NEW ALGO BOTS PAGE ============

def momentum_trader_bot(instrument_df, capital):
    """Momentum Trader Bot - Buys stocks with strong upward momentum"""
    st.info("🔄 Scanning for momentum opportunities...")
    
    # Get NIFTY 50 stocks
    nifty_stocks = get_nifty50_constituents(instrument_df)
    if nifty_stocks.empty:
        return []
    
    momentum_stocks = []
    
    for symbol in nifty_stocks['Symbol'].head(10):  # Check top 10 for performance
        try:
            token = get_instrument_token(symbol, instrument_df, 'NSE')
            if token:
                data = get_historical_data(token, 'day', period='3mo')
                if not data.empty and len(data) > 20:
                    # Calculate momentum indicators
                    data['rsi'] = ta.rsi(data['close'], length=14)
                    data['sma_20'] = ta.sma(data['close'], length=20)
                    data['sma_50'] = ta.sma(data['close'], length=50)
                    
                    latest = data.iloc[-1]
                    
                    # Momentum criteria
                    if (latest['rsi'] > 50 and  # Above neutral RSI
                        latest['close'] > latest['sma_20'] and  # Above short-term MA
                        latest['sma_20'] > latest['sma_50'] and  # Uptrend
                        latest['close'] > data['close'].iloc[-5]):  # Higher than 5 days ago
                        
                        # Get current price
                        quote = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                        if not quote.empty:
                            current_price = quote.iloc[0]['Price']
                            momentum_stocks.append({
                                'symbol': symbol,
                                'price': current_price,
                                'rsi': latest['rsi'],
                                'signal': 'BUY',
                                'reason': f'Momentum (RSI: {latest["rsi"]:.1f}, Above MAs)'
                            })
        except Exception:
            continue
    
    return momentum_stocks[:5]  # Return top 5 momentum stocks

def mean_reversion_bot(instrument_df, capital):
    """Mean Reversion Bot - Buys oversold stocks expecting reversion to mean"""
    st.info("📊 Scanning for mean reversion opportunities...")
    
    nifty_stocks = get_nifty50_constituents(instrument_df)
    if nifty_stocks.empty:
        return []
    
    reversion_stocks = []
    
    for symbol in nifty_stocks['Symbol'].head(10):
        try:
            token = get_instrument_token(symbol, instrument_df, 'NSE')
            if token:
                data = get_historical_data(token, 'day', period='3mo')
                if not data.empty and len(data) > 20:
                    # Calculate indicators
                    data['rsi'] = ta.rsi(data['close'], length=14)
                    data['bb'] = ta.bbands(data['close'], length=20)
                    
                    latest = data.iloc[-1]
                    
                    # Mean reversion criteria
                    if (latest['rsi'] < 35 and  # Oversold
                        'BBL_20_2.0' in data.columns and 
                        latest['close'] <= latest['BBL_20_2.0']):  # Below lower Bollinger Band
                        
                        quote = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                        if not quote.empty:
                            current_price = quote.iloc[0]['Price']
                            reversion_stocks.append({
                                'symbol': symbol,
                                'price': current_price,
                                'rsi': latest['rsi'],
                                'signal': 'BUY',
                                'reason': f'Oversold (RSI: {latest["rsi"]:.1f}, Below BB)'
                            })
        except Exception:
            continue
    
    return reversion_stocks[:5]

def volatility_breakout_bot(instrument_df, capital):
    """Volatility Breakout Bot - Buys stocks breaking out of volatility compression"""
    st.info("⚡ Scanning for volatility breakout opportunities...")
    
    nifty_stocks = get_nifty50_constituents(instrument_df)
    if nifty_stocks.empty:
        return []
    
    breakout_stocks = []
    
    for symbol in nifty_stocks['Symbol'].head(10):
        try:
            token = get_instrument_token(symbol, instrument_df, 'NSE')
            if token:
                data = get_historical_data(token, 'day', period='3mo')
                if not data.empty and len(data) > 20:
                    # Calculate volatility and breakout indicators
                    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
                    data['bb'] = ta.bbands(data['close'], length=20)
                    data['volume_sma'] = ta.sma(data['volume'], length=20)
                    
                    latest = data.iloc[-1]
                    
                    # Volatility breakout criteria
                    if ('BBU_20_2.0' in data.columns and 
                        latest['close'] >= latest['BBU_20_2.0'] and  # Above upper Bollinger Band
                        latest['volume'] > latest['volume_sma'] * 1.2):  # Volume surge
                        
                        quote = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                        if not quote.empty:
                            current_price = quote.iloc[0]['Price']
                            breakout_stocks.append({
                                'symbol': symbol,
                                'price': current_price,
                                'atr': latest['atr'],
                                'signal': 'BUY',
                                'reason': f'Volatility Breakout (High Volume)'
                            })
        except Exception:
            continue
    
    return breakout_stocks[:5]

def value_investor_bot(instrument_df, capital):
    """Value Investor Bot - Buys fundamentally strong stocks at good prices"""
    st.info("💰 Scanning for value investment opportunities...")
    
    # Use popular large-cap stocks for value investing
    value_stocks = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN']
    selected_stocks = []
    
    for symbol in value_stocks:
        try:
            # Get current price and simple value metrics
            quote = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
            if not quote.empty:
                current_price = quote.iloc[0]['Price']
                
                # Get historical data for basic analysis
                token = get_instrument_token(symbol, instrument_df, 'NSE')
                if token:
                    data = get_historical_data(token, 'day', period='1y')
                    if not data.empty:
                        # Simple value criteria (simplified for demo)
                        avg_price_3mo = data['close'].tail(63).mean()  # 3-month average
                        price_ratio = current_price / avg_price_3mo
                        
                        if price_ratio < 0.95:  # Trading below 3-month average
                            selected_stocks.append({
                                'symbol': symbol,
                                'price': current_price,
                                'avg_3mo': avg_price_3mo,
                                'discount': (1 - price_ratio) * 100,
                                'signal': 'BUY',
                                'reason': f'Value Buy ({price_ratio:.2%} of 3m avg)'
                            })
        except Exception:
            continue
    
    return selected_stocks[:5]

def execute_algo_bot_trades(bot_name, stocks, capital, instrument_df):
    """Execute trades for an algo bot"""
    if not stocks:
        st.warning(f"{bot_name}: No suitable stocks found")
        return
    
    st.success(f"{bot_name}: Found {len(stocks)} opportunities")
    
    # Calculate position sizing
    capital_per_stock = capital / len(stocks)
    
    for stock in stocks:
        try:
            # Calculate quantity based on capital allocation
            quantity = int(capital_per_stock / stock['price'])
            if quantity > 0:
                # Place order (in real implementation)
                st.info(f"📈 {bot_name}: {stock['signal']} {quantity} shares of {stock['symbol']} @ ₹{stock['price']:.2f}")
                st.write(f"   Reason: {stock['reason']}")
                
                # In a real implementation, you would place the order here:
                # place_order(instrument_df, stock['symbol'], quantity, 'MARKET', stock['signal'], 'MIS')
        except Exception as e:
            st.error(f"Error executing trade for {stock['symbol']}: {e}")

def page_algo_bots():
    """Algo Bots page with 4 pre-built trading bots"""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("🤖 Algo Trading Bots")
    st.info("Automated trading bots that scan the market and execute trades based on predefined strategies. Click 'Run Bot' to activate any strategy.", icon="🚀")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo bots.")
        return
    
    # Capital allocation
    st.subheader("💰 Capital Allocation")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.algo_bot_capital = st.number_input(
            "Trading Capital (₹)", 
            min_value=10000, 
            max_value=1000000, 
            value=st.session_state.algo_bot_capital,
            step=10000,
            help="Total capital to allocate across all bot trades"
        )
    with col2:
        st.metric("Available Capital", f"₹{st.session_state.algo_bot_capital:,.2f}")
    
    st.markdown("---")
    
    # Algo Bot Cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Momentum Trader Bot
        st.markdown(f"""
        <div class='algo-bot-card {'running' if st.session_state.algo_bots_running.get('momentum', False) else 'stopped'}'>
            <h3>🚀 Momentum Trader</h3>
            <p><strong>Strategy:</strong> Buys stocks with strong upward momentum using RSI and moving averages</p>
            <p><strong>Target:</strong> High-momentum NIFTY 50 stocks</p>
            <p><strong>Risk:</strong> Medium</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Momentum Bot", key="run_momentum", use_container_width=True):
            with st.spinner("Momentum Bot scanning..."):
                stocks = momentum_trader_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Momentum Trader", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['momentum'] = True
        
        # Mean Reversion Bot
        st.markdown(f"""
        <div class='algo-bot-card {'running' if st.session_state.algo_bots_running.get('reversion', False) else 'stopped'}'>
            <h3>📊 Mean Reversion</h3>
            <p><strong>Strategy:</strong> Buys oversold stocks expecting reversion to mean using RSI and Bollinger Bands</p>
            <p><strong>Target:</strong> Oversold NIFTY 50 stocks</p>
            <p><strong>Risk:</strong> Low-Medium</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Reversion Bot", key="run_reversion", use_container_width=True):
            with st.spinner("Mean Reversion Bot scanning..."):
                stocks = mean_reversion_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Mean Reversion", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['reversion'] = True
    
    with col2:
        # Volatility Breakout Bot
        st.markdown(f"""
        <div class='algo-bot-card {'running' if st.session_state.algo_bots_running.get('breakout', False) else 'stopped'}'>
            <h3>⚡ Volatility Breakout</h3>
            <p><strong>Strategy:</strong> Buys stocks breaking out of volatility compression with high volume</p>
            <p><strong>Target:</strong> High-volatility breakout stocks</p>
            <p><strong>Risk:</strong> High</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Breakout Bot", key="run_breakout", use_container_width=True):
            with st.spinner("Volatility Breakout Bot scanning..."):
                stocks = volatility_breakout_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Volatility Breakout", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['breakout'] = True
        
        # Value Investor Bot
        st.markdown(f"""
        <div class='algo-bot-card {'running' if st.session_state.algo_bots_running.get('value', False) else 'stopped'}'>
            <h3>💰 Value Investor</h3>
            <p><strong>Strategy:</strong> Buys fundamentally strong large-cap stocks trading below their averages</p>
            <p><strong>Target:</strong> Undervalued blue-chip stocks</p>
            <p><strong>Risk:</strong> Low</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Value Bot", key="run_value", use_container_width=True):
            with st.spinner("Value Investor Bot scanning..."):
                stocks = value_investor_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Value Investor", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['value'] = True
    
    st.markdown("---")
    
    # Run All Bots
    st.subheader("🎯 Multi-Strategy Execution")
    if st.button("Run All Bots", type="primary", use_container_width=True):
        st.info("🚀 Running all trading bots simultaneously...")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.spinner("Momentum..."):
                stocks = momentum_trader_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Momentum Trader", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['momentum'] = True
        
        with col2:
            with st.spinner("Reversion..."):
                stocks = mean_reversion_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Mean Reversion", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['reversion'] = True
        
        with col3:
            with st.spinner("Breakout..."):
                stocks = volatility_breakout_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Volatility Breakout", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['breakout'] = True
        
        with col4:
            with st.spinner("Value..."):
                stocks = value_investor_bot(instrument_df, st.session_state.algo_bot_capital)
                execute_algo_bot_trades("Value Investor", stocks, st.session_state.algo_bot_capital / 4, instrument_df)
                st.session_state.algo_bots_running['value'] = True
        
        st.success("✅ All bots completed execution!")
    
    # Bot Status
    st.markdown("---")
    st.subheader("📈 Bot Status")
    status_cols = st.columns(4)
    status_cols[0].metric("Momentum", "🟢 Running" if st.session_state.algo_bots_running.get('momentum') else "🔴 Stopped")
    status_cols[1].metric("Reversion", "🟢 Running" if st.session_state.algo_bots_running.get('reversion') else "🔴 Stopped")
    status_cols[2].metric("Breakout", "🟢 Running" if st.session_state.algo_bots_running.get('breakout') else "🔴 Stopped")
    status_cols[3].metric("Value", "🟢 Running" if st.session_state.algo_bots_running.get('value') else "🔴 Stopped")
    
    # Stop All Bots
    if st.button("🛑 Stop All Bots", use_container_width=True):
        for bot in ['momentum', 'reversion', 'breakout', 'value']:
            st.session_state.algo_bots_running[bot] = False
        st.success("All bots stopped!")
        st.rerun()

# ... (rest of the existing code remains the same, including other page functions)

def main_app():
    """The main application interface after successful login."""
    apply_custom_styling()
    display_overnight_changes_bar()
    
    # --- 2FA Check ---
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
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options", "HFT"], horizontal=True)
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
        refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Cash": {
            "Dashboard": page_dashboard,
            "Algo Bots": page_algo_bots,  # NEW PAGE ADDED
            "Premarket Pulse": page_premarket_pulse,
            "Advanced Charting": page_advanced_charting,
            "Market Scanners": page_momentum_and_trend_finder,
            "Portfolio & Risk": page_portfolio_and_risk,
            "Basket Orders": page_basket_orders,
            "Forecasting (ML)": page_forecasting_ml,
            "Algo Strategy Hub": page_algo_strategy_maker,
            "AI Discovery": page_ai_discovery,
            "AI Assistant": page_ai_assistant,
            "Economic Calendar": page_economic_calendar,
            "Fundamental Analytics": page_fundamental_analytics,  # NEW PAGE ADDED
        },
        "Options": {
            "F&O Analytics": page_fo_analytics,
            "Options Strategy Builder": page_option_strategy_builder,
            "Greeks Calculator": page_greeks_calculator,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
            "Fundamental Analytics": page_fundamental_analytics,  # NEW PAGE ADDED
        },
        "Futures": {
            "Futures Terminal": page_futures_terminal,
            "Advanced Charting": page_advanced_charting,
            "Algo Strategy Hub": page_algo_strategy_maker,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
            "Fundamental Analytics": page_fundamental_analytics,  # NEW PAGE ADDED
        },
        "HFT": {
            "HFT Terminal": page_hft_terminal,
            "Portfolio & Risk": page_portfolio_and_risk,
        }
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    no_refresh_pages = ["Forecasting (ML)", "AI Assistant", "AI Discovery", "Algo Strategy Hub", "Fundamental Analytics"]
    if auto_refresh and selection not in no_refresh_pages:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

# --- Application Entry Point ---
if __name__ == "__main__":
    initialize_session_state()
    
    if 'profile' in st.session_state and st.session_state.profile:
        if st.session_state.get('login_animation_complete', False):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
