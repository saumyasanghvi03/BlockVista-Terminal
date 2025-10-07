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
        
        .bot-active {
            border-left-color: #28a745 !important;
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(40, 167, 69, 0.1) 100%);
        }
        .bot-inactive {
            border-left-color: #6c757d !important;
        }
        .bot-profit {
            border-left-color: #28a745 !important;
        }
        .bot-loss {
            border-left-color: #dc3545 !important;
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

# ================ MISSING FUNCTION: get_global_indices_data ================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_global_indices_data(tickers_dict):
    """
    Fetches real-time data for global indices using yfinance.
    
    Parameters:
    - tickers_dict: Dictionary mapping display names to yfinance tickers
    
    Returns:
    - DataFrame with Ticker, Price, and % Change columns
    """
    data = []
    
    for display_name, yf_ticker in tickers_dict.items():
        try:
            # Get the ticker data
            ticker = yf.Ticker(yf_ticker)
            info = ticker.info
            
            # Get current price and previous close
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            previous_close = info.get('regularMarketPreviousClose')
            
            if current_price and previous_close:
                change = current_price - previous_close
                pct_change = (change / previous_close) * 100
                
                data.append({
                    'Ticker': display_name,
                    'Price': current_price,
                    'Change': change,
                    '% Change': pct_change
                })
            else:
                # Fallback: try to get historical data
                hist = ticker.history(period='2d')
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2]
                    change = current_price - previous_close
                    pct_change = (change / previous_close) * 100
                    
                    data.append({
                        'Ticker': display_name,
                        'Price': current_price,
                        'Change': change,
                        '% Change': pct_change
                    })
                
        except Exception as e:
            st.toast(f"Error fetching {display_name}: {e}", icon="⚠️")
            continue
    
    return pd.DataFrame(data) if data else pd.DataFrame()

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
    
    # Algo Bots state variables
    if 'algo_bots' not in st.session_state:
        st.session_state.algo_bots = {
            "momentum_trader": {
                "name": "Momentum Trader",
                "description": "Tracks stocks with strong momentum using RSI and moving averages",
                "status": "inactive",
                "capital": 1000,
                "trades": [],
                "pnl": 0,
                "active_since": None
            },
            "mean_reversion": {
                "name": "Mean Reversion",
                "description": "Capitalizes on price reversions using Bollinger Bands and RSI",
                "status": "inactive", 
                "capital": 1000,
                "trades": [],
                "pnl": 0,
                "active_since": None
            },
            "volatility_breakout": {
                "name": "Volatility Breakout", 
                "description": "Identifies breakout opportunities using volatility channels",
                "status": "inactive",
                "capital": 1000,
                "trades": [],
                "pnl": 0,
                "active_since": None
            },
            "value_investor": {
                "name": "Value Investor",
                "description": "Focuses on fundamentally strong stocks with good valuations",
                "status": "inactive",
                "capital": 1000,
                "trades": [],
                "pnl": 0,
                "active_since": None
            }
        }

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
    overnight_tickers = {
        "GIFT NIFTY": "NSEI",  # Nifty 50 index
        "S&P 500": "^GSPC",    # S&P 500 index
        "NASDAQ": "^IXIC",     # NASDAQ composite
        "DOW JONES": "^DJI",   # Dow Jones
        "SGX NIFTY": "SNIFT.NS" # SGX Nifty
    }
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
            st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Failed: {e}"})
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

# ================ 4. ALGO TRADING BOTS ================

def momentum_trader_strategy(symbol, capital, instrument_df):
    """Momentum trading strategy using RSI and moving averages."""
    try:
        # Get historical data
        token = get_instrument_token(symbol, instrument_df)
        if not token:
            return None, "Instrument not found"
        
        data = get_historical_data(token, '5minute', period='1d')
        if data.empty or len(data) < 20:
            return None, "Insufficient data"
        
        # Calculate indicators
        data['rsi_14'] = ta.rsi(data['close'], length=14)
        data['sma_20'] = ta.sma(data['close'], length=20)
        data['ema_12'] = ta.ema(data['close'], length=12)
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Strategy logic
        signal = "HOLD"
        reason = ""
        
        # Bullish momentum signal
        if (latest['rsi_14'] > 50 and 
            latest['close'] > latest['sma_20'] and 
            latest['ema_12'] > latest['sma_20'] and
            latest['close'] > prev['close']):
            signal = "BUY"
            reason = f"Strong momentum: RSI({latest['rsi_14']:.1f}) > 50, Price above SMA20"
        
        # Bearish momentum signal  
        elif (latest['rsi_14'] < 45 and 
              latest['close'] < latest['sma_20'] and
              latest['close'] < prev['close']):
            signal = "SELL"
            reason = f"Momentum weakening: RSI({latest['rsi_14']:.1f}) < 45, Price below SMA20"
        
        return signal, reason
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def mean_reversion_strategy(symbol, capital, instrument_df):
    """Mean reversion strategy using Bollinger Bands and RSI."""
    try:
        token = get_instrument_token(symbol, instrument_df)
        if not token:
            return None, "Instrument not found"
        
        data = get_historical_data(token, '5minute', period='1d')
        if data.empty or len(data) < 20:
            return None, "Insufficient data"
        
        # Calculate indicators
        bb = ta.bbands(data['close'], length=20, std=2)
        data = pd.concat([data, bb], axis=1)
        data['rsi_14'] = ta.rsi(data['close'], length=14)
        
        latest = data.iloc[-1]
        
        # Strategy logic
        signal = "HOLD"
        reason = ""
        
        # Oversold bounce (Buy signal)
        if (latest['close'] <= latest['BBL_20_2.0'] and 
            latest['rsi_14'] < 30):
            signal = "BUY"
            reason = f"Oversold bounce: Price at lower BB, RSI({latest['rsi_14']:.1f}) < 30"
        
        # Overbought rejection (Sell signal)
        elif (latest['close'] >= latest['BBU_20_2.0'] and 
              latest['rsi_14'] > 70):
            signal = "SELL" 
            reason = f"Overbought rejection: Price at upper BB, RSI({latest['rsi_14']:.1f}) > 70"
        
        return signal, reason
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def volatility_breakout_strategy(symbol, capital, instrument_df):
    """Volatility breakout strategy using ATR and price channels."""
    try:
        token = get_instrument_token(symbol, instrument_df)
        if not token:
            return None, "Instrument not found"
        
        data = get_historical_data(token, '5minute', period='1d')
        if data.empty or len(data) < 20:
            return None, "Insufficient data"
        
        # Calculate indicators
        data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['high_20'] = data['high'].rolling(20).max()
        data['low_20'] = data['low'].rolling(20).min()
        data['volume_sma'] = data['volume'].rolling(20).mean()
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signal = "HOLD"
        reason = ""
        
        # Breakout above resistance with volume
        if (latest['close'] > latest['high_20'] and 
            latest['volume'] > latest['volume_sma'] * 1.5):
            signal = "BUY"
            reason = f"Breakout: New 20-period high with volume surge"
        
        # Breakdown below support
        elif (latest['close'] < latest['low_20'] and
              latest['volume'] > latest['volume_sma']):
            signal = "SELL"
            reason = f"Breakdown: New 20-period low"
        
        return signal, reason
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def value_investor_strategy(symbol, capital, instrument_df):
    """Value investing strategy using fundamental-like metrics (simulated)."""
    try:
        token = get_instrument_token(symbol, instrument_df)
        if not token:
            return None, "Instrument not found"
        
        data = get_historical_data(token, 'day', period='3mo')
        if data.empty or len(data) < 50:
            return None, "Insufficient data"
        
        # Calculate "value" metrics (simulated)
        data['sma_50'] = ta.sma(data['close'], length=50)
        data['sma_200'] = ta.sma(data['close'], length=200)
        data['price_to_sma'] = data['close'] / data['sma_50']
        
        latest = data.iloc[-1]
        
        signal = "HOLD"
        reason = ""
        
        # Undervalued signal (price below historical average)
        if (latest['price_to_sma'] < 0.95 and 
            latest['close'] > latest['sma_200']):  # Above long-term trend
            signal = "BUY"
            reason = f"Undervalued: Price {latest['price_to_sma']:.3f}x of 50-day average, above 200-day SMA"
        
        # Overvalued signal
        elif (latest['price_to_sma'] > 1.05 and
              latest['close'] < latest['sma_200']):  # Below long-term trend
            signal = "SELL"
            reason = f"Overvalued: Price {latest['price_to_sma']:.3f}x of 50-day average, below 200-day SMA"
        
        return signal, reason
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def execute_algo_trade(bot_id, symbol, signal, capital, instrument_df):
    """Execute trade for algo bot."""
    if signal not in ["BUY", "SELL"]:
        return False, "No valid signal"
    
    try:
        # Calculate quantity based on capital and current price
        quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
        if quote_data.empty:
            return False, "Could not fetch current price"
        
        current_price = quote_data.iloc[0]['Price']
        quantity = max(1, int(capital / current_price))
        
        # Place order
        place_order(instrument_df, symbol, quantity, 'MARKET', signal, 'MIS')
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'quantity': quantity,
            'price': current_price,
            'capital_used': quantity * current_price
        }
        
        st.session_state.algo_bots[bot_id]['trades'].append(trade)
        
        return True, f"Successfully executed {signal} order for {quantity} shares of {symbol}"
        
    except Exception as e:
        return False, f"Trade execution failed: {str(e)}"

def page_algo_trading_bots():
    """Algo Trading Bots page with 4 pre-built strategies."""
    display_header()
    
    # Check for market timing notifications
    check_market_timing_notifications()
    
    st.title("🤖 Algo Trading Bots")
    st.info("Automated trading bots that can execute strategies with minimum ₹100 capital. Monitor and control your algo traders here.", icon="🚀")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo trading bots.")
        return
    
    # Bot selection and configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Available Bots")
        
        # Display all bots
        bots = st.session_state.algo_bots
        for bot_id, bot in bots.items():
            bot_class = "bot-active" if bot["status"] == "active" else "bot-inactive"
            if bot["pnl"] > 0:
                bot_class = "bot-profit"
            elif bot["pnl"] < 0:
                bot_class = "bot-loss"
                
            with st.container():
                st.markdown(f'<div class="trade-card {bot_class}">', unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns([3, 2, 1])
                
                with col_a:
                    st.subheader(bot["name"])
                    st.caption(bot["description"])
                    
                with col_b:
                    st.metric("Capital", f"₹{bot['capital']:,.0f}")
                    st.metric("P&L", f"₹{bot['pnl']:,.2f}")
                    
                with col_c:
                    status_color = "🟢" if bot["status"] == "active" else "🔴"
                    st.write(f"{status_color} {bot['status'].upper()}")
                    
                    if bot["status"] == "active":
                        if st.button("🛑 Stop", key=f"stop_{bot_id}"):
                            st.session_state.algo_bots[bot_id]["status"] = "inactive"
                            st.rerun()
                    else:
                        if st.button("▶️ Start", key=f"start_{bot_id}"):
                            st.session_state.algo_bots[bot_id]["status"] = "active"
                            st.session_state.algo_bots[bot_id]["active_since"] = datetime.now()
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Quick Controls")
        
        # Capital management
        selected_bot = st.selectbox(
            "Select Bot",
            options=list(bots.keys()),
            format_func=lambda x: bots[x]["name"]
        )
        
        new_capital = st.number_input(
            "Allocate Capital (₹)",
            min_value=100,
            max_value=100000,
            value=bots[selected_bot]["capital"],
            step=100
        )
        
        if st.button("Update Capital", use_container_width=True):
            st.session_state.algo_bots[selected_bot]["capital"] = new_capital
            st.success(f"Capital updated to ₹{new_capital:,.0f}")
            st.rerun()
        
        st.markdown("---")
        
        # Manual trade execution
        st.subheader("Manual Test")
        test_symbol = st.text_input("Symbol", "RELIANCE").upper()
        test_bot = st.selectbox("Test Strategy", list(bots.keys()), 
                               format_func=lambda x: bots[x]["name"])
        
        if st.button("Test Strategy", use_container_width=True):
            with st.spinner("Analyzing..."):
                if test_bot == "momentum_trader":
                    signal, reason = momentum_trader_strategy(test_symbol, bots[test_bot]["capital"], instrument_df)
                elif test_bot == "mean_reversion":
                    signal, reason = mean_reversion_strategy(test_symbol, bots[test_bot]["capital"], instrument_df)
                elif test_bot == "volatility_breakout":
                    signal, reason = volatility_breakout_strategy(test_symbol, bots[test_bot]["capital"], instrument_df)
                elif test_bot == "value_investor":
                    signal, reason = value_investor_strategy(test_symbol, bots[test_bot]["capital"], instrument_df)
                
                if signal:
                    st.info(f"**Signal**: {signal}")
                    st.write(f"**Reason**: {reason}")
                    
                    if signal in ["BUY", "SELL"] and st.button("Execute Trade", type="primary"):
                        success, message = execute_algo_trade(test_bot, test_symbol, signal, bots[test_bot]["capital"], instrument_df)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.warning(f"No trading signal: {reason}")
    
    st.markdown("---")
    
    # Active bot monitoring
    st.subheader("📊 Active Bot Performance")
    
    active_bots = {k: v for k, v in bots.items() if v["status"] == "active"}
    
    if active_bots:
        for bot_id, bot in active_bots.items():
            with st.expander(f"📈 {bot['name']} - Live Monitoring", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Get current signal for a sample stock
                    sample_symbol = "RELIANCE"
                    if bot_id == "momentum_trader":
                        signal, reason = momentum_trader_strategy(sample_symbol, bot["capital"], instrument_df)
                    elif bot_id == "mean_reversion":
                        signal, reason = mean_reversion_strategy(sample_symbol, bot["capital"], instrument_df)
                    elif bot_id == "volatility_breakout":
                        signal, reason = volatility_breakout_strategy(sample_symbol, bot["capital"], instrument_df)
                    elif bot_id == "value_investor":
                        signal, reason = value_investor_strategy(sample_symbol, bot["capital"], instrument_df)
                    
                    signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
                    st.metric("Current Signal", signal, delta=reason)
                
                with col2:
                    total_trades = len(bot["trades"])
                    winning_trades = len([t for t in bot["trades"] if t.get('profit', 0) > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col3:
                    if bot["active_since"]:
                        active_time = datetime.now() - bot["active_since"]
                        st.metric("Active Time", f"{active_time.days}d {active_time.seconds//3600}h")
                
                # Recent trades
                if bot["trades"]:
                    st.write("**Recent Trades**")
                    recent_trades = pd.DataFrame(bot["trades"][-5:])
                    st.dataframe(recent_trades, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades executed yet.")
                    
                # Auto-execution toggle
                auto_execute = st.checkbox(f"Enable Auto-Trading for {bot['name']}", value=False, key=f"auto_{bot_id}")
                if auto_execute:
                    st.warning("⚠️ Auto-trading will execute signals automatically. Use with caution!")
                    
                    # Sample execution on test symbol
                    if signal in ["BUY", "SELL"]:
                        if st.button(f"Execute {signal} for {sample_symbol}", key=f"execute_{bot_id}"):
                            success, message = execute_algo_trade(bot_id, sample_symbol, signal, bot["capital"], instrument_df)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
    else:
        st.info("No active bots. Start a bot to see live monitoring.")
    
    st.markdown("---")
    
    # Strategy explanations
    st.subheader("🎯 Strategy Details")
    
    strategy_tabs = st.tabs([bot["name"] for bot in bots.values()])
    
    for i, (bot_id, bot) in enumerate(bots.items()):
        with strategy_tabs[i]:
            st.write(f"**{bot['name']} Strategy**")
            
            if bot_id == "momentum_trader":
                st.markdown("""
                **Strategy Logic:**
                - Buys when RSI > 50 and price is above 20-period SMA with upward momentum
                - Sells when RSI < 45 and price is below 20-period SMA
                - Uses 5-minute timeframes for intraday momentum
                
                **Best For:** Trending markets with clear momentum
                **Risk Level:** Medium
                """)
                
            elif bot_id == "mean_reversion":
                st.markdown("""
                **Strategy Logic:**
                - Buys when price touches lower Bollinger Band and RSI < 30 (oversold)
                - Sells when price touches upper Bollinger Band and RSI > 70 (overbought)
                - Capitalizes on price returning to mean
                
                **Best For:** Range-bound markets
                **Risk Level:** Low-Medium
                """)
                
            elif bot_id == "volatility_breakout":
                st.markdown("""
                **Strategy Logic:**
                - Buys on breakout above 20-period high with volume surge
                - Sells on breakdown below 20-period low
                - Uses Average True Range (ATR) for volatility measurement
                
                **Best For:** High volatility periods, breakouts
                **Risk Level:** High
                """)
                
            elif bot_id == "value_investor":
                st.markdown("""
                **Strategy Logic:**
                - Buys when price is below 50-day average (undervalued) but above 200-day SMA
                - Sells when price is significantly above 50-day average (overvalued)
                - Simulated fundamental approach for intraday
                
                **Best For:** Longer-term positions, value opportunities
                **Risk Level:** Low
                """)
            
            st.metric("Minimum Capital", "₹100")
            st.metric("Recommended Capital", "₹1,000 - ₹10,000")

# ================ 5. PAGE DEFINITIONS ============

# [Previous page definitions remain the same, just adding the new algo bots page to navigation]

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

# [Rest of the existing pages remain exactly the same...]

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    secret = base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]
    return secret

def two_factor_dialog():
    """Dialog for 2FA login with improved UI."""
    if 'show_2fa_dialog' not in st.session_state:
        st.session_state.show_2fa_dialog = False
    
    if st.session_state.show_2fa_dialog:
        # Create a centered container for better UI
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; border-radius: 10px; background: var(--widget-bg); border: 1px solid var(--border-color);'>
                <h3>🔐 Two-Factor Authentication</h3>
                <p>Enter the 6-digit code from your authenticator app</p>
            </div>
            """, unsafe_allow_html=True)
            
            auth_code = st.text_input(
                "2FA Code", 
                max_chars=6, 
                key="2fa_code",
                placeholder="000000",
                label_visibility="collapsed"
            )
            
            btn_col1, btn_col2 = st.columns(2)
            if btn_col1.button("✅ Authenticate", use_container_width=True, type="primary"):
                if auth_code and len(auth_code) == 6:
                    try:
                        totp = pyotp.TOTP(st.session_state.pyotp_secret)
                        if totp.verify(auth_code):
                            st.session_state.authenticated = True
                            st.session_state.show_2fa_dialog = False
                            st.rerun()
                        else:
                            st.error("❌ Invalid code. Please try again.")
                    except Exception as e:
                        st.error(f"Authentication error: {e}")
                else:
                    st.warning("⚠️ Please enter a valid 6-digit code.")
            
            if btn_col2.button("❌ Cancel", use_container_width=True):
                st.session_state.show_2fa_dialog = False
                st.rerun()

def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup with improved UI."""
    if 'show_qr_dialog' not in st.session_state:
        st.session_state.show_qr_dialog = True
    
    if st.session_state.show_qr_dialog:
        # Centered container for QR setup
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; border-radius: 10px; background: var(--widget-bg); border: 1px solid var(--border-color);'>
                <h3>🔒 Set Up Two-Factor Authentication</h3>
                <p>Scan this QR code with your authenticator app for enhanced security</p>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.pyotp_secret is None:
                st.session_state.pyotp_secret = get_user_secret(st.session_state.get('profile', {}))
            
            secret = st.session_state.pyotp_secret
            user_name = st.session_state.get('profile', {}).get('user_name', 'User')
            uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
            
            # Generate QR code
            img = qrcode.make(uri)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            
            # Display QR code
            st.image(buf.getvalue(), caption="Scan with Google Authenticator or similar app", use_container_width=True)
            
            # Manual secret entry option
            with st.expander("Manual Setup"):
                st.code(secret, language="text")
                st.caption("If you can't scan the QR code, enter this secret manually in your authenticator app.")
            
            if st.button("✅ I've scanned the code. Continue to login.", use_container_width=True, type="primary"):
                st.session_state.two_factor_setup_complete = True
                st.session_state.show_qr_dialog = False
                st.session_state.show_2fa_dialog = True  # Immediately show 2FA entry
                st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login with improved design."""
    # Center the animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h1 style='color: var(--text-color); margin-bottom: 2rem;'>🚀 BlockVista Terminal</h1>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = {
            "🔐 Authenticating user credentials...": 20,
            "🌐 Establishing secure connection...": 40, 
            "📡 Fetching live market data feeds...": 65,
            "⚡ Initializing trading terminal...": 85,
            "✅ Terminal ready! Loading interface...": 100
        }
        
        for text, progress in steps.items():
            status_text.markdown(f"<div style='text-align: center; padding: 1rem;'>{text}</div>", unsafe_allow_html=True)
            progress_bar.progress(progress)
            a_time.sleep(0.8)
        
        a_time.sleep(0.5)
        st.session_state['login_animation_complete'] = True
        st.rerun()

def login_page():
    """Displays the login page for broker authentication with improved UI."""
    # Apply styling first
    apply_custom_styling()
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <h1>BlockVista Terminal</h1>
            <p style='color: var(--text-light);'>Professional Trading Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login card
        st.markdown("""
        <div style='
            background: var(--widget-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
        '>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔐 Broker Login")
        
        broker = st.selectbox("Select Your Broker", ["Zerodha"], key="broker_select")
        
        if broker == "Zerodha":
            api_key = st.secrets.get("ZERODHA_API_KEY")
            api_secret = st.secrets.get("ZERODHA_API_SECRET")
            
            if not api_key or not api_secret:
                st.error("""
                ❌ Kite API credentials not found. 
                
                Please set these secrets in your Streamlit app:
                - `ZERODHA_API_KEY`
                - `ZERODHA_API_SECRET`
                
                [Learn how to set secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
                """)
                st.stop()
                
            kite = KiteConnect(api_key=api_key)
            request_token = st.query_params.get("request_token")
            
            if request_token:
                try:
                    with st.spinner("🔄 Authenticating with Zerodha..."):
                        data = kite.generate_session(request_token, api_secret=api_secret)
                        st.session_state.access_token = data["access_token"]
                        kite.set_access_token(st.session_state.access_token)
                        st.session_state.kite = kite
                        st.session_state.profile = kite.profile()
                        st.session_state.broker = "Zerodha"
                        st.query_params.clear()
                        st.success("✅ Authentication successful!")
                        a_time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Authentication failed: {e}")
                    st.query_params.clear()
                    if st.button("🔄 Try Again"):
                        st.rerun()
            else:
                st.markdown("""
                <div style='
                    background: var(--secondary-bg);
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                '>
                    <h4>📋 Login Instructions:</h4>
                    <ol>
                        <li>Click the login button below</li>
                        <li>You'll be redirected to Zerodha</li>
                        <li>Login with your Kite credentials</li>
                        <li>You'll be redirected back automatically</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Login button with better styling
                login_url = kite.login_url()
                st.markdown(f"""
                <a href="{login_url}" style='
                    display: inline-block;
                    width: 100%;
                    padding: 0.75rem 1.5rem;
                    background: #387ed1;
                    color: white;
                    text-align: center;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                    border: none;
                    cursor: pointer;
                    margin: 1rem 0;
                '>🔗 Login with Zerodha Kite</a>
                """, unsafe_allow_html=True)
                
                st.caption("🔒 Your credentials are handled securely by Zerodha. We never store your password.")

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
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options", "HFT", "Algo Bots"], horizontal=True)
    st.sidebar.divider()
    
    # Dynamic refresh interval based on mode
    if st.session_state.terminal_mode == "HFT":
        refresh_interval = 2
        auto_refresh = True
        st.sidebar.header("HFT Mode Active")
        st.sidebar.caption(f"Refresh Interval: {refresh_interval}s")
    elif st.session_state.terminal_mode == "Algo Bots":
        refresh_interval = 10
        auto_refresh = True
        st.sidebar.header("Algo Bots Active")
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
            "Fundamental Analytics": page_fundamental_analytics,
        },
        "Options": {
            "F&O Analytics": page_fo_analytics,
            "Options Strategy Builder": page_option_strategy_builder,
            "Greeks Calculator": page_greeks_calculator,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
            "Fundamental Analytics": page_fundamental_analytics,
        },
        "Futures": {
            "Futures Terminal": page_futures_terminal,
            "Advanced Charting": page_advanced_charting,
            "Algo Strategy Hub": page_algo_strategy_maker,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
            "Fundamental Analytics": page_fundamental_analytics,
        },
        "HFT": {
            "HFT Terminal": page_hft_terminal,
            "Portfolio & Risk": page_portfolio_and_risk,
        },
        "Algo Bots": {
            "Algo Trading Bots": page_algo_trading_bots,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
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
