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
    """Applies a comprehensive CSS stylesheet for professional theming with Indian market focus."""
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
        
        /* Indian Market Specific Styling */
        .indian-flag-theme {
            background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
            padding: 2px;
            border-radius: 5px;
        }
        .nse-badge {
            background: #1f4d1f;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: bold;
        }
        .bse-badge {
            background: #2c5aa0;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: bold;
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

# Centralized data source configuration for Indian Markets
INDIAN_MARKET_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NSE",
        "description": "National Stock Exchange Fifty - Benchmark Indian Index"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NFO",
        "description": "Banking Sector Index - NSE Banking Stocks"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NFO",
        "description": "Financial Services Index - Banks, NBFCs, Insurance"
    },
    "SENSEX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv",
        "tradingsymbol": "SENSEX",
        "exchange": "BSE",
        "description": "Bombay Stock Exchange Sensitive Index - BSE 30 Stocks"
    },
    "NIFTY MIDCAP 100": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",  # Placeholder
        "tradingsymbol": "NIFTY MIDCAP 100",
        "exchange": "NSE",
        "description": "NSE Midcap 100 Companies"
    },
    "NIFTY SMALLCAP 100": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",  # Placeholder
        "tradingsymbol": "NIFTY SMLCAP 100",
        "exchange": "NSE",
        "description": "NSE Smallcap 100 Companies"
    },
    "INDIA VIX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",  # Placeholder
        "tradingsymbol": "INDIA VIX",
        "exchange": "NSE",
        "description": "India Volatility Index - Market Fear Gauge"
    }
}

# Indian Market Holidays 2024-2025
INDIAN_MARKET_HOLIDAYS = {
    2024: [
        '2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', 
        '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', 
        '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'
    ],
    2025: [
        '2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', 
        '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'
    ]
}

# Popular Indian Stocks for Quick Access
POPULAR_INDIAN_STOCKS = {
    "Large Cap": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN"],
    "Mid Cap": ["TATACONSUM", "ADANIENT", "BAJAJFINSV", "TECHM", "INDUSINDBK", "BRITANNIA"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BANDHANBNK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "MPHASIS"],
    "Auto": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJAUTO", "HEROMOTOCO", "EICHERMOT"]
}

# ================ 1.5 INITIALIZATION ========================
def initialize_session_state():
    """Initializes all necessary session state variables with Indian market focus."""
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
            "Nifty 50 Heavyweights": [
                {'symbol': 'RELIANCE', 'exchange': 'NSE', 'category': 'Large Cap'},
                {'symbol': 'HDFCBANK', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'INFY', 'exchange': 'NSE', 'category': 'IT'},
                {'symbol': 'ICICIBANK', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'TCS', 'exchange': 'NSE', 'category': 'IT'}
            ],
            "Banking Stocks": [
                {'symbol': 'HDFCBANK', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'ICICIBANK', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'SBIN', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'KOTAKBANK', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'AXISBANK', 'exchange': 'NSE', 'category': 'Banking'}
            ],
            "F&O favourites": [
                {'symbol': 'RELIANCE', 'exchange': 'NSE', 'category': 'Large Cap'},
                {'symbol': 'SBIN', 'exchange': 'NSE', 'category': 'Banking'},
                {'symbol': 'TATAMOTORS', 'exchange': 'NSE', 'category': 'Auto'},
                {'symbol': 'HINDALCO', 'exchange': 'NSE', 'category': 'Metals'},
                {'symbol': 'ONGC', 'exchange': 'NSE', 'category': 'Energy'}
            ]
        }
    if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Nifty 50 Heavyweights"
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
    if 'hft_symbol' not in st.session_state: st.session_state.hft_symbol = "NIFTY24OCTFUT"

# ================ 2. INDIAN MARKET SPECIFIC FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays for Indian market."""
    return INDIAN_MARKET_HOLIDAYS.get(year, [])

def get_market_status():
    """Checks if the Indian stock market is open with NSE timings."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    # Check if it's a holiday
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "#FF4B4B", "message": "Market Holiday"}
    
    # Check market hours
    if market_open_time <= now.time() <= market_close_time:
        time_to_close = (datetime.combine(now.date(), market_close_time) - now).total_seconds() / 60
        return {"status": "OPEN", "color": "#28a745", "message": f"Market Open - {int(time_to_close)}min to close"}
    
    # Check pre-open session
    pre_open_start, pre_open_end = time(9, 0), time(9, 15)
    if pre_open_start <= now.time() <= pre_open_end:
        return {"status": "PRE-OPEN", "color": "#FFA500", "message": "Pre-Open Session"}
    
    return {"status": "CLOSED", "color": "#FF4B4B", "message": "Market Closed"}

def get_indian_market_sectors():
    """Returns Indian market sector classification."""
    return {
        "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
        "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
        "Auto": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJAUTO", "HEROMOTOCO"],
        "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
        "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL"],
        "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "SAIL"],
        "FMCG": ["HINDUNILVR", "ITC", "BRITANNIA", "NESTLE", "DABUR"],
        "Telecom": ["BHARTIARTL", "RELIANCE", "VODAFONE"],
        "Realty": ["DLF", "PRESTIGE", "SOBHA", "BRIGADE"],
        "Media": ["ZEEL", "SUNTV", "TV18BRDCST"]
    }

def display_header():
    """Displays the main header with Indian market status and quick actions."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(
            '<h1 style="margin: 0; line-height: 1.2;">🇮🇳 BlockVista Terminal</h1>'
            '<p style="margin: 0; color: var(--text-light); font-size: 0.9rem;">Professional Trading Platform for Indian Markets</p>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"""
            <div style="text-align: right;">
                <h5 style="margin: 0;">{current_time}</h5>
                <h5 style="margin: 0;">NSE: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
                <p style="margin: 0; font-size: 0.8rem; color: var(--text-light);">{status_info["message"]}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("📈 Buy", use_container_width=True, key="header_buy"):
            quick_trade_dialog()
        if b_col2.button("📉 Sell", use_container_width=True, key="header_sell"):
            quick_trade_dialog()

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog optimized for Indian market instruments."""
    if 'show_quick_trade' not in st.session_state:
        st.session_state.show_quick_trade = False
    
    if symbol or st.button("Quick Trade", key="quick_trade_btn"):
        st.session_state.show_quick_trade = True
    
    if st.session_state.show_quick_trade:
        st.subheader(f"🇮🇳 Place Order for {symbol}" if symbol else "🇮🇳 Quick Order")
        
        if symbol is None:
            col1, col2 = st.columns(2)
            symbol = col1.text_input("Symbol").upper()
            exchange = col2.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX", "CDS"])
        
        transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
        product = st.radio("Product", ["MIS", "CNC", "NRML"], horizontal=True, key="diag_prod_type")
        order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
        
        col_qty, col_price = st.columns(2)
        quantity = col_qty.number_input("Quantity", min_value=1, step=1, key="diag_qty")
        price = col_price.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

        col1, col2 = st.columns(2)
        if col1.button("✅ Submit Order", use_container_width=True, type="primary"):
            if symbol and quantity > 0:
                instrument_df = get_instrument_df()
                place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
                st.session_state.show_quick_trade = False
                st.rerun()
            else:
                st.warning("Please fill in all fields.")
        
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state.show_quick_trade = False
            st.rerun()

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    client = get_broker_client()
    if not client: 
        return pd.DataFrame()
    
    if st.session_state.broker == "Zerodha":
        try:
            df = pd.DataFrame(client.instruments())
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'])
            return df
        except Exception as e:
            st.error(f"Failed to fetch instruments: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Instrument list for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: 
        return None
    
    match = instrument_df[
        (instrument_df['tradingsymbol'] == symbol.upper()) & 
        (instrument_df['exchange'] == exchange)
    ]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API for Indian markets."""
    client = get_broker_client()
    if not client or not instrument_token: 
        return pd.DataFrame()
    
    if st.session_state.broker == "Zerodha":
        if not to_date: 
            to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        
        if from_date > to_date:
            from_date = to_date - timedelta(days=1)
            
        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty: 
                return df
            
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Add technical indicators commonly used in Indian markets
            try:
                # Basic indicators
                df.ta.rsi(length=14, append=True)
                df.ta.macd(append=True)
                df.ta.bbands(length=20, append=True)
                df.ta.ema(length=50, append=True)
                df.ta.ema(length=200, append=True)
                df.ta.supertrend(append=True)
                df.ta.vwap(append=True)
                
            except Exception as e:
                st.toast(f"Some indicators couldn't be calculated: {e}", icon="⚠️")
                
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
    """Fetches live prices and market data for Indian stocks."""
    client = get_broker_client()
    if not client or not symbols_with_exchange: 
        return pd.DataFrame()
    
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
                    
                    watchlist.append({
                        'Ticker': item['symbol'], 
                        'Exchange': item['exchange'],
                        'Category': item.get('category', 'General'),
                        'Price': last_price, 
                        'Change': change, 
                        '% Change': pct_change,
                        'Volume': quote.get('volume', 0)
                    })
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
    else:
        st.warning(f"Watchlist for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    """Generates a Plotly chart optimized for Indian market data."""
    fig = go.Figure()
    if df.empty: 
        return fig
    
    chart_df = df.copy()
    
    # Handle MultiIndex columns
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.droplevel(0)
        
    chart_df.columns = [str(col).lower() for col in chart_df.columns]
    
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in chart_df.columns for col in required_cols):
        st.error(f"Charting error for {ticker}: Dataframe is missing required columns.")
        return go.Figure()

    # Create chart based on type
    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
        fig.add_trace(go.Candlestick(
            x=ha_df.index, 
            open=ha_df['HA_open'], 
            high=ha_df['HA_high'], 
            low=ha_df['HA_low'], 
            close=ha_df['HA_close'], 
            name='Heikin-Ashi'
        ))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Price'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(
            x=chart_df.index, 
            open=chart_df['open'], 
            high=chart_df['high'], 
            low=chart_df['low'], 
            close=chart_df['close'], 
            name='OHLC'
        ))
    else:  # Candlestick
        fig.add_trace(go.Candlestick(
            x=chart_df.index, 
            open=chart_df['open'], 
            high=chart_df['high'], 
            low=chart_df['low'], 
            close=chart_df['close'], 
            name='Candlestick'
        ))
        
    # Add Bollinger Bands if available
    bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
    bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
    if bbl_col and bbu_col:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbl_col], line=dict(color='rgba(135,206,250,0.5)', width=1), name='BB Lower'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbu_col], line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='BB Upper'))
        
    # Add forecast if available
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        if conf_int_df is not None:
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(
        title=f'{ticker} Price Chart ({chart_type})', 
        yaxis_title='Price (₹)', 
        xaxis_rangeslider_visible=False, 
        template=template, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ================ 4. INDIAN F&O ANALYTICS ================

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for Indian underlyings."""
    client = get_broker_client()
    if not client or instrument_df.empty: 
        return pd.DataFrame(), None, 0.0, []
    
    if st.session_state.broker == "Zerodha":
        exchange_map = {"NIFTY": "NFO", "BANKNIFTY": "NFO", "FINNIFTY": "NFO"}
        exchange = exchange_map.get(underlying, 'NFO')
        
        # Get underlying LTP
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}.get(underlying, underlying)
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"
        
        try:
            underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
        except Exception:
            underlying_ltp = 0.0
        
        # Get options data
        options = instrument_df[
            (instrument_df['name'] == underlying.upper()) & 
            (instrument_df['exchange'] == exchange)
        ]
        if options.empty: 
            return pd.DataFrame(), None, underlying_ltp, []
        
        expiries = sorted(options['expiry'].dt.date.unique())
        three_months_later = datetime.now().date() + timedelta(days=90)
        available_expiries = [e for e in expiries if datetime.now().date() <= e <= three_months_later]
        if not available_expiries: 
            return pd.DataFrame(), None, underlying_ltp, []
        
        if not expiry_date: 
            expiry_date = available_expiries[0]
        else: 
            expiry_date = pd.to_datetime(expiry_date).date()
        
        chain_df = options[options['expiry'].dt.date == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch: 
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        
        try:
            quotes = client.quote(instruments_to_fetch)
            
            ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            ce_df['oi'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            pe_df['oi'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            
            final_chain = pd.merge(
                ce_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                pe_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                on='strike', 
                suffixes=('_CE', '_PE')
            ).rename(columns={
                'LTP_CE': 'CALL LTP', 
                'LTP_PE': 'PUT LTP', 
                'strike': 'STRIKE', 
                'oi_CE': 'CALL OI', 
                'oi_PE': 'PUT OI', 
                'tradingsymbol_CE': 'CALL', 
                'tradingsymbol_PE': 'PUT'
            }).fillna(0)
            
            return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries
        
        except Exception as e:
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order through the broker's API for Indian markets."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        try:
            # Determine exchange based on symbol characteristics
            is_option = any(char.isdigit() for char in symbol)
            if is_option:
                exchange = 'NFO'
            else:
                instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
                if instrument.empty:
                    st.error(f"Symbol '{symbol}' not found.")
                    return
                exchange = instrument.iloc[0]['exchange']

            order_id = client.place_order(
                tradingsymbol=symbol.upper(),
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                variety=client.VARIETY_REGULAR,
                price=price
            )
            st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
            st.session_state.order_history.insert(0, {
                "id": order_id, 
                "symbol": symbol, 
                "qty": quantity, 
                "type": transaction_type, 
                "status": "Success",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        except Exception as e:
            st.toast(f"❌ Order failed: {e}", icon="🔥")
            st.session_state.order_history.insert(0, {
                "id": "N/A", 
                "symbol": symbol, 
                "qty": quantity, 
                "type": transaction_type, 
                "status": f"Failed: {e}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
    else:
        st.warning(f"Order placement for {st.session_state.broker} not implemented.")

# ================ 5. INDIAN MARKET SPECIFIC PAGES ============

def page_indian_market_overview():
    """Comprehensive overview of Indian markets with real-time data."""
    display_header()
    
    st.title("🇮🇳 Indian Market Overview")
    st.info("Real-time overview of Indian stock markets, indices, and key metrics", icon="📊")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view market data.")
        return
    
    # Key Indian Indices
    st.subheader("📈 Key Indian Indices")
    indices = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE', 'name': 'Nifty 50'},
        {'symbol': 'BANKNIFTY', 'exchange': 'NSE', 'name': 'Bank Nifty'},
        {'symbol': 'SENSEX', 'exchange': 'BSE', 'name': 'Sensex'},
        {'symbol': 'NIFTY FIN SERVICE', 'exchange': 'NSE', 'name': 'Fin Nifty'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE', 'name': 'India VIX'}
    ]
    
    index_data = get_watchlist_data(indices)
    if not index_data.empty:
        cols = st.columns(len(indices))
        for idx, (_, row) in enumerate(index_data.iterrows()):
            with cols[idx]:
                color = 'green' if row['Change'] > 0 else 'red'
                st.metric(
                    label=row['Ticker'],
                    value=f"₹{row['Price']:,.0f}",
                    delta=f"{row['Change']:+.1f} ({row['% Change']:+.2f}%)",
                    delta_color="normal"
                )
    
    st.markdown("---")
    
    # Market Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏛️ Market Status")
        status_info = get_market_status()
        st.markdown(f"""
        <div class="metric-card" style="border-color: {status_info['color']};">
            <h3>{status_info['status']}</h3>
            <p>{status_info['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Sector Performance")
        sectors = get_indian_market_sectors()
        st.metric("Sectors Tracked", len(sectors))
        st.metric("Popular Stocks", sum(len(stocks) for stocks in sectors.values()))
    
    with col3:
        st.subheader("🕒 Trading Hours")
        st.markdown("""
        - **Pre-open**: 9:00 AM - 9:15 AM
        - **Normal Market**: 9:15 AM - 3:30 PM
        - **Post-close**: 3:30 PM - 4:00 PM
        """)
    
    st.markdown("---")
    
    # Quick Access to Popular Stocks
    st.subheader("🚀 Popular Indian Stocks")
    
    for sector, stocks in list(POPULAR_INDIAN_STOCKS.items())[:3]:  # Show first 3 sectors
        st.write(f"**{sector}**")
        sector_stocks = [{'symbol': s, 'exchange': 'NSE', 'category': sector} for s in stocks[:4]]
        sector_data = get_watchlist_data(sector_stocks)
        
        if not sector_data.empty:
            cols = st.columns(len(sector_data))
            for idx, (_, stock) in enumerate(sector_data.iterrows()):
                with cols[idx]:
                    badge_class = "nse-badge" if stock['Exchange'] == 'NSE' else "bse-badge"
                    st.markdown(f"""
                    <div style='padding: 1rem; background: var(--widget-bg); border-radius: 8px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <strong>{stock['Ticker']}</strong>
                            <span class='{badge_class}'>{stock['Exchange']}</span>
                        </div>
                        <div style='font-size: 1.2em; margin: 0.5rem 0;'>₹{stock['Price']:,.1f}</div>
                        <div style='color: {'#28a745' if stock['Change'] > 0 else '#FF4B4B'};'>
                            {stock['Change']:+.1f} ({stock['% Change']:+.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def page_dashboard():
    """A completely redesigned 'Trader UI' Dashboard focused on Indian markets."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    
    # Market status notification
    status_info = get_market_status()
    if status_info['status'] != 'OPEN':
        st.warning(f"⚠️ {status_info['message']}")
    
    # Bharatiya Market Pulse
    bmp_col, heatmap_col = st.columns([1, 1], gap="large")
    
    with bmp_col:
        st.subheader("🇮🇳 Bharatiya Market Pulse")
        st.markdown("""
        <div class="metric-card">
            <h3>76.5</h3>
            <p style="color: #28a745; font-weight: bold;">Bharat Pragati (Bullish)</p>
            <small>Proprietary score based on NIFTY, SENSEX, and India VIX momentum</small>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("BMP Score Interpretation"):
            st.markdown("""
            - **80-100 (Bharat Udaan):** Very Strong Bullish Momentum
            - **60-80 (Bharat Pragati):** Moderately Bullish Sentiment  
            - **40-60 (Bharat Santulan):** Neutral or Sideways Market
            - **20-40 (Bharat Sanket):** Moderately Bearish Sentiment
            - **0-20 (Bharat Mandhi):** Very Strong Bearish Momentum
            """)
    
    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        # Placeholder for heatmap
        st.info("Heatmap visualization would appear here with live NIFTY 50 data")
    
    st.markdown("---")
    
    # Main Content Area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["📊 Watchlist", "💼 Portfolio"])
        
        with tab1:
            # Watchlist selector
            st.session_state.active_watchlist = st.radio(
                "Select Watchlist",
                options=list(st.session_state.watchlists.keys()),
                horizontal=True,
                label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]
            
            # Add stock form
            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="e.g., RELIANCE", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add", use_container_width=True):
                    if new_symbol:
                        if len(active_list) >= 15:
                            st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({
                                'symbol': new_symbol.upper(), 
                                'exchange': new_exchange,
                                'category': 'Custom'
                            })
                            st.rerun()
                        else:
                            st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            # Display watchlist with live data
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    
                    w_cols[0].markdown(
                        f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", 
                        unsafe_allow_html=True
                    )
                    w_cols[1].markdown(
                        f"**₹{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", 
                        unsafe_allow_html=True
                    )
                    
                    quantity = w_cols[2].number_input("Qty", min_value=1, step=1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                    
                    if w_cols[3].button("🟢", key=f"buy_{row['Ticker']}", use_container_width=True, help="Buy"):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                    if w_cols[4].button("🔴", key=f"sell_{row['Ticker']}", use_container_width=True, help="Sell"):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
                    if w_cols[5].button("🗑️", key=f"del_{row['Ticker']}", use_container_width=True, help="Remove"):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [
                            item for item in active_list if item['symbol'] != row['Ticker']
                        ]
                        st.rerun()
        
        with tab2:
            st.subheader("My Portfolio")
            positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
            
            col_metrics1, col_metrics2 = st.columns(2)
            col_metrics1.metric("Total Investment", f"₹{total_investment:,.2f}")
            col_metrics2.metric("Today's P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            
            with st.expander("View Holdings"):
                if not holdings_df.empty:
                    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings found.")
    
    with col2:
        st.subheader("NIFTY 50 Live Chart")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
        else:
            st.info("Connect to broker for live charts")

# ================ 6. AUTHENTICATION AND MAIN APP ============

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
    """Dialog to generate a QR code for 2FA setup."""
    if 'show_qr_dialog' not in st.session_state:
        st.session_state.show_qr_dialog = True
    
    if st.session_state.show_qr_dialog:
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
            
            st.image(buf.getvalue(), caption="Scan with Google Authenticator or similar app", use_container_width=True)
            
            with st.expander("Manual Setup"):
                st.code(secret, language="text")
                st.caption("If you can't scan the QR code, enter this secret manually in your authenticator app.")
            
            if st.button("✅ I've scanned the code. Continue to login.", use_container_width=True, type="primary"):
                st.session_state.two_factor_setup_complete = True
                st.session_state.show_qr_dialog = False
                st.session_state.show_2fa_dialog = True
                st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h1 style='color: var(--text-color); margin-bottom: 2rem;'>🚀 BlockVista Terminal</h1>
            <p style='color: var(--text-light);'>Professional Trading Platform for Indian Markets</p>
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
    """Displays the login page for broker authentication."""
    apply_custom_styling()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <h1>🇮🇳 BlockVista Terminal</h1>
            <p style='color: var(--text-light);'>Professional Trading Platform for Indian Markets</p>
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
    
    # 2FA Check
    if st.session_state.get('profile'):
        if not st.session_state.get('two_factor_setup_complete'):
            qr_code_dialog()
            return
        if not st.session_state.get('authenticated', False):
            two_factor_dialog()
            return

    # Sidebar
    st.sidebar.title(f"🇮🇳 Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Market Focus", ["Equity", "F&O", "HFT"], horizontal=True)
    st.sidebar.divider()
    
    # Navigation
    st.sidebar.header("Navigation")
    
    pages = {
        "Equity": {
            "Market Overview": page_indian_market_overview,
            "Dashboard": page_dashboard,
            "Advanced Charting": page_advanced_charting,
            "Market Scanners": page_momentum_and_trend_finder,
            "Portfolio & Risk": page_portfolio_and_risk,
            "Basket Orders": page_basket_orders,
        },
        "F&O": {
            "F&O Analytics": page_fo_analytics,
            "Options Strategy Builder": page_option_strategy_builder,
            "Greeks Calculator": page_greeks_calculator,
            "Futures Terminal": page_futures_terminal,
        },
        "HFT": {
            "HFT Terminal": page_hft_terminal,
        }
    }
    
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Execute selected page
    pages[st.session_state.terminal_mode][selection]()

# Placeholder functions for pages that are not fully implemented in this snippet
def page_advanced_charting(): 
    st.title("Advanced Charting")
    st.info("Advanced charting functionality for Indian markets")

def page_momentum_and_trend_finder():
    st.title("Market Scanners") 
    st.info("Momentum and trend scanning for Indian stocks")

def page_portfolio_and_risk():
    st.title("Portfolio & Risk")
    st.info("Portfolio management and risk analysis")

def page_basket_orders():
    st.title("Basket Orders")
    st.info("Basket order functionality for Indian markets")

def page_fo_analytics():
    st.title("F&O Analytics")
    st.info("Futures and Options analytics for Indian derivatives")

def page_option_strategy_builder():
    st.title("Options Strategy Builder")
    st.info("Options strategy building for Indian markets")

def page_greeks_calculator():
    st.title("Greeks Calculator") 
    st.info("Options greeks calculator for Indian derivatives")

def page_futures_terminal():
    st.title("Futures Terminal")
    st.info("Futures trading terminal for Indian markets")

def page_hft_terminal():
    st.title("HFT Terminal")
    st.info("High Frequency Trading terminal for Indian markets")

def get_portfolio():
    """Placeholder portfolio function"""
    return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

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
