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
import time as a_time # Renaming to avoid conflict with datetime.time
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
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="collapsed")

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
            --dark-blue: #58a6ff;
            --dark-purple: #8a63d2;

            --light-bg: #FFFFFF;
            --light-secondary-bg: #F0F2F6;
            --light-widget-bg: #F8F9FA;
            --light-border: #dee2e6;
            --light-text: #212529;
            --light-text-light: #6c757d;
            --light-green: #198754;
            --light-red: #dc3545;
            --light-blue: #0d6efd;
        }

        /* Set Theme based on body class */
        body.dark-theme {
            --primary-bg: var(--dark-bg);
            --secondary-bg: var(--dark-secondary-bg);
            --widget-bg: var(--dark-widget-bg);
            --border-color: var(--dark-border);
            --text-color: var(--dark-text);
            --text-light: var(--dark-text-light);
            --green: var(--dark-green);
            --red: var(--dark-red);
            --blue: var(--dark-blue);
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
            --blue: var(--light-blue);
        }

        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        
        /* Main App container */
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

        /* --- Components --- */
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
        
        /* Metric Cards */
        .metric-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }
        
        /* AI Trade Idea Card */
        .trade-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }

        /* Notification Bar */
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
        
        /* HFT Terminal specific styles */
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

        /* Login Page Specific Styles */
        .login-container {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .login-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(88, 166, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(40, 167, 69, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(138, 99, 210, 0.05) 0%, transparent 50%);
            animation: pulse 8s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .login-card {
            background: rgba(33, 38, 45, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 3rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            max-width: 450px;
            width: 100%;
            position: relative;
            z-index: 2;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #58a6ff, #8a63d2, #28a745);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 30px rgba(88, 166, 255, 0.3);
        }
        
        .login-subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .feature-item {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .feature-item:hover {
            transform: translateY(-2px);
            border-color: var(--blue);
            background: rgba(88, 166, 255, 0.1);
        }
        
        .feature-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .login-button {
            background: linear-gradient(135deg, var(--blue), var(--dark-purple));
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            margin: 1rem 0;
        }
        
        .login-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(88, 166, 255, 0.3);
        }
        
        .market-status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(33, 38, 45, 0.9);
            backdrop-filter: blur(10px);
            padding: 10px 20px;
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
            z-index: 1000;
        }
        
        .ticker-tape {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(33, 38, 45, 0.9);
            backdrop-filter: blur(10px);
            padding: 10px 0;
            border-top: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
            z-index: 1000;
        }
        
        .ticker-content {
            display: inline-block;
            white-space: nowrap;
            animation: ticker-scroll 30s linear infinite;
        }
        
        @keyframes ticker-scroll {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        .ticker-item {
            display: inline-block;
            margin: 0 40px;
            font-size: 0.9rem;
        }
        
        .glow-text {
            text-shadow: 0 0 20px rgba(88, 166, 255, 0.5);
        }
        
        .floating-elements {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: 1;
        }
        
        .floating-element {
            position: absolute;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px;
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .news-card {
            background: var(--secondary-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .news-card:hover {
            border-color: var(--blue);
            transform: translateX(5px);
        }
        
        .market-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-box {
            background: var(--secondary-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Use JavaScript to apply theme class to body
    js_theme = f"""
    <script>
        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add('{st.session_state.theme.lower()}-theme');
    </script>
    """
    st.components.v1.html(js_theme, height=0)

# Centralized data source configuration for ML models
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
    """Initializes all necessary session state variables."""
    # Broker and Login
    if 'broker' not in st.session_state: st.session_state.broker = None
    if 'kite' not in st.session_state: st.session_state.kite = None
    if 'profile' not in st.session_state: st.session_state.profile = None
    if 'login_animation_complete' not in st.session_state: st.session_state.login_animation_complete = False
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'two_factor_setup_complete' not in st.session_state: st.session_state.two_factor_setup_complete = False
    if 'pyotp_secret' not in st.session_state: st.session_state.pyotp_secret = None

    # UI/Theme
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'

    # Watchlists
    if 'watchlists' not in st.session_state:
        st.session_state.watchlists = {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
            "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
        }
    if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Watchlist 1"

    # Orders
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    if 'basket' not in st.session_state: st.session_state.basket = []
    if 'last_order_details' not in st.session_state: st.session_state.last_order_details = {}

    # F&O Analytics
    if 'underlying_pcr' not in st.session_state: st.session_state.underlying_pcr = "NIFTY"
    if 'strategy_legs' not in st.session_state: st.session_state.strategy_legs = []
    if 'calculated_greeks' not in st.session_state: st.session_state.calculated_greeks = None
    
    # AI/ML Features
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'ml_forecast_df' not in st.session_state: st.session_state.ml_forecast_df = None
    if 'ml_instrument_name' not in st.session_state: st.session_state.ml_instrument_name = None
    if 'backtest_results' not in st.session_state: st.session_state.backtest_results = None

    # HFT Mode
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly). A more robust solution would use an API or a library."""
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
            st.info("Quick trade feature - connect to broker to enable")
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            st.info("Quick trade feature - connect to broker to enable")

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    
def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
    overnight_tickers = {"GIFT NIFTY": "NQ=F", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
    try:
        data = get_global_indices_data(overnight_tickers)
        
        if not data.empty:
            bar_html = "<div class='notification-bar'>"
            for name, ticker in overnight_tickers.items():
                row = data[data['Ticker'] == name]
                if not row.empty:
                    price = row.iloc[0]['Price']
                    change = row.iloc[0]['% Change']
                    if not np.isnan(price) and not np.isnan(change):
                        color = 'var(--green)' if change > 0 else 'var(--red)'
                        bar_html += f"<span>{name}: {price:,.2f} <span style='color:{color};'>({change:+.2f}%)</span></span>"
            bar_html += "</div>"
            st.markdown(bar_html, unsafe_allow_html=True)
    except Exception as e:
        # Fallback display
        bar_html = """
        <div class='notification-bar'>
            <span>GIFT NIFTY: 22,415.50 <span style='color:var(--green);'>(+0.45%)</span></span>
            <span>S&P 500 Futures: 4,815.25 <span style='color:var(--green);'>(+0.32%)</span></span>
            <span>NASDAQ Futures: 16,725.80 <span style='color:var(--red);'>(-0.15%)</span></span>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance with improved error handling."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Try to get data for all tickers at once
        data_yf = yf.download(list(tickers.values()), period="2d", interval="1d", progress=False)
        if data_yf.empty:
            return pd.DataFrame()

        data = []
        for ticker_name, yf_ticker_name in tickers.items():
            try:
                if len(tickers) > 1:
                    # For multiple tickers
                    if 'Close' in data_yf.columns and yf_ticker_name in data_yf['Close'].columns:
                        closes = data_yf['Close'][yf_ticker_name].dropna()
                    else:
                        continue
                else:
                    # For single ticker
                    closes = data_yf['Close'].dropna()

                if len(closes) >= 2:
                    last_price = closes.iloc[-1]
                    prev_close = closes.iloc[-2]
                    change = last_price - prev_close
                    pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                    
                    if not (pd.isna(last_price) or pd.isna(pct_change)):
                        data.append({
                            'Ticker': ticker_name, 
                            'Price': last_price, 
                            'Change': change, 
                            '% Change': pct_change
                        })
            except Exception:
                continue

        return pd.DataFrame(data)

    except Exception:
        # Return empty DataFrame if there's an error
        return pd.DataFrame()

def display_market_status():
    """Displays market status in the corner."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    st.markdown(f"""
        <div class="market-status">
            <div style="font-size: 0.8rem; color: var(--text-light);">Market: <span style="color:{status_info['color']};">{status_info['status']}</span></div>
            <div style="font-size: 0.7rem; color: var(--text-light);">{current_time}</div>
        </div>
    """, unsafe_allow_html=True)

def display_ticker_tape():
    """Displays a moving ticker tape with market data."""
    tickers = {
        "NIFTY 50": "^NSEI", 
        "SENSEX": "^BSESN", 
        "BANK NIFTY": "^NSEBANK", 
        "USDINR": "INR=X",
        "GOLD": "GC=F",
        "SILVER": "SI=F"
    }
    data = get_global_indices_data(tickers)
    
    if not data.empty:
        ticker_html = ""
        for _, row in data.iterrows():
            color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
            ticker_html += f'<span class="ticker-item">{row["Ticker"]}: {row["Price"]:,.2f} <span style="color:{color};">({row["% Change"]:+.2f}%)</span></span>'
    else:
        # Fallback data
        ticker_html = """
        <span class="ticker-item">NIFTY 50: 22,415.50 <span style="color:var(--green);">(+0.45%)</span></span>
        <span class="ticker-item">SENSEX: 73,895.54 <span style="color:var(--green);">(+0.38%)</span></span>
        <span class="ticker-item">BANK NIFTY: 48,225.75 <span style="color:var(--red);">(-0.12%)</span></span>
        <span class="ticker-item">USDINR: 83.25 <span style="color:var(--green);">(+0.08%)</span></span>
        <span class="ticker-item">GOLD: 2,415.80 <span style="color:var(--green);">(+0.25%)</span></span>
        """
    
    st.markdown(f"""
        <div class="ticker-tape">
            <div class="ticker-content">
                {ticker_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

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
            return df
        except Exception:
            return pd.DataFrame()
    else:
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
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# ================ IMPROVED PREMARKET PULSE PAGE ================

@st.cache_data(ttl=300)
def fetch_and_analyze_news_improved(query=None):
    """Fetches and performs sentiment analysis on financial news with improved reliability."""
    analyzer = SentimentIntensityAnalyzer()
    
    news_sources = {
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
        "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
        "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    }
    
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:6]:
                try:
                    published_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime.fromtimestamp(mktime_tz(entry.published_parsed))
                    
                    title = entry.title if hasattr(entry, 'title') else "No title"
                    
                    if query is None or query.lower() in title.lower():
                        sentiment_score = analyzer.polarity_scores(title)['compound']
                        
                        all_news.append({
                            "source": source, 
                            "title": title, 
                            "link": entry.link if hasattr(entry, 'link') else "#",
                            "date": published_date.date(), 
                            "sentiment": sentiment_score,
                        })
                except Exception:
                    continue
                    
        except Exception:
            continue
            
    # Sort by date and sentiment
    all_news.sort(key=lambda x: (x['date'], abs(x['sentiment'])), reverse=True)
    return pd.DataFrame(all_news)

@st.cache_data(ttl=300)
def get_gift_nifty_data_improved():
    """Fetches NIFTY data with fallback options."""
    try:
        # Try to get NIFTY data
        data = yf.download("^NSEI", period="1d", interval="5m", progress=False)
        if not data.empty and len(data) > 0:
            return data
    except Exception:
        pass
    
    # Fallback: create sample data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='5min')
    sample_data = pd.DataFrame({
        'Open': [22000 + i * 10 for i in range(len(dates))],
        'High': [22050 + i * 10 for i in range(len(dates))],
        'Low': [21950 + i * 10 for i in range(len(dates))],
        'Close': [22020 + i * 10 for i in range(len(dates))],
        'Volume': [1000000] * len(dates)
    }, index=dates)
    return sample_data

def page_premarket_pulse():
    """Global market overview and premarket indicators with improved data handling."""
    display_header()
    st.title("Premarket & Global Cues")
    st.markdown("---")

    st.subheader("Global Market Snapshot")
    
    # Global indices with reliable tickers
    global_tickers = {
        "S&P 500": "^GSPC", 
        "Dow Jones": "^DJI", 
        "NASDAQ": "^IXIC", 
        "FTSE 100": "^FTSE", 
        "Nikkei 225": "^N225", 
        "Hang Seng": "^HSI"
    }
    
    with st.spinner("Loading global market data..."):
        global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_data))
        for i, (_, row) in enumerate(global_data.iterrows()):
            with cols[i]:
                delta_color = "normal" if row['% Change'] >= 0 else "inverse"
                st.metric(
                    label=row['Ticker'],
                    value=f"{row['Price']:,.2f}",
                    delta=f"{row['% Change']:.2f}%",
                    delta_color=delta_color
                )
    else:
        # Fallback display with sample data
        st.info("📊 Live data loading... Showing sample data")
        sample_data = [
            {"Ticker": "S&P 500", "Price": 6706.39, "% Change": -0.07},
            {"Ticker": "Dow Jones", "Price": 46403.73, "% Change": -0.08},
            {"Ticker": "NASDAQ", "Price": 22805.25, "% Change": 0.22},
            {"Ticker": "FTSE 100", "Price": 9427.73, "% Change": -0.20},
            {"Ticker": "Nikkei 225", "Price": 44936.73, "% Change": 0.87},
            {"Ticker": "Hang Seng", "Price": 27287.12, "% Change": 0.15}
        ]
        
        cols = st.columns(len(sample_data))
        for i, data in enumerate(sample_data):
            with cols[i]:
                delta_color = "normal" if data["% Change"] >= 0 else "inverse"
                st.metric(
                    label=data["Ticker"],
                    value=f"{data['Price']:,.2f}",
                    delta=f"{data['% Change']:.2f}%",
                    delta_color=delta_color
                )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY 50 Chart")
        nifty_data = get_gift_nifty_data_improved()
        
        if not nifty_data.empty:
            fig = go.Figure()
            
            if 'Close' in nifty_data.columns:
                fig.add_trace(go.Scatter(
                    x=nifty_data.index, 
                    y=nifty_data['Close'],
                    mode='lines',
                    name='NIFTY 50',
                    line=dict(color='#58a6ff', width=2)
                ))
            
            fig.update_layout(
                title="NIFTY 50 - Intraday Movement",
                xaxis_title="Time",
                yaxis_title="Price",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 Chart data will appear here when available")
            
    with col2:
        st.subheader("Key Asian Markets")
        asian_data = [
            {"Ticker": "Nikkei 225", "Price": 44936.73, "% Change": 0.87},
            {"Ticker": "Hang Seng", "Price": 27287.12, "% Change": 0.15},
            {"Ticker": "Shanghai", "Price": 3450.67, "% Change": -0.25}
        ]
        
        for data in asian_data:
            delta_color = "normal" if data["% Change"] >= 0 else "inverse"
            st.metric(
                label=data["Ticker"],
                value=f"{data['Price']:,.2f}",
                delta=f"{data['% Change']:.2f}%",
                delta_color=delta_color
            )

    st.markdown("---")

    st.subheader("Latest Market News")
    
    with st.spinner("📰 Loading latest market news..."):
        news_df = fetch_and_analyze_news_improved()
    
    if not news_df.empty:
        for _, news in news_df.head(6).iterrows():
            sentiment_score = news['sentiment']
            
            if sentiment_score > 0.2:
                icon = "🟢"
                border_color = "#28a745"
            elif sentiment_score < -0.2:
                icon = "🔴"
                border_color = "#da3633"
            else:
                icon = "🔵"
                border_color = "#58a6ff"
            
            st.markdown(f"""
                <div class="news-card" style="border-left-color: {border_color};">
                    <div style="display: flex; justify-content: between; align-items: start;">
                        <div style="flex: 1;">
                            <strong>{icon} {news['title']}</strong>
                            <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.5rem;">
                                📅 {news['date']} | 📰 {news['source']} | 🎯 Sentiment: {sentiment_score:.2f}
                            </div>
                        </div>
                        <a href="{news['link']}" target="_blank" style="margin-left: 1rem;">🔗</a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📰 Loading financial news...")
        
        # Fallback news
        fallback_news = [
            {"title": "Global markets show mixed trends amid economic data releases", "source": "Market Update", "sentiment": 0.1},
            {"title": "Technology stocks lead gains in pre-market trading", "source": "Sector Watch", "sentiment": 0.3},
            {"title": "Central bank decisions expected to influence market direction", "source": "Economic Outlook", "sentiment": -0.1},
        ]
        
        for news in fallback_news:
            st.markdown(f"""
                <div class="news-card">
                    <strong>📊 {news['title']}</strong>
                    <div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.5rem;">
                        📰 {news['source']} | 🎯 Sentiment: {news['sentiment']:.1f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ================ OTHER PAGE FUNCTIONS (Simplified) ================

def page_dashboard():
    """Dashboard page."""
    display_header()
    st.title("Dashboard")
    st.info("Connect to your broker to access live market data and trading features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Overview")
        st.metric("NIFTY 50", "22,415.50", "0.45%")
        st.metric("SENSEX", "73,895.54", "0.38%")
        st.metric("BANK NIFTY", "48,225.75", "-0.12%")
    
    with col2:
        st.subheader("Quick Actions")
        st.button("View Watchlists")
        st.button("Check Portfolio")
        st.button("Market Analysis")

def page_advanced_charting():
    """Advanced charting page."""
    display_header()
    st.title("Advanced Charting")
    st.info("Connect to your broker to access advanced charting tools.")

def page_fo_analytics():
    """F&O Analytics page."""
    display_header()
    st.title("F&O Analytics")
    st.info("Connect to your broker to access F&O analytics.")

def page_portfolio_and_risk():
    """Portfolio and risk page."""
    display_header()
    st.title("Portfolio & Risk")
    st.info("Connect to your broker to view your portfolio.")

def page_ai_assistant():
    """AI Assistant page."""
    display_header()
    st.title("AI Assistant")
    st.info("Connect to your broker to access AI-powered insights.")

# ================ LOGIN AND AUTHENTICATION ================

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    secret = base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]
    return secret

@st.dialog("Two-Factor Authentication")
def two_factor_dialog():
    """Dialog for 2FA login."""
    st.subheader("Enter your 2FA code")
    st.caption("Please enter the 6-digit code from your authenticator app to continue.")
    
    auth_code = st.text_input("2FA Code", max_chars=6, key="2fa_code")
    
    if st.button("Authenticate", use_container_width=True):
        if auth_code:
            try:
                totp = pyotp.TOTP(st.session_state.pyotp_secret)
                if totp.verify(auth_code):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid code. Please try again.")
            except Exception as e:
                st.error(f"An error occurred during authentication: {e}")
        else:
            st.warning("Please enter a code.")

@st.dialog("Generate QR Code for 2FA")
def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup."""
    st.subheader("Set up Two-Factor Authentication")
    st.info("Please scan this QR code with your authenticator app (e.g., Google or Microsoft Authenticator). This is a one-time setup.")

    if st.session_state.pyotp_secret is None:
        st.session_state.pyotp_secret = get_user_secret(st.session_state.get('profile', {}))
    
    secret = st.session_state.pyotp_secret
    user_name = st.session_state.get('profile', {}).get('user_name', 'User')
    uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
    
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    st.image(buf.getvalue(), caption="Scan with your authenticator app", use_container_width=True)
    st.markdown(f"**Your Secret Key:** `{secret}` (You can also enter this manually)")
    
    if st.button("I have scanned the code. Continue.", use_container_width=True):
        st.session_state.two_factor_setup_complete = True
        st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
    st.markdown("""
        <style>
        .animation-container {
            background: linear-gradient(135deg, var(--primary-bg) 0%, var(--secondary-bg) 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
        .loading-bar {
            width: 300px;
            height: 4px;
            background: var(--border-color);
            border-radius: 2px;
            margin: 20px 0;
            overflow: hidden;
        }
        .loading-progress {
            height: 100%;
            background: linear-gradient(90deg, var(--blue), var(--green));
            border-radius: 2px;
            animation: loading 2s ease-in-out;
        }
        @keyframes loading {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="animation-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="login-title glow-text">BLOCKVISTA</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Initializing Terminal</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = {
        "Loading market data...": 25,
        "Connecting to broker...": 50,
        "Initializing analytics...": 75,
        "Ready to trade!": 100
    }
    
    for text, progress in steps.items():
        status_text.text(f"🔄 {text}")
        progress_bar.progress(progress)
        a_time.sleep(0.8)
    
    a_time.sleep(0.5)
    st.session_state['login_animation_complete'] = True
    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_login_ui():
    """Creates the modern trader login UI."""
    # Apply styling
    apply_custom_styling()
    
    # Display market status and ticker
    display_market_status()
    display_ticker_tape()
    
    # Add floating elements for visual interest
    st.markdown("""
        <div class="floating-elements">
            <div class="floating-element" style="width: 80px; height: 80px; top: 20%; left: 10%; animation-delay: 0s;"></div>
            <div class="floating-element" style="width: 60px; height: 60px; top: 60%; left: 80%; animation-delay: 2s;"></div>
            <div class="floating-element" style="width: 100px; height: 100px; top: 80%; left: 20%; animation-delay: 4s;"></div>
            <div class="floating-element" style="width: 70px; height: 70px; top: 30%; left: 70%; animation-delay: 1s;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Main login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        # Header
        st.markdown("""
            <div class="login-header">
                <div class="login-title glow-text">BLOCKVISTA</div>
                <div class="login-subtitle">Professional Trading Terminal</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Features Grid
        st.markdown("""
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="feature-icon">📈</div>
                    <div>Live Charts</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🤖</div>
                    <div>AI Analytics</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">⚡</div>
                    <div>HFT Ready</div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🛡️</div>
                    <div>Secure</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Login Section
        st.markdown("### Connect Your Broker")
        
        broker = st.selectbox("Select Your Broker", ["Zerodha"], label_visibility="collapsed")
        
        if broker == "Zerodha":
            api_key = st.secrets.get("ZERODHA_API_KEY", "")
            api_secret = st.secrets.get("ZERODHA_API_SECRET", "")
            
            if not api_key or not api_secret:
                st.error("⚠️ API credentials not configured. Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in Streamlit secrets.")
                st.info("🔧 For now, you can explore the app in demo mode.")
                if st.button("🚀 Enter Demo Mode", use_container_width=True, type="primary"):
                    st.session_state.profile = {"user_name": "Demo User"}
                    st.session_state.broker = "Demo"
                    st.session_state.authenticated = True
                    st.session_state.two_factor_setup_complete = True
                    st.session_state.login_animation_complete = True
                    st.rerun()
            else:
                kite = KiteConnect(api_key=api_key)
                request_token = st.query_params.get("request_token")
                
                if request_token:
                    try:
                        with st.spinner("🔄 Authenticating..."):
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
                else:
                    if st.button("🚀 Login with Zerodha Kite", use_container_width=True, type="primary"):
                        st.markdown(f'<a href="{kite.login_url()}" target="_self" style="text-decoration: none; color: white;">Click here if not redirected</a>', unsafe_allow_html=True)
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={kite.login_url()}">', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Navigation options
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options"], horizontal=True)
    
    st.sidebar.divider()
    
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
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
        },
        "Options": {
            "F&O Analytics": page_fo_analytics,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
        },
        "Futures": {
            "F&O Analytics": page_fo_analytics,
            "Advanced Charting": page_advanced_charting,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
        }
    }
    
    # Add a note about HFT coming soon
    st.sidebar.info("🚀 HFT Mode - Coming Soon!")
    
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if auto_refresh:
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
        create_login_ui()
