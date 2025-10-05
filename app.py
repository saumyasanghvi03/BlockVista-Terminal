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
import time as a_time  # Renaming to avoid conflict with datetime.time
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# ================ 1. STYLING AND CONFIGURATION ===============

st.set_page_config(
    page_title="BlockVista Terminal Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

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
        --dark-orange: #db6d28;
        --dark-purple: #8957e5;
        --light-bg: #FFFFFF;
        --light-secondary-bg: #F0F2F6;
        --light-widget-bg: #F8F9FA;
        --light-border: #dee2e6;
        --light-text: #212529;
        --light-text-light: #6c757d;
        --light-green: #198754;
        --light-red: #dc3545;
        --light-orange: #fd7e14;
        --light-purple: #6f42c1;
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
        --orange: var(--dark-orange);
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
        --purple: var(--light-purple);
    }

    body {
        background-color: var(--primary-bg);
        color: var(--text-color);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main App container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3, h4, h5 {
        color: var(--text-color) !important;
        font-weight: 600;
    }

    hr {
        background: var(--border-color);
    }

    /* -- Components -- */
    .stButton>button {
        border-color: var(--border-color);
        background-color: var(--widget-bg);
        color: var(--text-color);
        font-weight: 500;
        border-radius: 8px;
    }

    .stButton>button:hover {
        border-color: var(--green);
        color: var(--green);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }

    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
        background-color: var(--widget-bg);
        border-color: var(--border-color);
        color: var(--text-color);
        border-radius: 8px;
    }

    .stRadio>div {
        background-color: var(--widget-bg);
        border: 1px solid var(--border-color);
        padding: 8px;
        border-radius: 8px;
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background-color: var(--secondary-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 12px;
        border-left-width: 5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    /* AI Trade Signal Card */
    .signal-card {
        background-color: var(--secondary-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .signal-buy {
        border-left: 5px solid var(--green);
    }

    .signal-sell {
        border-left: 5px solid var(--red);
    }

    .signal-neutral {
        border-left: 5px solid var(--orange);
    }

    /* Premium Badge */
    .premium-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #000;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 8px;
    }

    /* Subscription Tier Cards */
    .tier-card {
        background: var(--widget-bg);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .tier-card:hover {
        border-color: var(--green);
        transform: translateY(-3px);
    }

    .tier-card.featured {
        border-color: var(--orange);
        background: linear-gradient(135deg, var(--widget-bg), var(--secondary-bg));
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
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
    }

    .hft-depth-ask {
        background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05));
        padding: 4px 8px;
        margin: 2px 0;
        border-radius: 4px;
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

    /* Fundamental Analysis Styles */
    .fundamental-metric {
        background: var(--widget-bg);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--green);
        margin: 0.5rem 0;
    }

    .fundamental-metric.warning {
        border-left-color: var(--red);
    }

    .fundamental-metric.neutral {
        border-left-color: var(--orange);
    }

    /* ML Model Cards */
    .model-card {
        background: var(--secondary-bg);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
    }

    .model-card.arima {
        border-left: 5px solid #1f77b4;
    }

    .model-card.lstm {
        border-left: 5px solid #2ca02c;
    }

    .model-card.xgboost {
        border-left: 5px solid #d62728;
    }

    /* Smart Money Flow Styles */
    .smart-money-buy {
        background: linear-gradient(90deg, rgba(0,255,0,0.1), transparent);
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
    }

    .smart-money-sell {
        background: linear-gradient(90deg, rgba(255,0,0,0.1), transparent);
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
    }

    /* Sector Rotation Styles */
    .sector-hot {
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
    }

    .sector-warm {
        background: linear-gradient(90deg, #ffa500, #ffd93d);
        color: black;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
    }

    .sector-cold {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
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

# Subscription Plans Configuration
SUBSCRIPTION_PLANS = {
    "Basic": {
        "price": 999,
        "features": [
            "Real-time NSE/BSE Data",
            "Basic Charting & Indicators",
            "5 Watchlists",
            "Standard Options Chain",
            "Email Support"
        ],
        "limits": {
            "ai_signals": 0,
            "basket_orders": 0,
            "advanced_analytics": False,
            "api_access": False
        }
    },
    "Professional": {
        "price": 2499,
        "features": [
            "All Basic Features",
            "Advanced Charting",
            "AI Trade Signals (10/day)",
            "Options Flow Analytics",
            "Basket Orders",
            "F&O Analytics",
            "Priority Support",
            "Mobile App Access"
        ],
        "limits": {
            "ai_signals": 10,
            "basket_orders": 5,
            "advanced_analytics": True,
            "api_access": False
        }
    },
    "Institutional": {
        "price": 7999,
        "features": [
            "All Professional Features",
            "Unlimited AI Signals",
            "API Access",
            "White-label Solutions",
            "Custom Indicators",
            "Dedicated Account Manager",
            "Advanced Risk Analytics",
            "Multi-user Access"
        ],
        "limits": {
            "ai_signals": 9999,
            "basket_orders": 9999,
            "advanced_analytics": True,
            "api_access": True
        }
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
    
    # Subscription
    if 'subscription_tier' not in st.session_state: st.session_state.subscription_tier = "Professional"
    if 'ai_signals_used' not in st.session_state: st.session_state.ai_signals_used = 0
    if 'basket_orders_used' not in st.session_state: st.session_state.basket_orders_used = 0
    
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
    if 'ml_model_type' not in st.session_state: st.session_state.ml_model_type = None
    if 'ml_metrics' not in st.session_state: st.session_state.ml_metrics = {}
    if 'ml_message' not in st.session_state: st.session_state.ml_message = ""
    if 'ml_previous_models' not in st.session_state: st.session_state.ml_previous_models = []
    
    # Enhanced AI Features
    if 'ai_trade_signals' not in st.session_state: st.session_state.ai_trade_signals = []
    if 'smart_money_flow' not in st.session_state: st.session_state.smart_money_flow = {}
    if 'sector_rotation' not in st.session_state: st.session_state.sector_rotation = {}
    if 'options_flow' not in st.session_state: st.session_state.options_flow = {}
    
    # HFT Mode
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []
    
    # Fundamental Analysis
    if 'fundamental_data' not in st.session_state: st.session_state.fundamental_data = {}
    if 'comparison_symbols' not in st.session_state: st.session_state.comparison_symbols = []

# ================ 2. ENHANCED HELPER FUNCTIONS ================

def check_subscription_feature(feature_name):
    """Check if current subscription allows a feature."""
    tier = st.session_state.get('subscription_tier', 'Basic')
    limits = SUBSCRIPTION_PLANS[tier]['limits']
    
    if feature_name == "ai_signals":
        if st.session_state.ai_signals_used >= limits['ai_signals']:
            st.error(f"❌ AI Signals limit reached for {tier} plan. Upgrade to access more signals.")
            return False
        st.session_state.ai_signals_used += 1
        return True
    
    elif feature_name == "basket_orders":
        if st.session_state.basket_orders_used >= limits['basket_orders']:
            st.error(f"❌ Basket Orders limit reached for {tier} plan. Upgrade for unlimited orders.")
            return False
        st.session_state.basket_orders_used += 1
        return True
    
    elif feature_name == "advanced_analytics":
        if not limits['advanced_analytics']:
            st.error(f"❌ Advanced Analytics not available in {tier} plan. Upgrade to Professional or Institutional.")
            return False
        return True
    
    elif feature_name == "api_access":
        if not limits['api_access']:
            st.error(f"❌ API Access not available in {tier} plan. Upgrade to Institutional.")
            return False
        return True
    
    return True

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    instrument_df = get_instrument_df()
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    
    if symbol is None:
        symbol = st.text_input("Symbol").upper()
    
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
    order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
    quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
    price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0
    
    if st.button("Submit Order", use_container_width=True):
        if symbol and quantity > 0:
            place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

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

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal Pro</h1>', unsafe_allow_html=True)
        tier = st.session_state.get('subscription_tier', 'Basic')
        st.markdown(f'<small style="color: var(--text-light);">{tier} Plan • AI-Powered Trading</small>', unsafe_allow_html=True)
    
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

# ================ 3. ENHANCED CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None, comparison_data=None):
    """Generates a Plotly chart with various chart types and overlays."""
    fig = go.Figure()
    
    if df.empty and not comparison_data:
        return fig

    # Handle single symbol chart
    if not df.empty:
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
            fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], 
                                        low=ha_df['HA_low'], close=ha_df['HA_close'], name=f'{ticker} - Heikin-Ashi'))
        elif chart_type == 'Line':
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name=f'{ticker} - Line'))
        elif chart_type == 'Bar':
            fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], 
                                 low=chart_df['low'], close=chart_df['close'], name=f'{ticker} - Bar'))
        else:
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], 
                                        low=chart_df['low'], close=chart_df['close'], name=f'{ticker} - Candlestick'))

        # Add Bollinger Bands if available
        bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
        bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
        if bbl_col and bbu_col:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbl_col], line=dict(color='rgba(135,206,250,0.5)', width=1), name=f'{ticker} - Lower Band'))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbu_col], line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name=f'{ticker} - Upper Band'))

    # Handle comparison data
    if comparison_data:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, (comp_ticker, comp_df) in enumerate(comparison_data.items()):
            if not comp_df.empty:
                # Normalize prices for better comparison (start at 100)
                if 'close' in comp_df.columns:
                    normalized_close = (comp_df['close'] / comp_df['close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(
                        x=comp_df.index,
                        y=normalized_close,
                        mode='lines',
                        name=f'{comp_ticker} (Normalized)',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))

    # Add forecast if available
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', 
                                line=dict(color='yellow', dash='dash'), name='Forecast'))
    
    # Add confidence intervals if available
    if conf_int_df is not None:
        fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
        fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))

    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    chart_title = f'{ticker} Price Chart ({chart_type})' if not comparison_data else f'Price Comparison: {ticker} vs {", ".join(comparison_data.keys())}'
    
    fig.update_layout(title=chart_title, yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, 
                     template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    client = get_broker_client()
    if not client:
        return pd.DataFrame()
    
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
    if instrument_df.empty:
        return None
    
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API."""
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
            
            # Add technical indicators silently
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True)
                df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True)
                df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True)
                df.ta.coppock(append=True); df.ta.ema(length=50, append=True)
                df.ta.ema(length=200, append=True); df.ta.fisher(append=True)
                df.ta.kst(append=True); df.ta.macd(append=True); df.ta.mfi(append=True)
                df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True)
                df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
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

# ================ 4. ENHANCED AI & ML FEATURES ================

def calculate_technical_score(symbol, timeframe='1h'):
    """Calculate technical analysis score (0-100)."""
    try:
        instrument_df = get_instrument_df()
        token = get_instrument_token(symbol, instrument_df)
        
        if not token:
            return 50
        
        data = get_historical_data(token, '15minute', period='1mo')
        if data.empty or len(data) < 20:
            return 50
        
        # Calculate multiple technical indicators
        rsi = data.ta.rsi().iloc[-1] if 'RSI_14' in data.columns else 50
        macd = data.ta.macd().iloc[-1][0] if 'MACD_12_26_9' in data.columns else 0
        stoch = data.ta.stoch().iloc[-1][0] if 'STOCHk_14_3_3' in data.columns else 50
        adx = data.ta.adx().iloc[-1][0] if 'ADX_14' in data.columns else 25
        
        # Normalize scores
        rsi_score = 100 - abs(rsi - 50) * 2  # Closer to 50 is better
        macd_score = 75 if macd > 0 else 25
        stoch_score = 100 - abs(stoch - 50) * 2
        adx_score = min(adx * 2, 100)  # Higher ADX is better
        
        # Weighted average
        technical_score = (rsi_score * 0.3 + macd_score * 0.3 + stoch_score * 0.2 + adx_score * 0.2)
        
        return min(100, max(0, technical_score))
    
    except Exception as e:
        st.error(f"Technical analysis error: {e}")
        return 50

def calculate_fundamental_score(symbol):
    """Calculate fundamental analysis score (0-100)."""
    try:
        fundamental_data = get_fundamental_data(symbol)
        if not fundamental_data:
            return 50
        
        scores = []
        
        # PE Ratio scoring
        pe = fundamental_data.get('pe_ratio', 0)
        if 0 < pe < 25:
            scores.append(80)
        elif 25 <= pe < 40:
            scores.append(60)
        else:
            scores.append(40)
        
        # ROE scoring
        roe = fundamental_data.get('roe', 0)
        if roe > 15:
            scores.append(85)
        elif roe > 8:
            scores.append(65)
        else:
            scores.append(45)
        
        # Debt-to-Equity scoring
        debt_equity = fundamental_data.get('debt_to_equity', 0)
        if debt_equity < 1:
            scores.append(80)
        elif debt_equity < 2:
            scores.append(60)
        else:
            scores.append(40)
        
        return np.mean(scores) if scores else 50
    
    except Exception as e:
        st.error(f"Fundamental analysis error: {e}")
        return 50

def analyze_sentiment_flow(symbol):
    """Analyze news and social sentiment."""
    try:
        news_df = fetch_and_analyze_news(symbol)
        if not news_df.empty:
            avg_sentiment = news_df['sentiment'].mean()
            return (avg_sentiment + 1) * 50  # Convert from -1,1 to 0,100 scale
        return 50
    except:
        return 50

def analyze_smart_money_flow(symbol):
    """Analyze institutional/smart money activity."""
    # This would typically integrate with bulk deal data and FII/DII data
    # For now, return a simulated score
    return random.randint(40, 80)

def ensemble_ai_prediction(technical_score, fundamental_score, sentiment_score, smart_money_score):
    """Combine all analysis methods for final prediction."""
    weights = {
        'technical': 0.35,
        'fundamental': 0.25,
        'sentiment': 0.20,
        'smart_money': 0.20
    }
    
    final_score = (
        technical_score * weights['technical'] +
        fundamental_score * weights['fundamental'] +
        sentiment_score * weights['sentiment'] +
        smart_money_score * weights['smart_money']
    )
    
    return final_score

def generate_ai_trade_signal(symbol, timeframe='1h'):
    """Generate AI-powered trade signal with confidence score."""
    
    if not check_subscription_feature("ai_signals"):
        return None
    
    with st.spinner(f"🤖 AI analyzing {symbol}..."):
        # Calculate individual scores
        technical_score = calculate_technical_score(symbol, timeframe)
        fundamental_score = calculate_fundamental_score(symbol)
        sentiment_score = analyze_sentiment_flow(symbol)
        smart_money_score = analyze_smart_money_flow(symbol)
        
        # Ensemble prediction
        final_score = ensemble_ai_prediction(technical_score, fundamental_score, sentiment_score, smart_money_score)
        
        # Determine signal type
        if final_score >= 70:
            signal = "STRONG_BUY"
            confidence = final_score
        elif final_score >= 60:
            signal = "BUY"
            confidence = final_score
        elif final_score >= 40:
            signal = "HOLD"
            confidence = 100 - abs(final_score - 50) * 2
        elif final_score >= 30:
            signal = "SELL"
            confidence = 100 - final_score
        else:
            signal = "STRONG_SELL"
            confidence = 100 - final_score
        
        # Get current price data
        instrument_df = get_instrument_df()
        token = get_instrument_token(symbol, instrument_df)
        data = get_historical_data(token, '15minute', period='1d')
        
        current_price = data['close'].iloc[-1] if not data.empty else 0
        prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
        
        # Generate price targets (simplified)
        if signal in ["STRONG_BUY", "BUY"]:
            target_1 = current_price * 1.02
            target_2 = current_price * 1.04
            stop_loss = current_price * 0.98
        elif signal in ["STRONG_SELL", "SELL"]:
            target_1 = current_price * 0.98
            target_2 = current_price * 0.96
            stop_loss = current_price * 1.02
        else:
            target_1 = target_2 = stop_loss = current_price
        
        signal_data = {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'price_change': ((current_price - prev_close) / prev_close * 100) if prev_close else 0,
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
            'timeframe': timeframe,
            'technical_score': technical_score,
            'fundamental_score': fundamental_score,
            'sentiment_score': sentiment_score,
            'smart_money_score': smart_money_score,
            'timestamp': datetime.now()
        }
        
        # Store signal
        if 'ai_trade_signals' not in st.session_state:
            st.session_state.ai_trade_signals = []
        
        st.session_state.ai_trade_signals.append(signal_data)
        
        return signal_data

def track_smart_money_flow():
    """Track institutional and smart money activity."""
    try:
        # This would integrate with actual FII/DII data and bulk deals
        # For demo purposes, returning simulated data
        symbols = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK', 'SBIN']
        
        smart_money_data = {}
        for symbol in symbols:
            smart_money_data[symbol] = {
                'fii_net': random.randint(-100, 100) * 100000,
                'dii_net': random.randint(-50, 150) * 100000,
                'bulk_deals': random.randint(0, 5),
                'delivery_percentage': random.randint(30, 80),
                'sentiment': random.choice(['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL'])
            }
        
        st.session_state.smart_money_flow = smart_money_data
        return smart_money_data
    
    except Exception as e:
        st.error(f"Smart money tracking error: {e}")
        return {}

def analyze_sector_rotation():
    """Analyze sector momentum and rotation."""
    try:
        sectors = {
            'BANKING': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            'AUTO': ['TATAMOTORS', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN'],
            'METAL': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'SAIL'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR']
        }
        
        sector_data = {}
        for sector, stocks in sectors.items():
            # Calculate average performance
            total_change = 0
            count = 0
            
            for stock in stocks:
                instrument_df = get_instrument_df()
                token = get_instrument_token(stock, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period='1mo')
                    if not data.empty and len(data) > 1:
                        change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100
                        total_change += change
                        count += 1
            
            avg_change = total_change / count if count > 0 else 0
            sector_data[sector] = {
                'performance': avg_change,
                'momentum': 'HOT' if avg_change > 2 else 'WARM' if avg_change > 0 else 'COLD',
                'stocks_analyzed': count
            }
        
        st.session_state.sector_rotation = sector_data
        return sector_data
    
    except Exception as e:
        st.error(f"Sector rotation analysis error: {e}")
        return {}

# ================ 5. NEW ENHANCED PAGES ================

def page_ai_trade_signals():
    """AI-Powered Trade Signals page."""
    display_header()
    st.title("🤖 AI Trade Signals")
    st.info("Get AI-powered trading signals with multi-factor analysis", icon="🤖")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generate Signal")
        symbol = st.text_input("Stock Symbol", "RELIANCE", key="ai_signal_symbol").upper()
        timeframe = st.selectbox("Timeframe", ["15min", "1h", "4h", "1d"], index=1)
        
        if st.button("Generate AI Signal", type="primary", use_container_width=True):
            signal_data = generate_ai_trade_signal(symbol, timeframe)
            
            if signal_data:
                st.success("✅ AI Signal Generated Successfully!")
                
                # Display signal card
                signal_class = ""
                if "BUY" in signal_data['signal']:
                    signal_class = "signal-buy"
                elif "SELL" in signal_data['signal']:
                    signal_class = "signal-sell"
                else:
                    signal_class = "signal-neutral"
                
                st.markdown(f"""
                <div class="signal-card {signal_class}">
                    <h3>🎯 {signal_data['symbol']} - {signal_data['signal']}</h3>
                    <p><strong>Confidence:</strong> {signal_data['confidence']:.1f}%</p>
                    <p><strong>Current Price:</strong> ₹{signal_data['current_price']:.2f}</p>
                    <p><strong>Target 1:</strong> ₹{signal_data['target_1']:.2f}</p>
                    <p><strong>Target 2:</strong> ₹{signal_data['target_2']:.2f}</p>
                    <p><strong>Stop Loss:</strong> ₹{signal_data['stop_loss']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Subscription usage
        tier = st.session_state.get('subscription_tier', 'Basic')
        limits = SUBSCRIPTION_PLANS[tier]['limits']
        used = st.session_state.get('ai_signals_used', 0)
        
        st.metric("AI Signals Used", f"{used}/{limits['ai_signals']}")
        
        if tier == "Basic":
            st.warning("Upgrade to Professional for AI Signals")
    
    with col2:
        st.subheader("Recent Signals")
        
        if 'ai_trade_signals' in st.session_state and st.session_state.ai_trade_signals:
            signals_df = pd.DataFrame(st.session_state.ai_trade_signals)
            signals_df = signals_df.sort_values('timestamp', ascending=False).head(10)
            
            for _, signal in signals_df.iterrows():
                signal_color = "var(--green)" if "BUY" in signal['signal'] else "var(--red)" if "SELL" in signal['signal'] else "var(--orange)"
                
                st.markdown(f"""
                <div class="signal-card" style="border-left-color: {signal_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">{signal['symbol']} - {signal['signal']}</h4>
                        <span style="color: {signal_color}; font-weight: bold;">{signal['confidence']:.1f}%</span>
                    </div>
                    <p style="margin: 5px 0;">Price: ₹{signal['current_price']:.2f} | Change: {signal['price_change']:+.2f}%</p>
                    <small>Timeframe: {signal['timeframe']} | {signal['timestamp'].strftime('%H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No signals generated yet. Generate your first AI signal!")

def page_smart_money_flow():
    """Smart Money Flow Tracking page."""
    display_header()
    st.title("💰 Smart Money Flow")
    st.info("Track institutional and smart money activity", icon="💰")
    
    if not check_subscription_feature("advanced_analytics"):
        return
    
    if st.button("Refresh Smart Money Data", use_container_width=True):
        with st.spinner("Tracking smart money flow..."):
            smart_money_data = track_smart_money_flow()
    
    if 'smart_money_flow' in st.session_state and st.session_state.smart_money_flow:
        smart_money_data = st.session_state.smart_money_flow
        
        # Display smart money table
        data = []
        for symbol, flow in smart_money_data.items():
            data.append({
                'Symbol': symbol,
                'FII Net (Cr)': flow['fii_net'] / 10000000,
                'DII Net (Cr)': flow['dii_net'] / 10000000,
                'Bulk Deals': flow['bulk_deals'],
                'Delivery %': flow['delivery_percentage'],
                'Sentiment': flow['sentiment']
            })
        
        df = pd.DataFrame(data)
        
        # Apply styling
        def color_sentiment(val):
            if val == 'STRONG_BUY':
                return 'background-color: #d4edda'
            elif val == 'BUY':
                return 'background-color: #f8d7da'
            elif val == 'SELL':
                return 'background-color: #fff3cd'
            return ''
        
        styled_df = df.style.format({
            'FII Net (Cr)': '{:.2f}',
            'DII Net (Cr)': '{:.2f}'
        }).applymap(color_sentiment, subset=['Sentiment'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Smart money insights
        st.subheader("📊 Smart Money Insights")
        
        col1, col2, col3 = st.columns(3)
        
        total_fii = df['FII Net (Cr)'].sum()
        total_dii = df['DII Net (Cr)'].sum()
        
        with col1:
            st.metric("Total FII Flow", f"₹{total_fii:+.2f} Cr", 
                     delta=f"₹{total_fii:+.2f} Cr")
        
        with col2:
            st.metric("Total DII Flow", f"₹{total_dii:+.2f} Cr",
                     delta=f"₹{total_dii:+.2f} Cr")
        
        with col3:
            bullish_stocks = len(df[df['Sentiment'].isin(['STRONG_BUY', 'BUY'])])
            st.metric("Bullish Stocks", bullish_stocks)
    
    else:
        st.info("Click 'Refresh Smart Money Data' to load institutional flow data")

def page_sector_rotation():
    """Sector Rotation Analysis page."""
    display_header()
    st.title("📊 Sector Rotation")
    st.info("Identify sector momentum and rotation opportunities", icon="📊")
    
    if not check_subscription_feature("advanced_analytics"):
        return
    
    if st.button("Analyze Sector Rotation", use_container_width=True):
        with st.spinner("Analyzing sector momentum..."):
            sector_data = analyze_sector_rotation()
    
    if 'sector_rotation' in st.session_state and st.session_state.sector_rotation:
        sector_data = st.session_state.sector_rotation
        
        # Create sector performance chart
        sectors = list(sector_data.keys())
        performances = [sector_data[s]['performance'] for s in sectors]
        momentum = [sector_data[s]['momentum'] for s in sectors]
        
        fig = go.Figure()
        
        colors = []
        for mom in momentum:
            if mom == 'HOT':
                colors.append('#ff6b6b')
            elif mom == 'WARM':
                colors.append('#ffa500')
            else:
                colors.append('#4facfe')
        
        fig.add_trace(go.Bar(
            x=sectors,
            y=performances,
            marker_color=colors,
            text=[f'{p:.2f}%' for p in performances],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Sector Performance (%)",
            xaxis_title="Sectors",
            yaxis_title="Performance %",
            template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector insights
        st.subheader("🏆 Sector Insights")
        
        hot_sectors = [s for s in sectors if sector_data[s]['momentum'] == 'HOT']
        cold_sectors = [s for s in sectors if sector_data[s]['momentum'] == 'COLD']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 Hot Sectors:**")
            for sector in hot_sectors:
                st.markdown(f'<div class="sector-hot">{sector}: {sector_data[sector]["performance"]:.2f}%</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.write("**🧊 Cold Sectors:**")
            for sector in cold_sectors:
                st.markdown(f'<div class="sector-cold">{sector}: {sector_data[sector]["performance"]:.2f}%</div>', 
                           unsafe_allow_html=True)

def page_subscription_plans():
    """Subscription Plans and Pricing page."""
    display_header()
    st.title("💎 Subscription Plans")
    st.info("Choose the plan that fits your trading needs", icon="💎")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tier-card">
            <h3>Basic</h3>
            <h2>₹999<small>/month</small></h2>
            <ul>
                <li>Real-time NSE/BSE Data</li>
                <li>Basic Charting & Indicators</li>
                <li>5 Watchlists</li>
                <li>Standard Options Chain</li>
                <li>Email Support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.subscription_tier != "Basic":
            if st.button("Choose Basic", use_container_width=True):
                st.session_state.subscription_tier = "Basic"
                st.success("Switched to Basic Plan!")
                st.rerun()
        else:
            st.success("Current Plan")
    
    with col2:
        st.markdown("""
        <div class="tier-card featured">
            <h3>Professional <span class="premium-badge">POPULAR</span></h3>
            <h2>₹2,499<small>/month</small></h2>
            <ul>
                <li>All Basic Features</li>
                <li>Advanced Charting</li>
                <li>AI Trade Signals (10/day)</li>
                <li>Options Flow Analytics</li>
                <li>Basket Orders</li>
                <li>F&O Analytics</li>
                <li>Priority Support</li>
                <li>Mobile App Access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.subscription_tier != "Professional":
            if st.button("Choose Professional", use_container_width=True, type="primary"):
                st.session_state.subscription_tier = "Professional"
                st.success("Switched to Professional Plan!")
                st.rerun()
        else:
            st.success("Current Plan")
    
    with col3:
        st.markdown("""
        <div class="tier-card">
            <h3>Institutional</h3>
            <h2>₹7,999<small>/month</small></h2>
            <ul>
                <li>All Professional Features</li>
                <li>Unlimited AI Signals</li>
                <li>API Access</li>
                <li>White-label Solutions</li>
                <li>Custom Indicators</li>
                <li>Dedicated Account Manager</li>
                <li>Advanced Risk Analytics</li>
                <li>Multi-user Access</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.subscription_tier != "Institutional":
            if st.button("Choose Institutional", use_container_width=True):
                st.session_state.subscription_tier = "Institutional"
                st.success("Switched to Institutional Plan!")
                st.rerun()
        else:
            st.success("Current Plan")
    
    # Current usage statistics
    st.markdown("---")
    st.subheader("📈 Your Usage")
    
    tier = st.session_state.subscription_tier
    limits = SUBSCRIPTION_PLANS[tier]['limits']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        used_signals = st.session_state.get('ai_signals_used', 0)
        st.metric("AI Signals Used", f"{used_signals}/{limits['ai_signals']}")
    
    with col2:
        used_baskets = st.session_state.get('basket_orders_used', 0)
        st.metric("Basket Orders Used", f"{used_baskets}/{limits['basket_orders']}")
    
    with col3:
        status = "✅ Active" if limits['advanced_analytics'] else "❌ Inactive"
        st.metric("Advanced Analytics", status)
    
    with col4:
        status = "✅ Active" if limits['api_access'] else "❌ Inactive"
        st.metric("API Access", status)

# ================ 6. INTEGRATE ENHANCED PAGES ================

# [Include all the existing page functions from your original code here: 
# page_dashboard(), page_advanced_charting(), page_premarket_pulse(), 
# page_fo_analytics(), page_forecasting_ml(), page_hft_terminal(), 
# page_basket_orders(), page_fundamental_analysis(), page_ai_trade_assistant(), 
# page_settings(), login_page() - but make sure to update the navigation]

def main():
    """Main application controller."""
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    apply_custom_styling()
    
    # Check authentication
    if not st.session_state.get('authenticated'):
        login_page()
        return
    
    # Display overnight changes bar
    display_overnight_changes_bar()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        
        # User profile
        if st.session_state.profile:
            tier = st.session_state.get('subscription_tier', 'Basic')
            st.markdown(f"""
            <div style="background: var(--secondary-bg); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin: 0;">{st.session_state.profile.get('user_name', 'User')}</h4>
                <small style="color: var(--text-light);">{st.session_state.broker}</small><br>
                <span class="premium-badge">{tier} Plan</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced navigation options
        page_options = {
            "🏠 Dashboard": page_dashboard,
            "📈 Advanced Charting": page_advanced_charting,
            "🌍 Premarket Pulse": page_premarket_pulse,
            "📊 F&O Analytics": page_fo_analytics,
            "🤖 ML Forecasting": page_forecasting_ml,
            "⚡ HFT Terminal": page_hft_terminal,
            "🧺 Basket Orders": page_basket_orders,
            "🏢 Fundamental Analysis": page_fundamental_analysis,
            "💡 AI Trade Assistant": page_ai_trade_assistant,
            "🎯 AI Trade Signals": page_ai_trade_signals,
            "💰 Smart Money Flow": page_smart_money_flow,
            "📊 Sector Rotation": page_sector_rotation,
            "💎 Subscription Plans": page_subscription_plans,
            "⚙️ Settings": page_settings
        }
        
        selected_page = st.radio(
            "Navigate to",
            options=list(page_options.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Place Quick Order", use_container_width=True):
            quick_trade_dialog()
        
        if st.button("Generate AI Signal", use_container_width=True):
            st.session_state.active_page = "AI Trade Signals"
            st.rerun()
        
        # Market status
        status_info = get_market_status()
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: var(--widget-bg); border-radius: 8px;">
            <small>Market Status</small><br>
            <strong style="color: {status_info['color']};">{status_info['status']}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: var(--text-light); font-size: 0.8rem;">
            BlockVista Terminal Pro v2.0<br>
            <small>AI-Powered Trading Platform</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Display selected page
    page_function = page_options[selected_page]
    page_function()

# [Include all other existing functions from your original code that are not shown here]

if __name__ == "__main__":
    main()
