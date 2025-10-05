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

    js_theme = f"""
    <script>
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add('{st.session_state.get("theme", "Dark").lower()}-theme');
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
    if 'access_token' not in st.session_state: st.session_state.access_token = None
    
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
    
    # 2FA Dialogs
    if 'show_2fa_dialog' not in st.session_state: st.session_state.show_2fa_dialog = False
    if 'show_qr_dialog' not in st.session_state: st.session_state.show_qr_dialog = False

# ================ 2. ENHANCED HELPER FUNCTIONS ================

def check_subscription_feature(feature_name):
    """Check if current subscription allows a feature."""
    tier = st.session_state.get('subscription_tier', 'Basic')
    limits = SUBSCRIPTION_PLANS[tier]['limits']
    
    if feature_name == "ai_signals":
        if st.session_state.ai_signals_used >= limits['ai_signals']:
            st.error(f"AI Signals limit reached for {tier} plan. Upgrade to access more signals.")
            return False
        st.session_state.ai_signals_used += 1
        return True
    
    elif feature_name == "basket_orders":
        if st.session_state.basket_orders_used >= limits['basket_orders']:
            st.error(f"Basket Orders limit reached for {tier} plan. Upgrade for unlimited orders.")
            return False
        st.session_state.basket_orders_used += 1
        return True
    
    elif feature_name == "advanced_analytics":
        if not limits['advanced_analytics']:
            st.error(f"Advanced Analytics not available in {tier} plan. Upgrade to Professional or Institutional.")
            return False
        return True
    
    elif feature_name == "api_access":
        if not limits['api_access']:
            st.error(f"API Access not available in {tier} plan. Upgrade to Institutional.")
            return False
        return True
    
    return True

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    
    instrument_df = get_instrument_df()
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
            place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
            st.success("Order submitted!")
        else:
            st.warning("Please fill in all fields.")
    
    if col2.button("Cancel", use_container_width=True):
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

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
        tier = st.session_state.get('subscription_tier', 'Basic')
        st.markdown(f'<small style="color: var(--text-light);">{tier} Plan • Professional Trading</small>', unsafe_allow_html=True)
    
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
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            st.session_state.show_quick_trade = True
    
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
                if not np.isnan(price) and price > 0:
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
        try:
            df = pd.DataFrame(client.instruments())
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'])
            return df
        except Exception as e:
            st.error(f"Error fetching instruments: {e}")
            return pd.DataFrame()
    else:
        # Return demo instrument data
        demo_instruments = [
            {'tradingsymbol': 'RELIANCE', 'exchange': 'NSE', 'instrument_token': 738561},
            {'tradingsymbol': 'TCS', 'exchange': 'NSE', 'instrument_token': 2953217},
            {'tradingsymbol': 'INFY', 'exchange': 'NSE', 'instrument_token': 408065},
            {'tradingsymbol': 'HDFCBANK', 'exchange': 'NSE', 'instrument_token': 341249},
            {'tradingsymbol': 'NIFTY 50', 'exchange': 'NSE', 'instrument_token': 256265},
            {'tradingsymbol': 'BANKNIFTY', 'exchange': 'NFO', 'instrument_token': 260105},
        ]
        return pd.DataFrame(demo_instruments)

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
    
    # For demo mode, generate synthetic data
    if not client or st.session_state.broker == "Demo":
        return generate_demo_data(period or '1mo', interval)
    
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
        return generate_demo_data(period or '1mo', interval)

def generate_demo_data(period, interval):
    """Generate demo data for demo mode."""
    days = {'1d': 1, '5d': 5, '1mo': 30, '6mo': 180, '1y': 365, '5y': 1825}.get(period, 30)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = random.uniform(1000, 5000)
    
    data = []
    price = base_price
    for date in dates:
        change = random.uniform(-0.02, 0.02)
        price = price * (1 + change)
        high = price * (1 + random.uniform(0, 0.01))
        low = price * (1 - random.uniform(0, 0.01))
        open_price = price * (1 + random.uniform(-0.005, 0.005))
        volume = random.randint(100000, 1000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
    client = get_broker_client()
    
    # For demo mode, generate synthetic data
    if not client or st.session_state.broker == "Demo":
        watchlist = []
        for item in symbols_with_exchange:
            base_price = random.uniform(100, 5000)
            change = random.uniform(-50, 50)
            pct_change = (change / base_price) * 100
            
            watchlist.append({
                'Ticker': item['symbol'],
                'Exchange': item['exchange'],
                'Price': base_price + change,
                'Change': change,
                '% Change': pct_change
            })
        return pd.DataFrame(watchlist)
    
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
                        'Price': last_price,
                        'Change': change,
                        '% Change': pct_change
                    })
            
            return pd.DataFrame(watchlist)
            
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth (order book) for a given instrument."""
    client = get_broker_client()
    if not client or not instrument_token:
        return None

    try:
        depth = client.depth(instrument_token)
        return depth.get(str(instrument_token))
    except Exception as e:
        st.toast(f"Error fetching market depth: {e}", icon="⚠️")
        return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for a given underlying."""
    client = get_broker_client()
    
    # For demo mode, generate synthetic options chain
    if not client or st.session_state.broker == "Demo":
        return generate_demo_options_chain(underlying)
    
    if st.session_state.broker == "Zerodha" and instrument_df.empty:
        return pd.DataFrame(), None, 0.0, []

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
                on='strike', suffixes=('_CE', '_PE')
            ).rename(columns={
                'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP',
                'strike': 'STRIKE',
                'oi_CE': 'CALL OI', 'oi_PE': 'PUT OI',
                'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'
            }).fillna(0)

            return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries

        except Exception as e:
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        return generate_demo_options_chain(underlying)

def generate_demo_options_chain(underlying):
    """Generate demo options chain data."""
    base_price = 18000 if underlying == "NIFTY" else 40000 if underlying == "BANKNIFTY" else 18000
    expiry = datetime.now().date() + timedelta(days=7)
    
    strikes = list(range(int(base_price * 0.9), int(base_price * 1.1), 100))
    
    chain_data = []
    for strike in strikes:
        call_ltp = max(10, (base_price - strike) * 0.1 + random.uniform(-5, 5))
        put_ltp = max(10, (strike - base_price) * 0.1 + random.uniform(-5, 5))
        call_oi = random.randint(1000, 100000)
        put_oi = random.randint(1000, 100000)
        
        chain_data.append({
            'CALL': f"{underlying}{expiry.strftime('%d%b%y').upper()}{strike}CE",
            'CALL LTP': call_ltp,
            'CALL OI': call_oi,
            'STRIKE': strike,
            'PUT LTP': put_ltp,
            'PUT OI': put_oi,
            'PUT': f"{underlying}{expiry.strftime('%d%b%y').upper()}{strike}PE"
        })
    
    return pd.DataFrame(chain_data), expiry, base_price, [expiry]

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio positions and holdings from the broker."""
    client = get_broker_client()
    
    # For demo mode, generate synthetic portfolio
    if not client or st.session_state.broker == "Demo":
        demo_holdings = [
            {'tradingsymbol': 'RELIANCE', 'quantity': 10, 'average_price': 2450.50, 'last_price': 2520.75, 'pnl': 702.50},
            {'tradingsymbol': 'TCS', 'quantity': 5, 'average_price': 3250.00, 'last_price': 3315.25, 'pnl': 326.25},
            {'tradingsymbol': 'INFY', 'quantity': 8, 'average_price': 1650.75, 'last_price': 1620.50, 'pnl': -242.00}
        ]
        positions_df = pd.DataFrame(demo_holdings)
        holdings_df = pd.DataFrame(demo_holdings)
        total_pnl = positions_df['pnl'].sum()
        total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum()
        return positions_df, holdings_df, total_pnl, total_investment
    
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
            
            st.toast(f"Order placed successfully! ID: {order_id}", icon="✅")
            st.session_state.order_history.insert(0, {
                "id": order_id, "symbol": symbol, "qty": quantity, 
                "type": transaction_type, "status": "Success"
            })
            
        except Exception as e:
            st.toast(f"Order failed: {e}", icon="❌")
            st.session_state.order_history.insert(0, {
                "id": "N/A", "symbol": symbol, "qty": quantity, 
                "type": transaction_type, "status": f"Failed: {e}"
            })
    else:
        # Demo mode order simulation
        order_id = f"DEMO_{random.randint(10000, 99999)}"
        st.toast(f"Demo order placed! ID: {order_id}", icon="✅")
        st.session_state.order_history.insert(0, {
            "id": order_id, "symbol": symbol, "qty": quantity, 
            "type": transaction_type, "status": "Success (Demo)"
        })

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
            for entry in feed.entries[:10]:  # Limit to 10 articles per source
                published_date_tuple = entry.published_parsed if hasattr(entry, 'published_parsed') else entry.updated_parsed
                published_date = datetime.fromtimestamp(mktime_tz(published_date_tuple)) if published_date_tuple else datetime.now()
                
                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    all_news.append({
                        "source": source,
                        "title": entry.title,
                        "link": entry.link,
                        "date": published_date.date(),
                        "sentiment": analyzer.polarity_scores(entry.title)['compound']
                    })
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
        
        # Backtesting: Create predictions for historical data
        fitted_values = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values}).dropna()
        
        # Forecasting: Predict future values
        forecast_result = model.get_forecast(steps=forecast_steps)
        forecast_adjusted = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
        
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
    """Loads and combines historical data from a static CSV with live data from the broker."""
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
                live_df.index = live_df.index.tz_convert(None)  # Make timezone-naive
                live_df.columns = [col.lower() for col in live_df.columns]
                
    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max")
            if not live_df.empty:
                live_df.index = live_df.index.tz_localize(None)  # Make timezone-naive
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

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price and Greeks."""
    if sigma <= 0 or T <= 0:
        return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,
        "theta": theta / 365,
        "rho": rho / 100
    }

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    if T <= 0 or market_price <= 0:
        return np.nan
    
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError):
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty:
        return {}
    
    latest = df.iloc[-1].copy()
    latest.index = latest.index.str.lower()
    interpretation = {}
    
    # More robust column finding
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
    
    with st.spinner(f"AI analyzing {symbol}..."):
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
    st.title("AI Trade Signals")
    st.info("Get AI-powered trading signals with multi-factor analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generate Signal")
        symbol = st.text_input("Stock Symbol", "RELIANCE", key="ai_signal_symbol").upper()
        timeframe = st.selectbox("Timeframe", ["15min", "1h", "4h", "1d"], index=1)
        
        if st.button("Generate AI Signal", type="primary", use_container_width=True):
            signal_data = generate_ai_trade_signal(symbol, timeframe)
            
            if signal_data:
                st.success("AI Signal Generated Successfully!")
                
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
                    <h3>{signal_data['symbol']} - {signal_data['signal']}</h3>
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
    st.title("Smart Money Flow")
    st.info("Track institutional and smart money activity")
    
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
        st.subheader("Smart Money Insights")
        
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
    st.title("Sector Rotation")
    st.info("Identify sector momentum and rotation opportunities")
    
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
        st.subheader("Sector Insights")
        
        hot_sectors = [s for s in sectors if sector_data[s]['momentum'] == 'HOT']
        cold_sectors = [s for s in sectors if sector_data[s]['momentum'] == 'COLD']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Hot Sectors:")
            for sector in hot_sectors:
                st.markdown(f'<div class="sector-hot">{sector}: {sector_data[sector]["performance"]:.2f}%</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.write("Cold Sectors:")
            for sector in cold_sectors:
                st.markdown(f'<div class="sector-cold">{sector}: {sector_data[sector]["performance"]:.2f}%</div>', 
                           unsafe_allow_html=True)

def page_subscription_plans():
    """Subscription Plans and Pricing page."""
    display_header()
    st.title("Subscription Plans")
    st.info("Choose the plan that fits your trading needs")
    
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
    st.subheader("Your Usage")
    
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
        status = "Active" if limits['advanced_analytics'] else "Inactive"
        st.metric("Advanced Analytics", status)
    
    with col4:
        status = "Active" if limits['api_access'] else "Inactive"
        st.metric("API Access", status)

# ================ 6. EXISTING PAGES (UPDATED) ================

def page_dashboard():
    """A completely redesigned 'Trader UI' Dashboard."""
    display_header()
    instrument_df = get_instrument_df()
    
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return

    # Quick Trade Section
    if st.session_state.get('show_quick_trade'):
        st.subheader("Quick Trade")
        quick_trade_dialog()
        if st.button("Close"):
            st.session_state.show_quick_trade = False
            st.rerun()
        return

    # Fetch NIFTY, SENSEX, and VIX data for BMP calculation
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
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50']
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX']
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX']
            
            if not nifty_row.empty and not sensex_row.empty and not vix_row.empty:
                nifty_change = nifty_row.iloc[0]['% Change']
                sensex_change = sensex_row.iloc[0]['% Change']
                vix_value = vix_row.iloc[0]['Price']
                
                # Fetch historical data for normalization
                nifty_hist = get_historical_data(get_instrument_token('NIFTY 50', instrument_df, 'NSE'), 'day', period='1y')
                sensex_hist = get_historical_data(get_instrument_token('SENSEX', instrument_df, 'BSE'), 'day', period='1y')
                vix_hist = get_historical_data(get_instrument_token('INDIA VIX', instrument_df, 'NSE'), 'day', period='1y')
                
                if not nifty_hist.empty and not sensex_hist.empty and not vix_hist.empty:
                    lookback_data = pd.DataFrame({
                        'nifty_change': nifty_hist['close'].pct_change() * 100,
                        'sensex_change': sensex_hist['close'].pct_change() * 100,
                        'vix_value': vix_hist['close']
                    }).dropna()
                    
                    bmp_score, bmp_label, bmp_color = get_bmp_score_and_label(
                        nifty_change, sensex_change, vix_value, lookback_data
                    )
                    
                    st.markdown(f'<div class="metric-card" style="border-color:{bmp_color};"><h3>{bmp_score:.2f}</h3><p style="color:{bmp_color}; font-weight:bold;">{bmp_label}</p><small>Proprietary score from NIFTY, SENSEX, and India VIX.</small></div>', 
                               unsafe_allow_html=True)
                    
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
        else:
            st.info("BMP data is loading...")

    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        st.plotly_chart(create_nifty_heatmap(instrument_df), use_container_width=True)

    st.markdown("---")

    # Middle Row: Main Content Area
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

def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    
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

        # Comparison symbols
        with st.expander("Comparison Symbols (Max 4)"):
            comp_cols = st.columns(2)
            comp1 = comp_cols[0].text_input("Compare 1", key=f"comp1_{i}").upper()
            comp2 = comp_cols[1].text_input("Compare 2", key=f"comp2_{i}").upper()
            comp3 = comp_cols[0].text_input("Compare 3", key=f"comp3_{i}").upper()
            comp4 = comp_cols[1].text_input("Compare 4", key=f"comp4_{i}").upper()
            
            comparison_symbols = [sym for sym in [comp1, comp2, comp3, comp4] if sym]

        # Get main symbol data
        token = get_instrument_token(ticker, instrument_df)
        data = get_historical_data(token, interval, period=period)
        
        # Get comparison data
        comparison_data = {}
        for comp_symbol in comparison_symbols:
            comp_token = get_instrument_token(comp_symbol, instrument_df)
            if comp_token:
                comp_data = get_historical_data(comp_token, interval, period=period)
                if not comp_data.empty:
                    comparison_data[comp_symbol] = comp_data

        if data.empty and not comparison_data:
            st.warning(f"No data to display for {ticker} with selected parameters.")
        else:
            st.plotly_chart(
                create_chart(data, ticker, chart_type, comparison_data=comparison_data if comparison_data else None),
                use_container_width=True,
                key=f"chart_{i}"
            )

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
                if not np.isnan(price) and price > 0:
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
                    if not np.isnan(price) and price > 0:
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

            # Display options chain with ITM/OTM styling
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
            r = 0.07  # Assume a risk-free rate of 7%
            
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
    """A page for advanced ML forecasting with multiple models."""
    display_header()
    st.title("Advanced ML Forecasting Hub")
    st.info("""
    Train multiple machine learning models to forecast future prices.
    
    **Models Available:**
    - **Seasonal ARIMA**: Traditional time series model
    - **LSTM**: Deep learning model for sequence prediction
    - **XGBoost**: Gradient boosting for tabular data
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Seasonal ARIMA", "LSTM", "XGBoost"],
            help="ARIMA: Best for seasonal patterns. LSTM: Best for complex sequences. XGBoost: Best with features."
        )
        
        # Extended forecast durations
        forecast_durations = {
            "1 Week": 7,
            "2 Weeks": 14,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180
        }
        
        duration_key = st.radio("Forecast Duration", list(forecast_durations.keys()), horizontal=True, key="ml_duration")
        forecast_steps = forecast_durations[duration_key]
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            if model_type == "Seasonal ARIMA":
                st.slider("Seasonal Period", 5, 30, 7, key="seasonal_period", help="Seasonal cycle length")
            elif model_type == "LSTM":
                st.slider("Sequence Length", 30, 90, 60, key="lstm_seq_len", help="Look-back period for LSTM")
            elif model_type == "XGBoost":
                st.slider("Number of Estimators", 100, 2000, 1000, key="xgb_n_estimators", help="Number of boosting rounds")
        
        if st.button("Train Model & Forecast", use_container_width=True, type="primary"):
            with st.spinner(f"Training {model_type} model for {instrument_name}..."):
                data = load_and_combine_data(instrument_name)
                
                if data.empty or len(data) < 100:
                    st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
                else:
                    # Train selected model
                    if model_type == "Seasonal ARIMA":
                        forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                        model_message = "Seasonal ARIMA model trained successfully"
                    elif model_type == "LSTM":
                        forecast_df, backtest_df, conf_int_df, model_message = train_lstm_model(data, forecast_steps)
                    elif model_type == "XGBoost":
                        forecast_df, backtest_df, conf_int_df, model_message = train_xgboost_model(data, forecast_steps)
                    
                    if forecast_df is not None:
                        # Calculate model metrics
                        metrics = calculate_model_metrics(
                            backtest_df['Actual'].values if backtest_df is not None else [],
                            backtest_df['Predicted'].values if backtest_df is not None else []
                        )
                        
                        st.session_state.update({
                            'ml_forecast_df': forecast_df,
                            'ml_backtest_df': backtest_df,
                            'ml_conf_int_df': conf_int_df,
                            'ml_instrument_name': instrument_name,
                            'ml_historical_data': data,
                            'ml_duration_key': duration_key,
                            'ml_model_type': model_type,
                            'ml_metrics': metrics,
                            'ml_message': model_message
                        })
                        
                        # Store previous models for comparison
                        if 'ml_previous_models' not in st.session_state:
                            st.session_state.ml_previous_models = []
                        
                        st.session_state.ml_previous_models.append({
                            'type': model_type,
                            'metrics': metrics,
                            'timestamp': datetime.now()
                        })
                        
                        # Keep only last 5 models
                        if len(st.session_state.ml_previous_models) > 5:
                            st.session_state.ml_previous_models = st.session_state.ml_previous_models[-5:]
                        
                        st.success(f"{model_type} model trained successfully!")
                    else:
                        st.error(f"Model training failed: {model_message}")
        
        # Model comparison section
        if st.session_state.get('ml_previous_models'):
            st.subheader("Model Comparison")
            
            comparison_data = []
            for model_info in st.session_state.ml_previous_models:
                comparison_data.append({
                    'Model': model_info['type'],
                    'Accuracy': f"{100 - model_info['metrics'].get('MAPE', 0):.1f}%",
                    'MAE': f"₹{model_info['metrics'].get('MAE', 0):.2f}",
                    'Direction Accuracy': f"{model_info['metrics'].get('Direction_Accuracy', 0):.1f}%",
                    'R²': f"{model_info['metrics'].get('R2', 0):.3f}"
                })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col2:
        if 'ml_instrument_name' in st.session_state:
            instrument_name = st.session_state.ml_instrument_name
            model_type = st.session_state.get('ml_model_type', 'Seasonal ARIMA')
            metrics = st.session_state.get('ml_metrics', {})
            
            st.subheader(f"Forecast Results for {instrument_name} ({model_type})")
            
            forecast_df = st.session_state.get('ml_forecast_df')
            backtest_df = st.session_state.get('ml_backtest_df')
            conf_int_df = st.session_state.get('ml_conf_int_df')
            data = st.session_state.get('ml_historical_data')
            duration_key = st.session_state.get('ml_duration_key')
            
            if forecast_df is not None and backtest_df is not None and data is not None:
                # Display model metrics
                if metrics:
                    st.subheader("Model Performance")
                    metric_cols = st.columns(4)
                    
                    if 'MAPE' in metrics:
                        metric_cols[0].metric("Accuracy", f"{100 - metrics['MAPE']:.1f}%")
                    if 'MAE' in metrics:
                        metric_cols[1].metric("MAE", f"₹{metrics['MAE']:.2f}")
                    if 'RMSE' in metrics:
                        metric_cols[2].metric("RMSE", f"₹{metrics['RMSE']:.2f}")
                    if 'Direction_Accuracy' in metrics:
                        metric_cols[3].metric("Direction Accuracy", f"{metrics['Direction_Accuracy']:.1f}%")
                
                # Plot the results
                fig = create_chart(data.tail(252), instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                
                if backtest_df is not None:
                    fig.add_trace(go.Scatter(
                        x=backtest_df.index,
                        y=backtest_df['Predicted'],
                        mode='lines',
                        name='Model Fit',
                        line=dict(color='orange', dash='dot')
                    ))
                
                fig.update_layout(title=f"{instrument_name} {duration_key} Forecast ({model_type})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Price prediction summary
                st.subheader("Price Prediction Summary")
                
                current_price = data['close'].iloc[-1]
                predicted_prices = forecast_df['Predicted']
                
                pred_cols = st.columns(4)
                pred_cols[0].metric("Current Price", f"₹{current_price:.2f}")
                pred_cols[1].metric("Predicted High", f"₹{predicted_prices.max():.2f}")
                pred_cols[2].metric("Predicted Low", f"₹{predicted_prices.min():.2f}")
                pred_cols[3].metric("Predicted Avg", f"₹{predicted_prices.mean():.2f}")
                
                # Price change projection
                price_change = ((predicted_prices.mean() - current_price) / current_price) * 100
                st.metric(
                    "Projected Price Change",
                    f"₹{predicted_prices.mean() - current_price:+.2f}",
                    delta=f"{price_change:+.1f}%"
                )
                
                # Detailed forecast analysis
                with st.expander("Detailed Forecast Analysis"):
                    st.subheader("Forecast Statistics")
                    
                    # Volatility analysis
                    forecast_volatility = predicted_prices.pct_change().std() * np.sqrt(252) * 100  # Annualized
                    historical_volatility = data['close'].pct_change().std() * np.sqrt(252) * 100
                    
                    vol_cols = st.columns(2)
                    vol_cols[0].metric("Historical Volatility", f"{historical_volatility:.1f}%")
                    vol_cols[1].metric("Forecast Volatility", f"{forecast_volatility:.1f}%")
                    
                    # Price targets
                    st.subheader("Key Price Levels")
                    levels_cols = st.columns(3)
                    levels_cols[0].metric("Support Level", f"₹{predicted_prices.quantile(0.25):.2f}")
                    levels_cols[1].metric("Median Forecast", f"₹{predicted_prices.median():.2f}")
                    levels_cols[2].metric("Resistance Level", f"₹{predicted_prices.quantile(0.75):.2f}")
                    
                    # Forecast data table
                    st.subheader("Forecast Data")
                    display_df = forecast_df.copy()
                    display_df['Date'] = display_df.index.strftime('%Y-%m-%d')
                    display_df['Price'] = display_df['Predicted']
                    display_df = display_df[['Date', 'Price']]
                    
                    st.dataframe(display_df.style.format({'Price': '₹{:.2f}'}), use_container_width=True)
        else:
            st.info("Train a model to see the forecast results.")
    
    # Model insights and warnings
    st.markdown("---")
    st.subheader("Model Insights")
    
    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.info("""
        **Model Strengths:**
        - **ARIMA**: Best for seasonal patterns
        - **LSTM**: Captures complex temporal dependencies
        - **XGBoost**: Handles multiple feature types
        """)
    
    with insight_cols[1]:
        st.warning("""
        **Important Notes:**
        - Models are for educational purposes only
        - Past performance ≠ future results
        - Always use multiple analysis methods
        - Consider market fundamentals
        - 6-month forecasts have higher uncertainty
        """)

def page_hft_terminal():
    """High-Frequency Trading terminal with real-time market data."""
    display_header()
    st.title("HFT Terminal")
    st.info("Real-time market data and order execution for high-frequency trading strategies.")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the HFT Terminal.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Market Data")
        symbol = st.text_input("Symbol", "RELIANCE", key="hft_symbol").upper()
        exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"], key="hft_exchange")
        
        token = get_instrument_token(symbol, instrument_df, exchange)
        if not token:
            st.error(f"Could not find {symbol} on {exchange}")
            return
        
        # Auto-refresh every 2 seconds
        st_autorefresh(interval=2000, limit=100, key="hft_refresh")
        
        # Get live quote
        client = get_broker_client()
        if client:
            try:
                quote = client.quote(f"{exchange}:{symbol}")
                if quote:
                    data = quote[f"{exchange}:{symbol}"]
                    last_price = data['last_price']
                    ohlc = data['ohlc']
                    depth = data['depth']
                    
                    # Calculate tick change
                    prev_price = st.session_state.hft_last_price
                    tick_class = ""
                    if last_price > prev_price:
                        tick_class = "tick-up"
                    elif last_price < prev_price:
                        tick_class = "tick-down"
                    
                    st.session_state.hft_last_price = last_price
                    
                    # Display price with tick animation
                    st.markdown(f"""
                    <div class="{tick_class}" style="font-size: 2.5rem; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; background: var(--widget-bg);">
                        ₹{last_price:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # OHLC
                    ohlc_cols = st.columns(4)
                    ohlc_cols[0].metric("Open", f"₹{ohlc['open']:,.2f}")
                    ohlc_cols[1].metric("High", f"₹{ohlc['high']:,.2f}")
                    ohlc_cols[2].metric("Low", f"₹{ohlc['low']:,.2f}")
                    ohlc_cols[3].metric("Close", f"₹{ohlc['close']:,.2f}")
                    
                    # Volume and OI
                    vol_cols = st.columns(2)
                    vol_cols[0].metric("Volume", f"{data['volume']:,}")
                    if 'oi' in data:
                        vol_cols[1].metric("Open Interest", f"{data['oi']:,}")
                        
            except Exception as e:
                st.error(f"Error fetching live data: {e}")
    
    with col2:
        st.subheader("Quick Order")
        with st.form(key="hft_order_form"):
            hft_quantity = st.number_input("Quantity", min_value=1, value=1, key="hft_qty")
            hft_price = st.number_input("Price", min_value=0.01, value=st.session_state.hft_last_price, key="hft_price")
            
            hft_cols = st.columns(2)
            if hft_cols[0].form_submit_button("BUY", use_container_width=True, type="primary"):
                place_order(instrument_df, symbol, hft_quantity, 'LIMIT', 'BUY', 'MIS', hft_price)
            
            if hft_cols[1].form_submit_button("SELL", use_container_width=True, type="secondary"):
                place_order(instrument_df, symbol, hft_quantity, 'LIMIT', 'SELL', 'MIS', hft_price)
        
        # Market Depth
        st.subheader("Market Depth")
        if token:
            depth = get_market_depth(token)
            if depth:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Bids (Buyers)**")
                    bids = depth.get('buy', [])[:5]  # Top 5 bids
                    for i, bid in enumerate(bids):
                        st.markdown(f"""
                        <div class="hft-depth-bid">
                            {i+1}. ₹{bid['price']:,.2f} × {bid['quantity']:,}
                            {f'({bid["orders"]} orders)' if 'orders' in bid else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Asks (Sellers)**")
                    asks = depth.get('sell', [])[:5]  # Top 5 asks
                    for i, ask in enumerate(asks):
                        st.markdown(f"""
                        <div class="hft-depth-ask">
                            {i+1}. ₹{ask['price']:,.2f} × {ask['quantity']:,}
                            {f'({ask["orders"]} orders)' if 'orders' in ask else ''}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Recent Ticks
        st.subheader("Recent Ticks")
        if token and client:
            try:
                # Get recent ticks (last 10)
                ticks = client.historical_data(token, datetime.now() - timedelta(minutes=5), datetime.now(), 'minute')
                if ticks:
                    ticks_df = pd.DataFrame(ticks[-10:])  # Last 10 ticks
                    if not ticks_df.empty:
                        st.dataframe(ticks_df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(10),
                                   use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not fetch tick data: {e}")

def page_basket_orders():
    """Page for creating and executing basket orders."""
    display_header()
    st.title("Basket Orders")
    st.info("Execute multiple orders simultaneously for portfolio rebalancing or strategy implementation.")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use Basket Orders.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create Basket Order")
        with st.form(key="basket_order_form"):
            symbol = st.text_input("Symbol", placeholder="RELIANCE", key="basket_symbol").upper()
            exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX"], key="basket_exchange")
            quantity = st.number_input("Quantity", min_value=1, value=1, key="basket_qty")
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="basket_order_type")
            price = st.number_input("Price", min_value=0.01, value=0.0, key="basket_price") if order_type == "LIMIT" else 0.0
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="basket_trans_type")
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="basket_product")
            
            if st.form_submit_button("Add to Basket", use_container_width=True):
                if symbol:
                    basket_item = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'quantity': quantity,
                        'order_type': order_type,
                        'transaction_type': transaction_type,
                        'product': product,
                        'price': price if price > 0 else None
                    }
                    st.session_state.basket.append(basket_item)
                    st.success(f"Added {symbol} to basket!")
                    st.rerun()
                else:
                    st.warning("Please enter a symbol.")
    
    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty. Add orders to get started.")
        else:
            total_orders = len(st.session_state.basket)
            st.metric("Total Orders", total_orders)
            
            for i, item in enumerate(st.session_state.basket):
                with st.container(border=True):
                    cols = st.columns([3, 2, 1])
                    cols[0].markdown(f"**{item['symbol']}** ({item['exchange']})")
                    cols[1].markdown(f"{item['transaction_type']} {item['quantity']} @ {item['order_type']}")
                    
                    if cols[2].button("❌", key=f"del_{i}", use_container_width=True):
                        st.session_state.basket.pop(i)
                        st.rerun()
            
            # Basket actions
            st.markdown("---")
            action_cols = st.columns(2)
            
            if action_cols[0].button("Execute Basket", use_container_width=True, type="primary"):
                execute_basket_order(st.session_state.basket, instrument_df)
            
            if action_cols[1].button("Clear Basket", use_container_width=True, type="secondary"):
                st.session_state.basket = []
                st.rerun()
        
        # Predefined Baskets
        st.subheader("Predefined Baskets")
        predefined_cols = st.columns(3)
        
        with predefined_cols[0]:
            if st.button("NIFTY 50 Top 5", use_container_width=True):
                nifty_top = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS']
                for symbol in nifty_top:
                    st.session_state.basket.append({
                        'symbol': symbol, 'exchange': 'NSE', 'quantity': 1,
                        'order_type': 'MARKET', 'transaction_type': 'BUY', 'product': 'MIS'
                    })
                st.success("Added NIFTY 50 Top 5 to basket!")
                st.rerun()
        
        with predefined_cols[1]:
            if st.button("Banking Sector", use_container_width=True):
                banking_stocks = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK']
                for symbol in banking_stocks:
                    st.session_state.basket.append({
                        'symbol': symbol, 'exchange': 'NSE', 'quantity': 1,
                        'order_type': 'MARKET', 'transaction_type': 'BUY', 'product': 'MIS'
                    })
                st.success("Added Banking Sector to basket!")
                st.rerun()
        
        with predefined_cols[2]:
            if st.button("IT Sector", use_container_width=True):
                it_stocks = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']
                for symbol in it_stocks:
                    st.session_state.basket.append({
                        'symbol': symbol, 'exchange': 'NSE', 'quantity': 1,
                        'order_type': 'MARKET', 'transaction_type': 'BUY', 'product': 'MIS'
                    })
                st.success("Added IT Sector to basket!")
                st.rerun()

def page_fundamental_analysis():
    """Comprehensive fundamental analysis with financial metrics and valuation."""
    display_header()
    st.title("Fundamental Analysis")
    st.info("Deep dive into company fundamentals, financial health, and intrinsic valuation.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Company Analysis")
        symbol = st.text_input("Stock Symbol", "RELIANCE", key="fundamental_symbol").upper()
        
        if st.button("Analyze Company", use_container_width=True):
            with st.spinner(f"Fetching fundamental data for {symbol}..."):
                fundamental_data = get_fundamental_data(symbol)
                if fundamental_data:
                    st.session_state.fundamental_data = fundamental_data
                else:
                    st.error(f"Could not fetch fundamental data for {symbol}")
        
        # Comparison symbols
        st.subheader("Peer Comparison")
        comp_symbol = st.text_input("Compare with", key="comp_symbol").upper()
        
        if st.button("Add to Comparison", use_container_width=True):
            if comp_symbol and comp_symbol not in st.session_state.comparison_symbols:
                st.session_state.comparison_symbols.append(comp_symbol)
                st.rerun()
        
        # Display comparison list
        if st.session_state.comparison_symbols:
            st.write("**Comparison List:**")
            for comp in st.session_state.comparison_symbols:
                cols = st.columns([3, 1])
                cols[0].write(comp)
                if cols[1].button("Remove", key=f"rm_{comp}"):
                    st.session_state.comparison_symbols.remove(comp)
                    st.rerun()
    
    with col2:
        if 'fundamental_data' in st.session_state and st.session_state.fundamental_data:
            data = st.session_state.fundamental_data
            
            # Company Overview
            st.subheader(f"{data.get('company_name', symbol)} ({symbol})")
            overview_cols = st.columns(3)
            overview_cols[0].metric("Current Price", f"₹{data.get('current_price', 0):.2f}")
            overview_cols[1].metric("Market Cap", f"₹{data.get('market_cap', 0):,.0f} Cr")
            overview_cols[2].metric("Sector", data.get('sector', 'N/A'))
            
            # Key Metrics
            st.subheader("Valuation Metrics")
            metric_cols = st.columns(4)
            
            pe_ratio = data.get('pe_ratio', 0)
            metric_cols[0].metric("P/E Ratio", f"{pe_ratio:.1f}",
                                delta="Undervalued" if pe_ratio < 15 else "Overvalued" if pe_ratio > 25 else "Fair")
            
            pb_ratio = data.get('pb_ratio', 0)
            metric_cols[1].metric("P/B Ratio", f"{pb_ratio:.1f}",
                                delta="Undervalued" if pb_ratio < 1 else "Overvalued" if pb_ratio > 3 else "Fair")
            
            metric_cols[2].metric("Dividend Yield", f"{data.get('dividend_yield', 0):.2f}%")
            metric_cols[3].metric("Beta", f"{data.get('beta', 0):.2f}")
            
            # Profitability
            st.subheader("Profitability")
            profit_cols = st.columns(3)
            
            roe = data.get('roe', 0)
            profit_cols[0].metric("Return on Equity", f"{roe:.1f}%",
                                delta="Good" if roe > 15 else "Poor" if roe < 8 else "Average")
            
            profit_margin = data.get('profit_margin', 0)
            profit_cols[1].metric("Profit Margin", f"{profit_margin:.1f}%",
                                delta="Good" if profit_margin > 15 else "Poor" if profit_margin < 5 else "Average")
            
            operating_margin = data.get('operating_margin', 0)
            profit_cols[2].metric("Operating Margin", f"{operating_margin:.1f}%",
                                delta="Good" if operating_margin > 20 else "Poor" if operating_margin < 10 else "Average")
            
            # Growth
            st.subheader("Growth")
            growth_cols = st.columns(2)
            
            revenue_growth = data.get('revenue_growth', 0)
            growth_cols[0].metric("Revenue Growth", f"{revenue_growth:.1f}%",
                                delta="Strong" if revenue_growth > 15 else "Weak" if revenue_growth < 5 else "Moderate")
            
            profit_growth = data.get('profit_growth', 0)
            growth_cols[1].metric("Profit Growth", f"{profit_growth:.1f}%",
                                delta="Strong" if profit_growth > 15 else "Weak" if profit_growth < 5 else "Moderate")
            
            # Financial Health
            st.subheader("Financial Health")
            health_cols = st.columns(3)
            
            debt_to_equity = data.get('debt_to_equity', 0)
            health_cols[0].metric("Debt to Equity", f"{debt_to_equity:.1f}",
                                delta="High" if debt_to_equity > 2 else "Low" if debt_to_equity < 0.5 else "Moderate")
            
            current_ratio = data.get('current_ratio', 0)
            health_cols[1].metric("Current Ratio", f"{current_ratio:.1f}",
                                delta="Good" if current_ratio > 1.5 else "Poor" if current_ratio < 1 else "Adequate")
            
            free_cash_flow = data.get('free_cash_flow', 0)
            health_cols[2].metric("Free Cash Flow", f"₹{free_cash_flow:,.0f} Cr")
            
            # Intrinsic Value Calculation
            st.subheader("Intrinsic Value (DCF)")
            intrinsic_value, margin_of_safety = calculate_intrinsic_value(data)
            
            if intrinsic_value:
                value_cols = st.columns(2)
                value_cols[0].metric("Intrinsic Value", f"₹{intrinsic_value:,.2f}")
                value_cols[1].metric("Margin of Safety", f"{margin_of_safety:.1f}%",
                                   delta="Undervalued" if margin_of_safety > 20 else "Overvalued" if margin_of_safety < -20 else "Fairly Valued")
            
            # 52-week range
            st.subheader("52-Week Range")
            week_high = data.get('52_week_high', 0)
            week_low = data.get('52_week_low', 0)
            current_price = data.get('current_price', 0)
            
            if week_high > 0 and week_low > 0:
                range_position = (current_price - week_low) / (week_high - week_low) * 100
                st.metric("Range Position", f"{range_position:.1f}%")
                
                # Visual range indicator
                st.progress(range_position / 100)
                
                range_cols = st.columns(3)
                range_cols[0].write(f"Low: ₹{week_low:.2f}")
                range_cols[1].write(f"Current: ₹{current_price:.2f}")
                range_cols[2].write(f"High: ₹{week_high:.2f}")
            
            # Peer Comparison Table
            if st.session_state.comparison_symbols:
                st.markdown("---")
                st.subheader("Peer Comparison")
                
                comparison_data = []
                all_symbols = [symbol] + st.session_state.comparison_symbols
                
                for comp_symbol in all_symbols:
                    comp_data = get_fundamental_data(comp_symbol)
                    if comp_data:
                        comparison_data.append({
                            'Symbol': comp_symbol,
                            'Price': comp_data.get('current_price', 0),
                            'P/E': comp_data.get('pe_ratio', 0),
                            'P/B': comp_data.get('pb_ratio', 0),
                            'ROE %': comp_data.get('roe', 0),
                            'Debt/Equity': comp_data.get('debt_to_equity', 0),
                            'Market Cap (Cr)': comp_data.get('market_cap', 0) / 1e7
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df.style.format({
                        'Price': '₹{:.2f}',
                        'P/E': '{:.1f}',
                        'P/B': '{:.1f}',
                        'ROE %': '{:.1f}%',
                        'Debt/Equity': '{:.1f}',
                        'Market Cap (Cr)': '₹{:.0f}'
                    }), use_container_width=True, hide_index=True)

def page_ai_trade_assistant():
    """AI-powered trading assistant with strategy suggestions."""
    display_header()
    st.title("AI Trade Assistant")
    st.info("Get AI-powered trading insights, strategy suggestions, and market analysis.")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the AI Trade Assistant.")
       
