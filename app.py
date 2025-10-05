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
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
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

    .signal-buy { border-left: 5px solid var(--green); }
    .signal-sell { border-left: 5px solid var(--red); }
    .signal-neutral { border-left: 5px solid var(--orange); }

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
        position: sticky; top: 0; width: 100%;
        background-color: var(--secondary-bg);
        color: var(--text-color);
        padding: 8px 12px; z-index: 999;
        display: flex; justify-content: center;
        align-items: center; font-size: 0.9rem;
        border-bottom: 1px solid var(--border-color);
        margin-left: -20px; margin-right: -20px;
        width: calc(100% + 40px);
    }

    .notification-bar span { margin: 0 15px; white-space: nowrap; }
    .hft-depth-bid { background: linear-gradient(to left, rgba(0, 128, 0, 0.3), rgba(0, 128, 0, 0.05)); padding: 4px 8px; margin: 2px 0; border-radius: 4px; }
    .hft-depth-ask { background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05)); padding: 4px 8px; margin: 2px 0; border-radius: 4px; }
    .tick-up { color: var(--green); animation: flash-green 0.5s; }
    .tick-down { color: var(--red); animation: flash-red 0.5s; }

    @keyframes flash-green { 0% { background-color: rgba(40, 167, 69, 0.5); } 100% { background-color: transparent; } }
    @keyframes flash-red { 0% { background-color: rgba(218, 54, 51, 0.5); } 100% { background-color: transparent; } }

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
        "price": 0,
        "features": ["Real-time NSE/BSE Data (Demo)", "Basic Charting & Indicators", "3 Watchlists", "Standard Options Chain", "Email Support"],
        "limits": {"ai_signals": 0, "basket_orders": 0, "advanced_analytics": False, "api_access": False}
    },
    "Professional": {
        "price": 2499,
        "features": ["All Basic Features", "Advanced Charting", "AI Trade Signals (10/day)", "Options Flow Analytics", "Basket Orders", "F&O Analytics", "Priority Support"],
        "limits": {"ai_signals": 10, "basket_orders": 5, "advanced_analytics": True, "api_access": False}
    },
    "Institutional": {
        "price": 7999,
        "features": ["All Professional Features", "Unlimited AI Signals", "API Access", "Custom Indicators", "Dedicated Account Manager", "Advanced Risk Analytics"],
        "limits": {"ai_signals": 9999, "basket_orders": 9999, "advanced_analytics": True, "api_access": True}
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
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = "Cash"

    # Subscription
    if 'subscription_tier' not in st.session_state: st.session_state.subscription_tier = "Professional"
    if 'ai_signals_used' not in st.session_state: st.session_state.ai_signals_used = 0
    if 'basket_orders_used' not in st.session_state: st.session_state.basket_orders_used = 0
    
    # Watchlists
    if 'watchlists' not in st.session_state:
        st.session_state.watchlists = {
            "Indices": [{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'}],
            "Bank Stocks": [{'symbol': 'HDFCBANK', 'exchange': 'NSE'}, {'symbol': 'ICICIBANK', 'exchange': 'NSE'}],
            "IT Stocks": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}]
        }
    if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Indices"
    
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
    if 'ai_trade_signals' not in st.session_state: st.session_state.ai_trade_signals = []
    if 'smart_money_flow' not in st.session_state: st.session_state.smart_money_flow = {}
    if 'sector_rotation' not in st.session_state: st.session_state.sector_rotation = {}
    
    # HFT Mode
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []

# ================ 2. HELPER & UTILITY FUNCTIONS ================

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
    
    elif feature_name in ["advanced_analytics", "api_access"]:
        if not limits[feature_name]:
            st.error(f"{feature_name.replace('_', ' ').title()} not available in {tier} plan. Upgrade required.")
            return False
        return True
    
    return True

def get_broker_client():
    """Gets current broker client from session state."""
    return st.session_state.get('kite')

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None):
    """A quick trade dialog for placing orders."""
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    instrument_df = get_instrument_df()
    
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
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    }
    return holidays_by_year.get(year, [])

def get_market_status():
    """Checks if the Indian stock market is open."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "var(--red)"}
    if market_open_time <= now.time() <= market_close_time:
        return {"status": "OPEN", "color": "var(--green)"}
    return {"status": "CLOSED", "color": "var(--red)"}

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
        for _, row in data.iterrows():
            price = row['Price']
            change = row['% Change']
            if not np.isnan(price):
                color = 'var(--green)' if change > 0 else 'var(--red)'
                bar_html += f"<span>{row['Ticker']}: {price:,.2f} <span style='color:{color};'>({change:+.2f}%)</span></span>"
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile for 2FA."""
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    return base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None, comparison_data=None):
    """Generates a Plotly chart with various chart types, overlays, and comparisons."""
    fig = go.Figure()
    
    # Handle single symbol chart
    if not df.empty:
        chart_df = df.copy()
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df.columns = chart_df.columns.droplevel(0)
        chart_df.columns = [str(col).lower() for col in chart_df.columns]
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in chart_df.columns for col in required_cols):
            return go.Figure()

        if chart_type == 'Heikin-Ashi':
            ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
            fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name=f'{ticker} (HA)'))
        elif chart_type == 'Line':
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name=f'{ticker}'))
        else: # Candlestick or Bar
            trace_class = go.Ohlc if chart_type == 'Bar' else go.Candlestick
            fig.add_trace(trace_class(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name=f'{ticker}'))

    # Handle comparison data
    if comparison_data:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (comp_ticker, comp_df) in enumerate(comparison_data.items()):
            if not comp_df.empty and 'close' in comp_df.columns:
                normalized_close = (comp_df['close'] / comp_df['close'].iloc[0]) * 100
                fig.add_trace(go.Scatter(x=comp_df.index, y=normalized_close, mode='lines', name=f'{comp_ticker} (Norm)', line=dict(color=colors[i % len(colors)])))

    # Add forecast and confidence intervals
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    if conf_int_df is not None:
        fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence'))

    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    chart_title = f'Price Comparison' if comparison_data else f'{ticker} Chart'
    fig.update_layout(title=chart_title, yaxis_title='Price', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments."""
    client = get_broker_client()
    if not client: return pd.DataFrame()
    try:
        df = pd.DataFrame(client.instruments())
        if 'expiry' in df.columns: df['expiry'] = pd.to_datetime(df['expiry'])
        return df
    except Exception: return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data, supporting both live and demo modes."""
    client = get_broker_client()
    if st.session_state.broker == "Demo":
        return generate_demo_data(period or '1mo', interval)
    
    if not client or not instrument_token: return pd.DataFrame()
    
    if not to_date: to_date = datetime.now().date()
    if not from_date:
        days_map = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
        from_date = to_date - timedelta(days=days_map.get(period, 365))
    if from_date > to_date: from_date = to_date - timedelta(days=1)

    try:
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.ta.adx(append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
        df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
        df.ta.supertrend(append=True); df.ta.atr(append=True)
        return df
    except Exception: return pd.DataFrame()

def generate_demo_data(period, interval):
    """Generate synthetic data for demo mode."""
    days = {'1d': 1, '5d': 5, '1mo': 30, '6mo': 180, '1y': 365, '5y': 1825}.get(period, 30)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    price = random.uniform(1000, 5000)
    data = []
    for date in dates:
        change = price * random.uniform(-0.02, 0.02)
        open_price, close_price = price, price + change
        high = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        data.append({'date': date, 'open': open_price, 'high': high, 'low': low, 'close': close_price, 'volume': random.randint(100000, 5000000)})
        price = close_price
    df = pd.DataFrame(data).set_index('date')
    df.ta.adx(append=True); df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
    return df

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices for a list of symbols."""
    client = get_broker_client()
    if st.session_state.broker == "Demo":
        return pd.DataFrame([{'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': random.uniform(100, 5000), 'Change': random.uniform(-50, 50), '% Change': random.uniform(-2, 2)} for item in symbols_with_exchange])

    if not client or not symbols_with_exchange: return pd.DataFrame()
    
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
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth for an instrument."""
    client = get_broker_client()
    if not client or not instrument_token: return None
    try:
        return client.depth(instrument_token).get(str(instrument_token))
    except Exception: return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches the options chain."""
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []

    ltp_symbol_map = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}
    ltp_symbol = ltp_symbol_map.get(underlying, underlying)
    
    try:
        underlying_ltp = client.ltp(f"NSE:{ltp_symbol}")[f"NSE:{ltp_symbol}"]['last_price']
    except Exception: underlying_ltp = 0.0

    options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == 'NFO')]
    if options.empty: return pd.DataFrame(), None, underlying_ltp, []

    expiries = sorted(options['expiry'].dt.date.unique())
    available_expiries = [e for e in expiries if e >= datetime.now().date()]
    if not available_expiries: return pd.DataFrame(), None, underlying_ltp, []

    expiry_date = pd.to_datetime(expiry_date).date() if expiry_date else available_expiries[0]
    
    chain_df = options[options['expiry'].dt.date == expiry_date].sort_values(by='strike')
    ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
    pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()

    instruments = [f"NFO:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
    if not instruments: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries

    try:
        quotes = client.quote(instruments)
        ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        ce_df['oi'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('oi', 0))
        pe_df['oi'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('oi', 0))
        
        final_chain = pd.merge(
            ce_df[['tradingsymbol', 'strike', 'LTP', 'oi']],
            pe_df[['tradingsymbol', 'strike', 'LTP', 'oi']],
            on='strike', suffixes=('_CE', '_PE')
        ).rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'oi_CE': 'CALL OI', 'oi_PE': 'PUT OI', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
        return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries
    except Exception: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches portfolio positions and holdings."""
    client = get_broker_client()
    if st.session_state.broker == "Demo":
        demo_holdings = [
            {'tradingsymbol': 'RELIANCE', 'quantity': 10, 'average_price': 2850.50, 'last_price': 2920.75, 'pnl': 702.50},
            {'tradingsymbol': 'TCS', 'quantity': 5, 'average_price': 3850.00, 'last_price': 3915.25, 'pnl': 326.25},
        ]
        total_pnl = sum(d['pnl'] for d in demo_holdings)
        total_investment = sum(d['quantity'] * d['average_price'] for d in demo_holdings)
        return pd.DataFrame(demo_holdings), pd.DataFrame(demo_holdings), total_pnl, total_investment

    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

    try:
        positions = client.positions().get('net', [])
        holdings = client.holdings()
        positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
        total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
        holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
        total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
        return positions_df, holdings_df, total_pnl, total_investment
    except Exception: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order."""
    client = get_broker_client()
    if st.session_state.broker == "Demo":
        st.toast(f"✅ Demo Order: {transaction_type} {quantity} {symbol} @ {order_type}", icon="🧪")
        return

    if not client: st.error("Broker not connected."); return

    try:
        instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
        if instrument.empty: st.error(f"Symbol '{symbol}' not found."); return
        exchange = instrument.iloc[0]['exchange']
        
        order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")
        st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed"})

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
    analyzer = SentimentIntensityAnalyzer()
    news_sources = {
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                if query is None or query.lower() in entry.title.lower():
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception: continue
    return pd.DataFrame(all_news)

@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    if not tickers: return pd.DataFrame()
    try:
        data_yf = yf.download(list(tickers.values()), period="5d")
        if data_yf.empty: return pd.DataFrame()
        data = []
        for name, yf_ticker in tickers.items():
            hist = data_yf.loc[:, (slice(None), yf_ticker)] if len(tickers) > 1 else data_yf
            hist.columns = hist.columns.droplevel(1) if len(tickers) > 1 else hist.columns
            if len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
                data.append({'Ticker': name, 'Price': last_price, '% Change': pct_change})
        return pd.DataFrame(data)
    except Exception: return pd.DataFrame()

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty: return {}
    latest = df.iloc[-1].copy()
    latest.index = [c.lower() for c in latest.index]
    interpretation = {}
    
    rsi_col = next((c for c in latest.index if 'rsi' in c), None)
    macd_col = next((c for c in latest.index if 'macd' in c and 'macds' not in c and 'macdh' not in c), None)
    signal_col = next((c for c in latest.index if 'macds' in c), None)
    adx_col = next((c for c in latest.index if 'adx' in c), None)
    
    if rsi_col and (r := latest.get(rsi_col)) is not None:
        interpretation['RSI (14)'] = "Overbought" if r > 70 else "Oversold" if r < 30 else "Neutral"
    if macd_col and signal_col and (macd := latest.get(macd_col)) is not None and (signal := latest.get(signal_col)) is not None:
        interpretation['MACD'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    if adx_col and (adx := latest.get(adx_col)) is not None:
        interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else "Weak/No Trend"
        
    return interpretation

# ================ 4. AI, ML & ADVANCED FEATURES ================

# --- Black-Scholes and IV Functions ---
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
    """Calculates implied volatility using Newton-Raphson method."""
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try: return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError): return np.nan

# --- ML Forecasting Functions ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data, forecast_steps=30):
    if _data.empty or len(_data) < 100: return None, None, None
    df = _data.copy(); df.index = pd.to_datetime(df.index)
    try:
        decomposed = seasonal_decompose(df['close'], model='additive', period=7)
        model = ARIMA(df['close'] - decomposed.seasonal, order=(5, 1, 0)).fit()
        fitted = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted}).dropna()
        forecast = model.get_forecast(steps=forecast_steps)
        future_seasonal = np.tile(decomposed.seasonal.iloc[-7:].values, forecast_steps // 7 + 1)[:forecast_steps]
        future_forecast = forecast.predicted_mean + future_seasonal
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame({'Predicted': future_forecast.values}, index=future_dates)
        conf_int = forecast.conf_int(alpha=0.05)
        conf_int_df = pd.DataFrame({'lower': conf_int.iloc[:, 0] + future_seasonal, 'upper': conf_int.iloc[:, 1] + future_seasonal}, index=future_dates)
        return forecast_df, backtest_df, conf_int_df
    except Exception: return None, None, None

@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads and combines historical CSV with live broker data."""
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info: return pd.DataFrame()
    try:
        hist_df = pd.read_csv(source_info['github_url'])
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True).dt.tz_localize(None)
        hist_df = hist_df.set_index('Date').rename(columns=str.lower)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns: hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    except Exception: return pd.DataFrame()
    
    live_df = pd.DataFrame()
    # Live data fetching logic can be added here if broker is connected
    return hist_df.sort_index()

# --- AI Signal Generation ---
def generate_ai_trade_signal(symbol):
    if not check_subscription_feature("ai_signals"): return None
    with st.spinner(f"AI analyzing {symbol}..."):
        # Simulated analysis for demo
        signal = random.choice(["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])
        confidence = random.uniform(60, 95)
        current_price = random.uniform(100, 5000)
        signal_data = {
            'symbol': symbol, 'signal': signal, 'confidence': confidence, 'current_price': current_price,
            'target_1': current_price * 1.02 if "BUY" in signal else current_price * 0.98,
            'target_2': current_price * 1.04 if "BUY" in signal else current_price * 0.96,
            'stop_loss': current_price * 0.98 if "BUY" in signal else current_price * 1.02,
            'timestamp': datetime.now()
        }
        st.session_state.ai_trade_signals.append(signal_data)
        return signal_data

# --- Backtesting Functions ---
def run_backtest(strategy_func, data, **params):
    """Runs a backtest for a given strategy function."""
    df = data.copy()
    signals = strategy_func(df, **params)
    initial_capital = 100000.0; capital = initial_capital; position = 0; portfolio_value = []
    for i in range(len(df)):
        if signals[i] == 'BUY' and position == 0:
            position = capital / df['close'][i]; capital = 0
        elif signals[i] == 'SELL' and position > 0:
            capital = position * df['close'][i]; position = 0
        portfolio_value.append(capital + (position * df['close'][i]))
    pnl = (portfolio_value[-1] - initial_capital) / initial_capital * 100
    return pnl, pd.Series(portfolio_value, index=df.index)

def rsi_strategy(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    rsi = ta.rsi(df['close'], length=rsi_period); signals = [''] * len(df)
    for i in range(1, len(df)):
        if rsi[i-1] < rsi_oversold and rsi[i] > rsi_oversold: signals[i] = 'BUY'
        elif rsi[i-1] > rsi_overbought and rsi[i] < rsi_overbought: signals[i] = 'SELL'
    return signals

def macd_strategy(df, fast=12, slow=26, signal=9):
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal); signals = [''] * len(df)
    for i in range(1, len(df)):
        if macd[f'MACD_{fast}_{slow}_{signal}'][i-1] < macd[f'MACDs_{fast}_{slow}_{signal}'][i-1] and macd[f'MACD_{fast}_{slow}_{signal}'][i] > macd[f'MACDs_{fast}_{slow}_{signal}'][i]: signals[i] = 'BUY'
        elif macd[f'MACD_{fast}_{slow}_{signal}'][i-1] > macd[f'MACDs_{fast}_{slow}_{signal}'][i-1] and macd[f'MACD_{fast}_{slow}_{signal}'][i] < macd[f'MACDs_{fast}_{slow}_{signal}'][i]: signals[i] = 'SELL'
    return signals

def supertrend_strategy(df, period=7, multiplier=3):
    st = ta.supertrend(df['high'], df['low'], df['close'], length=period, multiplier=multiplier); signals = [''] * len(df)
    st_col = next((c for c in st.columns if 'SUPERT' in c), None)
    if not st_col: return signals
    for i in range(1, len(df)):
        if df['close'][i] > st[st_col][i-1] and df['close'][i-1] <= st[st_col][i-1]: signals[i] = 'BUY'
        elif df['close'][i] < st[st_col][i-1] and df['close'][i-1] >= st[st_col][i-1]: signals[i] = 'SELL'
    return signals

# ================ 5. PAGE DEFINITIONS ================

def page_dashboard():
    """Redesigned 'Trader UI' Dashboard."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Please connect to a broker to view the dashboard.")
        return

    # Top Row: Market Pulse & Key Indices
    col1, col2, col3, col4 = st.columns(4)
    index_data = get_watchlist_data([{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'}, {'symbol': 'INDIA VIX', 'exchange': 'NSE'}])
    if not index_data.empty:
        nifty = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
        sensex = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
        vix = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
        col1.metric("NIFTY 50", f"{nifty['Price']:,.2f}", f"{nifty['% Change']:.2f}%")
        col2.metric("SENSEX", f"{sensex['Price']:,.2f}", f"{sensex['% Change']:.2f}%")
        col3.metric("INDIA VIX", f"{vix['Price']:,.2f}", f"{vix['% Change']:.2f}%")
    
    with col4: # Bharatiya Market Pulse (BMP)
        if not index_data.empty:
            bmp_score = 50 + (nifty['% Change'] + sensex['% Change']) * 10 - vix['Price'] * 0.5
            bmp_score = max(0, min(100, bmp_score))
            if bmp_score > 70: label, color = "Bharat Udaan", "var(--green)"
            elif bmp_score > 40: label, color = "Bharat Santulan", "var(--orange)"
            else: label, color = "Bharat Mandhi", "var(--red)"
            st.markdown(f'<div class="metric-card" style="border-color:{color}; text-align:center;"><h4>BMP: {bmp_score:.1f}</h4><p style="color:{color}; font-weight:bold;">{label}</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Middle Row: Main Content
    col1, col2 = st.columns([1, 1], gap="large")
    with col1: # Watchlist & Portfolio
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio"])
        with tab1:
            st.session_state.active_watchlist = st.radio("Select Watchlist", options=st.session_state.watchlists.keys(), horizontal=True, label_visibility="collapsed")
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]
            
            with st.form(key="add_stock_form"):
                add_cols = st.columns([2, 1, 1])
                new_symbol = add_cols[0].text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_cols[1].selectbox("Exchange", ["NSE", "BSE"], label_visibility="collapsed")
                if add_cols[2].form_submit_button("Add"):
                    if new_symbol and not any(d['symbol'] == new_symbol.upper() for d in active_list):
                        active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange}); st.rerun()
            
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for _, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", unsafe_allow_html=True)
                    w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                    if w_cols[2].button("B", key=f"buy_{row['Ticker']}"): quick_trade_dialog(row['Ticker'])
                    if w_cols[3].button("S", key=f"sell_{row['Ticker']}"): quick_trade_dialog(row['Ticker'])
                    if w_cols[4].button("🗑️", key=f"del_{row['Ticker']}"):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != row['Ticker']]; st.rerun()
        with tab2:
            _, holdings_df, total_pnl, total_investment = get_portfolio()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
            st.metric("Today's P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            if not holdings_df.empty: st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    with col2: # Chart
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token or st.session_state.broker == "Demo":
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)

def page_advanced_charting():
    """Page for advanced multi-chart layout."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Connect to a broker to use charting tools."); return

    layout_option = st.radio("Layout", ["Single", "2 Charts", "4 Charts"], horizontal=True)
    num_charts = {"Single": 1, "2 Charts": 2, "4 Charts": 4}[layout_option]
    
    grid = [st.columns(2) for _ in range((num_charts + 1) // 2)]
    flat_grid = [col for row in grid for col in row]

    for i in range(num_charts):
        with flat_grid[i].container(border=True):
            st.subheader(f"Chart {i+1}")
            chart_cols = st.columns(4)
            ticker = chart_cols[0].text_input("Symbol", "NIFTY 50", key=f"ticker_{i}").upper()
            period = chart_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key=f"period_{i}")
            interval = chart_cols[2].selectbox("Interval", ["minute", "day", "week"], index=1, key=f"interval_{i}")
            chart_type = chart_cols[3].selectbox("Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{i}")
            
            token = get_instrument_token(ticker, instrument_df) if st.session_state.broker != "Demo" else 12345
            if token:
                data = get_historical_data(token, interval, period=period)
                if not data.empty:
                    st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)

def page_premarket_pulse():
    """Global market overview and premarket indicators."""
    display_header()
    st.title("Premarket & Global Cues")
    st.markdown("---")
    
    st.subheader("Global Market Snapshot")
    global_tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "Hang Seng": "^HSI"}
    global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_tickers))
        for i, row in global_data.iterrows():
            if not np.isnan(row['Price']):
                cols[i].metric(label=row['Ticker'], value=f"{row['Price']:,.2f}", delta=f"{row['% Change']:.2f}%")
    
    st.markdown("---")
    st.subheader("Latest Market News")
    news_df = fetch_and_analyze_news()
    if not news_df.empty:
        for _, news in news_df.head(5).iterrows():
            icon = "🔼" if news['sentiment'] > 0.2 else "🔽" if news['sentiment'] < -0.2 else "▶️"
            st.markdown(f"**{icon} [{news['title']}]({news['link']})** - *{news['source']}*")

def page_fo_analytics():
    """F&O Analytics page with options chain and analysis."""
    display_header()
    st.title("F&O Analytics Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Connect to a broker to access F&O Analytics."); return

    tab1, tab2 = st.tabs(["Options Chain", "PCR & Volatility Analysis"])
    with tab1:
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        st.session_state.underlying_pcr = underlying
        chain_df, expiry, ltp, _ = get_options_chain(underlying, instrument_df)
        if not chain_df.empty:
            st.metric("Current Price", f"₹{ltp:,.2f}", f"Expiry: {expiry.strftime('%d %b %Y') if expiry else 'N/A'}")
            st.dataframe(chain_df.style.format({'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}', 'STRIKE': '₹{:.0f}'}), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Put-Call Ratio & Volatility Surface")
        chain_df, expiry, ltp, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)
        if not chain_df.empty and 'CALL OI' in chain_df.columns:
            total_ce_oi = chain_df['CALL OI'].sum(); total_pe_oi = chain_df['PUT OI'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            cols = st.columns(3)
            cols[0].metric("Total CE OI", f"{total_ce_oi:,.0f}")
            cols[1].metric("Total PE OI", f"{total_pe_oi:,.0f}")
            cols[2].metric("PCR", f"{pcr:.2f}", "Bearish" if pcr > 1.3 else "Bullish" if pcr < 0.7 else "Neutral")

def page_ai_trade_signals():
    """AI-Powered Trade Signals page."""
    display_header()
    st.title("AI Trade Signals")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Generate Signal")
        symbol = st.text_input("Stock Symbol", "RELIANCE", key="ai_signal_symbol").upper()
        if st.button("Generate AI Signal", type="primary", use_container_width=True):
            generate_ai_trade_signal(symbol)
    
    with col2:
        st.subheader("Recent Signals")
        if st.session_state.ai_trade_signals:
            for signal in reversed(st.session_state.ai_trade_signals[-5:]):
                color = "var(--green)" if "BUY" in signal['signal'] else "var(--red)" if "SELL" in signal['signal'] else "var(--orange)"
                st.markdown(f"""
                <div class="signal-card" style="border-left-color: {color};">
                    <div style="display: flex; justify-content: space-between;">
                        <h4 style="margin:0;">{signal['symbol']} - {signal['signal']}</h4>
                        <span style="color:{color};">{signal['confidence']:.1f}%</span>
                    </div>
                    <p style="margin: 5px 0;">Price: ₹{signal['current_price']:.2f} | Target: ₹{signal['target_1']:.2f} | SL: ₹{signal['stop_loss']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

def page_forecasting_ml():
    """Page for advanced ML forecasting."""
    display_header()
    st.title("Advanced ML Forecasting")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select Instrument", list(ML_DATA_SOURCES.keys()))
        duration_key = st.radio("Forecast Duration", ["1 Month", "3 Months"], horizontal=True)
        forecast_steps = 30 if duration_key == "1 Month" else 90
        if st.button("Train & Forecast", use_container_width=True, type="primary"):
            with st.spinner("Training model..."):
                data = load_and_combine_data(instrument_name)
                if not data.empty and len(data) > 100:
                    forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                    st.session_state.update({'ml_forecast_df': forecast_df, 'ml_backtest_df': backtest_df, 'ml_conf_int_df': conf_int_df, 'ml_instrument_name': instrument_name, 'ml_historical_data': data})
                    st.success("Model trained!")
                else: st.error("Not enough data to train model.")
    
    with col2:
        if 'ml_instrument_name' in st.session_state:
            st.subheader(f"Forecast for {st.session_state.ml_instrument_name}")
            data, forecast_df, backtest_df, conf_int_df = st.session_state.ml_historical_data, st.session_state.ml_forecast_df, st.session_state.ml_backtest_df, st.session_state.ml_conf_int_df
            if forecast_df is not None:
                fig = create_chart(data.tail(252), st.session_state.ml_instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Model Fit', line=dict(color='orange', dash='dot')))
                st.plotly_chart(fig, use_container_width=True)
                mape = mean_absolute_percentage_error(backtest_df.tail(100)['Actual'], backtest_df.tail(100)['Predicted'])
                st.metric("Model Accuracy (Last 100 days)", f"{100 - mape:.2f}%")

def page_hft_terminal():
    """High-Frequency Trading terminal."""
    display_header()
    st.title("HFT Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Connect to a broker to use the HFT Terminal."); return
    st_autorefresh(interval=2000, key="hft_refresh")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Symbol", "RELIANCE", key="hft_symbol").upper()
        token = get_instrument_token(symbol, instrument_df)
        if not token and st.session_state.broker != "Demo":
            st.error(f"Could not find {symbol}"); return
        
        quote = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}]).iloc[0]
        last_price = quote['Price']
        tick_class = "tick-up" if last_price > st.session_state.hft_last_price else "tick-down" if last_price < st.session_state.hft_last_price else ""
        st.session_state.hft_last_price = last_price
        st.markdown(f'<div class="{tick_class}" style="font-size: 2.5rem; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; background: var(--widget-bg);">₹{last_price:,.2f}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Quick Order")
        quantity = st.number_input("Quantity", min_value=1, value=1, key="hft_qty")
        if st.button("BUY", use_container_width=True): place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS')
        if st.button("SELL", use_container_width=True): place_order(instrument_df, symbol, quantity, 'MARKET', 'SELL', 'MIS')

def page_basket_orders():
    """Page for creating and executing basket orders."""
    display_header()
    st.title("Basket Orders")
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Connect to a broker to use Basket Orders."); return
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Create Basket Order")
        with st.form(key="basket_form"):
            symbol = st.text_input("Symbol").upper()
            quantity = st.number_input("Quantity", min_value=1, value=1)
            transaction = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            if st.form_submit_button("Add to Basket"):
                st.session_state.basket.append({'symbol': symbol, 'quantity': quantity, 'transaction_type': transaction, 'order_type': 'MARKET', 'product': 'MIS'})
                st.rerun()
    with col2:
        st.subheader("Current Basket")
        if st.session_state.basket:
            for i, item in enumerate(st.session_state.basket):
                st.write(f"{i+1}. {item['transaction_type']} {item['quantity']} of {item['symbol']}")
            if st.button("Execute Basket", type="primary"):
                if check_subscription_feature("basket_orders"):
                    for item in st.session_state.basket:
                        place_order(instrument_df, **item)
                    st.session_state.basket = []
                    st.success("Basket executed!"); st.rerun()

def page_fundamental_analysis():
    """Page for fundamental analysis."""
    display_header()
    st.title("Fundamental Analysis")
    st.info("This is a demo page using simulated fundamental data.")

    symbol = st.text_input("Stock Symbol", "RELIANCE").upper()
    if st.button("Analyze"):
        # Simulated data
        data = {'P/E': random.uniform(10, 40), 'P/B': random.uniform(1, 5), 'ROE': random.uniform(5, 25), 'Debt/Equity': random.uniform(0.1, 2.0)}
        st.subheader(f"Key Metrics for {symbol}")
        cols = st.columns(4)
        for i, (metric, value) in enumerate(data.items()):
            cols[i].metric(metric, f"{value:.2f}")

def page_ai_trade_assistant():
    """AI-powered trading assistant with command parsing."""
    display_header()
    st.title("AI Trade Assistant")
    instrument_df = get_instrument_df()

    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help with your portfolio or the markets?"}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_lower = prompt.lower()
                response = "I can help with portfolio, orders, and market data. Try 'What are my positions?' or 'Buy 10 shares of RELIANCE'."
                client = get_broker_client()
                
                if not client and st.session_state.broker != "Demo":
                    response = "Please connect to a broker first."
                elif any(word in prompt_lower for word in ["holdings", "positions", "portfolio"]):
                    _, holdings_df, total_pnl, _ = get_portfolio()
                    response = f"Your total P&L is ₹{total_pnl:,.2f}. Holdings:\n```\n{tabulate(holdings_df, headers='keys')}\n```" if not holdings_df.empty else "You have no holdings."
                elif "buy" in prompt_lower or "sell" in prompt_lower:
                    match = re.search(r'(buy|sell)\s+(\d+)\s+(?:shares? of\s+)?([a-zA-Z0-9\-\&_]+)', prompt_lower)
                    if match:
                        trans_type, quantity, symbol = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
                        st.session_state.last_order_details = {"symbol": symbol, "quantity": quantity, "transaction_type": trans_type, "confirmed": False}
                        response = f"Ready to {trans_type} {quantity} shares of {symbol}. Please type 'confirm' to execute."
                    else:
                        response = "Please use the format: 'Buy 10 shares of RELIANCE'."
                elif prompt_lower == "confirm" and "last_order_details" in st.session_state and not st.session_state.last_order_details["confirmed"]:
                    order = st.session_state.last_order_details
                    place_order(instrument_df, order['symbol'], order['quantity'], 'MARKET', order['transaction_type'], 'MIS')
                    order['confirmed'] = True
                    response = f"Order placed for {order['quantity']} shares of {order['symbol']}."
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def page_algo_strategy_maker():
    """Page for backtesting and executing simple algo strategies."""
    display_header()
    st.title("Algo Strategy Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty and st.session_state.broker != "Demo":
        st.info("Connect to a broker to use this feature."); return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Strategy Configuration")
        strategy_map = {"RSI Crossover": rsi_strategy, "MACD Crossover": macd_strategy, "Supertrend": supertrend_strategy}
        strategy_name = st.selectbox("Select Strategy", list(strategy_map.keys()))
        symbol = st.text_input("Symbol", "RELIANCE").upper()
        quantity = st.number_input("Trade Quantity", min_value=1, value=1)
        if st.button("Run Backtest & Get Signal", type="primary"):
            token = get_instrument_token(symbol, instrument_df)
            data = get_historical_data(token, 'day', period='1y')
            if not data.empty:
                pnl, curve = run_backtest(strategy_map[strategy_name], data)
                signal = strategy_map[strategy_name](data)[-1]
                st.session_state['backtest_results'] = {'pnl': pnl, 'curve': curve, 'signal': signal, 'symbol': symbol, 'quantity': quantity}
    with col2:
        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            st.subheader("Backtest Results (1 Year)")
            st.metric("Total P&L", f"{results['pnl']:.2f}%")
            st.line_chart(results['curve'])
            st.subheader("Live Signal")
            signal = results['signal']
            color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
            st.markdown(f"### Latest Signal: <span style='color:{color};'>{signal if signal else 'HOLD'}</span>", unsafe_allow_html=True)
            if signal in ["BUY", "SELL"]:
                if st.button(f"Place {signal} Order for {results['quantity']} of {results['symbol']}"):
                    place_order(instrument_df, results['symbol'], results['quantity'], "MARKET", signal, "MIS")

def page_subscription_plans():
    """Subscription Plans and Pricing page."""
    display_header()
    st.title("Subscription Plans")
    
    cols = st.columns(3)
    plans = ["Basic", "Professional", "Institutional"]
    for i, plan_name in enumerate(plans):
        with cols[i]:
            plan = SUBSCRIPTION_PLANS[plan_name]
            is_featured = plan_name == "Professional"
            st.markdown(f'<div class="tier-card {"featured" if is_featured else ""}"><h3>{plan_name} {"<span class=\'premium-badge\'>POPULAR</span>" if is_featured else ""}</h3><h2>₹{plan["price"]}<small>/month</small></h2><ul>{"".join([f"<li>{feat}</li>" for feat in plan["features"]])}</ul></div>', unsafe_allow_html=True)
            if st.session_state.subscription_tier != plan_name:
                if st.button(f"Switch to {plan_name}", use_container_width=True, type="primary" if is_featured else "secondary"):
                    st.session_state.subscription_tier = plan_name; st.success(f"Switched to {plan_name} Plan!"); st.rerun()
            else:
                st.success("Current Plan")

# ================ 6. MAIN APP & SIDEBAR ================

def main():
    """Main application entry point and router."""
    initialize_session_state()
    apply_custom_styling()
    
    # --- AUTHENTICATION FLOW ---
    if not st.session_state.get('profile'):
        login_page()
        return

    if not st.session_state.get('login_animation_complete'):
        show_login_animation()
        return

    if not st.session_state.get('two_factor_setup_complete'):
        qr_code_dialog()
        return

    if not st.session_state.get('authenticated'):
        two_factor_dialog()
        return

    # --- MAIN APP UI ---
    display_overnight_changes_bar()
    
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.profile['user_name']}")
        st.caption(f"Connected via {st.session_state.broker}")
        st.divider()
        
        st.header("Terminal Controls")
        st.session_state.theme = st.radio("Theme", ["Dark", "Light"], horizontal=True, index=0 if st.session_state.theme == "Dark" else 1)
        st.session_state.terminal_mode = st.radio("Mode", ["Cash", "Options", "HFT"], horizontal=True)
        st.divider()
        
        page_mapping = {
            "Cash": ["📊 Dashboard", "📈 Adv. Charting", "🌍 Premarket Pulse", "🤖 AI Trade Signals", "🔮 Forecasting & ML", "🧺 Basket Orders", "📋 Fundamentals", "💡 Algo Hub", "🤝 AI Assistant"],
            "Options": ["📊 F&O Analytics", "📈 Adv. Charting", "💡 Algo Hub", "🤝 AI Assistant"],
            "HFT": ["⚡ HFT Terminal"]
        }
        selection = st.radio("Navigation", page_mapping[st.session_state.terminal_mode])
        
        st.divider()
        if st.sidebar.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    pages = {
        "📊 Dashboard": page_dashboard, "📈 Adv. Charting": page_advanced_charting,
        "🌍 Premarket Pulse": page_premarket_pulse, "📊 F&O Analytics": page_fo_analytics,
        "🤖 AI Trade Signals": page_ai_trade_signals, "🔮 Forecasting & ML": page_forecasting_ml,
        "⚡ HFT Terminal": page_hft_terminal, "🧺 Basket Orders": page_basket_orders,
        "📋 Fundamentals": page_fundamental_analysis, "🤝 AI Assistant": page_ai_trade_assistant,
        "💡 Algo Hub": page_algo_strategy_maker, "💎 Subscriptions": page_subscription_plans
    }
    
    # Execute the selected page function
    page_function = pages.get(selection)
    if page_function:
        page_function()
    else:
        page_dashboard()

def login_page():
    """Displays the login page for broker authentication."""
    st.title("BlockVista Terminal Login")
    broker = st.selectbox("Broker", ["Zerodha", "Demo"])
    
    if broker == "Zerodha":
        api_key = st.secrets.get("ZERODHA_API_KEY")
        api_secret = st.secrets.get("ZERODHA_API_SECRET")
        if not api_key or not api_secret:
            st.error("Kite API credentials not found in Streamlit secrets.")
            return
            
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                kite.set_access_token(st.session_state.access_token)
                st.session_state.kite = kite
                st.session_state.profile = kite.profile()
                st.session_state.broker = "Zerodha"
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())
    elif broker == "Demo":
        if st.button("Start Demo Mode"):
            st.session_state.broker = "Demo"
            st.session_state.profile = {'user_name': 'Demo User', 'user_id': 'DM0001'}
            st.session_state.authenticated = True # Bypass 2FA for demo
            st.session_state.two_factor_setup_complete = True
            st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
    progress_bar = st.progress(0, text="Authenticating user...")
    a_time.sleep(0.7)
    progress_bar.progress(50, text="Establishing secure connection...")
    a_time.sleep(0.7)
    progress_bar.progress(100, text="Initializing terminal... COMPLETE")
    a_time.sleep(1)
    st.session_state['login_animation_complete'] = True
    st.rerun()

@st.dialog("Two-Factor Authentication Setup")
def qr_code_dialog():
    st.info("Scan this QR code with your authenticator app (e.g., Google Authenticator). This is a one-time setup.")
    secret = get_user_secret(st.session_state.profile)
    st.session_state.pyotp_secret = secret
    uri = pyotp.totp.TOTP(secret).provisioning_uri(st.session_state.profile['user_name'], issuer_name="BlockVista")
    img = qrcode.make(uri)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    st.image(buf.getvalue())
    if st.button("Continue", use_container_width=True):
        st.session_state.two_factor_setup_complete = True; st.rerun()

@st.dialog("Two-Factor Authentication")
def two_factor_dialog():
    st.subheader("Enter your 6-digit code")
    auth_code = st.text_input("2FA Code", max_chars=6)
    if st.button("Authenticate"):
        if pyotp.TOTP(st.session_state.pyotp_secret).verify(auth_code):
            st.session_state.authenticated = True; st.rerun()
        else:
            st.error("Invalid code.")

if __name__ == "__main__":
    main()
