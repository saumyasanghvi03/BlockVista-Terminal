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

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

def display_header():
    """Displays the main header for the application."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #1e3c72, #2a5298); border-radius: 10px; margin-bottom: 1rem;">
        <h1 style="color: white; margin: 0;">🚀 BlockVista Terminal</h1>
        <p style="color: #cccccc; margin: 0;">Professional Trading & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
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

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    holidays_by_year = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
        2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    }
    return holidays_by_year.get(year, [])

# =============================================================================
    
def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
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

# =============================================================================
# MARKET TIMING FUNCTIONS - REPLACE EXISTING ONES
# =============================================================================

def is_market_hours():
    """Check if current time is within market hours (9:15 AM to 3:30 PM, Monday to Friday)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()  # Monday=0, Sunday=6
    
    # Check if it's a weekday (Monday to Friday)
    if current_day >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Market hours: 9:15 AM to 3:30 PM
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    return market_open <= current_time <= market_close

def is_pre_market_hours():
    """Check if current time is pre-market hours (9:00 AM to 9:15 AM, Monday to Friday)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    # Check if it's a weekday
    if current_day >= 5:
        return False
    
    # Pre-market hours: 9:00 AM to 9:15 AM
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    
    return pre_market_start <= current_time < market_open

def is_square_off_time():
    """Check if current time is square-off time for Equity/Cash (3:20 PM to 3:30 PM, Monday to Friday)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    # Check if it's a weekday
    if current_day >= 5:  # Saturday or Sunday
        return False
    
    # Square-off time for Equity/Cash: 3:20 PM to 3:30 PM
    square_off_start = time(15, 20)  # 3:20 PM
    square_off_end = time(15, 30)    # 3:30 PM
    
    return square_off_start <= current_time <= square_off_end

def is_derivatives_square_off_time():
    """Check if current time is square-off time for Equity/Index Derivatives (3:25 PM to 3:30 PM)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5:
        return False
    
    # Square-off time for Derivatives: 3:25 PM to 3:30 PM
    square_off_start = time(15, 25)  # 3:25 PM
    square_off_end = time(15, 30)    # 3:30 PM
    
    return square_off_start <= current_time <= square_off_end

def get_market_status():
    """Get current market status and next market event"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    current_date = now.date()
    
    # Get holidays for current year
    holidays = get_market_holidays(now.year)
    
    # Check if today is a holiday
    if current_date.strftime('%Y-%m-%d') in holidays:
        next_market = now + timedelta(days=1)
        # Skip weekends and holidays
        while next_market.weekday() >= 5 or next_market.strftime('%Y-%m-%d') in holidays:
            next_market += timedelta(days=1)
        next_market = next_market.replace(hour=9, minute=0, second=0, microsecond=0)
        return "holiday", next_market
    
    # Weekend check
    if current_day >= 5:  # Saturday or Sunday
        days_until_monday = (7 - current_day) % 7
        if days_until_monday == 0:  # Already Monday? (shouldn't happen but safe)
            days_until_monday = 7
        next_market = now + timedelta(days=days_until_monday)
        return "weekend", next_market.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # Market timing definitions
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    equity_square_off = time(15, 20)  # 3:20 PM for Equity/Cash
    derivatives_square_off = time(15, 25)  # 3:25 PM for Derivatives
    market_close = time(15, 30)
    
    if current_time < pre_market_start:
        # Before pre-market (overnight)
        next_market = now.replace(hour=9, minute=0, second=0, microsecond=0)
        return "market_closed", next_market
    elif pre_market_start <= current_time < market_open:
        # Pre-market hours (9:00 AM to 9:15 AM)
        next_market = now.replace(hour=9, minute=15, second=0, microsecond=0)
        return "pre_market", next_market
    elif market_open <= current_time < equity_square_off:
        # Market open (9:15 AM to 3:20 PM)
        next_market = now.replace(hour=15, minute=20, second=0, microsecond=0)
        return "market_open", next_market
    elif equity_square_off <= current_time < derivatives_square_off:
        # Equity square-off time (3:20 PM to 3:25 PM)
        next_market = now.replace(hour=15, minute=25, second=0, microsecond=0)
        return "equity_square_off", next_market
    elif derivatives_square_off <= current_time <= market_close:
        # Derivatives square-off time (3:25 PM to 3:30 PM)
        next_market = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return "derivatives_square_off", next_market
    else:
        # Market closed for the day (after 3:30 PM)
        next_day = now + timedelta(days=1)
        # Skip weekends and holidays
        while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in holidays:
            next_day += timedelta(days=1)
        next_market = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
        return "market_closed", next_market

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
        # Manual Heikin-Ashi calculation
        ha_close = (chart_df['open'] + chart_df['high'] + chart_df['low'] + chart_df['close']) / 4
        ha_open = (chart_df['open'].shift(1) + chart_df['close'].shift(1)) / 2
        ha_open.iloc[0] = (chart_df['open'].iloc[0] + chart_df['close'].iloc[0]) / 2
        ha_high = chart_df[['high', 'open', 'close']].max(axis=1)
        ha_low = chart_df[['low', 'open', 'close']].min(axis=1)
        
        fig.add_trace(go.Candlestick(x=chart_df.index, open=ha_open, high=ha_high, low=ha_low, close=ha_close, name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Bar'))
    else:
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
        
    # Bollinger Bands using TA-Lib
    if 'close' in chart_df.columns:
        upperband, middleband, lowerband = talib.BBANDS(chart_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        fig.add_trace(go.Scatter(x=chart_df.index, y=lowerband, line=dict(color='rgba(135,206,250,0.5)', width=1), name='Lower Band'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=upperband, line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='Upper Band'))
        
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
        depth = client.depth(instrument_token)
        return depth.get(str(instrument_token))
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

# ================ ALGO TRADING BOTS SECTION ================

def momentum_trader_bot(instrument_df, symbol, capital=100):
    """Momentum trading bot that buys on upward momentum and sells on downward momentum."""
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

def mean_reversion_bot(instrument_df, symbol, capital=100):
    """Mean reversion bot that trades on price returning to mean levels."""
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

def volatility_breakout_bot(instrument_df, symbol, capital=100):
    """Volatility breakout bot that trades on breakouts from consolidation."""
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

def value_investor_bot(instrument_df, symbol, capital=100):
    """Value investor bot focusing on longer-term value signals."""
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

def scalper_bot(instrument_df, symbol, capital=100):
    """High-frequency scalping bot for quick, small profits."""
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

def trend_follower_bot(instrument_df, symbol, capital=100):
    """Trend following bot that rides established trends."""
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

def page_algo_bots():
    """Main algo bots page where users can run different trading bots."""
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

# ================ AUTOMATED BOT FUNCTIONS ================

def automated_momentum_trader(instrument_df, symbol):
    """Enhanced momentum trader for automated mode."""
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

def automated_mean_reversion(instrument_df, symbol):
    """Enhanced mean reversion bot for automated mode."""
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

# Dictionary of automated bots
AUTOMATED_BOTS = {
    "Auto Momentum Trader": automated_momentum_trader,
    "Auto Mean Reversion": automated_mean_reversion
}

def initialize_automated_mode():
    """Initialize session state for fully automated trading with paper trading."""
    if 'automated_mode' not in st.session_state:
        st.session_state.automated_mode = {
            'enabled': False,
            'running': False,
            'live_trading': False,
            'bots_active': {},
            'total_capital': 10000.0,
            'risk_per_trade': 2.0,
            'max_open_trades': 5,
            'trade_history': [],
            'performance_metrics': {},
            'last_signal_check': None,
            'paper_portfolio': {
                'cash_balance': 10000.0,
                'positions': {},
                'initial_capital': 10000.0,
                'total_value': 10000.0
            }
        }
    else:
        # Migration: Ensure paper_portfolio exists for existing users
        if 'paper_portfolio' not in st.session_state.automated_mode:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': st.session_state.automated_mode.get('total_capital', 10000.0),
                'positions': {},
                'initial_capital': st.session_state.automated_mode.get('total_capital', 10000.0),
                'total_value': st.session_state.automated_mode.get('total_capital', 10000.0)
            }

def update_paper_portfolio_values(instrument_df):
    """Update paper portfolio values with current market prices."""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    if not paper_portfolio:
        return
    
    positions = paper_portfolio.get('positions', {})
    if not positions:
        paper_portfolio['total_value'] = paper_portfolio.get('cash_balance', 0.0)
        return
    
    # Get current prices for all positions
    symbols_with_exchange = []
    for symbol in positions.keys():
        symbols_with_exchange.append({'symbol': symbol, 'exchange': 'NSE'})
    
    if symbols_with_exchange:
        live_data = get_watchlist_data(symbols_with_exchange)
        
        if not live_data.empty:
            total_position_value = 0.0
            
            for symbol, position in positions.items():
                symbol_data = live_data[live_data['Ticker'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Price']
                    position_value = position['quantity'] * current_price
                    total_position_value += position_value
                    
                    # Update unrealized P&L for open trades
                    open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                                  if t.get('symbol') == symbol and t.get('status') == 'OPEN']
                    for trade in open_trades:
                        if trade.get('action') == 'BUY':
                            trade['pnl'] = (current_price - trade.get('entry_price', 0)) * trade.get('quantity', 0)
                        else:  # SELL (short)
                            trade['pnl'] = (trade.get('entry_price', 0) - current_price) * trade.get('quantity', 0)
            
            paper_portfolio['total_value'] = paper_portfolio.get('cash_balance', 0.0) + total_position_value

def close_paper_position(symbol, quantity=None):
    """Close a paper trading position."""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    
    if not paper_portfolio or symbol not in paper_portfolio.get('positions', {}):
        st.error(f"No position found for {symbol}")
        return False
    
    position = paper_portfolio['positions'][symbol]
    close_quantity = quantity if quantity else position['quantity']
    
    if close_quantity > position['quantity']:
        st.error(f"Cannot close more than current position: {position['quantity']}")
        return False
    
    # Get current price
    live_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
    if live_data.empty:
        st.error(f"Could not get current price for {symbol}")
        return False
    
    current_price = live_data.iloc[0]['Price']
    
    # Calculate P&L
    pnl = (current_price - position['avg_price']) * close_quantity
    if position.get('action') == 'SELL':  # For short positions, reverse the P&L
        pnl = -pnl
    
    # Update cash and position
    paper_portfolio['cash_balance'] += close_quantity * current_price
    paper_portfolio['positions'][symbol]['quantity'] -= close_quantity
    
    # Remove position if fully closed
    if paper_portfolio['positions'][symbol]['quantity'] == 0:
        del paper_portfolio['positions'][symbol]
    
    # Update trade history
    open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                  if t.get('symbol') == symbol and t.get('status') == 'OPEN']
    for trade in open_trades:
        if trade.get('action') == position.get('action'):  # Find matching trade
            # Close the trade
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.now().isoformat()
            trade['pnl'] = pnl
    
    st.success(f"✅ Closed {close_quantity} shares of {symbol} at ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
    return True

def get_automated_bot_performance():
    """Calculate performance metrics for automated bots with paper trading support."""
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    if not trade_history:
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        current_value = paper_portfolio.get('total_value', paper_portfolio.get('cash_balance', 0.0))
        initial_capital = paper_portfolio.get('initial_capital', current_value)
        paper_return_pct = ((current_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'open_trades': 0,
            'paper_portfolio_value': current_value,
            'paper_return_pct': paper_return_pct,
            'unrealized_pnl': 0.0
        }
    
    closed_trades = [t for t in trade_history if t.get('status') == 'CLOSED']
    open_trades = [t for t in trade_history if t.get('status') == 'OPEN']
    
    # Calculate metrics for closed trades
    winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
    
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0.0
    
    avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    # Paper trading metrics
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    initial_capital = paper_portfolio.get('initial_capital', 10000.0)
    current_value = paper_portfolio.get('total_value', paper_portfolio.get('cash_balance', initial_capital))
    paper_return_pct = ((current_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    return {
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'open_trades': len(open_trades),
        'paper_portfolio_value': current_value,
        'paper_return_pct': paper_return_pct,
        'unrealized_pnl': sum(t.get('pnl', 0) for t in open_trades)
    }

def execute_automated_trade(instrument_df, bot_result, risk_per_trade):
    """Execute trades automatically based on bot signals - with paper trading support."""
    if bot_result.get("error") or bot_result["action"] == "HOLD":
        return None
    
    try:
        symbol = bot_result["symbol"]
        action = bot_result["action"]
        current_price = bot_result["current_price"]
        
        # Calculate position size based on risk
        risk_amount = (risk_per_trade / 100.0) * st.session_state.automated_mode['total_capital']
        quantity = max(1, int(risk_amount / current_price))
        
        # Check if we have too many open trades
        open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                      if t.get('status') == 'OPEN']
        if len(open_trades) >= st.session_state.automated_mode.get('max_open_trades', 5):
            return None
        
        # Check for existing position in the same symbol
        existing_position = next((t for t in open_trades if t.get('symbol') == symbol), None)
        if existing_position:
            # Avoid opening same position multiple times
            if existing_position['action'] == action:
                return None
        
        # PLACE REAL ORDER if live trading is enabled
        order_type = "PAPER"
        if st.session_state.automated_mode.get('live_trading', False):
            try:
                # Place the real order
                place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
                order_type = "LIVE"
            except Exception as e:
                st.error(f"❌ Failed to place LIVE order for {symbol}: {e}")
                return None
        else:
            # PAPER TRADING - Simulate the trade
            paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
            trade_value = quantity * current_price
            
            if action == "BUY":
                if paper_portfolio.get('cash_balance', 0) >= trade_value:
                    # Deduct from cash, add to positions
                    paper_portfolio['cash_balance'] -= trade_value
                    if symbol in paper_portfolio.get('positions', {}):
                        paper_portfolio['positions'][symbol]['quantity'] += quantity
                        paper_portfolio['positions'][symbol]['avg_price'] = (
                            (paper_portfolio['positions'][symbol]['avg_price'] * 
                             paper_portfolio['positions'][symbol]['quantity'] + 
                             trade_value) / (paper_portfolio['positions'][symbol]['quantity'] + quantity)
                        )
                    else:
                        if 'positions' not in paper_portfolio:
                            paper_portfolio['positions'] = {}
                        paper_portfolio['positions'][symbol] = {
                            'quantity': quantity,
                            'avg_price': current_price,
                            'action': 'BUY'
                        }
                else:
                    st.error(f"❌ Paper trading: Insufficient cash for {symbol} buy order")
                    return None
            else:  # SELL action
                positions = paper_portfolio.get('positions', {})
                if symbol in positions and positions[symbol]['quantity'] >= quantity:
                    # Remove from positions, add to cash
                    position = positions[symbol]
                    paper_portfolio['cash_balance'] += quantity * current_price
                    
                    # Calculate P&L for the trade
                    pnl = (current_price - position['avg_price']) * quantity
                    if position.get('action') == 'SELL':  # For short positions, reverse the P&L
                        pnl = -pnl
                    
                    # Update position
                    paper_portfolio['positions'][symbol]['quantity'] -= quantity
                    if paper_portfolio['positions'][symbol]['quantity'] == 0:
                        del paper_portfolio['positions'][symbol]
                else:
                    st.error(f"❌ Paper trading: No position to sell for {symbol}")
                    return None
        
        # Record the trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'status': 'OPEN',
            'bot_name': bot_result['bot_name'],
            'risk_level': bot_result['risk_level'],
            'order_type': order_type,
            'pnl': 0.0,  # Initialize P&L
            'exit_price': None,
            'exit_time': None
        }
        
        if 'trade_history' not in st.session_state.automated_mode:
            st.session_state.automated_mode['trade_history'] = []
        st.session_state.automated_mode['trade_history'].append(trade_record)
        
        if order_type == "LIVE":
            st.toast(f"🤖 LIVE {action} order executed for {symbol} (Qty: {quantity})", icon="⚡")
        else:
            st.toast(f"🤖 PAPER {action} order simulated for {symbol} (Qty: {quantity})", icon="📄")
            
        return trade_record
        
    except Exception as e:
        st.error(f"Automated trade execution failed: {e}")
        return None

def run_automated_bots_cycle(instrument_df, watchlist_symbols):
    """Run one cycle of all active automated bots with paper trading updates."""
    if not st.session_state.automated_mode.get('running', False):
        return
    
    # Update paper portfolio values first
    update_paper_portfolio_values(instrument_df)
    
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    
    for bot_name in active_bots:
        for symbol in watchlist_symbols[:10]:  # Limit to first 10 symbols to avoid rate limits
            try:
                bot_function = AUTOMATED_BOTS[bot_name]
                bot_result = bot_function(instrument_df, symbol)
                
                if not bot_result.get("error") and bot_result["action"] != "HOLD":
                    execute_automated_trade(
                        instrument_df, 
                        bot_result, 
                        st.session_state.automated_mode.get('risk_per_trade', 2.0)
                    )
                
                # Small delay to avoid rate limiting
                a_time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Automated bot {bot_name} failed for {symbol}: {e}")
    
    # Update performance metrics
    st.session_state.automated_mode['performance_metrics'] = get_automated_bot_performance()
    st.session_state.automated_mode['last_signal_check'] = datetime.now().isoformat()

def page_fully_automated_bots(instrument_df):
    """Fully automated bots page with comprehensive paper trading simulation."""
    
    # Display current time and market status first
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    st.warning("🚨 **LIVE TRADING WARNING**: Automated bots will execute real trades with real money! Use at your own risk.", icon="⚠️")
    
    # 🎯 ENHANCED HEADER WITH TICKER THEME
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.subheader("🤖 Automated Trading Bots")
    with col_header2:
        st.caption(f"📅 {current_date}")
    with col_header3:
        # Live ticker-style time display
        if st.session_state.automated_mode.get('live_trading', False):
            st.caption(f"🔴 {current_time}")
        else:
            st.caption(f"🔵 {current_time}")
    
    # Initialize automated mode if not exists
    if 'automated_mode' not in st.session_state:
        initialize_automated_mode()
    else:
        # Ensure paper_portfolio exists (migration for existing sessions)
        if 'paper_portfolio' not in st.session_state.automated_mode:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': st.session_state.automated_mode.get('total_capital', 10000.0),
                'positions': {},
                'initial_capital': st.session_state.automated_mode.get('total_capital', 10000.0),
                'total_value': st.session_state.automated_mode.get('total_capital', 10000.0)
            }
    
    # Fix the total_capital value if it's below minimum
    current_capital = float(st.session_state.automated_mode.get('total_capital', 10000.0))
    if current_capital < 1000.0:
        st.session_state.automated_mode['total_capital'] = 10000.0
    
    # Get market status with error handling
    try:
        market_status, next_market = get_market_status()
    except Exception as e:
        st.error(f"Error getting market status: {e}")
        market_status = "unknown"
        next_market = datetime.now()
    
    # 🎯 ENHANCED MARKET STATUS WITH SEGMENT TIMING
    st.markdown("---")
    
    if market_status == "market_open":
        time_left = datetime.combine(datetime.now().date(), time(15, 20)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.success(f"🟢 **MARKET OPEN** | Equity square-off at 3:20 PM | {minutes_left} minutes | {current_time}")
        
    elif market_status == "equity_square_off":
        time_left = datetime.combine(datetime.now().date(), time(15, 25)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.error(f"🔴 **EQUITY SQUARE-OFF** | Derivatives square-off in {minutes_left} minutes | {current_time}")
        
    elif market_status == "derivatives_square_off":
        time_left = datetime.combine(datetime.now().date(), time(15, 30)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.error(f"🚨 **DERIVATIVES SQUARE-OFF** | Market closes in {minutes_left} minutes | {current_time}")
        
    elif market_status == "pre_market":
        time_left = datetime.combine(datetime.now().date(), time(9, 15)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.info(f"⏰ **PRE-MARKET** | Live trading starts in {minutes_left} minutes | {current_time}")
        
    elif market_status == "market_closed":
        st.info(f"🔴 **MARKET CLOSED** | Live trading available tomorrow at 9:15 AM | {current_time}")
        
    else:  # weekend or unknown
        st.info(f"🎉 **WEEKEND** | Markets closed | Paper trading available | {current_time}")
    
    # Trading status indicators with color themes
    if st.session_state.automated_mode.get('running', False):
        if st.session_state.automated_mode.get('live_trading', False):
            if is_market_hours():
                # Red theme for live trading
                st.error("**🔴 LIVE TRADING ACTIVE** - Real money at risk! Monitor positions carefully.")
            else:
                st.warning("**⏸️ LIVE TRADING PAUSED** - Outside market hours")
        else:
            # Blue theme for paper trading
            st.info("**🔵 PAPER TRADING ACTIVE** - Safe simulation running")
    
    # 🎯 ENHANCED CONTROL PANEL
    st.markdown("---")
    st.subheader("🎮 Control Panel")
    
    # Main control panel with better layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write("**🔧 Mode**")
        auto_enabled = st.toggle(
            "Enable Bots", 
            value=st.session_state.automated_mode.get('enabled', False),
            help="Enable automated trading bots",
            key="auto_enable"
        )
        st.session_state.automated_mode['enabled'] = auto_enabled
    
    with col2:
        st.write("**🎯 Trading Type**")
        is_market_open = is_market_hours()
        
        if is_market_open or is_pre_market_hours():
            live_trading = st.toggle(
                "Live Trading",
                value=st.session_state.automated_mode.get('live_trading', False),
                help="Real money trading (Market hours: 9:15 AM - 3:30 PM)",
                key="live_trading"
            )
        else:
            live_trading = False
            st.session_state.automated_mode['live_trading'] = False
            st.toggle(
                "Live Trading",
                value=False,
                help="❌ Available 9:15 AM - 3:30 PM only",
                key="live_trading_disabled",
                disabled=True
            )
            if market_status == "pre_market":
                st.caption("🕘 Starts 9:15 AM")
            elif market_status == "market_closed":
                st.caption("🕞 Available tomorrow")
            elif market_status == "weekend":
                st.caption("🎉 Markets closed")
        
        st.session_state.automated_mode['live_trading'] = live_trading
    
    with col3:
        st.write("**🚦 Actions**")
        if st.session_state.automated_mode['enabled']:
            if not st.session_state.automated_mode.get('running', False):
                # Start button with color themes
                if live_trading and (is_market_hours() or is_pre_market_hours()):
                    if st.button("🔴 Start Live", use_container_width=True, type="secondary"):
                        st.session_state.need_live_confirmation = True
                        st.rerun()
                elif live_trading and not is_market_hours():
                    st.button("⏸️ Market Closed", use_container_width=True, disabled=True)
                else:
                    if st.button("🔵 Start Paper", use_container_width=True, type="primary"):
                        st.session_state.automated_mode['running'] = True
                        st.success("Paper trading started!")
                        st.rerun()
            else:
                # Stop button
                if st.button("🛑 Stop", use_container_width=True, type="secondary"):
                    st.session_state.automated_mode['running'] = False
                    st.rerun()
        else:
            st.button("Start Trading", use_container_width=True, disabled=True)
    
    with col4:
        st.write("**💰 Capital**")
        current_capital = float(st.session_state.automated_mode.get('total_capital', 10000.0))
        current_capital = max(1000.0, current_capital)
        
        total_capital = st.number_input(
            "Trading Capital (₹)",
            min_value=100.0,
            max_value=1000000.0,
            value=current_capital,
            step=100.0,
            help="Total capital for trading",
            key="auto_capital",
            label_visibility="collapsed"
        )
        st.session_state.automated_mode['total_capital'] = float(total_capital)
    
    with col5:
        st.write("**⚡ Risk**")
        current_risk = float(st.session_state.automated_mode.get('risk_per_trade', 2.0))
        current_risk = max(0.5, min(5.0, current_risk))
        
        risk_per_trade = st.number_input(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=current_risk,
            step=0.5,
            help="Risk percentage per trade",
            key="auto_risk",
            label_visibility="collapsed"
        )
        st.session_state.automated_mode['risk_per_trade'] = float(risk_per_trade)
    
    # Update paper portfolio capital if not running
    if not st.session_state.automated_mode.get('running', False):
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        if not paper_portfolio:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': float(total_capital),
                'positions': {},
                'initial_capital': float(total_capital),
                'total_value': float(total_capital)
            }
        else:
            st.session_state.automated_mode['paper_portfolio']['initial_capital'] = float(total_capital)
            st.session_state.automated_mode['paper_portfolio']['cash_balance'] = float(total_capital)
            st.session_state.automated_mode['paper_portfolio']['total_value'] = float(total_capital)
    
    # Live trading confirmation dialog
    if st.session_state.get('need_live_confirmation', False):
        st.markdown("---")
        st.error("""
        🚨 **LIVE TRADING CONFIRMATION REQUIRED**
        
        **You are about to enable LIVE TRADING with real money!**
        
        **Risks:**
        • Real orders with real money
        • You are responsible for ALL losses
        • Market conditions can change rapidly
        
        **Market Hours:**
        • Live trading: 9:15 AM - 3:30 PM only
        • Auto-stop at market close
        • Square-off by 3:30 PM required
        """)
        
        col_confirm1, col_confirm2, col_confirm3 = st.columns([2, 1, 1])
        
        with col_confirm1:
            if st.button("✅ CONFIRM LIVE TRADING", type="primary", use_container_width=True):
                st.session_state.automated_mode['running'] = True
                st.session_state.automated_mode['live_trading'] = True
                st.session_state.need_live_confirmation = False
                st.session_state.live_trading_start_time = datetime.now().isoformat()
                st.success("🚀 LIVE TRADING ACTIVATED!")
                st.rerun()
        
        with col_confirm2:
            if st.button("📄 PAPER TRADING", use_container_width=True):
                st.session_state.automated_mode['running'] = True
                st.session_state.automated_mode['live_trading'] = False
                st.session_state.need_live_confirmation = False
                st.info("Paper trading started.")
                st.rerun()
                
        with col_confirm3:
            if st.button("❌ CANCEL", use_container_width=True):
                st.session_state.automated_mode['live_trading'] = False
                st.session_state.need_live_confirmation = False
                st.info("Live trading cancelled.")
                st.rerun()
        return
    
    # Auto-stop live trading if market closes
    if (st.session_state.automated_mode.get('running', False) and 
        st.session_state.automated_mode.get('live_trading', False) and 
        not is_market_hours() and not is_pre_market_hours()):
        
        st.session_state.automated_mode['running'] = False
        st.session_state.automated_mode['live_trading'] = False
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.automated_mode['enabled']:
        # 🎯 ENHANCED DASHBOARD LAYOUT WITH NEW FEATURES
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 Bot Configuration", "📊 Live Dashboard", "🔍 Live Thinking", "🎯 Symbol Override"])
        
        with tab1:
            display_bot_configuration_tab()
        
        with tab2:
            # Live performance dashboard
            try:
                display_enhanced_live_dashboard(instrument_df)
            except Exception as e:
                st.error(f"Error displaying dashboard: {e}")
        
        with tab3:
            # 🎯 ENHANCED LIVE THINKING TAB
            try:
                display_enhanced_live_thinking_tab(instrument_df)
            except Exception as e:
                st.error(f"Error in live thinking: {e}")
        
        with tab4:
            # 🎯 NEW SYMBOL OVERRIDE TAB
            try:
                display_symbol_override_tab(instrument_df)
            except Exception as e:
                st.error(f"Error in symbol override: {e}")
    
    else:
        # Setup guide when disabled
        display_setup_guide()

def display_bot_configuration_tab():
    """Display bot configuration tab"""
    st.subheader("⚙️ Bot Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.write("**🤖 Active Bots**")
        for bot_name in AUTOMATED_BOTS.keys():
            is_active = st.session_state.automated_mode.get('bots_active', {}).get(bot_name, False)
            if st.checkbox(bot_name, value=is_active, key=f"auto_{bot_name}"):
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = True
            else:
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = False
        
        st.markdown("---")
        st.write("**📊 Trading Limits**")
        max_trades = st.slider(
            "Max Open Trades",
            min_value=1,
            max_value=20,
            value=st.session_state.automated_mode.get('max_open_trades', 5),
            help="Maximum simultaneous trades",
            key="auto_max_trades"
        )
        st.session_state.automated_mode['max_open_trades'] = max_trades
    
    with col_config2:
        st.write("**⏰ Analysis Frequency**")
        current_interval = st.session_state.automated_mode.get('check_interval', '1 minute')
        frequency_options = ["15 seconds", "30 seconds", "1 minute", "5 minutes", "15 minutes"]
        
        current_index = 2
        if current_interval in frequency_options:
            current_index = frequency_options.index(current_interval)
        
        check_interval = st.selectbox(
            "Analysis Frequency",
            options=frequency_options,
            index=current_index,
            help="How often bots analyze the market",
            key="auto_freq"
        )
        st.session_state.automated_mode['check_interval'] = check_interval
        
        # Frequency warnings
        if check_interval == "15 seconds":
            st.warning("⚡ High frequency - May hit API limits")
        elif check_interval == "30 seconds":
            st.info("🚀 Active trading - Good balance")
        else:
            st.success("🔄 Standard frequency - Stable")
        
        st.markdown("---")
        st.write("**📋 Trading Symbols**")
        active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
        watchlist_symbols = [item['symbol'] for item in st.session_state.watchlists.get(active_watchlist, [])]
        
        if watchlist_symbols:
            st.success(f"Trading from: **{active_watchlist}**")
            with st.expander(f"View {len(watchlist_symbols)} symbols"):
                for symbol in watchlist_symbols:
                    st.write(f"• {symbol}")
        else:
            st.warning("No symbols in active watchlist")

def display_enhanced_live_dashboard(instrument_df):
    """Display enhanced live dashboard"""
    st.info("Live trading dashboard - Real-time performance metrics would display here")

def display_enhanced_live_thinking_tab(instrument_df):
    """Display enhanced live thinking tab"""
    st.info("Live bot thinking analysis would display here")

def display_symbol_override_tab(instrument_df):
    """Display symbol override tab"""
    st.info("Symbol override functionality would be here")

def display_setup_guide():
    """Display setup guide"""
    st.info("Automated trading setup guide would display here")

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
        parents=['NIFTY 50'] * len(full_data),
        values=full_data['size'],
        text=full_data['% Change'].apply(lambda x: f"{x:+.2f}%"),
        textinfo="label+text",
        marker=dict(
            colors=full_data['% Change'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="% Change")
        ),
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{text}<extra></extra>',
        customdata=np.stack((full_data['Price'], full_data['% Change']), axis=-1)
    ))
    
    fig.update_layout(
        title="NIFTY 50 Heatmap",
        height=500,
        margin=dict(t=50, l=25, r=25, b=25),
        template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    )
    
    return fig

def create_sector_allocation_chart(instrument_df):
    """Creates a sector allocation chart based on the SENSEX sector data."""
    sector_df = get_sector_data()
    if sector_df is None or sector_df.empty:
        return go.Figure()
    
    sector_counts = sector_df['Sector'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=sector_counts.index,
        values=sector_counts.values,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="SENSEX Sector Allocation",
        height=400,
        margin=dict(t=50, l=25, r=25, b=25),
        template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    )
    
    return fig

def page_dashboard():
    """Enhanced Dashboard with BMP and visualizations."""
    display_header()
    st.title("🏠 Bharatiya Market Pulse (BMP)")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    
    # Auto-refresh every 60 seconds
    st_autorefresh(interval=60000, key="dashboard_refresh")
    
    # Market status and timing
    market_status, next_market = get_market_status()
    status_colors = {
        "market_open": "🟢", "pre_market": "🟡", "equity_square_off": "🔴", 
        "derivatives_square_off": "🔴", "market_closed": "⚫", "weekend": "⚫", "holiday": "⚫"
    }
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{status_colors.get(market_status, '⚫')} Market Status: {market_status.replace('_', ' ').title()}")
    with col2:
        if market_status in ["market_closed", "weekend", "holiday"]:
            st.caption(f"Next market: {next_market.strftime('%a, %b %d, %H:%M')}")
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main BMP and indices row
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        # BMP Score Card
        try:
            nifty_quote = get_watchlist_data([{'symbol': 'NIFTY 50', 'exchange': 'NSE'}])
            sensex_quote = get_watchlist_data([{'symbol': 'SENSEX', 'exchange': 'BSE'}])
            vix_quote = get_watchlist_data([{'symbol': 'INDIAVIX', 'exchange': 'NSE'}])
            
            nifty_change = nifty_quote.iloc[0]['% Change'] if not nifty_quote.empty else 0
            sensex_change = sensex_quote.iloc[0]['% Change'] if not sensex_quote.empty else 0
            vix_value = vix_quote.iloc[0]['Price'] if not vix_quote.empty else 20
            
            # Create a simple lookback dataframe (in a real app, this would be historical)
            lookback_df = pd.DataFrame({
                'nifty_change': [nifty_change] * 30,
                'sensex_change': [sensex_change] * 30,
                'vix_value': [vix_value] * 30
            })
            
            bmp_score, bmp_label, bmp_color = get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df)
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {bmp_color}22, {bmp_color}44); border-radius: 15px; border: 2px solid {bmp_color};">
                <h1 style="margin: 0; color: {bmp_color}; font-size: 4rem;">{bmp_score:.0f}</h1>
                <h3 style="margin: 0; color: {bmp_color};">{bmp_label}</h3>
                <p style="margin: 0; color: #666;">Bharatiya Market Pulse Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error calculating BMP: {e}")
    
    with col_b:
        # Key indices
        indices_data = [
            {'Ticker': 'NIFTY 50', 'Exchange': 'NSE'},
            {'Ticker': 'BANKNIFTY', 'Exchange': 'NSE'},
            {'Ticker': 'INDIAVIX', 'Exchange': 'NSE'},
            {'Ticker': 'SENSEX', 'Exchange': 'BSE'}
        ]
        
        indices_df = get_watchlist_data(indices_data)
        if not indices_df.empty:
            for _, row in indices_df.iterrows():
                change_color = "green" if row['% Change'] > 0 else "red"
                st.metric(
                    label=row['Ticker'],
                    value=f"₹{row['Price']:.2f}",
                    delta=f"{row['% Change']:+.2f}%",
                    delta_color="normal"
                )
    
    # Visualizations row
    st.markdown("---")
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # NIFTY 50 Heatmap
        heatmap_fig = create_nifty_heatmap(instrument_df)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("NIFTY 50 heatmap data not available.")
    
    with col_viz2:
        # Sector Allocation
        sector_fig = create_sector_allocation_chart(instrument_df)
        if sector_fig:
            st.plotly_chart(sector_fig, use_container_width=True)
        else:
            st.info("Sector allocation data not available.")
    
    # News and Quick Actions
    st.markdown("---")
    col_news, col_actions = st.columns([2, 1])
    
    with col_news:
        st.subheader("📰 Latest Market News")
        news_df = fetch_and_analyze_news()
        if not news_df.empty:
            for _, article in news_df.head(5).iterrows():
                sentiment_icon = "🟢" if article['sentiment'] > 0.1 else "🔴" if article['sentiment'] < -0.1 else "🟡"
                st.markdown(f"""
                **{sentiment_icon} [{article['title']}]({article['link']})**  
                *{article['source']} • {article['date']}*
                """)
        else:
            st.info("No recent news available.")
    
    with col_actions:
        st.subheader("⚡ Quick Actions")
        quick_trade_dialog()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 View Portfolio", use_container_width=True):
            st.session_state.active_page = "Portfolio & Risk"
            st.rerun()

def page_advanced_charting():
    """Advanced charting page with multiple chart types and technical indicators."""
    display_header()
    st.title("📈 Advanced Charting")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use charting features.")
        return
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE', 'NFO'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox("Select Symbol", sorted(all_symbols), key="chart_symbol")
    
    with col2:
        interval = st.selectbox("Interval", ['1d', '5d', '1mo', '6mo', '1y', '5y'], key="chart_interval")
    
    with col3:
        chart_type = st.selectbox("Chart Type", ['Candlestick', 'Heikin-Ashi', 'Line', 'Bar'], key="chart_type")
    
    with col4:
        show_indicators = st.checkbox("Show Indicators", value=True, key="chart_indicators")
    
    if selected_symbol:
        # Determine exchange
        is_option = any(char.isdigit() for char in selected_symbol)
        if is_option:
            exchange = 'NFO'
        else:
            instrument = instrument_df[instrument_df['tradingsymbol'] == selected_symbol]
            exchange = instrument.iloc[0]['exchange'] if not instrument.empty else 'NSE'
        
        token = get_instrument_token(selected_symbol, instrument_df, exchange)
        if token:
            data = get_historical_data(token, interval, period=interval)
            if not data.empty:
                # Create chart
                fig = create_chart(data, selected_symbol, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators interpretation
                if show_indicators:
                    interpretation = interpret_indicators(data)
                    if interpretation:
                        st.subheader("📊 Technical Indicators Interpretation")
                        cols = st.columns(3)
                        for i, (indicator, signal) in enumerate(interpretation.items()):
                            with cols[i % 3]:
                                if "Bullish" in signal:
                                    st.success(f"**{indicator}:** {signal}")
                                elif "Bearish" in signal:
                                    st.error(f"**{indicator}:** {signal}")
                                else:
                                    st.info(f"**{indicator}:** {signal}")
            else:
                st.warning("No historical data available for the selected symbol and period.")
        else:
            st.error("Could not find instrument token for the selected symbol.")
    else:
        st.info("Please select a symbol to view charts.")

def page_premarket_pulse():
    """Premarket analysis page with gap-up/gap-down scanner and premarket movers."""
    display_header()
    st.title("🌅 Premarket Pulse")
    st.info("Analyze premarket trends and identify potential gap-up/gap-down opportunities.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use premarket analysis.")
        return
    
    # Market status check
    market_status, next_market = get_market_status()
    if market_status not in ["pre_market", "market_open"]:
        st.warning(f"Premarket analysis is most useful during pre-market hours (9:00 AM - 9:15 AM). Current status: {market_status.replace('_', ' ').title()}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Gap Analysis Scanner")
        gap_threshold = st.slider("Gap Threshold (%)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        
        if st.button("Scan for Gap Opportunities", use_container_width=True):
            with st.spinner("Scanning for gap-up/gap-down stocks..."):
                # Get NIFTY 50 constituents for scanning
                nifty_constituents = get_nifty50_constituents(instrument_df)
                if not nifty_constituents.empty:
                    gap_opportunities = []
                    symbols_to_scan = nifty_constituents['Symbol'].tolist()[:20]  # Limit for demo
                    
                    for symbol in symbols_to_scan:
                        try:
                            token = get_instrument_token(symbol, instrument_df, 'NSE')
                            if token:
                                # Get yesterday's close and current price
                                hist_data = get_historical_data(token, 'day', period='5d')
                                if not hist_data.empty and len(hist_data) >= 2:
                                    yesterday_close = hist_data['close'].iloc[-2]
                                    current_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                                    if not current_data.empty:
                                        current_price = current_data.iloc[0]['Price']
                                        gap_percentage = ((current_price - yesterday_close) / yesterday_close) * 100
                                        
                                        if abs(gap_percentage) >= gap_threshold:
                                            gap_opportunities.append({
                                                'Symbol': symbol,
                                                'Yesterday Close': yesterday_close,
                                                'Current Price': current_price,
                                                'Gap %': gap_percentage,
                                                'Type': 'Gap Up' if gap_percentage > 0 else 'Gap Down'
                                            })
                        except Exception:
                            continue
                    
                    if gap_opportunities:
                        gap_df = pd.DataFrame(gap_opportunities)
                        st.subheader(f"🎯 Gap Opportunities (≥ {gap_threshold}%)")
                        
                        # Display gap-up and gap-down separately
                        gap_up_df = gap_df[gap_df['Type'] == 'Gap Up'].sort_values('Gap %', ascending=False)
                        gap_down_df = gap_df[gap_df['Type'] == 'Gap Down'].sort_values('Gap %', ascending=True)
                        
                        if not gap_up_df.empty:
                            st.write("**🔼 Gap Up Stocks:**")
                            for _, row in gap_up_df.iterrows():
                                st.success(f"{row['Symbol']}: +{row['Gap %']:.2f}% (₹{row['Yesterday Close']:.2f} → ₹{row['Current Price']:.2f})")
                        
                        if not gap_down_df.empty:
                            st.write("**🔽 Gap Down Stocks:**")
                            for _, row in gap_down_df.iterrows():
                                st.error(f"{row['Symbol']}: {row['Gap %']:.2f}% (₹{row['Yesterday Close']:.2f} → ₹{row['Current Price']:.2f})")
                    else:
                        st.info(f"No gap opportunities found above {gap_threshold}% threshold.")
                else:
                    st.error("Could not fetch NIFTY 50 constituents for scanning.")
    
    with col2:
        st.subheader("⚡ Quick Actions")
        
        # Most active options
        if st.button("Most Active Options", use_container_width=True):
            show_most_active_dialog("NIFTY", instrument_df)
        
        # Global indices
        st.subheader("🌍 Global Indices")
        global_tickers = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "FTSE 100": "^FTSE"}
        global_data = get_global_indices_data(global_tickers)
        if not global_data.empty:
            for _, row in global_data.iterrows():
                st.metric(
                    label=row['Ticker'],
                    value=f"${row['Price']:.2f}",
                    delta=f"{row['% Change']:+.2f}%"
                )
    
    # News sentiment analysis
    st.markdown("---")
    st.subheader("📰 Overnight News Sentiment")
    
    news_query = st.text_input("Filter news by keyword (e.g., RBI, Budget, Results):", placeholder="Enter keyword...")
    
    if st.button("Analyze News Sentiment", use_container_width=True) or news_query:
        with st.spinner("Analyzing financial news..."):
            news_df = fetch_and_analyze_news(news_query)
            if not news_df.empty:
                # Calculate overall sentiment
                avg_sentiment = news_df['sentiment'].mean()
                sentiment_label = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
                sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "orange"
                
                st.metric("Overall News Sentiment", sentiment_label, delta=f"{avg_sentiment:.2f}")
                
                # Display top news
                st.subheader("Top Relevant News")
                for _, article in news_df.head(5).iterrows():
                    with st.expander(f"{article['source']}: {article['title']}"):
                        st.write(f"**Published:** {article['date']}")
                        st.write(f"**Sentiment Score:** {article['sentiment']:.2f}")
                        st.write(f"[Read more]({article['link']})")
            else:
                st.info("No relevant news found for the given keyword.")

def page_fo_analytics():
    """F&O Analytics page with options chain, PCR, and Greeks calculation."""
    display_header()
    st.title("📊 F&O Analytics")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use F&O analytics.")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        underlying_options = ["NIFTY", "BANKNIFTY", "FINNIFTY", "GOLDM", "CRUDEOIL", "USDINR"]
        selected_underlying = st.selectbox("Underlying", underlying_options, key="fo_underlying")
    
    with col2:
        # Get available expiries
        chain_df, selected_expiry, underlying_ltp, available_expiries = get_options_chain(selected_underlying, instrument_df)
        if available_expiries:
            expiry_date = st.selectbox("Expiry", available_expiries, key="fo_expiry")
        else:
            expiry_date = None
            st.warning("No expiries available")
    
    with col3:
        st.metric(f"{selected_underlying} LTP", f"₹{underlying_ltp:.2f}" if underlying_ltp > 0 else "N/A")
    
    if chain_df is not None and not chain_df.empty and expiry_date:
        # Calculate PCR
        total_ce_oi = chain_df['CALL OI'].sum()
        total_pe_oi = chain_df['PUT OI'].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        st.metric("Put-Call Ratio (PCR)", f"{pcr:.2f}")
        
        # PCR interpretation
        if pcr > 1.5:
            st.success("📈 High PCR: Potentially oversold, bullish signal")
        elif pcr < 0.7:
            st.error("📉 Low PCR: Potentially overbought, bearish signal")
        else:
            st.info("⚖️ Normal PCR: Market in balance")
        
        # Display options chain with styling
        st.subheader(f"Options Chain - {selected_underlying} ({expiry_date})")
        styled_chain = style_option_chain(chain_df, underlying_ltp)
        st.dataframe(styled_chain, use_container_width=True, height=400)
        
        # Greeks calculator
        st.markdown("---")
        st.subheader("🧮 Options Greeks Calculator")
        
        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
        
        with col_g1:
            S = st.number_input("Spot Price", value=underlying_ltp, min_value=0.01, step=1.0)
        with col_g2:
            K = st.number_input("Strike Price", value=round(underlying_ltp, -2), min_value=0.01, step=50.0)
        with col_g3:
            T = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
        with col_g4:
            r = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, max_value=15.0, step=0.1) / 100
        with col_g5:
            sigma = st.number_input("Volatility (%)", value=20.0, min_value=1.0, max_value=100.0, step=1.0) / 100
        
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
        
        if st.button("Calculate Greeks", use_container_width=True):
            T_years = T / 365.0
            greeks = black_scholes(S, K, T_years, r, sigma, option_type.lower())
            
            if greeks:
                st.session_state.calculated_greeks = greeks
                st.success("Greeks calculated successfully!")
        
        if st.session_state.calculated_greeks:
            greeks = st.session_state.calculated_greeks
            col_g6, col_g7, col_g8, col_g9, col_g10 = st.columns(5)
            
            with col_g6:
                st.metric("Delta", f"{greeks['delta']:.4f}")
            with col_g7:
                st.metric("Gamma", f"{greeks['gamma']:.4f}")
            with col_g8:
                st.metric("Vega", f"{greeks['vega']:.4f}")
            with col_g9:
                st.metric("Theta", f"{greeks['theta']:.4f}")
            with col_g10:
                st.metric("Rho", f"{greeks['rho']:.4f}")
    
    else:
        st.warning("Could not fetch options chain data. Please check if the market is open.")

def page_forecasting_ml():
    """Machine Learning Forecasting page with Seasonal ARIMA and backtesting."""
    display_header()
    st.title("🤖 ML Forecasting")
    st.info("Predict future price movements using Seasonal ARIMA models with backtesting.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        instrument_options = list(ML_DATA_SOURCES.keys())
        selected_instrument = st.selectbox("Select Instrument", instrument_options, key="ml_instrument")
        
        forecast_steps = st.slider("Forecast Period (Days)", min_value=7, max_value=90, value=30, step=7)
        
        if st.button("Generate Forecast", use_container_width=True):
            with st.spinner("Training model and generating forecast..."):
                # Load and combine data
                combined_data = load_and_combine_data(selected_instrument)
                
                if combined_data.empty:
                    st.error("Could not load data for the selected instrument.")
                else:
                    # Train model and get forecasts
                    forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(combined_data, forecast_steps)
                    
                    if forecast_df is not None:
                        st.session_state.ml_forecast_df = forecast_df
                        st.session_state.ml_backtest_df = backtest_df
                        st.session_state.ml_conf_int_df = conf_int_df
                        st.session_state.ml_instrument_name = selected_instrument
                        st.session_state.ml_historical_data = combined_data
                        st.success("Forecast generated successfully!")
                    else:
                        st.error("Model training failed. Please try with different parameters.")
    
    with col2:
        if st.session_state.ml_forecast_df is not None:
            st.subheader("📊 Forecast Summary")
            
            latest_actual = st.session_state.ml_historical_data['close'].iloc[-1]
            forecast_values = st.session_state.ml_forecast_df['Predicted']
            avg_forecast = forecast_values.mean()
            forecast_change = ((avg_forecast - latest_actual) / latest_actual) * 100
            
            st.metric(
                "Average Forecast Price",
                f"₹{avg_forecast:.2f}",
                delta=f"{forecast_change:+.2f}%"
            )
            
            st.metric("Forecast Period", f"{forecast_steps} days")
            st.metric("Confidence Level", "95%")
    
    # Display charts if forecast exists
    if (st.session_state.ml_forecast_df is not None and 
        st.session_state.ml_historical_data is not None):
        
        st.markdown("---")
        
        # Create forecast chart
        chart_tab, metrics_tab = st.tabs(["📈 Forecast Chart", "📊 Model Metrics"])
        
        with chart_tab:
            fig = create_chart(
                st.session_state.ml_historical_data, 
                st.session_state.ml_instrument_name,
                forecast_df=st.session_state.ml_forecast_df,
                conf_int_df=st.session_state.ml_conf_int_df
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with metrics_tab:
            if st.session_state.ml_backtest_df is not None:
                backtest_df = st.session_state.ml_backtest_df
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(backtest_df['Actual'], backtest_df['Predicted'])
                rmse = np.sqrt(np.mean((backtest_df['Actual'] - backtest_df['Predicted']) ** 2))
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col_m2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col_m3:
                    accuracy = max(0, 100 - mape)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Backtest chart
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Actual'],
                    mode='lines', name='Actual', line=dict(color='blue')
                ))
                fig_backtest.add_trace(go.Scatter(
                    x=backtest_df.index, y=backtest_df['Predicted'],
                    mode='lines', name='Predicted', line=dict(color='red', dash='dash')
                ))
                
                template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
                fig_backtest.update_layout(
                    title="Backtest: Actual vs Predicted",
                    template=template,
                    height=400
                )
                st.plotly_chart(fig_backtest, use_container_width=True)
    
    else:
        st.info("👆 Select an instrument and generate a forecast to see predictions.")

def page_portfolio_and_risk():
    """Portfolio and Risk Management page with real-time positions and risk metrics."""
    display_header()
    st.title("💼 Portfolio & Risk Management")
    
    client = get_broker_client()
    if not client:
        st.info("Please connect to a broker to view your portfolio.")
        return
    
    # Fetch portfolio data
    positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
    with col2:
        st.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{(total_pnl/total_investment*100) if total_investment > 0 else 0:.2f}%")
    with col3:
        open_positions = len(positions_df) if not positions_df.empty else 0
        st.metric("Open Positions", open_positions)
    with col4:
        total_holdings = len(holdings_df) if not holdings_df.empty else 0
        st.metric("Total Holdings", total_holdings)
    
    # Positions and Holdings tabs
    tab1, tab2 = st.tabs(["📈 Positions", "📊 Holdings"])
    
    with tab1:
        if not positions_df.empty:
            st.subheader("Open Positions")
            
            # Calculate additional metrics
            positions_df['Current Value'] = positions_df['quantity'] * positions_df['last_price']
            positions_df['Investment'] = positions_df['quantity'] * positions_df['average_price']
            positions_df['P&L %'] = (positions_df['pnl'] / positions_df['Investment']) * 100
            
            # Display positions with styling
            for _, position in positions_df.iterrows():
                pnl_color = "green" if position['pnl'] > 0 else "red"
                with st.expander(f"{position['tradingsymbol']} | P&L: ₹{position['pnl']:,.2f}"):
                    col_p1, col_p2, col_p3 = st.columns(3)
                    with col_p1:
                        st.write(f"**Quantity:** {position['quantity']}")
                        st.write(f"**Avg Price:** ₹{position['average_price']:.2f}")
                    with col_p2:
                        st.write(f"**LTP:** ₹{position['last_price']:.2f}")
                        st.write(f"**Current Value:** ₹{position['Current Value']:,.2f}")
                    with col_p3:
                        st.write(f"**P&L:** ₹{position['pnl']:,.2f}")
                        st.write(f"**P&L %:** {position['P&L %']:.2f}%")
                    
                    # Quick action buttons
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        if st.button(f"Quick Sell {position['tradingsymbol']}", key=f"sell_{position['tradingsymbol']}", use_container_width=True):
                            quick_trade_dialog(position['tradingsymbol'], 'NSE')
                    with col_a2:
                        if st.button(f"Add to Watchlist", key=f"watch_{position['tradingsymbol']}", use_container_width=True):
                            if position['tradingsymbol'] not in [item['symbol'] for item in st.session_state.watchlists[st.session_state.active_watchlist]]:
                                st.session_state.watchlists[st.session_state.active_watchlist].append({
                                    'symbol': position['tradingsymbol'], 
                                    'exchange': 'NSE'
                                })
                                st.success(f"Added {position['tradingsymbol']} to watchlist")
        else:
            st.info("No open positions found.")
    
    with tab2:
        if not holdings_df.empty:
            st.subheader("Stock Holdings")
            
            # Calculate holding metrics
            holdings_df['Current Value'] = holdings_df['quantity'] * holdings_df['last_price']
            holdings_df['Investment'] = holdings_df['quantity'] * holdings_df['average_price']
            holdings_df['P&L %'] = (holdings_df['pnl'] / holdings_df['Investment']) * 100
            
            # Display holdings
            holdings_display = holdings_df[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'Current Value', 'pnl', 'P&L %']].copy()
            holdings_display.columns = ['Symbol', 'Quantity', 'Avg Price', 'LTP', 'Current Value', 'P&L', 'P&L %']
            
            st.dataframe(holdings_display.style.format({
                'Avg Price': '₹{:.2f}',
                'LTP': '₹{:.2f}',
                'Current Value': '₹{:,.2f}',
                'P&L': '₹{:,.2f}',
                'P&L %': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.info("No holdings found.")
    
    # Risk metrics
    st.markdown("---")
    st.subheader("📊 Risk Metrics")
    
    if not positions_df.empty:
        # Basic risk calculations
        total_exposure = positions_df['Current Value'].sum()
        max_single_exposure = positions_df['Current Value'].max()
        max_single_symbol = positions_df.loc[positions_df['Current Value'].idxmax(), 'tradingsymbol']
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Total Exposure", f"₹{total_exposure:,.2f}")
        with col_r2:
            st.metric("Max Single Exposure", f"₹{max_single_exposure:,.2f}", delta=max_single_symbol)
        with col_r3:
            concentration_ratio = (max_single_exposure / total_exposure * 100) if total_exposure > 0 else 0
            st.metric("Concentration Ratio", f"{concentration_ratio:.1f}%")
    else:
        st.info("No open positions to calculate risk metrics.")

def page_ai_assistant():
    """AI Assistant page with chat interface and trading insights."""
    display_header()
    st.title("🤖 AI Trading Assistant")
    st.info("Get AI-powered insights about your portfolio, market conditions, and trading strategies.", icon="ℹ️")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your AI trading assistant. I can help you with market analysis, portfolio insights, and trading strategies. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about markets, your portfolio, or trading strategies..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Simple rule-based responses (in a real app, this would connect to an AI API)
                response = generate_ai_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("---")
    st.subheader("⚡ Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Portfolio Summary", use_container_width=True):
            portfolio_summary = generate_portfolio_summary()
            st.session_state.messages.append({"role": "user", "content": "Give me a portfolio summary"})
            st.session_state.messages.append({"role": "assistant", "content": portfolio_summary})
            st.rerun()
    
    with col2:
        if st.button("📈 Market Outlook", use_container_width=True):
            market_outlook = generate_market_outlook()
            st.session_state.messages.append({"role": "user", "content": "What's the market outlook?"})
            st.session_state.messages.append({"role": "assistant", "content": market_outlook})
            st.rerun()
    
    with col3:
        if st.button("💡 Trading Ideas", use_container_width=True):
            trading_ideas = generate_trading_ideas()
            st.session_state.messages.append({"role": "user", "content": "Suggest some trading ideas"})
            st.session_state.messages.append({"role": "assistant", "content": trading_ideas})
            st.rerun()

def generate_ai_response(prompt):
    """Generate AI response based on user prompt (rule-based for demo)."""
    prompt_lower = prompt.lower()
    
    # Portfolio-related queries
    if any(word in prompt_lower for word in ['portfolio', 'holding', 'position']):
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        
        if positions_df.empty and holdings_df.empty:
            return "I can see you don't have any open positions or holdings currently. Would you like me to suggest some trading ideas?"
        
        response = "Based on your portfolio:\n\n"
        
        if not positions_df.empty:
            response += "**Open Positions:**\n"
            for _, position in positions_df.iterrows():
                pnl_status = "profit" if position['pnl'] > 0 else "loss"
                response += f"- {position['tradingsymbol']}: ₹{position['pnl']:,.2f} {pnl_status}\n"
        
        if not holdings_df.empty:
            response += "\n**Stock Holdings:**\n"
            for _, holding in holdings_df.head(5).iterrows():
                pnl_status = "profit" if holding['pnl'] > 0 else "loss"
                response += f"- {holding['tradingsymbol']}: ₹{holding['pnl']:,.2f} {pnl_status}\n"
        
        total_pnl_status = "profit" if total_pnl > 0 else "loss"
        response += f"\n**Overall P&L:** ₹{total_pnl:,.2f} {total_pnl_status}"
        
        return response
    
    # Market-related queries
    elif any(word in prompt_lower for word in ['market', 'nifty', 'sensex', 'outlook']):
        return """**Current Market Analysis:**
        
- **NIFTY 50**: Trading in a range with support at 21,800 and resistance at 22,200
- **Sector Rotation**: IT and Pharma showing strength, while FMCG is under pressure
- **Global Cues**: US markets stable, crude prices elevated
- **Recommendation**: Consider defensive stocks in current volatility"""

    # Trading strategy queries
    elif any(word in prompt_lower for word in ['strategy', 'trade', 'idea']):
        return """**Trading Ideas for Today:**
        
1. **RELIANCE** - Breaking out of consolidation, target ₹2,900
2. **HDFCBANK** - Near support, good for swing trade
3. **TATASTEEL** - Momentum building, watch for breakout

*Always use proper risk management and stop losses.*"""

    # Risk management queries
    elif any(word in prompt_lower for word in ['risk', 'stop loss', 'management']):
        return """**Risk Management Guidelines:**
        
- Never risk more than 2% of capital on a single trade
- Always use stop losses (mental or automated)
- Diversify across sectors
- Monitor position sizing carefully
- Review your risk exposure daily"""

    # Default response
    else:
        return """I understand you're asking about trading and markets. I can help you with:

- Portfolio analysis and insights
- Market outlook and trends  
- Trading strategies and ideas
- Risk management guidance
- Technical analysis

Could you please rephrase your question or ask about one of these specific areas?"""

def generate_portfolio_summary():
    """Generate a summary of the user's portfolio."""
    positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
    
    if positions_df.empty and holdings_df.empty:
        return "Your portfolio is currently empty. Consider starting with bluechip stocks for long-term growth."
    
    summary = "**Portfolio Summary:**\n\n"
    
    if total_pnl != 0:
        summary += f"**Total P&L:** ₹{total_pnl:,.2f} ({(total_pnl/total_investment*100) if total_investment > 0 else 0:.2f}%)\n"
    
    if not positions_df.empty:
        winning_positions = positions_df[positions_df['pnl'] > 0]
        losing_positions = positions_df[positions_df['pnl'] < 0]
        
        summary += f"**Open Positions:** {len(positions_df)}\n"
        summary += f"**Winning Trades:** {len(winning_positions)}\n"
        summary += f"**Losing Trades:** {len(losing_positions)}\n"
    
    if not holdings_df.empty:
        summary += f"**Long-term Holdings:** {len(holdings_df)}\n"
    
    summary += "\n**Recommendation:** Consider rebalancing if any single position exceeds 10% of your portfolio value."
    
    return summary

def generate_market_outlook():
    """Generate current market outlook."""
    return """**Market Outlook Today:**
    
- **Short-term (1-2 days)**: Sideways to positive bias
- **Medium-term (1 week)**: Range-bound with stock-specific action
- **Key Levels**: NIFTY support 21,800, resistance 22,200
- **Sectors to Watch**: IT, Banking, Auto
- **Risk Factors**: Global volatility, currency movements

*Trade with caution and proper position sizing.*"""

def generate_trading_ideas():
    """Generate trading ideas based on current market conditions."""
    return """**Trading Ideas for Current Market:**
    
**Momentum Plays:**
1. INFY - Breaking resistance with volume
2. RELIANCE - Consolidation breakout expected

**Value Picks:**
1. HDFCBANK - Near 52-week low, good for accumulation
2. ITC - Stable dividend stock for portfolio

**Swing Trade Ideas:**
1. TATAMOTORS - Strong auto sector momentum
2. BAJFINANCE - Financial services recovery play

*Always conduct your own research and use stop losses.*"""

def page_fundamental_analytics():
    """Fundamental Analytics page with company financials and ratios."""
    display_header()
    st.title("📊 Fundamental Analytics")
    st.info("Analyze company fundamentals, financial ratios, and valuation metrics.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use fundamental analytics.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox("Select Company", sorted(all_symbols), key="fundamental_symbol")
        
        if selected_symbol:
            # Display basic company info (mock data for demo)
            st.subheader("Company Overview")
            
            # Mock fundamental data
            fundamental_data = {
                'Market Cap': '₹10,00,000 Cr',
                'P/E Ratio': '25.3',
                'P/B Ratio': '4.2',
                'ROE': '15.8%',
                'Debt/Equity': '0.45',
                'Dividend Yield': '1.2%'
            }
            
            for metric, value in fundamental_data.items():
                st.metric(metric, value)
    
    with col2:
        if selected_symbol:
            st.subheader("Financial Metrics")
            
            # Create tabs for different fundamental aspects
            tab1, tab2, tab3 = st.tabs(["Valuation", "Profitability", "Leverage"])
            
            with tab1:
                st.write("**Valuation Ratios**")
                valuation_data = {
                    'P/E Ratio': '25.3 (Sector Avg: 22.1)',
                    'P/B Ratio': '4.2 (Sector Avg: 3.8)',
                    'P/S Ratio': '3.1 (Sector Avg: 2.9)',
                    'EV/EBITDA': '12.5 (Sector Avg: 11.2)'
                }
                
                for ratio, value in valuation_data.items():
                    st.write(f"**{ratio}:** {value}")
            
            with tab2:
                st.write("**Profitability Metrics**")
                profitability_data = {
                    'ROE': '15.8% (5-yr Avg: 14.2%)',
                    'ROCE': '18.2% (5-yr Avg: 16.5%)',
                    'Net Margin': '12.3% (5-yr Avg: 11.8%)',
                    'Operating Margin': '18.5% (5-yr Avg: 17.2%)'
                }
                
                for metric, value in profitability_data.items():
                    st.write(f"**{metric}:** {value}")
            
            with tab3:
                st.write("**Leverage & Efficiency**")
                leverage_data = {
                    'Debt/Equity': '0.45 (Sector Avg: 0.60)',
                    'Interest Coverage': '8.2x (Sector Avg: 6.5x)',
                    'Current Ratio': '1.8 (Sector Avg: 1.5)',
                    'Asset Turnover': '1.2 (Sector Avg: 1.1)'
                }
                
                for metric, value in leverage_data.items():
                    st.write(f"**{metric}:** {value}")
            
            # Financial health indicator
            st.markdown("---")
            st.subheader("Financial Health Score")
            
            # Mock health score
            health_score = 78
            health_color = "green" if health_score >= 70 else "orange" if health_score >= 50 else "red"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {health_color}22; border-radius: 10px; border: 2px solid {health_color};">
                <h1 style="margin: 0; color: {health_color};">{health_score}/100</h1>
                <p style="margin: 0; color: #666;">Overall Financial Health</p>
            </div>
            """, unsafe_allow_html=True)
            
            if health_score >= 70:
                st.success("**Strong Fundamentals:** Company shows healthy financial metrics across all parameters.")
            elif health_score >= 50:
                st.warning("**Moderate Fundamentals:** Company has average financial health with some areas for improvement.")
            else:
                st.error("**Weak Fundamentals:** Company shows concerning financial metrics that need attention.")

def page_basket_orders():
    """Basket Orders page for placing multiple orders simultaneously."""
    display_header()
    st.title("🧺 Basket Orders")
    st.info("Place multiple orders across different symbols in a single execution.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use basket orders.")
        return
    
    # Basket management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Add to Basket")
        
        # Symbol selection
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox("Symbol", sorted(all_symbols), key="basket_symbol")
        
        # Order details
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            transaction_type = st.radio("Action", ["BUY", "SELL"], key="basket_action", horizontal=True)
        with col_b:
            product_type = st.radio("Product", ["MIS", "CNC"], key="basket_product", horizontal=True)
        with col_c:
            order_type = st.radio("Type", ["MARKET", "LIMIT"], key="basket_order_type", horizontal=True)
        with col_d:
            quantity = st.number_input("Quantity", min_value=1, value=1, key="basket_quantity")
        
        price = None
        if order_type == "LIMIT":
            price = st.number_input("Price", min_value=0.01, value=0.0, key="basket_price")
        
        if st.button("Add to Basket", use_container_width=True):
            if selected_symbol and quantity > 0:
                basket_item = {
                    'symbol': selected_symbol,
                    'transaction_type': transaction_type,
                    'product': product_type,
                    'order_type': order_type,
                    'quantity': quantity,
                    'price': price
                }
                
                if 'basket' not in st.session_state:
                    st.session_state.basket = []
                
                st.session_state.basket.append(basket_item)
                st.success(f"Added {transaction_type} {quantity} {selected_symbol} to basket")
    
    with col2:
        st.subheader("Basket Summary")
        
        if 'basket' in st.session_state and st.session_state.basket:
            total_orders = len(st.session_state.basket)
            buy_orders = len([item for item in st.session_state.basket if item['transaction_type'] == 'BUY'])
            sell_orders = total_orders - buy_orders
            
            st.metric("Total Orders", total_orders)
            st.metric("Buy Orders", buy_orders)
            st.metric("Sell Orders", sell_orders)
            
            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()
        else:
            st.info("Basket is empty")
    
    # Display current basket
    st.markdown("---")
    st.subheader("Current Basket")
    
    if 'basket' in st.session_state and st.session_state.basket:
        for i, item in enumerate(st.session_state.basket):
            with st.expander(f"{item['transaction_type']} {item['quantity']} {item['symbol']} ({item['product']})"):
                col_i1, col_i2, col_i3 = st.columns([2, 1, 1])
                
                with col_i1:
                    st.write(f"**Symbol:** {item['symbol']}")
                    st.write(f"**Type:** {item['order_type']}")
                with col_i2:
                    st.write(f"**Quantity:** {item['quantity']}")
                    if item['price']:
                        st.write(f"**Price:** ₹{item['price']:.2f}")
                with col_i3:
                    if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                        st.session_state.basket.pop(i)
                        st.rerun()
        
        # Execute basket
        st.markdown("---")
        st.subheader("Execute Basket")
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            if st.button("🚀 Execute All Orders", type="primary", use_container_width=True):
                execute_basket_order(st.session_state.basket, instrument_df)
        
        with col_e2:
            if st.button("📋 Export Orders", use_container_width=True):
                # Create CSV of basket orders
                basket_df = pd.DataFrame(st.session_state.basket)
                csv = basket_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"basket_orders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("Your basket is empty. Add some orders to get started.")

def page_algo_strategy_maker():
    """Algo Strategy Maker page for building and testing trading strategies."""
    display_header()
    st.title("⚡ Algo Strategy Maker")
    st.info("Build, test, and optimize your algorithmic trading strategies.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use strategy maker.")
        return
    
    # Strategy builder
    st.subheader("Strategy Builder")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        strategy_name = st.text_input("Strategy Name", placeholder="e.g., Momentum Breakout")
        
        # Entry conditions
        st.write("**Entry Conditions**")
        entry_condition = st.selectbox(
            "When to Enter",
            ["Price crosses above EMA", "RSI oversold", "MACD crossover", "Bollinger Band squeeze"]
        )
        
        # Parameters
        st.write("**Parameters**")
        param1 = st.number_input("Parameter 1", value=20, help="e.g., EMA period")
        param2 = st.number_input("Parameter 2", value=14, help="e.g., RSI period")
    
    with col2:
        # Exit conditions
        st.write("**Exit Conditions**")
        exit_condition = st.selectbox(
            "When to Exit",
            ["Price crosses below EMA", "RSI overbought", "Fixed target", "Trailing stop loss"]
        )
        
        # Risk management
        st.write("**Risk Management**")
        stop_loss = st.number_input("Stop Loss (%)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
        target = st.number_input("Target (%)", value=4.0, min_value=0.1, max_value=20.0, step=0.1)
    
    # Strategy testing
    st.markdown("---")
    st.subheader("Strategy Tester")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        test_symbol = st.selectbox("Test Symbol", ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY'], key="test_symbol")
    with col4:
        test_period = st.selectbox("Test Period", ['1 month', '3 months', '6 months', '1 year'], key="test_period")
    with col5:
        initial_capital = st.number_input("Initial Capital (₹)", value=100000, min_value=1000, step=1000)
    
    if st.button("Backtest Strategy", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            # Mock backtest results
            backtest_results = {
                'total_trades': 45,
                'winning_trades': 28,
                'losing_trades': 17,
                'win_rate': 62.2,
                'total_pnl': 12500,
                'max_drawdown': -8.5,
                'sharpe_ratio': 1.8
            }
            
            st.session_state.backtest_results = backtest_results
            st.success("Backtest completed!")
    
    # Display backtest results
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        
        st.markdown("---")
        st.subheader("Backtest Results")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            st.metric("Total Trades", results['total_trades'])
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
        with col_r2:
            st.metric("Winning Trades", results['winning_trades'])
            st.metric("Losing Trades", results['losing_trades'])
        with col_r3:
            st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
            st.metric("Return %", f"{(results['total_pnl']/initial_capital*100):.1f}%")
        with col_r4:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        # Strategy evaluation
        st.markdown("---")
        st.subheader("Strategy Evaluation")
        
        if results['win_rate'] > 60 and results['sharpe_ratio'] > 1.5:
            st.success("**Excellent Strategy:** High win rate with good risk-adjusted returns. Consider deploying with real capital.")
        elif results['win_rate'] > 50 and results['sharpe_ratio'] > 1.0:
            st.warning("**Good Strategy:** Moderate performance. May need further optimization.")
        else:
            st.error("**Needs Improvement:** Strategy performance below acceptable levels. Review parameters and conditions.")

def page_momentum_and_trend_finder():
    """Momentum and Trend Finder page for identifying strong trending stocks."""
    display_header()
    st.title("📊 Momentum & Trend Finder")
    st.info("Scan for stocks with strong momentum and trending behavior across different timeframes.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use momentum scanner.")
        return
    
    # Scanner configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        scanner_type = st.selectbox(
            "Scanner Type",
            [
                "Strong Momentum (All Timeframes)",
                "Daily Breakouts", 
                "Weekly Strength",
                "RSI Oversold/Oversold",
                "Volume Breakouts",
                "New 52-week Highs/Lows"
            ],
            key="momentum_scanner"
        )
    
    with col2:
        min_volume = st.number_input("Min Volume (Lakhs)", value=1.0, min_value=0.1, step=0.5)
    
    with col3:
        min_price = st.number_input("Min Price (₹)", value=50.0, min_value=1.0, step=10.0)
    
    # Additional filters
    col4, col5 = st.columns(2)
    
    with col4:
        rsi_range = st.slider("RSI Range", 0, 100, (30, 70))
    
    with col5:
        change_threshold = st.slider("Min Price Change %", 0.0, 10.0, 2.0, 0.5)
    
    if st.button("Run Momentum Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning for momentum opportunities..."):
            # Get NIFTY 50 constituents for scanning
            nifty_constituents = get_nifty50_constituents(instrument_df)
            
            if not nifty_constituents.empty:
                momentum_stocks = []
                symbols_to_scan = nifty_constituents['Symbol'].tolist()[:15]  # Limit for demo
                
                for symbol in symbols_to_scan:
                    try:
                        token = get_instrument_token(symbol, instrument_df, 'NSE')
                        if token:
                            # Get historical data for analysis
                            hist_data = get_historical_data(token, 'day', period='1mo')
                            if not hist_data.empty and len(hist_data) >= 20:
                                current_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                                
                                if not current_data.empty:
                                    current_price = current_data.iloc[0]['Price']
                                    current_change = current_data.iloc[0]['% Change']
                                    
                                    # Skip if below minimum price
                                    if current_price < min_price:
                                        continue
                                    
                                    # Calculate RSI using TA-Lib
                                    rsi = talib.RSI(hist_data['close'], timeperiod=14).iloc[-1]
                                    
                                    # Skip if RSI outside range
                                    if not (rsi_range[0] <= rsi <= rsi_range[1]):
                                        continue
                                    
                                    # Calculate momentum score based on multiple factors
                                    momentum_score = 0
                                    signals = []
                                    
                                    # Price change momentum
                                    if abs(current_change) >= change_threshold:
                                        momentum_score += 30
                                        signals.append(f"Price change: {current_change:+.2f}%")
                                    
                                    # RSI momentum
                                    if rsi > 70:
                                        momentum_score += 20
                                        signals.append("RSI Overbought")
                                    elif rsi < 30:
                                        momentum_score += 20
                                        signals.append("RSI Oversold")
                                    elif 40 <= rsi <= 60:
                                        momentum_score += 10
                                        signals.append("RSI Neutral")
                                    
                                    # Volume check (mock)
                                    if 'volume' in hist_data.columns:
                                        avg_volume = hist_data['volume'].mean()
                                        current_volume = hist_data['volume'].iloc[-1] if len(hist_data) > 0 else 0
                                        if current_volume > avg_volume * 1.5:
                                            momentum_score += 20
                                            signals.append("High Volume")
                                    
                                    # Trend analysis
                                    if len(hist_data) >= 50:
                                        sma_20 = talib.SMA(hist_data['close'], timeperiod=20).iloc[-1]
                                        sma_50 = talib.SMA(hist_data['close'], timeperiod=50).iloc[-1]
                                        
                                        if current_price > sma_20 > sma_50:
                                            momentum_score += 20
                                            signals.append("Strong Uptrend")
                                        elif current_price < sma_20 < sma_50:
                                            momentum_score += 10
                                            signals.append("Strong Downtrend")
                                    
                                    if momentum_score >= 40:  # Minimum threshold
                                        momentum_stocks.append({
                                            'Symbol': symbol,
                                            'Price': current_price,
                                            'Change %': current_change,
                                            'RSI': rsi,
                                            'Momentum Score': momentum_score,
                                            'Signals': ', '.join(signals)
                                        })
                    except Exception as e:
                        continue
                
                if momentum_stocks:
                    # Create and display results dataframe
                    results_df = pd.DataFrame(momentum_stocks)
                    results_df = results_df.sort_values('Momentum Score', ascending=False)
                    
                    st.subheader(f"🎯 Momentum Opportunities ({len(results_df)} found)")
                    
                    # Display results with styling
                    for _, stock in results_df.iterrows():
                        score_color = "green" if stock['Momentum Score'] >= 70 else "orange" if stock['Momentum Score'] >= 50 else "red"
                        
                        with st.expander(f"{stock['Symbol']} | Score: {stock['Momentum Score']} | RSI: {stock['RSI']:.1f}"):
                            col_s1, col_s2 = st.columns(2)
                            
                            with col_s1:
                                st.metric("Current Price", f"₹{stock['Price']:.2f}")
                                st.metric("Price Change", f"{stock['Change %']:+.2f}%")
                            
                            with col_s2:
                                st.metric("RSI", f"{stock['RSI']:.1f}")
                                st.markdown(f'<div style="color: {score_color}; font-weight: bold;">Momentum Score: {stock["Momentum Score"]}</div>', unsafe_allow_html=True)
                            
                            st.write(f"**Signals:** {stock['Signals']}")
                            
                            # Quick actions
                            col_a1, col_a2 = st.columns(2)
                            with col_a1:
                                if st.button(f"Quick Trade {stock['Symbol']}", key=f"trade_{stock['Symbol']}", use_container_width=True):
                                    quick_trade_dialog(stock['Symbol'], 'NSE')
                            with col_a2:
                                if st.button(f"Add to Watchlist", key=f"watch_{stock['Symbol']}", use_container_width=True):
                                    if stock['Symbol'] not in [item['symbol'] for item in st.session_state.watchlists[st.session_state.active_watchlist]]:
                                        st.session_state.watchlists[st.session_state.active_watchlist].append({
                                            'symbol': stock['Symbol'], 
                                            'exchange': 'NSE'
                                        })
                                        st.success(f"Added {stock['Symbol']} to watchlist")
                else:
                    st.info("No momentum opportunities found with the current filters. Try adjusting the parameters.")
            else:
                st.error("Could not fetch stock list for scanning.")
    
    # Quick actions section
    st.markdown("---")
    
    if 'momentum_stocks' in locals() and momentum_stocks:
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("📋 Export Results", use_container_width=True):
                # Create CSV of results
                results_df = pd.DataFrame(momentum_stocks)
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"momentum_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_act2:
            if st.button("👀 Add Top 5 to Watchlist", use_container_width=True):
                added = 0
                for stock in momentum_stocks[:5]:
                    if stock['Symbol'] not in [item['symbol'] for item in st.session_state.watchlists[st.session_state.active_watchlist]]:
                        st.session_state.watchlists[st.session_state.active_watchlist].append({
                            'symbol': stock['Symbol'], 
                            'exchange': 'NSE'
                        })
                        added += 1
                
                if added > 0:
                    st.success(f"Added {added} stocks to watchlist")
                else:
                    st.info("No new stocks to add. They are already in the watchlist.")

# ================ 6. AUTHENTICATION & SETUP ================

def setup_zerodha_2fa():
    """Setup 2FA for Zerodha using pyotp."""
    if st.session_state.pyotp_secret is None:
        st.session_state.pyotp_secret = pyotp.random_base32()
    
    totp = pyotp.TOTP(st.session_state.pyotp_secret)
    provisioning_uri = totp.provisioning_uri(
        name="blockvista@trader", 
        issuer_name="BlockVista Terminal"
    )
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL Image to bytes
    img_buffer = io.BytesIO()
    qr_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Display QR code
    st.subheader("🔐 Zerodha 2FA Setup")
    st.image(img_buffer, caption='Scan this QR code with Google Authenticator', use_column_width=True)
    
    st.info("""
    **Setup Instructions:**
    1. Download Google Authenticator on your phone
    2. Scan the QR code above
    3. Enter the 6-digit code from the app below to verify
    """)
    
    verification_code = st.text_input("Enter verification code from Authenticator app:", max_chars=6)
    
    if st.button("Verify 2FA Setup"):
        if verification_code and totp.verify(verification_code):
            st.session_state.two_factor_setup_complete = True
            st.session_state.show_qr_dialog = False
            st.success("✅ 2FA setup completed successfully!")
            st.rerun()
        else:
            st.error("❌ Invalid verification code. Please try again.")

def zerodha_login_flow():
    """Handles Zerodha login with API key and 2FA."""
    st.subheader("🔐 Zerodha Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("API Key", type="password", help="Get this from Kite Connect developer console")
        api_secret = st.text_input("API Secret", type="password", help="Get this from Kite Connect developer console")
    
    with col2:
        if st.session_state.two_factor_setup_complete and st.session_state.pyotp_secret:
            totp = pyotp.TOTP(st.session_state.pyotp_secret)
            twofa_code = totp.now()
            st.text_input("2FA Code", value=twofa_code, disabled=True, help="Auto-generated from your authenticator")
        else:
            twofa_code = st.text_input("2FA Code", max_chars=6, help="Enter code from your authenticator app or SMS")
    
    request_token = st.text_input("Request Token", help="Get this from the Kite Connect login URL after authorizing")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("Setup 2FA", use_container_width=True):
            st.session_state.show_qr_dialog = True
            st.rerun()
    
    with col4:
        if st.button("Login", type="primary", use_container_width=True):
            if not all([api_key, api_secret, twofa_code, request_token]):
                st.error("Please fill all fields")
                return
            
            try:
                kite = KiteConnect(api_key=api_key)
                
                # Generate session
                data = kite.generate_session(request_token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                
                # Store in session state
                st.session_state.kite = kite
                st.session_state.broker = "Zerodha"
                st.session_state.profile = data["user_name"]
                st.session_state.authenticated = True
                
                st.success(f"✅ Successfully logged in as {data['user_name']}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Login failed: {str(e)}")

def broker_selection_page():
    """Broker selection and authentication page."""
    display_header()
    st.title("🔐 Broker Authentication")
    
    if not st.session_state.authenticated:
        # Broker selection
        st.subheader("Select Your Broker")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Zerodha", use_container_width=True):
                st.session_state.broker = "Zerodha"
                st.rerun()
        
        with col2:
            if st.button("Angel One", use_container_width=True, disabled=True):
                st.info("Coming soon")
        
        with col3:
            if st.button("Fyers", use_container_width=True, disabled=True):
                st.info("Coming soon")
        
        # Broker-specific login
        if st.session_state.broker == "Zerodha":
            if st.session_state.show_qr_dialog:
                setup_zerodha_2fa()
            else:
                zerodha_login_flow()
        
        # Demo mode option
        st.markdown("---")
        st.subheader("🚀 Try Demo Mode")
        
        if st.button("Enter Demo Mode", use_container_width=True, type="secondary"):
            st.session_state.authenticated = True
            st.session_state.broker = "Demo"
            st.session_state.profile = "Demo User"
            st.success("Entering demo mode with sample data")
            st.rerun()
    
    else:
        st.success(f"✅ Already authenticated with {st.session_state.broker}")
        
        if st.button("Switch Broker", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.broker = None
            st.session_state.kite = None
            st.rerun()

# ================ 7. MAIN APPLICATION ================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom styling
    apply_custom_styling()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🚀 Navigation")
        
        if not st.session_state.authenticated:
            st.warning("Please authenticate with your broker")
            selected_page = "Broker Authentication"
        else:
            # Page selection
            pages = {
                "🏠 Dashboard": page_dashboard,
                "📈 Advanced Charting": page_advanced_charting,
                "🌅 Premarket Pulse": page_premarket_pulse,
                "📊 F&O Analytics": page_fo_analytics,
                "🤖 ML Forecasting": page_forecasting_ml,
                "💼 Portfolio & Risk": page_portfolio_and_risk,
                "🤖 AI Assistant": page_ai_assistant,
                "📊 Fundamental Analytics": page_fundamental_analytics,
                "🧺 Basket Orders": page_basket_orders,
                "⚡ Algo Strategy Maker": page_algo_strategy_maker,
                "📊 Momentum & Trend Finder": page_momentum_and_trend_finder,
                "🤖 Algo Bots": page_algo_bots
            }
            
            selected_page = st.selectbox("Go to", list(pages.keys()))
        
        # User profile section
        st.markdown("---")
        if st.session_state.authenticated:
            st.success(f"✅ {st.session_state.profile}")
            st.caption(f"Broker: {st.session_state.broker}")
            
            if st.button("Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.broker = None
                st.session_state.kite = None
                st.session_state.profile = None
                st.rerun()
        else:
            st.info("🔒 Not authenticated")
        
        # Theme selector
        st.markdown("---")
        theme = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_selector")
        st.session_state.theme = theme
        
        # Quick actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        quick_trade_dialog()
        
        if st.session_state.authenticated:
            # Watchlist management
            st.subheader("📋 Watchlists")
            
            # Active watchlist selector
            active_watchlist = st.selectbox(
                "Active Watchlist",
                list(st.session_state.watchlists.keys()),
                key="active_watchlist_selector"
            )
            st.session_state.active_watchlist = active_watchlist
            
            # Display active watchlist
            watchlist_data = get_watchlist_data(st.session_state.watchlists[active_watchlist])
            if not watchlist_data.empty:
                for _, row in watchlist_data.iterrows():
                    change_color = "green" if row['% Change'] > 0 else "red"
                    st.markdown(f"""
                    **{row['Ticker']}**  
                    ₹{row['Price']:.2f}  
                    <span style='color: {change_color};'>{row['% Change']:+.2f}%</span>
                    """, unsafe_allow_html=True)
            else:
                st.info("Watchlist is empty")
            
            # Add symbol to watchlist
            with st.expander("Add Symbol"):
                all_symbols = get_instrument_df()
                if not all_symbols.empty:
                    nse_symbols = all_symbols[all_symbols['exchange'] == 'NSE']['tradingsymbol'].unique()
                    new_symbol = st.selectbox("Symbol", sorted(nse_symbols), key="add_symbol")
                    
                    if st.button("Add to Watchlist"):
                        if new_symbol not in [item['symbol'] for item in st.session_state.watchlists[active_watchlist]]:
                            st.session_state.watchlists[active_watchlist].append({
                                'symbol': new_symbol, 
                                'exchange': 'NSE'
                            })
                            st.success(f"Added {new_symbol} to {active_watchlist}")
                            st.rerun()
    
    # Display selected page
    if not st.session_state.authenticated:
        broker_selection_page()
    else:
        pages = {
            "🏠 Dashboard": page_dashboard,
            "📈 Advanced Charting": page_advanced_charting,
            "🌅 Premarket Pulse": page_premarket_pulse,
            "📊 F&O Analytics": page_fo_analytics,
            "🤖 ML Forecasting": page_forecasting_ml,
            "💼 Portfolio & Risk": page_portfolio_and_risk,
            "🤖 AI Assistant": page_ai_assistant,
            "📊 Fundamental Analytics": page_fundamental_analytics,
            "🧺 Basket Orders": page_basket_orders,
            "⚡ Algo Strategy Maker": page_algo_strategy_maker,
            "📊 Momentum & Trend Finder": page_momentum_and_trend_finder,
            "🤖 Algo Bots": page_algo_bots
        }
        
        if selected_page in pages:
            pages[selected_page]()
        else:
            st.error("Page not found")

if __name__ == "__main__":
    main()
