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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

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
            border-left-color: #ffcc00;
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
    if 'ml_model_type' not in st.session_state: st.session_state.ml_model_type = None
    if 'ml_metrics' not in st.session_state: st.session_state.ml_metrics = {}
    if 'ml_message' not in st.session_state: st.session_state.ml_message = ""
    if 'ml_previous_models' not in st.session_state: st.session_state.ml_previous_models = []

    # HFT Mode
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []
    
    # Fundamental Analysis
    if 'fundamental_data' not in st.session_state: st.session_state.fundamental_data = {}
    if 'comparison_symbols' not in st.session_state: st.session_state.comparison_symbols = []

# ================ 2. HELPER FUNCTIONS ================

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

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None, comparison_data=None):
    """Generates a Plotly chart with various chart types and overlays."""
    fig = go.Figure()
    if df.empty and not comparison_data: return fig
    
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
            fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name=f'{ticker} - Heikin-Ashi'))
        elif chart_type == 'Line':
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name=f'{ticker} - Line'))
        elif chart_type == 'Bar':
            fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name=f'{ticker} - Bar'))
        else:
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name=f'{ticker} - Candlestick'))
            
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
    
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        if conf_int_df is not None:
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    chart_title = f'{ticker} Price Chart ({chart_type})' if not comparison_data else f'Price Comparison: {ticker} vs {", ".join(comparison_data.keys())}'
    fig.update_layout(title=chart_title, yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
            
            # Add technical indicators silently
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
        depth = client.depth(instrument_token)
        return depth.get(str(instrument_token))  # Zerodha returns a dict with token as string key
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
            for entry in feed.entries[:10]: # Limit to 10 articles per source
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
        
        # Backtesting: Create predictions for historical data
        fitted_values = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values}).dropna()
        
        # Forecasting: Predict future values
        forecast_result = model.get_forecast(steps=forecast_steps)
        forecast_adjusted = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05) # 95% confidence interval

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
                live_df.index = live_df.index.tz_convert(None) # Make timezone-naive
                live_df.columns = [col.lower() for col in live_df.columns]
    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max")
            if not live_df.empty: 
                live_df.index = live_df.index.tz_localize(None) # Make timezone-naive
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

# ================ 4. ADVANCED ML MODELS ================

def create_lstm_model(sequence_length, n_features):
    """Creates an LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def prepare_lstm_data(data, sequence_length=60):
    """Prepares data for LSTM training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def train_lstm_model(data, forecast_steps=180):
    """Trains LSTM model and generates forecasts."""
    try:
        # Use only closing prices for LSTM
        close_prices = data['close'].values.reshape(-1, 1)
        
        # Prepare data
        sequence_length = min(60, len(close_prices) // 3)
        if len(close_prices) < sequence_length + forecast_steps:
            return None, None, None, "Insufficient data for LSTM training"
        
        X, y, scaler = prepare_lstm_data(close_prices, sequence_length)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        model = create_lstm_model(sequence_length, 1)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate forecasts
        last_sequence = close_prices[-sequence_length:]
        last_sequence_scaled = scaler.transform(last_sequence)
        
        forecasts = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(forecast_steps):
            X_pred = current_sequence.reshape(1, sequence_length, 1)
            pred = model.predict(X_pred, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Inverse transform forecasts
        forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
        
        # Create forecast dataframe
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame({'Predicted': forecasts.flatten()}, index=future_dates)
        
        # Generate predictions for historical data
        y_pred = model.predict(X, verbose=0)
        y_pred = scaler.inverse_transform(y_pred)
        
        # Create backtest dataframe
        backtest_dates = data.index[sequence_length:sequence_length + len(y_pred)]
        backtest_df = pd.DataFrame({
            'Actual': data['close'].iloc[sequence_length:sequence_length + len(y_pred)],
            'Predicted': y_pred.flatten()
        }, index=backtest_dates)
        
        return forecast_df, backtest_df, None, "LSTM model trained successfully"
        
    except Exception as e:
        return None, None, None, f"LSTM training failed: {str(e)}"

def train_xgboost_model(data, forecast_steps=180):
    """Trains XGBoost model and generates forecasts."""
    try:
        # Create features
        df = data.copy()
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_of_week'] = df.index.dayofweek
        
        # Create lag features
        for lag in [1, 5, 10, 20]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Create rolling statistics
        df['close_rolling_mean_7'] = df['close'].rolling(window=7).mean()
        df['close_rolling_std_7'] = df['close'].rolling(window=7).std()
        df['close_rolling_mean_30'] = df['close'].rolling(window=30).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            return None, None, None, "Insufficient data for XGBoost training"
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'close']
        X = df[feature_columns]
        y = df['close']
        
        # Split data
        split_idx = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Generate forecasts
        forecasts = []
        last_row = df.iloc[-1:].copy()
        
        for i in range(forecast_steps):
            # Prepare features for next prediction
            next_date = data.index[-1] + timedelta(days=i+1)
            features = last_row[feature_columns].copy()
            
            # Update date features
            features['day'] = next_date.day
            features['month'] = next_date.month
            features['year'] = next_date.year
            features['day_of_week'] = next_date.dayofweek
            
            # Update lag features
            current_pred = forecasts[-1] if forecasts else data['close'].iloc[-1]
            for lag in [1, 5, 10, 20]:
                if lag == 1:
                    features[f'close_lag_{lag}'] = current_pred
                else:
                    # For longer lags, we'd need to maintain a history
                    # This is simplified - in production you'd maintain proper lag history
                    features[f'close_lag_{lag}'] = data['close'].iloc[-lag] if len(data) > lag else current_pred
            
            # Make prediction
            pred = model.predict(features)[0]
            forecasts.append(pred)
        
        # Create forecast dataframe
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame({'Predicted': forecasts}, index=future_dates)
        
        # Generate backtest predictions
        y_pred = model.predict(X)
        backtest_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred
        }, index=df.index)
        
        return forecast_df, backtest_df, None, "XGBoost model trained successfully"
        
    except Exception as e:
        return None, None, None, f"XGBoost training failed: {str(e)}"

def calculate_model_metrics(actual, predicted):
    """Calculate comprehensive model performance metrics."""
    metrics = {}
    
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return metrics
    
    # Basic metrics
    metrics['MAE'] = mean_absolute_error(actual_clean, predicted_clean)
    metrics['RMSE'] = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    metrics['MAPE'] = mean_absolute_percentage_error(actual_clean, predicted_clean)
    
    # Direction accuracy
    actual_dir = np.diff(actual_clean) > 0
    predicted_dir = np.diff(predicted_clean) > 0
    if len(actual_dir) > 0:
        metrics['Direction_Accuracy'] = np.mean(actual_dir == predicted_dir) * 100
    
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    if ss_tot != 0:
        metrics['R2'] = 1 - (ss_res / ss_tot)
    
    return metrics

def compare_ml_models(data, forecast_steps=180):
    """Compare multiple ML models and return their performance."""
    models = {}
    
    # Train ARIMA
    arima_forecast, arima_backtest, arima_conf_int = train_seasonal_arima_model(data, forecast_steps)
    if arima_backtest is not None:
        arima_metrics = calculate_model_metrics(arima_backtest['Actual'], arima_backtest['Predicted'])
        models['ARIMA'] = {
            'forecast': arima_forecast,
            'backtest': arima_backtest,
            'metrics': arima_metrics,
            'color': '#1f77b4'
        }
    
    # Train LSTM
    lstm_forecast, lstm_backtest, _, lstm_message = train_lstm_model(data, forecast_steps)
    if lstm_backtest is not None:
        lstm_metrics = calculate_model_metrics(lstm_backtest['Actual'], lstm_backtest['Predicted'])
        models['LSTM'] = {
            'forecast': lstm_forecast,
            'backtest': lstm_backtest,
            'metrics': lstm_metrics,
            'color': '#2ca02c'
        }
    
    # Train XGBoost
    xgb_forecast, xgb_backtest, _, xgb_message = train_xgboost_model(data, forecast_steps)
    if xgb_backtest is not None:
        xgb_metrics = calculate_model_metrics(xgb_backtest['Actual'], xgb_backtest['Predicted'])
        models['XGBoost'] = {
            'forecast': xgb_forecast,
            'backtest': xgb_backtest,
            'metrics': xgb_metrics,
            'color': '#d62728'
        }
    
    return models

def create_model_comparison_chart(data, models):
    """Create a comparison chart for multiple models."""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=data.index, y=data['close'],
        mode='lines', name='Historical',
        line=dict(color='black', width=2)
    ))
    
    # Add model forecasts
    for model_name, model_data in models.items():
        if model_data['forecast'] is not None:
            fig.add_trace(go.Scatter(
                x=model_data['forecast'].index,
                y=model_data['forecast']['Predicted'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=model_data['color'], dash='dash')
            ))
    
    fig.update_layout(
        title="Model Comparison - Price Forecasts",
        xaxis_title="Date",
        yaxis_title="Price",
        template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    )
    
    return fig

# ================ 5. FUNDAMENTAL ANALYSIS FUNCTIONS ================

@st.cache_data(ttl=3600)
def get_fundamental_data(symbol):
    """Fetches comprehensive fundamental data for a company."""
    try:
        # Add .NS suffix for NSE stocks
        ticker_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # Get financial statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cash_flow
        
        # Calculate key metrics
        fundamental_data = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'roe': calculate_roe(income_stmt, balance_sheet),
            'roa': calculate_roa(income_stmt, balance_sheet),
            'debt_to_equity': calculate_debt_to_equity(balance_sheet),
            'current_ratio': calculate_current_ratio(balance_sheet),
            'profit_margin': calculate_profit_margin(income_stmt),
            'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
            'revenue_growth': calculate_revenue_growth(income_stmt),
            'profit_growth': calculate_profit_growth(income_stmt),
            'free_cash_flow': calculate_free_cash_flow(cash_flow),
            'shares_outstanding': info.get('sharesOutstanding', 0),
            'book_value': info.get('bookValue', 0),
            'eps': info.get('trailingEps', 0),
            'beta': info.get('beta', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0)
        }
        
        return fundamental_data
    except Exception as e:
        st.error(f"Error fetching fundamental data for {symbol}: {e}")
        return None

def calculate_roe(income_stmt, balance_sheet):
    """Calculate Return on Equity."""
    try:
        if not income_stmt.empty and not balance_sheet.empty:
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            return (net_income / equity) * 100 if equity != 0 else 0
    except:
        pass
    return 0

def calculate_roa(income_stmt, balance_sheet):
    """Calculate Return on Assets."""
    try:
        if not income_stmt.empty and not balance_sheet.empty:
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
            return (net_income / total_assets) * 100 if total_assets != 0 else 0
    except:
        pass
    return 0

def calculate_debt_to_equity(balance_sheet):
    """Calculate Debt to Equity ratio."""
    try:
        if not balance_sheet.empty:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            return total_debt / equity if equity != 0 else 0
    except:
        pass
    return 0

def calculate_current_ratio(balance_sheet):
    """Calculate Current Ratio."""
    try:
        if not balance_sheet.empty:
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 0
            return current_assets / current_liabilities if current_liabilities != 0 else 0
    except:
        pass
    return 0

def calculate_profit_margin(income_stmt):
    """Calculate Profit Margin."""
    try:
        if not income_stmt.empty:
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
            return (net_income / revenue) * 100 if revenue != 0 else 0
    except:
        pass
    return 0

def calculate_revenue_growth(income_stmt):
    """Calculate Revenue Growth."""
    try:
        if not income_stmt.empty and len(income_stmt.columns) >= 2:
            current_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
            previous_revenue = income_stmt.loc['Total Revenue'].iloc[1] if 'Total Revenue' in income_stmt.index else 0
            return ((current_revenue - previous_revenue) / previous_revenue) * 100 if previous_revenue != 0 else 0
    except:
        pass
    return 0

def calculate_profit_growth(income_stmt):
    """Calculate Profit Growth."""
    try:
        if not income_stmt.empty and len(income_stmt.columns) >= 2:
            current_profit = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            previous_profit = income_stmt.loc['Net Income'].iloc[1] if 'Net Income' in income_stmt.index else 0
            return ((current_profit - previous_profit) / previous_profit) * 100 if previous_profit != 0 else 0
    except:
        pass
    return 0

def calculate_free_cash_flow(cash_flow):
    """Calculate Free Cash Flow."""
    try:
        if not cash_flow.empty:
            operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
            capital_expenditure = cash_flow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cash_flow.index else 0
            return operating_cash_flow + capital_expenditure  # Capital expenditure is typically negative
    except:
        pass
    return 0

def calculate_intrinsic_value(fundamental_data, discount_rate=0.12, growth_rate=0.08):
    """Calculate intrinsic value using DCF model."""
    free_cash_flow = fundamental_data.get('free_cash_flow', 0)
    
    if free_cash_flow <= 0:
        return None, None
    
    # Simple 10-year DCF model
    years = 10
    future_cash_flows = []
    
    for year in range(1, years + 1):
        future_fcf = free_cash_flow * ((1 + growth_rate) ** year)
        discounted_fcf = future_fcf / ((1 + discount_rate) ** year)
        future_cash_flows.append(discounted_fcf)
    
    # Terminal value (perpetual growth method)
    terminal_growth_rate = 0.03  # 3% perpetual growth
    terminal_value = (future_cash_flows[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** years)
    
    intrinsic_value = sum(future_cash_flows) + discounted_terminal_value
    
    # Calculate margin of safety
    current_price = fundamental_data.get('current_price', 0)
    margin_of_safety = ((intrinsic_value - current_price) / intrinsic_value) * 100 if intrinsic_value > 0 else 0
    
    return intrinsic_value, margin_of_safety

def get_industry_peers(sector):
    """Get industry peers for comparison (simplified)."""
    sector_peers = {
        'Technology': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM'],
        'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
        'Oil & Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL'],
        'Automobile': ['TATAMOTORS', 'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
        'Pharmaceuticals': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN'],
        'Consumer Goods': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR']
    }
    
    return sector_peers.get(sector, ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY'])

# ================ 6. HNI & PRO TRADER FEATURES ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        orders_to_place = []
        for item in basket_items:
            # Find exchange for each symbol
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
            # Clear basket after execution
            st.session_state.basket = []
            st.rerun()
        except Exception as e:
            st.toast(f"❌ Basket order failed: {e}", icon="🔥")

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    # This function expects a local file named 'sensex_sectors.csv'
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
        # Call ITM
        if row['STRIKE'] < ltp:
            styles[df.columns.get_loc('CALL LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('CALL OI')] = 'background-color: #2E4053'
        # Put ITM
        if row['STRIKE'] > ltp:
            styles[df.columns.get_loc('PUT LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('PUT OI')] = 'background-color: #2E4053'
        return styles

    return df.style.apply(highlight_itm, axis=1)

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    st.subheader(f"Most Active {underlying} Options (By Volume)")
    with st.spinner("Fetching data..."):
        active_df = get_most_active_options(underlying, instrument_df)
        if not active_df.empty:
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not retrieve data for most active options.")

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
                'OI': data.get('open_interest', 0) # Corrected key
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
                # Corrected line - proper indexing for MultiIndex DataFrame
                if isinstance(data_yf.columns, pd.MultiIndex):
                    hist = data_yf.xs(yf_ticker_name, level=1, axis=1)
                else:
                    hist = data_yf
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

# ================ 7. PAGE DEFINITIONS ============

# --- Bharatiya Market Pulse (BMP) Functions ---
def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns the score and a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30:
        return 50, "Calculating...", "#cccccc"
    # Normalize NIFTY and SENSEX
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    # Inversely normalize VIX
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
    full_data['size'] = full_data['Price'].astype(float) * 1000 # Using price as a proxy for size
    
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
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
            
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
    """, icon="🧠")
    
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
        else:
            st.info("Select an instrument and run the forecast to see results.")
            
        # Model insights and warnings
        st.markdown("---")
        st.subheader("Model Insights")
        
        insight_cols = st.columns(2)
        
        with insight_cols[0]:
            st.info("""
            **📊 Model Strengths:**
            - **ARIMA**: Best for seasonal patterns
            - **LSTM**: Captures complex temporal dependencies  
            - **XGBoost**: Handles multiple feature types
            """)
        
        with insight_cols[1]:
            st.warning("""
            **⚠️ Important Notes:**
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
    st.info("Real-time market data and order execution for high-frequency trading strategies.", icon="⚡")
    
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
                    {f'({bid['orders']} orders)' if 'orders' in bid else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Asks (Sellers)**")
                asks = depth.get('sell', [])[:5]  # Top 5 asks
                for i, ask in enumerate(asks):
                    st.markdown(f"""
                    <div class="hft-depth-ask">
                    {i+1}. ₹{ask['price']:,.2f} × {ask['quantity']:,} 
                    {f'({ask['orders']} orders)' if 'orders' in ask else ''}
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
    st.info("Execute multiple orders simultaneously for portfolio rebalancing or strategy implementation.", icon="🧺")
    
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
    st.info("Deep dive into company fundamentals, financial health, and intrinsic valuation.", icon="📊")
    
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
    st.info("Get AI-powered trading insights, strategy suggestions, and market analysis.", icon="🤖")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the AI Trade Assistant.")
        return

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Analysis")
        symbol = st.text_input("Symbol for Analysis", "NIFTY 50", key="ai_symbol").upper()
        
        if st.button("Generate Analysis", use_container_width=True):
            with st.spinner("AI is analyzing the market..."):
                # Get historical data
                token = get_instrument_token(symbol, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period='6mo')
                    if not data.empty:
                        # Technical analysis
                        indicators = interpret_indicators(data)
                        
                        # Generate AI insights
                        current_price = data['close'].iloc[-1]
                        price_change = ((current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]) * 100
                        
                        # Simple AI logic based on technical indicators
                        bullish_signals = 0
                        bearish_signals = 0
                        
                        for indicator, interpretation in indicators.items():
                            if "Bullish" in interpretation:
                                bullish_signals += 1
                            elif "Bearish" in interpretation:
                                bearish_signals += 1
                        
                        if bullish_signals > bearish_signals:
                            sentiment = "BULLISH"
                            confidence = min(100, (bullish_signals / len(indicators)) * 100)
                            recommendation = "Consider LONG positions"
                            reasoning = f"Strong bullish signals detected with {bullish_signals} out of {len(indicators)} indicators showing positive momentum."
                        elif bearish_signals > bullish_signals:
                            sentiment = "BEARISH" 
                            confidence = min(100, (bearish_signals / len(indicators)) * 100)
                            recommendation = "Consider SHORT positions or wait for better entry"
                            reasoning = f"Caution advised with {bearish_signals} out of {len(indicators)} indicators showing negative momentum."
                        else:
                            sentiment = "NEUTRAL"
                            confidence = 50
                            recommendation = "Wait for clearer signals or consider range-bound strategies"
                            reasoning = "Mixed signals detected. Market may be consolidating."
                        
                        # Store analysis in session state
                        st.session_state.ai_analysis = {
                            'symbol': symbol,
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'recommendation': recommendation,
                            'reasoning': reasoning,
                            'current_price': current_price,
                            'price_change': price_change,
                            'indicators': indicators
                        }
                    else:
                        st.error("Could not fetch data for analysis")

    with col2:
        st.subheader("Quick Actions")
        
        if st.button("Scan for Opportunities", use_container_width=True):
            with st.spinner("Scanning for trading opportunities..."):
                # Scan top NIFTY stocks
                watchlist_symbols = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK']
                opportunities = []
                
                for stock in watchlist_symbols:
                    token = get_instrument_token(stock, instrument_df)
                    if token:
                        data = get_historical_data(token, 'day', period='1mo')
                        if not data.empty and len(data) > 20:
                            # Simple momentum scan
                            current_price = data['close'].iloc[-1]
                            sma_20 = data['close'].rolling(20).mean().iloc[-1]
                            rsi = data.ta.rsi().iloc[-1] if 'RSI_14' in data.columns else 50
                            
                            if current_price > sma_20 and rsi < 70:
                                opportunities.append({
                                    'symbol': stock,
                                    'signal': 'BULLISH',
                                    'reason': 'Price above 20-day SMA with RSI not overbought'
                                })
                            elif current_price < sma_20 and rsi > 30:
                                opportunities.append({
                                    'symbol': stock, 
                                    'signal': 'BEARISH',
                                    'reason': 'Price below 20-day SMA with RSI not oversold'
                                })
                
                st.session_state.scan_results = opportunities

    # Display AI Analysis Results
    if 'ai_analysis' in st.session_state:
        analysis = st.session_state.ai_analysis
        
        st.markdown("---")
        st.subheader(f"AI Analysis for {analysis['symbol']}")
        
        # Sentiment with color coding
        sentiment_color = {
            'BULLISH': 'var(--green)',
            'BEARISH': 'var(--red)', 
            'NEUTRAL': 'var(--text-light)'
        }.get(analysis['sentiment'], 'var(--text-color)')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Sentiment", analysis['sentiment'])
        col2.metric("AI Confidence", f"{analysis['confidence']:.1f}%")
        col3.metric("Current Price", f"₹{analysis['current_price']:,.2f}", 
                   delta=f"{analysis['price_change']:+.2f}%")
        
        # Recommendation
        st.markdown(f"""
        <div class="trade-card" style="border-color: {sentiment_color};">
            <h4>🤖 AI Recommendation</h4>
            <p><strong>{analysis['recommendation']}</strong></p>
            <p><small>{analysis['reasoning']}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical Indicators
        st.subheader("Technical Indicators")
        if analysis['indicators']:
            indicator_cols = st.columns(3)
            col_idx = 0
            for indicator, interpretation in list(analysis['indicators'].items())[:6]:  # Show first 6
                with indicator_cols[col_idx]:
                    st.metric(indicator, interpretation.split('(')[0].strip())
                col_idx = (col_idx + 1) % 3
        
        # Trading Strategy Suggestions
        st.subheader("Suggested Strategies")
        strategy_cols = st.columns(2)
        
        with strategy_cols[0]:
            if analysis['sentiment'] == 'BULLISH':
                st.markdown("""
                **Bullish Strategies:**
                - Long Call Options
                - Bull Call Spread
                - Buying Stock/ETF
                - Covered Calls
                """)
            else:
                st.markdown("""
                **Bearish/Neutral Strategies:**
                - Long Put Options  
                - Bear Put Spread
                - Protective Puts
                - Cash-Secured Puts
                """)
        
        with strategy_cols[1]:
            st.markdown("""
            **Risk Management:**
            - Position sizing: 1-2% of capital per trade
            - Stop-loss: 2-3% below entry
            - Take-profit: 1:2 risk-reward ratio
            - Diversify across sectors
            """)

    # Display Scan Results
    if 'scan_results' in st.session_state:
        st.markdown("---")
        st.subheader("Opportunity Scan Results")
        
        opportunities = st.session_state.scan_results
        if opportunities:
            for opp in opportunities:
                signal_color = 'var(--green)' if opp['signal'] == 'BULLISH' else 'var(--red)'
                st.markdown(f"""
                <div class="trade-card" style="border-color: {signal_color};">
                    <h4>{opp['symbol']} - {opp['signal']}</h4>
                    <p>{opp['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong opportunities found in the current scan.")

    # Market Education
    with st.expander("📚 Trading Education"):
        st.markdown("""
        **Essential Trading Concepts:**
        
        **Technical Analysis:**
        - Support & Resistance levels
        - Trend identification
        - Volume confirmation
        - Multiple time frame analysis
        
        **Risk Management:**
        - Always use stop-loss orders
        - Risk only 1-2% of capital per trade
        - Maintain a trading journal
        - Emotional discipline is key
        
        **Market Psychology:**
        - Fear and greed drive markets
        - Be fearful when others are greedy
        - Be greedy when others are fearful
        - Patience is a trader's best friend
        """)

def page_settings():
    """User settings and configuration page."""
    display_header()
    st.title("Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Appearance")
        
        # Theme selection
        theme = st.radio(
            "Select Theme",
            ["Dark", "Light"],
            index=0 if st.session_state.get('theme') == 'Dark' else 1,
            key="theme_selector"
        )
        if theme != st.session_state.get('theme'):
            st.session_state.theme = theme
            st.rerun()
        
        # Chart configuration
        st.subheader("Chart Settings")
        default_interval = st.selectbox(
            "Default Chart Interval",
            ["1min", "5min", "15min", "1h", "1d", "1w"],
            index=3
        )
        
        chart_style = st.selectbox(
            "Default Chart Style",
            ["Candlestick", "Line", "Bar", "Heikin-Ashi"],
            index=0
        )
        
    with col2:
        st.subheader("Trading Preferences")
        
        # Default order settings
        default_product = st.radio(
            "Default Product Type",
            ["MIS", "CNC", "NRML"],
            horizontal=True
        )
        
        default_quantity = st.number_input(
            "Default Quantity",
            min_value=1,
            value=1
        )
        
        # Risk management
        st.subheader("Risk Management")
        max_position_size = st.number_input(
            "Maximum Position Size (%)",
            min_value=1,
            max_value=100,
            value=10
        )
        
        auto_stop_loss = st.checkbox(
            "Auto-calculate Stop Loss",
            value=True
        )
    
    # Data & API Settings
    st.subheader("Data & API Configuration")
    
    api_cols = st.columns(2)
    with api_cols[0]:
        st.text_input("Zerodha API Key", type="password")
        st.text_input("Zerodha API Secret", type="password")
    
    with api_cols[1]:
        st.text_input("News API Key", type="password")
        st.text_input("Alternative Data Source", placeholder="Optional")
    
    # Watchlist Management
    st.subheader("Watchlist Management")
    
    watchlist_cols = st.columns([2, 1])
    with watchlist_cols[0]:
        new_watchlist_name = st.text_input("Create New Watchlist")
    
    with watchlist_cols[1]:
        if st.button("Create Watchlist", use_container_width=True) and new_watchlist_name:
            if new_watchlist_name not in st.session_state.watchlists:
                st.session_state.watchlists[new_watchlist_name] = []
                st.success(f"Created watchlist: {new_watchlist_name}")
                st.rerun()
            else:
                st.error("Watchlist already exists!")
    
    # Display existing watchlists
    for watchlist_name in list(st.session_state.watchlists.keys()):
        if watchlist_name != "Watchlist 1":  # Don't allow deletion of default
            cols = st.columns([3, 1])
            cols[0].write(f"📊 {watchlist_name} ({len(st.session_state.watchlists[watchlist_name])} symbols)")
            if cols[1].button("Delete", key=f"del_{watchlist_name}"):
                del st.session_state.watchlists[watchlist_name]
                st.rerun()
    
    # Export/Import Settings
    st.subheader("Data Management")
    
    manage_cols = st.columns(2)
    with manage_cols[0]:
        if st.button("Export Settings", use_container_width=True):
            # In a real app, this would generate a settings file
            st.info("Settings export would be implemented in production")
    
    with manage_cols[1]:
        if st.button("Reset to Defaults", use_container_width=True):
            st.warning("This will reset all settings to defaults!")
            if st.button("Confirm Reset", type="primary"):
                initialize_session_state()
                st.success("Settings reset to defaults!")
                st.rerun()

    # About Section
    with st.expander("About BlockVista Terminal"):
        st.markdown("""
        **BlockVista Terminal v2.0**
        
        A comprehensive trading terminal with advanced features:
        
        - **Real-time Market Data**: Live prices, charts, and order book
        - **Advanced Charting**: Multiple timeframes and technical indicators
        - **F&O Analytics**: Options chain, PCR, and volatility analysis
        - **ML Forecasting**: AI-powered price predictions
        - **Fundamental Analysis**: Company valuation and financial metrics
        - **Basket Orders**: Execute multiple orders simultaneously
        - **HFT Terminal**: Professional trading interface
        
        **Built with:** Streamlit, Plotly, Pandas, Scikit-learn, TensorFlow
        
        **Disclaimer:** This is for educational purposes only. 
        Trading involves risk of financial loss.
        """)

# ================ 8. LOGIN & AUTHENTICATION ================

def login_page():
    """Handles user authentication and broker connection."""
    st.title("🔐 BlockVista Terminal Login")
    st.markdown("Connect your trading account to access real-time market data and trading features.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Broker Connection")
        broker = st.selectbox("Select Your Broker", ["Zerodha", "Angel One", "ICICI Direct", "Demo Mode"])
        
        if broker == "Zerodha":
            st.info("""
            **Zerodha Connection:**
            1. Login to [Kite Connect](https://kite.trade/)
            2. Go to Developer section
            3. Create an app to get API Key and Secret
            """)
            
            api_key = st.text_input("API Key", placeholder="Enter your API key")
            api_secret = st.text_input("API Secret", type="password", placeholder="Enter your API secret")
            redirect_url = st.text_input("Redirect URL", value="http://localhost:8501")
            
            if st.button("Connect to Zerodha", use_container_width=True):
                if api_key and api_secret:
                    try:
                        kite = KiteConnect(api_key=api_key)
                        request_token_url = kite.login_url()
                        
                        st.session_state.kite = kite
                        st.session_state.api_key = api_key
                        st.session_state.api_secret = api_secret
                        st.session_state.broker = broker
                        
                        st.success("Zerodha connection initialized!")
                        st.markdown(f"""
                        **Next Steps:**
                        1. [Click here to login and get request token]({request_token_url})
                        2. Enter the request token below after authorization
                        """)
                        
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
                else:
                    st.error("Please enter both API Key and Secret")
        
        elif broker == "Demo Mode":
            if st.button("Enter Demo Mode", use_container_width=True):
                st.session_state.broker = "Demo"
                st.session_state.authenticated = True
                st.session_state.profile = {"user_name": "Demo User", "email": "demo@blockvista.com"}
                st.success("Entering Demo Mode with sample data!")
                st.rerun()
    
    with col2:
        if st.session_state.get('broker') == "Zerodha" and st.session_state.get('kite'):
            st.subheader("Complete Authentication")
            request_token = st.text_input("Request Token", placeholder="Paste request token here")
            
            if st.button("Authenticate", use_container_width=True):
                if request_token:
                    try:
                        kite = st.session_state.kite
                        data = kite.generate_session(request_token, st.session_state.api_secret)
                        kite.set_access_token(data["access_token"])
                        
                        st.session_state.kite = kite
                        st.session_state.authenticated = True
                        st.session_state.profile = kite.profile()
                        
                        st.success("✅ Authentication successful!")
                        st.balloons()
                        a_time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")
                else:
                    st.error("Please enter the request token")
        
        # Two-Factor Authentication Setup
        if st.session_state.get('authenticated') and not st.session_state.get('two_factor_setup_complete'):
            st.subheader("🔒 Two-Factor Authentication")
            
            if st.session_state.pyotp_secret is None:
                st.session_state.pyotp_secret = pyotp.random_base32()
            
            totp = pyotp.TOTP(st.session_state.pyotp_secret)
            provisioning_uri = totp.provisioning_uri(
                name=st.session_state.profile.get('user_name', 'user'),
                issuer_name="BlockVista Terminal"
            )
            
            # Generate QR Code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64 for display
            buffered = io.BytesIO()
            qr_img.save(buffered, format="PNG")
            qr_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            st.image(f"data:image/png;base64,{qr_base64}", width=200)
            st.write("Scan this QR code with Google Authenticator or Authy")
            st.code(st.session_state.pyotp_secret, language="text")
            
            # Verify setup
            otp_input = st.text_input("Enter OTP from authenticator app")
            if st.button("Verify 2FA Setup"):
                if totp.verify(otp_input):
                    st.session_state.two_factor_setup_complete = True
                    st.success("✅ Two-factor authentication setup complete!")
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")

# ================ 9. MAIN APPLICATION ================

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
            st.markdown(f"""
            <div style="background: var(--secondary-bg); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin: 0;">{st.session_state.profile.get('user_name', 'User')}</h4>
                <small style="color: var(--text-light);">{st.session_state.broker}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Navigation options
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
        
        if st.button("View Portfolio", use_container_width=True):
            st.session_state.active_page = "Dashboard"
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
            BlockVista Terminal v2.0<br>
            <small>Educational Purpose Only</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Display selected page
    page_function = page_options[selected_page]
    page_function()

if __name__ == "__main__":
    main()
