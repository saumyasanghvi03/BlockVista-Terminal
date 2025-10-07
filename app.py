# ================ 0. REQUIRED LIBRARIES ================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time as dt_time
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

# ================ 2. ENHANCED DATA COLLECTION ================

# Data sources now primarily rely on yfinance for up-to-date, dynamic data.
ENHANCED_DATA_SOURCES = {
    "NIFTY 50": {"yfinance_ticker": "^NSEI", "tradingsymbol": "NIFTY 50", "exchange": "NSE"},
    "BANK NIFTY": {"yfinance_ticker": "^NSEBANK", "tradingsymbol": "BANKNIFTY", "exchange": "NFO"},
    "NIFTY Financial Services": {"yfinance_ticker": "FINNIFTY.NS", "tradingsymbol": "FINNIFTY", "exchange": "NFO"},
    "GOLD": {"yfinance_ticker": "GC=F", "tradingsymbol": "GOLDM", "exchange": "MCX"},
    "USDINR": {"yfinance_ticker": "INR=X", "tradingsymbol": "USDINR", "exchange": "CDS"},
    "SENSEX": {"yfinance_ticker": "^BSESN", "tradingsymbol": "SENSEX", "exchange": "BSE"},
    "S&P 500": {"yfinance_ticker": "^GSPC", "tradingsymbol": "^GSPC", "exchange": "yfinance"},
    "NIFTY MIDCAP 100": {"yfinance_ticker": "^CNXMD", "tradingsymbol": "NIFTYMID100", "exchange": "NSE"}
}

def get_enhanced_historical_data(instrument_name, data_type="daily", period="5y"):
    """
    Enhanced historical data fetcher using yfinance as the primary source.
    """
    source_info = ENHANCED_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()

    yf_ticker = source_info.get("yfinance_ticker")
    if not yf_ticker:
        st.error(f"yfinance ticker not configured for {instrument_name}")
        return pd.DataFrame()

    interval = "1h" if data_type == "hourly" else "1d"
    
    try:
        data = yf.download(yf_ticker, period=period, interval=interval, progress=False)
        if data.empty:
            st.warning(f"No data returned from yfinance for {instrument_name} ({yf_ticker})")
            return pd.DataFrame()

        data.columns = [col.lower() for col in data.columns]
        
        # Add technical indicators if we have data
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            data.ta.rsi(length=14, append=True)
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            data.ta.macd(append=True)
            data.ta.bbands(append=True)
        
        return data

    except Exception as e:
        st.error(f"Error downloading {instrument_name} from yfinance: {e}")
        return pd.DataFrame()

# Alias for ML module compatibility
ML_DATA_SOURCES = ENHANCED_DATA_SOURCES


# ================ 3. INITIALIZATION ================
def initialize_session_state():
    """Initializes all necessary session state variables."""
    defaults = {
        'broker': None, 'kite': None, 'profile': None, 'login_animation_complete': False,
        'authenticated': False, 'two_factor_setup_complete': False, 'pyotp_secret': None,
        'theme': 'Dark',
        'watchlists': {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
            "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
        },
        'active_watchlist': "Watchlist 1", 'order_history': [], 'basket': [],
        'last_order_details': {}, 'underlying_pcr': "NIFTY", 'strategy_legs': [],
        'calculated_greeks': None, 'messages': [], 'ml_forecast_df': None,
        'ml_instrument_name': None, 'backtest_results': None, 'fundamental_companies': ['RELIANCE', 'TCS'],
        'hft_last_price': 0, 'hft_tick_log': [], 'market_notifications_shown': {},
        'show_2fa_dialog': False, 'show_qr_dialog': False,
        # Bot states
        'bot_momentum_running': False, 'bot_momentum_log': [], 'bot_momentum_pnl': 0.0, 'bot_momentum_position': None,
        'bot_reversion_running': False, 'bot_reversion_log': [], 'bot_reversion_pnl': 0.0, 'bot_reversion_position': None,
        'bot_breakout_running': False, 'bot_breakout_log': [], 'bot_breakout_pnl': 0.0, 'bot_breakout_position': None,
        'bot_value_running': False, 'bot_value_log': [], 'bot_value_investments': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================ 4. CORE HELPER & UI FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    return st.session_state.get('kite') if st.session_state.get('broker') == "Zerodha" else None

def get_market_status():
    """Checks if the Indian stock market is open."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    }.get(now.year, [])
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    """Displays the main header with market status and a live clock."""
    status_info = get_market_status()
    current_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S IST")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right;">
            <h5 style="margin: 0;">{current_time}</h5>
            <h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    client = get_broker_client()
    if not client: return pd.DataFrame()
    try:
        df = pd.DataFrame(client.instruments())
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
        return df
    except Exception:
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period='1y'):
    """Fetches historical data from the broker's API."""
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    
    days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
    from_date = datetime.now().date() - timedelta(days=days_to_subtract.get(period, 365))
    to_date = datetime.now().date()
    
    try:
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Add a comprehensive set of indicators
        df.ta.adx(append=True)
        df.ta.bbands(append=True)
        df.ta.donchian(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.supertrend(append=True)

        return df
    except Exception as e:
        st.toast(f"API Error (Historical): {e}", icon="⚠️")
        return pd.DataFrame()

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None, tag="BlockVista"):
    """Places a single order through the broker's API with a tag."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return None
    
    try:
        instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
        if instrument.empty:
            st.error(f"Symbol '{symbol}' not found.")
            return None
        exchange = instrument.iloc[0]['exchange']

        order_id = client.place_order(
            tradingsymbol=symbol.upper(),
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=product,
            variety=client.VARIETY_REGULAR,
            price=price,
            tag=tag # Add a tag to identify bot trades
        )
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
        return order_id
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")
        st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})
        return None

# ================ 5. PAGE DEFINITIONS ================

# --- QUICK TRADE DIALOG ---
def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    if 'show_quick_trade' not in st.session_state:
        st.session_state.show_quick_trade = False
    
    if symbol or st.button("Quick Trade", key="quick_trade_btn"):
        st.session_state.show_quick_trade = True
    
    if st.session_state.show_quick_trade:
        with st.form("quick_trade_form"):
            st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
            
            if symbol is None:
                symbol = st.text_input("Symbol").upper()
            
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
            quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
            price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

            submitted = st.form_submit_button("Submit Order")
            if submitted:
                if symbol and quantity > 0:
                    instrument_df = get_instrument_df()
                    place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
                    st.session_state.show_quick_trade = False
                    st.rerun()
                else:
                    st.warning("Please fill in all fields.")
        
        if st.button("Cancel"):
            st.session_state.show_quick_trade = False
            st.rerun()

# --- OTHER CORE FUNCTIONS (UNCHANGED) ---
def check_market_timing_notifications():
    pass
def display_overnight_changes_bar():
    pass
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    fig = go.Figure()
    if df.empty: return fig
    chart_df = df.copy()
    chart_df.columns = [str(col).lower() for col in chart_df.columns]
    
    if not all(col in chart_df.columns for col in ['open', 'high', 'low', 'close']):
        return fig

    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template)
    return fig

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
    client = get_broker_client()
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
    except Exception:
        return pd.DataFrame()

# --- PAGES (SELECTED) ---

def page_dashboard():
    """Main dashboard page."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
        
    st.subheader("Watchlist")
    active_list = st.session_state.watchlists[st.session_state.active_watchlist]
    watchlist_data = get_watchlist_data(active_list)
    if not watchlist_data.empty:
        st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
    else:
        st.info("Watchlist is empty or data could not be fetched.")

def page_fo_analytics():
    """F&O Analytics page."""
    display_header()
    st.title("F&O Analytics Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to access F&O Analytics.")
        return
    st.info("Options Chain and other F&O tools will be displayed here.")

# --- ALGO BOTS PAGE ---

def page_algo_bots():
    """A page to run pre-built automated trading strategies."""
    display_header()
    st.title("🤖 Algo Trading Bots")
    st.info("Activate pre-built trading bots to scan markets and execute trades automatically. Use with caution.", icon="💡")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.warning("Please connect to a broker to use the Algo Bots.")
        return
    
    nifty50_stocks = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 
        'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK', 'MARUTI', 'ASIANPAINT',
        'TATAMOTORS', 'BHARTIARTL', 'ADANIENT', 'TATASTEEL', 'HCLTECH'
    ]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"])

    with tab1:
        st.subheader("Momentum Trader (RSI + EMA)")
        st.markdown("Buys stocks in a strong uptrend (Price > 20-EMA) with strong momentum (RSI > 60). Sells when momentum fades (RSI < 50).")
        
        capital1 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap1")
        scan_list1 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[:5], key="scan1")
        
        is_running = st.session_state.get('bot_momentum_running', False)
        button_text = "⏹️ Stop Bot" if is_running else "▶️ Run Bot"
        if st.button(button_text, key="run_mom_bot"):
            st.session_state.bot_momentum_running = not is_running
            st.rerun()

        if is_running:
            run_momentum_bot(instrument_df, scan_list1, capital1)
        
        st.write("**Activity Log:**")
        log_container = st.container(height=200)
        for log in st.session_state.get('bot_momentum_log', []):
            log_container.text(log)

    with tab2:
        st.subheader("Mean Reversion (Bollinger Bands)")
        st.markdown("Buys when price hits the lower Bollinger Band, expecting a bounce. Sells when price hits the middle band (20-SMA).")
        
        capital2 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap2")
        scan_list2 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[5:10], key="scan2")
        
        is_running2 = st.session_state.get('bot_reversion_running', False)
        if st.button("⏹️ Stop Bot" if is_running2 else "▶️ Run Bot", key="run_rev_bot"):
            st.session_state.bot_reversion_running = not is_running2
            st.rerun()

        if is_running2:
            run_mean_reversion_bot(instrument_df, scan_list2, capital2)
        
        st.write("**Activity Log:**")
        log_container2 = st.container(height=200)
        for log in st.session_state.get('bot_reversion_log', []):
            log_container2.text(log)
            
    with tab3:
        st.subheader("Volatility Breakout (Donchian Channel)")
        st.markdown("Buys when the stock breaks its 20-day high. Sells if it falls below the 10-day low.")
        
        capital3 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap3")
        scan_list3 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[10:15], key="scan3")
        
        is_running3 = st.session_state.get('bot_breakout_running', False)
        if st.button("⏹️ Stop Bot" if is_running3 else "▶️ Run Bot", key="run_break_bot"):
            st.session_state.bot_breakout_running = not is_running3
            st.rerun()

        if is_running3:
            run_volatility_breakout_bot(instrument_df, scan_list3, capital3)
        
        st.write("**Activity Log:**")
        log_container3 = st.container(height=200)
        for log in st.session_state.get('bot_breakout_log', []):
            log_container3.text(log)
            
    with tab4:
        st.subheader("Value Investor (Fundamental)")
        st.markdown("Scans for stocks with P/E < 25, P/B < 5, and ROE > 15%, then makes a long-term investment (`CNC`).")
        
        capital4 = st.number_input("Investment Amount (₹)", 20000, 200000, 50000, 1000, key="cap4")
        scan_list4 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks, key="scan4")
        
        is_running4 = st.session_state.get('bot_value_running', False)
        if st.button("⏹️ Stop Scanner" if is_running4 else "▶️ Run Scanner", key="run_val_bot"):
            st.session_state.bot_value_running = not is_running4
            st.rerun()
            
        if is_running4:
            run_value_investor_bot(instrument_df, scan_list4, capital4)
            
        st.write("**Activity Log:**")
        log_container4 = st.container(height=200)
        for log in st.session_state.get('bot_value_log', []):
            log_container4.text(log)

# --- BOT LOGIC ---

def log_bot_action(bot_key, message):
    log_list = st.session_state[f'bot_{bot_key}_log']
    log_list.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    if len(log_list) > 50: log_list.pop()

def run_momentum_bot(instrument_df, scan_list, capital):
    bot_key = "momentum"
    position = st.session_state.get(f'bot_{bot_key}_position')
    
    if position:
        token = get_instrument_token(position['symbol'], instrument_df)
        data = get_historical_data(token, '5minute', period='5d')
        if not data.empty and 'RSI_14' in data.columns:
            rsi = data['RSI_14'].iloc[-1]
            if rsi < 50:
                place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotMomentumExit")
                log_bot_action(bot_key, f"SELL SIGNAL: RSI ({rsi:.1f}) < 50. Closing {position['symbol']}.")
                st.session_state[f'bot_{bot_key}_position'] = None
                return
    else:
        for symbol in scan_list:
            token = get_instrument_token(symbol, instrument_df)
            data = get_historical_data(token, '5minute', period='5d')
            if data.empty or 'RSI_14' not in data.columns or 'EMA_20' not in data.columns: continue
            
            latest = data.iloc[-1]
            if latest['close'] > latest['EMA_20'] and latest['RSI_14'] > 60:
                quantity = int(capital / latest['close'])
                if quantity > 0:
                    if place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS', tag="BotMomentumEntry"):
                        log_bot_action(bot_key, f"BUY SIGNAL on {symbol}. Order placed.")
                        st.session_state[f'bot_{bot_key}_position'] = {'symbol': symbol, 'quantity': quantity}
                        return

def run_mean_reversion_bot(instrument_df, scan_list, capital):
    bot_key = "reversion"
    position = st.session_state.get(f'bot_{bot_key}_position')
    
    if position:
        token = get_instrument_token(position['symbol'], instrument_df)
        data = get_historical_data(token, '5minute', period='5d')
        if not data.empty and 'BBM_20_2.0' in data.columns:
            if data['close'].iloc[-1] >= data['BBM_20_2.0'].iloc[-1]:
                place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotReversionExit")
                log_bot_action(bot_key, f"SELL SIGNAL: Price hit middle band. Closing {position['symbol']}.")
                st.session_state[f'bot_{bot_key}_position'] = None
                return
    else:
        for symbol in scan_list:
            token = get_instrument_token(symbol, instrument_df)
            data = get_historical_data(token, '5minute', period='5d')
            if data.empty or 'BBL_20_2.0' not in data.columns: continue

            if data['close'].iloc[-1] <= data['BBL_20_2.0'].iloc[-1]:
                quantity = int(capital / data['close'].iloc[-1])
                if quantity > 0:
                    if place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS', tag="BotReversionEntry"):
                        log_bot_action(bot_key, f"BUY SIGNAL on {symbol}: Price hit lower Bollinger Band.")
                        st.session_state[f'bot_{bot_key}_position'] = {'symbol': symbol, 'quantity': quantity}
                        return

def run_volatility_breakout_bot(instrument_df, scan_list, capital):
    bot_key = "breakout"
    position = st.session_state.get(f'bot_{bot_key}_position')
    
    if position:
        token = get_instrument_token(position['symbol'], instrument_df)
        data = get_historical_data(token, 'day', period='6mo')
        if not data.empty and 'DCL_10_20' in data.columns:
            if data['close'].iloc[-1] < data['DCL_10_20'].iloc[-1]:
                place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotBreakoutExit")
                log_bot_action(bot_key, f"STOP SIGNAL: Price broke 10-day low. Closing {position['symbol']}.")
                st.session_state[f'bot_{bot_key}_position'] = None
                return
    else:
        for symbol in scan_list:
            token = get_instrument_token(symbol, instrument_df)
            data = get_historical_data(token, 'day', period='6mo')
            if data.empty or 'DCU_20_20' not in data.columns: continue
            
            if data['close'].iloc[-1] > data['DCU_20_20'].iloc[-2]:
                quantity = int(capital / data['close'].iloc[-1])
                if quantity > 0:
                    if place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS', tag="BotBreakoutEntry"):
                        log_bot_action(bot_key, f"BUY SIGNAL on {symbol}: Broke 20-day high.")
                        st.session_state[f'bot_{bot_key}_position'] = {'symbol': symbol, 'quantity': quantity}
                        return
                        
def run_value_investor_bot(instrument_df, scan_list, capital):
    bot_key = "value"
    log_bot_action(bot_key, "Starting fundamental scan...")
    
    for symbol in scan_list:
        try:
            if any(inv['symbol'] == symbol for inv in st.session_state.get('bot_value_investments', [])):
                continue
            
            fundamentals = get_fundamental_data(symbol)
            if fundamentals:
                pe = fundamentals.get('P/E Ratio', 999)
                pb = fundamentals.get('P/B Ratio', 99)
                roe = fundamentals.get('ROE', 0)
                
                if 0 < pe < 25 and 0 < pb < 5 and roe > 0.15:
                    ltp_df = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                    if not ltp_df.empty:
                        ltp = ltp_df.iloc[0]['Price']
                        quantity = int(capital / ltp)
                        if quantity > 0:
                            if place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'CNC', tag="BotValueEntry"):
                                log_bot_action(bot_key, f"VALUE FIND on {symbol}: P/E={pe:.1f}, ROE={roe*100:.1f}%. Investing.")
                                st.session_state.bot_value_investments.append({'symbol': symbol, 'quantity': quantity})
                                st.session_state.bot_value_running = False
                                st.rerun()
            a_time.sleep(1)
        except Exception:
            continue
    log_bot_action(bot_key, "Scan complete. No new value stocks found.")
    st.session_state.bot_value_running = False
    st.rerun()

# --- FUNDAMENTAL ANALYTICS PAGE (REPAIRED) ---

@st.cache_data(ttl=3600)
def get_fundamental_data(symbol):
    """Fetches key fundamental data from yfinance, handling errors."""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        
        data = {
            'Company Name': info.get('longName'), 'Sector': info.get('sector'),
            'Market Cap': info.get('marketCap'), 'P/E Ratio': info.get('trailingPE'),
            'P/B Ratio': info.get('priceToBook'), 'Dividend Yield': info.get('dividendYield'),
            'ROE': info.get('returnOnEquity'), 'Debt to Equity': info.get('debtToEquity'),
            'Profit Margins': info.get('profitMargins'), 'Revenue Growth': info.get('revenueGrowth'),
        }
        return {k: v for k, v in data.items() if v is not None and v != 0}
    except Exception:
        return None

def get_financial_statement(symbol, statement_type):
    """Fetches financial statements, robustly handling missing columns."""
    try:
        ticker = yf.Ticker(symbol + ".NS")
        
        if statement_type == 'balance_sheet':
            df = ticker.balance_sheet
            cols = ['Total Assets', 'Current Assets', 'Current Liabilities', 'Total Stockholder Equity', 'Net Debt']
        elif statement_type == 'income_stmt':
            df = ticker.income_stmt
            cols = ['Total Revenue', 'Cost Of Revenue', 'Operating Income', 'Net Income', 'Basic EPS']
        else: # cash_flow
            df = ticker.cash_flow
            cols = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
            
        if df.empty: return None
        
        available_cols = [col for col in cols if col in df.index]
        return df.loc[available_cols].T.tail(4) if available_cols else None
    except Exception:
        return None
        
def format_large_number(num):
    if pd.isna(num): return "N/A"
    num = float(num)
    if abs(num) >= 10_000_000: return f"₹{num/10_000_000:.2f} Cr"
    if abs(num) >= 100_000: return f"₹{num/100_000:.2f} L"
    return f"₹{num:,.2f}"

def page_fundamental_analytics():
    """Fundamental Analytics page with real data and robust error handling."""
    display_header()
    st.title("📊 Fundamental Analytics")
    
    popular_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BAJFINANCE']
    
    st.subheader("Select Companies for Analysis")
    selected_companies = st.multiselect("Enter stock symbols:", options=popular_stocks, default=st.session_state.fundamental_companies)
    st.session_state.fundamental_companies = selected_companies

    if not selected_companies:
        st.info("Please select one or more companies to begin analysis.")
        return

    st.markdown("---")
    st.subheader("📈 Key Metrics Comparison")

    with st.spinner("Fetching fundamental data..."):
        all_data = {symbol: get_fundamental_data(symbol) for symbol in selected_companies}
        valid_data = {s: d for s, d in all_data.items() if d}
        
        if valid_data:
            df_list = []
            for symbol, data in valid_data.items():
                data['Symbol'] = symbol
                df_list.append(data)
            comp_df = pd.DataFrame(df_list).set_index('Symbol')
            st.dataframe(comp_df.style.format(precision=2), use_container_width=True)
        else:
            st.error("Could not fetch data for any selected companies.")

    st.markdown("---")
    st.subheader("📋 Detailed Financial Statements")
    
    company_to_view = st.selectbox("Select a company for detailed view:", options=selected_companies)
    
    if company_to_view:
        tab1, tab2, tab3 = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])
        
        with tab1, st.spinner("Fetching Balance Sheet..."):
            bs_df = get_financial_statement(company_to_view, 'balance_sheet')
            if bs_df is not None:
                st.dataframe(bs_df.applymap(format_large_number), use_container_width=True)
            else:
                st.warning("Balance Sheet data not available for this company.")
                    
        with tab2, st.spinner("Fetching Income Statement..."):
            is_df = get_financial_statement(company_to_view, 'income_stmt')
            if is_df is not None:
                st.dataframe(is_df.applymap(format_large_number), use_container_width=True)
            else:
                st.warning("Income Statement data not available for this company.")

        with tab3, st.spinner("Fetching Cash Flow..."):
            cf_df = get_financial_statement(company_to_view, 'cash_flow')
            if cf_df is not None:
                st.dataframe(cf_df.applymap(format_large_number), use_container_width=True)
            else:
                st.warning("Cash Flow data not available for this company.")

# --- MAIN APP ---
def main_app():
    apply_custom_styling()
    
    if not st.session_state.get('authenticated'):
        st.title("Authentication Required")
        st.info("Please login to access the terminal.")
        return

    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.divider()
    
    st.sidebar.header("Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", 5, 60, 10, disabled=not auto_refresh)
    st.sidebar.divider()

    st.sidebar.header("Navigation")
    pages = {
        "Dashboard": page_dashboard,
        "Fundamental Analytics": page_fundamental_analytics,
        "Algo Trading Bots": page_algo_bots,
        "F&O Analytics": page_fo_analytics,
        # Add other pages back here as needed
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["Algo Trading Bots", "Fundamental Analytics"]:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
        
    pages[selection]()

def login_page():
    apply_custom_styling()
    st.title("BlockVista Terminal Login")
    
    api_key = st.secrets.get("ZERODHA_API_KEY")
    api_secret = st.secrets.get("ZERODHA_API_SECRET")

    if not api_key or not api_secret:
        st.error("Kite API credentials are not set in Streamlit secrets.")
        st.stop()

    kite = KiteConnect(api_key=api_key)
    request_token = st.query_params.get("request_token")

    if request_token:
        try:
            with st.spinner("Authenticating..."):
                data = kite.generate_session(request_token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                st.session_state.kite = kite
                st.session_state.profile = kite.profile()
                st.session_state.broker = "Zerodha"
                st.session_state.authenticated = True
                st.query_params.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    else:
        login_url = kite.login_url()
        st.link_button("Login with Zerodha Kite", login_url, use_container_width=True)


if __name__ == "__main__":
    initialize_session_state()
    
    if st.session_state.get('authenticated') and st.session_state.get('profile'):
        main_app()
    else:
        login_page()
