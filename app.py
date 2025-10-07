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
            --primary-bg: var(--dark-bg); --secondary-bg: var(--dark-secondary-bg); --widget-bg: var(--dark-widget-bg);
            --border-color: var(--dark-border); --text-color: var(--dark-text); --text-light: var(--dark-text-light);
            --green: var(--dark-green); --red: var(--dark-red);
        }
        body.light-theme {
            --primary-bg: var(--light-bg); --secondary-bg: var(--light-secondary-bg); --widget-bg: var(--light-widget-bg);
            --border-color: var(--light-border); --text-color: var(--light-text); --text-light: var(--light-text-light);
            --green: var(--light-green); --red: var(--light-red);
        }
        body { background-color: var(--primary-bg); color: var(--text-color); }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3, h4, h5 { color: var(--text-color) !important; }
        hr { background: var(--border-color); }
        .stButton>button { border-color: var(--border-color); background-color: var(--widget-bg); color: var(--text-color); }
        .stButton>button:hover { border-color: var(--green); color: var(--green); }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
            background-color: var(--widget-bg); border-color: var(--border-color); color: var(--text-color);
        }
        .stRadio>div { background-color: var(--widget-bg); border: 1px solid var(--border-color); padding: 8px; border-radius: 8px; }
        .metric-card { background-color: var(--secondary-bg); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 10px; border-left-width: 5px; }
        .trade-card { background-color: var(--secondary-bg); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 10px; border-left-width: 5px; }
        .notification-bar {
            position: sticky; top: 0; width: 100%; background-color: var(--secondary-bg); color: var(--text-color);
            padding: 8px 12px; z-index: 999; display: flex; justify-content: center; align-items: center; font-size: 0.9rem;
            border-bottom: 1px solid var(--border-color); margin-left: -20px; margin-right: -20px; width: calc(100% + 40px);
        }
        .notification-bar span { margin: 0 15px; white-space: nowrap; }
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

# ================ 2. DATA & API CONFIGURATION ================

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
        },
        'active_watchlist': "Watchlist 1", 'basket': [], 'underlying_pcr': "NIFTY",
        'fundamental_companies': ['RELIANCE', 'TCS'],
        # Bot states
        'bot_momentum_running': False, 'bot_momentum_log': [], 'bot_momentum_position': None,
        'bot_reversion_running': False, 'bot_reversion_log': [], 'bot_reversion_position': None,
        'bot_breakout_running': False, 'bot_breakout_log': [], 'bot_breakout_position': None,
        'bot_value_running': False, 'bot_value_log': [], 'bot_value_investments': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================ 4. CORE HELPER & API FUNCTIONS ================

def get_broker_client():
    return st.session_state.get('kite')

def display_header():
    """Displays the main header with market status and a live clock."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = ['2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    
    status = "CLOSED"
    color = "#FF4B4B"
    if now.weekday() < 5 and now.strftime('%Y-%m-%d') not in holidays:
        if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
            status = "OPEN"
            color = "#28a745"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right;">
            <h5 style="margin: 0;">{now.strftime("%H:%M:%S IST")}</h5>
            <h5 style="margin: 0;">Market: <span style='color:{color}; font-weight: bold;'>{status}</span></h5>
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
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period='1y'):
    """Fetches historical data from the broker's API."""
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    
    days_map = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
    from_date = datetime.now().date() - timedelta(days=days_map.get(period, 365))
    to_date = datetime.now().date()
    
    try:
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # Add a comprehensive set of indicators
        df.ta.adx(append=True); df.ta.bbands(append=True); df.ta.donchian(append=True)
        df.ta.ema(length=20, append=True); df.ta.ema(length=50, append=True)
        df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.supertrend(append=True)

        return df
    except Exception as e:
        st.toast(f"API Error (Historical Data): {e}", icon="⚠️")
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    
    instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
    try:
        quotes = client.quote(instrument_names)
        data = []
        for item in symbols_with_exchange:
            instrument = f"{item['exchange']}:{item['symbol']}"
            quote = quotes.get(instrument)
            if quote:
                change = quote['last_price'] - quote['ohlc']['close']
                pct_change = (change / quote['ohlc']['close'] * 100) if quote['ohlc']['close'] != 0 else 0
                data.append({'Ticker': item['symbol'], 'Price': quote['last_price'], 'Change': change, '% Change': pct_change})
        return pd.DataFrame(data)
    except Exception:
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
            tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type,
            quantity=quantity, order_type=order_type, product=product,
            variety=client.VARIETY_REGULAR, price=price, tag=tag
        )
        st.toast(f"✅ Order placed for {symbol}! ID: {order_id}", icon="🎉")
        return order_id
    except Exception as e:
        st.toast(f"❌ Order for {symbol} failed: {e}", icon="🔥")
        return None

# ================ 5. PAGE DEFINITIONS ================

def page_dashboard():
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
        
    st.subheader("Watchlist")
    active_list = st.session_state.watchlists.get(st.session_state.active_watchlist, [])
    watchlist_data = get_watchlist_data(active_list)
    if not watchlist_data.empty:
        st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
    else:
        st.info("Watchlist is empty or data could not be fetched.")

# --- ALGO TRADING BOTS PAGE ---

def page_algo_bots():
    """A page to run pre-built automated trading strategies."""
    display_header()
    st.title("🤖 Algo Trading Bots")
    st.info("Activate bots to scan markets and execute trades automatically. Trades are placed as MIS orders. Use with caution.", icon="💡")

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
        st.markdown("Buys stocks in an uptrend (Price > 20-EMA) with strong momentum (RSI > 60). Sells when momentum fades (RSI < 50).")
        
        capital1 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap1")
        scan_list1 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[:5], key="scan1")
        
        is_running = st.session_state.get('bot_momentum_running', False)
        if st.button("⏹️ Stop Bot" if is_running else "▶️ Run Bot", key="run_mom_bot"):
            st.session_state.bot_momentum_running = not is_running
            st.rerun()

        if is_running: run_momentum_bot(instrument_df, scan_list1, capital1)
        
        st.write("**Activity Log:**"); log_container = st.container(height=200)
        for log in st.session_state.get('bot_momentum_log', []): log_container.text(log)

    with tab2:
        st.subheader("Mean Reversion (Bollinger Bands)")
        st.markdown("Buys on dips to the lower Bollinger Band. Sells when the price reverts to the middle band (20-SMA).")

        capital2 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap2")
        scan_list2 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[5:10], key="scan2")
        
        is_running2 = st.session_state.get('bot_reversion_running', False)
        if st.button("⏹️ Stop Bot" if is_running2 else "▶️ Run Bot", key="run_rev_bot"):
            st.session_state.bot_reversion_running = not is_running2
            st.rerun()

        if is_running2: run_mean_reversion_bot(instrument_df, scan_list2, capital2)
        
        st.write("**Activity Log:**"); log_container2 = st.container(height=200)
        for log in st.session_state.get('bot_reversion_log', []): log_container2.text(log)
            
    with tab3:
        st.subheader("Volatility Breakout (Donchian Channel)")
        st.markdown("Buys when a stock breaks its 20-day high. Sells as a trailing stop if it falls below the 10-day low.")

        capital3 = st.number_input("Capital per Trade (₹)", 10000, 100000, 25000, 1000, key="cap3")
        scan_list3 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks[10:15], key="scan3")
        
        is_running3 = st.session_state.get('bot_breakout_running', False)
        if st.button("⏹️ Stop Bot" if is_running3 else "▶️ Run Bot", key="run_break_bot"):
            st.session_state.bot_breakout_running = not is_running3
            st.rerun()

        if is_running3: run_volatility_breakout_bot(instrument_df, scan_list3, capital3)
        
        st.write("**Activity Log:**"); log_container3 = st.container(height=200)
        for log in st.session_state.get('bot_breakout_log', []): log_container3.text(log)
            
    with tab4:
        st.subheader("Value Investor (Fundamental)")
        st.markdown("Scans for stocks with P/E < 25, P/B < 5, & ROE > 15%, then makes a long-term investment (`CNC`). Runs once per click.")
        
        capital4 = st.number_input("Investment Amount (₹)", 20000, 200000, 50000, 1000, key="cap4")
        scan_list4 = st.multiselect("Stocks to Scan", nifty50_stocks, default=nifty50_stocks, key="scan4")

        if st.button("▶️ Run Scanner Once", key="run_val_bot"):
            run_value_investor_bot(instrument_df, scan_list4, capital4)
            
        st.write("**Activity Log:**"); log_container4 = st.container(height=200)
        for log in st.session_state.get('bot_value_log', []): log_container4.text(log)

# --- BOT LOGIC FUNCTIONS ---
def log_bot_action(bot_key, message):
    log_list = st.session_state[f'bot_{bot_key}_log']
    log_list.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    if len(log_list) > 50: log_list.pop()

def run_momentum_bot(instrument_df, scan_list, capital):
    bot_key, position = "momentum", st.session_state.get('bot_momentum_position')
    if position:
        data = get_historical_data(get_instrument_token(position['symbol'], instrument_df), '5minute', '5d')
        if not data.empty and 'RSI_14' in data.columns and data['RSI_14'].iloc[-1] < 50:
            place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotMomentumExit")
            log_bot_action(bot_key, f"SELL SIGNAL: RSI < 50. Closing {position['symbol']}.")
            st.session_state.bot_momentum_position = None
            st.rerun()
    else:
        for symbol in scan_list:
            data = get_historical_data(get_instrument_token(symbol, instrument_df), '5minute', '5d')
            if data.empty or not all(k in data.columns for k in ['RSI_14', 'EMA_20']): continue
            latest = data.iloc[-1]
            if latest['close'] > latest['EMA_20'] and latest['RSI_14'] > 60:
                qty = int(capital / latest['close'])
                if qty > 0 and place_order(instrument_df, symbol, qty, 'MARKET', 'BUY', 'MIS', tag="BotMomentumEntry"):
                    log_bot_action(bot_key, f"BUY SIGNAL on {symbol}. Order placed for {qty} shares.")
                    st.session_state.bot_momentum_position = {'symbol': symbol, 'quantity': qty}
                    st.rerun()

def run_mean_reversion_bot(instrument_df, scan_list, capital):
    bot_key, position = "reversion", st.session_state.get('bot_reversion_position')
    if position:
        data = get_historical_data(get_instrument_token(position['symbol'], instrument_df), '5minute', '5d')
        if not data.empty and 'BBM_20_2.0' in data.columns and data['close'].iloc[-1] >= data['BBM_20_2.0'].iloc[-1]:
            place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotReversionExit")
            log_bot_action(bot_key, f"SELL SIGNAL: Price hit middle band. Closing {position['symbol']}.")
            st.session_state.bot_reversion_position = None
            st.rerun()
    else:
        for symbol in scan_list:
            data = get_historical_data(get_instrument_token(symbol, instrument_df), '5minute', '5d')
            if data.empty or 'BBL_20_2.0' not in data.columns: continue
            if data['close'].iloc[-1] <= data['BBL_20_2.0'].iloc[-1]:
                qty = int(capital / data['close'].iloc[-1])
                if qty > 0 and place_order(instrument_df, symbol, qty, 'MARKET', 'BUY', 'MIS', tag="BotReversionEntry"):
                    log_bot_action(bot_key, f"BUY SIGNAL on {symbol}: Price hit lower Bollinger Band.")
                    st.session_state.bot_reversion_position = {'symbol': symbol, 'quantity': qty}
                    st.rerun()

def run_volatility_breakout_bot(instrument_df, scan_list, capital):
    bot_key, position = "breakout", st.session_state.get('bot_breakout_position')
    if position:
        data = get_historical_data(get_instrument_token(position['symbol'], instrument_df), 'day', '6mo')
        if not data.empty and 'DCL_10_20' in data.columns and data['close'].iloc[-1] < data['DCL_10_20'].iloc[-1]:
            place_order(instrument_df, position['symbol'], position['quantity'], 'MARKET', 'SELL', 'MIS', tag="BotBreakoutExit")
            log_bot_action(bot_key, f"STOP SIGNAL: Price broke 10-day low. Closing {position['symbol']}.")
            st.session_state.bot_breakout_position = None
            st.rerun()
    else:
        for symbol in scan_list:
            data = get_historical_data(get_instrument_token(symbol, instrument_df), 'day', '6mo')
            if data.empty or 'DCU_20_20' not in data.columns: continue
            if data['close'].iloc[-1] > data['DCU_20_20'].iloc[-2]:
                qty = int(capital / data['close'].iloc[-1])
                if qty > 0 and place_order(instrument_df, symbol, qty, 'MARKET', 'BUY', 'MIS', tag="BotBreakoutEntry"):
                    log_bot_action(bot_key, f"BUY SIGNAL on {symbol}: Broke 20-day high.")
                    st.session_state.bot_breakout_position = {'symbol': symbol, 'quantity': qty}
                    st.rerun()

def run_value_investor_bot(instrument_df, scan_list, capital):
    bot_key = "value"
    log_bot_action(bot_key, "Scanning for value stocks...")
    for symbol in scan_list:
        if any(inv['symbol'] == symbol for inv in st.session_state.get('bot_value_investments', [])): continue
        fundamentals = get_fundamental_data(symbol)
        if fundamentals:
            pe, pb, roe = fundamentals.get('P/E Ratio', 999), fundamentals.get('P/B Ratio', 99), fundamentals.get('ROE', 0)
            if 0 < pe < 25 and 0 < pb < 5 and roe > 0.15:
                ltp_df = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                if not ltp_df.empty:
                    qty = int(capital / ltp_df.iloc[0]['Price'])
                    if qty > 0 and place_order(instrument_df, symbol, qty, 'MARKET', 'BUY', 'CNC', tag="BotValueEntry"):
                        log_bot_action(bot_key, f"VALUE FIND on {symbol}: P/E={pe:.1f}, ROE={roe*100:.1f}%. Investing.")
                        st.session_state.bot_value_investments.append({'symbol': symbol, 'quantity': qty})
                        st.rerun()
        a_time.sleep(1) # API rate limit
    log_bot_action(bot_key, "Scan complete.")
    st.rerun()

# --- FUNDAMENTAL ANALYTICS PAGE (REPAIRED) ---

@st.cache_data(ttl=3600)
def get_fundamental_data(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        data = {'Company Name': info.get('longName'), 'Sector': info.get('sector'), 'Market Cap': info.get('marketCap'), 'P/E Ratio': info.get('trailingPE'), 'P/B Ratio': info.get('priceToBook'), 'Dividend Yield': info.get('dividendYield'), 'ROE': info.get('returnOnEquity'), 'Debt to Equity': info.get('debtToEquity'), 'Profit Margins': info.get('profitMargins'), 'Revenue Growth': info.get('revenueGrowth')}
        return {k: v for k, v in data.items() if v is not None}
    except Exception:
        return None

def get_financial_statement(symbol, statement_type):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        df = getattr(ticker, statement_type, pd.DataFrame())
        if df.empty: return None
        
        if statement_type == 'balance_sheet': cols = ['Total Assets', 'Current Assets', 'Current Liabilities', 'Total Stock
