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
    
    if 'algo_bots' not in st.session_state:
        st.session_state.algo_bots = {
            "momentum_trader": {
                "name": "Momentum Trader",
                "description": "Tracks stocks with strong momentum using RSI and moving averages",
                "status": "inactive", "capital": 1000, "trades": [], "pnl": 0, "active_since": None,
                "auto_trade_enabled": False, "position": None, "entry_price": 0
            },
            "mean_reversion": {
                "name": "Mean Reversion",
                "description": "Capitalizes on price reversions using Bollinger Bands and RSI",
                "status": "inactive", "capital": 1000, "trades": [], "pnl": 0, "active_since": None,
                "auto_trade_enabled": False, "position": None, "entry_price": 0
            },
            "volatility_breakout": {
                "name": "Volatility Breakout", 
                "description": "Identifies breakout opportunities using volatility channels",
                "status": "inactive", "capital": 1000, "trades": [], "pnl": 0, "active_since": None,
                "auto_trade_enabled": False, "position": None, "entry_price": 0
            },
            "value_investor": {
                "name": "Value Investor",
                "description": "Focuses on fundamentally strong stocks with good valuations",
                "status": "inactive", "capital": 1000, "trades": [], "pnl": 0, "active_since": None,
                "auto_trade_enabled": False, "position": None, "entry_price": 0
            }
        }

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays_2025 = ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays_2025:
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if market_open_time <= now.time() <= market_close_time:
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    col1, col2 = st.columns([3, 1])
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
    
@st.cache_data(ttl=600)
def get_global_indices_data(tickers):
    data_list = []
    ticker_string = " ".join(tickers.values())
    try:
        yf_data = yf.download(ticker_string, period="2d", group_by='ticker')
        if yf_data.empty:
            return pd.DataFrame()
        for name, ticker in tickers.items():
            if ticker in yf_data.columns.get_level_values(0):
                stock_data = yf_data[ticker].dropna()
                if not stock_data.empty and len(stock_data) > 1:
                    last_price = stock_data['Close'].iloc[-1]
                    prev_close = stock_data['Close'].iloc[-2]
                    change_pct = ((last_price - prev_close) / prev_close) * 100
                    data_list.append({'Ticker': name, 'Price': last_price, '% Change': change_pct})
    except Exception as e:
        st.error(f"Error fetching global indices data: {e}")
    return pd.DataFrame(data_list)

def display_overnight_changes_bar():
    overnight_tickers = {"GIFT NIFTY": "IN=F", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
    data = get_global_indices_data(overnight_tickers)
    if not data.empty:
        bar_html = "<div class='notification-bar'>"
        for _, row in data.iterrows():
            color = 'var(--green)' if row['% Change'] > 0 else 'var(--red)'
            bar_html += f"<span>{row['Ticker']}: {row['Price']:,.2f} <span style='color:{color};'>({row['% Change']:+.2f}%)</span></span>"
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

@st.cache_resource(ttl=3600)
def get_instrument_df():
    client = get_broker_client()
    if not client: return pd.DataFrame()
    df = pd.DataFrame(client.instruments())
    if 'expiry' in df.columns:
        df['expiry'] = pd.to_datetime(df['expiry'])
    return df

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period='5y'):
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    try:
        to_date = datetime.now().date()
        days_map = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
        from_date = to_date - timedelta(days=days_map.get(period, 1825))
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.ta.adx(append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
    try:
        quotes = client.quote(instrument_names)
        watchlist = []
        for item in symbols_with_exchange:
            q = quotes.get(f"{item['exchange']}:{item['symbol']}")
            if q:
                change = q['last_price'] - q['ohlc']['close']
                pct_change = (change / q['ohlc']['close'] * 100) if q['ohlc']['close'] != 0 else 0
                watchlist.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': q['last_price'], 'Change': change, '% Change': pct_change})
        return pd.DataFrame(watchlist)
    except Exception:
        return pd.DataFrame()

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    try:
        match = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
        if match.empty:
            st.error(f"Symbol '{symbol}' not found.")
            return
        exchange = match.iloc[0]['exchange']
        
        order_id = client.place_order(
            tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type,
            quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price
        )
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")

# ================ 4. ALGO TRADING BOTS ================

def momentum_trader_strategy(symbol, capital, instrument_df):
    token = get_instrument_token(symbol, instrument_df)
    if not token: return None, "Instrument not found"
    data = get_historical_data(token, '5minute', period='1d')
    if data.empty or len(data) < 20: return None, "Insufficient data"
    
    latest = data.iloc[-1]
    # Check if 'RSI_14' column exists
    if 'RSI_14' not in data.columns:
        return "HOLD", "RSI indicator not available"

    if (latest['RSI_14'] > 60 and latest['close'] > data['close'].rolling(20).mean().iloc[-1]):
        return "BUY", f"RSI({latest['RSI_14']:.1f}) > 60 and Price > 20-SMA"
    elif (latest['RSI_14'] < 40):
        return "SELL", f"RSI({latest['RSI_14']:.1f}) < 40"
    return "HOLD", "Neutral"

def mean_reversion_strategy(symbol, capital, instrument_df):
    token = get_instrument_token(symbol, instrument_df)
    if not token: return None, "Instrument not found"
    data = get_historical_data(token, '5minute', period='1d')
    if data.empty or len(data) < 20: return None, "Insufficient data"
    
    bbands = ta.bbands(data['close'], length=20, std=2)
    if bbands is None or 'BBL_20_2.0' not in bbands.columns or 'BBU_20_2.0' not in bbands.columns: 
        return None, "Could not calculate Bollinger Bands"
    data = pd.concat([data, bbands], axis=1)
    latest = data.iloc[-1]
    
    # Check if 'RSI_14' column exists
    if 'RSI_14' not in data.columns:
        return "HOLD", "RSI indicator not available"

    if latest['close'] <= latest['BBL_20_2.0'] and latest['RSI_14'] < 35:
        return "BUY", f"Price at Lower BB and RSI({latest['RSI_14']:.1f}) < 35"
    elif latest['close'] >= latest['BBU_20_2.0'] and latest['RSI_14'] > 65:
        return "SELL", f"Price at Upper BB and RSI({latest['RSI_14']:.1f}) > 65"
    return "HOLD", "Neutral"

def volatility_breakout_strategy(symbol, capital, instrument_df):
    token = get_instrument_token(symbol, instrument_df)
    if not token: return None, "Instrument not found"
    data = get_historical_data(token, '5minute', period='1d')
    if data.empty or len(data) < 20: return None, "Insufficient data"

    if data['close'].iloc[-1] > data['high'].rolling(20).max().iloc[-2]:
        return "BUY", "Breakout above 20-period high"
    elif data['close'].iloc[-1] < data['low'].rolling(20).min().iloc[-2]:
        return "SELL", "Breakdown below 20-period low"
    return "HOLD", "Neutral"

def value_investor_strategy(symbol, capital, instrument_df):
    token = get_instrument_token(symbol, instrument_df)
    if not token: return None, "Instrument not found"
    data = get_historical_data(token, 'day', period='1y')
    if data.empty or len(data) < 200: return None, "Insufficient data"

    sma_50 = data['close'].rolling(50).mean().iloc[-1]
    sma_200 = data['close'].rolling(200).mean().iloc[-1]
    latest_price = data['close'].iloc[-1]

    if latest_price < sma_50 and latest_price > sma_200:
        return "BUY", "Price is below 50-day SMA but above 200-day SMA"
    elif latest_price > sma_50 * 1.2: # 20% above 50-day SMA as a simple overvalued metric
        return "SELL", "Price is overextended from 50-day SMA"
    return "HOLD", "Neutral"


def execute_algo_trade(bot_id, symbol, signal, capital, instrument_df):
    if signal not in ["BUY", "SELL"]: return False, "No valid signal"
    
    quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
    if quote_data.empty: return False, "Could not fetch current price"
    
    current_price = quote_data.iloc[0]['Price']
    quantity = max(1, int(capital / current_price))
    
    place_order(instrument_df, symbol, quantity, 'MARKET', signal, 'MIS')
    
    bot = st.session_state.algo_bots[bot_id]
    trade = {'timestamp': datetime.now(), 'symbol': symbol, 'signal': signal, 'quantity': quantity, 'price': current_price}
    
    if signal == "BUY":
        bot["position"] = "LONG"
        bot["entry_price"] = current_price
    elif signal == "SELL" and bot.get("position") == "LONG":
        profit = (current_price - bot.get("entry_price", 0)) * quantity
        bot["pnl"] = bot.get("pnl", 0) + profit
        trade['pnl'] = profit
        bot["position"] = None
        bot["entry_price"] = 0

    bot['trades'].append(trade)
    return True, f"Executed {signal} for {quantity} {symbol}"

def run_active_auto_traders(instrument_df, bots):
    if instrument_df.empty: return
    sample_symbol = "RELIANCE"

    for bot_id, bot in bots.items():
        if bot.get("status") == "active" and bot.get("auto_trade_enabled", False):
            strategy_func = {
                "momentum_trader": momentum_trader_strategy,
                "mean_reversion": mean_reversion_strategy,
                "volatility_breakout": volatility_breakout_strategy,
                "value_investor": value_investor_strategy
            }.get(bot_id)
            
            if strategy_func:
                signal, reason = strategy_func(sample_symbol, bot["capital"], instrument_df)
                if not signal: continue

                if bot.get("position") is None and signal == "BUY":
                    success, msg = execute_algo_trade(bot_id, sample_symbol, "BUY", bot["capital"], instrument_df)
                    if success: st.toast(f"{bot['name']} entered LONG on {sample_symbol}", icon="📈")
                elif bot.get("position") == "LONG" and signal == "SELL":
                    success, msg = execute_algo_trade(bot_id, sample_symbol, "SELL", bot["capital"], instrument_df)
                    if success: st.toast(f"{bot['name']} exited LONG on {sample_symbol}", icon="📉")

# ================ 5. PAGE DEFINITIONS ============

def page_algo_trading_bots():
    display_header()
    st.title("🤖 Algo Trading Bots")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo trading bots.")
        return
        
    run_active_auto_traders(instrument_df, st.session_state.algo_bots)
    
    bots = st.session_state.algo_bots
    for bot_id, bot in bots.items():
        bot_class = "bot-active" if bot.get("status") == "active" else "bot-inactive"
        if bot.get("pnl", 0) > 0: bot_class = "bot-profit"
        elif bot.get("pnl", 0) < 0: bot_class = "bot-loss"
            
        with st.container():
            st.markdown(f'<div class="trade-card {bot_class}">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.subheader(bot["name"])
                st.caption(bot["description"])
            with c2:
                st.metric("Capital", f"₹{bot['capital']:,.0f}")
                st.metric("P&L", f"₹{bot.get('pnl', 0):,.2f}")
            with c3:
                # Use a key to ensure the toggle state is managed correctly
                auto_trade_key = f"auto_{bot_id}"
                is_auto_trading = st.toggle('Enable Auto-Trading', value=st.session_state.algo_bots[bot_id].get('auto_trade_enabled', False), key=auto_trade_key)
                st.session_state.algo_bots[bot_id]['auto_trade_enabled'] = is_auto_trading

                if is_auto_trading:
                    st.success("Auto ON")
                else:
                    st.warning("Auto OFF")

                if bot.get("status") == "active":
                    if st.button("🛑 Stop", key=f"stop_{bot_id}"):
                        st.session_state.algo_bots[bot_id]["status"] = "inactive"
                        st.session_state.algo_bots[bot_id]["auto_trade_enabled"] = False
                        st.rerun()
                else:
                    if st.button("▶️ Start", key=f"start_{bot_id}"):
                        st.session_state.algo_bots[bot_id]["status"] = "active"
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def page_dashboard():
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    
    st.subheader("Watchlist")
    active_list = st.session_state.watchlists["Watchlist 1"]
    watchlist_data = get_watchlist_data(active_list)
    if not watchlist_data.empty:
        for _, row in watchlist_data.iterrows():
            c1, c2, c3, c4, c5 = st.columns([3,2,1,1,1])
            color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
            c1.markdown(f"**{row['Ticker']}**")
            c2.markdown(f"<span style='color:{color};'>{row['Price']:,.2f} ({row['% Change']:.2f}%)</span>", unsafe_allow_html=True)
            qty = c3.number_input("Qty", 1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
            if c4.button("B", key=f"b_{row['Ticker']}"): place_order(instrument_df, row['Ticker'], qty, 'MARKET', 'BUY', 'MIS')
            if c5.button("S", key=f"s_{row['Ticker']}"): place_order(instrument_df, row['Ticker'], qty, 'MARKET', 'SELL', 'MIS')

def page_not_implemented():
    st.title("Under Construction")
    st.info("This feature is not yet available.")

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

def login_page():
    apply_custom_styling()
    st.title("BlockVista Terminal Login")
    
    broker = st.selectbox("Select Broker", ["Zerodha"])
    api_key = st.secrets.get("ZERODHA_API_KEY")
    api_secret = st.secrets.get("ZERODHA_API_SECRET")

    if not api_key or not api_secret:
        st.error("Zerodha API credentials are not set in Streamlit secrets.")
        st.stop()
        
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
        st.link_button("Login with Zerodha", kite.login_url())

def main_app():
    apply_custom_styling()
    display_overnight_changes_bar()
    
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    mode = st.sidebar.radio("Mode", ["Cash", "Algo Bots"], horizontal=True)
    st.sidebar.divider()

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
        
    pages = {
        "Cash": page_dashboard,
        "Algo Bots": page_algo_trading_bots,
    }
    
    st_autorefresh(interval=15 * 1000, key="data_refresher")
    
    pages.get(mode, page_not_implemented)()

# --- Application Entry Point ---
if __name__ == "__main__":
    initialize_session_state()
    
    if 'profile' in st.session_state and st.session_state.profile:
        main_app()
    else:
        login_page()

