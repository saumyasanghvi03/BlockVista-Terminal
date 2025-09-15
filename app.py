# Updated app.py for BlockVista Terminal with quick fixes, streamlined logic, and live Zerodha + yfinance/Alpha Vantage integration

# --- Keep existing imports ---
import os, time, json, threading
from collections import deque
from datetime import datetime, timedelta
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from kiteconnect import KiteConnect, KiteTicker
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries
import plotly.graph_objs as go

# --- CONFIG ---
AV_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '2R0I2OXW1A1HMD9N')
MAX_TICKS_PER_SYMBOL = 4000
MAX_AGG_CANDLES = 500

# --- UTILITIES ---
def browser_notification(title, body, icon=None):
    icon_line = f'icon: "{icon}",' if icon else ''
    st.markdown(f"""
    <script>
    if(Notification.permission!="granted"){{Notification.requestPermission();}}
    if(Notification.permission=="granted"){{new Notification("{title}",{{body:"{body}",{icon_line}}});}}
    </script>""", unsafe_allow_html=True)

# --- Alpha Vantage fallback ---
def fetch_alpha_vantage_intraday(symbol, interval='1min', outputsize='compact'):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        symbol_av = symbol.upper()+'.NSE' if not symbol.upper().endswith('.NSE') else symbol.upper()
        data, _ = ts.get_intraday(symbol=symbol_av, interval=interval, outputsize=outputsize)
        data.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. volume':'Volume'}, inplace=True)
        data.index = pd.to_datetime(data.index)
        return data.sort_index()
    except: return None

# --- Streamlit Page Config ---
st.set_page_config(page_title="BLOCKVISTA TERMINAL", layout="wide")
if 'dark_theme' not in st.session_state: st.session_state.dark_theme=True

def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v3fvcr {background-color: #121212 !important;}
        .stSidebar, .css-15zrgzn {background: #191919 !important;}
        .stDataFrame tbody tr {background-color: #191919 !important;color: #FFD900 !important;}
        .stMetric, .stMetricLabel, .stMetricValue {color: #FFD900 !important;}
        h1,h2,h3,h4,h5,h6,label {color: #FFD900 !important;}
        </style>
        """, unsafe_allow_html=True)

# --- Terminal Theme ---
theme_choice = st.sidebar.selectbox("Terminal Theme", ["Black/Yellow/Green", "Streamlit Default"])
if theme_choice=="Black/Yellow/Green": set_terminal_style(True)

# --- Auto Refresh ---
st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.get('auto_refresh', True))
st.session_state.refresh_interval = st.sidebar.number_input("Refresh Sec", value=st.session_state.get('refresh_interval', 15), min_value=3, max_value=90, step=1)

# --- Zerodha Credentials ---
def get_api_creds():
    api_key = st.secrets.get("KITE_API_KEY") or os.getenv("KITE_API_KEY")
    api_secret = st.secrets.get("KITE_API_SECRET") or os.getenv("KITE_API_SECRET")
    refresh_token = st.secrets.get("KITE_REFRESH_TOKEN") or os.getenv("KITE_REFRESH_TOKEN")
    return api_key, api_secret, refresh_token

api_key, api_secret, refresh_token = get_api_creds()
if not api_key or not api_secret: st.error("Zerodha API credentials missing."); st.stop()

# --- Kite Session ---
def get_kite_session():
    kite_conn = KiteConnect(api_key=api_key)
    if 'access_token' in st.session_state:
        kite_conn.set_access_token(st.session_state['access_token'])
        return kite_conn
    if refresh_token:
        try:
            data = kite_conn.generate_session(refresh_token, api_secret=api_secret)
            st.session_state['access_token'] = data.get('access_token')
            kite_conn.set_access_token(st.session_state['access_token'])
            st.success("✅ Zerodha session auto-refreshed")
            return kite_conn
        except: st.warning("Auto-refresh failed.")
    login_url = kite_conn.login_url()
    st.markdown(f'<a href="{login_url}" target="_blank">Login & authorize BlockVista</a>', unsafe_allow_html=True)
    request_token = st.text_input("Paste request_token here:")
    if st.button("Generate Access Token") and request_token:
        try:
            data = kite_conn.generate_session(request_token, api_secret=api_secret)
            st.session_state['access_token'] = data.get('access_token')
            kite_conn.set_access_token(st.session_state['access_token'])
            st.success("Session started!")
            st.code(f"REFRESH_TOKEN: {data.get('refresh_token')}")
            return kite_conn
        except Exception as ex: st.error(f"Login failed: {ex}"); st.stop()
    st.stop()

kite = get_kite_session()

# --- Live Price / Screener Helpers ---
def get_live_price(symbol):
    try: return kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
    except: return np.nan

@st.cache_data(show_spinner=True)
def fetch_stock_data(symbol, period='1d', interval='5m'):
    df = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
    if df.empty: df = fetch_alpha_vantage_intraday(symbol, interval=interval)
    if df is not None and not df.empty:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_12_26_9'] = ta.macd(df['Close'])['MACD_12_26_9']
        df['SMA21'] = ta.sma(df['Close'], length=21)
    return df

# --- Watchlist P&L Tracker (Sidebar) ---
watchlist = st.sidebar.text_area("List NSE symbols", "RELIANCE, SBIN, TCS")
entry_prices = [float(x) for x in st.sidebar.text_area("Entry prices", "2550, 610, 3580").split(",")]
quantities = [float(x) for x in st.sidebar.text_area("Quantities", "10,20,5").split(",")]
symbols = [x.strip().upper() for x in watchlist.split(",")]
pnl_data = []
for i,s in enumerate(symbols):
    try:
        live = get_live_price(s) or fetch_stock_data(s)['Close'].iloc[-1]
        pnl_data.append({'Symbol':s,'Entry':entry_prices[i],'LTP':live,'Qty':quantities[i],'P&L ₹':round((live-entry_prices[i])*quantities[i],2)})
    except: pnl_data.append({'Symbol':s,'Entry':entry_prices[i],'LTP':'Err','Qty':quantities[i],'P&L ₹':'Err'})
if pnl_data: st.sidebar.dataframe(pd.DataFrame(pnl_data))

# --- KiteTicker Live (dynamic subscribe) ---
if 'live_ticks' not in st.session_state: st.session_state['live_ticks']={}
if 'ohlc_agg' not in st.session_state: st.session_state['ohlc_agg']={}
if 'token_to_symbol' not in st.session_state: st.session_state['token_to_symbol']={}
if 'kws_obj' not in st.session_state: st.session_state['kws_obj']=None
if 'kws_running' not in st.session_state: st.session_state['kws_running']=False
if 'subscribed_tokens' not in st.session_state: st.session_state['subscribed_tokens']=set()

# --- Lightweight Charts Renderer ---
def render_lightweight_candles(symbol):
    df = pd.DataFrame(st.session_state['ohlc_agg'].get(symbol,{})).T
    if df.empty: st.info("No live data yet"); return
    df['time'] = pd.to_datetime(df['time'], unit='s')
    js_data = df[['time','open','high','low','close']].to_dict(orient='records
