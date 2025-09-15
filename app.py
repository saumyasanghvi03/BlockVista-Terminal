# app.py - BlockVista Terminal Production-ready
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from datetime import datetime
import requests
import threading

st.set_page_config(page_title="BlockVista Terminal", layout="wide")

# ---- CONFIGURATION ----
API_KEY = "your_kite_api_key"
API_SECRET = "your_kite_api_secret"
ACCESS_TOKEN = "your_access_token"  # Can refresh manually first time

# ---- KITE SESSION HELPER ----
def get_kite_session():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

kite = get_kite_session()
st.success("✅ Kite session initialized")

# ---- NOTIFICATION HELPER ----
def browser_notification(title, body):
    st.write(f"🔔 {title}: {body}")

def telegram_notification(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except:
        pass

# ---- LIVE TICKER MANAGER ----
class LiveTickerManager:
    def __init__(self, kite_session):
        self.kite = kite_session
        self.data = {}  # symbol -> DataFrame
        self.lock = threading.Lock()

    def subscribe(self, symbols):
        for symbol in symbols:
            if symbol not in self.data:
                self.data[symbol] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    def update(self, symbol):
        try:
            ltp = self.kite.ltp(f"NSE:{symbol}")  # get last price
            now = datetime.now()
            row = {
                "timestamp": now,
                "open": ltp[f"NSE:{symbol}"]['ohlc']['open'],
                "high": ltp[f"NSE:{symbol}"]['ohlc']['high'],
                "low": ltp[f"NSE:{symbol}"]['ohlc']['low'],
                "close": ltp[f"NSE:{symbol}"]['last_price'],
                "volume": ltp[f"NSE:{symbol}"]['volume']
            }
            with self.lock:
                self.data[symbol] = pd.concat([self.data[symbol], pd.DataFrame([row])], ignore_index=True)
                # Keep last 500 rows to optimize memory
                if len(self.data[symbol]) > 500:
                    self.data[symbol] = self.data[symbol].iloc[-500:]
        except Exception as e:
            st.error(f"Error fetching {symbol} data: {e}")

    def get_aggregated(self, symbol):
        return self.data.get(symbol, pd.DataFrame())

# ---- TECHNICAL INDICATORS ----
def add_indicators(df):
    df = df.copy()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['BB_upper'], df['BB_lower'] = compute_bollinger(df['close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std*std
    lower = sma - num_std*std
    return upper, lower

# ---- CHART RENDERING ----
def render_chart(symbol, df):
    if df.empty:
        st.warning(f"No data yet for {symbol}")
        return

    df = add_indicators(df)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=symbol
    ))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_10'], line=dict(color='blue', width=1), name="SMA 10"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], line=dict(color='green', width=1), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_upper'], line=dict(color='red', width=1, dash='dot'), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_lower'], line=dict(color='red', width=1, dash='dot'), name="BB Lower"))
    fig.update_layout(title=f"{symbol} Live Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- SYMBOLS AND REFRESH ----
symbols = st.multiselect("Select Symbols", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"], default=["RELIANCE"])
refresh_interval = st.slider("Refresh Interval (seconds)", min_value=5, max_value=30, value=10)

ltm = LiveTickerManager(kite)
ltm.subscribe(symbols)

# ---- BACKGROUND THREAD TO UPDATE DATA ----
def background_fetch():
    while True:
        for sym in symbols:
            ltm.update(sym)
        time.sleep(refresh_interval)

thread = threading.Thread(target=background_fetch, daemon=True)
thread.start()

# ---- DISPLAY TABS PER SYMBOL ----
tabs = st.tabs(symbols)
for i, symbol in enumerate(symbols):
    with tabs[i]:
        df = ltm.get_aggregated(symbol)
        render_chart(symbol, df)

        # Screener metrics table
        st.write("Latest Data")
        st.dataframe(df.tail(5))

        # CSV Download
        st.download_button(f"Download CSV {symbol}", df.to_csv(index=False), f"{symbol}_data.csv", "text/csv")

        # Alerts Example
        if not df.empty and df['RSI'].iloc[-1] > 70:
            browser_notification(symbol, "RSI overbought!")
        elif not df.empty and df['RSI'].iloc[-1] < 30:
            browser_notification(symbol, "RSI oversold!")

# ---- AUTO REFRESH ----
st_autorefresh(interval=refresh_interval*1000, key="ticker_refresh")

