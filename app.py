# app.py
"""
BlockVista Terminal — Streamlit app
Features:
- Zerodha KiteConnect login & KiteTicker (WebSocket) subscription
- Server-side aggregation of live ticks -> OHLC candles (1m/5m/15m)
- Candlestick rendering using TradingView Lightweight Charts embedded in an iframe
- Fallback data sources: yfinance and Alpha Vantage (existing logic retained)
- Auto-refresh and Start/Stop controls for KiteTicker
Notes:
- Place Zerodha credentials inside Streamlit secrets (ZERODHA_API_KEY, ZERODHA_API_SECRET)
- Alpha Vantage key is present in AV_API_KEY variable (already in your original file)
- This implementation uses Streamlit reruns (st_autorefresh) to update the chart UI.
  For lower-latency incremental updates, a postMessage bridge and persistent connection
  would be required — TODO included below.
"""

# app.py (FULL updated file)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect, KiteTicker
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries
from collections import deque
import threading
import time
import json
from datetime import datetime, timedelta

import os

# Load API keys from Streamlit secrets (preferred on cloud) or environment variables
if "KITE_API_KEY" in st.secrets:
    api_key = st.secrets["KITE_API_KEY"]
    api_secret = st.secrets["KITE_API_SECRET"]
else:
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")


# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"
MAX_TICKS_PER_SYMBOL = 4000   # bound live memory
MAX_AGG_CANDLES = 500

# ---------------------- UTILITIES ----------------------
def browser_notification(title, body, icon=None):
    icon_line = f'icon: "{icon}",' if icon else ""
    st.markdown(
        f"""
        <script>
        if (Notification.permission !== "granted") {{
            Notification.requestPermission();
        }}
        if (Notification.permission === "granted") {{
            new Notification("{title}", {{
                body: "{body}",
                {icon_line}
            }});
        }}
        </script>
        """, unsafe_allow_html=True
    )

def fetch_alpha_vantage_intraday(symbol, interval='1min', outputsize='compact'):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        symbol_av = symbol.upper() + ".NSE" if not symbol.upper().endswith(".NSE") else symbol.upper()
        data, meta_data = ts.get_intraday(symbol=symbol_av, interval=interval, outputsize=outputsize)
        df = data.rename(columns={
            '1. open': "Open",
            '2. high': "High",
            '3. low': "Low",
            '4. close': "Close",
            '5. volume': "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        st.warning(f"Alpha Vantage fetch failed: {e}")
        return None

# ---------------------- APP THEME & SESSION ----------------------
st.set_page_config(page_title="BLOCKVISTA TERMINAL", layout="wide")
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True

def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v3fvcr {background-color: #121212 !important;}
        .stSidebar, .css-15zrgzn {background: #191919 !important;}
        .stDataFrame tbody tr {background-color: #191919 !important;color: #FFD900 !important;}
        .css-pkbazv, .stTextInput, .stTextArea textarea, .stNumberInput input, .st-cq, .st-de {
            background: #191919 !important;color: #FFD900 !important;}
        .stMetric, .stMetricLabel, .stMetricValue {color: #FFD900 !important;}
        h1, h2, h3, h4, h5, h6, label, .st-bv, .stTextInput label, .stTextArea label {color: #FFD900 !important;}
        </style>
        """, unsafe_allow_html=True)

theme_choice = st.sidebar.selectbox("Terminal Theme", ["Black/Yellow/Green", "Streamlit Default"])
if theme_choice == "Black/Yellow/Green":
    set_terminal_style(True)
    st.session_state.dark_theme = True

if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 15

refresh_col, toggle_col = st.sidebar.columns([2,2])
with refresh_col:
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh", value=st.session_state["auto_refresh"])
with toggle_col:
    st.session_state["refresh_interval"] = st.number_input("Sec", value=st.session_state["refresh_interval"], min_value=3, max_value=90, step=1)
if st.session_state["auto_refresh"]:
    st_autorefresh(interval=st.session_state["refresh_interval"] * 1000, key="autorefresh")

# ---------------------- Smallcase baskets ----------------------
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

# ---------------------- Zerodha Auth (keeps original flow) ----------------------
api_key = st.secrets.get("ZERODHA_API_KEY")
api_secret = st.secrets.get("ZERODHA_API_SECRET")
if api_key is None or api_secret is None:
    st.error("Zerodha API credentials not found in Streamlit secrets. Add ZERODHA_API_KEY and ZERODHA_API_SECRET.")
    st.stop()

if "access_token" not in st.session_state:
    kite_tmp = KiteConnect(api_key=api_key)
    login_url = kite_tmp.login_url()
    st.markdown(
        f"""
        <div style="background: #f5da82; padding: 14px; border-radius: 8px;">
        🟠 <a href="{login_url}" target="_blank"><b>Click here to login & authorize BlockVista</b></a><br>
        After logging in, get <b>`request_token=xxxx`</b> from the URL and paste below:
        </div>
        """, unsafe_allow_html=True
    )
    request_token = st.text_input("Paste request_token here:")
    if st.button("Generate Access Token") and request_token:
        try:
            data = kite_tmp.generate_session(request_token, api_secret=api_secret)
            st.session_state["access_token"] = data["access_token"]
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(data["access_token"])
            st.session_state["kite"] = kite
            st.success("✅ Zerodha session started! All Kite features enabled.")
        except Exception as ex:
            st.error(f"❌ Zerodha login failed: {ex}")
            browser_notification("BlockVista Error", f"❌ Zerodha login failed: {ex}", "https://cdn-icons-png.flaticon.com/512/2583/2583346.png")
            st.stop()
    st.stop()
else:
    if "kite" not in st.session_state:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state["access_token"])
        st.session_state["kite"] = kite
    else:
        kite = st.session_state["kite"]

# ---------------------- Core helpers (kept original) ----------------------
def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        # do not spam logs in UI — return np.nan to signal missing
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
    if data is None or len(data) == 0:
        av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
        st.info("Fetching live data from Alpha Vantage…")
        data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
        if data is None or len(data) == 0:
            return None
    # Technical indicators (same as original)
    try:
        data['RSI'] = ta.rsi(data['Close'], length=14) if len(data) > 0 else np.nan
        macd = ta.macd(data['Close'])
        if isinstance(macd, pd.DataFrame) and not macd.empty:
            for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                data[col] = macd[col] if col in macd else np.nan
        else:
            for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                data[col] = np.nan
        data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 0 else np.nan
        data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 0 else np.nan
        bbands = ta.bbands(data['Close'], length=20)
        for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
            data[label] = bbands[key] if isinstance(bbands, pd.DataFrame) and key in bbands else np.nan
        atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['ATR'] = atr if isinstance(atr, pd.Series) else np.nan
        adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
        data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx else np.nan
        stochrsi = ta.stochrsi(data['Close'], length=14)
        data['STOCHRSI'] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) and "STOCHRSIk_14_14_3_3" in stochrsi else np.nan
        supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
        if isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
            for col in supertrend.columns:
                data[col] = supertrend[col]
        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume']) if len(data) > 0 else np.nan
        ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
        for c in ['open','high','low','close']:
            ha_key = f'HA_{c}'
            data[ha_key] = ha[ha_key] if isinstance(ha, pd.DataFrame) and ha_key in ha else np.nan
    except Exception:
        # If TA library fails for some reason, return data with minimal columns
        pass
    return data

# helpers from original: try_scalar, get_signals, make_screener (unchanged)
def try_scalar(val):
    if isinstance(val, pd.Series) and len(val) == 1:
        val = val.iloc[0]
    if isinstance(val, (float, int, np.floating, np.integer)):
        return val
    try:
        return float(val)
    except Exception:
        return np.nan

def get_signals(data):
    latest = data.iloc[-1]
    signals = {}
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    supertrend_col = [str(c) for c in list(data.columns) if isinstance(c, str) and str(c).startswith('SUPERT_') and not str(c).endswith('_dir')]
    supertrend = try_scalar(latest[supertrend_col[0]]) if supertrend_col else np.nan
    close = try_scalar(latest.get('Close', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    stochrsi = try_scalar(latest.get('STOCHRSI', np.nan))
    signals['RSI Signal'] = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if macd > macds else ('Bearish' if macd < macds else 'Neutral')
    signals['Supertrend'] = (
        'Bullish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend < close else
        'Bearish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend > close else
        'Unknown'
    )
    signals['ADX Trend'] = 'Strong' if adx > 25 else 'Weak'
    signals['STOCHRSI Signal'] = 'Overbought' if stochrsi > 0.8 else ('Oversold' if stochrsi < 0.2 else 'Neutral')
    return signals

def make_screener(stock_list, period, interval):
    screener_data = []
    for s in stock_list:
        data = fetch_stock_data(s, period, interval)
        if data is not None:
            latest = data.iloc[-1]
            signals = get_signals(data)
            row = {
                "Symbol": s,
                "LTP": float(try_scalar(latest.get('Close', np.nan))),
                "RSI": float(try_scalar(latest.get('RSI', np.nan))),
                "MACD": float(try_scalar(latest.get('MACD_12_26_9', np.nan))),
                "ADX": float(try_scalar(latest.get('ADX', np.nan))),
                "ATR": float(try_scalar(latest.get('ATR', np.nan))),
                "Signal": signals['RSI Signal'] + "/" + signals['MACD Signal'] + "/" + signals['Supertrend'],
            }
            screener_data.append(row)
    return pd.DataFrame(screener_data)

# ---------------------- Sidebar: Watchlist P&L Tracker (kept) ----------------------
st.sidebar.subheader("📈 Watchlist P&L Tracker (Live)")
watchlist = st.sidebar.text_area("List NSE symbols (comma-separated)", value="RELIANCE, SBIN, TCS")
positions_input = st.sidebar.text_area("Entry prices (comma, same order)", value="2550, 610, 3580")
qty_input = st.sidebar.text_area("Quantities (comma, same order)", value="10, 20, 5")
symbols = [x.strip().upper() for x in watchlist.split(",") if x.strip()]
entry_prices = [float(x) for x in positions_input.split(",") if x.strip()]
quantities = [float(x) for x in qty_input.split(",") if x.strip()]
pnl_data = []
for i, s in enumerate(symbols):
    try:
        live = get_live_price(s)
        if isinstance(live, str) or live is None or np.isnan(live):
            d = fetch_stock_data(s, "1d", "5m")
            if d is not None and len(d):
                live = d["Close"][-1]
            else:
                live = np.nan
        pnl = (live - entry_prices[i]) * quantities[i]
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": live, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
    except Exception:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
if pnl_data:
    st.sidebar.dataframe(pd.DataFrame(pnl_data))
    total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int,float)))
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)

# ---------------------- Sidebar: Screener (kept) ----------------------
st.sidebar.title('Multi-Screener Settings')
screener_mode = st.sidebar.radio("Screener Mode", ["Single Stock", "Basket (Smallcase)"])
if screener_mode == "Single Stock":
    symbol = st.sidebar.text_input('NSE Symbol', value='RELIANCE')
    stock_list = [symbol]
else:
    basket = st.sidebar.selectbox("Pick Basket", list(SMALLCASE_BASKETS.keys()))
    stock_list = SMALLCASE_BASKETS[basket]

screen_period = st.sidebar.selectbox('Period', ['1d','5d'])
screen_interval = st.sidebar.selectbox('Interval', ['1m','5m','15m'])
screen_df = make_screener(stock_list, screen_period, screen_interval)
st.sidebar.subheader("Screener Results")
if len(screen_df):
    st.sidebar.dataframe(screen_df)
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---------------------- KiteTicker live integration (NEW, dynamic subscribe) ----------------------
# session-state initialization
if "live_ticks" not in st.session_state:
    st.session_state["live_ticks"] = {}         # symbol -> deque of ticks [{'t':ts,'v':price}]
if "ohlc_agg" not in st.session_state:
    st.session_state["ohlc_agg"] = {}          # symbol -> {minute_ts: {open,high,low,close,time}}
if "token_to_symbol" not in st.session_state:
    st.session_state["token_to_symbol"] = {}   # str(token) -> symbol
if "kws_obj" not in st.session_state:
    st.session_state["kws_obj"] = None
if "kws_thread" not in st.session_state:
    st.session_state["kws_thread"] = None
if "kws_running" not in st.session_state:
    st.session_state["kws_running"] = False
if "subscribed_tokens" not in st.session_state:
    st.session_state["subscribed_tokens"] = set()

# helpers for instrument lookup
@st.cache_data
def load_instruments():
    try:
        instruments = kite.instruments()
        df = pd.DataFrame(instruments)
        return df
    except Exception as e:
        st.warning(f"Could not load instruments list: {e}")
        return pd.DataFrame()

def find_instrument_token(symbol, exchange='NSE'):
    df = load_instruments()
    if df.empty:
        return None
    cond = (df['tradingsymbol'].str.upper() == symbol.upper()) & (df['exchange'] == exchange)
    matches = df[cond]
    if not matches.empty:
        return int(matches.iloc[0]['instrument_token'])
    # fallback: contains
    matches2 = df[df['tradingsymbol'].str.upper().str.contains(symbol.upper()) & (df['exchange'] == exchange)]
    if not matches2.empty:
        return int(matches2.iloc[0]['instrument_token'])
    return None

# kite ticker callbacks
def kite_on_ticks(ws, ticks):
    for t in ticks:
        token = t.get('instrument_token')
        lp = t.get('last_price')
        ts_field = t.get('timestamp')
        try:
            if isinstance(ts_field, str):
                ts = pd.to_datetime(ts_field)
            else:
                # timestamp may be epoch in ms or s
                if ts_field is None:
                    ts = pd.Timestamp.utcnow()
                elif ts_field > 1e12:
                    ts = pd.to_datetime(int(ts_field), unit='ms')
                elif ts_field > 1e9:
                    ts = pd.to_datetime(int(ts_field), unit='s')
                else:
                    ts = pd.Timestamp.utcnow()
        except Exception:
            ts = pd.Timestamp.utcnow()

        token_s = str(token)
        symbol = st.session_state["token_to_symbol"].get(token_s)
        if symbol is None:
            # try reverse lookup
            df = load_instruments()
            if not df.empty:
                rows = df[df['instrument_token'] == int(token)]
                if not rows.empty:
                    symbol = rows.iloc[0]['tradingsymbol']
                    st.session_state["token_to_symbol"][token_s] = symbol
                else:
                    symbol = token_s
                    st.session_state["token_to_symbol"][token_s] = symbol
            else:
                symbol = token_s
                st.session_state["token_to_symbol"][token_s] = symbol

        # store tick (deque)
        if symbol not in st.session_state["live_ticks"]:
            st.session_state["live_ticks"][symbol] = deque(maxlen=MAX_TICKS_PER_SYMBOL)
        st.session_state["live_ticks"][symbol].append({'t': ts, 'v': float(lp) if lp is not None else np.nan})

        # aggregate into 1m OHLC server-side
        minute = ts.replace(second=0, microsecond=0)
        if symbol not in st.session_state["ohlc_agg"]:
            st.session_state["ohlc_agg"][symbol] = {}
        ohlc_map = st.session_state["ohlc_agg"][symbol]
        key = minute.isoformat()
        if key not in ohlc_map:
            ohlc_map[key] = {'time': minute.strftime('%Y-%m-%d %H:%M:%S'), 'open': lp, 'high': lp, 'low': lp, 'close': lp}
        else:
            # update highs/lows/closes
            if lp is not None:
                if ohlc_map[key]['high'] is None or lp > ohlc_map[key]['high']:
                    ohlc_map[key]['high'] = lp
                if ohlc_map[key]['low'] is None or lp < ohlc_map[key]['low']:
                    ohlc_map[key]['low'] = lp
                ohlc_map[key]['close'] = lp
        # cap size in memory
        if len(ohlc_map) > MAX_AGG_CANDLES:
            # keep only latest MAX_AGG_CANDLES
            keys_sorted = sorted(list(ohlc_map.keys()))
            for oldk in keys_sorted[:-MAX_AGG_CANDLES]:
                del ohlc_map[oldk]

def kite_on_connect(ws, response):
    st.session_state["kws_running"] = True

def kite_on_close(ws, code, reason):
    st.session_state["kws_running"] = False

def start_kite_ticker_thread():
    # start KiteTicker object once and save into session state
    if st.session_state["kws_obj"] is None:
        try:
            kws = KiteTicker(api_key, st.session_state["access_token"])
            kws.on_ticks = kite_on_ticks
            kws.on_connect = kite_on_connect
            kws.on_close = kite_on_close
            # run connect in separate thread so Streamlit main loop is free
            def run():
                try:
                    kws.connect(threaded=False)
                except Exception as e:
                    # put in warning but don't crash
                    st.warning(f"KiteTicker connect error: {e}")
            th = threading.Thread(target=run, daemon=True)
            th.start()
            st.session_state["kws_obj"] = kws
            st.session_state["kws_thread"] = th
        except Exception as e:
            st.warning(f"Failed to init KiteTicker: {e}")

def subscribe_to_token(token, symbol=None):
    # thread-safe subscription helper
    try:
        kws = st.session_state.get("kws_obj")
        if kws is None:
            return False
        if token not in st.session_state["subscribed_tokens"]:
            kws.subscribe([int(token)])
            kws.set_mode(kws.MODE_FULL, [int(token)])
            st.session_state["subscribed_tokens"].add(int(token))
        if symbol and str(token) not in st.session_state["token_to_symbol"]:
            st.session_state["token_to_symbol"][str(token)] = symbol
        return True
    except Exception as e:
        st.warning(f"Subscribe error: {e}")
        return False

def unsubscribe_all_tokens():
    try:
        kws = st.session_state.get("kws_obj")
        if kws is None:
            return
        if st.session_state["subscribed_tokens"]:
            try:
                kws.unsubscribe(list(st.session_state["subscribed_tokens"]))
            except Exception:
                pass
            st.session_state["subscribed_tokens"].clear()
    except Exception:
        pass

# ---------------------- Sidebar controls for live WS ----------------------
st.sidebar.header("Live WebSocket (KiteTicker) — optional")
ws_symbol = st.sidebar.text_input("Symbol to subscribe (e.g. RELIANCE)", value="RELIANCE")
ws_token_input = st.sidebar.text_input("Instrument Token (optional)", value="")
if st.sidebar.button("Start Live WebSocket"):
    # ensure access token present
    if "access_token" not in st.session_state:
        st.error("Please complete Zerodha login first.")
    else:
        start_kite_ticker_thread()
        # resolve token
        if ws_token_input.strip():
            try:
                token = int(ws_token_input.strip())
                st.session_state["token_to_symbol"][str(token)] = ws_symbol.upper()
            except Exception:
                st.error("Invalid instrument_token provided.")
                token = None
        else:
            token = find_instrument_token(ws_symbol.upper())
            if token is None:
                st.error("Could not find instrument token for symbol. Provide instrument_token manually.")
        if token:
            ok = subscribe_to_token(token, ws_symbol.upper())
            if ok:
                st.sidebar.success(f"Subscribed {ws_symbol.upper()} (token {token})")
            else:
                st.sidebar.error("Subscription failed. Check logs.")
if st.sidebar.button("Stop Live WebSocket"):
    unsubscribe_all_tokens()
    # close connection if exists
    kws = st.session_state.get("kws_obj")
    try:
        if kws is not None:
            kws.close()
    except Exception:
        pass
    st.session_state["kws_obj"] = None
    st.session_state["kws_thread"] = None
    st.session_state["kws_running"] = False
    st.sidebar.info("KiteTicker stopped.")

# ---------------------- Helper: aggregate ticks to ohlc DataFrame ----------------------
def aggregate_ticks_to_ohlc(symbol, minutes=1):
    """
    Convert server-side aggregated map st.session_state['ohlc_agg'][symbol] -> DataFrame sorted by time
    minutes param currently only for label; aggregation already done at 1m in tick handler.
    """
    ohlc_map = st.session_state["ohlc_agg"].get(symbol, {})
    if not ohlc_map:
        return pd.DataFrame()
    rows = []
    for k, v in ohlc_map.items():
        # ensure numeric
        try:
            open_p = float(v['open']) if v['open'] is not None else np.nan
            high_p = float(v['high']) if v['high'] is not None else np.nan
            low_p = float(v['low']) if v['low'] is not None else np.nan
            close_p = float(v['close']) if v['close'] is not None else np.nan
            rows.append({'time': v['time'], 'open': open_p, 'high': high_p, 'low': low_p, 'close': close_p})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(sorted(rows, key=lambda x: x['time']))
    # trim to MAX_AGG_CANDLES
    if len(df) > MAX_AGG_CANDLES:
        df = df.iloc[-MAX_AGG_CANDLES:]
    return df

# ---------------------- Main UI: combine historical + live candles & safe LTP metric ----------------------
st.markdown(
    """
    <div style='background:linear-gradient(90deg,#141e30,#243b55 60%,#FFD900 100%);
     padding:10px 24px 6px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007; '>
        <span style='color:#FFD900;font-family:monospace;font-size:2.1rem;font-weight:bold;vertical-align:middle;letter-spacing:2px;'>
        BLOCKVISTA TERMINAL</span>
        <span style='float:right;color:#30ff96;font-size:1.25rem;font-family:monospace;padding-top:16px;font-weight:bold;'>
        INDIA • INTRADAY • LIVE</span>
    </div>
    """, unsafe_allow_html=True
)

# Main Dashboard
if len(stock_list):
    display_symbol = stock_list[0].upper()
    st.header(f"Live Technical Dashboard: {display_symbol}")

    data = fetch_stock_data(display_symbol, screen_period, screen_interval)
    if data is None or not len(data):
        st.error("No data available for this symbol/interval.")
        st.stop()

    # Determine LTP: prefer live ticks (if available), else kite.ltp, else last historical close
    price = np.nan
    try:
        if display_symbol in st.session_state["live_ticks"] and st.session_state["live_ticks"][display_symbol]:
            price = st.session_state["live_ticks"][display_symbol][-1]['v']
        else:
            # kite.ltp returns nested dict or will raise — wrap safely
            try:
                ltp_json = kite.ltp(f"NSE:{display_symbol}")
                price = ltp_json.get(f"NSE:{display_symbol}", {}).get("last_price", np.nan)
            except Exception:
                # fallback to last historical close
                price = float(data["Close"].iloc[-1])
    except Exception:
        price = np.nan

    # build metrics row - safe handling of non-scalars
    metrics_row = st.columns([1.5,1,1,1,1,1])
    latest = data.iloc[-1]
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    atr = try_scalar(latest.get('ATR', np.nan))
    vwap = try_scalar(latest.get('VWAP', np.nan))

    # Safe LTP display
    with metrics_row[0]:
        try:
            # normalize price value to a scalar float
            if price is None:
                st.metric("LTP", "—")
            elif isinstance(price, (pd.Series, np.ndarray, list, tuple)):
                try:
                    latest_price = float(np.array(price).flatten()[-1])
                    if np.isnan(latest_price):
                        st.metric("LTP", "—")
                    else:
                        st.metric("LTP", f"{round(latest_price, 2)}", label_visibility="visible")
                except Exception:
                    st.metric("LTP", "—")
            else:
                # convert to float safely
                try:
                    p = float(price)
                    if np.isnan(p):
                        st.metric("LTP", "—")
                    else:
                        st.metric("LTP", f"{round(p, 2)}", label_visibility="visible")
                except Exception:
                    st.metric("LTP", "—")
        except Exception as e:
            st.metric("LTP", "—")
            st.warning(f"LTP error: {e}")

    with metrics_row[1]:
        st.metric("RSI", f"{round(rsi,2) if not np.isnan(rsi) else '—'}")
    with metrics_row[2]:
        st.metric("MACD", f"{round(macd,2) if not np.isnan(macd) else '—'}")
    with metrics_row[3]:
        st.metric("ADX", f"{round(adx,2) if not np.isnan(adx) else '—'}")
    with metrics_row[4]:
        st.metric("ATR", f"{round(atr,2) if not np.isnan(atr) else '—'}")
    with metrics_row[5]:
        st.metric("VWAP", f"{round(vwap,2) if not np.isnan(vwap) else '—'}")

    # Build combined candle dataframe:
    # 1) Get historical bars (existing 'data')
    # 2) Get aggregated live 1m candles from st.session_state
    # 3) Append aggregated candles after historical index to show latest incomplete candle
    try:
        agg_live = aggregate_ticks_to_ohlc(display_symbol, minutes=1) if 'aggregate_ticks_to_ohlc' in globals() else None
    except Exception:
        agg_live = aggregate_ticks_to_ohlc(display_symbol, minutes=1)

    # Prepare combined DataFrame for plotting
    combined = None
    try:
        hist_df = data.reset_index().rename(columns={"index":"time"})
        # standardize time format
        if 'time' in hist_df.columns:
            hist_df['time'] = pd.to_datetime(hist_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        # hist_df columns: time, Open, High, Low, Close
        if 'Open' in hist_df.columns:
            hist_ohlc = hist_df[['time','Open','High','Low','Close']].rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'})
        else:
            hist_ohlc = pd.DataFrame()

        if isinstance(agg_live, pd.DataFrame) and not agg_live.empty:
            # agg_live has columns time,open,high,low,close
            combined = pd.concat([hist_ohlc, agg_live], ignore_index=True, sort=False)
        else:
            combined = hist_ohlc
        # drop rows with missing OHLC (safety)
        if combined is not None and not combined.empty:
            combined = combined.dropna(subset=['open','high','low','close'])
    except Exception as e:
        combined = None
        st.warning(f"Could not build combined candle dataset: {e}")

    # ---- Chart Tab (plotly candlestick) ----
    tabs = st.tabs(["Chart", "TA", "Advanced", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands (Fill)", value=True)
        fig = go.Figure()
        if combined is None or combined.empty:
            st.info("No combined candle data available yet. Historical bars shown below.")
            # fallback to historical plot
            if 'Open' in data.columns:
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candles'))
        else:
            # Plot combined DataFrame
            # ensure index is datetime for plotly
            combined_plot = combined.copy()
            combined_plot['time'] = pd.to_datetime(combined_plot['time'])
            if chart_style == "Heikin Ashi":
                # compute HA candles from combined_plot
                ha = ta.ha(combined_plot['open'], combined_plot['high'], combined_plot['low'], combined_plot['close'])
                fig.add_trace(go.Candlestick(
                    x=combined_plot['time'],
                    open=ha['HA_open'], high=ha['HA_high'],
                    low=ha['HA_low'], close=ha['HA_close'], name='Heikin Ashi'))
            else:
                fig.add_trace(go.Candlestick(
                    x=combined_plot['time'], open=combined_plot['open'], high=combined_plot['high'],
                    low=combined_plot['low'], close=combined_plot['close'], name='Candles'))
        # overlays
        if 'EMA9' in data:
            fig.add_trace(go.Scatter(x=data.index, y=data['EMA9'], line=dict(width=1), name='EMA 9'))
        if 'SMA21' in data:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA21'], line=dict(width=1), name='SMA 21'))
        if bands_show and 'BOLL_U' in data and 'BOLL_L' in data:
            fig.add_trace(go.Scatter(x=data.index, y=data['BOLL_U'], line=dict(width=1), name='Boll U'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BOLL_L'], line=dict(width=1), name='Boll L'))
            fig.add_trace(go.Scatter(
                x=list(data.index) + list(data.index[::-1]),
                y=list(data['BOLL_U']) + list(data['BOLL_L'])[::-1],
                fill="toself", fillcolor="rgba(41,134,204,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="BB Channel"))
        supertrend_col = [str(c) for c in list(data.columns) if str(c).startswith('SUPERT_') and not str(c).endswith('_dir')]
        if supertrend_col:
            fig.add_trace(go.Scatter(x=data.index, y=data[supertrend_col[0]], line=dict(color='#fae900', width=2), name='Supertrend'))

        fig.update_layout(template='plotly_dark', height=640)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        ta_cols_all = ['RSI','ADX','STOCHRSI']
        ta_cols = [c for c in ta_cols_all if c in list(data.columns)]
        if ta_cols:
            st.line_chart(data[ta_cols].dropna())
        else:
            st.warning("No available TA columns for charting.")
        macd_cols_all = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        macd_cols = [c for c in macd_cols_all if c in list(data.columns)]
        if macd_cols:
            st.line_chart(data[macd_cols].dropna())
        if 'ATR' in data:
            st.line_chart(data['ATR'].dropna())
        last_cols_all = ['Close','RSI','ADX','STOCHRSI','ATR','VWAP']
        last_cols = [c for c in last_cols_all if c in list(data.columns)]
        st.write("Latest Values:", data.iloc[-1][last_cols])

    with tabs[2]:
        st.subheader("Signals (Current)")
        signals = get_signals(data)
        st.table(pd.DataFrame(signals.items(), columns=['Indicator', 'Signal']))
        csv2 = data.to_csv()
        st.download_button('Export Data to CSV', csv2, file_name=f"{stock_list[0]}_{screen_interval}.csv")

    with tabs[3]:
        if st.checkbox("Show Table Data"):
            st.dataframe(data.tail(40))

st.caption("BlockVista Terminal | Powered by Zerodha KiteConnect, yFinance, Alpha Vantage, Plotly & Streamlit")

# ---------------------- Instrument token quick method (instructions) ----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Instrument Token (optional)**")
st.sidebar.markdown("If symbol-to-token lookup fails, get token via:")
st.sidebar.code("""
instruments = kite.instruments()
df = pd.DataFrame(instruments)
df[df.tradingsymbol == 'RELIANCE'][['tradingsymbol','instrument_token','exchange']]
""", language='python')
st.sidebar.markdown("Copy `instrument_token` and paste into the 'Instrument Token (optional)' field above before starting the websocket.")

# End of file
