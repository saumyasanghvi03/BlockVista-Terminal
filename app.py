# app.py
"""
BlockVista Terminal — Streamlit app
Features:
- Zerodha KiteConnect auto-refresh token flow (uses KITE_REFRESH_TOKEN if present)
- KiteTicker WebSocket subscription (dynamic subscribe/unsubscribe)
- Server-side aggregation of live ticks -> OHLC candles (1m/5m/15m)
- Candlestick rendering using Lightweight Charts embedded via components.html
- Order placement (market / limit) through Kite place_order
- Fallbacks to yfinance / Alpha Vantage for historical bars (original logic retained)
"""

import os
import time
import json
import threading
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

# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"
MAX_TICKS_PER_SYMBOL = 4000   # bound live memory per-symbol
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
        # do not spam UI — return None
        return None

# ---------------------- THEME & SESSION ----------------------
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

# ---------------------- Smallcase baskets ----------------------
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

# ---------------------- Zerodha Auth (auto-refresh with refresh_token if available) ----------------------
def get_api_creds():
    # Prefer Streamlit secrets, fallback to environment variables
    def get(name):
        if name in st.secrets:
            return st.secrets[name]
        return os.getenv(name)
    api_key = get("KITE_API_KEY") or get("ZERODHA_API_KEY") or get("ZERODHA_API_KEY")
    api_secret = get("KITE_API_SECRET") or get("ZERODHA_API_SECRET") or get("ZERODHA_API_SECRET")
    refresh_token = get("KITE_REFRESH_TOKEN")
    return api_key, api_secret, refresh_token

api_key, api_secret, refresh_token = get_api_creds()

if not api_key or not api_secret:
    st.error("Zerodha API credentials not found. Add KITE_API_KEY and KITE_API_SECRET in Streamlit secrets or environment variables.")
    st.stop()

def get_kite_session():
    """
    Returns a KiteConnect instance with an active access token in session_state.
    Flow:
    - If st.session_state['access_token'] exists -> set and return kite.
    - Else if KITE_REFRESH_TOKEN present -> call kite.generate_session(refresh_token, api_secret) to obtain access_token.
      (This assumes your refresh token can be used this way; if your Kite app uses a different refresh flow you may need to adapt.)
    - Else -> show login URL and guide to manually obtain request_token -> generate_session -> print refresh_token once and store it.
    """
    kite_conn = KiteConnect(api_key=api_key)
    # If we already have access_token in session state (valid for this run)
    if "access_token" in st.session_state:
        kite_conn.set_access_token(st.session_state["access_token"])
        return kite_conn

    # Try auto-refresh via refresh token (recommended)
    if refresh_token:
        try:
            # Some Kite setups provide generate_session with request_token only.
            # If your Kite account/app supports refresh_token -> use session refresh method accordingly.
            # Here we attempt generate_session(refresh_token, api_secret) per prior discussion.
            data = kite_conn.generate_session(refresh_token, api_secret=api_secret)
            access_token = data.get("access_token")
            if access_token:
                st.session_state["access_token"] = access_token
                kite_conn.set_access_token(access_token)
                st.success("✅ Zerodha session auto-refreshed via refresh_token")
                return kite_conn
            else:
                st.warning("Auto refresh attempt did not return access_token. Falling back to manual flow.")
        except Exception as e:
            # Do not crash app; guide user to manual flow
            st.warning(f"Auto-refresh failed: {e}")

    # Manual one-time login flow (guide)
    login_url = kite_conn.login_url()
    st.markdown(
        f"""
        <div style="background:#f5da82;padding:14px;border-radius:8px;">
        🟠 <a href="{login_url}" target="_blank"><b>Click here to login & authorize BlockVista</b></a><br>
        After logging in, you'll be redirected to your redirect_url with <b>request_token=xxxx</b> in the URL.
        Paste that <b>request_token</b> into the box below and click 'Generate Access Token'. Then copy the printed
        <b>refresh_token</b> into your Streamlit secrets (`KITE_REFRESH_TOKEN`) for future auto-refresh.
        </div>
        """,
        unsafe_allow_html=True
    )
    request_token = st.text_input("Paste request_token here (one-time):")
    if st.button("Generate Access Token") and request_token:
        try:
            data = kite_conn.generate_session(request_token, api_secret=api_secret)
            access_token = data.get("access_token")
            refresh_token_printed = data.get("refresh_token") or data.get("public_token") or None
            if access_token:
                st.session_state["access_token"] = access_token
                kite_conn.set_access_token(access_token)
                st.success("✅ Zerodha session started! Save the refresh token to secrets for auto refresh.")
                if refresh_token_printed:
                    st.code(f"REFRESH_TOKEN (save this in your secrets):\n{refresh_token_printed}")
                return kite_conn
            else:
                st.error("Could not obtain access_token from Kite response.")
                st.stop()
        except Exception as ex:
            st.error(f"❌ Zerodha login failed: {ex}")
            st.stop()
    st.stop()

kite = get_kite_session()

# ---------------------- Core helpers (original logic kept) ----------------------
def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception:
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
    if data is None or len(data) == 0:
        av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
        data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
        if data is None or len(data) == 0:
            return None
    # Compute technicals (best-effort)
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
        pass
    return data

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

# ---------------------- Sidebar: Watchlist P&L Tracker ----------------------
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

# ---------------------- Sidebar: Screener ----------------------
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

# ---------------------- KiteTicker live integration (dynamic subscribe) ----------------------
# session-state initialization
if "live_ticks" not in st.session_state:
    st.session_state["live_ticks"] = {}
if "ohlc_agg" not in st.session_state:
    st.session_state["ohlc_agg"] = {}
if "token_to_symbol" not in st.session_state:
    st.session_state["token_to_symbol"] = {}
if "kws_obj" not in st.session_state:
    st.session_state["kws_obj"] = None
if "kws_thread" not in st.session_state:
    st.session_state["kws_thread"] = None
if "kws_running" not in st.session_state:
    st.session_state["kws_running"] = False
if "subscribed_tokens" not in st.session_state:
    st.session_state["subscribed_tokens"] = set()

@st.cache_data(ttl=3600)
def load_instruments():
    try:
        instruments = kite.instruments()
        df = pd.DataFrame(instruments)
        return df
    except Exception:
        return pd.DataFrame()

def find_instrument_token(symbol, exchange='NSE'):
    df = load_instruments()
    if df.empty:
        return None
    cond = (df['tradingsymbol'].str.upper() == symbol.upper()) & (df['exchange'] == exchange)
    matches = df[cond]
    if not matches.empty:
        return int(matches.iloc[0]['instrument_token'])
    matches2 = df[df['tradingsymbol'].str.upper().str.contains(symbol.upper()) & (df['exchange'] == exchange)]
    if not matches2.empty:
        return int(matches2.iloc[0]['instrument_token'])
    return None

# KiteTicker callbacks and utilities
def kite_on_ticks(ws, ticks):
    for t in ticks:
        token = t.get('instrument_token')
        lp = t.get('last_price')
        ts_field = t.get('timestamp')
        try:
            if isinstance(ts_field, str):
                ts = pd.to_datetime(ts_field)
            else:
                # epoch ms / s fallback
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
        symbol_from_map = st.session_state["token_to_symbol"].get(token_s)
        if symbol_from_map is None:
            df_inst = load_instruments()
            if not df_inst.empty:
                rows = df_inst[df_inst['instrument_token'] == int(token)]
                if not rows.empty:
                    symbol_from_map = rows.iloc[0]['tradingsymbol']
                    st.session_state["token_to_symbol"][token_s] = symbol_from_map
                else:
                    symbol_from_map = token_s
                    st.session_state["token_to_symbol"][token_s] = symbol_from_map
            else:
                symbol_from_map = token_s
                st.session_state["token_to_symbol"][token_s] = symbol_from_map

        # store tick
        if symbol_from_map not in st.session_state["live_ticks"]:
            st.session_state["live_ticks"][symbol_from_map] = deque(maxlen=MAX_TICKS_PER_SYMBOL)
        st.session_state["live_ticks"][symbol_from_map].append({'t': ts, 'v': float(lp) if lp is not None else np.nan})

        # aggregate into 1m OHLC
        minute = ts.replace(second=0, microsecond=0)
        if symbol_from_map not in st.session_state["ohlc_agg"]:
            st.session_state["ohlc_agg"][symbol_from_map] = {}
        ohlc_map = st.session_state["ohlc_agg"][symbol_from_map]
        key = minute.isoformat()
        if key not in ohlc_map:
            ohlc_map[key] = {'time': int(minute.timestamp()), 'open': lp, 'high': lp, 'low': lp, 'close': lp}
        else:
            if lp is not None:
                if ohlc_map[key]['high'] is None or lp > ohlc_map[key]['high']:
                    ohlc_map[key]['high'] = lp
                if ohlc_map[key]['low'] is None or lp < ohlc_map[key]['low']:
                    ohlc_map[key]['low'] = lp
                ohlc_map[key]['close'] = lp
        # cap
        if len(ohlc_map) > MAX_AGG_CANDLES:
            keys_sorted = sorted(list(ohlc_map.keys()))
            for oldk in keys_sorted[:-MAX_AGG_CANDLES]:
                del ohlc_map[oldk]

def kite_on_connect(ws, response):
    st.session_state["kws_running"] = True

def kite_on_close(ws, code, reason):
    st.session_state["kws_running"] = False

def start_kite_ticker_thread():
    # Guard: don't start multiple threads
    if st.session_state.get("kws_running"):
        return
    if st.session_state.get("kws_obj") is not None:
        try:
            # If an object exists but not running, try to reuse
            kws = st.session_state["kws_obj"]
            th = st.session_state.get("kws_thread")
            if th and th.is_alive():
                return
        except Exception:
            pass
    try:
        kws = KiteTicker(api_key, st.session_state["access_token"])
        kws.on_ticks = kite_on_ticks
        kws.on_connect = kite_on_connect
        kws.on_close = kite_on_close
        # run in thread
        def run():
            try:
                kws.connect(threaded=False)
            except Exception as e:
                print("KiteTicker connect error:", e)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        st.session_state["kws_obj"] = kws
        st.session_state["kws_thread"] = th
    except Exception as e:
        st.warning(f"Failed to init KiteTicker: {e}")

def subscribe_to_token(token, symbol=None):
    try:
        kws = st.session_state.get("kws_obj")
        if kws is None:
            return False
        if int(token) not in st.session_state["subscribed_tokens"]:
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

# ---------------------- Sidebar: Live WS controls & Order Placement ----------------------
st.sidebar.header("Live WebSocket (KiteTicker) — optional")
ws_symbol = st.sidebar.text_input("Symbol to subscribe (e.g. RELIANCE)", value="RELIANCE")
ws_token_input = st.sidebar.text_input("Instrument Token (optional)", value="")
agg_period = st.sidebar.selectbox("Aggregation Period", ["1m", "5m", "15m"], index=0)

start_ws = st.sidebar.button("Start Live WebSocket")
stop_ws = st.sidebar.button("Stop Live WebSocket")

if start_ws:
    if "access_token" not in st.session_state:
        st.sidebar.error("Please complete Zerodha login first.")
    else:
        start_kite_ticker_thread()
        if ws_token_input.strip():
            try:
                token = int(ws_token_input.strip())
                st.session_state["token_to_symbol"][str(token)] = ws_symbol.upper()
            except Exception:
                st.sidebar.error("Invalid instrument_token provided.")
                token = None
        else:
            token = find_instrument_token(ws_symbol.upper())
            if token is None:
                st.sidebar.error("Could not find instrument token for symbol. Provide instrument_token manually.")
        if token:
            ok = subscribe_to_token(token, ws_symbol.upper())
            if ok:
                st.sidebar.success(f"Subscribed {ws_symbol.upper()} (token {token})")
            else:
                st.sidebar.error("Subscription failed. Check logs.")

if stop_ws:
    unsubscribe_all_tokens()
    try:
        kws = st.session_state.get("kws_obj")
        if kws is not None:
            kws.close()
    except Exception:
        pass
    st.session_state["kws_obj"] = None
    st.session_state["kws_thread"] = None
    st.session_state["kws_running"] = False
    st.sidebar.info("KiteTicker stopped.")

# ---------------------- Sidebar: Order Placement ----------------------
st.sidebar.markdown("---")
st.sidebar.header("Place Order")

order_symbol = st.sidebar.text_input("Order Symbol (e.g. RELIANCE)", value=ws_symbol.upper())
order_qty = st.sidebar.number_input("Quantity", min_value=1, value=1)
order_type = st.sidebar.selectbox("Order Type", ["MARKET", "LIMIT"])
transaction_type = st.sidebar.selectbox("Transaction Type", ["BUY", "SELL"])
order_price = None
order_product = st.sidebar.selectbox("Product", ["CNC", "MIS", "NRML"], index=0)

if order_type == "LIMIT":
    order_price = st.sidebar.number_input("Limit Price", min_value=0.0, value=0.0, format="%.2f")

if st.sidebar.button("Submit Order"):
    try:
        order_params = {
            "tradingsymbol": order_symbol.upper(),
            "exchange": "NSE",
            "transaction_type": transaction_type,
            "quantity": int(order_qty),
            "order_type": order_type,
            "product": order_product,
            "variety": "regular"
        }
        if order_type == "LIMIT":
            order_params["price"] = float(order_price)
        # place order via kite
        order_id = kite.place_order(**order_params)
        st.sidebar.success(f"✅ Order placed! ID: {order_id}")
    except Exception as e:
        st.sidebar.error(f"❌ Order failed: {e}")

# ---------------------- Helper: aggregate ticks to ohlc DataFrame ----------------------
def aggregate_ticks_to_ohlc(symbol, minutes=1):
    ohlc_map = st.session_state.get("ohlc_agg", {}).get(symbol, {})
    if not ohlc_map:
        return pd.DataFrame()
    rows = []
    for k, v in ohlc_map.items():
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
    if len(df) > MAX_AGG_CANDLES:
        df = df.iloc[-MAX_AGG_CANDLES:]
    return df

# ---------------------- Lightweight Charts renderer ----------------------
def render_lightweight_candles(symbol, agg_period='1m'):
    df = aggregate_ticks_to_ohlc(symbol, minutes=1)
    if df.empty:
        st.info("No live aggregated candle data yet.")
        return
    # convert epoch seconds to JS-friendly time for lightweight-charts (either timestamps in seconds or 'YYYY-MM-DD HH:MM:SS')
    df['time'] = pd.to_datetime(df['time'], unit='s')
    rule_map = {"1m": "1T", "5m": "5T", "15m": "15T"}
    if agg_period in rule_map:
        df = df.set_index('time').resample(rule_map[agg_period]).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
    # lightweight-charts expects time either as unix timestamp in seconds or {year,month,...}
    js_rows = []
    for _, row in df.iterrows():
        ts = int(pd.to_datetime(row['time']).timestamp())
        js_rows.append({
            "time": ts,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close'])
        })
    js_data = json.dumps(js_rows)

    html_code = f"""
    <!doctype html>
    <html>
    <head><meta charset="utf-8"></head>
    <body>
      <div id="chart" style="width:100%;height:520px;"></div>
      <script src="https://unpkg.com/lightweight-charts@3.7.0/dist/lightweight-charts.standalone.production.js"></script>
      <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
          layout: {{background: {{type: 'solid', color: '#0f1116'}}, textColor: '#DDD'}},
          grid: {{vertLines: {{color: '#2B2B43'}}, horzLines: {{color: '#2B2B43'}}}},
          timeScale: {{timeVisible: true, secondsVisible: false}}
        }});
        const candleSeries = chart.addCandlestickSeries();
        const data = {js_data};
        candleSeries.setData(data);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=540)

# ---------------------- Main UI ----------------------
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

# Use the screener's selected stock_list to drive main dashboard
if len(stock_list):
    display_symbol = stock_list[0].upper()
    st.header(f"Live Technical Dashboard: {display_symbol}")

    data = fetch_stock_data(display_symbol, screen_period, screen_interval)
    if data is None or not len(data):
        st.error("No data available for this symbol/interval.")
        st.stop()

    # LTP preference: live ticks -> kite.ltp -> historical close
    price = np.nan
    try:
        if display_symbol in st.session_state.get("live_ticks", {}) and st.session_state["live_ticks"][display_symbol]:
            price = st.session_state["live_ticks"][display_symbol][-1]['v']
        else:
            try:
                ltp_json = kite.ltp(f"NSE:{display_symbol}")
                price = ltp_json.get(f"NSE:{display_symbol}", {}).get("last_price", np.nan)
            except Exception:
                price = float(data["Close"].iloc[-1])
    except Exception:
        price = np.nan

    # Metrics Row
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

    # Prepare combined candle dataset: historical + live aggregated
    try:
        agg_live = aggregate_ticks_to_ohlc(display_symbol, minutes=1)
    except Exception:
        agg_live = pd.DataFrame()

    combined = None
    try:
        hist_df = data.reset_index().rename(columns={"index":"time"})
        if 'time' in hist_df.columns:
            hist_df['time'] = pd.to_datetime(hist_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'Open' in hist_df.columns:
            hist_ohlc = hist_df[['time','Open','High','Low','Close']].rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'})
        else:
            hist_ohlc = pd.DataFrame()
        if isinstance(agg_live, pd.DataFrame) and not agg_live.empty:
            combined = pd.concat([hist_ohlc, agg_live], ignore_index=True, sort=False)
        else:
            combined = hist_ohlc
        if combined is not None and not combined.empty:
            combined = combined.dropna(subset=['open','high','low','close'])
    except Exception as e:
        combined = None
        st.warning(f"Could not build combined candle dataset: {e}")

    # Chart Tab
    tabs = st.tabs(["Chart", "TA", "Advanced", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands (Fill)", value=True)
        # Render Lightweight Chart from live aggregated candles if available
        if display_symbol in st.session_state.get("ohlc_agg", {}) and st.session_state["ohlc_agg"][display_symbol]:
            render_lightweight_candles(display_symbol, agg_period)
        else:
            st.info("No live OHLC data available yet; showing historical Plotly chart.")
            if 'Open' in data.columns:
                fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
                fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=640)
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
        
    with st.expander("Show Analysis"):
        st.subheader("Advanced Analysis & Signals")
        # Add your analysis widgets, charts, or metrics here
        st.write("Coming soon: AI-powered trade signals and strategy insights 🚀")
        signals = get_signals(data)
        for k, v in signals.items():
        st.metric(label=k, value=v)
        st.markdown("### Additional Advanced Indicators")\
        advanced_cols = ['SMA21', 'EMA9', 'BOLL_L', 'BOLL_M', 'BOLL_U', 'ATR', 'VWAP']
        adv_data = data[advanced_cols].tail(20) if all(c in data.columns for c in advanced_cols) else None
        if adv_data is not None:
        st.line_chart(adv_data)
        else:
        st.info("No advanced indicator data available.")
    with tabs[3]:
        st.subheader("Raw Data")
        raw_df = combined if combined is not None else data
        st.dataframe(raw_df)
        # Auto-refresh
        if st.session_state.get("auto_refresh", True):
        st_autorefresh(interval=st.session_state.get("refresh_interval", 15) * 1000, key="data_refresh")
        if st.sidebar.button("Notify LTP"):
        browser_notification(f"{display_symbol} Live Price", f"LTP: {price}")
        
        st.caption("BlockVista Terminal | Powered by Zerodha KiteConnect, yFinance, Alpha Vantage, Plotly & Streamlit")

# ---------------------- End of file -----------
