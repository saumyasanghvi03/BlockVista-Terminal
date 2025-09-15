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

import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import json
import datetime as dt
import math

import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go

from kiteconnect import KiteConnect, KiteTicker
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries

# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"  # keep as-is (you may rotate to your own)
MAX_TICKS_PER_SYMBOL = 4000  # keep memory bounded
MAX_CANDLES = 500  # limit candles sent to front-end to keep iframe light

# ---------------------- Utility / Notification ----------------------
def browser_notification(title, body, icon=None):
    icon_line = f'icon: "{icon}",' if icon else ""
    st.markdown(
        f"""
        <script>
        if (Notification && Notification.permission !== "granted") {{
            Notification.requestPermission();
        }}
        if (Notification && Notification.permission === "granted") {{
            new Notification("{title}", {{
            body: "{body}",
            {icon_line}
            }});
        }}
        </script>
        """, unsafe_allow_html=True
    )

# ---------------------- Alpha Vantage fetch ----------------------
def fetch_alpha_vantage_intraday(symbol, interval='1min', outputsize='compact'):
    try:
        ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
        # Alpha Vantage uses suffixes like ".NSE" sometimes; attempt common pattern
        symbol_av = symbol.upper() if symbol.upper().endswith(".NSE") else f"{symbol.upper()}.NSE"
        data, _ = ts.get_intraday(symbol=symbol_av, interval=interval, outputsize=outputsize)
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

# ---------------------- Streamlit session defaults ----------------------
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 10

# Websocket/session state
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "kite" not in st.session_state:
    st.session_state["kite"] = None
if "kite_ws_running" not in st.session_state:
    st.session_state["kite_ws_running"] = False
if "live_ticks" not in st.session_state:
    # dict: SYMBOL -> list of ticks [{t:ISO8601, lp:float, size: int}]
    st.session_state["live_ticks"] = {}
if "token_to_symbol" not in st.session_state:
    st.session_state["token_to_symbol"] = {}
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = None

# ---------------------- UI Theme (kept from your original) ----------------------
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

# Minimal custom styling (you can preserve the rest of your CSS if you want)
def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container {background-color: #0f1115 !important;}
        .stSidebar {background: #0b0b0b !important;}
        h1,h2,h3,h4,h5,h6,label {color: #FFD900 !important;}
        .stDataFrame tbody tr {background-color: #111 !important;color: #FFD900 !important;}
        </style>
        """, unsafe_allow_html=True)

set_terminal_style(True)

# ---------------------- Zerodha Auth Flow (unchanged pattern) ----------------------
api_key = st.secrets.get("ZERODHA_API_KEY")
api_secret = st.secrets.get("ZERODHA_API_SECRET")

if not api_key or not api_secret:
    st.error("Zerodha credentials missing in Streamlit secrets. Add ZERODHA_API_KEY and ZERODHA_API_SECRET.")
    st.stop()

# If access_token not present, show login flow
if not st.session_state["access_token"]:
    kite_tmp = KiteConnect(api_key=api_key)
    login_url = kite_tmp.login_url()
    st.markdown(
        f"""
        <div style="background: #f5da82; padding: 14px; border-radius: 8px;">
        🟠 <a href="{login_url}" target="_blank"><b>Click here to login & authorize BlockVista</b></a><br>
        After logging in, copy the `request_token` parameter from the redirect URL and paste it below.
        </div>
        """, unsafe_allow_html=True
    )
    request_token = st.text_input("Paste Zerodha `request_token` here:")
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
            browser_notification("BlockVista Error", f"❌ Zerodha login failed: {ex}")
        st.stop()
    else:
        st.stop()
else:
    if not st.session_state["kite"]:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state["access_token"])
        st.session_state["kite"] = kite
    else:
        kite = st.session_state["kite"]

# ---------------------- Instruments loader / token lookup ----------------------
@st.cache_data
def load_instruments_cached():
    try:
        instruments = kite.instruments()  # returns list of dicts
        df = pd.DataFrame(instruments)
        return df
    except Exception as e:
        st.warning(f"Could not load instruments from Zerodha: {e}")
        return pd.DataFrame()

def find_instrument_token(symbol, exchange='NSE'):
    df = st.session_state.get("instruments_df")
    if df is None or df.empty:
        df = load_instruments_cached()
        st.session_state["instruments_df"] = df
    if df is None or df.empty:
        return None
    cond = (df['tradingsymbol'].str.upper() == symbol.upper()) & (df['exchange'] == exchange)
    matches = df[cond]
    if not matches.empty:
        return int(matches.iloc[0]['instrument_token'])
    # fallback: try contains
    matches2 = df[df['tradingsymbol'].str.upper().str.contains(symbol.upper()) & (df['exchange'] == exchange)]
    if not matches2.empty:
        return int(matches2.iloc[0]['instrument_token'])
    return None

# ---------------------- KiteTicker websocket management ----------------------
kws = None
kws_thread = None

def kite_on_ticks(ws, ticks):
    """
    Called by KiteTicker on every tick batch.
    We convert ticks -> append to st.session_state['live_ticks'][SYMBOL]
    Each tick structure has: instrument_token, last_price, timestamp, volume, etc.
    """
    for t in ticks:
        token = t.get('instrument_token')
        lp = t.get('last_price') or t.get('ohlc', {}).get('close')
        ts = t.get('timestamp')
        # timestamp might be string or epoch-like — normalize to datetime
        try:
            if isinstance(ts, str):
                ts_dt = pd.to_datetime(ts)
            else:
                # assume epoch in milliseconds or seconds based on magnitude
                if ts is None:
                    ts_dt = pd.Timestamp.utcnow()
                elif ts > 1e12:  # ms
                    ts_dt = pd.to_datetime(int(ts), unit='ms')
                elif ts > 1e9:
                    ts_dt = pd.to_datetime(int(ts), unit='s')
                else:
                    ts_dt = pd.Timestamp.utcnow()
        except Exception:
            ts_dt = pd.Timestamp.utcnow()
        # Map token -> symbol if we have mapping
        token_s = str(token)
        symbol = st.session_state["token_to_symbol"].get(token_s, None)
        if symbol is None:
            # attempt reverse lookup in instruments df (fast fallback)
            df = st.session_state.get("instruments_df")
            if df is None or df.empty:
                df = load_instruments_cached()
                st.session_state["instruments_df"] = df
            if not df.empty:
                rows = df[df['instrument_token'] == int(token)]
                if not rows.empty:
                    symbol = rows.iloc[0]['tradingsymbol']
                    st.session_state["token_to_symbol"][token_s] = symbol
                else:
                    symbol = token_s
            else:
                symbol = token_s
                st.session_state["token_to_symbol"][token_s] = symbol

        # create symbol list and append tick
        if symbol not in st.session_state["live_ticks"]:
            st.session_state["live_ticks"][symbol] = []
        tick_obj = {"t": ts_dt.isoformat(), "lp": float(lp) if lp is not None else None,
                    "volume": float(t.get("volume", 0) or 0)}
        st.session_state["live_ticks"][symbol].append(tick_obj)
        # enforce memory limit
        if len(st.session_state["live_ticks"][symbol]) > MAX_TICKS_PER_SYMBOL:
            st.session_state["live_ticks"][symbol] = st.session_state["live_ticks"][symbol][-MAX_TICKS_PER_SYMBOL:]

def kite_on_connect(ws, response):
    st.session_state["kite_ws_running"] = True

def kite_on_close(ws, code, reason):
    st.session_state["kite_ws_running"] = False

def start_kite_ticker(instrument_tokens):
    """
    Starts KiteTicker in a background thread and subscribes to given instrument_tokens.
    instrument_tokens: list[int]
    """
    global kws, kws_thread
    if st.session_state["kite_ws_running"]:
        return
    try:
        kws = KiteTicker(api_key, st.session_state["access_token"])
        kws.on_ticks = kite_on_ticks
        kws.on_connect = kite_on_connect
        kws.on_close = kite_on_close
    except Exception as e:
        st.warning(f"KiteTicker init failed: {e}")
        return

    def _run():
        try:
            # connect is blocking unless threaded=True; here we run in a dedicated thread
            kws.connect(threaded=False)
        except Exception as e:
            st.warning(f"KiteTicker connection error: {e}")
    def _subscribe_later():
        # small wait for connection establishment
        time.sleep(1.0)
        try:
            kws.subscribe(instrument_tokens)
            kws.set_mode(kws.MODE_FULL, instrument_tokens)
        except Exception as e:
            st.warning(f"KiteTicker subscribe error: {e}")

    kws_thread = threading.Thread(target=_run, daemon=True)
    kws_thread.start()
    sub_thread = threading.Thread(target=_subscribe_later, daemon=True)
    sub_thread.start()

def stop_kite_ticker():
    global kws
    try:
        if kws:
            kws.close()
    except Exception:
        pass
    st.session_state["kite_ws_running"] = False

# ---------------------- Aggregation: ticks -> OHLC candles ----------------------
def aggregate_ticks_to_ohlc(ticks, interval_minutes=1):
    """
    ticks: list of dicts with {'t': ISO8601, 'lp': float, 'volume': float}
    interval_minutes: int (1,5,15)
    Returns pandas.DataFrame with columns ['time','open','high','low','close','volume'] indexed by time (datetime)
    """
    if not ticks:
        return pd.DataFrame()
    # Convert to DataFrame
    df = pd.DataFrame(ticks)
    if df.empty:
        return pd.DataFrame()
    # Normalize timestamp
    df['t'] = pd.to_datetime(df['t'])
    # Drop rows without price
    df = df.dropna(subset=['lp'])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('t')
    # Choose the resample rule
    rule = f"{int(interval_minutes)}T"
    df = df.set_index('t')
    agg = df['lp'].resample(rule).ohlc()
    vol = df['volume'].resample(rule).sum().rename('volume')
    ohlc = pd.concat([agg, vol], axis=1).dropna()
    ohlc = ohlc.reset_index().rename(columns={'t': 'time', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    # Ensure ISO strings for times, and limit rows
    if not ohlc.empty:
        ohlc['time'] = ohlc['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if len(ohlc) > MAX_CANDLES:
            ohlc = ohlc.iloc[-MAX_CANDLES:]
    return ohlc

# ---------------------- Lightweight Charts HTML renderer (candles) ----------------------
from streamlit.components.v1 import html as st_html

LIGHTWEIGHT_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://unpkg.com/lightweight-charts@3.7.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; background: #0f1115; }}
    #chart {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const rawData = {candles_json};
    // Convert to lightweight format: {time: 'YYYY-MM-DD HH:MM:SS', open, high, low, close}
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
        layout: {{ backgroundColor: '#0f1115', textColor: '#f8f8f8' }},
        rightPriceScale: {{ visible: true }},
        timeScale: {{ timeVisible: true, secondsVisible: false }},
    }});
    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    }});
    function parseCandles(raw){
        return raw.map(r => {{
            // If time contains seconds or hours, pass as string; otherwise lightweight treats it as day
            return {{
                time: r.time.replace(' ', 'T'),
                open: r.open,
                high: r.high,
                low: r.low,
                close: r.close
            }};
        }});
    }
    const data = parseCandles(rawData);
    if(data && data.length) {{
        candleSeries.setData(data);
        chart.timeScale().fitContent();
    }} else {{
        // show empty message
        document.getElementById('chart').innerHTML = "<div style='color:#FFD900;padding:16px;font-family:monospace;'>No candle data available</div>"
    }}
    // Basic auto-refresh: the parent streamlit app will re-render this iframe with updated rawData on rerun.
  </script>
</body>
</html>
"""

def render_candlestick_in_iframe(candles_df, height=520):
    if candles_df is None or candles_df.empty:
        candles_json = "[]"
    else:
        # Only send limited columns and ensure numeric types
        df = candles_df[['time','open','high','low','close']].copy()
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        records = df.to_dict(orient='records')
        candles_json = json.dumps(records)
    html_code = LIGHTWEIGHT_TEMPLATE.format(candles_json=candles_json)
    st_html(html_code, height=height, scrolling=False)

# ---------------------- Legacy: fetch_stock_data w/ TA (kept mostly intact) ----------------------
@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    """
    Fetch historical bars using yfinance; fallback to AlphaVantage intraday if empty
    symbol: e.g. 'RELIANCE' -> yfinance uses 'RELIANCE.NS' convention
    period: '1d','5d'
    interval: '1m','5m','15m'
    """
    try:
        data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
    except Exception as e:
        st.warning(f"yfinance download error: {e}")
        data = pd.DataFrame()
    if data is None or len(data) == 0:
        av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
        st.info("Fetching live data from Alpha Vantage…")
        data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
        if data is None or len(data) == 0:
            return None
    # Add TA columns
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
    except Exception as e:
        st.warning(f"TA computation error: {e}")
    return data

# ---------------------- Helper: scalar conversion & signals (kept) ----------------------
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

# ---------------------- Sidebar: Live WS Controls / Watchlist ----------------------
st.sidebar.subheader("Live WebSocket (KiteTicker)")
ws_symbol = st.sidebar.text_input("Symbol for Live Tick (e.g. RELIANCE)", value="RELIANCE")
ws_token_input = st.sidebar.text_input("Instrument Token (optional)", value="")  # optional if lookup fails

if st.sidebar.button("Start Live WebSocket"):
    try:
        if ws_token_input.strip():
            token_list = [int(ws_token_input.strip())]
            st.session_state["token_to_symbol"][str(ws_token_input.strip())] = ws_symbol.upper()
        else:
            token = find_instrument_token(ws_symbol.upper())
            if token is None:
                st.error("Could not find instrument token for symbol. Provide instrument_token in the field above.")
                token_list = []
            else:
                token_list = [int(token)]
                st.session_state["token_to_symbol"][str(token)] = ws_symbol.upper()
        if token_list:
            start_kite_ticker(token_list)
            st.sidebar.success(f"Subscribed to {ws_symbol} (tokens: {token_list})")
    except Exception as e:
        st.sidebar.error(f"Start WS failed: {e}")

if st.sidebar.button("Stop Live WebSocket"):
    stop_kite_ticker()
    st.sidebar.info("KiteTicker stopped.")

# ---------------------- Watchlist P&L Tracker (kept) ----------------------
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
        # attempt to get live price from st.session_state live_ticks with fallback to kite.ltp or fetch_stock_data
        ltp = np.nan
        if s in st.session_state["live_ticks"] and st.session_state["live_ticks"][s]:
            ltp = st.session_state["live_ticks"][s][-1]["lp"]
        else:
            try:
                ltp_json = kite.ltp(f"NSE:{s}")
                ltp = ltp_json[f"NSE:{s}"]["last_price"]
            except Exception:
                d = fetch_stock_data(s, "1d", "5m")
                if d is not None and len(d):
                    ltp = float(d["Close"].iloc[-1])
        pnl = (ltp - entry_prices[i]) * quantities[i]
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": ltp, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
    except Exception:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
if pnl_data:
    st.sidebar.dataframe(pd.DataFrame(pnl_data))
    total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int,float)))
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)

# ---------------------- Main: Screener + Dashboard (kept) ----------------------
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

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
# Implement make_screener same as original (kept small for brevity)
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

screen_df = make_screener(stock_list, screen_period, screen_interval)
st.sidebar.subheader("Screener Results")
if len(screen_df):
    st.sidebar.dataframe(screen_df)
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---------------------- Main UI: Dashboard and Chart ----------------------
if len(stock_list):
    display_symbol = stock_list[0].upper()
    st.header(f"Live Technical Dashboard: {display_symbol}")

    data = fetch_stock_data(display_symbol, screen_period, screen_interval)
    if data is None or not len(data):
        st.error("No historical data available for this symbol/interval. Use the Screener or try another symbol.")
        st.stop()

    # Metrics panel (some values from historical TA + live LTP)
    price = None
    if display_symbol in st.session_state["live_ticks"] and st.session_state["live_ticks"][display_symbol]:
        price = st.session_state["live_ticks"][display_symbol][-1]["lp"]
    else:
        try:
            price = kite.ltp(f"NSE:{display_symbol}")[f"NSE:{display_symbol}"]["last_price"]
        except Exception:
            price = data["Close"].iloc[-1]

    latest = data.iloc[-1]
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    atr = try_scalar(latest.get('ATR', np.nan))
    vwap = try_scalar(latest.get('VWAP', np.nan))

    metrics_row = st.columns([1.6,1,1,1,1,1])

    with metrics_row[0]:
    if price is None or (isinstance(price, float) and np.isnan(price)):
    st.metric("LTP", "—")
    elif isinstance(price, (list, tuple, np.ndarray, pd.Series)):
    latest_price = float(np.array(price).flatten()[-1]) # take last element
    st.metric("LTP", f"{round(latest_price, 2)}")
    else:
    st.metric("LTP", f"{round(float(price), 2)}")

    with metrics_row[1]: st.metric("RSI", f"{round(rsi,2) if not np.isnan(rsi) else '—'}")
    with metrics_row[2]: st.metric("MACD", f"{round(macd,2) if not np.isnan(macd) else '—'}")
    with metrics_row[3]: st.metric("ADX", f"{round(adx,2) if not np.isnan(adx) else '—'}")
    with metrics_row[4]: st.metric("ATR", f"{round(atr,2) if not np.isnan(atr) else '—'}")
    with metrics_row[5]: st.metric("VWAP", f"{round(vwap,2) if not np.isnan(vwap) else '—'}")

    tabs = st.tabs(["Chart", "TA", "Advanced", "Raw"])
    with tabs[0]:
        # Chart settings
        chart_interval = st.selectbox("Live Chart Interval", ["1m","5m","15m"], index=0, key="chart_interval")
        # Determine interval minutes
        interval_minutes = int(chart_interval.replace("m",""))
        show_bands = st.checkbox("Show Bollinger Bands (Plotly fallback)", value=False)

        # Build aggregated candles using live ticks + recent historical data
        # Strategy: take recent historical bars (yfinance) for context + append current aggregated candles built from ticks
        try:
            hist_bars = data.reset_index().rename(columns={"index":"time"})
            # Standardize time column to string format without timezone for frontend
            if 'time' in hist_bars.columns:
                hist_bars['time'] = pd.to_datetime(hist_bars['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            # aggregate live ticks for display
            live_ticks_for_symbol = st.session_state["live_ticks"].get(display_symbol, [])
            agg_candles = aggregate_ticks_to_ohlc(live_ticks_for_symbol, interval_minutes=interval_minutes)
            # Combine: keep historical bars of same interval (if available) and append agg_candles (taking last N)
            # If yfinance provided bars of same interval, take last 200 for context
            # For simplicity, convert historical to required columns
            if not hist_bars.empty and 'Open' in hist_bars.columns:
                hist_ohlc = hist_bars[['time','Open','High','Low','Close']].rename(columns={
                    'Open':'open','High':'high','Low':'low','Close':'close'
                })
                # Ensure same granularity: we'll just append aggregated candles to hist_ohlc
                combined = pd.concat([hist_ohlc, agg_candles.rename(columns={'volume':'volume'})], ignore_index=True, sort=False)
            else:
                combined = agg_candles
            # Clean combined, drop rows where open/high/low/close missing
            if not combined.empty:
                combined = combined.dropna(subset=['open','high','low','close'])
                # Keep last MAX_CANDLES rows
                if len(combined) > MAX_CANDLES:
                    combined = combined.iloc[-MAX_CANDLES:]
        except Exception as e:
            st.warning(f"Chart aggregation error: {e}")
            combined = pd.DataFrame()

        # Render candlestick chart (Lightweight Charts) inside iframe
        render_candlestick_in_iframe(combined, height=540)

    with tabs[1]:
        # TA charts using Streamlit plotting (Plotly/line_chart)
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
        st.download_button('Export Data to CSV', csv2, file_name=f"{display_symbol}_{screen_interval}.csv")

    with tabs[3]:
        if st.checkbox("Show Table Data"):
            st.dataframe(data.tail(80))

st.caption("BlockVista Terminal | Zerodha KiteConnect + Lightweight Charts — Candlestick aggregation server-side (1m/5m/15m).")

# ---------------------- Auto refresh to update chart from websocket ticks ----------------------
# This auto-refresh causes Streamlit to rerun the app at interval seconds,
# updating embedded iframe HTML with latest aggregated candles.
with st.sidebar:
    st.markdown("---")
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh App (updates chart)", value=st.session_state["auto_refresh"])
    st.session_state["refresh_interval"] = st.number_input("Refresh (sec)", value=st.session_state["refresh_interval"], min_value=3, max_value=60, step=1)

if st.session_state["auto_refresh"]:
    # st_autorefresh calls cause reruns, which update the iframe HTML with new candle data
    st_autorefresh(interval=st.session_state["refresh_interval"] * 1000, key="autorefresh")

# ---------------------- Clean exit handlers ----------------------
# Ensure websocket stopped on app teardown (best-effort)
def _cleanup():
    try:
        stop_kite_ticker()
    except Exception:
        pass

# You can optionally call cleanup on exit, but Streamlit lifecycle is managed by server process.
# _cleanup()
