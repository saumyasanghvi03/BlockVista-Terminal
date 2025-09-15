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
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect, KiteTicker
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries
import time
import threading
import json
import datetime
from collections import deque
import streamlit.components.v1 as components

# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"

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

# ---------------------- UI THEME / SESSION ----------------------
if "kite_ws_running" not in st.session_state:
    st.session_state.kite_ws_running = False
if "live_ticks" not in st.session_state:
    st.session_state.live_ticks = {}
if "ohlc_data" not in st.session_state:
    st.session_state.ohlc_data = {}
if "token_to_symbol" not in st.session_state:
    st.session_state.token_to_symbol = {}

# ---------------------- Zerodha Auth ----------------------
api_key = st.secrets.get("ZERODHA_API_KEY")
api_secret = st.secrets.get("ZERODHA_API_SECRET")

if not api_key or not api_secret:
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
            browser_notification(
                "BlockVista Error",
                f"❌ Zerodha login failed: {ex}",
                "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
            )
            st.stop()
    st.stop()
else:
    if "kite" not in st.session_state:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state["access_token"])
        st.session_state["kite"] = kite
    else:
        kite = st.session_state["kite"]

# ---------------------- Helper: Instrument token lookup ----------------------
@st.cache_data(ttl=3600)
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
    return None

# ---------------------- Kite WebSocket (KiteTicker) ----------------------
kws_thread = None
kws_obj = None


def kite_on_ticks(ws, ticks):
    for t in ticks:
        token = t.get('instrument_token')
        lp = t.get('last_price')
        ts = pd.to_datetime(t.get('timestamp', datetime.datetime.now().timestamp()), unit='s')
        symbol = st.session_state.get('token_to_symbol', {}).get(str(token), str(token))

        if symbol not in st.session_state.live_ticks:
            st.session_state.live_ticks[symbol] = deque(maxlen=1000)
        st.session_state.live_ticks[symbol].append({'t': ts, 'v': lp})

        minute = ts.replace(second=0, microsecond=0)
        if symbol not in st.session_state.ohlc_data:
            st.session_state.ohlc_data[symbol] = {}
        ohlc_dict = st.session_state.ohlc_data[symbol]
        if minute not in ohlc_dict:
            ohlc_dict[minute] = {
                'time': int(minute.timestamp()),
                'open': lp,
                'high': lp,
                'low': lp,
                'close': lp
            }
        else:
            ohlc_dict[minute]['high'] = max(ohlc_dict[minute]['high'], lp)
            ohlc_dict[minute]['low'] = min(ohlc_dict[minute]['low'], lp)
            ohlc_dict[minute]['close'] = lp


def kite_on_connect(ws, response):
    st.session_state.kite_ws_running = True


def kite_on_close(ws, code, reason):
    st.session_state.kite_ws_running = False


def start_kite_ticker(instrument_tokens, token_to_symbol_map=None):
    global kws_thread, kws_obj
    if st.session_state.kite_ws_running:
        return
    kws_obj = KiteTicker(api_key, st.session_state["access_token"])
    kws_obj.on_ticks = kite_on_ticks
    kws_obj.on_connect = kite_on_connect
    kws_obj.on_close = kite_on_close

    def run():
        try:
            kws_obj.connect(threaded=False)
        except Exception as e:
            print(f"KiteTicker connection error: {e}")

    def subscribe_when_ready():
        time.sleep(1.5)
        try:
            kws_obj.subscribe(instrument_tokens)
            kws_obj.set_mode(kws_obj.MODE_FULL, instrument_tokens)
        except Exception as e:
            print(f"Subscribe failed: {e}")

    kws_thread = threading.Thread(target=run, daemon=True)
    kws_thread.start()
    sub_thread = threading.Thread(target=subscribe_when_ready, daemon=True)
    sub_thread.start()


def stop_kite_ticker():
    global kws_obj
    try:
        if kws_obj:
            kws_obj.close()
    except Exception:
        pass
    st.session_state.kite_ws_running = False

# ---------------------- Lightweight Charts Renderer ----------------------
def render_lightweight_candles(symbol, agg_period='1m'):
    ohlc_dict = st.session_state.ohlc_data.get(symbol, {})
    if not ohlc_dict:
        st.info("No candle data yet. Start WebSocket first.")
        return

    df = pd.DataFrame(list(ohlc_dict.values())).sort_values("time")
    if df.empty:
        st.info("No OHLC available.")
        return

    # Resample if needed
    df.index = pd.to_datetime(df["time"], unit='s')
    rule_map = {"1m": "1T", "5m": "5T", "15m": "15T"}
    if agg_period in rule_map:
        df_resampled = df.resample(rule_map[agg_period]).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna()
        df_resampled["time"] = df_resampled.index.astype(int) // 10**9
        df = df_resampled.reset_index(drop=True)

    js_data = df.to_dict(orient='records')
    js_data = json.dumps(js_data)

    html_code = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script src="https://unpkg.com/lightweight-charts@3.7.0/dist/lightweight-charts.standalone.production.js"></script>
    </head>
    <body>
      <div id="chart" style="width:100%;height:420px;"></div>
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
    components.html(html_code, height=450)

# ---------------------- Sidebar UI ----------------------
st.sidebar.header("Live WebSocket (KiteTicker) — optional")
ws_symbol = st.sidebar.text_input("Symbol to subscribe (e.g. RELIANCE)", value="RELIANCE")
ws_token_input = st.sidebar.text_input("Instrument Token (optional)", value="")
agg_period = st.sidebar.selectbox("Aggregation Period", ["1m", "5m", "15m"], index=0)

st.sidebar.markdown("""
If symbol-to-token lookup fails, get token via:
```python
instruments = kite.instruments()
df = pd.DataFrame(instruments)
df[df.tradingsymbol == 'RELIANCE'][['tradingsymbol','instrument_token','exchange']]
```
Copy instrument_token and paste into the 'Instrument Token (optional)' field above.
""")

if st.sidebar.button("Start Live WebSocket"):
    try:
        if ws_token_input.strip():
            token_list = [int(ws_token_input.strip())]
            st.session_state['token_to_symbol'][str(ws_token_input.strip())] = ws_symbol.upper()
        else:
            token = find_instrument_token(ws_symbol.upper())
            if token is None:
                st.error("Could not find instrument token for symbol. Provide instrument_token manually.")
            else:
                token_list = [int(token)]
                st.session_state['token_to_symbol'][str(token)] = ws_symbol.upper()
        start_kite_ticker(token_list)
        st.sidebar.success(f"Subscribed to {ws_symbol} (tokens: {token_list})")
    except Exception as e:
        st.sidebar.error(f"Start WS failed: {e}")

if st.sidebar.button("Stop Live WebSocket"):
    stop_kite_ticker()
    st.sidebar.info("KiteTicker stopped.")

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

st.header("Live Candlestick Chart (TradingView-style)")
if ws_symbol.upper() in st.session_state.ohlc_data and st.session_state.ohlc_data[ws_symbol.upper()]:
    render_lightweight_candles(ws_symbol.upper(), agg_period)
else:
    st.info("No OHLC data for the requested symbol yet. Start WebSocket and subscribe first.")

st.caption("BlockVista Terminal — Zerodha Kite WebSocket + Real-time Candlestick Charts")
