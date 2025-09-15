import os
import time
import json
import threading
import hashlib
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
import plotly.graph_objects as go
import feedparser
from dateutil.parser import parse

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
except ModuleNotFoundError:
    vader_available = False
    st.warning("vaderSentiment not installed. Sentiment meter disabled.")

# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"
MAX_TICKS_PER_SYMBOL = 4000
MAX_AGG_CANDLES = 500
USERS_FILE = "users.json"
SYMBOL_TO_COMPANY = {
    "RELIANCE": ["Reliance Industries", "Reliance"],
    "SBIN": ["State Bank of India", "SBI"],
    "TCS": ["Tata Consultancy Services", "TCS"],
    "HINDZINC": ["Hindustan Zinc"],
    "NMDC": ["NMDC"],
    "VEDL": ["Vedanta"],
    "MOIL": ["MOIL"],
    "RSWM": ["RSWM"],
    "MANAPPURAM": ["Manappuram Finance"],
    "MUTHOOTFIN": ["Muthoot Finance"],
    "MMTC": ["MMTC"],
    "HDFCBANK": ["HDFC Bank"],
    "ICICIBANK": ["ICICI Bank"],
    "KOTAKBANK": ["Kotak Mahindra Bank"],
    "AXISBANK": ["Axis Bank"],
    "HINDUNILVR": ["Hindustan Unilever"],
    "NESTLEIND": ["Nestle India"],
    "ITC": ["ITC"],
    "BRITANNIA": ["Britannia Industries"]
}
NIFTY50_TOP = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS"]

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
    except Exception:
        return None

# ---------------------- User Management ----------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_access_code(access_code):
    return hashlib.sha256(access_code.encode()).hexdigest()

def signup_user(username, access_code):
    if not username or not access_code:
        return False, "Username and access code cannot be empty."
    if len(access_code) < 6:
        return False, "Access code must be at least 6 characters."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_access_code(access_code)
    save_users(users)
    return True, "Account created successfully!"

def login_user(username, access_code):
    users = load_users()
    if username in users and users[username] == hash_access_code(access_code):
        return True
    return False

# ---------------------- Sentiment Analysis ----------------------
@st.cache_data(ttl=1800)
def get_nifty50_sentiment():
    if not vader_available:
        return 0.0, "Neutral"
    analyzer = SentimentIntensityAnalyzer()
    rss_urls = [
        "https://www.financialexpress.com/market/feed/",
        "http://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
        "https://www.livemint.com/rss/markets",
        "https://www.business-standard.com/rss/markets-102.cms",
        "https://www.cnbctv18.com/market/feed"
    ]
    scores = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = entry.title
            if hasattr(entry, 'description'):
                text += " " + entry.description
            text = text.upper()
            if "NIFTY 50" in text or any(SYMBOL_TO_COMPANY.get(s, [s])[0].upper() in text for s in NIFTY50_TOP):
                score = analyzer.polarity_scores(entry.title + (entry.description if hasattr(entry, 'description') else ""))
                scores.append(score['compound'])
    if not scores:
        return 0.0, "Neutral"
    avg_score = sum(scores) / len(scores)
    sentiment_label = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
    return avg_score, sentiment_label

# ---------------------- THEME & SESSION ----------------------
st.set_page_config(page_title="BlockVista Terminal", layout="wide")
if "theme" not in st.session_state:
    st.session_state.theme = "Bloomberg Dark"

def set_terminal_style():
    if st.session_state.theme == "Bloomberg Dark":
        st.markdown("""
        <style>
        .main, .block-container {background-color: #000000 !important;}
        .stSidebar {background: #1a1a1a !important;}
        .stDataFrame tbody tr {background-color: #1a1a1a !important; color: #FFFF00 !important;}
        .stTextInput input, .stNumberInput input, .stSelectbox {background: #2a2a2a !important; color: #FFFF00 !important;}
        .stMetric, .stMetricLabel {color: #FFFF00 !important;}
        .stMetricValue {color: #00CC00 !important;}
        h1, h2, h3, h4, h5, h6, label {color: #FFFF00 !important; font-family: 'Courier New', monospace;}
        .css-1v3fvcr {border: 1px solid #333333; border-radius: 8px;}
        .stButton button {background: #004d00; color: #FFFFFF; border-radius: 4px;}
        .stButton button:hover {background: #006600;}
        .stTabs [data-baseweb="tab"] {color: #FFFF00;}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #004d00; color: #FFFFFF;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main, .block-container {background-color: #F0F0F0 !important;}
        .stSidebar {background: #E0E0E0 !important;}
        .stDataFrame tbody tr {background-color: #E0E0E0 !important; color: #000000 !important;}
        .stTextInput input, .stNumberInput input, .stSelectbox {background: #FFFFFF !important; color: #000000 !important;}
        .stMetric, .stMetricLabel {color: #000000 !important;}
        .stMetricValue {color: #00CC00 !important;}
        h1, h2, h3, h4, h5, h6, label {color: #000000 !important; font-family: 'Courier New', monospace;}
        .css-1v3fvcr {border: 1px solid #333333; border-radius: 8px;}
        .stButton button {background: #004d00; color: #FFFFFF; border-radius: 4px;}
        .stButton button:hover {background: #006600;}
        .stTabs [data-baseweb="tab"] {color: #000000;}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #004d00; color: #FFFFFF;}
        </style>
        """, unsafe_allow_html=True)

if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 15
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "login_attempts" not in st.session_state:
    st.session_state["login_attempts"] = 0
MAX_LOGIN_ATTEMPTS = 3

# ---------------------- Login/Signup UI ----------------------
if not st.session_state["logged_in"]:
    st.subheader("BlockVista Terminal Account")
    auth_mode = st.selectbox("Select Action", ["Login", "Signup"])
    
    if auth_mode == "Signup":
        st.write("Create a new BlockVista Terminal account")
        signup_username = st.text_input("New Username")
        signup_access_code = st.text_input("New Access Code", type="password")
        if st.button("Sign Up"):
            success, message = signup_user(signup_username, signup_access_code)
            if success:
                st.success(message)
                st.session_state["logged_in"] = True
                st.session_state["username"] = signup_username
                st.session_state["login_attempts"] = 0
            else:
                st.error(message)
    
    else:
        st.write("Login to your BlockVista Terminal account")
        login_username = st.text_input("Username")
        login_access_code = st.text_input("Access Code", type="password")
        if st.button("Login"):
            if st.session_state["login_attempts"] >= MAX_LOGIN_ATTEMPTS:
                st.error("❌ Too many failed attempts. Please try again later.")
                st.stop()
            if login_user(login_username, login_access_code):
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_username
                st.session_state["login_attempts"] = 0
                st.success(f"✅ Welcome, {login_username}!")
            else:
                st.session_state["login_attempts"] += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state["login_attempts"]
                st.error(f"❌ Invalid username or access code. {remaining} attempts remaining.")
    
    st.stop()

theme_choice = st.sidebar.selectbox("Terminal Theme", ["Bloomberg Dark", "Bloomberg Light"])
st.session_state.theme = theme_choice
set_terminal_style()

refresh_col, toggle_col = st.sidebar.columns([2,2])
with refresh_col:
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh", value=st.session_state["auto_refresh"])
with toggle_col:
    st.session_state["refresh_interval"] = st.number_input("Sec", value=st.session_state["refresh_interval"], min_value=3, max_value=90, step=1)

# ---------------------- Smallcase Baskets ----------------------
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

# ---------------------- Zerodha Auth ----------------------
def get_api_creds():
    def get(name):
        if hasattr(st, 'secrets') and name in st.secrets:
            return st.secrets[name]
        return os.getenv(name)
    api_key = get("KITE_API_KEY") or get("ZERODHA_API_KEY")
    api_secret = get("KITE_API_SECRET") or get("ZERODHA_API_SECRET")
    refresh_token = get("KITE_REFRESH_TOKEN")
    return api_key, api_secret, refresh_token

api_key, api_secret, refresh_token = get_api_creds()

if not api_key or not api_secret:
    st.error("Zerodha API credentials missing. Set KITE_API_KEY and KITE_API_SECRET in Streamlit secrets or environment variables.")
    st.stop()

def get_kite_session():
    kite_conn = KiteConnect(api_key=api_key)
    
    # Check if already authenticated
    if "access_token" in st.session_state and "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
        try:
            kite_conn.set_access_token(st.session_state["access_token"])
            # Validate session with a simple API call
            kite_conn.profile()  # Will raise an exception if token is invalid
            return kite_conn
        except Exception as e:
            st.warning(f"Zerodha session invalid: {e}. Please re-authenticate.")
            st.session_state.pop("access_token", None)
            st.session_state.pop("zerodha_authenticated", None)

    if refresh_token:
        try:
            data = kite_conn.generate_session(refresh_token, api_secret=api_secret)
            access_token = data.get("access_token")
            if access_token:
                st.session_state["access_token"] = access_token
                st.session_state["zerodha_authenticated"] = True
                kite_conn.set_access_token(access_token)
                st.success("✅ Zerodha session auto-refreshed")
                return kite_conn
            else:
                st.warning("Auto-refresh did not return access_token.")
        except Exception as e:
            st.warning(f"Auto-refresh failed: {e}")

    # Show login UI only if not authenticated
    st.markdown(
        f"""
        <div style="background:#1a1a1a;padding:14px;border-radius:8px;border:1px solid #FFFF00;">
        🟢 <a href="{kite_conn.login_url()}" target="_blank"><b>Login to Zerodha</b></a><br>
        Paste the <b>request_token</b> from the redirect URL below and click 'Generate Access Token'. Save the <b>refresh_token</b> to secrets for auto-refresh.
        </div>
        """, unsafe_allow_html=True
    )
    request_token = st.text_input("Paste request_token here:")
    if st.button("Generate Access Token") and request_token:
        try:
            data = kite_conn.generate_session(request_token, api_secret=api_secret)
            access_token = data.get("access_token")
            refresh_token_printed = data.get("refresh_token")
            if access_token:
                st.session_state["access_token"] = access_token
                st.session_state["zerodha_authenticated"] = True
                kite_conn.set_access_token(access_token)
                st.success("✅ Zerodha session started!")
                if refresh_token_printed:
                    st.code(f"KITE_REFRESH_TOKEN={refresh_token_printed}")
                return kite_conn
            else:
                st.error("Failed to obtain access_token.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Zerodha login failed: {e}")
            st.stop()
    st.stop()

kite = get_kite_session()

# Add logout button in sidebar
st.sidebar.markdown("---")
if "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
    if st.sidebar.button("Logout from Zerodha"):
        st.session_state.pop("access_token", None)
        st.session_state.pop("zerodha_authenticated", None)
        st.session_state.pop("subscribed_tokens", None)
        st.session_state.pop("live_ticks", None)
        st.session_state.pop("ohlc_agg", None)
        st.session_state.pop("token_to_symbol", None)
        st.session_state["kws_running"] = False
        if st.session_state.get("kws_obj"):
            try:
                st.session_state["kws_obj"].close()
            except Exception:
                pass
        st.session_state["kws_obj"] = None
        st.session_state["kws_thread"] = None
        st.success("Logged out from Zerodha. Please re-authenticate.")

# ---------------------- News Parser from RSS ----------------------
@st.cache_data(ttl=600)
def get_news(symbol):
    rss_urls = [
        "https://www.financialexpress.com/market/feed/",
        "http://timesofindia.indiatimes.com/rssfeeds/1898055.cms",
        "https://www.livemint.com/rss/markets",
        "https://www.business-standard.com/rss/markets-102.cms",
        "https://www.cnbctv18.com/market/feed"
    ]
    news_items = []
    company_names = SYMBOL_TO_COMPANY.get(symbol.upper(), [symbol.upper()])
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title.upper()
            description = entry.description.upper() if hasattr(entry, 'description') else ""
            if any(name.upper() in title or name.upper() in description for name in company_names):
                published = entry.published if hasattr(entry, 'published') else "N/A"
                try:
                    published_dt = parse(published) if published != "N/A" else datetime.min
                except Exception:
                    published_dt = datetime.min
                news_items.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": published,
                    "published_dt": published_dt
                })
    news_items = sorted(news_items, key=lambda x: x["published_dt"], reverse=True)[:5]
    return news_items

# ---------------------- Core Helpers ----------------------
def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception:
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    try:
        data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
        if data.empty:
            av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
            data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
            if data is None or data.empty:
                st.warning(f"No data fetched for {symbol}.")
                return None
        
        # Validate columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            st.warning(f"Missing required columns for {symbol}: {list(data.columns)}")
            return None
        
        # Check for duplicate columns
        if len(data.columns[data.columns.duplicated()]) > 0:
            st.warning(f"Duplicate columns detected for {symbol}: {data.columns[data.columns.duplicated()].tolist()}")
            data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        # Ensure numeric data
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        try:
            data['RSI'] = ta.rsi(data['Close'], length=14) if len(data) > 14 else np.nan
            macd = ta.macd(data['Close'])
            if isinstance(macd, pd.DataFrame) and not macd.empty:
                for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                    data[col] = macd[col] if col in macd else np.nan
            data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 21 else np.nan
            data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 9 else np.nan
            bbands = ta.bbands(data['Close'], length=20)
            if isinstance(bbands, pd.DataFrame):
                for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
                    data[label] = bbands[key] if key in bbands else np.nan
            data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14) if len(data) > 14 else np.nan
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
            if isinstance(ha, pd.DataFrame):
                for c in ['open', 'high', 'low', 'close']:
                    data[f'HA_{c}'] = ha[f'HA_{c}'] if f'HA_{c}' in ha else np.nan
        except Exception as e:
            st.warning(f"Error computing indicators for {symbol}: {e}")
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None

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
    if data is None or data.empty:
        return {}
    latest = data.iloc[-1]
    signals = {}
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    supertrend_col = [c for c in data.columns if isinstance(c, str) and c.startswith('SUPERT_') and not c.endswith('_dir')]
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
        if data is not None and not data.empty:
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
try:
    entry_prices = [float(x) for x in positions_input.split(",") if x.strip()]
    quantities = [float(x) for x in qty_input.split(",") if x.strip()]
except ValueError:
    st.sidebar.error("Invalid entry prices or quantities. Use comma-separated numbers.")
    entry_prices, quantities = [], []

pnl_data = []
for i, s in enumerate(symbols):
    try:
        if i >= len(entry_prices) or i >= len(quantities):
            raise IndexError("Mismatch in symbols, prices, or quantities.")
        live = get_live_price(s)
        if np.isnan(live):
            d = fetch_stock_data(s, "1d", "5m")
            live = d["Close"].iloc[-1] if d is not None and not d.empty else np.nan
        pnl = (live - entry_prices[i]) * quantities[i]
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": live, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
    except Exception:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i] if i < len(entry_prices) else "—", "LTP": "Err", "Qty": quantities[i] if i < len(quantities) else "—", "P&L ₹": "Err"})
if pnl_data:
    st.sidebar.dataframe(pd.DataFrame(pnl_data))
    total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int, float)))
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)

# ---------------------- Sidebar: Screener ----------------------
st.sidebar.title('Multi-Screener Settings')
screener_mode = st.sidebar.radio("Screener Mode", ["Single Stock", "Basket"])
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
if not screen_df.empty:
    st.sidebar.dataframe(screen_df)
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---------------------- KiteTicker Live Integration ----------------------
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
        return pd.DataFrame(instruments)
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

def kite_on_ticks(ws, ticks):
    for t in ticks:
        token = t.get('instrument_token')
        lp = t.get('last_price')
        ts_field = t.get('timestamp')
        try:
            if isinstance(ts_field, str):
                ts = pd.to_datetime(ts_field)
            else:
                ts = pd.Timestamp.utcnow() if ts_field is None else pd.to_datetime(int(ts_field), unit='ms' if ts_field > 1e12 else 's')
        except Exception:
            ts = pd.Timestamp.utcnow()

        token_s = str(token)
        symbol_from_map = st.session_state["token_to_symbol"].get(token_s)
        if symbol_from_map is None:
            df_inst = load_instruments()
            if not df_inst.empty:
                rows = df_inst[df_inst['instrument_token'] == int(token)]
                symbol_from_map = rows.iloc[0]['tradingsymbol'] if not rows.empty else token_s
                st.session_state["token_to_symbol"][token_s] = symbol_from_map
            else:
                symbol_from_map = token_s
                st.session_state["token_to_symbol"][token_s] = symbol_from_map

        if symbol_from_map not in st.session_state["live_ticks"]:
            st.session_state["live_ticks"][symbol_from_map] = deque(maxlen=MAX_TICKS_PER_SYMBOL)
        st.session_state["live_ticks"][symbol_from_map].append({'t': ts, 'v': float(lp) if lp is not None else np.nan})

        minute = ts.replace(second=0, microsecond=0)
        if symbol_from_map not in st.session_state["ohlc_agg"]:
            st.session_state["ohlc_agg"][symbol_from_map] = {}
        ohlc_map = st.session_state["ohlc_agg"][symbol_from_map]
        key = minute.isoformat()
        if key not in ohlc_map:
            ohlc_map[key] = {'time': int(minute.timestamp()), 'open': lp, 'high': lp, 'low': lp, 'close': lp}
        else:
            if lp is not None:
                ohlc_map[key]['high'] = max(ohlc_map[key]['high'] or lp, lp)
                ohlc_map[key]['low'] = min(ohlc_map[key]['low'] or lp, lp)
                ohlc_map[key]['close'] = lp
        if len(ohlc_map) > MAX_AGG_CANDLES:
            keys_sorted = sorted(ohlc_map.keys())
            for oldk in keys_sorted[:-MAX_AGG_CANDLES]:
                del ohlc_map[oldk]

def kite_on_connect(ws, response):
    st.session_state["kws_running"] = True
    st.sidebar.info("KiteTicker connected.")

def kite_on_close(ws, code, reason):
    st.session_state["kws_running"] = False
    st.sidebar.warning(f"KiteTicker closed: {reason}")

def start_kite_ticker_thread():
    if st.session_state.get("kws_running") and st.session_state.get("kws_thread") and st.session_state["kws_thread"].is_alive():
        return
    try:
        kws = KiteTicker(api_key, st.session_state["access_token"])
        kws.on_ticks = kite_on_ticks
        kws.on_connect = kite_on_connect
        kws.on_close = kite_on_close
        def run():
            try:
                kws.connect(threaded=False)
            except Exception as e:
                st.error(f"KiteTicker connect error: {e}")
        th = threading.Thread(target=run, daemon=True)
        th.start()
        st.session_state["kws_obj"] = kws
        st.session_state["kws_thread"] = th
    except Exception as e:
        st.error(f"Failed to init KiteTicker: {e}")

def subscribe_to_token(token, symbol=None):
    try:
        kws = st.session_state.get("kws_obj")
        if kws is None or not st.session_state.get("kws_running"):
            start_kite_ticker_thread()
            kws = st.session_state.get("kws_obj")
        if kws and int(token) not in st.session_state["subscribed_tokens"]:
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
        if kws and st.session_state["subscribed_tokens"]:
            kws.unsubscribe(list(st.session_state["subscribed_tokens"]))
            st.session_state["subscribed_tokens"].clear()
    except Exception:
        pass

# ---------------------- Sidebar: Live WS Controls & Order Placement ----------------------
st.sidebar.header("Live WebSocket")
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
        token = int(ws_token_input.strip()) if ws_token_input.strip() else find_instrument_token(ws_symbol.upper())
        if token:
            ok = subscribe_to_token(token, ws_symbol.upper())
            if ok:
                st.sidebar.success(f"Subscribed {ws_symbol.upper()} (token {token})")
            else:
                st.sidebar.error("Subscription failed.")
        else:
            st.sidebar.error("Could not find instrument token. Provide it manually.")

if stop_ws:
    unsubscribe_all_tokens()
    try:
        kws = st.session_state.get("kws_obj")
        if kws:
            kws.close()
    except Exception:
        pass
    st.session_state["kws_obj"] = None
    st.session_state["kws_thread"] = None
    st.session_state["kws_running"] = False
    st.sidebar.info("KiteTicker stopped.")

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
        order_id = kite.place_order(**order_params)
        st.sidebar.success(f"✅ Order placed! ID: {order_id}")
    except Exception as e:
        st.sidebar.error(f"❌ Order failed: {e}")

# ---------------------- Helper: Aggregate Ticks to OHLC ----------------------
def aggregate_ticks_to_ohlc(symbol, minutes=1):
    ohlc_map = st.session_state.get("ohlc_agg", {}).get(symbol, {})
    if not ohlc_map:
        return pd.DataFrame()
    rows = []
    for k, v in ohlc_map.items():
        try:
            if any(x is None for x in [v['open'], v['high'], v['low'], 'close']):
                continue
            rows.append({
                'time': v['time'],
                'open': float(v['open']),
                'high': float(v['high']),
                'low': float(v['low']),
                'close': float(v['close'])
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(sorted(rows, key=lambda x: x['time']))
    if len(df) > MAX_AGG_CANDLES:
        df = df.iloc[-MAX_AGG_CANDLES:]
    return df

# ---------------------- Lightweight Charts Renderer ----------------------
def render_lightweight_candles(symbol, agg_period='1m'):
    df = aggregate_ticks_to_ohlc(symbol, minutes=1)
    if df.empty:
        st.info("No live aggregated candle data yet.")
        return
    df['time'] = pd.to_datetime(df['time'], unit='s')
    rule_map = {"1m": "1T", "5m": "5T", "15m": "15T"}
    if agg_period in rule_map:
        df = df.set_index('time').resample(rule_map[agg_period]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()
    js_rows = [
        {
            "time": int(pd.to_datetime(row['time']).timestamp()),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close'])
        }
        for _, row in df.iterrows()
    ]
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
          layout: {{background: {{type: 'solid', color: '{ '#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0' }'}}, textColor: '{ '#FFFF00' if st.session_state.theme == 'Bloomberg Dark' else '#000000' }'}},
          grid: {{vertLines: {{color: '#333333'}}, horzLines: {{color: '#333333'}}}},
          timeScale: {{timeVisible: true, secondsVisible: false}},
          rightPriceScale: {{borderColor: '#00CC00'}},
          crosshair: {{vertLine: {{color: '#00CC00'}}, horzLine: {{color: '#00CC00'}}}}
        }});
        const candleSeries = chart.addCandlestickSeries({{
          upColor: '#00CC00',
          downColor: '#FF3333',
          borderUpColor: '#00CC00',
          borderDownColor: '#FF3333',
          wickUpColor: '#00CC00',
          wickDownColor: '#FF3333'
        }});
        candleSeries.setData({js_data});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=540)

# ---------------------- Main UI ----------------------
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{ '#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0' },#1a1a1a 60%,#00CC00 100%);
     padding:10px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007;'>
        <span style='color:#FFFF00;font-family:"Courier New",monospace;font-size:2.1rem;font-weight:bold;letter-spacing:2px;'>
        BlockVista Terminal</span>
        <span style='float:right;color:#FFFF00;font-size:1.25rem;font-family:monospace;padding-top:16px;'>
        INDIA • INTRADAY • LIVE</span>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------- Sentiment Meter ----------------------
if vader_available:
    try:
        sentiment_score, sentiment_label = get_nifty50_sentiment()
        # Validate sentiment_score and sentiment_label
        if not isinstance(sentiment_score, (int, float)) or pd.isna(sentiment_score):
            sentiment_score, sentiment_label = 0.0, "Neutral"
            delta_color = "normal"
        else:
            sentiment_score = float(sentiment_score)  # Ensure float
            delta_color = "normal"  # Use Streamlit's default coloring
        if not isinstance(sentiment_label, str):
            sentiment_label = "Neutral"
        with st.expander("NIFTY 50 Sentiment Meter", expanded=True):
            st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color=delta_color)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#FFFF00"},
                    'bar': {'color': "#00CC00" if sentiment_score > 0 else "#FF3333"},
                    'bgcolor': "#000000" if st.session_state.theme == "Bloomberg Dark" else "#F0F0F0",
                    'bordercolor': "#FFFF00",
                    'steps': [
                        {'range': [-1, -0.05], 'color': "#FF3333"},
                        {'range': [-0.05, 0.05], 'color': "#666666"},
                        {'range': [0.05, 1], 'color': "#00CC00"}
                    ]
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to render sentiment meter: {e}")
        sentiment_score, sentiment_label = 0.0, "Neutral"
        with st.expander("NIFTY 50 Sentiment Meter", expanded=True):
            st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color="normal")
else:
    sentiment_score, sentiment_label = 0.0, "Neutral"
    with st.expander("NIFTY 50 Sentiment Meter", expanded=True):
        st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color="normal")

# ---------------------- News Section Below Sentiment Meter ----------------------
if len(stock_list):
    display_symbol = stock_list[0].upper()
    with st.expander(f"Latest News for {display_symbol}", expanded=False):
        if st.button("Refresh News"):
            st.cache_data.clear()
        news = get_news(display_symbol)
        if news:
            for n in news:
                st.markdown(f"[{n['title']}]({n['link']}) - {n['published']}")
        else:
            st.info("No recent news found for this symbol.")

if len(stock_list):
    display_symbol = stock_list[0].upper()
    st.header(f"Live Dashboard: {display_symbol}")

    data = fetch_stock_data(display_symbol, screen_period, screen_interval)
    if data is None or data.empty:
        st.error("No data available for this symbol/interval.")
        st.stop()

    # Debug DataFrame structure
    st.write(f"Debug: data.columns = {list(data.columns)}")
    st.write(f"Debug: data.index = {data.index[:5]}")
    st.write(f"Debug: data.tail(1) = {data.tail(1)}")

    price = np.nan
    try:
        if display_symbol in st.session_state.get("live_ticks", {}) and st.session_state["live_ticks"][display_symbol]:
            price = st.session_state["live_ticks"][display_symbol][-1]['v']
        else:
            ltp_json = kite.ltp(f"NSE:{display_symbol}")
            price = ltp_json.get(f"NSE:{display_symbol}", {}).get("last_price", np.nan)
            if np.isnan(price):
                price = float(data["Close"].iloc[-1])
    except Exception:
        price = np.nan

    metrics_row = st.columns([1.5,1,1,1,1,1])
    latest = data.iloc[-1]
    # Debug latest Series
    st.write(f"Debug: type(latest) = {type(latest)}, latest['Close'] = {latest['Close']}, type(latest['Close']) = {type(latest['Close'])}")
    
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    atr = try_scalar(latest.get('ATR', np.nan))
    vwap = try_scalar(latest.get('VWAP', np.nan))
    close_price = try_scalar(latest.get('Close', np.nan))  # Ensure scalar

    with metrics_row[0]:
        if np.isnan(close_price) or np.isnan(price):
            st.metric("LTP", f"{round(float(price), 2) if not np.isnan(price) else '—'}", delta=None, delta_color="off")
        else:
            delta = price - close_price
            st.metric("LTP", f"{round(float(price), 2)}", delta=f"{round(delta, 2)}", delta_color="normal")
    with metrics_row[1]:
        st.metric("RSI", f"{round(rsi, 2) if not np.isnan(rsi) else '—'}", delta_color="red" if rsi > 70 else "green" if rsi < 30 else "normal")
    with metrics_row[2]:
        st.metric("MACD", f"{round(macd, 2) if not np.isnan(macd) else '—'}", delta_color="green" if macd > try_scalar(latest.get('MACDs_12_26_9', np.nan)) else "red")
    with metrics_row[3]:
        st.metric("ADX", f"{round(adx, 2) if not np.isnan(adx) else '—'}", delta_color="green" if adx > 25 else "normal")
    with metrics_row[4]:
        st.metric("ATR", f"{round(atr, 2) if not np.isnan(atr) else '—'}")
    with metrics_row[5]:
        st.metric("VWAP", f"{round(vwap, 2) if not np.isnan(vwap) else '—'}")

    try:
        agg_live = aggregate_ticks_to_ohlc(display_symbol, minutes=1)
    except Exception:
        agg_live = pd.DataFrame()

    combined = None
    try:
        hist_df = data.reset_index().rename(columns={"index": "time"})
        if 'time' in hist_df.columns:
            hist_df['time'] = pd.to_datetime(hist_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        if all(col in hist_df for col in ['Open', 'High', 'Low', 'Close']):
            hist_ohlc = hist_df[['time', 'Open', 'High', 'Low', 'Close']].rename(
                columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
            )
        else:
            hist_ohlc = pd.DataFrame()
        if not agg_live.empty:
            combined = pd.concat([hist_ohlc, agg_live], ignore_index=True, sort=False)
        else:
            combined = hist_ohlc
        if combined is not None and not combined.empty:
            combined = combined.dropna(subset=['open', 'high', 'low', 'close'])
    except Exception as e:
        st.warning(f"Could not build combined candle dataset: {e}")
        combined = None

    tabs = st.tabs(["Chart", "TA", "Signals", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands", value=True)
        if display_symbol in st.session_state.get("ohlc_agg", {}) and st.session_state["ohlc_agg"][display_symbol]:
            render_lightweight_candles(display_symbol, agg_period)
        else:
            st.info("No live OHLC data. Showing historical chart.")
            required_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in data.columns for col in required_cols):
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close']
                    )
                ])
                if bands_show and all(col in data for col in ['BOLL_L', 'BOLL_M', 'BOLL_U']):
                    fig.add_trace(go.Scatter(x=data.index, y=data['BOLL_U'], name='Upper BB', line=dict(color='#00CC00')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BOLL_L'], name='Lower BB', line=dict(color='#FF3333')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['BOLL_M'], name='Middle BB', line=dict(color='#FFFF00')))
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    template='plotly_dark' if st.session_state.theme == "Bloomberg Dark" else 'plotly_white',
                    height=640
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for candlestick chart. Required columns: Open, High, Low, Close.")

    with tabs[1]:
        st.subheader("Technical Analysis")
        ta_cols = [c for c in ['RSI', 'ADX', 'STOCHRSI'] if c in data.columns]
        if ta_cols and not data[ta_cols].dropna().empty:
            st.line_chart(data[ta_cols].dropna())
        else:
            st.info("No valid technical analysis data available.")
        macd_cols = [c for c in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if c in data.columns]
        if macd_cols and not data[macd_cols].dropna().empty:
            st.line_chart(data[macd_cols].dropna())
        else:
            st.info("No valid MACD data available.")
        if 'ATR' in data.columns and not data['ATR'].dropna().empty:
            st.line_chart(data['ATR'].dropna())
        else:
            st.info("No valid ATR data available.")
        last_cols = [c for c in ['Close', 'RSI', 'ADX', 'STOCHRSI', 'ATR', 'VWAP'] if c in data.columns]
        if last_cols and not data[last_cols].dropna().empty:
            st.write("Latest Values:", data.iloc[-1][last_cols])
        else:
            st.info("No valid latest values available.")

    with tabs[2]:
        st.subheader("Signals & Analysis")
        signals = get_signals(data)
        cols = st.columns(3)
        for i, (k, v) in enumerate(signals.items()):
            with cols[i % 3]:
                delta_color = "green" if "Bullish" in v or "Strong" in v or "Oversold" in v else "red" if "Bearish" in v or "Overbought" in v else "normal"
                st.metric(label=k, value=v, delta_color=delta_color)
        advanced_cols = [c for c in ['SMA21', 'EMA9', 'BOLL_L', 'BOLL_M', 'BOLL_U', 'ATR', 'VWAP'] if c in data.columns]
        if advanced_cols and not data[advanced_cols].tail(20).dropna().empty:
            st.line_chart(data[advanced_cols].tail(20))
        else:
            st.info("No valid advanced indicators available.")

    with tabs[3]:
        st.subheader("Raw Data")
        raw_df = combined if combined is not None and not combined.empty else data
        if not raw_df.empty:
            st.dataframe(raw_df)
        else:
            st.info("No raw data available.")

    if st.session_state.get("auto_refresh", True):
        st_autorefresh(interval=st.session_state.get("refresh_interval", 15) * 1000, key="data_refresh")

    if st.sidebar.button("Notify LTP"):
        browser_notification(f"{display_symbol} Live Price", f"LTP: {price if not np.isnan(price) else '—'}")

    st.caption("BlockVista Terminal | Powered by Zerodha KiteConnect, yFinance, Alpha Vantage")
