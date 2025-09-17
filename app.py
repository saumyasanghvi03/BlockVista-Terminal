import os
import time
import json
import hashlib
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh
from alpha_vantage.timeseries import TimeSeries
import plotly.graph_objects as go
import feedparser
from dateutil.parser import parse
import pytz
import logging
import streamlit.components.v1 as components

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
except ModuleNotFoundError:
    vader_available = False
    st.warning("vaderSentiment not installed. Sentiment meter disabled.")

from kiteconnect import KiteTicker

# ---- WebSocket Session State ----
if "ws_ticks" not in st.session_state:
    st.session_state["ws_ticks"] = {}

def start_ws_ticker(symbols):
    tokens = []
    with st.spinner("Connecting WebSocket..."):
        profile = kite.profile()
        instruments = kite.instruments("NSE")
        symbol_to_token = {inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments}
        for s in symbols:
            if s in symbol_to_token:
                tokens.append(symbol_to_token[s])
        if not tokens:
            st.warning("No valid tokens found for selected symbols.")
            return
        api_key = kite._api_key
        access_token = kite._access_token
        kws = KiteTicker(api_key, access_token)
        def on_ticks(ws, ticks):
            for t in ticks:
                sym = next((k for k, v in symbol_to_token.items() if v == t["instrument_token"]), None)
                if sym:
                    st.session_state["ws_ticks"][sym] = t
        def on_connect(ws, response):
            ws.subscribe(tokens)
        def on_close(ws, code, reason):
            print("WebSocket closed", code, reason)
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        if not st.session_state.get("ws_running"):
            st.session_state["ws_running"] = True
            kws.connect(threaded=True)


# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"  # Ensure this is your valid Alpha Vantage API key
USERS_FILE = "users.json"
LOGIN_HISTORY_FILE = "login_history.json"
TRADE_LOGS_FILE = "trade_logs.json"
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
    "BRITANNIA": ["Britannia Industries"],
    "BEL": ["Bharat Electronics"],
    "HAL": ["Hindustan Aeronautics"]
}
NIFTY50_TOP = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS"]

# Setup logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Security questions
SECURITY_QUESTIONS = [
    "What is the name of your first pet?",
    "What is your mother's maiden name?",
    "What is the name of your high school?",
    "What is your favorite book?",
    "What is the city you were born in?"
]

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

def fetch_alpha_vantage_intraday(symbol, interval='1min', outputsize='compact', retries=3, delay=5):
    for attempt in range(retries):
        try:
            ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
            symbol_av = symbol.upper() + ".NSE" if not symbol.upper().endswith(".NSE") else symbol.upper()
            logging.debug(f"Fetching Alpha Vantage data for {symbol_av}, interval={interval}, attempt={attempt+1}")
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
            logging.debug(f"Alpha Vantage data for {symbol_av}: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Alpha Vantage error for {symbol_av}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            continue
    return None

# ---------------------- User Management ----------------------
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logging.error(f"Invalid users.json format: expected dict, got {type(data)}")
                return {}
            valid_users = {}
            for username, user_data in data.items():
                if isinstance(user_data, dict) and "access_code" in user_data:
                    user_data_clean = {
                        "access_code": user_data.get("access_code"),
                        "security_question": user_data.get("security_question"),
                        "security_answer": user_data.get("security_answer")
                    }
                    valid_users[username] = user_data_clean
                else:
                    logging.warning(f"Invalid user data for {username}: {user_data}")
            return valid_users
        return {}
    except Exception as e:
        logging.error(f"Error loading users: {e}")
        return {}

def save_users(users):
    try:
        if not isinstance(users, dict):
            logging.error(f"Attempted to save invalid users data: {type(users)}")
            return
        for username, user_data in users.items():
            if not isinstance(user_data, dict) or "access_code" not in user_data:
                logging.error(f"Invalid user data for {username}: {user_data}")
                return
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving users: {e}")

def load_login_history():
    try:
        if os.path.exists(LOGIN_HISTORY_FILE):
            with open(LOGIN_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading login history: {e}")
        return []

def save_login_history(history):
    try:
        with open(LOGIN_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving login history: {e}")

def load_trade_logs():
    try:
        if os.path.exists(TRADE_LOGS_FILE):
            with open(TRADE_LOGS_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading trade logs: {e}")
        return []

def save_trade_logs(logs):
    try:
        with open(TRADE_LOGS_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving trade logs: {e}")

def hash_access_code(access_code):
    return hashlib.sha256(access_code.encode()).hexdigest()

def signup_user(username, access_code, security_question, security_answer):
    if not all([username, access_code, security_question, security_answer]):
        return False, "All fields are required."
    if len(access_code) < 6:
        return False, "Access code must be at least 6 characters."
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "access_code": hash_access_code(access_code),
        "security_question": security_question,
        "security_answer": hash_access_code(security_answer)
    }
    save_users(users)
    if username not in st.session_state["user_activity"]:
        st.session_state["user_activity"][username] = []
    st.session_state["user_activity"][username].append({
        "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "action": "Signup",
        "details": "User account created"
    })
    return True, "Account created successfully!"

def login_user(username, access_code):
    users = load_users()
    login_history = load_login_history()
    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
    ip_address = "N/A"
    if username in users:
        if users[username]["access_code"] == hash_access_code(access_code):
            if username not in st.session_state["user_activity"]:
                st.session_state["user_activity"][username] = []
            login_history.append({
                "username": username,
                "timestamp": timestamp,
                "success": True,
                "reason": "Successful login",
                "ip_address": ip_address
            })
            save_login_history(login_history)
            st.session_state["user_activity"][username].append({
                "timestamp": timestamp,
                "action": "Login",
                "details": "User logged in"
            })
            return True, "Login successful."
        else:
            login_history.append({
                "username": username,
                "timestamp": timestamp,
                "success": False,
                "reason": "Invalid access code",
                "ip_address": ip_address
            })
            save_login_history(login_history)
            return False, "Invalid username or access code."
    login_history.append({
        "username": username,
        "timestamp": timestamp,
        "success": False,
        "reason": "Username does not exist",
        "ip_address": ip_address
    })
    save_login_history(login_history)
    return False, "Invalid username or access code."

def reset_user_password(username, security_answer, new_access_code):
    users = load_users()
    login_history = load_login_history()
    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
    ip_address = "N/A"
    if username not in users:
        login_history.append({
            "username": username,
            "timestamp": timestamp,
            "success": False,
            "reason": "Password reset failed: Username does not exist",
            "ip_address": ip_address
        })
        save_login_history(login_history)
        return False, "Username does not exist."
    if users[username]["security_answer"] != hash_access_code(security_answer):
        login_history.append({
            "username": username,
            "timestamp": timestamp,
            "success": False,
            "reason": "Password reset failed: Incorrect security answer",
            "ip_address": ip_address
        })
        save_login_history(login_history)
        return False, "Incorrect security answer."
    if len(new_access_code) < 6:
        login_history.append({
            "username": username,
            "timestamp": timestamp,
            "success": False,
            "reason": "Password reset failed: New access code too short",
            "ip_address": ip_address
        })
        save_login_history(login_history)
        return False, "New access code must be at least 6 characters."
    users[username]["access_code"] = hash_access_code(new_access_code)
    save_users(users)
    login_history.append({
        "username": username,
        "timestamp": timestamp,
        "success": True,
        "reason": "Password reset successful",
        "ip_address": ip_address
    })
    save_login_history(login_history)
    if username not in st.session_state["user_activity"]:
        st.session_state["user_activity"][username] = []
    st.session_state["user_activity"][username].append({
        "timestamp": timestamp,
        "action": "Password Reset",
        "details": "User reset password"
    })
    logging.info(f"User {username} reset their password")
    return True, "Password reset successfully."

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
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "login_attempts" not in st.session_state:
    st.session_state["login_attempts"] = 0
if "reset_attempts" not in st.session_state:
    st.session_state["reset_attempts"] = 0
if "user_activity" not in st.session_state:
    st.session_state["user_activity"] = {}
MAX_LOGIN_ATTEMPTS = 3
MAX_RESET_ATTEMPTS = 3

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

# ---------------------- Login/Signup/Reset UI ----------------------
if not st.session_state["logged_in"]:
    st.subheader("BlockVista Terminal Account")
    auth_mode = st.selectbox("Select Action", ["Login", "Signup", "Reset Password"])
    
    if auth_mode == "Signup":
        st.write("Create a new BlockVista Terminal account")
        signup_username = st.text_input("New Username")
        signup_access_code = st.text_input("New Access Code", type="password")
        security_question = st.selectbox("Security Question", SECURITY_QUESTIONS)
        security_answer = st.text_input("Security Answer", type="password")
        if st.button("Sign Up"):
            success, message = signup_user(signup_username, signup_access_code, security_question, security_answer)
            if success:
                st.success(message)
                st.session_state["logged_in"] = True
                st.session_state["username"] = signup_username
                st.session_state["login_attempts"] = 0
                st.session_state["reset_attempts"] = 0
            else:
                st.error(message)
    
    elif auth_mode == "Reset Password":
        st.write("Reset your password")
        reset_username = st.text_input("Username")
        users = load_users()
        if reset_username in users:
            st.write(f"Security Question: {users[reset_username]['security_question']}")
            reset_security_answer = st.text_input("Security Answer", type="password")
            new_access_code = st.text_input("New Access Code", type="password")
            if st.button("Reset Password"):
                if st.session_state["reset_attempts"] >= MAX_RESET_ATTEMPTS:
                    st.error("❌ Too many reset attempts. Please try again later.")
                    st.stop()
                success, message = reset_user_password(reset_username, reset_security_answer, new_access_code)
                if success:
                    st.session_state["reset_attempts"] = 0
                    st.success(message)
                    st.info("Please log in with your new access code.")
                else:
                    st.session_state["reset_attempts"] += 1
                    remaining = MAX_RESET_ATTEMPTS - st.session_state["reset_attempts"]
                    st.error(f"❌ {message}. {remaining} attempts remaining.")
        else:
            if reset_username:
                st.error("Username does not exist.")
    
    else:
        st.write("Login to your BlockVista Terminal account")
        login_username = st.text_input("Username")
        login_access_code = st.text_input("Access Code", type="password")
        if st.button("Login"):
            if st.session_state["login_attempts"] >= MAX_LOGIN_ATTEMPTS:
                st.error("❌ Too many failed attempts. Please try again later.")
                st.stop()
            success, message = login_user(login_username, login_access_code)
            if success:
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_username
                st.session_state["login_attempts"] = 0
                st.session_state["reset_attempts"] = 0
                st.success(f"✅ Welcome, {login_username}!")
            else:
                st.session_state["login_attempts"] += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state["login_attempts"]
                st.error(f"❌ {message}. {remaining} attempts remaining.")
    
    st.stop()

theme_choice = st.sidebar.selectbox("Terminal Theme", ["Bloomberg Dark", "Bloomberg Light"])
st.session_state.theme = theme_choice
set_terminal_style()

refresh_col, toggle_col = st.sidebar.columns([2,2])
with refresh_col:
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh", value=st.session_state.get("auto_refresh", True))
with toggle_col:
    st.session_state["refresh_interval"] = st.number_input("Sec", value=st.session_state.get("refresh_interval", 15), min_value=3, max_value=90, step=1)

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
    
    if "access_token" in st.session_state and "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
        try:
            kite_conn.set_access_token(st.session_state["access_token"])
            kite_conn.profile()
            return kite_conn
        except Exception as e:
            logging.error(f"Zerodha session invalid: {e}")
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
            logging.error(f"Zerodha auto-refresh failed: {e}")
            st.warning(f"Auto-refresh failed: {e}")

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
            logging.error(f"Zerodha login failed: {e}")
            st.error(f"❌ Zerodha login failed: {e}")
            st.stop()
    st.stop()

kite = get_kite_session()

st.sidebar.markdown("---")
if "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
    if st.sidebar.button("Logout from Zerodha"):
        st.session_state.pop("access_token", None)
        st.session_state.pop("zerodha_authenticated", None)
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

# ---- Live Ticker Display ----
selected_symbols = stock_list  # Adjust this if your variable name is different
if st.button("Start Live Ticker"):
    start_ws_ticker(selected_symbols)

if st.session_state.get("ws_ticks"):
    st.subheader("Live Ticker:")
    for sym, tick in st.session_state["ws_ticks"].items():
        st.write(f"**{sym}**: {tick.get('last_price', '—'):,} ({tick.get('change', 0):+})")

# ---------------------- Core Helpers ----------------------
def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        logging.error(f"Error getting live price for {symbol}: {e}")
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    try:
        logging.debug(f"Fetching yfinance data for {symbol}.NS, period={period}, interval={interval}")
        data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
        if data.empty or not isinstance(data, pd.DataFrame):
            logging.warning(f"yfinance returned empty or invalid data for {symbol}.NS")
            av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
            data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
            if data is None or data.empty:
                logging.warning(f"No data fetched for {symbol} from Alpha Vantage.")
                st.warning(f"No data fetched for {symbol}.")
                return None
        
        logging.debug(f"Data columns for {symbol}: {list(data.columns)}")
        logging.debug(f"Data head for {symbol}:\n{data.head().to_string()}")
        
        if isinstance(data.columns, pd.MultiIndex):
            logging.debug(f"MultiIndex detected for {symbol}. Attempting to flatten columns.")
            try:
                data = data.xs(f"{symbol}.NS", level=1, axis=1, drop_level=True)
            except KeyError:
                logging.warning(f"Symbol {symbol}.NS not found in MultiIndex columns: {data.columns}")
                available_cols = [col for col in data.columns if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']]
                if not available_cols:
                    logging.warning(f"No valid columns found for {symbol} in MultiIndex.")
                    st.warning(f"No valid columns found for {symbol}.")
                    return None
                data = data[available_cols]
                data.columns = [col[0] for col in data.columns]
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            logging.warning(f"Missing required columns for {symbol}: {list(data.columns)}")
            st.warning(f"Missing required columns for {symbol}: {list(data.columns)}")
            return None
        
        if len(data.columns[data.columns.duplicated()]) > 0:
            logging.warning(f"Duplicate columns detected for {symbol}: {data.columns[data.columns.duplicated()].tolist()}")
            data = data.loc[:, ~data.columns.duplicated(keep='first')]
        
        for col in required_cols:
            if not isinstance(data[col], pd.Series):
                logging.error(f"Column {col} for {symbol} is not a pandas Series: {type(data[col])}")
                st.warning(f"Invalid data type for {col} in {symbol}: {type(data[col])}")
                return None
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().all():
                logging.warning(f"Column {col} for {symbol} contains no valid numeric data.")
                st.warning(f"Column {col} for {symbol} contains no valid numeric data.")
                return None
        
        if len(data) < 2:
            logging.warning(f"Insufficient data for {symbol}: {len(data)} rows.")
            st.warning(f"Insufficient data for {symbol}: {len(data)} rows.")
            return None
        
        logging.debug(f"Processed data for {symbol}: {data.tail(5)}")
        
        try:
            if len(data) > 14:
                data['RSI'] = ta.rsi(data['Close'], length=14)
                data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
                data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx else np.nan
                stochrsi = ta.stochrsi(data['Close'], length=14)
                data['STOCHRSI'] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) and "STOCHRSIk_14_14_3_3" in stochrsi else np.nan
            else:
                data['RSI'] = np.nan
                data['ATR'] = np.nan
                data['ADX'] = np.nan
                data['STOCHRSI'] = np.nan
            
            if len(data) > 12:
                macd = ta.macd(data['Close'])
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                        data[col] = macd[col] if col in macd else np.nan
            else:
                data['MACD_12_26_9'] = np.nan
                data['MACDh_12_26_9'] = np.nan
                data['MACDs_12_26_9'] = np.nan
            
            if len(data) > 21:
                data['SMA21'] = ta.sma(data['Close'], length=21)
            else:
                data['SMA21'] = np.nan
            
            if len(data) > 9:
                data['EMA9'] = ta.ema(data['Close'], length=9)
            else:
                data['EMA9'] = np.nan
            
            if len(data) > 20:
                bbands = ta.bbands(data['Close'], length=20)
                if isinstance(bbands, pd.DataFrame):
                    for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
                        data[label] = bbands[key] if key in bbands else np.nan
            else:
                data['BOLL_L'] = np.nan
                data['BOLL_M'] = np.nan
                data['BOLL_U'] = np.nan
            
            if len(data) > 7:
                supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
                if isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
                    for col in supertrend.columns:
                        data[col] = supertrend[col]
            
            if len(data) > 0 and 'Volume' in data.columns:
                data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
            else:
                data['VWAP'] = np.nan
            
            ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
            if isinstance(ha, pd.DataFrame):
                for c in ['open', 'high', 'low', 'close']:
                    data[f'HA_{c}'] = ha[f'HA_{c}'] if f'HA_{c}' in ha else np.nan
        except Exception as e:
            logging.error(f"Error computing indicators for {symbol}: {e}")
            st.warning(f"Error computing indicators for {symbol}: {e}")
            return data
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
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
st.sidebar.subheader("📈 Watchlist P&L Tracker")
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
    except Exception as e:
        logging.error(f"Error calculating P&L for {s}: {e}")
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
    stock_list = [symbol.upper()]
else:
    basket = st.sidebar.selectbox("Pick Basket", list(SMALLCASE_BASKETS.keys()))
    stock_list = SMALLCASE_BASKETS[basket]

# Fallback: If nothing is chosen, make sure stock_list always exists!
if not stock_list or not isinstance(stock_list, list):
    stock_list = ["RELIANCE"]
selected_symbols = stock_list

    
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

if st.session_state["logged_in"] and len(stock_list) > 0:
    username = st.session_state["username"]
    if username not in st.session_state["user_activity"]:
        st.session_state["user_activity"][username] = []
    st.session_state["user_activity"][username].append({
        "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "action": "Screener Viewed",
        "details": f"Symbols: {', '.join(stock_list)}, Period: {screen_period}, Interval: {screen_interval}"
    })

# ---------------------- Sidebar: Order Placement ----------------------
st.sidebar.header("Place Order")
order_symbol = st.sidebar.text_input("Order Symbol (e.g. RELIANCE)", value="RELIANCE")
order_qty = st.sidebar.number_input("Quantity", min_value=1, value=1)
order_type = st.sidebar.selectbox("Order Type", ["MARKET", "LIMIT"])
transaction_type = st.sidebar.selectbox("Transaction Type", ["BUY", "SELL"])
order_price = None
order_product = st.sidebar.selectbox("Product", ["CNC", "MIS", "NRML"], index=0)

if order_type == "LIMIT":
    order_price = st.sidebar.number_input("Limit Price", min_value=0.0, value=0.0, format="%.2f")

if st.sidebar.button("Submit Order"):
    trade_logs = load_trade_logs()
    username = st.session_state["username"]
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
        trade_logs.append({
            "order_id": order_id,
            "username": username,
            "symbol": order_symbol.upper(),
            "quantity": int(order_qty),
            "order_type": order_type,
            "transaction_type": transaction_type,
            "product": order_product,
            "price": float(order_price) if order_type == "LIMIT" else None,
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "status": "Success"
        })
        save_trade_logs(trade_logs)
        st.sidebar.success(f"✅ Order placed! ID: {order_id}")
        logging.info(f"Order placed by {username}: {order_params}")
        if username not in st.session_state["user_activity"]:
            st.session_state["user_activity"][username] = []
        st.session_state["user_activity"][username].append({
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "action": "Order Placed",
            "details": f"Order ID: {order_id}, Symbol: {order_symbol}, Qty: {order_qty}, Type: {order_type}"
        })
    except Exception as e:
        trade_logs.append({
            "order_id": "N/A",
            "username": username,
            "symbol": order_symbol.upper(),
            "quantity": int(order_qty),
            "order_type": order_type,
            "transaction_type": transaction_type,
            "product": order_product,
            "price": float(order_price) if order_type == "LIMIT" else None,
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "status": f"Failed: {str(e)}"
        })
        save_trade_logs(trade_logs)
        st.sidebar.error(f"❌ Order failed: {e}")
        logging.error(f"Order failed by {username}: {e}")

if st.session_state.get("auto_refresh", True):
    st_autorefresh(interval=st.session_state.get("refresh_interval", 15) * 1000, key="chart_autorefresh")

df = fetch_stock_data(symbol, screen_period, screen_interval)
if df is None or df.empty:
    st.info("No chart data available")
else:
    if symbol in st.session_state.get("ws_ticks", {}):
        live_tick = st.session_state["ws_ticks"][symbol]
        df.iloc[-1, df.columns.get_loc("Close")] = live_tick["last_price"]

    render_lightweight_candles(df)


# ---------------------- Lightweight Charts Renderer ----------------------
def render_lightweight_candles(df, chart_style="Candlestick"):
    if df.empty:
        st.info("No data available for chart.")
        return
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
NSE_HOLIDAYS_2025 = [
    "2025-01-26", "2025-02-17", "2025-03-06", "2025-03-31", "2025-04-10",
    "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27",
    "2025-10-02", "2025-10-21", "2025-11-05", "2025-12-25"
]

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
current_date = now.strftime("%Y-%m-%d")
is_weekday = now.weekday() < 5
is_market_hours = (now.time() >= datetime.strptime("09:15", "%H:%M").time() and
                  now.time() <= datetime.strptime("15:30", "%H:%M").time())
is_holiday = current_date in NSE_HOLIDAYS_2025
market_status = "LIVE" if is_weekday and not is_holiday and is_market_hours else "Closed"
status_color = "#FFFF00" if market_status == "LIVE" else "#FF3333"

st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{ '#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0' },#1a1a1a 60%,#00CC00 100%);
     padding:10px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007;'>
        <span style='color:#FFFF00;font-family:"Courier New",monospace;font-size:2.1rem;font-weight:bold;letter-spacing:2px;'>
        BlockVista Terminal</span>
        <span style='float:right;color:{status_color};font-size:1.25rem;font-family:monospace;padding-top:16px;'>
        INDIA • INTRADAY • {market_status}</span>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------- Sentiment Meter ----------------------
with st.expander("NIFTY 50 Sentiment Meter", expanded=True):
    if vader_available:
        try:
            sentiment_score, sentiment_label = get_nifty50_sentiment()
            if not isinstance(sentiment_score, (int, float)) or pd.isna(sentiment_score):
                sentiment_score, sentiment_label = 0.0, "Neutral"
                delta_color = "normal"
            else:
                sentiment_score = float(sentiment_score)
                delta_color = "normal"
            if not isinstance(sentiment_label, str):
                sentiment_label = "Neutral"
            st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color=delta_color)
        except Exception as e:
            logging.error(f"Failed to render sentiment meter: {e}")
            st.warning(f"Failed to render sentiment meter: {e}")
            sentiment_score, sentiment_label = 0.0, "Neutral"
            st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}", delta_color="normal")
    else:
        sentiment_score, sentiment_label = 0.0, "Neutral"
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
    st.header(f"Dashboard: {display_symbol}")

    data = fetch_stock_data(display_symbol, screen_period, screen_interval)
    if data is None or data.empty:
        st.error("No data available for this symbol/interval.")
        st.stop()

    price = np.nan
    try:
        ltp_json = kite.ltp(f"NSE:{display_symbol}")
        price = ltp_json.get(f"NSE:{display_symbol}", {}).get("last_price", np.nan)
        if np.isnan(price):
            price = float(data["Close"].iloc[-1])
    except Exception as e:
        logging.error(f"Error getting LTP for {display_symbol}: {e}")
        price = float(data["Close"].iloc[-1]) if not data["Close"].empty else np.nan

    metrics_row = st.columns([1.5,1,1,1,1,1])
    latest = data.iloc[-1]
    
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    atr = try_scalar(latest.get('ATR', np.nan))
    vwap = try_scalar(latest.get('VWAP', np.nan))
    close_price = try_scalar(latest.get('Close', np.nan))

    with metrics_row[0]:
        if np.isnan(close_price) or np.isnan(price):
            st.metric("LTP", f"{round(float(price), 2) if not np.isnan(price) else '—'}", delta=None, delta_color="off")
        else:
            delta = price - close_price
            st.metric("LTP", f"{round(float(price), 2)}", delta=f"{round(delta, 2)}", delta_color="normal")
    with metrics_row[1]:
        rsi_label = f"{round(rsi, 2)} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else ''}" if not np.isnan(rsi) else '—'
        st.metric("RSI", rsi_label, delta=None, delta_color="off")
    with metrics_row[2]:
        if np.isnan(macd) or np.isnan(macds):
            st.metric("MACD", f"{round(macd, 2) if not np.isnan(macd) else '—'}", delta=None, delta_color="off")
        else:
            macd_delta = macd - macds
            st.metric("MACD", f"{round(macd, 2)}", delta=f"{round(macd_delta, 2)}", delta_color="normal")
    with metrics_row[3]:
        adx_label = f"{round(adx, 2)} {'(Strong)' if adx > 25 else ''}" if not np.isnan(adx) else '—'
        st.metric("ADX", adx_label, delta=None, delta_color="off")
    with metrics_row[4]:
        st.metric("ATR", f"{round(atr, 2) if not np.isnan(atr) else '—'}", delta=None, delta_color="off")
    with metrics_row[5]:
        st.metric("VWAP", f"{round(vwap, 2) if not np.isnan(vwap) else '—'}", delta=None, delta_color="off")

    tabs = st.tabs(["Chart", "TA", "Signals", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands", value=True)
        required_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in required_cols):
            hist_df = data.reset_index().rename(columns={"index": "time"})
            if 'time' not in hist_df.columns:
                hist_df['time'] = hist_df.index
            hist_df['time'] = pd.to_datetime(hist_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if chart_style == "Heikin Ashi":
                ohlc_cols = ['HA_open', 'HA_high', 'HA_low', 'HA_close']
                if all(col in data.columns for col in ohlc_cols):
                    chart_data = hist_df[['time', 'HA_open', 'HA_high', 'HA_low', 'HA_close']].rename(
                        columns={'HA_open': 'open', 'HA_high': 'high', 'HA_low': 'low', 'HA_close': 'close'}
                    ).dropna()
                else:
                    st.warning("Heikin Ashi data not available. Showing Candlestick chart.")
                    chart_data = hist_df[['time', 'Open', 'High', 'Low', 'Close']].rename(
                        columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
                    ).dropna()
            else:
                chart_data = hist_df[['time', 'Open', 'High', 'Low', 'Close']].rename(
                    columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
                ).dropna()
            render_lightweight_candles(chart_data, chart_style)
            if bands_show and all(col in data.columns for col in ['BOLL_L', 'BOLL_M', 'BOLL_U']):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df['time'], y=data['BOLL_U'], name='Upper Band', line=dict(color='rgba(255,165,0,0.5)')))
                fig.add_trace(go.Scatter(x=hist_df['time'], y=data['BOLL_M'], name='Middle Band', line=dict(color='rgba(255,255,255,0.5)')))
                fig.add_trace(go.Scatter(x=hist_df['time'], y=data['BOLL_L'], name='Lower Band', line=dict(color='rgba(255,165,0,0.5)'), fill='tonexty', fillcolor='rgba(255,165,0,0.2)'))
                fig.update_layout(
                    title="Bollinger Bands",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_dark" if st.session_state.theme == "Bloomberg Dark" else "plotly"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Required columns for chart not found.")
    
    with tabs[1]:
        if not data.empty:
            ta_data = data[['RSI', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'ADX', 'ATR', 'VWAP', 'SMA21', 'EMA9']].tail(10)
            st.dataframe(ta_data)
        else:
            st.info("No technical analysis data available.")
    
    with tabs[2]:
        signals = get_signals(data)
        if signals:
            st.json(signals)
        else:
            st.info("No signals available.")
    
    with tabs[3]:
        st.dataframe(data.tail(10))
