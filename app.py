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
from kiteconnect import KiteConnect, KiteTicker
import threading
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

# ---------------------- CONFIG ----------------------
AV_API_KEY = "2R0I2OXW1A1HMD9N"
USERS_FILE = "users.json"
LOGIN_HISTORY_FILE = "login_history.json"
TRADE_LOGS_FILE = "trade_logs.json"
LIVE_QUOTES = {} # Global dictionary for live prices
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

# ---------------------- UTILITIES (Includes New Kite Functions) ----------------------

@st.cache_data
def get_instrument_tokens():
    """Reads instruments.csv and creates a symbol-to-token map."""
    try:
        df = pd.read_csv("instruments.csv")
        eq_df = df[(df['exchange'] == 'NSE') & (df['instrument_type'] == 'EQ')]
        return dict(zip(eq_df['tradingsymbol'], eq_df['instrument_token']))
    except FileNotFoundError:
        st.error("`instruments.csv` not found. Please download it from Kite and place it in the root directory.")
        return {}
    except Exception as e:
        logging.error(f"Error reading instruments.csv: {e}")
        return {}

def fetch_kite_historical(symbol, interval, from_dt, to_dt, kite_client):
    """Fetches historical data from Kite, returns a standardized DataFrame."""
    try:
        instrument_token_map = st.session_state.get("instrument_token_map", {})
        if symbol.upper() not in instrument_token_map:
            logging.warning(f"Instrument token not found for {symbol}")
            return None
        
        instrument_token = instrument_token_map[symbol.upper()]
        data = kite_client.historical_data(instrument_token, from_dt, to_dt, interval)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return None
    except Exception as e:
        logging.error(f"Kite historical data error for {symbol}: {e}")
        return None

def fetch_alpha_vantage_intraday(symbol, interval='1min', outputsize='compact', retries=3, delay=5):
    for attempt in range(retries):
        try:
            ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
            symbol_av = symbol.upper() + ".NSE" if not symbol.upper().endswith(".NSE") else symbol.upper()
            logging.debug(f"Fetching Alpha Vantage data for {symbol_av}, interval={interval}, attempt={attempt+1}")
            data, _ = ts.get_intraday(symbol=symbol_av, interval=interval, outputsize=outputsize)
            df = data.rename(columns={'1. open': "Open", '2. high': "High", '3. low': "Low", '4. close': "Close", '5. volume': "Volume"})
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            logging.debug(f"Alpha Vantage data for {symbol_av}: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Alpha Vantage error for {symbol_av}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
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
    
    else: # Login
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

# ---------------------- Kite WebSocket Live Ticker ----------------------
def start_live_ticker(api_key, access_token, instrument_tokens):
    kws = KiteTicker(api_key, access_token)
    token_to_symbol_map = st.session_state.get("token_to_symbol_map", {})

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick.get("instrument_token")
            symbol = token_to_symbol_map.get(token)
            ltp = tick.get("last_price")
            if symbol and ltp is not None:
                LIVE_QUOTES[symbol] = ltp

    def on_connect(ws, response):
        logging.info("Kite Ticker: Connected.")
        ws.subscribe(instrument_tokens)
        ws.set_mode(ws.MODE_LTP, instrument_tokens)

    def on_close(ws, code, reason):
        logging.warning(f"Kite Ticker: Connection closed. Code: {code}, Reason: {reason}")

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    
    ticker_thread = threading.Thread(target=lambda: kws.connect(threaded=True), daemon=True)
    ticker_thread.start()
    logging.info("Kite Ticker thread started.")

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
    
    def initialize_ticker_and_tokens():
        """Loads tokens and starts the live ticker once per session."""
        if 'ticker_started' not in st.session_state:
            st.info("Initializing live data feed...")
            instrument_token_map = get_instrument_tokens()
            if not instrument_token_map:
                st.error("Could not load instrument tokens. Live data feed will not start.")
                return
            st.session_state["instrument_token_map"] = instrument_token_map
            st.session_state["token_to_symbol_map"] = {v: k for k, v in instrument_token_map.items()}
            
            watchlist_symbols = st.session_state.get("watchlist_symbols", NIFTY50_TOP)
            important_symbols = set(watchlist_symbols)
            tokens_to_track = [instrument_token_map[s] for s in important_symbols if s in instrument_token_map]
            
            if tokens_to_track:
                start_live_ticker(api_key, st.session_state["access_token"], tokens_to_track)
                st.session_state['ticker_started'] = True
                st.rerun()

    if "access_token" in st.session_state and st.session_state.get("zerodha_authenticated"):
        try:
            kite_conn.set_access_token(st.session_state["access_token"])
            kite_conn.profile()
            initialize_ticker_and_tokens()
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
                initialize_ticker_and_tokens()
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
        Paste the <b>request_token</b> from the redirect URL below and click 'Generate Access Token'.
        Save the <b>refresh_token</b> to secrets for auto-refresh.
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
                initialize_ticker_and_tokens()
                st.rerun()
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

# ---- Start Sidebar ----
st.sidebar.markdown("---")
if "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
    if st.sidebar.button("Logout from Zerodha"):
        st.session_state.pop("access_token", None)
        st.session_state.pop("zerodha_authenticated", None)
        st.session_state.pop("ticker_started", None)
        st.success("Logged out from Zerodha. Please re-authenticate.")
        st.rerun()

theme_choice = st.sidebar.selectbox("Terminal Theme", ["Bloomberg Dark", "Bloomberg Light"])
st.session_state.theme = theme_choice
set_terminal_style()

refresh_col, toggle_col = st.sidebar.columns([2,2])
with refresh_col:
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh", value=st.session_state.get("auto_refresh", True))
with toggle_col:
    st.session_state["refresh_interval"] = st.number_input("Sec", value=st.session_state.get("refresh_interval", 15), min_value=3, max_value=90, step=1)

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
    # 1. Check WebSocket quotes first
    if symbol in LIVE_QUOTES:
        return LIVE_QUOTES[symbol]
    
    # 2. Fallback to LTP API call
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        price = ltp[f"NSE:{symbol}"]["last_price"]
        LIVE_QUOTES[symbol] = price # Update cache
        return price
    except Exception as e:
        logging.error(f"Error getting live price for {symbol}: {e}")
        return np.nan

def process_indicators(data, symbol):
    """Helper function to clean and add indicators to a DataFrame."""
    try:
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            logging.warning(f"Indicator calc skipped: Missing required columns for {symbol}")
            return data
        
        if len(data) > 14:
            data['RSI'] = ta.rsi(data['Close'], length=14)
            data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
            adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
            data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx else np.nan
            stochrsi = ta.stochrsi(data['Close'], length=14)
            data['STOCHRSI'] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) and "STOCHRSIk_14_14_3_3" in stochrsi else np.nan
        else:
            data['RSI'], data['ATR'], data['ADX'], data['STOCHRSI'] = np.nan, np.nan, np.nan, np.nan
        
        if len(data) > 12:
            macd = ta.macd(data['Close'])
            if isinstance(macd, pd.DataFrame) and not macd.empty:
                for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                    data[col] = macd[col] if col in macd else np.nan
        else:
            data['MACD_12_26_9'], data['MACDh_12_26_9'], data['MACDs_12_26_9'] = np.nan, np.nan, np.nan
        
        data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 21 else np.nan
        data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 9 else np.nan

        if len(data) > 20:
            bbands = ta.bbands(data['Close'], length=20)
            if isinstance(bbands, pd.DataFrame):
                for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
                    data[label] = bbands[key] if key in bbands else np.nan
        else:
            data['BOLL_L'], data['BOLL_M'], data['BOLL_U'] = np.nan, np.nan, np.nan
        
        if len(data) > 7:
            supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
            if isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
                for col in supertrend.columns:
                    data[col] = supertrend[col]

        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume']) if 'Volume' in data.columns else np.nan
        
        ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
        if isinstance(ha, pd.DataFrame):
            for c in ['open', 'high', 'low', 'close']:
                data[f'HA_{c}'] = ha[f'HA_{c}'] if f'HA_{c}' in ha else np.nan
        
        return data
    except Exception as e:
        logging.error(f"Error computing indicators for {symbol}: {e}")
        return data

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    """Fetches stock data, prioritizing Kite, then yfinance, then Alpha Vantage."""
    if "zerodha_authenticated" in st.session_state and st.session_state["zerodha_authenticated"]:
        logging.debug(f"Attempting to fetch {symbol} from Kite...")
        to_dt = datetime.now()
        days_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
        from_dt = to_dt - timedelta(days=days_map.get(period, 5))
        interval_map = {"1m": "minute", "5m": "5minute", "15m": "15minute", "30m": "30minute", "60m": "60minute", "1d": "day"}
        kite_interval = interval_map.get(interval)
        if kite_interval:
            data = fetch_kite_historical(symbol, kite_interval, from_dt, to_dt, kite)
            if data is not None and not data.empty:
                logging.info(f"Successfully fetched {symbol} from Kite.")
                return process_indicators(data, symbol)
    
    try:
        logging.debug(f"Fetching yfinance data for {symbol}.NS...")
        data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
        if data.empty or not isinstance(data, pd.DataFrame):
            logging.warning(f"yfinance failed for {symbol}.NS, trying Alpha Vantage.")
            av_interval = {"1m": "1min", "5m": "5min", "15m": "15min"}.get(interval, "5min")
            data = fetch_alpha_vantage_intraday(symbol, interval=av_interval)
            if data is None or data.empty:
                st.warning(f"No data fetched for {symbol}.")
                return None
        return process_indicators(data.copy(), symbol)
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
    signals['Supertrend'] = 'Bullish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend < close else 'Bearish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend > close else 'Unknown'
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
st.session_state['watchlist_symbols'] = [x.strip().upper() for x in watchlist.split(",") if x.strip()]
positions_input = st.sidebar.text_area("Entry prices (comma, same order)", value="2550, 610, 3580")
qty_input = st.sidebar.text_area("Quantities (comma, same order)", value="10, 20, 5")
symbols = st.session_state['watchlist_symbols']
try:
    entry_prices = [float(x) for x in positions_input.split(",") if x.strip()]
    quantities = [float(x) for x in qty_input.split(",") if x.strip()]
except ValueError:
    st.sidebar.error("Invalid entry prices or quantities. Use comma-separated numbers.")
    entry_prices, quantities = [], []

pnl_data = []
if len(symbols) == len(entry_prices) == len(quantities):
    for i, s in enumerate(symbols):
        try:
            live = get_live_price(s)
            if np.isnan(live):
                d = fetch_stock_data(s, "1d", "5m")
                live = d["Close"].iloc[-1] if d is not None and not d.empty else np.nan
            pnl = (live - entry_prices[i]) * quantities[i]
            pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": live, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
        except Exception as e:
            logging.error(f"Error calculating P&L for {s}: {e}")
            pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
    if pnl_data:
        st.sidebar.dataframe(pd.DataFrame(pnl_data))
        total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int, float)))
        st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)
else:
    st.sidebar.warning("Mismatch in the number of symbols, prices, or quantities.")

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
            "tradingsymbol": order_symbol.upper(), "exchange": "NSE",
            "transaction_type": transaction_type, "quantity": int(order_qty),
            "order_type": order_type, "product": order_product, "variety": "regular"
        }
        if order_type == "LIMIT":
            order_params["price"] = float(order_price)
        order_id = kite.place_order(**order_params)
        trade_logs.append({
            "order_id": order_id, "username": username, "symbol": order_symbol.upper(),
            "quantity": int(order_qty), "order_type": order_type, "transaction_type": transaction_type,
            "product": order_product, "price": float(order_price) if order_type == "LIMIT" else None,
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(), "status": "Success"
        })
        save_trade_logs(trade_logs)
        st.sidebar.success(f"✅ Order placed! ID: {order_id}")
    except Exception as e:
        trade_logs.append({
            "order_id": "N/A", "username": username, "symbol": order_symbol.upper(),
            "quantity": int(order_qty), "order_type": order_type, "transaction_type": transaction_type,
            "product": order_product, "price": float(order_price) if order_type == "LIMIT" else None,
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(), "status": f"Failed: {str(e)}"
        })
        save_trade_logs(trade_logs)
        st.sidebar.error(f"❌ Order failed: {e}")

# ---------------------- Lightweight Charts Renderer (Repaired) ----------------------
def render_lightweight_chart(df, chart_style="Candlestick", show_bollinger=True):
    if df is None or df.empty:
        st.info("No data available for chart.")
        return

    # Prepare OHLC data
    ohlc_cols = ['HA_open', 'HA_high', 'HA_low', 'HA_close'] if chart_style == "Heikin Ashi" and all(c in df.columns for c in ['HA_open', 'HA_high', 'HA_low', 'HA_close']) else ['Open', 'High', 'Low', 'Close']
    df_chart = df.reset_index()
    df_chart = df_chart.rename(columns={'index': 'time', 'date': 'time'})
    
    ohlc_data = [
        {"time": int(pd.to_datetime(row['time']).timestamp()), "open": row[ohlc_cols[0]], "high": row[ohlc_cols[1]], "low": row[ohlc_cols[2]], "close": row[ohlc_cols[3]]}
        for _, row in df_chart.iterrows() if pd.notna(row[ohlc_cols[0]])
    ]
    
    # Prepare Bollinger Bands data if requested and available
    bollinger_upper, bollinger_middle, bollinger_lower = "[]", "[]", "[]"
    if show_bollinger and all(c in df.columns for c in ['BOLL_U', 'BOLL_M', 'BOLL_L']):
        bollinger_upper = json.dumps([{"time": int(pd.to_datetime(row['time']).timestamp()), "value": row['BOLL_U']} for _, row in df_chart.iterrows() if pd.notna(row['BOLL_U'])])
        bollinger_middle = json.dumps([{"time": int(pd.to_datetime(row['time']).timestamp()), "value": row['BOLL_M']} for _, row in df_chart.iterrows() if pd.notna(row['BOLL_M'])])
        bollinger_lower = json.dumps([{"time": int(pd.to_datetime(row['time']).timestamp()), "value": row['BOLL_L']} for _, row in df_chart.iterrows() if pd.notna(row['BOLL_L'])])
        
    html_code = f"""
    <!doctype html><html><head><meta charset="utf-8"></head><body>
      <div id="chart" style="width:100%;height:520px;"></div>
      <script src="https://unpkg.com/lightweight-charts@3.7.0/dist/lightweight-charts.standalone.production.js"></script>
      <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
          layout: {{background: {{type: 'solid', color: '{'#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0'}'}}, textColor: '{'#FFFF00' if st.session_state.theme == 'Bloomberg Dark' else '#000000'}'}},
          grid: {{vertLines: {{color: '#333333'}}, horzLines: {{color: '#333333'}}}},
          timeScale: {{timeVisible: true, secondsVisible: false}},
          rightPriceScale: {{borderColor: '#00CC00'}},
          crosshair: {{vertLine: {{color: '#00CC00'}}, horzLine: {{color: '#00CC00'}}}}
        }});
        const candleSeries = chart.addCandlestickSeries({{
          upColor: '#00CC00', downColor: '#FF3333', borderUpColor: '#00CC00', borderDownColor: '#FF3333',
          wickUpColor: '#00CC00', wickDownColor: '#FF3333'
        }});
        candleSeries.setData({json.dumps(ohlc_data)});

        if ({show_bollinger and bollinger_upper != "[]"}) {{
            const upperBand = chart.addLineSeries({{ color: 'rgba(255,165,0,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
            const middleBand = chart.addLineSeries({{ color: 'rgba(255,255,255,0.7)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false }});
            const lowerBand = chart.addLineSeries({{ color: 'rgba(255,165,0,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
            upperBand.setData({bollinger_upper});
            middleBand.setData({bollinger_middle});
            lowerBand.setData({bollinger_lower});
        }}
        chart.timeScale().fitContent();
      </script>
    </body></html>
    """
    components.html(html_code, height=540)


# ---------------------- Main UI ----------------------
NSE_HOLIDAYS_2025 = [
    "2025-01-26", "2025-03-14", "2025-03-31", "2025-04-14", "2025-04-18", 
    "2025-05-01", "2025-06-06", "2025-08-15", "2025-10-02", "2025-10-21", 
    "2025-11-05", "2025-12-25"
]

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
current_date = now.strftime("%Y-%m-%d")
is_weekday = now.weekday() < 5
is_market_hours = (now.time() >= datetime.strptime("09:15", "%H:%M").time() and now.time() <= datetime.strptime("15:30", "%H:%M").time())
is_holiday = current_date in NSE_HOLIDAYS_2025
market_status = "LIVE" if is_weekday and not is_holiday and is_market_hours else "Closed"
status_color = "#00CC00" if market_status == "LIVE" else "#FF3333"

st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{'#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0'},#1a1a1a 60%,#00CC00 100%); padding:10px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007;'>
        <span style='color:#FFFF00;font-family:"Courier New",monospace;font-size:2.1rem;font-weight:bold;letter-spacing:2px;'>BlockVista Terminal</span>
        <span style='float:right;color:{status_color};font-size:1.25rem;font-family:monospace;padding-top:16px;'>INDIA • INTRADAY • {market_status}</span>
    </div>
    """, unsafe_allow_html=True
)

with st.expander("NIFTY 50 Sentiment Meter", expanded=True):
    if vader_available:
        sentiment_score, sentiment_label = get_nifty50_sentiment()
        st.metric("NIFTY 50 Sentiment", sentiment_label, f"{sentiment_score:.2f}")

if len(stock_list):
    display_symbol = stock_list[0].upper()
    with st.expander(f"Latest News for {display_symbol}", expanded=False):
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

    price = get_live_price(display_symbol)
    if np.isnan(price):
        price = float(data["Close"].iloc[-1]) if not data.empty else np.nan

    metrics_row = st.columns([1.5,1,1,1,1,1])
    latest = data.iloc[-1]
    
    rsi, macd, macds, adx, atr, vwap, close_price = (try_scalar(latest.get(k, np.nan)) for k in ['RSI', 'MACD_12_26_9', 'MACDs_12_26_9', 'ADX', 'ATR', 'VWAP', 'Close'])

    with metrics_row[0]:
        delta = price - close_price if not np.isnan(price) and not np.isnan(close_price) else None
        st.metric("LTP", f"{round(price, 2) if not np.isnan(price) else '—'}", f"{round(delta, 2) if delta is not None else ''}")
    with metrics_row[1]:
        rsi_label = f"{round(rsi, 2)} {'(OB)' if rsi > 70 else '(OS)' if rsi < 30 else ''}" if not np.isnan(rsi) else '—'
        st.metric("RSI", rsi_label)
    with metrics_row[2]:
        macd_delta = macd - macds if not np.isnan(macd) and not np.isnan(macds) else None
        st.metric("MACD", f"{round(macd, 2) if not np.isnan(macd) else '—'}", f"{round(macd_delta, 2) if macd_delta is not None else ''}")
    with metrics_row[3]:
        adx_label = f"{round(adx, 2)} {'(Strong)' if adx > 25 else ''}" if not np.isnan(adx) else '—'
        st.metric("ADX", adx_label)
    with metrics_row[4]:
        st.metric("ATR", f"{round(atr, 2) if not np.isnan(atr) else '—'}")
    with metrics_row[5]:
        st.metric("VWAP", f"{round(vwap, 2) if not np.isnan(vwap) else '—'}")

    tabs = st.tabs(["Chart", "TA", "Signals", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands", value=True)
        render_lightweight_chart(data, chart_style, bands_show)
        
    with tabs[1]:
        st.dataframe(data[['RSI', 'MACD_12_26_9', 'MACDs_12_26_9', 'ADX', 'ATR', 'VWAP', 'SMA21', 'EMA9']].tail(10))
    
    with tabs[2]:
        st.json(get_signals(data))
    
    with tabs[3]:
        st.dataframe(data.tail(10))
