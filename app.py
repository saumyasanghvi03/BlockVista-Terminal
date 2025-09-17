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

import os
import requests
import streamlit as st

# --- Automatic instruments.csv downloader ---
def ensure_instruments_csv():
    fname = "instruments.csv"
    url = "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/instruments.csv"
    if not os.path.exists(fname):
        try:
            st.info("instruments.csv not found, attempting download from repo...")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(fname, "wb") as f:
                f.write(r.content)
            st.success("Successfully downloaded instruments.csv from repo.")
        except Exception as ex:
            st.error(f"Could not auto-download instruments.csv: {ex}")

ensure_instruments_csv()

# --- Zerodha WebSocket Live Data Setup ---
api_key = st.secrets.get("KITE_API_KEY") or st.secrets.get("ZERODHA_API_KEY")
access_token = st.secrets.get("KITE_ACCESS_TOKEN") or st.secrets.get("ZERODHA_ACCESS_TOKEN")
if not api_key or not access_token:
    st.error("Please set your Kite API Key and Access Token in Streamlit secrets!")
    st.stop()

kite = KiteConnect(api_key=api_key)
kws = KiteTicker(api_key, access_token)

tokens = []
if os.path.exists("instruments.csv"):
    df = pd.read_csv("instruments.csv")
    # Subscribe to all your watchlist tokens
    watchlist = ["RELIANCE", "SBIN", "TCS", "INFY"]  # Example - replace as needed
    for symbol in watchlist:
        row = df.query(f"tradingsymbol == '{symbol}' & exchange == 'NSE' & instrument_type == 'EQ'")
        if not row.empty:
            tokens.append(int(row["instrument_token"].iloc[0]))
else:
    tokens = [408065]  # Fallback: INFY

def on_ticks(ws, ticks):
    if "live_ticks" not in st.session_state:
        st.session_state["live_ticks"] = []
    st.session_state["live_ticks"].extend(ticks)

def on_connect(ws, response):
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_close(ws, code, reason):
    print("WebSocket closed", code, reason)

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

if "kws_connected" not in st.session_state:
    kws.connect(threaded=True)
    st.session_state["kws_connected"] = True


try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
except ModuleNotFoundError:
    vader_available = False
    pass

# ---------------------- CONFIG ----------------------
AV_API_KEY = "C1QB4N93H0VTOFS8"
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
        .stButton button {background: #004d00; color: #FFFFFF; border-radius: 4px;}
        .stButton button:hover {background: #006600;}
        .stTabs [data-baseweb="tab"] {color: #FFFF00;}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #004d00; color: #FFFFFF;}
        /* Style for the new containers */
        [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
            border: 1px solid #333333;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    else: # Bloomberg Light Theme
        st.markdown("""
        <style>
        .main, .block-container {background-color: #F0F0F0 !important;}
        .stSidebar {background: #E0E0E0 !important;}
        .stDataFrame tbody tr {background-color: #E0E0E0 !important; color: #000000 !important;}
        .stTextInput input, .stNumberInput input, .stSelectbox {background: #FFFFFF !important; color: #000000 !important;}
        .stMetric, .stMetricLabel {color: #000000 !important;}
        .stMetricValue {color: #008000 !important;}
        h1, h2, h3, h4, h5, h6, label {color: #000000 !important; font-family: 'Courier New', monospace;}
        .stButton button {background: #004d00; color: #FFFFFF; border-radius: 4px;}
        .stButton button:hover {background: #006600;}
        .stTabs [data-baseweb="tab"] {color: #000000;}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #004d00; color: #FFFFFF;}
        /* Style for the new containers */
        [data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

# ---------------------- Login/Signup/Reset UI (PRESERVED) ----------------------
if not st.session_state["logged_in"]:
    set_terminal_style() # Set theme for login page as well
    st.title("BlockVista Terminal")
    st.subheader("Account Access")
    
    auth_container = st.container()
    with auth_container:
        auth_mode = st.selectbox("Select Action", ["Login", "Signup", "Reset Password"], label_visibility="collapsed")
        
        if auth_mode == "Signup":
            st.write("Create a new BlockVista Terminal account")
            with st.form("signup_form"):
                signup_username = st.text_input("New Username")
                signup_access_code = st.text_input("New Access Code", type="password")
                security_question = st.selectbox("Security Question", SECURITY_QUESTIONS)
                security_answer = st.text_input("Security Answer", type="password")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    success, message = signup_user(signup_username, signup_access_code, security_question, security_answer)
                    if success:
                        st.success(message)
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = signup_username
                        st.session_state["login_attempts"] = 0
                        st.session_state["reset_attempts"] = 0
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
        
        elif auth_mode == "Reset Password":
            st.write("Reset your password")
            with st.form("reset_form"):
                reset_username = st.text_input("Username")
                users = load_users()
                if reset_username in users:
                    st.write(f"Security Question: {users[reset_username]['security_question']}")
                reset_security_answer = st.text_input("Security Answer", type="password")
                new_access_code = st.text_input("New Access Code", type="password")
                submitted = st.form_submit_button("Reset Password")
                if submitted:
                    if st.session_state["reset_attempts"] >= MAX_RESET_ATTEMPTS:
                        st.error("❌ Too many reset attempts. Please try again later.")
                    else:
                        success, message = reset_user_password(reset_username, reset_security_answer, new_access_code)
                        if success:
                            st.session_state["reset_attempts"] = 0
                            st.success(message)
                            st.info("Please log in with your new access code.")
                        else:
                            st.session_state["reset_attempts"] += 1
                            remaining = MAX_RESET_ATTEMPTS - st.session_state["reset_attempts"]
                            st.error(f"❌ {message}. {remaining} attempts remaining.")
        
        else: # Login
            st.write("Login to your BlockVista Terminal account")
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_access_code = st.text_input("Access Code", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if st.session_state["login_attempts"] >= MAX_LOGIN_ATTEMPTS:
                        st.error("❌ Too many failed attempts. Please try again later.")
                    else:
                        success, message = login_user(login_username, login_access_code)
                        if success:
                            st.session_state["logged_in"] = True
                            st.session_state["username"] = login_username
                            st.session_state["login_attempts"] = 0
                            st.session_state["reset_attempts"] = 0
                            st.success(f"✅ Welcome, {login_username}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.session_state["login_attempts"] += 1
                            remaining = MAX_LOGIN_ATTEMPTS - st.session_state["login_attempts"]
                            st.error(f"❌ {message}. {remaining} attempts remaining.")
    
    st.stop()

# --- MAIN APP EXECUTION STARTS HERE ---

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
            instrument_token_map = get_instrument_tokens()
            if not instrument_token_map:
                st.warning("Could not load instrument tokens. Live data feed will not start.")
                return
            st.session_state["instrument_token_map"] = instrument_token_map
            st.session_state["token_to_symbol_map"] = {v: k for k, v in instrument_token_map.items()}
            
            watchlist_symbols = st.session_state.get("watchlist_symbols", NIFTY50_TOP)
            important_symbols = set(watchlist_symbols)
            tokens_to_track = [instrument_token_map[s] for s in important_symbols if s in instrument_token_map]
            
            if tokens_to_track:
                start_live_ticker(api_key, st.session_state["access_token"], tokens_to_track)
                st.session_state['ticker_started'] = True

    if "access_token" in st.session_state and st.session_state.get("zerodha_authenticated"):
        try:
            kite_conn.set_access_token(st.session_state["access_token"])
            kite_conn.profile()
            initialize_ticker_and_tokens()
            return kite_conn
        except Exception as e:
            st.session_state.pop("access_token", None); st.session_state.pop("zerodha_authenticated", None)
            st.warning(f"Zerodha session invalid, please re-authenticate. Error: {e}")
    
    return kite_conn

kite = get_kite_session()
set_terminal_style()

# ---------------------- Sidebar Setup ----------------------
with st.sidebar:
    st.title(f"Welcome, {st.session_state.get('username', 'Guest')}")
    theme_choice = st.selectbox("Terminal Theme", ["Bloomberg Dark", "Bloomberg Light"])
    st.session_state.theme = theme_choice
    
    screener_mode = st.radio("Screener Mode", ["Single Stock", "Basket"])
    if screener_mode == "Single Stock":
        symbol = st.text_input('NSE Symbol', value='RELIANCE')
        stock_list = [symbol.strip().upper()]
    else:
        basket = st.selectbox("Pick Basket", list(SMALLCASE_BASKETS.keys()))
        stock_list = SMALLCASE_BASKETS[basket]
    
    screen_period = st.selectbox('Period', ['1d','5d'])
    screen_interval = st.selectbox('Interval', ['1m','5m','15m'])

    if not st.session_state.get("zerodha_authenticated"):
        with st.expander("Zerodha Login", expanded=True):
            st.markdown(f'<a href="{kite.login_url()}" target="_blank"><b>Click here to Login to Zerodha</b></a>', unsafe_allow_html=True)
            request_token = st.text_input("Paste request_token here:")
            if st.button("Generate Session"):
                try:
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    st.session_state["access_token"] = data["access_token"]
                    st.session_state["zerodha_authenticated"] = True
                    st.success("✅ Zerodha session started!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Login failed: {e}")

    if st.session_state.get("zerodha_authenticated"):
        if st.button("Logout from Zerodha"):
            st.session_state.pop("access_token", None); st.session_state.pop("zerodha_authenticated", None); st.session_state.pop("ticker_started", None)
            st.success("Logged out from Zerodha.")
            time.sleep(1); st.rerun()
            
    st.divider()
    st.subheader("Watchlist & P/L Controls")
    watchlist_input = st.text_area("Watchlist (comma-separated)", value="RELIANCE, SBIN, TCS")
    positions_input = st.text_area("Entry Prices (comma, same order)", value="2900, 850, 4000")
    qty_input = st.text_area("Quantities (comma, same order)", value="10, 20, 5")
    st.session_state['watchlist_symbols'] = [x.strip().upper() for x in watchlist_input.split(",") if x.strip()]

# ---------------------- News Parser & Core Helpers ----------------------
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
    return sorted(news_items, key=lambda x: x["published_dt"], reverse=True)[:5]

def get_live_price(symbol):
    if symbol in LIVE_QUOTES: return LIVE_QUOTES[symbol]
    try:
        if st.session_state.get("zerodha_authenticated"):
            ltp = kite.ltp(f"NSE:{symbol}")
            price = ltp[f"NSE:{symbol}"]["last_price"]
            LIVE_QUOTES[symbol] = price
            return price
    except Exception:
        return np.nan
    return np.nan

def process_indicators(data, symbol):
    try:
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            return data
        
        if len(data) > 14:
            data['RSI'] = ta.rsi(data['Close'], length=14)
            data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
            adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
            data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) else np.nan
        else:
            data['RSI'], data['ATR'], data['ADX'] = np.nan, np.nan, np.nan
        
        if len(data) > 26:
            macd = ta.macd(data['Close'])
            if isinstance(macd, pd.DataFrame):
                data = data.join(macd)
        
        data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 21 else np.nan
        data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 9 else np.nan

        if len(data) > 20:
            bbands = ta.bbands(data['Close'], length=20)
            if isinstance(bbands, pd.DataFrame):
                data = data.join(bbands)
        
        if 'Volume' in data.columns:
            data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
        if isinstance(ha, pd.DataFrame):
            data = data.join(ha)
            
        return data
    except Exception as e:
        logging.error(f"Error computing indicators for {symbol}: {e}")
        return data

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    if st.session_state.get("zerodha_authenticated"):
        to_dt = datetime.now()
        days_map = {'1d': 2, '5d': 7} # Fetch slightly more to ensure data
        from_dt = to_dt - timedelta(days=days_map.get(period, 30))
        interval_map = {"1m": "minute", "5m": "5minute", "15m": "15minute"}
        kite_interval = interval_map.get(interval)
        if kite_interval:
            data = fetch_kite_historical(symbol, kite_interval, from_dt, to_dt, kite)
            if data is not None and not data.empty:
                return process_indicators(data, symbol)
    try:
        data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            data = fetch_alpha_vantage_intraday(symbol, interval=f"{interval[:-1]}min")
        return process_indicators(data.copy(), symbol) if data is not None and not data.empty else None
    except Exception as e:
        logging.error(f"Error in fallback fetch for {symbol}: {e}")
        return None

def get_signals(data):
    if data is None or data.empty: return {}
    latest = data.iloc[-1]
    signals = {}
    rsi = try_scalar(latest.get('RSI_14', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    close = try_scalar(latest.get('Close', np.nan))
    signals['RSI Signal'] = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if macd > macds else ('Bearish' if macd < macds else 'Neutral')
    # Simplified Supertrend logic for example
    if 'SUPERTd_7_3.0' in latest and latest['SUPERTd_7_3.0'] == 1:
        signals['Trend'] = 'Bullish'
    elif 'SUPERTd_7_3.0' in latest and latest['SUPERTd_7_3.0'] == -1:
        signals['Trend'] = 'Bearish'
    else:
        signals['Trend'] = 'Neutral'
    return signals

def try_scalar(val):
    if isinstance(val, pd.Series) and len(val) >= 1: val = val.iloc[-1]
    if isinstance(val, (float, int, np.floating, np.integer)): return val
    try: return float(val)
    except: return np.nan

# ---------------------- Lightweight Charts Renderer ----------------------
def render_lightweight_chart(df, chart_style="Candlestick", show_bollinger=True):
    if df is None or df.empty: return
    df_chart = df.reset_index()
    df_chart = df_chart.rename(columns={'index': 'time', 'date': 'time'})
    ohlc_cols = ['HA_open', 'HA_high', 'HA_low', 'HA_close'] if chart_style == "Heikin Ashi" and 'HA_open' in df.columns else ['Open', 'High', 'Low', 'Close']
    ohlc_data = [{"time": int(t.timestamp()), "open": o, "high": h, "low": l, "close": c} for t, o, h, l, c in zip(pd.to_datetime(df_chart['time']), df_chart[ohlc_cols[0]], df_chart[ohlc_cols[1]], df_chart[ohlc_cols[2]], df_chart[ohlc_cols[3]])]
    bollinger_upper, bollinger_middle, bollinger_lower = "[]", "[]", "[]"
    if show_bollinger and 'BBU_20_2.0' in df.columns:
        bollinger_upper = json.dumps([{"time": int(t.timestamp()), "value": v} for t, v in zip(pd.to_datetime(df_chart['time']), df_chart['BBU_20_2.0']) if pd.notna(v)])
        bollinger_middle = json.dumps([{"time": int(t.timestamp()), "value": v} for t, v in zip(pd.to_datetime(df_chart['time']), df_chart['BBM_20_2.0']) if pd.notna(v)])
        bollinger_lower = json.dumps([{"time": int(t.timestamp()), "value": v} for t, v in zip(pd.to_datetime(df_chart['time']), df_chart['BBL_20_2.0']) if pd.notna(v)])
    html_code = f"""
    <!doctype html><html><head><meta charset="utf-8"></head><body>
      <div id="chart" style="width:100%;height:400px;"></div>
      <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
      <script>
        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
          layout: {{background: {{type: 'solid', color: '{'#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0'}'}}, textColor: '{'#FFFF00' if st.session_state.theme == 'Bloomberg Dark' else '#000000'}'}},
          grid: {{vertLines: {{color: '#333333'}}, horzLines: {{color: '#333333'}}}}, timeScale: {{timeVisible: true, secondsVisible: false}},
        }});
        const candleSeries = chart.addCandlestickSeries({{ upColor: '#00CC00', downColor: '#FF3333', borderVisible: false, wickUpColor: '#00CC00', wickDownColor: '#FF3333' }});
        candleSeries.setData({json.dumps(ohlc_data)});
        if ({show_bollinger and bollinger_upper != "[]"}) {{
            const upperBand = chart.addLineSeries({{ color: 'rgba(255,165,0,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }});
            const middleBand = chart.addLineSeries({{ color: 'rgba(255,255,255,0.7)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }});
            const lowerBand = chart.addLineSeries({{ color: 'rgba(255,165,0,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }});
            upperBand.setData({bollinger_upper}); middleBand.setData({bollinger_middle}); lowerBand.setData({bollinger_lower});
        }}
        chart.timeScale().fitContent();
      </script>
    </body></html>"""
    components.html(html_code, height=420)

# ---------------------- HEADER ----------------------
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
market_status = "LIVE" if (now.weekday() < 5 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()) else "Closed"
status_color = "#00CC00" if market_status == "LIVE" else "#FF3333"

st.markdown(f"""
    <div style='background:linear-gradient(90deg,{'#000000' if st.session_state.theme == 'Bloomberg Dark' else '#F0F0F0'},#1a1a1a 60%,#00CC00 100%); padding:10px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007;'>
        <span style='color:#FFFF00;font-family:"Courier New",monospace;font-size:2.1rem;font-weight:bold;letter-spacing:2px;'>BlockVista Terminal</span>
        <span style='float:right;color:{status_color};font-size:1.25rem;font-family:monospace;padding-top:16px;'>INDIA • {now.strftime("%d %b %Y, %H:%M")} • {market_status}</span>
    </div>""", unsafe_allow_html=True)

# ---------------------- Main 6-Panel UI ----------------------
if not stock_list or not stock_list[0]:
    st.warning("Please enter a stock symbol in the sidebar to begin.")
    st.stop()

display_symbol = stock_list[0]
data = fetch_stock_data(display_symbol, screen_period, screen_interval)

if data is None or data.empty:
    st.error(f"No data available for {display_symbol}. Please check the symbol or try a different period/interval.")
    st.stop()

col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        price = get_live_price(display_symbol)
        if np.isnan(price): price = data["Close"].iloc[-1]
        delta = price - data["Close"].iloc[-2] if len(data) > 1 else 0
        c1, c2 = st.columns([3, 2])
        c1.subheader(f"📊 {display_symbol} Chart")
        c2.metric("LTP", f"{price:.2f}", f"{delta:.2f}")
        chart_style = st.radio("Style", ["Candlestick", "Heikin Ashi"], horizontal=True, key="chart_style")
        bands_show = st.checkbox("Bollinger Bands", value=True, key="bands_show")
        render_lightweight_chart(data, chart_style, bands_show)

    with st.container(border=True):
        st.subheader("📈 Technical Analysis")
        st.dataframe(data.filter(regex='RSI|MACD|ADX|ATR|VWAP|SMA|EMA').tail(5), height=215, use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader("🛒 Place Order")
        if st.session_state.get("zerodha_authenticated"):
            with st.form("order_form"):
                c1, c2 = st.columns(2)
                transaction_type = c1.selectbox("Txn", ["BUY", "SELL"])
                order_product = c2.selectbox("Product", ["CNC", "MIS"])
                c1, c2 = st.columns(2)
                order_type = c1.selectbox("Type", ["MARKET", "LIMIT"])
                order_qty = c2.number_input("Qty", min_value=1, value=1)
                order_price = st.number_input("Price", value=round(price, 1), step=0.05, disabled=(order_type != "LIMIT"))
                if st.form_submit_button("Submit Order"):
                    try:
                        # Full order placement logic here
                        st.success("Order Submitted (Simulated)")
                    except Exception as e:
                        st.error(f"Order failed: {e}")
        else:
            st.warning("Please login to Zerodha via the sidebar to place orders.")

    with st.container(border=True):
        st.subheader("💰 Watchlist P/L")
        try:
            entry_prices = [float(x) for x in positions_input.split(",") if x.strip()]
            quantities = [float(x) for x in qty_input.split(",") if x.strip()]
            pnl_data = []
            if len(st.session_state['watchlist_symbols']) == len(entry_prices) == len(quantities):
                for i, s in enumerate(st.session_state['watchlist_symbols']):
                    live = get_live_price(s); pnl = (live - entry_prices[i]) * quantities[i] if not np.isnan(live) else "N/A"
                    pnl_data.append({"Symbol": s, "LTP": live, "P/L ₹": pnl})
                st.dataframe(pd.DataFrame(pnl_data), height=180, use_container_width=True)
            else:
                st.info("Enter matching watchlist symbols, prices, and quantities in the sidebar.")
        except:
            st.warning("Invalid P/L inputs in the sidebar.")

    with st.container(border=True):
        st.subheader(f"📰 News & Signals for {display_symbol}")
        news_items = get_news(display_symbol)
        if news_items:
            for item in news_items[:2]: st.markdown(f"• [{item['title']}]({item['link']})")
        st.divider()
        signals = get_signals(data)
        if signals:
            c1, c2, c3 = st.columns(3)
            c1.metric("RSI", signals.get('RSI Signal', 'N/A'))
            c2.metric("MACD", signals.get('MACD Signal', 'N/A'))
            c3.metric("Trend", signals.get('Trend', 'N/A'))
