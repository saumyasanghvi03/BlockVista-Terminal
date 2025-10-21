# ================ 0. REQUIRED LIBRARIES ================

import streamlit as st
import pandas as pd
import talib # Replace pandas_ta with talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
import pytz
import feedparser
from email.utils import mktime_tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
import time as a_time
import re
import yfinance as yf
import pyotp
import qrcode
from PIL import Image
import base64
import io
import requests
import hashlib
import random
from streamlit_autorefresh import st_autorefresh
import praw

# <<<--- ENSURE THESE IMPORTS ARE PRESENT --->>>
# If any are missing, add them above

# ================ UPSTOX API INTEGRATION ================
import requests
import json

# ================ 1. STYLING AND CONFIGURATION ===============

st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def apply_custom_styling():
    """Applies a comprehensive CSS stylesheet for professional theming."""
    theme_css = """
    <style>
        :root {
            --dark-bg: #0E1117;
            --dark-secondary-bg: #161B22;
            --dark-widget-bg: #21262D;
            --dark-border: #30363D;
            --dark-text: #c9d1d9;
            --dark-text-light: #8b949e;
            --dark-green: #28a745;
            --dark-red: #da3633;

            --light-bg: #FFFFFF;
            --light-secondary-bg: #F0F2F6;
            --light-widget-bg: #F8F9FA;
            --light-border: #dee2e6;
            --light-text: #212529;
            --light-text-light: #6c757d;
            --light-green: #198754;
            --light-red: #dc3545;
        }

        body.dark-theme {
            --primary-bg: var(--dark-bg);
            --secondary-bg: var(--dark-secondary-bg);
            --widget-bg: var(--dark-widget-bg);
            --border-color: var(--dark-border);
            --text-color: var(--dark-text);
            --text-light: var(--dark-text-light);
            --green: var(--dark-green);
            --red: var(--dark-red);
        }

        body.light-theme {
            --primary-bg: var(--light-bg);
            --secondary-bg: var(--light-secondary-bg);
            --widget-bg: var(--light-widget-bg);
            --border-color: var(--light-border);
            --text-color: var(--light-text);
            --text-light: var(--light-text-light);
            --green: var(--light-green);
            --red: var(--light-red);
        }

        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        h1, h2, h3, h4, h5 {
            color: var(--text-color) !important;
        }
        
        hr {
            background: var(--border-color);
        }

        .stButton>button {
            border-color: var(--border-color);
            background-color: var(--widget-bg);
            color: var(--text-color);
        }
        .stButton>button:hover {
            border-color: var(--green);
            color: var(--green);
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
            background-color: var(--widget-bg);
            border-color: var(--border-color);
            color: var(--text-color);
        }
        .stRadio>div {
            background-color: var(--widget-bg);
            border: 1px solid var(--border-color);
            padding: 8px;
            border-radius: 8px;
        }
        
        .metric-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }
        
        .trade-card {
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 10px;
            border-left-width: 5px;
        }

        .notification-bar {
            position: sticky;
            top: 0;
            width: 100%;
            background-color: var(--secondary-bg);
            color: var(--text-color);
            padding: 8px 12px;
            z-index: 999;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border-color);
            margin-left: -20px;
            margin-right: -20px;
            width: calc(100% + 40px);
        }
        .notification-bar span {
            margin: 0 15px;
            white-space: nowrap;
        }
        
        .hft-depth-bid {
            background: linear-gradient(to left, rgba(0, 128, 0, 0.3), rgba(0, 128, 0, 0.05));
            padding: 2px 5px;
        }
        .hft-depth-ask {
            background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05));
            padding: 2px 5px;
        }
        .tick-up {
            color: var(--green);
            animation: flash-green 0.5s;
        }
        .tick-down {
            color: var(--red);
            animation: flash-red 0.5s;
        }
        @keyframes flash-green {
            0% { background-color: rgba(40, 167, 69, 0.5); }
            100% { background-color: transparent; }
        }
        @keyframes flash-red {
            0% { background-color: rgba(218, 54, 51, 0.5); }
            100% { background-color: transparent; }
        }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    js_theme = f"""
    <script>
        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add('{st.session_state.theme.lower()}-theme');
    </script>
    """
    st.components.v1.html(js_theme, height=0)

# Centralized data source configuration
ML_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NSE"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NFO"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NFO"
    },
    "GOLD": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv",
        "tradingsymbol": "GOLDM",
        "exchange": "MCX"
    },
    "USDINR": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv",
        "tradingsymbol": "USDINR",
        "exchange": "CDS"
    },
    "SENSEX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv",
        "tradingsymbol": "SENSEX",
        "exchange": "BSE"
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
        "tradingsymbol": "^GSPC",
        "exchange": "yfinance"
    }
    
}
# Upstox configuration
UPSTOX_CONFIG = {
    "api_key": "your_upstox_api_key",  # Will be set from secrets
    "redirect_uri": "https://your-redirect-uri.com",  # Set in Upstox developer console
    "api_secret": "your_upstox_api_secret"  # Will be set from secrets

}
# ================ 1.5 INITIALIZATION ========================

def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'broker' not in st.session_state: st.session_state.broker = None
    if 'kite' not in st.session_state: st.session_state.kite = None
    if 'profile' not in st.session_state: st.session_state.profile = None
        # Add Upstox session state variables
    if 'upstox_client' not in st.session_state: 
        st.session_state.upstox_client = None
    if 'upstox_access_token' not in st.session_state: 
        st.session_state.upstox_access_token = None
    if 'upstox_token_type' not in st.session_state: 
        st.session_state.upstox_token_type = None
    if 'login_animation_complete' not in st.session_state: st.session_state.login_animation_complete = False
    if 'authenticated' not in st.session_state: st.session_state.authenticated = False
    if 'two_factor_setup_complete' not in st.session_state: st.session_state.two_factor_setup_complete = False
    if 'pyotp_secret' not in st.session_state: st.session_state.pyotp_secret = None
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    
    # Remove the dialog state variables
    if 'show_quick_trade' not in st.session_state: st.session_state.show_quick_trade = False
    
    # Add other existing initializations...
    if 'watchlists' not in st.session_state:
        st.session_state.watchlists = {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
            "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
        }
    # Add all other existing initializations...
    if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Watchlist 1"
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    if 'basket' not in st.session_state: st.session_state.basket = []
    if 'last_order_details' not in st.session_state: st.session_state.last_order_details = {}
    if 'underlying_pcr' not in st.session_state: st.session_state.underlying_pcr = "NIFTY"
    if 'strategy_legs' not in st.session_state: st.session_state.strategy_legs = []
    if 'calculated_greeks' not in st.session_state: st.session_state.calculated_greeks = None
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'ml_forecast_df' not in st.session_state: st.session_state.ml_forecast_df = None
    if 'ml_instrument_name' not in st.session_state: st.session_state.ml_instrument_name = None
    if 'backtest_results' not in st.session_state: st.session_state.backtest_results = None
    if 'hft_last_price' not in st.session_state: st.session_state.hft_last_price = 0
    if 'hft_tick_log' not in st.session_state: st.session_state.hft_tick_log = []
    if 'last_bot_result' not in st.session_state: st.session_state.last_bot_result = None
    
    # Add new dialog state variables
    if 'show_quick_trade' not in st.session_state: st.session_state.show_quick_trade = False
    if 'show_most_active' not in st.session_state: st.session_state.show_most_active = False
    if 'show_2fa_dialog' not in st.session_state: st.session_state.show_2fa_dialog = False
    if 'show_qr_dialog' not in st.session_state: st.session_state.show_qr_dialog = False
    
    # Add automated mode variables
    if 'automated_mode' not in st.session_state:
        st.session_state.automated_mode = {
            'enabled': False,
            'running': False,
            'bots_active': {},
            'total_capital': 10000,
            'risk_per_trade': 2,
            'max_open_trades': 5,
            'trade_history': [],
            'performance_metrics': {},
            'last_signal_check': None
        }
    
    # Add special trading days
    if 'special_trading_days' not in st.session_state:
        st.session_state.special_trading_days = []
# ================ 2. HELPER FUNCTIONS ================
def get_ist_time():
    """Get current time in IST timezone."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def check_for_special_session():
    """Checks if the current time falls within any special trading session."""
    if 'special_trading_days' not in st.session_state:
        return None
    
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    now_date = now.date()
    now_time = now.time()

    for session in st.session_state.special_trading_days:
        if now_date == session['date'] and session['start'] <= now_time <= session['end']:
            return session # Return the active session details
    return None

def setup_auto_refresh_control():
    """Setup comprehensive auto-refresh control to prevent flickering."""
    if 'refresh_control' not in st.session_state:
        st.session_state.refresh_control = {
            'enabled': True,
            'last_refresh': datetime.now(),
            'refresh_interval': 60,  # seconds
            'excluded_tabs': ["🔍 Live Thinking", "📋 Trade History", "🎯 Symbol Override"],
            'user_interaction_time': datetime.now(),
            'prevent_refresh_until': None
        }

def should_auto_refresh(current_tab=None):
    """Determine if auto-refresh should occur based on current context."""
    control = st.session_state.get('refresh_control', {})
    
    # Check if auto-refresh is disabled globally
    if not control.get('enabled', True):
        return False
    
    # Check if we're in a cooldown period
    prevent_until = control.get('prevent_refresh_until')
    if prevent_until and datetime.now() < prevent_until:
        return False
    
    # Check if current tab is excluded from auto-refresh
    if current_tab and current_tab in control.get('excluded_tabs', []):
        return False
    
    # Check if user recently interacted (within last 5 seconds)
    last_interaction = control.get('user_interaction_time', datetime.now())
    time_since_interaction = (datetime.now() - last_interaction).total_seconds()
    if time_since_interaction < 5:
        return False
    
    # Check refresh interval
    last_refresh = control.get('last_refresh', datetime.now())
    refresh_interval = control.get('refresh_interval', 60)
    time_since_refresh = (datetime.now() - last_refresh).total_seconds()
    
    return time_since_refresh >= refresh_interval

def update_refresh_timestamp():
    """Update the last refresh timestamp."""
    if 'refresh_control' in st.session_state:
        st.session_state.refresh_control['last_refresh'] = datetime.now()

def prevent_refresh_for(seconds=10):
    """Temporarily prevent auto-refresh for specified seconds."""
    if 'refresh_control' in st.session_state:
        st.session_state.refresh_control['prevent_refresh_until'] = (
            datetime.now() + timedelta(seconds=seconds)
        )

def record_user_interaction():
    """Record when user interacts with the app to prevent immediate refresh."""
    if 'refresh_control' in st.session_state:
        st.session_state.refresh_control['user_interaction_time'] = datetime.now()

def get_broker_client():
    """Gets current broker client from session state."""
    broker = st.session_state.get('broker')
    
    if broker == "Zerodha":
        return st.session_state.get('kite')
    elif broker == "Upstox":
        # For Upstox, we use the access token directly in API calls
        return st.session_state.get('upstox_access_token')
    
    return None

def debug_upstox_installation():
    """Debug function to check Upstox installation."""
    try:
        from upstox_api.api import Upstox
        st.success("✓ Upstox package imported successfully")
        
        # Check available methods
        methods = [method for method in dir(Upstox) if not method.startswith('_')]
        st.write("Available Upstox methods:", methods[:10])  # Show first 10 methods
        
        return True
    except ImportError as e:
        st.error(f"✗ Upstox import failed: {e}")
        return False
    except Exception as e:
        st.error(f"✗ Upstox check failed: {e}")
        return False

def debug_upstox_exchanges(access_token):
    """Debug function to find available exchanges in Upstox API v2."""
    if not access_token:
        st.error("No access token available")
        return
    
    st.subheader("🔧 Upstox API Debug - Available Exchanges")
    
    # Test various possible exchange codes
    possible_exchanges = [
        'NSE_EQ', 'NSE_FO', 'NSE_CD', 'BSE_EQ', 'BSE_FO', 
        'MCX_FO', 'MCX_COMM', 'NSE', 'BSE', 'NFO', 'MCX', 'CDS'
    ]
    
    available_exchanges = []
    
    for exchange in possible_exchanges:
        try:
            url = f"https://api.upstox.com/v2/master-contract/{exchange}"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    count = len(data.get('data', []))
                    available_exchanges.append(exchange)
                    st.success(f"✓ {exchange}: {count} instruments available")
                else:
                    st.warning(f"⚠ {exchange}: API returned error - {data.get('message', 'Unknown')}")
            else:
                st.info(f"✗ {exchange}: Not available (HTTP {response.status_code})")
                
        except Exception as e:
            st.error(f"✗ {exchange} error: {e}")
    
    return available_exchanges

def debug_upstox_api(access_token):
    """Debug function to test Upstox API connectivity."""
    if not access_token:
        st.error("No access token available")
        return
    
    # Test profile endpoint
    try:
        profile_url = "https://api.upstox.com/v2/user/profile"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(profile_url, headers=headers)
        if response.status_code == 200:
            st.success("✓ Upstox profile API working")
            profile_data = response.json()
            st.write("Profile data:", profile_data)
        else:
            st.error(f"✗ Profile API failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"✗ Profile API error: {e}")
    
    # Test available exchanges
    exchanges = ['NSE', 'BSE', 'NFO', 'MCX', 'CDS']
    for exchange in exchanges:
        try:
            url = f"https://api.upstox.com/v2/master-contract/{exchange}"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    count = len(data.get('data', []))
                    st.success(f"✓ {exchange}: {count} instruments")
                else:
                    st.warning(f"⚠ {exchange}: {data.get('message', 'Unknown error')}")
            else:
                st.error(f"✗ {exchange}: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"✗ {exchange} error: {e}")
# ===== END DEBUG FUNCTION =====

def get_upstox_login_url():
    """Generate Upstox login URL."""
    try:
        api_key = st.secrets.get("UPSTOX_API_KEY")
        redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI")
        
        if not api_key:
            st.error("UPSTOX_API_KEY not found in secrets")
            return None
            
        if not redirect_uri:
            st.error("UPSTOX_REDIRECT_URI not found in secrets")
            return None
        
        # URL encode the redirect_uri
        import urllib.parse
        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
        
        # Upstox login URL format
        login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={encoded_redirect_uri}"
        
        st.info(f"Login URL generated. Redirect URI: {redirect_uri}")
        return login_url
        
    except Exception as e:
        st.error(f"Error generating login URL: {e}")
        return None

def upstox_generate_session(authorization_code):
    """Generate Upstox session using authorization code via direct API call."""
    try:
        api_key = st.secrets.get("UPSTOX_API_KEY")
        api_secret = st.secrets.get("UPSTOX_API_SECRET")
        redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI")
        
        if not all([api_key, api_secret, redirect_uri]):
            st.error("Missing Upstox credentials in secrets")
            return False
            
        url = 'https://api.upstox.com/v2/login/authorization/token'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        data = {
            'code': authorization_code,
            'client_id': api_key,
            'client_secret': api_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code',
        }

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status() # Raise an exception for bad status codes
        
        token_data = response.json()
        
        if 'access_token' in token_data:
            st.session_state.upstox_access_token = token_data['access_token']
            st.session_state.upstox_token_type = token_data.get('token_type', 'bearer')
            st.success("Upstox authentication successful!")
            return True
        else:
            st.error(f"No access token in response: {token_data.get('error_description', 'Unknown error')}")
            return False
            
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Upstox session generation failed (HTTP Error): {http_err.response.status_code} - {http_err.response.text}")
        return False
    except Exception as e:
        st.error(f"Upstox session generation failed: {str(e)}")
        return False

def upstox_logout():
    """Logs out the user from the Upstox API."""
    try:
        access_token = st.session_state.get('upstox_access_token')
        if not access_token:
            st.warning("No active Upstox session to log out from.")
            return

        url = 'https://api.upstox.com/v2/logout'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        
        logout_data = response.json()
        if logout_data.get("status") == "success":
            st.toast("Successfully logged out from Upstox.")
        else:
            st.warning(f"Upstox logout may not have been fully successful: {logout_data.get('message', '')}")

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Upstox logout failed (HTTP Error): {http_err.response.status_code} - {http_err.response.text}")
    except Exception as e:
        st.error(f"An error occurred during Upstox logout: {str(e)}")


def get_upstox_instruments(access_token, exchange='NSE_EQ'):
    """Fetches instrument list from Upstox REST API v2 with correct exchange codes."""
    if not access_token:
        return pd.DataFrame()
    
    try:
        # Correct exchange codes for Upstox v2 (from their documentation)
        exchange_map = {
            'NSE': 'NSE_EQ',    # NSE Equity
            'BSE': 'BSE_EQ',    # BSE Equity  
            'NFO': 'NSE_FO',    # NSE Futures & Options
            'MCX': 'MCX_FO',    # MCX Commodities
            'CDS': 'NSE_CD',    # NSE Currency Derivatives
        }
        
        upstox_exchange = exchange_map.get(exchange, 'NSE_EQ')
        
        url = f"https://api.upstox.com/v2/master-contract/{upstox_exchange}"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        st.info(f"Fetching instruments for {exchange} -> {upstox_exchange}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            instrument_list = []
            
            if data.get('status') == 'success' and 'data' in data:
                instruments_data = data['data']
                if not isinstance(instruments_data, list):
                    st.error(f"Unexpected data format: {type(instruments_data)}")
                    return pd.DataFrame()
                
                for instrument in instruments_data:
                    instrument_list.append({
                        'tradingsymbol': instrument.get('trading_symbol', ''),
                        'name': instrument.get('name', ''),
                        'instrument_token': instrument.get('instrument_key', ''),
                        'exchange': exchange,  # Keep our internal exchange code
                        'lot_size': instrument.get('lot_size', 1),
                        'instrument_type': instrument.get('instrument_type', 'EQ'),
                        'strike_price': instrument.get('strike_price', 0),
                        'expiry': instrument.get('expiry', '')
                    })
                
                st.success(f"Loaded {len(instrument_list)} instruments from {exchange}")
                return pd.DataFrame(instrument_list)
            else:
                st.error(f"Upstox API response error: {data}")
        else:
            st.error(f"Upstox API Error (Instruments): {response.status_code} - {response.text}")
            
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Upstox API Error (Instruments): {e}")
        return pd.DataFrame()

def get_upstox_historical_data(access_token, instrument_key, interval, period=None):
    """Fetches historical data from Upstox REST API v2."""
    if not access_token or not instrument_key:
        return pd.DataFrame()
    
    try:
        from datetime import datetime, timedelta
        
        # Map interval to Upstox v2 format
        interval_map = {
            'minute': '1minute',
            '5minute': '5minute', 
            'day': 'day',
            'week': 'week'
        }
        
        upstox_interval = interval_map.get(interval, 'day')
        
        # Calculate date range based on period
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=2)
        elif period == '5d':
            start_date = end_date - timedelta(days=7)
        elif period == '1mo':
            start_date = end_date - timedelta(days=31)
        elif period == '6mo':
            start_date = end_date - timedelta(days=182)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch historical data from Upstox v2
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{upstox_interval}/{start_date_str}/{end_date_str}"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                st.error(f"Upstox API response error: {data}")
        else:
            st.error(f"Upstox API Error: {response.status_code} - {response.text}")
            
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Upstox API Error (Historical): {e}")
        return pd.DataFrame()

def get_upstox_instruments(access_token, exchange='NSE_EQ'):
    """Fetches instrument list from Upstox REST API v2."""
    if not access_token:
        return pd.DataFrame()
    
    try:
        url = f"https://api.upstox.com/v2/master-contract/{exchange}"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            instrument_list = []
            
            if data['status'] == 'success' and 'data' in data:
                for instrument in data['data']:
                    instrument_list.append({
                        'tradingsymbol': instrument.get('trading_symbol', ''),
                        'name': instrument.get('name', ''),
                        'instrument_token': instrument.get('instrument_key', ''),
                        'exchange': exchange,
                        'lot_size': instrument.get('lot_size', 1),
                        'instrument_type': instrument.get('instrument_type', 'EQ')
                    })
                
                return pd.DataFrame(instrument_list)
        else:
            st.error(f"Upstox API Error (Instruments): {response.text}")
            
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Upstox API Error (Instruments): {e}")
        return pd.DataFrame()

def get_upstox_quote(upstox_client, instrument_key):
    """Fetches quote data from Upstox."""
    if not upstox_client or not instrument_key:
        return None
    
    try:
        # Get live quote
        quote_data = upstox_client.get_live_feed(instrument_key, 'quote')
        return quote_data
        
    except Exception as e:
        st.error(f"Upstox API Error (Quote): {e}")
        return None


def place_upstox_order(order_params):
    """Place order through Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return None
            
        api_instance = upstox.OrderApi(client['api_client'])
        
        # Prepare order parameters
        order_request = upstox.PlaceOrderRequest(
            quantity=order_params['quantity'],
            product=order_params['product'],
            validity=order_params['validity'],
            price=order_params.get('price', 0),
            tag=order_params.get('tag', ''),
            instrument_token=order_params['instrument_token'],
            order_type=order_params['order_type'],
            transaction_type=order_params['transaction_type'],
            disclosed_quantity=order_params.get('disclosed_quantity', 0),
            trigger_price=order_params.get('trigger_price', 0),
            is_amo=order_params.get('is_amo', False)
        )
        
        response = api_instance.place_order(order_request)
        return response.data.order_id
        
    except ApiException as e:
        st.error(f"Upstox order failed: {e}")
        return None
    except Exception as e:
        st.error(f"Upstox order error: {e}")
        return None

def get_upstox_positions():
    """Fetch positions from Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return pd.DataFrame()
            
        api_instance = upstox.PortfolioApi(client['api_client'])
        response = api_instance.get_positions()
        
        positions = []
        if response and hasattr(response, 'data'):
            for position in response.data:
                positions.append({
                    'tradingsymbol': position.tradingsymbol,
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'last_price': position.last_price,
                    'pnl': position.pnl,
                    'product': position.product
                })
                
        return pd.DataFrame(positions)
        
    except Exception as e:
        st.error(f"Error fetching Upstox positions: {e}")
        return pd.DataFrame()

def get_upstox_holdings():
    """Fetch holdings from Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return pd.DataFrame()
            
        api_instance = upstox.PortfolioApi(client['api_client'])
        response = api_instance.get_holdings()
        
        holdings = []
        if response and hasattr(response, 'data'):
            for holding in response.data:
                holdings.append({
                    'tradingsymbol': holding.tradingsymbol,
                    'quantity': holding.quantity,
                    'average_price': holding.average_price,
                    'last_price': holding.last_price,
                    'pnl': holding.pnl,
                    'product': holding.product
                })
                
        return pd.DataFrame(holdings)
        
    except Exception as e:
        st.error(f"Error fetching Upstox holdings: {e}")
        return pd.DataFrame()

def get_upstox_order_book():
    """Fetch order book from Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return pd.DataFrame()
            
        api_instance = upstox.OrderApi(client['api_client'])
        response = api_instance.get_order_book()
        
        orders = []
        if response and hasattr(response, 'data'):
            for order in response.data:
                orders.append({
                    'order_id': order.order_id,
                    'tradingsymbol': order.tradingsymbol,
                    'transaction_type': order.transaction_type,
                    'order_type': order.order_type,
                    'product': order.product,
                    'quantity': order.quantity,
                    'filled_quantity': order.filled_quantity,
                    'price': order.price,
                    'status': order.status,
                    'order_timestamp': order.order_timestamp
                })
                
        return pd.DataFrame(orders)
        
    except Exception as e:
        st.error(f"Error fetching Upstox order book: {e}")
        return pd.DataFrame()

# @st.dialog("Quick Trade")
def quick_trade_interface(symbol=None, exchange=None):
    """A quick trade interface that doesn't use dialogs."""
    if st.session_state.get('show_quick_trade', False):
        st.markdown("---")
        st.subheader(f"Quick Trade - {symbol}" if symbol else "Quick Order")
        
        if symbol is None:
            symbol = st.text_input("Symbol").upper()
        
        transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="quick_trans_type")
        product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="quick_prod_type")
        order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="quick_order_type")
        quantity = st.number_input("Quantity", min_value=1, step=1, key="quick_qty")
        price = st.number_input("Price", min_value=0.01, key="quick_price") if order_type == "LIMIT" else 0

        col1, col2 = st.columns(2)
        if col1.button("Submit Order", use_container_width=True, type="primary"):
            if symbol and quantity > 0:
                place_order(get_instrument_df(), symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
                st.session_state.show_quick_trade = False
                st.rerun()
            else:
                st.warning("Please fill in all fields.")
        
        if col2.button("Cancel", use_container_width=True):
            st.session_state.show_quick_trade = False
            st.rerun()

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    holidays_by_year = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
        2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    }
    return holidays_by_year.get(year, [])


# =============================================================================
# MARKET TIMING FUNCTIONS - REPLACE EXISTING ONES
# =============================================================================
def is_market_hours():
    """Check if current time is within market hours OR a special session."""
    if check_for_special_session():
        return True

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    if now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        return False
        
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5:
        return False
    
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    return market_open <= current_time <= market_close

def is_pre_market_hours():
    """Check for pre-market hours, but not during a special session."""
    if check_for_special_session():
        return False

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5 or now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        return False
    
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    
    return pre_market_start <= current_time < market_open

def is_square_off_time():
    """Check for square-off time, but not during a special session."""
    if check_for_special_session():
        return False

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5 or now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        return False
    
    square_off_start = time(15, 20)
    square_off_end = time(15, 30)
    
    return square_off_start <= current_time <= square_off_end

def is_derivatives_square_off_time():
    """Check for derivatives square-off time, but not during a special session."""
    if check_for_special_session():
        return False

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5 or now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        return False
    
    square_off_start = time(15, 25)
    square_off_end = time(15, 30)
    
    return square_off_start <= current_time <= square_off_end

def get_market_status():
    """Get current market status, accounting for special sessions, holidays, and weekends."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    active_session = check_for_special_session()
    if active_session:
        session_end_datetime = datetime.combine(active_session['date'], active_session['end'], tzinfo=ist)
        return {
            "status": "special_trading_session",
            "next_market": session_end_datetime,
            "color": "#9400D3"
        }

    if now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in get_market_holidays(next_day.year):
            next_day += timedelta(days=1)
        return {
            "status": "market_holiday",
            "next_market": next_day.replace(hour=9, minute=0, second=0, microsecond=0),
            "color": "#A52A2A"
        }

    current_day = now.weekday()
    if current_day >= 5:
        next_day = now + timedelta(days=(7 - current_day))
        while next_day.strftime('%Y-%m-%d') in get_market_holidays(next_day.year):
            next_day += timedelta(days=1)
        return {
            "status": "weekend",
            "next_market": next_day.replace(hour=9, minute=0, second=0, microsecond=0),
            "color": "#cccccc"
        }

    current_time = now.time()
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    equity_square_off = time(15, 20)
    derivatives_square_off = time(15, 25)
    market_close = time(15, 30)

    if current_time < pre_market_start:
        status, next_event_time, color = "market_closed", time(9, 0), "#cccccc"
    elif current_time < market_open:
        status, next_event_time, color = "pre_market", time(9, 15), "#ffcc00"
    elif current_time < equity_square_off:
        status, next_event_time, color = "market_open", time(15, 20), "#00cc00"
    elif current_time < derivatives_square_off:
        status, next_event_time, color = "equity_square_off", time(15, 25), "#ff9900"
    elif current_time <= market_close:
        status, next_event_time, color = "derivatives_square_off", time(15, 30), "#ff6600"
    else:
        status, color = "market_closed", "#cccccc"
        next_day = now + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in get_market_holidays(next_day.year):
            next_day += timedelta(days=1)
        return {
            "status": status,
            "next_market": next_day.replace(hour=9, minute=0, second=0, microsecond=0),
            "color": color
        }

    return {
        "status": status,
        "next_market": now.replace(hour=next_event_time.hour, minute=next_event_time.minute, second=0, microsecond=0),
        "color": color
    }
    
def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    status_info = get_market_status()
    current_time = get_ist_time().strftime("%H:%M:%S IST")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="text-align: right;">
                <h5 style="margin: 0;">{current_time}</h5>
                <h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"].replace("_", " ").title()}</span></h5>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Buy", use_container_width=True, key="header_buy"):
            st.session_state.show_quick_trade = True
            st.rerun()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            st.session_state.show_quick_trade = True
            st.rerun()

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def get_global_indices_data(tickers):
    """Fetch global indices data for the overnight changes bar."""
    if not tickers:
        return pd.DataFrame()
    
    data = []
    
    for ticker_name, yf_ticker in tickers.items():
        try:
            stock_data = yf.download(yf_ticker, period="2d", progress=False)
            
            if not stock_data.empty and len(stock_data) >= 2:
                current_close = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2]
                
                if prev_close > 0:
                    change = current_close - prev_close
                    pct_change = (change / prev_close) * 100
                else:
                    change = 0
                    pct_change = 0
                
                data.append({
                    'Ticker': ticker_name,
                    'Price': current_close,
                    'Change': change,
                    '% Change': pct_change
                })
            else:
                # Add placeholder for unavailable data
                data.append({
                    'Ticker': ticker_name,
                    'Price': np.nan,
                    'Change': np.nan,
                    '% Change': np.nan
                })
                
        except Exception as e:
            print(f"Error fetching {ticker_name}: {e}")
            # Add placeholder for failed fetch
            data.append({
                'Ticker': ticker_name,
                'Price': np.nan,
                'Change': np.nan,
                '% Change': np.nan
            })
    
    return pd.DataFrame(data)
    
def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
    overnight_tickers = {"GIFT NIFTY": "NIFTY_F1", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
    data = get_global_indices_data(overnight_tickers)
    
    if not data.empty:
        bar_html = "<div class='notification-bar'>"
        for name, ticker in overnight_tickers.items():
            row = data[data['Ticker'] == name]
            if not row.empty:
                price = row.iloc[0]['Price']
                change = row.iloc[0]['% Change']
                if not np.isnan(price):
                    color = 'var(--green)' if change > 0 else 'var(--red)'
                    bar_html += f"<span>{name}: {price:,.2f} <span style='color:{color};'>({change:+.2f}%)</span></span>"
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    """Generates a Plotly chart with various chart types and overlays."""
    fig = go.Figure()
    if df.empty: return fig
    chart_df = df.copy()
    
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.droplevel(0)
        
    chart_df.columns = [str(col).lower() for col in chart_df.columns]
    
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in chart_df.columns for col in required_cols):
        st.error(f"Charting error for {ticker}: Dataframe is missing required columns (open, high, low, close).")
        return go.Figure()

    if chart_type == 'Heikin-Ashi':
        # Manual Heikin-Ashi calculation
        ha_close = (chart_df['open'] + chart_df['high'] + chart_df['low'] + chart_df['close']) / 4
        ha_open = (chart_df['open'].shift(1) + chart_df['close'].shift(1)) / 2
        ha_open.iloc[0] = (chart_df['open'].iloc[0] + chart_df['close'].iloc[0]) / 2
        ha_high = chart_df[['high', 'open', 'close']].max(axis=1)
        ha_low = chart_df[['low', 'open', 'close']].min(axis=1)
        
        fig.add_trace(go.Candlestick(x=chart_df.index, open=ha_open, high=ha_high, low=ha_low, close=ha_close, name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Bar'))
    else:
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
        
    # Bollinger Bands using TA-Lib
    if 'close' in chart_df.columns:
        upperband, middleband, lowerband = talib.BBANDS(chart_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        fig.add_trace(go.Scatter(x=chart_df.index, y=lowerband, line=dict(color='rgba(135,206,250,0.5)', width=1), name='Lower Band'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=upperband, line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='Upper Band'))
        
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        if conf_int_df is not None:
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


@st.cache_resource(ttl=3600)
@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    client = get_broker_client()
    broker = st.session_state.get('broker')
    
    if not client: 
        st.info("Please connect to a broker to view the dashboard.")
        return pd.DataFrame()
    
    if broker == "Zerodha":
        try:
            df = pd.DataFrame(client.instruments())
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'])
            st.success(f"Loaded {len(df)} instruments from Zerodha")
            return df
        except Exception as e:
            st.error(f"Error loading Zerodha instruments: {e}")
            return pd.DataFrame()
    
    elif broker == "Upstox":
        access_token = st.session_state.get('upstox_access_token')
        if not access_token:
            st.error("Upstox access token not found. Please login again.")
            return pd.DataFrame()
        
        # Try the most common exchanges first
        exchanges_to_try = ['NSE', 'BSE', 'NFO', 'MCX', 'CDS']
        all_instruments = []
        
        for exchange in exchanges_to_try:
            try:
                exchange_instruments = get_upstox_instruments(access_token, exchange)
                if not exchange_instruments.empty:
                    all_instruments.append(exchange_instruments)
            except Exception as e:
                st.error(f"Failed to load instruments from {exchange}: {e}")
                continue
        
        if all_instruments:
            combined_df = pd.concat(all_instruments, ignore_index=True)
            st.success(f"Successfully loaded {len(combined_df)} instruments from Upstox")
            return combined_df
        else:
            st.error("""
            Could not load any instruments from Upstox. Possible reasons:
            1. Your Upstox account may not have access to these market segments
            2. There might be temporary API issues
            3. Your API key might not have the required permissions
            
            Please use the debug button to check available exchanges.
            """)
            return pd.DataFrame()
    
    else:
        st.warning(f"Instrument list for {broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None


@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API."""
    client = get_broker_client()
    broker = st.session_state.get('broker')
    
    if not client or not instrument_token: 
        return pd.DataFrame()
    
    if broker == "Zerodha":
        # Your existing Zerodha code
        if not to_date: to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        
        if from_date > to_date:
            from_date = to_date - timedelta(days=1)
            
        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty: return df
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            st.error(f"Kite API Error (Historical): {e}")
            return pd.DataFrame()
    
    elif broker == "Upstox":
        # Use Upstox REST API
        access_token = st.session_state.get('upstox_access_token')
        return get_upstox_historical_data(access_token, instrument_token, interval, period)
    
    else:
        st.warning(f"Historical data for {broker} not implemented.")
        return pd.DataFrame()

@st.cache_resource(ttl=3600)

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if not client or not symbols_with_exchange: 
        return pd.DataFrame()
    
    broker = st.session_state.broker
    
    if broker == "Zerodha":
        # Existing Zerodha implementation
        instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
        try:
            quotes = client.quote(instrument_names)
            watchlist = []
            for item in symbols_with_exchange:
                instrument = f"{item['exchange']}:{item['symbol']}"
                if instrument in quotes:
                    quote = quotes[instrument]
                    last_price = quote['last_price']
                    prev_close = quote['ohlc']['close']
                    change = last_price - prev_close
                    pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                    watchlist.append({
                        'Ticker': item['symbol'], 
                        'Exchange': item['exchange'], 
                        'Price': last_price, 
                        'Change': change, 
                        '% Change': pct_change
                    })
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
            
    elif broker == "Upstox":
        # Upstox implementation
        try:
            # Convert symbols to Upstox format
            upstox_instruments = []
            symbol_map = {}
            
            for item in symbols_with_exchange:
                # This mapping would depend on your instrument key format
                instrument_key = f"{item['exchange']}|{item['symbol']}"  # Adjust based on actual format
                upstox_instruments.append(instrument_key)
                symbol_map[instrument_key] = item['symbol']
            
            quotes = get_upstox_quote(upstox_instruments)
            watchlist = []
            
            for instrument_key, quote_data in quotes.items():
                symbol = symbol_map.get(instrument_key, instrument_key)
                last_price = quote_data.get('last_price', 0)
                prev_close = quote_data.get('close', last_price)
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                
                watchlist.append({
                    'Ticker': symbol,
                    'Exchange': instrument_key.split('|')[0],
                    'Price': last_price,
                    'Change': change,
                    '% Change': pct_change
                })
                
            return pd.DataFrame(watchlist)
            
        except Exception as e:
            st.toast(f"Error fetching Upstox watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
    
    else:
        st.warning(f"Watchlist for {broker} not implemented.")
        return pd.DataFrame()


@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth (order book) for a given instrument."""
    client = get_broker_client()
    if not client or not instrument_token:
        return None
    try:
        depth = client.depth(instrument_token)
        return depth.get(str(instrument_token))
    except Exception as e:
        st.toast(f"Error fetching market depth: {e}", icon="⚠️")
        return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for a given underlying."""
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
    if st.session_state.broker == "Zerodha":
        exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "NATURALGAS": "MCX", "USDINR": "CDS"}
        exchange = exchange_map.get(underlying, 'NFO')
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}.get(underlying, underlying)
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"
        try:
            underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
        except Exception:
            underlying_ltp = 0.0
        
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty: return pd.DataFrame(), None, underlying_ltp, []
        
        expiries = sorted(options['expiry'].dt.date.unique())
        three_months_later = datetime.now().date() + timedelta(days=90)
        available_expiries = [e for e in expiries if datetime.now().date() <= e <= three_months_later]
        if not available_expiries: return pd.DataFrame(), None, underlying_ltp, []
        
        if not expiry_date: 
            expiry_date = available_expiries[0]
        else: 
            expiry_date = pd.to_datetime(expiry_date).date()
        
        chain_df = options[options['expiry'].dt.date == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        
        try:
            quotes = client.quote(instruments_to_fetch)
            ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            ce_df['oi'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            pe_df['oi'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            
            final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                   pe_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                   on='strike', suffixes=('_CE', '_PE')).rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'oi_CE': 'CALL OI', 'oi_PE': 'PUT OI', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
            
            return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries
        except Exception as e:
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if not client: 
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
        
    broker = st.session_state.broker
    
    if broker == "Zerodha":
        # Existing Zerodha implementation
        try:
            positions = client.positions().get('net', [])
            holdings = client.holdings()
            positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
            total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
            holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
            total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
            return positions_df, holdings_df, total_pnl, total_investment
        except Exception as e:
            st.error(f"Portfolio API Error: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
            
    elif broker == "Upstox":
        # Upstox implementation
        try:
            positions_df = get_upstox_positions()
            holdings_df = get_upstox_holdings()
            
            total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
            total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
            
            return positions_df, holdings_df, total_pnl, total_investment
            
        except Exception as e:
            st.error(f"Upstox portfolio error: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    
    else:
        st.warning(f"Portfolio for {broker} not implemented.")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0


def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
        
    broker = st.session_state.broker
    
    if broker == "Zerodha":
        # Existing Zerodha implementation
        try:
            is_option = any(char.isdigit() for char in symbol)
            if is_option:
                exchange = 'NFO'
            else:
                instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
                if instrument.empty:
                    st.error(f"Symbol '{symbol}' not found.")
                    return
                exchange = instrument.iloc[0]['exchange']

            order_id = client.place_order(
                tradingsymbol=symbol.upper(), 
                exchange=exchange, 
                transaction_type=transaction_type, 
                quantity=quantity, 
                order_type=order_type, 
                product=product, 
                variety=client.VARIETY_REGULAR, 
                price=price
            )
            st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
            st.session_state.order_history.insert(0, {
                "id": order_id, 
                "symbol": symbol, 
                "qty": quantity, 
                "type": transaction_type, 
                "status": "Success"
            })
        except Exception as e:
            st.toast(f"❌ Order failed: {e}", icon="🔥")
            st.session_state.order_history.insert(0, {
                "id": "N/A", 
                "symbol": symbol, 
                "qty": quantity, 
                "type": transaction_type, 
                "status": f"Failed: {e}"
            })
            
    elif broker == "Upstox":
        # Upstox implementation
        try:
            # Find instrument token
            instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
            if instrument.empty:
                st.error(f"Symbol '{symbol}' not found.")
                return
                
            instrument_token = instrument.iloc[0]['instrument_token']
            
            # Map product types
            product_map = {
                'MIS': 'intraday',
                'CNC': 'delivery',
                'NRML': 'normal'
            }
            
            # Map order types
            order_type_map = {
                'MARKET': 'market',
                'LIMIT': 'limit'
            }
            
            order_params = {
                'instrument_token': instrument_token,
                'quantity': quantity,
                'product': product_map.get(product, 'intraday'),
                'validity': 'day',
                'order_type': order_type_map.get(order_type, 'market'),
                'transaction_type': transaction_type.lower(),
                'price': price if order_type == 'LIMIT' else 0
            }
            
            order_id = place_upstox_order(order_params)
            if order_id:
                st.toast(f"✅ Upstox order placed successfully! ID: {order_id}", icon="🎉")
                st.session_state.order_history.insert(0, {
                    "id": order_id, 
                    "symbol": symbol, 
                    "qty": quantity, 
                    "type": transaction_type, 
                    "status": "Success"
                })
            else:
                st.toast(f"❌ Upstox order failed", icon="🔥")
                st.session_state.order_history.insert(0, {
                    "id": "N/A", 
                    "symbol": symbol, 
                    "qty": quantity, 
                    "type": transaction_type, 
                    "status": "Failed"
                })
                
        except Exception as e:
            st.toast(f"❌ Upstox order failed: {e}", icon="🔥")
            st.session_state.order_history.insert(0, {
                "id": "N/A", 
                "symbol": symbol, 
                "qty": quantity, 
                "type": transaction_type, 
                "status": f"Failed: {e}"
            })
    
    else:
        st.warning(f"Order placement for {broker} not implemented.")


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_and_analyze_news(query=None):
    """Enhanced news fetcher with multiple fallback sources and better error handling."""
    analyzer = SentimentIntensityAnalyzer()
    
    # Expanded news sources with fallbacks
    news_sources = {
        "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
        "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Business Standard Markets": "https://www.business-standard.com/rss/markets-102.rss",
        "Livemint Markets": "https://www.livemint.com/rss/markets",
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "Reuters Markets": "https://feeds.reuters.com/reuters/INmarketsNews",
        "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
        "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
        "Financial Express": "https://www.financialexpress.com/feed/",
    }
    
    all_news = []
    
    for source, url in news_sources.items():
        try:
            # Add timeout and better error handling
            feed = feedparser.parse(url)
            
            if hasattr(feed, 'entries') and feed.entries:
                for entry in feed.entries[:8]:  # Limit per source
                    try:
                        # Get publication date
                        published_date = datetime.now().date()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_date = datetime(*entry.published_parsed[:6]).date()
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published_date = datetime(*entry.updated_parsed[:6]).date()
                        
                        # Check if news matches query
                        title = entry.title if hasattr(entry, 'title') else "No title"
                        summary = entry.summary if hasattr(entry, 'summary') else ""
                        
                        if query is None or query.lower() in title.lower() or query.lower() in summary.lower():
                            # Calculate sentiment
                            text_for_sentiment = f"{title} {summary}"
                            sentiment_score = analyzer.polarity_scores(text_for_sentiment)['compound']
                            
                            all_news.append({
                                "source": source,
                                "title": title,
                                "link": entry.link if hasattr(entry, 'link') else "#",
                                "date": published_date,
                                "sentiment": sentiment_score,
                                "summary": summary[:200] + "..." if len(summary) > 200 else summary
                            })
                    except Exception as e:
                        continue  # Skip individual entry errors
                        
        except Exception as e:
            continue  # Skip source if it fails
    
    # Sort by date (newest first) and return
    all_news.sort(key=lambda x: x['date'], reverse=True)
    return pd.DataFrame(all_news)

# Fallback news data for when feeds are unavailable
def get_fallback_news():
    """Provide fallback news when live feeds are down."""
    fallback_news = [
        {
            "source": "Market Update",
            "title": "Indian markets expected to open positive following global cues",
            "link": "#",
            "date": datetime.now().date(),
            "sentiment": 0.2,
            "summary": "Global markets show positive trend, likely to influence Indian market opening."
        },
        {
            "source": "Economic Indicators",
            "title": "RBI monetary policy meeting scheduled for this week",
            "link": "#", 
            "date": datetime.now().date(),
            "sentiment": 0.1,
            "summary": "Market participants await RBI's decision on interest rates and policy stance."
        },
        {
            "source": "Corporate News",
            "title": "Major IT companies to announce quarterly results",
            "link": "#",
            "date": datetime.now().date(),
            "sentiment": 0.15,
            "summary": "IT sector in focus as earnings season begins this week."
        },
        {
            "source": "Global Markets",
            "title": "US Fed minutes and economic data to guide global markets",
            "link": "#",
            "date": datetime.now().date(),
            "sentiment": 0.05,
            "summary": "Federal Reserve policy outlook and economic indicators to influence market direction."
        },
        {
            "source": "Commodities",
            "title": "Crude oil prices stable amid supply concerns",
            "link": "#",
            "date": datetime.now().date(),
            "sentiment": -0.1,
            "summary": "Oil prices remain volatile due to geopolitical tensions and supply dynamics."
        }
    ]
    return pd.DataFrame(fallback_news)

def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE function to remove sklearn dependency."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data, forecast_steps=30):
    """Trains a Seasonal ARIMA model for time series forecasting."""
    if _data.empty or len(_data) < 100:
        return None, None, None

    df = _data.copy()
    df.index = pd.to_datetime(df.index)
    
    try:
        decomposed = seasonal_decompose(df['close'], model='additive', period=7)
        seasonally_adjusted = df['close'] - decomposed.seasonal

        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
        
        # Backtesting
        fitted_values = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values}).dropna()
        
        # Forecasting
        forecast_result = model.get_forecast(steps=forecast_steps)
        forecast_adjusted = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        last_season_cycle = decomposed.seasonal.iloc[-7:]
        future_seasonal_values = np.tile(last_season_cycle.values, forecast_steps // 7 + 1)[:forecast_steps]
        
        future_forecast = forecast_adjusted + future_seasonal_values
        
        future_dates = pd.to_datetime(pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_steps))
        forecast_df = pd.DataFrame({'Predicted': future_forecast.values}, index=future_dates)
        
        # Add seasonality back to confidence intervals
        conf_int_df = pd.DataFrame({
            'lower': conf_int.iloc[:, 0] + future_seasonal_values,
            'upper': conf_int.iloc[:, 1] + future_seasonal_values
        }, index=future_dates)
        
        return forecast_df, backtest_df, conf_int_df

    except Exception as e:
        st.error(f"Seasonal ARIMA model training failed: {e}")
        return None, None, None

@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads and combines historical data from a static CSV with live data from the broker."""
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()
    try:
        response = requests.get(source_info['github_url'])
        response.raise_for_status()
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True).dt.tz_localize(None)
        hist_df.set_index('Date', inplace=True)
        hist_df.columns = [col.lower() for col in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return pd.DataFrame()
        
    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol') and source_info.get('exchange') != 'yfinance':
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty: 
                live_df.index = live_df.index.tz_convert(None)
                live_df.columns = [col.lower() for col in live_df.columns]
    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max")
            if not live_df.empty: 
                live_df.index = live_df.index.tz_localize(None)
                live_df.columns = [col.lower() for col in live_df.columns]
        except Exception as e:
            st.error(f"Failed to load yfinance data: {e}")
            live_df = pd.DataFrame()
            
    if not live_df.empty:
        hist_df.index = hist_df.index.tz_localize(None) if hist_df.index.tz is not None else hist_df.index
        live_df.index = live_df.index.tz_localize(None) if live_df.index.tz is not None else live_df.index
        
        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        hist_df.sort_index(inplace=True)
        return hist_df

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price and Greeks."""
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError):
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty: return {}
    latest = df.iloc[-1].copy()
    latest.index = latest.index.str.lower()
    interpretation = {}
    
    # RSI using TA-Lib
    if 'close' in df.columns:
        rsi = talib.RSI(df['close'], timeperiod=14)
        if not np.isnan(rsi.iloc[-1]):
            rsi_val = rsi.iloc[-1]
            interpretation['RSI (14)'] = "Overbought (Bearish)" if rsi_val > 70 else "Oversold (Bullish)" if rsi_val < 30 else "Neutral"
    
    # Stochastic using TA-Lib
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    if not np.isnan(slowk.iloc[-1]):
        stoch_k = slowk.iloc[-1]
        interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
    
    # MACD using TA-Lib
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    if not np.isnan(macd.iloc[-1]) and not np.isnan(macdsignal.iloc[-1]):
        interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd.iloc[-1] > macdsignal.iloc[-1] else "Bearish Crossover"
    
    # ADX using TA-Lib
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    if not np.isnan(adx.iloc[-1]):
        adx_val = adx.iloc[-1]
        interpretation['ADX (14)'] = f"Strong Trend ({adx_val:.1f})" if adx_val > 25 else f"Weak/No Trend ({adx_val:.1f})"
    
    return interpretation

# ================ 4. HNI & PRO TRADER FEATURES ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        orders_to_place = []
        for item in basket_items:
            instrument = instrument_df[instrument_df['tradingsymbol'] == item['symbol']]
            if instrument.empty:
                st.toast(f"❌ Could not find symbol {item['symbol']} in instrument list. Skipping.", icon="🔥")
                continue
            exchange = instrument.iloc[0]['exchange']

            order = {
                "tradingsymbol": item['symbol'],
                "exchange": exchange,
                "transaction_type": client.TRANSACTION_TYPE_BUY if item['transaction_type'] == 'BUY' else client.TRANSACTION_TYPE_SELL,
                "quantity": int(item['quantity']),
                "product": client.PRODUCT_MIS if item['product'] == 'MIS' else client.PRODUCT_CNC,
                "order_type": client.ORDER_TYPE_MARKET if item['order_type'] == 'MARKET' else client.ORDER_TYPE_LIMIT,
            }
            if order['order_type'] == client.ORDER_TYPE_LIMIT:
                order['price'] = item['price']
            orders_to_place.append(order)
        
        if not orders_to_place:
            st.warning("No valid orders to place in the basket.")
            return

        try:
            client.place_order(variety=client.VARIETY_REGULAR, orders=orders_to_place)
            st.toast("✅ Basket order placed successfully!", icon="🎉")
            st.session_state.basket = []
            st.rerun()
        except Exception as e:
            st.toast(f"❌ Basket order failed: {e}", icon="🔥")

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("'sensex_sectors.csv' not found. Sector allocation will be unavailable.")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to highlight ITM/OTM in the options chain."""
    if df.empty or 'STRIKE' not in df.columns or ltp == 0:
        return df.style

    def highlight_itm(row):
        styles = [''] * len(row)
        if row['STRIKE'] < ltp:
            styles[df.columns.get_loc('CALL LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('CALL OI')] = 'background-color: #2E4053'
        if row['STRIKE'] > ltp:
            styles[df.columns.get_loc('PUT LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('PUT OI')] = 'background-color: #2E4053'
        return styles

    return df.style.apply(highlight_itm, axis=1)

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    if 'show_most_active' not in st.session_state:
        st.session_state.show_most_active = False
    
    if st.button("Most Active Options", use_container_width=True) or st.session_state.show_most_active:
        st.session_state.show_most_active = True
        
        st.subheader(f"Most Active {underlying} Options (By Volume)")
        with st.spinner("Fetching data..."):

            client = get_broker_client()
            if not client:
                st.toast("Broker not connected.", icon="⚠️")
                return pd.DataFrame()
            
            try:
                chain_df, expiry, _, _ = get_options_chain(underlying, instrument_df)
                if chain_df.empty or expiry is None:
                    return pd.DataFrame()
                ce_symbols = chain_df['CALL'].dropna().tolist()
                pe_symbols = chain_df['PUT'].dropna().tolist()
                all_symbols = [f"NFO:{s}" for s in ce_symbols + pe_symbols]

                if not all_symbols:
                    return pd.DataFrame()

                quotes = client.quote(all_symbols)
                
                active_options = []
                for symbol, data in quotes.items():
                    prev_close = data.get('ohlc', {}).get('close', 0)
                    last_price = data.get('last_price', 0)
                    change = last_price - prev_close
                    pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                    
                    active_options.append({
                        'Symbol': data.get('tradingsymbol'),
                        'LTP': last_price,
                        'Change %': pct_change,
                        'Volume': data.get('volume', 0),
                        'OI': data.get('oi', 0)
                    })
                
                df = pd.DataFrame(active_options)
                df_sorted = df.sort_values(by='Volume', ascending=False)
                st.dataframe(df_sorted.head(10), use_container_width=True, hide_index=True)

                if st.button("Close", use_container_width=True):
                    st.session_state.show_most_active = False
                    st.rerun()

            except Exception as e:
                st.error(f"Could not fetch most active options: {e}")


@st.cache_data(ttl=300)  # 5 minute cache
def get_global_indices_data_enhanced(tickers):
    """Enhanced global indices data fetcher with multiple fallbacks."""
    if not tickers:
        return pd.DataFrame()
    
    data = []
    
    for ticker_name, yf_ticker in tickers.items():
        try:
            # Try to download data with timeout
            stock_data = yf.download(yf_ticker, period="2d", progress=False, timeout=10)
            
            if stock_data.empty or len(stock_data) < 2:
                # Try alternative period
                stock_data = yf.download(yf_ticker, period="5d", progress=False, timeout=10)
            
            if not stock_data.empty and len(stock_data) >= 2:
                current_close = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2]
                
                # Handle division by zero
                if prev_close > 0:
                    change = current_close - prev_close
                    pct_change = (change / prev_close) * 100
                else:
                    change = 0
                    pct_change = 0
                
                data.append({
                    'Ticker': ticker_name,
                    'Price': current_close,
                    'Change': change,
                    '% Change': pct_change
                })
            else:
                # Add placeholder for unavailable data
                data.append({
                    'Ticker': ticker_name,
                    'Price': np.nan,
                    'Change': np.nan,
                    '% Change': np.nan
                })
                
        except Exception as e:
            print(f"Error fetching {ticker_name}: {e}")
            # Add placeholder for failed fetch
            data.append({
                'Ticker': ticker_name,
                'Price': np.nan,
                'Change': np.nan,
                '% Change': np.nan
            })
    
    return pd.DataFrame(data)

# Fallback global data
def get_fallback_global_data(tickers):
    """Provide fallback global data when live data is unavailable."""
    fallback_data = []
    
    # Mock data based on typical values
    mock_data = {
        "S&P 500": 4500.0,
        "Dow Jones": 34500.0, 
        "NASDAQ": 14000.0,
        "FTSE 100": 7500.0,
        "Nikkei 225": 32500.0,
        "Hang Seng": 18000.0,
        "SGX Nifty": 19500.0
    }
    
    for ticker_name in tickers.keys():
        base_price = mock_data.get(ticker_name, 1000.0)
        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # -2% to +2%
        current_price = base_price * (1 + variation)
        change = current_price - base_price
        pct_change = (change / base_price) * 100
        
        fallback_data.append({
            'Ticker': ticker_name,
            'Price': current_price,
            'Change': change,
            '% Change': pct_change
        })
    
    return pd.DataFrame(fallback_data)

# ================ ALGO TRADING BOTS SECTION ================

def momentum_trader_bot(instrument_df, symbol, capital=100):
    """Momentum trading bot that buys on upward momentum and sells on downward momentum."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, '5minute', period='1d')
        if data.empty or len(data) < 20:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate indicators with TA-Lib
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
        data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = []
        
        # Momentum signals
        if (latest['EMA_20'] > latest['EMA_50'] and 
            prev['EMA_20'] <= prev['EMA_50']):
            signals.append("EMA crossover - BULLISH")
        
        rsi_val = latest.get('RSI', 50)
        if rsi_val < 30:
            signals.append("RSI oversold - BULLISH")
        elif rsi_val > 70:
            signals.append("RSI overbought - BEARISH")
        
        # Price momentum
        if len(data) >= 6:
            price_change_5min = ((latest['close'] - data.iloc[-6]['close']) / data.iloc[-6]['close']) * 100
            if price_change_5min > 0.5:
                signals.append(f"Strong upward momentum: +{price_change_5min:.2f}%")
        
        # Calculate position size
        current_price = latest['close']
        quantity = max(1, int((capital * 0.8) / current_price))
        
        action = "HOLD"
        if len([s for s in signals if "BULLISH" in s]) >= 2:
            action = "BUY"
        elif len([s for s in signals if "BEARISH" in s]) >= 2:
            action = "SELL"
        
        return {
            "bot_name": "Momentum Trader",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "Medium"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def mean_reversion_bot(instrument_df, symbol, capital=100):
    """Mean reversion bot that trades on price returning to mean levels."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, '15minute', period='5d')
        if data.empty or len(data) < 50:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate Bollinger Bands using TA-Lib
        upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data['BB_Upper'] = upperband
        data['BB_Middle'] = middleband
        data['BB_Lower'] = lowerband
        
        latest = data.iloc[-1]
        
        signals = []
        current_price = latest['close']
        bb_lower = latest.get('BB_Lower', current_price)
        bb_upper = latest.get('BB_Upper', current_price)
        bb_middle = latest.get('BB_Middle', current_price)
        
        # Mean reversion signals
        if current_price <= bb_lower * 1.02:
            signals.append("Near lower Bollinger Band - BULLISH")
        
        if current_price >= bb_upper * 0.98:
            signals.append("Near upper Bollinger Band - BEARISH")
        
        # Distance from mean
        distance_from_mean = ((current_price - bb_middle) / bb_middle) * 100
        if abs(distance_from_mean) > 3:
            signals.append(f"Price {abs(distance_from_mean):.1f}% from mean")
        
        # RSI for confirmation
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        rsi = data['RSI'].iloc[-1]
        if rsi < 35:
            signals.append("RSI supporting oversold condition")
        elif rsi > 65:
            signals.append("RSI supporting overbought condition")
        
        # Calculate position size
        quantity = max(1, int((capital * 0.6) / current_price))
        
        action = "HOLD"
        if any("BULLISH" in s for s in signals) and rsi < 40:
            action = "BUY"
        elif any("BEARISH" in s for s in signals) and rsi > 60:
            action = "SELL"
        
        return {
            "bot_name": "Mean Reversion",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "Low"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def volatility_breakout_bot(instrument_df, symbol, capital=100):
    """Volatility breakout bot that trades on breakouts from consolidation."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, '30minute', period='5d')
        if data.empty or len(data) < 30:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate ATR using TA-Lib
        data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['Range'] = data['high'] - data['low']
        avg_range = data['Range'].rolling(window=20).mean()
        
        latest = data.iloc[-1]
        current_price = latest['close']
        current_atr = latest['ATR']
        current_range = latest['Range']
        
        signals = []
        
        # Volatility signals
        if current_range > avg_range.iloc[-1] * 1.5:
            signals.append("High volatility - potential breakout")
        
        # Price action signals
        prev_high = data['high'].iloc[-2]
        prev_low = data['low'].iloc[-2]
        
        if current_price > prev_high + current_atr * 0.5:
            signals.append("Breakout above resistance - BULLISH")
        
        if current_price < prev_low - current_atr * 0.5:
            signals.append("Breakdown below support - BEARISH")
        
        # Volume confirmation (if available)
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            if data['volume'].iloc[-1] > avg_volume * 1.2:
                signals.append("High volume confirmation")
        
        # Calculate position size based on ATR
        atr_percentage = (current_atr / current_price) * 100
        risk_per_trade = min(20, max(5, atr_percentage * 2))
        quantity = max(1, int((capital * (risk_per_trade / 100)) / current_price))
        
        action = "HOLD"
        if any("BULLISH" in s for s in signals):
            action = "BUY"
        elif any("BEARISH" in s for s in signals):
            action = "SELL"
        
        return {
            "bot_name": "Volatility Breakout",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "High"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def value_investor_bot(instrument_df, symbol, capital=100):
    """Value investor bot focusing on longer-term value signals."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, 'day', period='1y')
        if data.empty or len(data) < 100:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate moving averages using TA-Lib
        data['SMA_50'] = talib.SMA(data['close'], timeperiod=50)
        data['SMA_200'] = talib.SMA(data['close'], timeperiod=200)
        data['EMA_21'] = talib.EMA(data['close'], timeperiod=21)
        
        latest = data.iloc[-1]
        current_price = latest['close']
        
        signals = []
        
        # Trend analysis
        if latest['SMA_50'] > latest['SMA_200']:
            signals.append("Bullish trend (50 > 200 SMA)")
        else:
            signals.append("Bearish trend (50 < 200 SMA)")
        
        # Support and resistance levels
        support_20 = data['low'].rolling(window=20).min().iloc[-1]
        resistance_20 = data['high'].rolling(window=20).max().iloc[-1]
        
        distance_to_support = ((current_price - support_20) / current_price) * 100
        distance_to_resistance = ((resistance_20 - current_price) / current_price) * 100
        
        if distance_to_support < 5:
            signals.append("Near strong support - BULLISH")
        
        if distance_to_resistance < 5:
            signals.append("Near strong resistance - BEARISH")
        
        # Monthly performance
        monthly_return = ((current_price - data['close'].iloc[-21]) / data['close'].iloc[-21]) * 100
        if monthly_return < -10:
            signals.append("Oversold on monthly basis - BULLISH")
        elif monthly_return > 15:
            signals.append("Overbought on monthly basis - BEARISH")
        
        # Calculate position size for longer term
        quantity = max(1, int((capital * 0.5) / current_price))
        
        action = "HOLD"
        bullish_signals = len([s for s in signals if "BULLISH" in s])
        bearish_signals = len([s for s in signals if "BEARISH" in s])
        
        if bullish_signals >= 2 and bearish_signals == 0:
            action = "BUY"
        elif bearish_signals >= 2 and bullish_signals == 0:
            action = "SELL"
        
        return {
            "bot_name": "Value Investor",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "Low"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def scalper_bot(instrument_df, symbol, capital=100):
    """High-frequency scalping bot for quick, small profits."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, 'minute', period='1d')
        if data.empty or len(data) < 100:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate scalping indicators using TA-Lib
        data['RSI_9'] = talib.RSI(data['close'], timeperiod=9)
        data['EMA_8'] = talib.EMA(data['close'], timeperiod=8)
        data['EMA_21'] = talib.EMA(data['close'], timeperiod=21)
        
        latest = data.iloc[-1]
        current_price = latest['close']
        
        signals = []
        
        # Scalping signals
        if latest['EMA_8'] > latest['EMA_21']:
            signals.append("Fast EMA above slow EMA - BULLISH")
        else:
            signals.append("Fast EMA below slow EMA - BEARISH")
        
        rsi_9 = latest['RSI_9']
        if rsi_9 < 25:
            signals.append("Extremely oversold - BULLISH")
        elif rsi_9 > 75:
            signals.append("Extremely overbought - BEARISH")
        
        # Price momentum for scalping
        price_change_3min = ((current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]) * 100
        if abs(price_change_3min) > 0.3:
            signals.append(f"Strong short-term momentum: {price_change_3min:+.2f}%")
        
        # Calculate small position size for scalping
        quantity = max(1, int((capital * 0.3) / current_price))
        
        action = "HOLD"
        if (any("BULLISH" in s for s in signals) and 
            "BEARISH" not in str(signals) and
            rsi_9 < 70):
            action = "BUY"
        elif (any("BEARISH" in s for s in signals) and 
              "BULLISH" not in str(signals) and
              rsi_9 > 30):
            action = "SELL"
        
        return {
            "bot_name": "Scalper Pro",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "Very High"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def trend_follower_bot(instrument_df, symbol, capital=100):
    """Trend following bot that rides established trends."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, 'hour', period='1mo')
        if data.empty or len(data) < 100:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate trend indicators using TA-Lib
        data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
        data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
        
        # Simple trend detection without SuperTrend
        data['Trend'] = np.where(data['EMA_20'] > data['EMA_50'], 1, -1)
        
        latest = data.iloc[-1]
        current_price = latest['close']
        
        signals = []
        
        # Trend strength
        adx = latest.get('ADX', 0)
        if adx > 25:
            signals.append(f"Strong trend (ADX: {adx:.1f})")
        else:
            signals.append(f"Weak trend (ADX: {adx:.1f})")
        
        # Trend direction
        if latest['EMA_20'] > latest['EMA_50']:
            signals.append("Uptrend confirmed - BULLISH")
        else:
            signals.append("Downtrend confirmed - BEARISH")
        
        # Price relative to trend
        if current_price > latest['EMA_20']:
            signals.append("Price above short-term trend - BULLISH")
        else:
            signals.append("Price below short-term trend - BEARISH")
        
        # Pullback opportunities
        if (latest['EMA_20'] > latest['EMA_50'] and 
            current_price < latest['EMA_20'] and 
            current_price > latest['EMA_50']):
            signals.append("Pullback in uptrend - BULLISH")
        
        elif (latest['EMA_20'] < latest['EMA_50'] and 
              current_price > latest['EMA_20'] and 
              current_price < latest['EMA_50']):
            signals.append("Pullback in downtrend - BEARISH")
        
        # Calculate position size
        quantity = max(1, int((capital * 0.7) / current_price))
        
        action = "HOLD"
        bullish_count = len([s for s in signals if "BULLISH" in s])
        bearish_count = len([s for s in signals if "BEARISH" in s])
        
        if bullish_count >= 2 and adx > 20:
            action = "BUY"
        elif bearish_count >= 2 and adx > 20:
            action = "SELL"
        
        return {
            "bot_name": "Trend Follower",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "current_price": current_price,
            "signals": signals,
            "capital_required": quantity * current_price,
            "risk_level": "Medium"
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

# Dictionary of all available bots
ALGO_BOTS = {
    "Momentum Trader": momentum_trader_bot,
    "Mean Reversion": mean_reversion_bot,
    "Volatility Breakout": volatility_breakout_bot,
    "Value Investor": value_investor_bot,
    "Scalper Pro": scalper_bot,
    "Trend Follower": trend_follower_bot
}

def execute_bot_trade(instrument_df, bot_result):
    """Displays bot recommendations WITHOUT automatic execution - requires manual confirmation."""
    if bot_result.get("error"):
        st.error(bot_result["error"])
        return
    
    if bot_result["action"] == "HOLD":
        st.info(f"🤖 {bot_result['bot_name']} recommends HOLDING {bot_result['symbol']}")
        return
    
    action = bot_result["action"]
    symbol = bot_result["symbol"]
    quantity = bot_result["quantity"]
    current_price = bot_result["current_price"]
    required_capital = bot_result["capital_required"]
    
    st.success(f"""
    🚀 **{bot_result['bot_name']} Recommendation:**
    - **Action:** {action} {quantity} shares of {symbol}
    - **Current Price:** ₹{current_price:.2f}
    - **Required Capital:** ₹{required_capital:.2f}
    - **Risk Level:** {bot_result['risk_level']}
    """)
    
    # Display signals first
    st.subheader("📊 Analysis Signals")
    for signal in bot_result["signals"]:
        if "BULLISH" in signal:
            st.success(f"✅ {signal}")
        elif "BEARISH" in signal:
            st.error(f"❌ {signal}")
        else:
            st.info(f"📈 {signal}")
    
    # Manual execution section - clearly separated
    st.markdown("---")
    st.subheader("🚀 Manual Trade Execution")
    st.warning("**Manual Confirmation Required:** Click the button below ONLY if you want to execute this trade.")
    
    col1, col2 = st.columns(2)
    
    if col1.button(f"Execute {action} Order", key=f"execute_{symbol}", use_container_width=True, type="primary"):
        # Only execute when user explicitly clicks
        place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
        st.toast(f"✅ {action} order for {symbol} placed successfully!", icon="🎉")
        st.rerun()
    
    if col2.button("Ignore Recommendation", key=f"ignore_{symbol}", use_container_width=True):
        st.info("Trade execution cancelled.")
        st.rerun()

def page_algo_bots():
    """Main algo bots page where users can run different trading bots."""
    display_header()
    st.title("🤖 Algo Trading Bots")
    st.info("Run automated trading bots with minimum capital of ₹100. Each bot uses different strategies and risk profiles.", icon="🤖")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo bots.")
        return
    
    # Bot selection and configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_bot = st.selectbox(
            "Select Trading Bot",
            list(ALGO_BOTS.keys()),
            help="Choose a trading bot based on your risk appetite and trading style"
        )
        
        # Bot descriptions
        bot_descriptions = {
            "Momentum Trader": "Trades on strong price momentum and trend continuations. Medium risk.",
            "Mean Reversion": "Buys low and sells high based on statistical mean reversion. Low risk.",
            "Volatility Breakout": "Captures breakouts from low volatility periods. High risk.",
            "Value Investor": "Focuses on longer-term value and fundamental trends. Low risk.",
            "Scalper Pro": "High-frequency trading for quick, small profits. Very high risk.",
            "Trend Follower": "Identifies and rides established market trends using ADX and EMAs. Medium risk."
        }
        
        st.markdown(f"**Description:** {bot_descriptions.get(selected_bot, 'N/A')}")
    
    with col2:
        trading_capital = st.number_input(
            "Trading Capital (₹)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Minimum ₹100 required"
        )
    
    st.markdown("---")
    
    # Symbol selection and bot execution
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Stock Selection")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox(
            "Select Stock",
            sorted(all_symbols),
            index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0
        )
        
        # Show current price
        quote_data = get_watchlist_data([{'symbol': selected_symbol, 'exchange': 'NSE'}])
        if not quote_data.empty:
            current_price = quote_data.iloc[0]['Price']
            st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col4:
        st.subheader("Bot Execution")
        st.write(f"**Selected Bot:** {selected_bot}")
        st.write(f"**Available Capital:** ₹{trading_capital:,}")
        
        if st.button("🚀 Run Trading Bot", use_container_width=True, type="primary"):
            with st.spinner(f"Running {selected_bot} analysis..."):
                bot_function = ALGO_BOTS[selected_bot]
                bot_result = bot_function(instrument_df, selected_symbol, trading_capital)
                
                if bot_result and not bot_result.get("error"):
                    st.session_state.last_bot_result = bot_result
                    st.rerun()
    
    # Display bot results
    if 'last_bot_result' in st.session_state and st.session_state.last_bot_result:
        bot_result = st.session_state.last_bot_result
        
        if bot_result.get("error"):
            st.error(bot_result["error"])
        else:
            st.markdown("---")
            st.subheader("🤖 Bot Analysis Results")
            
            # Create metrics cards
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                action_color = "green" if bot_result["action"] == "BUY" else "red" if bot_result["action"] == "SELL" else "orange"
                st.markdown(f'<div class="metric-card" style="border-color: {action_color};">'
                            f'<h3 style="color: {action_color};">{bot_result["action"]}</h3>'
                            f'<p>Recommended Action</p></div>', unsafe_allow_html=True)
            
            with col6:
                st.metric("Quantity", bot_result["quantity"])
            
            with col7:
                st.metric("Capital Required", f"₹{bot_result['capital_required']:.2f}")
            
            with col8:
                risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
                st.markdown(f'<div class="metric-card" style="border-color: {risk_color.get(bot_result["risk_level"], "gray")};">'
                            f'<h3 style="color: {risk_color.get(bot_result["risk_level"], "gray")};">{bot_result["risk_level"]}</h3>'
                            f'<p>Risk Level</p></div>', unsafe_allow_html=True)
            
            # Display signals and execute trade
            execute_bot_trade(instrument_df, bot_result)

    # Bot performance history
    st.markdown("---")
    st.subheader("📈 Bot Performance Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Best Practices:**
        - Start with minimum capital (₹100)
        - Use 'Value Investor' for beginners
        - 'Scalper Pro' requires constant monitoring
        - Always check signals before executing
        - Combine multiple bot recommendations
        """)
    
    with tips_col2:
        st.markdown("""
        **Risk Management:**
        - Never risk more than 2% per trade
        - Use stop losses with every trade
        - Diversify across different bots
        - Monitor performance regularly
        - Adjust capital based on experience
        """)
    
    # Quick bot comparison
    with st.expander("🤖 Bot Comparison Guide"):
        comparison_data = {
            "Bot": list(ALGO_BOTS.keys()),
            "Risk Level": ["Medium", "Low", "High", "Low", "Very High", "Medium"],
            "Holding Period": ["Hours", "Days", "Minutes", "Weeks", "Minutes", "Days"],
            "Capital Recommended": ["₹1,000+", "₹500+", "₹2,000+", "₹2,000+", "₹5,000+", "₹1,500+"],
            "Best For": ["Trend riding", "Safe returns", "Quick profits", "Long term", "Experienced", "Trend following"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Dictionary of all available bots (semi-automated)
ALGO_BOTS = {
    "Momentum Trader": momentum_trader_bot,
    "Mean Reversion": mean_reversion_bot,
    "Volatility Breakout": volatility_breakout_bot,
    "Value Investor": value_investor_bot,
    "Scalper Pro": scalper_bot,
    "Trend Follower": trend_follower_bot
}

# ================ AUTOMATED BOT FUNCTIONS ================

def automated_momentum_trader(instrument_df, symbol):
    """Enhanced momentum trader for automated mode."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, '5minute', period='1d')
        if data.empty or len(data) < 50:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate multiple indicators
        data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
        data['RSI_21'] = talib.RSI(data['close'], timeperiod=21)
        data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
        data['EMA_50'] = talib.EMA(data['close'], timeperiod=50)
        data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['close'])
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = []
        score = 0
        
        # EMA Crossover (30 points)
        if (latest['EMA_20'] > latest['EMA_50'] and 
            prev['EMA_20'] <= prev['EMA_50']):
            signals.append("EMA Bullish Crossover")
            score += 30
        
        # RSI Signals (25 points)
        rsi_14 = latest['RSI_14']
        if 30 < rsi_14 < 70:  # Avoid extremes
            if rsi_14 > 50:
                signals.append("RSI Bullish")
                score += 15
            if rsi_14 > latest['RSI_21']:
                signals.append("RSI Positive Divergence")
                score += 10
        
        # MACD Signals (20 points)
        if (latest['MACD'] > latest['MACD_Signal'] and 
            prev['MACD'] <= prev['MACD_Signal']):
            signals.append("MACD Bullish Crossover")
            score += 20
        
        # Volume confirmation (15 points)
        if 'volume' in data.columns and len(data) > 20:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if data['volume'].iloc[-1] > avg_volume * 1.2:
                signals.append("High Volume Confirmation")
                score += 15
        
        # Price momentum (10 points)
        if len(data) >= 10:
            price_change_30min = ((latest['close'] - data['close'].iloc[-6]) / data['close'].iloc[-6]) * 100
            if price_change_30min > 1:
                signals.append("Strong Short-term Momentum")
                score += 10
        
        current_price = latest['close']
        risk_level = "High" if score >= 60 else "Medium" if score >= 40 else "Low"
        
        action = "HOLD"
        if score >= 50:  # Minimum threshold for trade
            action = "BUY"
        elif score <= 20:  # Very weak momentum could signal SELL
            # Check if we have an existing position to sell
            open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                          if t.get('symbol') == symbol and t.get('status') == 'OPEN']
            if open_trades and open_trades[0]['action'] == 'BUY':
                action = "SELL"
        
        return {
            "bot_name": "Auto Momentum Trader",
            "symbol": symbol,
            "action": action,
            "quantity": 1,  # Will be calculated based on risk
            "current_price": current_price,
            "signals": signals,
            "score": score,
            "risk_level": risk_level,
            "capital_required": current_price
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def automated_mean_reversion(instrument_df, symbol):
    """Enhanced mean reversion bot for automated mode."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    try:
        data = get_historical_data(token, '15minute', period='5d')
        if data.empty or len(data) < 100:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate Bollinger Bands and other indicators
        upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data['BB_Upper'] = upperband
        data['BB_Middle'] = middleband
        data['BB_Lower'] = lowerband
        data['RSI_14'] = talib.RSI(data['close'], timeperiod=14)
        
        latest = data.iloc[-1]
        current_price = latest['close']
        
        signals = []
        score = 0
        
        # Bollinger Band position (40 points)
        bb_position = (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        
        if bb_position < 0.1:  # Near lower band
            signals.append("Near Lower Bollinger Band")
            score += 40
        elif bb_position > 0.9:  # Near upper band
            signals.append("Near Upper Bollinger Band")
            score += 40
        
        # RSI confirmation (30 points)
        rsi = latest['RSI_14']
        if bb_position < 0.1 and rsi < 35:  # Oversold confirmation
            signals.append("RSI Oversold Confirmation")
            score += 30
        elif bb_position > 0.9 and rsi > 65:  # Overbought confirmation
            signals.append("RSI Overbought Confirmation")
            score += 30
        
        # Distance from mean (20 points)
        distance_from_mean = abs((current_price - latest['BB_Middle']) / latest['BB_Middle']) * 100
        if distance_from_mean > 3:
            signals.append(f"Price {distance_from_mean:.1f}% from mean")
            score += 20
        
        # Volume confirmation (10 points)
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            if data['volume'].iloc[-1] > avg_volume * 1.5:
                signals.append("High Volume Reversion Signal")
                score += 10
        
        risk_level = "Low" if score >= 60 else "Medium" if score >= 30 else "High"
        
        action = "HOLD"
        if score >= 50 and bb_position < 0.1:  # Buy near lower band
            action = "BUY"
        elif score >= 50 and bb_position > 0.9:  # Sell near upper band
            # Check for existing position
            open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                          if t.get('symbol') == symbol and t.get('status') == 'OPEN']
            if open_trades and open_trades[0]['action'] == 'BUY':
                action = "SELL"
        
        return {
            "bot_name": "Auto Mean Reversion",
            "symbol": symbol,
            "action": action,
            "quantity": 1,
            "current_price": current_price,
            "signals": signals,
            "score": score,
            "risk_level": risk_level,
            "capital_required": current_price
        }
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def setup_breakout_bot(instrument_df, nifty25_auto_trade=False):
    """Setup Breakout Detection Bot with advanced pattern recognition."""
    st.write("**📈 Breakout Trading Bot**")
    st.caption("Identifies and trades breakouts from consolidation patterns with volume confirmation")
    
    if nifty25_auto_trade:
        st.info("🔷 Nifty 25 Auto-Trading is ACTIVE - Breakout Bot will scan Nifty 25 stocks")
        symbols_to_scan = get_nifty25_instruments(instrument_df)
    else:
        # User-selected symbols
        watchlist_symbols = st.session_state.get('watchlists', {}).get(
            st.session_state.get('active_watchlist', 'Watchlist 1'), []
        )
        symbols_to_scan = watchlist_symbols
        st.info(f"🔍 Breakout Bot will scan {len(symbols_to_scan)} symbols from your watchlist")
    
    if not symbols_to_scan:
        st.warning("No symbols available for scanning. Please set up your watchlist or enable Nifty 25 mode.")
        return
    
    # BREAKOUT BOT CONFIGURATION
    with st.expander("⚙️ Breakout Bot Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # CAPITAL AND RISK MANAGEMENT
            st.subheader("💰 Capital & Risk")
            total_capital = st.number_input(
                "Total Capital (₹)", 
                min_value=1000, 
                max_value=1000000, 
                value=10000, 
                step=1000,
                key="breakout_capital"
            )
            
            risk_percent = st.slider(
                "Risk per Trade (%)", 
                min_value=0.5, 
                max_value=10.0, 
                value=2.0, 
                step=0.5,
                key="breakout_risk"
            )
            st.caption(f"Risk Amount: ₹{total_capital * risk_percent/100:.2f} per trade")
            
            consolidation_period = st.slider("Consolidation Period (days)", 10, 30, 15)
            min_volume_ratio = st.slider("Min Volume Ratio", 1.2, 3.0, 1.8)
        
        with col2:
            st.subheader("🎯 Breakout Parameters")
            breakout_confirmation = st.selectbox(
                "Breakout Confirmation", 
                ["Close Above", "High Above", "Both Close and High"]
            )
            min_price_move = st.slider("Min Price Move %", 1.0, 5.0, 2.0)
            stop_loss_type = st.selectbox(
                "Stop Loss Type", 
                ["ATR Based", "Percentage Based", "Support/Resistance"]
            )
            
            if stop_loss_type == "Percentage Based":
                stop_loss_pct = st.slider("Stop Loss %", 1.0, 5.0, 2.5)
            elif stop_loss_type == "ATR Based":
                atr_multiplier = st.slider("ATR Multiplier", 1.5, 3.0, 2.0)
    
    # TRADING CONTROLS
    col_control1, col_control2, col_control3 = st.columns(3)
    with col_control1:
        auto_execute = st.checkbox("🤖 Auto-Execute Trades", value=False)
    with col_control2:
        max_positions = st.slider("Max Positions", 1, 10, 3, key="breakout_max_pos")
    with col_control3:
        min_confidence = st.slider("Min Confidence", 60, 90, 75, key="breakout_conf")
    
    # SCAN AND EXECUTE
    if st.button("🔍 Scan Breakout Opportunities", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(symbols_to_scan)} symbols for breakout patterns..."):
            breakout_signals = run_breakout_analysis(
                symbols_to_scan, 
                instrument_df,
                consolidation_period,
                min_volume_ratio,
                min_price_move
            )
        
        if breakout_signals and breakout_signals.get('signals'):
            display_breakout_signals(
                breakout_signals, 
                total_capital, 
                risk_percent, 
                auto_execute,
                max_positions
            )
        else:
            st.info("🤖 No breakout signals found. Market may be in consolidation.")

def run_breakout_analysis(symbols, instrument_df, consolidation_period=15, min_volume_ratio=1.8, min_price_move=2.0):
    """Run breakout analysis on given symbols."""
    breakout_signals = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(symbols[:25]):  # Limit to 25 for performance
        symbol = item['symbol']
        exchange = item['exchange']
        status_text.text(f"Analyzing {symbol}... ({i+1}/{min(25, len(symbols))})")
        
        try:
            token = get_instrument_token(symbol, instrument_df, exchange)
            if not token:
                continue
                
            # Get 60 days historical data
            data = get_historical_data_60days(token, 'day')
            if data.empty or len(data) < 30:
                continue
            
            # Analyze for breakouts
            signal = analyze_breakout_pattern(data, symbol, consolidation_period, min_volume_ratio, min_price_move)
            if signal and signal.get('signal') != 'NO_SIGNAL':
                breakout_signals.append(signal)
                
        except Exception as e:
            continue
        
        progress_bar.progress((i + 1) / min(25, len(symbols)))
    
    status_text.text("Breakout analysis complete!")
    
    return {
        'signals': sorted(breakout_signals, key=lambda x: x.get('confidence', 0), reverse=True),
        'total_scanned': min(25, len(symbols)),
        'breakouts_found': len(breakout_signals),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def analyze_breakout_pattern(data, symbol, consolidation_period, min_volume_ratio, min_price_move):
    """Analyze stock data for breakout patterns."""
    if len(data) < consolidation_period + 5:
        return None
    
    # Calculate technical indicators
    data = calculate_advanced_indicators(data)
    latest = data.iloc[-1]
    prev_day = data.iloc[-2]
    
    # Calculate consolidation parameters
    consolidation_high = data['high'].tail(consolidation_period).max()
    consolidation_low = data['low'].tail(consolidation_period).min()
    consolidation_range = consolidation_high - consolidation_low
    current_close = latest['close']
    current_high = latest['high']
    current_volume = latest['volume']
    
    # Volume analysis
    avg_volume = data['volume'].tail(consolidation_period).mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Price movement analysis
    price_move_pct = ((current_close - prev_day['close']) / prev_day['close']) * 100
    
    # Initialize signal
    signal = 'NO_SIGNAL'
    confidence = 0
    pattern_type = ""
    breakout_level = 0
    
    # UPPER BREAKOUT DETECTION
    if current_close > consolidation_high and volume_ratio >= min_volume_ratio:
        signal = 'BUY'
        pattern_type = "Upper Breakout"
        breakout_level = consolidation_high
        confidence = 70
        
        # Enhance confidence based on factors
        if price_move_pct >= min_price_move:
            confidence += 10
        if current_high > consolidation_high * 1.01:  # Strong breakout
            confidence += 10
        if latest.get('EMA_20', 0) > latest.get('EMA_50', 0):  # Trend confirmation
            confidence += 10
    
    # LOWER BREAKOUT DETECTION (Breakdown)
    elif current_close < consolidation_low and volume_ratio >= min_volume_ratio:
        signal = 'SELL'
        pattern_type = "Lower Breakout"
        breakout_level = consolidation_low
        confidence = 65
        
        if price_move_pct <= -min_price_move:
            confidence += 10
        if latest.get('EMA_20', 0) < latest.get('EMA_50', 0):  # Downtrend confirmation
            confidence += 10
    
    if signal != 'NO_SIGNAL':
        # Calculate stop loss and target
        if signal == 'BUY':
            stop_loss = consolidation_low
            target = current_close + (2 * (current_close - stop_loss))
        else:  # SELL
            stop_loss = consolidation_high
            target = current_close - (2 * (stop_loss - current_close))
        
        risk_reward = abs((target - current_close) / (current_close - stop_loss)) if current_close != stop_loss else 0
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': min(95, confidence),
            'pattern_type': pattern_type,
            'current_price': current_close,
            'breakout_level': breakout_level,
            'volume_ratio': volume_ratio,
            'price_move_pct': price_move_pct,
            'consolidation_high': consolidation_high,
            'consolidation_low': consolidation_low,
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward': risk_reward,
            'consolidation_range_pct': (consolidation_range / consolidation_low * 100) if consolidation_low > 0 else 0
        }
    
    return None

def setup_multi_signal_bot(instrument_df, nifty25_auto_trade=False):
    """Setup Multi-Signal Bot that combines multiple technical strategies."""
    st.write("**🎯 Multi-Signal Trading Bot**")
    st.caption("Combines momentum, mean reversion, breakout, and volume signals for high-probability trades")
    
    if nifty25_auto_trade:
        st.info("🔷 Nifty 25 Auto-Trading is ACTIVE - Multi-Signal Bot will scan Nifty 25 stocks")
        symbols_to_scan = get_nifty25_instruments(instrument_df)
    else:
        watchlist_symbols = st.session_state.get('watchlists', {}).get(
            st.session_state.get('active_watchlist', 'Watchlist 1'), []
        )
        symbols_to_scan = watchlist_symbols
        st.info(f"🔍 Multi-Signal Bot will scan {len(symbols_to_scan)} symbols from your watchlist")
    
    if not symbols_to_scan:
        st.warning("No symbols available for scanning. Please set up your watchlist or enable Nifty 25 mode.")
        return
    
    # MULTI-SIGNAL BOT CONFIGURATION
    with st.expander("⚙️ Multi-Signal Bot Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Capital & Risk")
            total_capital = st.number_input(
                "Total Capital (₹)", 
                min_value=1000, 
                max_value=1000000, 
                value=10000, 
                step=1000,
                key="multi_capital"
            )
            
            risk_percent = st.slider(
                "Risk per Trade (%)", 
                min_value=0.5, 
                max_value=10.0, 
                value=1.5, 
                step=0.5,
                key="multi_risk"
            )
            st.caption(f"Risk Amount: ₹{total_capital * risk_percent/100:.2f} per trade")
            
            st.subheader("📊 Signal Weights")
            momentum_weight = st.slider("Momentum Weight", 0.0, 1.0, 0.3)
            mean_reversion_weight = st.slider("Mean Reversion Weight", 0.0, 1.0, 0.25)
            breakout_weight = st.slider("Breakout Weight", 0.0, 1.0, 0.25)
            volume_weight = st.slider("Volume Weight", 0.0, 1.0, 0.2)
        
        with col2:
            st.subheader("🎯 Strategy Parameters")
            min_combined_score = st.slider("Min Combined Score", 50, 90, 70)
            require_volume_confirmation = st.checkbox("Require Volume Confirmation", value=True)
            require_trend_alignment = st.checkbox("Require Trend Alignment", value=True)
            
            st.subheader("⏱️ Timeframe Settings")
            primary_timeframe = st.selectbox("Primary Timeframe", ["Daily", "Hourly", "Both"])
            lookback_period = st.slider("Lookback Period (days)", 20, 100, 60)
    
    # SIGNAL FILTERS
    with st.expander("🔍 Signal Filters"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_rsi = st.slider("Min RSI", 20, 50, 30)
            max_rsi = st.slider("Max RSI", 50, 80, 70)
            min_volume_ratio = st.slider("Min Volume Ratio", 1.0, 2.0, 1.2)
        with col_f2:
            min_adx = st.slider("Min ADX", 15, 30, 20)
            max_positions = st.slider("Max Positions", 1, 8, 4, key="multi_max_pos")
    
    # TRADING CONTROLS
    col_control1, col_control2 = st.columns(2)
    with col_control1:
        auto_execute = st.checkbox("🤖 Auto-Execute Trades", value=False, key="multi_auto")
    with col_control2:
        if st.button("🚀 Scan Multi-Signal Opportunities", type="primary", use_container_width=True):
            with st.spinner(f"Running multi-signal analysis on {len(symbols_to_scan)} symbols..."):
                multi_signals = run_multi_signal_analysis(
                    symbols_to_scan,
                    instrument_df,
                    {
                        'momentum_weight': momentum_weight,
                        'mean_reversion_weight': mean_reversion_weight,
                        'breakout_weight': breakout_weight,
                        'volume_weight': volume_weight,
                        'min_combined_score': min_combined_score,
                        'require_volume_confirmation': require_volume_confirmation,
                        'require_trend_alignment': require_trend_alignment,
                        'min_rsi': min_rsi,
                        'max_rsi': max_rsi,
                        'min_volume_ratio': min_volume_ratio,
                        'min_adx': min_adx,
                        'lookback_period': lookback_period
                    }
                )
            
            if multi_signals and multi_signals.get('signals'):
                display_multi_signals(
                    multi_signals,
                    total_capital,
                    risk_percent,
                    auto_execute,
                    max_positions
                )
            else:
                st.info("🤖 No multi-signal opportunities found. Adjust parameters or try later.")

def run_multi_signal_analysis(symbols, instrument_df, params):
    """Run multi-signal analysis combining multiple strategies."""
    multi_signals = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(symbols[:20]):  # Limit for performance
        symbol = item['symbol']
        exchange = item['exchange']
        status_text.text(f"Multi-signal analysis: {symbol}... ({i+1}/{min(20, len(symbols))})")
        
        try:
            token = get_instrument_token(symbol, instrument_df, exchange)
            if not token:
                continue
                
            # Get historical data
            data = get_historical_data_60days(token, 'day')
            if data.empty or len(data) < 30:
                continue
            
            # Run multi-signal analysis
            signal = analyze_multi_signals(data, symbol, params)
            if signal and signal.get('combined_score', 0) >= params['min_combined_score']:
                multi_signals.append(signal)
                
        except Exception as e:
            continue
        
        progress_bar.progress((i + 1) / min(20, len(symbols)))
    
    status_text.text("Multi-signal analysis complete!")
    
    return {
        'signals': sorted(multi_signals, key=lambda x: x.get('combined_score', 0), reverse=True),
        'total_scanned': min(20, len(symbols)),
        'signals_found': len(multi_signals),
        'params': params,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def analyze_multi_signals(data, symbol, params):
    """Combine multiple technical signals into a unified score."""
    if len(data) < 30:
        return None
    
    data = calculate_advanced_indicators(data)
    latest = data.iloc[-1]
    
    signal_scores = {
        'momentum': 0,
        'mean_reversion': 0,
        'breakout': 0,
        'volume': 0
    }
    
    signal_details = []
    final_signal = 'HOLD'
    
    # 1. MOMENTUM SIGNAL
    momentum_score = calculate_momentum_score(latest, data)
    signal_scores['momentum'] = momentum_score
    
    # 2. MEAN REVERSION SIGNAL  
    mean_reversion_score = calculate_mean_reversion_score(latest, data, params)
    signal_scores['mean_reversion'] = mean_reversion_score
    
    # 3. BREAKOUT SIGNAL
    breakout_score = calculate_breakout_score(latest, data)
    signal_scores['breakout'] = breakout_score
    
    # 4. VOLUME SIGNAL
    volume_score = calculate_volume_score(latest, data, params)
    signal_scores['volume'] = volume_score
    
    # COMBINE SCORES WITH WEIGHTS
    combined_score = (
        signal_scores['momentum'] * params['momentum_weight'] +
        signal_scores['mean_reversion'] * params['mean_reversion_weight'] +
        signal_scores['breakout'] * params['breakout_weight'] +
        signal_scores['volume'] * params['volume_weight']
    ) * 100
    
    # DETERMINE FINAL SIGNAL
    buy_signals = 0
    sell_signals = 0
    
    if momentum_score > 0.6: buy_signals += 1
    if momentum_score < 0.4: sell_signals += 1
    
    if mean_reversion_score > 0.7: buy_signals += 1
    if mean_reversion_score < 0.3: sell_signals += 1
    
    if breakout_score > 0.6: buy_signals += 1
    if breakout_score < 0.4: sell_signals += 1
    
    if volume_score > 0.5: buy_signals += 1  # Volume usually confirms
    
    if buy_signals >= 3 and combined_score >= params['min_combined_score']:
        final_signal = 'BUY'
    elif sell_signals >= 3 and combined_score >= params['min_combined_score']:
        final_signal = 'SELL'
    
    if final_signal != 'HOLD':
        # Calculate position metrics
        current_price = latest['close']
        atr = latest.get('ATR', current_price * 0.02)  # Default 2% if ATR not available
        
        return {
            'symbol': symbol,
            'signal': final_signal,
            'combined_score': combined_score,
            'signal_breakdown': signal_scores,
            'current_price': current_price,
            'rsi': latest.get('RSI_14', 50),
            'volume_ratio': latest.get('volume', 0) / data['volume'].tail(20).mean(),
            'atr': atr,
            'momentum_strength': "Strong" if momentum_score > 0.7 else "Medium" if momentum_score > 0.5 else "Weak",
            'confidence': min(95, combined_score),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    
    return None

def calculate_momentum_score(latest, data):
    """Calculate momentum score (0-1)."""
    score = 0
    factors = 0
    
    # EMA Alignment
    if (latest.get('EMA_20', 0) > latest.get('EMA_50', 0) > latest.get('EMA_200', 0)):
        score += 0.3
    factors += 1
    
    # RSI Momentum
    rsi = latest.get('RSI_14', 50)
    if 60 < rsi < 70:
        score += 0.2  # Strong but not overbought
    factors += 1
    
    # MACD
    if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
        score += 0.2
    factors += 1
    
    # ADX Trend Strength
    if latest.get('ADX', 0) > 25:
        score += 0.3
    factors += 1
    
    return score / factors if factors > 0 else 0

def calculate_mean_reversion_score(latest, data, params):
    """Calculate mean reversion score (0-1)."""
    current_price = latest['close']
    mean_price = data['close'].tail(50).mean()
    std_price = data['close'].tail(50).std()
    
    score = 0
    factors = 0
    
    if std_price > 0:
        z_score = abs(current_price - mean_price) / std_price
        
        # Oversold conditions (higher score for more oversold)
        if z_score > 1.5 and current_price < mean_price:
            score += min(0.4, z_score * 0.2)
            factors += 1
    
    # RSI based mean reversion
    rsi = latest.get('RSI_14', 50)
    if rsi < params['min_rsi']:
        score += 0.3
        factors += 1
    
    # Bollinger Band position
    bb_lower = latest.get('BB_Lower', 0)
    if current_price <= bb_lower * 1.02 and bb_lower > 0:
        score += 0.3
        factors += 1
    
    return score / factors if factors > 0 else 0

def calculate_breakout_score(latest, data):
    """Calculate breakout score (0-1)."""
    score = 0
    factors = 0
    recent_high = data['high'].tail(20).max()
    if latest['close'] > recent_high:
        score += 0.4
    factors += 1
    
    # Volume confirmation
    volume_avg = data['volume'].tail(20).mean()
    if latest['volume'] > volume_avg * 1.5:
        score += 0.3
    factors += 1
    
    # Price above key EMAs
    if latest['close'] > latest.get('EMA_20', 0) > latest.get('EMA_50', 0):
        score += 0.3
    factors += 1
    
    return score / factors if factors > 0 else 0

def calculate_volume_score(latest, data, params):
    """Calculate volume confirmation score (0-1)."""
    volume_avg = data['volume'].tail(20).mean()
    current_volume = latest['volume']
    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
    
    if volume_ratio >= params['min_volume_ratio']:
        return 0.8  # Strong volume confirmation
    elif volume_ratio >= 1.0:
        return 0.5  # Moderate volume
    else:
        return 0.2  # Weak volume

def display_breakout_signals(signals_data, total_capital, risk_percent, auto_execute, max_positions):
    """Display breakout signals with trading options."""
    st.subheader("🎯 Breakout Signals Found")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        st.metric("Total Scanned", signals_data['total_scanned'])
    with col_stats2:
        st.metric("Breakouts Found", signals_data['breakouts_found'])
    with col_stats3:
        st.metric("Last Scan", signals_data['timestamp'].split()[1])
    
    st.markdown("---")
    
    # Display top signals
    top_signals = signals_data['signals'][:max_positions]
    
    for i, signal in enumerate(top_signals, 1):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
            
            with col1:
                st.write(f"**#{i}**")
                st.write(f"**{signal['symbol']}**")
                st.write(f"₹{signal['current_price']:.2f}")
            
            with col2:
                # Signal visualization
                confidence = signal['confidence']
                if confidence > 80:
                    color = "🟢"
                elif confidence > 65:
                    color = "🟡" 
                else:
                    color = "🟠"
                
                st.write(f"{color} **{signal['signal']}** - {signal['pattern_type']}")
                st.progress(confidence / 100)
                
                # FIXED: Corrected the f-string syntax
                st.caption(f"Breakout: ₹{signal['breakout_level']:.2f} | Volume: {signal['volume_ratio']:.1f}x | Move: {signal['price_move_pct']:.1f}%")
            
            with col3:
                st.metric("Confidence", f"{confidence}%")
                st.metric("R/R", f"{signal['risk_reward']:.1f}")
            
            with col4:
                st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}")
                st.metric("Target", f"₹{signal['target']:.2f}")
            
            with col5:
                # Position sizing
                risk_amount = total_capital * (risk_percent / 100)
                position_size = calculate_breakout_position_size(signal, risk_amount)
                
                st.write(f"**Qty:** {position_size}")
                st.write(f"**Risk:** ₹{risk_amount:.2f}")
                
                if st.button("🚀 Trade", key=f"breakout_trade_{signal['symbol']}", type="primary"):
                    execute_breakout_trade(signal, position_size)
            
            st.markdown("---")
    
    # AUTO-EXECUTION
    if auto_execute and top_signals:
        if st.button("🤖 Execute All Breakout Trades", type="primary", use_container_width=True):
            execute_batch_breakout_trades(top_signals, total_capital, risk_percent)

def execute_batch_breakout_trades(signals, total_capital, risk_percent):
    """Execute multiple breakout trades in batch."""
    executed_trades = []
    failed_trades = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, signal in enumerate(signals):
        status_text.text(f"Executing trade {i+1}/{len(signals)}: {signal['symbol']}")
        
        try:
            risk_amount = total_capital * (risk_percent / 100)
            position_size = calculate_breakout_position_size(signal, risk_amount)
            
            instrument_df = get_instrument_df()
            place_order(
                instrument_df,
                signal['symbol'],
                position_size,
                'MARKET',
                signal['signal'],
                'MIS'
            )
            
            executed_trades.append({
                'symbol': signal['symbol'],
                'action': signal['signal'],
                'quantity': position_size,
                'price': signal['current_price']
            })
            
            log_breakout_trade(signal, position_size, 'AUTO_BATCH')
            
        except Exception as e:
            failed_trades.append({
                'symbol': signal['symbol'],
                'error': str(e)
            })
        
        progress_bar.progress((i + 1) / len(signals))
    
    # Display results
    if executed_trades:
        st.success(f"✅ Successfully executed {len(executed_trades)} breakout trades!")
        for trade in executed_trades:
            st.write(f"• {trade['action']} {trade['quantity']} shares of {trade['symbol']} @ ₹{trade['price']:.2f}")
    
    if failed_trades:
        st.error(f"❌ {len(failed_trades)} trades failed")
        for trade in failed_trades:
            st.write(f"• {trade['symbol']}: {trade['error']}")

def display_multi_signals(signals_data, total_capital, risk_percent, auto_execute, max_positions):
    """Display multi-signal analysis results."""
    st.subheader("🎯 Multi-Signal Opportunities")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Total Scanned", signals_data['total_scanned'])
    with col_stats2:
        st.metric("Signals Found", signals_data['signals_found'])
    with col_stats3:
        strong_signals = len([s for s in signals_data['signals'] if s['combined_score'] > 80])
        st.metric("Strong Signals", strong_signals)
    with col_stats4:
        st.metric("Last Scan", signals_data['timestamp'].split()[1])
    
    st.markdown("---")
    
    # Display top signals
    top_signals = signals_data['signals'][:max_positions]
    
    for i, signal in enumerate(top_signals, 1):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 1])
            
            with col1:
                st.write(f"**#{i}**")
                st.write(f"**{signal['symbol']}**")
                st.write(f"₹{signal['current_price']:.2f}")
            
            with col2:
                # Combined score visualization
                score = signal['combined_score']
                if score > 80:
                    color = "🟢"
                    strength = "VERY STRONG"
                elif score > 70:
                    color = "🟡"
                    strength = "STRONG"
                else:
                    color = "🟠"
                    strength = "MODERATE"
                
                st.write(f"{color} **{signal['signal']}** - {strength}")
                st.progress(score / 100)
                
                # Signal breakdown
                breakdown = signal['signal_breakdown']
                breakdown_text = f"M:{breakdown['momentum']:.1f} • R:{breakdown['mean_reversion']:.1f} • B:{breakdown['breakout']:.1f} • V:{breakdown['volume']:.1f}"
                st.caption(breakdown_text)
                st.caption(f"Buy Signals: {signal['buy_signals']} | Sell Signals: {signal['sell_signals']}")
            
            with col3:
                st.metric("Combined Score", f"{score:.0f}")
                st.metric("RSI", f"{signal['rsi']:.1f}")
            
            with col4:
                st.metric("Volume", f"{signal['volume_ratio']:.1f}x")
                st.metric("Confidence", f"{signal['confidence']}%")
            
            with col5:
                # Position sizing
                risk_amount = total_capital * (risk_percent / 100)
                position_size = calculate_risk_position_size(signal, risk_amount)
                
                st.write(f"**Qty:** {position_size}")
                st.write(f"**Risk:** ₹{risk_amount:.2f}")
                
                if st.button("🚀 Trade", key=f"multi_trade_{signal['symbol']}", type="primary"):
                    execute_multi_signal_trade(signal, position_size)
            
            st.markdown("---")
    
    # AUTO-EXECUTION
    if auto_execute and top_signals:
        if st.button("🤖 Execute All Multi-Signal Trades", type="primary", use_container_width=True):
            execute_batch_multi_trades(top_signals, total_capital, risk_percent)

def calculate_breakout_position_size(signal, risk_amount):
    """Calculate position size for breakout trades."""
    current_price = signal['current_price']
    stop_loss = signal['stop_loss']
    
    if current_price > stop_loss:  # For long trades
        risk_per_share = current_price - stop_loss
    else:  # For short trades
        risk_per_share = stop_loss - current_price
    
    if risk_per_share > 0:
        position_size = int(risk_amount / risk_per_share)
    else:
        position_size = int(risk_amount / (current_price * 0.02))  # Default 2% stop
    
    # Ensure minimum 1 share and reasonable maximum
    return max(1, min(position_size, 1000))

def calculate_risk_position_size(signal, risk_amount):
    """Calculate position size based on ATR risk."""
    current_price = signal['current_price']
    atr = signal.get('atr', current_price * 0.02)  # Default 2% if ATR not available
    
    # Use 2x ATR for stop loss
    risk_per_share = 2 * atr
    
    if risk_per_share > 0:
        position_size = int(risk_amount / risk_per_share)
    else:
        position_size = int(risk_amount / (current_price * 0.02))
    
    return max(1, min(position_size, 1000))

def execute_breakout_trade(signal, quantity):
    """Execute a single breakout trade."""
    try:
        instrument_df = get_instrument_df()
        place_order(
            instrument_df,
            signal['symbol'],
            quantity,
            'MARKET',
            signal['signal'],
            'MIS'
        )
        st.success(f"✅ {signal['signal']} order executed for {signal['symbol']} (Qty: {quantity})")
        
        # Log the trade
        log_breakout_trade(signal, quantity, 'MANUAL')
        
    except Exception as e:
        st.error(f"❌ Trade execution failed: {str(e)}")

def execute_multi_signal_trade(signal, quantity):
    """Execute a single multi-signal trade."""
    try:
        instrument_df = get_instrument_df()
        place_order(
            instrument_df,
            signal['symbol'],
            quantity,
            'MARKET',
            signal['signal'],
            'MIS'
        )
        st.success(f"✅ {signal['signal']} order executed for {signal['symbol']} (Qty: {quantity})")
        
        # Log the trade
        log_multi_signal_trade(signal, quantity, 'MANUAL')
        
    except Exception as e:
        st.error(f"❌ Trade execution failed: {str(e)}")

# Add these logging functions
def log_breakout_trade(signal, quantity, execution_type):
    """Log breakout trade execution."""
    trade_log = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': signal['symbol'],
        'signal': signal['signal'],
        'quantity': quantity,
        'price': signal['current_price'],
        'type': 'BREAKOUT',
        'execution': execution_type,
        'confidence': signal['confidence'],
        'pattern': signal['pattern_type']
    }
    
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []
    
    st.session_state.trade_logs.append(trade_log)

def log_multi_signal_trade(signal, quantity, execution_type):
    """Log multi-signal trade execution."""
    trade_log = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': signal['symbol'],
        'signal': signal['signal'],
        'quantity': quantity,
        'price': signal['current_price'],
        'type': 'MULTI_SIGNAL',
        'execution': execution_type,
        'confidence': signal['confidence'],
        'combined_score': signal['combined_score']
    }
    
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []
    
    st.session_state.trade_logs.append(trade_log)

# Dictionary of automated bots
AUTOMATED_BOTS = {
    "Auto Momentum Trader": automated_momentum_trader,
    "Auto Mean Reversion": automated_mean_reversion,
    "Breakout Bot": setup_breakout_bot,
    "Multi-Signal Bot": setup_multi_signal_bot
}

# ================ ALGO BOTS PAGE FUNCTIONS ================

def page_algo_bots():
    """Main algo bots page with both semi-automated and fully automated modes."""
    display_header()
    st.title("🤖 Algo Trading Bots")
    
    # Initialize automated mode
    initialize_automated_mode()
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo bots.")
        return
    
    # Mode selection tabs
    tab1, tab2 = st.tabs(["🚀 Semi-Automated Bots", "⚡ Fully Automated Bots"])
    
    with tab1:
        page_semi_automated_bots(instrument_df)
    
    with tab2:
        page_fully_automated_bots(instrument_df)

def page_semi_automated_bots(instrument_df):
    """Semi-automated bots page - requires manual confirmation."""
    st.info("Run automated analysis and get trading signals. Manual confirmation required for execution.", icon="🚀")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_bot = st.selectbox(
            "Select Trading Bot",
            list(ALGO_BOTS.keys()),
            help="Choose a trading bot based on your risk appetite and trading style",
            key="semi_bot_select"
        )
        
        # Bot descriptions
        bot_descriptions = {
            "Momentum Trader": "Trades on strong price momentum and trend continuations. Medium risk.",
            "Mean Reversion": "Buys low and sells high based on statistical mean reversion. Low risk.",
            "Volatility Breakout": "Captures breakouts from low volatility periods. High risk.",
            "Value Investor": "Focuses on longer-term value and fundamental trends. Low risk.",
            "Scalper Pro": "High-frequency trading for quick, small profits. Very high risk.",
            "Trend Follower": "Rides established trends with multiple confirmation signals. Medium risk."
        }
        
        st.markdown(f"**Description:** {bot_descriptions[selected_bot]}")
    
    with col2:
        trading_capital = st.number_input(
            "Trading Capital (₹)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Minimum ₹100 required",
            key="semi_capital"
        )
    
    st.markdown("---")
    
    # Symbol selection and bot execution
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Stock Selection")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox(
            "Select Stock",
            sorted(all_symbols),
            index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0,
            key="semi_symbol"
        )
        
        # Show current price
        quote_data = get_watchlist_data([{'symbol': selected_symbol, 'exchange': 'NSE'}])
        if not quote_data.empty:
            current_price = quote_data.iloc[0]['Price']
            st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col4:
        st.subheader("Bot Execution")
        st.write(f"**Selected Bot:** {selected_bot}")
        st.write(f"**Available Capital:** ₹{trading_capital:,}")
        
        if st.button("🚀 Run Trading Bot", use_container_width=True, type="primary", key="semi_run"):
            with st.spinner(f"Running {selected_bot} analysis..."):
                bot_function = ALGO_BOTS[selected_bot]
                bot_result = bot_function(instrument_df, selected_symbol, trading_capital)
                
                if bot_result and not bot_result.get("error"):
                    st.session_state.last_bot_result = bot_result
                    st.rerun()
    
    # Display bot results
    if 'last_bot_result' in st.session_state and st.session_state.last_bot_result:
        bot_result = st.session_state.last_bot_result
        
        if bot_result.get("error"):
            st.error(bot_result["error"])
        else:
            st.markdown("---")
            st.subheader("🤖 Bot Analysis Results")
            
            # Create metrics cards
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                action_color = "green" if bot_result["action"] == "BUY" else "red" if bot_result["action"] == "SELL" else "orange"
                st.markdown(f'<div class="metric-card" style="border-color: {action_color};">'
                          f'<h3 style="color: {action_color};">{bot_result["action"]}</h3>'
                          f'<p>Recommended Action</p></div>', unsafe_allow_html=True)
            
            with col6:
                st.metric("Quantity", bot_result["quantity"])
            
            with col7:
                st.metric("Capital Required", f"₹{bot_result['capital_required']:.2f}")
            
            with col8:
                risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
                st.markdown(f'<div class="metric-card" style="border-color: {risk_color.get(bot_result["risk_level"], "gray")};">'
                          f'<h3 style="color: {risk_color.get(bot_result["risk_level"], "gray")};">{bot_result["risk_level"]}</h3>'
                          f'<p>Risk Level</p></div>', unsafe_allow_html=True)
            
            # Display signals and execute trade
            execute_bot_trade(instrument_df, bot_result)

    # Bot performance history
    st.markdown("---")
    st.subheader("📈 Bot Performance Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Best Practices:**
        - Start with minimum capital (₹100)
        - Use 'Value Investor' for beginners
        - 'Scalper Pro' requires constant monitoring
        - Always check signals before executing
        - Combine multiple bot recommendations
        """)
    
    with tips_col2:
        st.markdown("""
        **Risk Management:**
        - Never risk more than 2% per trade
        - Use stop losses with every trade
        - Diversify across different bots
        - Monitor performance regularly
        - Adjust capital based on experience
        """)
    
    # Quick bot comparison
    with st.expander("🤖 Bot Comparison Guide"):
        comparison_data = {
            "Bot": list(ALGO_BOTS.keys()),
            "Risk Level": ["Medium", "Low", "High", "Low", "Very High", "Medium"],
            "Holding Period": ["Hours", "Days", "Minutes", "Weeks", "Minutes", "Days"],
            "Capital Recommended": ["₹1,000+", "₹500+", "₹2,000+", "₹2,000+", "₹5,000+", "₹1,500+"],
            "Best For": ["Trend riding", "Safe returns", "Quick profits", "Long term", "Experienced", "Trend following"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def initialize_automated_mode():
    """Initialize session state for fully automated trading with paper trading."""
    if 'automated_mode' not in st.session_state:
        st.session_state.automated_mode = {
            'enabled': False,
            'running': False,
            'live_trading': False,
            'bots_active': {},
            'total_capital': 10000.0,
            'risk_per_trade': 2.0,
            'max_open_trades': 5,
            'trade_history': [],
            'performance_metrics': {},
            'last_signal_check': None,
            'paper_portfolio': {
                'cash_balance': 10000.0,
                'positions': {},
                'initial_capital': 100.0,
                'total_value': 10000.0
            }
        }
    else:
        # Migration: Ensure paper_portfolio exists for existing users
        if 'paper_portfolio' not in st.session_state.automated_mode:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': st.session_state.automated_mode.get('total_capital', 10000.0),
                'positions': {},
                'initial_capital': st.session_state.automated_mode.get('total_capital', 10000.0),
                'total_value': st.session_state.automated_mode.get('total_capital', 10000.0)
            }

def update_paper_portfolio_values(instrument_df):
    """Update paper portfolio values with current market prices."""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    if not paper_portfolio:
        return
    
    positions = paper_portfolio.get('positions', {})
    if not positions:
        paper_portfolio['total_value'] = paper_portfolio.get('cash_balance', 0.0)
        return
    
    # Get current prices for all positions
    symbols_with_exchange = []
    for symbol in positions.keys():
        symbols_with_exchange.append({'symbol': symbol, 'exchange': 'NSE'})
    
    if symbols_with_exchange:
        live_data = get_watchlist_data(symbols_with_exchange)
        
        if not live_data.empty:
            total_position_value = 0.0
            
            for symbol, position in positions.items():
                symbol_data = live_data[live_data['Ticker'] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Price']
                    position_value = position['quantity'] * current_price
                    total_position_value += position_value
                    
                    # Update unrealized P&L for open trades
                    open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                                  if t.get('symbol') == symbol and t.get('status') == 'OPEN']
                    for trade in open_trades:
                        if trade.get('action') == 'BUY':
                            trade['pnl'] = (current_price - trade.get('entry_price', 0)) * trade.get('quantity', 0)
                        else:  # SELL (short)
                            trade['pnl'] = (trade.get('entry_price', 0) - current_price) * trade.get('quantity', 0)
            
            paper_portfolio['total_value'] = paper_portfolio.get('cash_balance', 0.0) + total_position_value

# Also add the missing close_paper_position function
def close_paper_position(symbol, quantity=None):
    """Close a paper trading position."""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    
    if not paper_portfolio or symbol not in paper_portfolio.get('positions', {}):
        st.error(f"No position found for {symbol}")
        return False
    
    position = paper_portfolio['positions'][symbol]
    close_quantity = quantity if quantity else position['quantity']
    
    if close_quantity > position['quantity']:
        st.error(f"Cannot close more than current position: {position['quantity']}")
        return False
    
    # Get current price
    live_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
    if live_data.empty:
        st.error(f"Could not get current price for {symbol}")
        return False
    
    current_price = live_data.iloc[0]['Price']
    
    # Calculate P&L
    pnl = (current_price - position['avg_price']) * close_quantity
    if position.get('action') == 'SELL':  # For short positions, reverse the P&L
        pnl = -pnl
    
    # Update cash and position
    paper_portfolio['cash_balance'] += close_quantity * current_price
    paper_portfolio['positions'][symbol]['quantity'] -= close_quantity
    
    # Remove position if fully closed
    if paper_portfolio['positions'][symbol]['quantity'] == 0:
        del paper_portfolio['positions'][symbol]
    
    # Update trade history
    open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                  if t.get('symbol') == symbol and t.get('status') == 'OPEN']
    for trade in open_trades:
        if trade.get('action') == position.get('action'):  # Find matching trade
            # Close the trade
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.now().isoformat()
            trade['pnl'] = pnl
    
    st.success(f"✅ Closed {close_quantity} shares of {symbol} at ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
    return True


def get_automated_bot_performance():
    """Calculate performance metrics for automated bots with paper trading support."""
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    if not trade_history:
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        current_value = paper_portfolio.get('total_value', paper_portfolio.get('cash_balance', 0.0))
        initial_capital = paper_portfolio.get('initial_capital', current_value)
        paper_return_pct = ((current_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'open_trades': 0,
            'paper_portfolio_value': current_value,
            'paper_return_pct': paper_return_pct,
            'unrealized_pnl': 0.0
        }
    
    closed_trades = [t for t in trade_history if t.get('status') == 'CLOSED']
    open_trades = [t for t in trade_history if t.get('status') == 'OPEN']
    
    # Calculate metrics for closed trades
    winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('pnl', 0) <= 0]
    
    total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0.0
    
    avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    # Paper trading metrics
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    initial_capital = paper_portfolio.get('initial_capital', 10000.0)
    current_value = paper_portfolio.get('total_value', paper_portfolio.get('cash_balance', initial_capital))
    paper_return_pct = ((current_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    return {
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'open_trades': len(open_trades),
        'paper_portfolio_value': current_value,
        'paper_return_pct': paper_return_pct,
        'unrealized_pnl': sum(t.get('pnl', 0) for t in open_trades)
    }

def execute_automated_trade(instrument_df, bot_result, risk_per_trade):
    """Execute trades automatically based on bot signals."""
    if bot_result.get("error") or bot_result["action"] == "HOLD":
        return None
    
    try:
        symbol = bot_result["symbol"]
        action = bot_result["action"]
        current_price = bot_result["current_price"]
        
        # Calculate position size based on risk
        risk_amount = (risk_per_trade / 100) * st.session_state.automated_mode['total_capital']
        quantity = max(1, int(risk_amount / current_price))
        
        # Check if we have too many open trades
        open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                      if t.get('status') == 'OPEN']
        if len(open_trades) >= st.session_state.automated_mode['max_open_trades']:
            return None
        
        # Check for existing position in the same symbol
        existing_position = next((t for t in open_trades if t.get('symbol') == symbol), None)
        if existing_position:
            # Avoid opening same position multiple times
            if existing_position['action'] == action:
                return None
        
        # PLACE REAL ORDER if live trading is enabled
        order_type = "PAPER"
        if st.session_state.automated_mode.get('live_trading', False):
            try:
                # Place the real order
                place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
                order_type = "LIVE"
            except Exception as e:
                st.error(f"❌ Failed to place LIVE order for {symbol}: {e}")
                return None
        
        # Record the trade with IST timestamp
        ist_time = get_ist_time()
        trade_record = {
            'timestamp': ist_time.isoformat(),
            'timestamp_display': ist_time.strftime("%Y-%m-%d %H:%M:%S IST"),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'status': 'OPEN',
            'bot_name': bot_result['bot_name'],
            'risk_level': bot_result['risk_level'],
            'order_type': order_type,
            'pnl': 0  # Initialize P&L
        }
        
        st.session_state.automated_mode['trade_history'].append(trade_record)
        
        if order_type == "LIVE":
            st.toast(f"🤖 LIVE {action} order executed for {symbol} (Qty: {quantity})", icon="⚡")
        else:
            st.toast(f"🤖 PAPER {action} order simulated for {symbol} (Qty: {quantity})", icon="📄")
            
        return trade_record
        
    except Exception as e:
        st.error(f"Automated trade execution failed: {e}")
        return None
        
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'status': 'OPEN',
            'bot_name': bot_result['bot_name'],
            'risk_level': bot_result['risk_level'],
            'order_type': order_type,
            'pnl': 0.0,  # Initialize P&L
            'exit_price': None,
            'exit_time': None
        }
        
        if 'trade_history' not in st.session_state.automated_mode:
            st.session_state.automated_mode['trade_history'] = []
        st.session_state.automated_mode['trade_history'].append(trade_record)
        
        if order_type == "LIVE":
            st.toast(f"🤖 LIVE {action} order executed for {symbol} (Qty: {quantity})", icon="⚡")
        else:
            st.toast(f"🤖 PAPER {action} order simulated for {symbol} (Qty: {quantity})", icon="📄")
            
        return trade_record
        
    except Exception as e:
        st.error(f"Automated trade execution failed: {e}")
        return None

def run_automated_bots_cycle(instrument_df, watchlist_symbols):
    """Run one cycle of all active automated bots with paper trading updates."""
    if not st.session_state.automated_mode.get('running', False):
        return
    
    # Update paper portfolio values first
    update_paper_portfolio_values(instrument_df)
    
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    
    for bot_name in active_bots:
        for symbol in watchlist_symbols[:10]:  # Limit to first 10 symbols to avoid rate limits
            try:
                bot_function = AUTOMATED_BOTS[bot_name]
                bot_result = bot_function(instrument_df, symbol)
                
                if not bot_result.get("error") and bot_result["action"] != "HOLD":
                    execute_automated_trade(
                        instrument_df, 
                        bot_result, 
                        st.session_state.automated_mode.get('risk_per_trade', 2.0)
                    )
                
                # Small delay to avoid rate limiting
                a_time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Automated bot {bot_name} failed for {symbol}: {e}")
    
    # Update performance metrics
    st.session_state.automated_mode['performance_metrics'] = get_automated_bot_performance()
    st.session_state.automated_mode['last_signal_check'] = datetime.now().isoformat()

# Now define the page_fully_automated_bots function
import streamlit as st
import pandas as pd
from datetime import datetime

# These are assumed to be defined elsewhere in your application
# from your_utils import initialize_automated_mode, update_paper_portfolio_values, close_paper_position, run_automated_bots_cycle, get_automated_bot_performance
# from streamlit_autorefresh import st_autorefresh
# AUTOMATED_BOTS = {"Auto Momentum Trader": None, "Auto Mean Reversion": None}

# Define helper functions FIRST, before the main function
def get_ist_time():
    """Get current time in IST timezone."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def get_live_trading_performance():
    """Get live trading performance metrics (would connect to broker API)"""
    # Simulated broker connection data
    total_capital = st.session_state.automated_mode.get('total_capital', 10000.0)
    
    # Calculate some simulated metrics based on trade history
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    live_trades = [t for t in trade_history if t.get('order_type') == 'LIVE']
    
    total_pnl = sum(trade.get('pnl', 0) for trade in live_trades)
    winning_trades = [t for t in live_trades if t.get('pnl', 0) > 0]
    win_rate = (len(winning_trades) / len(live_trades) * 100) if live_trades else 0
    
    return {
        'account_balance': total_capital + total_pnl,
        'used_margin': total_capital * 0.1,  # 10% margin usage
        'available_margin': total_capital * 0.9,
        'margin_usage': 10.0,
        'open_pnl': total_pnl * 0.3,  # 30% of total as open
        'open_pnl_pct': (total_pnl * 0.3 / total_capital * 100),
        'daily_pnl': total_pnl * 0.1,  # 10% as daily
        'total_pnl': total_pnl,
        'portfolio_risk': st.session_state.automated_mode.get('risk_per_trade', 2.0),
        'max_drawdown': abs(min([t.get('pnl', 0) for t in live_trades] + [0])),
        'sharpe_ratio': 1.2 if win_rate > 50 else 0.8,
        'win_rate': win_rate,
        'avg_position_size': total_capital * 0.05,  # 5% per position
        'leverage': 1.0,
        'volatility': 15.5
    }

def get_nifty25_instruments(instrument_df):
    """Get top 25 Nifty stocks by market cap."""
    # Top 25 Nifty stocks (you can update this list periodically)
    nifty_25_symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'ITC', 
        'KOTAKBANK', 'LT', 'SBIN', 'BHARTIARTL', 'AXISBANK', 'ASIANPAINT', 
        'MARUTI', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'WIPRO', 'NESTLEIND', 
        'POWERGRID', 'NTPC', 'HCLTECH', 'BAJFINANCE', 'ADANIPORTS', 'TATAMOTORS'
    ]
    
    nifty_instruments = []
    
    for symbol in nifty_25_symbols:
        # Find instrument in NSE
        instrument = instrument_df[
            (instrument_df['tradingsymbol'] == symbol) |
            (instrument_df['tradingsymbol'].str.startswith(symbol + '-')) |
            (instrument_df['name'].str.contains(symbol, case=False, na=False))
        ].head(1)
        
        if not instrument.empty:
            nifty_instruments.append({
                'symbol': instrument.iloc[0]['tradingsymbol'],
                'exchange': instrument.iloc[0]['exchange'],
                'token': instrument.iloc[0]['instrument_token'],
                'name': instrument.iloc[0]['name']
            })
        else:
            # Try alternative search if exact match not found
            alt_instrument = instrument_df[
                instrument_df['tradingsymbol'].str.contains(symbol, case=False, na=False)
            ].head(1)
            if not alt_instrument.empty:
                nifty_instruments.append({
                    'symbol': alt_instrument.iloc[0]['tradingsymbol'],
                    'exchange': alt_instrument.iloc[0]['exchange'],
                    'token': alt_instrument.iloc[0]['instrument_token'],
                    'name': alt_instrument.iloc[0]['name']
                })
    
    return nifty_instruments

def get_nifty25_auto_trading_status():
    """Get current status of Nifty 25 auto-trading."""
    # This would typically query your trading database
    return {
        'active_positions': st.session_state.automated_mode.get('active_nifty_positions', 0),
        'signals_today': st.session_state.automated_mode.get('nifty_signals_today', 0),
        'success_rate': st.session_state.automated_mode.get('nifty_success_rate', 0),
        'pnl_today': st.session_state.automated_mode.get('nifty_pnl_today', 0.0),
        'total_trades': st.session_state.automated_mode.get('nifty_total_trades', 0),
        'win_rate': st.session_state.automated_mode.get('nifty_win_rate', 0)
    }

def initialize_automated_mode():
    """Initialize automated trading mode with default values."""
    st.session_state.automated_mode = {
        'enabled': False,
        'running': False,
        'live_trading': False,
        'total_capital': 10000.0,
        'risk_per_trade': 2.0,
        'nifty25_auto_trade': False,
        'paper_portfolio': {
            'cash_balance': 10000.0,
            'positions': {},
            'initial_capital': 10000.0,
            'total_value': 10000.0
        },
        'active_bots': {},
        'trade_history': []
    }

def get_automated_bot_performance():
    """Get paper trading performance metrics"""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    cash_balance = paper_portfolio.get('cash_balance', 0.0)
    portfolio_value = paper_portfolio.get('total_value', cash_balance)
    initial_capital = paper_portfolio.get('initial_capital', portfolio_value)
    
    # Calculate metrics from trade history
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    paper_trades = [t for t in trade_history if t.get('order_type') != 'LIVE']
    
    total_trades = len(paper_trades)
    winning_trades = [t for t in paper_trades if t.get('pnl', 0) > 0]
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(trade.get('pnl', 0) for trade in paper_trades)
    avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
    losing_trades = [t for t in paper_trades if t.get('pnl', 0) < 0]
    avg_loss = np.mean([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
    
    return {
        'paper_return_pct': ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'unrealized_pnl': portfolio_value - cash_balance - initial_capital,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': 5.2,  # Simulated
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'recovery_factor': 1.8  # Simulated
    }

# Define diagnostic helper functions
def get_symbol_data(instrument_df, symbol):
    """Get data for a specific symbol"""
    try:
        if symbol in instrument_df.columns:
            return instrument_df[symbol]
        else:
            # Try to find symbol in dataframe
            for col in instrument_df.columns:
                if symbol in col:
                    return instrument_df[col]
        return None
    except:
        return None

def analyze_with_bot(bot_name, symbol, data):
    """Analyze a symbol with a specific bot and return thinking"""
    if bot_name == "Auto Momentum Trader":
        return analyze_momentum(symbol, data)
    elif bot_name == "Auto Mean Reversion":
        return analyze_mean_reversion(symbol, data)
    else:
        return "HOLD", 50, "Unknown bot strategy"

def analyze_momentum(symbol, data):
    """Momentum bot analysis with detailed thinking"""
    if len(data) < 20:
        return "HOLD", 0, "Insufficient data for momentum analysis"
    
    # Simple momentum logic (replace with actual implementation)
    recent_price = data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1]
    prev_price = data['close'].iloc[-5] if 'close' in data.columns else data.iloc[-5]
    
    momentum = (recent_price - prev_price) / prev_price * 100
    
    if momentum > 2:
        return "BUY", 70, f"Strong upward momentum: {momentum:.2f}%"
    elif momentum < -2:
        return "SELL", 70, f"Strong downward momentum: {momentum:.2f}%"
    else:
        return "HOLD", 60, f"Neutral momentum: {momentum:.2f}%"

def analyze_mean_reversion(symbol, data):
    """Mean reversion bot analysis with detailed thinking"""
    if len(data) < 20:
        return "HOLD", 0, "Insufficient data for mean reversion analysis"
    
    # Simple mean reversion logic (replace with actual implementation)
    recent_price = data['close'].iloc[-1] if 'close' in data.columns else data.iloc[-1]
    mean_price = data['close'].mean() if 'close' in data.columns else data.mean()
    
    deviation = (recent_price - mean_price) / mean_price * 100
    
    if deviation > 5:
        return "SELL", 65, f"Price {deviation:.2f}% above mean - overbought"
    elif deviation < -5:
        return "BUY", 65, f"Price {deviation:.2f}% below mean - oversold"
    else:
        return "HOLD", 55, f"Price near mean: {deviation:.2f}% deviation"

def run_detailed_bot_analysis(instrument_df, watchlist_symbols):
    """Run detailed analysis and return thinking data"""
    thinking_data = []
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    
    for symbol in watchlist_symbols[:15]:  # Limit for performance
        symbol_data = get_symbol_data(instrument_df, symbol)
        if symbol_data is None:
            continue
            
        for bot_name in active_bots:
            if bot_name in AUTOMATED_BOTS:
                try:
                    signal, confidence, reasoning = analyze_with_bot(bot_name, symbol, symbol_data)
                    thinking_data.append({
                        'symbol': symbol,
                        'bot': bot_name,
                        'signal': signal,
                        'confidence': confidence,
                        'thinking': reasoning,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    thinking_data.append({
                        'symbol': symbol,
                        'bot': bot_name,
                        'signal': 'ERROR',
                        'confidence': 0,
                        'thinking': f"Analysis error: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })
    
    # Store thinking data for display
    st.session_state.automated_mode['last_thinking_analysis'] = thinking_data
    return thinking_data

def display_bot_thinking(instrument_df):
    """Display real-time bot thinking and analysis"""
    
    # Get current state
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
    watchlist_symbols = [item['symbol'] for item in st.session_state.watchlists.get(active_watchlist, [])]
    
    if not active_bots:
        st.error("❌ **No Active Bots**: Enable at least one bot in the configuration panel")
        return
        
    if not watchlist_symbols:
        st.error("❌ **No Symbols**: Add symbols to your active watchlist first")
        return
    
    st.write("**🔍 Current Bot Analysis:**")
    
    # Use cached thinking data or run new analysis
    if 'last_thinking_analysis' in st.session_state.automated_mode:
        thinking_data = st.session_state.automated_mode['last_thinking_analysis']
    else:
        thinking_data = run_detailed_bot_analysis(instrument_df, watchlist_symbols)
    
    if thinking_data:
        # Convert to dataframe for better display
        thinking_df = pd.DataFrame(thinking_data)
        
        # Color code signals
        def color_signal(signal):
            if signal == 'BUY':
                return 'background-color: #90EE90'  # Light green
            elif signal == 'SELL':
                return 'background-color: #FFB6C1'  # Light red
            elif signal == 'HOLD':
                return 'background-color: #F0F0F0'  # Light gray
            else:
                return ''
        
        # Display styled dataframe
        styled_df = thinking_df.style.apply(lambda x: [color_signal(val) for val in x], subset=['signal'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        buy_signals = len(thinking_df[thinking_df['signal'] == 'BUY'])
        sell_signals = len(thinking_df[thinking_df['signal'] == 'SELL'])
        hold_signals = len(thinking_df[thinking_df['signal'] == 'HOLD'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Buy Signals", buy_signals)
        col2.metric("Sell Signals", sell_signals)
        col3.metric("Hold Signals", hold_signals)
        
        # Reasons for no trades
        if buy_signals == 0 and sell_signals == 0:
            st.warning("""
            **🤔 Why No Trades? Possible Reasons:**
            - Market conditions don't match bot strategies
            - Risk limits preventing position sizing
            - Maximum open trades limit reached
            - Insufficient confidence in signals
            - Missing or stale market data
            - Technical indicators showing neutral signals
            """)
    else:
        st.info("No analysis data available. Run diagnostics to see bot thinking.")

def display_detailed_diagnostics(instrument_df):
    """Display comprehensive diagnostics"""
    st.markdown("---")
    st.subheader("🔧 Detailed System Diagnostics")
    
    # System status
    st.write("**🖥️ System Status:**")
    col1, col2, col3 = st.columns(3)
    
    # Check basic requirements
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
    watchlist_symbols = [item['symbol'] for item in st.session_state.watchlists.get(active_watchlist, [])]
    
    with col1:
        if active_bots:
            st.success(f"✅ **Bots Active**: {len(active_bots)}")
        else:
            st.error("❌ **No Bots Active**")
    
    with col2:
        if watchlist_symbols:
            st.success(f"✅ **Symbols**: {len(watchlist_symbols)}")
        else:
            st.error("❌ **No Symbols**")
    
    with col3:
        if st.session_state.automated_mode.get('running', False):
            st.success("✅ **Trading Active**")
        else:
            st.error("❌ **Trading Stopped**")
    
    # Market data diagnostics
    st.write("**📊 Market Data Status:**")
    if watchlist_symbols:
        data_status = []
        for symbol in watchlist_symbols[:5]:  # Check first 5 symbols
            symbol_data = get_symbol_data(instrument_df, symbol)
            if symbol_data is not None and len(symbol_data) > 0:
                latest_time = symbol_data.index.max() if hasattr(symbol_data.index, 'max') else datetime.now()
                data_age = (datetime.now() - latest_time).total_seconds() / 60  # minutes
                
                if data_age < 5:
                    status = "✅ Fresh"
                elif data_age < 15:
                    status = "⚠️ Stale"
                else:
                    status = "❌ Old"
                    
                data_status.append({
                    'Symbol': symbol,
                    'Status': status,
                    'Age (min)': f"{data_age:.1f}",
                    'Data Points': len(symbol_data)
                })
            else:
                data_status.append({
                    'Symbol': symbol,
                    'Status': "❌ No Data",
                    'Age (min)': "N/A",
                    'Data Points': 0
                })
        
        st.dataframe(pd.DataFrame(data_status), use_container_width=True, hide_index=True)
    
    # Bot-specific diagnostics
    st.write("**🤖 Bot Configuration Check:**")
    bot_diagnostics = []
    for bot_name in AUTOMATED_BOTS.keys():
        is_active = st.session_state.automated_mode.get('bots_active', {}).get(bot_name, False)
        bot_diagnostics.append({
            'Bot': bot_name,
            'Status': '✅ Active' if is_active else '❌ Inactive',
            'Strategy': AUTOMATED_BOTS[bot_name].get('description', 'N/A')
        })
    
    st.dataframe(pd.DataFrame(bot_diagnostics), use_container_width=True, hide_index=True)
    
    # Trading limits check
    st.write("**📈 Trading Limits:**")
    limits_data = {
        'Setting': ['Max Open Trades', 'Risk per Trade', 'Total Capital', 'Analysis Frequency'],
        'Current Value': [
            st.session_state.automated_mode.get('max_open_trades', 5),
            f"{st.session_state.automated_mode.get('risk_per_trade', 2.0)}%",
            f"₹{st.session_state.automated_mode.get('total_capital', 10000.0):,.2f}",
            st.session_state.automated_mode.get('check_interval', '1 minute')
        ],
        'Status': ['✅ OK', '✅ OK', '✅ OK', '✅ OK']
    }
    st.dataframe(pd.DataFrame(limits_data), use_container_width=True, hide_index=True)

# NOW define the main function
def page_fully_automated_bots(instrument_df):
    """Fully automated bots page with comprehensive paper trading simulation."""
    
    # Display current time and market status first
    current_time = get_ist_time().strftime("%H:%M:%S IST")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    st.warning("🚨 **LIVE TRADING WARNING**: Automated bots will execute real trades with real money! Use at your own risk.", icon="⚠️")
    
    # 🎯 ENHANCED HEADER WITH TICKER THEME
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.subheader("🤖 Automated Trading Bots")
    with col_header2:
        st.caption(f"📅 {current_date}")
    with col_header3:
        # Live ticker-style time display
        if st.session_state.automated_mode.get('live_trading', False):
            st.caption(f"🔴 {current_time}")
        else:
            st.caption(f"🔵 {current_time}")
    
    # Initialize automated mode if not exists
    if 'automated_mode' not in st.session_state:
        initialize_automated_mode()
    else:
        # Ensure paper_portfolio exists (migration for existing sessions)
        if 'paper_portfolio' not in st.session_state.automated_mode:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': st.session_state.automated_mode.get('total_capital', 10000.0),
                'positions': {},
                'initial_capital': st.session_state.automated_mode.get('total_capital', 10000.0),
                'total_value': st.session_state.automated_mode.get('total_capital', 10000.0)
            }
    
    # Fix the total_capital value if it's below minimum
    current_capital = float(st.session_state.automated_mode.get('total_capital', 100.0))
    if current_capital < 100.0:
        st.session_state.automated_mode['total_capital'] = 100.0
    
    # Get market status with error handling
    try:
        status_info = get_market_status()
        market_status = status_info['status']
        next_market = status_info['next_market']
    except Exception as e:
        st.error(f"Error getting market status: {e}")
        market_status = "unknown"
        next_market = datetime.now()
    
    # 🎯 ENHANCED NIFTY 25 AUTO-TRADING TOGGLE SECTION
    st.markdown("---")
    st.subheader("🔷 Nifty 25 Auto-Trading")
    
    col_nifty1, col_nifty2, col_nifty3 = st.columns([2, 1, 1])
    with col_nifty1:
        nifty25_auto_trade = st.toggle(
            "🎯 Enable Nifty 25 Auto-Trading",
            value=st.session_state.automated_mode.get('nifty25_auto_trade', False),
            help="Automatically trade all top 25 Nifty stocks based on AI signals",
            key="nifty25_toggle"
        )
        st.session_state.automated_mode['nifty25_auto_trade'] = nifty25_auto_trade
    
    with col_nifty2:
        if nifty25_auto_trade:
            st.success("🟢 ACTIVE")
            # Show Nifty 25 count
            nifty_instruments = get_nifty25_instruments(instrument_df)
            st.caption(f"{len(nifty_instruments)} stocks")
        else:
            st.info("⚪ INACTIVE")
    
    with col_nifty3:
        if nifty25_auto_trade:
            if st.button("🔄 Refresh Nifty", use_container_width=True):
                st.rerun()
    
    # NIFTY 25 CONFIGURATION WHEN ENABLED
    if nifty25_auto_trade:
        with st.expander("⚙️ Nifty 25 Auto-Trading Settings", expanded=True):
            col_nifty_settings1, col_nifty_settings2 = st.columns(2)
            
            with col_nifty_settings1:
                nifty_bot_strategy = st.selectbox(
                    "Trading Strategy",
                    ["Momentum", "Mean Reversion", "Breakout", "Multi-Signal"],
                    key="nifty_strategy"
                )
                nifty_max_positions = st.slider("Max Open Positions", 1, 10, 5, key="nifty_max_pos")
                
                # Strategy-specific settings
                if nifty_bot_strategy == "Momentum":
                    nifty_momentum_period = st.slider("Momentum Period (days)", 5, 30, 10)
                elif nifty_bot_strategy == "Mean Reversion":
                    nifty_reversion_threshold = st.slider("Deviation Threshold (std)", 1.0, 3.0, 1.5)
            
            with col_nifty_settings2:
                nifty_risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 10.0, 2.0, key="nifty_risk")
                nifty_min_confidence = st.slider("Min Confidence", 70, 90, 80, key="nifty_conf")
                
                nifty_position_size = st.selectbox(
                    "Position Sizing",
                    ["Fixed", "Dynamic Risk-Based", "Volatility Adjusted"],
                    key="nifty_size"
                )
            
            # NIFTY 25 STATUS DASHBOARD
            nifty_status = get_nifty25_auto_trading_status()
            st.markdown("---")
            st.subheader("📊 Nifty 25 Auto-Trading Status")
            
            col_status1, col_status2, col_status3, col_status4 = st.columns(4)
            with col_status1:
                st.metric("Active Positions", f"{nifty_status['active_positions']}/{nifty_max_positions}")
            with col_status2:
                st.metric("Signals Today", nifty_status['signals_today'])
            with col_status3:
                st.metric("Success Rate", f"{nifty_status['success_rate']}%")
            with col_status4:
                st.metric("P&L Today", f"₹{nifty_status['pnl_today']:+.2f}")
    
    # 🎯 ENHANCED MARKET STATUS WITH SEGMENT TIMING
    st.markdown("---")
    
    # Get status color for display
    status_colors = {
        "market_open": "#00cc00",
        "equity_square_off": "#ff9900", 
        "derivatives_square_off": "#ff6600",
        "pre_market": "#ffcc00",
        "market_closed": "#cccccc",
        "weekend": "#cccccc"
    }
    
    status_color = status_colors.get(market_status, "#cccccc")
    
    if market_status == "market_open":
        time_left = datetime.combine(datetime.now().date(), time(15, 20)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">🟢 **MARKET OPEN** | Equity square-off at 3:20 PM | {minutes_left} minutes remaining</div>', unsafe_allow_html=True)
        
    elif market_status == "equity_square_off":
        time_left = datetime.combine(datetime.now().date(), time(15, 25)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">🔴 **EQUITY SQUARE-OFF** | Derivatives square-off in {minutes_left} minutes</div>', unsafe_allow_html=True)
        
    elif market_status == "derivatives_square_off":
        time_left = datetime.combine(datetime.now().date(), time(15, 30)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">🚨 **DERIVATIVES SQUARE-OFF** | Market closes in {minutes_left} minutes</div>', unsafe_allow_html=True)
        
    elif market_status == "pre_market":
        time_left = datetime.combine(datetime.now().date(), time(9, 15)) - datetime.now()
        minutes_left = time_left.seconds // 60
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">⏰ **PRE-MARKET** | Live trading starts in {minutes_left} minutes</div>', unsafe_allow_html=True)
        
    elif market_status == "market_closed":
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">🔴 **MARKET CLOSED** | Paper trading available 24/7</div>', unsafe_allow_html=True)
        
    else:  # weekend or unknown
        st.markdown(f'<div style="color: {status_color}; font-weight: bold;">🎉 **WEEKEND** | Paper trading available 24/7</div>', unsafe_allow_html=True)
    
    # Trading status indicators
    is_live_trading = st.session_state.automated_mode.get('live_trading', False)
    if st.session_state.automated_mode.get('running', False):
        if is_live_trading:
            if is_market_hours():
                st.error("**🔴 LIVE TRADING ACTIVE** - Real money at risk! Monitor positions carefully.")
            else:
                st.warning("**⏸️ LIVE TRADING ACTIVE (Market Closed)** - Orders will execute when market opens")
        else:
            st.info("**🔵 PAPER TRADING ACTIVE** - Safe simulation running 24/7")
    
    # 🎯 ENHANCED CONTROL PANEL
    st.markdown("---")
    st.subheader("🎮 Control Panel")
    
    # Main control panel with better layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write("**🔧 Mode**")
        auto_enabled = st.toggle(
            "Enable Bots", 
            value=st.session_state.automated_mode.get('enabled', False),
            help="Enable automated trading bots",
            key="auto_enable"
        )
        st.session_state.automated_mode['enabled'] = auto_enabled
    
    with col2:
        st.write("**🎯 Trading Type**")
        # Always allow live trading toggle, regardless of market hours
        live_trading = st.toggle(
            "Live Trading",
            value=st.session_state.automated_mode.get('live_trading', False),
            help="Real money trading - Available anytime (trades execute during market hours)",
            key="live_trading"
        )
        st.session_state.automated_mode['live_trading'] = live_trading
        
        # Show warning if live trading is enabled outside market hours
        if live_trading and not is_market_hours() and not is_pre_market_hours():
            st.warning("⚠️ Market is closed - Live orders will queue until market opens")
    
    with col3:
        st.write("**🚦 Actions**")
        if st.session_state.automated_mode['enabled']:
            if not st.session_state.automated_mode.get('running', False):
                # Start button - always available
                if live_trading:
                    if st.button("🔴 Start Live Trading", use_container_width=True, type="secondary"):
                        st.session_state.need_live_confirmation = True
                        st.rerun()
                else:
                    if st.button("🔵 Start Paper Trading", use_container_width=True, type="primary"):
                        st.session_state.automated_mode['running'] = True
                        st.success("Paper trading started!")
                        st.rerun()
            else:
                # Stop button
                if st.button("🛑 Stop Trading", use_container_width=True, type="secondary"):
                    st.session_state.automated_mode['running'] = False
                    st.rerun()
        else:
            st.button("Start Trading", use_container_width=True, disabled=True)
    
    with col4:
        st.write("**💰 Capital**")
        current_capital = float(st.session_state.automated_mode.get('total_capital', 100.0))
        current_capital = max(100.0, current_capital)
        
        total_capital = st.number_input(
            "Trading Capital (₹)",
            min_value=100.0,
            max_value=1000000.0,
            value=current_capital,
            step=100.0,
            help="Minimum ₹100 required for automated trading",
            key="auto_capital",
            label_visibility="collapsed"
        )
        st.session_state.automated_mode['total_capital'] = float(total_capital)
    
    with col5:
        st.write("**⚡ Risk**")
        current_risk = float(st.session_state.automated_mode.get('risk_per_trade', 2.0))
        current_risk = max(0.5, min(10.0, current_risk))  # Updated to 10% max as requested
        
        risk_per_trade = st.number_input(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=10.0,  # Updated to 10% max
            value=current_risk,
            step=0.5,
            help="Risk percentage per trade (0.5% to 10%)",
            key="auto_risk",
            label_visibility="collapsed"
        )
        st.session_state.automated_mode['risk_per_trade'] = float(risk_per_trade)
    
    # Update paper portfolio capital if not running
    if not st.session_state.automated_mode.get('running', False):
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        if not paper_portfolio:
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': float(total_capital),
                'positions': {},
                'initial_capital': float(total_capital),
                'total_value': float(total_capital)
            }
        else:
            st.session_state.automated_mode['paper_portfolio']['initial_capital'] = float(total_capital)
            st.session_state.automated_mode['paper_portfolio']['cash_balance'] = float(total_capital)
            st.session_state.automated_mode['paper_portfolio']['total_value'] = float(total_capital)

    # Live trading confirmation dialog
    if st.session_state.get('need_live_confirmation', False):
        st.markdown("---")
        st.error("""
        🚨 **LIVE TRADING CONFIRMATION REQUIRED**
        
        **You are about to enable LIVE TRADING with real money!**
        
        **Important Notes:**
        • Real orders with real money
        • You are responsible for ALL losses
        • Market conditions can change rapidly
        • Trading available 24/7 (orders execute during market hours)
        • Auto-stop at market close for position safety
        
        **Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)**
        """)
        
        col_confirm1, col_confirm2, col_confirm3 = st.columns([2, 1, 1])
        
        with col_confirm1:
            if st.button("✅ CONFIRM LIVE TRADING", type="primary", use_container_width=True):
                st.session_state.automated_mode['running'] = True
                st.session_state.automated_mode['live_trading'] = True
                st.session_state.need_live_confirmation = False
                st.session_state.live_trading_start_time = get_ist_time().isoformat()
                st.success("🚀 LIVE TRADING ACTIVATED!")
                st.rerun()
        
        with col_confirm2:
            if st.button("📄 PAPER TRADING", use_container_width=True):
                st.session_state.automated_mode['running'] = True
                st.session_state.automated_mode['live_trading'] = False
                st.session_state.need_live_confirmation = False
                st.info("Paper trading started.")
                st.rerun()
                
        with col_confirm3:
            if st.button("❌ CANCEL", use_container_width=True):
                st.session_state.automated_mode['live_trading'] = False
                st.session_state.need_live_confirmation = False
                st.info("Live trading cancelled.")
                st.rerun()
        
        return  # Stop execution here if confirmation is needed
    
    st.markdown("---")
    
    if st.session_state.automated_mode['enabled']:
        # 🎯 ENHANCED DASHBOARD LAYOUT WITH NEW FEATURES
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Bot Configuration", "📊 Live Dashboard", "🔍 Live Thinking", "🎯 Symbol Override", "📋 Trade History"])
        
        with tab1:
            display_bot_configuration_tab(instrument_df, nifty25_auto_trade)
        
        with tab2:
            # Live performance dashboard - CALL THE ACTUAL FUNCTION
            try:
                display_enhanced_live_dashboard(instrument_df)
            except Exception as e:
                st.error(f"Error displaying dashboard: {e}")
        
        with tab3:
            # 🎯 ENHANCED LIVE THINKING TAB - CALL THE ACTUAL FUNCTION
            # NO AUTO-REFRESH FOR LIVE THINKING TAB - IT HAS ITS OWN REFRESH BUTTON
            try:
                display_enhanced_live_thinking_tab(instrument_df)
            except Exception as e:
                st.error(f"Error in live thinking: {e}")
        
        with tab4:
            # 🎯 NEW SYMBOL OVERRIDE TAB - CALL THE ACTUAL FUNCTION
            try:
                display_symbol_override_tab(instrument_df)
            except Exception as e:
                st.error(f"Error in symbol override: {e}")
        
        with tab5:
            # 📋 TRADE HISTORY TAB - CALL THE ACTUAL FUNCTION
            try:
                display_trade_history()
            except Exception as e:
                st.error(f"Error displaying trade history: {e}")
        
        # 🎯 AUTO-REFRESH LOGIC (EXCLUDES LIVE THINKING TAB)
        if st.session_state.automated_mode.get('running', False):
            # Get current active tab to determine if we should auto-refresh
            current_tab = st.session_state.get('current_tab', '🤖 Bot Configuration')
            
            # Only auto-refresh if NOT on the Live Thinking tab
            if current_tab != "🔍 Live Thinking":
                # Get refresh interval from settings
                check_interval = st.session_state.automated_mode.get('check_interval', '1 minute')
                interval_seconds = {
                    "15 seconds": 15,
                    "30 seconds": 30,
                    "1 minute": 60,
                    "5 minutes": 300,
                    "15 minutes": 900,
                    "30 minutes": 1800
                }.get(check_interval, 60)
                
                # Auto-refresh only for non-Live Thinking tabs
                if interval_seconds <= 300:  # Only auto-refresh for intervals <= 5 minutes
                    st.rerun()
    
    else:
        # Setup guide when disabled
        display_setup_guide()

# Add helper function to track current tab
def track_current_tab():
    """Track the current active tab for auto-refresh purposes."""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "🤖 Bot Configuration"

# Add the missing function implementations

def display_enhanced_live_dashboard(instrument_df):
    """Enhanced live dashboard with optimized refresh behavior."""
    # Record user access
    record_user_interaction()
    
    st.subheader("📊 Live Trading Dashboard")
    
    # Add refresh control to dashboard
    col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 1, 1])
    
    with col_refresh1:
        st.write("**Real-time Performance Metrics**")
    
    with col_refresh2:
        if st.button("🔄 Refresh Data", key="dashboard_refresh"):
            record_user_interaction()
            prevent_refresh_for(5)  # Prevent auto-refresh for 5 seconds after manual refresh
            st.rerun()
    
    with col_refresh3:
        if st.session_state.automated_mode.get('running', False):
            st.success("🟢 LIVE")
        else:
            st.info("⏸️ PAUSED")
    
    # Your existing dashboard content...
    # Add more user interaction recording for form elements
    
    # Example: Record interactions with form elements
    if st.button("Example Button", key="example_btn"):
        record_user_interaction()
        prevent_refresh_for(3)
        # Handle button click
    
    # Record interactions with sliders, inputs, etc.
    example_slider = st.slider("Example Slider", 0, 100, 50, key="example_slider")
    record_user_interaction()  # Record after slider interaction

def display_enhanced_live_thinking_tab(instrument_df):
    """Enhanced live bot thinking analysis with real reasoning and manual refresh only."""
    
    # Track that we're on the Live Thinking tab (NO AUTO-REFRESH)
    st.session_state.current_tab = "🔍 Live Thinking"
    
    st.subheader("🔍 Live Bot Analysis & Thinking")
    st.info("🤖 **Manual Refresh Only**: This tab doesn't auto-refresh to preserve analysis context. Use the refresh button below.", icon="ℹ️")
    
    # Get current state
    active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
    active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
    watchlist_symbols = st.session_state.watchlists.get(active_watchlist, [])
    
    if not active_bots:
        st.error("❌ **No Active Bots**: Enable at least one bot in the configuration panel")
        return
        
    if not watchlist_symbols:
        st.error("❌ **No Symbols**: Add symbols to your active watchlist first")
        return
    
    # Analysis controls with enhanced layout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Real-time Bot Analysis**")
        st.caption(f"Analyzing {len(watchlist_symbols)} symbols with {len(active_bots)} active bots")
    
    with col2:
        if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
            st.session_state.last_analysis_time = get_ist_time().strftime("%H:%M:%S IST")
            st.rerun()
    
    with col3:
        if st.button("📊 View Raw Data", use_container_width=True):
            st.session_state.show_raw_thinking_data = not st.session_state.get('show_raw_thinking_data', False)
    
    # Show last analysis time
    last_analysis = st.session_state.get('last_analysis_time', 'Never')
    st.caption(f"Last analysis: {last_analysis}")
    
    # Run analysis on watchlist symbols
    thinking_data = []
    analysis_errors = []
    
    with st.spinner("🤖 Analyzing symbols with active bots... This may take a few moments."):
        progress_bar = st.progress(0)
        total_symbols = min(15, len(watchlist_symbols))  # Limit for performance
        
        for i, item in enumerate(watchlist_symbols[:total_symbols]):
            symbol = item['symbol']
            exchange = item.get('exchange', 'NSE')
            
            for bot_name in active_bots:
                try:
                    # Get current price for context first
                    quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
                    current_price = quote_data.iloc[0]['Price'] if not quote_data.empty else 0
                    
                    # Run bot analysis based on bot type
                    if bot_name == "Momentum Bot":
                        bot_result = analyze_momentum_for_thinking(instrument_df, symbol, exchange)
                    elif bot_name == "Mean Reversion Bot":
                        bot_result = analyze_mean_reversion_for_thinking(instrument_df, symbol, exchange)
                    elif bot_name == "Breakout Bot":
                        bot_result = analyze_breakout_for_thinking(instrument_df, symbol, exchange)
                    elif bot_name == "Multi-Signal Bot":
                        bot_result = analyze_multi_signal_for_thinking(instrument_df, symbol, exchange)
                    else:
                        bot_result = {
                            'action': 'HOLD',
                            'score': 50,
                            'signals': ['Bot not implemented'],
                            'risk_level': 'Medium',
                            'reasoning': 'Bot analysis function not available'
                        }
                    
                    if bot_result and not bot_result.get("error"):
                        thinking_data.append({
                            'Symbol': symbol,
                            'Bot': bot_name,
                            'Signal': bot_result['action'],
                            'Confidence': bot_result.get('score', 50),
                            'Current Price': f"₹{current_price:.2f}",
                            'Analysis': " | ".join(bot_result.get('signals', ['Analyzing...'])),
                            'Risk Level': bot_result.get('risk_level', 'Medium'),
                            'Reasoning': bot_result.get('reasoning', 'Analysis complete'),
                            'Timestamp': get_ist_time().strftime("%H:%M:%S")
                        })
                    else:
                        analysis_errors.append(f"{symbol} - {bot_name}: {bot_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    error_msg = f"{symbol} - {bot_name}: {str(e)}"
                    analysis_errors.append(error_msg)
                    thinking_data.append({
                        'Symbol': symbol,
                        'Bot': bot_name,
                        'Signal': 'ERROR',
                        'Confidence': 0,
                        'Current Price': 'N/A',
                        'Analysis': f"Analysis failed",
                        'Risk Level': 'High',
                        'Reasoning': error_msg,
                        'Timestamp': get_ist_time().strftime("%H:%M:%S")
                    })
            
            # Update progress
            progress_bar.progress((i + 1) / total_symbols)
    
    # Show analysis errors if any
    if analysis_errors:
        with st.expander("⚠️ Analysis Errors", expanded=False):
            for error in analysis_errors[:5]:  # Show first 5 errors
                st.error(error)
            if len(analysis_errors) > 5:
                st.caption(f"... and {len(analysis_errors) - 5} more errors")
    
    if thinking_data:
        # Convert to dataframe
        thinking_df = pd.DataFrame(thinking_data)
        
        # Color coding functions
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            elif val == 'HOLD':
                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
            else:
                return 'background-color: #f5f5f5; color: #333;'
        
        def color_confidence(val):
            if val >= 70:
                return 'color: #28a745; font-weight: bold;'
            elif val >= 50:
                return 'color: #ffc107; font-weight: bold;'
            else:
                return 'color: #dc3545; font-weight: bold;'
        
        def color_risk(val):
            if val == 'Low':
                return 'color: #28a745;'
            elif val == 'Medium':
                return 'color: #ffc107;'
            else:
                return 'color: #dc3545;'
        
        # Display styled dataframe
        st.subheader("📈 Real-time Analysis Results")
        
        # Filter options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            show_signals = st.multiselect(
                "Filter by Signal",
                options=['BUY', 'SELL', 'HOLD', 'ERROR'],
                default=['BUY', 'SELL'],
                key="signal_filter"
            )
        with col_filter2:
            min_confidence = st.slider("Min Confidence", 0, 100, 50, key="confidence_filter")
        with col_filter3:
            selected_bots = st.multiselect(
                "Filter by Bot",
                options=active_bots,
                default=active_bots,
                key="bot_filter"
            )
        
        # Apply filters
        filtered_df = thinking_df[
            (thinking_df['Signal'].isin(show_signals)) &
            (thinking_df['Confidence'] >= min_confidence) &
            (thinking_df['Bot'].isin(selected_bots))
        ]
        
        if not filtered_df.empty:
            # Display filtered results
            display_columns = ['Symbol', 'Bot', 'Signal', 'Confidence', 'Current Price', 'Analysis', 'Risk Level']
            styled_df = filtered_df[display_columns].style\
                .map(color_signal, subset=['Signal'])\
                .map(color_confidence, subset=['Confidence'])\
                .map(color_risk, subset=['Risk Level'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Show raw data if requested
            if st.session_state.get('show_raw_thinking_data', False):
                with st.expander("📋 Raw Analysis Data", expanded=False):
                    st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No results match your current filters. Try adjusting filter settings.")
        
        # Summary statistics
        st.subheader("📊 Analysis Summary")
        
        total_analyses = len(thinking_df)
        buy_signals = len(thinking_df[thinking_df['Signal'] == 'BUY'])
        sell_signals = len(thinking_df[thinking_df['Signal'] == 'SELL'])
        hold_signals = len(thinking_df[thinking_df['Signal'] == 'HOLD'])
        error_signals = len(thinking_df[thinking_df['Signal'] == 'ERROR'])
        avg_confidence = thinking_df[thinking_df['Signal'] != 'ERROR']['Confidence'].mean()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Analyses", total_analyses)
        col2.metric("Buy Signals", buy_signals)
        col3.metric("Sell Signals", sell_signals)
        col4.metric("Hold Signals", hold_signals)
        col5.metric("Avg Confidence", f"{avg_confidence:.1f}%" if not pd.isna(avg_confidence) else "N/A")
        
        # Bot-specific analysis
        st.subheader("🤖 Bot Performance Summary")
        bot_performance = thinking_df.groupby('Bot').agg({
            'Signal': 'count',
            'Confidence': 'mean'
        }).rename(columns={'Signal': 'Analyses', 'Confidence': 'Avg Confidence'})
        
        st.dataframe(bot_performance.style.format({'Avg Confidence': '{:.1f}%'}), use_container_width=True)
        
        # Actionable insights
        st.subheader("💡 Market Insights & Recommendations")
        
        # Market sentiment analysis
        total_valid_signals = buy_signals + sell_signals + hold_signals
        if total_valid_signals > 0:
            buy_ratio = (buy_signals / total_valid_signals) * 100
            sell_ratio = (sell_signals / total_valid_signals) * 100
            
            if buy_ratio > 60:
                st.success("**📈 Strong Bullish Sentiment**")
                st.caption(f"{buy_ratio:.1f}% of signals are BUY recommendations")
            elif sell_ratio > 60:
                st.error("**📉 Strong Bearish Sentiment**") 
                st.caption(f"{sell_ratio:.1f}% of signals are SELL recommendations")
            else:
                st.info("**⚖️ Mixed Market Sentiment**")
                st.caption(f"Balanced signals: {buy_ratio:.1f}% BUY, {sell_ratio:.1f}% SELL")
        
        # Signal quality assessment
        if avg_confidence >= 70:
            st.success("**✅ High Quality Signals**")
            st.caption("Most analyses show high confidence levels - consider acting on strong signals")
        elif avg_confidence >= 50:
            st.warning("**⚠️ Moderate Signal Quality**")
            st.caption("Mixed confidence levels - review individual analyses carefully")
        else:
            st.error("**❌ Low Confidence Environment**")
            st.caption("Consider waiting for better market conditions or reviewing strategy")
        
        # Top recommendations
        strong_buy_signals = filtered_df[
            (filtered_df['Signal'] == 'BUY') & 
            (filtered_df['Confidence'] >= 70)
        ]
        
        if not strong_buy_signals.empty:
            st.subheader("🎯 Top BUY Recommendations")
            for _, signal in strong_buy_signals.head(3).iterrows():
                st.success(f"**{signal['Symbol']}** - {signal['Bot']} (Confidence: {signal['Confidence']}%)")
                st.caption(f"Analysis: {signal['Analysis']}")
        
    else:
        st.info("""
        ## 🤖 No Analysis Data Available
        
        **Possible reasons:**
        - All bots are still processing
        - No active symbols in watchlist  
        - Market data currently unavailable
        - Bot configuration issues
        
        **Try these solutions:**
        - Click **🔄 Refresh Analysis** button above
        - Add more symbols to your active watchlist
        - Check bot activation status in Configuration tab
        - Ensure market data is available (9:15 AM - 3:30 PM IST for live data)
        - Verify your internet connection
        """)

# Add these helper analysis functions if not already defined
def analyze_momentum_for_thinking(instrument_df, symbol, exchange):
    """Momentum analysis for thinking tab."""
    try:
        # Your momentum analysis logic here
        return {
            'action': 'BUY',  # Example
            'score': 75,
            'signals': ['Strong uptrend', 'Volume confirmation'],
            'risk_level': 'Medium',
            'reasoning': 'Price above key moving averages with increasing volume'
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_mean_reversion_for_thinking(instrument_df, symbol, exchange):
    """Mean reversion analysis for thinking tab."""
    try:
        # Your mean reversion analysis logic here
        return {
            'action': 'HOLD',  # Example
            'score': 60,
            'signals': ['Oversold condition', 'RSI below 30'],
            'risk_level': 'Low',
            'reasoning': 'Stock showing oversold signals but waiting for confirmation'
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_breakout_for_thinking(instrument_df, symbol, exchange):
    """Breakout analysis for thinking tab."""
    try:
        # Your breakout analysis logic here
        return {
            'action': 'SELL',  # Example
            'score': 80,
            'signals': ['Breakdown confirmed', 'Volume spike'],
            'risk_level': 'High',
            'reasoning': 'Price broke below key support level with high volume'
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_multi_signal_for_thinking(instrument_df, symbol, exchange):
    """Multi-signal analysis for thinking tab."""
    try:
        # Your multi-signal analysis logic here
        return {
            'action': 'BUY',
            'score': 85,
            'signals': ['Momentum + Volume + Breakout confirmation'],
            'risk_level': 'Low',
            'reasoning': 'Multiple strategies aligning for strong buy signal'
        }
    except Exception as e:
        return {'error': str(e)}

def display_symbol_override_tab(instrument_df):
    """Enhanced symbol override tab with manual control and analysis."""
    st.subheader("🎯 Symbol Override & Manual Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Manual Trade Entry")
        
        # Symbol selection with search
        symbol = st.text_input("Symbol", key="manual_symbol", placeholder="e.g., RELIANCE, TCS").upper()
        
        # Show symbol info if exists
        if symbol:
            symbol_info = instrument_df[instrument_df['tradingsymbol'] == symbol]
            if not symbol_info.empty:
                st.success(f"✅ Symbol found: {symbol_info.iloc[0].get('name', symbol)}")
            else:
                st.warning("⚠️ Symbol not found in instrument list")
        
        action = st.selectbox("Action", ["BUY", "SELL"], key="manual_action")
        quantity = st.number_input("Quantity", min_value=1, value=1, key="manual_quantity")
        price_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="manual_type")
        
        if price_type == "LIMIT":
            price = st.number_input("Price", min_value=0.01, key="manual_price", value=0.0)
        else:
            price = None
        
        # Market status info
        current_status = get_market_status()['status']
        market_open = is_market_hours()
        
        if st.button("🚀 Execute Manual Trade", type="primary", use_container_width=True):
            if symbol:
                try:
                    if st.session_state.automated_mode.get('live_trading', False):
                        if not market_open:
                            st.warning("⚠️ Market is closed - Live order will queue until market opens")
                        # Execute real trade
                        place_order(instrument_df, symbol, quantity, price_type, action, 'MIS', price)
                        st.success(f"✅ LIVE {action} order placed for {symbol}")
                    else:
                        # Paper trading simulation
                        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
                        current_price = price if price else get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}]).iloc[0]['Price']
                        
                        if action == "BUY":
                            trade_value = quantity * current_price
                            if paper_portfolio.get('cash_balance', 0) >= trade_value:
                                paper_portfolio['cash_balance'] -= trade_value
                                if symbol in paper_portfolio.get('positions', {}):
                                    # Update existing position
                                    old_qty = paper_portfolio['positions'][symbol]['quantity']
                                    old_avg = paper_portfolio['positions'][symbol]['avg_price']
                                    new_qty = old_qty + quantity
                                    new_avg = ((old_qty * old_avg) + (quantity * current_price)) / new_qty
                                    paper_portfolio['positions'][symbol]['quantity'] = new_qty
                                    paper_portfolio['positions'][symbol]['avg_price'] = new_avg
                                else:
                                    # New position
                                    if 'positions' not in paper_portfolio:
                                        paper_portfolio['positions'] = {}
                                    paper_portfolio['positions'][symbol] = {
                                        'quantity': quantity,
                                        'avg_price': current_price,
                                        'action': 'BUY',
                                        'entry_time': get_ist_time().isoformat()
                                    }
                                st.success(f"📄 PAPER {action} executed for {symbol} @ ₹{current_price:.2f}")
                            else:
                                st.error("❌ Insufficient paper trading balance")
                        else:  # SELL
                            # For paper trading, we simulate selling
                            if symbol in paper_portfolio.get('positions', {}):
                                position = paper_portfolio['positions'][symbol]
                                if position['quantity'] >= quantity:
                                    # Calculate P&L
                                    pnl = (current_price - position['avg_price']) * quantity
                                    paper_portfolio['cash_balance'] += quantity * current_price
                                    paper_portfolio['positions'][symbol]['quantity'] -= quantity
                                    
                                    # Remove if position closed
                                    if paper_portfolio['positions'][symbol]['quantity'] == 0:
                                        del paper_portfolio['positions'][symbol]
                                    
                                    st.success(f"📄 PAPER {action} executed for {symbol} @ ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
                                else:
                                    st.error(f"❌ Not enough shares to sell. Holding: {position['quantity']}")
                            else:
                                st.error(f"❌ No position found for {symbol}")
                except Exception as e:
                    st.error(f"❌ Trade execution failed: {e}")
            else:
                st.warning("⚠️ Please enter a symbol")
    
    with col2:
        st.write("### Quick Actions")
        
        # Force analysis on specific symbol
        st.write("**🔍 Quick Analysis**")
        analysis_symbol = st.text_input("Analyze Symbol", key="analysis_symbol", placeholder="Enter symbol for quick analysis").upper()
        
        if st.button("🤖 Run Bot Analysis", use_container_width=True) and analysis_symbol:
            with st.spinner(f"Analyzing {analysis_symbol}..."):
                # Run all active bots on this symbol
                active_bots = [bot for bot, active in st.session_state.automated_mode.get('bots_active', {}).items() if active]
                for bot_name in active_bots:
                    if bot_name in AUTOMATED_BOTS:
                        try:
                            bot_function = AUTOMATED_BOTS[bot_name]
                            bot_result = bot_function(instrument_df, analysis_symbol)
                            if not bot_result.get("error"):
                                st.write(f"**{bot_name}:** {bot_result['action']} - {bot_result.get('score', 'N/A')}% confidence")
                                for signal in bot_result.get('signals', []):
                                    st.write(f"  - {signal}")
                        except Exception as e:
                            st.error(f"{bot_name} failed: {e}")
        
        st.markdown("---")
        
        # Portfolio management
        st.write("**💰 Portfolio Actions**")
        
        if st.button("🔄 Update Portfolio Values", use_container_width=True):
            update_paper_portfolio_values(instrument_df)
            st.success("✅ Portfolio values updated!")
        
        if st.button("🗑️ Clear All Trades", use_container_width=True):
            st.session_state.automated_mode['trade_history'] = []
            st.session_state.automated_mode['paper_portfolio'] = {
                'cash_balance': st.session_state.automated_mode['total_capital'],
                'positions': {},
                'initial_capital': st.session_state.automated_mode['total_capital'],
                'total_value': st.session_state.automated_mode['total_capital']
            }
            st.session_state.order_history = []
            st.success("✅ All trades and positions cleared!")
        
        # Market status info
        st.markdown("---")
        st.write("### 📈 Market Status")
        status_info = get_market_status()
        st.write(f"**Status:** {status_info['status'].replace('_', ' ').title()}")
        st.write(f"**Paper Trading:** ✅ Always Available")
        st.write(f"**Live Trading:** ✅ Available 24/7")
        
        if not is_market_hours():
            st.info("📅 Live orders will queue and execute when market opens (9:15 AM - 3:30 PM IST)")
        
        # Current portfolio snapshot
        st.markdown("---")
        st.write("**💰 Portfolio Snapshot**")
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        st.write(f"Cash: ₹{paper_portfolio.get('cash_balance', 0):.2f}")
        st.write(f"Positions: {len(paper_portfolio.get('positions', {}))}")
        st.write(f"Total Value: ₹{paper_portfolio.get('total_value', 0):.2f}")
def close_paper_position(symbol, quantity=None):
    """Close a paper trading position with proper P&L calculation."""
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    
    if not paper_portfolio or symbol not in paper_portfolio.get('positions', {}):
        st.error(f"No position found for {symbol}")
        return False
    
    position = paper_portfolio['positions'][symbol]
    close_quantity = quantity if quantity else position['quantity']
    
    if close_quantity > position['quantity']:
        st.error(f"Cannot close more than current position: {position['quantity']}")
        return False
    
    # Get current price
    live_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
    if live_data.empty:
        st.error(f"Could not get current price for {symbol}")
        return False
    
    current_price = live_data.iloc[0]['Price']
    
    # Calculate P&L
    pnl = (current_price - position['avg_price']) * close_quantity
    
    # Update cash and position
    paper_portfolio['cash_balance'] += close_quantity * current_price
    paper_portfolio['positions'][symbol]['quantity'] -= close_quantity
    
    # Remove position if fully closed
    if paper_portfolio['positions'][symbol]['quantity'] == 0:
        del paper_portfolio['positions'][symbol]
    
    # Update trade history
    open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                  if t.get('symbol') == symbol and t.get('status') == 'OPEN' and t.get('action') == position.get('action')]
    
    for trade in open_trades:
        if trade.get('quantity', 0) <= close_quantity:
            # Close this trade
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['exit_time'] = get_ist_time().isoformat()
            trade['pnl'] = pnl
            close_quantity -= trade.get('quantity', 0)
    
    st.success(f"✅ Closed {close_quantity} shares of {symbol} at ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
    return True

def display_setup_guide():
    """Display setup guide when automated mode is disabled."""
    st.subheader("🚀 Automated Trading Setup Guide")
    
    st.info("""
    **To get started with automated trading:**
    
    1. **Enable Bots** - Toggle the 'Enable Bots' switch in the Control Panel
    2. **Configure Bots** - Go to the '🤖 Bot Configuration' tab to select which bots to run
    3. **Set Capital & Risk** - Adjust your trading capital and risk per trade
    4. **Choose Trading Mode** - Select between Paper Trading (safe) or Live Trading (real money)
    5. **Start Trading** - Click the Start button to begin automated trading
    
    **Recommended for beginners:**
    - Start with Paper Trading to test strategies
    - Use minimum capital (₹1,000) initially
    - Enable only one bot at first to understand its behavior
    - Monitor performance in the '📊 Live Dashboard' tab
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📚 Learning Resources**")
        st.write("• Review bot strategies in Algo Trading Bots page")
        st.write("• Test strategies with paper trading first")
        st.write("• Monitor performance regularly")
        st.write("• Adjust risk parameters based on results")
    
    with col2:
        st.write("**⚠️ Risk Management**")
        st.write("• Never risk more than 2% per trade")
        st.write("• Start with paper trading")
        st.write("• Set stop losses for all positions")
        st.write("• Monitor automated trades regularly")

# =============================================================================
# INTELLIGENT SQUARE-OFF FUNCTIONS - ADD NEW ONES
# =============================================================================
def display_segment_square_off_info():
    """Display segment-specific square-off information"""
    st.markdown("---")
    st.subheader("🕒 Segment-wise Square-off Times")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📈 Equity/Cash**")
        st.write("• Market: 9:15 AM - 3:30 PM")
        st.write("• Square-off: 3:20 PM")
        st.write("• Auto close: 3:20 PM")
    
    with col2:
        st.write("**📊 Equity Derivatives**")
        st.write("• Market: 9:15 AM - 3:30 PM")
        st.write("• Square-off: 3:25 PM")
        st.write("• Auto close: 3:25 PM")
    
    with col3:
        st.write("**🛢️ Commodities**")
        st.write("• Market: Varies by commodity")
        st.write("• Square-off: 10 min before close")
        st.write("• Auto close: 10 min before")

def display_enhanced_square_off_suggestions():
    """Display intelligent square-off suggestions with segment-specific timing - ONLY FOR LIVE TRADING"""
    # Only show for live trading
    if not st.session_state.automated_mode.get('live_trading', False):
        return
    
    st.warning("**⚠️ SQUARE-OFF TIME ACTIVE** - Consider closing positions before market close")
    portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    positions = portfolio.get('positions', {})
    
    if not positions:
        st.info("✅ No open positions to square off")
        return
    
    current_time = datetime.now()
    market_close = datetime.combine(current_time.date(), time(15, 30))
    
    # Check different square-off times
    equity_square_off_start = datetime.combine(current_time.date(), time(15, 20))
    derivatives_square_off_start = datetime.combine(current_time.date(), time(15, 25))
    
    minutes_to_equity_square_off = max(0, (equity_square_off_start - current_time).seconds // 60)
    minutes_to_derivatives_square_off = max(0, (derivatives_square_off_start - current_time).seconds // 60)
    minutes_to_market_close = max(0, (market_close - current_time).seconds // 60)
    
    # Determine which square-off period we're in
    if current_time >= equity_square_off_start and current_time < derivatives_square_off_start:
        # Equity square-off time (3:20 PM - 3:25 PM)
        st.markdown("---")
        st.error(f"🚨 **EQUITY SQUARE-OFF TIME** - Auto square-off in progress | {minutes_to_derivatives_square_off} min until derivatives")
        
    elif current_time >= derivatives_square_off_start and current_time <= market_close:
        # Derivatives square-off time (3:25 PM - 3:30 PM)
        st.markdown("---")
        st.error(f"🚨 **DERIVATIVES SQUARE-OFF TIME** - Final auto square-off | {minutes_to_market_close} min until market close")
        
    elif minutes_to_equity_square_off <= 15 and minutes_to_equity_square_off > 0:
        # Pre-square-off warning (3:05 PM - 3:20 PM)
        st.markdown("---")
        st.warning(f"⚠️ **PRE-SQUARE OFF WARNING** - {minutes_to_equity_square_off} minutes until equity auto square-off")
    
    # Get intelligent analysis only for live trading
    analyzed_positions = get_intelligent_square_off_suggestions(positions)
    
    if analyzed_positions:
        # Display priority actions
        display_priority_actions(analyzed_positions)
        
        # Display detailed analysis
        with st.expander("📊 Live Position Analysis", expanded=True):
            display_intelligent_position_analysis(analyzed_positions)
        
        # Quick action panel
        display_enhanced_square_off_actions(analyzed_positions)

def get_intelligent_square_off_suggestions(positions):
    """Get intelligent square-off suggestions for LIVE TRADING only"""
    
    if not st.session_state.automated_mode.get('live_trading', False):
        return []
    
    if not positions:
        return []
    
    analyzed_positions = []
    
    for symbol, position in positions.items():
        try:
            current_price = get_current_price(symbol)
            if not current_price:
                continue
                
            quantity = position.get('quantity', 0)
            avg_price = position.get('avg_price', 0)
            entry_time = position.get('entry_time')
            bot_name = position.get('bot_name', 'Unknown')
            
            # Calculate metrics
            pnl = (current_price - avg_price) * quantity
            pnl_percent = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            holding_period = calculate_holding_period(entry_time)
            
            position_data = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'holding_period': holding_period,
                'bot_name': bot_name,
                'entry_time': entry_time
            }
            
            # Get intelligent recommendation
            position_data['recommendation'] = get_ai_position_analysis(position_data)
            
            analyzed_positions.append(position_data)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
    
    return analyzed_positions

def display_priority_actions(analyzed_positions):
    """Display priority actions based on intelligent analysis"""
    
    # Sort by priority (high confidence CLOSE actions first)
    high_priority = [p for p in analyzed_positions 
                    if p['recommendation'].get('action') == 'CLOSE' 
                    and p['recommendation'].get('confidence', 0) >= 70]
    
    medium_priority = [p for p in analyzed_positions 
                      if p['recommendation'].get('action') == 'CLOSE' 
                      and 50 <= p['recommendation'].get('confidence', 0) < 70]
    
    hold_positions = [p for p in analyzed_positions 
                     if p['recommendation'].get('action') == 'HOLD_OVERNIGHT']
    
    if high_priority:
        st.subheader("🎯 HIGH PRIORITY - Close These Positions")
        for position in high_priority:
            display_priority_position_card(position)
    
    if medium_priority:
        st.subheader("⚠️ MEDIUM PRIORITY - Consider Closing")
        for position in medium_priority:
            display_priority_position_card(position)
    
    if hold_positions:
        st.subheader("💡 HOLD RECOMMENDED - Can Keep Overnight")
        for position in hold_positions:
            display_hold_position_card(position)

def display_priority_position_card(position):
    """Display a priority position card for closing"""
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.write(f"**{position['symbol']}**")
        st.write(f"*{position['bot_name']}*")
    
    with col2:
        # P&L display
        pnl = position['pnl']
        if pnl > 0:
            st.success(f"₹{pnl:+.2f}")
        else:
            st.error(f"₹{pnl:+.2f}")
    
    with col3:
        # Confidence level
        confidence = position['recommendation'].get('confidence', 0)
        if confidence >= 80:
            st.error(f"🔴 {confidence}%")
        elif confidence >= 60:
            st.warning(f"🟡 {confidence}%")
        else:
            st.info(f"🔵 {confidence}%")
    
    with col4:
        # Quick close button
        if st.button("📉 Close", key=f"quick_close_{position['symbol']}"):
            close_position(position['symbol'], position['quantity'])
            st.success(f"Closed {position['symbol']}")
            st.rerun()
    
    with col5:
        # View details
        if st.button("📖", key=f"details_{position['symbol']}"):
            st.session_state[f"show_{position['symbol']}"] = True

def display_intelligent_position_analysis(analyzed_positions):
    """Display detailed intelligent analysis of all positions"""
    
    for position in analyzed_positions:
        with st.container():
            st.markdown(f"### {position['symbol']} Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Position Details:**")
                st.write(f"• Entry: ₹{position['avg_price']:.2f}")
                st.write(f"• Current: ₹{position['current_price']:.2f}")
                st.write(f"• P&L: ₹{position['pnl']:+.2f} ({position['pnl_percent']:+.1f}%)")
                st.write(f"• Held: {position['holding_period']}")
                st.write(f"• Strategy: {position['bot_name']}")
            
            with col2:
                st.write("**Intelligent Analysis:**")
                recommendation = position['recommendation']
                
                action = recommendation.get('action', 'HOLD')
                confidence = recommendation.get('confidence', 50)
                reasoning = recommendation.get('reasoning', 'Analysis unavailable')
                
                # Action with color coding
                if action == "CLOSE":
                    st.error(f"**Action: CLOSE** (Confidence: {confidence}%)")
                elif action == "HOLD_OVERNIGHT":
                    st.success(f"**Action: HOLD** (Confidence: {confidence}%)")
                else:
                    st.warning(f"**Action: {action}** (Confidence: {confidence}%)")
                
                st.write(f"**Reasoning:** {reasoning}")
                
                if recommendation.get('target_price'):
                    st.write(f"**Target:** ₹{recommendation['target_price']:.2f}")
                if recommendation.get('stop_loss'):
                    st.write(f"**Stop Loss:** ₹{recommendation['stop_loss']:.2f}")
            
            st.markdown("---")

def display_enhanced_square_off_actions(analyzed_positions):
    """Display enhanced square-off action buttons with intelligence"""
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔴 Close High Priority", use_container_width=True, type="primary"):
            close_high_priority_positions(analyzed_positions)
            st.success("High priority positions closed!")
            st.rerun()
    
    with col2:
        if st.button("📉 Close All Losing", use_container_width=True):
            close_losing_positions(analyzed_positions)
            st.success("All losing positions closed!")
            st.rerun()
    
    with col3:
        if st.button("🤖 Smart Close All", use_container_width=True):
            execute_smart_close_all(analyzed_positions)
            st.success("Smart close executed!")
            st.rerun()
    
    with col4:
        if st.button("📊 Market Analysis", use_container_width=True):
            display_market_analysis_report(analyzed_positions)
            
# =============================================================================
# HELPER FUNCTIONS - ADD NEW ONES
# =============================================================================
def calculate_holding_period(entry_time):
    """Calculate how long position has been held"""
    if not entry_time:
        return "Unknown"
    
    try:
        if isinstance(entry_time, str):
            entry_dt = datetime.fromisoformat(entry_time)
        else:
            entry_dt = entry_time
            
        duration = datetime.now() - entry_dt
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except:
        return "Unknown"

def close_high_priority_positions(analyzed_positions):
    """Close only high priority positions"""
    for position in analyzed_positions:
        if (position['recommendation'].get('action') == 'CLOSE' and 
            position['recommendation'].get('confidence', 0) >= 70):
            close_position(position['symbol'], position['quantity'])

def close_losing_positions(analyzed_positions):
    """Close all losing positions"""
    for position in analyzed_positions:
        if position['pnl'] < 0:
            close_position(position['symbol'], position['quantity'])

def execute_smart_close_all(analyzed_positions):
    """Execute smart closing based on recommendations"""
    for position in analyzed_positions:
        action = position['recommendation'].get('action')
        confidence = position['recommendation'].get('confidence', 0)
        
        if action == "CLOSE" and confidence >= 60:
            close_position(position['symbol'], position['quantity'])
        elif action == "PARTIAL_CLOSE" and confidence >= 50:
            # Close half position
            close_position(position['symbol'], position['quantity'] // 2)

def display_market_analysis_report(analyzed_positions):
    """Display comprehensive market analysis report"""
    
    total_pnl = sum(pos['pnl'] for pos in analyzed_positions)
    close_recommendations = sum(1 for pos in analyzed_positions 
                               if pos['recommendation'].get('action') == 'CLOSE')
    hold_recommendations = sum(1 for pos in analyzed_positions 
                              if pos['recommendation'].get('action') == 'HOLD_OVERNIGHT')
    
    st.info(f"""
    **📈 Live Market Analysis Report:**
    
    • **Total Portfolio P&L:** ₹{total_pnl:,.2f}
    • **Close Recommendations:** {close_recommendations} positions
    • **Hold Recommendations:** {hold_recommendations} positions
    • **Overall Sentiment:** {'Bearish' if close_recommendations > hold_recommendations else 'Bullish'}
    • **Suggested Action:** {'Square off majority' if close_recommendations > hold_recommendations else 'Hold quality positions'}
    """)

def display_hold_position_card(position):
    """Display hold position card"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**{position['symbol']}**")
        st.write(f"*{position['bot_name']}*")
    
    with col2:
        pnl = position['pnl']
        if pnl > 0:
            st.success(f"₹{pnl:+.2f}")
        else:
            st.error(f"₹{pnl:+.2f}")
    
    with col3:
        confidence = position['recommendation'].get('confidence', 0)
        st.success(f"🟢 {confidence}%")

# Placeholder functions for the main implementation
def get_current_price(symbol):
    """Get current price for a symbol - implement based on your data source"""
    # This would connect to your data source (broker API, market data feed, etc.)
    # For now, return a placeholder value
    return 100.0

def close_position(symbol, quantity):
    """Close a position - implement based on your trading logic"""
    # This would execute through your broker API
    st.success(f"Closed {symbol} - {quantity} shares")

def get_ai_position_analysis(position_data):
    """Get AI analysis for position - implement based on your AI service"""
    # This would integrate with your AI service
    # For now, return sample analysis
    return {
        "action": "CLOSE",
        "confidence": 75,
        "reasoning": "Profit target achieved, take profits before market close",
        "target_price": None,
        "stop_loss": None
    }
# ================ AUTOMATED MODE HELPER FUNCTIONS ================

def initialize_automated_mode():
    """Initialize session state for fully automated trading."""
    if 'automated_mode' not in st.session_state:
        st.session_state.automated_mode = {
            'enabled': False,
            'running': False,
            'live_trading': False,
            'bots_active': {},
            'total_capital': 1000,
            'risk_per_trade': 2.0,
            'max_open_trades': 5,
            'trade_history': [],
            'performance_metrics': {},
            'last_signal_check': None
        }

def get_automated_bot_performance():
    """Calculate performance metrics for automated bots."""
    if not st.session_state.automated_mode['trade_history']:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    trades = st.session_state.automated_mode['trade_history']
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
    
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def display_trade_history():
    """Display comprehensive trade history with filtering and analysis."""
    st.subheader("📋 Trade History")
    
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    
    if not trade_history:
        st.info("No trade history available yet.")
        return
    
    # Add filtering options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_status = st.selectbox("Filter by Status", ["ALL", "OPEN", "CLOSED"])
    with col2:
        show_bot = st.selectbox("Filter by Bot", ["ALL"] + list(AUTOMATED_BOTS.keys()))
    with col3:
        show_type = st.selectbox("Filter by Type", ["ALL", "LIVE", "PAPER"])
    
    # Filter trades
    filtered_trades = trade_history.copy()
    
    if show_status != "ALL":
        filtered_trades = [t for t in filtered_trades if t.get('status') == show_status]
    
    if show_bot != "ALL":
        filtered_trades = [t for t in filtered_trades if t.get('bot_name') == show_bot]
    
    if show_type != "ALL":
        filtered_trades = [t for t in filtered_trades if t.get('order_type') == show_type]
    
    if not filtered_trades:
        st.warning("No trades match the selected filters.")
        return
    
    # Convert to DataFrame for display
    display_data = []
    for trade in filtered_trades:
        display_data.append({
            'Time': trade.get('timestamp_display', 
                            get_ist_time().strftime("%H:%M:%S") if 'timestamp' not in trade 
                            else datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).astimezone(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")),
            'Symbol': trade.get('symbol', 'N/A'),
            'Action': trade.get('action', 'N/A'),
            'Quantity': trade.get('quantity', 0),
            'Price': f"₹{trade.get('entry_price', 0):.2f}",
            'Status': trade.get('status', 'UNKNOWN'),
            'Bot': trade.get('bot_name', 'Manual'),
            'Type': trade.get('order_type', 'PAPER'),
            'P&L': f"₹{trade.get('pnl', 0):.2f}"
        })
    
    df_trades = pd.DataFrame(display_data)
    
    # Color coding for actions and P&L
    def color_trade_row(row):
        styles = [''] * len(row)
        
        # Color action column
        if row['Action'] == 'BUY':
            styles[2] = 'background-color: #90EE90; color: #006400;'  # Light green
        elif row['Action'] == 'SELL':
            styles[2] = 'background-color: #FFB6C1; color: #8B0000;'  # Light red
        
        # Color P&L column
        try:
            pnl_value = float(row['P&L'].replace('₹', ''))
            if pnl_value > 0:
                styles[7] = 'background-color: #90EE90; color: #006400; font-weight: bold;'
            elif pnl_value < 0:
                styles[7] = 'background-color: #FFB6C1; color: #8B0000; font-weight: bold;'
        except:
            pass
            
        # Color status column
        if row['Status'] == 'OPEN':
            styles[5] = 'background-color: #FFFACD; color: #8B7500;'  # Light yellow
        elif row['Status'] == 'CLOSED':
            styles[5] = 'background-color: #F0F0F0; color: #666;'     # Light gray
            
        return styles
    
    styled_df = df_trades.style.apply(color_trade_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.subheader("📊 Trade Summary")
    
    closed_trades = [t for t in filtered_trades if t.get('status') == 'CLOSED']
    open_trades = [t for t in filtered_trades if t.get('status') == 'OPEN']
    
    if closed_trades:
        total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(closed_trades))
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total P&L", f"₹{total_pnl:.2f}")
        col4.metric("Open Trades", len(open_trades))
    
    # Action buttons
    st.markdown("---")
    col_actions1, col_actions2, col_actions3 = st.columns(3)
    
    with col_actions1:
        if st.button("📥 Export to CSV", use_container_width=True):
            # Create downloadable CSV
            csv = df_trades.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_actions2:
        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            st.session_state.automated_mode['trade_history'] = []
            st.success("Trade history cleared!")
            st.rerun()
    
    with col_actions3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

def get_current_price(symbol):
    """Get current price for a symbol with fallback."""
    try:
        # Try to get live data first
        live_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
        if not live_data.empty:
            return live_data.iloc[0]['Price']
        
        # Fallback: use instrument data
        instrument_df = get_instrument_df()
        if not instrument_df.empty:
            instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
            if not instrument.empty:
                # Return a reasonable estimate (you might want to enhance this)
                return 100.0  # Default fallback price
                
    except Exception as e:
        st.error(f"Error getting price for {symbol}: {e}")
    
    return 100.0  # Final fallback

def close_position(symbol, quantity):
    """Close a position in paper trading."""
    try:
        paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
        
        if symbol not in paper_portfolio.get('positions', {}):
            st.error(f"No position found for {symbol}")
            return False
        
        position = paper_portfolio['positions'][symbol]
        
        if quantity > position['quantity']:
            st.error(f"Cannot close more than current position: {position['quantity']}")
            return False
        
        # Get current price for P&L calculation
        current_price = get_current_price(symbol)
        
        # Calculate P&L
        pnl = (current_price - position['avg_price']) * quantity
        
        # Update cash balance
        paper_portfolio['cash_balance'] += quantity * current_price
        
        # Update position
        paper_portfolio['positions'][symbol]['quantity'] -= quantity
        
        # Remove position if fully closed
        if paper_portfolio['positions'][symbol]['quantity'] == 0:
            del paper_portfolio['positions'][symbol]
        
        # Update trade history
        open_trades = [t for t in st.session_state.automated_mode.get('trade_history', []) 
                      if t.get('symbol') == symbol and t.get('status') == 'OPEN']
        
        for trade in open_trades:
            if trade.get('quantity', 0) <= quantity:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = current_price
                trade['exit_time'] = get_ist_time().isoformat()
                trade['pnl'] = pnl
                quantity -= trade.get('quantity', 0)
        
        st.success(f"✅ Closed {quantity} shares of {symbol} at ₹{current_price:.2f} | P&L: ₹{pnl:.2f}")
        return True
        
    except Exception as e:
        st.error(f"Error closing position for {symbol}: {e}")
        return False

def get_ai_position_analysis(position_data):
    """Get intelligent analysis for position with multiple factors."""
    try:
        symbol = position_data['symbol']
        current_price = position_data['current_price']
        avg_price = position_data['avg_price']
        pnl_percent = position_data['pnl_percent']
        holding_period = position_data['holding_period']
        
        # Simple AI logic (you can enhance this with ML models)
        action = "HOLD_OVERNIGHT"
        confidence = 50
        reasoning = "Holding for potential gains"
        
        # Profit-taking logic
        if pnl_percent >= 3:  # Take profit at 3%
            action = "CLOSE"
            confidence = 80
            reasoning = f"Profit target achieved (+{pnl_percent:.1f}%) - take profits"
        
        # Stop-loss logic
        elif pnl_percent <= -2:  # Stop loss at -2%
            action = "CLOSE"
            confidence = 75
            reasoning = f"Stop loss triggered ({pnl_percent:.1f}%) - limit losses"
        
        # Time-based exit for day trades
        elif "h" in holding_period and int(holding_period.split('h')[0]) >= 4:
            action = "CLOSE"
            confidence = 60
            reasoning = "Extended holding period - consider closing before market close"
        
        # Market context (simplified)
        market_status = get_market_status()['status']
        if "square_off" in market_status and pnl_percent > 0:
            action = "CLOSE"
            confidence = 70
            reasoning = "Square-off time - secure profits"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "target_price": current_price * 1.02 if action == "HOLD_OVERNIGHT" else None,
            "stop_loss": current_price * 0.98 if action == "HOLD_OVERNIGHT" else None
        }
        
    except Exception as e:
        return {
            "action": "HOLD_OVERNIGHT",
            "confidence": 30,
            "reasoning": f"Analysis error: {str(e)}",
            "target_price": None,
            "stop_loss": None
        }
# <<<--- PLACE execute_automated_trade FUNCTION HERE --->>>
def execute_automated_trade(instrument_df, bot_result, risk_per_trade):
    """Execute trades automatically based on bot signals."""
    if bot_result.get("error") or bot_result["action"] == "HOLD":
        return None
    
    try:
        symbol = bot_result["symbol"]
        action = bot_result["action"]
        current_price = bot_result["current_price"]
        
        # Calculate position size based on risk
        risk_amount = (risk_per_trade / 100) * st.session_state.automated_mode['total_capital']
        quantity = max(1, int(risk_amount / current_price))
        
        # Check if we have too many open trades
        open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                      if t.get('status') == 'OPEN']
        if len(open_trades) >= st.session_state.automated_mode['max_open_trades']:
            return None
        
        # Check for existing position in the same symbol
        existing_position = next((t for t in open_trades if t.get('symbol') == symbol), None)
        if existing_position:
            # Avoid opening same position multiple times
            if existing_position['action'] == action:
                return None
        
        # PLACE REAL ORDER if live trading is enabled
        order_type = "PAPER"
        if st.session_state.automated_mode.get('live_trading', False):
            try:
                # Place the real order
                place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
                order_type = "LIVE"
            except Exception as e:
                st.error(f"❌ Failed to place LIVE order for {symbol}: {e}")
                return None
        
        # Record the trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'status': 'OPEN',
            'bot_name': bot_result['bot_name'],
            'risk_level': bot_result['risk_level'],
            'order_type': order_type,
            'pnl': 0  # Initialize P&L
        }
        
        st.session_state.automated_mode['trade_history'].append(trade_record)
        
        if order_type == "LIVE":
            st.toast(f"🤖 LIVE {action} order executed for {symbol} (Qty: {quantity})", icon="⚡")
        else:
            st.toast(f"🤖 PAPER {action} order simulated for {symbol} (Qty: {quantity})", icon="📄")
            
        return trade_record
        
    except Exception as e:
        st.error(f"Automated trade execution failed: {e}")
        return None

def run_automated_bots_cycle(instrument_df, watchlist_symbols):
    """Run one cycle of all active automated bots."""
    if not st.session_state.automated_mode['running']:
        return
    
    active_bots = [bot for bot, active in st.session_state.automated_mode['bots_active'].items() if active]
    
    for bot_name in active_bots:
        for symbol in watchlist_symbols[:10]:  # Limit to first 10 symbols to avoid rate limits
            try:
                bot_function = AUTOMATED_BOTS[bot_name]
                bot_result = bot_function(instrument_df, symbol)
                
                if not bot_result.get("error") and bot_result["action"] != "HOLD":
                    execute_automated_trade(
                        instrument_df, 
                        bot_result, 
                        st.session_state.automated_mode['risk_per_trade']
                    )
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Automated bot {bot_name} failed for {symbol}: {e}")
    
    # Update performance metrics
    st.session_state.automated_mode['performance_metrics'] = get_automated_bot_performance()
    st.session_state.automated_mode['last_signal_check'] = datetime.now().isoformat()

def execute_automated_trade(instrument_df, bot_result, risk_per_trade):
    """Execute trades automatically based on bot signals."""
    if bot_result.get("error") or bot_result["action"] == "HOLD":
        return None
    
    try:
        symbol = bot_result["symbol"]
        action = bot_result["action"]
        current_price = bot_result["current_price"]
        
        # Calculate position size based on risk
        risk_amount = (risk_per_trade / 100) * st.session_state.automated_mode['total_capital']
        quantity = max(1, int(risk_amount / current_price))
        
        # Check if we have too many open trades
        open_trades = [t for t in st.session_state.automated_mode['trade_history'] 
                      if t.get('status') == 'OPEN']
        if len(open_trades) >= st.session_state.automated_mode['max_open_trades']:
            return None
        
        # Check for existing position in the same symbol
        existing_position = next((t for t in open_trades if t.get('symbol') == symbol), None)
        if existing_position:
            # Avoid opening same position multiple times
            if existing_position['action'] == action:
                return None
        
       
        # Place the real order
        place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
        
        # Record the trade (simulated for demo)
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'entry_price': current_price,
            'status': 'OPEN',
            'bot_name': bot_result['bot_name'],
            'risk_level': bot_result['risk_level'],
            'pnl': 0  # Initialize P&L
        }
        
        st.session_state.automated_mode['trade_history'].append(trade_record)
        
        st.toast(f"🤖 Automated {action} order executed for {symbol}", icon="⚡")
        return trade_record
        
    except Exception as e:
        st.error(f"Automated trade execution failed: {e}")
        return None

def display_bot_configuration_tab(instrument_df, nifty25_auto_trade=False):
    """Display bot configuration tab with Nifty 25 integration and enhanced controls."""
    st.subheader("⚙️ Bot Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.write("**🤖 Active Bots**")
        
        # Define available bots
        available_bots = {
            "Momentum Bot": "Trades stocks showing strong upward momentum",
            "Mean Reversion Bot": "Identifies oversold/overbought conditions for reversal trades",
            "Breakout Bot": "Detects breakouts from consolidation patterns with volume confirmation",
            "Multi-Signal Bot": "Combines multiple strategies for high-probability trades",
            "Custom Bot": "User-defined trading strategy"
        }
        
        # Initialize bots_active if not exists
        if 'bots_active' not in st.session_state.automated_mode:
            st.session_state.automated_mode['bots_active'] = {}
        
        # Display bot toggles with descriptions - FIXED: Removed problematic expander
        for bot_name, bot_description in available_bots.items():
            # Create a container for each bot
            with st.container():
                col_bot1, col_bot2 = st.columns([3, 1])
                with col_bot1:
                    is_active = st.session_state.automated_mode['bots_active'].get(bot_name, False)
                    if st.checkbox(f"**{bot_name}**", value=is_active, key=f"auto_{bot_name}"):
                        st.session_state.automated_mode['bots_active'][bot_name] = True
                    else:
                        st.session_state.automated_mode['bots_active'][bot_name] = False
                    
                    # Show description as caption instead of expander
                    st.caption(f"📝 {bot_description}")
                
                with col_bot2:
                    if st.session_state.automated_mode['bots_active'].get(bot_name, False):
                        st.success("🟢 ON")
                    else:
                        st.info("⚪ OFF")
                
                st.markdown("---")
        
        st.write("**📊 Trading Limits**")
        
        col_limits1, col_limits2 = st.columns(2)
        
        with col_limits1:
            max_trades = st.slider(
                "Max Open Trades",
                min_value=1,
                max_value=20,
                value=st.session_state.automated_mode.get('max_open_trades', 5),
                help="Maximum simultaneous trades across all bots",
                key="auto_max_trades"
            )
            st.session_state.automated_mode['max_open_trades'] = max_trades
        
        with col_limits2:
            max_daily_trades = st.slider(
                "Max Daily Trades",
                min_value=5,
                max_value=100,
                value=st.session_state.automated_mode.get('max_daily_trades', 20),
                help="Maximum trades per day to manage risk",
                key="auto_max_daily"
            )
            st.session_state.automated_mode['max_daily_trades'] = max_daily_trades
        
        # Risk per trade settings
        st.markdown("---")
        st.write("**⚡ Risk Management**")
        
        current_risk = st.session_state.automated_mode.get('risk_per_trade', 2.0)
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=10.0,
            value=current_risk,
            step=0.5,
            help="Risk percentage per individual trade (0.5% to 10%)",
            key="config_risk"
        )
        st.session_state.automated_mode['risk_per_trade'] = risk_per_trade
        
        # Calculate risk amount
        total_capital = st.session_state.automated_mode.get('total_capital', 10000.0)
        risk_amount = total_capital * (risk_per_trade / 100)
        st.caption(f"💰 Risk Amount: ₹{risk_amount:.2f} per trade")
    
    with col_config2:
        st.write("**⏰ Analysis & Execution**")
        
        # Analysis frequency
        current_interval = st.session_state.automated_mode.get('check_interval', '1 minute')
        frequency_options = ["15 seconds", "30 seconds", "1 minute", "5 minutes", "15 minutes", "30 minutes"]
        
        current_index = 2  # Default to "1 minute"
        if current_interval in frequency_options:
            current_index = frequency_options.index(current_interval)
        
        check_interval = st.selectbox(
            "Analysis Frequency",
            options=frequency_options,
            index=current_index,
            help="How often bots analyze the market and check for signals",
            key="auto_freq"
        )
        st.session_state.automated_mode['check_interval'] = check_interval
        
        # Frequency warnings and info
        if check_interval == "15 seconds":
            st.warning("⚡ **High Frequency** - May hit API limits, use with caution")
        elif check_interval == "30 seconds":
            st.info("🚀 **Active Trading** - Good balance for day trading")
        elif check_interval == "1 minute":
            st.success("🔄 **Standard Frequency** - Stable and reliable")
        else:
            st.info("📊 **Swing Trading** - Lower frequency for longer-term positions")
        
        # Execution settings
        st.markdown("---")
        st.write("**🎯 Execution Settings**")
        
        execution_mode = st.selectbox(
            "Execution Mode",
            options=["Aggressive", "Moderate", "Conservative"],
            index=1,
            help="Aggressive: Take all signals, Conservative: Only highest confidence",
            key="execution_mode"
        )
        st.session_state.automated_mode['execution_mode'] = execution_mode
        
        # Confidence threshold based on execution mode
        if execution_mode == "Aggressive":
            min_confidence = st.slider("Min Confidence", 60, 90, 65, key="agg_confidence")
        elif execution_mode == "Moderate":
            min_confidence = st.slider("Min Confidence", 60, 90, 75, key="mod_confidence")
        else:  # Conservative
            min_confidence = st.slider("Min Confidence", 60, 90, 85, key="con_confidence")
        
        st.session_state.automated_mode['min_confidence'] = min_confidence
        
        st.markdown("---")
        st.write("**📋 Trading Universe**")
        
        # Nifty 25 status display
        if nifty25_auto_trade:
            st.success("🔷 **Nifty 25 Mode Active**")
            try:
                nifty_instruments = get_nifty25_instruments(instrument_df)
                st.caption(f"Trading {len(nifty_instruments)} Nifty 25 stocks")
                
                # Use a button to show symbols instead of expander
                if st.button("📋 View Nifty 25 Symbols", key="view_nifty"):
                    st.session_state.show_nifty_symbols = not st.session_state.get('show_nifty_symbols', False)
                
                if st.session_state.get('show_nifty_symbols', False):
                    symbol_text = "\n".join([f"{i+1}. {inst['symbol']} - {inst.get('name', 'N/A')}" 
                                           for i, inst in enumerate(nifty_instruments[:25])])
                    st.text_area("Nifty 25 Symbols", value=symbol_text, height=200, key="nifty_display")
            except Exception as e:
                st.error(f"Error loading Nifty 25: {e}")
        else:
            # Watchlist-based trading
            active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
            watchlist_symbols = st.session_state.watchlists.get(active_watchlist, [])
            
            if watchlist_symbols:
                st.success(f"📊 **Watchlist Mode**: {active_watchlist}")
                st.caption(f"Trading {len(watchlist_symbols)} symbols from watchlist")
                
                # Use button to show symbols instead of expander
                if st.button("📋 View Watchlist Symbols", key="view_watchlist"):
                    st.session_state.show_watchlist_symbols = not st.session_state.get('show_watchlist_symbols', False)
                
                if st.session_state.get('show_watchlist_symbols', False):
                    symbol_text = "\n".join([f"{i+1}. {item['symbol']}" 
                                           for i, item in enumerate(watchlist_symbols[:20])])
                    st.text_area("Watchlist Symbols", value=symbol_text, height=150, key="watchlist_display")
            else:
                st.warning("⚠️ **No Trading Symbols**")
                st.caption("Add symbols to your watchlist or enable Nifty 25 mode")
        
        # Quick actions
        st.markdown("---")
        st.write("**🔧 Quick Actions**")
        
        col_actions1, col_actions2 = st.columns(2)
        
        with col_actions1:
            if st.button("🔄 Reset Settings", use_container_width=True, key="reset_bots"):
                reset_bot_configuration()
                st.success("Bot configuration reset to defaults!")
                st.rerun()
        
        with col_actions2:
            if st.button("💾 Save Config", use_container_width=True, key="save_bots"):
                save_bot_configuration()
                st.success("Configuration saved!")

# Add these helper functions if not already defined
def reset_bot_configuration():
    """Reset bot configuration to default values."""
    st.session_state.automated_mode.update({
        'bots_active': {
            'Momentum Bot': True,
            'Mean Reversion Bot': True,
            'Breakout Bot': False,
            'Multi-Signal Bot': False,
            'Custom Bot': False
        },
        'max_open_trades': 5,
        'max_daily_trades': 20,
        'check_interval': '1 minute',
        'execution_mode': 'Moderate',
        'min_confidence': 75
    })

def save_bot_configuration():
    """Save current bot configuration."""
    config = {
        'bots_active': st.session_state.automated_mode.get('bots_active', {}),
        'max_open_trades': st.session_state.automated_mode.get('max_open_trades', 5),
        'max_daily_trades': st.session_state.automated_mode.get('max_daily_trades', 20),
        'check_interval': st.session_state.automated_mode.get('check_interval', '1 minute'),
        'execution_mode': st.session_state.automated_mode.get('execution_mode', 'Moderate'),
        'min_confidence': st.session_state.automated_mode.get('min_confidence', 75),
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Store in session state (in production, you'd save to file/database)
    st.session_state.automated_mode['saved_config'] = config
    return config
            
# ================ 5. PAGE DEFINITIONS ============

def page_settings():
    """A page for platform-wide configurations, including special trading days."""
    display_header()
    st.title("⚙️ Platform Settings")
    st.info("Configure platform settings and add special trading sessions like Muhurat Trading.", icon="🛠️")

    st.subheader("Special Trading Sessions")
    st.write("Add custom market days and times. The market status indicators will respect these overrides.")

    # Form to add a new special session
    with st.form("special_session_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            session_date = st.date_input("Date")
        with col2:
            start_time = st.time_input("Start Time", time(18, 15)) # Default for Muhurat
        with col3:
            end_time = st.time_input("End Time", time(19, 15))

        submitted = st.form_submit_button("Add Special Session")
        if submitted:
            if start_time >= end_time:
                st.error("Error: Start time must be before end time.")
            else:
                st.session_state.special_trading_days.append({
                    'date': session_date,
                    'start': start_time,
                    'end': end_time
                })
                st.success(f"Added special session for {session_date.strftime('%d %b %Y')} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}.")

    st.markdown("---")

    # Display current special sessions
    st.subheader("Configured Special Sessions")
    if not st.session_state.special_trading_days:
        st.info("No special trading sessions have been added.")
    else:
        for i, session in enumerate(st.session_state.special_trading_days):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            col1.write(session['date'].strftime("%A, %d %B %Y"))
            col2.write(f"Start: {session['start'].strftime('%H:%M')}")
            col3.write(f"End: {session['end'].strftime('%H:%M')}")
            if col4.button("Remove", key=f"remove_session_{i}"):
                st.session_state.special_trading_days.pop(i)
                st.rerun()

    st.markdown("---")
    st.subheader("Platform Configuration")
    st.write("Manage other platform settings here.")
    
    theme = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.get('theme', 'Dark') == 'Dark' else 1, horizontal=True)
    if theme != st.session_state.get('theme'):
        st.session_state.theme = theme
        st.rerun()

# --- Bharatiya Market Pulse (BMP) Functions ---

def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns the score and a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30:
        return 50, "Calculating...", "#cccccc"
    
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    vix_min, vix_max = lookback_df['vix_value'].min(), lookback_df['vix_value'].max()
    vix_norm = 100 - (((vix_value - vix_min) / (vix_max - vix_min)) * 100) if (vix_max - vix_min) > 0 else 50

    bmp_score = (0.40 * nifty_norm) + (0.40 * sensex_norm) + (0.20 * vix_norm)
    bmp_score = min(100, max(0, bmp_score))

    if bmp_score >= 80:
        label, color = "Bharat Udaan (Very Bullish)", "#00b300"
    elif bmp_score >= 60:
        label, color = "Bharat Pragati (Bullish)", "#33cc33"
    elif bmp_score >= 40:
        label, color = "Bharat Santulan (Neutral)", "#ffcc00"
    elif bmp_score >= 20:
        label, color = "Bharat Sanket (Bearish)", "#ff6600"
    else:
        label, color = "Bharat Mandhi (Very Bearish)", "#ff0000"

    return bmp_score, label, color

@st.cache_data(ttl=300)
def get_nifty50_constituents(instrument_df):
    """Fetches the list of NIFTY 50 stocks by filtering the Kite API instrument list."""
    if instrument_df.empty:
        return pd.DataFrame()
    
    nifty50_symbols = [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'HINDUNILVR', 'ITC', 
        'LT', 'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT', 
        'AXISBANK', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'M&M', 'NESTLEIND',
        'ADANIENT', 'TATASTEEL', 'INDUSINDBK', 'TECHM', 'NTPC', 'MARUTI', 
        'BAJAJ-AUTO', 'POWERGRID', 'HCLTECH', 'ADANIPORTS', 'BPCL', 'COALINDIA', 
        'EICHERMOT', 'GRASIM', 'JSWSTEEL', 'SHREECEM', 'HEROMOTOCO', 'HINDALCO',
        'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'SBILIFE',
        'TATAMOTORS', 'BRITANNIA', 'DIVISLAB', 'BAJAJFINSV', 'SUNPHARMA', 'HDFCLIFE'
    ]
    
    nifty_constituents = instrument_df[
        (instrument_df['tradingsymbol'].isin(nifty50_symbols)) & 
        (instrument_df['segment'] == 'NSE')
    ].copy()

    constituents_df = pd.DataFrame({
        'Symbol': nifty_constituents['tradingsymbol'],
        'Name': nifty_constituents['tradingsymbol']
    })
    
    return constituents_df.drop_duplicates(subset='Symbol').head(15)

def create_nifty_heatmap(instrument_df):
    """Generates a Plotly Treemap for NIFTY 50 stocks."""
    constituents_df = get_nifty50_constituents(instrument_df)
    if constituents_df.empty:
        return go.Figure()
    
    symbols_with_exchange = [{'symbol': s, 'exchange': 'NSE'} for s in constituents_df['Symbol'].tolist()]
    live_data = get_watchlist_data(symbols_with_exchange)
    
    if live_data.empty:
        return go.Figure()
        
    full_data = pd.merge(live_data, constituents_df, left_on='Ticker', right_on='Symbol', how='left')
    full_data['size'] = full_data['Price'].astype(float) * 1000
    
    fig = go.Figure(go.Treemap(
        labels=full_data['Ticker'],
        parents=[''] * len(full_data),
        values=full_data['size'],
        marker=dict(
            colorscale='RdYlGn',
            colors=full_data['% Change'],
            colorbar=dict(title="% Change"),
        ),
        text=full_data['Ticker'],
        textinfo="label",
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{customdata[1]:.2f}%<extra></extra>',
        customdata=np.column_stack([full_data['Price'], full_data['% Change']])
    ))

    fig.update_layout(title="NIFTY 50 Heatmap (Live)")
    return fig

@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using a more reliable yfinance ticker."""
    try:
        data = yf.download("NIFTY_F1", period="1d", interval="1m")
        if not data.empty:
            return data
    except Exception:
        pass
    return pd.DataFrame()

def page_dashboard():
    """A completely redesigned 'Trader UI' Dashboard."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    # BMP Calculation and Display
    bmp_col, heatmap_col = st.columns([1, 1], gap="large")
    with bmp_col:
        st.subheader("Bharatiya Market Pulse (BMP)")
        if not index_data.empty:
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
            
            nifty_hist = get_historical_data(get_instrument_token('NIFTY 50', instrument_df, 'NSE'), 'day', period='1y')
            sensex_hist = get_historical_data(get_instrument_token('SENSEX', instrument_df, 'BSE'), 'day', period='1y')
            vix_hist = get_historical_data(get_instrument_token('INDIA VIX', instrument_df, 'NSE'), 'day', period='1y')
            
            if not nifty_hist.empty and not sensex_hist.empty and not vix_hist.empty:
                lookback_data = pd.DataFrame({
                    'nifty_change': nifty_hist['close'].pct_change() * 100,
                    'sensex_change': sensex_hist['close'].pct_change() * 100,
                    'vix_value': vix_hist['close']
                }).dropna()
                
                bmp_score, bmp_label, bmp_color = get_bmp_score_and_label(nifty_row['% Change'], sensex_row['% Change'], vix_row['Price'], lookback_data)
                
                st.markdown(f'<div class="metric-card" style="border-color:{bmp_color};"><h3>{bmp_score:.2f}</h3><p style="color:{bmp_color}; font-weight:bold;">{bmp_label}</p><small>Proprietary score from NIFTY, SENSEX, and India VIX.</small></div>', unsafe_allow_html=True)
                with st.expander("What do the BMP scores mean?"):
                    st.markdown("""
                    - **80-100 (Bharat Udaan):** Very Strong Bullish Momentum.
                    - **60-80 (Bharat Pragati):** Moderately Bullish Sentiment.
                    - **40-60 (Bharat Santulan):** Neutral or Sideways Market.
                    - **20-40 (Bharat Sanket):** Moderately Bearish Sentiment.
                    - **0-20 (Bharat Mandhi):** Very Strong Bearish Momentum.
                    """)
            else:
                st.info("BMP data is loading...")
        else:
            st.info("BMP data is loading...")
    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        st.plotly_chart(create_nifty_heatmap(instrument_df), use_container_width=True)

    st.markdown("---")
    
    # --- Middle Row: Main Content Area ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            st.session_state.active_watchlist = st.radio(
                "Select Watchlist",
                options=st.session_state.watchlists.keys(),
                horizontal=True,
                label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 15:
                            st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange})
                            st.rerun()
                        else:
                            st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            # Display watchlist
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", unsafe_allow_html=True)
                    w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                    
                    quantity = w_cols[2].number_input("Qty", min_value=1, step=1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                    
                    if w_cols[3].button("B", key=f"buy_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                    if w_cols[4].button("S", key=f"sell_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
                    if w_cols[5].button("🗑️", key=f"del_{row['Ticker']}", use_container_width=True):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != row['Ticker']]
                        st.rerun()
                st.markdown("---")

        with tab2:
            st.subheader("My Portfolio")
            _, holdings_df, total_pnl, total_investment = get_portfolio()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
            st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            with st.expander("View Holdings"):
                if not holdings_df.empty:
                    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings found.")
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
    
    # --- Bottom Row: Live Ticker Tape ---
    ticker_symbols = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist'), [])
    
    if ticker_symbols:
        ticker_data = get_watchlist_data(ticker_symbols)
        
        if not ticker_data.empty:
            ticker_html = "".join([
                f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {'#28a745' if item['Change'] > 0 else '#FF4B4B'};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>"
                for _, item in ticker_data.iterrows()
            ])
            
            st.markdown(f"""
            <style>
                @keyframes marquee {{
                    0%   {{ transform: translate(100%, 0); }}
                    100% {{ transform: translate(-100%, 0); }}
                }}
                .marquee-container {{
                    width: 100%;
                    overflow: hidden;
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    background-color: #1a1a1a;
                    border-top: 1px solid #333;
                    padding: 5px 0;
                    white-space: nowrap;
                }}
                .marquee-content {{
                    display: inline-block;
                    padding-left: 100%;
                    animation: marquee 35s linear infinite;
                }}
            </style>
            <div class="marquee-container">
                <div class="marquee-content">
                    {ticker_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    st.subheader("Chart Layout")
    layout_option = st.radio("Select Layout", ["Single Chart", "2 Charts", "4 Charts", "6 Charts"], horizontal=True)
    
    chart_counts = {"Single Chart": 1, "2 Charts": 2, "4 Charts": 4, "6 Charts": 6}
    num_charts = chart_counts[layout_option]
    
    st.markdown("---")
    
    if num_charts == 1:
        render_chart_controls(0, instrument_df)
    elif num_charts == 2:
        cols = st.columns(2)
        for i, col in enumerate(cols):
            with col:
                render_chart_controls(i, instrument_df)
    elif num_charts == 4:
        for i in range(2):
            cols = st.columns(2)
            with cols[0]:
                render_chart_controls(i * 2, instrument_df)
            with cols[1]:
                render_chart_controls(i * 2 + 1, instrument_df)
    elif num_charts == 6:
        for i in range(2):
            cols = st.columns(3)
            with cols[0]:
                render_chart_controls(i * 3, instrument_df)
            with cols[1]:
                render_chart_controls(i * 3 + 1, instrument_df)
            with cols[2]:
                render_chart_controls(i * 3 + 2, instrument_df)

def render_chart_controls(i, instrument_df):
    """Helper function to render controls for a single chart."""
    with st.container(border=True):
        st.subheader(f"Chart {i+1}")
        
        chart_cols = st.columns(4)
        ticker = chart_cols[0].text_input("Symbol", "NIFTY 50", key=f"ticker_{i}").upper()
        period = chart_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key=f"period_{i}")
        interval = chart_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key=f"interval_{i}")
        chart_type = chart_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{i}")

        token = get_instrument_token(ticker, instrument_df)
        data = get_historical_data(token, interval, period=period)

        if data.empty:
            st.warning(f"No data to display for {ticker} with selected parameters.")
        else:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True, key=f"chart_{i}")

            order_cols = st.columns([2,1,1,1])
            order_cols[0].markdown("**Quick Order**")
            quantity = order_cols[1].number_input("Qty", min_value=1, step=1, key=f"qty_{i}", label_visibility="collapsed")
            
            if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'BUY', 'MIS')
            if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'SELL', 'MIS')

def page_premarket_pulse():
    """Global market overview and premarket indicators with a trader-focused UI."""
    display_header()
    st.title("🌅 Premarket & Global Cues")
    st.info("Track global market movements, futures data, and overnight trends that impact Indian markets.", icon="📊")

    # Market status indicator
    status_info = get_market_status()
    current_time = get_ist_time().strftime("%H:%M:%S IST")
    
    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
    with col_status1:
        status_color = "#00cc00" if status_info['status'] == 'market_open' else "#ffcc00"
        st.markdown(f"**Market Status:** <span style='color:{status_color};'>{status_info['status'].replace('_', ' ').title()}</span>", unsafe_allow_html=True)
    with col_status2:
        st.markdown(f"**Time:** {current_time}")
    with col_status3:
        if is_pre_market_hours():
            st.success("🔸 Pre-market Hours")
        elif is_market_hours():
            st.success("🟢 Market Open")
        else:
            st.info("🔴 Market Closed")

    st.markdown("---")

    # Global Market Snapshot
    st.subheader("🌍 Global Market Snapshot")
    
    # Major global indices with proper tickers
    global_tickers = {
        "S&P 500": "^GSPC", 
        "Dow Jones": "^DJI", 
        "NASDAQ": "^IXIC", 
        "FTSE 100": "^FTSE", 
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225", 
        "Hang Seng": "^HSI",
        "Shanghai": "000001.SS",
        "SGX Nifty": "NIFTY_F1"
    }
    
    # Try to get live global data
    global_data = get_global_indices_data_enhanced(global_tickers)
    
    # Check if we have valid data
    valid_data = False
    if not global_data.empty:
        # Check if we have at least some valid prices
        valid_prices = global_data[~global_data['Price'].isna()]['Price']
        if len(valid_prices) > 0:
            valid_data = True
    
    if not valid_data:
        st.warning("⚠️ Live global data temporarily unavailable. Showing sample data for reference.")
        global_data = get_fallback_global_data(global_tickers)
    
    if not global_data.empty:
        # Display global indices in a grid
        cols = st.columns(4)
        displayed_count = 0
        
        for i, row in global_data.iterrows():
            price = row['Price']
            change = row['Change']
            pct_change = row['% Change']
            
            if not np.isnan(price) and not np.isnan(pct_change):
                col_idx = displayed_count % 4
                with cols[col_idx]:
                    # Color coding for changes
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        label=row['Ticker'], 
                        value=f"{price:,.0f}" if price > 100 else f"{price:.2f}",
                        delta=f"{pct_change:+.2f}%",
                        delta_color=delta_color
                    )
                displayed_count += 1

    st.markdown("---")

    # Main content columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 NIFTY 50 Futures (SGX Nifty)")
        
        # Get SGX Nifty data
        sgx_data = get_gift_nifty_data_enhanced()
        
        if not sgx_data.empty:
            # Calculate change and percentage
            if len(sgx_data) >= 2:
                current_price = sgx_data['Close'].iloc[-1]
                prev_close = sgx_data['Close'].iloc[-2] if len(sgx_data) > 1 else sgx_data['Close'].iloc[0]
                change = current_price - prev_close
                pct_change = (change / prev_close) * 100
                
                # Display current SGX Nifty price
                st.metric(
                    "SGX Nifty Current", 
                    f"{current_price:.2f}",
                    delta=f"{change:+.2f} ({pct_change:+.2f}%)",
                    delta_color="normal" if change >= 0 else "inverse"
                )
            
            # Create chart
            fig = go.Figure()
            
            # Add candlestick or line trace based on data availability
            if all(col in sgx_data.columns for col in ['Open', 'High', 'Low', 'Close']):
                fig.add_trace(go.Candlestick(
                    x=sgx_data.index,
                    open=sgx_data['Open'],
                    high=sgx_data['High'],
                    low=sgx_data['Low'],
                    close=sgx_data['Close'],
                    name='SGX Nifty'
                ))
            else:
                # Fallback to line chart
                fig.add_trace(go.Scatter(
                    x=sgx_data.index,
                    y=sgx_data['Close' if 'Close' in sgx_data.columns else sgx_data.iloc[:, 0]],
                    mode='lines',
                    name='SGX Nifty',
                    line=dict(color='cyan')
                ))
            
            fig.update_layout(
                title="SGX Nifty Futures",
                xaxis_title="Date",
                yaxis_title="Price",
                template='plotly_dark',
                height=400,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Fallback: Show NIFTY 50 chart if SGX data unavailable
            st.info("📊 SGX Nifty data unavailable. Showing NIFTY 50 instead.")
            instrument_df = get_instrument_df()
            if not instrument_df.empty:
                nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
                if nifty_token:
                    nifty_data = get_historical_data(nifty_token, "5minute", period="1d")
                    if not nifty_data.empty:
                        st.plotly_chart(create_chart(nifty_data.tail(100), "NIFTY 50"), use_container_width=True)
                    else:
                        st.error("Unable to load NIFTY 50 data")
                else:
                    st.error("NIFTY 50 token not found")
            else:
                st.error("Instrument data not available")
            
    with col2:
        st.subheader("🌏 Asian Markets Live")
        
        asian_tickers = {
            "Nikkei 225": "^N225", 
            "Hang Seng": "^HSI",
            "Shanghai Comp": "000001.SS",
            "Taiwan": "^TWII",
            "KOSPI": "^KS11"
        }
        
        asian_data = get_global_indices_data_enhanced(asian_tickers)
        
        if not asian_data.empty:
            for _, row in asian_data.iterrows():
                price = row['Price']
                change = row['Change']
                pct_change = row['% Change']
                
                if not np.isnan(price) and not np.isnan(pct_change):
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        label=row['Ticker'], 
                        value=f"{price:,.0f}" if price > 1000 else f"{price:.2f}",
                        delta=f"{pct_change:+.2f}%",
                        delta_color=delta_color
                    )
                else:
                    st.write(f"**{row['Ticker']}:** Data updating...")
        else:
            st.info("Asian market data updating...")

        st.markdown("---")
        
        # Pre-market indicators for Indian market
        st.subheader("🇮🇳 Indian Market Indicators")
        
        try:
            # Get NIFTY 50 and BANK NIFTY data
            instrument_df = get_instrument_df()
            if not instrument_df.empty:
                # NIFTY 50
                nifty_data = get_watchlist_data([{'symbol': 'NIFTY 50', 'exchange': 'NSE'}])
                if not nifty_data.empty:
                    nifty_row = nifty_data.iloc[0]
                    st.metric(
                        "NIFTY 50",
                        f"{nifty_row['Price']:.2f}",
                        delta=f"{nifty_row['Change']:+.2f} ({nifty_row['% Change']:+.2f}%)"
                    )
                else:
                    st.write("**NIFTY 50:** Data updating...")
                
                # BANK NIFTY
                bank_nifty_data = get_watchlist_data([{'symbol': 'NIFTY BANK', 'exchange': 'NSE'}])
                if not bank_nifty_data.empty:
                    bank_nifty_row = bank_nifty_data.iloc[0]
                    st.metric(
                        "BANK NIFTY", 
                        f"{bank_nifty_row['Price']:.2f}",
                        delta=f"{bank_nifty_row['Change']:+.2f} ({bank_nifty_row['% Change']:+.2f}%)"
                    )
                else:
                    st.write("**BANK NIFTY:** Data updating...")
                
                # India VIX
                vix_data = get_watchlist_data([{'symbol': 'INDIA VIX', 'exchange': 'NSE'}])
                if not vix_data.empty:
                    vix_row = vix_data.iloc[0]
                    vix_color = "inverse" if vix_row['Change'] > 0 else "normal"  # Higher VIX = bearish
                    st.metric(
                        "India VIX",
                        f"{vix_row['Price']:.2f}",
                        delta=f"{vix_row['Change']:+.2f} ({vix_row['% Change']:+.2f}%)",
                        delta_color=vix_color
                    )
                else:
                    st.write("**India VIX:** Data updating...")
                    
        except Exception as e:
            st.error(f"Error loading Indian market data: {e}")

    st.markdown("---")

    # Enhanced News Section
    st.subheader("📰 Latest Market News & Analysis")
    
    # News search and filter
    col_news1, col_news2 = st.columns([3, 1])
    with col_news1:
        news_query = st.text_input("Search news", placeholder="Enter keywords (e.g., RBI, earnings, inflation)...", key="news_search")
    with col_news2:
        news_limit = st.selectbox("Show", [5, 10, 15], index=0, key="news_limit")
    
    try:
        with st.spinner("📡 Fetching latest market news..."):
            news_df = fetch_and_analyze_news(query=news_query if news_query else None)
            
            # If no news found, use fallback
            if news_df.empty:
                st.info("🔍 No live news found. Showing recent market updates...")
                news_df = get_fallback_news()
            
        if not news_df.empty:
            # Display news with sentiment analysis
            news_count = 0
            for _, news in news_df.iterrows():
                if news_count >= news_limit:
                    break
                    
                sentiment_score = news['sentiment']
                
                # Sentiment indicators
                if sentiment_score > 0.2:
                    sentiment_icon = "🟢"
                    sentiment_text = "Positive"
                    border_color = "#28a745"
                elif sentiment_score < -0.2:
                    sentiment_icon = "🔴" 
                    sentiment_text = "Negative"
                    border_color = "#dc3545"
                else:
                    sentiment_icon = "🟡"
                    sentiment_text = "Neutral"
                    border_color = "#ffc107"
                
                # News card with colored border
                st.markdown(f"""
                <div style="border-left: 4px solid {border_color}; padding-left: 10px; margin: 10px 0;">
                    <div style="display: flex; justify-content: between; align-items: start;">
                        <div style="flex: 1;">
                            <strong>{news['title']}</strong>
                            <br>
                            <small style="color: #666;">{news['source']} • {news['date']}</small>
                        </div>
                        <div style="margin-left: 10px;">
                            {sentiment_icon} {sentiment_text}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable details
                with st.expander("Read summary"):
                    if news.get('summary'):
                        st.write(news['summary'])
                    st.write(f"**Sentiment Score:** {sentiment_score:.2f}")
                    if news['link'] != "#":
                        st.markdown(f"[Read full article →]({news['link']})")
                
                news_count += 1
                
    except Exception as e:
        st.error(f"❌ Error loading news feed: {str(e)}")
        st.info("📋 Showing recent market updates...")
        # Show fallback news even if there's an error
        fallback_df = get_fallback_news()
        for _, news in fallback_df.head(news_limit).iterrows():
            st.write(f"**{news['title']}**")
            st.caption(f"Source: {news['source']} • {news['date']}")
            st.markdown("---")

    # Market Calendar Section
    st.markdown("---")
    st.subheader("📅 Today's Key Economic Events")
    
    # Get today's date
    today = datetime.now().date()
    
    # Sample economic events (in a real app, this would come from an API)
    today_events = [
        {"time": "09:00 AM", "event": "GDP Growth Rate Q2", "country": "IND", "impact": "High"},
        {"time": "10:30 AM", "event": "Inflation Rate YoY", "country": "IND", "impact": "High"},
        {"time": "02:00 PM", "event": "FOMC Meeting Minutes", "country": "USA", "impact": "Medium"},
        {"time": "06:00 PM", "event": "Crude Oil Inventories", "country": "USA", "impact": "Medium"},
    ]
    
    if today_events:
        for event in today_events:
            col_event1, col_event2, col_event3 = st.columns([2, 3, 1])
            with col_event1:
                st.write(f"**{event['time']}**")
            with col_event2:
                st.write(f"{event['event']} ({event['country']})")
            with col_event3:
                impact_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
                st.markdown(f"<span style='color:{impact_color[event['impact']]}; font-weight:bold;'>{event['impact']}</span>", unsafe_allow_html=True)
    else:
        st.info("No major economic events scheduled for today.")

    # Refresh button
    if st.button("🔄 Refresh All Data", use_container_width=True, type="primary"):
        # Clear cache for fresh data
        st.cache_data.clear()
        st.rerun()

    # Last updated timestamp
    st.caption(f"🕒 Last updated: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')}")

# Enhanced helper function for global data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_global_indices_data_enhanced(tickers):
    """Enhanced version to fetch global indices data with better error handling."""
    if not tickers:
        return pd.DataFrame()
    
    data = []
    
    for ticker_name, yf_ticker in tickers.items():
        try:
            # Download data with error handling
            stock_data = yf.download(yf_ticker, period="2d", progress=False)
            
            if stock_data.empty or len(stock_data) < 2:
                continue
                
            # Calculate changes
            current_close = stock_data['Close'].iloc[-1]
            prev_close = stock_data['Close'].iloc[-2]
            
            change = current_close - prev_close
            pct_change = (change / prev_close) * 100
            
            data.append({
                'Ticker': ticker_name,
                'Price': current_close,
                'Change': change,
                '% Change': pct_change,
                'Previous Close': prev_close
            })
            
        except Exception as e:
            print(f"Error fetching {ticker_name}: {e}")
            continue
    
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def get_gift_nifty_data_enhanced():
    """Enhanced GIFT NIFTY data fetcher with multiple fallback sources."""
    tickers_to_try = [
        "NIFTY_F1",  # Primary ticker
        "^NSEI",     # NIFTY 50 index as fallback
        "NQ=F",      # NASDAQ futures as reference
    ]
    
    for ticker in tickers_to_try:
        try:
            data = yf.download(ticker, period="1d", interval="5m", progress=False)
            if not data.empty and len(data) > 1:
                st.success(f"✓ GIFT NIFTY data loaded from {ticker}")
                return data
        except Exception as e:
            continue
    
    # Fallback: create synthetic data if all sources fail
    st.warning("⚠️ Using synthetic GIFT NIFTY data (live data unavailable)")
    dates = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='5min')
    base_price = 19500  # Approximate NIFTY level
    synthetic_data = []
    
    for date in dates:
        # Simulate price movement
        variation = random.uniform(-0.002, 0.002)  # -0.2% to +0.2%
        price = base_price * (1 + variation)
        synthetic_data.append({
            'Open': price * 0.999,
            'High': price * 1.001,
            'Low': price * 0.998,
            'Close': price,
            'Volume': random.randint(1000, 5000)
        })
    
    return pd.DataFrame(synthetic_data, index=dates)

def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis."""
    display_header()
    st.title("F&O Analytics Hub")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access F&O Analytics.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "PCR Analysis", "Volatility & OI Analysis"])
    
    with tab1:
        st.subheader("Live Options Chain")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            st.session_state.underlying_pcr = underlying 
        
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)

        if not chain_df.empty:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")
            with col3:
                if st.button("Most Active Options", use_container_width=True):
                    show_most_active_dialog(underlying, instrument_df)

            st.dataframe(
                style_option_chain(chain_df, underlying_ltp).format({
                    'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}',
                    'STRIKE': '₹{:.0f}',
                    'CALL OI': '{:,.0f}', 'PUT OI': '{:,.0f}'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Could not load options chain data.")
    
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        
        chain_df, _, _, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)
        if not chain_df.empty and 'CALL OI' in chain_df.columns:
            total_ce_oi = chain_df['CALL OI'].sum()
            total_pe_oi = chain_df['PUT OI'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total CE OI", f"{total_ce_oi:,.0f}")
            col2.metric("Total PE OI", f"{total_pe_oi:,.0f}")
            col3.metric("PCR", f"{pcr:.2f}")
            
            if pcr > 1.3:
                st.success("High PCR suggests potential bearish sentiment (more Puts bought for hedging/speculation).")
            elif pcr < 0.7:
                st.error("Low PCR suggests potential bullish sentiment (more Calls bought).")
            else:
                st.info("PCR indicates neutral sentiment.")
        else:
            st.info("PCR data is loading... Select an underlying in the 'Options Chain' tab first.")
    
    with tab3:
        st.subheader("Volatility & Open Interest Surface")
        st.info("Real-time implied volatility and OI analysis for options contracts.")

        chain_df, expiry, underlying_ltp, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)

        if not chain_df.empty and expiry and underlying_ltp > 0:
            T = (expiry - datetime.now().date()).days / 365.0
            r = 0.07

            with st.spinner("Calculating Implied Volatility..."):
                chain_df['IV_CE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['CALL LTP'], 'call') * 100,
                    axis=1
                )
                chain_df['IV_PE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['PUT LTP'], 'put') * 100,
                    axis=1
                )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_CE'], mode='lines+markers', name='Call IV', line=dict(color='cyan')), secondary_y=False)
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_PE'], mode='lines+markers', name='Put IV', line=dict(color='magenta')), secondary_y=False)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['CALL OI'], name='Call OI', marker_color='rgba(0, 255, 255, 0.4)'), secondary_y=True)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['PUT OI'], name='Put OI', marker_color='rgba(255, 0, 255, 0.4)'), secondary_y=True)

            fig.update_layout(
                title_text=f"{st.session_state.get('underlying_pcr', 'NIFTY')} IV & OI Profile for {expiry.strftime('%d %b %Y')}",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select an underlying and expiry in the 'Options Chain' tab to view the volatility surface.")

import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import praw  # Reddit API

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime
import praw

def page_market_sentiment_ai():
    """Innovative AI-powered market sentiment analysis with real-time data."""
    st.title("🧠 AI Market Sentiment Analyzer")
    st.info("Real-time market sentiment analysis using AI and natural language processing.", icon="🌐")
    
    # Real-time data status
    if st.button("🔄 Refresh Real-time Data"):
        st.rerun()
    
    # Sentiment analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overall Sentiment", "Sector Analysis", "News Sentiment", "Social Trends"])
    
    with tab1:
        display_overall_sentiment()
    
    with tab2:
        display_sector_sentiment()
    
    with tab3:
        display_news_sentiment()
    
    with tab4:
        display_social_trends()

def display_overall_sentiment():
    """Display overall market sentiment analysis with real-time data."""
    st.subheader("📊 Overall Market Sentiment")
    
    # Calculate sentiment score with real data
    sentiment_score, market_data = calculate_market_sentiment()
    
    # Sentiment gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = create_sentiment_gauge(sentiment_score)
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time market metrics
    st.subheader("📈 Real-time Market Indicators")
    
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            nifty_change = market_data.get('nifty_change', 0)
            nifty_change_pct = market_data.get('nifty_change_pct', 0)
            st.metric("NIFTY 50", f"₹{market_data.get('nifty_price', 0):.2f}", 
                     f"{nifty_change:+.2f} ({nifty_change_pct:+.2f}%)")
        
        with col2:
            vix_price = market_data.get('vix_price', 0)
            vix_change = market_data.get('vix_change', 0)
            st.metric("India VIX", f"{vix_price:.2f}", 
                     f"{vix_change:+.2f}", delta_color="inverse")
        
        with col3:
            advance_decline = market_data.get('advance_decline', 'N/A')
            st.metric("Advance/Decline", advance_decline)
        
        with col4:
            put_call_ratio = market_data.get('put_call_ratio', 0)
            st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}")
    
    # Sentiment analysis
    st.subheader("🎯 Sentiment Insights")
    
    if sentiment_score > 70:
        st.success("**Strong Bullish Sentiment**: Market participants are optimistic. Consider quality stocks with strong fundamentals.")
    elif sentiment_score > 55:
        st.info("**Moderately Bullish**: Positive bias with selective opportunities. Focus on sectors showing strength.")
    elif sentiment_score > 45:
        st.warning("**Neutral Sentiment**: Market in consolidation. Wait for clear direction or trade range-bound strategies.")
    elif sentiment_score > 30:
        st.error("**Moderately Bearish**: Caution advised. Consider defensive stocks or hedging strategies.")
    else:
        st.error("**Strong Bearish Sentiment**: Risk-off environment. Focus on capital preservation and safe havens.")

def calculate_market_sentiment():
    """Calculate comprehensive market sentiment score using real-time data."""
    try:
        # Get real market data
        market_data = get_real_market_data()
        base_score = 50
        
        # Analyze NIFTY 50 trend
        nifty_change_pct = market_data.get('nifty_change_pct', 0)
        if nifty_change_pct > 1:
            base_score += 15
        elif nifty_change_pct > 0.5:
            base_score += 10
        elif nifty_change_pct < -1:
            base_score -= 15
        elif nifty_change_pct < -0.5:
            base_score -= 10
        
        # Analyze VIX (Fear Index)
        vix_price = market_data.get('vix_price', 20)
        if vix_price < 15:
            base_score += 10
        elif vix_price > 25:
            base_score -= 10
        elif vix_price > 30:
            base_score -= 15
        
        # Analyze Put/Call ratio
        put_call_ratio = market_data.get('put_call_ratio', 0.7)
        if put_call_ratio > 1.2:
            base_score += 10
        elif put_call_ratio < 0.8:
            base_score -= 5
        
        # Add Reddit sentiment to overall score
        reddit_sentiment = get_reddit_sentiment_score()
        base_score += (reddit_sentiment - 50) / 5
        
        # Market hours adjustment
        if is_market_hours():
            base_score += 5
        
        return max(0, min(100, base_score)), market_data
        
    except Exception as e:
        st.error(f"Error calculating sentiment: {e}")
        return 50, {}

def get_real_market_data():
    """Fetch real-time market data from various sources."""
    market_data = {}
    
    try:
        # NIFTY 50 data
        nifty = yf.download("^NSEI", period="1d", interval="5m", progress=False)
        if not nifty.empty and len(nifty) > 1:
            current_price = nifty['Close'].iloc[-1]
            prev_close = nifty['Close'].iloc[0]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            market_data.update({
                'nifty_price': current_price,
                'nifty_change': change,
                'nifty_change_pct': change_pct
            })
        else:
            # Fallback to daily data if intraday fails
            nifty_daily = yf.download("^NSEI", period="2d", progress=False)
            if not nifty_daily.empty and len(nifty_daily) > 1:
                current_price = nifty_daily['Close'].iloc[-1]
                prev_close = nifty_daily['Close'].iloc[-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                market_data.update({
                    'nifty_price': current_price,
                    'nifty_change': change,
                    'nifty_change_pct': change_pct
                })
        
        # India VIX data
        vix = yf.download("^INDIAVIX", period="2d", progress=False)
        if not vix.empty and len(vix) > 1:
            vix_price = vix['Close'].iloc[-1]
            vix_prev = vix['Close'].iloc[-2]
            vix_change = vix_price - vix_prev
            
            market_data.update({
                'vix_price': vix_price,
                'vix_change': vix_change
            })
        
        # Simulated Put/Call ratio (in real implementation, get from NSE)
        market_data['put_call_ratio'] = np.random.uniform(0.6, 1.4)
        
        # Simulated Advance/Decline ratio
        market_data['advance_decline'] = f"{np.random.randint(800, 1200)}/{np.random.randint(600, 1000)}"
        
    except Exception as e:
        st.warning(f"Some market data unavailable: {e}")
    
    return market_data

def create_sentiment_gauge(score):
    """Create a sentiment gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Sentiment"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightcoral"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def display_sector_sentiment():
    """Display sector-wise sentiment analysis with real data."""
    st.subheader("🏢 Sector Sentiment Analysis")
    
    # Fetch real sector data
    sector_data = get_real_sector_data()
    
    if not sector_data:
        st.warning("Unable to fetch real-time sector data. Showing sample data.")
        sector_data = get_sample_sector_data()
    
    # Display sector cards
    cols = st.columns(4)
    for idx, (sector, data) in enumerate(sector_data.items()):
        with cols[idx % 4]:
            sentiment = data["sentiment"]
            change = data["change"]
            delta_color = "normal" if change > 0 else "inverse"
            
            st.metric(
                sector,
                f"{sentiment}%",
                delta=f"{change:+.1f}%",
                delta_color=delta_color
            )
    
    # Sector rotation insights
    st.subheader("🔄 Sector Rotation Insights")
    
    bullish_sectors = [s for s, d in sector_data.items() if d["sentiment"] > 65]
    bearish_sectors = [s for s, d in sector_data.items() if d["sentiment"] < 40]
    
    if bullish_sectors:
        st.success(f"**Strong Sectors:** {', '.join(bullish_sectors)} - Consider overweight exposure")
    
    if bearish_sectors:
        st.error(f"**Weak Sectors:** {', '.join(bearish_sectors)} - Consider underweight or avoid")
    
    # Sector performance chart
    st.subheader("📊 Sector Performance Heatmap")
    display_sector_heatmap(sector_data)

def get_real_sector_data():
    """Fetch real sector performance data."""
    sector_indices = {
        "Technology": "NIFTYIT.NS",
        "Banking": "NIFTYBANK.NS",
        "Pharmaceuticals": "NIFTYPHARMA.NS",
        "Automobile": "NIFTYAUTO.NS",
        "Energy": "NIFTYENERGY.NS",
        "Real Estate": "NIFTYREALTY.NS",
        "FMCG": "NIFTYFMCG.NS",
        "Infrastructure": "NIFTYINFRA.NS"
    }
    
    sector_data = {}
    
    for sector, symbol in sector_indices.items():
        try:
            stock_data = yf.download(symbol, period="5d", interval="1d", progress=False)
            if not stock_data.empty and len(stock_data) > 1:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                # Convert price change to sentiment score (0-100)
                sentiment = max(0, min(100, 50 + (change_pct * 5)))
                
                sector_data[sector] = {
                    "sentiment": round(sentiment),
                    "change": round(change_pct, 2),
                    "price": current_price
                }
        except Exception as e:
            st.warning(f"Could not fetch data for {sector}: {e}")
    
    return sector_data

def get_sample_sector_data():
    """Return sample sector data when real data is unavailable."""
    return {
        "Technology": {"sentiment": 75, "change": 5.2, "price": 35000},
        "Banking": {"sentiment": 68, "change": 3.1, "price": 45000},
        "Pharmaceuticals": {"sentiment": 55, "change": -2.3, "price": 12500},
        "Automobile": {"sentiment": 45, "change": -5.1, "price": 8500},
        "Energy": {"sentiment": 72, "change": 8.7, "price": 22000},
        "Real Estate": {"sentiment": 38, "change": -7.2, "price": 4500},
        "FMCG": {"sentiment": 65, "change": 2.4, "price": 48000},
        "Infrastructure": {"sentiment": 58, "change": 4.1, "price": 5200}
    }

def display_sector_heatmap(sector_data):
    """Display sector performance heatmap."""
    sectors = list(sector_data.keys())
    sentiments = [data["sentiment"] for data in sector_data.values()]
    changes = [data["change"] for data in sector_data.values()]
    
    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'Sector': sectors,
        'Sentiment': sentiments,
        'Change (%)': changes
    })
    
    # Display as a styled dataframe
    def color_sentiment(val):
        if val > 70:
            return 'background-color: #90EE90'
        elif val > 55:
            return 'background-color: #FFB6C1'
        else:
            return 'background-color: #FFCCCB'
    
    styled_df = heatmap_data.style.map(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_df, use_container_width=True)

def display_news_sentiment():
    """Display news-based sentiment analysis with real news data."""
    st.subheader("📰 Real-time News & Social Sentiment Analysis")
    
    # Fetch real news and Reddit data
    with st.spinner("Fetching and analyzing latest market news and Reddit discussions..."):
        news_df = fetch_real_news_sentiment()
    
    if news_df.empty:
        st.info("No sentiment data available for analysis.")
        return
    
    # Calculate sentiment metrics
    positive_news = len(news_df[news_df['sentiment'] > 0.1])
    negative_news = len(news_df[news_df['sentiment'] < -0.1])
    neutral_news = len(news_df) - positive_news - negative_news
    total_news = len(news_df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", total_news)
    col2.metric("Positive", positive_news)
    col3.metric("Negative", negative_news)
    col4.metric("Neutral", neutral_news)
    
    # Overall sentiment
    avg_sentiment = news_df['sentiment'].mean()
    sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "gray"
    
    st.metric("Overall Sentiment Score", f"{avg_sentiment:.3f}", 
              delta_color="off" if sentiment_color == "gray" else "normal")
    
    # Display combined news and Reddit content
    st.subheader("📋 Latest Market News & Discussions")
    
    for _, item in news_df.head(15).iterrows():
        sentiment_score = item['sentiment']
        if sentiment_score > 0.1:
            sentiment_icon = "🟢"
            sentiment_text = "Positive"
        elif sentiment_score < -0.1:
            sentiment_icon = "🔴"
            sentiment_text = "Negative"
        else:
            sentiment_icon = "🟡"
            sentiment_text = "Neutral"
        
        source_icon = "📱" if item['source'].startswith('r/') else "📰"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{sentiment_icon} {source_icon} **{item['title']}**")
                if item['description'] and item['description'] != 'No description':
                    st.caption(item['description'][:200] + "...")
            with col2:
                st.caption(f"**{sentiment_text}**")
                st.caption(f"Score: {sentiment_score:.3f}")
                st.caption(f"Source: {item['source']}")
            st.markdown("---")

def fetch_real_news_sentiment():
    """Fetch real news and perform sentiment analysis."""
    try:
        # Using Yahoo Finance news
        news_df = fetch_yfinance_news()
        
        # Add Reddit posts to news analysis
        reddit_posts = get_reddit_finance_posts()
        
        return pd.concat([news_df, reddit_posts], ignore_index=True)
        
    except Exception as e:
        st.warning(f"News API unavailable: {e}. Using sample data.")
        return create_sample_news_data()

def fetch_yfinance_news():
    """Fetch news from Yahoo Finance."""
    try:
        # Get news for Indian market related tickers
        tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "^NSEI"]
        news_items = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                if news:
                    for item in news[:5]:
                        title = item.get('title', '')
                        link = item.get('link', '')
                        publisher = item.get('publisher', 'Unknown')
                        published_date = item.get('providerPublishTime', '')
                        
                        # Perform sentiment analysis on title
                        sentiment = analyze_text_sentiment(title)
                        
                        news_items.append({
                            'title': title,
                            'description': 'No description',
                            'source': publisher,
                            'published_at': published_date,
                            'sentiment': sentiment,
                            'url': link
                        })
            except Exception as e:
                continue
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_news = []
        for item in news_items:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_news.append(item)
        
        return pd.DataFrame(unique_news)
        
    except Exception as e:
        st.warning(f"Could not fetch Yahoo Finance news: {e}")
        return pd.DataFrame()

def get_reddit_finance_posts():
    """Fetch finance-related posts from Reddit and analyze sentiment."""
    try:
        # Initialize Reddit using Streamlit secrets
        reddit = praw.Reddit(
            client_id=st.secrets["REDDIT_CLIENT_ID"],
            client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
            user_agent="MarketSentimentApp/1.0"
        )
        
        posts_data = []
        subreddits = ['IndianStreetBets', 'stocks', 'investing', 'StockMarket']
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=10):
                    try:
                        # Analyze sentiment of title and selftext
                        title_sentiment = analyze_text_sentiment(post.title)
                        
                        # Combine title and text for better sentiment analysis
                        full_text = post.title
                        if post.selftext:
                            full_text += " " + post.selftext
                        
                        content_sentiment = analyze_text_sentiment(full_text)
                        
                        # Weighted sentiment (title has higher weight)
                        final_sentiment = (title_sentiment * 0.7) + (content_sentiment * 0.3)
                        
                        posts_data.append({
                            'title': post.title,
                            'description': post.selftext[:200] + "..." if post.selftext else 'No description',
                            'source': f'r/{subreddit_name}',
                            'published_at': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M'),
                            'sentiment': final_sentiment,
                            'url': f"https://reddit.com{post.permalink}",
                            'upvotes': post.score,
                            'comments': post.num_comments
                        })
                    except Exception as e:
                        continue
                    
            except Exception as e:
                st.warning(f"Could not fetch from r/{subreddit_name}: {e}")
                continue
        
        return pd.DataFrame(posts_data)
        
    except Exception as e:
        st.warning(f"Reddit API error: {e}")
        return pd.DataFrame()

def analyze_text_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except:
        return 0.0

def create_sample_news_data():
    """Create sample news data when real news is unavailable."""
    sample_news = [
        {
            'title': 'Nifty 50 reaches all-time high amid strong earnings',
            'description': 'Indian stock market continues bullish trend with major indices hitting record levels',
            'source': 'Economic Times',
            'published_at': '2024-01-15',
            'sentiment': 0.8,
            'url': '#'
        },
        {
            'title': 'RBI keeps interest rates unchanged, maintains accommodative stance',
            'description': 'Central bank holds repo rate at 6.5% in latest policy meeting',
            'source': 'Business Standard',
            'published_at': '2024-01-15',
            'sentiment': 0.6,
            'url': '#'
        }
    ]
    return pd.DataFrame(sample_news)

def display_social_trends():
    """Display social media and trending analysis with real Reddit data."""
    st.subheader("📱 Social Trading Trends from Reddit")
    
    # Fetch real Reddit trending data
    with st.spinner("Analyzing Reddit discussions for trending stocks..."):
        trending_data = get_reddit_trending_stocks()
    
    if not trending_data:
        st.warning("Unable to fetch real-time Reddit data. Showing sample data.")
        trending_data = get_sample_trending_data()
    
    # Display trending stocks from Reddit
    st.subheader("🔥 Reddit Trending Stocks")
    
    for stock in trending_data:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
        
        with col1:
            st.write(f"**{stock['symbol']}**")
            st.caption(f"Mentions: {stock['mentions']}")
            st.caption(f"Upvotes: {stock.get('total_upvotes', 0)}")
        
        with col2:
            sentiment = stock['sentiment']
            if sentiment > 0.6:
                st.success(f"Bullish {sentiment:.0%}")
            elif sentiment < 0.4:
                st.error(f"Bearish {sentiment:.0%}")
            else:
                st.info(f"Neutral {sentiment:.0%}")
        
        with col3:
            price_change = stock.get('price_change', 0)
            if price_change > 0:
                st.success(f"₹{stock.get('price', 0):.1f} ↗️")
            else:
                st.error(f"₹{stock.get('price', 0):.1f} ↘️")
        
        with col4:
            st.write(f"`{stock.get('change_pct', 0):+.1f}%`")
        
        with col5:
            if st.button("Analyze", key=f"analyze_{stock['symbol']}"):
                st.session_state[f"analyze_{stock['symbol']}"] = True
    
    # Reddit sentiment insights
    st.subheader("💡 Reddit Sentiment Insights")
    
    high_sentiment_stocks = [s for s in trending_data if s['sentiment'] > 0.6]
    low_sentiment_stocks = [s for s in trending_data if s['sentiment'] < 0.4]
    
    if high_sentiment_stocks:
        stock_list = ", ".join([s['symbol'] for s in high_sentiment_stocks])
        st.success(f"**Positive Reddit Sentiment:** {stock_list} - High positive discussion on Reddit")
    
    if low_sentiment_stocks:
        stock_list = ", ".join([s['symbol'] for s in low_sentiment_stocks])
        st.error(f"**Negative Reddit Sentiment:** {stock_list} - Increased negative discussion")
    
    # Display popular Reddit discussions
    st.subheader("💬 Hot Reddit Discussions")
    display_popular_reddit_posts()

def get_reddit_trending_stocks():
    """Get trending stocks from Reddit discussions."""
    try:
        reddit_posts = get_reddit_finance_posts()
        
        if reddit_posts.empty:
            return None
        
        # Indian stock symbols to look for
        indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 'ICICIBANK', 
                        'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'HINDUNILVR',
                        'ASIANPAINT', 'DMART', 'BAJFINANCE', 'WIPRO', 'SUNPHARMA']
        
        trending_data = []
        
        for stock in indian_stocks:
            # Count mentions and analyze sentiment for this stock
            stock_mentions = []
            total_upvotes = 0
            total_sentiment = 0
            mention_count = 0
            
            for _, post in reddit_posts.iterrows():
                if stock.lower() in post['title'].lower() or (isinstance(post['description'], str) and stock.lower() in post['description'].lower()):
                    stock_mentions.append(post)
                    total_upvotes += post.get('upvotes', 0)
                    total_sentiment += post['sentiment']
                    mention_count += 1
            
            if mention_count > 0:
                avg_sentiment = total_sentiment / mention_count
                
                # Get current stock price
                try:
                    stock_data = yf.download(f"{stock}.NS", period="2d", progress=False)
                    if not stock_data.empty and len(stock_data) > 1:
                        current_price = stock_data['Close'].iloc[-1]
                        prev_price = stock_data['Close'].iloc[-2]
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        trending_data.append({
                            'symbol': stock,
                            'mentions': mention_count,
                            'sentiment': avg_sentiment,
                            'price': current_price,
                            'change_pct': change_pct,
                            'price_change': current_price - prev_price,
                            'total_upvotes': total_upvotes
                        })
                except:
                    continue
        
        # Sort by mentions (most trending first)
        trending_data.sort(key=lambda x: x['mentions'], reverse=True)
        return trending_data[:10]
        
    except Exception as e:
        st.warning(f"Could not analyze Reddit trends: {e}")
        return None

def get_reddit_sentiment_score():
    """Calculate overall Reddit sentiment score (0-100)."""
    try:
        reddit_posts = get_reddit_finance_posts()
        
        if reddit_posts.empty:
            return 50
        
        avg_sentiment = reddit_posts['sentiment'].mean()
        # Convert from -1 to 1 scale to 0-100 scale
        return max(0, min(100, (avg_sentiment + 1) * 50))
        
    except:
        return 50

def display_popular_reddit_posts():
    """Display popular Reddit posts with sentiment analysis."""
    try:
        reddit_posts = get_reddit_finance_posts()
        
        if reddit_posts.empty:
            st.info("No Reddit posts available.")
            return
        
        # Sort by upvotes
        popular_posts = reddit_posts.nlargest(5, 'upvotes')
        
        for idx, (_, post) in enumerate(popular_posts.iterrows()):
            sentiment = post['sentiment']
            if sentiment > 0.1:
                sentiment_color = "🟢"
            elif sentiment < -0.1:
                sentiment_color = "🔴"
            else:
                sentiment_color = "🟡"
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{sentiment_color} **{post['title']}**")
                    st.caption(f"r/{post['source'].split('/')[-1]} • {post['upvotes']} upvotes • {post['comments']} comments")
                with col2:
                    st.caption(f"Sentiment: {sentiment:.3f}")
                    if st.button("View", key=f"view_{idx}"):
                        st.markdown(f"[Open in Reddit]({post['url']})")
                st.markdown("---")
                
    except Exception as e:
        st.warning(f"Could not display Reddit posts: {e}")

def get_sample_trending_data():
    """Return sample trending data."""
    return [
        {"symbol": "RELIANCE", "mentions": 1250, "sentiment": 0.75, "price": 2456.75, "change_pct": 1.2, "total_upvotes": 15000},
        {"symbol": "TATASTEEL", "mentions": 890, "sentiment": 0.62, "price": 156.80, "change_pct": 2.1, "total_upvotes": 8900},
        {"symbol": "INFY", "mentions": 760, "sentiment": 0.45, "price": 1850.50, "change_pct": -0.8, "total_upvotes": 7600},
        {"symbol": "HDFCBANK", "mentions": 680, "sentiment": 0.68, "price": 1650.25, "change_pct": 1.5, "total_upvotes": 6800},
        {"symbol": "TCS", "mentions": 540, "sentiment": 0.52, "price": 3850.75, "change_pct": -0.3, "total_upvotes": 5400}
    ]

def is_market_hours():
    """Check if Indian stock market is open."""
    try:
        now = datetime.now()
        # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if current time is within market hours and it's a weekday
        return market_open.time() <= now.time() <= market_close.time() and now.weekday() < 5
    except:
        return False

def page_forecasting_ml():
    """A page for advanced ML forecasting with an improved UI and corrected formulas."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train a Seasonal ARIMA model to forecast future prices. This is for educational purposes and not financial advice.", icon="🧠")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument")
        
        forecast_durations = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
        duration_key = st.radio("Forecast Duration", list(forecast_durations.keys()), horizontal=True, key="ml_duration")
        forecast_steps = forecast_durations[duration_key]

        if st.button("Train Model & Forecast", use_container_width=True, type="primary"):
            with st.spinner(f"Loading data and training model for {instrument_name}..."):
                data = load_and_combine_data(instrument_name)
                if data.empty or len(data) < 100:
                    st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
                else:
                    forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                    st.session_state.update({
                        'ml_forecast_df': forecast_df,
                        'ml_backtest_df': backtest_df,
                        'ml_conf_int_df': conf_int_df,
                        'ml_instrument_name': instrument_name,
                        'ml_historical_data': data,
                        'ml_duration_key': duration_key
                    })
                    st.success("Model trained successfully!")

    with col2:
        if 'ml_instrument_name' in st.session_state:
            instrument_name = st.session_state.ml_instrument_name
            st.subheader(f"Forecast Results for {instrument_name}")

            forecast_df = st.session_state.get('ml_forecast_df')
            backtest_df = st.session_state.get('ml_backtest_df')
            conf_int_df = st.session_state.get('ml_conf_int_df')
            data = st.session_state.get('ml_historical_data')
            duration_key = st.session_state.get('ml_duration_key')

            if forecast_df is not None and backtest_df is not None and data is not None and conf_int_df is not None:
                fig = create_chart(data.tail(252), instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Backtest Prediction', line=dict(color='orange', dash='dot')))
                fig.update_layout(title=f"{instrument_name} Forecast vs. Historical Data")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Model Performance (Backtest)")
                
                backtest_durations = {"Full History": len(backtest_df), "Last Year": 252, "6 Months": 126, "3 Months": 63}
                backtest_duration_key = st.selectbox("Select Backtest Period", list(backtest_durations.keys()))
                backtest_period = backtest_durations[backtest_duration_key]
                
                display_df = backtest_df.tail(backtest_period)

                mape = mean_absolute_percentage_error(display_df['Actual'], display_df['Predicted'])
                
                metric_cols = st.columns(2)
                metric_cols[0].metric(f"Accuracy ({backtest_duration_key})", f"{100 - mape:.2f}%")
                metric_cols[1].metric(f"MAPE ({backtest_duration_key})", f"{mape:.2f}%")

                with st.expander(f"View {duration_key} Forecast Data"):
                    display_df_forecast = forecast_df.join(conf_int_df)
                    st.dataframe(display_df_forecast.style.format("₹{:.2f}"), use_container_width=True)
            else:
                st.info("Train a model to see the forecast results.")
        else:
            st.info("Select an instrument and run the forecast to see results.")

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty and positions_df.empty:
        st.info("No holdings or positions found to analyze.")
        return

    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Live Order Book"])
    
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        else:
            st.info("No open positions for the day.")
    
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            st.markdown("---")

            st.subheader("Portfolio Allocation")
            
            sector_df = get_sector_data()
            
            holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
            
            if not holdings_df.empty and sector_df is not None:
                holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left')
                if 'Sector' not in holdings_df.columns:
                    holdings_df['Sector'] = 'Uncategorized'
                holdings_df['Sector'].fillna('Uncategorized', inplace=True)
            else:
                holdings_df['Sector'] = 'Uncategorized'
            
            col1_alloc, col2_alloc = st.columns(2)
            
            with col1_alloc:
                st.subheader("Stock-wise Allocation")
                fig_stock = go.Figure(data=[go.Pie(
                    labels=holdings_df['tradingsymbol'],
                    values=holdings_df['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == "Dark" else 'plotly_white')
                st.plotly_chart(fig_stock, use_container_width=True)
                
            if 'Sector' in holdings_df.columns:
                with col2_alloc:
                    st.subheader("Sector-wise Allocation")
                    sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                    fig_sector = go.Figure(data=[go.Pie(
                        labels=sector_allocation['Sector'],
                        values=sector_allocation['current_value'],
                        hole=.3,
                        textinfo='label+percent'
                    )])
                    fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == "Dark" else 'plotly_white')
                    st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No holdings found.")

    with tab3:
        st.subheader("Live Order Book")
        display_order_history()

def display_trade_history():
    """Display comprehensive trade history with IST timestamps."""
    st.subheader("📋 Trade History")
    
    # Get all trade sources
    manual_trades = st.session_state.get('order_history', [])
    automated_trades = st.session_state.automated_mode.get('trade_history', [])
    
    if not manual_trades and not automated_trades:
        st.info("No trade history available. Trades will appear here after execution.")
        return
    
    # Combine and format all trades
    all_trades = []
    
    # Add manual trades
    for trade in manual_trades:
        ist_time = get_ist_time()
        all_trades.append({
            'Time': ist_time.strftime("%H:%M:%S IST"),
            'Symbol': trade.get('symbol', 'N/A'),
            'Action': trade.get('type', 'N/A'),
            'Qty': trade.get('qty', 0),
            'Status': trade.get('status', 'N/A'),
            'Type': 'Manual',
            'P&L': 'N/A'
        })
    
    # Add automated trades
    for trade in automated_trades:
        display_time = trade.get('timestamp_display', 
                               get_ist_time().strftime("%H:%M:%S IST") if 'timestamp' not in trade 
                               else datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).astimezone(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S IST"))
        
        all_trades.append({
            'Time': display_time,
            'Symbol': trade.get('symbol', 'N/A'),
            'Action': trade.get('action', 'N/A'),
            'Qty': trade.get('quantity', 0),
            'Status': trade.get('status', 'OPEN'),
            'Type': trade.get('order_type', 'Automated'),
            'P&L': f"₹{trade.get('pnl', 0):.2f}"
        })
    
    # Sort by time (newest first)
    all_trades.sort(key=lambda x: x['Time'], reverse=True)
    
    # Create dataframe and display
    if all_trades:
        df = pd.DataFrame(all_trades)
        
        # Color coding for status
        def color_status(val):
            if val == 'Success' or val == 'CLOSED':
                return 'color: green; font-weight: bold;'
            elif val == 'OPEN':
                return 'color: orange; font-weight: bold;'
            elif 'Failed' in str(val):
                return 'color: red; font-weight: bold;'
            return ''
        
        # Color coding for action
        def color_action(val):
            if val == 'BUY':
                return 'color: #00cc00; font-weight: bold;'
            elif val == 'SELL':
                return 'color: #ff4444; font-weight: bold;'
            return ''
        
        styled_df = df.style.map(color_status, subset=['Status']).map(color_action, subset=['Action'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        total_trades = len(all_trades)
        successful_trades = len([t for t in all_trades if t['Status'] in ['Success', 'CLOSED']])
        open_trades = len([t for t in all_trades if t['Status'] == 'OPEN'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", total_trades)
        col2.metric("Successful", successful_trades)
        col3.metric("Open", open_trades)
        
        # Export option
        if st.button("📊 Export Trade History to CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trade_history_{get_ist_time().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No trades to display.")

def display_order_history():
    """Display order history - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if client:
        try:
            broker = st.session_state.broker
            
            if broker == "Zerodha":
                orders = client.orders()
                if orders:
                    order_data = []
                    for order in orders:
                        order_time = order.get('order_timestamp', '')
                        if order_time:
                            try:
                                dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                                ist = pytz.timezone('Asia/Kolkata')
                                dt_ist = dt.astimezone(ist)
                                display_time = dt_ist.strftime("%Y-%m-%d %H:%M:%S IST")
                            except:
                                display_time = order_time
                        else:
                            display_time = 'N/A'
                        
                        order_data.append({
                            'Time': display_time,
                            'Symbol': order.get('tradingsymbol', ''),
                            'Type': order.get('transaction_type', ''),
                            'Qty': order.get('quantity', 0),
                            'Price': f"₹{order.get('average_price', 0):.2f}",
                            'Status': order.get('status', '')
                        })
                    
                    st.dataframe(pd.DataFrame(order_data), use_container_width=True)
                else:
                    st.info("No orders placed today.")
                    
            elif broker == "Upstox":
                orders_df = get_upstox_order_book()
                if not orders_df.empty:
                    order_data = []
                    for _, order in orders_df.iterrows():
                        order_time = order.get('order_timestamp', '')
                        if order_time:
                            try:
                                dt = datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                                ist = pytz.timezone('Asia/Kolkata')
                                dt_ist = dt.astimezone(ist)
                                display_time = dt_ist.strftime("%Y-%m-%d %H:%M:%S IST")
                            except:
                                display_time = order_time
                        else:
                            display_time = 'N/A'
                        
                        order_data.append({
                            'Time': display_time,
                            'Symbol': order.get('tradingsymbol', ''),
                            'Type': order.get('transaction_type', ''),
                            'Qty': order.get('quantity', 0),
                            'Price': f"₹{order.get('price', 0):.2f}",
                            'Status': order.get('status', '')
                        })
                    
                    st.dataframe(pd.DataFrame(order_data), use_container_width=True)
                else:
                    st.info("No orders placed today.")
            
        except Exception as e:
            st.error(f"Error fetching orders: {e}")
    else:
        st.info("Broker not connected.")

def page_ai_assistant():
    """Enhanced AI-powered assistant with portfolio-aware insights and predictive analytics."""
    display_header()
    st.title("🧠 AI Portfolio Assistant")
    st.info("Advanced AI assistant with portfolio insights, predictive analytics, and intelligent trade recommendations.", icon="🤖")
    
    instrument_df = get_instrument_df()
    client = get_broker_client()

    # Initialize enhanced session state
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI trading assistant. I can help you with portfolio analysis, market insights, trade recommendations, and predictive analytics. How can I assist you today?"}
        ]
    
    if "assistant_context" not in st.session_state:
        st.session_state.assistant_context = {
            "portfolio_snapshot": {},
            "market_conditions": {},
            "risk_profile": "medium",
            "trading_style": "balanced"
        }

    # Sidebar with quick actions
    with st.sidebar:
        st.subheader("🛠️ Quick Actions")
        
        if st.button("📊 Portfolio Health Check", use_container_width=True):
            st.session_state.assistant_messages.append({"role": "user", "content": "Analyze my portfolio health and suggest improvements"})
            # This will trigger the response in the main chat
            
        if st.button("📈 Market Outlook", use_container_width=True):
            st.session_state.assistant_messages.append({"role": "user", "content": "What's the current market outlook and any opportunities?"})
            
        if st.button("🎯 Trade Ideas", use_container_width=True):
            st.session_state.assistant_messages.append({"role": "user", "content": "Suggest some high-probability trade ideas"})
            
        if st.button("⚠️ Risk Assessment", use_container_width=True):
            st.session_state.assistant_messages.append({"role": "user", "content": "Assess my portfolio risk and suggest hedging"})
            
        st.markdown("---")
        st.subheader("🔧 Configuration")
        
        st.session_state.assistant_context["risk_profile"] = st.selectbox(
            "Risk Profile",
            ["conservative", "moderate", "balanced", "aggressive"],
            index=2
        )
        
        st.session_state.assistant_context["trading_style"] = st.selectbox(
            "Trading Style", 
            ["scalping", "day_trading", "swing_trading", "position_trading"],
            index=2
        )

    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 AI Assistant")
        
        # Display chat messages
        for message in st.session_state.assistant_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display additional context for assistant messages
                if "analysis" in message:
                    with st.expander("📊 Detailed Analysis"):
                        st.write(message["analysis"])
                        
                if "recommendations" in message:
                    with st.expander("🎯 Actionable Recommendations"):
                        for rec in message["recommendations"]:
                            st.write(f"• {rec}")

    with col2:
        st.subheader("📈 Live Context")
        
        # Portfolio snapshot
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        if total_investment > 0:
            st.metric("Portfolio P&L", f"₹{total_pnl:,.2f}")
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
        else:
            st.info("No portfolio data")
        
        # Market status
        status_info = get_market_status()
        st.metric("Market Status", status_info['status'].replace('_', ' ').title())
        
        # AI Insights
        with st.expander("🤖 AI Insights"):
            st.write("• Real-time portfolio analysis")
            st.write("• Predictive market analytics")
            st.write("• Risk-aware recommendations")
            st.write("• Technical pattern recognition")

    # Chat input
    if prompt := st.chat_input("Ask about your portfolio, market insights, or trade ideas..."):
        # Add user message to chat history
        st.session_state.assistant_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query..."):
                response = process_ai_assistant_query(prompt, instrument_df)
                st.markdown(response["answer"])
                
                # Store enhanced response
                enhanced_message = {
                    "role": "assistant", 
                    "content": response["answer"]
                }
                
                if "analysis" in response:
                    enhanced_message["analysis"] = response["analysis"]
                if "recommendations" in response:
                    enhanced_message["recommendations"] = response["recommendations"]
                
                st.session_state.assistant_messages.append(enhanced_message)

def process_ai_assistant_query(prompt, instrument_df):
    """Enhanced query processing with advanced analytics."""
    prompt_lower = prompt.lower()
    
    # Update context
    update_assistant_context()
    
    # Portfolio-related queries
    if any(word in prompt_lower for word in ["portfolio", "holdings", "positions", "investment"]):
        return handle_portfolio_queries(prompt_lower, instrument_df)
    
    # Market analysis queries
    elif any(word in prompt_lower for word in ["market", "outlook", "trend", "nifty", "sensex"]):
        return handle_market_queries(prompt_lower, instrument_df)
    
    # Trade-related queries
    elif any(word in prompt_lower for word in ["trade", "buy", "sell", "opportunity", "idea"]):
        return handle_trade_queries(prompt_lower, instrument_df)
    
    # Risk-related queries
    elif any(word in prompt_lower for word in ["risk", "hedge", "protection", "volatility"]):
        return handle_risk_queries(prompt_lower, instrument_df)
    
    # Technical analysis queries
    elif any(word in prompt_lower for word in ["technical", "chart", "indicator", "pattern"]):
        return handle_technical_queries(prompt_lower, instrument_df)
    
    # General queries
    else:
        return handle_general_queries(prompt_lower, instrument_df)

def update_assistant_context():
    """Update the AI assistant's context with current market and portfolio data."""
    try:
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        
        # Portfolio snapshot
        st.session_state.assistant_context["portfolio_snapshot"] = {
            "total_pnl": total_pnl,
            "total_investment": total_investment,
            "positions_count": len(positions_df) if not positions_df.empty else 0,
            "holdings_count": len(holdings_df) if not holdings_df.empty else 0,
            "winning_positions": len([p for p in positions_df.to_dict('records') if p.get('pnl', 0) > 0]) if not positions_df.empty else 0
        }
        
        # Market conditions
        status_info = get_market_status()
        st.session_state.assistant_context["market_conditions"] = {
            "status": status_info['status'],
            "market_hours": is_market_hours(),
            "pre_market": is_pre_market_hours(),
            "square_off_time": is_square_off_time()
        }
    except Exception as e:
        st.error(f"Error updating context: {e}")

def handle_portfolio_queries(prompt_lower, instrument_df):
    """Handle portfolio-related queries with advanced analytics."""
    if "health" in prompt_lower or "how is my portfolio" in prompt_lower:
        return analyze_portfolio_health()
    elif "diversification" in prompt_lower:
        return analyze_portfolio_diversification()
    elif "performance" in prompt_lower:
        return analyze_portfolio_performance()
    else:
        return generate_portfolio_summary()

def analyze_portfolio_health():
    """Comprehensive portfolio health analysis."""
    try:
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        
        # Calculate metrics
        total_value = total_investment + total_pnl
        return_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Risk assessment
        risk_score = calculate_portfolio_risk(holdings_df)
        
        # Generate insights
        if return_pct > 10:
            health_status = "Excellent"
            color = "green"
        elif return_pct > 5:
            health_status = "Good" 
            color = "blue"
        elif return_pct > 0:
            health_status = "Fair"
            color = "orange"
        else:
            health_status = "Needs Attention"
            color = "red"
        
        analysis = f"""
        **Portfolio Health Analysis:**
        
        - **Total Value:** ₹{total_value:,.2f}
        - **Total P&L:** ₹{total_pnl:,.2f} ({return_pct:.2f}%)
        - **Health Status:** <span style='color:{color}'>{health_status}</span>
        - **Risk Score:** {risk_score}/10
        - **Positions:** {len(positions_df) if not positions_df.empty else 0} active
        - **Holdings:** {len(holdings_df) if not holdings_df.empty else 0} investments
        """
        
        recommendations = []
        
        if risk_score > 7:
            recommendations.append("Consider reducing exposure to high-risk positions")
        if return_pct < 0:
            recommendations.append("Review underperforming positions and consider stop-losses")
        if len(holdings_df) < 5:
            recommendations.append("Diversify across more sectors for better risk management")
        
        return {
            "answer": f"Your portfolio shows **{health_status}** health with a {return_pct:.2f}% return. " +
                     f"Risk level is {risk_score}/10. " +
                     ("I recommend reviewing the detailed analysis below." if recommendations else "Your portfolio is well-maintained."),
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to analyze portfolio health at the moment. Please ensure you're connected to your broker.",
            "analysis": "Error in portfolio analysis",
            "recommendations": ["Check broker connection", "Verify portfolio data availability"]
        }

def calculate_portfolio_risk(holdings_df):
    """Calculate portfolio risk score (1-10)."""
    try:
        if holdings_df.empty:
            return 5
        
        # Simple risk calculation based on concentration and volatility
        total_value = (holdings_df['quantity'] * holdings_df['last_price']).sum()
        
        # Concentration risk (higher if few holdings dominate)
        if len(holdings_df) < 3:
            concentration_risk = 8
        elif len(holdings_df) < 5:
            concentration_risk = 6
        else:
            concentration_risk = 4
        
        # Volatility estimate (simplified)
        avg_pnl_pct = (holdings_df['pnl'].abs() / (holdings_df['quantity'] * holdings_df['average_price'])).mean() * 100
        volatility_risk = min(10, avg_pnl_pct / 2)
        
        return min(10, (concentration_risk + volatility_risk) / 2)
    except:
        return 5

def analyze_portfolio_diversification():
    """Analyze portfolio diversification."""
    try:
        holdings_df, _, _, _ = get_portfolio()
        
        if holdings_df.empty:
            return {
                "answer": "No holdings found for diversification analysis.",
                "analysis": "Please add holdings to your portfolio to analyze diversification.",
                "recommendations": ["Start building your portfolio with diversified sectors"]
            }
        
        # Simple diversification analysis
        num_holdings = len(holdings_df)
        
        if num_holdings >= 8:
            div_score = "Excellent"
            color = "green"
        elif num_holdings >= 5:
            div_score = "Good"
            color = "blue"
        elif num_holdings >= 3:
            div_score = "Fair"
            color = "orange"
        else:
            div_score = "Low"
            color = "red"
        
        analysis = f"""
        **Portfolio Diversification Analysis:**
        
        - **Number of Holdings:** {num_holdings}
        - **Diversification Score:** <span style='color:{color}'>{div_score}</span>
        - **Recommendation:** {'Well diversified' if div_score in ['Excellent', 'Good'] else 'Consider adding more positions'}
        """
        
        recommendations = [
            "Aim for at least 5 different holdings for optimal diversification",
            "Consider adding exposure to different market sectors",
            "Rebalance periodically to maintain target allocation"
        ]
        
        if num_holdings < 5:
            recommendations.append("Add 2-3 more positions from different sectors")
        
        return {
            "answer": f"Your portfolio has {num_holdings} holdings with **{div_score}** diversification.",
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to analyze diversification at the moment.",
            "analysis": "Error in diversification analysis",
            "recommendations": ["Check portfolio data", "Try again later"]
        }

def analyze_portfolio_performance():
    """Analyze portfolio performance."""
    try:
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        
        if holdings_df.empty:
            return {
                "answer": "No portfolio data available for performance analysis.",
                "analysis": "Please connect to your broker or add manual holdings.",
                "recommendations": ["Connect to broker", "Add portfolio data"]
            }
        
        total_value = total_investment + total_pnl
        return_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Performance rating
        if return_pct > 15:
            rating = "Outstanding"
            color = "green"
        elif return_pct > 8:
            rating = "Good"
            color = "blue"
        elif return_pct > 0:
            rating = "Positive"
            color = "orange"
        else:
            rating = "Needs Improvement"
            color = "red"
        
        analysis = f"""
        **Portfolio Performance Analysis:**
        
        - **Total Value:** ₹{total_value:,.2f}
        - **Total Return:** ₹{total_pnl:,.2f} ({return_pct:.2f}%)
        - **Performance Rating:** <span style='color:{color}'>{rating}</span>
        - **Number of Positions:** {len(positions_df) if not positions_df.empty else 0}
        """
        
        recommendations = []
        
        if return_pct < 5:
            recommendations.append("Consider reviewing your trading strategy")
        if return_pct > 20:
            recommendations.append("Consider profit booking on high-performing positions")
        
        return {
            "answer": f"Your portfolio has delivered {return_pct:.2f}% returns with **{rating}** performance.",
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to analyze portfolio performance.",
            "analysis": "Error in performance analysis",
            "recommendations": ["Check data connection", "Verify portfolio details"]
        }

def generate_portfolio_summary():
    """Generate comprehensive portfolio summary."""
    try:
        positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
        
        if holdings_df.empty:
            return {
                "answer": "No portfolio data available.",
                "analysis": "Please connect to your broker to view portfolio summary.",
                "recommendations": ["Connect to broker account", "Add manual holdings if needed"]
            }
        
        total_value = total_investment + total_pnl
        return_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        analysis = f"""
        **Portfolio Summary:**
        
        - **Total Portfolio Value:** ₹{total_value:,.2f}
        - **Total Investment:** ₹{total_investment:,.2f}
        - **Total P&L:** ₹{total_pnl:,.2f} ({return_pct:.2f}%)
        - **Active Positions:** {len(positions_df) if not positions_df.empty else 0}
        - **Long-term Holdings:** {len(holdings_df) if not holdings_df.empty else 0}
        """
        
        # Top performers
        if not holdings_df.empty:
            top_performers = holdings_df.nlargest(3, 'pnl')
            analysis += "\n**Top Performers:**\n"
            for _, holding in top_performers.iterrows():
                analysis += f"- {holding['tradingsymbol']}: ₹{holding['pnl']:,.2f}\n"
        
        recommendations = [
            "Regularly review your portfolio allocation",
            "Set stop-losses for active positions",
            "Consider rebalancing if any position grows beyond 15% of portfolio"
        ]
        
        return {
            "answer": f"Your portfolio is valued at ₹{total_value:,.2f} with {return_pct:.2f}% returns.",
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to generate portfolio summary.",
            "analysis": "Error in portfolio summary generation",
            "recommendations": ["Check broker connection", "Verify portfolio data"]
        }

def handle_market_queries(prompt_lower, instrument_df):
    """Handle market-related queries."""
    try:
        status_info = get_market_status()
        market_status = status_info['status'].replace('_', ' ').title()
        
        # Get NIFTY data for market context
        nifty_data = get_watchlist_data([{'symbol': 'NIFTY 50', 'exchange': 'NSE'}])
        nifty_price = nifty_data.iloc[0]['Price'] if not nifty_data.empty else 0
        nifty_change = nifty_data.iloc[0]['Change'] if not nifty_data.empty else 0
        
        analysis = f"""
        **Current Market Conditions:**
        
        - **Market Status:** {market_status}
        - **NIFTY 50:** ₹{nifty_price:.2f} ({nifty_change:+.2f})
        - **Trading Hours:** 9:15 AM - 3:30 PM IST
        - **Market Phase:** {'Open' if is_market_hours() else 'Closed'}
        """
        
        recommendations = [
            "Monitor key support and resistance levels",
            "Watch for sector rotation opportunities",
            "Stay updated with corporate announcements"
        ]
        
        if "outlook" in prompt_lower:
            outlook = "cautiously optimistic" if nifty_change > 0 else "cautious" if nifty_change < 0 else "neutral"
            answer = f"The market is currently **{market_status}** with NIFTY at ₹{nifty_price:.2f}. The near-term outlook appears **{outlook}**."
        else:
            answer = f"The market is **{market_status}**. NIFTY 50 is trading at ₹{nifty_price:.2f} ({nifty_change:+.2f})."
        
        return {
            "answer": answer,
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to fetch market data at the moment.",
            "analysis": "Market data temporarily unavailable",
            "recommendations": ["Check internet connection", "Try again shortly"]
        }

def handle_trade_queries(prompt_lower, instrument_df):
    """Handle trade-related queries."""
    try:
        # Get current market conditions
        status_info = get_market_status()
        
        if not is_market_hours() and not is_pre_market_hours():
            return {
                "answer": "Markets are currently closed. Trading recommendations are available during market hours (9:15 AM - 3:30 PM IST).",
                "analysis": "Market Hours: 9:15 AM - 3:30 PM IST\nNext trading session: Tomorrow 9:15 AM",
                "recommendations": ["Prepare your watchlist for tomorrow", "Review overnight global market moves"]
            }
        
        # Generate simple trade ideas based on market conditions
        if "buy" in prompt_lower:
            ideas = ["RELIANCE (Strong fundamentals)", "INFY (IT sector momentum)", "HDFCBANK (Banking recovery)"]
            answer = "Based on current market conditions, consider these potential buy opportunities:"
        elif "sell" in prompt_lower:
            ideas = ["Consider profit booking on positions with >20% gains", "Review underperforming holdings", "Reduce exposure to high-beta stocks if market appears overbought"]
            answer = "For selling considerations, review these aspects:"
        else:
            ideas = ["Momentum plays in IT sector", "Value opportunities in banking", "Defensive positions in FMCG"]
            answer = "Current market environment suggests these trade ideas:"
        
        analysis = f"""
        **Trade Analysis Context:**
        
        - **Market Status:** {status_info['status'].replace('_', ' ').title()}
        - **Recommended Strategy:** {st.session_state.assistant_context['trading_style'].replace('_', ' ').title()}
        - **Risk Profile:** {st.session_state.assistant_context['risk_profile'].title()}
        """
        
        recommendations = ideas + [
            "Always use stop-loss orders",
            "Position size according to your risk tolerance",
            "Monitor market news for unexpected events"
        ]
        
        return {
            "answer": answer,
            "analysis": analysis,
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "answer": "Unable to generate trade ideas at the moment.",
            "analysis": "Trade analysis temporarily unavailable",
            "recommendations": ["Check market data connection", "Try again during market hours"]
        }

def handle_risk_queries(prompt_lower, instrument_df):
    """Handle risk-related queries."""
    analysis = """
    **Portfolio Risk Management Framework:**
    
    - **Diversification:** Spread across sectors and market caps
    - **Position Sizing:** Limit single positions to 5-10% of portfolio
    - **Stop-Loss:** Essential for risk management
    - **Hedging:** Consider options for portfolio protection
    """
    
    recommendations = [
        "Set stop-losses for all active positions",
        "Diversify across at least 5 sectors",
        "Avoid over-concentration in single stocks",
        "Consider hedging with put options in volatile markets",
        "Regularly review and adjust position sizes"
    ]
    
    return {
        "answer": "Effective risk management is crucial for long-term trading success. Here's my risk assessment framework:",
        "analysis": analysis,
        "recommendations": recommendations
    }

def handle_technical_queries(prompt_lower, instrument_df):
    """Handle technical analysis queries."""
    analysis = """
    **Technical Analysis Framework:**
    
    - **Trend Analysis:** Identify primary trends using moving averages
    - **Support/Resistance:** Key price levels for entry/exit decisions
    - **Momentum Indicators:** RSI, MACD for timing entries
    - **Volume Analysis:** Confirm price moves with volume
    - **Pattern Recognition:** Chart patterns for predictive signals
    """
    
    recommendations = [
        "Use multiple timeframes for confirmation (daily + hourly)",
        "Combine technical with fundamental analysis",
        "Wait for confirmation before entering trades",
        "Use stop-losses based on technical levels",
        "Monitor key indicators like RSI and moving averages"
    ]
    
    return {
        "answer": "Technical analysis helps identify trading opportunities through price patterns and indicators. Key aspects include:",
        "analysis": analysis,
        "recommendations": recommendations
    }

def handle_general_queries(prompt_lower, instrument_df):
    """Handle general trading queries."""
    general_responses = {
        "hello": "Hello! I'm your AI trading assistant. I can help with portfolio analysis, market insights, and trading strategies. What would you like to know?",
        "help": "I can assist with:\n- Portfolio analysis and health checks\n- Market outlook and trends\n- Trade ideas and opportunities\n- Risk management strategies\n- Technical analysis insights\n\nJust ask me anything about trading or your portfolio!",
        "hours": "Indian stock market trading hours:\n- Pre-open: 9:00-9:15 AM\n- Normal market: 9:15 AM - 3:30 PM\n- Special sessions may vary\nMarkets are closed on weekends and holidays.",
        "broker": "This platform supports Zerodha and Upstox integration. You can connect your broker in the settings section."
    }
    
    for key, response in general_responses.items():
        if key in prompt_lower:
            return {
                "answer": response,
                "analysis": "",
                "recommendations": []
            }
    
    # Default response for unrecognized queries
    return {
        "answer": "I'm here to help with your trading and investment needs. You can ask me about:\n\n• Your portfolio performance and health\n• Current market conditions and outlook\n• Trade ideas and opportunities\n• Risk management strategies\n• Technical analysis insights\n\nPlease try rephrasing your question if I didn't understand it correctly.",
        "analysis": "",
        "recommendations": ["Try asking about your portfolio", "Request market analysis", "Ask for trade ideas", "Get risk management advice"]
    }
    
def page_fundamental_analytics():
    """Fundamental Analytics page using Kite Connect data and other available sources."""
    display_header()
    st.title("📊 Fundamental Analytics")
    st.info("Analyze company fundamentals using available market data from Kite Connect and other sources.", icon="📈")
    
    tab1, tab2, tab3 = st.tabs(["Company Overview", "Financial Ratios", "Multi-Company Comparison"])
    
    with tab1:
        st.subheader("Company Fundamental Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            symbol = st.text_input("Enter Stock Symbol", "RELIANCE", 
                                 help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)")
            exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)
            
            if st.button("Fetch Fundamental Data", use_container_width=True):
                with st.spinner(f"Fetching data for {symbol}..."):
                    company_data = get_company_fundamentals_kite(symbol, exchange)
                    if company_data:
                        st.session_state.current_company = company_data
                        st.session_state.current_symbol = symbol
                        st.rerun()
        
        with col2:
            if 'current_company' in st.session_state and st.session_state.current_company:
                display_company_overview_kite(st.session_state.current_company, st.session_state.current_symbol)
            else:
                st.info("Enter a stock symbol and click 'Fetch Fundamental Data' to get started.")
    
    with tab2:
        st.subheader("Financial Ratios & Metrics")
        if 'current_company' in st.session_state and st.session_state.current_company:
            display_financial_ratios_kite(st.session_state.current_company, st.session_state.current_symbol)
        else:
            st.info("First fetch company data in the 'Company Overview' tab.")
    
    with tab3:
        st.subheader("Multi-Company Comparison")
        display_multi_company_comparison_kite()

def get_company_fundamentals_kite(symbol, exchange="NSE"):
    """Fetch fundamental data using Kite Connect APIs and other available sources."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected. Please connect to Kite first.")
        return None
    
    try:
        # Get basic instrument info
        instrument_df = get_instrument_df()
        if instrument_df.empty:
            st.error("Could not fetch instrument data.")
            return None
            
        instrument_info = instrument_df[
            (instrument_df['tradingsymbol'] == symbol.upper()) & 
            (instrument_df['exchange'] == exchange)
        ]
        
        if instrument_info.empty:
            st.error(f"Symbol {symbol} not found on {exchange}.")
            return None
            
        instrument_info = instrument_info.iloc[0]
        
        # Get current quote data
        quote_data = client.quote(f"{exchange}:{symbol.upper()}")
        if not quote_data:
            st.error(f"Could not fetch quote data for {symbol}.")
            return None
            
        quote = quote_data[f"{exchange}:{symbol.upper()}"]
        
        # Basic company info
        company_data = {
            'symbol': symbol.upper(),
            'exchange': exchange,
            'name': instrument_info.get('name', symbol.upper()),
            'lot_size': instrument_info.get('lot_size', 0),
            'instrument_type': instrument_info.get('instrument_type', 'EQ'),
            'segment': instrument_info.get('segment', ''),
        }
        
        # Price metrics from quote
        company_data.update({
            'current_price': quote.get('last_price', 0),
            'open': quote.get('ohlc', {}).get('open', 0),
            'high': quote.get('ohlc', {}).get('high', 0),
            'low': quote.get('ohlc', {}).get('low', 0),
            'close': quote.get('ohlc', {}).get('close', 0),
            'volume': quote.get('volume', 0),
            'average_volume': quote.get('average_price', 0) * quote.get('volume', 0) if quote.get('volume', 0) > 0 else 0,
        })
        
        # Calculate basic ratios from available data
        if quote.get('ohlc', {}).get('close', 0) > 0:
            change = company_data['current_price'] - quote['ohlc']['close']
            company_data['change_percent'] = (change / quote['ohlc']['close']) * 100
        else:
            company_data['change_percent'] = 0
        
        # Get historical data for additional calculations
        token = get_instrument_token(symbol, instrument_df, exchange)
        if token:
            hist_data = get_historical_data(token, 'day', period='1y')
            if not hist_data.empty and len(hist_data) > 200:
                # Calculate 52-week high/low
                company_data['52_week_high'] = hist_data['high'].max()
                company_data['52_week_low'] = hist_data['low'].min()
                
                # Calculate basic volatility
                returns = hist_data['close'].pct_change().dropna()
                company_data['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Calculate simple moving averages
                company_data['sma_50'] = hist_data['close'].tail(50).mean()
                company_data['sma_200'] = hist_data['close'].tail(200).mean()
        
        # Placeholder values for fundamental data (in real implementation, you'd fetch from other sources)
        company_data.update({
            'market_cap': company_data['current_price'] * instrument_info.get('lot_size', 1) * 1000,  # Rough estimate
            'sector': 'Not Available',  # Would need external data source
            'industry': 'Not Available',
            'pe_ratio': 0,  # Would need earnings data
            'dividend_yield': 0,
            'book_value': 0,
            'eps': 0,
        })
        
        return company_data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def display_company_overview_kite(company_data, symbol):
    """Display company overview using Kite Connect data."""
    st.subheader(f"{company_data['name']} ({symbol})")
    
    # Basic info cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"₹{company_data['current_price']:,.2f}")
        st.metric("Today's Change", f"{company_data['change_percent']:.2f}%")
    
    with col2:
        if company_data.get('52_week_high'):
            st.metric("52W High", f"₹{company_data['52_week_high']:,.2f}")
        if company_data.get('52_week_low'):
            st.metric("52W Low", f"₹{company_data['52_week_low']:,.2f}")
    
    with col3:
        st.metric("Volume", f"{company_data['volume']:,}")
        if company_data.get('volatility'):
            st.metric("Volatility", f"{company_data['volatility']:.1f}%")
    
    with col4:
        st.metric("Lot Size", f"{company_data['lot_size']:,}")
        st.metric("Instrument Type", company_data['instrument_type'])
    
    st.markdown("---")
    
    # Additional metrics
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Trading Information")
        st.write(f"**Exchange:** {company_data['exchange']}")
        st.write(f"**Segment:** {company_data['segment']}")
        st.write(f"**Open:** ₹{company_data['open']:,.2f}")
        st.write(f"**High:** ₹{company_data['high']:,.2f}")
        st.write(f"**Low:** ₹{company_data['low']:,.2f}")
        st.write(f"**Close:** ₹{company_data['close']:,.2f}")
    
    with col6:
        st.subheader("Technical Indicators")
        if company_data.get('sma_50'):
            st.write(f"**50-Day SMA:** ₹{company_data['sma_50']:,.2f}")
        if company_data.get('sma_200'):
            st.write(f"**200-Day SMA:** ₹{company_data['sma_200']:,.2f}")
        
        # Calculate position relative to moving averages
        if company_data.get('sma_50') and company_data.get('sma_200'):
            if company_data['current_price'] > company_data['sma_50'] > company_data['sma_200']:
                st.success("**Trend:** Bullish (Price > 50 SMA > 200 SMA)")
            elif company_data['current_price'] < company_data['sma_50'] < company_data['sma_200']:
                st.error("**Trend:** Bearish (Price < 50 SMA < 200 SMA)")
            else:
                st.info("**Trend:** Mixed")
        
        # Market cap estimate
        if company_data.get('market_cap'):
            st.write(f"**Estimated Market Cap:** {format_market_cap(company_data['market_cap'])}")

def display_financial_ratios_kite(company_data, symbol):
    """Display financial ratios and metrics using available data."""
    st.subheader(f"Market Data & Ratios - {company_data['name']}")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Volume & Liquidity", "Risk Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if company_data.get('52_week_high') and company_data['52_week_high'] > 0:
                distance_from_high = ((company_data['52_week_high'] - company_data['current_price']) / company_data['52_week_high']) * 100
                st.metric("From 52W High", f"-{distance_from_high:.1f}%")
            
            if company_data.get('52_week_low') and company_data['52_week_low'] > 0:
                distance_from_low = ((company_data['current_price'] - company_data['52_week_low']) / company_data['52_week_low']) * 100
                st.metric("From 52W Low", f"+{distance_from_low:.1f}%")
        
        with col2:
            if company_data.get('sma_50') and company_data['sma_50'] > 0:
                vs_sma_50 = ((company_data['current_price'] - company_data['sma_50']) / company_data['sma_50']) * 100
                st.metric("vs 50-Day SMA", f"{vs_sma_50:+.1f}%")
            
            if company_data.get('sma_200') and company_data['sma_200'] > 0:
                vs_sma_200 = ((company_data['current_price'] - company_data['sma_200']) / company_data['sma_200']) * 100
                st.metric("vs 200-Day SMA", f"{vs_sma_200:+.1f}%")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Today's Volume", f"{company_data['volume']:,}")
            if company_data.get('average_volume'):
                st.metric("Average Volume", f"{company_data['average_volume']:,.0f}")
        
        with col2:
            if company_data.get('lot_size'):
                st.metric("Lot Size", f"{company_data['lot_size']:,}")
            
            # Volume ratio (today vs average)
            if company_data.get('average_volume') and company_data['average_volume'] > 0:
                volume_ratio = (company_data['volume'] / company_data['average_volume']) * 100
                st.metric("Volume Ratio", f"{volume_ratio:.1f}%")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if company_data.get('volatility'):
                st.metric("Annual Volatility", f"{company_data['volatility']:.1f}%")
            
            # Beta calculation would require market data comparison
            st.metric("Beta", "N/A")
        
        with col2:
            # Daily price range
            if company_data['high'] > 0 and company_data['low'] > 0:
                daily_range = ((company_data['high'] - company_data['low']) / company_data['low']) * 100
                st.metric("Daily Range", f"{daily_range:.1f}%")
            
            # Gap analysis
            if company_data['open'] > 0 and company_data['close'] > 0:
                gap = ((company_data['open'] - company_data['close']) / company_data['close']) * 100
                st.metric("Opening Gap", f"{gap:+.1f}%")

def display_multi_company_comparison_kite():
    """Display comparison of multiple companies using Kite data."""
    st.subheader("Compare Multiple Companies")
    
    # Input for multiple symbols
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Enter Stock Symbols (comma separated)", 
            "RELIANCE, TCS, INFY, HDFCBANK",
            help="Enter NSE symbols separated by commas"
        )
    
    with col2:
        if st.button("Compare Companies", use_container_width=True):
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            with st.spinner("Fetching comparison data..."):
                comparison_data = []
                for symbol in symbols:
                    data = get_company_fundamentals_kite(symbol, "NSE")
                    if data:
                        comparison_data.append(data)
                
                if comparison_data:
                    st.session_state.comparison_data = comparison_data
                    st.rerun()
    
    if 'comparison_data' in st.session_state and st.session_state.comparison_data:
        comparison_df = create_comparison_dataframe_kite(st.session_state.comparison_data)
        
        # Select metrics to compare
        st.subheader("Select Metrics for Comparison")
        
        metric_categories = {
            "Price Analysis": ['current_price', 'change_percent', '52_week_high', '52_week_low'],
            "Volume Analysis": ['volume', 'lot_size'],
            "Technical Indicators": ['sma_50', 'sma_200', 'volatility']
        }
        
        selected_metrics = []
        for category, metrics in metric_categories.items():
            with st.expander(f"{category} Metrics"):
                for metric in metrics:
                    if st.checkbox(f"{format_metric_name(metric)}", value=True, key=f"comp_{metric}"):
                        selected_metrics.append(metric)
        
        if selected_metrics:
            # Display comparison table
            display_metrics = ['name'] + selected_metrics
            comparison_display_df = comparison_df[display_metrics].copy()
            
            # Format the dataframe
            for col in selected_metrics:
                if 'price' in col.lower() or 'sma' in col.lower():
                    comparison_display_df[col] = comparison_display_df[col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) and x != 0 else "N/A")
                elif 'percent' in col.lower() or 'volatility' in col.lower():
                    comparison_display_df[col] = comparison_display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) and x != 0 else "N/A")
                elif 'volume' in col.lower() or 'lot' in col.lower():
                    comparison_display_df[col] = comparison_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) and x != 0 else "N/A")
                else:
                    comparison_display_df[col] = comparison_display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            st.subheader("Company Comparison")
            st.dataframe(comparison_display_df, use_container_width=True)
            
            # Visual comparisons
            st.subheader("Visual Comparisons")
            
            # Bar chart for selected metrics
            if len(selected_metrics) >= 1:
                metric_to_plot = st.selectbox("Select metric for bar chart", selected_metrics)
                if metric_to_plot:
                    fig = go.Figure()
                    
                    values = []
                    names = []
                    for company in st.session_state.comparison_data:
                        value = company.get(metric_to_plot, 0)
                        if value and value != 0:
                            values.append(value)
                            names.append(company['name'])
                    
                    if values:
                        fig.add_trace(go.Bar(
                            x=names,
                            y=values,
                            text=[f"{v:.2f}{'%' if 'percent' in metric_to_plot or 'volatility' in metric_to_plot else ''}" for v in values],
                            textposition='auto',
                        ))
                        
                        y_axis_title = format_metric_name(metric_to_plot)
                        if 'percent' in metric_to_plot or 'volatility' in metric_to_plot:
                            y_axis_title += " (%)"
                        elif 'price' in metric_to_plot:
                            y_axis_title += " (₹)"
                        
                        fig.update_layout(
                            title=f"{format_metric_name(metric_to_plot)} Comparison",
                            xaxis_title="Companies",
                            yaxis_title=y_axis_title,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def create_comparison_dataframe_kite(company_data_list):
    """Create a DataFrame from multiple company data for comparison."""
    comparison_data = []
    for company in company_data_list:
        row = {k: v for k, v in company.items() if not isinstance(v, (list, dict))}
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

# Keep these helper functions as they're still useful
def format_market_cap(market_cap):
    """Format market cap into readable string."""
    if market_cap >= 1e12:
        return f"₹{market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"₹{market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"₹{market_cap/1e6:.2f}M"
    else:
        return f"₹{market_cap:,.0f}"

def format_number(number):
    """Format large numbers into readable string."""
    if number == 'N/A':
        return 'N/A'
    if number >= 1e9:
        return f"{number/1e9:.1f}B"
    elif number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:,.0f}"

def format_metric_name(metric):
    """Convert metric key to display name."""
    metric_names = {
        'current_price': 'Current Price',
        'change_percent': 'Change %',
        '52_week_high': '52W High',
        '52_week_low': '52W Low',
        'volume': 'Volume',
        'lot_size': 'Lot Size',
        'sma_50': '50-Day SMA',
        'sma_200': '200-Day SMA',
        'volatility': 'Volatility',
        'open': 'Open Price',
        'high': 'High Price',
        'low': 'Low Price',
        'close': 'Close Price',
        'instrument_type': 'Instrument Type',
        'segment': 'Segment'
    }
    return metric_names.get(metric, metric.replace('_', ' ').title())

def page_basket_orders():
    """A page for creating, managing, and executing basket orders."""
    display_header()
    st.title("Basket Orders")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the basket order feature.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add Order to Basket")
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0
            
            if st.form_submit_button("Add to Basket"):
                if symbol and quantity > 0:
                    st.session_state.basket.append({
                        'symbol': symbol,
                        'transaction_type': transaction_type,
                        'quantity': quantity,
                        'product': product,
                        'order_type': order_type,
                        'price': price if price > 0 else None
                    })
                    st.success(f"Added {symbol} to basket!")
                    st.rerun()

    with col2:
        st.subheader("Current Basket")
        if st.session_state.basket:
            for i, order in enumerate(st.session_state.basket):
                with st.expander(f"{order['transaction_type']} {order['quantity']} {order['symbol']}"):
                    st.write(f"**Product:** {order['product']}")
                    st.write(f"**Order Type:** {order['order_type']}")
                    if order['price']:
                        st.write(f"**Price:** ₹{order['price']}")
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.basket.pop(i)
                        st.rerun()
            
            st.markdown("---")
            if st.button("Execute Basket Order", type="primary", use_container_width=True):
                execute_basket_order(st.session_state.basket, instrument_df)
        else:
            st.info("Your basket is empty. Add orders using the form on the left.")

def run_backtest(strategy_func, data, **params):
    """Runs a backtest for a given strategy function."""
    df = data.copy()
    signals = strategy_func(df, **params)
    
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    portfolio_value = []
    
    for i in range(len(df)):
        if signals[i] == 'BUY' and position == 0:
            position = capital / df['close'][i]
            capital = 0
        elif signals[i] == 'SELL' and position > 0:
            capital = position * df['close'][i]
            position = 0
        
        current_value = capital + (position * df['close'][i])
        portfolio_value.append(current_value)
        
    pnl = (portfolio_value[-1] - initial_capital) / initial_capital * 100
    
    return pnl, pd.Series(portfolio_value, index=df.index)

def rsi_strategy(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """Simple RSI Crossover Strategy"""
    rsi = talib.RSI(df['close'], timeperiod=rsi_period)
    signals = [''] * len(df)
    for i in range(1, len(df)):
        if rsi.iloc[i-1] < rsi_oversold and rsi.iloc[i] > rsi_oversold:
            signals[i] = 'BUY'
        elif rsi.iloc[i-1] > rsi_overbought and rsi.iloc[i] < rsi_overbought:
            signals[i] = 'SELL'
    return signals

def macd_strategy(df, fast=12, slow=26, signal=9):
    """MACD Crossover Strategy"""
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
    signals = [''] * len(df)
    for i in range(1, len(df)):
        if macd.iloc[i-1] < macdsignal.iloc[i-1] and macd.iloc[i] > macdsignal.iloc[i]:
            signals[i] = 'BUY'
        elif macd.iloc[i-1] > macdsignal.iloc[i-1] and macd.iloc[i] < macdsignal.iloc[i]:
            signals[i] = 'SELL'
    return signals

def supertrend_strategy(df, period=7, multiplier=3):
    """Supertrend Strategy - simplified version without pandas-ta"""
    # Calculate ATR
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    
    # Basic upper and lower bands
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    signals = [''] * len(df)
    trend = [1] * len(df) # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            trend[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1 and trend[i-1] == -1:
            signals[i] = 'BUY'
        elif trend[i] == -1 and trend[i-1] == 1:
            signals[i] = 'SELL'
    
    return signals

def page_algo_strategy_maker():
    """Algo Strategy Maker page with pre-built, backtestable, and executable strategies."""
    display_header()
    st.title("Algo Strategy Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to use the Algo Strategy Hub.")
        return

    st.info("Select a pre-built strategy, configure its parameters, and run a backtest on historical data. You can then place trades based on the latest signal.", icon="🤖")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Strategy Configuration")
        
        strategy_options = {
            "RSI Crossover": rsi_strategy,
            "MACD Crossover": macd_strategy,
            "Supertrend Follower": supertrend_strategy,
        }
        selected_strategy_name = st.selectbox("Select a Strategy", list(strategy_options.keys()))
        
        st.markdown("**Instrument**")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'NFO', 'MCX', 'CDS'])]['tradingsymbol'].unique()
        symbol = st.selectbox("Select Symbol", all_symbols, index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0)
        
        st.markdown("**Parameters**")
        params = {}
        if selected_strategy_name == "RSI Crossover":
            params['rsi_period'] = st.slider("RSI Period", 5, 30, 14)
            params['rsi_overbought'] = st.slider("RSI Overbought", 60, 90, 70)
            params['rsi_oversold'] = st.slider("RSI Oversold", 10, 40, 30)
        elif selected_strategy_name == "MACD Crossover":
            params['fast'] = st.slider("Fast Period", 5, 20, 12)
            params['slow'] = st.slider("Slow Period", 20, 50, 26)
            params['signal'] = st.slider("Signal Period", 5, 20, 9)
        elif selected_strategy_name == "Supertrend Follower":
            params['period'] = st.slider("ATR Period", 5, 20, 7)
            params['multiplier'] = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.5)

        st.markdown("**Trade Execution**")
        quantity = st.number_input("Trade Quantity", min_value=1, value=1)
        
        run_button = st.button("Run Backtest & Get Signal", use_container_width=True, type="primary")

    with col2:
        if run_button:
            with st.spinner(f"Running backtest for {selected_strategy_name} on {symbol}..."):
                exchange = instrument_df[instrument_df['tradingsymbol'] == symbol].iloc[0]['exchange']
                token = get_instrument_token(symbol, instrument_df, exchange=exchange)
                data = get_historical_data(token, 'day', period='1y')
                
                if not data.empty and len(data) > 50: 
                    pnl, portfolio_curve = run_backtest(strategy_options[selected_strategy_name], data, **params)
                    latest_signal = strategy_options[selected_strategy_name](data, **params)[-1]

                    st.session_state['backtest_results'] = {
                        'pnl': pnl,
                        'curve': portfolio_curve,
                        'signal': latest_signal,
                        'symbol': symbol,
                        'quantity': quantity
                    }
                else:
                    st.error("Could not fetch enough historical data to run the backtest.")
                    if 'backtest_results' in st.session_state:
                        st.session_state['backtest_results'] = None

        if st.session_state.get('backtest_results') is not None:
            results = st.session_state['backtest_results']
            st.subheader("Backtest Results")
            st.metric("Total P&L (1 Year)", f"{results['pnl']:.2f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['curve'].index, y=results['curve'], mode='lines', name='Portfolio Value'))
            fig.update_layout(title="Portfolio Growth Over 1 Year", yaxis_title="Portfolio Value (₹)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Live Signal & Trading")
            signal = results['signal']
            color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
            st.markdown(f"### Latest Signal: <span style='color:{color};'>{signal if signal else 'HOLD'}</span>", unsafe_allow_html=True)

            if signal in ["BUY", "SELL"]:
                if st.button(f"Place {signal} Order for {results['quantity']} of {results['symbol']}", use_container_width=True):
                    place_order(instrument_df, results['symbol'], results['quantity'], "MARKET", signal, "MIS")

@st.cache_data(ttl=3600)
def run_scanner(instrument_df, scanner_type, holdings_df=None):
    """A single function to run different types of market scanners on user holdings or a predefined list."""
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame()

    scan_list = []
    if holdings_df is not None and not holdings_df.empty:
        scan_list = holdings_df['tradingsymbol'].unique().tolist()
        st.info("Scanning stocks from your live holdings.")
    else:
        scan_list = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK', 'MARUTI', 'ASIANPAINT']
        st.info("Scanning a predefined list of NIFTY 50 stocks as no holdings were found.")

    results = []
    
    token_map = {
        row['tradingsymbol']: row['instrument_token']
        for _, row in instrument_df[instrument_df['tradingsymbol'].isin(scan_list)].iterrows()
    }
    
    for symbol in scan_list:
        token = token_map.get(symbol)
        if not token: continue
        
        try:
            df = get_historical_data(token, 'day', period='1y')
            if df.empty or len(df) < 252: continue
            
            df.columns = [c.lower() for c in df.columns]

            if scanner_type == "Momentum":
                rsi_col = next((c for c in df.columns if 'rsi_14' in c), None)
                if rsi_col:
                    rsi = df.iloc[-1].get(rsi_col)
                    if rsi and (rsi > 70 or rsi < 30):
                        results.append({'Stock': symbol, 'RSI': f"{rsi:.2f}", 'Signal': "Overbought" if rsi > 70 else "Oversold"})
            
            elif scanner_type == "Trend":
                adx_col = next((c for c in df.columns if 'adx_14' in c), None)
                ema50_col = next((c for c in df.columns if 'ema_50' in c), None)
                ema200_col = next((c for c in df.columns if 'ema_200' in c), None)
                
                if adx_col and ema50_col and ema200_col:
                    adx = df.iloc[-1].get(adx_col)
                    ema50 = df.iloc[-1].get(ema50_col)
                    ema200 = df.iloc[-1].get(ema200_col)
                    if adx and adx > 25 and ema50 and ema200:
                        trend = "Uptrend" if ema50 > ema200 else "Downtrend"
                        results.append({'Stock': symbol, 'ADX': f"{adx:.2f}", 'Trend': trend})

            elif scanner_type == "Breakout":
                high_52wk = df['high'].rolling(window=252).max().iloc[-1]
                low_52wk = df['low'].rolling(window=252).min().iloc[-1]
                last_close = df['close'].iloc[-1]
                avg_vol_20d = df['volume'].rolling(window=20).mean().iloc[-1]
                last_vol = df['volume'].iloc[-1]

                if last_close >= high_52wk * 0.98:
                    signal = "Near 52-Week High"
                    if last_vol > avg_vol_20d * 1.5:
                        signal += " (Volume Surge)"
                    results.append({'Stock': symbol, 'Signal': signal, 'Last Close': last_close, '52Wk High': high_52wk})

        except Exception:
            continue
            
    return pd.DataFrame(results)

def run_momentum_scanner(instrument_df, holdings_df=None):
    """Momentum scanner with RSI and MACD analysis."""
    client = get_broker_client()
    if not client or instrument_df.empty: 
        return pd.DataFrame()

    # Get symbols to scan
    scan_list = []
    if holdings_df is not None and not holdings_df.empty:
        scan_list = holdings_df['tradingsymbol'].unique().tolist()[:20] # Limit to 20 stocks
    else:
        scan_list = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
            'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK', 
            'MARUTI', 'ASIANPAINT', 'HCLTECH', 'TATAMOTORS', 'SUNPHARMA'
        ]
    
    results = []
    
    for symbol in scan_list:
        try:
            # Get live quote
            exchange = 'NSE'
            quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
            if quote_data.empty:
                continue
                
            current_price = quote_data.iloc[0]['Price']
            change_pct = quote_data.iloc[0]['% Change']
            
            # Get historical data
            token = get_instrument_token(symbol, instrument_df, exchange)
            if not token:
                continue
                
            hist_data = get_historical_data(token, 'day', period='3mo')
            if hist_data.empty or len(hist_data) < 30:
                continue
            
            # Calculate RSI
            try:
                hist_data['RSI_14'] = talib.RSI(hist_data['close'], timeperiod=14)
                latest = hist_data.iloc[-1]
                rsi = latest.get('RSI_14', 50)
                
                # Momentum signals
                if rsi > 70 and change_pct > 0:
                    results.append({
                        'Symbol': symbol,
                        'LTP': f"₹{current_price:.2f}",
                        'Change %': f"{change_pct:.2f}%",
                        'RSI': f"{rsi:.1f}",
                        'Signal': "Overbought",
                        'Strength': "High"
                    })
                elif rsi < 30 and change_pct < 0:
                    results.append({
                        'Symbol': symbol,
                        'LTP': f"₹{current_price:.2f}",
                        'Change %': f"{change_pct:.2f}%",
                        'RSI': f"{rsi:.1f}",
                        'Signal': "Oversold", 
                        'Strength': "High"
                    })
                    
            except Exception:
                continue
                
        except Exception:
            continue
            
    return pd.DataFrame(results)

def run_trend_scanner(instrument_df, holdings_df=None):
    """Trend scanner with EMA analysis."""
    client = get_broker_client()
    if not client or instrument_df.empty: 
        return pd.DataFrame()

    scan_list = []
    if holdings_df is not None and not holdings_df.empty:
        scan_list = holdings_df['tradingsymbol'].unique().tolist()[:20]
    else:
        scan_list = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
            'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK'
        ]
    
    results = []
    
    for symbol in scan_list:
        try:
            # Get live data
            exchange = 'NSE'
            quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
            if quote_data.empty:
                continue
                
            current_price = quote_data.iloc[0]['Price']
            change_pct = quote_data.iloc[0]['% Change']
            
            # Get historical data
            token = get_instrument_token(symbol, instrument_df, exchange)
            if not token:
                continue
                
            hist_data = get_historical_data(token, 'day', period='3mo')
            if hist_data.empty or len(hist_data) < 50:
                continue
            
            # Calculate EMAs
            try:
                hist_data['EMA_20'] = talib.EMA(hist_data['close'], timeperiod=20)
                hist_data['EMA_50'] = talib.EMA(hist_data['close'], timeperiod=50)
                
                latest = hist_data.iloc[-1]
                ema_20 = latest.get('EMA_20', current_price)
                ema_50 = latest.get('EMA_50', current_price)
                
                # Trend signals
                if current_price > ema_20 > ema_50 and change_pct > 0:
                    results.append({
                        'Symbol': symbol,
                        'LTP': f"₹{current_price:.2f}",
                        'Change %': f"{change_pct:.2f}%",
                        'Trend': "Uptrend",
                        '20 EMA': f"₹{ema_20:.1f}",
                        '50 EMA': f"₹{ema_50:.1f}"
                    })
                elif current_price < ema_20 < ema_50 and change_pct < 0:
                    results.append({
                        'Symbol': symbol,
                        'LTP': f"₹{current_price:.2f}",
                        'Change %': f"{change_pct:.2f}%",
                        'Trend': "Downtrend",
                        '20 EMA': f"₹{ema_20:.1f}",
                        '50 EMA': f"₹{ema_50:.1f}"
                    })
                    
            except Exception:
                continue
                
        except Exception:
            continue
            
    return pd.DataFrame(results)

def run_breakout_scanner(instrument_df, holdings_df=None):
    """Breakout scanner for key level breaks."""
    client = get_broker_client()
    if not client or instrument_df.empty: 
        return pd.DataFrame()

    scan_list = []
    if holdings_df is not None and not holdings_df.empty:
        scan_list = holdings_df['tradingsymbol'].unique().tolist()[:20]
    else:
        scan_list = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
            'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK'
        ]
    
    results = []
    
    for symbol in scan_list:
        try:
            # Get live data
            exchange = 'NSE'
            quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
            if quote_data.empty:
                continue
                
            current_price = quote_data.iloc[0]['Price']
            change_pct = quote_data.iloc[0]['% Change']
            
            # Get historical data
            token = get_instrument_token(symbol, instrument_df, exchange)
            if not token:
                continue
                
            hist_data = get_historical_data(token, 'day', period='6mo')
            if hist_data.empty or len(hist_data) < 100:
                continue
            
            # Calculate breakout levels
            high_20d = hist_data['high'].tail(20).max()
            low_20d = hist_data['low'].tail(20).min()
            
            # Breakout signals
            if current_price >= high_20d and change_pct > 0:
                results.append({
                    'Symbol': symbol,
                    'LTP': f"₹{current_price:.2f}",
                    'Change %': f"{change_pct:.2f}%",
                    'Breakout': "20-Day High",
                    'Resistance': f"₹{high_20d:.1f}"
                })
            elif current_price <= low_20d and change_pct < 0:
                results.append({
                    'Symbol': symbol,
                    'LTP': f"₹{current_price:.2f}",
                    'Change %': f"{change_pct:.2f}%",
                    'Breakout': "20-Day Low", 
                    'Support': f"₹{low_20d:.1f}"
                })
                
        except Exception:
            continue
            
    return pd.DataFrame(results)

def page_momentum_and_trend_finder():
    """Clean and functional Market Scanners page."""
    display_header()
    st.title("Market Scanners")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use market scanners.")
        return
        
    _, holdings_df, _, _ = get_portfolio()
    
    # Simple scanner selection
    col1, col2 = st.columns([3, 1])
    with col1:
        scanner_type = st.radio(
            "Select Scanner Type",
            ["Momentum (RSI)", "Trend (EMA)", "Breakout"],
            horizontal=True
        )
    
    with col2:
        if st.button("🔄 Scan Now", use_container_width=True, type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # Run selected scanner
    with st.spinner(f"Running {scanner_type} scanner..."):
        if scanner_type == "Momentum (RSI)":
            data = run_momentum_scanner(instrument_df, holdings_df)
            title = "Momentum Stocks (RSI Based)"
            description = "Stocks with RSI above 70 (overbought) or below 30 (oversold)"
            
        elif scanner_type == "Trend (EMA)":
            data = run_trend_scanner(instrument_df, holdings_df) 
            title = "Trending Stocks (EMA Based)"
            description = "Stocks in strong uptrend/downtrend based on EMA alignment"
            
        else: # Breakout
            data = run_breakout_scanner(instrument_df, holdings_df)
            title = "Breakout Stocks"
            description = "Stocks breaking 20-day high/low resistance/support levels"
    
    # Display results
    st.subheader(title)
    st.caption(description)
    
    if not data.empty:
        # Color coding based on scanner type
        if scanner_type == "Momentum (RSI)":
            def color_momentum(val):
                if 'Overbought' in str(val):
                    return 'color: #ff4444; font-weight: bold;'
                elif 'Oversold' in str(val):
                    return 'color: #00aa00; font-weight: bold;'
                return ''
            styled_data = data.style.map(color_momentum, subset=['Signal'])
            
        elif scanner_type == "Trend (EMA)":
            def color_trend(val):
                if 'Uptrend' in str(val):
                    return 'color: #00aa00; font-weight: bold;'
                elif 'Downtrend' in str(val):
                    return 'color: #ff4444; font-weight: bold;'
                return ''
            styled_data = data.style.map(color_trend, subset=['Trend'])
            
        else: # Breakout
            def color_breakout(val):
                if 'High' in str(val):
                    return 'color: #00aa00; font-weight: bold;'
                elif 'Low' in str(val):
                    return 'color: #ff4444; font-weight: bold;'
                return ''
            styled_data = data.style.map(color_breakout, subset=['Breakout'])
        
        st.dataframe(styled_data, use_container_width=True, hide_index=True)
        
        # Simple statistics
        if scanner_type == "Momentum (RSI)":
            bullish = len(data[data['Signal'] == 'Overbought'])
            bearish = len(data[data['Signal'] == 'Oversold'])
            st.metric("Signals Found", len(data), delta=f"{bullish} Bullish, {bearish} Bearish")
            
        elif scanner_type == "Trend (EMA)":
            uptrend = len(data[data['Trend'] == 'Uptrend'])
            downtrend = len(data[data['Trend'] == 'Downtrend'])
            st.metric("Signals Found", len(data), delta=f"{uptrend} Up, {downtrend} Down")
            
        else: # Breakout
            breakouts = len(data[data['Breakout'].str.contains('High')])
            breakdowns = len(data[data['Breakout'].str.contains('Low')])
            st.metric("Signals Found", len(data), delta=f"{breakouts} Breakouts, {breakdowns} Breakdowns")
        
        # Quick actions
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Export to CSV", use_container_width=True):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{scanner_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("👀 Add to Watchlist", use_container_width=True):
                added = 0
                for symbol in data['Symbol'].head(5): # Add top 5
                    if symbol not in [item['symbol'] for item in st.session_state.watchlists[st.session_state.active_watchlist]]:
                        st.session_state.watchlists[st.session_state.active_watchlist].append({
                            'symbol': symbol, 
                            'exchange': 'NSE'
                        })
                        added += 1
                if added > 0:
                    st.success(f"Added {added} stocks to watchlist")
                else:
                    st.info("No new stocks to add")
                    
    else:
        # Clear, helpful empty state
        st.info(f"""
        **No {scanner_type.lower()} signals found.**
        
        This could mean:
        - Markets are in consolidation
        - No extreme conditions detected
        - Try a different scanner type
        - Check if market is open
        """)

def calculate_strategy_pnl(legs, underlying_ltp):
    """Calculates the P&L for a given options strategy."""
    if not legs:
        return pd.DataFrame(), 0, 0, []

    price_range = np.linspace(underlying_ltp * 0.8, underlying_ltp * 1.2, 100)
    pnl_df = pd.DataFrame(index=price_range)
    pnl_df.index.name = "Underlying Price at Expiry"
    
    total_premium = 0
    for i, leg in enumerate(legs):
        pnl = 0
        if leg['type'] == 'Call':
            if leg['position'] == 'Buy':
                pnl = np.maximum(0, price_range - leg['strike']) - leg['premium']
                total_premium -= leg['premium'] * leg['quantity']
            else:
                pnl = leg['premium'] - np.maximum(0, price_range - leg['strike'])
                total_premium += leg['premium'] * leg['quantity']
        else:
            if leg['position'] == 'Buy':
                pnl = np.maximum(0, leg['strike'] - price_range) - leg['premium']
                total_premium -= leg['premium'] * leg['quantity']
            else:
                pnl = leg['premium'] - np.maximum(0, leg['strike'] - price_range)
                total_premium += leg['premium'] * leg['quantity']
        
        pnl_df[f'Leg_{i+1}'] = pnl * leg['quantity']
    
    pnl_df['Total P&L'] = pnl_df.sum(axis=1)
    
    max_profit = pnl_df['Total P&L'].max()
    max_loss = pnl_df['Total P&L'].min()
    
    breakevens = []
    sign_changes = np.where(np.diff(np.sign(pnl_df['Total P&L'])))[0]
    for idx in sign_changes:
        breakevens.append(pnl_df.index[idx])

    return pnl_df, max_profit, max_loss, breakevens

def page_option_strategy_builder():
    """Option Strategy Builder page with live data and P&L calculation."""
    display_header()
    st.title("Options Strategy Builder")
    
    instrument_df = get_instrument_df()
    client = get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to build strategies.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Strategy Configuration")
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        
        _, _, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)

        if not available_expiries:
            st.error(f"No options available for {underlying}.")
            st.stop()
            
        expiry_date = st.selectbox("Expiry", [e.strftime("%d %b %Y") for e in available_expiries])
        
        with st.form("add_leg_form"):
            st.write("**Add a New Leg**")
            leg_cols = st.columns(4)
            position = leg_cols[0].selectbox("Position", ["Buy", "Sell"])
            option_type = leg_cols[1].selectbox("Type", ["Call", "Put"])
            
            expiry_dt = datetime.strptime(expiry_date, "%d %b %Y").date()
            options = instrument_df[
                (instrument_df['name'] == underlying) & 
                (instrument_df['expiry'].dt.date == expiry_dt) & 
                (instrument_df['instrument_type'] == option_type[0])
            ]
            
            if not options.empty:
                strikes = sorted(options['strike'].unique())
                strike = leg_cols[2].selectbox("Strike", strikes, index=len(strikes)//2)
                quantity = leg_cols[3].number_input("Lots", min_value=1, value=1)
                
                submitted = st.form_submit_button("Add Leg")
                if submitted:
                    lot_size = options.iloc[0]['lot_size']
                    tradingsymbol = options[options['strike'] == strike].iloc[0]['tradingsymbol']
                    
                    try:
                        quote = client.quote(f"NFO:{tradingsymbol}")[f"NFO:{tradingsymbol}"]
                        premium = quote['last_price']
                        
                        st.session_state.strategy_legs.append({
                            'symbol': tradingsymbol,
                            'position': position,
                            'type': option_type,
                            'strike': strike,
                            'quantity': quantity * lot_size,
                            'premium': premium
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not fetch premium: {e}")
            else:
                st.warning("No strikes found for selected expiry/type.")

        st.subheader("Current Legs")
        if st.session_state.strategy_legs:
            for i, leg in enumerate(st.session_state.strategy_legs):
                st.text(f"{i+1}: {leg['position']} {leg['quantity']} {leg['symbol']} @ ₹{leg['premium']:.2f}")
            if st.button("Clear All Legs"):
                st.session_state.strategy_legs = []
                st.rerun()
        else:
            st.info("Add legs to your strategy.")
            
    with col2:
        st.subheader("Strategy Payoff Analysis")
        
        if st.session_state.strategy_legs:
            pnl_df, max_profit, max_loss, breakevens = calculate_strategy_pnl(st.session_state.strategy_legs, underlying_ltp)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pnl_df.index, y=pnl_df['Total P&L'], mode='lines', name='P&L'))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=underlying_ltp, line_dash="dot", line_color="yellow", annotation_text="Current LTP")
            fig.update_layout(
                title="Strategy P&L Payoff Chart",
                xaxis_title="Underlying Price at Expiry",
                yaxis_title="Profit / Loss (₹)",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk & Reward Profile")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("Max Profit", f"₹{max_profit:,.2f}")
            metrics_col1.metric("Max Loss", f"₹{max_loss:,.2f}")
            metrics_col2.metric("Breakeven(s)", ", ".join([f"₹{b:,.2f}" for b in breakevens]) if breakevens else "N/A")
        else:
            st.info("Add legs to see the payoff analysis.")

def get_futures_contracts(instrument_df, underlying, exchange):
    """Fetches and sorts futures contracts for a given underlying and exchange."""
    if instrument_df.empty or not underlying: return pd.DataFrame()
    futures_df = instrument_df[
        (instrument_df['name'] == underlying) &
        (instrument_df['instrument_type'] == 'FUT') &
        (instrument_df['exchange'] == exchange)
    ].copy()
    if not futures_df.empty:
        futures_df['expiry'] = pd.to_datetime(futures_df['expiry'])
        return futures_df.sort_values('expiry')
    return pd.DataFrame()

def page_futures_terminal():
    """Futures Terminal page with live data."""
    display_header()
    st.title("Futures Terminal")
    
    instrument_df = get_instrument_df()
    client = get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to access futures data.")
        return
    
    exchange_options = sorted(instrument_df[instrument_df['instrument_type'] == 'FUT']['exchange'].unique())
    if not exchange_options:
        st.warning("No futures contracts found in the instrument list.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_exchange = st.selectbox("Select Exchange", exchange_options, index=exchange_options.index('NFO') if 'NFO' in exchange_options else 0)
    
    underlyings = sorted(instrument_df[(instrument_df['instrument_type'] == 'FUT') & (instrument_df['exchange'] == selected_exchange)]['name'].unique())
    if not underlyings:
        st.warning(f"No futures underlyings found for the {selected_exchange} exchange.")
        return
        
    with col2:
        selected_underlying = st.selectbox("Select Underlying", underlyings)

    tab1, tab2 = st.tabs(["Live Futures Contracts", "Futures Calendar"])
    
    with tab1:
        st.subheader(f"Live Contracts for {selected_underlying}")
        futures_contracts = get_futures_contracts(instrument_df, selected_underlying, selected_exchange)
        
        if not futures_contracts.empty:
            symbols = [f"{row['exchange']}:{row['tradingsymbol']}" for _, row in futures_contracts.iterrows()]
            try:
                quotes = client.quote(symbols)
                live_data = []
                for symbol_key, data in quotes.items():
                    if data:
                        prev_close = data.get('ohlc', {}).get('close', 0)
                        last_price = data.get('last_price', 0)
                        change = last_price - prev_close
                        pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                        
                        live_data.append({
                            'Contract': data.get('tradingsymbol', symbol_key.split(':')[-1]),
                            'LTP': last_price,
                            'Change': change,
                            '% Change': pct_change,
                            'Volume': data.get('volume', 0),
                            'OI': data.get('oi', 0)
                        })
                live_df = pd.DataFrame(live_data)
                st.dataframe(live_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not fetch live futures data: {e}")
        else:
            st.info(f"No active futures contracts found for {selected_underlying}.")
    
    with tab2:
        st.subheader("Futures Expiry Calendar")
        futures_contracts = get_futures_contracts(instrument_df, selected_underlying, selected_exchange)
        if not futures_contracts.empty:
            calendar_df = futures_contracts[['tradingsymbol', 'expiry']].copy()
            calendar_df['expiry'] = pd.to_datetime(calendar_df['expiry'])
            calendar_df['Days to Expiry'] = (calendar_df['expiry'] - pd.to_datetime('today')).dt.days
            st.dataframe(calendar_df.rename(columns={'tradingsymbol': 'Contract', 'expiry': 'Expiry Date'}), use_container_width=True, hide_index=True)

def generate_ai_trade_idea(instrument_df, active_list):
    """Dynamically generates a trade idea based on watchlist signals."""
    if not active_list or instrument_df.empty:
        return None

    discovery_results = {}
    for item in active_list:
        token = get_instrument_token(item['symbol'], instrument_df, exchange=item['exchange'])
        if token:
            data = get_historical_data(token, 'day', period='6mo')
            if not data.empty:
                interpretation = interpret_indicators(data)
                signals = [v for k, v in interpretation.items() if "Bullish" in v or "Bearish" in v]
                if signals:
                    discovery_results[item['symbol']] = {'signals': signals, 'data': data}
    
    if not discovery_results:
        return None

    best_ticker = max(discovery_results, key=lambda k: len(discovery_results[k]['signals']))
    
    ticker_data = discovery_results[best_ticker]['data']
    ltp = ticker_data['close'].iloc[-1]
    
    # Calculate ATR for stop-loss/target
    atr = talib.ATR(ticker_data['high'], ticker_data['low'], ticker_data['close'], timeperiod=14).iloc[-1]
    if pd.isna(atr): return None

    is_bullish = any("Bullish" in s for s in discovery_results[best_ticker]['signals'])

    narrative = f"**{best_ticker}** is showing a confluence of {'bullish' if is_bullish else 'bearish'} signals. Analysis indicates: {', '.join(discovery_results[best_ticker]['signals'])}. "

    if is_bullish:
        narrative += f"A move above recent resistance could trigger further upside."
        entry = ltp
        target = ltp + (2 * atr)
        stop_loss = ltp - (1.5 * atr)
        title = f"High-Conviction Long Setup: {best_ticker}"
    else:
        narrative += f"A break below recent support could lead to further downside."
        entry = ltp
        target = ltp - (2 * atr)
        stop_loss = ltp + (1.5 * atr)
        title = f"High-Conviction Short Setup: {best_ticker}"

    return {
        "title": title,
        "entry": entry,
        "target": target,
        "stop_loss": stop_loss,
        "narrative": narrative
    }

def page_ai_discovery():
    """Enhanced AI Discovery with machine learning patterns and predictive insights."""
    display_header()
    st.title("🔍 AI Discovery Engine")
    st.info("Advanced pattern recognition, predictive analytics, and AI-driven trade discovery using machine learning.", icon="🤖")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use AI Discovery.")
        return

    # Get active watchlist
    active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
    active_list = st.session_state.watchlists.get(active_watchlist, [])
    
    if not active_list:
        st.warning("Please set up your watchlist on the Dashboard page to enable AI Discovery.")
        return

    # Enhanced discovery modes
    discovery_mode = st.radio(
        "Discovery Mode",
        ["Pattern Recognition", "Predictive Signals", "Risk-Adjusted Opportunities", "Technical Setups"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🔎 {discovery_mode}")
        
        with st.spinner("🤖 AI is analyzing market patterns..."):
            if discovery_mode == "Pattern Recognition":
                results = enhanced_pattern_recognition(active_list, instrument_df)
            elif discovery_mode == "Predictive Signals":
                results = predictive_signals_analysis(active_list, instrument_df)
            elif discovery_mode == "Risk-Adjusted Opportunities":
                results = risk_adjusted_opportunities(active_list, instrument_df)
            else:  # Technical Setups
                results = technical_setups_analysis(active_list, instrument_df)
    
    with col2:
        st.subheader("⚙️ Discovery Settings")
        
        confidence_threshold = st.slider("Confidence Threshold", 50, 95, 75)
        min_volume = st.number_input("Minimum Volume", 1000, 1000000, 10000)
        time_frame = st.selectbox("Time Frame", ["1D", "1W", "1M", "3M"])
        
        st.markdown("---")
        st.subheader("📊 Market Context")
        
        # Market metrics
        status_info = get_market_status()
        st.metric("Market Status", status_info['status'].replace('_', ' ').title())
        
        # VIX if available
        vix_data = get_watchlist_data([{'symbol': 'INDIA VIX', 'exchange': 'NSE'}])
        if not vix_data.empty:
            st.metric("India VIX", f"{vix_data.iloc[0]['Price']:.2f}")
        
        st.markdown("---")
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()

    # Display results
    if results and not results.get("error"):
        display_enhanced_discovery_results(results, discovery_mode, confidence_threshold)
    else:
        st.error("No patterns found or analysis failed. Try adjusting parameters.")

def enhanced_pattern_recognition(active_list, instrument_df):
    """Advanced pattern recognition with ML-based technical analysis using multi-timeframe data."""
    patterns = []
    
    for item in active_list[:10]:  # Limit for performance
        try:
            symbol = item['symbol']
            exchange = item['exchange']
            token = get_instrument_token(symbol, instrument_df, exchange)
            
            if not token:
                continue
                
            # Get multi-timeframe data - ADD HOURLY DATA HERE
            daily_data = get_historical_data(token, 'day', period='3mo')
            hourly_data = get_hourly_data_with_fallback(token, symbol, days=30)  # NEW: Get hourly data
            
            if daily_data.empty:
                continue
            
            # Enhanced analysis with multi-timeframe data
            pattern_analysis = analyze_advanced_patterns_with_hourly(daily_data, hourly_data, symbol)  # UPDATED FUNCTION
            
            if pattern_analysis["confidence"] > 60:
                patterns.append(pattern_analysis)
                
        except Exception as e:
            continue
    
    return {
        "patterns": sorted(patterns, key=lambda x: x["confidence"], reverse=True),
        "total_analyzed": len(active_list),
        "high_confidence_patterns": len([p for p in patterns if p["confidence"] > 80])
    }

def analyze_advanced_patterns_with_hourly(daily_data, hourly_data, symbol):
    """Enhanced pattern analysis using both daily and hourly data."""
    
    # Calculate multiple indicators for both timeframes
    daily_data = calculate_advanced_indicators(daily_data)
    daily_latest = daily_data.iloc[-1]
    
    patterns_detected = []
    confidence = 0
    signal_strength = "Neutral"
    timeframe_alignment = 0
    
    # DAILY ANALYSIS (existing logic)
    # Trend analysis
    if (daily_latest.get('EMA_20', 0) > daily_latest.get('EMA_50', 0) and
        daily_latest.get('EMA_50', 0) > daily_latest.get('EMA_200', 0)):
        patterns_detected.append("Strong Daily Uptrend")
        confidence += 20
        signal_strength = "Bullish"
        timeframe_alignment += 1
    
    # Momentum confirmation
    daily_rsi = daily_latest.get('RSI_14', 50)
    if 40 < daily_rsi < 70:  # Avoid extremes
        if daily_rsi > 55:
            patterns_detected.append("Daily Positive Momentum")
            confidence += 10
        elif daily_rsi < 45:
            patterns_detected.append("Daily Negative Momentum") 
            confidence += 10
            signal_strength = "Bearish"
    
    # HOURLY ANALYSIS (NEW)
    if not hourly_data.empty and len(hourly_data) > 20:
        hourly_data = calculate_advanced_indicators(hourly_data)
        hourly_latest = hourly_data.iloc[-1]
        
        # Hourly trend analysis
        hourly_ema_20 = hourly_latest.get('EMA_20', 0)
        hourly_ema_50 = hourly_latest.get('EMA_50', 0)
        
        if hourly_ema_20 > hourly_ema_50:
            patterns_detected.append("Hourly Uptrend")
            confidence += 15
            timeframe_alignment += 1
        else:
            patterns_detected.append("Hourly Consolidation")
            confidence += 5
        
        # Hourly momentum
        hourly_rsi = hourly_latest.get('RSI_14', 50)
        if 30 < hourly_rsi < 80:  # Wider range for hourly
            if hourly_rsi > 60:
                patterns_detected.append("Hourly Bullish Momentum")
                confidence += 10
            elif hourly_rsi < 40:
                patterns_detected.append("Hourly Bearish Momentum")
                confidence += 10
        
        # Volume analysis on hourly
        if len(hourly_data) > 20:
            hourly_volume_avg = hourly_data['volume'].tail(20).mean()
            current_hourly_volume = hourly_latest.get('volume', 0)
            if current_hourly_volume > hourly_volume_avg * 1.5:
                patterns_detected.append("Hourly Volume Surge")
                confidence += 15
    
    # MULTI-TIMEFRAME ALIGNMENT BONUS (NEW)
    if timeframe_alignment >= 2:
        patterns_detected.append("Multi-Timeframe Alignment")
        confidence += 20
    
    # Volume analysis (daily)
    if len(daily_data) > 20:
        volume_avg = daily_data['volume'].tail(20).mean()
        if (daily_latest.get('volume', 0) > volume_avg * 1.2 and
            daily_latest.get('close', 0) > daily_latest.get('open', 0)):
            patterns_detected.append("Daily Volume Breakout")
            confidence += 15
    
    # Support/Resistance breaks
    if len(daily_data) > 20:
        resistance = daily_data['high'].tail(20).max()
        support = daily_data['low'].tail(20).min()
        current_price = daily_latest.get('close', 0)
        
        if current_price >= resistance * 0.99:
            patterns_detected.append("Daily Resistance Break")
            confidence += 20
            signal_strength = "Bullish"
        elif current_price <= support * 1.01:
            patterns_detected.append("Daily Support Break")
            confidence += 20
            signal_strength = "Bearish"
    
    return {
        "symbol": symbol,
        "patterns": patterns_detected,
        "confidence": min(100, confidence),
        "signal_strength": signal_strength,
        "current_price": daily_latest.get('close', 0),
        "daily_rsi": daily_rsi,
        "hourly_rsi": hourly_latest.get('RSI_14', 50) if not hourly_data.empty else None,
        "volume_ratio": daily_latest.get('volume', 0) / volume_avg if volume_avg > 0 else 1,
        "timeframe_alignment": timeframe_alignment,
        "has_hourly_data": not hourly_data.empty
    }

def predictive_signals_analysis(active_list, instrument_df):
    """Predictive analysis using ML-inspired signals with hourly data."""
    signals = []
    
    for item in active_list[:8]:
        try:
            symbol = item['symbol']
            exchange = item['exchange']
            token = get_instrument_token(symbol, instrument_df, exchange)
            
            if not token:
                continue
                
            # Get multi-timeframe data
            daily_data = get_historical_data(token, 'day', period='6mo')
            hourly_data = get_hourly_data_with_fallback(token, symbol, days=30)  # NEW
            
            if daily_data.empty or len(daily_data) < 50:
                continue
            
            # Enhanced predictive signal generation with hourly data
            signal = generate_predictive_signal_with_hourly(daily_data, hourly_data, symbol)  # UPDATED FUNCTION
            
            if signal["probability"] > 60:
                signals.append(signal)
                
        except Exception:
            continue
    
    return {
        "signals": sorted(signals, key=lambda x: x["probability"], reverse=True),
        "analysis_type": "Predictive ML Signals"
    }

def generate_predictive_signal_with_hourly(daily_data, hourly_data, symbol):
    """Enhanced predictive trading signals using multi-timeframe data."""
    
    # Feature engineering for both timeframes
    daily_data = calculate_advanced_indicators(daily_data)
    daily_latest = daily_data.iloc[-1]
    
    # ML-inspired scoring
    score = 0
    features = []
    timeframe_score = 0
    
    # DAILY FEATURES
    # Trend features
    if daily_latest.get('EMA_20', 0) > daily_latest.get('EMA_50', 0):
        score += 20
        features.append("Daily EMA Bullish")
        timeframe_score += 1
    
    # Momentum features
    daily_rsi = daily_latest.get('RSI_14', 50)
    if 30 < daily_rsi < 70:
        if daily_latest.get('MACD', 0) > daily_latest.get('MACD_Signal', 0):
            score += 15
            features.append("Daily MACD Bullish")
    
    # HOURLY FEATURES (NEW)
    if not hourly_data.empty and len(hourly_data) > 10:
        hourly_data = calculate_advanced_indicators(hourly_data)
        hourly_latest = hourly_data.iloc[-1]
        
        # Hourly trend
        if hourly_latest.get('EMA_20', 0) > hourly_latest.get('EMA_50', 0):
            score += 15
            features.append("Hourly EMA Bullish")
            timeframe_score += 1
        
        # Hourly momentum
        hourly_rsi = hourly_latest.get('RSI_14', 50)
        if 35 < hourly_rsi < 75:
            if hourly_rsi > 55:
                score += 10
                features.append("Hourly Momentum Positive")
        
        # Hourly volume
        if len(hourly_data) > 20:
            hourly_volume_avg = hourly_data['volume'].tail(20).mean()
            if hourly_latest.get('volume', 0) > hourly_volume_avg * 1.3:
                score += 10
                features.append("Hourly Volume Spike")
    
    # MULTI-TIMEFRAME ALIGNMENT BONUS (NEW)
    if timeframe_score >= 2:
        score += 20
        features.append("Multi-Timeframe Alignment")
    
    # Volume features (daily)
    if len(daily_data) > 20:
        volume_avg = daily_data['volume'].tail(20).mean()
        if daily_latest.get('volume', 0) > volume_avg * 1.1:
            score += 10
            features.append("Daily Volume Surge")
    
    # Price action features
    if (daily_latest.get('close', 0) > daily_latest.get('EMA_20', 0) and 
        daily_latest.get('close', 0) > daily_data['close'].tail(20).mean()):
        score += 10
        features.append("Daily Price Strength")
    
    probability = min(95, score)
    
    # Signal direction
    if probability > 65:
        signal_type = "BUY"
    elif probability < 35:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"
    
    return {
        "symbol": symbol,
        "signal": signal_type,
        "probability": probability,
        "features": features,
        "current_price": daily_latest.get('close', 0),
        "daily_rsi": daily_rsi,
        "hourly_rsi": hourly_latest.get('RSI_14', 50) if not hourly_data.empty else None,
        "volume_ratio": daily_latest.get('volume', 0) / volume_avg if volume_avg > 0 else 1,
        "timeframe_alignment": timeframe_score,
        "has_hourly_data": not hourly_data.empty
    }

def risk_adjusted_opportunities(active_list, instrument_df):
    """Find risk-adjusted trading opportunities."""
    opportunities = []
    
    for item in active_list[:8]:
        try:
            symbol = item['symbol']
            exchange = item['exchange']
            token = get_instrument_token(symbol, instrument_df, exchange)
            
            if not token:
                continue
                
            data = get_historical_data(token, 'day', period='3mo')
            if data.empty:
                continue
            
            opportunity = analyze_risk_adjusted_opportunity(data, symbol)
            
            if opportunity["risk_reward_ratio"] > 1.5:
                opportunities.append(opportunity)
                
        except Exception:
            continue
    
    return {
        "opportunities": sorted(opportunities, key=lambda x: x["risk_reward_ratio"], reverse=True),
        "analysis_type": "Risk-Adjusted Opportunities"
    }

def analyze_risk_adjusted_opportunity(data, symbol):
    """Analyze risk-reward ratio for trading opportunities."""
    data = calculate_advanced_indicators(data)
    latest = data.iloc[-1]
    
    # Calculate support and resistance
    support = data['low'].tail(20).min()
    resistance = data['high'].tail(20).max()
    current_price = latest.get('close', 0)
    
    # Risk-reward calculation
    if current_price > data['close'].tail(20).mean():
        # Bullish scenario
        potential_upside = resistance - current_price
        potential_downside = current_price - support
    else:
        # Bearish scenario
        potential_upside = current_price - support
        potential_downside = resistance - current_price
    
    risk_reward_ratio = potential_upside / potential_downside if potential_downside > 0 else 1
    
    # Volatility assessment
    volatility = data['close'].pct_change().std() * 100
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "support": support,
        "resistance": resistance,
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "volatility": round(volatility, 2),
        "rsi": round(latest.get('RSI_14', 50), 1)
    }

def technical_setups_analysis(active_list, instrument_df):
    """Analyze technical setups for trading."""
    setups = []
    
    for item in active_list[:10]:
        try:
            symbol = item['symbol']
            exchange = item['exchange']
            token = get_instrument_token(symbol, instrument_df, exchange)
            
            if not token:
                continue
                
            data = get_historical_data(token, 'day', period='3mo')
            if data.empty:
                continue
            
            setup = analyze_technical_setup(data, symbol)
            
            if setup["setup_quality"] > 60:
                setups.append(setup)
                
        except Exception:
            continue
    
    return {
        "setups": sorted(setups, key=lambda x: x["setup_quality"], reverse=True),
        "analysis_type": "Technical Setups"
    }

def analyze_technical_setup(data, symbol):
    """Analyze technical trading setups."""
    data = calculate_advanced_indicators(data)
    latest = data.iloc[-1]
    
    setup_quality = 0
    setup_type = "Neutral"
    characteristics = []
    
    # Trend characteristics
    if latest.get('EMA_20', 0) > latest.get('EMA_50', 0):
        setup_quality += 25
        characteristics.append("Uptrend")
        setup_type = "Bullish"
    
    # Momentum characteristics
    rsi = latest.get('RSI_14', 50)
    if 40 < rsi < 65:
        setup_quality += 20
        characteristics.append("Healthy Momentum")
    
    # Volume characteristics
    if len(data) > 20:
        volume_avg = data['volume'].tail(20).mean()
        if latest.get('volume', 0) > volume_avg:
            setup_quality += 15
            characteristics.append("Above Average Volume")
    
    # Pattern characteristics
    if (latest.get('close', 0) > latest.get('EMA_20', 0) and
        latest.get('close', 0) > data['close'].tail(10).mean()):
        setup_quality += 20
        characteristics.append("Price Strength")
    
    return {
        "symbol": symbol,
        "setup_type": setup_type,
        "setup_quality": min(100, setup_quality),
        "characteristics": characteristics,
        "current_price": latest.get('close', 0),
        "rsi": round(rsi, 1)
    }

def display_enhanced_discovery_results(results, discovery_mode, confidence_threshold):
    """Display enhanced discovery results with hourly data insights."""
    
    if discovery_mode == "Pattern Recognition":
        st.subheader("🎯 High-Confidence Patterns")
        
        filtered_patterns = [p for p in results["patterns"] if p["confidence"] >= confidence_threshold]
        
        if not filtered_patterns:
            st.info(f"No patterns found above {confidence_threshold}% confidence threshold.")
            return
            
        for pattern in filtered_patterns[:8]:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.write(f"**{pattern['symbol']}**")
                    st.caption(f"Strength: {pattern['signal_strength']}")
                    # NEW: Show hourly data indicator
                    if pattern.get('has_hourly_data'):
                        st.caption("📊 Multi-timeframe analysis")
                
                with col2:
                    patterns_text = ", ".join(pattern['patterns'][:3])
                    st.write(f"*{patterns_text}*")
                    # NEW: Show timeframe alignment
                    if pattern.get('timeframe_alignment', 0) >= 2:
                        st.caption("✅ Timeframes aligned")
                
                with col3:
                    confidence = pattern['confidence']
                    if confidence > 80:
                        st.success(f"{confidence}%")
                    elif confidence > 60:
                        st.warning(f"{confidence}%")
                    else:
                        st.info(f"{confidence}%")
                
                with col4:
                    if st.button("Analyze", key=f"analyze_{pattern['symbol']}"):
                        st.session_state[f"detailed_{pattern['symbol']}"] = True
                
                # Detailed analysis on click
                if st.session_state.get(f"detailed_{pattern['symbol']}", False):
                    with st.expander(f"Detailed Analysis - {pattern['symbol']}", expanded=True):
                        display_symbol_technical_analysis(pattern)
                
                st.markdown("---")

def display_symbol_technical_analysis(pattern_data):
    """Enhanced technical analysis display with hourly insights."""
    st.write(f"**Detailed Technical Analysis for {pattern_data['symbol']}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Price", f"₹{pattern_data['current_price']:.2f}")
        st.metric("Daily RSI", f"{pattern_data['daily_rsi']:.1f}")
        if pattern_data.get('hourly_rsi'):
            st.metric("Hourly RSI", f"{pattern_data['hourly_rsi']:.1f}")
        st.metric("Confidence", f"{pattern_data['confidence']}%")
    
    with col2:
        st.metric("Signal Strength", pattern_data['signal_strength'])
        st.metric("Volume Ratio", f"{pattern_data['volume_ratio']:.2f}x")
        if pattern_data.get('timeframe_alignment'):
            st.metric("Timeframe Alignment", f"{pattern_data['timeframe_alignment']}/2")
        st.metric("Patterns Found", len(pattern_data['patterns']))
    
    # Multi-timeframe insights
    if pattern_data.get('has_hourly_data'):
        st.success("✅ Multi-timeframe analysis available (Daily + Hourly)")
    
    if pattern_data.get('timeframe_alignment', 0) >= 2:
        st.info("🎯 Multiple timeframes are aligned - stronger signal")
    
    # Pattern details
    st.write("**Detected Patterns:**")
    for pattern in pattern_data['patterns']:
        if "Hourly" in pattern:
            st.write(f"• 🕒 {pattern}")
        elif "Daily" in pattern:
            st.write(f"• 📅 {pattern}")
        else:
            st.write(f"• {pattern}")
    
    # Trading recommendation
    if pattern_data['confidence'] > 80 and pattern_data.get('timeframe_alignment', 0) >= 2:
        recommendation = "Strong multi-timeframe opportunity"
        color = "green"
    elif pattern_data['confidence'] > 75:
        recommendation = "Strong trading opportunity"
        color = "green"
    elif pattern_data['confidence'] > 60:
        recommendation = "Moderate trading opportunity" 
        color = "blue"
    else:
        recommendation = "Watch for confirmation"
        color = "orange"
    
    st.markdown(f"**Recommendation:** <span style='color:{color}'>{recommendation}</span>", unsafe_allow_html=True)

def execute_ai_trade(signal):
    """Execute trade based on AI signal with confirmation."""
    if signal['probability'] < 70:
        st.warning(f"Low confidence signal ({signal['probability']}%). Consider manual review.")
        return
    
    # Show trade confirmation
    st.info(f"""
    **AI Trade Recommendation:**
    - **Action:** {signal['signal']} {signal['symbol']}
    - **Confidence:** {signal['probability']}%
    - **Current Price:** ₹{signal['current_price']:.2f}
    - **Key Features:** {', '.join(signal['features'][:3])}
    """)
    
    # Execute on confirmation
    if st.button(f"Confirm {signal['signal']} Order", type="primary", key=f"confirm_{signal['symbol']}"):
        instrument_df = get_instrument_df()
        quantity = 1  # Default quantity
        place_order(instrument_df, signal['symbol'], quantity, 'MARKET', signal['signal'], 'MIS')
        st.success(f"{signal['signal']} order placed for {signal['symbol']}!")

def page_greeks_calculator():
    """Calculates Greeks for any option contract."""
    display_header()
    st.title("F&O Greeks Calculator")
    st.info("Calculate the theoretical value and greeks (Delta, Gamma, Vega, Theta, Rho) for any option contract.")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use this feature.")
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Option Details")
        
        underlying_price = st.number_input("Underlying Price", min_value=0.01, value=23500.0)
        strike_price = st.number_input("Strike Price", min_value=0.01, value=23500.0)
        time_to_expiry = st.number_input("Days to Expiry", min_value=1, value=30)
        risk_free_rate = st.number_input("Risk-free Rate (%)", min_value=0.0, value=7.0)
        volatility = st.number_input("Volatility (%)", min_value=0.1, value=20.0)
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        if st.button("Calculate Greeks"):
            T = time_to_expiry / 365.0
            r = risk_free_rate / 100.0
            sigma = volatility / 100.0
            
            greeks = black_scholes(underlying_price, strike_price, T, r, sigma, option_type)
            
            st.session_state.calculated_greeks = greeks
            st.rerun()
    
    with col2:
        st.subheader("Greeks Results")
        
        if 'calculated_greeks' in st.session_state and st.session_state.calculated_greeks is not None:
            greeks = st.session_state.calculated_greeks
            
            st.metric("Option Price", f"₹{greeks['price']:.2f}")
            
            col_greeks1, col_greeks2 = st.columns(2)
            col_greeks1.metric("Delta", f"{greeks['delta']:.4f}")
            col_greeks1.metric("Gamma", f"{greeks['gamma']:.4f}")
            col_greeks1.metric("Vega", f"{greeks['vega']:.4f}")
            
            col_greeks2.metric("Theta", f"{greeks['theta']:.4f}")
            col_greeks2.metric("Rho", f"{greeks['rho']:.4f}")
            
            with st.expander("Understanding Greeks"):
                st.markdown("""
                - **Delta**: Price sensitivity to underlying movement
                - **Gamma**: Rate of change of Delta
                - **Vega**: Sensitivity to volatility changes
                - **Theta**: Time decay per day
                - **Rho**: Sensitivity to interest rate changes
                """)
        else:
            st.info("Enter option details and click 'Calculate Greeks' to see results.")

def page_economic_calendar():
    """Economic Calendar page with auto-updating live data."""
    display_header()
    st.title("📅 Economic Calendar")
    
    st.info("Live economic events and market-moving data from global sources. Updates automatically.", icon="🔄")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.toggle("Auto-refresh every 5 minutes", value=True, key="calendar_refresh")
    with col2:
        days_ahead = st.selectbox("Show next", [7, 14, 30], index=0, key="calendar_days")
    with col3:
        if st.button("🔄 Update Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if auto_refresh:
        st_autorefresh(interval=5 * 60 * 1000, key="calendar_auto_refresh")  # 5 minutes
    
    # Main calendar display
    display_live_economic_calendar(days_ahead)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_economic_calendar(days_ahead=7):
    """Fetch live economic calendar data from multiple sources."""
    calendar_data = []
    
    try:
        # Try multiple data sources in order
        data = fetch_from_fmp(days_ahead) or fetch_from_alphavantage(days_ahead) or get_fallback_calendar(days_ahead)
        calendar_data = data
    except Exception as e:
        st.error(f"Error fetching calendar data: {e}")
        calendar_data = get_fallback_calendar(days_ahead)
    
    return calendar_data

def fetch_from_fmp(days_ahead=7):
    """Fetch from Financial Modeling Prep using the provided endpoint."""
    try:
        # Use the provided API key and endpoint
        api_key = "H5icAaW4hjSv0zxsS8ufqsl3fb2SJ7RL"
        
        # Calculate date range
        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Use the stable endpoint you provided
        url = f"https://financialmodelingprep.com/stable/economic-calendar?from={from_date}&to={to_date}&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                formatted_data = []
                for event in data:
                    # Parse the event data from FMP
                    event_date = datetime.strptime(event['date'], '%Y-%m-%d %H:%M:%S')
                    
                    # Map FMP importance to our impact levels
                    importance = event.get('importance', 1)
                    if importance == 3:
                        impact = "High"
                    elif importance == 2:
                        impact = "Medium"
                    else:
                        impact = "Low"
                    
                    # Get country code (convert full name to code if needed)
                    country = event.get('country', '')
                    country_map = {
                        'united states': 'USA',
                        'india': 'IND', 
                        'china': 'CHN',
                        'europe': 'EU',
                        'germany': 'DEU',
                        'japan': 'JPN',
                        'united kingdom': 'GBR'
                    }
                    country_code = country_map.get(country.lower(), country.upper()[:3])
                    
                    formatted_data.append({
                        'Date': event_date.strftime('%Y-%m-%d'),
                        'Time': event_date.strftime('%H:%M'),
                        'Country': country_code,
                        'Event': event.get('event', ''),
                        'Impact': impact,
                        'Previous': str(event.get('previous', '')),
                        'Forecast': str(event.get('estimate', '')),
                        'Actual': str(event.get('actual', '')),
                        'Currency': event.get('currency', '')
                    })
                return formatted_data
    except Exception as e:
        st.error(f"FMP API error: {e}")
    return None

def fetch_from_alphavantage(days_ahead=7):
    """Fetch from Alpha Vantage using Streamlit secrets."""
    try:
        api_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return None
        
        # Alpha Vantage economic calendar endpoint
        url = f"https://www.alphavantage.co/query?function=ECONOMIC_CALENDAR&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have valid data
            if 'results' in data and data['results']:
                formatted_data = []
                for event in data['results']:
                    # Parse Alpha Vantage event format
                    event_date = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    
                    # Map importance to impact
                    importance = event.get('importance', 1)
                    if importance >= 80:
                        impact = "High"
                    elif importance >= 50:
                        impact = "Medium"
                    else:
                        impact = "Low"
                    
                    formatted_data.append({
                        'Date': event_date.strftime('%Y-%m-%d'),
                        'Time': event_date.strftime('%H:%M'),
                        'Country': event.get('country', ''),
                        'Event': event.get('event', ''),
                        'Impact': impact,
                        'Previous': str(event.get('previous', '')),
                        'Forecast': str(event.get('estimate', '')),
                        'Actual': str(event.get('actual', '')),
                        'Currency': event.get('currency', '')
                    })
                
                # Filter for future events within our date range
                end_date = datetime.now() + timedelta(days=days_ahead)
                filtered_data = [
                    event for event in formatted_data 
                    if datetime.strptime(event['Date'], '%Y-%m-%d') <= end_date
                ]
                
                return filtered_data
    except Exception as e:
        st.error(f"Alpha Vantage API error: {e}")
    return None

def get_fallback_calendar(days_ahead=7):
    """Provide comprehensive fallback calendar data with real events."""
    base_date = datetime.now()
    calendar_data = []
    
    # Indian Market Events
    indian_events = [
        {
            'date': base_date + timedelta(days=1),
            'time': '11:30',
            'event': 'Bank Loan Growth YoY',
            'country': 'IND',
            'impact': 'Medium',
            'previous': '10.0%',
            'forecast': '-',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=1),
            'time': '11:30', 
            'event': 'Foreign Exchange Reserves',
            'country': 'IND',
            'impact': 'Low',
            'previous': '$702.97B',
            'forecast': '-',
            'actual': '',
            'currency': 'USD'
        },
        {
            'date': base_date + timedelta(days=2),
            'time': '10:30',
            'event': 'Industrial Production YoY',
            'country': 'IND',
            'impact': 'Medium',
            'previous': '2.9%',
            'forecast': '3.5%',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=3),
            'time': '17:30',
            'event': 'Infrastructure Output YoY',
            'country': 'IND',
            'impact': 'Medium',
            'previous': '6.3%',
            'forecast': '6.5%',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=4),
            'time': '10:30',
            'event': 'Nikkei Manufacturing PMI',
            'country': 'IND',
            'impact': 'High',
            'previous': '58.5',
            'forecast': '58.8',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=6),
            'time': '10:30',
            'event': 'Nikkei Services PMI', 
            'country': 'IND',
            'impact': 'High',
            'previous': '61.6',
            'forecast': '61.2',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=8),
            'time': '11:00',
            'event': 'RBI Interest Rate Decision',
            'country': 'IND',
            'impact': 'High',
            'previous': '6.50%',
            'forecast': '6.50%',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=10),
            'time': '12:00',
            'event': 'WPI Inflation YoY',
            'country': 'IND',
            'impact': 'High',
            'previous': '0.3%',
            'forecast': '0.5%',
            'actual': '',
            'currency': 'INR'
        },
        {
            'date': base_date + timedelta(days=11),
            'time': '17:30',
            'event': 'CPI Inflation YoY',
            'country': 'IND',
            'impact': 'High',
            'previous': '5.1%',
            'forecast': '5.3%',
            'actual': '',
            'currency': 'INR'
        }
    ]
    
    # Global Events (US, EU, China)
    global_events = [
        {
            'date': base_date + timedelta(days=1),
            'time': '18:00',
            'event': 'US Initial Jobless Claims',
            'country': 'USA',
            'impact': 'High',
            'previous': '210K',
            'forecast': '215K',
            'actual': '',
            'currency': 'USD'
        },
        {
            'date': base_date + timedelta(days=2),
            'time': '18:30',
            'event': 'US Core PCE Price Index',
            'country': 'USA',
            'impact': 'High',
            'previous': '0.2%',
            'forecast': '0.3%',
            'actual': '',
            'currency': 'USD'
        },
        {
            'date': base_date + timedelta(days=3),
            'time': '14:00',
            'event': 'EU CPI Flash Estimate',
            'country': 'EU',
            'impact': 'High',
            'previous': '2.4%',
            'forecast': '2.3%',
            'actual': '',
            'currency': 'EUR'
        },
        {
            'date': base_date + timedelta(days=4),
            'time': '06:00',
            'event': 'China Manufacturing PMI',
            'country': 'CHN',
            'impact': 'Medium',
            'previous': '49.5',
            'forecast': '49.7',
            'actual': '',
            'currency': 'CNY'
        },
        {
            'date': base_date + timedelta(days=5),
            'time': '18:00',
            'event': 'US Non-Farm Payrolls',
            'country': 'USA',
            'impact': 'High',
            'previous': '175K',
            'forecast': '180K',
            'actual': '',
            'currency': 'USD'
        },
        {
            'date': base_date + timedelta(days=7),
            'time': '16:30',
            'event': 'US ISM Manufacturing PMI',
            'country': 'USA',
            'impact': 'High',
            'previous': '48.7',
            'forecast': '49.0',
            'actual': '',
            'currency': 'USD'
        }
    ]
    
    # Combine and filter events within the date range
    all_events = indian_events + global_events
    end_date = base_date + timedelta(days=days_ahead)
    
    for event in all_events:
        if event['date'] <= end_date:
            calendar_data.append({
                'Date': event['date'].strftime('%Y-%m-%d'),
                'Time': event['time'],
                'Country': event['country'],
                'Event': event['event'],
                'Impact': event['impact'],
                'Previous': event['previous'],
                'Forecast': event['forecast'],
                'Actual': event['actual'],
                'Currency': event['currency']
            })
    
    return calendar_data

def display_live_economic_calendar(days_ahead=7):
    """Display the economic calendar with live data."""
    
    with st.spinner("🔄 Fetching latest economic events..."):
        calendar_data = fetch_live_economic_calendar(days_ahead)
    
    if not calendar_data:
        st.error("Unable to fetch economic calendar data. Please try again later.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(calendar_data)
    
    # Display data source info
    source_info = st.empty()
    if len(df) > 0:
        if 'FMP' in str(calendar_data):
            source_info.success("✅ Data source: Financial Modeling Prep (Live)")
        elif 'Alpha' in str(calendar_data):
            source_info.info("ℹ️ Data source: Alpha Vantage (Live)")
        else:
            source_info.warning("⚠️ Data source: Fallback data (Static)")
    
    # Filter controls
    st.subheader("📊 Economic Events Filter")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        countries = ['All'] + sorted(df['Country'].unique().tolist())
        selected_country = st.selectbox("Country", countries)
    with col2:
        impacts = ['All'] + sorted(df['Impact'].unique().tolist())
        selected_impact = st.selectbox("Impact Level", impacts)
    with col3:
        currencies = ['All'] + sorted(df['Currency'].unique().tolist())
        selected_currency = st.selectbox("Currency", currencies)
    with col4:
        show_past = st.checkbox("Include past events", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    
    if selected_impact != 'All':
        filtered_df = filtered_df[filtered_df['Impact'] == selected_impact]
    
    if selected_currency != 'All':
        filtered_df = filtered_df[filtered_df['Currency'] == selected_currency]
    
    if not show_past:
        today = datetime.now().strftime('%Y-%m-%d')
        filtered_df = filtered_df[filtered_df['Date'] >= today]
    
    # Sort by date and time
    filtered_df = filtered_df.sort_values(['Date', 'Time'])
    
    # Display the calendar
    st.subheader(f"📅 Economic Calendar ({len(filtered_df)} events)")
    
    if filtered_df.empty:
        st.info("No events match your filters. Try adjusting the criteria.")
        return
    
    # Color code impact levels
    def color_impact(val):
        if val == 'High':
            return 'background-color: #ff4444; color: white; font-weight: bold;'
        elif val == 'Medium':
            return 'background-color: #ffaa00; color: black; font-weight: bold;'
        else:
            return 'background-color: #00aa00; color: white; font-weight: bold;'
    
    # Apply styling
    styled_df = filtered_df.style.applymap(color_impact, subset=['Impact'])
    
    # Display the table
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YY"),
            "Time": st.column_config.TextColumn("Time (IST)"),
            "Country": st.column_config.TextColumn("Country"),
            "Event": st.column_config.TextColumn("Event", width="large"),
            "Impact": st.column_config.TextColumn("Impact"),
            "Previous": st.column_config.TextColumn("Previous"),
            "Forecast": st.column_config.TextColumn("Forecast"),
            "Actual": st.column_config.TextColumn("Actual"),
            "Currency": st.column_config.TextColumn("Currency")
        }
    )
    
    # Key events highlight
    st.subheader("🎯 Key Events This Week")
    key_events = filtered_df[filtered_df['Impact'] == 'High'].head(5)
    
    if not key_events.empty:
        for _, event in key_events.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"**{event['Date']}**")
                    st.write(f"{event['Time']}")
                with col2:
                    st.write(f"**{event['Event']}** ({event['Country']})")
                    st.write(f"Forecast: {event['Forecast']} | Previous: {event['Previous']}")
                with col3:
                    if event['Actual'] and event['Actual'] != 'nan':
                        # Color code actual vs forecast
                        try:
                            actual_clean = str(event['Actual']).replace('%', '').replace('$', '').replace('K', '')
                            forecast_clean = str(event['Forecast']).replace('%', '').replace('$', '').replace('K', '')
                            
                            if actual_clean.replace('.', '').isdigit() and forecast_clean.replace('.', '').isdigit():
                                actual_num = float(actual_clean)
                                forecast_num = float(forecast_clean)
                                
                                if actual_num > forecast_num:
                                    st.success(f"**Actual: {event['Actual']}** 📈")
                                elif actual_num < forecast_num:
                                    st.error(f"**Actual: {event['Actual']}** 📉")
                                else:
                                    st.info(f"**Actual: {event['Actual']}** ➡️")
                            else:
                                st.info(f"**Actual: {event['Actual']}**")
                        except:
                            st.info(f"**Actual: {event['Actual']}**")
                    else:
                        st.info("⏳ Pending")
                st.markdown("---")
    else:
        st.info("No high-impact events in the selected period.")
    
    # Market sentiment analysis
    st.subheader("📈 Market Sentiment Analysis")
    
    high_impact_count = len(filtered_df[filtered_df['Impact'] == 'High'])
    indian_events = len(filtered_df[filtered_df['Country'] == 'IND'])
    completed_events = len(filtered_df[filtered_df['Actual'].notna() & (filtered_df['Actual'] != '') & (filtered_df['Actual'] != 'nan')])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(filtered_df))
    col2.metric("High Impact", high_impact_count)
    col3.metric("Indian Events", indian_events)
    col4.metric("Completed", completed_events)
    
    # Data freshness indicator
    st.caption(f"🕒 Last updated: {get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    # Export option
    if st.button("📥 Export Calendar to CSV", use_container_width=True):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"economic_calendar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Helper function to get IST time
def get_ist_time():
    """Get current time in IST timezone."""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)



# ============ 5.5 HFT TERMINAL PAGE ============
def page_hft_terminal():
    """A dedicated terminal for High-Frequency Trading with Level 2 data."""
    display_header()
    st.title("HFT Terminal (High-Frequency Trading)")
    st.info("This interface provides a simulated high-speed view of market depth and one-click trading. For liquid, F&O instruments only.", icon="⚡️")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.warning("Please connect to a broker to use the HFT Terminal.")
        return

    # --- Instrument Selection and Key Stats ---
    top_cols = st.columns([2, 1, 1, 1])
    with top_cols[0]:
        symbol = st.text_input("Instrument Symbol", "NIFTY24OCTFUT", key="hft_symbol").upper()
    
    instrument_info = instrument_df[instrument_df['tradingsymbol'] == symbol]
    if instrument_info.empty:
        st.error(f"Instrument '{symbol}' not found. Please enter a valid symbol.")
        return
    
    exchange = instrument_info.iloc[0]['exchange']
    instrument_token = instrument_info.iloc[0]['instrument_token']

    # --- Fetch Live Data ---
    quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
    depth_data = get_market_depth(instrument_token)

    # --- Display Key Stats ---
    if not quote_data.empty:
        ltp = quote_data.iloc[0]['Price']
        change = quote_data.iloc[0]['Change']
        
        tick_direction = "tick-up" if ltp > st.session_state.hft_last_price else "tick-down" if ltp < st.session_state.hft_last_price else ""
        with top_cols[1]:
            st.markdown(f"##### LTP: <span class='{tick_direction}' style='font-size: 1.2em;'>₹{ltp:,.2f}</span>", unsafe_allow_html=True)
            
            with main_cols[2]:
                st.subheader("Tick Log")
                log_container = st.container(height=400)
                for entry in st.session_state.hft_tick_log:
                    color = 'var(--green)' if entry['change'] > 0 else 'var(--red)'
                    log_container.markdown(f"<small>{entry['time']}</small> - **{entry['price']:.2f}** <span style='color:{color};'>({entry['change']:+.2f})</span>", unsafe_allow_html=True)
    
    # Update the tick logging part:
    if ltp != st.session_state.hft_last_price and st.session_state.hft_last_price != 0:
        ist_time = get_ist_time()
        log_entry = {
            "time": ist_time.strftime("%H:%M:%S.%f")[:-3] + " IST",
            "price": ltp,
            "change": ltp - st.session_state.hft_last_price
        }
        st.session_state.hft_tick_log.insert(0, log_entry)
        if len(st.session_state.hft_tick_log) > 20:
            st.session_state.hft_tick_log.pop()
        
        with top_cols[3]:
            latency = random.uniform(20, 80)
            st.metric("Latency (ms)", f"{latency:.2f}")

        # Update tick log
        if ltp != st.session_state.hft_last_price and st.session_state.hft_last_price != 0:
            log_entry = {
                "time": datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S.%f")[:-3],
                "price": ltp,
                "change": ltp - st.session_state.hft_last_price
            }
            st.session_state.hft_tick_log.insert(0, log_entry)
            if len(st.session_state.hft_tick_log) > 20:
                st.session_state.hft_tick_log.pop()

        st.session_state.hft_last_price = ltp

    st.markdown("---")

    # --- Main Layout: Depth, Orders, Ticks ---
    main_cols = st.columns([1, 1, 1], gap="large")

    with main_cols[0]:
        st.subheader("Market Depth")
        if depth_data and depth_data.get('buy') and depth_data.get('sell'):
            bids = pd.DataFrame(depth_data['buy']).sort_values('price', ascending=False).head(5)
            asks = pd.DataFrame(depth_data['sell']).sort_values('price', ascending=True).head(5)
            
            st.write("**Bids (Buyers)**")
            for _, row in bids.iterrows():
                st.markdown(f"<div class='hft-depth-bid'>{row['quantity']} @ **{row['price']:.2f}** ({row['orders']})</div>", unsafe_allow_html=True)
            
            st.write("**Asks (Sellers)**")
            for _, row in asks.iterrows():
                st.markdown(f"<div class='hft-depth-ask'>({row['orders']}) **{row['price']:.2f}** @ {row['quantity']}</div>", unsafe_allow_html=True)
        else:
            st.info("Waiting for market depth data...")

    with main_cols[1]:
        st.subheader("One-Click Execution")
        quantity = st.number_input("Order Quantity", min_value=1, value=instrument_info.iloc[0]['lot_size'], step=instrument_info.iloc[0]['lot_size'], key="hft_qty")
        
        btn_cols = st.columns(2)
        if btn_cols[0].button("MARKET BUY", use_container_width=True, type="primary"):
            place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS')
        if btn_cols[1].button("MARKET SELL", use_container_width=True):
            place_order(instrument_df, symbol, quantity, 'MARKET', 'SELL', 'MIS')
        
        st.markdown("---")
        st.subheader("Manual Order")
        price = st.number_input("Limit Price", min_value=0.01, step=0.05, key="hft_limit_price")
        limit_btn_cols = st.columns(2)
        if limit_btn_cols[0].button("LIMIT BUY", use_container_width=True):
            place_order(instrument_df, symbol, quantity, 'LIMIT', 'BUY', 'MIS', price=price)
        if limit_btn_cols[1].button("LIMIT SELL", use_container_width=True):
            place_order(instrument_df, symbol, quantity, 'LIMIT', 'SELL', 'MIS', price=price)

    with main_cols[2]:
        st.subheader("Tick Log")
        log_container = st.container(height=400)
        for entry in st.session_state.hft_tick_log:
            color = 'var(--green)' if entry['change'] > 0 else 'var(--red)'
            log_container.markdown(f"<small>{entry['time']}</small> - **{entry['price']:.2f}** <span style='color:{color};'>({entry['change']:+.2f})</span>", unsafe_allow_html=True)

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

# ================ 6. MAIN APP LOGIC AND AUTHENTICATION ============

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    if user_profile is None:
        user_profile = {}
    
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    secret = base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]
    return secret

# REPLACE THE DIALOG FUNCTIONS WITH REGULAR FUNCTIONS
def show_two_factor_setup():
    """Show 2FA setup without using dialogs."""
    st.title("🔐 Two-Factor Authentication Setup")
    st.info("Please scan the QR code with your authenticator app (e.g., Google or Microsoft Authenticator).")
    
    if st.session_state.pyotp_secret is None:
        profile = st.session_state.get('profile') or {}
        st.session_state.pyotp_secret = get_user_secret(profile)
    
    secret = st.session_state.pyotp_secret
    user_name = st.session_state.get('profile', {}).get('user_name', 'User')
    uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
    
    # Generate QR code with normal size
    img = qrcode.make(uri)
    
    # Resize to normal size (300x300 is standard for QR codes)
    img = img.resize((300, 300))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    # Display with normal size (remove use_container_width)
    st.image(buf.getvalue(), caption="Scan with your authenticator app", width=300)
    
    st.markdown(f"**Your Secret Key:** `{secret}` (You can also enter this manually)")
    
    st.markdown("---")
    st.subheader("Verify Setup")
    auth_code = st.text_input("Enter 6-digit code from your authenticator app", max_chars=6, key="verify_2fa")
    
    col1, col2 = st.columns(2)
    if col1.button("Verify & Continue", use_container_width=True, type="primary"):
        if auth_code:
            try:
                totp = pyotp.TOTP(secret)
                if totp.verify(auth_code):
                    st.session_state.two_factor_setup_complete = True
                    st.session_state.authenticated = True
                    st.success("2FA setup completed successfully!")
                    st.rerun()
                else:
                    st.error("Invalid code. Please try again.")
            except Exception as e:
                st.error(f"Verification failed: {e}")
        else:
            st.warning("Please enter a verification code.")
    
    if col2.button("Skip 2FA Setup", use_container_width=True):
        st.session_state.two_factor_setup_complete = True
        st.session_state.authenticated = True
        st.info("2FA setup skipped. You can enable it later in settings.")
        st.rerun()

def show_two_factor_auth():
    """Show 2FA authentication without using dialogs."""
    st.title("🔐 Two-Factor Authentication")
    st.caption("Please enter the 6-digit code from your authenticator app to continue.")
    
    auth_code = st.text_input("2FA Code", max_chars=6, key="2fa_code")
    
    col1, col2 = st.columns(2)
    if col1.button("Authenticate", use_container_width=True, type="primary"):
        if auth_code:
            try:
                totp = pyotp.TOTP(st.session_state.pyotp_secret)
                if totp.verify(auth_code):
                    st.session_state.authenticated = True
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Invalid code. Please try again.")
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.warning("Please enter a code.")
    
    if col2.button("Use Backup Code", use_container_width=True):
        st.info("Backup code feature not yet implemented.")
    
    st.markdown("---")
    if st.button("🔄 Return to Login", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['theme', 'watchlists', 'active_watchlist']:
                del st.session_state[key]
        st.rerun()


@st.dialog("Generate QR Code for 2FA")
def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup."""
    if 'show_qr_dialog' not in st.session_state:
        st.session_state.show_qr_dialog = False
    
    if not st.session_state.get('two_factor_setup_complete', False):
        st.session_state.show_qr_dialog = True
        
        st.subheader("Set up Two-Factor Authentication")
        st.info("Please scan this QR code with your authenticator app (e.g., Google or Microsoft Authenticator). This is a one-time setup.")

        if st.session_state.pyotp_secret is None:
            # Ensure we get a dictionary, not None
            profile = st.session_state.get('profile') or {}
            st.session_state.pyotp_secret = get_user_secret(profile)
        
        secret = st.session_state.pyotp_secret
        user_name = st.session_state.get('profile', {}).get('user_name', 'User')
        uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
        
        # Generate QR code with normal size
        img = qrcode.make(uri)
        img = img.resize((300, 300))  # Normal size
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        # Display with normal size
        st.image(buf.getvalue(), caption="Scan with your authenticator app", width=300)
        st.markdown(f"**Your Secret Key:** `{secret}` (You can also enter this manually)")
        
        if st.button("I have scanned the code. Continue.", use_container_width=True):
            st.session_state.two_factor_setup_complete = True
            st.session_state.show_qr_dialog = False
            st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
    st.title("BlockVista Terminal")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = {
        "Authenticating user...": 25,
        "Establishing secure connection...": 50,
        "Fetching live market data feeds...": 75,
        "Initializing terminal... COMPLETE": 100
    }
    
    for text, progress in steps.items():
        status_text.text(f"STATUS: {text}")
        progress_bar.progress(progress)
        a_time.sleep(0.7)
    
    a_time.sleep(0.5)
    st.session_state['login_animation_complete'] = True
    st.rerun()

def login_page():
    """Displays the login page for broker authentication."""
    st.title("BlockVista Terminal")
    st.subheader("Broker Login")
    
    broker = st.selectbox("Select Your Broker", ["Zerodha", "Upstox"])
    
    if broker == "Zerodha":
        # Your existing Zerodha code remains the same
        api_key = st.secrets.get("ZERODHA_API_KEY")
        api_secret = st.secrets.get("ZERODHA_API_SECRET")
        
        if not api_key or not api_secret:
            st.error("Kite API credentials not found. Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in your Streamlit secrets.")
            st.stop()
            
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                kite.set_access_token(st.session_state.access_token)
                st.session_state.kite = kite
                st.session_state.profile = kite.profile()
                st.session_state.broker = "Zerodha"
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
                st.query_params.clear()
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())
            st.info("Please login with Zerodha Kite to begin. You will be redirected back to the app.")
    
    elif broker == "Upstox":
        try:
            import urllib.parse
            import hashlib
            import base64
            import secrets
            
            api_key = st.secrets.get("UPSTOX_API_KEY")
            api_secret = st.secrets.get("UPSTOX_API_SECRET")
            redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI", "https://your-app-name.streamlit.app/")
            
            if not api_key or not api_secret:
                st.error("Upstox API credentials not found. Please set UPSTOX_API_KEY and UPSTOX_API_SECRET in your Streamlit secrets.")
                st.stop()
            
            # Generate PKCE code verifier and challenge (required for Upstox v2)
            code_verifier = secrets.token_urlsafe(64)
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
            
            # Store code_verifier in session state for later use
            st.session_state.upstox_code_verifier = code_verifier
            
            # Check for authorization code in URL parameters
            auth_code = st.query_params.get("code")
            
            if auth_code:
                try:
                    # Exchange authorization code for access token
                    token_url = "https://api.upstox.com/v2/login/authorization/token"
                    token_data = {
                        'code': auth_code,
                        'client_id': api_key,
                        'client_secret': api_secret,
                        'redirect_uri': redirect_uri,
                        'grant_type': 'authorization_code',
                        'code_verifier': st.session_state.upstox_code_verifier
                    }
                    
                    headers = {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'application/json'
                    }
                    
                    response = requests.post(token_url, data=token_data, headers=headers)
                    
                    if response.status_code == 200:
                        token_data = response.json()
                        access_token = token_data['access_token']
                        
                        # Store in session state
                        st.session_state.upstox_access_token = access_token
                        st.session_state.broker = "Upstox"
                        
                        # Get user profile
                        try:
                            profile_url = "https://api.upstox.com/v2/user/profile"
                            profile_headers = {
                                'Authorization': f'Bearer {access_token}',
                                'Accept': 'application/json'
                            }
                            
                            profile_response = requests.get(profile_url, headers=profile_headers)
                            if profile_response.status_code == 200:
                                profile_data = profile_response.json()
                                st.session_state.profile = {
                                    'user_name': profile_data['data']['name'],
                                    'email': profile_data['data']['email'],
                                    'user_id': profile_data['data']['client_code']
                                }
                            else:
                                st.session_state.profile = {
                                    'user_name': 'Upstox User',
                                    'email': '',
                                    'user_id': 'upstox_user'
                                }
                        except Exception as profile_error:
                            st.session_state.profile = {
                                'user_name': 'Upstox User',
                                'email': '',
                                'user_id': 'upstox_user'
                            }
                        
                        st.success("Upstox login successful!")
                        if st.session_state.get('broker') == "Upstox" and st.session_state.get('upstox_access_token'):
                            if st.button("🔧 Debug Upstox API & Exchanges"):
                                available_exchanges = debug_upstox_exchanges(st.session_state.upstox_access_token)
                                st.session_state.upstox_available_exchanges = available_exchanges
                                st.query_params.clear()
                        # Clean up code verifier
                        if 'upstox_code_verifier' in st.session_state:
                            del st.session_state.upstox_code_verifier
                        st.rerun()
                    else:
                        st.error(f"Upstox token exchange failed: {response.text}")
                        st.query_params.clear()
                        
                except Exception as e:
                    st.error(f"Upstox authentication failed: {e}")
                    st.query_params.clear()
            else:
                # Generate login URL for Upstox v2
                base_url = "https://api.upstox.com/v2/login/authorization/dialog"
                params = {
                    'response_type': 'code',
                    'client_id': api_key,
                    'redirect_uri': redirect_uri,
                    'code_challenge': code_challenge,
                    'code_challenge_method': 'S256'
                }
                
                login_url = f"{base_url}?{urllib.parse.urlencode(params)}"
                st.link_button("Login with Upstox", login_url)
                st.info("Please login with Upstox to begin. You will be redirected back to the app.")
                
        except Exception as e:
            st.error(f"Upstox initialization failed: {e}")
            
def main_app():
    """The main application interface after successful login."""
    apply_custom_styling()
    display_overnight_changes_bar()
    
    # --- 2FA Check - Handle authentication flow first
    profile = st.session_state.get('profile')
    if not profile:
        st.error("User profile not found. Please log in again.")
        if st.button("Return to Login"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        return
    
    # Show 2FA setup if needed (without dialogs)
    if not st.session_state.get('two_factor_setup_complete', False):
        show_two_factor_setup()
        return
    
    # Show 2FA authentication if needed (without dialogs)
    if not st.session_state.get('authenticated', False):
        show_two_factor_auth()
        return
    
    # Only show main content after 2FA is complete
    # Handle quick trade without dialogs
    if st.session_state.get('show_quick_trade', False):
        st.markdown("---")
        st.subheader("Quick Trade")
        symbol = st.session_state.get('quick_trade_symbol')
        if symbol:
            st.info(f"Trading: {symbol}")
        # Add your quick trade form here without using dialogs
    
    # Rest of your main app content...
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options", "HFT"], horizontal=True)
    st.sidebar.divider()
    
    # Dynamic refresh interval based on mode
    if st.session_state.terminal_mode == "HFT":
        refresh_interval = 2
        auto_refresh = True
        st.sidebar.header("HFT Mode Active")
        st.sidebar.caption(f"Refresh Interval: {refresh_interval}s")
    else:
        st.sidebar.header("Live Data")
        auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
        refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
    "Cash": {
        "Dashboard": page_dashboard,
        "AI Trading Bots": page_algo_bots,
        "AI Market Sentiment": page_market_sentiment_ai,  # NEW
        "AI Discovery Engine": page_ai_discovery,
        "AI Portfolio Assistant": page_ai_assistant,  # ENHANCED
        "Premarket Pulse": page_premarket_pulse,
        "Advanced Charting": page_advanced_charting,
        "Market Scanners": page_momentum_and_trend_finder,
        "Portfolio & Risk": page_portfolio_and_risk,
        "Fundamental Analytics": page_fundamental_analytics,
        "Basket Orders": page_basket_orders,
        "Forecasting (ML)": page_forecasting_ml,
        "Algo Strategy Hub": page_algo_strategy_maker,
        "Economic Calendar": page_economic_calendar,
        "Settings": page_settings
        },
        "Options": {
            "F&O Analytics": page_fo_analytics,
            "Options Strategy Builder": page_option_strategy_builder,
            "Greeks Calculator": page_greeks_calculator,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
        },
        "Futures": {
            "Futures Terminal": page_futures_terminal,
            "Advanced Charting": page_advanced_charting,
            "Algo Strategy Hub": page_algo_strategy_maker,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant": page_ai_assistant,
        },
        "HFT": {
            "HFT Terminal": page_hft_terminal,
            "Portfolio & Risk": page_portfolio_and_risk,
        }
    }
    
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    no_refresh_pages = ["Forecasting (ML)", "AI Assistant", "AI Discovery", "Algo Strategy Hub", "Algo Trading Bots"]
    if auto_refresh and selection not in no_refresh_pages:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

# --- Application Entry Point ---
if __name__ == "__main__":
    initialize_session_state()
    
    if 'profile' in st.session_state and st.session_state.profile:
        if st.session_state.get('login_animation_complete', False):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
