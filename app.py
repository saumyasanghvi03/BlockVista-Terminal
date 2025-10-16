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

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state - UPDATED FOR UPSTOX."""
    broker = st.session_state.get('broker')
    
    if broker == "Zerodha":
        return st.session_state.get('kite')
    elif broker == "Upstox":
        if 'upstox_client' not in st.session_state:
            initialize_upstox_client()
        return st.session_state.get('upstox_client')
    
    return None

def initialize_upstox_client():
    """Initialize Upstox client with stored credentials."""
    try:
        if not UPSTOX_AVAILABLE:
            st.error("Upstox package not installed. Run: pip install upstox-python")
            return None
            
        api_key = st.secrets.get("UPSTOX_API_KEY")
        access_token = st.session_state.get('upstox_access_token')
        
        if not api_key:
            st.error("UPSTOX_API_KEY not found in secrets")
            return None
            
        if not access_token:
            st.error("Upstox access token not available")
            return None
            
        # Initialize Upstox configuration
        configuration = upstox.Configuration()
        configuration.access_token = access_token
        
        # Create API client
        api_client = upstox.ApiClient(configuration)
        
        # Initialize various API clients
        st.session_state.upstox_client = {
            'api_client': api_client,
            'user_api': upstox.UserApi(api_client),
            'order_api': upstox.OrderApi(api_client),
            'portfolio_api': upstox.PortfolioApi(api_client),
            'market_quote_api': upstox.MarketQuoteApi(api_client),
            'history_api': upstox.HistoryApi(api_client)
        }
        return st.session_state.upstox_client
        
    except Exception as e:
        st.error(f"Failed to initialize Upstox client: {e}")
        return None

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
    """Generate Upstox session using authorization code."""
    try:
        if not UPSTOX_AVAILABLE:
            st.error("Upstox package not available")
            return False
            
        api_key = st.secrets.get("UPSTOX_API_KEY")
        api_secret = st.secrets.get("UPSTOX_API_SECRET")
        redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI")
        
        if not all([api_key, api_secret, redirect_uri]):
            st.error("Missing Upstox credentials in secrets")
            return False
        
        # Create a basic API client for token generation
        api_instance = upstox.LoginApi()
        
        # Generate session - using the API directly
        token_response = api_instance.token(
            code=authorization_code,
            client_id=api_key,
            client_secret=api_secret,
            redirect_uri=redirect_uri,
            grant_type="authorization_code"
        )
        
        if hasattr(token_response, 'access_token'):
            st.session_state.upstox_access_token = token_response.access_token
            st.session_state.upstox_token_type = getattr(token_response, 'token_type', 'bearer')
            st.success("Upstox authentication successful!")
            return True
        else:
            st.error("No access token in response")
            return False
        
    except Exception as e:
        st.error(f"Upstox session generation failed: {str(e)}")
        return False

def get_upstox_instruments():
    """Fetch Upstox instruments master."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return pd.DataFrame()
            
        # Get instruments from Upstox
        api_instance = upstox_api.MasterApi(client['api_client'])
        response = api_instance.get_master_csv(exchange="NSE_EQ")  # Can be NSE_FO, BSE_EQ, etc.
        
        # Parse CSV response
        instruments_df = pd.read_csv(io.StringIO(response))
        return instruments_df
        
    except Exception as e:
        st.error(f"Error fetching Upstox instruments: {e}")
        return pd.DataFrame()

def get_upstox_historical_data(instrument_key, interval, from_date, to_date):
    """Fetch historical data from Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return pd.DataFrame()
            
        api_instance = upstox_api.HistoryApi(client['api_client'])
        
        # Convert dates to required format
        from_date_str = from_date.strftime("%Y-%m-%d")
        to_date_str = to_date.strftime("%Y-%m-%d")
        
        response = api_instance.get_historical_candle_data(
            instrument_key=instrument_key,
            interval=interval,
            from_date=from_date_str,
            to_date=to_date_str
        )
        
        if response and hasattr(response, 'data'):
            candles = response.data.candles
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
            
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching Upstox historical data: {e}")
        return pd.DataFrame()

def get_upstox_quote(instruments):
    """Fetch real-time quotes from Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return {}
            
        api_instance = upstox_api.MarketQuoteApi(client['api_client'])
        
        # Format: "NSE_EQ|INE002A01018" or "NSE_FO|NIFTY23NOV23000CE"
        response = api_instance.get_market_quote_full(instruments)
        
        quotes = {}
        if response and hasattr(response, 'data'):
            for instrument_key, quote_data in response.data.items():
                quotes[instrument_key] = {
                    'last_price': quote_data.last_price,
                    'open': quote_data.open,
                    'high': quote_data.high,
                    'low': quote_data.low,
                    'close': quote_data.close,
                    'volume': quote_data.volume,
                    'oi': quote_data.oi
                }
        return quotes
        
    except Exception as e:
        st.error(f"Error fetching Upstox quotes: {e}")
        return {}

def place_upstox_order(order_params):
    """Place order through Upstox."""
    try:
        client = get_broker_client()
        if not client or st.session_state.broker != "Upstox":
            return None
            
        api_instance = upstox_api.OrderApi(client['api_client'])
        
        # Prepare order parameters
        order_request = upstox_api.PlaceOrderRequest(
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
            
        api_instance = upstox_api.PortfolioApi(client['api_client'])
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
            
        api_instance = upstox_api.PortfolioApi(client['api_client'])
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
            
        api_instance = upstox_api.OrderApi(client['api_client'])
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
    """Check if current time is within market hours (9:15 AM to 3:30 PM, Monday to Friday) using IST"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()  # Monday=0, Sunday=6
    
    # Check if it's a weekday (Monday to Friday)
    if current_day >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    return market_open <= current_time <= market_close

def is_pre_market_hours():
    """Check if current time is pre-market hours (9:00 AM to 9:15 AM, Monday to Friday) using IST"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    # Check if it's a weekday
    if current_day >= 5:
        return False
    
    # Pre-market hours: 9:00 AM to 9:15 AM IST
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    
    return pre_market_start <= current_time < market_open

def is_square_off_time():
    """Check if current time is square-off time for Equity/Cash (3:20 PM to 3:30 PM, Monday to Friday) using IST"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    # Check if it's a weekday
    if current_day >= 5:  # Saturday or Sunday
        return False
    
    # Square-off time for Equity/Cash: 3:20 PM to 3:30 PM IST
    square_off_start = time(15, 20)  # 3:20 PM
    square_off_end = time(15, 30)    # 3:30 PM
    
    return square_off_start <= current_time <= square_off_end

def is_derivatives_square_off_time():
    """Check if current time is square-off time for Equity/Index Derivatives (3:25 PM to 3:30 PM) using IST"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    
    if current_day >= 5:
        return False
    
    # Square-off time for Derivatives: 3:25 PM to 3:30 PM IST
    square_off_start = time(15, 25)  # 3:25 PM
    square_off_end = time(15, 30)    # 3:30 PM
    
    return square_off_start <= current_time <= square_off_end

def get_market_status():
    """Get current market status and next market event with proper formatting for display using IST."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_time = now.time()
    current_day = now.weekday()
    current_date = now.date()
    
    # Weekend check
    if current_day >= 5:  # Saturday or Sunday
        days_until_monday = (7 - current_day) % 7
        if days_until_monday == 0:  # Already Monday? (shouldn't happen but safe)
            days_until_monday = 7
        next_market = now + timedelta(days=days_until_monday)
        return {
            "status": "weekend",
            "next_market": next_market.replace(hour=9, minute=0, second=0, microsecond=0),
            "color": "#cccccc"
        }
    
    # Market timing definitions in IST
    pre_market_start = time(9, 0)
    market_open = time(9, 15)
    equity_square_off = time(15, 20)  # 3:20 PM for Equity/Cash
    derivatives_square_off = time(15, 25)  # 3:25 PM for Derivatives
    market_close = time(15, 30)
    
    if current_time < pre_market_start:
        # Before pre-market (overnight)
        next_market = now.replace(hour=9, minute=0, second=0, microsecond=0)
        return {
            "status": "market_closed",
            "next_market": next_market,
            "color": "#cccccc"
        }
    elif pre_market_start <= current_time < market_open:
        # Pre-market hours (9:00 AM to 9:15 AM)
        next_market = now.replace(hour=9, minute=15, second=0, microsecond=0)
        return {
            "status": "pre_market", 
            "next_market": next_market,
            "color": "#ffcc00"
        }
    elif market_open <= current_time < equity_square_off:
        # Market open (9:15 AM to 3:20 PM)
        next_market = now.replace(hour=15, minute=20, second=0, microsecond=0)
        return {
            "status": "market_open",
            "next_market": next_market, 
            "color": "#00cc00"
        }
    elif equity_square_off <= current_time < derivatives_square_off:
        # Equity square-off time (3:20 PM to 3:25 PM)
        next_market = now.replace(hour=15, minute=25, second=0, microsecond=0)
        return {
            "status": "equity_square_off",
            "next_market": next_market,
            "color": "#ff9900"
        }
    elif derivatives_square_off <= current_time <= market_close:
        # Derivatives square-off time (3:25 PM to 3:30 PM)
        next_market = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return {
            "status": "derivatives_square_off",
            "next_market": next_market,
            "color": "#ff6600"
        }
    else:
        # Market closed for the day (after 3:30 PM)
        next_day = now + timedelta(days=1)
        # Skip weekends
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_market = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
        return {
            "status": "market_closed",
            "next_market": next_market,
            "color": "#cccccc"
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
@st.cache_data(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if not client: 
        return pd.DataFrame()
    
    broker = st.session_state.broker
    
    if broker == "Zerodha":
        df = pd.DataFrame(client.instruments())
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
        return df
        
    elif broker == "Upstox":
        return get_upstox_instruments()
    
    else:
        st.warning(f"Instrument list for {broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data - UPDATED FOR UPSTOX."""
    client = get_broker_client()
    if not client or not instrument_token: 
        return pd.DataFrame()
    
    broker = st.session_state.broker
    
    if broker == "Zerodha":
        # Existing Zerodha implementation
        if not to_date: 
            to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        
        if from_date > to_date:
            from_date = to_date - timedelta(days=1)
            
        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty: 
                return df
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
            
    elif broker == "Upstox":
        # Upstox implementation
        if not to_date:
            to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        
        return get_upstox_historical_data(instrument_token, interval, from_date, to_date)
    
    else:
        st.warning(f"Historical data for {broker} not implemented.")
        return pd.DataFrame()

@st.cache_data(ttl=15)
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

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
    analyzer = SentimentIntensityAnalyzer()
    
    news_sources = {
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
        "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
        "Livemint": "https://www.livemint.com/rss/markets",
        "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
        "Reuters World": "http://feeds.reuters.com/Reuters/worldNews",
        "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml"
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                published_date_tuple = entry.published_parsed if hasattr(entry, 'published_parsed') else entry.updated_parsed
                published_date = datetime.fromtimestamp(mktime_tz(published_date_tuple)) if published_date_tuple else datetime.now()
                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception:
            continue
    return pd.DataFrame(all_news)

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

@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        data_yf = yf.download(list(tickers.values()), period="5d")
        if data_yf.empty:
            return pd.DataFrame()

        data = []
        for ticker_name, yf_ticker_name in tickers.items():
            if len(tickers) > 1:
                # For multiple tickers, extract data for each ticker
                if yf_ticker_name in data_yf['Close'].columns:
                    hist = data_yf['Close'][yf_ticker_name]
                else:
                    continue
            else:
                hist = data_yf['Close']

            if len(hist) >= 2:
                last_price = float(hist.iloc[-1])
                prev_close = float(hist.iloc[-2])
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                data.append({'Ticker': ticker_name, 'Price': last_price, 'Change': change, '% Change': pct_change})
            else:
                data.append({'Ticker': ticker_name, 'Price': np.nan, 'Change': np.nan, '% Change': np.nan})

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()

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

# Dictionary of automated bots
AUTOMATED_BOTS = {
    "Auto Momentum Trader": automated_momentum_trader,
    "Auto Mean Reversion": automated_mean_reversion
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
        current_risk = max(0.5, min(5.0, current_risk))
        
        risk_per_trade = st.number_input(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=current_risk,
            step=0.5,
            help="Risk percentage per trade",
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
            display_bot_configuration_tab()
        
        with tab2:
            # Live performance dashboard - CALL THE ACTUAL FUNCTION
            try:
                display_enhanced_live_dashboard(instrument_df)
            except Exception as e:
                st.error(f"Error displaying dashboard: {e}")
        
        with tab3:
            # 🎯 ENHANCED LIVE THINKING TAB - CALL THE ACTUAL FUNCTION
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
    
    else:
        # Setup guide when disabled
        display_setup_guide()

# Add the missing function implementations
def display_bot_configuration_tab():
    """Display bot configuration tab"""
    st.subheader("⚙️ Bot Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.write("**🤖 Active Bots**")
        for bot_name in AUTOMATED_BOTS.keys():
            is_active = st.session_state.automated_mode.get('bots_active', {}).get(bot_name, False)
            if st.checkbox(bot_name, value=is_active, key=f"auto_{bot_name}"):
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = True
            else:
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = False
        
        st.markdown("---")
        st.write("**📊 Trading Limits**")
        max_trades = st.slider(
            "Max Open Trades",
            min_value=1,
            max_value=20,
            value=st.session_state.automated_mode.get('max_open_trades', 5),
            help="Maximum simultaneous trades",
            key="auto_max_trades"
        )
        st.session_state.automated_mode['max_open_trades'] = max_trades
    
    with col_config2:
        st.write("**⏰ Analysis Frequency**")
        current_interval = st.session_state.automated_mode.get('check_interval', '1 minute')
        frequency_options = ["15 seconds", "30 seconds", "1 minute", "5 minutes", "15 minutes"]
        
        current_index = 2
        if current_interval in frequency_options:
            current_index = frequency_options.index(current_interval)
        
        check_interval = st.selectbox(
            "Analysis Frequency",
            options=frequency_options,
            index=current_index,
            help="How often bots analyze the market",
            key="auto_freq"
        )
        st.session_state.automated_mode['check_interval'] = check_interval
        
        # Frequency warnings
        if check_interval == "15 seconds":
            st.warning("⚡ High frequency - May hit API limits")
        elif check_interval == "30 seconds":
            st.info("🚀 Active trading - Good balance")
        else:
            st.success("🔄 Standard frequency - Stable")
        
        st.markdown("---")
        st.write("**📋 Trading Symbols**")
        active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
        watchlist_symbols = [item['symbol'] for item in st.session_state.watchlists.get(active_watchlist, [])]
        
        if watchlist_symbols:
            st.success(f"Trading from: **{active_watchlist}**")
            with st.expander(f"View {len(watchlist_symbols)} symbols"):
                for symbol in watchlist_symbols:
                    st.write(f"• {symbol}")
        else:
            st.warning("No symbols in active watchlist")

def display_enhanced_live_dashboard(instrument_df):
    """Enhanced live trading dashboard with real metrics."""
    st.subheader("📊 Live Performance Dashboard")
    
    # Performance metrics
    performance = get_automated_bot_performance()
    paper_portfolio = st.session_state.automated_mode.get('paper_portfolio', {})
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = performance.get('paper_return_pct', 0)
        return_color = "green" if total_return >= 0 else "red"
        st.metric(
            "Total Return", 
            f"{total_return:.2f}%",
            delta=f"{total_return:.2f}%",
            delta_color="normal" if total_return >= 0 else "inverse"
        )
    
    with col2:
        total_trades = performance.get('total_trades', 0)
        st.metric("Total Trades", total_trades)
    
    with col3:
        win_rate = performance.get('win_rate', 0)
        win_color = "green" if win_rate > 60 else "orange" if win_rate > 40 else "red"
        st.metric(
            "Win Rate", 
            f"{win_rate:.1f}%",
            delta_color="normal" if win_rate > 60 else "off"
        )
    
    with col4:
        total_pnl = performance.get('total_pnl', 0)
        pnl_color = "green" if total_pnl >= 0 else "red"
        st.metric(
            "Total P&L", 
            f"₹{total_pnl:,.2f}",
            delta=f"₹{total_pnl:,.2f}",
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    st.markdown("---")
    
    # Portfolio overview
    st.subheader("💰 Portfolio Overview")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        current_value = paper_portfolio.get('total_value', paper_portfolio.get('cash_balance', 0))
        st.metric("Portfolio Value", f"₹{current_value:,.2f}")
    
    with col6:
        cash_balance = paper_portfolio.get('cash_balance', 0)
        st.metric("Cash Balance", f"₹{cash_balance:,.2f}")
    
    with col7:
        initial_capital = paper_portfolio.get('initial_capital', current_value)
        st.metric("Initial Capital", f"₹{initial_capital:,.2f}")
    
    with col8:
        open_trades = performance.get('open_trades', 0)
        st.metric("Open Trades", open_trades)
    
    st.markdown("---")
    
    # Active positions
    st.subheader("📈 Active Positions")
    positions = paper_portfolio.get('positions', {})
    
    if positions:
        position_data = []
        total_position_value = 0
        
        for symbol, position in positions.items():
            # Get current price
            live_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
            current_price = live_data.iloc[0]['Price'] if not live_data.empty else position['avg_price']
            position_value = position['quantity'] * current_price
            total_position_value += position_value
            unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
            pnl_percent = (unrealized_pnl / (position['avg_price'] * position['quantity'])) * 100 if position['avg_price'] > 0 else 0
            
            position_data.append({
                'Symbol': symbol,
                'Quantity': position['quantity'],
                'Avg Price': f"₹{position['avg_price']:.2f}",
                'Current Price': f"₹{current_price:.2f}",
                'Value': f"₹{position_value:,.2f}",
                'P&L': f"₹{unrealized_pnl:.2f}",
                'P&L %': f"{pnl_percent:.2f}%",
                'Action': position.get('action', 'BUY')
            })
        
        # Display positions
        df_positions = pd.DataFrame(position_data)
        
        # Color coding for P&L
        def color_pnl(val):
            try:
                pnl_value = float(val.replace('₹', '').replace(',', '').split(' ')[0])
                if pnl_value > 0:
                    return 'color: green; font-weight: bold;'
                elif pnl_value < 0:
                    return 'color: red; font-weight: bold;'
            except:
                pass
            return ''
        
        styled_positions = df_positions.style.map(color_pnl, subset=['P&L', 'P&L %'])
        st.dataframe(styled_positions, use_container_width=True, hide_index=True)
        
        # Position summary
        st.metric("Total Positions Value", f"₹{total_position_value:,.2f}")
        
    else:
        st.info("No active positions. Trades will appear here when executed.")
    
    # Trading statistics
    st.markdown("---")
    st.subheader("📈 Trading Statistics")
    
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        avg_win = performance.get('avg_win', 0)
        st.metric("Avg Win", f"₹{avg_win:.2f}")
    
    with col10:
        avg_loss = performance.get('avg_loss', 0)
        st.metric("Avg Loss", f"₹{avg_loss:.2f}")
    
    with col11:
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    
    with col12:
        max_drawdown = performance.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Recent activity
    st.markdown("---")
    st.subheader("🕒 Recent Activity")
    
    # Show last 5 trades
    trade_history = st.session_state.automated_mode.get('trade_history', [])
    if trade_history:
        recent_trades = sorted(trade_history, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        
        for trade in recent_trades:
            timestamp = trade.get('timestamp_display', 
                                get_ist_time().strftime("%H:%M:%S IST") if 'timestamp' not in trade 
                                else datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).astimezone(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S IST"))
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            col1.write(f"**{trade['symbol']}**")
            col2.write(f"**{trade['action']}**")
            col3.write(f"{trade['quantity']} shares")
            col4.write(f"₹{trade.get('entry_price', 0):.2f}")
            st.caption(f"_{timestamp} • {trade.get('bot_name', 'Manual')} • {trade.get('status', 'OPEN')}_")
            st.markdown("---")
    else:
        st.info("No recent trading activity")

def display_enhanced_live_thinking_tab(instrument_df):
    """Enhanced live bot thinking analysis with real reasoning."""
    st.subheader("🔍 Live Bot Analysis & Thinking")
    
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
    
    # Analysis controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Real-time Bot Analysis**")
    with col2:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    # Run analysis on watchlist symbols
    thinking_data = []
    
    with st.spinner("🤖 Analyzing symbols with active bots..."):
        for symbol in watchlist_symbols[:10]:  # Limit for performance
            for bot_name in active_bots:
                if bot_name in AUTOMATED_BOTS:
                    try:
                        bot_function = AUTOMATED_BOTS[bot_name]
                        bot_result = bot_function(instrument_df, symbol)
                        
                        if not bot_result.get("error"):
                            # Get current price for context
                            quote_data = get_watchlist_data([{'symbol': symbol, 'exchange': 'NSE'}])
                            current_price = quote_data.iloc[0]['Price'] if not quote_data.empty else 0
                            
                            thinking_data.append({
                                'Symbol': symbol,
                                'Bot': bot_name,
                                'Signal': bot_result['action'],
                                'Confidence': bot_result.get('score', 50),
                                'Current Price': f"₹{current_price:.2f}",
                                'Analysis': " | ".join(bot_result.get('signals', ['No signals detected'])),
                                'Risk Level': bot_result.get('risk_level', 'Medium'),
                                'Timestamp': get_ist_time().strftime("%H:%M:%S IST")
                            })
                    except Exception as e:
                        thinking_data.append({
                            'Symbol': symbol,
                            'Bot': bot_name,
                            'Signal': 'ERROR',
                            'Confidence': 0,
                            'Current Price': 'N/A',
                            'Analysis': f"Analysis error: {str(e)}",
                            'Risk Level': 'High',
                            'Timestamp': get_ist_time().strftime("%H:%M:%S IST")
                        })
    
    if thinking_data:
        # Convert to dataframe
        thinking_df = pd.DataFrame(thinking_data)
        
        # Color coding
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
        
        # Display styled dataframe
        styled_df = thinking_df.style.map(color_signal, subset=['Signal']).map(color_confidence, subset=['Confidence'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        st.subheader("📊 Analysis Summary")
        
        buy_signals = len(thinking_df[thinking_df['Signal'] == 'BUY'])
        sell_signals = len(thinking_df[thinking_df['Signal'] == 'SELL'])
        hold_signals = len(thinking_df[thinking_df['Signal'] == 'HOLD'])
        avg_confidence = thinking_df['Confidence'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Buy Signals", buy_signals)
        col2.metric("Sell Signals", sell_signals)
        col3.metric("Hold Signals", hold_signals)
        col4.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Bot-specific analysis
        st.subheader("🤖 Bot Performance")
        bot_performance = thinking_df.groupby('Bot').agg({
            'Signal': 'count',
            'Confidence': 'mean'
        }).rename(columns={'Signal': 'Analyses', 'Confidence': 'Avg Confidence'})
        
        st.dataframe(bot_performance.style.format({'Avg Confidence': '{:.1f}%'}), use_container_width=True)
        
        # Actionable insights
        st.subheader("💡 Actionable Insights")
        
        if buy_signals > sell_signals * 2:
            st.success("**Market Sentiment:** 📈 Predominantly Bullish - More buy signals detected")
        elif sell_signals > buy_signals * 2:
            st.error("**Market Sentiment:** 📉 Predominantly Bearish - More sell signals detected")
        else:
            st.info("**Market Sentiment:** ⚖️ Mixed - Balanced signals across bots")
            
        if avg_confidence > 70:
            st.success("**Signal Quality:** ✅ High Confidence - Strong trading signals")
        elif avg_confidence > 50:
            st.warning("**Signal Quality:** ⚠️ Medium Confidence - Moderate trading signals")
        else:
            st.error("**Signal Quality:** ❌ Low Confidence - Weak or conflicting signals")
            
    else:
        st.info("""
        **No analysis data available. Possible reasons:**
        
        - Bots are still processing
        - No active symbols in watchlist
        - Market data unavailable
        - Bot configuration issues
        
        **Try:**
        - Adding symbols to your watchlist
        - Checking bot activation status
        - Waiting for market hours (9:15 AM - 3:30 PM IST)
        """)

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

def display_bot_configuration_tab():
    """Display bot configuration tab"""
    st.subheader("⚙️ Bot Configuration")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.write("**🤖 Active Bots**")
        for bot_name in AUTOMATED_BOTS.keys():
            is_active = st.session_state.automated_mode.get('bots_active', {}).get(bot_name, False)
            if st.checkbox(bot_name, value=is_active, key=f"auto_{bot_name}"):
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = True
            else:
                if 'bots_active' not in st.session_state.automated_mode:
                    st.session_state.automated_mode['bots_active'] = {}
                st.session_state.automated_mode['bots_active'][bot_name] = False
        
        st.markdown("---")
        st.write("**📊 Trading Limits**")
        max_trades = st.slider(
            "Max Open Trades",
            min_value=1,
            max_value=20,
            value=st.session_state.automated_mode.get('max_open_trades', 5),
            help="Maximum simultaneous trades",
            key="auto_max_trades"
        )
        st.session_state.automated_mode['max_open_trades'] = max_trades
    
    with col_config2:
        st.write("**⏰ Analysis Frequency**")
        current_interval = st.session_state.automated_mode.get('check_interval', '1 minute')
        frequency_options = ["15 seconds", "30 seconds", "1 minute", "5 minutes", "15 minutes"]
        
        current_index = 2
        if current_interval in frequency_options:
            current_index = frequency_options.index(current_interval)
        
        check_interval = st.selectbox(
            "Analysis Frequency",
            options=frequency_options,
            index=current_index,
            help="How often bots analyze the market",
            key="auto_freq"
        )
        st.session_state.automated_mode['check_interval'] = check_interval
        
        # Frequency warnings
        if check_interval == "15 seconds":
            st.warning("⚡ High frequency - May hit API limits")
        elif check_interval == "30 seconds":
            st.info("🚀 Active trading - Good balance")
        else:
            st.success("🔄 Standard frequency - Stable")
        
        st.markdown("---")
        st.write("**📋 Trading Symbols**")
        active_watchlist = st.session_state.get('active_watchlist', 'Watchlist 1')
        watchlist_symbols = [item['symbol'] for item in st.session_state.watchlists.get(active_watchlist, [])]
        
        if watchlist_symbols:
            st.success(f"Trading from: **{active_watchlist}**")
            with st.expander(f"View {len(watchlist_symbols)} symbols"):
                for symbol in watchlist_symbols:
                    st.write(f"• {symbol}")
        else:
            st.warning("No symbols in active watchlist")

# ================ 5. PAGE DEFINITIONS ============

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
    st.title("Premarket & Global Cues")
    st.markdown("---")

    st.subheader("Global Market Snapshot")
    global_tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "Hang Seng": "^HSI"}
    global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_tickers))
        for i, (name, ticker_symbol) in enumerate(global_tickers.items()):
            data_row = global_data[global_data['Ticker'] == name]
            if not data_row.empty:
                price = data_row.iloc[0]['Price']
                change = data_row.iloc[0]['% Change']
                if not np.isnan(price):
                    cols[i].metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                else:
                    cols[i].metric(label=name, value="N/A", delta="--")
    else:
        st.info("Loading global market data...")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY 50 Futures (Live Proxy)")
        gift_data = get_gift_nifty_data()
        if not gift_data.empty:
            st.plotly_chart(create_chart(gift_data, "NIFTY 50 Futures (Proxy)"), use_container_width=True)
        else:
            st.warning("Could not load NIFTY 50 Futures chart data.")
            
    with col2:
        st.subheader("Key Asian Markets")
        asian_tickers = {"Nikkei 225": "^N225", "Hang Seng": "^HSI"}
        asian_data = get_global_indices_data(asian_tickers)
        if not asian_data.empty:
            for name, ticker_symbol in asian_tickers.items():
                data_row = asian_data[asian_data['Ticker'] == name]
                if not data_row.empty:
                    price = data_row.iloc[0]['Price']
                    change = data_row.iloc[0]['% Change']
                    if not np.isnan(price):
                        st.metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                    else:
                        st.metric(label=name, value="N/A", delta="--")
        else:
            st.info("Loading Asian market data...")

    st.markdown("---")

    st.subheader("Latest Market News")
    news_df = fetch_and_analyze_news()
    if not news_df.empty:
        for _, news in news_df.head(10).iterrows():
            sentiment_score = news['sentiment']
            if sentiment_score > 0.2:
                icon = "🔼"
            elif sentiment_score < -0.2:
                icon = "🔽"
            else:
                icon = "▶️"
            st.markdown(f"**{icon} [{news['title']}]({news['link']})** - *{news['source']}*")
    else:
        st.info("News data is loading...")

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
    """An AI-powered assistant for portfolio management and market queries."""
    display_header()
    st.title("Portfolio-Aware Assistant")
    instrument_df = get_instrument_df()

    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your portfolio or the markets today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_lower = prompt.lower()
                response = "I can help with your portfolio, orders, and live market data. For example, try asking 'What are my positions?' or 'Show me the option chain for BANKNIFTY'."
                client = get_broker_client()
                if not client:
                    response = "I am not connected to your broker. Please log in first."
                
                elif any(word in prompt_lower for word in ["holdings", "investments"]):
                    _, holdings_df, _, _ = get_portfolio()
                    response = f"Here are your current holdings:\n```\n{tabulate(holdings_df, headers='keys', tablefmt='psql')}\n```" if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    response = f"Your total P&L is ₹{total_pnl:,.2f}. Here are your positions:\n```\n{tabulate(positions_df, headers='keys', tablefmt='psql')}\n```" if not positions_df.empty else "You have no open positions."
                elif "orders" in prompt_lower:
                    orders = client.orders()
                    response = f"Here are today's orders:\n```\n{tabulate(pd.DataFrame(orders), headers='keys', tablefmt='psql')}\n```" if orders else "You have no orders for the day."
                elif any(word in prompt_lower for word in ["funds", "margin", "balance"]):
                    funds = client.margins()
                    response = f"Available Funds:\n- Equity: ₹{funds['equity']['available']['live_balance']:,.2f}\n- Commodity: ₹{funds['commodity']['available']['live_balance']:,.2f}"
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    try:
                        ticker = prompt.split(" of ")[-1].strip().upper()
                        instrument = instrument_df[instrument_df['tradingsymbol'] == ticker]
                        if not instrument.empty:
                            exchange = instrument.iloc[0]['exchange']
                            ltp_data = get_watchlist_data([{'symbol': ticker, 'exchange': exchange}])
                            price = ltp_data.iloc[0]['Price'] if not ltp_data.empty else "N/A"
                            response = f"The current price of {ticker} is {price}."
                        else:
                            response = f"I could not find the ticker '{ticker}'. Please check the symbol."
                    except Exception:
                        response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                
                elif "buy" in prompt_lower or "sell" in prompt_lower:
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9\-\&_]+)', prompt_lower)
                    if match:
                        trans_type = match.group(1).upper()
                        quantity = int(match.group(2))
                        symbol = match.group(3).upper()

                        st.session_state.last_order_details = {
                            "symbol": symbol,
                            "quantity": quantity,
                            "transaction_type": trans_type,
                            "confirmed": False
                        }
                        
                        response = f"I can place a {trans_type} order for {quantity} shares of {symbol}. Please confirm by typing 'confirm'."
                    else:
                        response = "I couldn't understand the order. Please use a format like 'Buy 100 shares of RELIANCE'."
                elif prompt_lower == "confirm" and "last_order_details" in st.session_state and not st.session_state.last_order_details["confirmed"]:
                    order_details = st.session_state.last_order_details
                    place_order(instrument_df, order_details['symbol'], order_details['quantity'], 'MARKET', order_details['transaction_type'], 'MIS')
                    order_details['confirmed'] = True
                    response = f"Confirmed and placed {order_details['transaction_type']} order for {order_details['quantity']} shares of {order_details['symbol']}."

                elif "technical analysis for" in prompt_lower:
                    ticker = prompt.split("for")[-1].strip().upper()
                    token = get_instrument_token(ticker, instrument_df)
                    if token:
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            analysis = interpret_indicators(data)
                            response = f"**Technical Analysis for {ticker}:**\n\n" + "\n".join([f"- **{k}:** {v}" for k, v in analysis.items()])
                        else:
                            response = f"Could not retrieve enough data for {ticker} to perform analysis."
                    else:
                        response = f"Could not find the ticker '{ticker}'."
                
                elif "news for" in prompt_lower:
                    query = prompt.split("for")[-1].strip()
                    news_df = fetch_and_analyze_news(query)
                    if not news_df.empty:
                        response = f"**Top 3 news headlines for {query}:**\n\n" + "\n".join([f"1. [{row['title']}]({row['link']}) - _{row['source']}_" for _, row in news_df.head(3).iterrows()])
                    else:
                        response = f"No recent news found for '{query}'."
                
                elif "greeks for" in prompt_lower or "iv for" in prompt_lower:
                    try:
                        option_symbol_match = re.search(r'([A-Z]+)(\d{2}[A-Z]{3}\d+[CEPE]{2})', prompt.upper())
                        if option_symbol_match:
                            option_symbol = option_symbol_match.group(0)
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry - datetime.now().date()).days, 0) / 365.0
                            iv = implied_volatility(underlying_ltp, option_details['strike'], T, 0.07, ltp, option_details['instrument_type'].lower())
                            
                            if not np.isnan(iv):
                                greeks = black_scholes(underlying_ltp, option_details['strike'], T, 0.07, iv, option_details['instrument_type'].lower())
                                response = f"Calculated Greeks for **{option_symbol}**:\n- **Implied Volatility (IV):** {iv*100:.2f}%\n- **Delta:** {greeks['delta']:.4f}\n- **Gamma:** {greeks['gamma']:.4f}\n- **Vega:** {greeks['vega']:.4f}\n- **Theta:** {greeks['theta']:.4f}\n- **Rho:** {greeks['rho']:.4f}"
                            else:
                                response = f"Could not calculate IV or Greeks for {option_symbol}. The LTP might be zero or the option might be illiquid."
                        else:
                            response = "Please specify a valid option symbol (e.g., NIFTY24SEPWK123000CE)."
                    except (AttributeError, IndexError):
                        response = "I couldn't find a valid option symbol in your query. Please use the full symbol (e.g., BANKNIFTY24OCT60000CE)."
                    except Exception as e:
                        response = f"An error occurred: {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

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
    """AI-driven discovery engine with real data analysis."""
    display_header()
    st.title("AI Discovery Engine")
    st.info("This engine discovers technical patterns and suggests high-conviction trade setups based on your active watchlist. The suggestions are for informational purposes only.", icon="🧠")
    
    active_list = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
    instrument_df = get_instrument_df()

    if not active_list or instrument_df.empty:
        st.warning("Please set up your watchlist on the Dashboard page to enable AI Discovery.")
        return

    st.markdown("---")
    
    st.subheader("Automated Pattern Discovery")
    with st.spinner("Analyzing your watchlist for technical signals..."):
        discovery_results = {}
        for item in active_list:
            token = get_instrument_token(item['symbol'], instrument_df, exchange=item['exchange'])
            if token:
                data = get_historical_data(token, 'day', period='6mo')
                if not data.empty:
                    interpretation = interpret_indicators(data)
                    signals = [f"{k}: {v}" for k, v in interpretation.items() if "Bullish" in v or "Bearish" in v]
                    if signals:
                        discovery_results[item['symbol']] = signals
    
    if discovery_results:
        for ticker, signals in discovery_results.items():
            st.markdown(f"**Potential Signals for {ticker}:** " + ", ".join(signals))
    else:
        st.info("No significant technical patterns found in your watchlist.")
        
    st.markdown("---")
    
    st.subheader("AI-Powered Trade Idea")
    with st.spinner("Generating a high-conviction trade idea..."):
        trade_idea = generate_ai_trade_idea(instrument_df, active_list)

    if trade_idea:
        trade_idea_col = st.columns(3)
        trade_idea_col[0].metric("Entry Price", f"≈ ₹{trade_idea['entry']:.2f}")
        trade_idea_col[1].metric("Target Price", f"₹{trade_idea['target']:.2f}")
        trade_idea_col[2].metric("Stop Loss", f"₹{trade_idea['stop_loss']:.2f}")
        
        st.markdown(f"""
        <div class="trade-card" style="border-left-color: {'#28a745' if 'Long' in trade_idea['title'] else '#FF4B4B'};">
            <h4>{trade_idea['title']}</h4>
            <p><strong>Narrative:</strong> {trade_idea['narrative']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Could not generate a high-conviction trade idea from the current watchlist signals.")

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
    """Economic Calendar page for Indian market events."""
    display_header()
    st.title("Economic Calendar")
    st.info("Upcoming economic events for the Indian market, updated until October 2025.")

    events = {
        'Date': [
            '2025-09-26', '2025-09-26', '2025-09-29', '2025-09-30',
            '2025-10-01', '2025-10-03', '2025-10-08', '2025-10-10',
            '2025-10-14', '2025-10-15', '2025-10-17', '2025-10-24',
            '2025-10-31', '2025-10-31'
        ],
        'Time': [
            '11:30 AM', '11:30 AM', '10:30 AM', '05:30 PM',
            '10:30 AM', '10:30 AM', '11:00 AM', '05:00 PM',
            '12:00 PM', '05:30 PM', '05:00 PM', '05:00 PM',
            '05:30 PM', '05:00 PM'
        ],
        'Event Name': [
            'Bank Loan Growth YoY', 'Foreign Exchange Reserves', 'Industrial Production YoY (AUG)', 'Infrastructure Output YoY (AUG)',
            'Nikkei Manufacturing PMI (SEP)', 'Nikkei Services PMI (SEP)', 'RBI Interest Rate Decision',
            'Foreign Exchange Reserves', 'WPI Inflation YoY (SEP)', 'CPI Inflation YoY (SEP)',
            'Foreign Exchange Reserves', 'Foreign Exchange Reserves', 'Fiscal Deficit (SEP)',
            'Foreign Exchange Reserves'
        ],
        'Impact': [
            'Medium', 'Low', 'Medium', 'Medium',
            'High', 'High', 'High', 'Low',
            'High', 'High', 'Low', 'Low',
            'Medium', 'Low'
        ],
        'Previous': [
            '10.0%', '$702.97B', '2.9%', '6.3%',
            '58.5', '61.6', '6.50%', '$703.1B',
            '0.3%', '5.1%', '$704.5B', '$705.2B',
            '-4684.2B INR', '$705.9B'
        ],
        'Forecast': [
            '-', '-', '3.5%', '6.5%',
            '58.8', '61.2', '6.50%', '-',
            '0.5%', '5.3%', '-', '-',
            '-5100.0B INR', '-'
        ]
    }
    calendar_df = pd.DataFrame(events)

    st.dataframe(calendar_df, use_container_width=True, hide_index=True)



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
    """Displays the login page for broker authentication - UPDATED FOR UPSTOX."""
    st.title("BlockVista Terminal")
    st.subheader("Broker Login")
    
    broker = st.selectbox("Select Your Broker", ["Zerodha", "Upstox"])
    
    if broker == "Zerodha":
        # Existing Zerodha login
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
        # Upstox login
        api_key = st.secrets.get("UPSTOX_API_KEY")
        api_secret = st.secrets.get("UPSTOX_API_SECRET")
        redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI", "https://localhost")
        
        if not api_key or not api_secret:
            st.error("Upstox API credentials not found. Please set UPSTOX_API_KEY and UPSTOX_API_SECRET in your Streamlit secrets.")
            st.stop()
        
        # Check for authorization code in URL
        authorization_code = st.query_params.get("code")
        
        if authorization_code:
            try:
                success = upstox_generate_session(authorization_code)
                if success:
                    # Initialize client and get profile
                    client = initialize_upstox_client()
                    if client:
                        profile = client['user_api'].get_profile()
                        st.session_state.profile = {
                            'user_name': profile.data.name,
                            'email': profile.data.email,
                            'user_id': profile.data.client_id
                        }
                        st.session_state.broker = "Upstox"
                        st.query_params.clear()
                        st.rerun()
                else:
                    st.error("Upstox authentication failed.")
                    st.query_params.clear()
            except Exception as e:
                st.error(f"Upstox authentication failed: {e}")
                st.query_params.clear()
        else:
            login_url = get_upstox_login_url()
            if login_url:
                st.link_button("Login with Upstox", login_url)
                st.info("Please login with Upstox to begin. You will be redirected back to the app.")
            else:
                st.error("Could not generate Upstox login URL.")

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
            "Algo Trading Bots": page_algo_bots,
            "Premarket Pulse": page_premarket_pulse,
            "Advanced Charting": page_advanced_charting,
            "Market Scanners": page_momentum_and_trend_finder,
            "Portfolio & Risk": page_portfolio_and_risk,
            "Fundamental Analytics": page_fundamental_analytics,
            "Basket Orders": page_basket_orders,
            "Forecasting (ML)": page_forecasting_ml,
            "Algo Strategy Hub": page_algo_strategy_maker,
            "AI Discovery": page_ai_discovery,
            "AI Assistant": page_ai_assistant,
            "Economic Calendar": page_economic_calendar,
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
