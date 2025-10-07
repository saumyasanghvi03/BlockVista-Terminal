# ================ 0. REQUIRED LIBRARIES ================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time as dt_time
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
import plotly.express as px

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
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div, .stMultiSelect>div>div {
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
        
        .market-notification {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--widget-bg);
            border: 2px solid;
            border-radius: 15px;
            padding: 2rem;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
            min-width: 300px;
            max-width: 500px;
        }
        .market-notification.open {
            border-color: var(--green);
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(40, 167, 69, 0.1) 100%);
        }
        .market-notification.warning {
            border-color: #ffc107;
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(255, 193, 7, 0.1) 100%);
        }
        .market-notification.closed {
            border-color: var(--red);
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(220, 53, 69, 0.1) 100%);
        }
        .market-notification.info {
            border-color: #17a2b8;
            background: linear-gradient(135deg, var(--widget-bg) 0%, rgba(23, 162, 184, 0.1) 100%);
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

# ================ 2. DATA SOURCES & INITIALIZATION ================

ENHANCED_DATA_SOURCES = {
    "NIFTY 50": {"yfinance_ticker": "^NSEI", "tradingsymbol": "NIFTY 50", "exchange": "NSE"},
    "BANK NIFTY": {"yfinance_ticker": "^NSEBANK", "tradingsymbol": "BANKNIFTY", "exchange": "NFO"},
    "NIFTY Financial Services": {"yfinance_ticker": "NIFTY_FIN_SERVICE.NS", "tradingsymbol": "FINNIFTY", "exchange": "NFO"},
    "GOLD": {"yfinance_ticker": "GC=F", "tradingsymbol": "GOLDM", "exchange": "MCX"},
    "USDINR": {"yfinance_ticker": "INR=X", "tradingsymbol": "USDINR", "exchange": "CDS"},
    "SENSEX": {"yfinance_ticker": "^BSESN", "tradingsymbol": "SENSEX", "exchange": "BSE"},
    "S&P 500": {"yfinance_ticker": "^GSPC", "tradingsymbol": "^GSPC", "exchange": "yfinance"},
    "NIFTY MIDCAP 100": {"yfinance_ticker": "^CNXMIDCAP", "tradingsymbol": "NIFTYMID100", "exchange": "NSE"}
}
ML_DATA_SOURCES = ENHANCED_DATA_SOURCES

def initialize_session_state():
    """Initializes all necessary session state variables."""
    defaults = {
        'broker': None, 'kite': None, 'profile': None, 'login_animation_complete': False,
        'authenticated': False, 'two_factor_setup_complete': False, 'pyotp_secret': None,
        'theme': 'Dark', 'paper_trading': True,
        'watchlists': {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
        },
        'active_watchlist': "Watchlist 1", 'order_history': [], 'basket': [],
        'last_order_details': {}, 'underlying_pcr': "NIFTY", 'strategy_legs': [],
        'calculated_greeks': None, 'messages': [], 'ml_forecast_df': None,
        'ml_instrument_name': None, 'backtest_results': None, 'fundamental_companies': ['RELIANCE', 'TCS'],
        'hft_last_price': 0, 'hft_tick_log': [], 'market_notifications_shown': {},
        'show_2fa_dialog': False, 'show_qr_dialog': False,
        # Bot states with P&L tracking
        'bot_momentum_running': False, 'bot_momentum_log': [], 'bot_momentum_position': None, 'bot_momentum_trades': [],
        'bot_reversion_running': False, 'bot_reversion_log': [], 'bot_reversion_position': None, 'bot_reversion_trades': [],
        'bot_breakout_running': False, 'bot_breakout_log': [], 'bot_breakout_position': None, 'bot_breakout_trades': [],
        'bot_value_running': False, 'bot_value_log': [], 'bot_value_investments': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================ 3. CORE HELPER & UI FUNCTIONS ================

def get_broker_client():
    return st.session_state.get('kite') if st.session_state.get('broker') == "Zerodha" else None

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    holidays_by_year = {
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
    }
    return holidays_by_year.get(year, [])

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "var(--red)"}
    if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
        return {"status": "OPEN", "color": "var(--green)"}
    return {"status": "CLOSED", "color": "var(--red)"}

def display_header():
    status_info = get_market_status()
    current_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S IST")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align: right;">
            <h5 style="margin: 0;">{current_time}</h5>
            <h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Buy", use_container_width=True, key="header_buy"):
            quick_trade_dialog()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            quick_trade_dialog()
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None):
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    if symbol is None:
        symbol = st.text_input("Symbol").upper()
    
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
    order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
    quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
    price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

    if st.button("Submit Order", use_container_width=True):
        if symbol and quantity > 0:
            instrument_df = get_instrument_df()
            place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

def check_market_timing_notifications(): pass
def show_market_notification(title, message, notification_class, duration=10): pass
def display_overnight_changes_bar(): pass

# ================ 4. DATA, CALCULATION & ORDER FUNCTIONS ================

@st.cache_resource(ttl=3600)
def get_instrument_df():
    client = get_broker_client()
    if not client: return pd.DataFrame()
    try:
        df = pd.DataFrame(client.instruments())
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
        return df
    except Exception:
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None, tag="BlockVista"):
    if st.session_state.get('paper_trading', True):
        trade_price = price
        if order_type == 'MARKET':
            try:
                instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()].iloc[0]
                exchange = instrument['exchange']
                ltp_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}])
                trade_price = ltp_data.iloc[0]['Price']
            except Exception:
                st.toast(f"Could not get LTP for paper trade on {symbol}. Order not placed.", icon="⚠️")
                return None
        
        log_message = f"[PAPER] {transaction_type} {quantity} of {symbol} @ {trade_price:.2f}"
        st.toast(f"📄 {log_message}", icon="🧾")
        st.session_state.order_history.insert(0, {"id": f"PAPER_{random.randint(1000, 9999)}", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "PAPER_FILLED", "price": trade_price})
        return "PAPER_ORDER_ID"
    
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return None
    try:
        instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()].iloc[0]
        exchange = instrument['exchange']
        order_id = client.place_order(
            tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type,
            quantity=int(quantity), order_type=order_type, product=product,
            variety=client.VARIETY_REGULAR, price=price, tag=tag
        )
        st.toast(f"✅ REAL Order placed! ID: {order_id}", icon="🎉")
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "SUBMITTED"})
        return order_id
    except Exception as e:
        st.toast(f"❌ REAL Order failed: {e}", icon="🔥")
        return None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period='1y'):
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    
    days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825*2}
    from_date = datetime.now().date() - timedelta(days=days_to_subtract.get(period, 365))
    to_date = datetime.now().date()
    
    try:
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index).tz_convert('Asia/Kolkata')
        
        df.ta.adx(append=True)
        df.ta.bbands(append=True)
        df.ta.donchian(append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ichimoku(append=True)
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.supertrend(append=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    
    instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
    try:
        quotes = client.quote(instrument_names)
        watchlist = []
        for item in symbols_with_exchange:
            instrument = f"{item['exchange']}:{item['symbol']}"
            if instrument in quotes and quotes[instrument]:
                quote = quotes[instrument]
                prev_close = quote['ohlc']['close']
                change = quote['last_price'] - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                watchlist.append({
                    'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': quote['last_price'],
                    'Change': change, '% Change': pct_change, 'OI': quote.get('oi', 0),
                    'Prev OI': quote.get('oi_day_low', 0),
                })
        return pd.DataFrame(watchlist)
    except Exception:
        return pd.DataFrame()

def calculate_pivot_points(df):
    if df.empty: return {}
    last_day = df.iloc[-1]
    P = (last_day['high'] + last_day['low'] + last_day['close']) / 3
    R1 = (2 * P) - last_day['low']
    S1 = (2 * P) - last_day['high']
    R2 = P + (last_day['high'] - last_day['low'])
    S2 = P - (last_day['high'] - last_day['low'])
    return {'S2': S2, 'S1': S1, 'P': P, 'R1': R1, 'R2': R2}

@st.cache_data(ttl=300)
def get_oi_buildup(instrument_df):
    client = get_broker_client()
    if not client: return pd.DataFrame()
    results = []
    for underlying in ["NIFTY", "BANKNIFTY"]:
        chain, _, ltp, _ = get_options_chain(underlying, instrument_df)
        if chain.empty or ltp == 0: continue
        
        atm_strike = round(ltp / (50 if underlying == "NIFTY" else 100)) * (50 if underlying == "NIFTY" else 100)
        chain_slice = chain[(chain['STRIKE'] >= atm_strike - 500) & (chain['STRIKE'] <= atm_strike + 500)]

        for _, row in chain_slice.iterrows():
            for option_type in ['CALL', 'PUT']:
                symbol = row[option_type]
                try:
                    quote = client.quote(f"NFO:{symbol}")[f"NFO:{symbol}"]
                    oi_change = quote['oi'] - quote.get('open_interest', quote['oi'])
                    price_change = quote['last_price'] - quote['ohlc']['close']

                    if price_change > 0 and oi_change > 0: buildup = "Long Buildup"
                    elif price_change < 0 and oi_change > 0: buildup = "Short Buildup"
                    elif price_change < 0 and oi_change < 0: buildup = "Long Unwinding"
                    elif price_change > 0 and oi_change < 0: buildup = "Short Covering"
                    else: buildup = "Neutral"
                    
                    if abs(oi_change) > 100:
                        results.append({'Contract': symbol, 'Buildup': buildup, 'OI Change': oi_change, 'Price Change': price_change})
                except Exception:
                    continue
    return pd.DataFrame(results).sort_values(by='OI Change', key=abs, ascending=False)

@st.cache_data(ttl=300)
def calculate_option_pain(chain_df):
    if chain_df.empty: return 0, pd.DataFrame()
    strikes = sorted(chain_df['STRIKE'].unique())
    total_loss = []
    for strike_price in strikes:
        loss = 0
        calls_at_strike = chain_df[chain_df['STRIKE'] < strike_price]
        loss += ((strike_price - calls_at_strike['STRIKE']) * calls_at_strike['CALL OI']).sum()
        puts_at_strike = chain_df[chain_df['STRIKE'] > strike_price]
        loss += ((puts_at_strike['STRIKE'] - strike_price) * puts_at_strike['PUT OI']).sum()
        total_loss.append({'Strike': strike_price, 'Total Loss': loss})
    loss_df = pd.DataFrame(total_loss)
    if loss_df.empty: return 0, pd.DataFrame()
    max_pain_strike = loss_df.loc[loss_df['Total Loss'].idxmin()]
    return max_pain_strike['Strike'], loss_df

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

# ... (All other helper functions from the original script are assumed here for brevity)

# ================ 5. PAGE DEFINITIONS ================

def page_dashboard():
    display_header()
    st.title("Dashboard")
    st.info("Dashboard showing market status, watchlist, and key indices.")

def page_advanced_charting():
    display_header()
    st.title("Advanced Charting")
    st.info("Advanced charting with multiple indicators.")

def page_fo_analytics():
    display_header()
    st.title("F&O Analytics Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to access F&O Analytics.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Options Chain", "PCR Analysis", "Option Pain", "OI Buildup"])
    
    with tab1:
        st.subheader("Live Options Chain")
        # Full implementation would be here
        
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        # Full implementation would be here

    with tab3:
        st.subheader("Maximum Option Pain")
        underlying = st.selectbox("Select Underlying for Pain Analysis", ["NIFTY", "BANKNIFTY"], key="pain_ul")
        chain_df, _, ltp, _ = get_options_chain(underlying, instrument_df)
        if not chain_df.empty and ltp > 0:
            with st.spinner("Calculating Option Pain..."):
                max_pain_strike, loss_df = calculate_option_pain(chain_df)
            st.metric(f"Max Pain Strike for {underlying}", f"₹{max_pain_strike:,.0f}")
            fig = px.line(loss_df, x='Strike', y='Total Loss', title='Total Loss for Option Writers at Expiry')
            fig.add_vline(x=max_pain_strike, line_dash="dash", annotation_text="Max Pain")
            fig.add_vline(x=ltp, line_dash="dot", line_color="yellow", annotation_text="Current LTP")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Live OI Buildup Scanner")
        if st.button("Scan F&O Contracts for OI Buildup"):
            with st.spinner("Scanning..."):
                oi_df = get_oi_buildup(instrument_df)
                if not oi_df.empty:
                    st.dataframe(oi_df, use_container_width=True)
                else:
                    st.warning("Could not fetch OI buildup data.")

def page_fundamental_analytics():
    pass

def page_algo_bots():
    pass

def page_advanced_backtester():
    pass

def page_correlation_matrix():
    pass
    
# ... (All other original page functions: page_premarket_pulse, page_forecasting_ml, etc.)

# ================ 6. MAIN APP LOGIC ================

def main_app():
    apply_custom_styling()
    
    if not st.session_state.get('authenticated'):
        login_page()
        return

    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.divider()
    
    st.sidebar.header("Global Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.paper_trading = st.sidebar.toggle("Paper Trading Mode", value=True, help="If ON, all trades are simulated. If OFF, real orders are placed.")
    
    st.sidebar.header("Live Data Refresh")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", 5, 60, 15, disabled=not auto_refresh)
    st.sidebar.divider()

    st.sidebar.header("Navigation")
    pages = {
        "Dashboard": page_dashboard,
        "Advanced Charting": page_advanced_charting,
        "F&O Analytics": page_fo_analytics,
        "Fundamental Analytics": page_fundamental_analytics,
        "Algo Trading Bots": page_algo_bots,
        "Advanced Backtester": page_advanced_backtester,
        "Correlation Matrix": page_correlation_matrix,
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.divider()
    if st.sidebar.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    no_refresh_pages = ["Algo Trading Bots", "Fundamental Analytics", "Advanced Backtester", "Correlation Matrix"]
    if auto_refresh and selection not in no_refresh_pages:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
        
    pages[selection]()

def login_page():
    apply_custom_styling()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("BlockVista Terminal Login")
        api_key = st.secrets.get("ZERODHA_API_KEY")
        api_secret = st.secrets.get("ZERODHA_API_SECRET")

        if not api_key or not api_secret:
            st.error("Kite API credentials not set in Streamlit secrets.")
            st.stop()

        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")

        if request_token:
            try:
                with st.spinner("Authenticating..."):
                    data = kite.generate_session(request_token, api_secret=api_secret)
                    kite.set_access_token(data["access_token"])
                    st.session_state.kite = kite
                    st.session_state.profile = kite.profile()
                    st.session_state.broker = "Zerodha"
                    st.session_state.authenticated = True
                    st.query_params.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            login_url = kite.login_url()
            st.link_button("Login with Zerodha Kite", login_url, use_container_width=True)

if __name__ == "__main__":
    initialize_session_state()
    if st.session_state.get('authenticated') and st.session_state.get('profile'):
        main_app()
    else:
        login_page()
