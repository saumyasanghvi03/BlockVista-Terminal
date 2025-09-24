# ================ 0. REQUIRED LIBRARIES ================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
import pytz
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
from time import mktime
import requests
import io
import time as a_time # Renaming to avoid conflict
import re
import yfinance as yf
import json
import qrcode
import pyotp
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. For the best UI, please create it.")

load_css("style.css")

# Centralized data source configuration for ML models
ML_DATA_SOURCES = {
    "NIFTY 50": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv", "tradingsymbol": "NIFTY 50", "exchange": "NFO"},
    "BANK NIFTY": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv", "tradingsymbol": "BANKNIFTY", "exchange": "NFO"},
    "NIFTY Financial Services": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv", "tradingsymbol": "FINNIFTY", "exchange": "NFO"},
    "GOLD": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv", "tradingsymbol": "GOLDM", "exchange": "MCX"},
    "USDINR": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv", "tradingsymbol": "USDINR", "exchange": "CDS"},
    "SENSEX": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv", "tradingsymbol": None, "exchange": None},
    "S&P 500": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv", "tradingsymbol": None, "exchange": None}
}

# ================ 2. USER AUTHENTICATION & 2FA FUNCTIONS ================
USERS_FILE = "users.json"

def load_users():
    try:
        with open(USERS_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(USERS_FILE, 'w') as f: json.dump({}, f)
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f: json.dump(users, f, indent=4)

def show_qr_setup(username):
    users = load_users(); secret = users[username]['otp_secret']
    st.subheader(f"Setup Authenticator App for {username}")
    st.warning("This QR code will only be shown once. Please scan it now.", icon="⚠️")
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name="BlockVista Terminal")
    img = qrcode.make(uri); buf = io.BytesIO(); img.save(buf); buf.seek(0)
    st.image(buf)
    st.code(secret, language=None)
    st.info("Scan the QR code with an authenticator app (e.g., Google Authenticator). You can also manually enter the secret key.")
    if st.button("Continue to Login"):
        del st.session_state['new_user_setup']; st.rerun()

def user_authentication_flow():
    st.title("BlockVista Terminal Access")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        users = load_users()
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            otp = st.text_input("6-Digit Authenticator Code")
            if st.form_submit_button("Login"):
                if username in users and check_password_hash(users[username]['password_hash'], password):
                    if pyotp.TOTP(users[username]['otp_secret']).verify(otp):
                        st.session_state['user_authenticated'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else: st.error("Invalid Authenticator Code.")
                else: st.error("Invalid username or password.")
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")
            if st.form_submit_button("Register"):
                if new_username and new_password:
                    users = load_users()
                    if new_username in users: st.error("Username already exists.")
                    else:
                        users[new_username] = {"password_hash": generate_password_hash(new_password), "otp_secret": pyotp.random_base32()}
                        save_users(users); st.session_state['new_user_setup'] = new_username; st.rerun()
                else: st.warning("Username and password cannot be empty.")
    return False

# ================ 3. HELPER FUNCTIONS ================

def get_broker_client(): return st.session_state.get('kite')

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None):
    instrument_df = get_instrument_df()
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    if symbol is None: symbol = st.text_input("Symbol").upper()
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
    order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
    quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
    price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0
    if st.button("Submit Order", use_container_width=True):
        if symbol and quantity > 0: place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None); st.rerun()
        else: st.warning("Please fill in all fields.")

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    holidays_by_year = {2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'], 2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']}
    return holidays_by_year.get(year, [])

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata'); now = datetime.now(ist); holidays = get_market_holidays(now.year)
    market_open, market_close = time(9, 15), time(15, 30)
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays: return {"status": "CLOSED", "color": "#FF4B4B"}
    if market_open <= now.time() <= market_close: return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    status_info = get_market_status(); ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    with col2: st.markdown(f"""<div style="text-align: right;"><h5 style="margin: 0;">{current_time}</h5><h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5></div>""", unsafe_allow_html=True)
    with col3:
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Buy", use_container_width=True, key="header_buy"): quick_trade_dialog()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"): quick_trade_dialog()
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# ================ 4. CORE DATA & CHARTING FUNCTIONS ================
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
    fig = go.Figure(); template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
    if df.empty: return fig
    chart_df = df.copy(); chart_df.columns = [col.lower() for col in chart_df.columns]
    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    else: fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
    if forecast_df is not None: fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    client = get_broker_client(); return pd.DataFrame(client.instruments()) if client else pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period='6mo', from_date=None, to_date=None):
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    try:
        if not to_date: to_date = datetime.now().date()
        if not from_date:
            days_map = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_map.get(period, 1825))
        
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records);
        if df.empty: return df
        df.set_index('date', inplace=True); df.index = pd.to_datetime(df.index)
        try: 
            df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True); df.ta.coppock(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True); df.ta.macd(append=True); df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
        except Exception: pass
        return df
    except Exception as e: st.error(f"Kite API Error (Historical): {e}"); return pd.DataFrame()

def get_watchlist_data(symbols_with_exchange):
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
    try:
        quotes = client.quote(instrument_names)
        watchlist = []
        for item in symbols_with_exchange:
            instrument = f"{item['exchange']}:{item['symbol']}"
            if instrument in quotes:
                q = quotes[instrument]; lp = q['last_price']; pc = q['ohlc']['close']; ch = lp - pc
                watchlist.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': lp, 'Change': ch, '% Change': (ch / pc * 100) if pc != 0 else 0})
        return pd.DataFrame(watchlist)
    except Exception as e: st.toast(f"Error fetching watchlist data: {e}", icon="⚠️"); return pd.DataFrame()

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
    try:
        ltp_map = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}
        ul_name = f"NSE:{ltp_map.get(underlying, underlying)}"
        ul_ltp = client.ltp(ul_name)[ul_name]['last_price']
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == 'NFO')]
        if options.empty: return pd.DataFrame(), None, ul_ltp, []
        expiries = sorted(pd.to_datetime(options['expiry'].unique()))
        available_expiries = [e for e in expiries if e.date() >= datetime.now().date()]
        if not expiry_date: expiry_date = available_expiries[0] if available_expiries else None
        if not expiry_date: return pd.DataFrame(), None, ul_ltp, available_expiries
        chain_df = options[options['expiry'] == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments = [f"NFO:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments: return pd.DataFrame(), expiry_date, ul_ltp, available_expiries
        quotes = client.quote(instruments)
        ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP']], pe_df[['tradingsymbol', 'strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
        return final_chain[['CALL', 'CALL LTP', 'STRIKE', 'PUT LTP', 'PUT']], expiry_date, ul_ltp, available_expiries
    except Exception: return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    try:
        pos = client.positions().get('net', []); hld = client.holdings()
        pos_df = pd.DataFrame(pos)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if pos else pd.DataFrame()
        hld_df = pd.DataFrame(hld)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if hld else pd.DataFrame()
        return pos_df, hld_df, pos_df['pnl'].sum() if not pos_df.empty else 0.0, (hld_df['quantity'] * hld_df['average_price']).sum() if not hld_df.empty else 0.0
    except Exception as e: st.error(f"Kite API Error (Portfolio): {e}"); return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    client = get_broker_client()
    if not client: st.error("Broker not connected."); return
    try:
        is_option = any(char.isdigit() for char in symbol)
        exchange = 'NFO' if is_option else instrument_df[instrument_df['tradingsymbol'] == symbol.upper()].iloc[0]['exchange']
        if exchange is None: st.error(f"Symbol '{symbol}' not found."); return
        order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")
        st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    analyzer = SentimentIntensityAnalyzer()
    sources = {"ET": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "MC": "https://www.moneycontrol.com/rss/business.xml"}
    news = []
    for source, url in sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if query is None or query.lower() in entry.title.lower():
                    news.append({"source": source, "title": entry.title, "link": entry.link, "date": datetime.fromtimestamp(mktime(entry.published_parsed)) if hasattr(entry, 'published_parsed') else datetime.now(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception: continue
    return pd.DataFrame(news)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call": price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else: price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    if T <= 0 or market_price <= 0: return np.nan
    func = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try: return newton(func, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError): return np.nan

def interpret_indicators(df):
    if df.empty: return {}
    latest = df.iloc[-1].copy(); latest.index = latest.index.str.lower(); interpretation = {}
    r = latest.get('rsi_14'); macd = latest.get('macd_12_26_9'); signal = latest.get('macds_12_26_9'); adx = latest.get('adx_14')
    if r is not None: interpretation['RSI (14)'] = "Overbought (Bearish)" if r > 70 else "Oversold (Bullish)" if r < 30 else "Neutral"
    if macd is not None and signal is not None: interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    if adx is not None: interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else "Weak Trend"
    return interpretation

@st.cache_data(ttl=3600)
def get_sector_data():
    try: return pd.read_csv("sectors.csv")
    except FileNotFoundError: return None

def style_option_chain(df, ltp):
    atm_strike = abs(df['STRIKE'] - ltp).idxmin()
    return df.style.apply(lambda x: ['background-color: #2c3e50' if x.name < atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['CALL', 'CALL LTP']], axis=1)\
                   .apply(lambda x: ['background-color: #2c3e50' if x.name > atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['PUT', 'PUT LTP']], axis=1)

@st.cache_data(ttl=1800)
def get_global_indices_data(tickers):
    try:
        df = yf.download(tickers, period="2d", progress=False)['Close']
        if df.empty or len(df) < 2: return pd.DataFrame()
        data = []
        for ticker in tickers:
            if ticker in df.columns:
                lp, pc = df[ticker].iloc[-1], df[ticker].iloc[-2]; chg = lp - pc
                data.append({'Ticker': ticker, 'Price': lp, 'Change': chg, '% Change': (chg / pc) * 100})
        return pd.DataFrame(data)
    except Exception: return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS (FULL VERSIONS RESTORED) ============

def page_dashboard():
    display_header()
    instrument_df = get_instrument_df();
    if instrument_df.empty: st.info("Please connect to a broker to view the dashboard."); return
    index_symbols = [{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'}, {'symbol': 'INDIA VIX', 'exchange': 'NSE'}]
    index_data = get_watchlist_data(index_symbols)
    if not index_data.empty:
        cols = st.columns(len(index_data))
        for i, col in enumerate(cols):
            with col:
                change = index_data.iloc[i]['Change']; blink_class = "positive-blink" if change > 0 else "negative-blink" if change < 0 else ""
                st.markdown(f"""<div class="metric-card {blink_class}"><h4>{index_data.iloc[i]['Ticker']}</h4><h2>{index_data.iloc[i]['Price']:,.2f}</h2><p style="color: {'#28a745' if change > 0 else '#FF4B4B'}; margin: 0;">{change:,.2f} ({index_data.iloc[i]['% Change']:.2f}%)</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])
        with tab1:
            if 'watchlists' not in st.session_state:
                st.session_state.watchlists = {"Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}]}
            if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]
            st.session_state.active_watchlist = st.radio("Select Watchlist", options=st.session_state.watchlists.keys(), horizontal=True, label_visibility="collapsed")
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]
            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 15: st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list): active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange}); st.rerun()
                        else: st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            if active_list:
                watchlist_data = get_watchlist_data(active_list)
                if not watchlist_data.empty:
                    for _, row in watchlist_data.iterrows():
                        w_cols = st.columns([3, 2, 1, 1, 1, 1]); color = '#28a745' if row['Change'] > 0 else '#FF4B4B'
                        w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:gray;'>{row['Exchange']}</small>", unsafe_allow_html=True)
                        w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                        quantity = w_cols[2].number_input("Qty", min_value=1, step=1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                        if w_cols[3].button("B", key=f"buy_{row['Ticker']}", use_container_width=True): place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                        if w_cols[4].button("S", key=f"sell_{row['Ticker']}", use_container_width=True): place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
                        if w_cols[5].button("🗑️", key=f"del_{row['Ticker']}", use_container_width=True): st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != row['Ticker']]; st.rerun()
        with tab2:
            st.subheader("My Portfolio")
            _, holdings_df, total_pnl, total_investment = get_portfolio()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
            st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            with st.expander("View Holdings"): st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty: st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else: st.warning("Could not load NIFTY 50 chart. Market might be closed.")

def page_advanced_charting():
    display_header(); st.title("Advanced Charting"); instrument_df = get_instrument_df()
    if instrument_df.empty: st.info("Please connect to a broker to use the charting tools."); return
    num_charts = st.radio("Select Chart Layout", [1, 2, 4], index=0, horizontal=True)
    def display_chart_widget(index):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Symbol", "RELIANCE", key=f"ticker_{index}").upper()
        period = c2.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key=f"period_{index}")
        interval = c3.selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key=f"interval_{index}")
        chart_type = c4.selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{index}")
        token = get_instrument_token(ticker, instrument_df)
        if token:
            data = get_historical_data(token, interval, period=period)
            if not data.empty: st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True, key=f"chart_{index}")
            else: st.warning(f"No chart data for {ticker}.")
        else: st.error(f"Ticker '{ticker}' not found.")
    if num_charts == 1: display_chart_widget(0)
    elif num_charts == 2:
        cols = st.columns(2)
        with cols[0]: display_chart_widget(0)
        with cols[1]: display_chart_widget(1)
    elif num_charts == 4:
        row1 = st.columns(2)
        with row1[0]: display_chart_widget(0)
        with row1[1]: display_chart_widget(1)
        st.markdown("---")
        row2 = st.columns(2)
        with row2[0]: display_chart_widget(2)
        with row2[1]: display_chart_widget(3)

def page_options_hub():
    display_header(); st.title("Options Hub"); instrument_df = get_instrument_df()
    if instrument_df.empty: st.info("Please connect to a broker to use the Options Hub."); return
    col1, col2 = st.columns([1, 2]);
    with col1:
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "CRUDEOIL", "USDINR"])
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        if available_expiries:
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries, format_func=lambda d: d.strftime('%d %b %Y'))
            if selected_expiry != expiry: chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
        else: st.warning(f"No upcoming expiries found for {underlying}.")
        if not chain_df.empty and underlying_ltp > 0 and expiry:
            st.subheader("Greeks & Quick Trade")
            option_list = ["-Select-"] + chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist()
            option_selection = st.selectbox("Analyze or Trade an Option", option_list)
            if option_selection and option_selection != "-Select-":
                option_details = instrument_df[instrument_df['tradingsymbol'] == option_selection].iloc[0]
                strike_price, option_type = option_details['strike'], option_details['instrument_type'].lower()
                ltp_col = 'CALL LTP' if option_type == 'ce' else 'PUT LTP'
                symbol_col = 'CALL' if option_type == 'ce' else 'PUT'
                ltp = chain_df[chain_df[symbol_col] == option_selection][ltp_col].iloc[0]
                T = max((pd.to_datetime(expiry).date() - datetime.now().date()).days, 0) / 365.0
                iv = implied_volatility(underlying_ltp, strike_price, T, 0.07, ltp, option_type)
                if not np.isnan(iv) and iv > 0:
                    greeks = black_scholes(underlying_ltp, strike_price, T, 0.07, iv, option_type)
                    c1, c2, c3 = st.columns(3); c1.metric("Delta", f"{greeks['delta']:.3f}"); c2.metric("IV", f"{iv*100:.2f}%"); c3.metric("Vega", f"{greeks['vega']:.3f}");
                with st.form(key="option_trade_form"):
                    q_cols = st.columns([1,1,1])
                    quantity = q_cols[0].number_input("Lots", min_value=1, step=1, key="opt_qty")
                    buy_btn = q_cols[1].form_submit_button("Buy")
                    sell_btn = q_cols[2].form_submit_button("Sell")
                    if buy_btn: place_order(instrument_df, option_selection, quantity * option_details['lot_size'], 'MARKET', 'BUY', 'MIS')
                    if sell_btn: place_order(instrument_df, option_selection, quantity * option_details['lot_size'], 'MARKET', 'SELL', 'MIS')
    with col2:
        st.subheader(f"{underlying} Options Chain")
        if not chain_df.empty and expiry:
            st.caption(f"Expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')} | Spot: {underlying_ltp:,.2f}")
            st.dataframe(style_option_chain(chain_df, underlying_ltp), use_container_width=True, hide_index=True)
        else: st.warning("Could not fetch options chain.")

def page_portfolio_and_risk():
    display_header(); st.title("Portfolio & Risk");
    if not get_broker_client(): st.info("Connect to broker to view your portfolio."); return
    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Live Order Book"])
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty: st.dataframe(positions_df, use_container_width=True, hide_index=True); st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        else: st.info("No open positions for the day.")
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty: st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        else: st.info("No holdings found.")
    with tab3:
        st.subheader("Today's Order Book")
        client = get_broker_client()
        if client:
            try:
                orders = client.orders()
                if orders: st.dataframe(pd.DataFrame(orders)[['order_timestamp', 'tradingsymbol', 'transaction_type', 'order_type', 'quantity', 'average_price', 'status']], use_container_width=True, hide_index=True)
                else: st.info("No orders placed today.")
            except Exception as e: st.error(f"Failed to fetch order book: {e}")

def page_alpha_engine():
    display_header(); st.title("Alpha Engine: News Sentiment"); query = st.text_input("Enter a stock, commodity, or currency to analyze", "NIFTY")
    with st.spinner("Fetching and analyzing news..."):
        news_df = fetch_and_analyze_news(query)
        if not news_df.empty:
            avg_sentiment = news_df['sentiment'].mean(); sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            st.metric(f"Overall News Sentiment for '{query}'", sentiment_label, f"{avg_sentiment:.3f}")
            st.dataframe(news_df.drop(columns=['date']), use_container_width=True, hide_index=True, column_config={"link": st.column_config.LinkColumn("Link", display_text="Read Article")})
        else: st.info(f"No recent news found for '{query}'.")

def page_portfolio_analytics():
    display_header(); st.title("Portfolio Analytics"); _, holdings_df, _, _ = get_portfolio(); sector_df = get_sector_data()
    if holdings_df.empty: st.info("No holdings found to analyze."); return
    if sector_df is None: st.warning("`sectors.csv` not found."); return
    holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
    holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left'); holdings_df['Sector'].fillna('Uncategorized', inplace=True)
    st.metric("Total Portfolio Value", f"₹{holdings_df['current_value'].sum():,.2f}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock-wise Allocation"); fig = go.Figure(data=[go.Pie(labels=holdings_df['tradingsymbol'], values=holdings_df['current_value'], hole=.3, textinfo='label+percent')]); fig.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'); st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Sector-wise Allocation"); sector_alloc = holdings_df.groupby('Sector')['current_value'].sum().reset_index(); fig = go.Figure(data=[go.Pie(labels=sector_alloc['Sector'], values=sector_alloc['current_value'], hole=.3, textinfo='label+percent')]); fig.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'); st.plotly_chart(fig, use_container_width=True)

def page_option_strategy_builder():
    display_header(); st.title("Options Strategy Builder")
    st.info("This tool visualizes the payoff for various option strategies.", icon="ℹ️")
    st.warning("Option Strategy Builder logic needs to be fully implemented for payoff calculations.", icon="⚠️")
    instrument_df = get_instrument_df();
    if instrument_df.empty: st.info("Please connect to a broker to use the strategy builder."); return
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Strategy Configuration")
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        strategy = st.selectbox("Select Strategy", ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", "Short Straddle", "Iron Condor"])
        _, expiry_date, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        if available_expiries:
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries, format_func=lambda d: d.strftime('%d %b %Y'))
            if selected_expiry != expiry_date: _, expiry_date, underlying_ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
        st.metric(f"{underlying} Spot Price", f"{underlying_ltp:,.2f}")
    with col2: st.subheader("Payoff Diagram")

def page_premarket_pulse():
    display_header(); st.title("Premarket Pulse")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Cues (Live)")
        global_indices_tickers = {"NASDAQ": "^IXIC", "NIKKEI 225": "^N225", "Dow Jones": "^DJI", "Gold Futures": "GC=F", "Crude Oil": "CL=F"}
        global_indices_data = get_global_indices_data(list(global_indices_tickers.values()))
        if not global_indices_data.empty:
            global_indices_data['Ticker'] = global_indices_data['Ticker'].map({v: k for k, v in global_indices_tickers.items()})
            for _, row in global_indices_data.iterrows(): st.metric(f"{row['Ticker']} Price", f"{row['Price']:,.2f}", delta=f"{row['Change']:,.2f} ({row['% Change']:.2f}%)")
        else: st.warning("Could not retrieve data for all global indices.")
    with col2:
        st.subheader("GIFT NIFTY Chart (Proxy)")
        st.caption("Displaying NIFTY 50 futures as a proxy for GIFT NIFTY.")
        try:
            nifty_data = yf.download("^CNXNIFTY", period="1d", interval="1m", progress=False)
            if not nifty_data.empty: st.plotly_chart(create_chart(nifty_data, "GIFT NIFTY"), use_container_width=True)
            else: st.warning("Could not load GIFT NIFTY chart.")
        except Exception as e: st.error(f"Error fetching GIFT NIFTY data: {e}")

def page_ai_trading_journal():
    display_header(); st.title("AI Trading Journal")
    st.info("This journal helps you reflect on your trading psychology throughout the day.")
    if "journal_log" not in st.session_state: st.session_state.journal_log = []
    user_response = st.text_area("Your thoughts...", key="journal_response")
    if st.button("Submit Entry"):
        if user_response:
            st.session_state.journal_log.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "response": user_response})
            st.success("Entry saved!")
    st.markdown("---"); st.subheader("Your Past Entries")
    if st.session_state.journal_log:
        for entry in reversed(st.session_state.journal_log):
            st.expander(f"Entry on {entry['timestamp']}").write(f"{entry['response']}")
    else: st.info("Your journal is currently empty.")

def page_ai_discovery():
    display_header(); st.title("AI Discovery Engine")
    st.info("This engine simulates advanced AI analysis for educational purposes.", icon="🧠")
    st.warning("AI Discovery logic needs to be fully implemented.", icon="⚠️")

def page_forecasting_ml():
    display_header(); st.title("Advanced ML Forecasting")
    st.info("This is for educational purposes and is not financial advice.", icon="ℹ️")
    st.warning("ML model training and forecasting logic needs to be fully implemented.", icon="⚠️")

def page_basket_orders():
    display_header(); st.title("Basket Orders")
    if 'basket' not in st.session_state: st.session_state.basket = []
    instrument_df = get_instrument_df();
    if instrument_df.empty: st.info("Please connect to a broker to use basket orders."); return
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
                if symbol:
                    order = {"tradingsymbol": symbol, "transaction_type": transaction_type, "quantity": quantity, "product": product, "order_type": order_type}
                    if order_type == "LIMIT": order["price"] = price
                    st.session_state.basket.append(order); st.success(f"Added {symbol} to basket.")
                else: st.warning("Please enter a symbol.")
    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket: st.info("Your basket is empty.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.basket), use_container_width=True)
            if st.button("Execute Basket", use_container_width=True, type="primary"): st.warning("Basket execution logic needs to be fully implemented.")
            if st.button("Clear Basket", use_container_width=True): st.session_state.basket = []; st.rerun()

def page_ai_assistant():
    display_header(); st.title("Portfolio-Aware Assistant")
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = "Sorry, I can only help with basic queries like 'show holdings' or 'price of RELIANCE'."
                client = get_broker_client()
                if not client: response = "Please connect to a broker first."
                prompt_lower = prompt.lower()
                if "holdings" in prompt_lower:
                    _, h, _, _ = get_portfolio(); response = f"Holdings:\n```\n{tabulate(h, headers='keys')}\n```" if not h.empty else "No holdings."
                elif "price of" in prompt_lower:
                    try:
                        ticker = prompt.split("of")[-1].strip().upper()
                        ltp = get_watchlist_data([{'symbol': ticker, 'exchange': 'NSE'}])
                        response = f"Price of {ticker} is {ltp.iloc[0]['Price'] if not ltp.empty else 'N/A'}."
                    except Exception: response = "Could not fetch price."
                st.markdown(response); st.session_state.messages.append({"role": "assistant", "content": response})

# ============ 6. BROKER LOGIN AND MAIN APP ============

def show_broker_login_animation():
    st.title("BlockVista Terminal")
    progress_bar = st.progress(0); status_text = st.empty()
    steps = {"Authenticating Broker...": 25, "Establishing connection...": 50, "Fetching feeds...": 75, "Initializing...": 100}
    for text, progress in steps.items():
        status_text.text(f"STATUS: {text}"); progress_bar.progress(progress); a_time.sleep(0.8)
    st.session_state['broker_login_animation_complete'] = True; st.rerun()

def broker_login_page():
    st.title("BlockVista Terminal"); st.subheader("Broker Login")
    st.info(f"Welcome, {st.session_state.get('username', 'user')}! Please connect your broker account.")
    broker = st.selectbox("Select Broker", ["Zerodha"])
    if broker == "Zerodha":
        try: api_key = st.secrets["ZERODHA_API_KEY"]; api_secret = st.secrets["ZERODHA_API_SECRET"]
        except (FileNotFoundError, KeyError): st.error("Kite API credentials not found in secrets."); st.stop()
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                kite.set_access_token(st.session_state.access_token)
                st.session_state.kite = kite; st.session_state.profile = kite.profile(); st.session_state.broker = "Zerodha"
                st.query_params.clear(); st.rerun()
            except Exception as e: st.error(f"Broker authentication failed: {e}. Please try again.")
        else: st.link_button("Login with Zerodha Kite", kite.login_url(), use_container_width=True)

def main_app():
    # Initialize session state variables
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = 'Cash'
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    if 'watchlists' not in st.session_state: st.session_state.watchlists = {"Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}]}

    st.markdown(f'<body class="{"light-theme" if st.session_state.get("theme") == "Light" else ""}"></body>', unsafe_allow_html=True)
    
    st.sidebar.title(f"Welcome, {st.session_state.get('username')}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options"], horizontal=True)
    st.sidebar.divider()
    
    st.sidebar.header("Live Data")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Cash": {
            "Dashboard": page_dashboard, "Premarket Pulse": page_premarket_pulse,
            "Advanced Charting": page_advanced_charting, "Basket Orders": page_basket_orders,
            "Portfolio Analytics": page_portfolio_analytics, "Alpha Engine": page_alpha_engine,
            "Portfolio & Risk": page_portfolio_and_risk, "Forecasting & ML": page_forecasting_ml,
            "AI Trading Journal": page_ai_trading_journal, "AI Discovery": page_ai_discovery, "AI Assistant": page_ai_assistant
        },
        "Futures": {
            "Dashboard": page_dashboard, "Premarket Pulse": page_premarket_pulse,
            "Advanced Charting": page_advanced_charting, "Basket Orders": page_basket_orders,
            "Portfolio & Risk": page_portfolio_and_risk, "Alpha Engine": page_alpha_engine, "AI Assistant": page_ai_assistant
        },
        "Options": {
            "Options Hub": page_options_hub, "Strategy Builder": page_option_strategy_builder,
            "Portfolio & Risk": page_portfolio_and_risk, "AI Trading Journal": page_ai_trading_journal,
            "AI Discovery": page_ai_discovery, "AI Assistant": page_ai_assistant
        }
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector', label_visibility="collapsed")
    
    st.sidebar.divider()
    if st.sidebar.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["Forecasting & ML", "AI Assistant", "AI Discovery", "AI Trading Journal"]:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

# ============ 7. EXECUTION FLOW ============

if __name__ == "__main__":
    if not st.session_state.get("user_authenticated"):
        if 'new_user_setup' in st.session_state: show_qr_setup(st.session_state['new_user_setup'])
        else: user_authentication_flow()
    elif 'profile' not in st.session_state:
        broker_login_page()
    else:
        if st.session_state.get('broker_login_animation_complete', False): main_app()
        else: show_broker_login_animation()

