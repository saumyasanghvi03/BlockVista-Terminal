# ================ 0. REQUIRED LIBRARIES (MINIMAL FOR FAST LOGIN) ================
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
import pytz
import time as a_time # Renaming to avoid conflict with datetime.time

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

# --- UI ENHANCEMENT: Load Custom CSS for Trader UI ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. For the best UI, please create it.")

load_css("style.css")


# Centralized data source configuration for ML models
ML_DATA_SOURCES = {
    "NIFTY 50": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv"},
    "BANK NIFTY": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv"},
    "NIFTY Financial Services": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv"},
}

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    return st.session_state.get('kite')

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None, exchange=None):
    instrument_df = st.session_state.get('instrument_df', pd.DataFrame())
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    
    if symbol is None: symbol = st.text_input("Symbol").upper()
    
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
    order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
    quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
    price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

    if st.button("Submit Order", use_container_width=True):
        if symbol and quantity > 0:
            place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    return {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
    }.get(year, [])

def get_market_status():
    """Checks market status with a 60-second cache."""
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    if 'market_status' in st.session_state and (now_ist - st.session_state.market_status['timestamp'] < timedelta(seconds=60)):
        return st.session_state.market_status['data']

    holidays = get_market_holidays(now_ist.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    status = {"status": "CLOSED", "color": "#FF4B4B"}
    if now_ist.weekday() < 5 and now_ist.strftime('%Y-%m-%d') not in holidays:
        if market_open_time <= now_ist.time() <= market_close_time:
            status = {"status": "OPEN", "color": "#28a745"}

    st.session_state.market_status = {'timestamp': now_ist, 'data': status}
    return status

def display_header():
    """Displays the main header."""
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
        if b_col1.button("Buy", use_container_width=True, key="header_buy"): quick_trade_dialog()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"): quick_trade_dialog()
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

@st.cache_data
def calculate_indicators(_df):
    """Calculates a minimal set of indicators."""
    import pandas_ta as ta
    df = _df.copy()
    try:
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
    except Exception: pass
    return df

@st.cache_data(ttl=5)
def get_historical_data_raw(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches raw OHLCV data."""
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    if not to_date: to_date = datetime.now().date()
    if not from_date:
        days = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365}.get(period, 365)
        from_date = to_date - timedelta(days=days)
    try:
        records = client.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception: return pd.DataFrame()

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================
def create_chart(df, ticker, forecast_df=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    if df.empty: return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def get_instrument_token(symbol, exchange='NSE'):
    instrument_df = st.session_state.get('instrument_df', pd.DataFrame())
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=5)
def fetch_dashboard_data(index_symbols, watchlist_symbols):
    """Fetches all dashboard quotes in a single API call."""
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame()
    
    all_symbols_map = {f"{s['exchange']}:{s['symbol']}": s for s in index_symbols}
    all_symbols_map.update({f"{s['exchange']}:{s['symbol']}": s for s in watchlist_symbols})
    if not all_symbols_map: return pd.DataFrame(), pd.DataFrame()

    try:
        quotes = client.quote(list(all_symbols_map.keys()))
        processed_data = []
        for instrument, quote in quotes.items():
            change = quote['last_price'] - quote['ohlc']['close']
            pct_change = (change / quote['ohlc']['close'] * 100) if quote['ohlc']['close'] != 0 else 0
            original_symbol = all_symbols_map[instrument]
            processed_data.append({'Ticker': original_symbol['symbol'], 'Exchange': original_symbol['exchange'], 'Price': quote['last_price'], 'Change': change, '% Change': pct_change})
        
        all_data_df = pd.DataFrame(processed_data)
        index_tickers = [s['symbol'] for s in index_symbols]
        return all_data_df[all_data_df['Ticker'].isin(index_tickers)], all_data_df[~all_data_df['Ticker'].isin(index_tickers)]
    except: return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_option_chain_nse(underlying):
    """Fallback function to get option chain data directly from NSE."""
    from nselib import trading_info
    try:
        symbol_map = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY", "FINNIFTY": "FINNIFTY"}
        chain_data = trading_info.get_option_chain(symbol_map.get(underlying, underlying))
        df = chain_data.rename(columns={'CE_expiryDate': 'expiry_CE', 'CE_strikePrice': 'STRIKE', 'CE_lastPrice': 'CALL LTP', 'PE_lastPrice': 'PUT LTP'})
        df['CALL'] = df.apply(lambda row: f"{underlying}{pd.to_datetime(row['expiry_CE']).strftime('%d%b%y').upper()}{int(row['STRIKE'])}CE", axis=1)
        df['PUT'] = df.apply(lambda row: f"{underlying}{pd.to_datetime(row['expiry_CE']).strftime('%d%b%y').upper()}{int(row['STRIKE'])}PE", axis=1)
        expiries = sorted(pd.to_datetime(chain_data['CE_expiryDate'].unique()))
        return df[['CALL', 'CALL LTP', 'STRIKE', 'PUT LTP', 'PUT']].fillna(0), expiries[0], chain_data['underlyingValue'].iloc[0], expiries
    except: return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=5)
def get_options_chain(underlying, expiry_date_str=None):
    """Main function to get option chain. Tries KiteConnect first, then falls back to NSE."""
    client, instrument_df = get_broker_client(), st.session_state.get('instrument_df')
    try:
        if not client or instrument_df is None or instrument_df.empty: raise ConnectionError("KiteConnect client not available.")
        
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == 'NFO')]
        if options.empty: raise ValueError("No options found.")
            
        expiries = sorted(pd.to_datetime(options['expiry'].unique()).date)
        selected_expiry = pd.to_datetime(expiry_date_str).date() if expiry_date_str else expiries[0]
        chain_df = options[options['expiry'].dt.date == selected_expiry].sort_values(by='strike')
        
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY"}.get(underlying, underlying)
        underlying_ltp = client.ltp(f"NSE:{ltp_symbol}")[f"NSE:{ltp_symbol}"]['last_price']
        
        atm_strikes = chain_df[abs(chain_df['strike'] - underlying_ltp) < (underlying_ltp * 0.10)]
        
        instruments_to_fetch = [f"NFO:{s}" for s in atm_strikes['tradingsymbol']]
        quotes = client.quote(instruments_to_fetch)
        
        atm_strikes['LTP'] = atm_strikes['tradingsymbol'].apply(lambda x: quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        
        ce_df = atm_strikes[atm_strikes['instrument_type'] == 'CE']
        pe_df = atm_strikes[atm_strikes['instrument_type'] == 'PE']
        
        final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP']], pe_df[['tradingsymbol', 'strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer')
        final_chain.rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}, inplace=True)
        
        return final_chain[['CALL', 'CALL LTP', 'STRIKE', 'PUT LTP', 'PUT']].fillna(0), "KiteConnect (Live)", selected_expiry, underlying_ltp, expiries
    except Exception as e:
        st.toast(f"KiteConnect failed: {e}. Falling back to NSE.", icon="⚠️")
        final_chain, expiry, ltp, expiries = fetch_option_chain_nse(underlying)
        return final_chain, "NSE (Delayed)", expiry, ltp, expiries

@st.cache_data(ttl=10)
def get_portfolio():
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    try:
        positions = client.positions().get('net', [])
        holdings = client.holdings()
        positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
        holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
        return positions_df, holdings_df, positions_df['pnl'].sum(), (holdings_df['quantity'] * holdings_df['average_price']).sum()
    except: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    client = get_broker_client()
    if not client: st.error("Broker not connected."); return
    try:
        is_option = any(char.isdigit() for char in symbol)
        exchange = 'NFO' if is_option else instrument_df[instrument_df['tradingsymbol'] == symbol.upper()].iloc[0]['exchange']
        order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    news_sources = {"Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml"}
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if query is None or query.lower() in entry.title.lower():
                    all_news.append({"source": source, "title": entry.title, "link": entry.link})
        except: continue
    return pd.DataFrame(all_news)

@st.cache_data(show_spinner=False)
def train_simple_forecast_model(_data, forecast_horizon):
    from sklearn.linear_model import LinearRegression
    import numpy as np
    if _data.empty or len(_data) < 10: return None, None
    try:
        df = _data[['close']].copy()
        df['time'] = np.arange(len(df.index))
        recent_df = df.tail(10)
        X = recent_df[['time']]
        y = recent_df['close']
        model = LinearRegression().fit(X, y)
        last_time = df['time'].iloc[-1]
        future_time = np.arange(last_time + 1, last_time + 1 + forecast_horizon).reshape(-1, 1)
        future_predictions = model.predict(future_time)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
        return pd.DataFrame({'Predicted': future_predictions}, index=future_dates), None
    except: return None, None


@st.cache_data
def load_and_combine_data(instrument_name):
    import requests
    import io
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info: return pd.DataFrame()
    try:
        response = requests.get(source_info['github_url'])
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True)
        hist_df.set_index('Date', inplace=True)
        hist_df.columns = [col.lower() for col in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        return hist_df.sort_index()
    except: return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS ================
# NOTE: All page functions are defined here. They are not in a separate file.

def page_dashboard():
    display_header()
    instrument_df = st.session_state.get('instrument_df', pd.DataFrame())
    if instrument_df.empty: st.info("Instruments loading in background... please wait."); return

    index_symbols = [{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'}, {'symbol': 'INDIA VIX', 'exchange': 'NSE'}]
    active_list = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
    
    index_data, watchlist_data = fetch_dashboard_data(index_symbols, active_list)
    
    if not index_data.empty:
        cols = st.columns(len(index_data))
        for i, col in enumerate(cols):
            with col:
                change = index_data.iloc[i]['Change']
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{index_data.iloc[i]['Ticker']}</h4>
                    <h2>{index_data.iloc[i]['Price']:,.2f}</h2>
                    <p style="color: {'#28a745' if change > 0 else '#FF4B4B'}; margin: 0;">{change:,.2f} ({index_data.iloc[i]['% Change']:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])
        with tab1:
            if 'watchlists' not in st.session_state: st.session_state.watchlists = {"Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}]}
            if 'active_watchlist' not in st.session_state: st.session_state.active_watchlist = "Watchlist 1"
            st.session_state.active_watchlist = st.radio("Select Watchlist", options=st.session_state.watchlists.keys(), horizontal=True, label_visibility="collapsed")
            # ... Rest of watchlist UI
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50')
        if nifty_token:
            nifty_data = get_historical_data_raw(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart.")

def page_pulse():
    # ... Full code for this page ...
    pass

# ... And so on for ALL other page functions ...
# page_ai_discovery(), page_advanced_charting(), page_basket_orders(), etc.

# ================ 6. MAIN APP LOGIC AND AUTHENTICATION ============

def show_login_animation():
    """Shows a purely visual, fast boot-up animation."""
    st.title("BlockVista Terminal")
    progress_bar = st.progress(0)
    status_text = st.empty()
    steps = {"Initializing terminal...": 50, "Connecting to data feeds...": 100}
    for text, progress in steps.items():
        status_text.text(f"STATUS: {text}")
        progress_bar.progress(progress)
        a_time.sleep(0.4)
    st.session_state['login_animation_complete'] = True
    st.rerun()

def login_page():
    """Displays the login page."""
    from kiteconnect import KiteConnect
    st.title("BlockVista Terminal")
    st.subheader("Broker Login")
    broker = st.selectbox("Select Your Broker", ["Zerodha"])
    if broker == "Zerodha":
        try:
            api_key = st.secrets["ZERODHA_API_KEY"]
            api_secret = st.secrets["ZERODHA_API_SECRET"]
        except (FileNotFoundError, KeyError):
            st.error("Kite API credentials not found in st.secrets.")
            st.stop()
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                st.session_state.kite = kite
                st.session_state.broker = "Zerodha"
                st.session_state.profile = kite.profile()
                st.session_state.login_successful = True
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())

def load_heavy_data_in_background():
    """Fetches and stores the large instrument list."""
    try:
        st.session_state.instrument_df = pd.DataFrame(st.session_state.kite.instruments())
    except Exception as e:
        st.error(f"Failed to load instrument data: {e}")
    st.session_state.heavy_data_loaded = True

def main_app():
    """The main application interface."""
    from streamlit_autorefresh import st_autorefresh
    st.markdown(f'<body class="{"light-theme" if st.session_state.get("theme") == "Light" else ""}"></body>', unsafe_allow_html=True)
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    if not st.session_state.get('heavy_data_loaded'):
        with st.sidebar:
            with st.spinner("Loading instruments..."):
                load_heavy_data_in_background()
        st.rerun()
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Intraday", "Options"], horizontal=True)
    st.sidebar.divider()
    
    st.sidebar.header("Live Data")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=5, disabled=not auto_refresh)
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Intraday": {
            "Dashboard": page_dashboard, "Pre-Market Pulse": page_pulse, "AI Discovery": page_ai_discovery,
            "Advanced Charting": page_advanced_charting, "Basket Orders": page_basket_orders,
            "Portfolio Analytics": page_portfolio_analytics, "Alpha Engine": page_alpha_engine, 
            "Portfolio & Risk": page_portfolio_and_risk, "Forecasting & ML": page_forecasting_ml, 
            "AI Assistant": page_ai_assistant, "Journal Assistant": page_journal_assistant,
        },
        "Options": {
            "Options Hub": page_options_hub, "Strategy Builder": page_option_strategy_builder,
            "Portfolio & Risk": page_portfolio_and_risk, "AI Assistant": page_ai_assistant
        }
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["Forecasting & ML", "AI Assistant"]:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

if __name__ == "__main__":
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'order_history' not in st.session_state: st.session_state.order_history = []

    if st.session_state.get('login_successful'):
        if st.session_state.get('login_animation_complete'):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
