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
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
from time import mktime
import requests
import io
import time as a_time # Renaming to avoid conflict with datetime.time
import re
import yfinance as yf
from nselib import trading_info

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
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NFO"
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
        "tradingsymbol": None,
        "exchange": None
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
        "tradingsymbol": None,
        "exchange": None
    }
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

@st.cache_data(ttl=60)
def get_global_indices_data():
    """Fetches live data for major world indices using yfinance."""
    indices = {
        'S&P 500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW JONES': '^DJI',
        'FTSE 100': '^FTSE', 'DAX': '^GDAXI', 'NIKKEI 225': '^N225'
    }
    data = []
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100
                data.append({'Name': name, 'Price': last_price, 'Change': change, '% Change': pct_change})
        except Exception:
            continue
    return pd.DataFrame(data)

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================
def create_chart(df, ticker, forecast_df=None):
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
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import feedparser
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
def train_auto_arima_model(_data, forecast_horizon):
    if _data.empty or len(_data) < 30: return None, None
    try:
        ts_data = _data['close']
        auto_arima_model = pm.auto_arima(ts_data, seasonal=False, stepwise=True, suppress_warnings=True, trace=False)
        model = ARIMA(ts_data, order=auto_arima_model.order).fit()
        forecast_result = model.get_forecast(steps=forecast_horizon)
        forecast_df = pd.DataFrame({'Predicted': forecast_result.predicted_mean})
        in_sample_preds = model.predict(start=ts_data.index[0], end=ts_data.index[-1])
        backtest_df = pd.DataFrame({'Actual': ts_data, 'Predicted': in_sample_preds}).dropna()
        return forecast_df, backtest_df
    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        return None, None

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

def black_scholes(S, K, T, r, sigma, option_type="call"):
    from scipy.stats import norm
    import numpy as np
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    from scipy.optimize import newton
    import numpy as np
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except: return np.nan

def interpret_indicators(df):
    if df.empty: return {}
    latest = df.iloc[-1].copy(); latest.index = latest.index.str.lower(); interpretation = {}
    rsi = latest.get('rsi_14')
    if rsi is not None: interpretation['RSI (14)'] = "Overbought (Bearish)" if rsi > 70 else "Oversold (Bullish)" if rsi < 30 else "Neutral"
    macd = latest.get('macd_12_26_9'); signal = latest.get('macds_12_26_9')
    if macd is not None and signal is not None: interpretation['MACD'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    return interpretation

def place_basket_order(orders, variety):
    client = get_broker_client()
    if not client: st.error("Broker not connected."); return
    try:
        client.place_order(variety=variety, orders=orders)
        st.toast("✅ Basket order placed successfully!", icon="🎉")
    except Exception as e:
        st.toast(f"❌ Basket order failed: {e}", icon="🔥")

@st.cache_data(ttl=3600)
def get_sector_data():
    try: return pd.read_csv("sectors.csv")
    except: return None

def style_option_chain(df, ltp):
    atm_strike = abs(df['STRIKE'] - ltp).idxmin()
    return df.style.apply(lambda x: ['background-color: #2c3e50' if x.name < atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['CALL', 'CALL LTP']], axis=1)\
                   .apply(lambda x: ['background-color: #2c3e50' if x.name > atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['PUT', 'PUT LTP']], axis=1)

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying):
    st.subheader(f"Most Active {underlying} Options (By Volume)")
    with st.spinner("Fetching data..."):
        active_df = get_most_active_options(underlying)
        if not active_df.empty:
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not retrieve data.")

def get_most_active_options(underlying):
    client = get_broker_client()
    if not client: return pd.DataFrame()
    try:
        chain_df, _, _, _, _ = get_options_chain(underlying)
        if chain_df.empty: return pd.DataFrame()
        symbols = [f"NFO:{s}" for s in pd.concat([chain_df['CALL'], chain_df['PUT']]).dropna() if isinstance(s, str) and s.strip()]
        if not symbols: return pd.DataFrame()
        quotes = client.quote(symbols)
        active_options = []
        for data in quotes.values():
            change = data['last_price'] - data['ohlc']['close']
            pct_change = (change / data['ohlc']['close'] * 100) if data['ohlc']['close'] != 0 else 0
            active_options.append({'Symbol': data['tradingsymbol'], 'LTP': data['last_price'], 'Change %': pct_change, 'Volume': data['volume'], 'OI': data['open_interest']})
        return pd.DataFrame(active_options).sort_values(by='Volume', ascending=False).head(10)
    except: return pd.DataFrame()


# ================ 5. PAGE DEFINITIONS ================

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
    import yfinance as yf
    import plotly.graph_objects as go
    display_header()
    st.title("Pre-Market Pulse")
    st.subheader("Global Market Cues")
    with st.spinner("Fetching live global indices..."):
        # This helper function is defined outside and cached
        indices_df = get_global_indices_data()
        if not indices_df.empty:
            cols = st.columns(len(indices_df))
            for i, col in enumerate(cols):
                with col:
                    row = indices_df.iloc[i]
                    change = row['Change']
                    st.metric(label=row['Name'], value=f"{row['Price']:,.2f}", delta=f"{change:,.2f} ({row['% Change']:.2f}%)")
        else:
            st.warning("Could not fetch global market data.")
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("GIFT Nifty Live Chart")
        try:
            gift_nifty = yf.Ticker("NIFTY=F") 
            nifty_data = gift_nifty.history(period="1d", interval="5m")
            if not nifty_data.empty:
                fig = go.Figure(go.Scatter(x=nifty_data.index, y=nifty_data['Close'], mode='lines', name='GIFT Nifty'))
                template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                fig.update_layout(title="GIFT Nifty Real-time Price", yaxis_title='Price', template=template, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not fetch live GIFT Nifty data.")
        except Exception as e:
            st.error(f"Error fetching GIFT Nifty chart: {e}")
    with col2:
        st.subheader("Top Financial News")
        with st.spinner("Fetching latest headlines..."):
            news_df = fetch_and_analyze_news()
            if not news_df.empty:
                for _, row in news_df.head(10).iterrows():
                    st.markdown(f"**[{row['title']}]({row['link']})** <br> <small>*{row['source']}*</small>", unsafe_allow_html=True)
                    st.divider()
            else:
                st.info("No news articles found.")

def page_ai_discovery():
    display_header()
    st.title("AI Discovery Engine")
    if 'watchlists' not in st.session_state or not st.session_state.watchlists.get(st.session_state.get('active_watchlist')):
        st.info("Please add stocks to your watchlist on the Dashboard page to use this feature.")
        return
    active_watchlist = st.session_state.watchlists.get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
    watchlist_symbols = [item['symbol'] for item in active_watchlist]
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Automated Pattern Scan")
        with st.spinner(f"Scanning {len(watchlist_symbols)} stocks in your watchlist..."):
            all_signals = []
            for symbol in watchlist_symbols:
                token = get_instrument_token(symbol)
                if token:
                    data = get_historical_data_raw(token, 'day', period='6mo')
                    if not data.empty:
                        data_with_indicators = calculate_indicators(data)
                        if 'RSI_14' in data_with_indicators.columns:
                            latest_rsi = data_with_indicators['RSI_14'].iloc[-1]
                            if latest_rsi > 70:
                                all_signals.append({'symbol': symbol, 'signal': 'RSI Overbought', 'type': 'Bearish', 'value': f'RSI: {latest_rsi:.2f}'})
                            elif latest_rsi < 30:
                                all_signals.append({'symbol': symbol, 'signal': 'RSI Oversold', 'type': 'Bullish', 'value': f'RSI: {latest_rsi:.2f}'})
            if all_signals:
                st.success(f"Found {len(all_signals)} potential signal(s)!")
                st.dataframe(pd.DataFrame(all_signals), use_container_width=True, hide_index=True)
                st.session_state['ai_signals'] = all_signals
            else:
                st.info("No strong RSI patterns found in your watchlist.")
                st.session_state['ai_signals'] = []
    with col2:
        st.subheader("Data-Driven Trade of the Day")
        if not st.session_state.get('ai_signals'):
            st.info("Run the pattern scan first to generate a trade suggestion.")
            return
        st.info("AI analysis of signals and news sentiment coming soon.")

def page_advanced_charting():
    display_header()
    st.title("Advanced Charting")
    pass
def page_basket_orders():
    display_header()
    st.title("Basket Orders")
    pass
def page_portfolio_analytics():
    display_header()
    st.title("Portfolio Analytics")
    pass
def page_alpha_engine():
    display_header(); st.title("Alpha Engine: News Sentiment")
    pass
def page_portfolio_and_risk():
    display_header(); st.title("Portfolio & Risk")
    pass
def page_forecasting_ml():
    display_header()
    st.title("Auto-ARIMA Forecasting")
    st.info("Automatically find the best ARIMA model to forecast future prices. This is for educational purposes only.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Forecast Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        horizon_map = {"1 Day": 1, "5 Days": 5, "30 Days": 30, "60 Days": 60, "90 Days": 90}
        forecast_horizon_str = st.selectbox("Select Forecast Duration", list(horizon_map.keys()))
        forecast_horizon = horizon_map[forecast_horizon_str]

        with st.spinner(f"Loading 1 year of data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name).tail(365)
        
        if data.empty or len(data) < 30:
            st.error(f"Could not load sufficient historical data for {instrument_name}.")
            return
            
        if st.button("Generate Forecast"):
            with st.spinner("Finding best ARIMA model and forecasting..."):
                forecast_df, backtest_df = train_auto_arima_model(data, forecast_horizon)
                
                st.session_state.update({
                    'ml_forecast_df': forecast_df, 
                    'ml_backtest_df': backtest_df, 
                    'ml_instrument_name': instrument_name
                })
                st.rerun()

    with col2:
        if 'ml_instrument_name' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast for {instrument_name}")
            
            forecast_df = st.session_state.get('ml_forecast_df')
            if forecast_df is not None and not forecast_df.empty:
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest on Training Data)")
            backtest_df = st.session_state.get('ml_backtest_df')

            if backtest_df is not None and not backtest_df.empty:
                from sklearn.metrics import mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(backtest_df['Actual'], backtest_df['Predicted']) * 100
                accuracy = 100 - mape
                
                cum_returns = (1 + backtest_df['Actual'].pct_change().fillna(0)).cumprod()
                peak = cum_returns.cummax()
                drawdown = (cum_returns - peak) / peak
                max_drawdown = 0 if drawdown.empty else drawdown.min() * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy:.2f}%")
                m2.metric("MAPE", f"{mape:.2f}%")
                m3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Historical Price'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', name='Forecasted Price', line=dict(color='yellow', dash='dash')))
                template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                fig.update_layout(title="Price Forecast", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate performance metrics.")
        else:
            st.subheader(f"Historical Data for {instrument_name}")
            st.plotly_chart(create_chart(data, instrument_name), use_container_width=True)
            
def page_ai_assistant():
    display_header(); st.title("AI Portfolio-Aware Assistant")
    pass
def page_journal_assistant():
    display_header(); st.title("Trading Journal & Focus Assistant")
    pass
def page_options_hub():
    display_header()
    st.title("Options Hub")
    pass
def page_option_strategy_builder():
    display_header()
    st.title("Options Strategy Builder")
    pass

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
