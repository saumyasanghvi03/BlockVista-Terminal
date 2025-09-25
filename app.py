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
from email.utils import mktime_tz, parsedate_tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
import time as a_time # Renaming to avoid conflict with datetime.time
import re
import yfinance as yf
import pyotp
import io
import requests

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

# --- UI ENHANCEMENT: Load Custom CSS for Trader UI ---
def load_css(file_name):
    """Loads a custom CSS file to style the Streamlit app."""
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
        "exchange": "NSE"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "tradingsymbol": "BANK NIFTY", # --- FIX: Corrected trading symbol for Bank Nifty
        "exchange": "NSE"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "tradingsymbol": "FINNIFTY", # --- FIX: Corrected trading symbol for Fin Nifty
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


# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None, exchange=None):
    """
    A quick trade dialog for placing market or limit orders.
    """
    instrument_df = get_instrument_df()
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    
    if symbol is None:
        symbol = st.text_input("Symbol").upper()
    
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "NRML"], horizontal=True, key="diag_prod_type") # --- FIX: Changed CNC to NRML for F&O/MIS
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
    """NSE holidays (update yearly). A more robust solution would use an API or a library."""
    holidays_by_year = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
        2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    }
    return holidays_by_year.get(year, [])

def get_market_status():
    """Checks if the Indian stock market is open."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if market_open_time <= now.time() <= market_close_time:
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
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

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
    """Generates a Plotly chart with various chart types and overlays."""
    fig = go.Figure()
    if df.empty: return fig
    chart_df = df.copy()
    chart_df.columns = [col.lower() for col in chart_df.columns]
    
    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Bar'))
    else:
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
        
    bbl_col = next((col for col in chart_df.columns if 'bbl' in col.lower()), None)
    bbu_col = next((col for col in chart_df.columns if 'bbu' in col.lower()), None)
    if bbl_col and bbu_col:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbl_col], line=dict(color='rgba(135,206,250,0.5)', width=1), name='Lower Band'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbu_col], line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='Upper Band'))
        
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    client = get_broker_client()
    if not client: return pd.DataFrame()
    if st.session_state.broker == "Zerodha":
        return pd.DataFrame(client.instruments())
    else:
        st.warning(f"Instrument list for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty or not symbol: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """
    Fetches historical data from the broker's API.
    """
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
    if st.session_state.broker == "Zerodha":
        if not to_date: to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty: return df
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            # Apply all technical indicators
            try:
                df.ta.strategy("all") # --- FIX: Use a more robust way to add all indicators
            except Exception as e:
                st.toast(f"Could not calculate some indicators: {e}", icon="⚠️")
            return df
        except Exception as e:
            st.error(f"Kite API Error (Historical): {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Historical data for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    if st.session_state.broker == "Zerodha":
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
                    watchlist.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': last_price, 'Change': change, '% Change': pct_change})
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
    else:
        st.warning(f"Watchlist for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """
    Fetches and processes the options chain for a given underlying.
    """
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
    if st.session_state.broker == "Zerodha":
        exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "NATURALGAS": "MCX", "USDINR": "CDS"}
        exchange = exchange_map.get(underlying, 'NFO')
        
        # --- FIX: Simplify LTP fetching logic
        ltp_symbol = "NIFTY 50" if underlying == "NIFTY" else "NIFTY BANK" if underlying == "BANKNIFTY" else underlying
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"

        try:
            underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
        except Exception:
            underlying_ltp = 0.0
            
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty: return pd.DataFrame(), None, underlying_ltp, []
        
        expiries = sorted(pd.to_datetime(options['expiry'].unique()))
        three_months_later = datetime.now() + timedelta(days=90)
        available_expiries = [e for e in expiries if datetime.now().date() <= e.date() <= three_months_later.date()]
        if not available_expiries: return pd.DataFrame(), None, underlying_ltp, []
        if not expiry_date: expiry_date = available_expiries[0]
        
        chain_df = options[options['expiry'] == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        
        try:
            quotes = client.quote(instruments_to_fetch)
            
            def get_quote_value(symbol, key):
                return quotes.get(f"{exchange}:{symbol}", {}).get(key, 0)

            for df, suffix in [(ce_df, '_CE'), (pe_df, '_PE')]:
                df['LTP'] = df['tradingsymbol'].apply(get_quote_value, args=('last_price',))
                df[f'open_interest{suffix}'] = df['tradingsymbol'].apply(get_quote_value, args=('oi',))

            final_chain = pd.merge(
                ce_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_CE']], 
                pe_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_PE']], 
                on='strike', suffixes=('_CE', '_PE'), how='outer'
            ).rename(columns={
                'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 
                'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'
            }).fillna(0)
            
            return final_chain[['CALL', 'CALL LTP', 'open_interest_CE', 'STRIKE', 'PUT LTP', 'open_interest_PE', 'PUT']], expiry_date, underlying_ltp, available_expiries
        except Exception as e:
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio positions and holdings from the broker."""
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    if st.session_state.broker == "Zerodha":
        try:
            positions = client.positions().get('net', [])
            holdings = client.holdings()
            
            positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
            total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
            
            holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
            total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
            
            return positions_df, holdings_df, total_pnl, total_investment
        except Exception as e:
            st.error(f"Kite API Error (Portfolio): {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    else:
        st.warning(f"Portfolio for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order through the broker's API."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    if st.session_state.broker == "Zerodha":
        try:
            # --- FIX: More robust exchange lookup
            instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
            if instrument.empty:
                st.error(f"Symbol '{symbol}' not found in instrument list.")
                return
            exchange = instrument.iloc[0]['exchange']
            
            order_params = {
                "tradingsymbol": symbol.upper(),
                "exchange": exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "variety": client.VARIETY_REGULAR
            }
            if order_type == 'LIMIT' and price is not None:
                order_params["price"] = price

            order_id = client.place_order(**order_params)
            st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
            st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
        except Exception as e:
            st.toast(f"❌ Order failed: {e}", icon="🔥")
            st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})
    else:
        st.warning(f"Order placement for {st.session_state.broker} not implemented.")

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
    analyzer = SentimentIntensityAnalyzer()
    
    news_sources = {
        "Reuters": "http://feeds.reuters.com/reuters/businessNews",
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
        "Livemint": "https://www.livemint.com/rss/markets"
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_date_tuple = entry.get('published_parsed') or entry.get('updated_parsed')
                published_date = datetime.fromtimestamp(mktime_tz(published_date_tuple)) if published_date_tuple else datetime.now()

                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception:
            continue
    return pd.DataFrame(all_news)

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
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors='coerce', dayfirst=True)
        hist_df.dropna(subset=['Date'], inplace=True)
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
            if not live_df.empty: live_df.columns = [col.lower() for col in live_df.columns]
    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max", auto_adjust=True)
            if not live_df.empty: live_df.columns = [col.lower() for col in live_df.columns]
        except Exception as e:
            st.error(f"Failed to load yfinance data: {e}")
            live_df = pd.DataFrame()
            
    if not live_df.empty:
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
    else: # Put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError, ValueError):
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty: return {}
    latest = df.iloc[-1].copy()
    latest.index = latest.index.str.lower()
    interpretation = {}
    
    r = latest.get('rsi_14')
    if r is not None:
        interpretation['RSI (14)'] = "Overbought (Bearish)" if r > 70 else "Oversold (Bullish)" if r < 30 else "Neutral"
    
    stoch_k = latest.get('stok_14_3_3')
    if stoch_k is not None:
        interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
    
    macd = latest.get('macd_12_26_9')
    signal = latest.get('macds_12_26_9')
    if macd is not None and signal is not None:
        interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    
    adx = latest.get('adx_14')
    if adx is not None:
        interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
    
    return interpretation

# ================ 4. HNI & PRO TRADER FEATURES ================

def place_basket_order(orders):
    """Places a basket of orders by iterating and placing them individually."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return

    # --- FIX: KiteConnect V3 does not support basket orders in a single call.
    # We must loop through the orders and place them one by one.
    for order in orders:
        try:
            # Re-using the single place_order function for each item in the basket
            place_order(
                instrument_df=get_instrument_df(), # Ensure the function has access to the instrument list
                symbol=order['tradingsymbol'],
                quantity=order['quantity'],
                order_type=order['order_type'],
                transaction_type=order['transaction_type'],
                product=order['product'],
                price=order.get('price') # Use .get() for optional price
            )
            a_time.sleep(0.3) # Small delay to avoid hitting rate limits
        except Exception as e:
            st.error(f"Failed to place order for {order['tradingsymbol']}: {e}")

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
        # --- FIX: Assuming the file is in the same directory.
        return pd.read_csv("ind_nifty500list.csv")
    except FileNotFoundError:
        st.warning("Sector data file 'ind_nifty500list.csv' not found. Sector analysis will not be available.")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to the options chain dataframe."""
    if df.empty or 'STRIKE' not in df.columns or ltp == 0:
        return df
    
    # --- FIX: Corrected and simplified styling logic.
    def highlight_itm(row):
        style = ''
        if row.STRIKE < ltp: # ITM Calls
            style += 'background-color: #2E2249;'
        return [style] * len(row)
        
    def highlight_itm_puts(row):
        style = ''
        if row.STRIKE > ltp: # ITM Puts
            style += 'background-color: #2E2249;'
        return [style] * len(row)

    df_styled = df.style.apply(highlight_itm, axis=1, subset=['CALL', 'CALL LTP', 'open_interest_CE'])
    df_styled = df_styled.apply(highlight_itm_puts, axis=1, subset=['PUT', 'PUT LTP', 'open_interest_PE'])
    return df_styled

# ================ 5. PAGE DEFINITIONS ============

def page_dashboard():
    """--- UI ENHANCEMENT: A completely redesigned 'Trader UI' Dashboard ---"""
    display_header()
    instrument_df = get_instrument_df()
    client = get_broker_client()
    if not client or instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return

    # --- Top Row: Key Market Metrics ---
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    if not index_data.empty:
        cols = st.columns(len(index_data))
        for i, col in enumerate(cols):
            with col:
                change = index_data.iloc[i]['Change']
                st.metric(
                    label=index_data.iloc[i]['Ticker'],
                    value=f"{index_data.iloc[i]['Price']:,.2f}",
                    delta=f"{change:,.2f} ({index_data.iloc[i]['% Change']:.2f}%)"
                )
    
    st.markdown("---")
    
    # --- Middle Row: Main Content Area ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            if 'watchlists' not in st.session_state:
                st.session_state.watchlists = {
                    "My Watchlist": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
                    "Indices": [{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'NIFTY BANK', 'exchange': 'NSE'}],
                }
            if 'active_watchlist' not in st.session_state:
                st.session_state.active_watchlist = "My Watchlist"

            st.session_state.active_watchlist = st.radio(
                "Select Watchlist", options=st.session_state.watchlists.keys(),
                horizontal=True, label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "NFO", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 50:
                            st.toast("Watchlist full (max 50 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange})
                            st.rerun()
                        else:
                            st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                st.dataframe(watchlist_data, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("My Portfolio")
            positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
            current_value = (holdings_df['quantity'] * holdings_df['last_price']).sum() if not holdings_df.empty else 0
            
            p1, p2, p3 = st.columns(3)
            p1.metric("Investment", f"₹{total_investment:,.2f}")
            p2.metric("Current Value", f"₹{current_value:,.2f}")
            p3.metric("Overall P&L", f"₹{holdings_df['pnl'].sum():,.2f}")
            st.metric("Today's P&L (Intraday)", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            
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
                st.plotly_chart(create_chart(nifty_data.tail(400), "NIFTY 50"), use_container_width=True) # --- FIX: Increased data points
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
        
def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    with st.container():
        st.subheader("Global Chart Controls")
        global_cols = st.columns(4)
        global_ticker = global_cols[0].text_input("Symbol", "NIFTY 50", key="global_ticker").upper()
        global_period = global_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key="global_period")
        global_interval = global_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key="global_interval")
        global_chart_type = global_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key="global_chart_type")

        # --- FIX: Find exchange for the given ticker
        instrument = instrument_df[instrument_df['tradingsymbol'] == global_ticker]
        if not instrument.empty:
            exchange = instrument.iloc[0]['exchange']
            global_token = get_instrument_token(global_ticker, instrument_df, exchange)
            global_data = get_historical_data(global_token, global_interval, period=global_period)
        else:
            global_data = pd.DataFrame()
            st.error(f"Symbol '{global_ticker}' not found.")

    st.markdown("---")
    
    # --- FIX: Only show 2 charts for a cleaner UI ---
    chart_columns = st.columns(2, gap="large")
    
    for i in range(2):
        with chart_columns[i % 2]:
            if global_data.empty:
                st.warning(f"No data to display for {global_ticker} with selected parameters.")
            else:
                st.plotly_chart(create_chart(global_data, global_ticker, global_chart_type), use_container_width=True)

def page_alpha_engine():
    """Analyzes market sentiment from live news headlines and finds trading opportunities."""
    display_header()
    st.title("Alpha Engine")

    tab1, tab2 = st.tabs(["News Sentiment", "AI Discovery"])
    
    with tab1:
        st.subheader("News Sentiment")
        query = st.text_input("Enter a stock, commodity, or currency to analyze", "NIFTY")
        
        with st.spinner("Fetching and analyzing news..."):
            news_df = fetch_and_analyze_news(query)
            if not news_df.empty:
                avg_sentiment = news_df['sentiment'].mean()
                sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                st.metric(f"Overall News Sentiment for '{query}'", sentiment_label, f"{avg_sentiment:.3f}")
                st.dataframe(news_df.drop(columns=['date']), use_container_width=True, hide_index=True, column_config={"link": st.column_config.LinkColumn("Link", display_text="Read Article")})
            else:
                st.info(f"No recent news found for '{query}'.")

    with tab2:
        st.subheader("AI Discovery Engine")
        st.info("This engine discovers technical patterns based on your active watchlist.", icon="🧠")
        
        active_list = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist', 'My Watchlist'), [])
        instrument_df = get_instrument_df()

        if not active_list or instrument_df.empty:
            st.warning("Please set up your watchlist on the Dashboard page to enable AI Discovery.")
        else:
            st.subheader("Automated Pattern Discovery")
            
            with st.spinner("Analyzing data..."):
                discovery_results = {}
                for item in active_list:
                    ticker = item['symbol']
                    token = get_instrument_token(ticker, instrument_df, exchange=item['exchange'])
                    if token:
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            interpretation = interpret_indicators(data)
                            signals = [f"{k}: {v}" for k, v in interpretation.items() if "Neutral" not in v and "Trend" not in v]
                            if signals:
                                discovery_results[ticker] = signals
            
            if discovery_results:
                for ticker, signals in discovery_results.items():
                    st.markdown(f"**Potential Signals for {ticker}:**")
                    for signal in signals:
                        st.markdown(f"- {signal}")
            else:
                st.info("No significant technical patterns found in the last 6 months for your watchlist.")
                
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
        st.info("No holdings or positions found to analyze. Please check your portfolio.")
        return

    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Analytics & Allocation"])
    
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}", delta_color='normal' if total_pnl >= 0 else 'inverse')
        else:
            st.info("No open positions for the day.")
    
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty:
             st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        else:
            st.info("No investment holdings found.")

    with tab3:
        st.subheader("Portfolio Analytics")
        if holdings_df.empty:
            st.info("No holdings to analyze.")
            return

        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
            # --- FIX: Renamed sector file column from 'Industry' to 'Sector' for consistency
            if 'Industry' in sector_df.columns:
                sector_df.rename(columns={'Industry': 'Sector'}, inplace=True)
            holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left')
            holdings_df['Sector'].fillna('Uncategorized', inplace=True)
        else:
            holdings_df['Sector'] = 'Uncategorized'
        
        st.metric("Total Portfolio Value", f"₹{holdings_df['current_value'].sum():,.2f}")

        col1_alloc, col2_alloc = st.columns(2)
        
        with col1_alloc:
            st.subheader("Stock-wise Allocation")
            fig_stock = go.Figure(data=[go.Pie(
                labels=holdings_df['tradingsymbol'],
                values=holdings_df['current_value'],
                hole=.3, textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        with col2_alloc:
            if 'Sector' in holdings_df.columns and holdings_df['Sector'].nunique() > 1:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3, textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train a Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        with st.spinner(f"Loading data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            st.stop()
        
        if st.button("Train Seasonal ARIMA Model & Forecast"):
            with st.spinner("Training Seasonal ARIMA model... This may take a moment."):
                # --- FIX: Removed redundant model training logic. It's all in the function now.
                try:
                    decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                    seasonally_adjusted = data['close'] - decomposed.seasonal
                    model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                    
                    forecast_steps = 30
                    forecast_adjusted = model.forecast(steps=forecast_steps)
                    
                    last_season_cycle = decomposed.seasonal.iloc[-7:]
                    future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                    future_seasonal.index = forecast_adjusted.index
                    forecast_final = forecast_adjusted + future_seasonal
                    
                    fitted_values = model.fittedvalues + decomposed.seasonal.reindex(model.fittedvalues.index)
                    backtest_df = pd.DataFrame({'Actual': data['close'], 'Predicted': fitted_values}).dropna()
                    
                    st.session_state.update({
                        'ml_forecast': forecast_final.to_frame(name='Predicted Price'),
                        'ml_backtest_df': backtest_df,
                        'ml_instrument_name': instrument_name,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Seasonal ARIMA model training failed: {e}")
                    st.session_state.update({'ml_forecast': None})

    with col2:
        if st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast Results for {instrument_name}")
            
            forecast_df = st.session_state.get('ml_forecast')
            if forecast_df is not None:
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest)")
            backtest_df = st.session_state.get('ml_backtest_df')
            if backtest_df is not None and not backtest_df.empty:
                # --- FIX: Corrected MAPE calculation and added metrics
                mape = mean_absolute_percentage_error(backtest_df['Actual'], backtest_df['Predicted']) * 100
                accuracy = 100 - mape

                c1, c2 = st.columns(2)
                c1.metric("Backtest Accuracy", f"{accuracy:.2f}%")
                c2.metric("Mean Absolute Pct Error", f"{mape:.2f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual'], mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Predicted (In-sample)', line=dict(dash='dash')))
                template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                fig.update_layout(title="Backtest Results", yaxis_title='Price (INR)', template=template)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate performance metrics.")
        else:
            st.subheader(f"Historical Data for {instrument_name}")
            st.plotly_chart(create_chart(data.tail(252), instrument_name), use_container_width=True)

def page_ai_assistant():
    """An AI-powered assistant for portfolio management and market queries."""
    display_header()
    st.title("Portfolio-Aware Assistant")
    instrument_df = get_instrument_df()

    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your portfolio or the markets today?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_lower = prompt.lower()
                response = "I can help with your portfolio, orders, and live market data. For example, try asking 'What are my positions?' or 'Show me the technical analysis for RELIANCE'."
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
                elif "technical analysis for" in prompt_lower:
                    ticker = prompt.split("for")[-1].strip().upper()
                    instrument = instrument_df[instrument_df['tradingsymbol'] == ticker]
                    if not instrument.empty:
                        token = instrument.iloc[0]['instrument_token']
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            analysis = interpret_indicators(data)
                            response = f"**Technical Analysis for {ticker}:**\n\n" + "\n".join([f"- **{k}:** {v}" for k, v in analysis.items()])
                        else:
                            response = f"Could not retrieve enough data for {ticker} to perform analysis."
                    else:
                        response = f"Could not find the ticker '{ticker}'."
                
                # --- FIX: More robust regex for options and better error handling
                elif "greeks for" in prompt_lower or "iv for" in prompt_lower:
                    try:
                        option_symbol_match = re.search(r'([A-Z&]+)(\d{2}[A-Z]{3}\d+[CP]E)', prompt.upper())
                        if option_symbol_match:
                            option_symbol = "".join(option_symbol_match.groups())
                            option_details_row = instrument_df[instrument_df['tradingsymbol'] == option_symbol]
                            if not option_details_row.empty:
                                option_details = option_details_row.iloc[0]
                                _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, option_details['expiry'])
                                
                                ltp = client.ltp(f"NFO:{option_symbol}")[f"NFO:{option_symbol}"]['last_price']
                                T = max((option_details['expiry'].date() - datetime.now().date()).days, 0) / 365.0
                                
                                iv = implied_volatility(underlying_ltp, option_details['strike'], T, 0.07, ltp, option_details['instrument_type'].lower())
                                
                                if not np.isnan(iv):
                                    greeks = black_scholes(underlying_ltp, option_details['strike'], T, 0.07, iv, option_details['instrument_type'].lower())
                                    response = f"Calculated Greeks for **{option_symbol}**:\n- **Implied Volatility (IV):** {iv*100:.2f}%\n- **Delta:** {greeks['delta']:.4f}\n- **Gamma:** {greeks['gamma']:.4f}\n- **Vega:** {greeks['vega']:.4f}\n- **Theta:** {greeks['theta']:.4f}\n- **Rho:** {greeks['rho']:.4f}"
                                else:
                                    response = f"Could not calculate IV for {option_symbol}. LTP might be zero or option is illiquid."
                            else:
                                response = f"Option symbol '{option_symbol}' not found."
                        else:
                            response = "Please specify a valid option symbol (e.g., NIFTY25SEPWK123000CE)."
                    except Exception as e:
                        response = f"An error occurred calculating greeks: {e}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def page_basket_orders():
    """A page for creating, managing, and executing basket orders."""
    display_header()
    st.title("Basket Orders")

    if 'basket' not in st.session_state:
        st.session_state.basket = []

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the basket order feature.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Add Order to Basket")
        with st.form("add_to_basket_form", clear_on_submit=True):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "NRML"], horizontal=True) # --- FIX: Changed CNC to NRML
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0

            if st.form_submit_button("Add to Basket"):
                if symbol:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
                    if not instrument.empty:
                        exchange = instrument.iloc[0]['exchange']
                        order = {
                            "tradingsymbol": symbol, "exchange": exchange,
                            "transaction_type": transaction_type, "quantity": quantity,
                            "product": product, "order_type": order_type,
                        }
                        if order_type == "LIMIT": order["price"] = price
                        st.session_state.basket.append(order)
                        st.success(f"Added {symbol} to basket.")
                    else:
                        st.error(f"Symbol '{symbol}' not found.")
                else:
                    st.warning("Please enter a symbol.")

    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty.")
        else:
            basket_df = pd.DataFrame(st.session_state.basket)
            st.dataframe(basket_df, use_container_width=True, hide_index=True)

            if st.button("Execute Basket Order", use_container_width=True, type="primary"):
                with st.spinner("Placing basket order..."):
                    place_basket_order(st.session_state.basket) # --- FIX: Pass the basket list
                st.session_state.basket = []
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_options_chain():
    """An advanced options chain page with Greek calculations and OI analysis."""
    display_header()
    st.title("Advanced Options Chain")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to view the options chain.")
        return

    options_underlyings = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "USDINR"]
    col1, col2 = st.columns(2)
    underlying = col1.selectbox("Select Underlying", options_underlyings)
    
    chain_df, expiry_date, ltp, available_expiries = get_options_chain(underlying, instrument_df)
    
    if available_expiries:
        selected_expiry_str = col2.selectbox("Select Expiry", [d.strftime('%d-%b-%Y') for d in available_expiries])
        selected_expiry = datetime.strptime(selected_expiry_str, '%d-%b-%Y')
        
        if selected_expiry != expiry_date:
            chain_df, expiry_date, ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
    else:
        st.warning(f"No available expiries found for {underlying}.")
        return

    if chain_df.empty:
        st.warning(f"Could not load options chain for {underlying}.")
        return

    total_ce_oi = chain_df['open_interest_CE'].sum()
    total_pe_oi = chain_df['open_interest_PE'].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

    st.metric(f"Spot Price: {underlying}", f"{ltp:,.2f}", f"PCR: {pcr:.2f}")

    st.subheader("Open Interest Profile")
    atm_strike_index = abs(chain_df['STRIKE'] - ltp).idxmin()
    strikes_to_show = chain_df.loc[max(0, atm_strike_index-10):atm_strike_index+10]
    
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=strikes_to_show['STRIKE'], y=strikes_to_show['open_interest_CE'], name='CE OI', marker_color='#FF4B4B'))
    fig_oi.add_trace(go.Bar(x=strikes_to_show['STRIKE'], y=strikes_to_show['open_interest_PE'], name='PE OI', marker_color='#28a745'))
    fig_oi.update_layout(barmode='group', title=f"Open Interest for {underlying} ({expiry_date.strftime('%d-%b-%Y')})", template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white', yaxis_title="Contracts", xaxis_title="Strike Price")
    st.plotly_chart(fig_oi, use_container_width=True)
    
    st.subheader("Options Chain")
    st.dataframe(style_option_chain(chain_df, ltp), use_container_width=True, height=600)

def page_market_overview():
    """Provides a high-level overview of Indian and Global markets."""
    display_header()
    st.title("Market Overview")

    tab1, tab2 = st.tabs(["Indian Markets", "Global Markets"])

    with tab1:
        st.subheader("Indian Indices")
        indian_indices = [
            {'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'NIFTY BANK', 'exchange': 'NSE'},
            {'symbol': 'SENSEX', 'exchange': 'BSE'}, {'symbol': 'INDIA VIX', 'exchange': 'NSE'}
        ]
        indian_data = get_watchlist_data(indian_indices)
        if not indian_data.empty:
            st.dataframe(indian_data, hide_index=True, use_container_width=True)
        else:
            st.warning("Could not fetch Indian indices data. Broker may not be connected.")

    with tab2:
        st.subheader("Global Indices")
        global_indices_tickers = ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^N225', '^HSI']
        global_data = get_global_indices_data(global_indices_tickers)
        if not global_data.empty:
            st.dataframe(global_data, hide_index=True, use_container_width=True)
        else:
            st.warning("Could not fetch global indices data from yfinance.")

# ================ 6. MAIN APP LOGIC AND ROUTING ================

def main():
    """The main function that runs the Streamlit app."""
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'kite' not in st.session_state: st.session_state.kite = None
    if 'broker' not in st.session_state: st.session_state.broker = None
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state: st.session_state.refresh_interval = 30

    with st.sidebar:
        st.header("BlockVista Settings")
        st.session_state.theme = st.radio("Theme", ["Light", "Dark"], index=1, horizontal=True)
        st.markdown("---")
        
        if not st.session_state.logged_in:
            st.subheader("Connect to Broker")
            st.session_state.broker = st.selectbox("Select Broker", ["Zerodha"])
            if st.session_state.broker == "Zerodha":
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
                
                # --- FIX: Handle TOTP Login Flow
                query_params = st.query_params
                request_token = query_params.get("request_token")

                if st.button("Login with Kite"):
                    if api_key and api_secret:
                        try:
                            kite = KiteConnect(api_key=api_key)
                            st.session_state.kite_pre = kite
                            st.session_state.api_secret = api_secret
                            login_url = kite.login_url()
                            st.markdown(f"**[Click here to login]({login_url})**")
                            st.info("After logging in, you will be redirected back here. The request token will appear in the URL.")
                        except Exception as e:
                            st.error(f"Could not generate login URL: {e}")
                    else:
                        st.warning("Please enter API Key and Secret.")

                if request_token and 'kite_pre' in st.session_state:
                    try:
                        user_data = st.session_state.kite_pre.generate_session(request_token, api_secret=st.session_state.api_secret)
                        st.session_state.kite = st.session_state.kite_pre
                        st.session_state.kite.set_access_token(user_data["access_token"])
                        st.session_state.logged_in = True
                        st.query_params.clear() # Clear token from URL
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")
        else:
            st.success(f"✅ Connected to {st.session_state.broker}")
            profile = st.session_state.kite.profile()
            st.write(f"Welcome, {profile.get('user_name', 'User')}!")
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    if key not in ['theme']: # Persist theme choice
                        del st.session_state[key]
                st.rerun()

        st.markdown("---")
        
        st.subheader("Navigation")
        pages = {
            "📊 Dashboard": page_dashboard, "📈 Advanced Charting": page_advanced_charting,
            "⛓️ Options Chain": page_options_chain, "🌍 Market Overview": page_market_overview,
            "💡 Alpha Engine": page_alpha_engine, "📦 Basket Orders": page_basket_orders,
            "💼 Portfolio & Risk": page_portfolio_and_risk, "🤖 AI Assistant": page_ai_assistant,
            "🔮 ML Forecasting": page_forecasting_ml,
        }
        page_selection = st.radio("Go to", list(pages.keys()), label_visibility="collapsed")
        st.markdown("---")

        st.subheader("Auto-Refresh")
        st.session_state.auto_refresh = st.toggle("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider("Refresh Interval (s)", 5, 120, st.session_state.refresh_interval)
            st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="data_refresher")
        
        with st.expander("Recent Orders"):
            if st.session_state.order_history:
                for order in st.session_state.order_history[:5]:
                    st.info(f"{order['type']} {order['symbol']} ({order['qty']}) - {order['status']}")
            else:
                st.write("No orders placed in this session.")

    pages[page_selection]()

if __name__ == "__main__":
    main()
