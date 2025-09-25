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
import time as a_time # Renaming to avoid conflict with datetime.time
import re
import yfinance as yf
import pyotp
import qrcode
from PIL import Image
import base64
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

# --- Theme and UI Configuration ---
# This dictionary holds the color palettes for the light and dark themes.
# These themes are applied globally using custom CSS.
THEMES = {
    "light": {
        "primaryColor": "#6C63FF",  # A vibrant purple
        "backgroundColor": "#F4F4F4",  # A soft light gray
        "secondaryBackgroundColor": "#FFFFFF",  # Pure white for cards/elements
        "textColor": "#2C3E50",  # Dark gray text
        "font": "sans serif"
    },
    "dark": {
        "primaryColor": "#6C63FF",  # Same vibrant purple for consistency
        "backgroundColor": "#1A1A1A",  # A deep dark gray
        "secondaryBackgroundColor": "#2C2C2C",  # A slightly lighter dark gray for elements
        "textColor": "#ECF0F1",  # Soft white text
        "font": "sans serif"
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
        
    bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
    bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
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
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """
    Fetches historical data from the broker's API.
    
    Note: For `yfinance`, a different function is used. This function
    is specific to the connected broker (KiteConnect).
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
            # Apply all technical indicators with a single try-except block
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True); df.ta.coppock(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True); df.ta.macd(append=True); df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
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
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY", "FINNIFTY": "FINNIFTY"}.get(underlying, underlying)
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
            ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            
            ce_df['open_interest_CE'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            ce_df['open_interest_CE_change'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('net_change_oi', 0))
            pe_df['open_interest_PE'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            pe_df['open_interest_PE_change'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('net_change_oi', 0))

            final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_CE', 'open_interest_CE_change']], 
                                   pe_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_PE', 'open_interest_PE_change']], 
                                   on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
            
            return final_chain[['CALL', 'CALL LTP', 'open_interest_CE', 'STRIKE', 'PUT LTP', 'open_interest_PE', 'PUT', 'open_interest_CE_change', 'open_interest_PE_change']], expiry_date, underlying_ltp, available_expiries
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
            is_option = any(char.isdigit() for char in symbol)
            if is_option:
                exchange = 'NFO'
            else:
                instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
                if instrument.empty:
                    st.error(f"Symbol '{symbol}' not found.")
                    return
                exchange = instrument.iloc[0]['exchange']

            order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
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
    
    # Updated and expanded news sources
    news_sources = {
        "Reuters": "http://feeds.reuters.com/reuters/businessNews",
        "Bloomberg": "https://www.bloomberg.com/feeds/bpol/markets.xml",
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
        "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
        "Livemint": "https://www.livemint.com/rss/markets"
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_date_tuple = entry.published_parsed if hasattr(entry, 'published_parsed') else entry.updated_parsed
                published_date = datetime.fromtimestamp(mktime_tz(published_date_tuple)) if published_date_tuple else datetime.now()

                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception:
            continue
    return pd.DataFrame(all_news)

def create_features(df, ticker):
    """Creates features for the ML model from a historical DataFrame."""
    df_feat = df.copy()
    df_feat.columns = [col.lower() for col in df_feat.columns]
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['dayofyear'] = df_feat.index.dayofyear
    for lag in range(1, 6):
        df_feat[f'lag_{lag}'] = df_feat['close'].shift(lag)
    df_feat['rolling_mean_7'] = df_feat['close'].rolling(window=7).mean()
    df_feat['rolling_std_7'] = df_feat['close'].rolling(window=7).std()
    
    # Calculate technical indicators and handle potential errors
    for indicator in [ta.rsi, ta.macd, ta.bbands, ta.atr]:
        try:
            indicator(df_feat, append=True)
        except Exception:
            pass # Silently fail if an indicator cannot be computed

    news_df = fetch_and_analyze_news(ticker)
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date'])
        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].mean().to_frame()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        df_feat = df_feat.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        df_feat['sentiment'] = df_feat['sentiment'].fillna(method='ffill')
        df_feat['sentiment_rolling_3d'] = df_feat['sentiment'].rolling(window=3, min_periods=1).mean()
    else:
        df_feat['sentiment'] = 0
        df_feat['sentiment_rolling_3d'] = 0
    df_feat.bfill(inplace=True); df_feat.ffill(inplace=True); df_feat.dropna(inplace=True)
    return df_feat

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data):
    """Trains a Seasonal ARIMA model for time series forecasting."""
    if _data.empty or len(_data) < 100:
        return {}, pd.DataFrame()

    df = _data.copy()
    df.index = pd.to_datetime(df.index)
    
    predictions = {}

    try:
        decomposed = seasonal_decompose(df['close'], model='additive', period=7)
        seasonally_adjusted = df['close'] - decomposed.seasonal

        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
        
        forecast_steps = 30
        forecast_adjusted = model.forecast(steps=forecast_steps)
        
        last_season_cycle = decomposed.seasonal.iloc[-7:]
        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
        future_seasonal.index = forecast_adjusted.index
        
        forecast_final = forecast_adjusted + future_seasonal
        
        predictions["1-Day Close"] = forecast_final.iloc[0]
        predictions["5-Day Close"] = forecast_final.iloc[4]
        predictions["15-Day Close"] = forecast_final.iloc[14]
        predictions["30-Day Close"] = forecast_final.iloc[29]

        fitted_values = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values})
        backtest_df.dropna(inplace=True)

    except Exception as e:
        st.error(f"Seasonal ARIMA model training failed: {e}")
        return {}, pd.DataFrame()

    return predictions, backtest_df

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
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True)
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
        # Use yfinance for non-Indian indices
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max")
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

def place_basket_order(orders, variety):
    """Places a basket of orders."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        try:
            order_responses = client.place_order(variety=variety, orders=orders)
            st.toast("✅ Basket order placed successfully!", icon="🎉")
            
            for i, resp in enumerate(order_responses):
                if resp.get('status') == 'success':
                    order = orders[i]
                    st.session_state.order_history.insert(0, {"id": resp['order_id'], "symbol": order['tradingsymbol'], "qty": order['quantity'], "type": order['transaction_type'], "status": "Success"})
        except Exception as e:
            st.toast(f"❌ Basket order failed: {e}", icon="🔥")
            
@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("Sector data file 'sensex_sectors.csv' not found. Sector analysis will not be available.")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to the options chain dataframe."""
    if df.empty or 'STRIKE' not in df.columns:
        return df
    atm_strike_index = abs(df['STRIKE'] - ltp).idxmin()
    atm_strike_value = df.loc[atm_strike_index, 'STRIKE']
    
    df_styled = df.style.apply(lambda x: ['background-color: #2c3e50' if x['STRIKE'] < atm_strike_value else '' for i in x], axis=1, subset=pd.IndexSlice[:, ['CALL', 'CALL LTP', 'open_interest_CE']])\
                     .apply(lambda x: ['background-color: #2c3e50' if x['STRIKE'] > atm_strike_value else '' for i in x], axis=1, subset=pd.IndexSlice[:, ['PUT', 'PUT LTP', 'open_interest_PE']])
    return df_styled

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    st.subheader(f"Most Active {underlying} Options (By Volume)")
    with st.spinner("Fetching data..."):
        active_df = get_most_active_options(underlying, instrument_df)
        if not active_df.empty:
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not retrieve data for most active options.")

def get_most_active_options(underlying, instrument_df):
    """Fetches the most active options by volume for a given underlying."""
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
                'OI': data.get('open_interest', 0)
            })
        
        df = pd.DataFrame(active_options)
        df_sorted = df.sort_values(by='Volume', ascending=False)
        return df_sorted.head(10)

    except Exception as e:
        st.error(f"Could not fetch most active options: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        df = yf.download(tickers, period="2d")
        if df.empty:
            return pd.DataFrame()

        close_data = df['Close'].iloc[-1]
        prev_close_data = df['Close'].iloc[-2]
    except Exception as e:
        st.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()
    
    data = []
    if isinstance(close_data, pd.Series): # Handle multiple tickers
        for ticker in tickers:
            last_price = close_data.get(ticker, None)
            prev_close = prev_close_data.get(ticker, None)

            if last_price is not None and prev_close is not None:
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                data.append({'Ticker': ticker, 'Price': last_price, 'Change': change, '% Change': pct_change})
    else: # Handle single ticker case
        ticker = tickers[0]
        last_price = close_data
        prev_close = prev_close_data
        if last_price is not None and prev_close is not None:
            change = last_price - prev_close
            pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
            data.append({'Ticker': ticker, 'Price': last_price, 'Change': change, '% Change': pct_change})
        
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def get_indian_indices_data(symbols_with_exchange):
    """Fetches real-time data for Indian indices using the broker's API."""
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()
    
    instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
    try:
        quotes = client.quote(instrument_names)
        indices_data = []
        for item in symbols_with_exchange:
            instrument = f"{item['exchange']}:{item['symbol']}"
            if instrument in quotes:
                quote = quotes[instrument]
                last_price = quote['last_price']
                prev_close = quote['ohlc']['close']
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                indices_data.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': last_price, 'Change': change, '% Change': pct_change})
        return pd.DataFrame(indices_data)
    except Exception as e:
        st.toast(f"Error fetching Indian indices data: {e}", icon="⚠️")
        return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS ============

def page_dashboard():
    """--- UI ENHANCEMENT: A completely redesigned 'Trader UI' Dashboard ---"""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return

    # --- Top Row: Key Market Metrics ---
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    cols = st.columns(len(index_data))
    for i, col in enumerate(cols):
        with col:
            change = index_data.iloc[i]['Change']
            blink_class = "positive-blink" if change > 0 else "negative-blink" if change < 0 else ""
            st.markdown(f"""
            <div class="metric-card {blink_class}">
                <h4>{index_data.iloc[i]['Ticker']}</h4>
                <h2>{index_data.iloc[i]['Price']:,.2f}</h2>
                <p style="color: {'#28a745' if change > 0 else '#FF4B4B'}; margin: 0;">
                    {change:,.2f} ({index_data.iloc[i]['% Change']:.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- Middle Row: Main Content Area ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            # Initialize watchlists in session state
            if 'watchlists' not in st.session_state:
                st.session_state.watchlists = {
                    "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
                    "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
                    "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
                }
            if 'active_watchlist' not in st.session_state:
                st.session_state.active_watchlist = "Watchlist 1"

            # Watchlist selector
            st.session_state.active_watchlist = st.radio(
                "Select Watchlist",
                options=st.session_state.watchlists.keys(),
                horizontal=True,
                label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            # Add symbol form
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
            
            # Remove symbol dropdown
            if active_list:
                with st.form(key="remove_stock_form"):
                    rm_col1, rm_col2 = st.columns([3, 1])
                    symbol_to_remove = rm_col1.selectbox("Remove Symbol", [item['symbol'] for item in active_list], label_visibility="collapsed")
                    if rm_col2.form_submit_button("Remove"):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != symbol_to_remove]
                        st.rerun()

            # Display watchlist
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = '#28a745' if row['Change'] > 0 else '#FF4B4B'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:gray;'>{row['Exchange']}</small>", unsafe_allow_html=True)
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
    st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    # Global controls for all 4 charts
    with st.container():
        st.subheader("Global Chart Controls")
        global_cols = st.columns(4)
        global_ticker = global_cols[0].text_input("Symbol", "NIFTY 50", key="global_ticker").upper()
        global_period = global_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key="global_period")
        global_interval = global_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key="global_interval")
        global_chart_type = global_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key="global_chart_type")

        global_token = get_instrument_token(global_ticker, instrument_df)
        global_data = get_historical_data(global_token, global_interval, period=global_period)

    st.markdown("---")
    
    chart_columns = st.columns(2, gap="large")
    
    for i in range(4):
        with chart_columns[i % 2]:
            st.subheader(f"Chart {i+1}")
            if global_data.empty:
                st.warning(f"No data to display for {global_ticker} with selected parameters.")
            else:
                st.plotly_chart(create_chart(global_data, global_ticker, global_chart_type), use_container_width=True, key=f"chart_{i}")

                order_cols = st.columns(5)
                order_cols[0].markdown("Quick Order:")
                quantity = order_cols[1].number_input("Qty", min_value=1, step=1, key=f"qty_{i}", label_visibility="collapsed")
                
                if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True, type="primary"):
                    place_order(instrument_df, global_ticker, quantity, 'MARKET', 'BUY', 'MIS')
                if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True, type="secondary"):
                    place_order(instrument_df, global_ticker, quantity, 'MARKET', 'SELL', 'MIS')
    
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
        st.info("This engine simulates advanced AI analysis by discovering technical patterns and suggesting high-conviction trade setups based on your active watchlist. The suggestions are for informational purposes only.", icon="🧠")
        
        active_list = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
        instrument_df = get_instrument_df()

        if not active_list or instrument_df.empty:
            st.warning("Please set up your watchlist on the Dashboard page to enable AI Discovery.")
        else:
            st.markdown("---")
            st.subheader("Automated Pattern Discovery")
            st.markdown("Scanning your watchlist for potential technical signals...")
            
            with st.spinner("Analyzing data..."):
                discovery_results = {}
                for item in active_list:
                    ticker = item['symbol']
                    token = get_instrument_token(ticker, instrument_df, exchange=item['exchange'])
                    if token:
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            interpretation = interpret_indicators(data)
                            signals = [f"{k}: {v}" for k, v in interpretation.items() if v in ["Overbought (Bearish)", "Oversold (Bullish)", "Bullish Crossover", "Bearish Crossover"]]
                            if signals:
                                discovery_results[ticker] = signals
            
            if discovery_results:
                for ticker, signals in discovery_results.items():
                    st.markdown(f"**Potential Signals for {ticker}:**")
                    for signal in signals:
                        st.markdown(f"- {signal}")
            else:
                st.info("No significant technical patterns found in the last 6 months for your watchlist.")
                
            st.markdown("---")
            
            st.subheader("AI-Powered Trade Idea")
            st.warning("This is a simulated trade idea for educational purposes. It does not constitute financial advice.")
            
            if active_list:
                selected_ticker = active_list[0]['symbol']
                ltp_data = get_watchlist_data([active_list[0]])
                if not ltp_data.empty:
                    ltp = ltp_data.iloc[0]['Price']
                    
                    trade_setup = {
                        "title": f"High-Conviction Long Setup: {selected_ticker}",
                        "conviction": "High",
                        "score": 8.5,
                        "entry_range": [ltp * 0.99, ltp * 1.01],
                        "target": ltp * 1.05,
                        "stop_loss": ltp * 0.98,
                        "narrative": f"**{selected_ticker}** is showing a strong confluence of bullish signals, including a recent RSI crossover from the oversold region and a positive MACD divergence. A breakout above the 20-day EMA could confirm a move towards the target price."
                    }
                    
                    trade_idea_col = st.columns([1, 1, 1])
                    trade_idea_col[0].metric("Conviction Score", trade_setup['score'])
                    trade_idea_col[1].metric("Entry Range", f"₹{trade_setup['entry_range'][0]:.2f} - ₹{trade_setup['entry_range'][1]:.2f}")
                    trade_idea_col[2].metric("Target Price", f"₹{trade_setup['target']:.2f}")

                    st.markdown(f"""
                    <div class="trade-card">
                        <h4>{trade_setup['title']}</h4>
                        <p><strong>Narrative:</strong> {trade_setup['narrative']}</p>
                        <p style='color:#FF4B4B;'><strong>Stop Loss:</strong> ₹{trade_setup['stop_loss']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Could not retrieve live price for the selected ticker.")
            else:
                st.info("Please add stocks to your watchlist to generate trade ideas.")

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
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
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Portfolio Analytics")
        
        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
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
                hole=.3,
                textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        if sector_df is not None:
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train an advanced Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        with st.spinner(f"Loading real-time data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            st.stop()
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=30)
        forecast_date = st.date_input("Select a date to forecast", value=today, min_value=today, max_value=max_forecast_date)

        if st.button("Train Seasonal ARIMA Model & Forecast"):
            forecast_steps = (forecast_date - today).days + 1
            if forecast_steps <= 0:
                st.warning("Please select a future date to forecast.")
            else:
                with st.spinner("Training Seasonal ARIMA model... This may take a moment."):
                    predictions, backtest_df = train_seasonal_arima_model(data)
                    
                    try:
                        decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                        seasonally_adjusted = data['close'] - decomposed.seasonal
                        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                        forecast_adjusted = model.forecast(steps=forecast_steps)
                        
                        last_season_cycle = decomposed.seasonal.iloc[-7:]
                        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                        future_seasonal.index = forecast_adjusted.index
                        
                        forecast_final = forecast_adjusted + future_seasonal
                        
                        st.session_state.update({
                            'ml_predictions_by_date': forecast_final.to_frame(name='Predicted Price'),
                            'ml_backtest_df': backtest_df, 
                            'ml_instrument_name': instrument_name, 
                            'ml_model_choice': "Seasonal ARIMA"
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Seasonal ARIMA model forecasting failed: {e}")
                        st.session_state.update({'ml_predictions_by_date': None})

    with col2:
        if 'ml_model_choice' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast Results for {instrument_name} (Seasonal ARIMA)")
            
            if st.session_state.get('ml_predictions_by_date') is not None:
                forecast_df = st.session_state['ml_predictions_by_date']
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest)")
            backtest_df = st.session_state.get('ml_backtest_df')

            if backtest_df is not None and not backtest_df.empty:
                period_options = {
                    "Full History": len(backtest_df),
                    "Last Year": 252,
                    "Last 6 Months": 126,
                    "Last 3 Months": 63,
                    "Last Month": 21,
                    "Last 5 Days": 5
                }
                selected_period_name = st.selectbox("Select Backtest Period", list(period_options.keys()), key="backtest_period_select")
                days_to_display = period_options[selected_period_name]
                
                display_df = backtest_df.tail(days_to_display)
                
                if not display_df.empty:
                    mape_period = mean_squared_error(display_df['Actual'], display_df['Predicted']) * 100
                    accuracy_period = 100 - mape_period
                    cum_returns_period = (1 + (display_df['Actual'].pct_change().fillna(0))).cumprod()
                    peak_period = cum_returns_period.cummax()
                    drawdown_period = (cum_returns_period - peak_period) / peak_period
                    max_drawdown_period = drawdown_period.min()
                    
                    max_gains_period = ((1 + display_df['Predicted'].pct_change().fillna(0))).cumprod().max() - 1
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Accuracy ({selected_period_name})", f"{accuracy_period:.2f}%")
                    c2.metric(f"MAPE ({selected_period_name})", f"{mape_period:.2f}%")
                    c3.metric(f"Max Drawdown ({selected_period_name})", f"{max_drawdown_period*100:.2f}%")
                    st.metric(f"Max Gains ({selected_period_name})", f"{max_gains_period*100:.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Actual'], mode='lines', name='Actual Price'))
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                    fig.update_layout(title=f"Backtest Results ({selected_period_name})", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for the selected period.")
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
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9]+)', prompt_lower)
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
                        option_symbol = re.search(r'\b([A-Z]+)(\d{2}[-a-zA-Z]{3}\d+)\b', prompt.upper()).group(0)
                        if option_symbol:
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry.date() - datetime.now().date()).days, 0) / 365.0
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
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0

            if st.form_submit_button("Add to Basket"):
                if symbol:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
                    if not instrument.empty:
                        exchange = instrument.iloc[0]['exchange']
                        order = {
                            "tradingsymbol": symbol,
                            "exchange": exchange,
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "product": product,
                            "order_type": order_type,
                        }
                        if order_type == "LIMIT":
                            order["price"] = price
                        st.session_state.basket.append(order)
                        st.success(f"Added {symbol} to basket.")
                    else:
                        st.error(f"Symbol '{symbol}' not found.")
                else:
                    st.warning("Please enter a symbol.")

    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty. Add orders using the form on the left.")
        else:
            basket_df = pd.DataFrame(st.session_state.basket)
            st.dataframe(basket_df[['tradingsymbol', 'transaction_type', 'quantity', 'order_type', 'product']], use_container_width=True)

            if st.button("Execute Basket Order", use_container_width=True, type="primary"):
                with st.spinner("Placing basket order..."):
                    place_basket_order(st.session_state.basket, variety="regular")
                st.session_state.basket = []
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
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
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Portfolio Analytics")
        
        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
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
                hole=.3,
                textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        if sector_df is not None:
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train an advanced Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        with st.spinner(f"Loading real-time data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            st.stop()
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=30)
        forecast_date = st.date_input("Select a date to forecast", value=today, min_value=today, max_value=max_forecast_date)

        if st.button("Train Seasonal ARIMA Model & Forecast"):
            forecast_steps = (forecast_date - today).days + 1
            if forecast_steps <= 0:
                st.warning("Please select a future date to forecast.")
            else:
                with st.spinner("Training Seasonal ARIMA model... This may take a moment."):
                    predictions, backtest_df = train_seasonal_arima_model(data)
                    
                    try:
                        decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                        seasonally_adjusted = data['close'] - decomposed.seasonal
                        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                        forecast_adjusted = model.forecast(steps=forecast_steps)
                        
                        last_season_cycle = decomposed.seasonal.iloc[-7:]
                        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                        future_seasonal.index = forecast_adjusted.index
                        
                        forecast_final = forecast_adjusted + future_seasonal
                        
                        st.session_state.update({
                            'ml_predictions_by_date': forecast_final.to_frame(name='Predicted Price'),
                            'ml_backtest_df': backtest_df, 
                            'ml_instrument_name': instrument_name, 
                            'ml_model_choice': "Seasonal ARIMA"
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Seasonal ARIMA model forecasting failed: {e}")
                        st.session_state.update({'ml_predictions_by_date': None})

    with col2:
        if 'ml_model_choice' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast Results for {instrument_name} (Seasonal ARIMA)")
            
            if st.session_state.get('ml_predictions_by_date') is not None:
                forecast_df = st.session_state['ml_predictions_by_date']
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest)")
            backtest_df = st.session_state.get('ml_backtest_df')

            if backtest_df is not None and not backtest_df.empty:
                period_options = {
                    "Full History": len(backtest_df),
                    "Last Year": 252,
                    "Last 6 Months": 126,
                    "Last 3 Months": 63,
                    "Last Month": 21,
                    "Last 5 Days": 5
                }
                selected_period_name = st.selectbox("Select Backtest Period", list(period_options.keys()), key="backtest_period_select")
                days_to_display = period_options[selected_period_name]
                
                display_df = backtest_df.tail(days_to_display)
                
                if not display_df.empty:
                    mape_period = mean_squared_error(display_df['Actual'], display_df['Predicted']) * 100
                    accuracy_period = 100 - mape_period
                    cum_returns_period = (1 + (display_df['Actual'].pct_change().fillna(0))).cumprod()
                    peak_period = cum_returns_period.cummax()
                    drawdown_period = (cum_returns_period - peak_period) / peak_period
                    max_drawdown_period = drawdown_period.min()
                    
                    max_gains_period = ((1 + display_df['Predicted'].pct_change().fillna(0))).cumprod().max() - 1
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Accuracy ({selected_period_name})", f"{accuracy_period:.2f}%")
                    c2.metric(f"MAPE ({selected_period_name})", f"{mape_period:.2f}%")
                    c3.metric(f"Max Drawdown ({selected_period_name})", f"{max_drawdown_period*100:.2f}%")
                    st.metric(f"Max Gains ({selected_period_name})", f"{max_gains_period*100:.2f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Actual'], mode='lines', name='Actual Price'))
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                    fig.update_layout(title=f"Backtest Results ({selected_period_name})", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for the selected period.")
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
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9]+)', prompt_lower)
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
                        option_symbol = re.search(r'\b([A-Z]+)(\d{2}[-a-zA-Z]{3}\d+)\b', prompt.upper()).group(0)
                        if option_symbol:
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry.date() - datetime.now().date()).days, 0) / 365.0
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
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0

            if st.form_submit_button("Add to Basket"):
                if symbol:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
                    if not instrument.empty:
                        exchange = instrument.iloc[0]['exchange']
                        order = {
                            "tradingsymbol": symbol,
                            "exchange": exchange,
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "product": product,
                            "order_type": order_type,
                        }
                        if order_type == "LIMIT":
                            order["price"] = price
                        st.session_state.basket.append(order)
                        st.success(f"Added {symbol} to basket.")
                    else:
                        st.error(f"Symbol '{symbol}' not found.")
                else:
                    st.warning("Please enter a symbol.")

    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty. Add orders using the form on the left.")
        else:
            basket_df = pd.DataFrame(st.session_state.basket)
            st.dataframe(basket_df[['tradingsymbol', 'transaction_type', 'quantity', 'order_type', 'product']], use_container_width=True)

            if st.button("Execute Basket Order", use_container_width=True, type="primary"):
                with st.spinner("Placing basket order..."):
                    place_basket_order(st.session_state.basket, variety="regular")
                st.session_state.basket = []
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
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
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Portfolio Analytics")
        
        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
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
                hole=.3,
                textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        if sector_df is not None:
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train an advanced Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        with st.spinner(f"Loading real-time data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            st.stop()
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=30)
        forecast_date = st.date_input("Select a date to forecast", value=today, min_value=today, max_value=max_forecast_date)

        if st.button("Train Seasonal ARIMA Model & Forecast"):
            forecast_steps = (forecast_date - today).days + 1
            if forecast_steps <= 0:
                st.warning("Please select a future date to forecast.")
            else:
                with st.spinner("Training Seasonal ARIMA model... This may take a moment."):
                    predictions, backtest_df = train_seasonal_arima_model(data)
                    
                    try:
                        decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                        seasonally_adjusted = data['close'] - decomposed.seasonal
                        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                        forecast_adjusted = model.forecast(steps=forecast_steps)
                        
                        last_season_cycle = decomposed.seasonal.iloc[-7:]
                        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                        future_seasonal.index = forecast_adjusted.index
                        
                        forecast_final = forecast_adjusted + future_seasonal
                        
                        st.session_state.update({
                            'ml_predictions_by_date': forecast_final.to_frame(name='Predicted Price'),
                            'ml_backtest_df': backtest_df, 
                            'ml_instrument_name': instrument_name, 
                            'ml_model_choice': "Seasonal ARIMA"
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Seasonal ARIMA model forecasting failed: {e}")
                        st.session_state.update({'ml_predictions_by_date': None})

    with col2:
        if 'ml_model_choice' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast Results for {instrument_name} (Seasonal ARIMA)")
            
            if st.session_state.get('ml_predictions_by_date') is not None:
                forecast_df = st.session_state['ml_predictions_by_date']
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest)")
            backtest_df = st.session_state.get('ml_backtest_df')

            if backtest_df is not None and not backtest_df.empty:
                period_options = {
                    "Full History": len(backtest_df),
                    "Last Year": 252,
                    "Last 6 Months": 126,
                    "Last 3 Months": 63,
                    "Last Month": 21,
                    "Last 5 Days": 5
                }
                selected_period_name = st.selectbox("Select Backtest Period", list(period_options.keys()), key="backtest_period_select")
                days_to_display = period_options[selected_period_name]
                
                display_df = backtest_df.tail(days_to_display)
                
                if not display_df.empty:
                    mape_period = mean_squared_error(display_df['Actual'], display_df['Predicted']) * 100
                    accuracy_period = 100 - mape_period
                    cum_returns_period = (1 + (display_df['Actual'].pct_change().fillna(0))).cumprod()
                    peak_period = cum_returns_period.cummax()
                    drawdown_period = (cum_returns_period - peak_period) / peak_period
                    max_drawdown_period = drawdown_period.min()
                    
                    max_gains_period = ((1 + display_df['Predicted'].pct_change().fillna(0))).cumprod().max() - 1
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Accuracy ({selected_period_name})", f"{accuracy_period:.2f}%")
                    c2.metric(f"MAPE ({selected_period_name})", f"{mape_period:.2f}%")
                    c3.metric(f"Max Drawdown ({selected_period_name})", f"{max_drawdown_period*100:.2f}%", delta_color='inverse')
                    st.metric(f"Max Gains ({selected_period_name})", f"{max_gains_period*100:.2f}%", delta_color='normal')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Actual'], mode='lines', name='Actual Price'))
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                    fig.update_layout(title=f"Backtest Results ({selected_period_name})", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for the selected period.")
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
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9]+)', prompt_lower)
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
                        option_symbol = re.search(r'\b([A-Z]+)(\d{2}[-a-zA-Z]{3}\d+)\b', prompt.upper()).group(0)
                        if option_symbol:
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry.date() - datetime.now().date()).days, 0) / 365.0
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
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0

            if st.form_submit_button("Add to Basket"):
                if symbol:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
                    if not instrument.empty:
                        exchange = instrument.iloc[0]['exchange']
                        order = {
                            "tradingsymbol": symbol,
                            "exchange": exchange,
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "product": product,
                            "order_type": order_type,
                        }
                        if order_type == "LIMIT":
                            order["price"] = price
                        st.session_state.basket.append(order)
                        st.success(f"Added {symbol} to basket.")
                    else:
                        st.error(f"Symbol '{symbol}' not found.")
                else:
                    st.warning("Please enter a symbol.")

    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty. Add orders using the form on the left.")
        else:
            basket_df = pd.DataFrame(st.session_state.basket)
            st.dataframe(basket_df[['tradingsymbol', 'transaction_type', 'quantity', 'order_type', 'product']], use_container_width=True)

            if st.button("Execute Basket Order", use_container_width=True, type="primary"):
                with st.spinner("Placing basket order..."):
                    place_basket_order(st.session_state.basket, variety="regular")
                st.session_state.basket = []
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
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
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Portfolio Analytics")
        
        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
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
                hole=.3,
                textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        if sector_df is not None:
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train an advanced Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        
        with st.spinner(f"Loading real-time data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            st.stop()
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=30)
        forecast_date = st.date_input("Select a date to forecast", value=today, min_value=today, max_value=max_forecast_date)

        if st.button("Train Seasonal ARIMA Model & Forecast"):
            forecast_steps = (forecast_date - today).days + 1
            if forecast_steps <= 0:
                st.warning("Please select a future date to forecast.")
            else:
                with st.spinner("Training Seasonal ARIMA model... This may take a moment."):
                    predictions, backtest_df = train_seasonal_arima_model(data)
                    
                    try:
                        decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                        seasonally_adjusted = data['close'] - decomposed.seasonal
                        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                        forecast_adjusted = model.forecast(steps=forecast_steps)
                        
                        last_season_cycle = decomposed.seasonal.iloc[-7:]
                        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                        future_seasonal.index = forecast_adjusted.index
                        
                        forecast_final = forecast_adjusted + future_seasonal
                        
                        st.session_state.update({
                            'ml_predictions_by_date': forecast_final.to_frame(name='Predicted Price'),
                            'ml_backtest_df': backtest_df, 
                            'ml_instrument_name': instrument_name, 
                            'ml_model_choice': "Seasonal ARIMA"
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Seasonal ARIMA model forecasting failed: {e}")
                        st.session_state.update({'ml_predictions_by_date': None})

    with col2:
        if 'ml_model_choice' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            st.subheader(f"Forecast Results for {instrument_name} (Seasonal ARIMA)")
            
            if st.session_state.get('ml_predictions_by_date') is not None:
                forecast_df = st.session_state['ml_predictions_by_date']
                st.dataframe(forecast_df.style.format("₹{:.2f}"))
            else:
                st.error("Model training failed to produce forecasts.")

            st.subheader("Model Performance (Backtest)")
            backtest_df = st.session_state.get('ml_backtest_df')

            if backtest_df is not None and not backtest_df.empty:
                period_options = {
                    "Full History": len(backtest_df),
                    "Last Year": 252,
                    "Last 6 Months": 126,
                    "Last 3 Months": 63,
                    "Last Month": 21,
                    "Last 5 Days": 5
                }
                selected_period_name = st.selectbox("Select Backtest Period", list(period_options.keys()), key="backtest_period_select")
                days_to_display = period_options[selected_period_name]
                
                display_df = backtest_df.tail(days_to_display)
                
                if not display_df.empty:
                    mape_period = mean_squared_error(display_df['Actual'], display_df['Predicted']) * 100
                    accuracy_period = 100 - mape_period
                    cum_returns_period = (1 + (display_df['Actual'].pct_change().fillna(0))).cumprod()
                    peak_period = cum_returns_period.cummax()
                    drawdown_period = (cum_returns_period - peak_period) / peak_period
                    max_drawdown_period = drawdown_period.min()
                    
                    max_gains_period = ((1 + display_df['Predicted'].pct_change().fillna(0))).cumprod().max() - 1
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Accuracy ({selected_period_name})", f"{accuracy_period:.2f}%")
                    c2.metric(f"MAPE ({selected_period_name})", f"{mape_period:.2f}%")
                    c3.metric(f"Max Drawdown ({selected_period_name})", f"{max_drawdown_period*100:.2f}%", delta_color='inverse')
                    st.metric(f"Max Gains ({selected_period_name})", f"{max_gains_period*100:.2f}%", delta_color='normal')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Actual'], mode='lines', name='Actual Price'))
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                    fig.update_layout(title=f"Backtest Results ({selected_period_name})", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for the selected period.")
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
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9]+)', prompt_lower)
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
                        option_symbol = re.search(r'\b([A-Z]+)(\d{2}[-a-zA-Z]{3}\d+)\b', prompt.upper()).group(0)
                        if option_symbol:
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            expiry_date_from_symbol = option_details['expiry'].date() if hasattr(option_details['expiry'], 'date') else option_details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, expiry_date_from_symbol)
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((expiry.date() - datetime.now().date()).days, 0) / 365.0
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
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0

            if st.form_submit_button("Add to Basket"):
                if symbol:
                    instrument = instrument_df[instrument_df['tradingsymbol'] == symbol]
                    if not instrument.empty:
                        exchange = instrument.iloc[0]['exchange']
                        order = {
                            "tradingsymbol": symbol,
                            "exchange": exchange,
                            "transaction_type": transaction_type,
                            "quantity": quantity,
                            "product": product,
                            "order_type": order_type,
                        }
                        if order_type == "LIMIT":
                            order["price"] = price
                        st.session_state.basket.append(order)
                        st.success(f"Added {symbol} to basket.")
                    else:
                        st.error(f"Symbol '{symbol}' not found.")
                else:
                    st.warning("Please enter a symbol.")

    with col2:
        st.subheader("Current Basket")
        if not st.session_state.basket:
            st.info("Your basket is empty. Add orders using the form on the left.")
        else:
            basket_df = pd.DataFrame(st.session_state.basket)
            st.dataframe(basket_df[['tradingsymbol', 'transaction_type', 'quantity', 'order_type', 'product']], use_container_width=True)

            if st.button("Execute Basket Order", use_container_width=True, type="primary"):
                with st.spinner("Placing basket order..."):
                    place_basket_order(st.session_state.basket, variety="regular")
                st.session_state.basket = []
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_portfolio_and_risk():
    """A page for portfolio and risk management, including live P&L and holdings."""
    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    
    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
        return

    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Analytics & Allocation"])
    
    with tab1:
        st.subheader("Live Intraday Positions")
        # --- FIX WAS HERE ---
        # This entire if/else block is now correctly indented to be inside 'with tab1:'
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}", delta_color='normal' if total_pnl >= 0 else 'inverse')
        else:
            st.info("No open positions for the day.")

    with tab2:
        st.subheader("Investment Holdings")
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Portfolio Analytics")
        
        holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
        
        sector_df = get_sector_data()
        
        if sector_df is not None and not holdings_df.empty:
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
                hole=.3,
                textinfo='label+percent'
            )])
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white')
            st.plotly_chart(fig_stock, use_container_width=True)
            
        if sector_df is not None:
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig_sector = go.Figure(data=[go.Pie(
                    labels=sector_allocation['Sector'],
                    values=sector_allocation['current_value'],
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)

def page_options_chain():
    """An advanced options chain page with Greek calculations and OI analysis."""
    display_header()
    st.title("Advanced Options Chain")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to view the options chain.")
        return

    options_underlyings = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "GOLDM", "USDINR"]
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    underlying = col1.selectbox("Select Underlying", options_underlyings)
    
    chain_df, expiry_date, ltp, available_expiries = get_options_chain(underlying, instrument_df)
    
    if chain_df.empty:
        st.warning(f"Could not load options chain for {underlying}. It may not be an F&O symbol.")
        return

    selected_expiry = col2.selectbox("Select Expiry", [d.strftime('%d-%b-%Y') for d in available_expiries])
    if selected_expiry:
        expiry_date = datetime.strptime(selected_expiry, '%d-%b-%Y')
        chain_df, _, ltp, _ = get_options_chain(underlying, instrument_df, expiry_date)

    if col3.button("Most Active", use_container_width=True):
        show_most_active_dialog(underlying, instrument_df)

    total_ce_oi = chain_df['open_interest_CE'].sum()
    total_pe_oi = chain_df['open_interest_PE'].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

    st.metric(f"Spot Price: {underlying}", f"{ltp:,.2f}", f"PCR: {pcr:.2f}")

    # OI Bar Chart
    st.subheader("Open Interest Profile")
    atm_strike_index = abs(chain_df['STRIKE'] - ltp).idxmin()
    strikes_to_show = chain_df.loc[atm_strike_index-10:atm_strike_index+10]
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=strikes_to_show['STRIKE'], y=strikes_to_show['open_interest_CE'], name='CE OI', marker_color='#FF4B4B'))
    fig_oi.add_trace(go.Bar(x=strikes_to_show['STRIKE'], y=strikes_to_show['open_interest_PE'], name='PE OI', marker_color='#28a745'))
    fig_oi.update_layout(barmode='group', title=f"Open Interest for {underlying} ({expiry_date.strftime('%d-%b-%Y')})", template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white', yaxis_title="Contracts", xaxis_title="Strike Price")
    st.plotly_chart(fig_oi, use_container_width=True)
    
    st.subheader("Options Chain")
    st.dataframe(style_option_chain(chain_df, ltp), use_container_width=True, height=600, hide_index=True)

def page_market_overview():
    """Provides a high-level overview of Indian and Global markets."""
    display_header()
    st.title("Market Overview")

    tab1, tab2 = st.tabs(["Indian Markets", "Global Markets"])

    with tab1:
        st.subheader("Indian Indices")
        indian_indices = [
            {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
            {'symbol': 'NIFTY BANK', 'exchange': 'NSE'},
            {'symbol': 'NIFTY FIN SERVICE', 'exchange': 'NSE'},
            {'symbol': 'SENSEX', 'exchange': 'BSE'},
            {'symbol': 'INDIA VIX', 'exchange': 'NSE'}
        ]
        indian_data = get_indian_indices_data(indian_indices)
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
    # --- SESSION STATE INITIALIZATION ---
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'kite' not in st.session_state: st.session_state.kite = None
    if 'broker' not in st.session_state: st.session_state.broker = None
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
    if 'refresh_interval' not in st.session_state: st.session_state.refresh_interval = 30

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("BlockVista Settings")
        st.session_state.theme = st.radio("Theme", ["Light", "Dark"], index=1, horizontal=True)
        st.markdown("---")
        
        # --- Broker Connection ---
        if not st.session_state.logged_in:
            st.subheader("Connect to Broker")
            st.session_state.broker = st.selectbox("Select Broker", ["Zerodha"])
            if st.session_state.broker == "Zerodha":
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
                totp_secret = st.text_input("TOTP Secret (optional)", type="password")

                if st.button("Generate Login URL"):
                    if api_key and api_secret:
                        try:
                            kite = KiteConnect(api_key=api_key)
                            st.session_state.kite_pre = kite
                            st.session_state.api_secret = api_secret
                            st.session_state.totp_secret = totp_secret
                            login_url = st.session_state.kite_pre.login_url()
                            st.markdown(f"**[Click here to login]({login_url})**")
                        except Exception as e:
                            st.error(f"Login failed: {e}")
                    else:
                        st.warning("Please enter API Key and Secret.")

                request_token = st.text_input("Enter Request Token")
                if st.button("Login"):
                    if request_token and 'kite_pre' in st.session_state:
                        try:
                            totp = pyotp.TOTP(st.session_state.totp_secret).now() if st.session_state.totp_secret else None
                            user_data = st.session_state.kite_pre.generate_session(request_token, api_secret=st.session_state.api_secret, totp=totp)
                            st.session_state.kite = st.session_state.kite_pre
                            st.session_state.kite.set_access_token(user_data["access_token"])
                            st.session_state.logged_in = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Authentication failed: {e}")
        else:
            st.success(f"✅ Connected to {st.session_state.broker}")
            profile = st.session_state.kite.profile()
            st.write(f"Welcome, {profile.get('user_name', 'User')}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.kite = None
                st.rerun()

        st.markdown("---")
        
        # --- Page Navigation ---
        st.subheader("Navigation")
        pages = {
            "📊 Dashboard": page_dashboard,
            "📈 Advanced Charting": page_advanced_charting,
            "⛓️ Options Chain": page_options_chain,
            "🌍 Market Overview": page_market_overview,
            "💡 Alpha Engine": page_alpha_engine,
            "📦 Basket Orders": page_basket_orders,
            "💼 Portfolio & Risk": page_portfolio_and_risk,
            "🤖 AI Assistant": page_ai_assistant,
            "🔮 ML Forecasting": page_forecasting_ml,
        }
        page_selection = st.radio("Go to", list(pages.keys()), label_visibility="collapsed")
        st.markdown("---")

        # --- Auto-Refresh Controls ---
        st.subheader("Auto-Refresh")
        st.session_state.auto_refresh = st.toggle("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider("Refresh Interval (s)", 5, 120, st.session_state.refresh_interval)
            st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="data_refresher")
        
        # --- Order History ---
        with st.expander("Recent Orders"):
            if st.session_state.order_history:
                for order in st.session_state.order_history[:5]:
                    st.info(f"{order['type']} {order['symbol']} ({order['qty']}) - {order['status']}")
            else:
                st.write("No orders placed in this session.")

    # --- PAGE ROUTING ---
    pages[page_selection]()

if __name__ == "__main__":
    main()
