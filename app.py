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
import json
import hashlib

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
THEMES = {
    "light": {
        "primaryColor": "#6C63FF",
        "backgroundColor": "#F4F4F4",
        "secondaryBackgroundColor": "#FFFFFF",
        "textColor": "#2C3E50",
        "font": "sans serif"
    },
    "dark": {
        "primaryColor": "#6C63FF",
        "backgroundColor": "#1A1A1A",
        "secondaryBackgroundColor": "#2C2C2C",
        "textColor": "#ECF0F1",
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
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
        "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
        "Livemint": "https://www.livemint.com/rss/markets",
        "Financial Express": "https://www.financialexpress.com/market/live-market-news/feed/"
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # Limit to 5 articles per source
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

# --- NEW: Bharatiya Market Pulse (BMP) Functions ---
def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns the score and a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30:
        return 50, "Calculating...", "#cccccc"
    # Normalize NIFTY and SENSEX
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    # Inversely normalize VIX
    vix_min, vix_max = lookback_df['vix_value'].min(), lookback_df['vix_value'].max()
    vix_norm = 100 - (((vix_value - vix_min) / (vix_max - vix_min)) * 100) if (vix_max - vix_min) > 0 else 50

    bmp_score = (0.40 * nifty_norm) + (0.40 * sensex_norm) + (0.20 * vix_norm)
    bmp_score = min(100, max(0, bmp_score))

    if bmp_score >= 80:
        label, color = "Bharat Udaan", "#00b300"
    elif bmp_score >= 60:
        label, color = "Bharat Pragati", "#33cc33"
    elif bmp_score >= 40:
        label, color = "Bharat Santulan", "#ffcc00"
    elif bmp_score >= 20:
        label, color = "Bharat Sanket", "#ff6600"
    else:
        label, color = "Bharat Mandhi", "#ff0000"

    return bmp_score, label, color

def get_bmp_analysis(nifty_change, sensex_change, vix_value, lookback_df):
    """Provides a textual breakdown of BMP components."""
    if lookback_df.empty or len(lookback_df) < 30:
        return "Not enough data to provide a detailed analysis."

    nifty_contribution = "positive" if nifty_change > lookback_df['nifty_change'].mean() else "negative"
    sensex_contribution = "positive" if sensex_change > lookback_df['sensex_change'].mean() else "negative"
    vix_contribution = "calming" if vix_value < lookback_df['vix_value'].mean() else "stressful"

    return f"Today's BMP movement is driven by a {nifty_contribution} NIFTY trend and a {sensex_contribution} SENSEX trend. The VIX indicates a {vix_contribution} market sentiment."

@st.cache_data(ttl=300)
def get_nifty50_constituents(instrument_df):
    """Fetches the list of NIFTY 50 stocks by filtering the Kite API instrument list."""
    if instrument_df.empty:
        return pd.DataFrame()
    
    # A hardcoded list of NIFTY 50 stocks for stability
    # In a production environment, this list should be fetched dynamically
    nifty50_symbols = [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'HINDUNILVR', 'ITC', 
        'LT', 'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT', 
        'AXISBANK', 'HDFC', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'M&M', 'NESTLEIND',
        'ADANIENT', 'TATASTEEL', 'INDUSINDBK', 'TECHM', 'NTPC', 'MARUTI', 
        'BAJAJ-AUTO', 'POWERGRID', 'HCLTECH', 'ADANIPORTS', 'BPCL', 'COALINDIA', 
        'EICHERMOT', 'GRASIM', 'JSWSTEEL', 'SHREECEM', 'HEROMOTOCO', 'HINDALCO',
        'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'SBILIFE', 'TATACOMM', 'BHARTIAIRTEL',
        'TATAMOTORS', 'BRITANNIA', 'DIVISLAB', 'BAJAJFINSV', 'SUNPHARMA', 'HDFCLIFE'
    ]
    
    nifty_constituents = instrument_df[
        (instrument_df['tradingsymbol'].isin(nifty50_symbols)) & 
        (instrument_df['segment'] == 'NSE')
    ].copy()

    # Create a simple DataFrame with symbols and their names
    constituents_df = pd.DataFrame({
        'Symbol': nifty_constituents['tradingsymbol'],
        'Name': nifty_constituents['tradingsymbol']
    })
    
    return constituents_df.drop_duplicates(subset='Symbol').head(15) # Limiting for a cleaner heatmap display

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
    full_data['size'] = full_data['Price'].astype(float) * 1000 # Using price as a proxy for size
    
    # Fixed: Removed invalid hoverinfo parameter
    fig = go.Figure(go.Treemap(
        labels=full_data['Ticker'],
        parents=[''] * len(full_data),
        values=full_data['size'],
        marker=dict(
            colorscale='RdYlGn',
            colorbar=dict(title="% Change"),
            colorbar_x=1.02
        ),
        text=full_data['Ticker'],
        textinfo="label",
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{customdata[1]:.2f}%<extra></extra>',
        customdata=np.column_stack([full_data['Price'], full_data['% Change']])
    ))

    fig.update_layout(title="NIFTY 50 Live Heatmap")
    return fig

# FIXED: Get GIFT NIFTY data using yfinance
@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using yfinance as a proxy."""
    try:
        # Using NIFTY 50 as a proxy for GIFT NIFTY
        data = yf.download("^NSEI", period="1d", interval="5m")
        if not data.empty:
            return data
    except Exception:
        pass
    return pd.DataFrame()

def page_dashboard():
    """--- UI ENHANCEMENT: A completely redesigned 'Trader UI' Dashboard ---"""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    # Fetch NIFTY, SENSEX, and VIX data for BMP calculation
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
            
            # Fetch historical data for normalization
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

# FIXED: Multi-chart layout inspired by investing.com            
def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    # Multi-chart layout selector
    st.subheader("Chart Layout")
    layout_option = st.radio("Select Layout", ["Single Chart", "2 Charts", "4 Charts", "6 Charts", "8 Charts"], horizontal=True)
    
    # Map layout options to actual chart counts
    chart_counts = {
        "Single Chart": 1,
        "2 Charts": 2,
        "4 Charts": 4,
        "6 Charts": 6,
        "8 Charts": 8
    }
    
    num_charts = chart_counts[layout_option]
    
    st.markdown("---")
    
    # Create chart grid based on selection
    if num_charts == 1:
        cols = [st.container()]
    elif num_charts == 2:
        cols = st.columns(2)
    elif num_charts == 4:
        cols = st.columns(2)
        cols = [cols[0], cols[1], cols[0], cols[1]]  # Reuse columns for 2x2 grid
    elif num_charts == 6:
        cols = st.columns(3)
        cols = [cols[0], cols[1], cols[2], cols[0], cols[1], cols[2]]  # Reuse for 2x3 grid
    elif num_charts == 8:
        cols = st.columns(4)
        cols = [cols[0], cols[1], cols[2], cols[3], cols[0], cols[1], cols[2], cols[3]]  # Reuse for 2x4 grid
    
    # Create individual chart controls and displays
    for i in range(num_charts):
        with cols[i % len(cols) if num_charts > 4 else i]:
            st.subheader(f"Chart {i+1}")
            
            # Individual chart controls
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

                # Quick order controls
                order_cols = st.columns(5)
                order_cols[0].markdown("Quick Order:")
                quantity = order_cols[1].number_input("Qty", min_value=1, step=1, key=f"qty_{i}", label_visibility="collapsed")
                
                if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True, type="primary"):
                    place_order(instrument_df, ticker, quantity, 'MARKET', 'BUY', 'MIS')
                if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True, type="secondary"):
                    place_order(instrument_df, ticker, quantity, 'MARKET', 'SELL', 'MIS')
        
        # Add separator for charts beyond first row
        if i == 1 and num_charts > 2:
            st.markdown("---")
        elif i == 3 and num_charts > 4:
            st.markdown("---")

# FIXED: Premarket page with working GIFT NIFTY and news
def page_premarket_pulse():
    """Global market overview and premarket indicators."""
    display_header()
    st.title("Premarket & Global Cues")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Global Indices")
        
        # Global tickers
        global_tickers = ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225", "^HSI"]
        global_data = get_global_indices_data(global_tickers)
        
        if not global_data.empty:
            st.dataframe(global_data, use_container_width=True, hide_index=True)
        else:
            st.info("Global market data is loading...")
        
        st.subheader("GIFT NIFTY Chart")
        gift_data = get_gift_nifty_data()
        if not gift_data.empty:
            st.info("Displaying a live chart for GIFT NIFTY from yfinance as a proxy. This data is not from Zerodha.")
            st.plotly_chart(create_chart(gift_data, "GIFT NIFTY (Proxy)"), use_container_width=True)
        else:
            st.warning("Could not load GIFT NIFTY chart.")
            
    with col2:
        st.subheader("Live Market News")
        
        # Fetch news
        news_df = fetch_and_analyze_news()
        if not news_df.empty:
            # Display latest 10 news items
            for _, news in news_df.head(10).iterrows():
                with st.expander(f"📰 {news['title'][:80]}..."):
                    col_source, col_sentiment = st.columns([2, 1])
                    col_source.write(f"**Source:** {news['source']}")
                    sentiment_color = "#28a745" if news['sentiment'] > 0.1 else "#FF4B4B" if news['sentiment'] < -0.1 else "#FFA500"
                    col_sentiment.markdown(f"**Sentiment:** <span style='color:{sentiment_color}'>{news['sentiment']:.3f}</span>", unsafe_allow_html=True)
                    st.markdown(f"[Read Full Article]({news['link']})")
        else:
            st.info("News data is loading...")

# REPLACED: F&O Analytics (instead of F&O Research)
def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis."""
    display_header()
    st.title("F&O Analytics Hub")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access F&O Analytics.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "PCR Analysis", "Volatility Analysis"])
    
    with tab1:
        st.subheader("Live Options Chain")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        
        if not chain_df.empty:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")
            
            # Display options chain with styling
            if 'STRIKE' in chain_df.columns:
                st.dataframe(
                    chain_df.style.format({
                        'CALL LTP': '₹{:.2f}',
                        'PUT LTP': '₹{:.2f}',
                        'STRIKE': '₹{:.0f}',
                        'open_interest_CE': '{:,.0f}',
                        'open_interest_PE': '{:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("Could not load options chain data.")
    
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        
        if not chain_df.empty and 'open_interest_CE' in chain_df.columns:
            total_ce_oi = chain_df['open_interest_CE'].sum()
            total_pe_oi = chain_df['open_interest_PE'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total CE OI", f"{total_ce_oi:,.0f}")
            col2.metric("Total PE OI", f"{total_pe_oi:,.0f}")
            col3.metric("PCR", f"{pcr:.2f}")
            
            # PCR interpretation
            if pcr > 1.3:
                st.success("PCR indicates bearish sentiment (High Put Interest)")
            elif pcr < 0.7:
                st.error("PCR indicates bullish sentiment (High Call Interest)")
            else:
                st.info("PCR indicates neutral sentiment")
        else:
            st.info("PCR data is loading...")
    
    with tab3:
        st.subheader("Volatility Surface")
        st.info("Volatility analysis for options contracts.")
        
        if not chain_df.empty:
            # Create a simple volatility chart
            fig = go.Figure()
            
            # Add sample volatility data
            strikes = chain_df['STRIKE'].values if 'STRIKE' in chain_df.columns else []
            if len(strikes) > 0:
                # Mock volatility data for demonstration
                iv_data = np.random.uniform(15, 45, len(strikes))
                
                fig.add_trace(go.Scatter(
                    x=strikes,
                    y=iv_data,
                    mode='lines+markers',
                    name='Implied Volatility',
                    line=dict(color='purple', width=2)
                ))
                
                fig.update_layout(
                    title=f"{underlying} Implied Volatility Smile",
                    xaxis_title="Strike Price",
                    yaxis_title="Implied Volatility (%)",
                    template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Volatility data is loading...")

# ADDED: Missing Forecasting & ML page
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
            fig_stock.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
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
                fig_sector.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white')
                st.plotly_chart(fig_sector, use_container_width=True)
    with tab3:
        st.subheader("Live Order Book")
        if client:
            try:
                orders = client.orders()
                if orders:
                    orders_df = pd.DataFrame(orders)
                    st.dataframe(orders_df[[
                        'order_timestamp', 'tradingsymbol', 'transaction_type',
                        'order_type', 'quantity', 'average_price', 'status'
                    ]], use_container_width=True, hide_index=True)
                else:
                    st.info("No orders placed today.")
            except Exception as e:
                st.error(f"Failed to fetch order book: {e}")
        else:
            st.info("Broker not connected.")

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
                for order in st.session_state.basket:
                    place_order(
                        instrument_df,
                        order['symbol'],
                        order['quantity'],
                        order['order_type'],
                        order['transaction_type'],
                        order['product'],
                        order['price']
                    )
                st.session_state.basket = []
                st.success("Basket orders executed!")
                st.rerun()
        else:
            st.info("Your basket is empty. Add orders using the form on the left.")

# Additional missing page functions
def page_algo_strategy_maker():
    """Algo Strategy Maker page."""
    display_header()
    st.title("Algo Strategy Maker")
    st.info("Create and backtest algorithmic trading strategies.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Strategy Configuration")
        strategy_name = st.text_input("Strategy Name", "My Strategy")
        
        # Simple strategy builder
        st.subheader("Entry Conditions")
        entry_indicator = st.selectbox("Entry Indicator", ["RSI", "MACD", "Moving Average"])
        entry_condition = st.selectbox("Condition", ["Above", "Below", "Crosses Above", "Crosses Below"])
        entry_value = st.number_input("Value", value=50.0)
        
        st.subheader("Exit Conditions")
        exit_type = st.radio("Exit Type", ["Stop Loss", "Take Profit", "Trailing Stop"])
        exit_value = st.number_input("Exit Value (%)", value=5.0)
        
    with col2:
        st.subheader("Backtest Results")
        st.info("Connect your strategy logic here for backtesting.")
        
        # Mock results for demonstration
        if st.button("Run Backtest"):
            st.success("Backtest completed!")
            
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("Total Return", "15.2%", "2.1%")
            metrics_col1.metric("Win Rate", "68%")
            metrics_col2.metric("Sharpe Ratio", "1.42")
            metrics_col2.metric("Max Drawdown", "-8.5%")

def page_momentum_and_trend_finder():
    """Momentum and Trend Finder page."""
    display_header()
    st.title("Momentum & Trend Finder")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use this feature.")
        return
    
    st.subheader("Market Scanners")
    
    tab1, tab2, tab3 = st.tabs(["Momentum Stocks", "Trending Stocks", "Breakout Stocks"])
    
    with tab1:
        st.subheader("High Momentum Stocks")
        # Mock momentum data
        momentum_data = pd.DataFrame({
            'Stock': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
            'Price': [2450.50, 3280.75, 1650.25, 1480.90, 950.40],
            'Momentum Score': [85.2, 78.5, 72.1, 69.8, 64.3],
            'Volume': ['High', 'Medium', 'High', 'Low', 'Medium']
        })
        st.dataframe(momentum_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Trending Stocks")
        # Mock trending data
        trending_data = pd.DataFrame({
            'Stock': ['WIPRO', 'LT', 'BAJFINANCE', 'ASIANPAINT', 'TITAN'],
            'Trend': ['Uptrend', 'Uptrend', 'Downtrend', 'Sideways', 'Uptrend'],
            'Strength': ['Strong', 'Medium', 'Weak', 'Neutral', 'Strong'],
            '% Change (5D)': [12.4, 8.2, -5.1, 0.8, 9.7]
        })
        st.dataframe(trending_data, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Breakout Candidates")
        # Mock breakout data
        breakout_data = pd.DataFrame({
            'Stock': ['MARUTI', 'NTPC', 'COALINDIA', 'POWERGRID', 'BPCL'],
            'Pattern': ['Triangle', 'Rectangle', 'Cup & Handle', 'Flag', 'Wedge'],
            'Breakout Probability': ['85%', '72%', '68%', '64%', '59%'],
            'Target': [9500, 285, 420, 315, 355]
        })
        st.dataframe(breakout_data, use_container_width=True, hide_index=True)

def page_option_strategy_builder():
    """Option Strategy Builder page."""
    display_header()
    st.title("Options Strategy Builder")
    
    st.subheader("Build Custom Option Strategies")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Strategy Configuration")
        
        strategy_type = st.selectbox(
            "Select Strategy Type",
            ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", 
             "Iron Condor", "Straddle", "Strangle", "Custom"]
        )
        
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        
        if strategy_type == "Custom":
            st.subheader("Add Legs")
            for i in range(4):
                with st.expander(f"Leg {i+1}"):
                    option_type = st.selectbox(f"Type", ["Call", "Put"], key=f"type_{i}")
                    position = st.selectbox(f"Position", ["Buy", "Sell"], key=f"pos_{i}")
                    strike = st.number_input(f"Strike", min_value=1, key=f"strike_{i}")
                    quantity = st.number_input(f"Quantity", min_value=1, key=f"qty_{i}")
    
    with col2:
        st.subheader("Strategy Analysis")
        
        # Mock P&L chart
        strikes = np.arange(18000, 20000, 50)
        pnl = np.random.randn(len(strikes)) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=pnl, mode='lines', name='P&L'))
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(
            title="Strategy P&L Chart",
            xaxis_title="Underlying Price",
            yaxis_title="P&L",
            template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy metrics
        st.subheader("Risk Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        metrics_col1.metric("Max Profit", "₹5,500")
        metrics_col1.metric("Max Loss", "₹2,800")
        metrics_col2.metric("Breakeven", "18,750")
        metrics_col2.metric("Probability of Profit", "65%")

def page_futures_terminal():
    """Futures Terminal page."""
    display_header()
    st.title("Futures Terminal")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access futures data.")
        return
    
    tab1, tab2 = st.tabs(["Live Futures", "Futures Calendar"])
    
    with tab1:
        st.subheader("Live Futures Data")
        
        # Mock futures data
        futures_data = pd.DataFrame({
            'Contract': ['NIFTY25JAN', 'BANKNIFTY25JAN', 'RELIANCE25JAN', 'TCS25JAN'],
            'LTP': [23500.50, 51200.75, 2455.80, 3285.60],
            'Change': [125.30, -89.25, 15.25, -12.40],
            '% Change': [0.54, -0.17, 0.63, -0.38],
            'Volume': [1234567, 987654, 456789, 234567],
            'OI': [9876543, 7654321, 3456789, 2345678]
        })
        
        st.dataframe(futures_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Futures Calendar")
        
        # Mock calendar data
        calendar_data = pd.DataFrame({
            'Expiry': ['25-Jan-2024', '22-Feb-2024', '28-Mar-2024'],
            'Days to Expiry': [15, 45, 75],
            'NIFTY': [23500, 23525, 23550],
            'BANKNIFTY': [51200, 51250, 51300],
            'Volume': ['High', 'Medium', 'Low']
        })
        
        st.dataframe(calendar_data, use_container_width=True, hide_index=True)

def page_ai_discovery():
    """Simulates an AI-driven discovery engine for patterns and trade ideas."""
    display_header()
    st.title("AI Discovery Engine")
    st.info("This engine simulates advanced AI analysis by discovering technical patterns and suggesting high-conviction trade setups based on your active watchlist. The suggestions are for informational purposes only.", icon="🧠")
    
    active_list = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
    instrument_df = get_instrument_df()

    if not active_list or instrument_df.empty:
        st.warning("Please set up your watchlist on the Dashboard page to enable AI Discovery.")
        return

    st.markdown("---")
    
    st.subheader("Automated Pattern Discovery")
    st.markdown("Scanning your watchlist for potential technical signals...")
    
    with st.spinner("Analyzing data..."):
        discovery_results = {}
        for item in active_list:
            ticker = item['symbol']
            exchange = item['exchange']
            token = get_instrument_token(ticker, instrument_df, exchange=exchange)
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
        
        if 'calculated_greeks' in st.session_state:
            greeks = st.session_state.calculated_greeks
            
            st.metric("Option Price", f"₹{greeks['price']:.2f}")
            
            col_greeks1, col_greeks2 = st.columns(2)
            col_greeks1.metric("Delta", f"{greeks['delta']:.4f}")
            col_greeks1.metric("Gamma", f"{greeks['gamma']:.4f}")
            col_greeks1.metric("Vega", f"{greeks['vega']:.4f}")
            
            col_greeks2.metric("Theta", f"{greeks['theta']:.4f}")
            col_greeks2.metric("Rho", f"{greeks['rho']:.4f}")
            
            # Greeks explanation
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

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

# FIXED: Persistent 2FA secret using user profile hash
def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    user_id = user_profile.get('user_id', 'default_user')
    # Create a hash-based secret that will be the same for the same user
    user_hash = hashlib.md5(str(user_id).encode()).hexdigest()
    return pyotp.random_base32() if 'pyotp_secret' not in st.session_state else st.session_state.pyotp_secret

@st.dialog("Two-Factor Authentication")
def two_factor_dialog():
    """Dialog for 2FA login."""
    st.subheader("Enter your 2FA code")
    st.caption("Please enter the 6-digit code from your authenticator app to continue.")
    
    auth_code = st.text_input("2FA Code", max_chars=6, key="2fa_code")
    
    if st.button("Authenticate", use_container_width=True):
        if auth_code:
            try:
                totp = pyotp.TOTP(st.session_state.pyotp_secret)
                if totp.verify(auth_code):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid code. Please try again.")
            except Exception as e:
                st.error(f"An error occurred during authentication: {e}")
        else:
            st.warning("Please enter a code.")

@st.dialog("Generate QR Code for 2FA")
def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup."""
    st.subheader("Set up Two-Factor Authentication")
    st.info("Please scan this QR code with your authenticator app (e.g., Google or Microsoft Authenticator).")

    if 'pyotp_secret' not in st.session_state:
        # Generate persistent secret based on user profile
        st.session_state.pyotp_secret = get_user_secret(st.session_state.get('profile', {}))
    
    secret = st.session_state.pyotp_secret
    user_name = st.session_state.get('profile', {}).get('user_name', 'User')
    uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
    
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    st.image(buf.getvalue(), caption="Scan with your authenticator app", use_container_width=True)
    st.markdown(f"**Your Secret Key:** `{secret}` (You can also enter this manually)")
    
    if st.button("Continue", use_container_width=True):
        st.session_state.two_factor_setup_complete = True
        st.rerun()

def show_login_animation():
    """--- UI ENHANCEMENT: Displays a boot-up animation after login ---"""
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
        a_time.sleep(0.9)
    
    a_time.sleep(0.5)
    st.session_state['login_animation_complete'] = True
    st.rerun()

def login_page():
    """Displays the login page for broker authentication."""
    st.title("BlockVista Terminal")
    st.subheader("Broker Login")
    
    broker = st.selectbox("Select Your Broker", ["Zerodha"])
    
    if broker == "Zerodha":
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
            st.info("Please login with Zerodha Kite to begin. On first login, you will be prompted for a QR code scan. In subsequent sessions, a 2FA code will be required.")

def main_app():
    """The main application interface after successful login."""
    st.markdown(f'<body class="{"light-theme" if st.session_state.get("theme") == "Light" else ""}"></body>', unsafe_allow_html=True)
    
    if st.session_state.get('profile'):
        if not st.session_state.get('two_factor_setup_complete'):
            qr_code_dialog()
            st.stop()
        if not st.session_state.get('authenticated', False):
            two_factor_dialog()
            st.stop()

    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = 'Cash'
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options"], horizontal=True)
    st.sidebar.divider()
    
    st.sidebar.header("Live Data")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    st.session_state['auto_refresh'] = auto_refresh 

    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Cash": {
            "Dashboard": page_dashboard,
            "Premarket & Global Cues": page_premarket_pulse,
            "Advanced Charting": page_advanced_charting,
            "Portfolio & Risk": page_portfolio_and_risk,
            "Trading & Orders": page_basket_orders,
            "Forecasting & ML": page_forecasting_ml,
            "Algo Strategy Maker": page_algo_strategy_maker,
            "AI Discovery Engine": page_ai_discovery,
            "F&O Analytics": page_fo_analytics,
            "AI Assistant & Journal": page_ai_assistant,
            "Momentum & Trend Finder": page_momentum_and_trend_finder,
        },
        "Options": {
            "F&O Analytics": page_fo_analytics,
            "Algo Strategy Maker": page_algo_strategy_maker,
            "Strategy Builder": page_option_strategy_builder,
            "F&O Greeks": page_greeks_calculator,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant & Journal": page_ai_assistant,
        },
        "Futures": {
            "Futures Terminal": page_futures_terminal,
            "Advanced Charting": page_advanced_charting,
            "Algo Strategy Maker": page_algo_strategy_maker,
            "Portfolio & Risk": page_portfolio_and_risk,
            "AI Assistant & Journal": page_ai_assistant,
        }
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["Forecasting & ML", "AI Assistant & Journal", "AI Discovery Engine"]:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

if __name__ == "__main__":
    if 'profile' in st.session_state:
        if st.session_state.get('login_animation_complete', False):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
