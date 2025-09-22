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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
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
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None, exchange=None):
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

@st.cache_data(ttl=60)
def get_global_indices_data():
    """Fetches live data for major world indices using yfinance."""
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW JONES': '^DJI',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'NIKKEI 225': '^N225'
    }
    data = []
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
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
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
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
    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    client = get_broker_client()
    if not client: return pd.DataFrame()
    if st.session_state.broker == "Zerodha":
        return pd.DataFrame(client.instruments())
    else:
        st.warning(f"Instrument list for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
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

def get_watchlist_data(symbols_with_exchange):
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
        if not expiry_date: expiry_date = available_expiries[0] if available_expiries else None
        if not expiry_date: return pd.DataFrame(), None, underlying_ltp, available_expiries
        chain_df = options[options['expiry'] == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        quotes = client.quote(instruments_to_fetch)
        ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
        pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
        final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP']], pe_df[['tradingsymbol', 'strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
        return final_chain[['CALL', 'CALL LTP', 'STRIKE', 'PUT LTP', 'PUT']], expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
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
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    if st.session_state.broker == "Zerodha":
        try:
            # For options, the exchange is NFO, not derived from instrument_df
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
    analyzer = SentimentIntensityAnalyzer()
    news_sources = {"Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml", "Business Standard": "https://www.business-standard.com/rss/markets-102.cms", "Livemint": "https://www.livemint.com/rss/markets"}
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    published_date = datetime.fromtimestamp(mktime(entry.published_parsed)) if hasattr(entry, 'published_parsed') and entry.published_parsed else datetime.now()
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception:
            continue
    return pd.DataFrame(all_news)

@st.cache_data(show_spinner=False)
def train_rapid_forecast_model(_data, forecast_horizon):
    """
    Trains a simple and fast linear regression model for forecasting.
    """
    if _data.empty or len(_data) < 20:
        st.warning("Not enough data for Rapid Forecast. Minimum 20 data points required.")
        return None, None
    
    try:
        # Feature Engineering
        df = _data[['close']].copy()
        df['time'] = np.arange(len(df.index)) # Time trend
        df['lag_1'] = df['close'].shift(1) # Previous day's close
        df['lag_5'] = df['close'].shift(5) # 5 days ago close
        df.dropna(inplace=True)
        
        # Define X and y
        X = df[['time', 'lag_1', 'lag_5']]
        y = df['close']
        
        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        
        # --- Forecasting ---
        last_features = X.iloc[-1:].copy()
        future_predictions = []
        
        for i in range(forecast_horizon):
            next_pred = model.predict(last_features)[0]
            future_predictions.append(next_pred)
            
            # Update the features for the next prediction
            new_row = last_features.iloc[0].copy()
            new_row['time'] += 1
            new_row['lag_1'] = next_pred # The new lag_1 is our prediction
            # To update lag_5, we need more history, so we'll approximate
            if len(future_predictions) >= 5:
                new_row['lag_5'] = future_predictions[-5]
            else: # If we haven't predicted 5 steps yet, use a mix of history and prediction
                new_row['lag_5'] = y.iloc[-5 + i + 1] if i < 4 else future_predictions[0]

            last_features = pd.DataFrame([new_row])

        future_dates = pd.to_datetime(pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon))
        forecast_df = pd.DataFrame({'Predicted': future_predictions}, index=future_dates)
        
        # --- Backtesting ---
        in_sample_preds = model.predict(X)
        backtest_df = pd.DataFrame({'Actual': y, 'Predicted': in_sample_preds})

        return forecast_df, backtest_df

    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        return None, None


@st.cache_data
def load_and_combine_data(instrument_name):
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
    if get_broker_client() and source_info.get('tradingsymbol'):
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty: live_df.columns = [col.lower() for col in live_df.columns]
    if not live_df.empty:
        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        hist_df.sort_index(inplace=True)
        return hist_df

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError):
        return np.nan

def interpret_indicators(df):
    if df.empty: return {}
    latest = df.iloc[-1].copy(); latest.index = latest.index.str.lower(); interpretation = {}
    rsi = latest.get('rsi_14')
    if rsi is not None: interpretation['RSI (14)'] = "Overbought (Bearish)" if rsi > 70 else "Oversold (Bullish)" if rsi < 30 else "Neutral"
    stoch_k = latest.get('stok_14_3_3')
    if stoch_k is not None: interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
    macd = latest.get('macd_12_26_9'); signal = latest.get('macds_12_26_9')
    if macd is not None and signal is not None: interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    adx = latest.get('adx_14')
    if adx is not None: interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
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
            # Log successful orders
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
        # Create or upload a CSV with 'Symbol' and 'Sector' columns
        return pd.read_csv("sectors.csv")
    except FileNotFoundError:
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to the options chain dataframe."""
    atm_strike = abs(df['STRIKE'] - ltp).idxmin()
    df_styled = df.style.apply(lambda x: ['background-color: #2c3e50' if x.name < atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['CALL', 'CALL LTP']], axis=1)\
                         .apply(lambda x: ['background-color: #2c3e50' if x.name > atm_strike else '' for i in x], subset=pd.IndexSlice[:, ['PUT', 'PUT LTP']], axis=1)
    return df_styled

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    st.subheader(f"Most Active {underlying} Options (By Volume)")
    with st.spinner("Fetching data..."):
        active_df = get_most_active_options(underlying, instrument_df)
        if not active_df.empty:
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not retrieve data for most active options.")

def get_most_active_options(underlying, instrument_df):
    client = get_broker_client()
    if not client:
        st.toast("Broker not connected.", icon="⚠️")
        return pd.DataFrame()
    
    try:
        # Get chain for nearest expiry
        chain_df, expiry, _, _ = get_options_chain(underlying, instrument_df)
        if chain_df.empty or expiry is None:
            return pd.DataFrame()

        # Get all symbols from the chain
        ce_symbols = chain_df['CALL'].dropna().tolist()
        pe_symbols = chain_df['PUT'].dropna().tolist()
        all_symbols = [f"NFO:{s}" for s in ce_symbols + pe_symbols]

        if not all_symbols:
            return pd.DataFrame()

        # Get detailed quotes
        quotes = client.quote(all_symbols)
        
        active_options = []
        for symbol, data in quotes.items():
            prev_close = data['ohlc']['close']
            last_price = data['last_price']
            change = last_price - prev_close
            pct_change = (change / prev_close * 100) if prev_close != 0 else 0
            
            active_options.append({
                'Symbol': data['tradingsymbol'],
                'LTP': last_price,
                'Change %': pct_change,
                'Volume': data['volume'],
                'OI': data['open_interest']
            })
        
        df = pd.DataFrame(active_options)
        df_sorted = df.sort_values(by='Volume', ascending=False)
        return df_sorted.head(10)

    except Exception as e:
        st.error(f"Could not fetch most active options: {e}")
        return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS ============
def page_pulse():
    display_header()
    st.title("Pre-Market Pulse")

    # --- Global Market Cues ---
    st.subheader("Global Market Cues")
    with st.spinner("Fetching live global indices..."):
        indices_df = get_global_indices_data()
        if not indices_df.empty:
            cols = st.columns(len(indices_df))
            for i, col in enumerate(cols):
                with col:
                    row = indices_df.iloc[i]
                    change = row['Change']
                    st.metric(
                        label=row['Name'],
                        value=f"{row['Price']:,.2f}",
                        delta=f"{change:,.2f} ({row['% Change']:.2f}%)"
                    )
        else:
            st.warning("Could not fetch global market data.")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        # --- GIFT Nifty Tracker ---
        st.subheader("GIFT Nifty Live Chart")
        try:
            # Using a common futures ticker for Nifty, adjust if necessary
            gift_nifty = yf.Ticker("NIFTY=F") 
            nifty_data = gift_nifty.history(period="1d", interval="5m")
            if not nifty_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nifty_data.index, y=nifty_data['Close'], mode='lines', name='GIFT Nifty'))
                template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                fig.update_layout(
                    title="GIFT Nifty Real-time Price",
                    yaxis_title='Price',
                    xaxis_title='Time',
                    template=template,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not fetch live GIFT Nifty data. It may be outside trading hours.")
        except Exception as e:
            st.error(f"Error fetching GIFT Nifty chart: {e}")

    with col2:
        # --- Top Financial News ---
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
    instrument_df = get_instrument_df()

    if 'watchlists' not in st.session_state or not st.session_state.watchlists.get(st.session_state.get('active_watchlist')):
        st.info("Please add stocks to your watchlist on the Dashboard page to use the AI Discovery Engine.")
        return

    active_watchlist = st.session_state.watchlists.get(st.session_state.get('active_watchlist', 'Watchlist 1'), [])
    watchlist_symbols = [item['symbol'] for item in active_watchlist]

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Automated Pattern Scan")
        with st.spinner(f"Scanning {len(watchlist_symbols)} stocks in your watchlist..."):
            all_signals = []
            for symbol in watchlist_symbols:
                token = get_instrument_token(symbol, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period='6mo')
                    if not data.empty and 'RSI_14' in data.columns:
                        latest_rsi = data['RSI_14'].iloc[-1]
                        if latest_rsi > 70:
                            all_signals.append({'symbol': symbol, 'signal': 'RSI Overbought', 'type': 'Bearish', 'value': f'RSI: {latest_rsi:.2f}'})
                        elif latest_rsi < 30:
                            all_signals.append({'symbol': symbol, 'signal': 'RSI Oversold', 'type': 'Bullish', 'value': f'RSI: {latest_rsi:.2f}'})
            
            if all_signals:
                st.success(f"Found {len(all_signals)} potential signal(s)!")
                signals_df = pd.DataFrame(all_signals)
                st.dataframe(signals_df, use_container_width=True, hide_index=True)
                st.session_state['ai_signals'] = all_signals
            else:
                st.info("No strong technical patterns (RSI Overbought/Oversold) found in your watchlist at the moment.")
                st.session_state['ai_signals'] = []

    with col2:
        st.subheader("Data-Driven Trade of the Day")
        if not st.session_state.get('ai_signals'):
            st.info("Run the pattern scan first to generate a trade suggestion.")
            return

        with st.spinner("Analyzing signals and news sentiment for confluence..."):
            best_trade = None
            highest_confluence_score = 0

            for signal in st.session_state['ai_signals']:
                symbol = signal['symbol']
                signal_type = signal['type']

                news_df = fetch_and_analyze_news(symbol)
                if not news_df.empty:
                    avg_sentiment = news_df['sentiment'].mean()

                    is_bullish_confluence = signal_type == 'Bullish' and avg_sentiment > 0.1
                    is_bearish_confluence = signal_type == 'Bearish' and avg_sentiment < -0.1

                    if is_bullish_confluence or is_bearish_confluence:
                        confluence_score = abs(avg_sentiment)
                        if confluence_score > highest_confluence_score:
                            highest_confluence_score = confluence_score
                            best_trade = {
                                'symbol': symbol,
                                'signal_type': "BULLISH" if is_bullish_confluence else "BEARISH",
                                'technical_reason': signal['signal'],
                                'sentiment_reason': f"News Sentiment: {avg_sentiment:.2f} ({'Positive' if avg_sentiment > 0 else 'Negative'})",
                                'combined_reason': f"{signal['signal']} & {'Positive' if avg_sentiment > 0 else 'Negative'} News"
                            }
            
            if best_trade:
                card_color = "#28a745" if best_trade['signal_type'] == "BULLISH" else "#FF4B4B"
                with st.container(border=True):
                    st.markdown(f"#### {best_trade['symbol']} <span style='color:{card_color}; float:right;'>{best_trade['signal_type']}</span>", unsafe_allow_html=True)
                    st.caption(f"**Reason:** {best_trade['combined_reason']}")
                    st.markdown(f"- **Technical:** {best_trade['technical_reason']}")
                    st.markdown(f"- **Sentiment:** {best_trade['sentiment_reason']}")
                    
                    st.divider()
                    
                    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
                    quantity = b_col1.number_input("Quantity", min_value=1, step=1, key=f"trade_card_qty_{best_trade['symbol']}", label_visibility="collapsed")
                    if b_col2.button("Buy", use_container_width=True, key=f"trade_card_buy_{best_trade['symbol']}"):
                        place_order(instrument_df, best_trade['symbol'], quantity, 'MARKET', 'BUY', 'MIS')
                    if b_col3.button("Sell", use_container_width=True, key=f"trade_card_sell_{best_trade['symbol']}"):
                        place_order(instrument_df, best_trade['symbol'], quantity, 'MARKET', 'SELL', 'MIS')
            else:
                st.info("No strong confluence of technical signals and news sentiment found to suggest a high-probability trade.")

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
                st.dataframe(holdings_df, use_container_width=True, hide_index=True)

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
    display_header()
    st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return

    # Chart Layout Selector
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
            if not data.empty:
                st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True, key=f"chart_{index}")
            else:
                st.warning(f"No chart data for {ticker}.")
        else:
            st.error(f"Ticker '{ticker}' not found.")

    if num_charts == 1:
        display_chart_widget(0)
    elif num_charts == 2:
        cols = st.columns(2)
        with cols[0]:
            display_chart_widget(0)
        with cols[1]:
            display_chart_widget(1)
    elif num_charts == 4:
        row1 = st.columns(2)
        with row1[0]:
            display_chart_widget(0)
        with row1[1]:
            display_chart_widget(1)
        st.markdown("---")
        row2 = st.columns(2)
        with row2[0]:
            display_chart_widget(2)
        with row2[1]:
            display_chart_widget(3)

def page_options_hub():
    display_header(); st.title("Options Hub"); instrument_df = get_instrument_df()
    if instrument_df.empty: st.info("Please connect to a broker to use the Options Hub."); return
    
    col1, col2 = st.columns([1, 2]);
    
    with col1:
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "CRUDEOIL", "USDINR"])
        
        if st.button("Check Most Active Options", use_container_width=True):
            show_most_active_dialog(underlying, instrument_df)

        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        
        if available_expiries:
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries, format_func=lambda d: d.strftime('%d %b %Y'))
            if selected_expiry != expiry:
                chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
        else:
            st.warning(f"No upcoming expiries found for {underlying}.")

        if not chain_df.empty and underlying_ltp > 0 and expiry:
            st.subheader("Greeks & Quick Trade")
            option_list = ["-Select-"] + chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist()
            option_selection = st.selectbox("Analyze or Trade an Option", option_list)

            if option_selection and option_selection != "-Select-":
                # Greeks calculation
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
                
                # Quick Trade Form
                with st.form(key="option_trade_form"):
                    q_cols = st.columns([1,1,1])
                    quantity = q_cols[0].number_input("Lots", min_value=1, step=1, key="opt_qty")
                    buy_btn = q_cols[1].form_submit_button("Buy")
                    sell_btn = q_cols[2].form_submit_button("Sell")

                    if buy_btn:
                        place_order(instrument_df, option_selection, quantity * option_details['lot_size'], 'MARKET', 'BUY', 'MIS')
                    if sell_btn:
                        place_order(instrument_df, option_selection, quantity * option_details['lot_size'], 'MARKET', 'SELL', 'MIS')

    with col2:
        st.subheader(f"{underlying} Options Chain")
        if not chain_df.empty and expiry:
            st.caption(f"Expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')} | Spot: {underlying_ltp:,.2f}")
            st.dataframe(style_option_chain(chain_df, underlying_ltp), use_container_width=True, hide_index=True)
        else:
            st.warning("Could not fetch options chain.")

def page_alpha_engine():
    display_header(); st.title("Alpha Engine: News Sentiment"); query = st.text_input("Enter a stock, commodity, or currency to analyze", "NIFTY")
    with st.spinner("Fetching and analyzing news..."):
        news_df = fetch_and_analyze_news(query)
        if not news_df.empty:
            avg_sentiment = news_df['sentiment'].mean(); sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            st.metric(f"Overall News Sentiment for '{query}'", sentiment_label, f"{avg_sentiment:.3f}")
            st.dataframe(news_df.drop(columns=['date']), use_container_width=True, hide_index=True, column_config={"link": st.column_config.LinkColumn("Link", display_text="Read Article")})
        else:
            st.info(f"No recent news found for '{query}'.")

def page_portfolio_and_risk():
    display_header(); st.title("Portfolio & Risk")
    if not get_broker_client(): st.info("Connect to a broker to view your portfolio and positions."); return
    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Live Order Book"])
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True); st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        else:
            st.info("No open positions for the day.")
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        else:
            st.info("No holdings found.")
    with tab3:
        st.subheader("Today's Order Book")
        client = get_broker_client()
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

def page_forecasting_ml():
    display_header()
    st.title("Rapid Forecast")
    st.info("A fast, momentum-based forecast using Linear Regression. Ideal for short-term trend analysis.", icon="ℹ️")
    
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
            with st.spinner("Training model and forecasting..."):
                forecast_df, backtest_df = train_rapid_forecast_model(data, forecast_horizon)
                
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
                # --- Calculate Metrics ---
                mape = mean_absolute_percentage_error(backtest_df['Actual'], backtest_df['Predicted']) * 100
                accuracy = 100 - mape
                
                cum_returns = (1 + backtest_df['Actual'].pct_change().fillna(0)).cumprod()
                peak = cum_returns.cummax()
                drawdown = (cum_returns - peak) / peak
                max_drawdown = 0 if drawdown.empty else drawdown.min() * 100

                # --- Display Metrics ---
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy:.2f}%")
                m2.metric("MAPE", f"{mape:.2f}%")
                m3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                # --- Display Chart ---
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
                
                # General keywords
                elif any(word in prompt_lower for word in ["holdings", "investments"]):
                    _, holdings_df, _, _ = get_portfolio(); response = f"Here are your current holdings:\n```\n{tabulate(holdings_df, headers='keys', tablefmt='psql')}\n```" if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio(); response = f"Your total P&L is ₹{total_pnl:,.2f}. Here are your positions:\n```\n{tabulate(positions_df, headers='keys', tablefmt='psql')}\n```" if not positions_df.empty else "You have no open positions."
                elif "orders" in prompt_lower:
                    orders = client.orders(); response = f"Here are today's orders:\n```\n{tabulate(pd.DataFrame(orders), headers='keys', tablefmt='psql')}\n```" if orders else "You have no orders for the day."
                elif any(word in prompt_lower for word in ["funds", "margin", "balance"]):
                    funds = client.margins(); response = f"Available Funds:\n- Equity: ₹{funds['equity']['available']['live_balance']:,.2f}\n- Commodity: ₹{funds['commodity']['available']['live_balance']:,.2f}"
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
                
                # Advanced Assistant Features
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

                # Options-specific keywords
                elif "option chain for" in prompt_lower:
                    underlying = prompt.split("for")[-1].strip().upper()
                    chain_df, _, ltp, _ = get_options_chain(underlying, instrument_df)
                    if not chain_df.empty:
                        response = f"Here is the current option chain for {underlying} (LTP: {ltp:,.2f}):\n```\n{tabulate(chain_df.head(5), headers='keys', tablefmt='psql')}\n...\n{tabulate(chain_df.tail(5), headers='keys', tablefmt='psql')}\n```"
                    else:
                        response = f"Sorry, I could not fetch the option chain for {underlying}."
                
                elif "greeks for" in prompt_lower or "iv for" in prompt_lower:
                    try:
                        option_symbol = re.search(r'\b([A-Z]+)(\d{2}[A-Z]{3}\d+)\b', prompt.upper()).group(0)
                        if option_symbol:
                            option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                            _, expiry, underlying_ltp, _ = get_options_chain(option_details['name'], instrument_df, option_details['expiry'].date())
                            
                            ltp_data = client.ltp(f"NFO:{option_symbol}")
                            ltp = ltp_data[f"NFO:{option_symbol}"]['last_price']
                            T = max((option_details['expiry'].date() - datetime.now().date()).days, 0) / 365.0
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
                    place_basket_order(st.session_state.basket, variety="regular") # Use "amo" for After Market Orders if needed
                st.session_state.basket = [] # Clear basket after execution
                st.rerun()

            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

def page_portfolio_analytics():
    """A page for advanced portfolio analysis and visualization."""
    display_header()
    st.title("Portfolio Analytics")

    _, holdings_df, _, total_investment = get_portfolio()
    sector_df = get_sector_data()

    if holdings_df.empty:
        st.info("No holdings found to analyze. Please check your portfolio.")
        return
        
    if sector_df is None:
        st.warning("`sectors.csv` not found. Cannot perform sector-wise analysis. Please create this file.")

    # Calculate current value and merge with sector data
    holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
    
    if sector_df is not None:
        holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left')
        holdings_df['Sector'].fillna('Uncategorized', inplace=True)

    st.metric("Total Portfolio Value", f"₹{holdings_df['current_value'].sum():,.2f}")

    col1, col2 = st.columns(2)

    with col1:
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
        with col2:
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

def page_option_strategy_builder():
    """A tool to build and visualize option strategy payoffs."""
    display_header()
    st.title("Options Strategy Builder")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the strategy builder.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Strategy Configuration")
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        strategy = st.selectbox("Select Strategy", ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", "Short Straddle", "Iron Condor"])
        
        _, _, underlying_ltp, _ = get_options_chain(underlying, instrument_df)
        st.metric(f"{underlying} Spot Price", f"{underlying_ltp:,.2f}")

        # Strategy leg inputs
        if strategy in ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", "Short Straddle"]:
            strike1 = st.number_input("Strike Price 1 (K1)", value=int(round(underlying_ltp, -2)))
            premium1 = st.number_input("Premium 1", min_value=0.0)
        
        if strategy in ["Bull Call Spread", "Bear Put Spread"]:
            strike2 = st.number_input("Strike Price 2 (K2)", value=int(round(underlying_ltp, -2)) + 100)
            premium2 = st.number_input("Premium 2", min_value=0.0)

        if strategy == "Short Straddle":
            premium2 = st.number_input("Premium 2 (Put)", min_value=0.0)

        if strategy == "Iron Condor":
            k1 = st.number_input("K1 (Long Put)", value=int(round(underlying_ltp, -2)) - 200)
            p1 = st.number_input("Premium K1", min_value=0.0)
            k2 = st.number_input("K2 (Short Put)", value=int(round(underlying_ltp, -2)) - 100)
            p2 = st.number_input("Premium K2", min_value=0.0)
            k3 = st.number_input("K3 (Short Call)", value=int(round(underlying_ltp, -2)) + 100)
            p3 = st.number_input("Premium K3", min_value=0.0)
            k4 = st.number_input("K4 (Long Call)", value=int(round(underlying_ltp, -2)) + 200)
            p4 = st.number_input("Premium K4", min_value=0.0)

    with col2:
        st.subheader("Payoff Diagram")
        
        if underlying_ltp > 0:
            s_range = np.arange(underlying_ltp * 0.9, underlying_ltp * 1.1, 1) # Price range for plot
            payoff = np.zeros_like(s_range)

            try:
                if strategy == "Long Call":
                    payoff = np.maximum(s_range - strike1, 0) - premium1
                elif strategy == "Long Put":
                    payoff = np.maximum(strike1 - s_range, 0) - premium1
                elif strategy == "Bull Call Spread":
                    payoff = (np.maximum(s_range - strike1, 0) - premium1) - (np.maximum(s_range - strike2, 0) - premium2)
                elif strategy == "Bear Put Spread":
                    payoff = (np.maximum(strike2 - s_range, 0) - premium2) - (np.maximum(strike1 - s_range, 0) - premium1)
                elif strategy == "Short Straddle":
                    payoff_call = -(np.maximum(s_range - strike1, 0) - premium1)
                    payoff_put = -(np.maximum(strike1 - s_range, 0) - premium2)
                    payoff = payoff_call + payoff_put
                elif strategy == "Iron Condor":
                    p_long_put = np.maximum(k1 - s_range, 0) - p1
                    p_short_put = -(np.maximum(k2 - s_range, 0) - p2)
                    p_short_call = -(np.maximum(s_range - k3, 0) - p3)
                    p_long_call = np.maximum(s_range - k4, 0) - p4
                    payoff = p_long_put + p_short_put + p_short_call + p_long_call

                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s_range, y=payoff, mode='lines', name='Payoff', line=dict(color='cyan')))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                max_profit = payoff.max()
                max_loss = payoff.min()
                
                fig.update_layout(
                    title=f"{strategy} Payoff",
                    xaxis_title="Underlying Price at Expiry",
                    yaxis_title="Profit / Loss",
                    template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                c1, c2 = st.columns(2)
                c1.metric("Maximum Profit", f"₹{max_profit:,.2f}")
                c2.metric("Maximum Loss", f"₹{max_loss:,.2f}")

            except NameError:
                st.info("Enter premiums and strike prices to see the payoff chart.")

@st.dialog("Hourly Check-in")
def journal_prompt():
    """The pop-up dialog for the hourly journal entry."""
    st.write("Let's log your current trading mindset to track performance.")
    
    focus = st.text_input("Primary market focus right now?", placeholder="e.g., NIFTY Options, RELIANCE")
    bias = st.selectbox("Current market bias?", ["Bullish", "Bearish", "Neutral"])
    emotion_scale = {
        "1: Disciplined, Calm": 1, 
        "2: Focused": 2, 
        "3: Neutral": 3, 
        "4: Anxious, Hopeful": 4, 
        "5: Stressed, Fearful": 5
    }
    emotion = st.select_slider("Emotional state?", options=emotion_scale.keys())
    
    if st.button("Log Entry", use_container_width=True):
        entry = {
            "Timestamp": datetime.now(pytz.timezone('Asia/Kolkata')),
            "Focus": focus,
            "Bias": bias,
            "Emotion Score": emotion_scale[emotion]
        }
        st.session_state.journal_entries.append(entry)
        st.session_state.last_journal_prompt = datetime.now(pytz.timezone('Asia/Kolkata'))
        st.rerun()

def page_journal_assistant():
    """A page to log and review trading mindset and focus."""
    display_header()
    st.title("Trading Journal & Focus Assistant")

    st.info("""
        This assistant helps you build discipline by prompting you for a quick journal entry every hour during your session. 
        Over time, this log will reveal powerful insights into how your focus and emotional state impact your trading results.
    """)

    # --- Pop-up Trigger Logic ---
    # Initialize session state variables if they don't exist
    if 'last_journal_prompt' not in st.session_state:
        st.session_state.last_journal_prompt = datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(hours=1) # Ensure first prompt
    if 'journal_entries' not in st.session_state:
        st.session_state.journal_entries = []

    # Check if an hour has passed since the last prompt
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    if now - st.session_state.last_journal_prompt > timedelta(hours=1):
        journal_prompt()
        
    st.divider()
    
    # --- Performance Review Dashboard ---
    st.subheader("Performance Review Dashboard")

    if not st.session_state.journal_entries:
        st.write("Your journal is empty. The first hourly prompt will appear soon. You can also add a manual entry below.")
    else:
        journal_df = pd.DataFrame(st.session_state.journal_entries)
        journal_df['Timestamp'] = pd.to_datetime(journal_df['Timestamp'])

        st.subheader("Your Emotional State Over Time")
        st.caption("Track when you feel stressed or disciplined. Are there patterns?")
        st.bar_chart(journal_df, x="Timestamp", y="Emotion Score", color="#ff4b4b")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Market Bias Distribution")
            bias_counts = journal_df['Bias'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=bias_counts.index, values=bias_counts.values, hole=.3)])
            fig.update_layout(
                showlegend=True, 
                template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white',
                annotations=[dict(text='Bias', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Primary Focus Distribution")
            focus_counts = journal_df['Focus'].value_counts()
            fig2 = go.Figure(data=[go.Pie(labels=focus_counts.index, values=focus_counts.values, hole=.3)])
            fig2.update_layout(
                showlegend=True, 
                template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white',
                annotations=[dict(text='Focus', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    st.divider()
    st.subheader("Journal Log")
    if st.session_state.journal_entries:
        st.dataframe(
            pd.DataFrame(st.session_state.journal_entries).sort_values(by="Timestamp", ascending=False),
            use_container_width=True
        )

    with st.expander("Add a Manual Journal Entry"):
        with st.form("manual_journal_form"):
            focus = st.text_input("Primary market focus?", key="manual_focus")
            bias = st.selectbox("Market bias?", ["Bullish", "Bearish", "Neutral"], key="manual_bias")
            emotion_scale = {"1: Disciplined": 1, "2: Focused": 2, "3: Neutral": 3, "4: Anxious": 4, "5: Stressed": 5}
            emotion = st.select_slider("Emotional state?", options=emotion_scale.keys(), key="manual_emotion")
            
            if st.form_submit_button("Add Manual Entry"):
                entry = {
                    "Timestamp": datetime.now(pytz.timezone('Asia/Kolkata')),
                    "Focus": focus,
                    "Bias": bias,
                    "Emotion Score": emotion_scale[emotion]
                }
                st.session_state.journal_entries.append(entry)
                st.success("Manual entry added!")
                st.rerun()

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

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
        try:
            api_key = st.secrets["ZERODHA_API_KEY"]
            api_secret = st.secrets["ZERODHA_API_SECRET"]
        except (FileNotFoundError, KeyError):
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
                st.rerun() # Rerun to trigger the animation
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())

def main_app():
    """The main application interface after successful login."""
    st.markdown(f'<body class="{"light-theme" if st.session_state.get("theme") == "Light" else ""}"></body>', unsafe_allow_html=True)

    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = 'Intraday'
    if 'order_history' not in st.session_state: st.session_state.order_history = []
    
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Intraday", "Options"], horizontal=True)
    st.sidebar.divider()
    
    st.sidebar.header("Live Data")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Intraday": {
            "Dashboard": page_dashboard, 
            "Pre-Market Pulse": page_pulse,
            "AI Discovery": page_ai_discovery,
            "Advanced Charting": page_advanced_charting, 
            "Basket Orders": page_basket_orders,
            "Portfolio Analytics": page_portfolio_analytics,
            "Alpha Engine": page_alpha_engine, 
            "Portfolio & Risk": page_portfolio_and_risk, 
            "Forecasting & ML": page_forecasting_ml, 
            "AI Assistant": page_ai_assistant,
            "Journal Assistant": page_journal_assistant,
        },
        "Options": {
            "Options Hub": page_options_hub, 
            "Strategy Builder": page_option_strategy_builder,
            "Portfolio & Risk": page_portfolio_and_risk, 
            "AI Assistant": page_ai_assistant
        }
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["Forecasting & ML", "AI Assistant"]:
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
