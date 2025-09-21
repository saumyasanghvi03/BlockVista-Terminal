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
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from tabulate import tabulate
import os
import praw
from time import mktime
from urllib.parse import quote
import requests
import io
import time as a_time # Renaming to avoid conflict with datetime.time

# ==============================================================================
# 1. STYLING AND CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="BlockVista Terminal", layout="wide")

# --- ML Data Configuration ---
# Maps user-friendly names to correct GitHub URLs and broker-specific details
ML_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NFO"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/BANKNIFTY.csv",
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NFO"
    },
    "NIFTY Financial Services": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/NIFTY%20Financial%20Services.csv",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NFO"
    },
    "GOLD": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/GOLD.csv",
        "tradingsymbol": "GOLDM",
        "exchange": "MCX"
    },
    "USDINR": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/USDINR.csv",
        "tradingsymbol": "USDINR",
        "exchange": "CDS"
    },
    "SENSEX": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/SENSEX.csv",
        "tradingsymbol": None, # Not available on Zerodha for live data
        "exchange": None
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/ML%20Interface%20Data/SP500.csv",
        "tradingsymbol": None, # Not available on Zerodha for live data
        "exchange": None
    }
}


def set_blockvista_style(theme='Dark'):
    """ Sets the dark or light theme for the BlockVista Terminal """
    light_text_color = "#FFFFFF" if theme == 'Dark' else "#000000"
    if theme == 'Dark':
        st.markdown(f"""
            <style>
                .main .block-container {{ background-color: #0d1117; color: #c9d1d9; }}
                .stSidebar {{ background-color: #010409; }}
                .stSidebar * {{ color: {light_text_color}; }}
                .stMetric {{ border-left: 3px solid #58a6ff; padding-left: 10px; }}
                /* Make input text readable in Dark Mode */
                .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{
                    color: #FFFFFF !important;
                    -webkit-text-fill-color: #FFFFFF !important;
                }}
            </style>
        """, unsafe_allow_html=True)
    else: # Light Theme
        st.markdown("""
            <style>
                .main .block-container {{ background-color: #ffffff; color: #000000; }}
                .stSidebar {{ background-color: #f0f2f6; }}
                .stMetric {{ border-left: 3px solid #1c64f2; padding-left: 10px; }}
                h1, h2, h3, h4, h5, h6 {{ color: #1c64f2; }}
                /* Ensure input text is readable in Light Mode */
                .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{
                    color: #000000 !important;
                    -webkit-text-fill-color: #000000 !important;
                }}
            </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

# --- Broker Client Abstraction ---
def get_broker_client():
    """Gets the appropriate broker client from session state."""
    broker = st.session_state.get('broker')
    if broker == "Zerodha":
        return st.session_state.get('kite')
    return None

# --- Market Status and Header ---
@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """
    Returns a list of NSE market holidays for a given year.
    NOTE: This list should be updated annually for future years.
    """
    holidays_by_year = {
        2024: [
            '2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29',
            '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17',
            '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15',
            '2024-12-25'
        ],
        2025: [
            '2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18',
            '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05',
            '2025-12-25'
        ],
        2026: [
            '2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01',
            '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25'
        ]
    }
    return holidays_by_year.get(year, []) # Return list for the year, or empty list if not found

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now, market_open_time, market_close_time = datetime.now(ist), time(9, 15), time(15, 30)
    holidays = get_market_holidays(now.year)
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays: return {"status": "Closed", "color": "red"}
    if market_open_time <= now.time() <= market_close_time: return {"status": "Open", "color": "lightgreen"}
    return {"status": "Closed", "color": "red"}

def display_header():
    status_info = get_market_status()
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0;">BlockVista Terminal</h2>
            <div style="text-align: right;">
                <span style="margin: 0;">India | Market Status: <span style="color:{status_info['color']}; font-weight: bold;">{status_info['status']}</span></span>
            </div>
        </div><hr>
    """, unsafe_allow_html=True)

# --- Charting Functions ---
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
    fig = go.Figure()
    if df.empty: return fig
    # Ensure columns are lowercase for charting
    df.columns = [col.lower() for col in df.columns]
    
    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    elif chart_type == 'Line': fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar': fig.add_trace(go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Bar'))
    else: fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    
    # Check for Bollinger Bands columns (case-insensitive)
    bbl_col = next((col for col in df.columns if 'bbl' in col.lower()), None)
    bbu_col = next((col for col in df.columns if 'bbu' in col.lower()), None)
    
    if bbl_col: fig.add_trace(go.Scatter(x=df.index, y=df[bbl_col], line=dict(color='rgba(135, 206, 250, 0.5)', width=1), name='Lower Band'))
    if bbu_col: fig.add_trace(go.Scatter(x=df.index, y=df[bbu_col], line=dict(color='rgba(135, 206, 250, 0.5)', width=1), fill='tonexty', fillcolor='rgba(135, 206, 250, 0.1)', name='Upper Band'))
    
    if forecast_df is not None: fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    
    template = 'plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Broker API Functions ---
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
    if not match.empty: return match.iloc[0]['instrument_token']
    return None

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
            df.set_index('date', inplace=True); df.index = pd.to_datetime(df.index)
            # Add technical indicators
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True); df.ta.coppock(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True); df.ta.macd(append=True); df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
            except Exception as e:
                st.toast(f"Could not calculate some indicators: {e}", icon="⚠️")
            return df
        except Exception as e:
            st.error(f"Kite API Error (Historical): {e}"); return pd.DataFrame()
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
                    watchlist.append({'Ticker': item['symbol'], 'Price': last_price, 'Change': change, '% Change': pct_change})
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️"); return pd.DataFrame()
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
        ce_df, pe_df = chain_df[chain_df['instrument_type'] == 'CE'].copy(), chain_df[chain_df['instrument_type'] == 'PE'].copy()
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
            positions, holdings = client.positions()['net'], client.holdings()
            positions_df, total_pnl = (pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], pd.DataFrame(positions)['pnl'].sum()) if positions else (pd.DataFrame(), 0.0)
            holdings_df, total_investment = (pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], (pd.DataFrame(holdings)['quantity'] * pd.DataFrame(holdings)['average_price']).sum()) if holdings else (pd.DataFrame(), 0.0)
            return positions_df, holdings_df, total_pnl, total_investment
        except Exception as e:
            st.error(f"Kite API Error (Portfolio): {e}"); return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
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
            instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
            if instrument.empty:
                st.error(f"Symbol '{symbol}' not found.")
                return
            exchange = instrument.iloc[0]['exchange']
            order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
            st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
            if 'order_history' not in st.session_state: st.session_state.order_history = []
            st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
        except Exception as e:
            st.toast(f"❌ Order failed: {e}", icon="🔥")
            if 'order_history' not in st.session_state: st.session_state.order_history = []
            st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})
    else:
        st.warning(f"Order placement for {st.session_state.broker} not implemented.")

# --- Analytics, ML, News & Greeks Functions ---
@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    analyzer = SentimentIntensityAnalyzer()
    news_sources = {
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
                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    published_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime.fromtimestamp(mktime(entry.published_parsed))
                    
                    all_news.append({
                        "source": source, 
                        "title": entry.title, 
                        "link": entry.link, 
                        "date": published_date.date(),
                        "sentiment": analyzer.polarity_scores(entry.title)['compound']
                    })
        except Exception:
            continue
    return pd.DataFrame(all_news)

def create_features(df, ticker):
    """Create time series features, technical indicators, and news sentiment from price data."""
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
    
    try:
        df_feat.ta.rsi(length=14, append=True)
        df_feat.ta.macd(append=True)
        df_feat.ta.bbands(length=20, append=True)
        df_feat.ta.atr(length=14, append=True)
        df_feat.ta.stoch(append=True)
    except Exception as e:
        st.toast(f"Could not calculate all TA indicators for features: {e}", icon="⚠️")

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
def train_xgboost_model(_data, ticker):
    """Trains an XGBoost model for multiple forecast horizons."""
    df_features = create_features(_data.copy(), ticker)
    
    horizons = {
        "1-Day Open": ("open", 1), "1-Day Close": ("close", 1),
        "5-Day Close": ("close", 5), "15-Day Close": ("close", 15),
        "30-Day Close": ("close", 30)
    }
    
    predictions = {}
    backtest_results = {}

    for name, (target_col, shift_val) in horizons.items():
        df = df_features.copy()
        target_name = f"target_{name.replace(' ', '_').lower()}"
        df[target_name] = df[target_col].shift(-shift_val)
        df.dropna(subset=[target_name], inplace=True)

        features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume'] and 'target' not in col]
        
        X = df[features]
        y = df[target_name]

        # Ensure X and y are properly aligned before the split
        X = X.loc[y.index]
        
        # New robust check to prevent ValueError from train_test_split
        if len(y) < 5 or X.empty:
            st.warning(f"Not enough data to train for {name}. Skipping.")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
                                 max_depth=4, random_state=42, n_jobs=-1, early_stopping_rounds=20)
        
        model.fit(X_train_scaled, y_train, eval_set=[(scaler.transform(X_test), y_test)], verbose=False)

        # Use the last row of the original, aligned feature data to predict the next price
        last_features_scaled = scaler.transform(X.iloc[-1].values.reshape(1, -1))
        predictions[name] = float(model.predict(last_features_scaled)[0])

        if name == "1-Day Close":
            preds_test = model.predict(scaler.transform(X_test))
            
            # Use MAPE and RMSE, not an incorrect "accuracy"
            mape = mean_absolute_percentage_error(y_test, preds_test) * 100
            rmse = np.sqrt(mean_squared_error(y_test, preds_test))
            
            backtest_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds_test}, index=y_test.index)
            
            cumulative_returns = (1 + (y_test.pct_change().fillna(0))).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            backtest_results = {
                "mape": mape, "rmse": rmse, 
                "max_drawdown": max_drawdown, "backtest_df": backtest_df
            }
            
    return predictions, backtest_results.get("mape"), backtest_results.get("rmse"), backtest_results.get("max_drawdown"), backtest_results.get("backtest_df")

@st.cache_data(show_spinner=False)
def train_arima_model(_data):
    df = _data.copy()
    df.columns = [col.lower() for col in df.columns]
    
    predictions = {}
    
    df_close = df['close']
    if len(df_close) > 50:
        train_data_close, test_data_close = df_close[:-30], df_close[-30:]
        try:
            model_close = ARIMA(train_data_close, order=(5,1,0)).fit()
            forecasts_close = model_close.forecast(steps=30)
            
            predictions["1-Day Close"] = forecasts_close.iloc[0]
            predictions["5-Day Close"] = forecasts_close.iloc[4]
            predictions["15-Day Close"] = forecasts_close.iloc[14]
            predictions["30-Day Close"] = forecasts_close.iloc[29]
            
            preds_test = model_close.forecast(steps=len(test_data_close))
            mape = mean_absolute_percentage_error(test_data_close, preds_test) * 100
            rmse = np.sqrt(mean_squared_error(test_data_close, preds_test))
            backtest_df = pd.DataFrame({'Actual': test_data_close, 'Predicted': preds_test}, index=test_data_close.index)
            max_drawdown = (1 + (test_data_close.pct_change().fillna(0))).cumprod()
            max_drawdown = (max_drawdown / max_drawdown.cummax() - 1).min()

        except Exception:
            mape, rmse, backtest_df, max_drawdown = None, None, None, None

    df_open = df['open']
    if len(df_open) > 50:
        try:
            model_open = ARIMA(df_open, order=(5,1,0)).fit()
            predictions["1-Day Open"] = model_open.forecast(steps=1).iloc[0]
        except Exception:
            pass
    
    return predictions, mape, rmse, max_drawdown, backtest_df

@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads historical data from GitHub and combines it with recent live data."""
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return None

    try:
        url = source_info['github_url']
        response = requests.get(url)
        response.raise_for_status() 

        hist_df = pd.read_csv(io.StringIO(response.text))

        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True)
        hist_df.set_index('Date', inplace=True)
        
        hist_df.columns = [col.lower() for col in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                 hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        hist_df.sort_index(inplace=True)

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Failed to load historical data from GitHub. Please check the URL and your connection. Error: {http_err}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the data: {e}")
        return None

    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol'):
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty:
                live_df.columns = [col.lower() for col in live_df.columns]
    
    if not live_df.empty:
        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        return hist_df

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price, delta = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else: # put
        price, delta = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma, vega = norm.pdf(d1) / (S * sigma * np.sqrt(T)), S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    if T <= 0 or market_price <= 0: return np.nan
    def equation(sigma): return black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try: return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError): return np.nan

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

def page_dashboard():
    display_header()
    st.title("Dashboard")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
        
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'NIFTY BANK', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    if not index_data.empty:
        cols = st.columns(len(index_data))
        for i, col in enumerate(cols):
            index_name = index_data.iloc[i]['Ticker']
            price = index_data.iloc[i]['Price']
            change = index_data.iloc[i]['Change']
            pct_change_val = index_data.iloc[i]['% Change']
            pct_change_str = f"{pct_change_val:.2f}%"
            col.metric(label=index_name, value=f"{price:,.2f}", delta=f"{change:,.2f} ({pct_change_str})")

    st.markdown("---")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            st.subheader("My Watchlist")
            watchlist_symbols = [
                {'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'},
                {'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'},
                {'symbol': 'ICICIBANK', 'exchange': 'NSE'}
            ]
            watchlist_data = get_watchlist_data(watchlist_symbols)
            
            def style_change(val):
                color = '#28a745' if val > 0 else '#FF4B4B' if val < 0 else 'gray'
                return f'color: {color}'

            if not watchlist_data.empty:
                st.dataframe(
                    watchlist_data.style.format({'Price': '₹{:,.2f}', 'Change': '{:,.2f}', '% Change': '{:.2f}%'}).applymap(style_change, subset=['Change', '% Change']),
                    use_container_width=True, hide_index=True
                )

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
    
    st.markdown("<br>", unsafe_allow_html=True)

    ticker_symbols = [
        {'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'TCS', 'exchange': 'NSE'},
        {'symbol': 'HDFCBANK', 'exchange': 'NSE'}, {'symbol': 'ICICIBANK', 'exchange': 'NSE'},
        {'symbol': 'INFY', 'exchange': 'NSE'}, {'symbol': 'BHARTIARTL', 'exchange': 'NSE'},
        {'symbol': 'SBIN', 'exchange': 'NSE'}, {'symbol': 'ITC', 'exchange': 'NSE'}
    ]
    ticker_data = get_watchlist_data(ticker_symbols)
    
    if not ticker_data.empty:
        ticker_html = ""
        for i in range(len(ticker_data)):
            item = ticker_data.iloc[i]
            color = '#28a745' if item['Change'] > 0 else '#FF4B4B'
            ticker_html += f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {color};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>"
        
        st.markdown(f"""
        <style>
            @keyframes marquee {{
                0%  {{ transform: translate(100%, 0); }}
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
                animation: marquee 25s linear infinite;
            }}
        </style>
        <div class="marquee-container">
            <div class="marquee-content">
                {ticker_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_advanced_charting():
    display_header(); st.title("Advanced Charting"); instrument_df = get_instrument_df()
    st.sidebar.header("Chart Controls"); ticker = st.sidebar.text_input("Select Ticker", "RELIANCE").upper(); period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4); interval = st.sidebar.selectbox("Interval", ["5minute", "day", "week"], index=1); chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"])
    if instrument_df.empty: st.info("Please connect to a broker to use the charting tools."); return
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period=period)
        if not data.empty:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)
            st.subheader("Technical Indicator Analysis"); st.dataframe(pd.DataFrame([interpret_indicators(data)], index=["Interpretation"]).T, use_container_width=True)
        else: st.warning(f"No chart data available for {ticker} in the selected period/interval.")
    else: st.error(f"Ticker '{ticker}' not found. Please check the symbol.")

def page_options_hub():
    display_header(); st.title("Options Hub"); instrument_df = get_broker_client()
    if instrument_df is None: st.info("Please connect to a broker to use the Options Hub."); return
    col1, col2 = st.columns([1, 2]);
    with col1:
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "CRUDEOIL", "USDINR"])
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        if available_expiries:
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries, format_func=lambda d: d.strftime('%d %b %Y'))
            if selected_expiry != expiry:
                chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
        else:
            st.warning(f"No upcoming expiries found for {underlying}.")
    with col2:
        st.subheader("Greeks Calculator")
        if not chain_df.empty and underlying_ltp > 0 and expiry:
            option_list = chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist()
            if option_list:
                option_selection = st.selectbox("Select an option to analyze", option_list)
                if option_selection:
                    option_details_df = instrument_df[instrument_df['tradingsymbol'] == option_selection]
                    if not option_details_df.empty:
                        option_details = option_details_df.iloc[0]; strike_price, option_type = option_details['strike'], option_details['instrument_type'].lower()
                        ltp_col = 'CALL LTP' if option_type == 'ce' else 'PUT LTP'; symbol_col = 'CALL' if option_type == 'ce' else 'PUT'; ltp = chain_df[chain_df[symbol_col] == option_selection][ltp_col].iloc[0]
                        T = max((pd.to_datetime(expiry).date() - datetime.now().date()).days, 0) / 365.0; r = 0.07
                        iv = implied_volatility(underlying_ltp, strike_price, T, r, ltp, option_type)
                        if not np.isnan(iv) and iv > 0:
                            greeks = black_scholes(underlying_ltp, strike_price, T, r, iv, option_type)
                            st.metric("Implied Volatility (IV)", f"{iv*100:.2f}%"); c1, c2, c3 = st.columns(3); c1.metric("Delta", f"{greeks['delta']:.4f}"); c2.metric("Gamma", f"{greeks['gamma']:.4f}"); c3.metric("Vega", f"{greeks['vega']:.4f}"); c4, c5, _ = st.columns(3); c4.metric("Theta", f"{greeks['theta']:.4f}"); c5.metric("Rho", f"{greeks['rho']:.4f}")
                        else:
                            st.warning("Could not calculate Implied Volatility for this option (LTP might be zero or expiry too close).")
            else:
                st.warning(f"No options available for {underlying} on the selected expiry.")
    st.subheader(f"{underlying} Options Chain")
    if not chain_df.empty and expiry:
        st.caption(f"Displaying expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')} | Underlying LTP: {underlying_ltp:,.2f}"); st.dataframe(chain_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Could not fetch options chain for {underlying}.")

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
    display_header(); st.title("Portfolio & Risk Journal")
    if not get_broker_client(): st.info("Connect to a broker to view your portfolio and positions."); return
    positions_df, holdings_df, total_pnl, _ = get_portfolio(); tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Trading Journal"])
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
        st.subheader("Trading Journal")
        if 'journal' not in st.session_state: st.session_state.journal = []
        with st.form("journal_form"):
            entry = st.text_area("Log your thoughts or trade ideas:")
            if st.form_submit_button("Add Entry") and entry:
                st.session_state.journal.insert(0, (datetime.now(), entry))
        for ts, text in st.session_state.journal:
            st.info(text); st.caption(f"Logged: {ts.strftime('%d %b %Y, %H:%M')}")

def page_forecasting_ml():
    display_header()
    st.title("📈 Advanced ML Forecasting")
    st.info("Train advanced models using a hybrid of historical and live data to forecast the next closing price. This is for educational purposes and is not financial advice.", icon="ℹ️")

    market_status = get_market_status()
    if market_status['status'] == 'Closed':
        st.warning("The market is currently closed. Forecasts are based on the latest available historical data and will not include today's live price action.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Configuration")
        
        instrument_name = st.selectbox("Select an Instrument for Forecasting", list(ML_DATA_SOURCES.keys()))
        model_choice = st.selectbox("Select a Forecasting Model", ["XGBoost", "ARIMA"])
        
        with st.spinner(f"Loading data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)
        
        if data is None or data.empty or len(data) < 50:
            st.error(f"Could not load sufficient data for {instrument_name} to train a model.")
            st.stop()
            
        if st.button(f"Train {model_choice} Model & Forecast {instrument_name}"):
            with st.spinner(f"Training {model_choice} model... This may take a moment."):
                if model_choice == "XGBoost":
                    predictions, mape, rmse, max_drawdown, backtest_df = train_xgboost_model(data, instrument_name)
                else:
                    predictions, mape, rmse, max_drawdown, backtest_df = train_arima_model(data)
                
                # Store results in session state to display them
                st.session_state['ml_predictions'] = predictions
                st.session_state['ml_mape'] = mape
                st.session_state['ml_rmse'] = rmse
                st.session_state['ml_max_drawdown'] = max_drawdown
                st.session_state['ml_backtest_df'] = backtest_df
                st.session_state['ml_instrument_name'] = instrument_name
                st.session_state['ml_model_choice'] = model_choice
                st.experimental_rerun()

        # Display historical data chart before training
        with col2:
            if 'ml_model_choice' not in st.session_state or st.session_state['ml_instrument_name'] != instrument_name:
                st.subheader(f"Historical Data for {instrument_name}")
                st.plotly_chart(create_chart(data.tail(252), instrument_name), use_container_width=True)
                st.info("Click 'Train Model' to generate a forecast.", icon="💡")
            else:
                st.subheader("Multi-Horizon Forecast")
                if st.session_state.get('ml_predictions'):
                    forecast_df = pd.DataFrame.from_dict(st.session_state['ml_predictions'], orient='index', columns=['Predicted Price'])
                    forecast_df.index.name = "Forecast Horizon"
                    st.dataframe(forecast_df.style.format("₹{:.2f}"))
                else:
                    st.error("Model training failed to produce forecasts.")
                
                st.subheader(f"Model Performance ({st.session_state['ml_model_choice']} Backtest)")
                if st.session_state.get('ml_rmse') is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", f"{st.session_state['ml_rmse']:.2f}")
                    c2.metric("MAPE", f"{st.session_state['ml_mape']:.2f}%")
                    c3.metric("Max Drawdown", f"{st.session_state['ml_max_drawdown']*100:.2f}%")
                    
                    st.subheader("Backtest: Predicted vs. Actual Prices")
                    backtest_df = st.session_state.get('ml_backtest_df')
                    if backtest_df is not None and not backtest_df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual'], mode='lines', name='Actual Price'))
                        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                        template = 'plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
                        fig.update_layout(title=f"{st.session_state['ml_instrument_name']} 1-Day Backtest Results", yaxis_title='Price (INR)', template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not generate performance metrics.")

def page_ai_assistant():
    display_header(); st.title("🤖 Portfolio-Aware Assistant")
    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Ask me about your portfolio or stock prices (e.g., 'what are my holdings?' or 'current price of RELIANCE')."}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Accessing data..."):
                prompt_lower, response = prompt.lower(), "I can provide information on your holdings, positions, orders, funds, and current stock prices. Please ask a specific question."
                client = get_broker_client()
                if not client:
                    response = "I am not connected to your broker. Please log in."
                elif "holdings" in prompt_lower:
                    _, holdings_df, _, _ = get_portfolio()
                    response = "Here are your current holdings:\n```\n" + tabulate(holdings_df, headers='keys', tablefmt='psql') + "\n```" if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    response = f"Here are your current positions. Your total P&L is ₹{total_pnl:,.2f}:\n```\n" + tabulate(positions_df, headers='keys', tablefmt='psql') + "\n```" if not positions_df.empty else "You have no open positions."
                elif "orders" in prompt_lower:
                    if st.session_state.broker == "Zerodha":
                        orders = client.orders()
                        response = "Here are your orders for the day:\n```\n" + tabulate(pd.DataFrame(orders), headers='keys', tablefmt='psql') + "\n```" if orders else "You have no orders for the day."
                    else:
                        response = f"Fetching orders for {st.session_state.broker} is not yet implemented."
                elif "funds" in prompt_lower or "margin" in prompt_lower:
                    if st.session_state.broker == "Zerodha":
                        funds = client.margins()
                        response = f"Here are your available funds:\n- Equity: ₹{funds['equity']['available']['live_balance']:,.2f}\n- Commodity: ₹{funds['commodity']['available']['live_balance']:,.2f}"
                    else:
                        response = f"Fetching funds for {st.session_state.broker} is not yet implemented."
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    try:
                        ticker = prompt.split(" of ")[-1].strip().upper()
                        instrument_df = get_instrument_df()
                        instrument = instrument_df[instrument_df['tradingsymbol'] == ticker]
                        if not instrument.empty:
                            exchange = instrument.iloc[0]['exchange']
                            ltp_data = get_watchlist_data([{'symbol': ticker, 'exchange': exchange}])
                            price = ltp_data.iloc[0]['Price'] if not ltp_data.empty else "N/A"
                            response = f"The current price of {ticker} is {price:,.2f}."
                        else:
                            response = f"I could not find the ticker symbol '{ticker}'. Please check the symbol and try again."
                    except (ValueError, IndexError): response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ============ 7. MAIN APP LOGIC AND AUTHENTICATION ============

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
    
    instrument_df = get_instrument_df()
    # The emoji for "Place Order" has been updated to be a more universal right arrow.
    with st.sidebar.expander("→ Place Order", expanded=False):
        with st.form("order_form"):
            symbol = st.text_input("Symbol").upper()
            c1, c2 = st.columns(2)
            transaction_type = c1.radio("Transaction", ["BUY", "SELL"])
            product = c2.radio("Product", ["MIS", "CNC"])
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            qty = st.number_input("Quantity", min_value=1, step=1)
            price = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0
            
            if st.form_submit_button("Submit Order"):
                if symbol and not instrument_df.empty:
                    place_order(instrument_df, symbol, qty, order_type, transaction_type, product, price if price > 0 else None)
                elif not symbol:
                    st.warning("Please enter a symbol.")
    st.sidebar.divider()
    
    st.sidebar.header("Navigation")
    pages = {
        "Intraday": {"Dashboard": page_dashboard, "Advanced Charting": page_advanced_charting, "Alpha Engine": page_alpha_engine, "Portfolio & Risk": page_portfolio_and_risk, "Forecasting & ML": page_forecasting_ml, "AI Assistant": page_ai_assistant},
        "Options": {"Options Hub": page_options_hub, "Portfolio & Risk": page_portfolio_and_risk, "AI Assistant": page_ai_assistant}
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if auto_refresh and selection != "Forecasting & ML":
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
