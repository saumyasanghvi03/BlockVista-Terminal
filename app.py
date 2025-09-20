# app.py
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
import tweepy
from time import mktime
from urllib.parse import quote

# ==============================================================================
# 1. STYLING AND CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="BlockVista Terminal", layout="wide")

# --- ML Data Configuration ---
# Maps user-friendly names to GitHub URLs and broker-specific details
ML_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY_50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NFO"
    },
    "BANK NIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANK_NIFTY.csv",
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
        "tradingsymbol": None, # Not available on Zerodha for live data
        "exchange": None
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
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
            </style>
        """, unsafe_allow_html=True)
    else: # Light Theme
        st.markdown("""
            <style>
                .main .block-container {{ background-color: #ffffff; color: #000000; }}
                .stSidebar {{ background-color: #f0f2f6; }}
                .stMetric {{ border-left: 3px solid #1c64f2; padding-left: 10px; }}
                h1, h2, h3, h4, h5, h6 {{ color: #1c64f2; }}
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
    if year == 2025: return ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    if year == 2026: return ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    return []

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
            df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True)
            df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True); df.ta.cmf(append=True); df.ta.coppock(append=True)
            df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True); df.ta.macd(append=True)
            df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True)
            df.ta.supertrend(append=True); df.ta.willr(append=True)
            return df
        except Exception as e:
            st.error(f"Kite API Error (Historical): {e}"); return pd.DataFrame()
    else:
        st.warning(f"Historical data for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

# ... (Other broker functions like get_watchlist_data, get_options_chain, etc. remain the same)
def get_watchlist_data(symbols_with_tokens, exchange="NSE"):
    client = get_broker_client()
    if not client or not symbols_with_tokens: return pd.DataFrame()

    if st.session_state.broker == "Zerodha":
        instrument_names = [f"{exchange}:{item['symbol']}" for item in symbols_with_tokens]
        try:
            ltp_data, quotes, watchlist = client.ltp(instrument_names), client.quote(instrument_names), []
            for item in symbols_with_tokens:
                instrument = f"{exchange}:{item['symbol']}"
                if instrument in ltp_data and instrument in quotes:
                    last_price, prev_close = ltp_data[instrument]['last_price'], quotes[instrument]['ohlc']['close']
                    change, pct_change = last_price - prev_close, (last_price - prev_close) / prev_close * 100 if prev_close != 0 else 0
                    watchlist.append({'Ticker': item['symbol'], 'Price': f"₹{last_price:,.2f}", 'Change': f"{change:,.2f}", 'Pct Change': f"{pct_change:.2f}"})
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching LTP: {e}", icon="⚠️"); return pd.DataFrame()
    else:
        st.warning(f"Watchlist for {st.session_state.broker} not implemented.")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None, exchange=None):
    client = get_broker_client()
    if not client: return pd.DataFrame(), None, 0.0, []

    if st.session_state.broker == "Zerodha":
        if not exchange:
            exchange = 'MCX' if underlying in ["GOLDM", "CRUDEOIL", "SILVERM", "NATURALGAS"] else 'CDS' if underlying == 'USDINR' else 'NFO'
        
        underlying_instrument_name = f"NSE:{underlying}" if exchange == 'NFO' and underlying not in ["NIFTY", "BANKNIFTY", "FINNIFTY"] else f"{exchange}:{underlying}"
        try:
            ltp_data = client.ltp(underlying_instrument_name)
            underlying_ltp = ltp_data[underlying_instrument_name]['last_price']
        except Exception: underlying_ltp = 0.0

        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty: return pd.DataFrame(), None, underlying_ltp, []

        expiries = sorted(pd.to_datetime(options['expiry'].unique()))
        three_months_later = datetime.now() + timedelta(days=90)
        available_expiries = [e for e in expiries if e.date() >= datetime.now().date() and e.date() <= three_months_later.date()]

        if not expiry_date: expiry_date = available_expiries[0] if available_expiries else None
        if not expiry_date: return pd.DataFrame(), None, underlying_ltp, available_expiries

        chain_df = options[options['expiry'] == expiry_date].sort_values(by='strike')
        ce_df, pe_df = chain_df[chain_df['instrument_type'] == 'CE'].copy(), chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]

        if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        
        quotes = client.quote(instruments_to_fetch)
        ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
        pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
        final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP']], pe_df[['tradingsymbol', 'strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'})
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


def place_order(symbol, quantity, order_type, transaction_type, product, price=None):
    client = get_broker_client()
    if not client: 
        st.error("Broker not connected.")
        return

    if st.session_state.broker == "Zerodha":
        try:
            order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=client.EXCHANGE_NSE, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
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
    analyzer, news_sources, all_news = SentimentIntensityAnalyzer(), {"Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml", "Business Standard": "https://www.business-standard.com/rss/markets-102.cms", "Reuters": "http://feeds.reuters.com/reuters/businessNews", "WSJ": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "Bloomberg": "https://feeds.bloomberg.com/wealth/news.rss"}, []
    for source, url in news_sources.items():
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
    return pd.DataFrame(all_news)

def fetch_social_media_sentiment(query):
    analyzer, results = SentimentIntensityAnalyzer(), []
    try:
        reddit = praw.Reddit(client_id=st.secrets["REDDIT_CLIENT_ID"], client_secret=st.secrets["REDDIT_CLIENT_SECRET"], user_agent=st.secrets["REDDIT_USER_AGENT"])
        for submission in reddit.subreddit("wallstreetbets+IndianStockMarket").search(query, limit=25):
            results.append({"source": "Reddit", "text": submission.title, "sentiment": analyzer.polarity_scores(submission.title)['compound']})
    except Exception as e: st.toast(f"Could not connect to Reddit: {e}", icon="⚠️")
    try:
        client = tweepy.Client(bearer_token=st.secrets["TWITTER_BEARER_TOKEN"])
        response = client.search_recent_tweets(f'"{query}" lang:en -is:retweet', max_results=25)
        if response.data:
            for tweet in response.data:
                results.append({"source": "Twitter", "text": tweet.text, "sentiment": analyzer.polarity_scores(tweet.text)['compound']})
    except Exception as e: st.toast(f"Could not connect to Twitter: {e}", icon="⚠️")
    return pd.DataFrame(results)

def create_features(df, ticker):
    """Create time series features, technical indicators, and news sentiment from price data."""
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Lag features
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['close'].shift(lag)
        
    # Rolling window features
    df['rolling_mean_7'] = df['close'].rolling(window=7).mean()
    df['rolling_std_7'] = df['close'].rolling(window=7).std()
    df['rolling_mean_30'] = df['close'].rolling(window=30).mean()
    df['rolling_std_30'] = df['close'].rolling(window=30).std()

    # Technical Indicators using pandas-ta
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.stoch(append=True)
    
    # --- NEW: Add News Sentiment ---
    news_df = fetch_and_analyze_news(ticker)
    if not news_df.empty:
        news_df['date'] = pd.to_datetime(news_df['date'])
        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].mean().to_frame()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        df = df.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        
        df['sentiment'] = df['sentiment'].fillna(method='ffill')
        df['sentiment_rolling_3d'] = df['sentiment'].rolling(window=3, min_periods=1).mean()
        df.fillna(0, inplace=True)
    else:
        df['sentiment'] = 0
        df['sentiment_rolling_3d'] = 0

    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    df.dropna(inplace=True)
    return df

@st.cache_data(show_spinner=False)
def train_xgboost_model(_data, ticker):
    """Trains an XGBoost model for multiple forecast horizons."""
    df_features = create_features(_data.copy(), ticker)
    
    horizons = {
        "1-Day Open": ("open", 1), "1-Day Close": ("close", 1),
        "5-Day Close": ("close", 5), "15-Day Close": ("close", 15),
        "30-Day Close": ("close", 30), "45-Day Close": ("close", 45),
        "60-Day Close": ("close", 60)
    }
    
    predictions = {}
    backtest_results = {}

    for name, (target_col, shift_val) in horizons.items():
        df = df_features.copy()
        target_name = f"target_{name.replace(' ', '_').lower()}"
        df[target_name] = df[target_col].shift(-shift_val)
        df.dropna(subset=[target_name], inplace=True)

        features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume'] and 'target' not in col]
        
        if len(df) < 100: continue

        X = df[features]
        y = df[target_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
                                 max_depth=4, random_state=42, n_jobs=-1, early_stopping_rounds=20)
        
        model.fit(X_train_scaled, y_train, eval_set=[(scaler.transform(X_test), y_test)], verbose=False)

        last_features_scaled = scaler.transform(X.iloc[-1].values.reshape(1, -1))
        predictions[name] = float(model.predict(last_features_scaled)[0])

        if name == "1-Day Close":
            preds_test = model.predict(scaler.transform(X_test))
            accuracy = 100 - (mean_absolute_percentage_error(y_test, preds_test) * 100)
            rmse = np.sqrt(mean_squared_error(y_test, preds_test))
            backtest_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds_test}, index=y_test.index)
            
            cumulative_returns = (1 + (y_test.pct_change().fillna(0))).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            backtest_results = {
                "accuracy": accuracy, "rmse": rmse, 
                "max_drawdown": max_drawdown, "backtest_df": backtest_df
            }

    return predictions, backtest_results.get("accuracy"), backtest_results.get("rmse"), backtest_results.get("max_drawdown"), backtest_results.get("backtest_df")

@st.cache_data(show_spinner=False)
def train_arima_model(_data):
    df = _data.copy()
    df.columns = [col.lower() for col in df.columns]
    
    predictions = {}
    
    # Forecast Closing Prices
    df_close = df['close']
    if len(df_close) > 50:
        train_data_close, test_data_close = df_close[:-30], df_close[-30:]
        try:
            model_close = ARIMA(train_data_close, order=(5,1,0)).fit()
            forecasts_close = model_close.forecast(steps=60)
            
            predictions["1-Day Close"] = forecasts_close.iloc[0]
            predictions["5-Day Close"] = forecasts_close.iloc[4]
            predictions["15-Day Close"] = forecasts_close.iloc[14]
            predictions["30-Day Close"] = forecasts_close.iloc[29]
            predictions["45-Day Close"] = forecasts_close.iloc[44]
            predictions["60-Day Close"] = forecasts_close.iloc[59]

            # Backtest for 1-day close
            preds_test = model_close.forecast(steps=len(test_data_close))
            accuracy = 100 - (mean_absolute_percentage_error(test_data_close, preds_test) * 100)
            rmse = np.sqrt(mean_squared_error(test_data_close, preds_test))
            backtest_df = pd.DataFrame({'Actual': test_data_close, 'Predicted': preds_test}, index=test_data_close.index)
            max_drawdown = (1 + (test_data_close.pct_change().fillna(0))).cumprod()
            max_drawdown = (max_drawdown / max_drawdown.cummax() - 1).min()

        except Exception:
            accuracy, rmse, backtest_df, max_drawdown = None, None, None, None

    # Forecast Opening Price
    df_open = df['open']
    if len(df_open) > 50:
        try:
            model_open = ARIMA(df_open, order=(5,1,0)).fit()
            predictions["1-Day Open"] = model_open.forecast(steps=1).iloc[0]
        except Exception:
            pass

    return predictions, accuracy, rmse, max_drawdown, backtest_df


@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads historical data from GitHub and combines it with recent live data."""
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()

    # 1. Load historical data from GitHub
    try:
        hist_df = pd.read_csv(source_info['github_url'])
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed') # FIX: Use mixed format parsing
        hist_df.set_index('Date', inplace=True)
        # Standardize column names
        hist_df.columns = [col.lower().replace(' ', '_').replace('%', 'pct') for col in hist_df.columns]
    except Exception as e:
        st.error(f"Failed to load historical data from GitHub. Please check the URL and your connection. Error: {e}")
        return pd.DataFrame()

    # 2. Fetch recent live data if broker is connected and symbol exists
    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol'):
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            # Fetch last year of data to ensure overlap
            from_date = datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty:
                live_df.columns = [col.lower() for col in live_df.columns]


    # 3. Combine and de-duplicate
    if not live_df.empty:
        combined_df = pd.concat([hist_df, live_df])
        # Remove duplicates, keeping the entry from the live data (last one)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        hist_df.sort_index(inplace=True)
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
    latest, interpretation = df.iloc[-1], {}
    df.columns = [col.lower() for col in df.columns]
    if 'rsi_14' in df.columns: interpretation['RSI (14)'] = "Overbought (Bearish)" if df['rsi_14'].iloc[-1] > 70 else "Oversold (Bullish)" if df['rsi_14'].iloc[-1] < 30 else "Neutral"
    if 'stok_14_3_3' in df.columns:
        if df['stok_14_3_3'].iloc[-1] > 80: interpretation['Stochastic (14,3,3)'] = "Overbought (Bearish)"
        elif df['stok_14_3_3'].iloc[-1] < 20: interpretation['Stochastic (14,3,3)'] = "Oversold (Bullish)"
    if 'macd_12_26_9' in df.columns and 'macds_12_26_9' in df.columns:
        if df['macd_12_26_9'].iloc[-1] > df['macds_12_26_9'].iloc[-1]: interpretation['MACD (12,26,9)'] = "Bullish Crossover"
        else: interpretation['MACD (12,26,9)'] = "Bearish Crossover"
    if 'adx_14' in df.columns:
        adx = df['adx_14'].iloc[-1]
        interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
    if 'supertd_7_3.0' in df.columns:
        supertrend_dir = df['supertd_7_3.0'].iloc[-1]
        interpretation['Supertrend (7,3)'] = "Bullish" if supertrend_dir == 1 else "Bearish"
    return interpretation

def get_greek_interpretations(greeks):
    interpretations = {
        "Delta (Δ)": f"This option's price is expected to change by ₹{greeks['delta']:.2f} for every ₹1 change in the underlying stock price. It also represents the approximate probability of the option expiring in-the-money.",
        "Gamma (Γ)": f"For every ₹1 change in the underlying stock price, this option's Delta is expected to change by {greeks['gamma']:.4f}. It measures the rate of change of Delta.",
        "Vega (ν)": f"For every 1% change in implied volatility, this option's price is expected to change by ₹{greeks['vega']:.2f}. Vega is highest for at-the-money options with longer time to expiration.",
        "Theta (Θ)": f"This option's price is expected to decrease by ₹{abs(greeks['theta']):.2f} each day due to the passage of time (time decay).",
        "Rho (ρ)": f"For every 1% change in the risk-free interest rate, this option's price is expected to change by ₹{greeks['rho']:.2f}."
    }
    return interpretations

# ==============================================================================
# 3. PAGE DEFINITIONS
# ==============================================================================
def page_dashboard():
    display_header()
    st.title("Dashboard")
    instrument_df = get_instrument_df()
    watchlist_symbols = ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']
    watchlist_tokens = [{'symbol': s, 'token': get_instrument_token(s, instrument_df)} for s in watchlist_symbols]
    nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NFO')

    watchlist_data = get_watchlist_data(watchlist_tokens)
    nifty_data = get_historical_data(nifty_token, "minute", period="1d")
    _, _, total_pnl, total_investment = get_portfolio()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Live Watchlist")
        st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
        st.subheader("Portfolio Overview")
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
        st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}")

    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        if not nifty_data.empty:
            st.plotly_chart(create_chart(nifty_data.tail(75), "NIFTY 50"), use_container_width=True)
        else:
            st.warning("Could not load NIFTY 50 chart. The market is currently closed or live data is unavailable.")

def page_advanced_charting():
    display_header(); st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    st.sidebar.header("Chart Controls")
    ticker, period, interval, chart_type = st.sidebar.text_input("Select Ticker", "RELIANCE"), st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4), st.sidebar.selectbox("Interval", ["5minute", "day", "week"], index=1), st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"])
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period=period)
        if not data.empty:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)
            st.subheader("Technical Indicator Analysis"); st.dataframe(pd.DataFrame([interpret_indicators(data)], index=["Interpretation"]).T, use_container_width=True)
        else: st.warning(f"No chart data for {ticker}.")
    else: st.error(f"Ticker '{ticker}' not found.")

def page_options_hub():
    display_header(); st.title("Options Hub")
    instrument_df = get_instrument_df()
    
    col1, col2 = st.columns([1,2])
    with col1:
        underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "SILVERM", "CRUDEOIL", "NATURALGAS", "USDINR"])
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        
        selected_expiry = expiry
        if available_expiries:
            selected_expiry = st.selectbox("Select Expiry Date", available_expiries, format_func=lambda d: d.strftime('%d %b %Y'))
            if selected_expiry != expiry:
                chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df, selected_expiry)
        else:
            st.warning("No expiries found for this instrument.")

    with col2:
        st.subheader("Greeks Calculator")
        if not chain_df.empty and underlying_ltp > 0 and expiry:
            option_list = chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist()
            if not option_list:
                st.warning(f"No options available for {underlying} on {expiry.strftime('%d %b %Y')}.")
            else:
                option_selection = st.selectbox("Select an option to analyze", option_list)
                if option_selection:
                    option_details_df = instrument_df[instrument_df['tradingsymbol'] == option_selection]
                    if not option_details_df.empty:
                        option_details = option_details_df.iloc[0]
                        strike_price, option_type = option_details['strike'], option_details['instrument_type'].lower()
                        ltp = 0
                        if option_type == 'ce' and not chain_df[chain_df['CALL'] == option_selection].empty: ltp = chain_df[chain_df['CALL'] == option_selection]['CALL LTP'].iloc[0]
                        elif option_type == 'pe' and not chain_df[chain_df['PUT'] == option_selection].empty: ltp = chain_df[chain_df['PUT'] == option_selection]['PUT LTP'].iloc[0]
                        
                        expiry_date = pd.to_datetime(expiry).date()
                        days_to_expiry = (expiry_date - datetime.now().date()).days
                        T = max(days_to_expiry, 0) / 365.0
                        r = 0.07
                        
                        iv = implied_volatility(underlying_ltp, strike_price, T, r, ltp, option_type)
                        if not np.isnan(iv):
                            greeks = black_scholes(underlying_ltp, strike_price, T, r, iv, option_type)
                            st.metric("Implied Volatility (IV)", f"{iv*100:.2f}%")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Delta", f"{greeks['delta']:.4f}"); c2.metric("Gamma", f"{greeks['gamma']:.4f}"); c3.metric("Vega", f"{greeks['vega']:.4f}")
                            c4, c5, _ = st.columns(3)
                            c4.metric("Theta", f"{greeks['theta']:.4f}"); c5.metric("Rho", f"{greeks['rho']:.4f}")
                        else: st.warning("Could not calculate Implied Volatility for this option.")
    
    st.subheader(f"{underlying} Options Chain")
    if not chain_df.empty and expiry: st.caption(f"Displaying expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')}"); st.dataframe(chain_df, use_container_width=True, hide_index=True)
    else: st.warning(f"Could not fetch options chain for {underlying}.")

def page_alpha_engine():
    display_header(); st.title("Alpha Engine: News & Social Sentiment")
    query = st.text_input("Enter a stock, commodity, or currency to analyze", "NIFTY")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("News Sentiment")
        with st.spinner("Fetching and analyzing news..."):
            news_df = fetch_and_analyze_news(query)
            if not news_df.empty:
                avg_sentiment = news_df['sentiment'].mean()
                sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                st.metric(f"News Sentiment for '{query}'", sentiment_label, f"{avg_sentiment:.3f}")
                st.dataframe(news_df.drop(columns=['date']), use_container_width=True, hide_index=True, column_config={"link": st.column_config.LinkColumn("Link")})
            else: st.info(f"No recent news found for '{query}'.")

    with col2:
        st.subheader("Social Media Sentiment")
        with st.spinner("Fetching and analyzing social media..."):
            social_df = fetch_social_media_sentiment(query)
            if not social_df.empty:
                avg_sentiment = social_df['sentiment'].mean()
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                st.metric(f"Social Media Sentiment for '{query}'", sentiment_label, f"{avg_sentiment:.3f}")
                st.dataframe(social_df, use_container_width=True, hide_index=True)
            else: st.info(f"No recent social media posts found for '{query}'.")

def page_portfolio_and_risk():
    display_header(); st.title("Portfolio & Risk Journal")
    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Trading Journal"])
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty: st.dataframe(positions_df, use_container_width=True, hide_index=True); st.metric("Total Day P&L", f"₹{total_pnl:,.2f}")
        else: st.info("No open positions for the day.")
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty: st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        else: st.info("No holdings found.")
    with tab3:
        st.subheader("Trading Journal")
        if 'journal' not in st.session_state: st.session_state.journal = []
        with st.form("journal_form"):
            entry = st.text_area("Log your thoughts or trade ideas:")
            if st.form_submit_button("Add Entry") and entry: st.session_state.journal.insert(0, (datetime.now(), entry))
        for ts, text in st.session_state.journal:
            st.info(text); st.caption(f"Logged: {ts.strftime('%d %b %H:%M')}")

def page_forecasting_ml():
    display_header()
    st.title("📈 Advanced ML Forecasting")
    st.info("Train advanced models using a hybrid of historical and live data to forecast the next closing price. This is for educational purposes and is not financial advice.", icon="ℹ️")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Configuration")
        
        instrument_name = st.selectbox("Select an Instrument for Forecasting", list(ML_DATA_SOURCES.keys()))
        model_choice = st.selectbox("Select a Forecasting Model", ["XGBoost", "ARIMA"])
        
        with st.spinner(f"Loading data for {instrument_name}..."):
            data = load_and_combine_data(instrument_name)

        if data is not None and not data.empty:
            if st.button(f"Train {model_choice} Model & Forecast {instrument_name}"):
                with st.spinner(f"Training {model_choice} model... This may take a moment."):
                    predictions, accuracy, rmse, max_drawdown, backtest_df = (
                        train_xgboost_model(data, instrument_name) if model_choice == "XGBoost" 
                        else train_arima_model(data)
                    )
                
                with col2:
                    st.subheader("Multi-Horizon Forecast")
                    if predictions:
                        forecast_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Price'])
                        forecast_df.index.name = "Forecast Horizon"
                        st.dataframe(forecast_df.style.format("₹{:.2f}"))
                    else:
                        st.error("Model training failed to produce forecasts.")
                    
                    st.subheader("Model Performance (Based on 1-Day Close)")
                    if accuracy is not None:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Model Accuracy", f"{accuracy:.2f}%")
                        c2.metric("RMSE", f"{rmse:.2f}")
                        c3.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
                        
                        st.subheader("Backtest: Predicted vs. Actual Prices")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual'], mode='lines', name='Actual Price'))
                        fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
                        template = 'plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
                        fig.update_layout(title=f'{instrument_name} 1-Day Backtest Results', yaxis_title='Price (INR)', template=template)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not generate performance metrics.")
            else:
                with col2:
                    st.subheader(f"Historical Data for {instrument_name}")
                    st.plotly_chart(create_chart(data.tail(252), instrument_name), use_container_width=True)
        else:
            st.warning(f"Could not load sufficient data for {instrument_name}.")

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
                        token = get_instrument_token(ticker, instrument_df)
                        if token:
                            ltp_data = get_watchlist_data([{'symbol': ticker, 'token': token}])
                            if not ltp_data.empty:
                                price = ltp_data.iloc[0]['Price']
                                response = f"The current price of {ticker} is {price}."
                            else:
                                response = f"Could not fetch the live price for {ticker}."
                        else:
                            response = f"I could not find the ticker symbol '{ticker}'. Please check the symbol and try again."
                    except (ValueError, IndexError): response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# 4. MAIN APP LOGIC AND AUTHENTICATION
# ==============================================================================

def main():
    if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = 'Intraday'
    set_blockvista_style(st.session_state.theme)

    if 'profile' in st.session_state and 'broker' in st.session_state:
        st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
        st.sidebar.caption(f"Connected via {st.session_state.broker}")
        st.sidebar.header("Terminal Controls")
        st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], key='theme_selector')
        st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Intraday", "Options"], key='mode_selector')

        st.sidebar.header("Live Data")
        auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
        refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=5, disabled=not auto_refresh)
        if auto_refresh: st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
        with st.sidebar.expander("🚀 Place Order", expanded=False):
            with st.form("order_form"):
                symbol, qty, order_type = st.text_input("Symbol"), st.number_input("Quantity", min_value=1, step=1), st.radio("Order Type", ["MARKET", "LIMIT"])
                price, product, transaction_type = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0, st.radio("Product", ["MIS", "CNC"]), st.radio("Transaction", ["BUY", "SELL"])
                if st.form_submit_button("Submit Order") and symbol: place_order(symbol, qty, order_type, transaction_type, product, price if price > 0 else None)
        
        if st.session_state.terminal_mode == "Intraday":
            pages = {"Dashboard": page_dashboard, "Advanced Charting": page_advanced_charting, "Alpha Engine": page_alpha_engine, "Portfolio & Risk": page_portfolio_and_risk, "Forecasting & ML": page_forecasting_ml, "AI Assistant": page_ai_assistant}
        else: # Options Mode
            pages = {"Options Hub": page_options_hub, "Portfolio & Risk": page_portfolio_and_risk, "AI Assistant": page_ai_assistant}
        st.sidebar.header("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()), key='nav_selector')
        
        pages[selection]()
        
        if st.sidebar.button("Logout"):
            for key in ['access_token', 'kite', 'profile', 'messages', 'journal', 'broker']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    else:
        st.title("BlockVista Terminal")
        st.subheader("Broker Login")
        
        broker = st.selectbox("Select Your Broker", ["Zerodha", "Angelone", "Groww", "Motilal Oswal"])

        if broker == "Zerodha":
            try:
                api_key, api_secret = st.secrets["ZERODHA_API_KEY"], st.secrets["ZERODHA_API_SECRET"]
            except (FileNotFoundError, KeyError):
                st.error("Kite API credentials not found. Set ZERODHA_API_KEY and ZERODHA_API_SECRET in Streamlit secrets.")
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
                    
                    st.toast("Broker Login Successful!", icon="✅")
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Authentication failed: {e}")
            else:
                st.link_button("Login with Zerodha Kite", kite.login_url())

        elif broker in ["Angelone", "Groww", "Motilal Oswal"]:
            st.info(f"{broker} integration is a work in progress.")
            st.write(f"To enable {broker} login, you would need to add the respective API keys to your Streamlit secrets and implement their authentication flow here.")
            if st.button(f"Login with {broker} (Placeholder)"):
                st.warning("This button is a placeholder and does not initiate a real login.")


if __name__ == "__main__":
    main()

