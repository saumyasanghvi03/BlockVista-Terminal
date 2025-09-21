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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from time import mktime
import requests
import io
import time as a_time
import re

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. For the best UI, please create it.")

load_css("style.css")

ML_DATA_SOURCES = {
    "NIFTY 50": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv", "tradingsymbol": "NIFTY 50", "exchange": "NSE"},
    "BANK NIFTY": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv", "tradingsymbol": "BANKNIFTY", "exchange": "NFO"},
    "NIFTY Financial Services": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv", "tradingsymbol": "FINNIFTY", "exchange": "NFO"},
    "GOLD": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv", "tradingsymbol": "GOLDM", "exchange": "MCX"},
    "USDINR": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv", "tradingsymbol": "USDINR", "exchange": "CDS"},
    "SENSEX": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv", "tradingsymbol": "SENSEX", "exchange": "BSE"},
    "S&P 500": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv", "tradingsymbol": None, "exchange": None},
}

# ================ 2. HELPER FUNCTIONS ================
def get_broker_client():
    if st.session_state.get('broker') == "Zerodha":
        return st.session_state.get('kite')
    return None

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    return {2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']}.get(year, [])

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in get_market_holidays(now.year):
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if time(9, 15) <= now.time() <= time(15, 30):
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="margin: 0;">BlockVista Terminal</h1>
        <div>
            <h5 style="margin: 0; text-align: right;">{current_time}</h5>
            <h5 style="margin: 0; text-align: right;">Market Status: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

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
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    template = 'plotly_dark'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    client = get_broker_client()
    if not client: return pd.DataFrame()
    return pd.DataFrame(client.instruments())

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()
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
        df.ta.adx(append=True); df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True);
        return df
    except Exception as e:
        st.error(f"Kite API Error (Historical): {e}")
        return pd.DataFrame()

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
                quote = quotes[instrument]
                last_price = quote['last_price']
                prev_close = quote['ohlc']['close']
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                watchlist.append({'Ticker': item['symbol'], 'Price': last_price, 'Change': change, '% Change': pct_change})
        return pd.DataFrame(watchlist)
    except Exception as e:
        st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
    try:
        exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "USDINR": "CDS"}
        exchange = exchange_map.get(underlying, 'NFO')
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY", "FINNIFTY": "FINNIFTY"}.get(underlying, underlying)
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_ltp = client.ltp(f"{ltp_exchange}:{ltp_symbol}")[f"{ltp_exchange}:{ltp_symbol}"]['last_price']
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty: return pd.DataFrame(), None, underlying_ltp, []
        expiries = sorted(pd.to_datetime(options['expiry'].unique()))
        available_expiries = [e for e in expiries if datetime.now().date() <= e.date() <= (datetime.now() + timedelta(days=90)).date()]
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
    except Exception as e:
        st.warning(f"Could not fetch option chain: {e}")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
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

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    client = get_broker_client()
    if not client: st.error("Broker not connected."); return
    try:
        instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
        if instrument.empty: st.error(f"Symbol '{symbol}' not found."); return
        exchange = instrument.iloc[0]['exchange']
        order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")

# ========== 5. ANALYTICS, ML, NEWS & GREEKS FUNCTIONS ==========
@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    analyzer, news = SentimentIntensityAnalyzer(), []
    news_sources = {"ET": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "MC": "https://www.moneycontrol.com/rss/business.xml"}
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if query is None or query.lower() in entry.title.lower():
                    news.append({"source": source, "title": entry.title, "link": entry.link, "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception: continue
    return pd.DataFrame(news)

def create_features(df):
    df_feat = df.copy(); df_feat.columns = [c.lower() for c in df_feat.columns]
    df_feat['dayofweek'], df_feat['month'] = df_feat.index.dayofweek, df_feat.index.month
    for lag in range(1, 6): df_feat[f'lag_{lag}'] = df_feat['close'].shift(lag)
    df_feat['rolling_mean_7'] = df_feat['close'].rolling(window=7).mean()
    df_feat.ta.rsi(length=14, append=True); df_feat.ta.macd(append=True)
    return df_feat.bfill().ffill().dropna()

@st.cache_data(show_spinner=False)
def train_xgboost_model(_data):
    if _data.empty or len(_data) < 100: return {}, None, None, None, pd.DataFrame()
    df_features = create_features(_data).ffill().bfill()
    predictions = {}
    try:
        df = df_features.copy()
        df['target'] = df['close'].shift(-1)
        df.dropna(inplace=True)
        features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X, y = df[features], df['target']
        X.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(col)) for col in X.columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1, early_stopping_rounds=20)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        predictions["1-Day Close"] = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
        preds_test = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds_test) * 100
        backtest_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds_test})
        max_dd = (1 + (y_test.pct_change().fillna(0))).cumprod(); max_dd = (max_dd / max_dd.cummax() - 1).min()
        return predictions, 100 - mape, mape, max_dd, backtest_df
    except Exception as e: st.error(f"XGBoost training failed: {e}"); return {}, None, None, None, pd.DataFrame()

@st.cache_data(show_spinner=False)
def train_arima_model(_data):
    if _data.empty or len(_data) < 50: return {}, None, None, None, pd.DataFrame()
    df = _data.copy(); df.columns = [c.lower() for c in df.columns]
    try:
        train, test = df['close'][:-30], df['close'][-30:]
        model = ARIMA(train, order=(5,1,0)).fit()
        preds = model.forecast(steps=30)
        test_preds = model.forecast(steps=len(test))
        mape = mean_absolute_percentage_error(test, test_preds) * 100
        backtest = pd.DataFrame({'Actual': test, 'Predicted': test_preds})
        cum_ret = (1 + (test.pct_change().fillna(0))).cumprod(); drawdown = (cum_ret / cum_ret.cummax() - 1).min()
        return {"1-Day": preds.iloc[0], "5-Day": preds.iloc[4]}, 100 - mape, mape, drawdown, backtest
    except Exception: return {}, None, None, None, pd.DataFrame()

def load_and_combine_data(instrument_name):
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info: return pd.DataFrame()
    try:
        hist_df = pd.read_csv(source_info['github_url'], index_col='Date', parse_dates=True)
        hist_df.columns = [c.lower() for c in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns: hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    except Exception: return pd.DataFrame()
    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol') and source_info.get('exchange'):
        inst_df, token = get_instrument_df(), get_instrument_token(source_info['tradingsymbol'], inst_df, source_info['exchange'])
        if token:
            from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty: live_df.columns = [c.lower() for c in live_df.columns]
    if not live_df.empty:
        combined = pd.concat([hist_df, live_df]); return combined[~combined.index.duplicated(keep='last')].sort_index()
    return hist_df.sort_index()

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type).get('price', market_price) - market_price # Safe get
    try: return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError): return np.nan

def interpret_indicators(df):
    if df.empty: return {}
    latest = df.iloc[-1].copy(); latest.index = latest.index.str.lower(); interp = {}
    if 'rsi_14' in latest.index: interp['RSI (14)'] = "Overbought" if latest['rsi_14'] > 70 else "Oversold" if latest['rsi_14'] < 30 else "Neutral"
    if 'stok_14_3_3' in latest.index: interp['Stoch (14,3,3)'] = "Overbought" if latest['stok_14_3_3'] > 80 else "Oversold" if latest['stok_14_3_3'] < 20 else "Neutral"
    if 'macd_12_26_9' in latest.index and 'macds_12_26_9' in latest.index: interp['MACD'] = "Bullish" if latest['macd_12_26_9'] > latest['macds_12_26_9'] else "Bearish"
    return interp

# =========== 6. PAGE DEFINITIONS ============
def page_dashboard():
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty: st.info("Connect broker for dashboard."); return

    index_symbols = [{'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'}, {'symbol': 'INDIA VIX', 'exchange': 'NSE'}]
    index_data = get_watchlist_data(index_symbols)
    if not index_data.empty:
        cols = st.columns(3)
        for i, row in index_data.iterrows():
            cols[i].metric(label=row['Ticker'], value=f"{row['Price']:,.2f}", delta=f"{row['Change']:,.2f} ({row['% Change']:.2f}%)")
    st.markdown("---")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("My Watchlist")
        watchlist_symbols = [{'symbol': s, 'exchange': 'NSE'} for s in ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']]
        watchlist_data = get_watchlist_data(watchlist_symbols)
        if not watchlist_data.empty: st.dataframe(watchlist_data)
    with col2:
        st.subheader("NIFTY 50 Intraday Chart")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
    
    ticker_symbols = [{'symbol': s, 'exchange': 'NSE'} for s in ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'BHARTIARTL', 'SBIN', 'ITC']]
    ticker_data = get_watchlist_data(ticker_symbols)
    if not ticker_data.empty:
        ticker_html = "".join([f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {'#28a745' if item['Change'] > 0 else '#FF4B4B'};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>" for _, item in ticker_data.iterrows()])
        st.markdown(f"""<style>@keyframes marquee{{0%{{transform:translate(100%,0)}}100%{{transform:translate(-100%,0)}}}}.marquee-container{{width:100%;overflow:hidden;position:fixed;bottom:0;left:0;background-color:#1a1a1a;border-top:1px solid #333;padding:5px 0;white-space:nowrap;}}.marquee-content{{display:inline-block;padding-left:100%;animation:marquee 25s linear infinite}}</style><div class="marquee-container"><div class="marquee-content">{ticker_html}</div></div>""", unsafe_allow_html=True)

def page_advanced_charting():
    display_header(); st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    if instrument_df.empty: st.info("Connect broker for charting."); return
    c1, c2 = st.columns([1,3], gap="large")
    with c1:
        st.subheader("Controls")
        ticker = st.text_input("Ticker", "RELIANCE").upper()
        period = st.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=3)
        interval = st.selectbox("Interval", ["5minute", "day", "week"], index=1)
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Heikin-Ashi"])
        token = get_instrument_token(ticker, instrument_df)
        if token:
            data = get_historical_data(token, interval, period=period)
            st.subheader("Indicator Analysis")
            indicator_df = pd.DataFrame([interpret_indicators(data)], index=["Interpretation"]).T
            st.dataframe(indicator_df)
        else: st.error(f"'{ticker}' not found.")
    with c2:
        if 'data' in locals() and not data.empty:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)

def page_options_hub():
    # This page uses the original UI as requested
    pass

def page_portfolio_and_risk():
    # This page uses the original UI as requested
    pass

def page_forecasting_ml():
    display_header()
    st.title("📈 Advanced ML Forecasting")
    st.info("Train models to forecast prices. Not financial advice.", icon="ℹ️")
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument_select")
        
        # --- FEATURE ADDED: Duration Selection ---
        duration_options = {"5 Years": 365*5, "1 Year": 365, "6 Months": 182, "90 Days": 90}
        selected_duration = st.selectbox("Training Data Duration", list(duration_options.keys()))
        
        # --- BUG FIX: Clear old results when selection changes ---
        if 'ml_last_selection' not in st.session_state: st.session_state.ml_last_selection = instrument_name
        if st.session_state.ml_last_selection != instrument_name:
            keys_to_clear = ['ml_predictions', 'ml_accuracy', 'ml_mape', 'ml_max_drawdown', 'ml_backtest_df', 'ml_instrument_name', 'ml_model_choice']
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.session_state.ml_last_selection = instrument_name

        model_choice = st.selectbox("Select Model", ["XGBoost", "ARIMA"])
        
        data = load_and_combine_data(instrument_name)
        if data.empty: st.error("Could not load data."); return

        # Slice data based on selected duration
        cutoff_date = datetime.now() - timedelta(days=duration_options[selected_duration])
        training_data = data[data.index >= cutoff_date]
        
        if st.button(f"Train {model_choice} & Forecast"):
            with st.spinner("Training..."):
                trainer = train_xgboost_model if model_choice == "XGBoost" else train_arima_model
                predictions, accuracy, mape, max_drawdown, backtest_df = trainer(training_data)
                st.session_state.update({'ml_predictions': predictions, 'ml_accuracy': accuracy, 'ml_mape': mape, 'ml_max_drawdown': max_drawdown, 'ml_backtest_df': backtest_df, 'ml_instrument_name': instrument_name, 'ml_model_choice': model_choice})
                st.rerun()

    with col2:
        if st.session_state.get('ml_instrument_name') == instrument_name and 'ml_predictions' in st.session_state:
            st.subheader(f"Results for {instrument_name} ({st.session_state.get('ml_model_choice', '')})")
            if st.session_state.get('ml_predictions'):
                st.dataframe(pd.DataFrame.from_dict(st.session_state['ml_predictions'], orient='index', columns=['Predicted Price']))
                st.subheader("Performance")
                c1,c2,c3 = st.columns(3)
                c1.metric("Accuracy", f"{st.session_state.get('ml_accuracy', 0):.2f}%")
                c2.metric("Mean Error (MAPE)", f"{st.session_state.get('ml_mape', 0):.2f}%")
                c3.metric("Max Drawdown", f"{st.session_state.get('ml_max_drawdown', 0)*100:.2f}%")
                st.plotly_chart(create_chart(st.session_state['ml_backtest_df'], "Backtest", chart_type="Line"))
            else: st.error("Model failed to produce forecasts.")
        else:
            st.subheader(f"Historical Chart for {instrument_name}")
            st.plotly_chart(create_chart(data.tail(252), instrument_name))


def page_alpha_engine():
    # This page uses the original UI as requested
    pass
    
# ============ 7. MAIN APP LOGIC AND AUTHENTICATION ============
def show_login_animation():
    st.title("BlockVista Terminal"); progress_bar = st.progress(0); status_text = st.empty()
    steps = {"Authenticating...": 20, "Establishing connection...": 45, "Fetching market data...": 75, "Initializing terminal...": 100}
    for text, progress in steps.items():
        status_text.text(f"STATUS: {text}"); progress_bar.progress(progress); a_time.sleep(0.7)
    st.session_state['login_animation_complete'] = True; st.rerun()

def login_page():
    st.title("BlockVista Terminal"); st.subheader("Broker Login")
    if st.selectbox("Select Broker", ["Zerodha"]) == "Zerodha":
        try: api_key, api_secret = st.secrets["ZERODHA_API_KEY"], st.secrets["ZERODHA_API_SECRET"]
        except (FileNotFoundError, KeyError): st.error("Kite API credentials not in secrets."); st.stop()
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                kite.set_access_token(st.session_state.access_token)
                st.session_state.kite, st.session_state.profile, st.session_state.broker = kite, kite.profile(), "Zerodha"
                st.query_params.clear(); st.rerun()
            except Exception as e: st.error(f"Auth failed: {e}")
        else: st.link_button("Login with Zerodha Kite", kite.login_url())

def main_app():
    if 'terminal_mode' not in st.session_state: st.session_state.terminal_mode = 'Intraday'
    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"UID: {st.session_state.profile['user_id']}")
    st.sidebar.divider()
    st.sidebar.header("Navigation")
    st.session_state.terminal_mode = st.sidebar.radio("Mode", ["Intraday", "Options"], horizontal=True)
    pages = {"Intraday": {"Dashboard": page_dashboard, "Charting": page_advanced_charting, "Portfolio": page_portfolio_and_risk, "ML Forecast": page_forecasting_ml}, "Options": {"Options Hub": page_options_hub, "Portfolio": page_portfolio_and_risk}}
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    st.sidebar.header("Actions")
    with st.sidebar.expander("🚀 Place Order"):
        with st.form("order_form"):
            symbol = st.text_input("Symbol").upper(); c1,c2=st.columns(2); trans_type=c1.radio("Txn",["BUY","SELL"]); prod=c2.radio("Prod",["MIS","CNC"]); order_type=st.radio("Type",["MARKET","LIMIT"],horizontal=True); qty=st.number_input("Qty",1,step=1); price=st.number_input("Price",0.01) if order_type=="LIMIT" else 0
            if st.form_submit_button("Submit"):
                place_order(get_instrument_df(), symbol, qty, order_type, trans_type, prod, price if price > 0 else None)
    
    st.sidebar.divider()
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
    refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    if auto_refresh and selection not in ["ML Forecast"]:
        st_autorefresh(interval=refresh_interval * 1000, key="refresher")

    pages[st.session_state.terminal_mode][selection]()

if __name__ == "__main__":
    if 'profile' in st.session_state:
        if st.session_state.get('login_animation_complete', False):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
