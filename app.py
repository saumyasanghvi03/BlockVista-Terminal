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
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import os
import praw
import tweepy

# ==============================================================================
# 1. STYLING AND CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="BlockVista Terminal", layout="wide")

def set_blockvista_style():
    """ Sets the dark theme for the BlockVista Terminal """
    st.markdown("""
        <style>
            .main .block-container {
                padding-top: 2rem; padding-bottom: 2rem;
                padding-left: 3rem; padding-right: 3rem;
            }
            .stMetric {
                border-left: 3px solid #58a6ff; padding-left: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

set_blockvista_style()

# ==============================================================================
# 2. HELPER FUNCTIONS (CHARTS, KITE API, ML, NEWS, AI, GREEKS, MARKET STATUS)
# ==============================================================================

# --- Market Status and Header ---
@st.cache_data(ttl=3600) # Cache holidays for 1 hour
def get_market_holidays(year):
    if year == 2025:
        return ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25']
    if year == 2026:
        return ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
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
    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Bar'))
    else: # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBL_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBU_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), fill='tonexty', fillcolor='rgba(135, 206, 250, 0.1)', name='Upper Band'))
    if forecast_df is not None: fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_ranges_slider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Kite Connect API Functions ---
@st.cache_resource(ttl=3600)
def get_instrument_df(_kite):
    kite = st.session_state.get('kite');
    if not kite: return pd.DataFrame()
    return pd.DataFrame(kite.instruments())

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    if not match.empty: return match.iloc[0]['instrument_token']
    return None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period):
    kite = st.session_state.get('kite')
    if not kite or not instrument_token: return pd.DataFrame()
    to_date, days_to_subtract = datetime.now().date(), {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
    from_date = to_date - timedelta(days=days_to_subtract.get(period, 365))
    try:
        records = kite.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True); df.index = pd.to_datetime(df.index)
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.bbands(length=20, append=True); df.ta.adx(append=True)
        return df
    except Exception as e:
        st.error(f"Kite API Error (Historical): {e}"); return pd.DataFrame()

@st.cache_data(ttl=5)
def get_watchlist_data(symbols_with_tokens, exchange="NSE"):
    kite = st.session_state.get('kite')
    if not kite or not symbols_with_tokens: return pd.DataFrame()
    instrument_names = [f"{exchange}:{item['symbol']}" for item in symbols_with_tokens]
    try:
        ltp_data, quotes, watchlist = kite.ltp(instrument_names), kite.quote(instrument_names), []
        for item in symbols_with_tokens:
            instrument = f"{exchange}:{item['symbol']}"
            if instrument in ltp_data and instrument in quotes:
                last_price, prev_close = ltp_data[instrument]['last_price'], quotes[instrument]['ohlc']['close']
                change, pct_change = last_price - prev_close, (last_price - prev_close) / prev_close * 100 if prev_close != 0 else 0
                watchlist.append({'Ticker': item['symbol'], 'Price': f"₹{last_price:,.2f}", 'Change': f"{change:,.2f}", '% Change': f"{pct_change:.2f}%"})
        return pd.DataFrame(watchlist)
    except Exception as e:
        st.toast(f"Error fetching LTP: {e}", icon="⚠️"); return pd.DataFrame()

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df):
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame(), None, 0.0
    exchange = 'MCX' if underlying in ["GOLDM", "CRUDEOIL", "SILVERM", "NATURALGAS", "USDINR"] else 'NFO'
    underlying_instrument_name = f"NSE:{underlying}" if exchange == 'NFO' and underlying not in ["NIFTY", "BANKNIFTY", "FINNIFTY", "USDINR"] else (f"CDS:{underlying}" if underlying == "USDINR" else f"{exchange}:{underlying}")
    try:
        underlying_ltp = kite.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
    except Exception: underlying_ltp = 0.0
    options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
    if options.empty: return pd.DataFrame(), None, underlying_ltp
    expiries = sorted(options['expiry'].unique())
    nearest_expiry = expiries[0]
    chain_df = options[options['expiry'] == nearest_expiry].sort_values(by='strike')
    ce_df, pe_df = chain_df[chain_df['instrument_type'] == 'CE'].copy(), chain_df[chain_df['instrument_type'] == 'PE'].copy()
    instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
    if not instruments_to_fetch: return pd.DataFrame(), nearest_expiry, underlying_ltp
    quotes = kite.quote(instruments_to_fetch)
    ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
    pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
    final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP']], pe_df[['tradingsymbol', 'strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'})
    return final_chain[['CALL', 'CALL LTP', 'STRIKE', 'PUT LTP', 'PUT']], nearest_expiry, underlying_ltp

@st.cache_data(ttl=10)
def get_portfolio():
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    try:
        positions, holdings = kite.positions()['net'], kite.holdings()
        positions_df, total_pnl = (pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], pd.DataFrame(positions)['pnl'].sum()) if positions else (pd.DataFrame(), 0.0)
        holdings_df, total_investment = (pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], (pd.DataFrame(holdings)['quantity'] * pd.DataFrame(holdings)['average_price']).sum()) if holdings else (pd.DataFrame(), 0.0)
        return positions_df, holdings_df, total_pnl, total_investment
    except Exception as e:
        st.error(f"Kite API Error (Portfolio): {e}"); return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_zerodha_order(symbol, quantity, order_type, transaction_type, product, price=None):
    kite = st.session_state.get('kite')
    try:
        order_id = kite.place_order(tradingsymbol=symbol.upper(), exchange=kite.EXCHANGE_NSE, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=kite.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        if 'order_history' not in st.session_state: st.session_state.order_history = []
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
    except Exception as e:
        st.toast(f"❌ Order failed: {e}", icon="🔥")
        if 'order_history' not in st.session_state: st.session_state.order_history = []
        st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})

# --- Analytics, ML, News & Greeks Functions ---
@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    analyzer, news_sources, all_news = SentimentIntensityAnalyzer(), {"Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml", "Business Standard": "https://www.business-standard.com/rss/markets-102.cms", "Reuters": "http://feeds.reuters.com/reuters/businessNews", "WSJ": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "Bloomberg": "https://feeds.bloomberg.com/wealth/news.rss"}, []
    for source, url in news_sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if query is None or query.lower() in entry.title.lower() or query.lower() in entry.summary.lower():
                all_news.append({"source": source, "title": entry.title, "link": entry.link, "sentiment": analyzer.polarity_scores(entry.title)['compound']})
    return pd.DataFrame(all_news)

@st.cache_data(ttl=900)
def fetch_social_media_sentiment(query):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    # Reddit
    try:
        reddit = praw.Reddit(client_id=st.secrets["REDDIT_CLIENT_ID"], client_secret=st.secrets["REDDIT_CLIENT_SECRET"], user_agent=st.secrets["REDDIT_USER_AGENT"])
        for submission in reddit.subreddit("wallstreetbets+IndianStockMarket").search(query, limit=25):
            sentiment = analyzer.polarity_scores(submission.title)['compound']
            results.append({"source": "Reddit", "text": submission.title, "sentiment": sentiment})
    except Exception as e: st.toast(f"Could not connect to Reddit: {e}", icon="⚠️")
    # Twitter
    try:
        client = tweepy.Client(bearer_token=st.secrets["TWITTER_BEARER_TOKEN"])
        response = client.search_recent_tweets(f'"{query}" lang:en -is:retweet', max_results=25)
        if response.data:
            for tweet in response.data:
                sentiment = analyzer.polarity_scores(tweet.text)['compound']
                results.append({"source": "Twitter", "text": tweet.text, "sentiment": sentiment})
    except Exception as e: st.toast(f"Could not connect to Twitter: {e}", icon="⚠️")
    return pd.DataFrame(results)

def create_features(df):
    df['dayofweek'], df['quarter'], df['month'], df['year'], df['dayofyear'] = df.index.dayofweek, df.index.quarter, df.index.month, df.index.year, df.index.dayofyear
    for lag in range(1, 6): df[f'lag_{lag}'] = df['close'].shift(lag)
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean(); df.dropna(inplace=True)
    return df

@st.cache_data
def train_xgboost_model(_data):
    df = create_features(_data.copy())
    features, target = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']], 'close'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, objective='reg:squarederror', eval_metric='rmse')
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    preds, mape = reg.predict(X_test), mean_absolute_percentage_error(y_test, preds) * 100
    last_features, future_pred = df[features].iloc[-1], reg.predict(pd.DataFrame([last_features]))[0]
    return future_pred, mape

@st.cache_data
def train_arima_model(_data):
    df = _data['close']
    train_data, test_data = df[:-30], df[-30:]
    model, model_fit = ARIMA(train_data, order=(5,1,0)), None
    try: model_fit = model.fit()
    except Exception as e: st.warning(f"ARIMA model failed to converge: {e}"); return None, None
    preds, mape = model_fit.forecast(steps=30), mean_absolute_percentage_error(test_data, preds) * 100
    future_pred = model_fit.forecast(steps=1).iloc[0]
    return future_pred, mape

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) if sigma > 0 and T > 0 else 0
    d2 = d1 - sigma * np.sqrt(T) if sigma > 0 and T > 0 else 0
    if option_type == "call":
        price, delta = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) if T > 0 else 0
    else: # put
        price, delta = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) if T > 0 else 0
    gamma, vega = norm.pdf(d1) / (S * sigma * np.sqrt(T)) if sigma > 0 and T > 0 else 0, S * norm.pdf(d1) * np.sqrt(T) if T > 0 else 0
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365}

def implied_volatility(S, K, T, r, market_price, option_type):
    if T <= 0: return np.nan
    def equation(sigma): return black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try: return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except RuntimeError: return np.nan

def interpret_indicators(df):
    latest, interpretation = df.iloc[-1], {}
    if (rsi := latest.get('RSI_14')) is not None: interpretation['RSI (14)'] = "Overbought (Bearish)" if rsi > 70 else "Oversold (Bullish)" if rsi < 30 else "Neutral"
    if (macd := latest.get('MACD_12_26_9')) is not None and (signal := latest.get('MACDs_12_26_9')) is not None: interpretation['MACD (12,26,9)'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    if (adx := latest.get('ADX_14')) is not None: interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
    return interpretation

# ==============================================================================
# 3. PAGE DEFINITIONS
# ==============================================================================
def page_dashboard():
    display_header(); st.title("Dashboard")
    instrument_df, watchlist_symbols = get_instrument_df(st.session_state.kite), ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']
    watchlist_tokens, nifty_token = [{'symbol': s, 'token': get_instrument_token(s, instrument_df)} for s in watchlist_symbols], get_instrument_token('NIFTY 50', instrument_df, 'NFO')
    watchlist_data, nifty_data, (_, _, total_pnl, total_investment) = get_watchlist_data(watchlist_tokens), get_historical_data(nifty_token, "5minute", "1d"), get_portfolio()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Live Watchlist"); st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
        st.subheader("Portfolio Overview"); st.metric("Total Investment", f"₹{total_investment:,.2f}"); st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}")
    with col2:
        st.subheader("NIFTY 50 Live Chart (5-min)");
        if not nifty_data.empty: st.plotly_chart(create_chart(nifty_data.tail(75), "NIFTY 50"), use_container_width=True)

def page_advanced_charting():
    display_header(); st.title("Advanced Charting")
    instrument_df = get_instrument_df(st.session_state.kite)
    st.sidebar.header("Chart Controls")
    ticker, period, interval, chart_type = st.sidebar.text_input("Select Ticker", "RELIANCE"), st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4), st.sidebar.selectbox("Interval", ["5minute", "15minute", "day", "week"], index=2), st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"])
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period)
        if not data.empty:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)
            st.subheader("Technical Indicator Analysis"); st.dataframe(pd.DataFrame([interpret_indicators(data)], index=["Interpretation"]).T, use_container_width=True)
        else: st.warning(f"No chart data for {ticker}.")
    else: st.error(f"Ticker '{ticker}' not found.")

def page_options_hub():
    display_header(); st.title("Options Hub")
    instrument_df = get_instrument_df(st.session_state.kite)
    underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "SILVERM", "CRUDEOIL", "NATURALGAS", "USDINR"])
    chain_df, expiry, underlying_ltp = get_options_chain(underlying, instrument_df)
    tab1, tab2 = st.tabs(["Options Chain", "Greeks Calculator & Analysis"])
    with tab1:
        st.subheader(f"{underlying} Options Chain")
        if not chain_df.empty: st.caption(f"Displaying nearest expiry: {expiry.strftime('%d %b %Y')}"); st.dataframe(chain_df, use_container_width=True, hide_index=True)
        else: st.warning(f"Could not fetch options chain for {underlying}.")
    with tab2:
        st.subheader("Option Greeks Calculator (Black-Scholes)")
        if not chain_df.empty:
            option_selection = st.selectbox("Select an option to analyze", chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist())
            option_details = instrument_df[instrument_df['tradingsymbol'] == option_selection].iloc[0]
            strike_price, option_type, ltp = option_details['strike'], option_details['instrument_type'].lower(), chain_df[chain_df['CALL']==option_selection]['CALL LTP'].iloc[0] if option_type=='ce' and not chain_df[chain_df['CALL']==option_selection].empty else chain_df[chain_df['PUT']==option_selection]['PUT LTP'].iloc[0]
            days_to_expiry, T, r = (expiry.date() - datetime.now().date()).days, (expiry.date() - datetime.now().date()).days / 365, 0.07
            iv = implied_volatility(underlying_ltp, strike_price, T, r, ltp, option_type)
            if not np.isnan(iv):
                greeks = black_scholes(underlying_ltp, strike_price, T, r, iv, option_type)
                st.metric("Implied Volatility (IV)", f"{iv*100:.2f}%")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Delta", f"{greeks['delta']:.4f}"); col2.metric("Gamma", f"{greeks['gamma']:.4f}"); col3.metric("Vega", f"{greeks['vega']:.4f}"); col4.metric("Theta", f"{greeks['theta']:.4f}")
            else: st.warning("Could not calculate Implied Volatility for this option.")

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
                st.dataframe(news_df, use_container_width=True, hide_index=True)
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
    display_header(); st.title("📈 Advanced ML Forecasting")
    st.info("Train advanced models on historical data to forecast the next closing price. This is for educational purposes and is not financial advice.", icon="ℹ️")
    instrument_df, ticker = get_instrument_df(st.session_state.kite), st.text_input("Enter a stock symbol to analyze", "TCS")
    model_choice = st.selectbox("Select a Forecasting Model", ["XGBoost", "ARIMA"])
    
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, "day", "5y" if model_choice == "XGBoost" else "1y")
        if not data.empty:
            if st.button(f"Train {model_choice} Model & Forecast {ticker}"):
                with st.spinner(f"Training {model_choice} model... This may take a moment."):
                    prediction, accuracy = train_xgboost_model(data) if model_choice == "XGBoost" else train_arima_model(data)
                    if prediction is not None:
                        st.success(f"Model trained successfully!")
                        col1, col2 = st.columns(2)
                        col1.metric(f"Forecasted Next Day's Close for {ticker}", f"₹{prediction:,.2f}")
                        col2.metric("Model Accuracy (MAPE on Test Set)", f"{accuracy:.2f}%")
            
            st.subheader("Historical Data Used for Training"); st.plotly_chart(create_chart(data.tail(252), ticker), use_container_width=True)
        else: st.warning("Could not fetch sufficient data for training.")
    else: st.error("Ticker not found.")

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
                prompt_lower, response = prompt.lower(), "Sorry, I can't answer that. I can provide info on holdings, positions, and stock prices."
                if "holdings" in prompt_lower:
                    _, holdings_df, _, _ = get_portfolio()
                    response = "Here are your current holdings:\n" + holdings_df.to_markdown() if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    response = f"Here are your current positions. Your total P&L is ₹{total_pnl:,.2f}:\n" + positions_df.to_markdown() if not positions_df.empty else "You have no open positions."
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    try:
                        ticker = prompt.split()[-1].upper()
                        ltp_df = get_watchlist_data([{'symbol': ticker}])
                        response = f"The current price of {ticker} is {ltp_df.iloc[0]['Price']}." if not ltp_df.empty else f"Could not fetch price for {ticker}."
                    except (ValueError, IndexError): response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# 4. MAIN APP LOGIC AND AUTHENTICATION
# ==============================================================================

def main():
    if 'kite' in st.session_state:
        st.sidebar.title(f"Welcome {st.session_state.profile['user_name']}")
        st.sidebar.header("Live Data")
        auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
        refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=5, disabled=not auto_refresh)
        if auto_refresh: st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
        with st.sidebar.expander("🚀 Place Order", expanded=False):
            with st.form("order_form"):
                symbol, qty, order_type = st.text_input("Symbol"), st.number_input("Quantity", min_value=1, step=1), st.radio("Order Type", ["MARKET", "LIMIT"])
                price, product, transaction_type = st.number_input("Price", min_value=0.01) if order_type == "LIMIT" else 0, st.radio("Product", ["MIS", "CNC"]), st.radio("Transaction", ["BUY", "SELL"])
                if st.form_submit_button("Submit Order") and symbol: place_zerodha_order(symbol, qty, order_type, transaction_type, product, price if price > 0 else None)
        pages = {"Dashboard": page_dashboard, "Advanced Charting": page_advanced_charting, "Options Hub": page_options_hub, "Alpha Engine": page_alpha_engine, "Portfolio & Risk": page_portfolio_and_risk, "Forecasting & ML": page_forecasting_ml, "AI Assistant": page_ai_assistant}
        st.sidebar.header("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        pages[selection]()
        if st.sidebar.button("Logout"):
            for key in ['access_token', 'kite', 'profile']:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
    else:
        st.title("BlockVista Terminal")
        st.subheader("Zerodha Kite Authentication")
        try:
            api_key, api_secret = st.secrets["ZERODHA_API_KEY"], st.secrets["ZERODHA_API_SECRET"]
        except (FileNotFoundError, KeyError):
            st.error("Kite API credentials not found. Set ZERODHA_API_KEY and ZERODHA_API_SECRET in Streamlit secrets.")
            st.stop()
        kite, request_token = KiteConnect(api_key=api_key), st.query_params.get("request_token")
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token, st.session_state.kite, st.session_state.profile = data["access_token"], kite, kite.profile()
                st.query_params.clear(); st.rerun()
            except Exception as e: st.error(f"Authentication failed: {e}")
        else: st.link_button("Login with Kite", kite.login_url())

if __name__ == "__main__":
    main()
