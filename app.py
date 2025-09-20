# app.py
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import os

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
# 2. HELPER FUNCTIONS (CHARTS, KITE API, ML, NEWS, AI, GREEKS)
# ==============================================================================

# --- Charting Functions ---
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
    fig = go.Figure()

    # Calculate Heikin-Ashi candles if requested
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
    
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))

    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Kite Connect API Functions ---
@st.cache_resource(ttl=3600)
def get_instrument_df(_kite):
    kite = st.session_state.get('kite')
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
    to_date = datetime.now().date()
    days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
    from_date = to_date - timedelta(days=days_to_subtract.get(period, 365))
    try:
        records = kite.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return df
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.adx(append=True)
        return df
    except Exception as e:
        st.error(f"Kite API Error (Historical): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def get_ltp(symbols_with_tokens, exchange="NSE"):
    kite = st.session_state.get('kite')
    if not kite or not symbols_with_tokens: return pd.DataFrame()
    instrument_names = [f"{exchange}:{item['symbol']}" for item in symbols_with_tokens]
    try:
        ltp_data = kite.ltp(instrument_names)
        quotes = kite.quote(instrument_names)
        watchlist = []
        for item in symbols_with_tokens:
            instrument = f"{exchange}:{item['symbol']}"
            if instrument in ltp_data and instrument in quotes:
                last_price = ltp_data[instrument]['last_price']
                prev_close = quotes[instrument]['ohlc']['close']
                change = last_price - prev_close
                pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                watchlist.append({'Ticker': item['symbol'], 'Price': f"₹{last_price:,.2f}", 'Change': f"{change:,.2f}", '% Change': f"{pct_change:.2f}%"})
        return pd.DataFrame(watchlist)
    except Exception as e:
        st.toast(f"Error fetching LTP: {e}", icon="⚠️")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df):
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame(), None, 0.0
    exchange = 'MCX' if underlying in ["GOLDM", "CRUDEOIL", "SILVERM", "NATURALGAS"] else 'NFO'
    underlying_instrument_name = f"NSE:{underlying}" if exchange == 'NFO' else f"{exchange}:{underlying}"
    try:
        underlying_ltp = kite.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
    except Exception:
        underlying_ltp = 0.0 # Fallback

    options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
    if options.empty: return pd.DataFrame(), None, underlying_ltp
    expiries = sorted(options['expiry'].unique())
    nearest_expiry = expiries[0]
    chain_df = options[options['expiry'] == nearest_expiry].sort_values(by='strike')
    ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
    pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
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
        positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
        total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
        holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
        total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
        return positions_df, holdings_df, total_pnl, total_investment
    except Exception as e:
        st.error(f"Kite API Error (Portfolio): {e}")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_zerodha_order(symbol, quantity, order_type, transaction_type, product, price=None):
    kite = st.session_state.get('kite')
    try:
        order_id = kite.place_order(tradingsymbol=symbol.upper(), exchange=kite.EXCHANGE_NSE, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=kite.VARIETY_REGULAR, price=price)
        st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
        if 'order_history' not in st.session_state: st.session_state.order_history = []
        st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
    except Exception as e:
        st.toast(f"❌ Order placement failed: {e}", icon="🔥")
        if 'order_history' not in st.session_state: st.session_state.order_history = []
        st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})

# --- ML, News & Greeks Functions ---
@st.cache_data(ttl=900)
def fetch_and_analyze_news():
    analyzer = SentimentIntensityAnalyzer()
    news_sources = {"Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml", "Business Standard": "https://www.business-standard.com/rss/markets-102.cms"}
    all_news = []
    for source, url in news_sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            sentiment = analyzer.polarity_scores(entry.title)['compound']
            all_news.append({"source": source, "title": entry.title, "link": entry.link, "sentiment": sentiment})
    return pd.DataFrame(all_news)

def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['close'].shift(lag)
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df.dropna(inplace=True)
    return df

@st.cache_data
def train_xgboost_model(_data):
    df = create_features(_data.copy())
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    target = 'close'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, objective='reg:squarederror', eval_metric='rmse')
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    df['prediction'] = reg.predict(df[features])
    mse = mean_squared_error(df[target], df['prediction'])
    return reg, mse

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) if option_type == "call" else (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365}

def implied_volatility(S, K, T, r, market_price, option_type):
    def equation(sigma):
        return black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except RuntimeError:
        return np.nan

def interpret_indicators(df):
    latest = df.iloc[-1]
    interpretation = {}
    rsi = latest.get('RSI_14')
    if rsi > 70: interpretation['RSI (14)'] = "Overbought (Bearish)"
    elif rsi < 30: interpretation['RSI (14)'] = "Oversold (Bullish)"
    else: interpretation['RSI (14)'] = "Neutral"

    macd = latest.get('MACD_12_26_9')
    signal = latest.get('MACDs_12_26_9')
    if macd > signal: interpretation['MACD (12,26,9)'] = "Bullish Crossover"
    elif macd < signal: interpretation['MACD (12,26,9)'] = "Bearish Crossover"
    else: interpretation['MACD (12,26,9)'] = "Neutral"

    adx = latest.get('ADX_14')
    if adx > 25: interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})"
    else: interpretation['ADX (14)'] = f"Weak/No Trend ({adx:.1f})"
    return interpretation

# ==============================================================================
# 3. PAGE DEFINITIONS
# ==============================================================================

def page_dashboard():
    st.title("Main Dashboard")
    instrument_df, watchlist_symbols = get_instrument_df(st.session_state.kite), ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']
    watchlist_tokens = [{'symbol': s, 'token': get_instrument_token(s, instrument_df)} for s in watchlist_symbols]
    watchlist_data, nifty_token = get_watchlist_data(watchlist_tokens), get_instrument_token('NIFTY 50', instrument_df, 'NFO')
    nifty_data, (_, _, total_pnl, total_investment) = get_historical_data(nifty_token, "5minute", "1d"), get_portfolio()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Live Watchlist"); st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
        st.subheader("Portfolio Overview"); st.metric("Total Investment", f"₹{total_investment:,.2f}"); st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}")
    with col2:
        st.subheader("NIFTY 50 Live Chart (5-min)")
        if not nifty_data.empty: st.plotly_chart(create_chart(nifty_data.tail(75), "NIFTY 50"), use_container_width=True)

def page_advanced_charting():
    st.title("Advanced Charting")
    instrument_df, st.sidebar.header("Chart Controls") = get_instrument_df(st.session_state.kite), None
    ticker = st.sidebar.text_input("Select Ticker", "RELIANCE")
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4)
    interval = st.sidebar.selectbox("Interval", ["5minute", "15minute", "day", "week"], index=2)
    chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"])
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period)
        if not data.empty:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True)
            st.subheader("Technical Indicator Analysis")
            interp_df = pd.DataFrame([interpret_indicators(data)], index=["Interpretation"]).T
            st.dataframe(interp_df, use_container_width=True)
        else: st.warning(f"No chart data for {ticker}.")
    else: st.error(f"Ticker '{ticker}' not found.")

def page_options_hub():
    st.title("Options Hub")
    instrument_df = get_instrument_df(st.session_state.kite)
    underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "GOLDM", "SILVERM", "CRUDEOIL", "NATURALGAS", "USDINR"])
    chain_df, expiry, underlying_ltp = get_options_chain(underlying, instrument_df)
    tab1, tab2 = st.tabs(["Options Chain", "Greeks Calculator"])
    with tab1:
        st.subheader(f"{underlying} Options Chain")
        if not chain_df.empty:
            st.caption(f"Displaying nearest expiry: {expiry.strftime('%d %b %Y')}")
            st.dataframe(chain_df, use_container_width=True, hide_index=True)
        else: st.warning(f"Could not fetch options chain for {underlying}.")
    with tab2:
        st.subheader("Option Greeks Calculator (Black-Scholes)")
        if not chain_df.empty:
            option_selection = st.selectbox("Select an option to analyze", chain_df['CALL'].dropna().tolist() + chain_df['PUT'].dropna().tolist())
            option_details = instrument_df[instrument_df['tradingsymbol'] == option_selection].iloc[0]
            strike_price, option_type, ltp = option_details['strike'], option_details['instrument_type'].lower(), option_details['LTP']
            days_to_expiry = (expiry - datetime.now().date()).days
            T, r = days_to_expiry / 365, 0.07 # Time in years, Risk-free rate (approx)
            iv = implied_volatility(underlying_ltp, strike_price, T, r, ltp, option_type)
            if not np.isnan(iv):
                greeks = black_scholes(underlying_ltp, strike_price, T, r, iv, option_type)
                st.metric("Implied Volatility (IV)", f"{iv*100:.2f}%")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Delta", f"{greeks['delta']:.4f}")
                col2.metric("Gamma", f"{greeks['gamma']:.4f}")
                col3.metric("Vega", f"{greeks['vega']:.4f}")
                col4.metric("Theta", f"{greeks['theta']:.4f}")
            else: st.warning("Could not calculate Implied Volatility for this option (likely due to low liquidity or price).")

def page_alpha_engine():
    st.title("Alpha Engine: News & Sentiment")
    news_df = fetch_and_analyze_news()
    avg_sentiment, sentiment_label = news_df['sentiment'].mean(), "Positive" if news_df['sentiment'].mean() > 0.05 else "Negative" if news_df['sentiment'].mean() < -0.05 else "Neutral"
    st.metric("Aggregate News Sentiment", sentiment_label, f"{avg_sentiment:.3f}")
    st.subheader("Latest Headlines")
    for _, row in news_df.head(20).iterrows():
        color = "green" if row['sentiment'] > 0.2 else "red" if row['sentiment'] < -0.2 else "gray"
        st.markdown(f"<span style='color:{color};'>●</span> <a href='{row['link']}' target='_blank'>{row['title']}</a> ({row['source']})", unsafe_allow_html=True)

def page_portfolio_and_risk():
    st.title("Portfolio & Risk Journal")
    positions_df, holdings_df, total_pnl, total_investment = get_portfolio()
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
    st.title("📈 Advanced ML Forecasting (XGBoost)")
    st.info("This tool trains an XGBoost model on historical data to predict the next closing price. This is for educational purposes and is not financial advice.", icon="ℹ️")
    instrument_df, ticker = get_instrument_df(st.session_state.kite), st.text_input("Enter a stock symbol to analyze", "TCS")
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, "day", "5y")
        if not data.empty:
            if st.button(f"Train XGBoost Model & Forecast {ticker}"):
                with st.spinner("Training advanced model on 5 years of data... This may take a moment."):
                    model, mse = train_xgboost_model(data)
                    st.success(f"Model trained. Test Set MSE: {mse:.2f}")
                    # Create future forecast
                    last_features = create_features(data.tail(10)).iloc[-1]
                    features_for_pred = [col for col in last_features.index if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
                    future_preds = model.predict(pd.DataFrame([last_features[features_for_pred]]))
                    st.metric(f"Forecasted next closing price for {ticker}", f"₹{future_preds[0]:.2f}")
            st.subheader("Historical Data for Training"); st.plotly_chart(create_chart(data.tail(252), ticker), use_container_width=True)
        else: st.warning("Could not fetch sufficient data for training.")
    else: st.error("Ticker not found.")

def page_ai_assistant():
    st.title("🤖 Portfolio-Aware Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about your portfolio or stock prices (e.g., 'what are my holdings?' or 'current price of RELIANCE')."}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Accessing data..."):
                # Rule-based intent detection
                prompt_lower = prompt.lower()
                response = "Sorry, I can't answer that. I can provide information on your holdings, positions, and current stock prices."
                if "holdings" in prompt_lower:
                    _, holdings_df, _, _ = get_portfolio()
                    if not holdings_df.empty: response = "Here are your current holdings:\n" + holdings_df.to_markdown()
                    else: response = "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    if not positions_df.empty: response = f"Here are your current positions. Your total P&L is ₹{total_pnl:,.2f}:\n" + positions_df.to_markdown()
                    else: response = "You have no open positions."
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    words = prompt_lower.split()
                    try:
                        ticker_index = words.index("of") + 1
                        ticker = words[ticker_index].upper()
                        ltp_df = get_watchlist_data([{'symbol': ticker}])
                        if not ltp_df.empty: response = f"The current price of {ticker} is {ltp_df.iloc[0]['Price']}."
                        else: response = f"Could not fetch the price for {ticker}."
                    except (ValueError, IndexError):
                        response = "Please specify a stock ticker, for example: 'price of RELIANCE'."
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# 4. MAIN APP LOGIC AND AUTHENTICATION
# ==============================================================================

def main():
    if 'kite' in st.session_state:
        st.sidebar.success(f"Logged in as {st.session_state.profile['user_name']}")
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
