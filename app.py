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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import google.generativeai as genai

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
# 2. HELPER FUNCTIONS (CHARTS, KITE API, ML, NEWS, AI)
# ==============================================================================

# --- Charting Functions ---
def create_candlestick_chart(df, ticker, forecast_df=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBL_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBU_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), fill='tonexty', fillcolor='rgba(135, 206, 250, 0.1)', name='Upper Band'))
    
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))

    fig.update_layout(title=f'{ticker} Price Chart', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Kite Connect API Functions ---
@st.cache_resource(ttl=3600)
def get_instrument_df(_kite):
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame()
    instruments = kite.instruments()
    return pd.DataFrame(instruments)

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
        return df
    except Exception as e:
        st.error(f"Kite API Error (Historical): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def get_watchlist_data(symbols_with_tokens):
    kite = st.session_state.get('kite')
    if not kite or not symbols_with_tokens: return pd.DataFrame()
    instrument_names = [f"NSE:{item['symbol']}" for item in symbols_with_tokens]
    try:
        ltp_data = kite.ltp(instrument_names)
        quotes = kite.quote(instrument_names)
        watchlist = []
        for item in symbols_with_tokens:
            instrument = f"NSE:{item['symbol']}"
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
    if not kite: return pd.DataFrame(), None
    exchange = 'MCX' if underlying in ["GOLDM", "CRUDEOIL"] else 'NFO'
    options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
    if options.empty: return pd.DataFrame(), None
    expiries = sorted(options['expiry'].unique())
    nearest_expiry = expiries[0]
    chain_df = options[options['expiry'] == nearest_expiry].sort_values(by='strike')
    ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
    pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
    instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
    if not instruments_to_fetch: return pd.DataFrame(), nearest_expiry
    quotes = kite.quote(instruments_to_fetch)
    ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
    pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
    final_chain = pd.merge(ce_df[['strike', 'LTP']], pe_df[['strike', 'LTP']], on='strike', suffixes=('_CE', '_PE'), how='outer').rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE'})
    return final_chain[['CALL LTP', 'STRIKE', 'PUT LTP']], nearest_expiry

@st.cache_data(ttl=10)
def get_portfolio():
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    try:
        positions = kite.positions()['net']
        holdings = kite.holdings()
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

# --- ML, News & AI Functions ---
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

def train_and_forecast_model(data):
    df = data[['close']].copy()
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    X, y = df[['close']], df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    last_val, forecast = df['close'].iloc[-1], []
    for _ in range(5):
        next_pred = model.predict(np.array([[last_val]]))[0]
        forecast.append(next_pred)
        last_val = next_pred
    forecast_dates = pd.to_datetime([df.index[-1] + timedelta(days=i+1) for i in range(5)])
    forecast_df = pd.DataFrame({'predicted': forecast}, index=forecast_dates)
    return forecast_df, mse

def get_gemini_response(prompt):
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        contextual_prompt = f"You are a helpful financial assistant for the 'BlockVista Terminal'. Provide concise and accurate information about financial markets, stocks, and trading concepts. User's question: {prompt}"
        response = model.generate_content(contextual_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error connecting to Gemini API: {e}")
        return "Sorry, I'm having trouble connecting to the AI service right now."

# ==============================================================================
# 3. PAGE DEFINITIONS
# ==============================================================================

def page_dashboard():
    st.title("Main Dashboard")
    instrument_df = get_instrument_df(st.session_state.kite)
    watchlist_symbols = ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']
    watchlist_tokens = [{'symbol': s, 'token': get_instrument_token(s, instrument_df)} for s in watchlist_symbols]
    watchlist_data = get_watchlist_data(watchlist_tokens)
    nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NFO')
    nifty_data = get_historical_data(nifty_token, "5minute", "1d")
    _, _, total_pnl, total_investment = get_portfolio()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Live Watchlist"); st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
        st.subheader("Portfolio Overview"); st.metric("Total Investment", f"₹{total_investment:,.2f}"); st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}")
    with col2:
        st.subheader("NIFTY 50 Live Chart (5-min)")
        if not nifty_data.empty: st.plotly_chart(create_candlestick_chart(nifty_data.tail(75), "NIFTY 50"), use_container_width=True)

def page_advanced_charting():
    st.title("Advanced Charting")
    # CORRECTED LINE:
    instrument_df = get_instrument_df(st.session_state.kite)
    st.sidebar.header("Chart Controls")
    
    ticker = st.sidebar.text_input("Select Ticker", "RELIANCE")
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4)
    interval = st.sidebar.selectbox("Interval", ["5minute", "15minute", "day", "week"], index=2)
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period)
        if not data.empty: st.plotly_chart(create_candlestick_chart(data, ticker), use_container_width=True)
        else: st.warning(f"No chart data for {ticker}.")
    else: st.error(f"Ticker '{ticker}' not found.")

def page_options_hub():
    st.title("Options Hub")
    instrument_df = get_instrument_df(st.session_state.kite)
    underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "GOLDM", "CRUDEOIL"])
    st.subheader(f"{underlying} Options Chain")
    with st.spinner(f"Fetching options chain for {underlying}..."):
        chain_df, expiry = get_options_chain(underlying, instrument_df)
        if not chain_df.empty:
            st.caption(f"Displaying nearest expiry: {expiry.strftime('%d %b %Y')}")
            st.dataframe(chain_df, use_container_width=True, hide_index=True)
        else: st.warning(f"Could not fetch options chain for {underlying}.")

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
    st.title("📈 ML Price Forecasting")
    st.info("This is a simplified educational tool, not financial advice.", icon="ℹ️")
    instrument_df = get_instrument_df(st.session_state.kite)
    ticker = st.text_input("Enter a stock symbol to analyze", "TCS")
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, "day", "1y")
        if not data.empty:
            if st.button(f"Train Model & Forecast {ticker}"):
                with st.spinner("Training model on 1 year of historical data..."):
                    forecast_df, mse = train_and_forecast_model(data)
                    st.success(f"Model trained. Mean Squared Error: {mse:.2f}")
                    st.subheader("Forecasted Price Movements")
                    st.plotly_chart(create_candlestick_chart(data.tail(90), ticker, forecast_df), use_container_width=True)
                    st.dataframe(forecast_df)
            else:
                st.subheader("Historical Data"); st.plotly_chart(create_candlestick_chart(data.tail(90), ticker), use_container_width=True)
        else: st.warning("Could not fetch data for training.")
    else: st.error("Ticker not found.")

def page_ai_assistant():
    st.title("🤖 AI Assistant (Powered by Gemini)")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze the market today?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about market trends, strategies, or specific stocks..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response(prompt)
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
                if st.form_submit_button("Submit Order") and symbol:
                    place_zerodha_order(symbol, qty, order_type, transaction_type, product, price if price > 0 else None)
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
                st.query_params.clear()
                st.rerun()
            except Exception as e: st.error(f"Authentication failed: {e}")
        else: st.link_button("Login with Kite", kite.login_url())

if __name__ == "__main__":
    main()
