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
# 2. HELPER FUNCTIONS (CHARTS & KITE API) - OPTIMIZED WITH CACHING
# ==============================================================================

# --- Charting Functions ---
def create_candlestick_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBL_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df.get('BBU_20_2.0'), line=dict(color='rgba(135, 206, 250, 0.5)', width=1), fill='tonexty', fillcolor='rgba(135, 206, 250, 0.1)', name='Upper Band'))
    fig.update_layout(title=f'{ticker} Price Chart', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template='plotly_dark', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Kite Connect API Functions ---
@st.cache_resource(ttl=3600)  # Cache this resource for 1 hour
def get_instrument_df(_kite):
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame()
    instruments = kite.instruments()
    return pd.DataFrame(instruments)

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    if not match.empty: return match.iloc[0]['instrument_token']
    return None

@st.cache_data(ttl=60)  # Cache historical data for 60 seconds
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

@st.cache_data(ttl=5)  # Cache watchlist data for just 5 seconds for freshness
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

@st.cache_data(ttl=10) # Cache positions for 10 seconds
def get_positions():
    kite = st.session_state.get('kite')
    if not kite: return pd.DataFrame()
    try:
        positions = kite.positions()['net']
        if positions: return pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'pnl']]
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Kite API Error (Positions): {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. PAGE DEFINITIONS (Functions for each screen)
# ==============================================================================

def page_dashboard():
    st.title("Main Dashboard")
    instrument_df = get_instrument_df(st.session_state.kite)
    watchlist_symbols = ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'ICICIBANK']
    watchlist_tokens = [{'symbol': s, 'token': get_instrument_token(s, instrument_df)} for s in watchlist_symbols]
    watchlist_data = get_watchlist_data(watchlist_tokens)
    nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NFO')
    nifty_data = get_historical_data(nifty_token, "5minute", "1d")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Live Watchlist")
        st.dataframe(watchlist_data, use_container_width=True, hide_index=True)
        st.subheader("Portfolio Overview (Mock)")
        st.metric("Current Value", "₹13,10,500", "+₹15,200 (+1.16%)")
    with col2:
        st.subheader("NIFTY 50 Live Chart (5-min)")
        if not nifty_data.empty:
            st.plotly_chart(create_candlestick_chart(nifty_data.tail(75), "NIFTY 50"), use_container_width=True)

def page_advanced_charting():
    st.title("Advanced Charting")
    instrument_df = get_instrument_df(st.session_state.kite)
    st.sidebar.header("Chart Controls")
    ticker = st.sidebar.text_input("Select Ticker", "RELIANCE")
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4)
    interval = st.sidebar.selectbox("Interval", ["5minute", "15minute", "day", "week"], index=2)
    token = get_instrument_token(ticker, instrument_df)
    if token:
        data = get_historical_data(token, interval, period)
        if not data.empty:
            st.plotly_chart(create_candlestick_chart(data, ticker), use_container_width=True)
        else: st.warning(f"No chart data available for {ticker}.")
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
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live News Feed")
        rss_url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:10]:
            st.markdown(f"[{entry.title}]({entry.link})")
            st.caption(f"Published: {entry.published}")
            st.divider()
    with col2:
        st.subheader("Live Market Sentiment")
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(e.title)['compound'] for e in feed.entries]
        if scores:
            avg_score = sum(scores) / len(scores)
            sentiment = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
            st.metric("Overall News Sentiment", sentiment, f"{avg_score:.2f}")

def page_risk_and_journal():
    st.title("Risk & Trading Journal")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Positions")
        positions_df = get_positions()
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
        else: st.info("No open positions.")
    with col2:
        st.subheader("Portfolio Risk (Mock Greeks)")
        st.metric("Net Delta", "+150"); st.metric("Net Theta", "-₹3,500/day")

    st.subheader("Trading Journal")
    if 'journal' not in st.session_state: st.session_state.journal = []
    with st.form("journal_form"):
        entry = st.text_area("Log your thoughts or trade ideas:")
        if st.form_submit_button("Add Entry") and entry:
            st.session_state.journal.insert(0, (datetime.now(), entry))
    for ts, text in st.session_state.journal:
        st.info(text); st.caption(f"Logged: {ts.strftime('%d %b %H:%M')}")

def page_ai_assistant():
    st.title("🤖 AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze the market today?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about market trends..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = "I am a prototype. My capabilities are currently limited."
            if "risk" in prompt.lower(): response = "Based on mock data, your portfolio has a high negative Theta, indicating time decay risk."
            elif "nifty" in prompt.lower(): response = "NIFTY 50 is consolidating. Key support is at 24,000."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ==============================================================================
# 4. MAIN APP LOGIC AND AUTHENTICATION
# ==============================================================================

def main():
    """Main function to run the app."""
    if 'kite' in st.session_state:
        # --- LOGGED IN STATE ---
        st.sidebar.success(f"Logged in as {st.session_state.profile['user_name']}")
        
        # Auto-Refresh Controls
        st.sidebar.header("Live Data")
        auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
        refresh_interval = st.sidebar.number_input("Interval (s)", min_value=5, max_value=60, value=5, disabled=not auto_refresh)
        if auto_refresh:
            st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")

        # Page Navigation
        pages = {
            "Dashboard": page_dashboard,
            "Advanced Charting": page_advanced_charting,
            "Options Hub": page_options_hub,
            "Alpha Engine": page_alpha_engine,
            "Risk & Journal": page_risk_and_journal,
            "AI Assistant": page_ai_assistant,
        }
        st.sidebar.header("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Run the selected page function
        page_function = pages[selection]
        page_function()

        if st.sidebar.button("Logout"):
            # Clear all session state keys related to login
            for key in ['access_token', 'kite', 'profile']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        # --- LOGIN STATE ---
        st.subheader("Zerodha Kite Authentication")
        try:
            api_key = st.secrets["KITE_API_KEY"]
            api_secret = st.secrets["KITE_API_SECRET"]
        except (FileNotFoundError, KeyError):
            st.error("Kite API credentials not found. Please create `.streamlit/secrets.toml`.")
            st.stop()
        
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                st.session_state.kite = kite
                st.session_state.profile = kite.profile()
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.link_button("Login with Kite", kite.login_url())

if __name__ == "__main__":
    main()
