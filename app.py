# BlockVista-Terminal/app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import talib
from datetime import datetime, timedelta
from kiteconnect import KiteConnect, KiteTicker
import threading
import logging
import os

# --- Configuration and Initial Setup ---

st.set_page_config(layout="wide", page_title="BlockVista Terminal", page_icon="📈")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global dictionary to store live quotes from Kite WebSocket
LIVE_QUOTES = {}

# --- KiteConnect API Functions ---

@st.cache_data
def get_instrument_tokens():
    """
    Reads the instruments CSV file from Kite, filters for NSE equity,
    and returns a dictionary mapping trading symbols to instrument tokens.
    
    This function is cached to avoid reading the file on every rerun.
    """
    try:
        df = pd.read_csv("instruments.csv")
        eq_df = df[df['exchange'] == 'NSE']
        return dict(zip(eq_df['tradingsymbol'], eq_df['instrument_token']))
    except FileNotFoundError:
        st.error("`instruments.csv` not found. Please download it from Kite Dev and place it in the same directory.")
        return {}

def fetch_kite_historical(symbol, interval, from_dt, to_dt, kite_client, instrument_token_map):
    """
    Fetches historical data from Kite Connect API.
    Returns a pandas DataFrame or None if an error occurs.
    """
    try:
        if symbol.upper() not in instrument_token_map:
            logging.warning(f"Instrument token not found for {symbol}. Will use yfinance.")
            return None
            
        instrument_token = instrument_token_map[symbol.upper()]
        data = kite_client.historical_data(instrument_token, from_dt, to_dt, interval)
        df = pd.DataFrame(data)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            # Standardize column names to match yfinance
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df
        
    except Exception as e:
        logging.error(f"Kite API error for {symbol}: {e}")
        return None

# --- Kite Ticker (WebSocket for Live Data) ---

def start_live_ticker(api_key, access_token, instrument_tokens):
    """
    Starts a WebSocket client in a separate thread to receive live ticks.
    """
    kws = KiteTicker(api_key, access_token)

    def on_ticks(ws, ticks):
        for tick in ticks:
            # Find symbol from token (more reliable than 'tradingsymbol')
            token = tick.get("instrument_token")
            ltp = tick.get("last_price")
            # This is a reverse lookup, might be slow. A token->symbol map would be faster.
            # For now, we update the global dict which is keyed by symbol.
            # A better approach would be to have a global token -> symbol map created at startup.
            for symbol, tkn in st.session_state.instrument_token_map.items():
                if tkn == token:
                    LIVE_QUOTES[symbol] = ltp
                    # st.experimental_rerun() # Use with caution, can lead to high resource usage.
                    break

    def on_connect(ws, response):
        logging.info("Kite Ticker: Connected. Subscribing to tokens.")
        ws.subscribe(instrument_tokens)
        ws.set_mode(ws.MODE_LTP, instrument_tokens)

    def on_close(ws, code, reason):
        logging.warning(f"Kite Ticker: Connection closed. Code: {code}, Reason: {reason}")

    def on_error(ws, code, reason):
        logging.error(f"Kite Ticker: Error. Code: {code}, Reason: {reason}")
    
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    # Run the WebSocket connection in a non-blocking thread
    st.session_state.kws_thread = threading.Thread(target=lambda: kws.connect(threaded=True))
    st.session_state.kws_thread.daemon = True # Ensure thread closes when main app exits
    st.session_state.kws_thread.start()
    logging.info("Kite Ticker thread started.")


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("📈 BlockVista Terminal")

    # Load instrument tokens at the start of the app
    if 'instrument_token_map' not in st.session_state:
        st.session_state.instrument_token_map = get_instrument_tokens()

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Configuration")

    # --- Kite Login ---
    if 'kite' not in st.session_state:
        api_key = st.sidebar.text_input("Kite API Key", type="password")
        api_secret = st.sidebar.text_input("Kite API Secret", type="password")
        if st.sidebar.button("Login with Kite"):
            if api_key and api_secret:
                try:
                    kite = KiteConnect(api_key=api_key)
                    login_url = kite.login_url()
                    st.sidebar.markdown(f"[Click here to generate request token]({login_url})")
                    st.session_state.kite_pre = kite
                    st.session_state.api_key = api_key # Store for ticker
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
        
        request_token = st.sidebar.text_input("Paste Request Token here")
        if st.sidebar.button("Generate Session"):
            if 'kite_pre' in st.session_state and request_token:
                try:
                    user_session = st.session_state.kite_pre.generate_session(request_token, api_secret=api_secret)
                    kite = st.session_state.kite_pre
                    kite.set_access_token(user_session["access_token"])
                    st.session_state.kite = kite
                    st.session_state.access_token = user_session["access_token"] # Store for ticker
                    st.sidebar.success("Kite login successful!")
                    st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Authentication failed: {e}")
    else:
        st.sidebar.success("✅ Logged in to Kite")
        profile = st.session_state.kite.profile()
        st.sidebar.write(f"Welcome, {profile['user_name']}!")

    # --- Stock Selection and Parameters ---
    # Using a text area for symbols allows copy-pasting
    symbols_input = st.sidebar.text_area("Enter Stock Symbols (e.g., INFY, RELIANCE, TCS)", "RELIANCE,INFY,HDFCBANK")
    selected_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

    period = st.sidebar.selectbox("Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max'], index=3)
    interval = st.sidebar.selectbox("Interval", ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'], index=7)
    
    # --- Start WebSocket Ticker (if logged in and symbols selected) ---
    if 'kite' in st.session_state and selected_symbols and 'ticker_started' not in st.session_state:
        tokens_to_subscribe = [st.session_state.instrument_token_map[s] for s in selected_symbols if s in st.session_state.instrument_token_map]
        if tokens_to_subscribe:
            start_live_ticker(st.session_state.api_key, st.session_state.access_token, tokens_to_subscribe)
            st.session_state.ticker_started = True
            st.sidebar.info(f"Live ticker started for {len(tokens_to_subscribe)} symbols.")
        else:
            st.sidebar.warning("None of the selected symbols found for live data.")


    # --- Main Panel: Data Fetching and Display ---
    if not selected_symbols:
        st.info("Please enter at least one stock symbol in the sidebar to begin.")
        return

    all_data = {}
    
    # --- Data Fetching Logic (Kite with yfinance Fallback) ---
    with st.spinner(f"Fetching data for {', '.join(selected_symbols)}..."):
        # Calculate date range for Kite API
        # Note: yfinance period mapping to dates is not 1:1, this is an approximation
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, 
            '1y': 365, '2y': 730, '5y': 1825, 'ytd': datetime.now().timetuple().tm_yday, 'max': 5000
        }
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=period_map.get(period, 90))

        # Convert interval for Kite
        # Example mapping, needs to be more robust for all cases
        interval_map = {'1m': 'minute', '5m': '5minute', '15m': '15minute', '60m': '60minute', '1h': 'hour', '1d': 'day'}
        kite_interval = interval_map.get(interval, 'day')

        for symbol in selected_symbols:
            df = None
            # 1. Try fetching from Kite
            if 'kite' in st.session_state:
                df = fetch_kite_historical(symbol, kite_interval, from_dt, to_dt, st.session_state.kite, st.session_state.instrument_token_map)

            # 2. Fallback to yfinance if Kite fails or is not logged in
            if df is None or df.empty:
                try:
                    # Append .NS for NSE stocks for yfinance
                    yf_symbol = f"{symbol}.NS"
                    df = yf.download(tickers=yf_symbol, period=period, interval=interval, progress=False)
                    if df.empty:
                        st.warning(f"Could not fetch data for {symbol} from any source.")
                        continue
                    logging.info(f"Fetched data for {symbol} using yfinance fallback.")
                except Exception as e:
                    st.error(f"yfinance error for {symbol}: {e}")
                    continue

            if not df.empty:
                all_data[symbol] = df

    if not all_data:
        st.error("Failed to fetch data for all selected symbols. Please check symbols or API credentials.")
        return
        
    # --- Display Current Prices ---
    st.subheader("Current Prices")
    price_cols = st.columns(min(len(all_data), 4)) # Show up to 4 price metrics per row
    
    col_idx = 0
    for symbol, data in all_data.items():
        if len(data) < 2:
            continue
            
        last_close = data['Close'].iloc[-1]
        previous_close = data['Close'].iloc[-2]

        # Use live price from WebSocket if available, otherwise use last fetched close
        live_price = LIVE_QUOTES.get(symbol, last_close)
        
        price_change = live_price - previous_close
        percent_change = (price_change / previous_close) * 100
        
        delta_color = "normal"
        if price_change < 0:
            delta_color = "inverse"
            
        metric_label = f"LIVE: {symbol}" if symbol in LIVE_QUOTES else symbol
        price_cols[col_idx].metric(
            label=metric_label,
            value=f"₹{live_price:,.2f}",
            delta=f"₹{price_change:,.2f} ({percent_change:.2f}%)",
            delta_color=delta_color
        )
        col_idx = (col_idx + 1) % len(price_cols)

    st.markdown("---")

    # --- Display Charts and Indicators for each Symbol ---
    for symbol in selected_symbols:
        if symbol not in all_data:
            continue
            
        st.header(f"Analysis for {symbol}")
        
        data = all_data[symbol]
        
        # Calculate Indicators
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Close'], timeperiod=20)
        
        # --- Main Candlestick Chart ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='purple', width=1.5)))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['upper_band'], mode='lines', name='Upper BB', line=dict(color='gray', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['lower_band'], mode='lines', name='Lower BB', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
        
        fig.update_layout(title=f'{symbol} Candlestick Chart', yaxis_title='Price (₹)', xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Indicator Subplots ---
        col1, col2 = st.columns(2)

        with col1:
            # RSI Chart
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='cyan')))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(title='Relative Strength Index (RSI)', yaxis_title='RSI', height=300)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col2:
            # MACD Chart
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['macd'], mode='lines', name='MACD', line=dict(color='blue')))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['macdsignal'], mode='lines', name='Signal', line=dict(color='red')))
            macd_fig.add_trace(go.Bar(x=data.index, y=data['macdhist'], name='Histogram', marker_color='grey'))
            macd_fig.update_layout(title='MACD', height=300)
            st.plotly_chart(macd_fig, use_container_width=True)

        st.markdown("---")

if __name__ == "__main__":
    main()
