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
import time as a_time
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
        "tradingsymbol": "^NIFTY50",
        "exchange": "yfinance"
    },
    "S&P 500": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv",
        "tradingsymbol": "^GSPC",
        "exchange": "yfinance"
    }
}

# ================ 2. UTILITY FUNCTIONS ===============
def get_broker_client():
    """Returns the KiteConnect client if authenticated."""
    return st.session_state.get('kite_client')

def get_instrument_df():
    """Returns the instrument DataFrame if available."""
    return st.session_state.get('instrument_df', pd.DataFrame())

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Fetches instrument token for a given symbol."""
    try:
        token = instrument_df[
            (instrument_df['tradingsymbol'] == symbol) & 
            (instrument_df['exchange'] == exchange)
        ]['instrument_token'].iloc[0]
        return token
    except (IndexError, KeyError):
        st.error(f"Instrument token not found for {symbol} ({exchange})")
        return None

def get_historical_data(instrument_token, interval, period='1mo'):
    """Fetches historical data using KiteConnect."""
    client = get_broker_client()
    if not client:
        st.error("Broker client not initialized.")
        return pd.DataFrame()
    
    try:
        end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
        if period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif period == '5y':
            start_date = end_date - timedelta(days=5*365)
        else:
            start_date = end_date - timedelta(days=7)
        
        data = client.historical_data(
            instrument_token=instrument_token,
            from_date=start_date,
            to_date=end_date,
            interval=interval
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

def implied_volatility(S, K, T, r, option_price, option_type='call', max_iter=100, tol=1e-5):
    """Calculates implied volatility using Newton-Raphson method."""
    def black_scholes_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price - option_price
    
    sigma = 0.2  # Initial guess
    try:
        return newton(black_scholes_price, sigma, maxiter=max_iter, tol=tol)
    except:
        return np.nan

def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches options chain data for the specified underlying."""
    client = get_broker_client()
    if not client or instrument_df.empty:
        return pd.DataFrame(), None, None, []
    
    try:
        options = instrument_df[
            (instrument_df['tradingsymbol'].str.startswith(underlying)) &
            (instrument_df['segment'] == 'NFO-OPT')
        ]
        if options.empty:
            return pd.DataFrame(), None, None, []
        
        available_expiries = sorted(options['expiry'].unique())
        if not expiry_date:
            expiry_date = available_expiries[0] if available_expiries else None
        
        if expiry_date:
            options = options[options['expiry'] == expiry_date]
        
        chain_data = []
        underlying_token = get_instrument_token(underlying, instrument_df)
        underlying_quote = client.quote(f"NSE:{underlying}") if underlying_token else {}
        underlying_ltp = underlying_quote.get(f"NSE:{underlying}", {}).get('last_price', 0)
        
        strikes = sorted(options['strike'].unique())
        for strike in strikes:
            ce = options[(options['strike'] == strike) & (options['instrument_type'] == 'CE')]
            pe = options[(options['strike'] == strike) & (options['instrument_type'] == 'PE')]
            
            if not ce.empty and not pe.empty:
                ce_token = ce['instrument_token'].iloc[0]
                pe_token = pe['instrument_token'].iloc[0]
                ce_symbol = ce['tradingsymbol'].iloc[0]
                pe_symbol = pe['tradingsymbol'].iloc[0]
                
                quotes = client.quote([f"NFO:{ce_symbol}", f"NFO:{pe_symbol}"])
                ce_data = quotes.get(f"NFO:{ce_symbol}", {})
                pe_data = quotes.get(f"NFO:{pe_symbol}", {})
                
                chain_data.append({
                    'STRIKE': strike,
                    'CALL': ce_symbol,
                    'CALL LTP': ce_data.get('last_price', 0),
                    'open_interest_CE': ce_data.get('oi', 0),
                    'open_interest_CE_change': ce_data.get('oi_day_high', 0) - ce_data.get('oi_day_low', 0),
                    'PUT': pe_symbol,
                    'PUT LTP': pe_data.get('last_price', 0),
                    'open_interest_PE': pe_data.get('oi', 0),
                    'open_interest_PE_change': pe_data.get('oi_day_high', 0) - pe_data.get('oi_day_low', 0)
                })
        
        chain_df = pd.DataFrame(chain_data)
        return chain_df, expiry_date, underlying_ltp, available_expiries
    except Exception as e:
        st.error(f"Error fetching options chain: {str(e)}")
        return pd.DataFrame(), None, None, []

def style_option_chain(chain_df, underlying_ltp):
    """Styles the options chain DataFrame."""
    def highlight_atm(row):
        if abs(row['STRIKE'] - underlying_ltp) < 50:
            return ['background-color: #2E2E2E' for _ in row]
        return ['background-color: #1E1E1E' for _ in row]
    
    return chain_df.style.apply(highlight_atm, axis=1)

def quick_trade_dialog(symbol, exchange):
    """Displays a dialog for placing quick trades."""
    with st.form(key=f"trade_form_{symbol}"):
        st.write(f"Place Trade for {symbol}")
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"], key=f"order_type_{symbol}")
        quantity = st.number_input("Quantity", min_value=1, value=25, step=1, key=f"qty_{symbol}")
        price = st.number_input("Price", min_value=0.0, value=0.0, step=0.05, key=f"price_{symbol}") if order_type == "LIMIT" else 0
        transaction_type = st.selectbox("Transaction Type", ["BUY", "SELL"], key=f"trans_{symbol}")
        submitted = st.form_submit_button("Place Order")
        
        if submitted:
            client = get_broker_client()
            if client:
                try:
                    order_id = client.place_order(
                        variety=client.VARIETY_REGULAR,
                        exchange=exchange,
                        tradingsymbol=symbol,
                        transaction_type=client.TRANSACTION_TYPE_BUY if transaction_type == "BUY" else client.TRANSACTION_TYPE_SELL,
                        quantity=quantity,
                        product=client.PRODUCT_MIS,
                        order_type=client.ORDER_TYPE_MARKET if order_type == "MARKET" else client.ORDER_TYPE_LIMIT,
                        price=price if order_type == "LIMIT" else None
                    )
                    st.success(f"Order placed successfully! Order ID: {order_id}")
                except Exception as e:
                    st.error(f"Error placing order: {str(e)}")
            else:
                st.error("Broker client not initialized.")

# ================ 3. DATA FETCHING FUNCTIONS ===============
@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using yfinance as a proxy with enhanced error handling."""
    try:
        ticker = yf.Ticker("^NIFTY50")
        data = ticker.history(period="1d", interval="5m")
        
        if data.empty:
            st.error("No data retrieved for GIFT NIFTY (Proxy). Market data may be unavailable.")
            return pd.DataFrame()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error("Incomplete data for GIFT NIFTY (Proxy). Missing required columns.")
            return pd.DataFrame()
        
        data.index = pd.to_datetime(data.index)
        data.columns = [col.lower() for col in data.columns]
        return data
    
    except Exception as e:
        st.error(f"Error fetching GIFT NIFTY data: {str(e)}")
        return pd.DataFrame()

def get_indian_indices_data(indices):
    """Fetches Indian indices data using KiteConnect."""
    client = get_broker_client()
    instrument_df = get_instrument_df()
    if not client or instrument_df.empty:
        return pd.DataFrame()
    
    data = []
    for index in indices:
        token = get_instrument_token(index['symbol'], instrument_df, index['exchange'])
        if token:
            try:
                quote = client.quote(f"{index['exchange']}:{index['symbol']}")
                quote_data = quote.get(f"{index['exchange']}:{index['symbol']}", {})
                last_price = quote_data.get('last_price', 0)
                prev_close = quote_data.get('close_price', 0)
                pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0
                data.append({
                    'Ticker': index['symbol'],
                    'Exchange': index['exchange'],
                    'Price': last_price,
                    '% Change': pct_change
                })
            except Exception as e:
                st.error(f"Error fetching data for {index['symbol']}: {str(e)}")
    return pd.DataFrame(data)

def get_global_indices_data(tickers):
    """Fetches global indices data using yfinance."""
    data = []
    for ticker in tickers:
        try:
            index = yf.Ticker(ticker)
            info = index.history(period="1d")
            if not info.empty:
                last_price = info['Close'].iloc[-1]
                prev_close = info['Close'].iloc[0]
                pct_change = ((last_price - prev_close) / prev_close * 100) if prev_close else 0
                data.append({
                    'Ticker': ticker,
                    'Price': last_price,
                    '% Change': pct_change
                })
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def fetch_and_analyze_news():
    """Fetches and analyzes news sentiment."""
    feeds = [
        'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        'https://www.moneycontrol.com/rss/buzzingstocks.xml'
    ]
    news_data = []
    analyzer = SentimentIntensityAnalyzer()
    
    for feed in feeds:
        try:
            parsed_feed = feedparser.parse(feed)
            for entry in parsed_feed.entries[:10]:
                timestamp = datetime.fromtimestamp(mktime_tz(parsedate_tz(entry.published)))
                title = entry.title
                summary = entry.get('summary', '')
                sentiment = analyzer.polarity_scores(summary)['compound']
                news_data.append({
                    'title': title,
                    'source': parsed_feed.feed.title,
                    'link': entry.link,
                    'published': timestamp,
                    'sentiment': sentiment
                })
        except Exception as e:
            st.error(f"Error fetching news from {feed}: {str(e)}")
    
    news_df = pd.DataFrame(news_data)
    if not news_df.empty:
        news_df.sort_values(by='published', ascending=False, inplace=True)
    return news_df

# ================ 4. CHARTING FUNCTION ===============
def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None):
    """Generates a Plotly chart with various chart types and overlays."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.warning(f"No valid data to display for {ticker} chart.")
        return go.Figure()
    
    fig = go.Figure()
    chart_df = df.copy()
    
    try:
        chart_df.columns = [col.lower() for col in chart_df.columns]
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in chart_df.columns for col in required_columns):
            st.warning(f"Missing required columns for {ticker} chart: {required_columns}")
            return fig
    except AttributeError:
        st.warning(f"Invalid data format for {ticker} chart.")
        return fig
    
    try:
        if chart_type == 'Heikin-Ashi':
            ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
            fig.add_trace(go.Candlestick(
                x=ha_df.index,
                open=ha_df['HA_open'],
                high=ha_df['HA_high'],
                low=ha_df['HA_low'],
                close=ha_df['HA_close'],
                name='Heikin-Ashi'
            ))
        elif chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df['close'],
                mode='lines',
                name='Line',
                line=dict(color='#00b300')
            ))
        elif chart_type == 'Bar':
            fig.add_trace(go.Ohlc(
                x=chart_df.index,
                open=chart_df['open'],
                high=chart_df['high'],
                low=chart_df['low'],
                close=chart_df['close'],
                name='Bar'
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=chart_df.index,
                open=chart_df['open'],
                high=chart_df['high'],
                low=chart_df['low'],
                close=chart_df['close'],
                name='Candlestick'
            ))
        
        bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
        bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
        if bbl_col and bbu_col:
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df[bbl_col],
                line=dict(color='rgba(135,206,250,0.5)', width=1),
                name='Lower Band'
            ))
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df[bbu_col],
                line=dict(color='rgba(135,206,250,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(135,206,250,0.1)',
                name='Upper Band'
            ))
        
        if forecast_df is not None and not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Predicted'],
                mode='lines',
                line=dict(color='yellow', dash='dash'),
                name='Forecast'
            ))
        
        template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
        fig.update_layout(
            title={
                'text': f'{ticker} Price Chart ({chart_type})',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title='Price (INR)',
            xaxis_title='Time',
            xaxis_rangeslider_visible=False,
            template=template,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50),
            height=600
        )
        fig.update_xaxes(
            tickangle=45,
            tickformat="%H:%M",
            rangeslider_visible=False
        )
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart for {ticker}: {str(e)}")
        return go.Figure()

# ================ 5. PAGE FUNCTIONS ===============
def display_header():
    """Displays the app header."""
    st.markdown(
        """
        <div style='text-align: center; padding: 10px; background-color: #1E1E1E; border-radius: 10px;'>
            <h1 style='color: #FFFFFF; margin: 0;'>BlockVista Terminal</h1>
            <p style='color: #BBBBBB; margin: 0;'>Your Advanced Trading Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def login_page():
    """Handles user authentication with Zerodha."""
    st.title("Login to BlockVista Terminal")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Zerodha Login")
        client_id = st.text_input("Client ID", key="client_id")
        password = st.text_input("Password", type="password", key="password")
        totp_secret = st.text_input("TOTP Secret (Optional)", key="totp_secret")
        
        if st.button("Generate Login URL"):
            try:
                kite = KiteConnect(api_key=st.secrets["ZERODHA_API_KEY"])
                login_url = kite.login_url()
                st.session_state['kite_client'] = kite
                st.markdown(f"[Click here to login with Zerodha]({login_url})")
            except Exception as e:
                st.error(f"Error generating login URL: {str(e)}")
        
        request_token = st.text_input("Enter Request Token (after Zerodha login)", key="request_token")
        if st.button("Authenticate"):
            try:
                kite = st.session_state.get('kite_client')
                if not kite:
                    st.error("Kite client not initialized. Generate login URL first.")
                    return
                
                # Generate TOTP if provided
                totp_code = None
                if totp_secret:
                    totp = pyotp.TOTP(totp_secret)
                    totp_code = totp.now()
                
                # Simulate Zerodha login (actual API call requires user interaction)
                # Here, we assume request_token is obtained after login
                access_token = kite.generate_session(
                    request_token,
                    api_secret=st.secrets["ZERODHA_API_SECRET"]
                )['access_token']
                kite.set_access_token(access_token)
                
                # Fetch instruments
                instruments = kite.instruments()
                instrument_df = pd.DataFrame(instruments)
                instrument_df['expiry'] = pd.to_datetime(instrument_df['expiry'])
                
                # Fetch user profile
                profile = kite.profile()
                
                # Store in session state
                st.session_state.update({
                    'kite_client': kite,
                    'instrument_df': instrument_df,
                    'profile': profile,
                    'access_token': access_token,
                    'login_animation_complete': False
                })
                st.success("Successfully authenticated with Zerodha!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
    
    with col2:
        st.subheader("Broker Connection Status")
        if 'profile' in st.session_state:
            st.success(f"Connected as {st.session_state['profile']['user_name']}")
        else:
            st.info("Not connected. Please authenticate.")

def show_login_animation():
    """Displays a login animation."""
    st.markdown(
        """
        <style>
        .loader {
            border: 8px solid #f3f3f3;
            border-top: 8px solid #6C63FF;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
            margin: auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div style='text-align: center; padding: 50px;'>
            <h3>Connecting to BlockVista Terminal...</h3>
            <div class='loader'></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    a_time.sleep(2)
    st.session_state['login_animation_complete'] = True
    st.rerun()

def page_premarket_pulse():
    """Global market overview and premarket indicators with improved UI/UX."""
    display_header()
    st.title("Premarket & Global Cues")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view premarket data.")
        return
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Global Indices")
        
        indian_indices = [
            {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
            {'symbol': 'SENSEX', 'exchange': 'BSE'},
            {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
            {'symbol': 'BANKNIFTY', 'exchange': 'NSE'}
        ]
        indian_data = get_indian_indices_data(indian_indices)
        
        global_tickers = ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225", "^HSI"]
        global_data = get_global_indices_data(global_tickers)
        
        combined_indices = pd.concat([indian_data, global_data], ignore_index=True)
        if not combined_indices.empty:
            combined_indices['Color'] = combined_indices['% Change'].apply(lambda x: '#28a745' if x > 0 else '#FF4B4B')
            for _, row in combined_indices.iterrows():
                st.markdown(
                    f"<div style='padding: 10px; border-bottom: 1px solid #333;'>"
                    f"<b>{row['Ticker']}</b> ({row.get('Exchange', 'Global')}): "
                    f"₹{row['Price']:,.2f} "
                    f"<span style='color:{row['Color']};'>({row['% Change']:.2f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Indices data is loading...")
        
        st.subheader("GIFT NIFTY Live Chart (5-min)")
        with st.spinner("Fetching GIFT NIFTY data..."):
            gift_data = get_gift_nifty_data()
        
        if not gift_data.empty:
            st.info("Displaying a live chart for GIFT NIFTY (proxy via NIFTY 50 futures) from yfinance. This is not Zerodha data.")
            chart = create_chart(gift_data.tail(150), "GIFT NIFTY (Proxy)")
            if chart.data:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("Unable to display GIFT NIFTY chart due to data issues.")
        else:
            st.warning("Could not load GIFT NIFTY chart. Market data may be unavailable.")
    
    with col2:
        st.subheader("Live Market News")
        news_df = fetch_and_analyze_news()
        if not news_df.empty:
            for _, news in news_df.head(10).iterrows():
                with st.expander(f"📰 {news['title'][:80]}..."):
                    col_source, col_sentiment = st.columns([2, 1])
                    col_source.write(f"**Source:** {news['source']}")
                    sentiment_color = "#28a745" if news['sentiment'] > 0.1 else "#FF4B4B" if news['sentiment'] < -0.1 else "#FFA500"
                    col_sentiment.markdown(
                        f"**Sentiment:** <span style='color:{sentiment_color}'>{news['sentiment']:.3f}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"[Read Full Article]({news['link']})")
        else:
            st.info("News data is loading...")

def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis using KiteConnect."""
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
            expiry = st.selectbox("Select Expiry", 
                                [e.strftime("%d %b %Y") for e in get_options_chain(underlying, instrument_df)[3]],
                                key="expiry_select")
        
        chain_df, expiry_date, underlying_ltp, available_expiries = get_options_chain(
            underlying, instrument_df, 
            pd.to_datetime(expiry) if expiry else None
        )
        
        if not chain_df.empty:
            with col2:
                st.metric("Underlying Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry_date.strftime("%d %b %Y") if expiry_date else "N/A")
            
            styled_chain = style_option_chain(chain_df, underlying_ltp)
            st.dataframe(
                styled_chain.format({
                    'CALL LTP': '₹{:.2f}',
                    'PUT LTP': '₹{:.2f}',
                    'STRIKE': '₹{:.0f}',
                    'open_interest_CE': '{:,.0f}',
                    'open_interest_PE': '{:,.0f}',
                    'open_interest_CE_change': '{:,.0f}',
                    'open_interest_PE_change': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            for _, row in chain_df.iterrows():
                trade_cols = st.columns([2, 1, 1])
                trade_cols[0].markdown(f"**Strike: ₹{row['STRIKE']:.0f}**")
                if trade_cols[1].button(f"Buy {row['CALL']}", key=f"buy_{row['CALL']}"):
                    quick_trade_dialog(row['CALL'], 'NFO')
                if trade_cols[2].button(f"Sell {row['PUT']}", key=f"sell_{row['PUT']}"):
                    quick_trade_dialog(row['PUT'], 'NFO')
        else:
            st.warning("Could not load options chain data. Please check your connection.")
    
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
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pcr,
                title={'text': "Put-Call Ratio"},
                gauge={
                    'axis': {'range': [0, 2]},
                    'bar': {'color': "#6C63FF"},
                    'steps': [
                        {'range': [0, 0.7], 'color': "#28a745"},
                        {'range': [0.7, 1.3], 'color': "#FFA500"},
                        {'range': [1.3, 2], 'color': "#FF4B4B"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
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
        st.info("Volatility analysis for options contracts using KiteConnect data.")
        
        if not chain_df.empty:
            client = get_broker_client()
            if client:
                try:
                    vol_data = []
                    for _, row in chain_df.iterrows():
                        if row['CALL LTP'] > 0:
                            T = max((expiry_date.date() - datetime.now().date()).days, 0.01) / 365.0
                            iv_call = implied_volatility(
                                underlying_ltp, row['STRIKE'], T, 0.07, row['CALL LTP'], 'call'
                            )
                            if not np.isnan(iv_call):
                                vol_data.append({'Strike': row['STRIKE'], 'IV': iv_call * 100, 'Type': 'Call'})
                        if row['PUT LTP'] > 0:
                            T = max((expiry_date.date() - datetime.now().date()).days, 0.01) / 365.0
                            iv_put = implied_volatility(
                                underlying_ltp, row['STRIKE'], T, 0.07, row['PUT LTP'], 'put'
                            )
                            if not np.isnan(iv_put):
                                vol_data.append({'Strike': row['STRIKE'], 'IV': iv_put * 100, 'Type': 'Put'})
                    
                    vol_df = pd.DataFrame(vol_data)
                    if not vol_df.empty:
                        fig = go.Figure()
                        for opt_type in ['Call', 'Put']:
                            subset = vol_df[vol_df['Type'] == opt_type]
                            if not subset.empty:
                                fig.add_trace(go.Scatter(
                                    x=subset['Strike'],
                                    y=subset['IV'],
                                    mode='lines+markers',
                                    name=f'{opt_type} IV',
                                    line=dict(color='#00b300' if opt_type == 'Call' else '#FF4B4B')
                                ))
                        
                        fig.update_layout(
                            title=f"{underlying} Implied Volatility Smile",
                            xaxis_title="Strike Price",
                            yaxis_title="Implied Volatility (%)",
                            template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No valid IV data available for plotting.")
                except Exception as e:
                    st.error(f"Error calculating volatility surface: {str(e)}")
            else:
                st.info("Broker not connected.")
        else:
            st.info("Volatility data is loading...")

def page_forecasting_ml():
    """A page for advanced ML forecasting using a Seasonal ARIMA model."""
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train an advanced Seasonal ARIMA model to forecast future prices. This is for educational purposes only and is not financial advice.", icon="ℹ️")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access forecasting tools.")
        return
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()))
        source_info = ML_DATA_SOURCES.get(instrument_name)
        
        data = pd.DataFrame()
        if source_info.get('exchange') != 'yfinance':
            token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
            if token:
                with st.spinner(f"Fetching real-time data for {instrument_name}..."):
                    data = get_historical_data(token, 'day', period='5y')
        else:
            with st.spinner(f"Fetching yfinance data for {instrument_name}..."):
                data = yf.download(source_info['tradingsymbol'], period="5y")
                if not data.empty:
                    data.columns = [col.lower() for col in data.columns]
        
        if data.empty or len(data) < 100:
            st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
            return
        
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=30)
        forecast_date = st.date_input(
            "Select a date to forecast",
            value=today,
            min_value=today,
            max_value=max_forecast_date
        )
        
        if st.button("Train Seasonal ARIMA Model & Forecast"):
            forecast_steps = (forecast_date - today).days + 1
            if forecast_steps <= 0:
                st.warning("Please select a future date to forecast.")
            else:
                with st.spinner("Training Seasonal ARIMA model..."):
                    try:
                        decomposed = seasonal_decompose(data['close'], model='additive', period=7)
                        seasonally_adjusted = data['close'] - decomposed.seasonal
                        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
                        forecast_adjusted = model.forecast(steps=forecast_steps)
                        
                        last_season_cycle = decomposed.seasonal.iloc[-7:]
                        future_seasonal = pd.concat([last_season_cycle] * (forecast_steps // 7 + 1))[:forecast_steps]
                        future_seasonal.index = forecast_adjusted.index
                        
                        forecast_final = forecast_adjusted + future_seasonal
                        
                        predictions = {
                            "1-Day Close": forecast_final.iloc[0],
                            "5-Day Close": forecast_final.iloc[4] if forecast_steps > 4 else np.nan,
                            "15-Day Close": forecast_final.iloc[14] if forecast_steps > 14 else np.nan,
                            "30-Day Close": forecast_final.iloc[29] if forecast_steps > 29 else np.nan
                        }
                        
                        fitted_values = model.fittedvalues + decomposed.seasonal
                        backtest_df = pd.DataFrame({
                            'Actual': data['close'],
                            'Predicted': fitted_values
                        }).dropna()
                        
                        st.session_state.update({
                            'ml_predictions': predictions,
                            'ml_forecast_df': forecast_final.to_frame(name='Predicted'),
                            'ml_backtest_df': backtest_df,
                            'ml_instrument_name': instrument_name,
                            'ml_model_choice': "Seasonal ARIMA"
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Seasonal ARIMA model training failed: {str(e)}")
                        st.session_state.update({'ml_predictions': None})
    
    with col2:
        st.subheader(f"Forecast Results for {instrument_name} (Seasonal ARIMA)")
        
        if 'ml_predictions' in st.session_state and st.session_state.get('ml_instrument_name') == instrument_name:
            predictions = st.session_state['ml_predictions']
            forecast_df = st.session_state.get('ml_forecast_df')
            
            if predictions:
                pred_cols = st.columns(4)
                pred_cols[0].metric("1-Day Forecast", f"₹{predictions['1-Day Close']:.2f}")
                if not np.isnan(predictions['5-Day Close']):
                    pred_cols[1].metric("5-Day Forecast", f"₹{predictions['5-Day Close']:.2f}")
                if not np.isnan(predictions['15-Day Close']):
                    pred_cols[2].metric("15-Day Forecast", f"₹{predictions['15-Day Close']:.2f}")
                if not np.isnan(predictions['30-Day Close']):
                    pred_cols[3].metric("30-Day Forecast", f"₹{predictions['30-Day Close']:.2f}")
                
                if forecast_df is not None and not forecast_df.empty:
                    st.plotly_chart(
                        create_chart(data.tail(252), instrument_name, forecast_df=forecast_df),
                        use_container_width=True
                    )
            
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
                    mape = mean_absolute_percentage_error(display_df['Actual'], display_df['Predicted']) * 100
                    mse = mean_squared_error(display_df['Actual'], display_df['Predicted'])
                    accuracy = 100 - mape
                    cum_returns = (1 + (display_df['Actual'].pct_change().fillna(0))).cumprod()
                    peak = cum_returns.cummax()
                    drawdown = (cum_returns - peak) / peak
                    max_drawdown = drawdown.min()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Accuracy ({selected_period_name})", f"{accuracy:.2f}%")
                    c2.metric(f"MAPE ({selected_period_name})", f"{mape:.2f}%")
                    c3.metric(f"Max Drawdown ({selected_period_name})", f"{max_drawdown*100:.2f}%")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=display_df.index,
                        y=display_df['Actual'],
                        mode='lines',
                        name='Actual Price',
                        line=dict(color='#00b300')
                    ))
                    fig.add_trace(go.Scatter(
                        x=display_df.index,
                        y=display_df['Predicted'],
                        mode='lines',
                        name='Predicted Price',
                        line=dict(color='yellow', dash='dash')
                    ))
                    template = 'plotly_dark' if st.session_state.get('theme', 'Dark') == 'Dark' else 'plotly_white'
                    fig.update_layout(
                        title=f"Backtest Results ({selected_period_name})",
                        yaxis_title='Price (INR)',
                        xaxis_title='Date',
                        template=template,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for the selected period.")
            else:
                st.error("Could not generate performance metrics.")
        else:
            st.subheader(f"Historical Data for {instrument_name}")
            st.plotly_chart(create_chart(data.tail(252), instrument_name), use_container_width=True)

def main_app():
    """Main application interface."""
    st.sidebar.image("https://via.placeholder.com/150", caption="BlockVista Terminal")
    
    menu_options = [
        "Premarket & Global Cues",
        "F&O Analytics Hub",
        "Advanced ML Forecasting"
    ]
    page = st.sidebar.selectbox("Select Module", menu_options)
    
    theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])
    st.session_state['theme'] = theme
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    if page == "Premarket & Global Cues":
        page_premarket_pulse()
    elif page == "F&O Analytics Hub":
        page_fo_analytics()
    elif page == "Advanced ML Forecasting":
        page_forecasting_ml()

if __name__ == "__main__":
    if 'profile' in st.session_state:
        if st.session_state.get('login_animation_complete', False):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()
