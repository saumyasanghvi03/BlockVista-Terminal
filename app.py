```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from kiteconnect import KiteConnect
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
from scipy.optimize import newton
import pytz
import tabulate
from scipy.stats import norm
from math import log, sqrt, exp

# Constants
TWO_FACTOR_SECRET = "JBSWY3DPEHPK3PXP"  # Replace with your actual 2FA secret

ML_DATA_SOURCES = {
    "NIFTY 50": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/data/NIFTY50.csv"},
    "BANK NIFTY": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/data/BANKNIFTY.csv"},
    "S&P 500": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/data/SP500.csv"}
}

# Helper Functions
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using default styling.")
        st.markdown("""
        <style>
        .metric-card {
            background-color: #2c3e50;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .positive-blink { color: #28a745; }
        .negative-blink { color: #FF4B4B; }
        /* 2FA Animation */
        .qr-pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .fade-in {
            animation: fadeIn 1s ease-in;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        /* Zerodha Login Animation */
        .slide-in {
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            0% { opacity: 0; transform: translateX(-20px); }
            100% { opacity: 1; transform: translateX(0); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)

def get_sector_data():
    try:
        return pd.read_csv("sectors.csv")
    except FileNotFoundError:
        st.warning("sectors.csv not found. Sector data unavailable.")
        return pd.DataFrame(columns=['sector', 'symbol', 'weight', 'change'])

def setup_2fa():
    totp = pyotp.TOTP(TWO_FACTOR_SECRET)
    uri = totp.provisioning_uri(name="BlockVista", issuer_name="BlockVistaTerminal")
    qr = qrcode.QRCode()
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str, totp

def authenticate_broker():
    st.subheader("Broker Authentication")
    if 'setup_2fa' not in st.session_state:
        st.session_state['setup_2fa'] = True  # Show QR code on first visit
    if '2fa_verified' not in st.session_state:
        st.session_state['2fa_verified'] = False

    # 2FA Verification
    qr_code, totp = setup_2fa()
    if st.session_state['setup_2fa']:
        st.markdown("<div class='qr-pulse'>", unsafe_allow_html=True)
        st.image(f"data:image/png;base64,{qr_code}", caption="Scan with Authenticator App (One-Time Setup)")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    two_factor_code = st.text_input("Enter 2FA Code", key="2fa_code")
    if st.button("Verify 2FA"):
        with st.spinner("Verifying 2FA..."):
            if totp.verify(two_factor_code):
                st.session_state['2fa_verified'] = True
                st.session_state['setup_2fa'] = False  # Hide QR code after successful setup
                st.success("2FA Verified!")
                st.rerun()
            else:
                st.error("Invalid 2FA Code")
    st.markdown("</div>", unsafe_allow_html=True)

    # Zerodha Login
    if st.session_state['2fa_verified']:
        api_key = st.secrets.get("zerodha_api_key")
        api_secret = st.secrets.get("zerodha_api_secret")
        if not api_key or not api_secret:
            st.error("API key and secret not found in Streamlit secrets. Please add 'zerodha_api_key' and 'zerodha_api_secret' in your Streamlit secrets.")
            return
        st.markdown("<div class='slide-in'>", unsafe_allow_html=True)
        request_token = st.text_input("Enter Zerodha Request Token (Auth Code)", key="request_token")
        if st.button("Authenticate with Zerodha"):
            st.markdown("<span class='spinner'></span>", unsafe_allow_html=True)
            try:
                kite = KiteConnect(api_key=api_key)
                data = kite.generate_session(request_token, api_secret)
                kite.set_access_token(data["access_token"])
                st.session_state['kite'] = kite
                st.session_state['broker'] = "Zerodha"
                st.success("Authenticated with Zerodha!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

def get_instrument_df():
    if 'kite' not in st.session_state or not st.session_state['kite']:
        return pd.DataFrame()
    try:
        kite = st.session_state['kite']
        instruments = kite.instruments()
        return pd.DataFrame(instruments)
    except Exception as e:
        st.error(f"Failed to fetch instruments: {e}")
        return pd.DataFrame()

def get_historical_data(symbol, from_date, to_date, interval="day"):
    kite = st.session_state.get('kite')
    if not kite:
        return pd.DataFrame()
    try:
        instrument_df = get_instrument_df()
        instrument_token = instrument_df[instrument_df['tradingsymbol'] == symbol]['instrument_token'].iloc[0]
        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to fetch historical data for {symbol}: {e}")
        return pd.DataFrame()

def get_options_chain(underlying, instrument_df):
    try:
        symbols = instrument_df[instrument_df['name'] == underlying]['tradingsymbol'].values
        expiry = sorted(set([pd.to_datetime(s.split()[-1]).date() for s in symbols if 'CE' in s or 'PE' in s]))[0]
        chain = []
        kite = st.session_state['kite']
        quotes = kite.quote([f"NFO:{s}" for s in symbols])
        underlying_quote = kite.quote(f"NSE:{underlying}")["NSE:" + underlying]
        underlying_ltp = underlying_quote['last_price']
        for symbol in symbols:
            if pd.to_datetime(symbol.split()[-1]).date() == expiry:
                strike = float(symbol.split()[-2])
                option_type = 'CE' if 'CE' in symbol else 'PE' if 'PE' in symbol else None
                if option_type:
                    quote = quotes.get(f"NFO:{symbol}", {})
                    chain.append({
                        'STRIKE': strike,
                        'CALL' if option_type == 'CE' else 'PUT': symbol,
                        'CALL LTP' if option_type == 'CE' else 'PUT LTP': quote.get('last_price', 0)
                    })
        chain_df = pd.DataFrame(chain).groupby('STRIKE').first().reset_index()
        return chain_df, expiry, underlying_ltp, quotes
    except Exception as e:
        st.error(f"Failed to fetch options chain for {underlying}: {e}")
        return pd.DataFrame(), None, None, {}

def get_futures_chain(underlying, instrument_df):
    try:
        symbols = instrument_df[(instrument_df['name'] == underlying) & (instrument_df['instrument_type'] == 'FUT')]['tradingsymbol'].values
        expiry = sorted(set([pd.to_datetime(s.split()[-1]).date() for s in symbols]))[0]
        chain = []
        kite = st.session_state['kite']
        quotes = kite.quote([f"NFO:{s}" for s in symbols])
        for symbol in symbols:
            if pd.to_datetime(symbol.split()[-1]).date() == expiry:
                quote = quotes.get(f"NFO:{symbol}", {})
                chain.append({
                    'SYMBOL': symbol,
                    'LTP': quote.get('last_price', 0),
                    'OPEN': quote.get('ohlc', {}).get('open', 0),
                    'HIGH': quote.get('ohlc', {}).get('high', 0),
                    'LOW': quote.get('ohlc', {}).get('low', 0),
                    'VOLUME': quote.get('volume', 0)
                })
        chain_df = pd.DataFrame(chain)
        return chain_df, expiry
    except Exception as e:
        st.error(f"Failed to fetch futures chain for {underlying}: {e}")
        return pd.DataFrame(), None

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product):
    kite = st.session_state.get('kite')
    if not kite:
        st.error("Broker not connected.")
        return
    try:
        exchange = instrument_df[instrument_df['tradingsymbol'] == symbol]['exchange'].iloc[0]
        order_id = kite.place_order(
            variety="regular",
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=product,
            order_type=order_type,
            price=0
        )
        st.session_state['order_history'] = st.session_state.get('order_history', []) + [{
            'id': order_id,
            'symbol': symbol,
            'qty': quantity,
            'type': transaction_type,
            'status': 'PLACED'
        }]
        st.success(f"Order placed: {transaction_type} {quantity} {symbol}")
    except Exception as e:
        st.error(f"Failed to place order: {e}")

def place_basket_order(orders, variety='regular'):
    kite = st.session_state.get('kite')
    if not kite:
        st.error("Broker not connected.")
        return
    try:
        order_ids = kite.place_order(
            variety=variety,
            orders=orders
        )
        for order_id in order_ids:
            st.session_state['order_history'] = st.session_state.get('order_history', []) + [{
                'id': order_id,
                'symbol': 'BASKET',
                'qty': 1,
                'type': 'BASKET',
                'status': 'PLACED'
            }]
        st.success(f"Placed basket order: {len(order_ids)} orders")
    except Exception as e:
        st.error(f"Failed to place basket order: {e}")

def get_global_indices_data():
    indices = ["^GSPC", "^IXIC", "^DJI", "^NSEI"]
    data = []
    for index in indices:
        try:
            ticker = yf.Ticker(index)
            info = ticker.info
            data.append({
                'Index': index,
                'Price': info.get('regularMarketPrice', 0),
                'Change %': info.get('regularMarketChangePercent', 0)
            })
        except Exception:
            data.append({'Index': index, 'Price': 0, 'Change %': 0})
    return pd.DataFrame(data)

def black_scholes(S, K, T, r, sigma, option_type):
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type.lower() == 'ce':
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * exp(-r * T) * norm.cdf(d2) / 100 if option_type.lower() == 'ce' else -K * T * exp(-r * T) * norm.cdf(-d2) / 100
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}
    except Exception as e:
        st.error(f"Error calculating Black-Scholes: {e}")
        return {'price': 0, 'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

def implied_volatility(S, K, T, r, option_price, option_type):
    try:
        def objective(sigma):
            return black_scholes(S, K, T, r, sigma, option_type)['price'] - option_price
        return newton(objective, 0.2, tol=1e-6)
    except Exception:
        return np.nan

def rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return {'MACD': macd_line, 'Signal': signal_line, 'Histogram': histogram}

def bbands(close, window=20, std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * std)
    lower = rolling_mean - (rolling_std * std)
    middle = rolling_mean
    return {'Upper': upper, 'Middle': middle, 'Lower': lower}

def create_chart(df, symbol, timeframe, indicators):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ))
    if 'RSI' in indicators:
        df['RSI'] = rsi(df['close'])
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', yaxis='y2'))
    if 'MACD' in indicators:
        macd_data = macd(df['close'])
        fig.add_trace(go.Scatter(x=df['date'], y=macd_data['MACD'], name='MACD', yaxis='y3'))
        fig.add_trace(go.Scatter(x=df['date'], y=macd_data['Signal'], name='Signal', yaxis='y3'))
    if 'BBANDS' in indicators:
        bbands_data = bbands(df['close'])
        fig.add_trace(go.Scatter(x=df['date'], y=bbands_data['Upper'], name='Upper BB', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df['date'], y=bbands_data['Lower'], name='Lower BB', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df['date'], y=bbands_data['Middle'], name='Middle BB', line=dict(color='gray')))
    fig.update_layout(
        title=f"{symbol} - {timeframe}",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="RSI", overlaying='y', side='right', range=[0, 100]),
        yaxis3=dict(title="MACD", overlaying='y', side='right', position=0.95),
        xaxis_rangeslider_visible=False
    )
    return fig

def load_and_combine_data(instrument_name):
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()
    try:
        response = requests.get(source_info['github_url'])
        response.raise_for_status()
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df['date'] = pd.to_datetime(hist_df['date'] if 'date' in hist_df.columns else hist_df.index)
    except Exception:
        ticker_map = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "S&P 500": "^GSPC"}
        if instrument_name in ticker_map:
            hist_df = yf.download(ticker_map[instrument_name], period="1y")
            hist_df.reset_index(inplace=True)
            hist_df['date'] = pd.to_datetime(hist_df['Date'])
        else:
            st.error(f"Failed to load historical data for {instrument_name}")
            return pd.DataFrame()
    return hist_df.sort_values('date')

def train_seasonal_arima_model(data, forecast_date):
    try:
        decomposition = seasonal_decompose(data['close'], model='additive', period=252)
        trend = decomposition.trend.dropna()
        scaler = MinMaxScaler()
        scaled_trend = scaler.fit_transform(trend.values.reshape(-1, 1))
        model = ARIMA(scaled_trend, order=(5, 1, 0))
        model_fit = model.fit()
        today = datetime.now().date()
        forecast_steps = min((forecast_date - today).days + 1, 15)
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
        forecast_dates = [today + timedelta(days=x) for x in range(forecast_steps)]
        return pd.DataFrame({'date': forecast_dates, 'forecast': forecast})
    except Exception as e:
        st.error(f"Failed to train ARIMA model: {e}")
        return pd.DataFrame()

def fetch_and_analyze_news(symbol):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={symbol}")
        analyzer = SentimentIntensityAnalyzer()
        news_items = []
        for entry in feed.entries[:5]:
            sentiment = analyzer.polarity_scores(entry.title)
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'sentiment': sentiment['compound']
            })
        return pd.DataFrame(news_items)
    except Exception as e:
        st.error(f"Failed to fetch news for {symbol}: {e}")
        return pd.DataFrame()

def get_market_status():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return "Open" if market_open <= now <= market_close else "Closed"

def display_header():
    load_css()
    st.markdown("<h1 style='text-align: center;'>BlockVista Terminal</h1>", unsafe_allow_html=True)
    market_status = get_market_status()
    st.markdown(f"<p style='text-align: center;'>Market Status: {market_status}</p>", unsafe_allow_html=True)

# Pages
def page_dashboard():
    display_header()
    st.title("Dashboard")
    st_autorefresh(interval=60000, key="dashboard_refresh")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Indices")
        indices_df = get_global_indices_data()
        for _, row in indices_df.iterrows():
            change_class = "positive-blink" if row['Change %'] > 0 else "negative-blink"
            st.markdown(f"<div class='metric-card'>{row['Index']}: {row['Price']:.2f} <span class='{change_class}'>({row['Change %']:.2f}%)</span></div>", unsafe_allow_html=True)
    with col2:
        st.subheader("Sector Performance")
        sector_data = get_sector_data()
        if not sector_data.empty:
            for sector in sector_data['sector'].unique():
                sector_df = sector_data[sector_data['sector'] == sector]
                weighted_change = (sector_df['weight'] * sector_df['change']).sum()
                change_class = "positive-blink" if weighted_change > 0 else "negative-blink"
                st.markdown(f"<div class='metric-card'>{sector}: <span class='{change_class}'>{weighted_change:.2f}%</span></div>", unsafe_allow_html=True)
        else:
            st.info("No sector data available.")

def page_advanced_charting():
    display_header()
    st.title("Advanced Charting")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the Advanced Charting feature.")
        return
    symbol = st.selectbox("Select Symbol", instrument_df['tradingsymbol'].unique())
    timeframe = st.selectbox("Timeframe", ["1minute", "5minute", "15minute", "day"])
    indicators = st.multiselect("Indicators", ["RSI", "MACD", "BBANDS"])
    if symbol:
        from_date = datetime.now() - timedelta(days=30)
        df = get_historical_data(symbol, from_date, datetime.now(), timeframe)
        if not df.empty:
            st.plotly_chart(create_chart(df, symbol, timeframe, indicators), use_container_width=True)
        else:
            st.info("No data available for the selected timeframe.")

def page_options_hub():
    display_header()
    st.title("Options Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the Options Hub feature.")
        return
    underlying = st.selectbox("Select Underlying", ["NIFTY 50", "BANK NIFTY"])
    chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df)
    if not chain_df.empty:
        st.markdown(f"**Underlying LTP**: {underlying_ltp:.2f} | **Expiry**: {pd.to_datetime(expiry).strftime('%d %b %Y')}")
        st.dataframe(chain_df, use_container_width=True)
    else:
        st.info("No options data available.")

def page_futures_hub():
    display_header()
    st.title("Futures Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the Futures Hub feature.")
        return
    underlying = st.selectbox("Select Underlying", ["NIFTY 50", "BANK NIFTY"])
    chain_df, expiry = get_futures_chain(underlying, instrument_df)
    if not chain_df.empty:
        st.markdown(f"**Expiry**: {pd.to_datetime(expiry).strftime('%d %b %Y')}")
        st.dataframe(chain_df, use_container_width=True)
    else:
        st.info("No futures data available.")

def page_alpha_engine():
    display_header()
    st.title("Alpha Engine: News Sentiment")
    symbol = st.text_input("Enter Symbol (e.g., RELIANCE)")
    if symbol:
        news_df = fetch_and_analyze_news(symbol)
        if not news_df.empty:
            st.subheader(f"News Sentiment for {symbol}")
            for _, row in news_df.iterrows():
                sentiment = "Positive" if row['sentiment'] > 0 else "Negative" if row['sentiment'] < 0 else "Neutral"
                st.markdown(f"[{row['title']}]({row['link']}) - Sentiment: {sentiment} ({row['sentiment']:.2f})")
        else:
            st.info("No news available.")

def page_portfolio_and_risk():
    display_header()
    st.title("Portfolio & Risk")
    kite = st.session_state.get('kite')
    if not kite:
        st.info("Please connect to a broker to view portfolio.")
        return
    try:
        positions = kite.positions().get('net', [])
        holdings = kite.holdings()
        portfolio_df = pd.DataFrame(positions)
        holdings_df = pd.DataFrame(holdings)
        st.subheader("Positions")
        if not portfolio_df.empty:
            st.dataframe(portfolio_df[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], use_container_width=True)
        else:
            st.info("No positions available.")
        st.subheader("Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df[['tradingsymbol', 'quantity', 'average_price', 'last_price']], use_container_width=True)
        else:
            st.info("No holdings available.")
    except Exception as e:
        st.error(f"Failed to fetch portfolio data: {e}")

def page_forecasting_ml():
    display_header()
    st.title("Forecasting (ML)")
    instrument = st.selectbox("Select Instrument", list(ML_DATA_SOURCES.keys()))
    forecast_date = st.date_input("Forecast Until", value=datetime.now().date() + timedelta(days=7))
    if st.button("Generate Forecast"):
        with st.spinner("Training model..."):
            hist_df = load_and_combine_data(instrument)
            if not hist_df.empty:
                forecast_df = train_seasonal_arima_model(hist_df, forecast_date)
                if not forecast_df.empty:
                    st.subheader(f"Forecast for {instrument}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['close'], name="Historical"))
                    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], name="Forecast", line=dict(dash='dash')))
                    fig.update_layout(title=f"{instrument} Price Forecast", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Unable to generate forecast.")
            else:
                st.info("No historical data available.")

def page_ai_assistant():
    display_header()
    st.title("Portfolio-Aware Assistant")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about portfolio, orders, or market...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            prompt_lower = prompt.lower()
            instrument_df = get_instrument_df()
            kite = st.session_state.get('kite')
            if "portfolio" in prompt_lower or "holdings" in prompt_lower:
                if not kite:
                    response = "Please connect to a broker to view portfolio."
                else:
                    try:
                        holdings = kite.holdings()
                        holdings_df = pd.DataFrame(holdings)
                        if holdings_df.empty:
                            response = "No holdings found."
                        else:
                            response = tabulate.tabulate(holdings_df[['tradingsymbol', 'quantity', 'average_price', 'last_price']], headers='keys', tablefmt='pretty')
                    except Exception as e:
                        response = f"Failed to fetch holdings: {e}"
            elif "positions" in prompt_lower:
                if not kite:
                    response = "Please connect to a broker to view positions."
                else:
                    try:
                        positions = kite.positions().get('net', [])
                        positions_df = pd.DataFrame(positions)
                        if positions_df.empty:
                            response = "No positions found."
                        else:
                            response = tabulate.tabulate(positions_df[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']], headers='keys', tablefmt='pretty')
                    except Exception as e:
                        response = f"Failed to fetch positions: {e}"
            elif "orders" in prompt_lower:
                if 'order_history' in st.session_state and st.session_state['order_history']:
                    history_df = pd.DataFrame(st.session_state['order_history'])
                    response = tabulate.tabulate(history_df[['id', 'symbol', 'qty', 'type', 'status']], headers='keys', tablefmt='pretty')
                else:
                    response = "No orders placed yet."
            elif "funds" in prompt_lower:
                if not kite:
                    response = "Please connect to a broker to view funds."
                else:
                    try:
                        margins = kite.margins()
                        cash = margins['equity']['available']['cash']
                        response = f"Available Funds: ₹{cash:.2f}"
                    except Exception as e:
                        response = f"Failed to fetch funds: {e}"
            elif "price" in prompt_lower:
                try:
                    symbol = next((part.upper() for part in prompt_lower.split() if part.upper() in instrument_df['tradingsymbol'].values), None)
                    if symbol and kite:
                        quote = kite.quote(f"NSE:{symbol}")["NSE:" + symbol]
                        response = f"Current price of {symbol}: ₹{quote['last_price']:.2f}"
                    else:
                        response = "Please specify a valid symbol, e.g., 'Price of RELIANCE'."
                except Exception as e:
                    response = f"Failed to fetch price: {e}"
            elif any(word in prompt_lower for word in ["technical", "indicators", "rsi", "macd", "bollinger"]):
                try:
                    symbol = next((part.upper() for part in prompt_lower.split() if part.upper() in instrument_df['tradingsymbol'].values), None)
                    if symbol:
                        df = get_historical_data(symbol, datetime.now() - timedelta(days=30), datetime.now(), "day")
                        if not df.empty:
                            df['RSI'] = rsi(df['close'])
                            macd_data = macd(df['close'])
                            bbands_data = bbands(df['close'])
                            latest = df.iloc[-1]
                            response = (
                                f"**Technical Indicators for {symbol}**\n"
                                f"- RSI: {latest['RSI']:.2f}\n"
                                f"- MACD: {macd_data['MACD'].iloc[-1]:.2f}, Signal: {macd_data['Signal'].iloc[-1]:.2f}\n"
                                f"- Bollinger Bands: Upper {bbands_data['Upper'].iloc[-1]:.2f}, Lower {bbands_data['Lower'].iloc[-1]:.2f}"
                            )
                        else:
                            response = f"No data available for {symbol}."
                    else:
                        response = "Please specify a valid symbol, e.g., 'Technical indicators for RELIANCE'."
                except Exception as e:
                    response = f"Failed to fetch technical indicators: {e}"
            elif "news" in prompt_lower:
                try:
                    symbol = next((part.upper() for part in prompt_lower.split() if part.upper() in instrument_df['tradingsymbol'].values), None)
                    if symbol:
                        news_df = fetch_and_analyze_news(symbol)
                        if not news_df.empty:
                            response = "\n".join([f"- [{row['title']}]({row['link']}) (Sentiment: {row['sentiment']:.2f})" for _, row in news_df.iterrows()])
                        else:
                            response = f"No news available for {symbol}."
                    else:
                        response = "Please specify a valid symbol, e.g., 'News for RELIANCE'."
                except Exception as e:
                    response = f"Failed to fetch news: {e}"
            elif any(word in prompt_lower for word in ["option chain", "options chain"]):
                try:
                    underlying = next((part.upper() for part in prompt_lower.split() if part.upper() in ["NIFTY 50", "BANK NIFTY"]), None)
                    if underlying:
                        chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df)
                        if not chain_df.empty:
                            response = (
                                f"**{underlying} Options Chain (Expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')}, LTP: {underlying_ltp:.2f})**\n"
                                f"{tabulate.tabulate(chain_df, headers='keys', tablefmt='pretty')}"
                            )
                        else:
                            response = f"No options chain data available for {underlying}."
                    else:
                        response = "Please specify a valid underlying, e.g., 'NIFTY 50 options chain'."
                except Exception as e:
                    response = f"Failed to fetch options chain: {e}"
            elif any(word in prompt_lower for word in ["greeks for", "iv for"]):
                try:
                    option_symbol = next((part.upper() for part in prompt_lower.split() if part.upper() in instrument_df['tradingsymbol'].values), None)
                    if option_symbol:
                        option_details = instrument_df[instrument_df['tradingsymbol'] == option_symbol].iloc[0]
                        underlying = option_details['name']
                        strike_price = option_details['strike']
                        option_type = option_details['instrument_type'].lower()
                        chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df)
                        ltp_col = 'CALL LTP' if option_type == 'ce' else 'PUT LTP'
                        symbol_col = 'CALL' if option_type == 'ce' else 'PUT'
                        ltp = chain_df[chain_df[symbol_col] == option_symbol][ltp_col].iloc[0]
                        T = max((pd.to_datetime(expiry).date() - datetime.now().date()).days, 0) / 365.0
                        iv = implied_volatility(underlying_ltp, strike_price, T, 0.07, ltp, option_type)
                        if not np.isnan(iv) and iv > 0:
                            greeks = black_scholes(underlying_ltp, strike_price, T, 0.07, iv, option_type)
                            response = (
                                f"**Greeks for {option_symbol} ({option_type.upper()})**\n"
                                f"Strike: {strike_price}, Spot: {underlying_ltp:.2f}, Expiry: {pd.to_datetime(expiry).strftime('%d %b %Y')}\n"
                                f"- **Price**: {greeks['price']:.2f}\n"
                                f"- **Delta**: {greeks['delta']:.3f}\n"
                                f"- **Gamma**: {greeks['gamma']:.3f}\n"
                                f"- **Vega**: {greeks['vega']:.3f}\n"
                                f"- **Theta**: {greeks['theta']:.3f}\n"
                                f"- **Rho**: {greeks['rho']:.3f}\n"
                                f"- **Implied Volatility**: {iv*100:.2f}%"
                            )
                        else:
                            response = f"Could not calculate Greeks or IV for {option_symbol}. Market price or time to expiry may be invalid."
                    else:
                        response = "Please specify a valid option symbol, e.g., 'Greeks for NIFTY25OCT25100CE'."
                except Exception as e:
                    response = f"Error calculating Greeks: {e}"
            elif any(word in prompt_lower for word in ["buy", "sell", "trade"]):
                try:
                    parts = prompt_lower.split()
                    action = "BUY" if "buy" in prompt_lower else "SELL"
                    symbol = next((part.upper() for part in parts if part.upper() in instrument_df['tradingsymbol'].values), None)
                    quantity = next((int(part) for part in parts if part.isdigit()), 1)
                    if symbol:
                        place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
                        response = f"Order placed: {action} {quantity} {symbol}."
                    else:
                        response = "Please specify a valid symbol to trade, e.g., 'Buy 100 RELIANCE'."
                except Exception as e:
                    response = f"Failed to place order: {e}"
            else:
                response = "I didn't understand your request. Try asking about portfolio, positions, orders, funds, price, technical indicators, news, option chain, or Greeks."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def page_basket_orders():
    display_header()
    st.title("Basket Orders")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the Basket Orders feature.")
        return
    if 'basket_orders' not in st.session_state:
        st.session_state.basket_orders = []
    if 'basket_name' not in st.session_state:
        st.session_state.basket_name = "Basket 1"
    st.subheader("Create Basket Order")
    with st.form(key="basket_form"):
        basket_name = st.text_input("Basket Name", value=st.session_state.basket_name)
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
        symbol = col1.text_input("Symbol", placeholder="e.g., RELIANCE").upper()
        quantity = col2.number_input("Quantity", min_value=1, step=1, value=1)
        transaction_type = col3.selectbox("Type", ["BUY", "SELL"])
        order_type = col4.selectbox("Order Type", ["MARKET", "LIMIT"])
        product = col5.selectbox("Product", ["MIS", "CNC"])
        price = col6.number_input("Price", min_value=0.0, step=0.05, value=0.0) if order_type == "LIMIT" else 0.0
        if st.form_submit_button("Add to Basket"):
            if symbol in instrument_df['tradingsymbol'].values:
                basket_order = {
                    'tradingsymbol': symbol,
                    'quantity': quantity,
                    'transaction_type': transaction_type,
                    'order_type': order_type,
                    'product': product,
                    'price': price if order_type == "LIMIT" else None,
                    'exchange': instrument_df[instrument_df['tradingsymbol'] == symbol]['exchange'].iloc[0]
                }
                st.session_state.basket_orders.append(basket_order)
                st.session_state.basket_name = basket_name
                st.success(f"Added {symbol} to basket '{basket_name}'.")
                st.rerun()
            else:
                st.error("Please enter a valid symbol.")
    if st.session_state.basket_orders:
        st.subheader(f"Basket: {st.session_state.basket_name}")
        basket_df = pd.DataFrame(st.session_state.basket_orders)
        st.dataframe(basket_df[['tradingsymbol', 'exchange', 'quantity', 'transaction_type', 'order_type', 'product', 'price']], use_container_width=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Execute Basket Order"):
                with st.spinner("Executing basket order..."):
                    orders = [
                        {
                            'tradingsymbol': row['tradingsymbol'],
                            'exchange': row['exchange'],
                            'quantity': row['quantity'],
                            'transaction_type': row['transaction_type'],
                            'order_type': row['order_type'],
                            'product': row['product'],
                            'price': row['price'] if row['price'] is not None else 0
                        } for _, row in basket_df.iterrows()
                    ]
                    place_basket_order(orders, variety='regular')
                    st.session_state.basket_orders = []
                    st.rerun()
        with col2:
            if st.button("Clear Basket"):
                st.session_state.basket_orders = []
                st.success(f"Cleared basket '{st.session_state.basket_name}'.")
                st.rerun()
        with col3:
            with st.form(key="remove_from_basket"):
                symbol_to_remove = st.selectbox("Remove Symbol", [order['tradingsymbol'] for order in st.session_state.basket_orders])
                if st.form_submit_button("Remove"):
                    st.session_state.basket_orders = [order for order in st.session_state.basket_orders if order['tradingsymbol'] != symbol_to_remove]
                    st.success(f"Removed {symbol_to_remove} from basket.")
                    st.rerun()
    st.subheader("Order History")
    if 'order_history' in st.session_state and st.session_state.order_history:
        history_df = pd.DataFrame(st.session_state.order_history)
        st.dataframe(history_df[['id', 'symbol', 'qty', 'type', 'status']], use_container_width=True)
    else:
        st.info("No orders have been placed yet.")

def main():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Dark'
    if 'order_history' not in st.session_state:
        st.session_state.order_history = []
    with st.sidebar:
        st.header("BlockVista Terminal")
        if 'kite' not in st.session_state or not st.session_state.get('kite'):
            authenticate_broker()
        else:
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()
        page = st.selectbox(
            "Navigate",
            [
                "Dashboard",
                "Advanced Charting",
                "Options Hub",
                "Futures Hub",
                "Alpha Engine: News Sentiment",
                "Portfolio & Risk",
                "Forecasting (ML)",
                "Portfolio-Aware Assistant",
                "Basket Orders"
            ]
        )
        st.markdown("---")
        st.markdown(f"**Broker Connected:** {st.session_state.get('broker', 'None')}")
        st.markdown(f"**Theme:** {st.session_state.theme}")
        if st.button("Toggle Theme"):
            st.session_state.theme = 'Light' if st.session_state.theme == 'Dark' else 'Dark'
            st.rerun()
    if page == "Dashboard":
        page_dashboard()
    elif page == "Advanced Charting":
        page_advanced_charting()
    elif page == "Options Hub":
        page_options_hub()
    elif page == "Futures Hub":
        page_futures_hub()
    elif page == "Alpha Engine: News Sentiment":
        page_alpha_engine()
    elif page == "Portfolio & Risk":
        page_portfolio_and_risk()
    elif page == "Forecasting (ML)":
        page_forecasting_ml()
    elif page == "Portfolio-Aware Assistant":
        page_ai_assistant()
    elif page == "Basket Orders":
        page_basket_orders()

if __name__ == "__main__":
    main()
