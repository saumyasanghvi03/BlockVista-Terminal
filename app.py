import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import time
import datetime
import hashlib
import base64
import pyotp
import qrcode
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import kiteconnect, but provide fallback for demo mode
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.warning("KiteConnect not available. Running in demo mode.")

# ============ 1. CONFIGURATION AND STYLING ============

def apply_custom_styling():
    """Apply professional trading terminal styling."""
    st.set_page_config(
        page_title="BlockVista Terminal Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    dark_theme = """
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1668A4;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1F77B4;
    }
    .positive {
        color: #00D474;
    }
    .negative {
        color: #FF4B4B;
    }
    </style>
    """
    
    light_theme = """
    <style>
    .main {
        background-color: #FFFFFF;
        color: #31333F;
    }
    .stSidebar {
        background-color: #F0F2F6;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1F77B4;
    }
    .positive {
        color: #00D474;
    }
    .negative {
        color: #FF4B4B;
    }
    </style>
    """
    
    theme = st.session_state.get('theme', 'Dark')
    if theme == 'Dark':
        st.markdown(dark_theme, unsafe_allow_html=True)
    else:
        st.markdown(light_theme, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'broker' not in st.session_state:
        st.session_state.broker = None
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'kite' not in st.session_state:
        st.session_state.kite = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Dark'
    if 'login_animation_complete' not in st.session_state:
        st.session_state.login_animation_complete = False
    if 'two_factor_setup_complete' not in st.session_state:
        st.session_state.two_factor_setup_complete = False
    if 'show_2fa_dialog' not in st.session_state:
        st.session_state.show_2fa_dialog = False
    if 'show_qr_dialog' not in st.session_state:
        st.session_state.show_qr_dialog = False
    if 'pyotp_secret' not in st.session_state:
        st.session_state.pyotp_secret = None
    if 'subscription_tier' not in st.session_state:
        st.session_state.subscription_tier = 'Professional'

# ============ 2. MARKET DATA FUNCTIONS ============

def get_market_status():
    """Get current market status with error handling."""
    try:
        # Mock market status for demo
        current_time = datetime.datetime.now().time()
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        
        if market_open <= current_time <= market_close:
            return {'status': 'OPEN', 'message': 'Market is currently open'}
        else:
            return {'status': 'CLOSED', 'message': 'Market is currently closed'}
    except Exception as e:
        return {'status': 'UNKNOWN', 'message': f'Status check failed: {str(e)}'}

def get_futures_data(symbol):
    """Safe futures data fetching with error handling."""
    try:
        # Mock data for demo
        if "NIFTY" in symbol:
            return {
                'last_price': 22150.45,
                'change': 125.30,
                'change_percent': 0.57,
                'volume': 125000
            }
        elif "BANKNIFTY" in symbol:
            return {
                'last_price': 47230.15,
                'change': 280.45,
                'change_percent': 0.60,
                'volume': 89000
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching {symbol} Futures: {str(e)}")
        return None

def get_live_quotes(symbols):
    """Get live quotes for symbols."""
    # Mock data for demo
    quotes = {}
    for symbol in symbols:
        base_price = 1000 if symbol == "RELIANCE" else 500
        change = np.random.uniform(-10, 10)
        quotes[symbol] = {
            'last_price': base_price + change,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'volume': np.random.randint(10000, 100000)
        }
    return quotes

def get_options_chain(symbol):
    """Get options chain data."""
    # Mock options chain for demo
    strikes = [22000, 22100, 22200, 22300, 22400]
    chain_data = []
    
    for strike in strikes:
        chain_data.append({
            'strike': strike,
            'call_oi': np.random.randint(1000, 10000),
            'put_oi': np.random.randint(1000, 10000),
            'call_volume': np.random.randint(100, 1000),
            'put_volume': np.random.randint(100, 1000),
            'call_iv': np.random.uniform(10, 20),
            'put_iv': np.random.uniform(10, 20)
        })
    
    return pd.DataFrame(chain_data)

# ============ 3. PAGE FUNCTIONS ============

def page_dashboard():
    """Main dashboard page."""
    st.title("Trading Dashboard")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nifty_data = get_futures_data("NIFTY")
        if nifty_data:
            st.metric("NIFTY 50", f"{nifty_data['last_price']:,.2f}", 
                     f"{nifty_data['change']:+.2f} ({nifty_data['change_percent']:+.2f}%)")
    
    with col2:
        banknifty_data = get_futures_data("BANKNIFTY")
        if banknifty_data:
            st.metric("BANK NIFTY", f"{banknifty_data['last_price']:,.2f}", 
                     f"{banknifty_data['change']:+.2f} ({banknifty_data['change_percent']:+.2f}%)")
    
    with col3:
        st.metric("SENSEX", "73,142.45", "+245.18 (+0.34%)")
    
    with col4:
        st.metric("NSE Advance/Decline", "1,245/856", "+389")
    
    st.markdown("---")
    
    # Portfolio overview and charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Performance")
        # Sample price chart
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='NIFTY 50',
                                line=dict(color='#1F77B4', width=2)))
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Watchlist")
        watchlist_symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"]
        quotes = get_live_quotes(watchlist_symbols)
        
        for symbol in watchlist_symbols:
            if symbol in quotes:
                quote = quotes[symbol]
                change_color = "positive" if quote['change'] >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{symbol}</h4>
                    <h3>{quote['last_price']:.2f}</h3>
                    <p class="{change_color}">{quote['change']:+.2f} ({quote['change_percent']:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

def page_advanced_charting():
    """Advanced charting page."""
    st.title("Advanced Charting")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        symbol = st.selectbox("Select Symbol", ["NIFTY 50", "BANK NIFTY", "RELIANCE", "TCS", "INFY"])
        timeframe = st.selectbox("Timeframe", ["1min", "5min", "15min", "1H", "1D"])
        indicators = st.multiselect("Technical Indicators", 
                                  ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"])
    
    with col2:
        # Generate sample OHLC data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(100) * 0.3)
        low_prices = close_prices - np.abs(np.random.randn(100) * 0.3)
        open_prices = close_prices - np.random.randn(100) * 0.2
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name=symbol
        )])
        
        # Add selected indicators
        if "SMA" in indicators:
            sma = close_prices.rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=dates, y=sma, mode='lines', name='SMA 20',
                                   line=dict(color='orange', width=1)))
        
        if "Bollinger Bands" in indicators:
            sma = close_prices.rolling(window=20).mean()
            std = close_prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            fig.add_trace(go.Scatter(x=dates, y=upper_band, mode='lines', 
                                   name='Upper BB', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=dates, y=lower_band, mode='lines', 
                                   name='Lower BB', line=dict(color='gray', width=1),
                                   fill='tonexty'))
        
        fig.update_layout(
            height=600,
            title=f"{symbol} Price Chart - {timeframe}",
            xaxis_title="Date",
            yaxis_title="Price",
            template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def page_premarket_pulse():
    """Premarket analysis page."""
    st.title("Premarket Pulse")
    
    # Overnight global markets
    st.subheader("Global Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("S&P 500 Futures", "4,825.45", "+15.25 (+0.32%)")
    with col2:
        st.metric("NASDAQ Futures", "17,245.65", "+45.85 (+0.27%)")
    with col3:
        st.metric("Dow Futures", "38,425.75", "+125.45 (+0.33%)")
    with col4:
        st.metric("SGX NIFTY", "22,185.25", "+85.45 (+0.39%)")
    
    st.markdown("---")
    
    # Most active premarket
    st.subheader("Most Active Premarket")
    
    active_stocks = [
        {"symbol": "RELIANCE", "price": 2850.45, "change": 25.45, "change_percent": 0.90, "volume": 1250000},
        {"symbol": "TCS", "price": 3850.75, "change": -45.25, "change_percent": -1.16, "volume": 985000},
        {"symbol": "INFY", "price": 1685.25, "change": 12.85, "change_percent": 0.77, "volume": 756000},
        {"symbol": "HDFC", "price": 1685.45, "change": 8.75, "change_percent": 0.52, "volume": 642000},
    ]
    
    for stock in active_stocks:
        change_color = "positive" if stock['change'] >= 0 else "negative"
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{stock['symbol']}**")
        with col2:
            st.write(f"₹{stock['price']:.2f}")
        with col3:
            st.markdown(f'<p class="{change_color}">{stock["change"]:+.2f} ({stock["change_percent"]:+.2f}%)</p>', 
                       unsafe_allow_html=True)
        with col4:
            st.write(f"{stock['volume']:,}")

def page_fo_analytics():
    """F&O Analytics page."""
    st.title("F&O Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "Futures OI", "PCR Analysis"])
    
    with tab1:
        st.subheader("NIFTY Options Chain")
        
        chain_data = get_options_chain("NIFTY")
        
        # Display options chain
        st.dataframe(chain_data, use_container_width=True)
        
        # PCR Chart
        st.subheader("Put-Call Ratio (PCR)")
        pcr_values = np.random.uniform(0.8, 1.2, 20)
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=pcr_values, mode='lines', name='PCR',
                                line=dict(color='red', width=2)))
        fig.add_hline(y=1.0, line_dash="dash", line_color="white")
        fig.update_layout(height=300, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Futures Open Interest Analysis")
        # Add futures OI analysis content here
        st.info("Futures Open Interest analysis coming soon...")
    
    with tab3:
        st.subheader("PCR Trend Analysis")
        # Add PCR analysis content here
        st.info("PCR Trend analysis coming soon...")

def page_ai_trade_signals():
    """AI Trade Signals page."""
    st.title("AI Trade Signals")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current AI Signals")
        
        signals = [
            {"symbol": "RELIANCE", "signal": "BUY", "confidence": 0.85, "target": 2950, "stop_loss": 2750},
            {"symbol": "TCS", "signal": "SELL", "confidence": 0.72, "target": 3750, "stop_loss": 3950},
            {"symbol": "INFY", "signal": "BUY", "confidence": 0.68, "target": 1750, "stop_loss": 1600},
            {"symbol": "HDFC", "signal": "HOLD", "confidence": 0.55, "target": 1700, "stop_loss": 1650},
        ]
        
        for signal in signals:
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                with col1:
                    st.write(f"**{signal['symbol']}**")
                with col2:
                    color = "positive" if signal['signal'] == 'BUY' else "negative" if signal['signal'] == 'SELL' else "white"
                    st.markdown(f'<p class="{color}"><strong>{signal["signal"]}</strong></p>', unsafe_allow_html=True)
                with col3:
                    st.write(f"{signal['confidence']:.0%} conf")
                with col4:
                    st.write(f"Target: ₹{signal['target']} | SL: ₹{signal['stop_loss']}")
                st.markdown("---")
    
    with col2:
        st.subheader("Signal Filters")
        st.selectbox("Timeframe", ["Intraday", "Swing", "Positional"])
        st.selectbox("Risk Level", ["Low", "Medium", "High"])
        st.multiselect("Sectors", ["IT", "Banking", "Auto", "Pharma", "Energy"])
        
        st.markdown("---")
        st.subheader("Performance")
        st.metric("Win Rate", "68.5%", "2.3%")
        st.metric("Avg Return", "3.2%", "0.4%")
        st.metric("Sharpe Ratio", "1.45", "0.15")

def page_smart_money_flow():
    """Smart Money Flow page."""
    st.title("Smart Money Flow")
    st.info("Smart Money Flow analysis coming soon...")

def page_sector_rotation():
    """Sector Rotation page."""
    st.title("Sector Rotation")
    
    # Sector performance data
    sectors = ['IT', 'Banking', 'Auto', 'Pharma', 'Energy', 'FMCG', 'Metal', 'Realty']
    performance = np.random.uniform(-3, 3, len(sectors))
    
    fig = px.bar(x=sectors, y=performance, 
                 title="Sector Performance (%)",
                 color=performance,
                 color_continuous_scale='RdYlGn')
    fig.update_layout(height=400, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def page_forecasting_ml():
    """ML Forecasting page."""
    st.title("Machine Learning Forecasting")
    st.info("Machine Learning Forecasting models coming soon...")

def page_hft_terminal():
    """HFT Terminal page."""
    st.title("High Frequency Trading Terminal")
    st.warning("HFT Terminal - Professional and Institutional tier only")

def page_basket_orders():
    """Basket Orders page."""
    st.title("Basket Order Management")
    st.info("Basket order functionality coming soon...")

def page_fundamental_analysis():
    """Fundamental Analysis page."""
    st.title("Fundamental Analysis")
    st.info("Fundamental analysis tools coming soon...")

def page_ai_trade_assistant():
    """AI Assistant page."""
    st.title("AI Trading Assistant")
    
    st.subheader("Chat with AI Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask me about trading, analysis, or market insights..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Simulate AI response
        response = f"I understand you're asking about: {prompt}. This is a demo response. In production, this would connect to a real AI model."
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def page_subscription_plans():
    """Subscription Plans page."""
    st.title("Subscription Plans")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic")
        st.metric("Price", "₹999/month")
        st.write("• Live NSE/BSE Data")
        st.write("• Basic Charts")
        st.write("• Watchlist (10 symbols)")
        st.write("• Email Support")
        if st.button("Select Basic", key="basic"):
            st.session_state.subscription_tier = "Basic"
            st.success("Subscription updated to Basic")
    
    with col2:
        st.subheader("Professional")
        st.metric("Price", "₹2,999/month")
        st.write("• All Basic features")
        st.write("• Advanced Charting")
        st.write("• Options Chain")
        st.write("• AI Trade Signals")
        st.write("• Priority Support")
        if st.button("Select Professional", key="pro", type="primary"):
            st.session_state.subscription_tier = "Professional"
            st.success("Subscription updated to Professional")
    
    with col3:
        st.subheader("Institutional")
        st.metric("Price", "₹9,999/month")
        st.write("• All Professional features")
        st.write("• HFT Terminal")
        st.write("• Basket Orders")
        st.write("• API Access")
        st.write("• Dedicated Support")
        if st.button("Select Institutional", key="inst"):
            st.session_state.subscription_tier = "Institutional"
            st.success("Subscription updated to Institutional")

def display_overnight_changes_bar():
    """Display overnight changes bar at the top."""
    changes = [
        {"market": "US Markets", "change": "+0.45%"},
        {"market": "Europe", "change": "+0.23%"},
        {"market": "Asia", "change": "-0.12%"},
        {"market": "SGX Nifty", "change": "+0.38%"},
        {"market": "Crude Oil", "change": "-1.2%"},
        {"market": "Gold", "change": "+0.8%"},
        {"market": "USD/INR", "change": "-0.15%"},
    ]
    
    html = """
    <div style="background-color: #1F77B4; padding: 8px; border-radius: 4px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-around; color: white; font-weight: 500;">
    """
    
    for change in changes:
        color = "color: #00D474" if change["change"].startswith("+") else "color: #FF4B4B"
        html += f'<span>{change["market"]}: <span style="{color}">{change["change"]}</span></span>'
    
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

# ============ 4. AUTHENTICATION FUNCTIONS ============

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    secret = base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]
    return secret

def two_factor_dialog():
    """Dialog for 2FA login using container instead of st.dialog."""
    with st.container():
        st.subheader("Enter your 2FA code")
        st.caption("Please enter the 6-digit code from your authenticator app to continue.")
        
        auth_code = st.text_input("2FA Code", max_chars=6, key="2fa_code")
        
        col1, col2 = st.columns(2)
        
        if col1.button("Authenticate", use_container_width=True, type="primary"):
            if auth_code:
                try:
                    totp = pyotp.TOTP(st.session_state.pyotp_secret)
                    if totp.verify(auth_code):
                        st.session_state.authenticated = True
                        st.success("Authentication successful!")
                        st.rerun()
                    else:
                        st.error("Invalid code. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred during authentication: {e}")
            else:
                st.warning("Please enter a code.")
        
        if col2.button("Cancel", use_container_width=True):
            st.session_state.show_2fa_dialog = False
            st.rerun()

def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup using container."""
    with st.container():
        st.subheader("Set up Two-Factor Authentication")
        st.info("Please scan this QR code with your authenticator app (e.g., Google or Microsoft Authenticator). This is a one-time setup.")

        if st.session_state.pyotp_secret is None:
            st.session_state.pyotp_secret = get_user_secret(st.session_state.get('profile', {}))
        
        secret = st.session_state.pyotp_secret
        user_name = st.session_state.get('profile', {}).get('user_name', 'User')
        uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
        
        img = qrcode.make(uri)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        st.image(buf.getvalue(), caption="Scan with your authenticator app", use_container_width=True)
        st.markdown(f"**Your Secret Key:** `{secret}` (You can also enter this manually)")
        
        col1, col2 = st.columns(2)
        
        if col1.button("I have scanned the code. Continue.", use_container_width=True, type="primary"):
            st.session_state.two_factor_setup_complete = True
            st.session_state.show_qr_dialog = False
            st.success("2FA setup completed!")
            st.rerun()
        
        if col2.button("Setup Later", use_container_width=True):
            st.session_state.two_factor_setup_complete = True
            st.session_state.show_qr_dialog = False
            st.info("2FA setup skipped. You can set it up later in settings.")
            st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
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
        time.sleep(0.7)
    
    time.sleep(0.5)
    st.session_state['login_animation_complete'] = True
    st.rerun()

# ============ 5. LOGIN PAGE ============

def login_page():
    """Displays the login page for broker authentication."""
    st.title("BlockVista Terminal Pro")
    st.subheader("AI-Powered Trading Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Broker Login")
        broker = st.selectbox("Select Your Broker", ["Zerodha", "Angel One", "Upstox", "ICICI Direct", "Demo Mode"])
        
        if broker == "Zerodha":
            # Using Streamlit secrets for API credentials
            api_key = st.secrets.get("ZERODHA_API_KEY", "")
            api_secret = st.secrets.get("ZERODHA_API_SECRET", "")
            
            if api_key and api_secret:
                st.info("Using secured API credentials from environment")
                use_env_creds = True
            else:
                st.warning("API credentials not found in secrets. Please enter manually.")
                use_env_creds = False
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
            
            # Request token handling
            request_token = st.text_input("Request Token", type="password")
            
            if st.button("Connect to Zerodha", type="primary", use_container_width=True):
                if not api_key or not api_secret:
                    st.error("API Key and Secret are required")
                    return
                
                try:
                    kite = KiteConnect(api_key=api_key)
                    
                    if request_token:
                        with st.spinner("Authenticating with Zerodha..."):
                            data = kite.generate_session(request_token, api_secret=api_secret)
                            st.session_state.access_token = data["access_token"]
                            kite.set_access_token(st.session_state.access_token)
                            st.session_state.kite = kite
                            st.session_state.profile = kite.profile()
                            st.session_state.broker = "Zerodha"
                            st.session_state.authenticated = True
                            st.success("Successfully connected to Zerodha!")
                            st.rerun()
                    else:
                        # Show login URL if no request token provided
                        login_url = kite.login_url()
                        st.markdown(f"""
                        **Login Steps:**
                        1. Use the login link below
                        2. You'll be redirected to Zerodha
                        3. Login with your Zerodha credentials
                        4. Copy the request token from the redirect URL and paste above
                        """)
                        st.markdown(f"[Login with Zerodha Kite]({login_url})")
                        
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
        
        elif broker in ["Angel One", "Upstox", "ICICI Direct"]:
            st.warning(f"{broker} integration coming soon!")
            st.info("Currently, only Zerodha integration is available. Please select Zerodha or try Demo Mode.")
        
        elif broker == "Demo Mode":
            st.info("Entering demo mode with sample data")
            if st.button("Enter Demo Mode", type="primary", use_container_width=True):
                st.session_state.broker = "Demo"
                st.session_state.authenticated = True
                st.session_state.profile = {'user_name': 'Demo User', 'user_id': 'demo_001'}
                st.session_state.two_factor_setup_complete = True
                st.success("Welcome to Demo Mode!")
                st.rerun()
    
    with col2:
        st.subheader("Features")
        st.markdown("""
        - **AI-Powered Trade Signals**
        - **Advanced Charting & Analytics**
        - **Real-time Market Data**
        - **F&O Analytics & Options Chain**
        - **Basket Orders & Portfolio Management**
        - **Fundamental Analysis**
        - **Mobile Responsive Design**
        """)
        
        st.markdown("---")
        st.subheader("Security")
        st.markdown("""
        - Bank-grade encryption
        - Two-factor authentication
        - Secure API connections
        - No data storage on servers
        """)
        
        st.markdown("---")
        st.markdown("**Need help?**")
        st.markdown("[Documentation](https://github.com/saumyasanghvi03/BlockVista-Terminal) • [Support](mailto:support@blockvista.com)")

# ============ 6. MAIN APPLICATION ============

def main_app():
    """The main application interface after successful login."""
    apply_custom_styling()
    display_overnight_changes_bar()
    
    # Show login animation if not completed
    if not st.session_state.get('login_animation_complete', False):
        show_login_animation()
        return
    
    # Handle 2FA setup and authentication
    if st.session_state.get('profile') and not st.session_state.get('two_factor_setup_complete', False):
        if not st.session_state.get('show_qr_dialog', True):
            st.session_state.show_qr_dialog = True
        if st.session_state.show_qr_dialog:
            qr_code_dialog()
            return
    
    if st.session_state.get('profile') and not st.session_state.get('authenticated', False):
        if not st.session_state.get('show_2fa_dialog', True):
            st.session_state.show_2fa_dialog = True
        if st.session_state.show_2fa_dialog:
            two_factor_dialog()
            return

    # Main application sidebar and content
    with st.sidebar:
        st.title("Navigation")
        
        # User profile section
        if st.session_state.profile:
            user_name = st.session_state.profile.get('user_name', 'Trader')
            broker = st.session_state.get('broker', 'Demo')
            st.markdown(f"**Welcome, {user_name}**")
            st.caption(f"Connected via {broker}")
        
        # Subscription tier
        tier = st.session_state.get('subscription_tier', 'Professional')
        st.markdown(f"**Plan:** {tier}")
        
        st.markdown("---")
        
        # Navigation options
        nav_options = {
            "Dashboard": page_dashboard,
            "Advanced Charting": page_advanced_charting,
            "Premarket Pulse": page_premarket_pulse,
            "F&O Analytics": page_fo_analytics,
            "AI Trade Signals": page_ai_trade_signals,
            "Smart Money Flow": page_smart_money_flow,
            "Sector Rotation": page_sector_rotation,
            "ML Forecasting": page_forecasting_ml,
            "HFT Terminal": page_hft_terminal,
            "Basket Orders": page_basket_orders,
            "Fundamental Analysis": page_fundamental_analysis,
            "AI Assistant": page_ai_trade_assistant,
            "Subscription": page_subscription_plans
        }
        
        selected_page = st.radio("Go to", list(nav_options.keys()))
        
        st.markdown("---")
        
        # Theme selector
        theme = st.radio("Theme", ["Dark", "Light"], horizontal=True)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        
        if col1.button("Refresh", use_container_width=True):
            st.rerun()
        
        if col2.button("Trade", use_container_width=True):
            st.session_state.show_quick_trade = True
            st.rerun()
        
        # Market status
        st.markdown("---")
        st.subheader("Market Status")
        market_status = get_market_status()
        st.markdown(f"**Status:** {market_status['status']}")
        
        # Logout
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Logged out successfully!")
            st.rerun()
    
    # Main content area - show selected page
    if selected_page in nav_options:
        nav_options[selected_page]()
    else:
        page_dashboard()  # Default page

def main():
    """Main entry point."""
    initialize_session_state()
    apply_custom_styling()
    
    if not st.session_state.get('authenticated', False):
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
