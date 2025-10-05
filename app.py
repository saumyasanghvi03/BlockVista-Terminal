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
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# Try to import kiteconnect, but provide fallback for demo mode
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
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
        margin-bottom: 0.5rem;
    }
    .positive {
        color: #00D474;
    }
    .negative {
        color: #FF4B4B;
    }
    .header-bar {
        background: linear-gradient(90deg, #1F77B4 0%, #1668A4 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .market-status-open {
        color: #00D474;
        font-weight: bold;
    }
    .market-status-closed {
        color: #FF4B4B;
        font-weight: bold;
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
        margin-bottom: 0.5rem;
    }
    .positive {
        color: #00D474;
    }
    .negative {
        color: #FF4B4B;
    }
    .header-bar {
        background: linear-gradient(90deg, #1F77B4 0%, #1668A4 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: white;
    }
    .market-status-open {
        color: #00D474;
        font-weight: bold;
    }
    .market-status-closed {
        color: #FF4B4B;
        font-weight: bold;
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
    default_states = {
        'authenticated': False,
        'broker': None,
        'profile': None,
        'kite': None,
        'access_token': None,
        'theme': 'Dark',
        'login_animation_complete': False,
        'two_factor_setup_complete': False,
        'show_2fa_dialog': False,
        'show_qr_dialog': False,
        'pyotp_secret': None,
        'subscription_tier': 'Professional',
        'watchlist': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],
        'portfolio': {},
        'orders': [],
        'positions': [],
        'ai_chat_history': []
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============ 2. MARKET DATA FUNCTIONS ============

@st.cache_data(ttl=60)
def get_market_status() -> Dict:
    """Get current market status with error handling."""
    try:
        current_time = datetime.datetime.now().time()
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        
        if market_open <= current_time <= market_close:
            return {'status': 'OPEN', 'message': 'Market is currently open', 'css_class': 'market-status-open'}
        else:
            return {'status': 'CLOSED', 'message': 'Market is currently closed', 'css_class': 'market-status-closed'}
    except Exception as e:
        return {'status': 'UNKNOWN', 'message': f'Status check failed: {str(e)}', 'css_class': 'market-status-closed'}

@st.cache_data(ttl=30)
def get_futures_data(symbol: str) -> Optional[Dict]:
    """Safe futures data fetching with error handling."""
    try:
        # Mock data for demo - in production, replace with actual API calls
        futures_data = {
            "NIFTY": {'last_price': 22150.45, 'change': 125.30, 'change_percent': 0.57, 'volume': 125000},
            "BANKNIFTY": {'last_price': 47230.15, 'change': 280.45, 'change_percent': 0.60, 'volume': 89000},
            "SENSEX": {'last_price': 73142.45, 'change': 245.18, 'change_percent': 0.34, 'volume': 0},
            "S&P500": {'last_price': 4825.45, 'change': 15.25, 'change_percent': 0.32, 'volume': 0},
            "NASDAQ": {'last_price': 17245.65, 'change': 45.85, 'change_percent': 0.27, 'volume': 0}
        }
        
        return futures_data.get(symbol)
    except Exception as e:
        st.error(f"Error fetching {symbol} Futures: {str(e)}")
        return None

@st.cache_data(ttl=10)
def get_live_quotes(symbols: List[str]) -> Dict:
    """Get live quotes for symbols."""
    quotes = {}
    for symbol in symbols:
        # Base prices for common stocks
        base_prices = {
            'RELIANCE': 2850, 'TCS': 3850, 'INFY': 1685, 'HDFCBANK': 1685, 
            'ICICIBANK': 1050, 'HINDUNILVR': 2450, 'ITC': 430, 'SBIN': 750,
            'BHARTIARTL': 1150, 'KOTAKBANK': 1750
        }
        
        base_price = base_prices.get(symbol, 1000)
        change = np.random.uniform(-20, 20)
        change_percent = (change / base_price) * 100
        
        quotes[symbol] = {
            'last_price': base_price + change,
            'change': change,
            'change_percent': change_percent,
            'volume': np.random.randint(10000, 1000000),
            'high': base_price + abs(change) + np.random.uniform(5, 20),
            'low': base_price - abs(change) - np.random.uniform(5, 20),
            'open': base_price + np.random.uniform(-10, 10)
        }
    
    return quotes

@st.cache_data(ttl=60)
def get_options_chain(symbol: str) -> pd.DataFrame:
    """Get options chain data."""
    try:
        strikes = list(range(22000, 22500, 100))
        chain_data = []
        
        for strike in strikes:
            call_oi = np.random.randint(1000, 50000)
            put_oi = np.random.randint(1000, 50000)
            
            chain_data.append({
                'strike': strike,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'call_volume': np.random.randint(100, 5000),
                'put_volume': np.random.randint(100, 5000),
                'call_iv': np.random.uniform(10, 25),
                'put_iv': np.random.uniform(10, 25),
                'call_ltp': max(0.05, np.random.uniform(0.1, 500)),
                'put_ltp': max(0.05, np.random.uniform(0.1, 500))
            })
        
        return pd.DataFrame(chain_data)
    except Exception as e:
        st.error(f"Error fetching options chain: {str(e)}")
        return pd.DataFrame()

def get_sector_performance() -> Dict:
    """Get sector performance data."""
    sectors = ['IT', 'Banking', 'Auto', 'Pharma', 'Energy', 'FMCG', 'Metal', 'Realty', 'Media', 'Chemicals']
    performance = {sector: np.random.uniform(-4, 4) for sector in sectors}
    return performance

# ============ 3. TRADING FUNCTIONS ============

def place_order(symbol: str, transaction_type: str, quantity: int, order_type: str = "MARKET", price: float = 0):
    """Place a mock order (in demo mode)."""
    try:
        order_id = f"ORD{int(time.time())}{np.random.randint(1000, 9999)}"
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'transaction_type': transaction_type,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'status': 'COMPLETE',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to session state orders
        if 'orders' not in st.session_state:
            st.session_state.orders = []
        st.session_state.orders.append(order)
        
        # Update portfolio
        if symbol not in st.session_state.portfolio:
            st.session_state.portfolio[symbol] = {
                'quantity': 0,
                'average_price': 0,
                'current_value': 0
            }
        
        if transaction_type.upper() == 'BUY':
            st.session_state.portfolio[symbol]['quantity'] += quantity
        else:  # SELL
            st.session_state.portfolio[symbol]['quantity'] -= quantity
        
        return order_id
    except Exception as e:
        st.error(f"Error placing order: {str(e)}")
        return None

def get_portfolio_summary():
    """Get portfolio summary."""
    portfolio = st.session_state.get('portfolio', {})
    total_value = 0
    total_pnl = 0
    
    quotes = get_live_quotes(list(portfolio.keys()))
    
    for symbol, holding in portfolio.items():
        if symbol in quotes:
            current_price = quotes[symbol]['last_price']
            current_value = holding['quantity'] * current_price
            holding['current_value'] = current_value
            holding['current_price'] = current_price
            holding['pnl'] = current_value - (holding.get('average_price', current_price) * holding['quantity'])
            
            total_value += current_value
            total_pnl += holding['pnl']
    
    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'holdings': portfolio
    }

# ============ 4. PAGE FUNCTIONS ============

def page_dashboard():
    """Main dashboard page."""
    st.title("Trading Dashboard")
    
    # Market overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nifty_data = get_futures_data("NIFTY")
        if nifty_data:
            st.metric("NIFTY 50", f"₹{nifty_data['last_price']:,.2f}", 
                     f"₹{nifty_data['change']:+.2f} ({nifty_data['change_percent']:+.2f}%)")
    
    with col2:
        banknifty_data = get_futures_data("BANKNIFTY")
        if banknifty_data:
            st.metric("BANK NIFTY", f"₹{banknifty_data['last_price']:,.2f}", 
                     f"₹{banknifty_data['change']:+.2f} ({banknifty_data['change_percent']:+.2f}%)")
    
    with col3:
        sensex_data = get_futures_data("SENSEX")
        if sensex_data:
            st.metric("SENSEX", f"₹{sensex_data['last_price']:,.2f}", 
                     f"₹{sensex_data['change']:+.2f} ({sensex_data['change_percent']:+.2f}%)")
    
    with col4:
        market_status = get_market_status()
        st.metric("Market Status", market_status['status'])
    
    st.markdown("---")
    
    # Portfolio and Market Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Performance")
        
        # Generate sample price data
        dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=30), 
                             periods=30, freq='D')
        prices = 22000 + np.cumsum(np.random.randn(30) * 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='NIFTY 50',
                                line=dict(color='#1F77B4', width=3)))
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick Trade Section
        st.subheader("Quick Trade")
        trade_col1, trade_col2, trade_col3, trade_col4 = st.columns(4)
        
        with trade_col1:
            trade_symbol = st.selectbox("Symbol", st.session_state.watchlist, key="trade_symbol")
        
        with trade_col2:
            trade_action = st.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        
        with trade_col3:
            trade_qty = st.number_input("Quantity", min_value=1, value=1, key="trade_qty")
        
        with trade_col4:
            st.write("")  # Spacer
            st.write("")
            if st.button("Execute Trade", type="primary", use_container_width=True):
                quote = get_live_quotes([trade_symbol])[trade_symbol]
                order_id = place_order(trade_symbol, trade_action, trade_qty, "MARKET", quote['last_price'])
                if order_id:
                    st.success(f"Order {order_id} executed successfully!")

    with col2:
        st.subheader("Watchlist")
        watchlist_symbols = st.session_state.watchlist
        quotes = get_live_quotes(watchlist_symbols)
        
        for symbol in watchlist_symbols:
            if symbol in quotes:
                quote = quotes[symbol]
                change_color = "positive" if quote['change'] >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{symbol}</h4>
                    <h3>₹{quote['last_price']:.2f}</h3>
                    <p class="{change_color}">{quote['change']:+.2f} ({quote['change_percent']:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Portfolio Summary
        st.subheader("Portfolio Summary")
        portfolio = get_portfolio_summary()
        st.metric("Total Value", f"₹{portfolio['total_value']:,.2f}")
        pnl_color = "positive" if portfolio['total_pnl'] >= 0 else "negative"
        st.markdown(f'<p class="{pnl_color}">P&L: ₹{portfolio["total_pnl"]:+,.2f}</p>', unsafe_allow_html=True)

def page_advanced_charting():
    """Advanced charting page."""
    st.title("Advanced Charting")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"])
        timeframe = st.selectbox("Timeframe", ["1min", "5min", "15min", "30min", "1H", "1D"])
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        indicators = st.multiselect("Technical Indicators", 
                                  ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Volume"])
    
    with col2:
        # Generate sample OHLC data
        periods = 100
        dates = pd.date_range(end=datetime.datetime.now(), periods=periods, freq='D')
        
        # Generate realistic price data with trends
        returns = np.random.normal(0.001, 0.02, periods)
        prices = 1000 * (1 + np.cumsum(returns))
        
        # Generate OHLC from price series
        closes = prices
        opens = closes * (1 + np.random.normal(0, 0.005, periods))
        highs = np.maximum(opens, closes) * (1 + abs(np.random.normal(0, 0.01, periods)))
        lows = np.minimum(opens, closes) * (1 - abs(np.random.normal(0, 0.01, periods)))
        volumes = np.random.randint(100000, 1000000, periods)
        
        # Create main chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=(f'{symbol} Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=dates, open=opens, high=highs, low=lows, close=closes,
                name=symbol
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=dates, y=closes, mode='lines', name=symbol,
                line=dict(color='#1F77B4', width=2)
            ), row=1, col=1)
        
        # Add volume
        if "Volume" in indicators:
            colors = ['red' if closes[i] < opens[i] else 'green' for i in range(len(closes))]
            fig.add_trace(go.Bar(
                x=dates, y=volumes, name='Volume', marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
        
        # Add technical indicators
        if "SMA" in indicators:
            sma_20 = pd.Series(closes).rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=dates, y=sma_20, mode='lines', 
                                   name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        
        if "EMA" in indicators:
            ema_12 = pd.Series(closes).ewm(span=12).mean()
            fig.add_trace(go.Scatter(x=dates, y=ema_12, mode='lines', 
                                   name='EMA 12', line=dict(color='purple', width=1)), row=1, col=1)
        
        if "Bollinger Bands" in indicators:
            sma_20 = pd.Series(closes).rolling(window=20).mean()
            std_20 = pd.Series(closes).rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            fig.add_trace(go.Scatter(x=dates, y=upper_band, mode='lines', 
                                   name='Upper BB', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=dates, y=lower_band, mode='lines', 
                                   name='Lower BB', line=dict(color='gray', width=1, dash='dash'),
                                   fill='tonexty'), row=1, col=1)
        
        fig.update_layout(
            height=700,
            title=f"{symbol} - {timeframe} Chart",
            xaxis_rangeslider_visible=False,
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
        sp500_data = get_futures_data("S&P500")
        if sp500_data:
            st.metric("S&P 500 Futures", f"${sp500_data['last_price']:,.2f}", 
                     f"{sp500_data['change']:+.2f} ({sp500_data['change_percent']:+.2f}%)")
    
    with col2:
        nasdaq_data = get_futures_data("NASDAQ")
        if nasdaq_data:
            st.metric("NASDAQ Futures", f"${nasdaq_data['last_price']:,.2f}", 
                     f"{nasdaq_data['change']:+.2f} ({nasdaq_data['change_percent']:+.2f}%)")
    
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
        {"symbol": "HDFCBANK", "price": 1685.45, "change": 8.75, "change_percent": 0.52, "volume": 642000},
        {"symbol": "ICICIBANK", "price": 1052.30, "change": 15.20, "change_percent": 1.47, "volume": 589000},
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
    
    st.markdown("---")
    
    # Market sentiment
    st.subheader("Market Sentiment")
    sentiment_data = {
        'Bullish': 65,
        'Neutral': 25,
        'Bearish': 10
    }
    
    fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                 color_discrete_map={'Bullish': '#00D474', 'Neutral': '#FFB800', 'Bearish': '#FF4B4B'})
    fig.update_layout(height=300, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def page_fo_analytics():
    """F&O Analytics page."""
    st.title("F&O Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "Futures OI", "PCR Analysis"])
    
    with tab1:
        st.subheader("NIFTY Options Chain")
        
        expiry_date = st.selectbox("Select Expiry", ["25-Jan-2024", "01-Feb-2024", "08-Feb-2024"])
        chain_data = get_options_chain("NIFTY")
        
        if not chain_data.empty:
            # Display options chain
            st.dataframe(chain_data.style.format({
                'call_oi': '{:,}',
                'put_oi': '{:,}',
                'call_volume': '{:,}',
                'put_volume': '{:,}',
                'call_iv': '{:.2f}',
                'put_iv': '{:.2f}',
                'call_ltp': '{:.2f}',
                'put_ltp': '{:.2f}'
            }), use_container_width=True, height=400)
            
            # OI Analysis Chart
            st.subheader("Open Interest Analysis")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=chain_data['strike'], y=chain_data['call_oi'], 
                               name='Call OI', marker_color='red'))
            fig.add_trace(go.Bar(x=chain_data['strike'], y=chain_data['put_oi'], 
                               name='Put OI', marker_color='green'))
            fig.update_layout(barmode='group', height=400, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Futures Open Interest Analysis")
        
        # Futures OI data
        futures_data = {
            'Symbol': ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY'],
            'OI': [1250000, 890000, 450000, 320000, 280000],
            'Change_OI': [25000, -15000, 12000, 8000, -5000],
            'Volume': [2850000, 1950000, 980000, 750000, 620000]
        }
        
        futures_df = pd.DataFrame(futures_data)
        st.dataframe(futures_df.style.format({
            'OI': '{:,}',
            'Change_OI': '{:+,}',
            'Volume': '{:,}'
        }), use_container_width=True)
        
        # OI Change Chart
        fig = px.bar(futures_df, x='Symbol', y='Change_OI', 
                     title='Futures OI Change',
                     color='Change_OI',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Put-Call Ratio Analysis")
        
        # PCR Data
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        pcr_values = np.random.uniform(0.7, 1.5, 20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=pcr_values, mode='lines+markers', 
                               name='PCR', line=dict(color='red', width=3)))
        fig.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Neutral Level")
        fig.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Overbought")
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="Oversold")
        
        fig.update_layout(
            height=500,
            title="Put-Call Ratio Trend",
            xaxis_title="Date",
            yaxis_title="PCR Value",
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

def page_ai_trade_signals():
    """AI Trade Signals page."""
    st.title("AI Trade Signals")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current AI Signals")
        
        signals = [
            {"symbol": "RELIANCE", "signal": "BUY", "confidence": 0.85, "target": 2950, "stop_loss": 2750, "timeframe": "Swing"},
            {"symbol": "TCS", "signal": "SELL", "confidence": 0.72, "target": 3750, "stop_loss": 3950, "timeframe": "Intraday"},
            {"symbol": "INFY", "signal": "BUY", "confidence": 0.68, "target": 1750, "stop_loss": 1600, "timeframe": "Positional"},
            {"symbol": "HDFCBANK", "signal": "HOLD", "confidence": 0.55, "target": 1700, "stop_loss": 1650, "timeframe": "Swing"},
            {"symbol": "ICICIBANK", "signal": "BUY", "confidence": 0.78, "target": 1100, "stop_loss": 980, "timeframe": "Intraday"},
        ]
        
        for signal in signals:
            with st.container():
                col_a, col_b, col_c, col_d, col_e = st.columns([1, 1, 1, 1, 2])
                with col_a:
                    st.write(f"**{signal['symbol']}**")
                with col_b:
                    color = "positive" if signal['signal'] == 'BUY' else "negative" if signal['signal'] == 'SELL' else "white"
                    st.markdown(f'<p class="{color}"><strong>{signal["signal"]}</strong></p>', unsafe_allow_html=True)
                with col_c:
                    # Confidence bar
                    confidence_color = "#00D474" if signal['confidence'] > 0.7 else "#FFB800" if signal['confidence'] > 0.5 else "#FF4B4B"
                    st.markdown(f"""
                    <div style="background: #262730; border-radius: 10px; padding: 2px;">
                        <div style="background: {confidence_color}; width: {signal['confidence']*100}%; 
                                 border-radius: 10px; padding: 5px; text-align: center; color: white;">
                            {signal['confidence']:.0%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d:
                    st.write(f"{signal['timeframe']}")
                with col_e:
                    st.write(f"Target: ₹{signal['target']} | SL: ₹{signal['stop_loss']}")
                st.markdown("---")
    
    with col2:
        st.subheader("Signal Filters")
        timeframe_filter = st.selectbox("Timeframe", ["All", "Intraday", "Swing", "Positional"])
        risk_level = st.selectbox("Risk Level", ["All", "Low", "Medium", "High"])
        sectors = st.multiselect("Sectors", ["IT", "Banking", "Auto", "Pharma", "Energy", "FMCG"])
        
        st.markdown("---")
        st.subheader("Performance Metrics")
        st.metric("Win Rate", "68.5%", "2.3%")
        st.metric("Avg Return per Trade", "3.2%", "0.4%")
        st.metric("Sharpe Ratio", "1.45", "0.15")
        st.metric("Max Drawdown", "-8.2%", "-0.3%")
        
        st.markdown("---")
        st.subheader("Model Confidence")
        st.progress(0.85, text="Overall Model Confidence: 85%")

def page_smart_money_flow():
    """Smart Money Flow page."""
    st.title("Smart Money Flow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Institutional Activity")
        
        institutional_data = {
            'Stock': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
            'FII_Buy': [1250, 890, 750, 620, 580],
            'FII_Sell': [980, 750, 820, 550, 420],
            'DII_Buy': [850, 620, 580, 490, 380],
            'DII_Sell': [720, 580, 490, 380, 320]
        }
        
        inst_df = pd.DataFrame(institutional_data)
        st.dataframe(inst_df.style.format({
            'FII_Buy': '₹{:,}Cr',
            'FII_Sell': '₹{:,}Cr', 
            'DII_Buy': '₹{:,}Cr',
            'DII_Sell': '₹{:,}Cr'
        }), use_container_width=True)
    
    with col2:
        st.subheader("Block Deals")
        
        block_deals = [
            {"stock": "RELIANCE", "quantity": "25L", "price": "₹2,845", "buyer": "FII", "seller": "Promoter"},
            {"stock": "TCS", "quantity": "15L", "price": "₹3,820", "buyer": "DII", "seller": "FII"},
            {"stock": "INFY", "quantity": "12L", "price": "₹1,680", "buyer": "FII", "seller": "DII"},
            {"stock": "HDFCBANK", "quantity": "8L", "price": "₹1,675", "buyer": "DII", "seller": "FII"},
        ]
        
        for deal in block_deals:
            with st.container():
                st.write(f"**{deal['stock']}** - {deal['quantity']} @ {deal['price']}")
                st.write(f"Buyer: {deal['buyer']} | Seller: {deal['seller']}")
                st.markdown("---")
    
    # Money Flow Chart
    st.subheader("Sector-wise Money Flow")
    sectors = ['Banking', 'IT', 'Auto', 'Pharma', 'Energy', 'FMCG']
    money_flow = [1250, 980, -450, 320, -280, 150]  # Positive = inflow, Negative = outflow
    
    fig = px.bar(x=sectors, y=money_flow, 
                 title="Money Flow by Sector (₹ Cr)",
                 color=money_flow,
                 color_continuous_scale='RdYlGn')
    fig.update_layout(height=400, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def page_sector_rotation():
    """Sector Rotation page."""
    st.title("Sector Rotation Analysis")
    
    # Sector performance data
    sector_performance = get_sector_performance()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sector performance chart
        fig = px.bar(x=list(sector_performance.keys()), y=list(sector_performance.values()),
                     title="Sector Performance (%) - Last 30 Days",
                     color=list(sector_performance.values()),
                     color_continuous_scale='RdYlGn')
        fig.update_layout(height=500, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Performers")
        # Sort sectors by performance
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        for sector, perf in sorted_sectors[:5]:
            color = "positive" if perf >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{sector}</h4>
                <p class="{color}">{perf:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Bottom Performers")
        for sector, perf in sorted_sectors[-5:]:
            color = "positive" if perf >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{sector}</h4>
                <p class="{color}">{perf:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

def page_forecasting_ml():
    """ML Forecasting page."""
    st.title("Machine Learning Forecasting")
    
    st.subheader("Price Prediction Models")
    
    model_type = st.selectbox("Select Model", 
                            ["LSTM Neural Network", "Random Forest", "XGBoost", "Prophet"])
    
    symbol = st.selectbox("Select Stock", ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"])
    periods = st.slider("Forecast Period (Days)", 7, 90, 30)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training model and generating forecast..."):
            time.sleep(2)  # Simulate model training
            
            # Generate sample forecast data
            dates = pd.date_range(start=datetime.datetime.now(), periods=periods, freq='D')
            # Start from a realistic price
            start_price = 1000 if symbol == "RELIANCE" else 500
            # Generate realistic forecast with trend and noise
            trend = np.random.normal(0.001, 0.005, periods)
            forecast_prices = start_price * (1 + np.cumsum(trend))
            
            # Create forecast chart
            fig = go.Figure()
            
            # Historical data (mock)
            hist_dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=60), 
                                     end=datetime.datetime.now(), freq='D')
            hist_prices = start_price + np.cumsum(np.random.normal(0, 2, len(hist_dates)))
            
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_prices, mode='lines', name='Historical',
                line=dict(color='#1F77B4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=forecast_prices, mode='lines', name='Forecast',
                line=dict(color='#FF4B4B', width=2, dash='dash')
            ))
            
            # Confidence interval
            upper_band = forecast_prices * (1 + np.random.uniform(0.02, 0.05, periods))
            lower_band = forecast_prices * (1 - np.random.uniform(0.02, 0.05, periods))
            
            fig.add_trace(go.Scatter(
                x=dates, y=upper_band, mode='lines', line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=lower_band, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255,75,75,0.2)',
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                height=500,
                title=f"{symbol} Price Forecast - {model_type}",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            price_change = forecast_prices[-1] - hist_prices[-1]
            change_percent = (price_change / hist_prices[-1]) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{hist_prices[-1]:.2f}")
            with col2:
                st.metric("Predicted Price", f"₹{forecast_prices[-1]:.2f}", 
                         f"{price_change:+.2f} ({change_percent:+.2f}%)")
            with col3:
                confidence = np.random.uniform(0.75, 0.92)
                st.metric("Model Confidence", f"{confidence:.1%}")

def page_hft_terminal():
    """HFT Terminal page."""
    st.title("High Frequency Trading Terminal")
    
    if st.session_state.subscription_tier != "Institutional":
        st.warning("This feature is available only for Institutional subscribers.")
        st.info("Upgrade your subscription to access HFT capabilities.")
        return
    
    st.subheader("HFT Dashboard")
    
    # Market depth
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Depth - RELIANCE")
        
        # Bid side
        st.write("**Bids (Buy)**")
        for i in range(5, 0, -1):
            price = 2845 - i
            quantity = np.random.randint(100, 1000)
            st.write(f"₹{price:.2f} | {quantity} shares")
        
        # Current price
        st.markdown("---")
        st.metric("Current Price", "₹2,845.50", "+12.25 (+0.43%)")
        st.markdown("---")
        
        # Ask side
        st.write("**Asks (Sell)**")
        for i in range(1, 6):
            price = 2846 + i
            quantity = np.random.randint(100, 1000)
            st.write(f"₹{price:.2f} | {quantity} shares")
    
    with col2:
        st.subheader("HFT Controls")
        
        symbol = st.selectbox("Symbol", st.session_state.watchlist, key="hft_symbol")
        strategy = st.selectbox("Trading Strategy", 
                              ["Market Making", "Arbitrage", "Momentum", "Mean Reversion"])
        
        st.number_input("Order Size", min_value=1, value=100, key="hft_size")
        st.number_input("Max Position", min_value=100, value=1000, key="hft_max_pos")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start HFT", type="primary", use_container_width=True):
                st.success("HFT strategy activated")
        with col_b:
            if st.button("Stop HFT", type="secondary", use_container_width=True):
                st.info("HFT strategy stopped")
        
        st.markdown("---")
        st.subheader("Performance")
        st.metric("Trades Today", "1,245")
        st.metric("Success Rate", "92.3%")
        st.metric("Avg Trade Size", "₹45,250")

def page_basket_orders():
    """Basket Orders page."""
    st.title("Basket Order Management")
    
    if st.session_state.subscription_tier == "Basic":
        st.warning("Basket orders are available for Professional and Institutional subscribers.")
        st.info("Upgrade your subscription to access basket order functionality.")
        return
    
    st.subheader("Create Basket Order")
    
    # Basket configuration
    col1, col2 = st.columns(2)
    
    with col1:
        basket_name = st.text_input("Basket Name", "Technology Basket")
        strategy_type = st.selectbox("Strategy Type", 
                                   ["Equally Weighted", "Market Cap Weighted", "Custom Weights"])
    
    with col2:
        total_investment = st.number_input("Total Investment (₹)", min_value=1000, value=100000)
        execution_type = st.selectbox("Execution Type", ["MARKET", "LIMIT"])
    
    # Stock selection and weights
    st.subheader("Select Stocks and Weights")
    
    selected_stocks = st.multiselect("Select Stocks", st.session_state.watchlist, 
                                   default=st.session_state.watchlist[:3])
    
    weights = {}
    if strategy_type == "Equally Weighted":
        equal_weight = 100 / len(selected_stocks) if selected_stocks else 0
        for stock in selected_stocks:
            weights[stock] = equal_weight
    elif strategy_type == "Custom Weights":
        for stock in selected_stocks:
            weights[stock] = st.slider(f"Weight for {stock}", 0, 100, 
                                     int(100/len(selected_stocks)) if selected_stocks else 0)
    
    # Display basket summary
    if selected_stocks:
        st.subheader("Basket Summary")
        
        quotes = get_live_quotes(selected_stocks)
        basket_value = 0
        
        for stock in selected_stocks:
            if stock in quotes:
                investment = (weights[stock] / 100) * total_investment
                price = quotes[stock]['last_price']
                quantity = int(investment / price)
                actual_investment = quantity * price
                basket_value += actual_investment
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**{stock}**")
                with col2:
                    st.write(f"₹{price:.2f}")
                with col3:
                    st.write(f"{weights[stock]:.1f}%")
                with col4:
                    st.write(f"{quantity} shares (₹{actual_investment:,.0f})")
        
        st.metric("Total Basket Value", f"₹{basket_value:,.2f}")
        
        if st.button("Execute Basket Order", type="primary"):
            order_ids = []
            for stock in selected_stocks:
                if stock in quotes:
                    investment = (weights[stock] / 100) * total_investment
                    price = quotes[stock]['last_price']
                    quantity = int(investment / price)
                    order_id = place_order(stock, "BUY", quantity, execution_type, price)
                    if order_id:
                        order_ids.append(order_id)
            
            if order_ids:
                st.success(f"Basket order executed successfully! Order IDs: {', '.join(order_ids)}")

def page_fundamental_analysis():
    """Fundamental Analysis page."""
    st.title("Fundamental Analysis")
    
    symbol = st.selectbox("Select Company", st.session_state.watchlist)
    
    if symbol:
        quotes = get_live_quotes([symbol])
        if symbol in quotes:
            current_price = quotes[symbol]['last_price']
            
            # Fundamental metrics (mock data)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("P/E Ratio", "23.5", "-0.8")
                st.metric("EPS (TTM)", "₹125.60", "+₹8.45")
            
            with col2:
                st.metric("P/B Ratio", "4.2", "+0.1")
                st.metric("Book Value", "₹685.25", "+₹45.30")
            
            with col3:
                st.metric("ROE", "18.5%", "+1.2%")
                st.metric("ROCE", "22.3%", "+0.8%")
            
            with col4:
                st.metric("Debt/Equity", "0.45", "-0.05")
                st.metric("Dividend Yield", "1.2%", "+0.1%")
            
            st.markdown("---")
            
            # Financial charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue growth
                years = ['2019', '2020', '2021', '2022', '2023']
                revenue = [100, 120, 145, 165, 195]  # in 1000 Cr
                
                fig = px.line(x=years, y=revenue, title=f"{symbol} Revenue Growth (₹ Cr)",
                            markers=True)
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Profit margins
                years = ['2019', '2020', '2021', '2022', '2023']
                operating_margin = [22, 24, 26, 25, 27]
                net_margin = [18, 19, 21, 20, 22]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years, y=operating_margin, mode='lines+markers', 
                                       name='Operating Margin'))
                fig.add_trace(go.Scatter(x=years, y=net_margin, mode='lines+markers', 
                                       name='Net Margin'))
                fig.update_layout(title="Profit Margins (%)", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyst recommendations
            st.subheader("Analyst Recommendations")
            recommendations = {
                'Strong Buy': 12,
                'Buy': 8,
                'Hold': 3,
                'Sell': 1,
                'Strong Sell': 0
            }
            
            fig = px.pie(values=list(recommendations.values()), names=list(recommendations.keys()),
                        color_discrete_sequence=['#00D474', '#89F5B2', '#FFB800', '#FF7B7B', '#FF4B4B'])
            fig.update_layout(height=300, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

def page_ai_trade_assistant():
    """AI Assistant page."""
    st.title("AI Trading Assistant")
    
    st.subheader("Chat with AI Assistant")
    
    # Initialize chat history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    
    # Display chat messages from history
    for message in st.session_state.ai_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Market Analysis", use_container_width=True):
            user_message = "Give me today's market analysis"
            st.session_state.ai_chat_history.append({"role": "user", "content": user_message})
            response = "Based on current market data, NIFTY is trading at 22,150.45 (+0.57%). Key sectors showing strength are Banking and IT. Reliance and HDFC Bank are leading the gains. I recommend watching the 22,200 resistance level for NIFTY."
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("Stock Recommendation", use_container_width=True):
            user_message = "What stocks do you recommend for swing trading?"
            st.session_state.ai_chat_history.append({"role": "user", "content": user_message})
            response = "For swing trading, I recommend: 1) RELIANCE - Strong breakout above ₹2,850, target ₹2,950. 2) INFY - Oversold bounce expected, target ₹1,750. 3) ICICIBANK - Momentum building, target ₹1,100. Always use proper stop losses."
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("Options Strategy", use_container_width=True):
            user_message = "Suggest a conservative options strategy"
            st.session_state.ai_chat_history.append({"role": "user", "content": user_message})
            response = "For conservative options trading, consider a Bull Put Spread on BANKNIFTY: Sell 47,000 Put and Buy 46,500 Put. This gives you a limited risk profile with high probability of profit in a neutral to bullish market."
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col4:
        if st.button("Portfolio Review", use_container_width=True):
            user_message = "Review my current portfolio"
            st.session_state.ai_chat_history.append({"role": "user", "content": user_message})
            portfolio = get_portfolio_summary()
            response = f"Your portfolio shows a total value of ₹{portfolio['total_value']:,.2f} with P&L of ₹{portfolio['total_pnl']:+,.2f}. The portfolio is well-diversified across {len(portfolio['holdings'])} stocks. Consider rebalancing if any single stock exceeds 15% of your total portfolio value."
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me about trading, analysis, or market insights..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
        
        # Generate AI response (mock for demo)
        responses = [
            f"I understand you're asking about: {prompt}. Based on current market conditions, I would recommend careful position sizing and using stop losses.",
            f"Regarding {prompt}, my analysis suggests watching key support and resistance levels. The market is showing mixed signals currently.",
            f"For your question about {prompt}, I've analyzed the technical indicators and fundamentals. The risk-reward ratio appears favorable with proper risk management.",
            f"Based on your query about {prompt}, I recommend consulting the detailed charts in the Advanced Charting section and considering the overall market trend."
        ]
        
        response = np.random.choice(responses)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})

def page_subscription_plans():
    """Subscription Plans page."""
    st.title("Subscription Plans")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic")
        st.metric("Price", "₹999/month")
        st.write("• Live NSE/BSE Data")
        st.write("• Basic Charts & Watchlists")
        st.write("• Portfolio Tracking")
        st.write("• Email Support")
        st.write("• 10 Symbols in Watchlist")
        
        current_tier = st.session_state.subscription_tier
        if current_tier == "Basic":
            st.success("Current Plan")
        else:
            if st.button("Select Basic", key="basic", use_container_width=True):
                st.session_state.subscription_tier = "Basic"
                st.success("Subscription updated to Basic")
                st.rerun()
    
    with col2:
        st.subheader("Professional")
        st.metric("Price", "₹2,999/month")
        st.write("• All Basic features")
        st.write("• Advanced Charting & Analytics")
        st.write("• Options Chain & F&O Analytics")
        st.write("• AI Trade Signals")
        st.write("• Basket Orders")
        st.write("• Priority Support")
        st.write("• 50 Symbols in Watchlist")
        
        current_tier = st.session_state.subscription_tier
        if current_tier == "Professional":
            st.success("Current Plan")
        else:
            if st.button("Select Professional", key="pro", type="primary", use_container_width=True):
                st.session_state.subscription_tier = "Professional"
                st.success("Subscription updated to Professional")
                st.rerun()
    
    with col3:
        st.subheader("Institutional")
        st.metric("Price", "₹9,999/month")
        st.write("• All Professional features")
        st.write("• HFT Terminal")
        st.write("• Advanced Basket Orders")
        st.write("• API Access")
        st.write("• Custom Alerts & Scanners")
        st.write("• Dedicated Support Manager")
        st.write("• Unlimited Watchlist")
        
        current_tier = st.session_state.subscription_tier
        if current_tier == "Institutional":
            st.success("Current Plan")
        else:
            if st.button("Select Institutional", key="inst", use_container_width=True):
                st.session_state.subscription_tier = "Institutional"
                st.success("Subscription updated to Institutional")
                st.rerun()
    
    st.markdown("---")
    st.subheader("Current Usage")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Watchlist Symbols", f"{len(st.session_state.watchlist)}/{'∞' if st.session_state.subscription_tier == 'Institutional' else '50' if st.session_state.subscription_tier == 'Professional' else '10'}")
    with col2:
        st.metric("AI Signals Used", "24/50")
    with col3:
        st.metric("Data Refresh Rate", "Real-time")

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
    <div class="header-bar">
        <div style="display: flex; justify-content: space-around; font-weight: 500;">
    """
    
    for change in changes:
        color = "#00D474" if change["change"].startswith("+") else "#FF4B4B"
        html += f'<span>{change["market"]}: <span style="color: {color}">{change["change"]}</span></span>'
    
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

# ============ 5. AUTHENTICATION FUNCTIONS ============

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

def login_page():
    """Displays the login page for broker authentication."""
    st.title("BlockVista Terminal Pro")
    st.subheader("AI-Powered Trading Platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Broker Login")
        broker = st.selectbox("Select Your Broker", ["Zerodha", "Angel One", "Upstox", "ICICI Direct", "Demo Mode"])
        
        if broker == "Zerodha":
            if not KITE_AVAILABLE:
                st.error("KiteConnect is not available. Please install it with: pip install kiteconnect")
                st.info("Running in Demo Mode instead.")
                if st.button("Enter Demo Mode", type="primary", use_container_width=True):
                    st.session_state.broker = "Demo"
                    st.session_state.authenticated = True
                    st.session_state.profile = {'user_name': 'Demo User', 'user_id': 'demo_001'}
                    st.session_state.two_factor_setup_complete = True
                    st.success("Welcome to Demo Mode!")
                    st.rerun()
            else:
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
        st.markdown(f"**Status:** <span class='{market_status['css_class']}'>{market_status['status']}</span>", unsafe_allow_html=True)
        
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
