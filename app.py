# ================ 0. REQUIRED LIBRARIES ================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(
    page_title="BlockVista Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📈"
)

def apply_custom_styling():
    """Applies comprehensive CSS styling for professional trading terminal look."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .market-status {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .market-open {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .market-closed {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .watchlist-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .indian-flag-theme {
        background: linear-gradient(135deg, #ff9933 0%, #ffffff 50%, #138808 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ================ 2. DATA FUNCTIONS FOR INDIAN MARKETS ================

@st.cache_data(ttl=300)
def get_indian_indices_data():
    """Get real-time data for major Indian indices"""
    indices = {
        'NIFTY 50': '^NSEI',
        'BANK NIFTY': '^NSEBANK', 
        'SENSEX': '^BSESN',
        'NIFTY IT': '^CNXIT',
        'INDIA VIX': '^INDIAVIX'
    }
    
    data = []
    for name, ticker in indices.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2d')
            if len(hist) >= 2:
                current_price = hist['Close'][-1]
                prev_close = hist['Close'][-2]
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                data.append({
                    'Index': name,
                    'Price': current_price,
                    'Change': change,
                    'Change %': change_pct
                })
        except:
            continue
            
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def get_nse_top_gainers_losers():
    """Get top gainers and losers from NSE using official API"""
    try:
        # NSE API endpoint for Nifty 50 stocks
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers, timeout=10)
        data = response.json()
        
        gainers = []
        losers = []
        
        for stock in data['data']:
            change_pct = stock['pChange']
            stock_data = {
                'Symbol': stock['symbol'],
                'LTP': stock['lastPrice'],
                'Change %': change_pct
            }
            
            if change_pct > 0:
                gainers.append(stock_data)
            else:
                losers.append(stock_data)
                
        gainers_df = pd.DataFrame(gainers).nlargest(5, 'Change %')
        losers_df = pd.DataFrame(losers).nsmallest(5, 'Change %')
        
        return gainers_df, losers_df
        
    except Exception as e:
        st.error(f"Could not fetch live NSE data: {e}")
        # Fallback to static popular Indian stocks
        popular_stocks = {
            'RELIANCE': 2450.50, 'TCS': 3250.75, 'HDFCBANK': 1650.25, 
            'INFY': 1850.60, 'ICICIBANK': 950.80, 'HINDUNILVR': 2450.00
        }
        gainers = [{'Symbol': k, 'LTP': v, 'Change %': 2.5} for k, v in list(popular_stocks.items())[:3]]
        losers = [{'Symbol': k, 'LTP': v, 'Change %': -1.8} for k, v in list(popular_stocks.items())[3:]]
        return pd.DataFrame(gainers), pd.DataFrame(losers)

@st.cache_data(ttl=600)
def get_indian_stock_data(symbol, period='1y'):
    """Get historical data for Indian stocks"""
    try:
        if not symbol.endswith('.NS'):
            symbol += '.NS'
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_indian_market_holidays():
    """Get NSE market holidays for current year"""
    current_year = datetime.now().year
    holidays = [
        f"{current_year}-01-26",  # Republic Day
        f"{current_year}-03-08",  # Mahashivratri
        f"{current_year}-03-25",  # Holi
        f"{current_year}-04-11",  # Id-ul-Fitr
        f"{current_year}-04-17",  # Ram Navami
        f"{current_year}-05-01",  # Maharashtra Day
        f"{current_year}-06-17",  # Bakri Id
        f"{current_year}-07-17",  # Muharram
        f"{current_year}-08-15",  # Independence Day
        f"{current_year}-10-02",  # Mahatma Gandhi Jayanti
        f"{current_year}-11-01",  # Diwali
        f"{current_year}-11-15",  # Gurunanak Jayanti
        f"{current_year}-12-25",  # Christmas
    ]
    return holidays

def get_indian_market_status():
    """Check if Indian stock market is currently open"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Check if weekend
    if now.weekday() >= 5:
        return False, "Market Closed (Weekend)"
    
    # Check if holiday
    today_str = now.strftime('%Y-%m-%d')
    if today_str in get_indian_market_holidays():
        return False, "Market Closed (Holiday)"
    
    # Check market hours (9:15 AM to 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if market_open <= now <= market_close:
        return True, "Market Open"
    else:
        return False, "Market Closed"

@st.cache_data(ttl=3600)
def get_popular_indian_stocks():
    """Get list of popular Indian stocks for suggestions"""
    return [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
        'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK',
        'MARUTI', 'ASIANPAINT', 'HCLTECH', 'TATAMOTORS', 'SUNPHARMA', 'BHARTIARTL',
        'BAJAJFINSV', 'ADANIENT', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'POWERGRID'
    ]

# ================ 3. CHARTING FUNCTIONS ================

def create_indian_stock_chart(data, symbol, chart_type='Candlestick'):
    """Create interactive stock chart for Indian stocks"""
    if data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4')
        ))
    
    # Add Indian market specific indicators
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        mode='lines',
        name='MA20',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA50'],
        mode='lines',
        name='MA50',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{symbol} Price Chart - NSE',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_technical_indicators(data):
    """Calculate and display technical indicators for Indian stocks"""
    if data.empty:
        return pd.DataFrame()
    
    try:
        # RSI
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # MACD
        macd = ta.macd(data['Close'])
        if macd is not None:
            data = pd.concat([data, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(data['Close'], length=20)
        if bb is not None:
            data = pd.concat([data, bb], axis=1)
        
        return data.tail(1)
    except:
        return pd.DataFrame()

# ================ 4. PAGE FUNCTIONS FOR INDIAN MARKET ================

def render_indian_dashboard():
    """Main dashboard page for Indian market"""
    st.markdown('<div class="main-header">🇮🇳 BlockVista Terminal - Indian Markets</div>', unsafe_allow_html=True)
    
    # Market status
    is_open, status_msg = get_indian_market_status()
    status_class = "market-open" if is_open else "market-closed"
    st.markdown(f'<div class="market-status {status_class}">{status_msg}</div>', unsafe_allow_html=True)
    
    # Real-time Indian indices
    st.subheader("📊 Indian Market Indices")
    indices_df = get_indian_indices_data()
    
    if not indices_df.empty:
        cols = st.columns(len(indices_df))
        for idx, (_, row) in enumerate(indices_df.iterrows()):
            with cols[idx]:
                change_color = "positive" if row['Change'] > 0 else "negative"
                st.metric(
                    label=row['Index'],
                    value=f"₹{row['Price']:,.0f}",
                    delta=f"{row['Change %']:.2f}%"
                )
    
    # Top gainers and losers from NSE
    st.subheader("🔥 NSE Top Gainers & Losers")
    gainers_df, losers_df = get_nse_top_gainers_losers()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📈 Top Gainers**")
        if not gainers_df.empty:
            for _, row in gainers_df.iterrows():
                st.markdown(f"""
                <div class="watchlist-item">
                    <strong>{row['Symbol']}</strong><br>
                    ₹{row['LTP']:,.2f} <span class="positive">+{row['Change %']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No gainers data available")
    
    with col2:
        st.write("**📉 Top Losers**")
        if not losers_df.empty:
            for _, row in losers_df.iterrows():
                st.markdown(f"""
                <div class="watchlist-item">
                    <strong>{row['Symbol']}</strong><br>
                    ₹{row['LTP']:,.2f} <span class="negative">{row['Change %']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No losers data available")

def render_indian_charting():
    """Advanced charting page for Indian stocks"""
    st.header("📈 Indian Stocks Charting")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Chart Settings")
        popular_stocks = get_popular_indian_stocks()
        symbol = st.selectbox("Select Stock", popular_stocks, index=0)
        period = st.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
        chart_type = st.selectbox("Chart Type", ['Candlestick', 'Line'])
        
        # Quick analysis
        st.subheader("Quick Analysis")
        if st.button("Run Technical Analysis"):
            stock_data = get_indian_stock_data(symbol, period)
            if not stock_data.empty:
                indicators = create_technical_indicators(stock_data)
                if not indicators.empty:
                    st.write("Latest Technical Indicators:")
                    st.dataframe(indicators[['RSI', 'MA20', 'MA50']], use_container_width=True)
    
    with col2:
        stock_data = get_indian_stock_data(symbol, period)
        if not stock_data.empty:
            fig = create_indian_stock_chart(stock_data, symbol, chart_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price summary
            current_price = stock_data['Close'][-1]
            prev_close = stock_data['Close'][-2] if len(stock_data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Change", f"₹{change:.2f}")
            col3.metric("Change %", f"{change_pct:.2f}%")
        else:
            st.error(f"Could not fetch data for {symbol}")

def render_indian_watchlist():
    """Personal watchlist page for Indian stocks"""
    st.header("👀 My Indian Stocks Watchlist")
    
    # Initialize watchlist in session state
    if 'indian_watchlist' not in st.session_state:
        st.session_state.indian_watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HINDUNILVR']
    
    # Add stock to watchlist
    col1, col2 = st.columns([3, 1])
    with col1:
        popular_stocks = get_popular_indian_stocks()
        new_stock = st.selectbox("Add stock to watchlist", popular_stocks)
    with col2:
        if st.button("Add") and new_stock:
            if new_stock not in st.session_state.indian_watchlist:
                st.session_state.indian_watchlist.append(new_stock)
                st.rerun()
    
    # Display watchlist
    st.subheader("Your Indian Stocks Watchlist")
    
    if st.session_state.indian_watchlist:
        for symbol in st.session_state.indian_watchlist:
            with st.expander(f"📊 {symbol}"):
                stock_data = get_indian_stock_data(symbol, '1mo')
                if not stock_data.empty:
                    current_price = stock_data['Close'][-1]
                    prev_close = stock_data['Close'][-2] if len(stock_data) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    col1.metric("Current Price", f"₹{current_price:.2f}")
                    col2.metric("Change", f"₹{change:.2f}")
                    col3.metric("Change %", f"{change_pct:.2f}%")
                    
                    # Remove button
                    if col4.button(f"Remove", key=f"remove_{symbol}"):
                        st.session_state.indian_watchlist.remove(symbol)
                        st.rerun()
                    
                    # Mini chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        line=dict(color='green' if change > 0 else 'red')
                    ))
                    fig.update_layout(
                        height=200, 
                        showlegend=False, 
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False)
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your watchlist is empty. Add some Indian stocks to get started!")

def render_indian_screener():
    """Stock screener page for Indian market"""
    st.header("🔍 Indian Stock Screener")
    
    st.info("Screen Indian stocks based on various fundamental and technical parameters")
    
    # Screening criteria
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_cap = st.selectbox("Market Cap", [
            "Any", "Large Cap (>₹20,000 Cr)", "Mid Cap (₹5,000-20,000 Cr)", "Small Cap (<₹5,000 Cr)"
        ])
        sector = st.selectbox("Sector", [
            "Any", "Banking", "IT", "Pharma", "Auto", "FMCG", "Energy", "Metals"
        ])
    
    with col2:
        min_price = st.number_input("Min Price (₹)", min_value=0, value=100)
        max_price = st.number_input("Max Price (₹)", min_value=0, value=5000)
        volume = st.selectbox("Volume", ["Any", "High (>1L shares)", "Very High (>10L shares)"])
    
    with col3:
        pe_ratio = st.selectbox("P/E Ratio", ["Any", "Low (<15)", "Medium (15-25)", "High (>25)"])
        dividend_yield = st.selectbox("Dividend Yield", ["Any", "High (>3%)", "Very High (>5%)"])
    
    if st.button("Run Screener", type="primary"):
        # Simulate screening results
        st.success(f"Screening Indian stocks with: {market_cap}, {sector}, Price: ₹{min_price}-{max_price}")
        
        # Show sample results
        sample_stocks = [
            {'Symbol': 'RELIANCE', 'Sector': 'Energy', 'Market Cap': '₹15L Cr', 'Price': '₹2,450', 'P/E': '25.6'},
            {'Symbol': 'TCS', 'Sector': 'IT', 'Market Cap': '₹12L Cr', 'Price': '₹3,250', 'P/E': '28.9'},
            {'Symbol': 'HDFCBANK', 'Sector': 'Banking', 'Market Cap': '₹11L Cr', 'Price': '₹1,650', 'P/E': '18.2'},
        ]
        
        st.subheader("📋 Screening Results")
        st.dataframe(pd.DataFrame(sample_stocks), use_container_width=True)

def render_fno_analysis():
    """F&O analysis page for Indian derivatives"""
    st.header("📊 F&O Analytics - Indian Derivatives")
    
    tab1, tab2, tab3 = st.tabs(["NIFTY Options", "BANKNIFTY Options", "Stock Futures"])
    
    with tab1:
        st.subheader("NIFTY 50 Options Chain")
        st.info("Live NIFTY options chain analysis")
        
        # Simulated options data
        nifty_spot = 19500
        st.metric("NIFTY Spot", f"₹{nifty_spot:,}")
        
        # Options chain table
        strikes = [19300, 19400, 19500, 19600, 19700]
        options_data = []
        
        for strike in strikes:
            options_data.append({
                'Strike': strike,
                'Call OI': f"{np.random.randint(1000, 10000):,}",
                'Call LTP': f"₹{np.random.randint(50, 500):,}",
                'Put LTP': f"₹{np.random.randint(50, 500):,}",
                'Put OI': f"{np.random.randint(1000, 10000):,}"
            })
        
        st.dataframe(pd.DataFrame(options_data), use_container_width=True)
    
    with tab2:
        st.subheader("BANKNIFTY Options Chain")
        st.info("Live BANKNIFTY options chain analysis")
        
        banknifty_spot = 43500
        st.metric("BANKNIFTY Spot", f"₹{banknifty_spot:,}")
        
    with tab3:
        st.subheader("Stock Futures")
        st.info("Live stock futures analysis")
        
        futures_data = [
            {'Stock': 'RELIANCE', 'Future': '₹2,455', 'Spot': '₹2,450', 'Premium': '₹5'},
            {'Stock': 'TCS', 'Future': '₹3,255', 'Spot': '₹3,250', 'Premium': '₹5'},
            {'Stock': 'INFY', 'Future': '₹1,855', 'Spot': '₹1,850', 'Premium': '₹5'},
        ]
        
        st.dataframe(pd.DataFrame(futures_data), use_container_width=True)

def render_market_news():
    """Indian market news and updates"""
    st.header("📰 Indian Market News & Updates")
    
    # Simulated news data
    news_items = [
        {
            'title': 'RBI keeps repo rate unchanged at 6.5%',
            'source': 'Economic Times',
            'time': '2 hours ago',
            'impact': 'High'
        },
        {
            'title': 'NIFTY 50 hits new all-time high',
            'source': 'Moneycontrol',
            'time': '4 hours ago',
            'impact': 'Medium'
        },
        {
            'title': 'Reliance announces new green energy initiative',
            'source': 'Business Standard',
            'time': '6 hours ago',
            'impact': 'High'
        },
        {
            'title': 'IT stocks rally on strong earnings',
            'source': 'Financial Express',
            'time': '8 hours ago',
            'impact': 'Medium'
        }
    ]
    
    for news in news_items:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{news['title']}**")
                st.caption(f"Source: {news['source']} • {news['time']}")
            with col2:
                impact_color = "red" if news['impact'] == 'High' else "orange"
                st.markdown(f"<span style='color: {impact_color}; font-weight: bold;'>{news['impact']} Impact</span>", 
                          unsafe_allow_html=True)
            st.divider()

# ================ 5. MAIN APPLICATION ================

def main():
    """Main application function for Indian Market Terminal"""
    apply_custom_styling()
    
    # Sidebar navigation
    st.sidebar.title("🇮🇳 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Go to", [
        "Indian Dashboard", 
        "Charting", 
        "Watchlist", 
        "Stock Screener",
        "F&O Analytics",
        "Market News"
    ])
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "BlockVista Terminal - Professional Indian Stock Market Analysis Platform. "
        "Data is for educational purposes only. Focused exclusively on Indian markets."
    )
    
    # Market status in sidebar
    is_open, status_msg = get_indian_market_status()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏛️ NSE Market Status")
    st.sidebar.write(f"**Status:** {status_msg}")
    
    current_time_ist = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    st.sidebar.write(f"**IST:** {current_time_ist}")
    
    # Render selected page
    if page == "Indian Dashboard":
        render_indian_dashboard()
    elif page == "Charting":
        render_indian_charting()
    elif page == "Watchlist":
        render_indian_watchlist()
    elif page == "Stock Screener":
        render_indian_screener()
    elif page == "F&O Analytics":
        render_fno_analysis()
    elif page == "Market News":
        render_market_news()

if __name__ == "__main__":
    main()
