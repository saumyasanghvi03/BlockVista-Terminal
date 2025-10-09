# ================ 0. REQUIRED LIBRARIES (Lightweight Only) ================
# Speed Optimization: Heavy libraries are imported inside functions (lazy loading)
# to ensure a fast startup on cloud platforms like Render.
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import re
import time as a_time # Renamed to avoid conflict

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def apply_custom_styling():
    """Applies a comprehensive CSS stylesheet for professional theming."""
    theme_css = """
    <style>
        :root {
            --dark-bg: #0E1117; --dark-secondary-bg: #161B22; --dark-widget-bg: #21262D;
            --dark-border: #30363D; --dark-text: #c9d1d9; --dark-text-light: #8b949e;
            --dark-green: #28a745; --dark-red: #da3633;
            --light-bg: #FFFFFF; --light-secondary-bg: #F0F2F6; --light-widget-bg: #F8F9FA;
            --light-border: #dee2e6; --light-text: #212529; --light-text-light: #6c757d;
            --light-green: #198754; --light-red: #dc3545;
        }
        body.dark-theme {
            --primary-bg: var(--dark-bg); --secondary-bg: var(--dark-secondary-bg);
            --widget-bg: var(--dark-widget-bg); --border-color: var(--dark-border);
            --text-color: var(--dark-text); --text-light: var(--dark-text-light);
            --green: var(--dark-green); --red: var(--dark-red);
        }
        body.light-theme {
            --primary-bg: var(--light-bg); --secondary-bg: var(--light-secondary-bg);
            --widget-bg: var(--light-widget-bg); --border-color: var(--light-border);
            --text-color: var(--light-text); --text-light: var(--light-text-light);
            --green: var(--light-green); --red: var(--light-red);
        }
        body { background-color: var(--primary-bg); color: var(--text-color); }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3, h4, h5 { color: var(--text-color) !important; }
        hr { background: var(--border-color); }
        .stButton>button { border-color: var(--border-color); background-color: var(--widget-bg); color: var(--text-color); }
        .stButton>button:hover { border-color: var(--green); color: var(--green); }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div { background-color: var(--widget-bg); border-color: var(--border-color); color: var(--text-color); }
        .stRadio>div { background-color: var(--widget-bg); border: 1px solid var(--border-color); padding: 8px; border-radius: 8px; }
        .metric-card { background-color: var(--secondary-bg); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 10px; border-left-width: 5px; }
        .trade-card { background-color: var(--secondary-bg); border: 1px solid var(--border-color); padding: 1.5rem; border-radius: 10px; border-left-width: 5px; }
        .notification-bar { position: sticky; top: 0; width: 100%; background-color: var(--secondary-bg); color: var(--text-color); padding: 8px 12px; z-index: 999; display: flex; justify-content: center; align-items: center; font-size: 0.9rem; border-bottom: 1px solid var(--border-color); margin-left: -20px; margin-right: -20px; width: calc(100% + 40px); }
        .notification-bar span { margin: 0 15px; white-space: nowrap; }
        .hft-depth-bid { background: linear-gradient(to left, rgba(0, 128, 0, 0.3), rgba(0, 128, 0, 0.05)); padding: 2px 5px; }
        .hft-depth-ask { background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.05)); padding: 2px 5px; }
        .tick-up { color: var(--green); animation: flash-green 0.5s; }
        .tick-down { color: var(--red); animation: flash-red 0.5s; }
        @keyframes flash-green { 0% { background-color: rgba(40, 167, 69, 0.5); } 100% { background-color: transparent; } }
        @keyframes flash-red { 0% { background-color: rgba(218, 54, 51, 0.5); } 100% { background-color: transparent; } }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    
    js_theme = f"""
    <script>
        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add('{st.session_state.get('theme', 'Dark').lower()}-theme');
    </script>
    """
    st.components.v1.html(js_theme, height=0)

# Centralized data source configuration
ML_DATA_SOURCES = {
    "NIFTY 50": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv", "tradingsymbol": "NIFTY 50", "exchange": "NSE"},
    "BANK NIFTY": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv", "tradingsymbol": "BANKNIFTY", "exchange": "NFO"},
    "NIFTY Financial Services": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv", "tradingsymbol": "FINNIFTY", "exchange": "NFO"},
    "GOLD": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/GOLD.csv", "tradingsymbol": "GOLDM", "exchange": "MCX"},
    "USDINR": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/USDINR.csv", "tradingsymbol": "USDINR", "exchange": "CDS"},
    "SENSEX": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SENSEX.csv", "tradingsymbol": "SENSEX", "exchange": "BSE"},
    "S&P 500": {"github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/SP500.csv", "tradingsymbol": "^GSPC", "exchange": "yfinance"}
}

# ================ 1.5 INITIALIZATION ========================
def initialize_session_state():
    """Initializes all necessary session state variables."""
    defaults = {
        'broker': None, 'kite': None, 'profile': None,
        'login_animation_complete': False, 'authenticated': False,
        'two_factor_setup_complete': False, 'pyotp_secret': None,
        'theme': 'Dark', 'terminal_mode': "Cash",
        'watchlists': {
            "Watchlist 1": [{'symbol': 'RELIANCE', 'exchange': 'NSE'}, {'symbol': 'HDFCBANK', 'exchange': 'NSE'}],
            "Watchlist 2": [{'symbol': 'TCS', 'exchange': 'NSE'}, {'symbol': 'INFY', 'exchange': 'NSE'}],
            "Watchlist 3": [{'symbol': 'SENSEX', 'exchange': 'BSE'}]
        },
        'active_watchlist': "Watchlist 1",
        'order_history': [], 'basket': [], 'last_order_details': {},
        'underlying_pcr': "NIFTY", 'strategy_legs': [], 'calculated_greeks': None,
        'messages': [], 'ml_forecast_df': None, 'ml_instrument_name': None,
        'backtest_results': None, 'hft_last_price': 0, 'hft_tick_log': [],
        'algo_bots_status': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================ 2. HELPER FUNCTIONS ================

def get_broker_client():
    """Gets current broker client from session state."""
    return st.session_state.get('kite') if st.session_state.get('broker') == "Zerodha" else None

@st.dialog("Quick Trade")
def quick_trade_dialog(symbol=None, exchange=None):
    """A quick trade dialog for placing market or limit orders."""
    instrument_df = get_instrument_df()
    st.subheader(f"Place Order for {symbol}" if symbol else "Quick Order")
    
    if symbol is None:
        symbol = st.text_input("Symbol").upper()
    
    transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True, key="diag_trans_type")
    product = st.radio("Product", ["MIS", "CNC"], horizontal=True, key="diag_prod_type")
    order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True, key="diag_order_type")
    quantity = st.number_input("Quantity", min_value=1, step=1, key="diag_qty")
    price = st.number_input("Price", min_value=0.01, key="diag_price") if order_type == "LIMIT" else 0

    if st.button("Submit Order", use_container_width=True):
        if symbol and quantity > 0:
            place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price if price > 0 else None)
            st.rerun()
        else:
            st.warning("Please fill in all fields.")

@st.cache_data(ttl=3600)
def get_market_holidays(year):
    """NSE holidays (update yearly)."""
    holidays_by_year = {
        2024: ['2024-01-22', '2024-01-26', '2024-03-08', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-05-01', '2024-05-20', '2024-06-17', '2024-07-17', '2024-08-15', '2024-10-02', '2024-11-01', '2024-11-15', '2024-12-25'],
        2025: ['2025-01-26', '2025-03-06', '2025-03-21', '2025-04-14', '2025-04-18', '2025-05-01', '2025-08-15', '2025-10-02', '2025-10-21', '2025-11-05', '2025-12-25'],
        2026: ['2026-01-26', '2026-02-24', '2026-04-03', '2026-04-14', '2026-05-01', '2026-08-15', '2026-10-02', '2026-11-09', '2026-11-24', '2026-12-25']
    }
    return holidays_by_year.get(year, [])

def get_market_status():
    """Checks if the Indian stock market is open."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    holidays = get_market_holidays(now.year)
    market_open_time, market_close_time = time(9, 15), time(15, 30)
    
    if now.weekday() >= 5 or now.strftime('%Y-%m-%d') in holidays:
        return {"status": "CLOSED", "color": "#FF4B4B"}
    if market_open_time <= now.time() <= market_close_time:
        return {"status": "OPEN", "color": "#28a745"}
    return {"status": "CLOSED", "color": "#FF4B4B"}

def display_header():
    """Displays the main header with market status, a live clock, and trade buttons."""
    status_info = get_market_status()
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime("%H:%M:%S IST")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown('<h1 style="margin: 0; line-height: 1.2;">BlockVista Terminal</h1>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="text-align: right;">
                <h5 style="margin: 0;">{current_time}</h5>
                <h5 style="margin: 0;">Market: <span style='color:{status_info["color"]}; font-weight: bold;'>{status_info["status"]}</span></h5>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        b_col1, b_col2 = st.columns(2)
        if b_col1.button("Buy", use_container_width=True, key="header_buy"):
            quick_trade_dialog()
        if b_col2.button("Sell", use_container_width=True, key="header_sell"):
            quick_trade_dialog()

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)
    
def display_overnight_changes_bar():
    """Displays a notification bar with overnight market changes."""
    overnight_tickers = {"GIFT NIFTY": "IN=F", "S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F"}
    data = get_global_indices_data(overnight_tickers)
    
    if not data.empty:
        bar_html = "<div class='notification-bar'>"
        for _, row in data.iterrows():
            color = 'var(--green)' if row['% Change'] > 0 else 'var(--red)'
            bar_html += f"<span>{row['Ticker']}: {row['Price']:,.2f} <span style='color:{color};'>({row['% Change']:+.2f}%)</span></span>"
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

# ================ 3. CORE DATA & CHARTING FUNCTIONS ================

def create_chart(df, ticker, chart_type='Candlestick', forecast_df=None, conf_int_df=None):
    """Generates a Plotly chart with various chart types and overlays."""
    import plotly.graph_objects as go
    import pandas as pd
    import pandas_ta as ta

    fig = go.Figure()
    if df.empty: return fig
    chart_df = df.copy()
    
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.droplevel(0)
        
    chart_df.columns = [str(col).lower() for col in chart_df.columns]
    
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in chart_df.columns for col in required_cols):
        st.error(f"Charting error for {ticker}: Dataframe is missing required columns (open, high, low, close).")
        return go.Figure()

    if chart_type == 'Heikin-Ashi':
        ha_df = ta.ha(chart_df['open'], chart_df['high'], chart_df['low'], chart_df['close'])
        fig.add_trace(go.Candlestick(x=ha_df.index, open=ha_df['HA_open'], high=ha_df['HA_high'], low=ha_df['HA_low'], close=ha_df['HA_close'], name='Heikin-Ashi'))
    elif chart_type == 'Line':
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['close'], mode='lines', name='Line'))
    elif chart_type == 'Bar':
        fig.add_trace(go.Ohlc(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Bar'))
    else: # Candlestick
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'], name='Candlestick'))
        
    bbl_col = next((col for col in chart_df.columns if 'bbl' in col), None)
    bbu_col = next((col for col in chart_df.columns if 'bbu' in col), None)
    if bbl_col and bbu_col:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbl_col], line=dict(color='rgba(135,206,250,0.5)', width=1), name='Lower Band'))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[bbu_col], line=dict(color='rgba(135,206,250,0.5)', width=1), fill='tonexty', fillcolor='rgba(135,206,250,0.1)', name='Upper Band'))
        
    if forecast_df is not None:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], mode='lines', line=dict(color='yellow', dash='dash'), name='Forecast'))
        if conf_int_df is not None:
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['lower'], line=dict(color='rgba(255,255,0,0.2)', width=1), name='Lower CI', showlegend=False))
            fig.add_trace(go.Scatter(x=conf_int_df.index, y=conf_int_df['upper'], line=dict(color='rgba(255,255,0,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'))
        
    template = st.session_state.get('theme', 'Dark').lower()
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=f"plotly_{template}", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

@st.cache_resource(ttl=3600)
def get_instrument_df():
    """Fetches the full list of tradable instruments from the broker."""
    import pandas as pd
    client = get_broker_client()
    if not client: return pd.DataFrame()
    
    if st.session_state.broker == "Zerodha":
        df = pd.DataFrame(client.instruments())
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
        return df
    else:
        st.warning(f"Instrument list for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

def get_instrument_token(symbol, instrument_df, exchange='NSE'):
    """Finds the instrument token for a given symbol and exchange."""
    if instrument_df.empty: return None
    match = instrument_df[(instrument_df['tradingsymbol'] == symbol.upper()) & (instrument_df['exchange'] == exchange)]
    return match.iloc[0]['instrument_token'] if not match.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """Fetches historical data from the broker's API."""
    import pandas as pd
    import pandas_ta as ta
    from kiteconnect import exceptions as kite_exceptions

    client = get_broker_client()
    if not client or not instrument_token: return pd.DataFrame()

    if st.session_state.broker == "Zerodha":
        if not to_date: to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))
        
        if from_date > to_date:
            from_date = to_date - timedelta(days=1)
            
        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty: return df
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Add technical indicators silently
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True); df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cmf(append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True); df.ta.macd(append=True); df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True)
                if 'minute' in interval:
                    df.ta.vwap(append=True)
            except Exception:
                pass 
            return df
        except kite_exceptions.KiteException as e:
            st.error(f"Kite API Error (Historical): {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"An unexpected error occurred fetching historical data: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Historical data for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_watchlist_data(symbols_with_exchange):
    """Fetches live prices and market data for a list of symbols."""
    import pandas as pd
    client = get_broker_client()
    if not client or not symbols_with_exchange: return pd.DataFrame()

    if st.session_state.broker == "Zerodha":
        instrument_names = [f"{item['exchange']}:{item['symbol']}" for item in symbols_with_exchange]
        try:
            quotes = client.quote(instrument_names)
            watchlist = []
            for item in symbols_with_exchange:
                instrument = f"{item['exchange']}:{item['symbol']}"
                if instrument in quotes and quotes[instrument]:
                    quote = quotes[instrument]
                    last_price = quote['last_price']
                    prev_close = quote['ohlc']['close']
                    change = last_price - prev_close
                    pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                    watchlist.append({'Ticker': item['symbol'], 'Exchange': item['exchange'], 'Price': last_price, 'Change': change, '% Change': pct_change})
            return pd.DataFrame(watchlist)
        except Exception as e:
            st.toast(f"Error fetching watchlist data: {e}", icon="⚠️")
            return pd.DataFrame()
    else:
        st.warning(f"Watchlist for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

@st.cache_data(ttl=2)
def get_market_depth(instrument_token):
    """Fetches market depth (order book) for a given instrument."""
    client = get_broker_client()
    if not client or not instrument_token: return None
    try:
        depth = client.depth(instrument_token)
        return depth.get(str(instrument_token))
    except Exception as e:
        st.toast(f"Error fetching market depth: {e}", icon="⚠️")
        return None

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """Fetches and processes the options chain for a given underlying."""
    import pandas as pd
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame(), None, 0.0, []
    
    if st.session_state.broker == "Zerodha":
        exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "NATURALGAS": "MCX", "USDINR": "CDS"}
        exchange = exchange_map.get(underlying, 'NFO')
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}.get(underlying, underlying)
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"
        try:
            underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
        except Exception:
            underlying_ltp = 0.0
        
        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty: return pd.DataFrame(), None, underlying_ltp, []
        
        expiries = sorted(options['expiry'].dt.date.unique())
        three_months_later = datetime.now().date() + timedelta(days=90)
        available_expiries = [e for e in expiries if datetime.now().date() <= e <= three_months_later]
        if not available_expiries: return pd.DataFrame(), None, underlying_ltp, []
        
        if not expiry_date: expiry_date = available_expiries[0]
        else: expiry_date = pd.to_datetime(expiry_date).date()
        
        chain_df = options[options['expiry'].dt.date == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch: return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
        
        try:
            quotes = client.quote(instruments_to_fetch)
            ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            ce_df['oi'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            pe_df['oi'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            
            final_chain = pd.merge(ce_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                   pe_df[['tradingsymbol', 'strike', 'LTP', 'oi']], 
                                   on='strike', suffixes=('_CE', '_PE')).rename(columns={'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE', 'oi_CE': 'CALL OI', 'oi_PE': 'PUT OI', 'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'}).fillna(0)
            
            return final_chain[['CALL', 'CALL LTP', 'CALL OI', 'STRIKE', 'PUT LTP', 'PUT OI', 'PUT']], expiry_date, underlying_ltp, available_expiries
        except Exception as e:
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

@st.cache_data(ttl=10)
def get_portfolio():
    """Fetches real-time portfolio positions and holdings from the broker."""
    import pandas as pd
    client = get_broker_client()
    if not client: return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    
    if st.session_state.broker == "Zerodha":
        try:
            positions = client.positions().get('net', [])
            holdings = client.holdings()
            positions_df = pd.DataFrame(positions)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if positions else pd.DataFrame()
            total_pnl = positions_df['pnl'].sum() if not positions_df.empty else 0.0
            holdings_df = pd.DataFrame(holdings)[['tradingsymbol', 'quantity', 'average_price', 'last_price', 'pnl']] if holdings else pd.DataFrame()
            total_investment = (holdings_df['quantity'] * holdings_df['average_price']).sum() if not holdings_df.empty else 0.0
            return positions_df, holdings_df, total_pnl, total_investment
        except Exception as e:
            st.error(f"Kite API Error (Portfolio): {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0
    else:
        st.warning(f"Portfolio for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

def place_order(instrument_df, symbol, quantity, order_type, transaction_type, product, price=None):
    """Places a single order through the broker's API."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
        
    if st.session_state.broker == "Zerodha":
        try:
            # A simple way to check if it's an F&O symbol
            is_option_or_future = any(char.isdigit() for char in symbol) or "FUT" in symbol
            if is_option_or_future:
                exchange = 'NFO' # Assume NFO for F&O, can be refined
            else:
                instrument = instrument_df[instrument_df['tradingsymbol'] == symbol.upper()]
                if instrument.empty:
                    st.error(f"Symbol '{symbol}' not found.")
                    return
                exchange = instrument.iloc[0]['exchange']

            order_id = client.place_order(tradingsymbol=symbol.upper(), exchange=exchange, transaction_type=transaction_type, quantity=quantity, order_type=order_type, product=product, variety=client.VARIETY_REGULAR, price=price)
            st.toast(f"✅ Order placed successfully! ID: {order_id}", icon="🎉")
            st.session_state.order_history.insert(0, {"id": order_id, "symbol": symbol, "qty": quantity, "type": transaction_type, "status": "Success"})
        except Exception as e:
            st.toast(f"❌ Order failed: {e}", icon="🔥")
            st.session_state.order_history.insert(0, {"id": "N/A", "symbol": symbol, "qty": quantity, "type": transaction_type, "status": f"Failed: {e}"})
    else:
        st.warning(f"Order placement for {st.session_state.broker} not implemented.")

@st.cache_data(ttl=900)
def fetch_and_analyze_news(query=None):
    """Fetches and performs sentiment analysis on financial news."""
    import pandas as pd
    import feedparser
    from email.utils import mktime_tz
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    
    news_sources = {
        "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
        "Business Standard": "https://www.business-standard.com/rss/markets-102.cms",
        "Livemint": "https://www.livemint.com/rss/markets",
        "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
    }
    all_news = []
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                published_tuple = entry.published_parsed if hasattr(entry, 'published_parsed') else entry.updated_parsed
                published_date = datetime.fromtimestamp(mktime_tz(published_tuple)) if published_tuple else datetime.now()
                if query is None or query.lower() in entry.title.lower() or (hasattr(entry, 'summary') and query.lower() in entry.summary.lower()):
                    all_news.append({"source": source, "title": entry.title, "link": entry.link, "date": published_date.date(), "sentiment": analyzer.polarity_scores(entry.title)['compound']})
        except Exception:
            continue
    return pd.DataFrame(all_news)

def mean_absolute_percentage_error(y_true, y_pred):
    """Custom MAPE function."""
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data(show_spinner=False)
def train_seasonal_arima_model(_data, forecast_steps=30):
    """Trains a Seasonal ARIMA model for time series forecasting."""
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA

    if _data.empty or len(_data) < 100: return None, None, None

    df = _data.copy()
    df.index = pd.to_datetime(df.index)
    
    try:
        decomposed = seasonal_decompose(df['close'], model='additive', period=7)
        seasonally_adjusted = df['close'] - decomposed.seasonal
        model = ARIMA(seasonally_adjusted, order=(5, 1, 0)).fit()
        
        fitted_values = model.fittedvalues + decomposed.seasonal
        backtest_df = pd.DataFrame({'Actual': df['close'], 'Predicted': fitted_values}).dropna()
        
        forecast_result = model.get_forecast(steps=forecast_steps)
        forecast_adjusted = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        last_season_cycle = decomposed.seasonal.iloc[-7:]
        future_seasonal_values = np.tile(last_season_cycle.values, forecast_steps // 7 + 1)[:forecast_steps]
        future_forecast = forecast_adjusted + future_seasonal_values
        
        future_dates = pd.to_datetime(pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_steps))
        forecast_df = pd.DataFrame({'Predicted': future_forecast.values}, index=future_dates)
        
        conf_int_df = pd.DataFrame({
            'lower': conf_int.iloc[:, 0] + future_seasonal_values,
            'upper': conf_int.iloc[:, 1] + future_seasonal_values
        }, index=future_dates)
        
        return forecast_df, backtest_df, conf_int_df
    except Exception as e:
        st.error(f"Seasonal ARIMA model training failed: {e}")
        return None, None, None

@st.cache_data
def load_and_combine_data(instrument_name):
    """Loads and combines historical data from a static CSV with live data from the broker."""
    import pandas as pd
    import requests
    import io
    import yfinance as yf

    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()
    try:
        response = requests.get(source_info['github_url'])
        response.raise_for_status()
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True).dt.tz_localize(None)
        hist_df.set_index('Date', inplace=True)
        hist_df.columns = [col.lower() for col in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return pd.DataFrame()
        
    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol') and source_info.get('exchange') != 'yfinance':
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            from_date = hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date)
            if not live_df.empty: 
                live_df.index = live_df.index.tz_convert(None) # Make timezone-naive
                live_df.columns = [col.lower() for col in live_df.columns]
    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['tradingsymbol'], period="max", progress=False)
            if not live_df.empty: 
                live_df.index = live_df.index.tz_localize(None) # Make timezone-naive
                live_df.columns = [col.lower() for col in live_df.columns]
        except Exception as e:
            st.error(f"Failed to load yfinance data: {e}")
            live_df = pd.DataFrame()
            
    if not live_df.empty:
        hist_df.index = hist_df.index.tz_localize(None) if hist_df.index.tz is not None else hist_df.index
        live_df.index = live_df.index.tz_localize(None) if live_df.index.tz is not None else live_df.index
        
        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        hist_df.sort_index(inplace=True)
        return hist_df

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculates Black-Scholes option price and Greeks."""
    import numpy as np
    from scipy.stats import norm
    if sigma <= 0 or T <= 0: return {key: 0 for key in ["price", "delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2); delta = norm.cdf(d1); theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)); rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1); delta = norm.cdf(d1) - 1; theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)); rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)); vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega / 100, "theta": theta / 365, "rho": rho / 100}

def implied_volatility(S, K, T, r, market_price, option_type):
    """Calculates implied volatility using the Newton-Raphson method."""
    import numpy as np
    from scipy.optimize import newton
    if T <= 0 or market_price <= 0: return np.nan
    equation = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)['price'] - market_price
    try:
        return newton(equation, 0.5, tol=1e-5, maxiter=100)
    except (RuntimeError, TypeError):
        return np.nan

def interpret_indicators(df):
    """Interprets the latest values of various technical indicators."""
    if df.empty: return {}
    latest = df.iloc[-1].copy()
    latest.index = latest.index.str.lower()
    interpretation = {}
    
    rsi_col = next((col for col in latest.index if 'rsi' in col), None)
    stoch_k_col = next((col for col in latest.index if 'stok' in col), None)
    macd_col = next((col for col in latest.index if 'macd' in col and 'macds' not in col and 'macdh' not in col), None)
    signal_col = next((col for col in latest.index if 'macds' in col), None)
    adx_col = next((col for col in latest.index if 'adx' in col), None)

    r = latest.get(rsi_col)
    if r is not None: interpretation['RSI (14)'] = "Overbought (Bearish)" if r > 70 else "Oversold (Bullish)" if r < 30 else "Neutral"
    
    stoch_k = latest.get(stoch_k_col)
    if stoch_k is not None: interpretation['Stochastic'] = "Overbought (Bearish)" if stoch_k > 80 else "Oversold (Bullish)" if stoch_k < 20 else "Neutral"
    
    macd, signal = latest.get(macd_col), latest.get(signal_col)
    if macd is not None and signal is not None: interpretation['MACD'] = "Bullish Crossover" if macd > signal else "Bearish Crossover"
    
    adx = latest.get(adx_col)
    if adx is not None: interpretation['ADX (14)'] = f"Strong Trend ({adx:.1f})" if adx > 25 else f"Weak/No Trend ({adx:.1f})"
    
    return interpretation

# ================ 4. HNI & PRO TRADER FEATURES ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        orders_to_place = []
        for item in basket_items:
            instrument = instrument_df[instrument_df['tradingsymbol'] == item['symbol']]
            if instrument.empty:
                st.toast(f"❌ Could not find symbol {item['symbol']}. Skipping.", icon="🔥")
                continue
            exchange = instrument.iloc[0]['exchange']

            order = {
                "tradingsymbol": item['symbol'], "exchange": exchange,
                "transaction_type": client.TRANSACTION_TYPE_BUY if item['transaction_type'] == 'BUY' else client.TRANSACTION_TYPE_SELL,
                "quantity": int(item['quantity']), "product": client.PRODUCT_MIS if item['product'] == 'MIS' else client.PRODUCT_CNC,
                "order_type": client.ORDER_TYPE_MARKET if item['order_type'] == 'MARKET' else client.ORDER_TYPE_LIMIT,
            }
            if order['order_type'] == client.ORDER_TYPE_LIMIT:
                order['price'] = item['price']
            orders_to_place.append(order)
        
        if not orders_to_place:
            st.warning("No valid orders to place in the basket.")
            return

        try:
            client.place_order(variety=client.VARIETY_REGULAR, orders=orders_to_place)
            st.toast("✅ Basket order placed successfully!", icon="🎉")
            st.session_state.basket = []
            st.rerun()
        except Exception as e:
            st.toast(f"❌ Basket order failed: {e}", icon="🔥")

@st.cache_data(ttl=3600)
def get_sector_data():
    """
    Loads stock-to-sector mapping from a CSV file.
    RENDER DEPLOYMENT NOTE: This file ('sensex_sectors.csv') MUST be in your GitHub repository to be accessible.
    """
    import pandas as pd
    try:
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("'sensex_sectors.csv' not found in the repository. Sector allocation will be unavailable.")
        return None

def style_option_chain(df, ltp):
    """Applies conditional styling to highlight ITM/OTM in the options chain."""
    if df.empty or 'STRIKE' not in df.columns or ltp == 0:
        return df.style

    def highlight_itm(row):
        styles = [''] * len(row)
        if row['STRIKE'] < ltp:
            styles[df.columns.get_loc('CALL LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('CALL OI')] = 'background-color: #2E4053'
        if row['STRIKE'] > ltp:
            styles[df.columns.get_loc('PUT LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('PUT OI')] = 'background-color: #2E4053'
        return styles

    return df.style.apply(highlight_itm, axis=1)

@st.dialog("Most Active Options")
def show_most_active_dialog(underlying, instrument_df):
    """Dialog to display the most active options by volume."""
    st.subheader(f"Most Active {underlying} Options (By Volume)")
    with st.spinner("Fetching data..."):
        active_df = get_most_active_options(underlying, instrument_df)
        if not active_df.empty:
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not retrieve data for most active options.")

def get_most_active_options(underlying, instrument_df):
    """Fetches the most active options by volume for a given underlying."""
    import pandas as pd
    client = get_broker_client()
    if not client:
        st.toast("Broker not connected.", icon="⚠️")
        return pd.DataFrame()
    
    try:
        chain_df, expiry, _, _ = get_options_chain(underlying, instrument_df)
        if chain_df.empty or expiry is None: return pd.DataFrame()
        
        ce_symbols = chain_df['CALL'].dropna().tolist()
        pe_symbols = chain_df['PUT'].dropna().tolist()
        all_symbols = [f"NFO:{s}" for s in ce_symbols + pe_symbols]
        if not all_symbols: return pd.DataFrame()

        quotes = client.quote(all_symbols)
        active_options = []
        for symbol, data in quotes.items():
            prev_close = data.get('ohlc', {}).get('close', 0)
            last_price = data.get('last_price', 0)
            change = last_price - prev_close
            pct_change = (change / prev_close * 100) if prev_close != 0 else 0
            
            active_options.append({
                'Symbol': data.get('tradingsymbol'), 'LTP': last_price,
                'Change %': pct_change, 'Volume': data.get('volume', 0),
                'OI': data.get('oi', 0)
            })
        
        df = pd.DataFrame(active_options)
        return df.sort_values(by='Volume', ascending=False).head(10)

    except Exception as e:
        st.error(f"Could not fetch most active options: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    import pandas as pd
    import numpy as np
    import yfinance as yf

    if not tickers: return pd.DataFrame()
    
    try:
        data_yf = yf.download(list(tickers.values()), period="5d", progress=False)
        if data_yf.empty: return pd.DataFrame()

        data = []
        for ticker_name, yf_ticker_name in tickers.items():
            hist = data_yf.loc[:, (slice(None), yf_ticker_name)] if len(tickers) > 1 else data_yf
            if len(tickers) > 1: hist.columns = hist.columns.droplevel(1)

            if len(hist) >= 2:
                last_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = last_price - prev_close
                pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                data.append({'Ticker': ticker_name, 'Price': last_price, 'Change': change, '% Change': pct_change})
            else:
                data.append({'Ticker': ticker_name, 'Price': np.nan, 'Change': np.nan, '% Change': np.nan})

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()

# ================ 5. PAGE DEFINITIONS (INCLUDES ALGO BOTS) ================

# --- Algo Bot Signal Helpers ---
def rsi_signal(df, period=14, overbought=70, oversold=30):
    df.ta.rsi(length=period, append=True)
    rsi_col = f'RSI_{period}'
    if rsi_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last, prev = df[rsi_col].iloc[-2:]
    if prev < oversold and last >= oversold: return 'BUY', f"RSI crossed above {oversold}."
    if prev > overbought and last <= overbought: return 'SELL', f"RSI crossed below {overbought}."
    return 'HOLD', f"RSI is neutral at {last:.2f}."

def macd_signal(df, fast=12, slow=26, signal=9):
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    macd_col, signal_col = f'MACD_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}'
    if macd_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_macd, prev_macd = df[macd_col].iloc[-1], df[macd_col].iloc[-2]
    last_signal, prev_signal = df[signal_col].iloc[-1], df[signal_col].iloc[-2]
    if prev_macd < prev_signal and last_macd >= last_signal: return 'BUY', "MACD crossed above Signal."
    if prev_macd > prev_signal and last_macd <= last_signal: return 'SELL', "MACD crossed below Signal."
    return 'HOLD', "No MACD crossover."

def supertrend_signal(df, period=7, multiplier=3):
    import pandas_ta as ta
    st_result = df.ta.supertrend(length=period, multiplier=multiplier)
    st_col = st_result.columns[0]
    if st_col not in st_result.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
    last_st, prev_st = st_result[st_col].iloc[-1], st_result[st_col].iloc[-2]
    if prev_close <= prev_st and last_close > last_st: return 'BUY', "Price crossed above Supertrend."
    if prev_close >= prev_st and last_close < last_st: return 'SELL', "Price crossed below Supertrend."
    return 'HOLD', f"Price {'above' if last_close > last_st else 'below'} Supertrend."

def ema_crossover_signal(df, fast_period=50, slow_period=200):
    df.ta.ema(length=fast_period, append=True); df.ta.ema(length=slow_period, append=True)
    fast_col, slow_col = f'EMA_{fast_period}', f'EMA_{slow_period}'
    if fast_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_fast, prev_fast = df[fast_col].iloc[-1], df[fast_col].iloc[-2]
    last_slow, prev_slow = df[slow_col].iloc[-1], df[slow_col].iloc[-2]
    if prev_fast < prev_slow and last_fast >= last_slow: return 'BUY', "Golden Cross."
    if prev_fast > prev_slow and last_fast <= last_slow: return 'SELL', "Death Cross."
    return 'HOLD', "No EMA crossover."

def bollinger_band_signal(df, period=20, std=2):
    df.ta.bbands(length=period, std=std, append=True)
    lower_col, upper_col = f'BBL_{period}_{float(std)}', f'BBU_{period}_{float(std)}'
    if lower_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
    last_lower, last_upper = df[lower_col].iloc[-1], df[upper_col].iloc[-1]
    if prev_close > last_lower and last_close <= last_lower: return 'BUY', "Crossed below lower band."
    if prev_close < last_upper and last_close >= last_upper: return 'SELL', "Crossed above upper band."
    return 'HOLD', "Price within bands."

def vwap_scalper_signal(df):
    df.ta.vwap(append=True)
    vwap_cols = df.columns[df.columns.str.contains('VWAP')]
    if vwap_cols.empty or len(df) < 2: return 'HOLD', "Insufficient data."
    vwap_col = vwap_cols[0]
    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
    last_vwap = df[vwap_col].iloc[-1]
    if prev_close < last_vwap and last_close >= last_vwap: return 'BUY', "Crossed above VWAP."
    if prev_close > last_vwap and last_close <= last_vwap: return 'SELL', "Crossed below VWAP."
    return 'HOLD', "No VWAP crossover."

# --- Page Definitions (including Algo Bots) ---
# ... (All page functions from the previous version are assumed to be here) ...

def page_algo_bots():
    """A dedicated page for semi-automated and fully-automated trading bots."""
    display_header()
    st.title("Algo Trading Bots")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty or not get_broker_client():
        st.info("Please connect to a broker to run the Algo Bots."); return

    BOTS = {
        "RSI Momentum Bot": {"desc": "Trades on momentum shifts (RSI).", "func": rsi_signal, "params": {"period": 14, "overbought": 70, "oversold": 30}, "interval": "day"},
        "MACD Crossover Bot": {"desc": "Trend-following strategy on MACD crossovers.", "func": macd_signal, "params": {"fast": 12, "slow": 26, "signal": 9}, "interval": "day"},
        "Supertrend Trend Rider": {"desc": "Follows the trend until a reversal is signaled.", "func": supertrend_signal, "params": {"period": 7, "multiplier": 3.0}, "interval": "day"},
        "EMA Crossover Bot": {"desc": "Trades on 'Golden/Death Cross' signals.", "func": ema_crossover_signal, "params": {"fast_period": 50, "slow_period": 200}, "interval": "day"},
        "Bollinger Band Reversion": {"desc": "Mean-reversion strategy buying at lower and selling at upper bands.", "func": bollinger_band_signal, "params": {"period": 20, "std": 2.0}, "interval": "day"},
        "VWAP Intraday Scalper": {"desc": "Intraday strategy trading on VWAP crossovers.", "func": vwap_scalper_signal, "params": {}, "interval": "minute"}
    }
    
    bot_tabs = st.tabs(list(BOTS.keys()))

    for i, (bot_name, bot_info) in enumerate(BOTS.items()):
        with bot_tabs[i]:
            st.subheader(bot_name); st.caption(bot_info['desc'])
            
            if bot_name not in st.session_state.algo_bots_status:
                st.session_state.algo_bots_status[bot_name] = {"status": "Stopped", "log": [], "position": "FLAT"}
            bot_state = st.session_state.algo_bots_status[bot_name]

            with st.form(key=f"form_{bot_name}"):
                cols_config = st.columns(3)
                symbol = cols_config[0].text_input("Symbol", "RELIANCE", key=f"symbol_{bot_name}").upper()
                quantity = cols_config[1].number_input("Quantity", 1, min_value=1, key=f"qty_{bot_name}")
                mode = cols_config[2].radio("Mode", ["Semi-Automated", "Fully-Automated"], horizontal=True, label_visibility="collapsed")
                
                params = {}
                if bot_info['params']:
                    cols_params = st.columns(len(bot_info['params']))
                    for j, (p_name, p_val) in enumerate(bot_info['params'].items()):
                        params[p_name] = cols_params[j].number_input(p_name.replace("_", " ").title(), value=p_val, key=f"param_{p_name}_{bot_name}")

                is_running = bot_state["status"] == "Running"
                submit_text = "Get Live Signal" if mode == "Semi-Automated" else ("Stop Bot" if is_running else "Start Bot")
                submitted = st.form_submit_button(submit_text, use_container_width=True, type="primary" if not is_running else "secondary")

                if submitted:
                    if mode == "Semi-Automated":
                        with st.spinner(f"Fetching signal for {symbol}..."):
                            token = get_instrument_token(symbol, instrument_df)
                            if token:
                                data = get_historical_data(token, bot_info['interval'], period='1y' if bot_info['interval'] == 'day' else '5d')
                                if not data.empty:
                                    signal, reason = bot_info['func'](data.copy(), **params)
                                    bot_state['last_signal'] = {"signal": signal, "reason": reason, "symbol": symbol, "quantity": quantity}
                                else: st.error("Could not fetch data.")
                            else: st.error(f"Symbol {symbol} not found.")
                    else: # Fully-Automated
                        if is_running:
                            bot_state["status"] = "Stopped"; st.warning(f"{bot_name} stopped."); st.rerun()
                        else:
                            bot_state.update({
                                "status": "Running",
                                "config": {"symbol": symbol, "quantity": quantity, "params": params, "interval": bot_info['interval']},
                                "func": bot_info['func']
                            })
                            st.success(f"{bot_name} started. It will now trade automatically."); st.rerun()

            st.markdown("---")
            if mode == "Semi-Automated" and "last_signal" in bot_state:
                st.subheader("Live Signal")
                signal_info = bot_state['last_signal']
                if signal_info['signal'] == 'BUY': st.success(f"**BUY Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                elif signal_info['signal'] == 'SELL': st.error(f"**SELL Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                else: st.info(f"**HOLD Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                
                if signal_info['signal'] in ['BUY', 'SELL'] and st.button(f"Execute {signal_info['signal']} Order", key=f"exec_{bot_name}"):
                    place_order(instrument_df, signal_info['symbol'], signal_info['quantity'], "MARKET", signal_info['signal'], "MIS")
            
            elif mode == "Fully-Automated":
                st.subheader("Bot Status & Log")
                status_color = "green" if bot_state['status'] == 'Running' else 'orange'
                st.markdown(f"**Status:** <span style='color:{status_color};'>{bot_state['status']}</span>", unsafe_allow_html=True)
                if is_running: st.code(f"Watching: {bot_state['config']['symbol']} | Qty: {bot_state['config']['quantity']} | Position: {bot_state['position']}")
                
                log_container = st.container(height=200)
                if bot_state['log']:
                    for log_entry in reversed(bot_state['log']): log_container.code(log_entry)
                else: log_container.info("No trades executed yet.")

    # --- Fully-Automated Runner Logic (runs on every app refresh) ---
    for bot_name, bot_state in st.session_state.algo_bots_status.items():
        if bot_state.get("status") == "Running":
            config = bot_state['config']
            token = get_instrument_token(config['symbol'], instrument_df)
            if not token: continue
            data = get_historical_data(token, config['interval'], period='1y' if config['interval'] == 'day' else '5d')
            
            if not data.empty:
                signal, reason = bot_state['func'](data.copy(), **config['params'])
                
                if signal == 'BUY' and bot_state.get('position') == 'FLAT':
                    place_order(instrument_df, config['symbol'], config['quantity'], 'MARKET', 'BUY', 'MIS')
                    log_msg = f"[{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')}] EXECUTED BUY: {reason}"
                    bot_state['log'].append(log_msg); bot_state['position'] = 'LONG'
                    st.toast(f"🤖 {bot_name} executed BUY!", icon="✅")
                
                elif signal == 'SELL' and bot_state.get('position') == 'LONG':
                    place_order(instrument_df, config['symbol'], config['quantity'], 'MARKET', 'SELL', 'MIS')
                    log_msg = f"[{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')}] EXECUTED SELL (Exit): {reason}"
                    bot_state['log'].append(log_msg); bot_state['position'] = 'FLAT'
                    st.toast(f"🤖 {bot_name} EXITED position!", icon="✅")

# (The rest of the page functions: page_dashboard, page_advanced_charting, etc., would be here)

# ============ 6. MAIN APP LOGIC AND AUTHENTICATION ============

def get_user_secret(user_profile):
    """Generate a persistent secret based on user profile."""
    import hashlib, base64
    user_id = user_profile.get('user_id', 'default_user')
    user_hash = hashlib.sha256(str(user_id).encode()).digest()
    return base64.b32encode(user_hash).decode('utf-8').replace('=', '')[:16]

@st.dialog("Two-Factor Authentication")
def two_factor_dialog():
    """Dialog for 2FA login."""
    import pyotp
    st.subheader("Enter your 2FA code"); st.caption("Enter the 6-digit code from your authenticator app.")
    auth_code = st.text_input("2FA Code", max_chars=6, key="2fa_code")
    if st.button("Authenticate", use_container_width=True):
        if auth_code:
            try:
                if pyotp.TOTP(st.session_state.pyotp_secret).verify(auth_code):
                    st.session_state.authenticated = True; st.rerun()
                else: st.error("Invalid code.")
            except Exception as e: st.error(f"Authentication error: {e}")
        else: st.warning("Please enter a code.")

@st.dialog("Generate QR Code for 2FA")
def qr_code_dialog():
    """Dialog to generate a QR code for 2FA setup."""
    import pyotp, qrcode, io
    st.subheader("Set up Two-Factor Authentication")
    st.info("Scan this QR code with your authenticator app (e.g., Google Authenticator). This is a one-time setup.")

    if st.session_state.pyotp_secret is None:
        st.session_state.pyotp_secret = get_user_secret(st.session_state.get('profile', {}))
    
    secret, user_name = st.session_state.pyotp_secret, st.session_state.get('profile', {}).get('user_name', 'User')
    uri = pyotp.totp.TOTP(secret).provisioning_uri(user_name, issuer_name="BlockVista Terminal")
    
    img = qrcode.make(uri); buf = io.BytesIO(); img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="Scan with authenticator app", use_container_width=True)
    st.markdown(f"**Secret Key:** `{secret}` (Or enter manually)")
    
    if st.button("I have scanned the code. Continue.", use_container_width=True):
        st.session_state.two_factor_setup_complete = True; st.rerun()

def show_login_animation():
    """Displays a boot-up animation after login."""
    st.title("BlockVista Terminal")
    progress_bar, status_text = st.progress(0), st.empty()
    steps = {"Authenticating...": 25, "Establishing connection...": 50, "Fetching data feeds...": 75, "Initializing... COMPLETE": 100}
    for text, progress in steps.items():
        status_text.text(f"STATUS: {text}"); progress_bar.progress(progress); a_time.sleep(0.7)
    a_time.sleep(0.5); st.session_state['login_animation_complete'] = True; st.rerun()

def login_page():
    """Displays the login page for broker authentication."""
    from kiteconnect import KiteConnect
    st.title("BlockVista Terminal"); st.subheader("Broker Login")
    broker = st.selectbox("Select Your Broker", ["Zerodha"])
    
    if broker == "Zerodha":
        # RENDER DEPLOYMENT NOTE:
        # st.secrets will automatically read environment variables on Render.
        # Ensure ZERODHA_API_KEY and ZERODHA_API_SECRET are set in your Render dashboard.
        api_key, api_secret = st.secrets.get("ZERODHA_API_KEY"), st.secrets.get("ZERODHA_API_SECRET")
        if not api_key or not api_secret:
            st.error("Kite API credentials not found in environment variables."); st.stop()
            
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                st.session_state.update({
                    'access_token': data["access_token"], 'kite': kite,
                    'profile': kite.profile(), 'broker': "Zerodha"
                })
                st.query_params.clear(); st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}"); st.query_params.clear()
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())
            st.info("Login with Zerodha Kite to begin. You will be redirected back.")

def main_app():
    """The main application interface after successful login."""
    apply_custom_styling()
    display_overnight_changes_bar()
    
    if st.session_state.get('profile'):
        if not st.session_state.get('two_factor_setup_complete'):
            qr_code_dialog(); return
        if not st.session_state.get('authenticated'):
            two_factor_dialog(); return

    st.sidebar.title(f"Welcome, {st.session_state.profile['user_name']}")
    st.sidebar.caption(f"Connected via {st.session_state.broker}")
    st.sidebar.divider()
    
    st.sidebar.header("Terminal Controls")
    st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options", "HFT"], horizontal=True)
    st.sidebar.divider()
    
    if st.session_state.terminal_mode == "HFT":
        refresh_interval, auto_refresh = 2, True
        st.sidebar.header("HFT Mode Active"); st.sidebar.caption(f"Refresh: {refresh_interval}s")
    else:
        st.sidebar.header("Live Data")
        auto_refresh = st.sidebar.toggle("Auto Refresh", value=True)
        refresh_interval = st.sidebar.number_input("Interval (s)", 5, 60, 10, disabled=not auto_refresh)
    
    st.sidebar.divider()
    st.sidebar.header("Navigation")
    pages = {
        "Cash": {"Dashboard": page_dashboard, "Algo Bots": page_algo_bots, "Premarket Pulse": page_premarket_pulse, "Advanced Charting": page_advanced_charting, "Market Scanners": page_market_scanners, "Portfolio & Risk": page_portfolio_and_risk, "Basket Orders": page_basket_orders, "Forecasting (ML)": page_forecasting_ml, "Algo Strategy Hub": page_algo_strategy_maker, "AI Discovery": page_ai_discovery, "AI Assistant": page_ai_assistant, "Economic Calendar": page_economic_calendar},
        "Options": {"F&O Analytics": page_fo_analytics, "Options Strategy Builder": page_option_strategy_builder, "Greeks Calculator": page_greeks_calculator, "Portfolio & Risk": page_portfolio_and_risk, "AI Assistant": page_ai_assistant},
        "Futures": {"Futures Terminal": page_futures_terminal, "Algo Bots": page_algo_bots, "Advanced Charting": page_advanced_charting, "Algo Strategy Hub": page_algo_strategy_maker, "Portfolio & Risk": page_portfolio_and_risk, "AI Assistant": page_ai_assistant},
        "HFT": {"HFT Terminal": page_hft_terminal, "Portfolio & Risk": page_portfolio_and_risk}
    }
    selection = st.sidebar.radio("Go to", list(pages[st.session_state.terminal_mode].keys()), key='nav_selector')
    
    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    no_refresh_pages = ["Forecasting (ML)", "AI Assistant", "AI Discovery", "Algo Strategy Hub", "Algo Bots"]
    if auto_refresh and selection not in no_refresh_pages:
        # RENDER DEPLOYMENT NOTE: Replaced streamlit_autorefresh with a more robust JS-based solution
        from streamlit.components.v1 import html
        js_code = f"setTimeout(function(){{window.parent.location.reload()}}, {refresh_interval * 1000});"
        html(f"<script>{js_code}</script>", height=0)
    
    # Execute the selected page function
    # NOTE: The full code for all page_...() functions must be present in this file.
    # For this example, only the new page_algo_bots() is shown in full.
    # You would need to paste the code for all your other pages (page_dashboard, etc.) here.
    if selection in pages[st.session_state.terminal_mode]:
        pages[st.session_state.terminal_mode][selection]()
    else:
        st.error("Page not found.")


# --- Application Entry Point ---
if __name__ == "__main__":
    initialize_session_state()
    
    if 'profile' in st.session_state and st.session_state.profile:
        if st.session_state.get('login_animation_complete'):
            main_app()
        else:
            show_login_animation()
    else:
        login_page()

