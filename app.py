# ================ 0. REQUIRED LIBRARIES (Lightweight Only) ================
# Speed Optimization: Heavy libraries are now imported inside functions where they are used (lazy loading).
# This dramatically improves the initial startup time of the application.
import streamlit as st
from datetime import datetime, timedelta, time
import pytz
import re
import time as a_time # Renamed to avoid conflict with datetime.time

# ================ 1. STYLING AND CONFIGURATION ===============
st.set_page_config(page_title="BlockVista Terminal", layout="wide", initial_sidebar_state="expanded")

def apply_custom_styling():
    """Applies a comprehensive CSS stylesheet for professional theming."""
    theme_css = """
    <style>
        :root {
            --dark-bg: #0E1117;
            --dark-secondary-bg: #161B22;
            --dark-widget-bg: #21262D;
            --dark-border: #30363D;
            --dark-text: #c9d1d9;
            --dark-text-light: #8b949e;
            --dark-green: #28a745;
            --dark-red: #da3633;
            --light-bg: #FFFFFF;
            --light-secondary-bg: #F0F2F6;
            --light-widget-bg: #F8F9FA;
            --light-border: #dee2e6;
            --light-text: #212529;
            --light-text-light: #6c757d;
            --light-green: #198754;
            --light-red: #dc3545;
        }
        body.dark-theme {
            --primary-bg: var(--dark-bg);
            --secondary-bg: var(--dark-secondary-bg);
            --widget-bg: var(--dark-widget-bg);
            --border-color: var(--dark-border);
            --text-color: var(--dark-text);
            --text-light: var(--dark-text-light);
            --green: var(--dark-green);
            --red: var(--dark-red);
        }
        body.light-theme {
            --primary-bg: var(--light-bg);
            --secondary-bg: var(--light-secondary-bg);
            --widget-bg: var(--light-widget-bg);
            --border-color: var(--light-border);
            --text-color: var(--light-text);
            --text-light: var(--light-text-light);
            --green: var(--light-green);
            --red: var(--light-red);
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
        document.body.classList.add('{st.session_state.theme.lower()}-theme');
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
        
    template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
    fig.update_layout(title=f'{ticker} Price Chart ({chart_type})', yaxis_title='Price (INR)', xaxis_rangeslider_visible=False, template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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
            is_option = any(char.isdigit() for char in symbol)
            if is_option:
                exchange = 'NFO'
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
            live_df = yf.download(source_info['tradingsymbol'], period="max")
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
    """Loads stock-to-sector mapping from a local CSV file."""
    import pandas as pd
    try:
        # Assuming 'sensex_sectors.csv' is in the same directory as the script.
        return pd.read_csv("sensex_sectors.csv")
    except FileNotFoundError:
        st.warning("'sensex_sectors.csv' not found. Sector allocation will be unavailable.")
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

# ================ 5. PAGE DEFINITIONS ============

# --- Bharatiya Market Pulse (BMP) Functions ---
def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30: return 50, "Calculating...", "#cccccc"
    
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    vix_min, vix_max = lookback_df['vix_value'].min(), lookback_df['vix_value'].max()
    vix_norm = 100 - (((vix_value - vix_min) / (vix_max - vix_min)) * 100) if (vix_max - vix_min) > 0 else 50

    bmp_score = min(100, max(0, (0.40 * nifty_norm) + (0.40 * sensex_norm) + (0.20 * vix_norm)))

    if bmp_score >= 80: label, color = "Bharat Udaan (Very Bullish)", "#00b300"
    elif bmp_score >= 60: label, color = "Bharat Pragati (Bullish)", "#33cc33"
    elif bmp_score >= 40: label, color = "Bharat Santulan (Neutral)", "#ffcc00"
    elif bmp_score >= 20: label, color = "Bharat Sanket (Bearish)", "#ff6600"
    else: label, color = "Bharat Mandhi (Very Bearish)", "#ff0000"

    return bmp_score, label, color

@st.cache_data(ttl=300)
def get_nifty50_constituents(instrument_df):
    """Fetches a static list of NIFTY 50 stocks."""
    import pandas as pd
    if instrument_df.empty: return pd.DataFrame()
    
    nifty50_symbols = [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'HINDUNILVR', 'ITC', 
        'LT', 'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT', 
        'AXISBANK', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'M&M', 'NESTLEIND',
        'ADANIENT', 'TATASTEEL', 'INDUSINDBK', 'TECHM', 'NTPC', 'MARUTI', 
        'BAJAJ-AUTO', 'POWERGRID', 'HCLTECH', 'ADANIPORTS', 'BPCL', 'COALINDIA', 
        'EICHERMOT', 'GRASIM', 'JSWSTEEL', 'SHREECEM', 'HEROMOTOCO', 'HINDALCO',
        'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'SBILIFE', 'TATAMOTORS', 'BRITANNIA',
        'DIVISLAB', 'BAJAJFINSV', 'SUNPHARMA', 'HDFCLIFE'
    ]
    return pd.DataFrame({'Symbol': nifty50_symbols}).head(15)

def create_nifty_heatmap(instrument_df):
    """Generates a Plotly Treemap for NIFTY 50 stocks."""
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    constituents_df = get_nifty50_constituents(instrument_df)
    if constituents_df.empty: return go.Figure()
    
    symbols_with_exchange = [{'symbol': s, 'exchange': 'NSE'} for s in constituents_df['Symbol'].tolist()]
    live_data = get_watchlist_data(symbols_with_exchange)
    
    if live_data.empty: return go.Figure()
        
    full_data = pd.merge(live_data, constituents_df, left_on='Ticker', right_on='Symbol', how='left')
    full_data['size'] = full_data['Price'].astype(float).fillna(1) * 1000 # Use price as proxy for size
    
    fig = go.Figure(go.Treemap(
        labels=full_data['Ticker'], parents=[''] * len(full_data), values=full_data['size'],
        marker=dict(colorscale='RdYlGn', colors=full_data['% Change'], colorbar=dict(title="% Change")),
        text=full_data['Ticker'], textinfo="label",
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{customdata[1]:.2f}%<extra></extra>',
        customdata=np.column_stack([full_data['Price'], full_data['% Change']])
    ))
    fig.update_layout(title="NIFTY 50 Heatmap (Live)")
    return fig

@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using yfinance."""
    import yfinance as yf
    try:
        data = yf.download("IN=F", period="1d", interval="1m", progress=False)
        return data if not data.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def page_dashboard():
    """A redesigned 'Trader UI' Dashboard."""
    import pandas as pd
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
        
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'}, {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    bmp_col, heatmap_col = st.columns([1, 1], gap="large")
    with bmp_col:
        st.subheader("Bharatiya Market Pulse (BMP)")
        if not index_data.empty:
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
            
            nifty_hist = get_historical_data(get_instrument_token('NIFTY 50', instrument_df, 'NSE'), 'day', period='1y')
            sensex_hist = get_historical_data(get_instrument_token('SENSEX', instrument_df, 'BSE'), 'day', period='1y')
            vix_hist = get_historical_data(get_instrument_token('INDIA VIX', instrument_df, 'NSE'), 'day', period='1y')
            
            if not nifty_hist.empty and not sensex_hist.empty and not vix_hist.empty:
                lookback_data = pd.DataFrame({
                    'nifty_change': nifty_hist['close'].pct_change() * 100,
                    'sensex_change': sensex_hist['close'].pct_change() * 100,
                    'vix_value': vix_hist['close']
                }).dropna()
                
                bmp_score, bmp_label, bmp_color = get_bmp_score_and_label(nifty_row['% Change'], sensex_row['% Change'], vix_row['Price'], lookback_data)
                
                st.markdown(f'<div class="metric-card" style="border-color:{bmp_color};"><h3>{bmp_score:.2f}</h3><p style="color:{bmp_color}; font-weight:bold;">{bmp_label}</p><small>Proprietary score from NIFTY, SENSEX, and India VIX.</small></div>', unsafe_allow_html=True)
                with st.expander("What do the BMP scores mean?"):
                    st.markdown("- **80-100 (Bharat Udaan):** Very Strong Bullish Momentum.\n- **60-80 (Bharat Pragati):** Moderately Bullish Sentiment.\n- **40-60 (Bharat Santulan):** Neutral or Sideways Market.\n- **20-40 (Bharat Sanket):** Moderately Bearish Sentiment.\n- **0-20 (Bharat Mandhi):** Very Strong Bearish Momentum.")
            else:
                st.info("BMP data is loading...")
        else:
            st.info("BMP data is loading...")
    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        st.plotly_chart(create_nifty_heatmap(instrument_df), use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            st.session_state.active_watchlist = st.radio("Select Watchlist", options=st.session_state.watchlists.keys(), horizontal=True, label_visibility="collapsed")
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 15: st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange})
                            st.rerun()
                        else: st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", unsafe_allow_html=True)
                    w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                    quantity = w_cols[2].number_input("Qty", 1, 1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                    if w_cols[3].button("B", key=f"buy_{row['Ticker']}", use_container_width=True): place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                    if w_cols[4].button("S", key=f"sell_{row['Ticker']}", use_container_width=True): place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
                    if w_cols[5].button("🗑️", key=f"del_{row['Ticker']}", use_container_width=True):
                        st.session_state.watchlists[st.session_state.active_watchlist] = [item for item in active_list if item['symbol'] != row['Ticker']]
                        st.rerun()
                st.markdown("---")

        with tab2:
            st.subheader("My Portfolio")
            _, holdings_df, total_pnl, total_investment = get_portfolio()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
            st.metric("Today's Profit & Loss", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
            with st.expander("View Holdings"):
                st.dataframe(holdings_df, use_container_width=True, hide_index=True) if not holdings_df.empty else st.info("No holdings found.")
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
    
    ticker_symbols = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist'), [])
    if ticker_symbols:
        ticker_data = get_watchlist_data(ticker_symbols)
        if not ticker_data.empty:
            ticker_html = "".join([f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {'#28a745' if item['Change'] > 0 else '#FF4B4B'};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>" for _, item in ticker_data.iterrows()])
            st.markdown(f"""
            <style>
                @keyframes marquee {{ 0% {{ transform: translate(100%, 0); }} 100% {{ transform: translate(-100%, 0); }} }}
                .marquee-container {{ width: 100%; overflow: hidden; position: fixed; bottom: 0; left: 0; background-color: #1a1a1a; border-top: 1px solid #333; padding: 5px 0; white-space: nowrap; }}
                .marquee-content {{ display: inline-block; padding-left: 100%; animation: marquee 35s linear infinite; }}
            </style>
            <div class="marquee-container"><div class="marquee-content">{ticker_html}</div></div>
            """, unsafe_allow_html=True)

def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    layout_option = st.radio("Select Layout", ["Single Chart", "2 Charts", "4 Charts", "6 Charts"], horizontal=True)
    num_charts = {"Single Chart": 1, "2 Charts": 2, "4 Charts": 4, "6 Charts": 6}[layout_option]
    st.markdown("---")
    
    if num_charts == 1:
        render_chart_controls(0, instrument_df)
    elif num_charts == 2:
        cols = st.columns(2)
        for i, col in enumerate(cols):
            with col: render_chart_controls(i, instrument_df)
    elif num_charts == 4:
        for i in range(2):
            cols = st.columns(2)
            with cols[0]: render_chart_controls(i * 2, instrument_df)
            with cols[1]: render_chart_controls(i * 2 + 1, instrument_df)
    elif num_charts == 6:
        for i in range(2):
            cols = st.columns(3)
            with cols[0]: render_chart_controls(i * 3, instrument_df)
            with cols[1]: render_chart_controls(i * 3 + 1, instrument_df)
            with cols[2]: render_chart_controls(i * 3 + 2, instrument_df)

def render_chart_controls(i, instrument_df):
    """Helper function to render controls for a single chart."""
    with st.container(border=True):
        st.subheader(f"Chart {i+1}")
        
        chart_cols = st.columns(4)
        ticker = chart_cols[0].text_input("Symbol", "NIFTY 50", key=f"ticker_{i}").upper()
        period = chart_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], 4, key=f"period_{i}")
        interval = chart_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], 2, key=f"interval_{i}")
        chart_type = chart_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{i}")

        token = get_instrument_token(ticker, instrument_df)
        data = get_historical_data(token, interval, period=period)

        if data.empty:
            st.warning(f"No data for {ticker} with selected parameters.")
        else:
            st.plotly_chart(create_chart(data, ticker, chart_type), use_container_width=True, key=f"chart_{i}")

            order_cols = st.columns([2,1,1,1])
            order_cols[0].markdown("**Quick Order**")
            quantity = order_cols[1].number_input("Qty", 1, 1, key=f"qty_{i}", label_visibility="collapsed")
            if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'BUY', 'MIS')
            if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'SELL', 'MIS')

def page_premarket_pulse():
    """Global market overview and premarket indicators."""
    import numpy as np
    display_header()
    st.title("Premarket & Global Cues")
    st.markdown("---")

    st.subheader("Global Market Snapshot")
    global_tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "Hang Seng": "^HSI"}
    global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_tickers))
        for i, (name, _) in enumerate(global_tickers.items()):
            data_row = global_data[global_data['Ticker'] == name]
            if not data_row.empty:
                price, change = data_row.iloc[0]['Price'], data_row.iloc[0]['% Change']
                cols[i].metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%") if not np.isnan(price) else cols[i].metric(label=name, value="N/A", delta="--")
    else:
        st.info("Loading global market data...")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("NIFTY 50 Futures (Live Proxy)")
        gift_data = get_gift_nifty_data()
        st.plotly_chart(create_chart(gift_data, "NIFTY 50 Futures (Proxy)"), use_container_width=True) if not gift_data.empty else st.warning("Could not load NIFTY 50 Futures chart data.")
    with col2:
        st.subheader("Key Asian Markets")
        asian_tickers = {"Nikkei 225": "^N225", "Hang Seng": "^HSI"}
        asian_data = get_global_indices_data(asian_tickers)
        if not asian_data.empty:
            for name, _ in asian_tickers.items():
                data_row = asian_data[asian_data['Ticker'] == name]
                if not data_row.empty:
                    price, change = data_row.iloc[0]['Price'], data_row.iloc[0]['% Change']
                    st.metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%") if not np.isnan(price) else st.metric(label=name, value="N/A", delta="--")
        else:
            st.info("Loading Asian market data...")

    st.markdown("---")
    st.subheader("Latest Market News")
    news_df = fetch_and_analyze_news()
    if not news_df.empty:
        for _, news in news_df.head(10).iterrows():
            score = news['sentiment']
            icon = "🔼" if score > 0.2 else "🔽" if score < -0.2 else "▶️"
            st.markdown(f"**{icon} [{news['title']}]({news['link']})** - *{news['source']}*")
    else:
        st.info("News data is loading...")

def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    display_header()
    st.title("F&O Analytics Hub")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access F&O Analytics.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Options Chain", "PCR Analysis", "Volatility & OI Analysis"])
    
    with tab1:
        st.subheader("Live Options Chain")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            st.session_state.underlying_pcr = underlying 
        
        chain_df, expiry, underlying_ltp, _ = get_options_chain(underlying, instrument_df)

        if not chain_df.empty:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")
            with col3:
                if st.button("Most Active Options", use_container_width=True):
                    show_most_active_dialog(underlying, instrument_df)

            st.dataframe(
                style_option_chain(chain_df, underlying_ltp).format({'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}', 'STRIKE': '₹{:.0f}', 'CALL OI': '{:,.0f}', 'PUT OI': '{:,.0f}'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Could not load options chain data.")
    
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        chain_df, _, _, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)
        if not chain_df.empty and 'CALL OI' in chain_df.columns:
            total_ce_oi, total_pe_oi = chain_df['CALL OI'].sum(), chain_df['PUT OI'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total CE OI", f"{total_ce_oi:,.0f}"); col2.metric("Total PE OI", f"{total_pe_oi:,.0f}"); col3.metric("PCR", f"{pcr:.2f}")
            
            if pcr > 1.3: st.success("High PCR suggests potential bearish sentiment.")
            elif pcr < 0.7: st.error("Low PCR suggests potential bullish sentiment.")
            else: st.info("PCR indicates neutral sentiment.")
        else:
            st.info("PCR data is loading... Select an underlying in the 'Options Chain' tab first.")
    
    with tab3:
        st.subheader("Volatility & Open Interest Surface")
        st.info("Real-time implied volatility and OI analysis for options contracts.")
        chain_df, expiry, underlying_ltp, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)

        if not chain_df.empty and expiry and underlying_ltp > 0:
            T, r = (expiry - datetime.now().date()).days / 365.0, 0.07
            with st.spinner("Calculating Implied Volatility..."):
                chain_df['IV_CE'] = chain_df.apply(lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['CALL LTP'], 'call') * 100, axis=1)
                chain_df['IV_PE'] = chain_df.apply(lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['PUT LTP'], 'put') * 100, axis=1)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_CE'], mode='lines+markers', name='Call IV', line=dict(color='cyan')), secondary_y=False)
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_PE'], mode='lines+markers', name='Put IV', line=dict(color='magenta')), secondary_y=False)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['CALL OI'], name='Call OI', marker_color='rgba(0, 255, 255, 0.4)'), secondary_y=True)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['PUT OI'], name='Put OI', marker_color='rgba(255, 0, 255, 0.4)'), secondary_y=True)

            template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
            fig.update_layout(title_text=f"{st.session_state.get('underlying_pcr', 'NIFTY')} IV & OI Profile for {expiry.strftime('%d %b %Y')}", template=template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select an underlying in the 'Options Chain' tab to view the volatility surface.")

def page_forecasting_ml():
    """A page for advanced ML forecasting."""
    import plotly.graph_objects as go
    display_header()
    st.title("Advanced ML Forecasting")
    st.info("Train a Seasonal ARIMA model to forecast future prices. This is for educational purposes and not financial advice.", icon="🧠")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument")
        
        forecast_durations = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90}
        duration_key = st.radio("Forecast Duration", list(forecast_durations.keys()), horizontal=True, key="ml_duration")
        forecast_steps = forecast_durations[duration_key]

        if st.button("Train Model & Forecast", use_container_width=True, type="primary"):
            with st.spinner(f"Loading data and training model for {instrument_name}..."):
                data = load_and_combine_data(instrument_name)
                if data.empty or len(data) < 100:
                    st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
                else:
                    forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                    st.session_state.update({'ml_forecast_df': forecast_df, 'ml_backtest_df': backtest_df, 'ml_conf_int_df': conf_int_df, 'ml_instrument_name': instrument_name, 'ml_historical_data': data, 'ml_duration_key': duration_key})
                    st.success("Model trained successfully!")

    with col2:
        if 'ml_instrument_name' in st.session_state and st.session_state.ml_instrument_name:
            instrument_name = st.session_state.ml_instrument_name
            st.subheader(f"Forecast Results for {instrument_name}")

            forecast_df, backtest_df, conf_int_df, data, duration_key = (
                st.session_state.get('ml_forecast_df'), st.session_state.get('ml_backtest_df'),
                st.session_state.get('ml_conf_int_df'), st.session_state.get('ml_historical_data'),
                st.session_state.get('ml_duration_key')
            )

            if all(v is not None for v in [forecast_df, backtest_df, conf_int_df, data]):
                fig = create_chart(data.tail(252), instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Backtest Prediction', line=dict(color='orange', dash='dot')))
                fig.update_layout(title=f"{instrument_name} Forecast vs. Historical Data")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Model Performance (Backtest)")
                backtest_durations = {"Full History": len(backtest_df), "Last Year": 252, "6 Months": 126, "3 Months": 63}
                backtest_duration_key = st.selectbox("Select Backtest Period", list(backtest_durations.keys()))
                display_df = backtest_df.tail(backtest_durations[backtest_duration_key])
                mape = mean_absolute_percentage_error(display_df['Actual'], display_df['Predicted'])
                
                metric_cols = st.columns(2)
                metric_cols[0].metric(f"Accuracy ({backtest_duration_key})", f"{100 - mape:.2f}%")
                metric_cols[1].metric(f"MAPE ({backtest_duration_key})", f"{mape:.2f}%")

                with st.expander(f"View {duration_key} Forecast Data"):
                    st.dataframe(forecast_df.join(conf_int_df).style.format("₹{:.2f}"), use_container_width=True)
            else:
                st.info("Train a model to see the forecast results.")
        else:
            st.info("Select an instrument and run the forecast to see results.")

def page_portfolio_and_risk():
    """Portfolio and risk management page."""
    import pandas as pd
    import plotly.graph_objects as go

    display_header()
    st.title("Portfolio & Risk")

    client = get_broker_client()
    if not client:
        st.info("Connect to a broker to view your portfolio and positions.")
        return

    positions_df, holdings_df, total_pnl, _ = get_portfolio()
    if holdings_df.empty and positions_df.empty:
        st.info("No holdings or positions found to analyze.")
        return

    tab1, tab2, tab3 = st.tabs(["Day Positions", "Holdings (Investments)", "Live Order Book"])
    
    with tab1:
        st.subheader("Live Intraday Positions")
        if not positions_df.empty:
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            st.metric("Total Day P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        else:
            st.info("No open positions for the day.")
    
    with tab2:
        st.subheader("Investment Holdings")
        if not holdings_df.empty:
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            st.markdown("---")
            st.subheader("Portfolio Allocation")
            
            sector_df = get_sector_data()
            holdings_df['current_value'] = holdings_df['quantity'] * holdings_df['last_price']
            
            if sector_df is not None:
                holdings_df = pd.merge(holdings_df, sector_df, left_on='tradingsymbol', right_on='Symbol', how='left')
                holdings_df['Sector'].fillna('Uncategorized', inplace=True)
            else:
                holdings_df['Sector'] = 'Uncategorized'
            
            col1_alloc, col2_alloc = st.columns(2)
            with col1_alloc:
                st.subheader("Stock-wise Allocation")
                fig = go.Figure(data=[go.Pie(labels=holdings_df['tradingsymbol'], values=holdings_df['current_value'], hole=.3, textinfo='label+percent')])
                fig.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == "Dark" else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            with col2_alloc:
                st.subheader("Sector-wise Allocation")
                sector_allocation = holdings_df.groupby('Sector')['current_value'].sum().reset_index()
                fig = go.Figure(data=[go.Pie(labels=sector_allocation['Sector'], values=sector_allocation['current_value'], hole=.3, textinfo='label+percent')])
                fig.update_layout(showlegend=False, template='plotly_dark' if st.session_state.theme == "Dark" else 'plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings found.")

    with tab3:
        st.subheader("Live Order Book")
        try:
            orders = client.orders()
            if orders:
                orders_df = pd.DataFrame(orders)
                st.dataframe(orders_df[['order_timestamp', 'tradingsymbol', 'transaction_type', 'order_type', 'quantity', 'average_price', 'status']], use_container_width=True, hide_index=True)
            else:
                st.info("No orders placed today.")
        except Exception as e:
            st.error(f"Failed to fetch order book: {e}")

def page_ai_assistant():
    """AI-powered assistant for portfolio and market queries."""
    import pandas as pd
    from tabulate import tabulate
    import numpy as np

    display_header()
    st.title("Portfolio-Aware Assistant")
    instrument_df = get_instrument_df()

    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your portfolio or the markets today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_lower = prompt.lower()
                response = "I can help with your portfolio, orders, and market data. Try 'What are my positions?' or 'Show me the option chain for BANKNIFTY'."
                client = get_broker_client()
                if not client:
                    response = "I am not connected to your broker. Please log in first."
                
                elif any(word in prompt_lower for word in ["holdings", "investments"]):
                    _, holdings_df, _, _ = get_portfolio()
                    response = f"Here are your holdings:\n```\n{tabulate(holdings_df, headers='keys', tablefmt='psql')}\n```" if not holdings_df.empty else "You have no holdings."
                elif "positions" in prompt_lower:
                    positions_df, _, total_pnl, _ = get_portfolio()
                    response = f"Total P&L is ₹{total_pnl:,.2f}. Positions:\n```\n{tabulate(positions_df, headers='keys', tablefmt='psql')}\n```" if not positions_df.empty else "You have no open positions."
                elif "orders" in prompt_lower:
                    orders = client.orders()
                    response = f"Today's orders:\n```\n{tabulate(pd.DataFrame(orders), headers='keys', tablefmt='psql')}\n```" if orders else "No orders today."
                elif any(word in prompt_lower for word in ["funds", "margin", "balance"]):
                    funds = client.margins()
                    response = f"Available Funds:\n- Equity: ₹{funds['equity']['available']['live_balance']:,.2f}\n- Commodity: ₹{funds['commodity']['available']['live_balance']:,.2f}"
                elif "price of" in prompt_lower or "ltp of" in prompt_lower:
                    try:
                        ticker = prompt.split(" of ")[-1].strip().upper()
                        instrument = instrument_df[instrument_df['tradingsymbol'] == ticker]
                        if not instrument.empty:
                            exchange = instrument.iloc[0]['exchange']
                            ltp_data = get_watchlist_data([{'symbol': ticker, 'exchange': exchange}])
                            price = ltp_data.iloc[0]['Price'] if not ltp_data.empty else "N/A"
                            response = f"The price of {ticker} is {price}."
                        else: response = f"Ticker '{ticker}' not found."
                    except Exception: response = "Please specify a ticker, e.g., 'price of RELIANCE'."
                
                elif "buy" in prompt_lower or "sell" in prompt_lower:
                    match = re.search(r'(buy|sell)\s+(\d+)\s+shares?\s+of\s+([a-zA-Z0-9\-\&_]+)', prompt_lower)
                    if match:
                        trans_type, quantity, symbol = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
                        st.session_state.last_order_details = {"symbol": symbol, "quantity": quantity, "transaction_type": trans_type, "confirmed": False}
                        response = f"I can place a {trans_type} order for {quantity} shares of {symbol}. Please type 'confirm'."
                    else: response = "I couldn't understand. Use a format like 'Buy 100 shares of RELIANCE'."
                elif prompt_lower == "confirm" and "last_order_details" in st.session_state and not st.session_state.last_order_details.get("confirmed"):
                    order_details = st.session_state.last_order_details
                    place_order(instrument_df, order_details['symbol'], order_details['quantity'], 'MARKET', order_details['transaction_type'], 'MIS')
                    order_details['confirmed'] = True
                    response = f"Confirmed and placed {order_details['transaction_type']} order for {order_details['quantity']} shares of {order_details['symbol']}."

                elif "technical analysis for" in prompt_lower:
                    ticker = prompt.split("for")[-1].strip().upper()
                    token = get_instrument_token(ticker, instrument_df)
                    if token:
                        data = get_historical_data(token, 'day', period='6mo')
                        if not data.empty:
                            analysis = interpret_indicators(data)
                            response = f"**Technical Analysis for {ticker}:**\n\n" + "\n".join([f"- **{k}:** {v}" for k, v in analysis.items()])
                        else: response = f"Could not retrieve data for {ticker}."
                    else: response = f"Could not find ticker '{ticker}'."
                
                elif "news for" in prompt_lower:
                    query = prompt.split("for")[-1].strip()
                    news_df = fetch_and_analyze_news(query)
                    if not news_df.empty:
                        response = f"**Top 3 news headlines for {query}:**\n\n" + "\n".join([f"1. [{row['title']}]({row['link']}) - _{row['source']}_" for _, row in news_df.head(3).iterrows()])
                    else: response = f"No recent news found for '{query}'."
                
                elif "greeks for" in prompt_lower or "iv for" in prompt_lower:
                    try:
                        match = re.search(r'([A-Z]+)(\d{2}[A-Z]{3}\d+[CEPE]{2})', prompt.upper())
                        if match:
                            symbol = match.group(0)
                            details = instrument_df[instrument_df['tradingsymbol'] == symbol].iloc[0]
                            expiry_date = details['expiry'].date() if hasattr(details['expiry'], 'date') else details['expiry']
                            _, expiry, underlying_ltp, _ = get_options_chain(details['name'], instrument_df, expiry_date)
                            
                            ltp = client.ltp(f"NFO:{symbol}")[f"NFO:{symbol}"]['last_price']
                            T = max((expiry - datetime.now().date()).days, 0) / 365.0
                            iv = implied_volatility(underlying_ltp, details['strike'], T, 0.07, ltp, details['instrument_type'].lower())
                            
                            if not np.isnan(iv):
                                greeks = black_scholes(underlying_ltp, details['strike'], T, 0.07, iv, details['instrument_type'].lower())
                                response = f"Greeks for **{symbol}**:\n- **IV:** {iv*100:.2f}%\n- **Delta:** {greeks['delta']:.4f}\n- **Gamma:** {greeks['gamma']:.4f}\n- **Vega:** {greeks['vega']:.4f}\n- **Theta:** {greeks['theta']:.4f}"
                            else: response = f"Could not calculate IV for {symbol}. It might be illiquid."
                        else: response = "Please specify a valid option symbol (e.g., NIFTY24OCT...CE)."
                    except Exception as e: response = f"An error occurred: {e}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def page_basket_orders():
    """A page for creating, managing, and executing basket orders."""
    display_header()
    st.title("Basket Orders")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the basket order feature.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Add Order to Basket")
        with st.form("add_to_basket_form"):
            symbol = st.text_input("Symbol").upper()
            transaction_type = st.radio("Transaction", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, step=1)
            product = st.radio("Product", ["MIS", "CNC"], horizontal=True)
            order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
            price = st.number_input("Price", 0.01) if order_type == "LIMIT" else 0
            
            if st.form_submit_button("Add to Basket"):
                if symbol and quantity > 0:
                    st.session_state.basket.append({'symbol': symbol, 'transaction_type': transaction_type, 'quantity': quantity, 'product': product, 'order_type': order_type, 'price': price if price > 0 else None})
                    st.success(f"Added {symbol} to basket!")
                    st.rerun()

    with col2:
        st.subheader("Current Basket")
        if st.session_state.basket:
            for i, order in enumerate(st.session_state.basket):
                with st.expander(f"{order['transaction_type']} {order['quantity']} {order['symbol']}"):
                    st.write(f"**Product:** {order['product']}"); st.write(f"**Order Type:** {order['order_type']}")
                    if order['price']: st.write(f"**Price:** ₹{order['price']}")
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.basket.pop(i)
                        st.rerun()
            
            st.markdown("---")
            if st.button("Execute Basket Order", type="primary", use_container_width=True):
                execute_basket_order(st.session_state.basket, instrument_df)
        else:
            st.info("Your basket is empty.")

def run_backtest(strategy_func, data, **params):
    """Runs a backtest for a given strategy function."""
    import pandas as pd
    df = data.copy()
    signals = strategy_func(df, **params)
    
    initial_capital, capital, position = 100000.0, 100000.0, 0
    portfolio_value = []
    
    for i in range(len(df)):
        if signals[i] == 'BUY' and position == 0:
            position = capital / df['close'][i]; capital = 0
        elif signals[i] == 'SELL' and position > 0:
            capital = position * df['close'][i]; position = 0
        
        portfolio_value.append(capital + (position * df['close'][i]))
        
    pnl = (portfolio_value[-1] - initial_capital) / initial_capital * 100
    return pnl, pd.Series(portfolio_value, index=df.index)

def rsi_strategy(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """Simple RSI Crossover Strategy"""
    import pandas_ta as ta
    rsi = ta.rsi(df['close'], length=rsi_period)
    signals = [''] * len(df)
    for i in range(1, len(df)):
        if rsi[i-1] < rsi_oversold and rsi[i] > rsi_oversold: signals[i] = 'BUY'
        elif rsi[i-1] > rsi_overbought and rsi[i] < rsi_overbought: signals[i] = 'SELL'
    return signals

def macd_strategy(df, fast=12, slow=26, signal=9):
    """MACD Crossover Strategy"""
    import pandas_ta as ta
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
    signals, macd_col, signal_col = [''] * len(df), f'MACD_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}'
    for i in range(1, len(df)):
        if macd[macd_col][i-1] < macd[signal_col][i-1] and macd[macd_col][i] > macd[signal_col][i]: signals[i] = 'BUY'
        elif macd[macd_col][i-1] > macd[signal_col][i-1] and macd[macd_col][i] < macd[signal_col][i]: signals[i] = 'SELL'
    return signals

def supertrend_strategy(df, period=7, multiplier=3):
    """Supertrend Strategy"""
    import pandas_ta as ta
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=period, multiplier=multiplier)
    signals, st_col = [''] * len(df), next((col for col in supertrend.columns if 'SUPERT' in col), None)
    if not st_col: return signals 
    for i in range(1, len(df)):
        if df['close'][i] > supertrend[st_col][i-1] and df['close'][i-1] <= supertrend[st_col][i-1]: signals[i] = 'BUY'
        elif df['close'][i] < supertrend[st_col][i-1] and df['close'][i-1] >= supertrend[st_col][i-1]: signals[i] = 'SELL'
    return signals

def page_algo_strategy_maker():
    """Algo Strategy Maker page with backtesting and execution."""
    import plotly.graph_objects as go
    display_header()
    st.title("Algo Strategy Hub")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to use the Algo Strategy Hub.")
        return

    st.info("Select a strategy, configure parameters, run a backtest, and place trades based on the latest signal.", icon="🤖")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Strategy Configuration")
        strategy_options = {"RSI Crossover": rsi_strategy, "MACD Crossover": macd_strategy, "Supertrend Follower": supertrend_strategy}
        selected_strategy = st.selectbox("Select a Strategy", list(strategy_options.keys()))
        
        st.markdown("**Instrument**")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'NFO', 'MCX', 'CDS'])]['tradingsymbol'].unique()
        symbol = st.selectbox("Select Symbol", all_symbols, index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0)
        
        st.markdown("**Parameters**")
        params = {}
        if selected_strategy == "RSI Crossover":
            params['rsi_period'] = st.slider("RSI Period", 5, 30, 14); params['rsi_overbought'] = st.slider("Overbought", 60, 90, 70); params['rsi_oversold'] = st.slider("Oversold", 10, 40, 30)
        elif selected_strategy == "MACD Crossover":
            params['fast'] = st.slider("Fast Period", 5, 20, 12); params['slow'] = st.slider("Slow Period", 20, 50, 26); params['signal'] = st.slider("Signal Period", 5, 20, 9)
        elif selected_strategy == "Supertrend Follower":
            params['period'] = st.slider("ATR Period", 5, 20, 7); params['multiplier'] = st.slider("Multiplier", 1.0, 5.0, 3.0, 0.5)

        st.markdown("**Trade Execution**"); quantity = st.number_input("Trade Quantity", 1, value=1)
        
        run_button = st.button("Run Backtest & Get Signal", use_container_width=True, type="primary")

    with col2:
        if run_button:
            with st.spinner(f"Running backtest for {selected_strategy} on {symbol}..."):
                exchange = instrument_df[instrument_df['tradingsymbol'] == symbol].iloc[0]['exchange']
                token = get_instrument_token(symbol, instrument_df, exchange=exchange)
                data = get_historical_data(token, 'day', period='1y')
                
                if not data.empty and len(data) > 50: 
                    pnl, portfolio_curve = run_backtest(strategy_options[selected_strategy], data, **params)
                    latest_signal = strategy_options[selected_strategy](data, **params)[-1]
                    st.session_state['backtest_results'] = {'pnl': pnl, 'curve': portfolio_curve, 'signal': latest_signal, 'symbol': symbol, 'quantity': quantity}
                else:
                    st.error("Could not fetch enough data for the backtest.")
                    if 'backtest_results' in st.session_state: st.session_state['backtest_results'] = None

        if st.session_state.get('backtest_results'):
            results = st.session_state['backtest_results']
            st.subheader("Backtest Results")
            st.metric("Total P&L (1 Year)", f"{results['pnl']:.2f}%")

            fig = go.Figure().add_trace(go.Scatter(x=results['curve'].index, y=results['curve'], mode='lines', name='Portfolio Value')).update_layout(title="Portfolio Growth Over 1 Year", yaxis_title="Portfolio Value (₹)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Live Signal & Trading")
            signal = results['signal']
            color = "green" if signal == "BUY" else "red" if signal == "SELL" else "orange"
            st.markdown(f"### Latest Signal: <span style='color:{color};'>{signal if signal else 'HOLD'}</span>", unsafe_allow_html=True)

            if signal in ["BUY", "SELL"]:
                if st.button(f"Place {signal} Order for {results['quantity']} of {results['symbol']}", use_container_width=True):
                    place_order(instrument_df, results['symbol'], results['quantity'], "MARKET", signal, "MIS")

@st.cache_data(ttl=3600)
def run_scanner(instrument_df, scanner_type, holdings_df=None):
    """Single function to run different market scanners."""
    import pandas as pd
    client = get_broker_client()
    if not client or instrument_df.empty: return pd.DataFrame()

    if holdings_df is not None and not holdings_df.empty:
        scan_list = holdings_df['tradingsymbol'].unique().tolist()
        st.info("Scanning stocks from your live holdings.")
    else:
        scan_list = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'LT', 'WIPRO', 'AXISBANK', 'MARUTI', 'ASIANPAINT']
        st.info("Scanning a predefined list of NIFTY 50 stocks as no holdings were found.")

    results = []
    token_map = {row['tradingsymbol']: row['instrument_token'] for _, row in instrument_df[instrument_df['tradingsymbol'].isin(scan_list)].iterrows()}
    
    for symbol in scan_list:
        token = token_map.get(symbol)
        if not token: continue
        
        try:
            df = get_historical_data(token, 'day', period='1y')
            if df.empty or len(df) < 252: continue
            df.columns = [c.lower() for c in df.columns]

            if scanner_type == "Momentum":
                rsi = df.iloc[-1].get(next((c for c in df.columns if 'rsi_14' in c), None))
                if rsi and (rsi > 70 or rsi < 30):
                    results.append({'Stock': symbol, 'RSI': f"{rsi:.2f}", 'Signal': "Overbought" if rsi > 70 else "Oversold"})
            
            elif scanner_type == "Trend":
                adx, ema50, ema200 = df.iloc[-1].get(next((c for c in df.columns if 'adx_14' in c), None)), df.iloc[-1].get(next((c for c in df.columns if 'ema_50' in c), None)), df.iloc[-1].get(next((c for c in df.columns if 'ema_200' in c), None))
                if adx and adx > 25 and ema50 and ema200:
                    results.append({'Stock': symbol, 'ADX': f"{adx:.2f}", 'Trend': "Uptrend" if ema50 > ema200 else "Downtrend"})

            elif scanner_type == "Breakout":
                high_52wk, last_close = df['high'].rolling(252).max().iloc[-1], df['close'].iloc[-1]
                if last_close >= high_52wk * 0.98: # Within 2% of 52-week high
                    signal = "Near 52-Week High" + (" (Volume Surge)" if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5 else "")
                    results.append({'Stock': symbol, 'Signal': signal, 'Last Close': last_close, '52Wk High': high_52wk})
        except Exception:
            continue
            
    return pd.DataFrame(results)

def page_market_scanners():
    """Simplified Market Scanners page using the unified scanner function."""
    display_header()
    st.title("Market Scanners")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use this feature.")
        return
        
    _, holdings_df, _, _ = get_portfolio()
    
    st.subheader("Live Market Scanners")
    st.caption("Scanners run on your holdings or a predefined NIFTY 50 list.")
    
    tab1, tab2, tab3 = st.tabs(["Momentum Stocks", "Trending Stocks", "Breakout Stocks"])
    
    with tab1:
        st.subheader("High Momentum Stocks (RSI)")
        with st.spinner("Scanning for momentum..."):
            data = run_scanner(instrument_df, "Momentum", holdings_df)
            st.dataframe(data, use_container_width=True, hide_index=True) if not data.empty else st.info("No stocks with strong momentum signals found.")
    
    with tab2:
        st.subheader("Trending Stocks (ADX & EMA)")
        with st.spinner("Scanning for trends..."):
            data = run_scanner(instrument_df, "Trend", holdings_df)
            st.dataframe(data, use_container_width=True, hide_index=True) if not data.empty else st.info("No stocks with strong trend signals found.")
    
    with tab3:
        st.subheader("Breakout Candidates (52-Week High)")
        with st.spinner("Scanning for breakouts..."):
            data = run_scanner(instrument_df, "Breakout", holdings_df)
            st.dataframe(data, use_container_width=True, hide_index=True) if not data.empty else st.info("No stocks nearing their 52-week high found.")

def calculate_strategy_pnl(legs, underlying_ltp):
    """Calculates the P&L for a given options strategy."""
    import pandas as pd
    import numpy as np
    if not legs: return pd.DataFrame(), 0, 0, []

    price_range = np.linspace(underlying_ltp * 0.8, underlying_ltp * 1.2, 100)
    pnl_df = pd.DataFrame(index=price_range); pnl_df.index.name = "Price at Expiry"
    
    for i, leg in enumerate(legs):
        pnl = np.maximum(0, price_range - leg['strike']) - leg['premium'] if leg['type'] == 'Call' else np.maximum(0, leg['strike'] - price_range) - leg['premium']
        if leg['position'] == 'Sell': pnl = -pnl
        pnl_df[f'Leg_{i+1}'] = pnl * leg['quantity']
    
    pnl_df['Total P&L'] = pnl_df.sum(axis=1)
    
    max_profit, max_loss = pnl_df['Total P&L'].max(), pnl_df['Total P&L'].min()
    
    sign_changes = np.where(np.diff(np.sign(pnl_df['Total P&L'])))[0]
    breakevens = [pnl_df.index[idx] for idx in sign_changes]

    return pnl_df, max_profit, max_loss, breakevens

def page_option_strategy_builder():
    """Option Strategy Builder page with live data and P&L calculation."""
    import pandas as pd
    import plotly.graph_objects as go
    display_header()
    st.title("Options Strategy Builder")
    
    instrument_df = get_instrument_df()
    client = get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to build strategies.")
        return
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Strategy Configuration")
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
        _, _, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        
        if not available_expiries:
            st.error(f"No options available for {underlying}."); st.stop()
            
        expiry_date = st.selectbox("Expiry", [e.strftime("%d %b %Y") for e in available_expiries])
        
        with st.form("add_leg_form"):
            st.write("**Add a New Leg**")
            pos, o_type = st.columns(2)
            position = pos.selectbox("Position", ["Buy", "Sell"])
            option_type = o_type.selectbox("Type", ["Call", "Put"])
            
            expiry_dt = datetime.strptime(expiry_date, "%d %b %Y").date()
            options = instrument_df[(instrument_df['name'] == underlying) & (instrument_df['expiry'].dt.date == expiry_dt) & (instrument_df['instrument_type'] == option_type[0])]
            
            if not options.empty:
                s, q = st.columns(2)
                strikes = sorted(options['strike'].unique())
                strike = s.selectbox("Strike", strikes, index=len(strikes)//2)
                quantity = q.number_input("Lots", 1, value=1)
                
                if st.form_submit_button("Add Leg"):
                    lot_size = options.iloc[0]['lot_size']
                    tradingsymbol = options[options['strike'] == strike].iloc[0]['tradingsymbol']
                    try:
                        premium = client.quote(f"NFO:{tradingsymbol}")[f"NFO:{tradingsymbol}"]['last_price']
                        st.session_state.strategy_legs.append({'symbol': tradingsymbol, 'position': position, 'type': option_type, 'strike': strike, 'quantity': quantity * lot_size, 'premium': premium})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not fetch premium: {e}")
            else:
                st.warning("No strikes found for selected expiry/type.")

        st.subheader("Current Legs")
        if st.session_state.strategy_legs:
            for i, leg in enumerate(st.session_state.strategy_legs):
                st.text(f"{i+1}: {leg['position']} {leg['quantity']} {leg['symbol']} @ ₹{leg['premium']:.2f}")
            if st.button("Clear All Legs"):
                st.session_state.strategy_legs = []; st.rerun()
        else:
            st.info("Add legs to your strategy.")
            
    with col2:
        st.subheader("Strategy Payoff Analysis")
        if st.session_state.strategy_legs:
            pnl_df, max_profit, max_loss, breakevens = calculate_strategy_pnl(st.session_state.strategy_legs, underlying_ltp)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pnl_df.index, y=pnl_df['Total P&L'], mode='lines', name='P&L'))
            fig.add_hline(y=0, line_dash="dash", line_color="gray"); fig.add_vline(x=underlying_ltp, line_dash="dot", line_color="yellow", annotation_text="Current LTP")
            template = 'plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
            fig.update_layout(title="Strategy P&L Payoff Chart", xaxis_title="Price at Expiry", yaxis_title="Profit / Loss (₹)", template=template)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk & Reward Profile")
            c1, c2 = st.columns(2)
            c1.metric("Max Profit", f"₹{max_profit:,.2f}"); c1.metric("Max Loss", f"₹{max_loss:,.2f}")
            c2.metric("Breakeven(s)", ", ".join([f"₹{b:,.2f}" for b in breakevens]) if breakevens else "N/A")
        else:
            st.info("Add legs to see the payoff analysis.")

def get_futures_contracts(instrument_df, underlying, exchange):
    """Fetches and sorts futures contracts."""
    import pandas as pd
    if instrument_df.empty or not underlying: return pd.DataFrame()
    futures_df = instrument_df[(instrument_df['name'] == underlying) & (instrument_df['instrument_type'] == 'FUT') & (instrument_df['exchange'] == exchange)].copy()
    if not futures_df.empty:
        futures_df['expiry'] = pd.to_datetime(futures_df['expiry'])
        return futures_df.sort_values('expiry')
    return pd.DataFrame()

def page_futures_terminal():
    """Futures Terminal page with live data."""
    import pandas as pd
    display_header()
    st.title("Futures Terminal")
    
    instrument_df, client = get_instrument_df(), get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to access futures data.")
        return
    
    exchange_options = sorted(instrument_df[instrument_df['instrument_type'] == 'FUT']['exchange'].unique())
    if not exchange_options:
        st.warning("No futures contracts found."); return
    
    c1, c2 = st.columns(2)
    selected_exchange = c1.selectbox("Select Exchange", exchange_options, index=exchange_options.index('NFO') if 'NFO' in exchange_options else 0)
    underlyings = sorted(instrument_df[(instrument_df['instrument_type'] == 'FUT') & (instrument_df['exchange'] == selected_exchange)]['name'].unique())
    if not underlyings:
        st.warning(f"No futures underlyings found for {selected_exchange}."); return
    selected_underlying = c2.selectbox("Select Underlying", underlyings)

    tab1, tab2 = st.tabs(["Live Futures Contracts", "Futures Calendar"])
    with tab1:
        st.subheader(f"Live Contracts for {selected_underlying}")
        futures_contracts = get_futures_contracts(instrument_df, selected_underlying, selected_exchange)
        
        if not futures_contracts.empty:
            symbols = [f"{row['exchange']}:{row['tradingsymbol']}" for _, row in futures_contracts.iterrows()]
            try:
                quotes = client.quote(symbols)
                live_data = []
                for key, data in quotes.items():
                    if data:
                        prev_close, last_price = data.get('ohlc', {}).get('close', 0), data.get('last_price', 0)
                        change = last_price - prev_close
                        pct_change = (change / prev_close * 100) if prev_close != 0 else 0
                        live_data.append({'Contract': data.get('tradingsymbol', key.split(':')[-1]), 'LTP': last_price, 'Change': change, '% Change': pct_change, 'Volume': data.get('volume', 0), 'OI': data.get('oi', 0)})
                st.dataframe(pd.DataFrame(live_data), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not fetch live futures data: {e}")
        else:
            st.info(f"No active futures contracts found for {selected_underlying}.")
    
    with tab2:
        st.subheader("Futures Expiry Calendar")
        futures_contracts = get_futures_contracts(instrument_df, selected_underlying, selected_exchange)
        if not futures_contracts.empty:
            calendar_df = futures_contracts[['tradingsymbol', 'expiry']].copy()
            calendar_df['expiry'] = pd.to_datetime(calendar_df['expiry'])
            calendar_df['Days to Expiry'] = (calendar_df['expiry'] - pd.to_datetime('today')).dt.days
            st.dataframe(calendar_df.rename(columns={'tradingsymbol': 'Contract', 'expiry': 'Expiry Date'}), use_container_width=True, hide_index=True)

def generate_ai_trade_idea(instrument_df, active_list):
    """Dynamically generates a trade idea based on watchlist signals."""
    import pandas as pd
    if not active_list or instrument_df.empty: return None

    discovery_results = {}
    for item in active_list:
        token = get_instrument_token(item['symbol'], instrument_df, exchange=item['exchange'])
        if token:
            data = get_historical_data(token, 'day', period='6mo')
            if not data.empty:
                interpretation = interpret_indicators(data)
                signals = [v for k, v in interpretation.items() if "Bullish" in v or "Bearish" in v]
                if signals: discovery_results[item['symbol']] = {'signals': signals, 'data': data}
    
    if not discovery_results: return None

    best_ticker = max(discovery_results, key=lambda k: len(discovery_results[k]['signals']))
    ticker_data = discovery_results[best_ticker]['data']
    ltp = ticker_data['close'].iloc[-1]
    atr_col = next((c for c in ticker_data.columns if 'atr' in c), None) 
    if not atr_col or pd.isna(ticker_data[atr_col].iloc[-1]): return None 
    atr = ticker_data[atr_col].iloc[-1]
    
    is_bullish = any("Bullish" in s for s in discovery_results[best_ticker]['signals'])
    narrative = f"**{best_ticker}** shows a confluence of {'bullish' if is_bullish else 'bearish'} signals: {', '.join(discovery_results[best_ticker]['signals'])}. "

    if is_bullish:
        narrative += "A move above resistance could trigger upside."
        entry, target, stop_loss = ltp, ltp + (2 * atr), ltp - (1.5 * atr)
        title = f"High-Conviction Long Setup: {best_ticker}"
    else: # Bearish
        narrative += "A break below support could lead to downside."
        entry, target, stop_loss = ltp, ltp - (2 * atr), ltp + (1.5 * atr)
        title = f"High-Conviction Short Setup: {best_ticker}"

    return {"title": title, "entry": entry, "target": target, "stop_loss": stop_loss, "narrative": narrative}

def page_ai_discovery():
    """AI-driven discovery engine with real data analysis."""
    display_header()
    st.title("AI Discovery Engine")
    st.info("Discovers technical patterns from your active watchlist and suggests high-conviction trade setups.", icon="🧠")
    
    active_list = st.session_state.watchlists.get(st.session_state.active_watchlist, [])
    instrument_df = get_instrument_df()

    if not active_list or instrument_df.empty:
        st.warning("Please set up your watchlist on the Dashboard to enable AI Discovery.")
        return

    st.markdown("---")
    st.subheader("Automated Pattern Discovery")
    with st.spinner("Analyzing your watchlist for technical signals..."):
        discovery_results = {}
        for item in active_list:
            token = get_instrument_token(item['symbol'], instrument_df, exchange=item['exchange'])
            if token:
                data = get_historical_data(token, 'day', period='6mo')
                if not data.empty:
                    interpretation = interpret_indicators(data)
                    signals = [f"{k}: {v}" for k, v in interpretation.items() if "Bullish" in v or "Bearish" in v]
                    if signals: discovery_results[item['symbol']] = signals
    
    if discovery_results:
        for ticker, signals in discovery_results.items():
            st.markdown(f"**Potential Signals for {ticker}:** " + ", ".join(signals))
    else:
        st.info("No significant technical patterns found in your watchlist.")
        
    st.markdown("---")
    st.subheader("AI-Powered Trade Idea")
    with st.spinner("Generating a high-conviction trade idea..."):
        trade_idea = generate_ai_trade_idea(instrument_df, active_list)

    if trade_idea:
        cols = st.columns(3)
        cols[0].metric("Entry Price", f"≈ ₹{trade_idea['entry']:.2f}")
        cols[1].metric("Target Price", f"₹{trade_idea['target']:.2f}")
        cols[2].metric("Stop Loss", f"₹{trade_idea['stop_loss']:.2f}")
        
        st.markdown(f"""
        <div class="trade-card" style="border-left-color: {'#28a745' if 'Long' in trade_idea['title'] else '#FF4B4B'};">
            <h4>{trade_idea['title']}</h4>
            <p><strong>Narrative:</strong> {trade_idea['narrative']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Could not generate a high-conviction trade idea from the current signals.")

def page_greeks_calculator():
    """Calculates Greeks for any option contract."""
    display_header()
    st.title("F&O Greeks Calculator")
    st.info("Calculate the theoretical value and greeks for any option contract.")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use this feature.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Option Details")
        underlying_price = st.number_input("Underlying Price", 0.01, value=23500.0)
        strike_price = st.number_input("Strike Price", 0.01, value=23500.0)
        days_to_expiry = st.number_input("Days to Expiry", 1, value=30)
        risk_free_rate = st.number_input("Risk-free Rate (%)", 0.0, value=7.0)
        volatility = st.number_input("Volatility (%)", 0.1, value=20.0)
        option_type = st.selectbox("Option Type", ["call", "put"])
        
        if st.button("Calculate Greeks"):
            T, r, sigma = days_to_expiry / 365.0, risk_free_rate / 100.0, volatility / 100.0
            st.session_state.calculated_greeks = black_scholes(underlying_price, strike_price, T, r, sigma, option_type)
            st.rerun()
    
    with col2:
        st.subheader("Greeks Results")
        if 'calculated_greeks' in st.session_state and st.session_state.calculated_greeks:
            greeks = st.session_state.calculated_greeks
            st.metric("Option Price", f"₹{greeks['price']:.2f}")
            c1, c2 = st.columns(2)
            c1.metric("Delta", f"{greeks['delta']:.4f}"); c1.metric("Gamma", f"{greeks['gamma']:.4f}"); c1.metric("Vega", f"{greeks['vega']:.4f}")
            c2.metric("Theta", f"{greeks['theta']:.4f}"); c2.metric("Rho", f"{greeks['rho']:.4f}")
            
            with st.expander("Understanding Greeks"):
                st.markdown("- **Delta**: Price sensitivity to underlying movement\n- **Gamma**: Rate of change of Delta\n- **Vega**: Sensitivity to volatility\n- **Theta**: Time decay\n- **Rho**: Sensitivity to interest rates")
        else:
            st.info("Enter option details and click 'Calculate Greeks'.")

def page_economic_calendar():
    """Economic Calendar page for Indian market events."""
    import pandas as pd
    display_header()
    st.title("Economic Calendar")
    st.info("Upcoming economic events for the Indian market, updated until October 2025.")

    events = {'Date': ['2025-10-10', '2025-10-14', '2025-10-15', '2025-10-17', '2025-10-24', '2025-10-31', '2025-10-31'],
              'Time': ['05:00 PM', '12:00 PM', '05:30 PM', '05:00 PM', '05:00 PM', '05:30 PM', '05:00 PM'],
              'Event Name': ['Foreign Exchange Reserves', 'WPI Inflation YoY (SEP)', 'CPI Inflation YoY (SEP)', 'Foreign Exchange Reserves', 'Foreign Exchange Reserves', 'Fiscal Deficit (SEP)', 'Foreign Exchange Reserves'],
              'Impact': ['Low', 'High', 'High', 'Low', 'Low', 'Medium', 'Low'],
              'Previous': ['$703.1B', '0.3%', '5.1%', '$704.5B', '$705.2B', '-4684.2B INR', '$705.9B'],
              'Forecast': ['-', '0.5%', '5.3%', '-', '-', '-5100.0B INR', '-']}
    calendar_df = pd.DataFrame(events)
    st.dataframe(calendar_df, use_container_width=True, hide_index=True)

# ============ 5.5 HFT TERMINAL PAGE ============
def page_hft_terminal():
    """A dedicated terminal for High-Frequency Trading."""
    import pandas as pd
    import random
    display_header()
    st.title("HFT Terminal (High-Frequency Trading)")
    st.info("Simulated high-speed view of market depth and one-click trading for liquid F&O instruments.", icon="⚡️")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.warning("Please connect to a broker to use the HFT Terminal."); return

    top_cols = st.columns([2, 1, 1, 1])
    symbol = top_cols[0].text_input("Instrument Symbol", "NIFTY24OCTFUT", key="hft_symbol").upper()
    
    instrument_info = instrument_df[instrument_df['tradingsymbol'] == symbol]
    if instrument_info.empty:
        st.error(f"Instrument '{symbol}' not found."); return
    
    exchange, token = instrument_info.iloc[0]['exchange'], instrument_info.iloc[0]['instrument_token']
    quote_data, depth_data = get_watchlist_data([{'symbol': symbol, 'exchange': exchange}]), get_market_depth(token)

    if not quote_data.empty:
        ltp, change = quote_data.iloc[0]['Price'], quote_data.iloc[0]['Change']
        tick_direction = "tick-up" if ltp > st.session_state.hft_last_price else "tick-down" if ltp < st.session_state.hft_last_price else ""
        top_cols[1].markdown(f"##### LTP: <span class='{tick_direction}' style='font-size: 1.2em;'>₹{ltp:,.2f}</span>", unsafe_allow_html=True)
        top_cols[2].markdown(f"##### Change: <span style='color:{'var(--green)' if change > 0 else 'var(--red)'}; font-size: 1.2em;'>{change:,.2f}</span>", unsafe_allow_html=True)
        top_cols[3].metric("Latency (ms)", f"{random.uniform(20, 80):.2f}")

        if ltp != st.session_state.hft_last_price and st.session_state.hft_last_price != 0:
            log_entry = {"time": datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S.%f")[:-3], "price": ltp, "change": ltp - st.session_state.hft_last_price}
            st.session_state.hft_tick_log.insert(0, log_entry)
            st.session_state.hft_tick_log = st.session_state.hft_tick_log[:20]
        st.session_state.hft_last_price = ltp

    st.markdown("---")
    main_cols = st.columns([1, 1, 1], gap="large")

    with main_cols[0]:
        st.subheader("Market Depth")
        if depth_data and depth_data.get('buy') and depth_data.get('sell'):
            bids, asks = pd.DataFrame(depth_data['buy']).sort_values('price', ascending=False).head(5), pd.DataFrame(depth_data['sell']).sort_values('price', ascending=True).head(5)
            st.write("**Bids (Buyers)**"); [st.markdown(f"<div class='hft-depth-bid'>{row['quantity']} @ **{row['price']:.2f}** ({row['orders']})</div>", unsafe_allow_html=True) for _, row in bids.iterrows()]
            st.write("**Asks (Sellers)**"); [st.markdown(f"<div class='hft-depth-ask'>({row['orders']}) **{row['price']:.2f}** @ {row['quantity']}</div>", unsafe_allow_html=True) for _, row in asks.iterrows()]
        else:
            st.info("Waiting for market depth data...")

    with main_cols[1]:
        st.subheader("One-Click Execution")
        quantity = st.number_input("Order Qty", 1, value=instrument_info.iloc[0]['lot_size'], step=instrument_info.iloc[0]['lot_size'], key="hft_qty")
        
        c1, c2 = st.columns(2)
        if c1.button("MARKET BUY", use_container_width=True, type="primary"): place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS')
        if c2.button("MARKET SELL", use_container_width=True): place_order(instrument_df, symbol, quantity, 'MARKET', 'SELL', 'MIS')
        
        st.markdown("---"); st.subheader("Manual Order")
        price = st.number_input("Limit Price", 0.01, step=0.05, key="hft_limit_price")
        c3, c4 = st.columns(2)
        if c3.button("LIMIT BUY", use_container_width=True): place_order(instrument_df, symbol, quantity, 'LIMIT', 'BUY', 'MIS', price=price)
        if c4.button("LIMIT SELL", use_container_width=True): place_order(instrument_df, symbol, quantity, 'LIMIT', 'SELL', 'MIS', price=price)

    with main_cols[2]:
        st.subheader("Tick Log")
        log_container = st.container(height=400)
        for entry in st.session_state.hft_tick_log:
            log_container.markdown(f"<small>{entry['time']}</small> - **{entry['price']:.2f}** <span style='color:{'var(--green)' if entry['change'] > 0 else 'var(--red)'};'>({entry['change']:+.2f})</span>", unsafe_allow_html=True)

# ================ 5.6 ALGO BOTS PAGE AND HELPERS ================
def rsi_signal(df, period=14, overbought=70, oversold=30):
    import pandas_ta as ta
    df.ta.rsi(length=period, append=True); rsi_col = f'RSI_{period}'
    if rsi_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_rsi, prev_rsi = df[rsi_col].iloc[-1], df[rsi_col].iloc[-2]
    if prev_rsi < oversold and last_rsi >= oversold: return 'BUY', f"RSI crossed above {oversold}."
    if prev_rsi > overbought and last_rsi <= overbought: return 'SELL', f"RSI crossed below {overbought}."
    return 'HOLD', f"RSI is neutral at {last_rsi:.2f}."

def macd_signal(df, fast=12, slow=26, signal=9):
    import pandas_ta as ta
    df.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
    macd_col, signal_col = f'MACD_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}'
    if macd_col not in df.columns or signal_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
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
    if prev_close <= prev_st and last_close > last_st: return 'BUY', f"Price crossed above Supertrend."
    if prev_close >= prev_st and last_close < last_st: return 'SELL', f"Price crossed below Supertrend."
    return 'HOLD', f"Price is {'above' if last_close > last_st else 'below'} Supertrend."

def ema_crossover_signal(df, fast_period=50, slow_period=200):
    import pandas_ta as ta
    df.ta.ema(length=fast_period, append=True); df.ta.ema(length=slow_period, append=True)
    fast_ema_col, slow_ema_col = f'EMA_{fast_period}', f'EMA_{slow_period}'
    if fast_ema_col not in df.columns or slow_ema_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_fast, prev_fast = df[fast_ema_col].iloc[-1], df[fast_ema_col].iloc[-2]
    last_slow, prev_slow = df[slow_ema_col].iloc[-1], df[slow_ema_col].iloc[-2]
    if prev_fast < prev_slow and last_fast >= last_slow: return 'BUY', f"Golden Cross: EMA({fast_period}) above EMA({slow_period})."
    if prev_fast > prev_slow and last_fast <= last_slow: return 'SELL', f"Death Cross: EMA({fast_period}) below EMA({slow_period})."
    return 'HOLD', "No EMA crossover."
    
def bollinger_band_signal(df, period=20, std=2):
    import pandas_ta as ta
    df.ta.bbands(length=period, std=std, append=True)
    lower_col, upper_col = f'BBL_{period}_{float(std)}', f'BBU_{period}_{float(std)}'
    if lower_col not in df.columns or upper_col not in df.columns or len(df) < 2: return 'HOLD', "Insufficient data."
    last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
    last_lower, last_upper = df[lower_col].iloc[-1], df[upper_col].iloc[-1]
    if prev_close > last_lower and last_close <= last_lower: return 'BUY', "Price crossed below lower band."
    if prev_close < last_upper and last_close >= last_upper: return 'SELL', "Price crossed above upper band."
    return 'HOLD', "Price is within bands."
    
def vwap_scalper_signal(df):
    import pandas_ta as ta
    df.ta.vwap(append=True)
    vwap_col = df.columns[df.columns.str.contains('VWAP')][0] if not df.columns[df.columns.str.contains('VWAP')].empty else None
    if not vwap_col or len(df) < 2: return 'HOLD', "Insufficient data."
    last_close, prev_close, last_vwap = df['close'].iloc[-1], df['close'].iloc[-2], df[vwap_col].iloc[-1]
    if prev_close < last_vwap and last_close >= last_vwap: return 'BUY', "Price crossed above VWAP."
    if prev_close > last_vwap and last_close <= last_vwap: return 'SELL', "Price crossed below VWAP."
    return 'HOLD', "No VWAP crossover."

def page_algo_bots():
    """A dedicated page for semi-automated and fully-automated trading bots."""
    display_header()
    st.title("Algo Trading Bots")
    
    instrument_df, client = get_instrument_df(), get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to run the Algo Bots."); return

    BOTS = {
        "RSI Momentum": {"desc": "Trades on momentum shifts (overbought/oversold).", "func": rsi_signal, "params": {"period": 14, "overbought": 70, "oversold": 30}, "interval": "day"},
        "MACD Crossover": {"desc": "Trend-following strategy based on MACD line crossovers.", "func": macd_signal, "params": {"fast": 12, "slow": 26, "signal": 9}, "interval": "day"},
        "Supertrend Rider": {"desc": "Follows the prevailing trend and trades on reversals.", "func": supertrend_signal, "params": {"period": 7, "multiplier": 3.0}, "interval": "day"},
        "EMA Crossover": {"desc": "Trades on 'Golden Cross' (bullish) and 'Death Cross' (bearish) signals.", "func": ema_crossover_signal, "params": {"fast_period": 50, "slow_period": 200}, "interval": "day"},
        "Bollinger Reversion": {"desc": "Mean-reversion strategy using Bollinger Bands.", "func": bollinger_band_signal, "params": {"period": 20, "std": 2.0}, "interval": "day"},
        "VWAP Scalper": {"desc": "Intraday strategy trading on price crossing the VWAP.", "func": vwap_scalper_signal, "params": {}, "interval": "minute"}
    }
    
    bot_tabs = st.tabs(list(BOTS.keys()))

    for i, (bot_name, bot_info) in enumerate(BOTS.items()):
        with bot_tabs[i]:
            st.subheader(bot_name); st.caption(bot_info['desc'])
            
            if bot_name not in st.session_state.algo_bots_status:
                st.session_state.algo_bots_status[bot_name] = {"status": "Stopped", "log": [], "position": "FLAT"}
            bot_state = st.session_state.algo_bots_status[bot_name]

            with st.form(key=f"form_{bot_name}"):
                c1, c2, c3 = st.columns(3)
                symbol = c1.text_input("Symbol", "RELIANCE", key=f"symbol_{bot_name}").upper()
                quantity = c2.number_input("Quantity", 1, key=f"qty_{bot_name}")
                mode = c3.radio("Mode", ["Semi-Auto", "Full-Auto"], key=f"mode_{bot_name}", horizontal=True, label_visibility="collapsed")
                params = bot_info['params'] # Simplified for this version
                
                is_running = bot_state["status"] == "Running"
                submit_text = "Get Signal" if mode == "Semi-Auto" else ("Stop Bot" if is_running else "Start Bot")
                submitted = st.form_submit_button(submit_text, use_container_width=True, type="primary" if not is_running else "secondary")

                if submitted:
                    if mode == "Semi-Auto":
                        with st.spinner(f"Fetching signal..."):
                            token = get_instrument_token(symbol, instrument_df)
                            data = get_historical_data(token, bot_info['interval'], '1y' if bot_info['interval'] == 'day' else '5d')
                            if not data.empty:
                                signal, reason = bot_info['func'](data.copy(), **params)
                                bot_state['last_signal'] = {"signal": signal, "reason": reason, "symbol": symbol, "quantity": quantity}
                            else: st.error("Could not fetch data.")
                    else: # Full-Auto
                        if is_running:
                            bot_state["status"] = "Stopped"; st.warning(f"Bot stopped."); st.rerun()
                        else:
                            bot_state["status"] = "Running"; bot_state["config"] = {"symbol": symbol, "quantity": quantity, "params": params, "interval": bot_info['interval'], "func": bot_info['func']}
                            st.success(f"Bot started."); st.rerun()

            st.markdown("---")
            if mode == "Semi-Auto" and "last_signal" in bot_state:
                st.subheader("Live Signal")
                signal_info = bot_state['last_signal']
                if signal_info['signal'] == 'BUY': st.success(f"**BUY Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                elif signal_info['signal'] == 'SELL': st.error(f"**SELL Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                else: st.info(f"**HOLD Signal for {signal_info['symbol']}:** {signal_info['reason']}")
                if signal_info['signal'] in ['BUY', 'SELL'] and st.button(f"Execute {signal_info['signal']} Order", key=f"exec_{bot_name}"):
                    place_order(instrument_df, signal_info['symbol'], signal_info['quantity'], "MARKET", signal_info['signal'], "MIS")

            elif mode == "Full-Auto":
                st.subheader("Bot Status & Log")
                st.markdown(f"**Status:** <span style='color:{'green' if bot_state['status'] == 'Running' else 'orange'};'>{bot_state['status']}</span>", unsafe_allow_html=True)
                log_container = st.container(height=200)
                for log_entry in reversed(bot_state.get('log', [])): log_container.code(log_entry)
    
    # --- Fully-Automated Runner Logic (runs on every script rerun) ---
    for bot_name, bot_state in st.session_state.algo_bots_status.items():
        if bot_state.get("status") == "Running":
            st.toast(f"Checking signal for {bot_name}...")
            config = bot_state['config']
            token = get_instrument_token(config['symbol'], instrument_df)
            data = get_historical_data(token, config['interval'], '1y' if config['interval'] == 'day' else '5d')
            if not data.empty:
                signal, reason = config['func'](data.copy(), **config['params'])
                if signal == 'BUY' and bot_state.get('position') == 'FLAT':
                    place_order(instrument_df, config['symbol'], config['quantity'], 'MARKET', 'BUY', 'MIS')
                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTED BUY: {config['quantity']} of {config['symbol']}. Reason: {reason}"
                    bot_state['log'].append(log_msg); bot_state['position'] = 'LONG'
                    st.toast(f"✅ {bot_name} executed BUY!", icon="🤖")
                elif signal == 'SELL' and bot_state.get('position') == 'LONG':
                    place_order(instrument_df, config['symbol'], config['quantity'], 'MARKET', 'SELL', 'MIS')
                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTED SELL (Exit): {config['quantity']} of {config['symbol']}. Reason: {reason}"
                    bot_state['log'].append(log_msg); bot_state['position'] = 'FLAT'
                    st.toast(f"✅ {bot_name} exited position!", icon="🤖")

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
        api_key, api_secret = st.secrets.get("ZERODHA_API_KEY"), st.secrets.get("ZERODHA_API_SECRET")
        if not api_key or not api_secret:
            st.error("Kite API credentials not found in Streamlit secrets."); st.stop()
            
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
    from streamlit_autorefresh import st_autorefresh

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
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresher")
    
    pages[st.session_state.terminal_mode][selection]()

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
