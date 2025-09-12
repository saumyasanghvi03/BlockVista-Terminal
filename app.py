import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

# ---- Custom Bloomberg Terminal Style ----
def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v3fvcr {background-color: #121212 !important;}
        .stSidebar, .css-15zrgzn {background: #191919 !important;}
        .stDataFrame tbody tr {background-color: #191919 !important;color: #00FF99 !important;}
        .css-pkbazv, .stTextInput, .stTextArea textarea, .stNumberInput input, .st-cq, .st-de {
            background: #191919 !important;color: #00FF99 !important;}
        .stMetric, .stMetricLabel, .stMetricValue {color: #00FF99 !important;}
        h1, h2, h3, h4, h5, h6, label, .st-bv, .stTextInput label, .stTextArea label {color: #00FF99 !important;}
        </style>
        """, unsafe_allow_html=True)

if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True

theme_choice = st.sidebar.selectbox("Terminal Theme", ["Black/Green", "Streamlit Default"])
if theme_choice == "Black/Green" and not st.session_state.dark_theme:
    set_terminal_style(True)
    st.session_state.dark_theme = True
elif theme_choice != "Black/Green" and st.session_state.dark_theme:
    set_terminal_style(False)
    st.session_state.dark_theme = False
elif theme_choice == "Black/Green":
    set_terminal_style(True)

# ---- AUTO REFRESH ----
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 15

refresh_col, toggle_col = st.sidebar.columns([2,2])
with refresh_col:
    st.session_state["auto_refresh"] = st.checkbox("Auto Refresh", value=st.session_state["auto_refresh"])
with toggle_col:
    st.session_state["refresh_interval"] = st.number_input("Sec", value=st.session_state["refresh_interval"], min_value=5, max_value=90, step=1)

if st.session_state["auto_refresh"]:
    st_autorefresh(interval=st.session_state["refresh_interval"] * 1000, key="autorefresh")

# ---- Smallcase-Like Baskets ----
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

# ---- Zerodha connection ----
api_key = st.secrets["ZERODHA_API_KEY"]
access_token = st.secrets["ZERODHA_ACCESS_TOKEN"]
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception:
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval)
    if len(data) == 0:
        return None
    # All indicators with robust guards
    data['RSI'] = ta.rsi(data['Close'], length=14) if len(data) > 0 else np.nan
    
    # MACD
    macd = ta.macd(data['Close'])
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            data[col] = macd[col] if col in macd else np.nan
    else:
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            data[col] = np.nan
    
    data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 0 else np.nan
    data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 0 else np.nan

    # BBands
    bbands = ta.bbands(data['Close'], length=20)
    for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
        data[label] = bbands[key] if isinstance(bbands, pd.DataFrame) and key in bbands else np.nan
    
    atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['ATR'] = atr if isinstance(atr, pd.Series) else np.nan

    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx else np.nan

    stochrsi = ta.stochrsi(data['Close'], length=14)
    data['STOCHRSI'] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) and "STOCHRSIk_14_14_3_3" in stochrsi else np.nan

    # Supertrend
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
    if isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
        for col in supertrend.columns:
            data[col] = supertrend[col]
    # VWAP
    data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume']) if len(data) > 0 else np.nan
    # Heikin-Ashi
    ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
    for c in ['open','high','low','close']:
        ha_key = f'HA_{c}'
        data[ha_key] = ha[ha_key] if isinstance(ha, pd.DataFrame) and ha_key in ha else np.nan
    return data

def get_signals(data):
    latest = data.iloc[-1]
    signals = {}
    signals['RSI Signal'] = 'Overbought' if latest['RSI'] > 70 else ('Oversold' if latest['RSI'] < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if latest.get('MACD_12_26_9',np.nan) > latest.get('MACDs_12_26_9',np.nan) else (
        'Bearish' if latest.get('MACD_12_26_9',np.nan) < latest.get('MACDs_12_26_9',np.nan) else 'Neutral')
    supertrend_col = [c for c in data.columns if c.startswith('SUPERT_') and not c.endswith('_dir')]
    if supertrend_col:
        signals['Supertrend'] = 'Bullish' if latest[supertrend_col[0]] < latest['Close'] else (
            'Bearish' if latest[supertrend_col[0]] > latest['Close'] else 'Neutral')
    else:
        signals['Supertrend'] = 'Unknown'
    signals['ADX Trend'] = 'Strong' if latest.get('ADX',np.nan) > 25 else 'Weak'
    signals['STOCHRSI Signal'] = 'Overbought' if latest.get('STOCHRSI',np.nan) > 0.8 else (
        'Oversold' if latest.get('STOCHRSI',np.nan) < 0.2 else 'Neutral')
    return signals

def make_screener(stock_list, period, interval):
    screener_data = []
    for s in stock_list:
        data = fetch_stock_data(s, period, interval)
        if data is not None:
            latest = data.iloc[-1]
            signals = get_signals(data)
            row = {
                "Symbol": s,
                "LTP": float(latest['Close']),
                "RSI": float(latest['RSI']),
                "MACD": float(latest.get('MACD_12_26_9', np.nan)),
                "ADX": float(latest.get('ADX', np.nan)),
                "ATR": float(latest['ATR']),
                "Signal": signals['RSI Signal'] + "/" + signals['MACD Signal'] + "/" + signals['Supertrend'],
            }
            screener_data.append(row)
    return pd.DataFrame(screener_data)

st.sidebar.subheader("📈 Watchlist P&L Tracker (Live)")
watchlist = st.sidebar.text_area("List NSE symbols (comma-separated)", value="RELIANCE, SBIN, TCS")
positions_input = st.sidebar.text_area("Entry prices (comma, same order)", value="2550, 610, 3580")
qty_input = st.sidebar.text_area("Quantities (comma, same order)", value="10, 20, 5")
symbols = [x.strip().upper() for x in watchlist.split(",") if x.strip()]
entry_prices = [float(x) for x in positions_input.split(",") if x.strip()]
quantities = [float(x) for x in qty_input.split(",") if x.strip()]
pnl_data = []
for i, s in enumerate(symbols):
    try:
        live = get_live_price(s)
        if isinstance(live, str) or live is None or np.isnan(live):
            d = fetch_stock_data(s, "1d", "5m")
            if d is not None and len(d):
                live = d["Close"][-1]
            else:
                live = np.nan
        pnl = (live - entry_prices[i]) * quantities[i]
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": live, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
    except Exception:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
if pnl_data:
    st.sidebar.dataframe(pd.DataFrame(pnl_data))
    total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int,float)))
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)

# ---- Sidebar: Screener ----
st.sidebar.title('Multi-Screener Settings')
screener_mode = st.sidebar.radio("Screener Mode", ["Single Stock", "Basket (Smallcase)"])
if screener_mode == "Single Stock":
    symbol = st.sidebar.text_input('NSE Symbol', value='RELIANCE')
    stock_list = [symbol]
else:
    basket = st.sidebar.selectbox("Pick Basket", list(SMALLCASE_BASKETS.keys()))
    stock_list = SMALLCASE_BASKETS[basket]
screen_period = st.sidebar.selectbox('Period', ['1d','5d'])
screen_interval = st.sidebar.selectbox('Interval', ['1m','5m','15m'])

screen_df = make_screener(stock_list, screen_period, screen_interval)
st.sidebar.subheader("Screener Results")
if len(screen_df):
    st.sidebar.dataframe(screen_df)
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---- Sidebar: Order Placement ----
st.sidebar.subheader("Order Placement (Kite)")
trade_type = st.sidebar.selectbox("Type", ['BUY','SELL'])
order_qty = st.sidebar.number_input("Quantity", value=1, step=1, min_value=1)
order_type = st.sidebar.selectbox("Order Type",['MARKET', 'LIMIT'])
limit_price = None
if order_type == 'LIMIT' and len(screen_df):
    limit_price = st.sidebar.number_input("Limit Price", value=float(screen_df.iloc[0]['LTP']))
elif order_type == 'LIMIT':
    limit_price = st.sidebar.number_input("Limit Price", value=0)
symbol_for_order = stock_list[0] if stock_list else ""
if st.sidebar.button("PLACE ORDER") and len(screen_df):
    try:
        placed_order = kite.place_order(
            tradingsymbol=symbol_for_order,
            exchange="NSE",
            transaction_type=trade_type,
            quantity=int(order_qty),
            order_type=order_type,
            price=limit_price if order_type == 'LIMIT' else None,
            variety="regular",
            product="CNC",
            validity="DAY"
        )
        st.sidebar.success(f"Order placed: {placed_order}")
    except Exception as e:
        st.sidebar.error(f"Order failed: {e}")

# ---- Sidebar: Alerts ----
st.sidebar.subheader("Simple Price Alert")
if len(screen_df):
    alert_price = st.sidebar.number_input('Alert above price', value=float(screen_df.iloc[0]['LTP']))
else:
    alert_price = st.sidebar.number_input('Alert above price', value=0.0)
if st.sidebar.button("Set/Check Alert") and len(screen_df):
    curr_price = get_live_price(stock_list[0])
    if curr_price != 'Error' and curr_price is not None and curr_price > alert_price:
        st.sidebar.warning(f"ALERT: {stock_list[0]} > {alert_price}")

# ---- Main UI: TABS ----
if len(stock_list):
    st.header(f"Live Technical Dashboard: {stock_list[0]}")
    data = fetch_stock_data(stock_list[0], screen_period, screen_interval)
    if data is None or not len(data):
        st.error("No data available for this symbol/interval.")
        st.stop()
    price = get_live_price(stock_list[0])
    st.metric(label="Current LTP", value=price)
    tabs = st.tabs(["Chart", "TA", "Advanced", "Raw"])
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands (Fill)", value=True)
        fig = go.Figure()
        if chart_style == "Heikin Ashi":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['HA_open'], high=data['HA_high'],
                low=data['HA_low'], close=data['HA_close'], name='Heikin Ashi'))
        else:
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='Candles'))
        if 'EMA9' in data:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['EMA9'], line=dict(color='#00FF99', width=1), name='EMA 9'))
        if 'SMA21' in data:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA21'], line=dict(color='#FFA500', width=1), name='SMA 21'))
        if bands_show and 'BOLL_U' in data and 'BOLL_L' in data:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BOLL_U'], line=dict(color='#2986cc', width=1), name='Boll U'))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BOLL_L'], line=dict(color='#cc2929', width=1), name='Boll L'))
            fig.add_trace(go.Scatter(
                x=list(data.index) + list(data.index[::-1]),
                y=list(data['BOLL_U']) + list(data['BOLL_L'])[::-1],
                fill="toself", fillcolor="rgba(41,134,204,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="BB Channel"))
        supertrend_col = [c for c in data.columns if c.startswith('SUPERT_') and not c.endswith('_dir')]
        if supertrend_col:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[supertrend_col[0]],
                line=dict(color='#fae900', width=2), name='Supertrend'))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        ta_cols = [c for c in ['RSI','ADX','STOCHRSI'] if c in data]
        if ta_cols:
            st.line_chart(data[ta_cols].dropna())
        macd_cols = [c for c in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'] if c in data]
        if macd_cols:
            st.line_chart(data[macd_cols].dropna())
        if 'ATR' in data:
            st.line_chart(data['ATR'].dropna())
        last_cols = [c for c in ['Close','RSI','ADX','STOCHRSI','ATR','VWAP'] if c in data]
        st.write("Latest Values:", data.iloc[-1][last_cols])
    with tabs[2]:
        st.subheader("Signals (Current)")
        signals = get_signals(data)
        st.table(pd.DataFrame(signals.items(), columns=['Indicator', 'Signal']))
        csv2 = data.to_csv()
        st.download_button('Export Data to CSV', csv2, file_name=f"{stock_list[0]}_{screen_interval}.csv")
    with tabs[3]:
        if st.checkbox("Show Table Data"):
            st.dataframe(data.tail(40))
