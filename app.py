import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect

# --- Custom THEMING ---
def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v3fvcr {
            background-color: #121212 !important;
        }
        .stSidebar, .css-15zrgzn {
            background: #191919 !important;
        }
        .stDataFrame tbody tr {background-color: #191919 !important;color: #00FF99 !important;}
        .css-pkbazv, .stTextInput, .stTextArea textarea, .stNumberInput input, .st-cq, .st-de {
            background: #191919 !important;
            color: #00FF99 !important;
        }
        .stMetric, .stMetricLabel, .stMetricValue {color: #00FF99 !important;}
        h1, h2, h3, h4, h5, h6, label, .st-bv, .stTextInput label, .stTextArea label {color: #00FF99 !important;}
        </style>
        """, unsafe_allow_html=True)
set_terminal_style()

# ---- Smallcase-Like Baskets ----
SMALLCASE_BASKETS = {
    "Precious Metals": ["HINDZINC", "NMDC", "VEDL", "MOIL", "RSWM"],
    "Gold Related": ["MANAPPURAM", "MUTHOOTFIN", "MMTC"],
    "Top Equity": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Banking": ["KOTAKBANK", "SBIN", "AXISBANK", "ICICIBANK"],
    "FMCG": ["HINDUNILVR", "NESTLEIND", "ITC", "BRITANNIA"],
}

# ---- ZERODHA INIT ----
api_key = st.secrets["ZERODHA_API_KEY"]
access_token = st.secrets["ZERODHA_ACCESS_TOKEN"]
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# ---- Utility Functions ----
def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        return f"Error: {e}"

@st.cache_data()
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval)
    if len(data) == 0:
        return None
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'])
    data = pd.concat([data, macd], axis=1)
    data['SMA21'] = ta.sma(data['Close'], length=21)
    data['EMA9'] = ta.ema(data['Close'], length=9)
    bbands = ta.bbands(data['Close'], length=20)
    data['BOLL_L'], data['BOLL_M'], data['BOLL_U'] = bbands['BBL_20_2.0'], bbands['BBM_20_2.0'], bbands['BBU_20_2.0']
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'], length=14)["ADX_14"]
    data['STOCHRSI'] = ta.stochrsi(data['Close'], length=14)["STOCHRSIk_14_14_3_3"]
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
    if not supertrend.empty:
        data = pd.concat([data, supertrend], axis=1)
    data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
    # Heikin Ashi
    ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
    for c in ha.columns:
        data[f'HA_{c}'] = ha[c]
    return data

def get_signals(data):
    latest = data.iloc[-1]
    signals = {}
    signals['RSI Signal'] = 'Overbought' if latest['RSI'] > 70 else ('Oversold' if latest['RSI'] < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] else (
        'Bearish' if latest['MACD_12_26_9'] < latest['MACDs_12_26_9'] else 'Neutral')
    signals['Supertrend'] = 'Bullish' if latest.get('SUPERT_7_3.0') and latest['SUPERT_7_3.0'] < latest['Close'] else (
        'Bearish' if latest.get('SUPERT_7_3.0') and latest['SUPERT_7_3.0'] > latest['Close'] else 'Neutral')
    signals['ADX Trend'] = 'Strong' if latest['ADX'] > 25 else 'Weak'
    signals['STOCHRSI Signal'] = 'Overbought' if latest['STOCHRSI'] > 0.8 else (
        'Oversold' if latest['STOCHRSI'] < 0.2 else 'Neutral')
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
                "MACD": float(latest['MACD_12_26_9']),
                "ADX": float(latest['ADX']),
                "ATR": float(latest['ATR']),
                "Signal": signals['RSI Signal'] + "/" + signals['MACD Signal'] + "/" + signals['Supertrend'],
            }
            screener_data.append(row)
    return pd.DataFrame(screener_data)

# ---- SIDEBAR: Theme Toggle ----
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True
theme_choice = st.sidebar.selectbox("Terminal Theme", ["Black/Green", "Streamlit Default"])
if theme_choice == "Black/Green" and not st.session_state.dark_theme:
    set_terminal_style(True)
    st.session_state.dark_theme = True
elif theme_choice != "Black/Green" and st.session_state.dark_theme:
    set_terminal_style(False)
    st.session_state.dark_theme = False

# ---- SIDEBAR: Fast Refresh ----
if st.sidebar.button("🔄 Refresh All"):
    st.experimental_rerun()

# ---- SIDEBAR: Watchlist P&L Tracker ----
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
        if isinstance(live, str):
            live = float(fetch_stock_data(s, "1d", "5m")["Close"][-1]) # fallback
        pnl = (live - entry_prices[i]) * quantities[i]
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": live, "Qty": quantities[i], "P&L ₹": round(pnl,2)})
    except Exception as e:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
if pnl_data:
    st.sidebar.dataframe(pd.DataFrame(pnl_data))
    total_pnl = sum(x["P&L ₹"] for x in pnl_data if isinstance(x["P&L ₹"], (int,float)))
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl,2)}</b>", unsafe_allow_html=True)

# ---- SIDEBAR: Screener Selection ----
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

# ---- SIDEBAR: Screener Output ----
screen_df = make_screener(stock_list, screen_period, screen_interval)
st.sidebar.subheader("Screener Results")
if len(screen_df):
    st.sidebar.dataframe(screen_df)
    # Advanced: Download
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener to CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---- SIDEBAR: Order Placement ----
st.sidebar.subheader("Order Placement (Kite)")
trade_type = st.sidebar.selectbox("Type", ['BUY','SELL'])
order_qty = st.sidebar.number_input("Quantity", value=1, step=1, min_value=1)
order_type = st.sidebar.selectbox("Order Type",['MARKET', 'LIMIT'])
limit_price = None
if order_type == 'LIMIT':
    limit_price = st.sidebar.number_input("Limit Price", value=float(screen_df.iloc[0]['LTP']) if len(screen_df) else 0)
symbol_for_order = stock_list[0]
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

# ---- SIDEBAR: Alerts (Single symbol) ----
st.sidebar.subheader("Simple Price Alert")
alert_price = st.sidebar.number_input('Alert above price', value=float(screen_df.iloc[0]['LTP']) if len(screen_df) else 0)
if st.sidebar.button("Set/Check Alert") and len(screen_df):
    curr_price = get_live_price(stock_list[0])
    if curr_price != 'Error' and float(curr_price) > alert_price:
        st.sidebar.warning(f"ALERT: {stock_list[0]} > {alert_price}")

# ---- Main UI: All in Tabs -------------------------------------------------
if len(stock_list):
    st.header(f"Live Technical Dashboard: {stock_list[0]}")
    data = fetch_stock_data(stock_list[0], screen_period, screen_interval)
    price = get_live_price(stock_list[0])
    st.metric(label="Current LTP", value=price)
    tabs = st.tabs(["Price Chart", "TA Indicators", "Advanced", "Raw Data"])
    # ---- Chart Tab ----
    with tabs[0]:
        chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
        bands_show = st.checkbox("Show Bollinger Bands (Fill)", value=True)
        fig = go.Figure()
        if chart_style == "Heikin Ashi":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['HA_open'], high=data['HA_high'], low=data['HA_low'], close=data['HA_close'],
                name='Heikin Ashi'
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='Candles'))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA9'], line=dict(color='#00FF99', width=1), name='EMA 9'))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA21'], line=dict(color='#FFA500', width=1), name='SMA 21'))
        if bands_show:
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
        if 'SUPERT_7_3.0' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SUPERT_7_3.0'], line=dict(color='#fae900', width=2), name='Supertrend'))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    # ---- TA Tab ----
    with tabs[1]:
        st.line_chart(data[['RSI','ADX','STOCHRSI']].dropna())
        st.line_chart(data[['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']].dropna())
        st.line_chart(data['ATR'].dropna())
        st.write("Latest Values:", data.iloc[-1][['Close','RSI','ADX','STOCHRSI','ATR','VWAP']])

    # ---- Advanced Tab ----
    with tabs[2]:
        st.subheader("Signals (Current)")
        signals = get_signals(data)
        st.table(pd.DataFrame(signals.items(), columns=['Indicator', 'Signal']))
        # Download data
        csv = data.to_csv()
        st.download_button('Export Data to CSV', csv, file_name=f"{stock_list[0]}_{screen_interval}.csv")

    # ---- Raw Data Tab ----
    with tabs[3]:
        if st.checkbox("Show Table Data"):
            st.dataframe(data.tail(40))
