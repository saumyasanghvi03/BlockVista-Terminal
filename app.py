import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

# ---- Browser Notification Helper ----
def browser_notification(title, body, icon=None):
    icon_line = f'icon: "{icon}",' if icon else ""
    st.markdown(
        f"""
        <script>
        if (Notification.permission !== "granted") {{
            Notification.requestPermission();
        }}
        if (Notification.permission === "granted") {{
            new Notification("{title}", {{
            body: "{body}",
            {icon_line}
            }});
        }}
        </script>
        """, unsafe_allow_html=True
    )

# ---- Theming ----
st.markdown(
    """
    <div style='background:linear-gradient(90deg,#141e30,#243b55 60%,#FFD900 100%);
     padding:10px 24px 6px 24px;border-radius:8px;margin-bottom:18px;box-shadow:0 4px 10px #0007; '>
        <span style='color:#FFD900;font-family:monospace;font-size:2.1rem;font-weight:bold;vertical-align:middle;letter-spacing:2px;'>
        BLOCKVISTA TERMINAL</span>
        <span style='float:right;color:#30ff96;font-size:1.25rem;font-family:monospace;padding-top:16px;font-weight:bold;'>
        INDIA • INTRADAY • LIVE</span>
    </div>
    """, unsafe_allow_html=True
)
def set_terminal_style(custom_dark=True):
    if custom_dark:
        st.markdown("""
        <style>
        .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v3fvcr {background-color: #121212 !important;}
        .stSidebar, .css-15zrgzn {background: #191919 !important;}
        .stDataFrame tbody tr {background-color: #191919 !important;color: #FFD900 !important;}
        .css-pkbazv, .stTextInput, .stTextArea textarea, .stNumberInput input, .st-cq, .st-de {
            background: #191919 !important;color: #FFD900 !important;}
        .stMetric, .stMetricLabel, .stMetricValue {color: #FFD900 !important;}
        h1, h2, h3, h4, h5, h6, label, .st-bv, .stTextInput label, .stTextArea label {color: #FFD900 !important;}
        </style>
        """, unsafe_allow_html=True)
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True
theme_choice = st.sidebar.selectbox("Terminal Theme", ["Black/Yellow/Green", "Streamlit Default"])
if theme_choice == "Black/Yellow/Green" and not st.session_state.dark_theme:
    set_terminal_style(True)
    st.session_state.dark_theme = True
elif theme_choice != "Black/Yellow/Green" and st.session_state.dark_theme:
    set_terminal_style(False)
    st.session_state.dark_theme = False
elif theme_choice == "Black/Yellow/Green":
    set_terminal_style(True)

# ---- Auto Refresh ----
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

# ---- Zerodha Login ----
api_key = st.secrets["ZERODHA_API_KEY"]
api_secret = st.secrets["ZERODHA_API_SECRET"]
if "access_token" not in st.session_state:
    kite_tmp = KiteConnect(api_key=api_key)
    login_url = kite_tmp.login_url()
    st.markdown(
        f"""
        <div style="background: #f5da82; padding: 14px; border-radius: 8px;">
        🟠 <a href="{login_url}" target="_blank"><b>Click here to login & authorize BlockVista</b></a><br>
        After logging in, get <b>`request_token=xxxx`</b> from the URL and paste below:
        </div>
        """, unsafe_allow_html=True
    )
    request_token = st.text_input("Paste request_token here:")
    if st.button("Generate Access Token") and request_token:
        try:
            data = kite_tmp.generate_session(request_token, api_secret=api_secret)
            st.session_state["access_token"] = data["access_token"]
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(data["access_token"])
            st.session_state["kite"] = kite
            st.success("✅ Zerodha session started! All Kite features enabled.")
        except Exception as ex:
            st.error(f"❌ Zerodha login failed: {ex}")
            browser_notification(
                "BlockVista Error",
                f"❌ Zerodha login failed: {ex}",
                "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
            )
            st.stop()
    st.stop()
else:
    if "kite" not in st.session_state:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state["access_token"])
        st.session_state["kite"] = kite
    else:
        kite = st.session_state["kite"]

def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        browser_notification(
            "BlockVista Error",
            f"❌ Live price fetch failed: {e}",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
        return np.nan

@st.cache_data(show_spinner="⏳ Loading data...")
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval)
    if len(data) == 0:
        return None
    data['RSI'] = ta.rsi(data['Close'], length=14) if len(data) > 0 else np.nan
    macd = ta.macd(data['Close'])
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            data[col] = macd[col] if col in macd else np.nan
    else:
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            data[col] = np.nan
    data['SMA21'] = ta.sma(data['Close'], length=21) if len(data) > 0 else np.nan
    data['EMA9'] = ta.ema(data['Close'], length=9) if len(data) > 0 else np.nan
    bbands = ta.bbands(data['Close'], length=20)
    for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
        data[label] = bbands[key] if isinstance(bbands, pd.DataFrame) and key in bbands else np.nan
    atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['ATR'] = atr if isinstance(atr, pd.Series) else np.nan
    adx = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    data['ADX'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx else np.nan
    stochrsi = ta.stochrsi(data['Close'], length=14)
    data['STOCHRSI'] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) and "STOCHRSIk_14_14_3_3" in stochrsi else np.nan
    supertrend = ta.supertrend(data['High'], data['Low'], data['Close'])
    if isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
        for col in supertrend.columns:
            data[col] = supertrend[col]
    data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume']) if len(data) > 0 else np.nan
    ha = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
    for c in ['open','high','low','close']:
        ha_key = f'HA_{c}'
        data[ha_key] = ha[ha_key] if isinstance(ha, pd.DataFrame) and ha_key in ha else np.nan
    return data

def try_scalar(val):
    if isinstance(val, pd.Series) and len(val) == 1:
        val = val.iloc[0]
    if isinstance(val, (float, int, np.floating, np.integer)):
        return val
    try:
        return float(val)
    except Exception:
        return np.nan

def get_signals(data):
    latest = data.iloc[-1]
    signals = {}
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    supertrend_col = [str(c) for c in list(data.columns) if isinstance(c, str) and str(c).startswith('SUPERT_') and not str(c).endswith('_dir')]
    supertrend = try_scalar(latest[supertrend_col[0]]) if supertrend_col else np.nan
    close = try_scalar(latest.get('Close', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    stochrsi = try_scalar(latest.get('STOCHRSI', np.nan))
    signals['RSI Signal'] = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if macd > macds else ('Bearish' if macd < macds else 'Neutral')
    signals['Supertrend'] = (
        'Bullish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend < close else
        'Bearish' if not np.isnan(supertrend) and not np.isnan(close) and supertrend > close else
        'Unknown'
    )
    signals['ADX Trend'] = 'Strong' if adx > 25 else 'Weak'
    signals['STOCHRSI Signal'] = 'Overbought' if stochrsi > 0.8 else ('Oversold' if stochrsi < 0.2 else 'Neutral')
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
                "LTP": float(try_scalar(latest.get('Close', np.nan))),
                "RSI": float(try_scalar(latest.get('RSI', np.nan))),
                "MACD": float(try_scalar(latest.get('MACD_12_26_9', np.nan))),
                "ADX": float(try_scalar(latest.get('ADX', np.nan))),
                "ATR": float(try_scalar(latest.get('ATR', np.nan))),
                "Signal": signals['RSI Signal'] + "/" + signals['MACD Signal'] + "/" + signals['Supertrend'],
            }
            screener_data.append(row)
    return pd.DataFrame(screener_data)

# ---- Sidebar: Watchlist P&L Tracker ----
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
    except Exception as e:
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
        browser_notification(
            "BlockVista Error",
            f"❌ P&L Watchlist: {s} fetch/update failed.",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
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

# ---- Sidebar: Price Alert with Browser Notification ----
st.sidebar.subheader("Simple Price Alert")
if len(screen_df):
    alert_price = st.sidebar.number_input(
        'Alert above price',
        value=float(screen_df.iloc[0]['LTP']),
        key="alert_price_value"
    )
else:
    alert_price = st.sidebar.number_input(
        'Alert above price',
        value=0.0,
        key="alert_price_fallback"
    )
if st.sidebar.button("Set/Check Alert") and len(screen_df):
    curr_price = get_live_price(stock_list[0])
    if curr_price != 'Error' and curr_price is not None and curr_price > alert_price:
        st.sidebar.warning(f"ALERT: {stock_list[0]} > {alert_price}")
        browser_notification(
            "Stock Price Alert",
            f"🚨 {stock_list[0]} > ₹{alert_price} (Now ₹{curr_price})",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
    elif curr_price == 'Error' or curr_price is None:
        st.sidebar.error("Live price fetch failed!")
        browser_notification(
            "BlockVista Error",
            "❌ Live price fetch failed.",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )

# ---- Sidebar: Order Placement (Kite, with browser notification for errors) ----
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
        browser_notification(
            "BlockVista Error",
            f"❌ Order failed: {e}",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )

# ---- Main UI: TABS + Bloomberg Metric Bar ----
if len(stock_list):
    st.header(f"Live Technical Dashboard: {stock_list[0]}")
    data = fetch_stock_data(stock_list[0], screen_period, screen_interval)
    if data is None or not len(data):
        st.error("No data available for this symbol/interval.")
        browser_notification(
            "BlockVista Error",
            "❌ No data available for this symbol/interval.",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
        st.stop()
    price = get_live_price(stock_list[0])
    metrics_row = st.columns([1.5,1,1,1,1,1])
    latest = data.iloc[-1]
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    atr = try_scalar(latest.get('ATR', np.nan))
    vwap = try_scalar(latest.get('VWAP', np.nan))
    with metrics_row[0]: st.metric("LTP", f"{price}", label_visibility="visible")
    with metrics_row[1]: st.metric("RSI", f"{round(rsi,2) if not np.isnan(rsi) else '—'}")
    with metrics_row[2]: st.metric("MACD", f"{round(macd,2) if not np.isnan(macd) else '—'}")
    with metrics_row[3]: st.metric("ADX", f"{round(adx,2) if not np.isnan(adx) else '—'}")
    with metrics_row[4]: st.metric("ATR", f"{round(atr,2) if not np.isnan(atr) else '—'}")
    with metrics_row[5]: st.metric("VWAP", f"{round(vwap,2) if not np.isnan(vwap) else '—'}")
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
        supertrend_col = [str(c) for c in list(data.columns) if str(c).startswith('SUPERT_') and not str(c).endswith('_dir')]
        if supertrend_col:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[supertrend_col[0]],
                line=dict(color='#fae900', width=2), name='Supertrend'))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        ta_cols_all = ['RSI','ADX','STOCHRSI']
        ta_cols = [c for c in ta_cols_all if c in list(data.columns)]
        if ta_cols:
            st.line_chart(data[ta_cols].dropna())
        else:
            st.warning("No available TA columns for charting.")
            browser_notification(
                "BlockVista Warning",
                "No available TA columns for charting."
            )
        macd_cols_all = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        macd_cols = [c for c in macd_cols_all if c in list(data.columns)]
        if macd_cols:
            st.line_chart(data[macd_cols].dropna())
        if 'ATR' in data:
            st.line_chart(data['ATR'].dropna())
        last_cols_all = ['Close','RSI','ADX','STOCHRSI','ATR','VWAP']
        last_cols = [c for c in last_cols_all if c in list(data.columns)]
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
st.caption("BlockVista Terminal | Powered by Zerodha KiteConnect & Streamlit")
