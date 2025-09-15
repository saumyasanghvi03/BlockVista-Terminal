import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objs as go
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh
import feedparser
from datetime import datetime, timedelta

# ---- Browser Notification Helper (Corrected) ----
def browser_notification(title, body, icon=None):
    if not st.session_state.get("show_notifications", True):
        return

    icon_line = f'icon: "{icon}",' if icon else ""
    script_str = f"""
    <script>
    if ('Notification' in window) {{
        if (Notification.permission === "granted") {{
            new Notification("{title}", {{
                body: "{body}",
                {icon_line}
            }});
        }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(permission => {{
                if (permission === "granted") {{
                    new Notification("{title}", {{
                        body: "{body}",
                        {icon_line}
                    }});
                }}
            }});
        }}
    }}
    </script>
    """
    st.markdown(script_str, unsafe_allow_html=True)

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

st.markdown("""
<style>
div.block-container > div:nth-child(2) {margin-top: -36px !important;}
h1 {margin-top: -20px !important;}
.block-container > div:nth-child(2) {margin-bottom: -36px !important;}
.block-container h1 {margin-top: -28px !important;}
section.main > div:first-child {margin-bottom: -28px !important;}

.stExpander {
    margin-top: -42px !important;
}
.stExpander > div:first-child {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}
</style>
""", unsafe_allow_html=True)

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
    "Defense": ["HAL", "BEL", "BEML", "MTARTECH", "BDL", "MAZDOCK", "SOLARA", "COCHINSHIP"],
}

# ---- Zerodha Login ----
try:
    api_key = st.secrets["ZERODHA_API_KEY"]
    api_secret = st.secrets["ZERODHA_API_SECRET"]
except KeyError:
    st.error("Please add ZERODHA_API_KEY and ZERODHA_API_SECRET to your Streamlit secrets.")
    st.stop()

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
            st.experimental_rerun()
        except Exception as ex:
            st.error(f"❌ Zerodha login failed: {ex}")
            browser_notification(
                "BlockVista Error",
                f"❌ Zerodha login failed: {ex}",
                "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
            )
    st.stop()
else:
    if "kite" not in st.session_state:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state["access_token"])
        st.session_state["kite"] = kite
    kite = st.session_state["kite"]

def get_live_price(symbol):
    try:
        ltp = kite.ltp(f"NSE:{symbol}")
        return ltp[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        browser_notification(
            "BlockVista Error",
            f"❌ Live price fetch failed for {symbol}: {e}",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
        return np.nan

@st.cache_data(ttl=60) # Cache the data for 60 seconds
def fetch_stock_data(symbol, period, interval):
    data = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False)
    if data.empty:
        st.warning(f"No {interval} data for {symbol} for {period}. Fetching last 5 days.")
        data = yf.download(f"{symbol}.NS", period='5d', interval=interval, progress=False)

    if data.empty:
        return None
        
    data.columns = [c.replace(' ', '_').replace('.', '_').replace('-', '_') for c in data.columns]
    
    # Calculate indicators, handling potential empty DataFrames
    if not data.empty:
        data['RSI'] = ta.rsi(data['Close'], length=14)
        macd_res = ta.macd(data['Close'])
        if isinstance(macd_res, pd.DataFrame) and not macd_res.empty:
            for col in macd_res.columns:
                data[col] = macd_res[col]
        
        data['SMA21'] = ta.sma(data['Close'], length=21)
        data['EMA9'] = ta.ema(data['Close'], length=9)
        
        bbands_res = ta.bbands(data['Close'], length=20)
        if isinstance(bbands_res, pd.DataFrame) and not bbands_res.empty:
            for label, key in zip(['BOLL_L', 'BOLL_M', 'BOLL_U'], ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
                if key in bbands_res.columns:
                    data[label] = bbands_res[key]
                else:
                    data[label] = np.nan
        
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        adx_res = ta.adx(data['High'], data['Low'], data['Close'], length=14)
        if isinstance(adx_res, pd.DataFrame) and 'ADX_14' in adx_res.columns:
            data['ADX'] = adx_res['ADX_14']
        
        stochrsi_res = ta.stochrsi(data['Close'], length=14)
        if isinstance(stochrsi_res, pd.DataFrame) and 'STOCHRSIk_14_14_3_3' in stochrsi_res.columns:
            data['STOCHRSI'] = stochrsi_res["STOCHRSIk_14_14_3_3"]
        
        supertrend_res = ta.supertrend(data['High'], data['Low'], data['Close'])
        if isinstance(supertrend_res, pd.DataFrame) and not supertrend_res.empty:
            for col in supertrend_res.columns:
                data[col] = supertrend_res[col]
        
        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        ha_res = ta.ha(data['Open'], data['High'], data['Low'], data['Close'])
        if isinstance(ha_res, pd.DataFrame) and not ha_res.empty:
            for c in ['open','high','low','close']:
                ha_key = f'HA_{c}'
                if ha_key in ha_res.columns:
                    data[ha_key] = ha_res[ha_key]
        
    return data

def try_scalar(val):
    if isinstance(val, (pd.Series, pd.DataFrame)) and not val.empty:
        val = val.iloc[-1]
    if isinstance(val, (float, int, np.floating, np.integer)):
        return val
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def get_signals(data):
    if data is None or data.empty:
        return {}
    latest = data.iloc[-1]
    signals = {}
    rsi = try_scalar(latest.get('RSI', np.nan))
    macd = try_scalar(latest.get('MACD_12_26_9', np.nan))
    macds = try_scalar(latest.get('MACDs_12_26_9', np.nan))
    
    supertrend_col = [c for c in data.columns if c.startswith('SUPERT_') and not c.endswith('_dir')]
    supertrend = try_scalar(latest.get(supertrend_col[0], np.nan)) if supertrend_col else np.nan
    
    close = try_scalar(latest.get('Close', np.nan))
    adx = try_scalar(latest.get('ADX', np.nan))
    stochrsi = try_scalar(latest.get('STOCHRSI', np.nan))
    
    signals['RSI Signal'] = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    signals['MACD Signal'] = 'Bullish' if macd > macds else ('Bearish' if macd < macds else 'Neutral')
    
    if not np.isnan(supertrend) and not np.isnan(close):
        signals['Supertrend'] = 'Bullish' if supertrend < close else 'Bearish'
    else:
        signals['Supertrend'] = 'Unknown'

    signals['ADX Trend'] = 'Strong' if adx > 25 else 'Weak'
    signals['STOCHRSI Signal'] = 'Overbought' if stochrsi > 0.8 else ('Oversold' if stochrsi < 0.2 else 'Neutral')
    return signals

def make_screener(stock_list, period, interval):
    screener_data = []
    for s in stock_list:
        data = fetch_stock_data(s, period, interval)
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            signals = get_signals(data)
            row = {
                "Symbol": s,
                "LTP": try_scalar(latest.get('Close', np.nan)),
                "RSI": try_scalar(latest.get('RSI', np.nan)),
                "MACD": try_scalar(latest.get('MACD_12_26_9', np.nan)),
                "ADX": try_scalar(latest.get('ADX', np.nan)),
                "ATR": try_scalar(latest.get('ATR', np.nan)),
                "Signal": f"{signals.get('RSI Signal', 'N/A')}/{signals.get('MACD Signal', 'N/A')}/{signals.get('Supertrend', 'N/A')}",
            }
            screener_data.append(row)
    return pd.DataFrame(screener_data)

# ---- Sidebar: Calculators ----
with st.sidebar.expander('🔢 Calculators: Brokerage, SIP, ROI', expanded=False):
    st.subheader("Brokerage Calculator")
    trade_value = st.number_input("Trade Amount (₹)", min_value=1)
    brokerage_rate = st.number_input("Brokerage Rate (%)", value=0.03)
    other_charges = st.number_input("Other Charges (₹)", value=20.0)
    if st.button("Calculate Brokerage"):
        total_brokerage = trade_value * (brokerage_rate / 100) + other_charges
        st.success(f"Total Cost: ₹{total_brokerage:.2f}")

    st.divider()
    st.subheader("SIP ROI Calculator")
    sip_amt = st.number_input("SIP/Month (₹)", key="sip_amt")
    months = st.number_input("Months", value=12, min_value=1)
    expected_cagr = st.number_input("Expected CAGR (%)", value=12.0)
    if st.button("Calculate SIP Returns"):
        r = (expected_cagr / 100) / 12
        final_amt = sip_amt * (((1 + r) ** months - 1) / r) * (1 + r)
        st.success(f"Projected Value: ₹{final_amt:,.2f}")

    st.divider()
    st.subheader("ROI Calculator")
    inv = st.number_input("Initial Investment (₹)", key="inv")
    final = st.number_input("Final Value (₹)", key="final")
    if st.button("Calculate ROI"):
        try:
            if inv != 0:
                roi_val = ((final - inv) / inv) * 100
                st.success(f"ROI: {roi_val:.2f}%")
            else:
                st.error("Initial investment cannot be zero.")
        except Exception:
            st.error("Enter valid numbers for calculation.")

# ---- Sidebar: Watchlist P&L Tracker (Live) ----
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
        if i >= len(entry_prices) or i >= len(quantities):
            continue
        live = get_live_price(s)
        if pd.isna(live) or live is None:
            d = fetch_stock_data(s, "1d", "5m")
            if d is not None and not d.empty:
                live = d["Close"].iloc[-1]
            else:
                live = np.nan
        
        if not pd.isna(live):
            pnl = (live - entry_prices[i]) * quantities[i]
            pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": round(live, 2), "Qty": quantities[i], "P&L ₹": round(pnl, 2)})
        else:
            pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "N/A", "Qty": quantities[i], "P&L ₹": "N/A"})
    except Exception as e:
        st.sidebar.error(f"Error for {s}: {e}")
        pnl_data.append({"Symbol": s, "Entry": entry_prices[i], "LTP": "Err", "Qty": quantities[i], "P&L ₹": "Err"})
        browser_notification(
            "BlockVista Error",
            f"❌ P&L Watchlist: {s} fetch/update failed.",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
if pnl_data:
    df_pnl = pd.DataFrame(pnl_data)
    st.sidebar.dataframe(df_pnl.set_index('Symbol'))
    total_pnl = df_pnl['P&L ₹'].sum() if all(isinstance(x, (int, float)) for x in df_pnl['P&L ₹']) else "N/A"
    st.sidebar.markdown(f"<b>Total P&L ₹: {round(total_pnl, 2)}</b>", unsafe_allow_html=True)

# ---- Sidebar: Order Placement (Kite) ----
st.sidebar.subheader("Order Placement (Kite)")
trade_type = st.sidebar.selectbox("Type", ['BUY', 'SELL'])
order_qty = st.sidebar.number_input("Quantity", value=1, step=1, min_value=1, key="order_qty_sidebar")
order_type = st.sidebar.selectbox("Order Type", ['MARKET', 'LIMIT'])
limit_price = st.sidebar.number_input("Limit Price", value=0.0, key="order_limit_price") if order_type == 'LIMIT' else None
symbol_for_order = st.sidebar.text_input("Stock Symbol", value="RELIANCE", key="order_symbol_sidebar")
if st.sidebar.button("PLACE ORDER"):
    try:
        placed_order = kite.place_order(
            tradingsymbol=symbol_for_order,
            exchange="NSE",
            transaction_type=trade_type,
            quantity=int(order_qty),
            order_type=order_type,
            price=limit_price,
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

# ---- Sidebar: GTT Orders ----
st.sidebar.subheader("GTT (Good Till Triggered) Order")
gtt_symbol = st.sidebar.text_input("GTT Symbol", value="RELIANCE", key="gtt_symbol")
gtt_qty = st.sidebar.number_input("GTT Quantity", value=1, min_value=1, key="gtt_qty")
gtt_trigger = st.sidebar.number_input("Trigger Price", value=0.0, key="gtt_trigger")
gtt_limit = st.sidebar.number_input("Limit Price (Executes At)", value=0.0, key="gtt_limit")
gtt_type = st.sidebar.selectbox("GTT Side", ['BUY', 'SELL'], key="gtt_side")
if st.sidebar.button("PLACE GTT"):
    try:
        placed_gtt = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_SINGLE,
            tradingsymbol=gtt_symbol,
            exchange="NSE",
            trigger_values=[gtt_trigger],
            last_price=get_live_price(gtt_symbol),
            orders=[{
                "transaction_type": gtt_type,
                "quantity": int(gtt_qty),
                "order_type": "LIMIT",
                "product": "CNC",
                "price": gtt_limit
            }]
        )
        st.sidebar.success(f"GTT Order placed: ID {placed_gtt['trigger_id']}")
    except Exception as e:
        st.sidebar.error(f"GTT Order failed: {e}")
        browser_notification(
            "BlockVista Error",
            f"GTT Order failed: {e}",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )

# ---- Quick News Deck ----
def get_news_headlines_rss(ticker='^NSEI'):
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}.NS"
    try:
        feed = feedparser.parse(rss_url)
        return [(e['title'], e['link']) for e in feed['entries'][:5]]
    except Exception as e:
        st.error(f"Could not fetch news: {e}")
        return []

with st.expander("📰 Quick News Deck (Yahoo Finance Headlines)", expanded=False):
    news_query = st.text_input("Symbol for News", "RELIANCE")
    news = get_news_headlines_rss(news_query)
    if news:
        for (headline, link) in news:
            st.markdown(f"- [{headline}]({link})")
    else:
        st.info("No headlines found or invalid symbol. Try ^NSEI or another stock!")

# ---- Sentiment Meter (Live Market Mood) ----
@st.cache_data(ttl=60)
def get_market_sentiment(stock_list):
    pos, neg = 0, 0
    for sym in stock_list:
        try:
            d = fetch_stock_data(sym, "1d", "5m")
            if d is not None and len(d) > 1:
                diff = d["Close"].iloc[-1] - d["Close"].iloc[0]
                if diff > 0: pos += 1
                elif diff < 0: neg += 1
        except Exception:
            pass
    if pos+neg == 0: return "Neutral", 0.5
    ratio = pos / (pos + neg)
    if ratio > 0.7: return "Strongly Bullish", ratio
    elif ratio > 0.5: return "Bullish", ratio
    elif ratio == 0.5: return "Neutral", ratio
    elif ratio > 0.3: return "Bearish", ratio
    else: return "Strongly Bearish", ratio

st.subheader("📊 Market Sentiment Meter")
stock_list_sent = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
sent_txt, sent_val = get_market_sentiment(stock_list_sent)
st.markdown(f"**Market Sentiment:** {sent_txt}")
st.progress(sent_val)

from datetime import datetime
if "sentiment_history" not in st.session_state:
    st.session_state["sentiment_history"] = []

now = datetime.now()
if not st.session_state["sentiment_history"] or \
   st.session_state["sentiment_history"][-1][1] != sent_val:
    st.session_state["sentiment_history"].append((now, sent_val))

df_senti = pd.DataFrame(st.session_state["sentiment_history"], columns=["time", "sentiment"]).set_index("time")
if not df_senti.empty:
    st.line_chart(df_senti["sentiment"])

# ---- One-Click Custom Alerts ----
st.sidebar.subheader("🔔 One-Click Price Alert (this session only)")
alert_symbol = st.sidebar.text_input("Alert Symbol", value="RELIANCE")
alert_price = st.sidebar.number_input("Target Price", value=0.0)
if st.sidebar.button("Set Alert"):
    if "alerts" not in st.session_state: st.session_state["alerts"] = {}
    st.session_state["alerts"][alert_symbol.upper()] = alert_price
    st.sidebar.success(f"Alert set for {alert_symbol} > ₹{alert_price}")
if "alerts" in st.session_state:
    for sym, target in list(st.session_state["alerts"].items()):
        try:
            ltp = get_live_price(sym)
            if ltp is not None and not pd.isna(ltp) and ltp > target:
                st.sidebar.warning(f"ALERT: {sym} > ₹{target} (Now ₹{ltp})")
                browser_notification(
                    f"Price Alert Hit: {sym}",
                    f"Hit ₹{target} (Now ₹{ltp})"
                )
                del st.session_state["alerts"][sym]
        except Exception:
            pass

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
if not screen_df.empty:
    st.sidebar.dataframe(screen_df)
    csv = screen_df.to_csv(index=False)
    st.sidebar.download_button("Export Screener CSV", csv, file_name="screener_results.csv")
else:
    st.sidebar.info("No data found for selection.")

# ---- Main UI: TABS, Chart + Trade-from-Chart ----
if len(stock_list) > 0:
    st.header(f"Live Technical Dashboard: {stock_list[0]}")
    data = fetch_stock_data(stock_list[0], screen_period, screen_interval)
    if data is None or data.empty:
        st.error("No data available for this symbol/interval.")
        browser_notification(
            "BlockVista Error",
            "❌ No data available for this symbol/interval.",
            "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
        )
    else:
        ltp_tab = get_live_price(stock_list[0])
        
        tabs = st.tabs(["Chart", "TA", "Advanced", "Raw"])

        with tabs[0]:
            st.metric(
                "Live Price (LTP)",
                f"₹{ltp_tab:.2f}" if ltp_tab is not None and not pd.isna(ltp_tab) else "N/A"
            )
            latest = data.iloc[-1]
            macd_val = try_scalar(latest.get('MACD_12_26_9', np.nan))
            rsi_val = try_scalar(latest.get('RSI', np.nan))
            
            macd_val_formatted = f"{macd_val:.2f}" if not pd.isna(macd_val) else "N/A"
            rsi_val_formatted = f"{rsi_val:.2f}" if not pd.isna(rsi_val) else "N/A"
            
            st.write(
                f"**RSI:** {rsi_val_formatted} &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; **MACD:** {macd_val_formatted}"
            )
            
            def plot_live_chart(symbol, data, ltp):
                chart_style = st.radio("Chart Style", ["Candlestick", "Heikin Ashi"], horizontal=True)
                bands_show = st.checkbox("Show Bollinger Bands (Fill)", value=True)
                
                fig = go.Figure()

                if not data.empty:
                    if chart_style == "Heikin Ashi" and 'HA_open' in data.columns:
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['HA_open'], high=data['HA_high'],
                            low=data['HA_low'], close=data['HA_close'], name='Heikin Ashi'))
                    else:
                        fig.add_trace(go.Candlestick(
                            x=data.index, open=data['Open'], high=data['High'],
                            low=data['Low'], close=data['Close'], name='Candles'))
                    
                    if bands_show and 'BOLL_U' in data.columns and 'BOLL_L' in data.columns:
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
                    if 'EMA9' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['EMA9'], line=dict(color='#00FF99', width=1), name='EMA 9'))
                    if 'SMA21' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data['SMA21'], line=dict(color='#FFA500', width=1), name='SMA 21'))
                    supertrend_col = [c for c in data.columns if c.startswith('SUPERT_') and not c.endswith('_dir')]
                    if supertrend_col:
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data[supertrend_col[0]],
                            line=dict(color='#fae900', width=2), name='Supertrend'))

                if pd.notna(ltp) and not data.empty:
                    # Use the last timestamp from the data to extend the line, if available
                    last_time = data.index[-1] if not data.empty else datetime.now()
                    fig.add_trace(
                        go.Scatter(
                            x=[data.index.min(), last_time + timedelta(minutes=5)], # Extends line slightly to the right
                            y=[ltp, ltp], 
                            mode='lines',
                            line=dict(color="#FFD900", width=2, dash='dash'),
                            name=f"Live ₹{ltp:.2f}",
                            showlegend=True
                        )
                    )
                    
                fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                return ltp if pd.notna(ltp) else (data['Close'].iloc[-1] if not data.empty else 0)

            last_price = plot_live_chart(stock_list[0], data, ltp_tab)
            
            st.markdown("#### Place Order Directly from Chart")
            chosen_price = st.number_input(
                "Trade Price (pick from chart, autofilled with last close)",
                value=float(last_price) if not pd.isna(last_price) else 0.0
            )
            side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
            qty = st.number_input("Quantity", min_value=1, value=1)
            
            if st.button(f"{side} Now (Chart Tab)"):
                try:
                    placed_order = kite.place_order(
                        tradingsymbol=stock_list[0], exchange="NSE",
                        transaction_type=side, quantity=int(qty),
                        order_type="LIMIT", price=chosen_price,
                        variety="regular", product="CNC", validity="DAY"
                    )
                    st.success(f"Order {side} placed for {qty} at ₹{chosen_price:.2f}. ID: {placed_order}")
                except Exception as e:
                    st.error(f"Order failed: {e}")
                    browser_notification(
                        "BlockVista Error",
                        f"❌ Order failed: {e}",
                        "https://cdn-icons-png.flaticon.com/512/2583/2583346.png"
                    )
        
        with tabs[1]:
            ta_cols_all = ['RSI', 'ADX', 'STOCHRSI']
            ta_cols = [c for c in ta_cols_all if c in data.columns]
            if ta_cols:
                st.line_chart(data[ta_cols].dropna())
            else:
                st.warning("No available TA columns for charting.")
                browser_notification(
                    "BlockVista Warning",
                    "No available TA columns for charting."
                )
            
            macd_cols_all = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
            macd_cols = [c for c in macd_cols_all if c in data.columns]
            if macd_cols:
                st.line_chart(data[macd_cols].dropna())
            
            if 'ATR' in data.columns:
                st.line_chart(data['ATR'].dropna())
            
            last_cols_all = ['Close', 'RSI', 'ADX', 'STOCHRSI', 'ATR', 'VWAP']
            last_cols = [c for c in last_cols_all if c in data.columns]
            
            latest_values = data.iloc[-1].filter(items=last_cols)
            st.write("Latest Values:", latest_values)
            
        with tabs[2]:
            st.subheader("Signals (Current)")
            signals = get_signals(data)
            st.table(pd.DataFrame(signals.items(), columns=['Indicator', 'Signal']))
            csv2 = data.to_csv()
            st.download_button('Export Data to CSV', csv2, file_name=f"{stock_list[0]}_{screen_interval}.csv")
            
        with tabs[3]:
            if st.checkbox("Show Table Data"):
                st.dataframe(data.tail(40))

st.caption("BlockVista Terminal | Powered by Zerodha KiteConnect, yFinance, Plotly & Streamlit")
