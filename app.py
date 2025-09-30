import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import io
import logging
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas_ta as ta
from streamlit import session_state as session
from scipy import optimize

# Configure logging
logging.basicConfig(filename='blockvista_errors.log', level=logging.ERROR)

# Mock implied volatility function (replace with actual implementation if available)
def implied_volatility(S, K, T, r, option_price, option_type):
    def bs_model(sigma, S, K, T, r, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        return optimize.newton(lambda sigma: bs_model(sigma, S, K, T, r, option_type) - option_price, 0.2)
    except:
        return np.nan

# Mock calculate_strategy_pnl function (replace with actual implementation if available)
def calculate_strategy_pnl(legs, underlying_ltp):
    price_range = np.linspace(underlying_ltp * 0.8, underlying_ltp * 1.2, 100)
    pnl_df = pd.DataFrame(index=price_range, columns=['Total P&L'])
    pnl_df['Total P&L'] = 0
    for leg in legs:
        strike = leg['strike']
        premium = leg['premium']
        quantity = leg['quantity']
        multiplier = 1 if leg['position'] == 'Buy' else -1
        if leg['type'] == 'Call':
            leg_pnl = np.maximum(0, price_range - strike) - premium
        else:
            leg_pnl = np.maximum(0, strike - price_range) - premium
        pnl_df['Total P&L'] += leg_pnl * quantity * multiplier
    max_profit = pnl_df['Total P&L'].max()
    max_loss = pnl_df['Total P&L'].min()
    breakevens = price_range[np.abs(pnl_df['Total P&L']) < 1e-2].tolist()
    return pnl_df, max_profit, max_loss, breakevens

# Data sources (update with actual URLs and symbols as needed)
ML_DATA_SOURCES = {
    "NIFTY 50": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/NIFTY50.csv",
        "tradingsymbol": "NIFTY 50",
        "exchange": "NSE"
    },
    "BANKNIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/BANKNIFTY.csv",
        "tradingsymbol": "BANKNIFTY",
        "exchange": "NSE"
    },
    "FINNIFTY": {
        "github_url": "https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/FINNIFTY.csv",
        "tradingsymbol": "FINNIFTY",
        "exchange": "NSE"
    }
}

@st.cache_data
def get_instrument_df():
    client = get_broker_client()
    if not client:
        return pd.DataFrame()
    try:
        instruments = client.instruments()
        df = pd.DataFrame(instruments)
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Failed to fetch instruments: {e}")
        st.error(f"Failed to fetch instrument data: {e}")
        return pd.DataFrame()

def get_broker_client():
    client = st.session_state.get('kite')
    if client:
        try:
            client.ltp("NSE:NIFTY 50")
            return client
        except Exception:
            st.error("Broker connection lost. Please reconnect.")
            return None
    return None

def get_instrument_token(tradingsymbol, instrument_df, exchange):
    if instrument_df.empty:
        return None
    result = instrument_df[
        (instrument_df['tradingsymbol'] == tradingsymbol) & 
        (instrument_df['exchange'] == exchange)
    ]
    return result['instrument_token'].iloc[0] if not result.empty else None

@st.cache_data(ttl=60)
def get_historical_data(instrument_token, interval, period=None, from_date=None, to_date=None):
    """
    Fetches historical data from the broker's API.
    """
    client = get_broker_client()
    if not client or not instrument_token:
        st.error("Broker client or instrument token not available.")
        return pd.DataFrame()

    if st.session_state.broker == "Zerodha":
        if not to_date:
            to_date = datetime.now().date()
        if not from_date:
            days_to_subtract = {'1d': 2, '5d': 7, '1mo': 31, '6mo': 182, '1y': 365, '5y': 1825}
            from_date = to_date - timedelta(days=days_to_subtract.get(period, 1825))

        # Ensure from_date is not after to_date
        if from_date > to_date:
            st.error(f"Invalid date range: from_date ({from_date}) is after to_date ({to_date}). Adjusting from_date.")
            from_date = to_date - timedelta(days=365)

        try:
            records = client.historical_data(instrument_token, from_date, to_date, interval)
            df = pd.DataFrame(records)
            if df.empty:
                st.warning(f"No historical data found for token {instrument_token}.")
                return df

            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            try:
                df.ta.adx(append=True); df.ta.apo(append=True); df.ta.aroon(append=True); df.ta.atr(append=True)
                df.ta.bbands(append=True); df.ta.cci(append=True); df.ta.chop(append=True); df.ta.cksp(append=True)
                df.ta.cmf(append=True); df.ta.coppock(append=True); df.ta.ema(length=50, append=True)
                df.ta.ema(length=200, append=True); df.ta.fisher(append=True); df.ta.kst(append=True)
                df.ta.macd(append=True); df.ta.mfi(append=True); df.ta.mom(append=True); df.ta.obv(append=True)
                df.ta.rsi(append=True); df.ta.stoch(append=True); df.ta.supertrend(append=True); df.ta.willr(append=True)
            except Exception as e:
                st.toast(f"Could not calculate some indicators: {e}", icon="⚠️")
            return df
        except Exception as e:
            logging.error(f"Kite API Error (Historical): {e}, Token: {instrument_token}, From: {from_date}, To: {to_date}")
            st.error(f"Kite API Error (Historical): {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Historical data for {st.session_state.broker} not implemented.")
        return pd.DataFrame()

@st.cache_data
def load_and_combine_data(instrument_name):
    """
    Loads and combines historical data from a static CSV with live data from the broker.
    """
    source_info = ML_DATA_SOURCES.get(instrument_name)
    if not source_info:
        st.error(f"No data source configured for {instrument_name}")
        return pd.DataFrame()

    try:
        response = requests.get(source_info['github_url'])
        response.raise_for_status()
        hist_df = pd.read_csv(io.StringIO(response.text))
        hist_df['Date'] = pd.to_datetime(hist_df['Date'], format='mixed', dayfirst=True)
        hist_df.set_index('Date', inplace=True)
        hist_df.columns = [col.lower() for col in hist_df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in hist_df.columns:
                hist_df[col] = pd.to_numeric(hist_df[col].astype(str).str.replace(',', ''), errors='coerce')
        hist_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    except Exception as e:
        logging.error(f"Failed to load historical data: {e}")
        st.error(f"Failed to load historical data: {e}")
        hist_df = pd.DataFrame()

    live_df = pd.DataFrame()
    if get_broker_client() and source_info.get('tradingsymbol') and source_info.get('exchange') != 'yfinance':
        instrument_df = get_instrument_df()
        token = get_instrument_token(source_info['tradingsymbol'], instrument_df, source_info['exchange'])
        if token:
            from_date = (hist_df.index.max().date() if not hist_df.empty else datetime.now().date() - timedelta(days=365))
            to_date = datetime.now().date()
            if from_date > to_date:
                from_date = to_date - timedelta(days=365)
            live_df = get_historical_data(token, 'day', from_date=from_date, to_date=to_date)
            if not live_df.empty:
                live_df.columns = [col.lower() for col in live_df.columns]

    elif source_info.get('exchange') == 'yfinance':
        try:
            live_df = yf.download(source_info['SSID'], period="max")
            if not live_df.empty:
                live_df.columns = [col.lower() for col in live_df.columns]
        except Exception as e:
            logging.error(f"Failed to load yfinance data: {e}")
            st.error(f"Failed to load yfinance data: {e}")
            live_df = pd.DataFrame()

    if not live_df.empty and not hist_df.empty:
        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df
    elif not hist_df.empty:
        hist_df.sort_index(inplace=True)
        return hist_df
    else:
        return live_df

@st.cache_data(ttl=30)
def get_options_chain(underlying, instrument_df, expiry_date=None):
    """
    Fetches and processes the options chain for a given underlying.
    """
    client = get_broker_client()
    if not client or instrument_df.empty:
        st.error("Broker client or instrument data not available.")
        return pd.DataFrame(), None, 0.0, []

    if st.session_state.broker == "Zerodha":
        exchange_map = {"GOLDM": "MCX", "CRUDEOIL": "MCX", "SILVERM": "MCX", "NATURALGAS": "MCX", "USDINR": "CDS"}
        exchange = exchange_map.get(underlying, 'NFO')
        ltp_symbol = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY", "FINNIFTY": "FINNIFTY"}.get(underlying, underlying)
        ltp_exchange = "NSE" if exchange == "NFO" else exchange
        underlying_instrument_name = f"{ltp_exchange}:{ltp_symbol}"

        try:
            underlying_ltp = client.ltp(underlying_instrument_name)[underlying_instrument_name]['last_price']
        except Exception as e:
            logging.error(f"Could not fetch LTP for {ltp_symbol}: {e}")
            st.warning(f"Could not fetch LTP for {ltp_symbol}: {e}")
            underlying_ltp = 0.0

        options = instrument_df[(instrument_df['name'] == underlying.upper()) & (instrument_df['exchange'] == exchange)]
        if options.empty:
            st.warning(f"No options data found for {underlying} in exchange {exchange}.")
            return pd.DataFrame(), None, underlying_ltp, []

        expiries = sorted(pd.to_datetime(options['expiry'].unique()))
        three_months_later = datetime.now() + timedelta(days=90)
        available_expiries = [e for e in expiries if datetime.now().date() <= e.date() <= three_months_later.date()]
        if not available_expiries:
            st.warning(f"No valid expiry dates found for {underlying} within the next 90 days.")
            return pd.DataFrame(), None, underlying_ltp, []

        if not expiry_date:
            expiry_date = available_expiries[0]
        elif expiry_date not in available_expiries:
            st.warning(f"Selected expiry date {expiry_date} is not available. Using nearest expiry.")
            expiry_date = available_expiries[0]

        chain_df = options[options['expiry'] == expiry_date].sort_values(by='strike')
        ce_df = chain_df[chain_df['instrument_type'] == 'CE'].copy()
        pe_df = chain_df[chain_df['instrument_type'] == 'PE'].copy()
        instruments_to_fetch = [f"{exchange}:{s}" for s in list(ce_df['tradingsymbol']) + list(pe_df['tradingsymbol'])]
        if not instruments_to_fetch:
            st.warning(f"No instruments found for {underlying} on expiry {expiry_date}.")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries

        try:
            quotes = client.quote(instruments_to_fetch)
            ce_df['LTP'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            pe_df['LTP'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('last_price', 0))
            ce_df['open_interest_CE'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            ce_df['open_interest_CE_change'] = ce_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('net_change_oi', 0))
            pe_df['open_interest_PE'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('oi', 0))
            pe_df['open_interest_PE_change'] = pe_df['tradingsymbol'].apply(lambda x: quotes.get(f"{exchange}:{x}", {}).get('net_change_oi', 0))

            final_chain = pd.merge(
                ce_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_CE', 'open_interest_CE_change']],
                pe_df[['tradingsymbol', 'strike', 'LTP', 'open_interest_PE', 'open_interest_PE_change']],
                on='strike', suffixes=('_CE', '_PE'), how='outer'
            ).rename(columns={
                'LTP_CE': 'CALL LTP', 'LTP_PE': 'PUT LTP', 'strike': 'STRIKE',
                'tradingsymbol_CE': 'CALL', 'tradingsymbol_PE': 'PUT'
            }).fillna(0)

            return (
                final_chain[['CALL', 'CALL LTP', 'open_interest_CE', 'STRIKE', 'PUT LTP', 'open_interest_PE', 'PUT', 'open_interest_CE_change', 'open_interest_PE_change']],
                expiry_date,
                underlying_ltp,
                available_expiries
            )
        except Exception as e:
            logging.error(f"Failed to fetch real-time OI data: {e}")
            st.error(f"Failed to fetch real-time OI data: {e}")
            return pd.DataFrame(), expiry_date, underlying_ltp, available_expiries
    else:
        st.warning(f"Options chain for {st.session_state.broker} not implemented.")
        return pd.DataFrame(), None, 0.0, []

def style_option_chain(chain_df, underlying_ltp):
    def highlight_strike(row):
        styles = [''] * len(row)
        if 'STRIKE' in row.index:
            strike = row['STRIKE']
            if abs(strike - underlying_ltp) == min(abs(chain_df['STRIKE'] - underlying_ltp)):
                styles[row.index.get_loc('STRIKE')] = 'background-color: rgba(255, 255, 0, 0.3)'
        return styles
    return chain_df.style.apply(highlight_strike, axis=1)

def display_header():
    if st.session_state.get('profile'):
        st.sidebar.write(f"**User**: {st.session_state.profile.get('user_name', 'N/A')}")
        st.sidebar.write(f"**Broker**: {st.session_state.broker}")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.write("Please login to access all features.")

def login_page():
    st.title("BlockVista Terminal")
    st.subheader("Broker Login")
    broker = st.selectbox("Select Your Broker", ["Zerodha"])
    
    if broker == "Zerodha":
        api_key = st.secrets.get("ZERODHA_API_KEY")
        api_secret = st.secrets.get("ZERODHA_API_SECRET")
        
        if not api_key or not api_secret:
            st.error("Zerodha API credentials not configured. Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in Streamlit secrets.")
            st.stop()
        
        kite = KiteConnect(api_key=api_key)
        request_token = st.query_params.get("request_token")
        
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                st.session_state.access_token = data["access_token"]
                kite.set_access_token(st.session_state.access_token)
                st.session_state.kite = kite
                st.session_state.profile = kite.profile()
                st.session_state.broker = "Zerodha"
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                logging.error(f"Authentication failed: {e}")
                st.error(f"Authentication failed: {e}")
                st.query_params.clear()
        else:
            st.link_button("Login with Zerodha Kite", kite.login_url())
            st.info("Please login with Zerodha Kite to begin. On first login, you will be prompted for a QR code scan. In subsequent sessions, a 2FA code will be required.")

def qr_code_dialog():
    st.session_state.two_factor_setup_complete = True  # Mock implementation

def two_factor_dialog():
    st.session_state.authenticated = True  # Mock implementation

def page_fo_analytics():
    """F&O Analytics page with comprehensive options analysis."""
    display_header()
    st.title("F&O Analytics Hub")

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access F&O Analytics.")
        return

    tab1, tab2, tab3 = st.tabs(["Options Chain", "PCR Analysis", "Volatility & OI Analysis"])

    with tab1:
        st.subheader("Live Options Chain")

        col1, col2 = st.columns([1, 3])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="fo_underlying")
            expiry_options = get_options_chain(underlying, instrument_df)[3]
            expiry_str = st.selectbox(
                "Select Expiry",
                [e.strftime("%d %b %Y") for e in expiry_options] if expiry_options else ["No expiries available"],
                key="fo_expiry"
            )
            expiry_date = datetime.strptime(expiry_str, "%d %b %Y") if expiry_str != "No expiries available" else None

        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df, expiry_date)

        if not chain_df.empty and 'STRIKE' in chain_df.columns:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}" if underlying_ltp > 0 else "N/A")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")

            styled_chain = style_option_chain(chain_df, underlying_ltp)
            st.dataframe(
                styled_chain.format({
                    'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}', 'STRIKE': '₹{:.0f}',
                    'open_interest_CE': '{:,.0f}', 'open_interest_PE': '{:,.0f}',
                    'open_interest_CE_change': '{:,.0f}', 'open_interest_PE_change': '{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(f"Could not load options chain data for {underlying}. Please try a different underlying or expiry.")

    with tab2:
        st.subheader("Put-Call Ratio Analysis")

        chain_df, _, _, _ = get_options_chain(st.session_state.get('fo_underlying', "NIFTY"), instrument_df, expiry_date)
        if not chain_df.empty and 'open_interest_CE' in chain_df.columns:
            total_ce_oi = chain_df['open_interest_CE'].sum()
            total_pe_oi = chain_df['open_interest_PE'].sum()
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total CE OI", f"{total_ce_oi:,.0f}")
            col2.metric("Total PE OI", f"{total_pe_oi:,.0f}")
            col3.metric("PCR", f"{pcr:.2f}")

            if pcr > 1.3:
                st.success("High PCR suggests potential bearish sentiment (more Puts bought for hedging/speculation).")
            elif pcr < 0.7:
                st.error("Low PCR suggests potential bullish sentiment (more Calls bought).")
            else:
                st.info("PCR indicates neutral sentiment.")
        else:
            st.info("PCR data is loading... Select an underlying and expiry in the 'Options Chain' tab first.")

    with tab3:
        st.subheader("Volatility & Open Interest Surface")
        st.info("Real-time implied volatility and OI analysis for options contracts.")

        if not chain_df.empty and expiry and underlying_ltp > 0:
            T = (expiry.date() - datetime.now().date()).days / 365.0
            r = 0.07

            with st.spinner("Calculating Implied Volatility..."):
                chain_df['IV_CE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['CALL LTP'], 'call') * 100,
                    axis=1
                )
                chain_df['IV_PE'] = chain_df.apply(
                    lambda row: implied_volatility(underlying_ltp, row['STRIKE'], T, r, row['PUT LTP'], 'put') * 100,
                    axis=1
                )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_CE'], mode='lines+markers', name='Call IV', line=dict(color='cyan')), secondary_y=False)
            fig.add_trace(go.Scatter(x=chain_df['STRIKE'], y=chain_df['IV_PE'], mode='lines+markers', name='Put IV', line=dict(color='magenta')), secondary_y=False)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['open_interest_CE'], name='Call OI', marker_color='rgba(0, 255, 255, 0.4)'), secondary_y=True)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['open_interest_PE'], name='Put OI', marker_color='rgba(255, 0, 255, 0.4)'), secondary_y=True)

            fig.update_layout(
                title_text=f"{underlying} IV & OI Profile for {expiry.strftime('%d %b %Y')}",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select an underlying and expiry in the 'Options Chain' tab to view the volatility surface.")

def page_option_strategy_builder():
    """Option Strategy Builder page with live data and P&L calculation."""
    display_header()
    st.title("Options Strategy Builder")

    instrument_df = get_instrument_df()
    client = get_broker_client()
    if instrument_df.empty or not client:
        st.info("Please connect to a broker to build strategies.")
        return

    if 'strategy_legs' not in st.session_state:
        st.session_state.strategy_legs = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Strategy Configuration")
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="strategy_underlying")
        
        chain_df, _, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)
        
        if not available_expiries:
            st.error(f"No options available for {underlying}. Please select a different underlying.")
            return

        expiry_str = st.selectbox(
            "Expiry",
            [e.strftime("%d %b %Y") for e in available_expiries],
            key="strategy_expiry"
        )
        expiry_date = datetime.strptime(expiry_str, "%d %b %Y")

        with st.form("add_leg_form"):
            st.write("**Add a New Leg**")
            leg_cols = st.columns(4)
            position = leg_cols[0].selectbox("Position", ["Buy", "Sell"])
            option_type = leg_cols[1].selectbox("Type", ["Call", "Put"])
            
            options = instrument_df[
                (instrument_df['name'] == underlying) &
                (instrument_df['expiry'] == expiry_date) &
                (instrument_df['instrument_type'] == option_type[0])
            ]

            if not options.empty:
                strikes = sorted(options['strike'].unique())
                if not strikes:
                    st.warning(f"No strikes available for {underlying} {option_type} on {expiry_date.strftime('%d %b %Y')}.")
                else:
                    strike = leg_cols[2].selectbox("Strike", strikes, index=len(strikes)//2)
                    quantity = leg_cols[3].number_input("Lots", min_value=1, value=1)
                    
                    submitted = st.form_submit_button("Add Leg")
                    if submitted:
                        lot_size = options.iloc[0]['lot_size']
                        tradingsymbol = options[options['strike'] == strike].iloc[0]['tradingsymbol']
                        
                        try:
                            quote = client.quote(f"NFO:{tradingsymbol}")[f"NFO:{tradingsymbol}"]
                            premium = quote['last_price']
                            
                            st.session_state.strategy_legs.append({
                                'symbol': tradingsymbol,
                                'position': position,
                                'type': option_type,
                                'strike': strike,
                                'quantity': quantity * lot_size,
                                'premium': premium
                            })
                            st.success(f"Added {tradingsymbol} to strategy!")
                            st.rerun()
                        except Exception as e:
                            logging.error(f"Could not fetch premium for {tradingsymbol}: {e}")
                            st.error(f"Could not fetch premium for {tradingsymbol}: {e}")
            else:
                st.warning(f"No {option_type} options found for {underlying} on {expiry_date.strftime('%d %b %Y')}.")
                st.form_submit_button("Add Leg", disabled=True)

        st.subheader("Current Legs")
        if st.session_state.strategy_legs:
            for i, leg in enumerate(st.session_state.strategy_legs):
                st.text(f"{i+1}: {leg['position']} {leg['quantity']} {leg['symbol']} @ ₹{leg['premium']:.2f}")
            if st.button("Clear All Legs"):
                st.session_state.strategy_legs = []
                st.rerun()
        else:
            st.info("Add legs to your strategy.")

    with col2:
        st.subheader("Strategy Payoff Analysis")
        
        if st.session_state.strategy_legs:
            pnl_df, max_profit, max_loss, breakevens = calculate_strategy_pnl(st.session_state.strategy_legs, underlying_ltp)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pnl_df.index, y=pnl_df['Total P&L'], mode='lines', name='P&L'))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=underlying_ltp, line_dash="dot", line_color="yellow", annotation_text="Current LTP")
            fig.update_layout(
                title="Strategy P&L Payoff Chart",
                xaxis_title="Underlying Price at Expiry",
                yaxis_title="Profit / Loss (₹)",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Risk & Reward Profile")
            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("Max Profit", f"₹{max_profit:,.2f}")
            metrics_col1.metric("Max Loss", f"₹{max_loss:,.2f}")
            metrics_col2.metric("Breakeven(s)", ", ".join([f"₹{b:,.2f}" for b in breakevens]) if breakevens else "N/A")
        else:
            st.info("Add legs to see the payoff analysis.")

def main_app():
    if st.session_state.get('profile'):
        if not st.session_state.get('two_factor_setup_complete'):
            qr_code_dialog()
            st.stop()
        if not st.session_state.get('authenticated', False):
            two_factor_dialog()
            st.stop()

    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.error("Failed to load instrument data. Please ensure the broker connection is active and try again.")
        return

    st.sidebar.title("BlockVista Terminal")
    page = st.sidebar.radio("Navigate", ["F&O Analytics", "Option Strategy Builder"])

    if page == "F&O Analytics":
        page_fo_analytics()
    elif page == "Option Strategy Builder":
        page_option_strategy_builder()

if __name__ == "__main__":
    st.set_page_config(page_title="BlockVista Terminal", layout="wide")
    
    if 'authenticated' not in st.session_state or not st.session_state.get('authenticated'):
        login_page()
    else:
        main_app()
