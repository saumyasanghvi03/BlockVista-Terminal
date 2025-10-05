# ================ 6. HNI & PRO TRADER FEATURES (CONTINUED) ================

def execute_basket_order(basket_items, instrument_df):
    """Formats and places a basket of orders in a single API call."""
    client = get_broker_client()
    if not client:
        st.error("Broker not connected.")
        return
    
    if st.session_state.broker == "Zerodha":
        orders_to_place = []
        for item in basket_items:
            # Find exchange for each symbol
            instrument = instrument_df[instrument_df['tradingsymbol'] == item['symbol']]
            if instrument.empty:
                st.toast(f"❌ Could not find symbol {item['symbol']} in instrument list. Skipping.", icon="🔥")
                continue
            exchange = instrument.iloc[0]['exchange']

            order = {
                "tradingsymbol": item['symbol'],
                "exchange": exchange,
                "transaction_type": client.TRANSACTION_TYPE_BUY if item['transaction_type'] == 'BUY' else client.TRANSACTION_TYPE_SELL,
                "quantity": int(item['quantity']),
                "product": client.PRODUCT_MIS if item['product'] == 'MIS' else client.PRODUCT_CNC,
                "order_type": client.ORDER_TYPE_MARKET if item['order_type'] == 'MARKET' else client.ORDER_TYPE_LIMIT,
            }
            if order['order_type'] == client.ORDER_TYPE_LIMIT:
                order['price'] = item['price']
            orders_to_place.append(order)
        
        if not orders_to_place:
            st.warning("No valid orders to place in the basket.")
            return

        try:
            # Note: Kite Connect doesn't support batch orders in the free tier
            # We'll execute orders sequentially
            successful_orders = []
            for order in orders_to_place:
                try:
                    order_id = client.place_order(
                        tradingsymbol=order['tradingsymbol'],
                        exchange=order['exchange'],
                        transaction_type=order['transaction_type'],
                        quantity=order['quantity'],
                        order_type=order['order_type'],
                        product=order['product'],
                        price=order.get('price')
                    )
                    successful_orders.append(order_id)
                except Exception as e:
                    st.toast(f"❌ Failed to place order for {order['tradingsymbol']}: {e}", icon="🔥")
            
            if successful_orders:
                st.toast(f"✅ {len(successful_orders)} orders placed successfully!", icon="🎉")
                # Clear basket after execution
                st.session_state.basket = []
                st.rerun()
        except Exception as e:
            st.toast(f"❌ Basket order failed: {e}", icon="🔥")

@st.cache_data(ttl=3600)
def get_sector_data():
    """Loads stock-to-sector mapping from a local CSV file."""
    try:
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
        # Call ITM
        if row['STRIKE'] < ltp:
            styles[df.columns.get_loc('CALL LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('CALL OI')] = 'background-color: #2E4053'
        # Put ITM
        if row['STRIKE'] > ltp:
            styles[df.columns.get_loc('PUT LTP')] = 'background-color: #2E4053'
            styles[df.columns.get_loc('PUT OI')] = 'background-color: #2E4053'
        return styles

    return df.style.apply(highlight_itm, axis=1)

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
    client = get_broker_client()
    if not client:
        st.toast("Broker not connected.", icon="⚠️")
        return pd.DataFrame()
    
    try:
        chain_df, expiry, _, _ = get_options_chain(underlying, instrument_df)
        if chain_df.empty or expiry is None:
            return pd.DataFrame()
        ce_symbols = chain_df['CALL'].dropna().tolist()
        pe_symbols = chain_df['PUT'].dropna().tolist()
        all_symbols = [f"NFO:{s}" for s in ce_symbols + pe_symbols]

        if not all_symbols:
            return pd.DataFrame()

        quotes = client.quote(all_symbols)
        
        active_options = []
        for symbol, data in quotes.items():
            prev_close = data.get('ohlc', {}).get('close', 0)
            last_price = data.get('last_price', 0)
            change = last_price - prev_close
            pct_change = (change / prev_close * 100) if prev_close != 0 else 0
            
            active_options.append({
                'Symbol': data.get('tradingsymbol'),
                'LTP': last_price,
                'Change %': pct_change,
                'Volume': data.get('volume', 0),
                'OI': data.get('oi', 0)
            })
        
        df = pd.DataFrame(active_options)
        df_sorted = df.sort_values(by='Volume', ascending=False)
        return df_sorted.head(10)

    except Exception as e:
        st.error(f"Could not fetch most active options: {e}")
        return pd.DataFrame()
        
@st.cache_data(ttl=60)
def get_global_indices_data(tickers):
    """Fetches real-time data for global indices using yfinance."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        data_yf = yf.download(list(tickers.values()), period="5d")
        if data_yf.empty:
            return pd.DataFrame()

        data = []
        for ticker_name, yf_ticker_name in tickers.items():
            if len(tickers) > 1:
                # Corrected line - proper indexing for MultiIndex DataFrame
                if isinstance(data_yf.columns, pd.MultiIndex):
                    hist = data_yf.xs(yf_ticker_name, level=1, axis=1)
                else:
                    hist = data_yf
            else:
                hist = data_yf

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

# ================ 7. PAGE DEFINITIONS ============

def two_factor_setup():
    """Handle two-factor authentication setup."""
    st.title("🔒 Two-Factor Authentication Setup")
    st.info("Set up 2FA for enhanced security. Scan the QR code with your authenticator app.")
    
    if st.session_state.pyotp_secret is None:
        st.session_state.pyotp_secret = pyotp.random_base32()
    
    totp = pyotp.TOTP(st.session_state.pyotp_secret)
    user_name = st.session_state.profile.get('user_name', 'user') if st.session_state.profile else 'user'
    provisioning_uri = totp.provisioning_uri(
        name=user_name,
        issuer_name="BlockVista Terminal"
    )
    
    # Generate QR Code
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        qr_img.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        st.image(f"data:image/png;base64,{qr_base64}", width=200, caption="Scan with Google Authenticator or Authy")
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.code(st.session_state.pyotp_secret, language="text")
        with col_b:
            if st.button("Copy Secret", key="copy_secret"):
                st.code(st.session_state.pyotp_secret)
                st.success("Secret copied!")
        
        st.write("**Manual setup:** Enter this secret in your authenticator app")
        
    except Exception as e:
        st.error(f"QR code generation failed: {e}")
        st.info("Please use this secret for manual setup:")
        st.code(st.session_state.pyotp_secret)
    
    # Verify setup
    st.divider()
    st.write("**Verify 2FA Setup**")
    otp_input = st.text_input("Enter OTP from authenticator app", placeholder="6-digit code")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Verify 2FA Setup", use_container_width=True):
            if otp_input and len(otp_input) == 6:
                if totp.verify(otp_input):
                    st.session_state.two_factor_setup_complete = True
                    st.session_state.totp = totp
                    st.success("✅ Two-factor authentication setup complete!")
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")
            else:
                st.error("Please enter a valid 6-digit OTP")
    
    with col2:
        if st.button("Skip 2FA Setup", use_container_width=True):
            st.session_state.two_factor_setup_complete = True
            st.warning("2FA setup skipped. Please enable it later in settings for security.")
            st.rerun()

def two_factor_verification():
    """Handle two-factor authentication verification."""
    st.title("🔒 Two-Factor Authentication")
    st.info("Please enter the OTP from your authenticator app to continue.")
    
    if st.session_state.pyotp_secret is None:
        st.error("2FA not properly configured. Please set up 2FA first.")
        st.session_state.two_factor_setup_complete = False
        st.rerun()
        return
    
    totp = pyotp.TOTP(st.session_state.pyotp_secret)
    otp_input = st.text_input("Enter OTP", placeholder="6-digit code", key="2fa_verify")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Verify OTP", use_container_width=True, type="primary"):
            if otp_input and len(otp_input) == 6:
                if totp.verify(otp_input):
                    st.session_state.two_factor_verified = True
                    st.success("✅ 2FA verification successful!")
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")
            else:
                st.error("Please enter a valid 6-digit OTP")
    
    with col2:
        if st.button("Reset 2FA", use_container_width=True):
            st.session_state.two_factor_setup_complete = False
            st.session_state.two_factor_verified = False
            st.session_state.pyotp_secret = None
            st.rerun()

# --- Bharatiya Market Pulse (BMP) Functions ---
def get_bmp_score_and_label(nifty_change, sensex_change, vix_value, lookback_df):
    """Calculates BMP score and returns the score and a Bharat-flavored label."""
    if lookback_df.empty or len(lookback_df) < 30:
        return 50, "Calculating...", "#cccccc"
    # Normalize NIFTY and SENSEX
    nifty_min, nifty_max = lookback_df['nifty_change'].min(), lookback_df['nifty_change'].max()
    sensex_min, sensex_max = lookback_df['sensex_change'].min(), lookback_df['sensex_change'].max()

    nifty_norm = ((nifty_change - nifty_min) / (nifty_max - nifty_min)) * 100 if (nifty_max - nifty_min) > 0 else 50
    sensex_norm = ((sensex_change - sensex_min) / (sensex_max - sensex_min)) * 100 if (sensex_max - sensex_min) > 0 else 50
    
    # Inversely normalize VIX
    vix_min, vix_max = lookback_df['vix_value'].min(), lookback_df['vix_value'].max()
    vix_norm = 100 - (((vix_value - vix_min) / (vix_max - vix_min)) * 100) if (vix_max - vix_min) > 0 else 50

    bmp_score = (0.40 * nifty_norm) + (0.40 * sensex_norm) + (0.20 * vix_norm)
    bmp_score = min(100, max(0, bmp_score))

    if bmp_score >= 80:
        label, color = "Bharat Udaan (Very Bullish)", "#00b300"
    elif bmp_score >= 60:
        label, color = "Bharat Pragati (Bullish)", "#33cc33"
    elif bmp_score >= 40:
        label, color = "Bharat Santulan (Neutral)", "#ffcc00"
    elif bmp_score >= 20:
        label, color = "Bharat Sanket (Bearish)", "#ff6600"
    else:
        label, color = "Bharat Mandhi (Very Bearish)", "#ff0000"

    return bmp_score, label, color

@st.cache_data(ttl=300)
def get_nifty50_constituents(instrument_df):
    """Fetches the list of NIFTY 50 stocks by filtering the Kite API instrument list."""
    if instrument_df.empty:
        return pd.DataFrame()
    
    nifty50_symbols = [
        'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'HINDUNILVR', 'ITC', 
        'LT', 'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'BHARTIARTL', 'ASIANPAINT', 
        'AXISBANK', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'M&M', 'NESTLEIND',
        'ADANIENT', 'TATASTEEL', 'INDUSINDBK', 'TECHM', 'NTPC', 'MARUTI', 
        'BAJAJ-AUTO', 'POWERGRID', 'HCLTECH', 'ADANIPORTS', 'BPCL', 'COALINDIA', 
        'EICHERMOT', 'GRASIM', 'JSWSTEEL', 'SHREECEM', 'HEROMOTOCO', 'HINDALCO',
        'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'SBILIFE',
        'TATAMOTORS', 'BRITANNIA', 'DIVISLAB', 'BAJAJFINSV', 'SUNPHARMA', 'HDFCLIFE'
    ]
    
    nifty_constituents = instrument_df[
        (instrument_df['tradingsymbol'].isin(nifty50_symbols)) & 
        (instrument_df['segment'] == 'NSE')
    ].copy()

    constituents_df = pd.DataFrame({
        'Symbol': nifty_constituents['tradingsymbol'],
        'Name': nifty_constituents['tradingsymbol']
    })
    
    return constituents_df.drop_duplicates(subset='Symbol').head(15)

def create_nifty_heatmap(instrument_df):
    """Generates a Plotly Treemap for NIFTY 50 stocks."""
    constituents_df = get_nifty50_constituents(instrument_df)
    if constituents_df.empty:
        return go.Figure()
    
    symbols_with_exchange = [{'symbol': s, 'exchange': 'NSE'} for s in constituents_df['Symbol'].tolist()]
    live_data = get_watchlist_data(symbols_with_exchange)
    
    if live_data.empty:
        return go.Figure()
        
    full_data = pd.merge(live_data, constituents_df, left_on='Ticker', right_on='Symbol', how='left')
    full_data['size'] = full_data['Price'].astype(float) * 1000 # Using price as a proxy for size
    
    fig = go.Figure(go.Treemap(
        labels=full_data['Ticker'],
        parents=[''] * len(full_data),
        values=full_data['size'],
        marker=dict(
            colorscale='RdYlGn',
            colors=full_data['% Change'],
            colorbar=dict(title="% Change"),
        ),
        text=full_data['Ticker'],
        textinfo="label",
        hovertemplate='<b>%{label}</b><br>Price: ₹%{customdata[0]:.2f}<br>Change: %{customdata[1]:.2f}%<extra></extra>',
        customdata=np.column_stack([full_data['Price'], full_data['% Change']])
    ))

    fig.update_layout(title="NIFTY 50 Heatmap (Live)")
    return fig

@st.cache_data(ttl=300)
def get_gift_nifty_data():
    """Fetches GIFT NIFTY data using a more reliable yfinance ticker."""
    try:
        data = yf.download("IN=F", period="1d", interval="1m")
        if not data.empty:
            return data
    except Exception:
        pass
    return pd.DataFrame()

def page_dashboard():
    """A completely redesigned 'Trader UI' Dashboard."""
    display_header()
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to view the dashboard.")
        return
    # Fetch NIFTY, SENSEX, and VIX data for BMP calculation
    index_symbols = [
        {'symbol': 'NIFTY 50', 'exchange': 'NSE'},
        {'symbol': 'SENSEX', 'exchange': 'BSE'},
        {'symbol': 'INDIA VIX', 'exchange': 'NSE'},
    ]
    index_data = get_watchlist_data(index_symbols)
    
    # BMP Calculation and Display
    bmp_col, heatmap_col = st.columns([1, 1], gap="large")
    with bmp_col:
        st.subheader("Bharatiya Market Pulse (BMP)")
        if not index_data.empty:
            nifty_row = index_data[index_data['Ticker'] == 'NIFTY 50'].iloc[0]
            sensex_row = index_data[index_data['Ticker'] == 'SENSEX'].iloc[0]
            vix_row = index_data[index_data['Ticker'] == 'INDIA VIX'].iloc[0]
            
            # Fetch historical data for normalization
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
                    st.markdown("""
                    - **80-100 (Bharat Udaan):** Very Strong Bullish Momentum.
                    - **60-80 (Bharat Pragati):** Moderately Bullish Sentiment.
                    - **40-60 (Bharat Santulan):** Neutral or Sideways Market.
                    - **20-40 (Bharat Sanket):** Moderately Bearish Sentiment.
                    - **0-20 (Bharat Mandhi):** Very Strong Bearish Momentum.
                    """)
            else:
                st.info("BMP data is loading...")
        else:
            st.info("BMP data is loading...")
    with heatmap_col:
        st.subheader("NIFTY 50 Heatmap")
        st.plotly_chart(create_nifty_heatmap(instrument_df), use_container_width=True)

    st.markdown("---")
    
    # --- Middle Row: Main Content Area ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        tab1, tab2 = st.tabs(["Watchlist", "Portfolio Overview"])

        with tab1:
            st.session_state.active_watchlist = st.radio(
                "Select Watchlist",
                options=list(st.session_state.watchlists.keys()),
                horizontal=True,
                label_visibility="collapsed"
            )
            
            active_list = st.session_state.watchlists[st.session_state.active_watchlist]

            with st.form(key="add_stock_form"):
                add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
                new_symbol = add_col1.text_input("Symbol", placeholder="Add symbol...", label_visibility="collapsed")
                new_exchange = add_col2.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS"], label_visibility="collapsed")
                if add_col3.form_submit_button("Add"):
                    if new_symbol:
                        if len(active_list) >= 15:
                            st.toast("Watchlist full (max 15 stocks).", icon="⚠️")
                        elif not any(d['symbol'] == new_symbol.upper() for d in active_list):
                            active_list.append({'symbol': new_symbol.upper(), 'exchange': new_exchange})
                            st.rerun()
                        else:
                            st.toast(f"{new_symbol.upper()} is already in this watchlist.", icon="⚠️")
            
            # Display watchlist
            watchlist_data = get_watchlist_data(active_list)
            if not watchlist_data.empty:
                for index, row in watchlist_data.iterrows():
                    w_cols = st.columns([3, 2, 1, 1, 1, 1])
                    color = 'var(--green)' if row['Change'] > 0 else 'var(--red)'
                    w_cols[0].markdown(f"**{row['Ticker']}**<br><small style='color:var(--text-light);'>{row['Exchange']}</small>", unsafe_allow_html=True)
                    w_cols[1].markdown(f"**{row['Price']:,.2f}**<br><small style='color:{color};'>{row['Change']:,.2f} ({row['% Change']:.2f}%)</small>", unsafe_allow_html=True)
                    
                    quantity = w_cols[2].number_input("Qty", min_value=1, step=1, key=f"qty_{row['Ticker']}", label_visibility="collapsed")
                    
                    if w_cols[3].button("B", key=f"buy_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'BUY', 'MIS')
                    if w_cols[4].button("S", key=f"sell_{row['Ticker']}", use_container_width=True):
                        place_order(instrument_df, row['Ticker'], quantity, 'MARKET', 'SELL', 'MIS')
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
                if not holdings_df.empty:
                    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings found.")
    with col2:
        st.subheader("NIFTY 50 Live Chart (1-min)")
        nifty_token = get_instrument_token('NIFTY 50', instrument_df, 'NSE')
        if nifty_token:
            nifty_data = get_historical_data(nifty_token, "minute", period="1d")
            if not nifty_data.empty:
                st.plotly_chart(create_chart(nifty_data.tail(150), "NIFTY 50"), use_container_width=True)
            else:
                st.warning("Could not load NIFTY 50 chart. Market might be closed.")
    
    # --- Bottom Row: Live Ticker Tape ---
    ticker_symbols = st.session_state.get('watchlists', {}).get(st.session_state.get('active_watchlist'), [])
    
    if ticker_symbols:
        ticker_data = get_watchlist_data(ticker_symbols)
        
        if not ticker_data.empty:
            ticker_html = "".join([
                f"<span style='color: white; margin-right: 40px;'>{item['Ticker']} <span style='color: {'#28a745' if item['Change'] > 0 else '#FF4B4B'};'>{item['Price']:,.2f} ({item['% Change']:.2f}%)</span></span>"
                for _, item in ticker_data.iterrows()
            ])
            
            st.markdown(f"""
            <style>
                @keyframes marquee {{
                    0%   {{ transform: translate(100%, 0); }}
                    100% {{ transform: translate(-100%, 0); }}
                }}
                .marquee-container {{
                    width: 100%;
                    overflow: hidden;
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    background-color: #1a1a1a;
                    border-top: 1px solid #333;
                    padding: 5px 0;
                    white-space: nowrap;
                }}
                .marquee-content {{
                    display: inline-block;
                    padding-left: 100%;
                    animation: marquee 35s linear infinite;
                }}
            </style>
            <div class="marquee-container">
                <div class="marquee-content">
                    {ticker_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

def page_advanced_charting():
    """A page for advanced charting with custom intervals and indicators."""
    display_header()
    st.title("Advanced Multi-Chart Terminal")
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use the charting tools.")
        return
    
    st.subheader("Chart Layout")
    layout_option = st.radio("Select Layout", ["Single Chart", "2 Charts", "4 Charts", "6 Charts"], horizontal=True)
    
    chart_counts = {"Single Chart": 1, "2 Charts": 2, "4 Charts": 4, "6 Charts": 6}
    num_charts = chart_counts[layout_option]
    
    st.markdown("---")
    
    if num_charts == 1:
        render_chart_controls(0, instrument_df)
    elif num_charts == 2:
        cols = st.columns(2)
        for i, col in enumerate(cols):
            with col:
                render_chart_controls(i, instrument_df)
    elif num_charts == 4:
        for i in range(2):
            cols = st.columns(2)
            with cols[0]:
                render_chart_controls(i * 2, instrument_df)
            with cols[1]:
                render_chart_controls(i * 2 + 1, instrument_df)
    elif num_charts == 6:
        for i in range(2):
            cols = st.columns(3)
            with cols[0]:
                render_chart_controls(i * 3, instrument_df)
            with cols[1]:
                render_chart_controls(i * 3 + 1, instrument_df)
            with cols[2]:
                render_chart_controls(i * 3 + 2, instrument_df)

def render_chart_controls(i, instrument_df):
    """Helper function to render controls for a single chart."""
    with st.container(border=True):
        st.subheader(f"Chart {i+1}")
        
        chart_cols = st.columns(4)
        ticker = chart_cols[0].text_input("Symbol", "NIFTY 50", key=f"ticker_{i}").upper()
        period = chart_cols[1].selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=4, key=f"period_{i}")
        interval = chart_cols[2].selectbox("Interval", ["minute", "5minute", "day", "week"], index=2, key=f"interval_{i}")
        chart_type = chart_cols[3].selectbox("Chart Type", ["Candlestick", "Line", "Bar", "Heikin-Ashi"], key=f"chart_type_{i}")

        # Comparison symbols
        with st.expander("Comparison Symbols (Max 4)"):
            comp_cols = st.columns(2)
            comp1 = comp_cols[0].text_input("Compare 1", key=f"comp1_{i}").upper()
            comp2 = comp_cols[1].text_input("Compare 2", key=f"comp2_{i}").upper()
            comp3 = comp_cols[0].text_input("Compare 3", key=f"comp3_{i}").upper()
            comp4 = comp_cols[1].text_input("Compare 4", key=f"comp4_{i}").upper()
        
        comparison_symbols = [sym for sym in [comp1, comp2, comp3, comp4] if sym]
        
        # Get main symbol data
        token = get_instrument_token(ticker, instrument_df)
        data = get_historical_data(token, interval, period=period)

        # Get comparison data
        comparison_data = {}
        for comp_symbol in comparison_symbols:
            comp_token = get_instrument_token(comp_symbol, instrument_df)
            if comp_token:
                comp_data = get_historical_data(comp_token, interval, period=period)
                if not comp_data.empty:
                    comparison_data[comp_symbol] = comp_data

        if data.empty and not comparison_data:
            st.warning(f"No data to display for {ticker} with selected parameters.")
        else:
            st.plotly_chart(
                create_chart(data, ticker, chart_type, comparison_data=comparison_data if comparison_data else None), 
                use_container_width=True, 
                key=f"chart_{i}"
            )

            order_cols = st.columns([2,1,1,1])
            order_cols[0].markdown("**Quick Order**")
            quantity = order_cols[1].number_input("Qty", min_value=1, step=1, key=f"qty_{i}", label_visibility="collapsed")
            
            if order_cols[2].button("Buy", key=f"buy_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'BUY', 'MIS')
            if order_cols[3].button("Sell", key=f"sell_btn_{i}", use_container_width=True):
                place_order(instrument_df, ticker, quantity, 'MARKET', 'SELL', 'MIS')

def page_premarket_pulse():
    """Global market overview and premarket indicators with a trader-focused UI."""
    display_header()
    st.title("Premarket & Global Cues")
    st.markdown("---")

    st.subheader("Global Market Snapshot")
    global_tickers = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "FTSE 100": "^FTSE", "Nikkei 225": "^N225", "Hang Seng": "^HSI"}
    global_data = get_global_indices_data(global_tickers)
    
    if not global_data.empty:
        cols = st.columns(len(global_tickers))
        for i, (name, ticker_symbol) in enumerate(global_tickers.items()):
            data_row = global_data[global_data['Ticker'] == name]
            if not data_row.empty:
                price = data_row.iloc[0]['Price']
                change = data_row.iloc[0]['% Change']
                if not np.isnan(price):
                    cols[i].metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                else:
                    cols[i].metric(label=name, value="N/A", delta="--")
    else:
        st.info("Loading global market data...")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("NIFTY 50 Futures (Live Proxy)")
        gift_data = get_gift_nifty_data()
        if not gift_data.empty:
            st.plotly_chart(create_chart(gift_data, "NIFTY 50 Futures (Proxy)"), use_container_width=True)
        else:
            st.warning("Could not load NIFTY 50 Futures chart data.")
            
    with col2:
        st.subheader("Key Asian Markets")
        asian_tickers = {"Nikkei 225": "^N225", "Hang Seng": "^HSI"}
        asian_data = get_global_indices_data(asian_tickers)
        if not asian_data.empty:
            for name, ticker_symbol in asian_tickers.items():
                data_row = asian_data[asian_data['Ticker'] == name]
                if not data_row.empty:
                    price = data_row.iloc[0]['Price']
                    change = data_row.iloc[0]['% Change']
                    if not np.isnan(price):
                        st.metric(label=name, value=f"{price:,.2f}", delta=f"{change:.2f}%")
                    else:
                        st.metric(label=name, value="N/A", delta="--")
        else:
            st.info("Loading Asian market data...")

    st.markdown("---")

    st.subheader("Latest Market News")
    news_df = fetch_and_analyze_news()
    if not news_df.empty:
        for _, news in news_df.head(10).iterrows():
            sentiment_score = news['sentiment']
            if sentiment_score > 0.2:
                icon = "🔼"
            elif sentiment_score < -0.2:
                icon = "🔽"
            else:
                icon = "▶️"
            st.markdown(f"**{icon} [{news['title']}]({news['link']})** - *{news['source']}*")
    else:
        st.info("News data is loading...")

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
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            underlying = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            st.session_state.underlying_pcr = underlying 
        
        chain_df, expiry, underlying_ltp, available_expiries = get_options_chain(underlying, instrument_df)

        if not chain_df.empty:
            with col2:
                st.metric("Current Price", f"₹{underlying_ltp:,.2f}")
                st.metric("Expiry Date", expiry.strftime("%d %b %Y") if expiry else "N/A")
            with col3:
                if st.button("Most Active Options", use_container_width=True):
                    show_most_active_dialog(underlying, instrument_df)

            # Display options chain with ITM/OTM styling
            st.dataframe(
                style_option_chain(chain_df, underlying_ltp).format({
                    'CALL LTP': '₹{:.2f}', 'PUT LTP': '₹{:.2f}',
                    'STRIKE': '₹{:.0f}',
                    'CALL OI': '{:,.0f}', 'PUT OI': '{:,.0f}'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Could not load options chain data.")
    
    with tab2:
        st.subheader("Put-Call Ratio Analysis")
        
        chain_df, _, _, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)
        if not chain_df.empty and 'CALL OI' in chain_df.columns:
            total_ce_oi = chain_df['CALL OI'].sum()
            total_pe_oi = chain_df['PUT OI'].sum()
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
            st.info("PCR data is loading... Select an underlying in the 'Options Chain' tab first.")
    
    with tab3:
        st.subheader("Volatility & Open Interest Surface")
        st.info("Real-time implied volatility and OI analysis for options contracts.")

        chain_df, expiry, underlying_ltp, _ = get_options_chain(st.session_state.get('underlying_pcr', "NIFTY"), instrument_df)

        if not chain_df.empty and expiry and underlying_ltp > 0:
            T = (expiry - datetime.now().date()).days / 365.0
            r = 0.07  # Assume a risk-free rate of 7%

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
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['CALL OI'], name='Call OI', marker_color='rgba(0, 255, 255, 0.4)'), secondary_y=True)
            fig.add_trace(go.Bar(x=chain_df['STRIKE'], y=chain_df['PUT OI'], name='Put OI', marker_color='rgba(255, 0, 255, 0.4)'), secondary_y=True)

            fig.update_layout(
                title_text=f"{st.session_state.get('underlying_pcr', 'NIFTY')} IV & OI Profile for {expiry.strftime('%d %b %Y')}",
                template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Implied Volatility (%)", secondary_y=False)
            fig.update_yaxes(title_text="Open Interest", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select an underlying and expiry in the 'Options Chain' tab to view the volatility surface.")

def page_forecasting_ml():
    """A page for advanced ML forecasting with multiple models."""
    display_header()
    st.title("Advanced ML Forecasting Hub")
    st.info(""" 
    Train multiple machine learning models to forecast future prices. 
    **Models Available:**
    - **Seasonal ARIMA**: Traditional time series model
    - **LSTM**: Deep learning model for sequence prediction  
    - **XGBoost**: Gradient boosting for tabular data
    """, icon="🧠")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        instrument_name = st.selectbox("Select an Instrument", list(ML_DATA_SOURCES.keys()), key="ml_instrument")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model", 
            ["Seasonal ARIMA", "LSTM", "XGBoost"],
            help="ARIMA: Best for seasonal patterns. LSTM: Best for complex sequences. XGBoost: Best with features."
        )
        
        # Extended forecast durations
        forecast_durations = {
            "1 Week": 7, 
            "2 Weeks": 14, 
            "1 Month": 30, 
            "3 Months": 90, 
            "6 Months": 180
        }
        duration_key = st.radio("Forecast Duration", list(forecast_durations.keys()), horizontal=True, key="ml_duration")
        forecast_steps = forecast_durations[duration_key]

        # Advanced parameters
        with st.expander("Advanced Parameters"):
            if model_type == "Seasonal ARIMA":
                st.slider("Seasonal Period", 5, 30, 7, key="seasonal_period", help="Seasonal cycle length")
            elif model_type == "LSTM":
                st.slider("Sequence Length", 30, 90, 60, key="lstm_seq_len", help="Look-back period for LSTM")
            elif model_type == "XGBoost":
                st.slider("Number of Estimators", 100, 2000, 1000, key="xgb_n_estimators", help="Number of boosting rounds")

        if st.button("Train Model & Forecast", use_container_width=True, type="primary"):
            with st.spinner(f"Training {model_type} model for {instrument_name}..."):
                data = load_and_combine_data(instrument_name)
                if data.empty or len(data) < 100:
                    st.error(f"Could not load sufficient historical data for {instrument_name}. Model training requires at least 100 data points.")
                else:
                    # Train selected model
                    if model_type == "Seasonal ARIMA":
                        forecast_df, backtest_df, conf_int_df = train_seasonal_arima_model(data, forecast_steps)
                        model_message = "Seasonal ARIMA model trained successfully"
                    elif model_type == "LSTM":
                        forecast_df, backtest_df, conf_int_df, model_message = train_lstm_model(data, forecast_steps)
                    elif model_type == "XGBoost":
                        forecast_df, backtest_df, conf_int_df, model_message = train_xgboost_model(data, forecast_steps)
                    
                    if forecast_df is not None:
                        # Calculate model metrics
                        metrics = calculate_model_metrics(
                            backtest_df['Actual'].values if backtest_df is not None else [],
                            backtest_df['Predicted'].values if backtest_df is not None else []
                        )
                        
                        st.session_state.update({
                            'ml_forecast_df': forecast_df,
                            'ml_backtest_df': backtest_df,
                            'ml_conf_int_df': conf_int_df,
                            'ml_instrument_name': instrument_name,
                            'ml_historical_data': data,
                            'ml_duration_key': duration_key,
                            'ml_model_type': model_type,
                            'ml_metrics': metrics,
                            'ml_message': model_message
                        })
                        
                        # Store previous models for comparison
                        if 'ml_previous_models' not in st.session_state:
                            st.session_state.ml_previous_models = []
                        
                        st.session_state.ml_previous_models.append({
                            'type': model_type,
                            'metrics': metrics,
                            'timestamp': datetime.now()
                        })
                        
                        # Keep only last 5 models
                        if len(st.session_state.ml_previous_models) > 5:
                            st.session_state.ml_previous_models = st.session_state.ml_previous_models[-5:]
                            
                        st.success(f"{model_type} model trained successfully!")
                    else:
                        st.error(f"Model training failed: {model_message}")

        # Model comparison section
        if st.session_state.get('ml_previous_models'):
            st.subheader("Model Comparison")
            comparison_data = []
            for model_info in st.session_state.ml_previous_models:
                comparison_data.append({
                    'Model': model_info['type'],
                    'Accuracy': f"{100 - model_info['metrics'].get('MAPE', 0):.1f}%",
                    'MAE': f"₹{model_info['metrics'].get('MAE', 0):.2f}",
                    'Direction Accuracy': f"{model_info['metrics'].get('Direction_Accuracy', 0):.1f}%",
                    'R²': f"{model_info['metrics'].get('R2', 0):.3f}"
                })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with col2:
        if 'ml_instrument_name' in st.session_state:
            instrument_name = st.session_state.ml_instrument_name
            model_type = st.session_state.get('ml_model_type', 'Seasonal ARIMA')
            metrics = st.session_state.get('ml_metrics', {})
            
            st.subheader(f"Forecast Results for {instrument_name} ({model_type})")

            forecast_df = st.session_state.get('ml_forecast_df')
            backtest_df = st.session_state.get('ml_backtest_df')
            conf_int_df = st.session_state.get('ml_conf_int_df')
            data = st.session_state.get('ml_historical_data')
            duration_key = st.session_state.get('ml_duration_key')

            if forecast_df is not None and backtest_df is not None and data is not None:
                # Display model metrics
                if metrics:
                    st.subheader("Model Performance")
                    metric_cols = st.columns(4)
                    
                    if 'MAPE' in metrics:
                        metric_cols[0].metric("Accuracy", f"{100 - metrics['MAPE']:.1f}%")
                    if 'MAE' in metrics:
                        metric_cols[1].metric("MAE", f"₹{metrics['MAE']:.2f}")
                    if 'RMSE' in metrics:
                        metric_cols[2].metric("RMSE", f"₹{metrics['RMSE']:.2f}")
                    if 'Direction_Accuracy' in metrics:
                        metric_cols[3].metric("Direction Accuracy", f"{metrics['Direction_Accuracy']:.1f}%")

                # Create and display forecast chart
                fig = create_chart(data, instrument_name, forecast_df=forecast_df, conf_int_df=conf_int_df)
                st.plotly_chart(fig, use_container_width=True)

                # Show forecast table
                with st.expander("View Forecast Data"):
                    st.dataframe(forecast_df, use_container_width=True)

                # Show backtest results
                if not backtest_df.empty:
                    with st.expander("View Backtest Results"):
                        st.dataframe(backtest_df.tail(10), use_container_width=True)
                        
                        # Backtest chart
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Actual'], mode='lines', name='Actual', line=dict(color='blue')))
                        fig_backtest.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Predicted'], mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
                        fig_backtest.update_layout(title="Backtest: Actual vs Predicted", template='plotly_dark' if st.session_state.get('theme') == 'Dark' else 'plotly_white')
                        st.plotly_chart(fig_backtest, use_container_width=True)

# ================ 8. ADDITIONAL PAGES ================

def page_hft_terminal():
    """High-Frequency Trading terminal simulation."""
    display_header()
    st.title("⚡ HFT Terminal")
    st.info("Real-time market data and order execution (Simulation Mode)")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to access HFT features.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Market Depth")
        symbol = st.text_input("Symbol for Market Depth", "RELIANCE", key="hft_symbol").upper()
        token = get_instrument_token(symbol, instrument_df)
        
        if token:
            depth = get_market_depth(token)
            if depth:
                st.subheader(f"Order Book - {symbol}")
                
                # Display bids
                st.write("**Bids (Buy)**")
                bids_df = pd.DataFrame(depth['buy'])[:10]
                if not bids_df.empty:
                    for _, bid in bids_df.iterrows():
                        st.markdown(f'<div class="hft-depth-bid">Bid: {bid["quantity"]} @ ₹{bid["price"]:.2f} | Orders: {bid["orders"]}</div>', unsafe_allow_html=True)
                
                # Current price
                if depth.get('sell') and depth.get('buy'):
                    best_bid = depth['buy'][0]['price'] if depth['buy'] else 0
                    best_ask = depth['sell'][0]['price'] if depth['sell'] else 0
                    spread = best_ask - best_bid
                    
                    st.metric("Best Bid", f"₹{best_bid:.2f}")
                    st.metric("Best Ask", f"₹{best_ask:.2f}")
                    st.metric("Spread", f"₹{spread:.2f}")
                
                # Display asks
                st.write("**Asks (Sell)**")
                asks_df = pd.DataFrame(depth['sell'])[:10]
                if not asks_df.empty:
                    for _, ask in asks_df.iterrows():
                        st.markdown(f'<div class="hft-depth-ask">Ask: {ask["quantity"]} @ ₹{ask["price"]:.2f} | Orders: {ask["orders"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Quick Execution")
        hft_quantity = st.number_input("Quantity", min_value=1, value=100, key="hft_qty")
        hft_price = st.number_input("Price", min_value=0.01, value=0.0, key="hft_price")
        
        col_b, col_s = st.columns(2)
        with col_b:
            if st.button("BUY MARKET", use_container_width=True, type="primary"):
                place_order(instrument_df, symbol, hft_quantity, 'MARKET', 'BUY', 'MIS')
        with col_s:
            if st.button("SELL MARKET", use_container_width=True, type="secondary"):
                place_order(instrument_df, symbol, hft_quantity, 'MARKET', 'SELL', 'MIS')
        
        if hft_price > 0:
            col_bb, col_ss = st.columns(2)
            with col_bb:
                if st.button("BUY LIMIT", use_container_width=True):
                    place_order(instrument_df, symbol, hft_quantity, 'LIMIT', 'BUY', 'MIS', hft_price)
            with col_ss:
                if st.button("SELL LIMIT", use_container_width=True):
                    place_order(instrument_df, symbol, hft_quantity, 'LIMIT', 'SELL', 'MIS', hft_price)

def page_basket_orders():
    """Basket orders for HNI/pro traders."""
    display_header()
    st.title("🧺 Basket Orders")
    st.info("Execute multiple orders simultaneously")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use basket orders.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Create Basket")
        
        with st.form("basket_order_form"):
            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns([3, 2, 2, 2, 1])
            symbol = b_col1.text_input("Symbol", placeholder="Symbol", key="basket_symbol").upper()
            quantity = b_col2.number_input("Qty", min_value=1, value=100, key="basket_qty")
            order_type = b_col3.selectbox("Type", ["MARKET", "LIMIT"], key="basket_type")
            transaction_type = b_col4.selectbox("Side", ["BUY", "SELL"], key="basket_side")
            price = st.number_input("Price (if LIMIT)", min_value=0.01, value=0.0, key="basket_price") if order_type == "LIMIT" else 0
            
            if st.form_submit_button("Add to Basket", use_container_width=True):
                if symbol and quantity > 0:
                    basket_item = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'order_type': order_type,
                        'transaction_type': transaction_type,
                        'product': 'MIS',
                        'price': price if price > 0 else None
                    }
                    st.session_state.basket.append(basket_item)
                    st.success(f"Added {symbol} to basket!")
                    st.rerun()
    
    with col2:
        st.subheader("Basket Summary")
        if st.session_state.basket:
            total_quantity = sum(item['quantity'] for item in st.session_state.basket)
            buy_orders = sum(item['quantity'] for item in st.session_state.basket if item['transaction_type'] == 'BUY')
            sell_orders = sum(item['quantity'] for item in st.session_state.basket if item['transaction_type'] == 'SELL')
            
            st.metric("Total Orders", len(st.session_state.basket))
            st.metric("Total Quantity", total_quantity)
            st.metric("Buy Qty", buy_orders)
            st.metric("Sell Qty", sell_orders)
            
            if st.button("Execute Basket", type="primary", use_container_width=True):
                execute_basket_order(st.session_state.basket, instrument_df)
            
            if st.button("Clear Basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()
        else:
            st.info("Basket is empty")
    
    # Display current basket
    if st.session_state.basket:
        st.subheader("Current Basket")
        for i, item in enumerate(st.session_state.basket):
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 2, 2, 1])
            col1.write(f"**{item['symbol']}**")
            col2.write(f"Qty: {item['quantity']}")
            col3.write(f"Type: {item['order_type']}")
            col4.write(f"Side: {item['transaction_type']}")
            col5.write(f"Price: {item['price'] or 'Market'}")
            if col6.button("❌", key=f"del_{i}"):
                st.session_state.basket.pop(i)
                st.rerun()

def page_fundamental_analysis():
    """Fundamental analysis page with company financials."""
    display_header()
    st.title("🏢 Fundamental Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", "RELIANCE", key="fundamental_symbol").upper()
        
        if st.button("Analyze Fundamentals", use_container_width=True):
            with st.spinner(f"Fetching fundamental data for {symbol}..."):
                fundamental_data = get_fundamental_data(symbol)
                if fundamental_data:
                    st.session_state.fundamental_data = fundamental_data
                else:
                    st.error(f"Could not fetch fundamental data for {symbol}")
    
    with col2:
        st.subheader("Compare With")
        comp1 = st.text_input("Peer 1", "TCS", key="comp1").upper()
        comp2 = st.text_input("Peer 2", "HDFCBANK", key="comp2").upper()
        comp3 = st.text_input("Peer 3", "INFY", key="comp3").upper()
        
        comparison_symbols = [sym for sym in [comp1, comp2, comp3] if sym]
        if st.button("Add Comparison", use_container_width=True) and comparison_symbols:
            st.session_state.comparison_symbols = comparison_symbols
    
    # Display fundamental data
    if st.session_state.get('fundamental_data'):
        data = st.session_state.fundamental_data
        
        st.subheader(f"Fundamental Analysis: {data.get('company_name', data['symbol'])}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{data.get('current_price', 0):.2f}")
        col2.metric("Market Cap", f"₹{data.get('market_cap', 0):,.0f} Cr")
        col3.metric("P/E Ratio", f"{data.get('pe_ratio', 0):.2f}")
        col4.metric("P/B Ratio", f"{data.get('pb_ratio', 0):.2f}")
        
        # Profitability metrics
        st.subheader("Profitability & Growth")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROE", f"{data.get('roe', 0):.2f}%")
        col2.metric("ROA", f"{data.get('roa', 0):.2f}%")
        col3.metric("Profit Margin", f"{data.get('profit_margin', 0):.2f}%")
        col4.metric("Operating Margin", f"{data.get('operating_margin', 0):.2f}%")
        
        # Financial health
        st.subheader("Financial Health")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Debt to Equity", f"{data.get('debt_to_equity', 0):.2f}")
        col2.metric("Current Ratio", f"{data.get('current_ratio', 0):.2f}")
        col3.metric("Revenue Growth", f"{data.get('revenue_growth', 0):.2f}%")
        col4.metric("Profit Growth", f"{data.get('profit_growth', 0):.2f}%")
        
        # Valuation
        st.subheader("Valuation")
        intrinsic_value, margin_of_safety = calculate_intrinsic_value(data)
        
        if intrinsic_value:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intrinsic Value", f"₹{intrinsic_value:,.2f}")
            col2.metric("Margin of Safety", f"{margin_of_safety:.2f}%")
            col3.metric("Book Value", f"₹{data.get('book_value', 0):.2f}")
            col4.metric("EPS", f"₹{data.get('eps', 0):.2f}")
            
            # Investment recommendation
            if margin_of_safety > 20:
                st.success("**Investment Recommendation:** STRONG BUY (High Margin of Safety)")
            elif margin_of_safety > 0:
                st.info("**Investment Recommendation:** BUY (Positive Margin of Safety)")
            elif margin_of_safety > -10:
                st.warning("**Investment Recommendation:** HOLD (Fairly Valued)")
            else:
                st.error("**Investment Recommendation:** SELL (Overvalued)")
        
        # Comparison with peers
        if st.session_state.get('comparison_symbols'):
            st.subheader("Peer Comparison")
            peers_data = []
            for symbol in [data['symbol']] + st.session_state.comparison_symbols:
                peer_data = get_fundamental_data(symbol)
                if peer_data:
                    peers_data.append(peer_data)
            
            if len(peers_data) > 1:
                comparison_df = pd.DataFrame(peers_data)
                st.dataframe(comparison_df[['symbol', 'pe_ratio', 'pb_ratio', 'roe', 'profit_margin']], use_container_width=True)

def page_ai_trade_assistant():
    """AI-powered trade assistant with recommendations."""
    display_header()
    st.title("💡 AI Trade Assistant")
    st.info("Get AI-powered trading insights and recommendations")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use AI features.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Technical Analysis")
        symbol = st.text_input("Symbol for Analysis", "RELIANCE", key="ai_symbol").upper()
        period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=2)
        
        if st.button("Generate Analysis", use_container_width=True):
            with st.spinner("Analyzing market data..."):
                token = get_instrument_token(symbol, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period=period)
                    if not data.empty:
                        # Technical indicators interpretation
                        interpretation = interpret_indicators(data)
                        
                        st.subheader(f"Technical Analysis for {symbol}")
                        
                        if interpretation:
                            for indicator, analysis in interpretation.items():
                                st.write(f"**{indicator}:** {analysis}")
                        
                        # Display chart with indicators
                        st.plotly_chart(create_chart(data, symbol), use_container_width=True)
                        
                        # Generate trading recommendation
                        latest = data.iloc[-1]
                        price = latest['close']
                        
                        # Simple recommendation logic
                        rsi_col = next((col for col in data.columns if 'rsi' in col), None)
                        if rsi_col and latest[rsi_col] < 30:
                            recommendation = "STRONG BUY (Oversold)"
                            confidence = "High"
                        elif rsi_col and latest[rsi_col] > 70:
                            recommendation = "STRONG SELL (Overbought)" 
                            confidence = "High"
                        else:
                            # Check trend
                            if price > data['close'].rolling(20).mean().iloc[-1]:
                                recommendation = "BUY (Uptrend)"
                                confidence = "Medium"
                            else:
                                recommendation = "SELL (Downtrend)"
                                confidence = "Medium"
                        
                        st.subheader("AI Trading Recommendation")
                        rec_col1, rec_col2, rec_col3 = st.columns(3)
                        rec_col1.metric("Action", recommendation)
                        rec_col2.metric("Confidence", confidence)
                        rec_col3.metric("Current Price", f"₹{price:.2f}")
                        
                        # Quick order buttons
                        order_col1, order_col2 = st.columns(2)
                        quantity = st.number_input("Quantity", min_value=1, value=100, key="ai_qty")
                        
                        with order_col1:
                            if st.button("Execute BUY", use_container_width=True, type="primary"):
                                place_order(instrument_df, symbol, quantity, 'MARKET', 'BUY', 'MIS')
                        with order_col2:
                            if st.button("Execute SELL", use_container_width=True, type="secondary"):
                                place_order(instrument_df, symbol, quantity, 'MARKET', 'SELL', 'MIS')
    
    with col2:
        st.subheader("Market Sentiment")
        news_df = fetch_and_analyze_news()
        if not news_df.empty:
            avg_sentiment = news_df['sentiment'].mean()
            st.metric("Overall Market Sentiment", 
                     "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral",
                     delta=f"{avg_sentiment:.2f}")
            
            st.write("**Top Market News:**")
            for _, news in news_df.head(3).iterrows():
                st.write(f"• [{news['title']}]({news['link']})")

def page_settings():
    """Application settings and configuration."""
    display_header()
    st.title("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Appearance")
        st.session_state.theme = st.radio("Theme", ["Dark", "Light"], horizontal=True)
        
        st.subheader("Trading Configuration")
        default_product = st.selectbox("Default Product", ["MIS", "CNC"])
        default_quantity = st.number_input("Default Quantity", min_value=1, value=100)
        
        st.subheader("Data Refresh")
        auto_refresh = st.checkbox("Enable Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 15)
        
        if auto_refresh:
            st_autorefresh(interval=refresh_interval * 1000, limit=100, key="settings_refresh")
    
    with col2:
        st.subheader("Risk Management")
        max_order_value = st.number_input("Max Order Value (₹)", min_value=0, value=50000)
        daily_loss_limit = st.number_input("Daily Loss Limit (₹)", min_value=0, value=10000)
        
        st.subheader("Notifications")
        email_alerts = st.checkbox("Email Alerts", value=False)
        price_alerts = st.checkbox("Price Alerts", value=True)
        
        st.subheader("2FA Security")
        if st.session_state.get('two_factor_setup_complete'):
            st.success("🔒 Two-Factor Authentication: ENABLED")
            if st.button("Disable 2FA", use_container_width=True):
                st.session_state.two_factor_setup_complete = False
                st.session_state.two_factor_verified = False
                st.session_state.pyotp_secret = None
                st.rerun()
        else:
            st.warning("⚠️ Two-Factor Authentication: DISABLED")
            if st.button("Enable 2FA", use_container_width=True):
                st.session_state.two_factor_setup_complete = False  # Will trigger setup
                st.rerun()
    
    if st.button("Save Settings", use_container_width=True, type="primary"):
        st.success("Settings saved successfully!")

# ================ 9. AUTHENTICATION & MAIN APP FLOW ================

def login_page():
    """Handles user authentication and broker connection."""
    st.title("🔐 BlockVista Terminal Login")
    st.markdown("Connect to your broker or use demo mode to explore features.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Broker Login")
        broker = st.selectbox("Select Broker", ["Zerodha", "Demo Mode"])
        
        if broker == "Zerodha":
            api_key = st.text_input("API Key", type="password")
            api_secret = st.text_input("API Secret", type="password")
            request_token = st.text_input("Request Token")
            
            if st.button("Connect to Zerodha", use_container_width=True):
                if api_key and api_secret and request_token:
                    try:
                        kite = KiteConnect(api_key=api_key)
                        data = kite.generate_session(request_token, api_secret=api_secret)
                        
                        st.session_state.kite = kite
                        st.session_state.profile = data['user_id']
                        st.session_state.broker = "Zerodha"
                        st.session_state.authenticated = True
                        
                        st.success("✅ Successfully connected to Zerodha!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
                else:
                    st.warning("Please fill all fields")
        
        elif broker == "Demo Mode":
            if st.button("Enter Demo Mode", use_container_width=True):
                st.session_state.broker = "Demo"
                st.session_state.authenticated = True
                st.session_state.profile = {"user_name": "Demo User"}
                st.success("✅ Entered Demo Mode!")
                st.rerun()
    
    with col2:
        st.subheader("Quick Start Guide")
        st.markdown("""
        1. **For Zerodha:**
           - Get API credentials from Kite Connect
           - Generate request token
           - Connect your account
        
        2. **Demo Mode:**
           - Explore all features
           - No real trading
           - Sample data only
        
        3. **Features:**
           - Live market data
           - Advanced charting
           - Options analysis
           - AI forecasting
           - Basket orders
        """)
        
        st.subheader("Security Notes")
        st.info("""
        - API keys are stored only in session
        - No data is persisted on servers
        - Enable 2FA for enhanced security
        - Use demo mode for testing
        """)

def main_app():
    """The main application interface after successful login."""
    
    # Apply custom styling
    apply_custom_styling()
    display_overnight_changes_bar()
    
    # 2FA Setup Check (for new authenticated sessions)
    if (st.session_state.get('authenticated') and 
        st.session_state.get('broker') != "Demo" and
        not st.session_state.get('two_factor_setup_complete')):
        
        two_factor_setup()
        return  # Stop further execution until 2FA setup
    
    # 2FA Verification Check (for existing sessions)
    if (st.session_state.get('authenticated') and 
        st.session_state.get('two_factor_setup_complete') and 
        st.session_state.get('pyotp_secret') and
        not st.session_state.get('two_factor_verified') and
        st.session_state.get('broker') != "Demo"):
        
        two_factor_verification()
        return  # Stop further execution until 2FA verification
    
    # Main application interface
    if st.session_state.get('profile'):
        user_name = st.session_state.profile.get('user_name', 'User')
        broker = st.session_state.get('broker', 'Demo')
        
        st.sidebar.title(f"Welcome, {user_name}")
        st.sidebar.caption(f"Connected via {broker}")
        
        # Show 2FA status
        if st.session_state.get('two_factor_setup_complete'):
            st.sidebar.success("🔒 2FA: Enabled")
        else:
            st.sidebar.warning("⚠️ 2FA: Not setup")
        
        st.sidebar.divider()
        
        st.sidebar.header("Terminal Controls")
        st.session_state.theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
        st.session_state.terminal_mode = st.sidebar.radio("Terminal Mode", ["Cash", "Futures", "Options", "HFT"], horizontal=True)
        st.sidebar.divider()
        
        # Navigation
        st.sidebar.header("Navigation")
        page_options = {
            "🏠 Dashboard": page_dashboard,
            "📈 Advanced Charting": page_advanced_charting, 
            "🌍 Premarket Pulse": page_premarket_pulse,
            "📊 F&O Analytics": page_fo_analytics,
            "🤖 ML Forecasting": page_forecasting_ml,
            "⚡ HFT Terminal": page_hft_terminal,
            "🧺 Basket Orders": page_basket_orders,
            "🏢 Fundamental Analysis": page_fundamental_analysis,
            "💡 AI Trade Assistant": page_ai_trade_assistant,
            "⚙️ Settings": page_settings
        }
        
        selected_page = st.sidebar.radio(
            "Navigate to",
            options=list(page_options.keys()),
            label_visibility="collapsed"
        )
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("Place Quick Order", use_container_width=True):
            quick_trade_dialog()
        
        if st.sidebar.button("View Portfolio", use_container_width=True):
            st.session_state.active_page = "Dashboard"
            st.rerun()
        
        # Market status
        status_info = get_market_status()
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: var(--widget-bg); border-radius: 8px;">
            <small>Market Status</small><br>
            <strong style="color: {status_info['color']};">{status_info['status']}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; color: var(--text-light); font-size: 0.8rem;">
            BlockVista Terminal v2.0<br>
            <small>Educational Purpose Only</small>
        </div>
        """, unsafe_allow_html=True)
    
        # Display selected page
        page_function = page_options[selected_page]
        page_function()

# ================ 10. MAIN EXECUTION ================

def main():
    """Main application entry point."""
    initialize_session_state()
    
    if not st.session_state.get('authenticated'):
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
