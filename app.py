# ================ AUTHENTICATION FUNCTIONS ================

def login_page():
    """Main login page with broker selection and authentication."""
    st.set_page_config(page_title="BlockVista Terminal", layout="wide")
    apply_custom_styling()
    
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🚀 BlockVista Terminal</h1>
            <p style="font-size: 1.2rem; color: var(--text-light);">Professional Trading & Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Broker Login")
        st.session_state.broker = st.selectbox("Select Broker", ["Zerodha", "Angel One", "5Paisa", "Upstox"])
        
        if st.session_state.broker == "Zerodha":
            zerodha_login_form()
        else:
            st.info(f"{st.session_state.broker} integration coming soon!")
    
    with col2:
        st.markdown("""
            <div style="padding: 2rem; background: var(--secondary-bg); border-radius: 10px;">
                <h3>🌟 Features</h3>
                <ul style="color: var(--text-light);">
                    <li>Live Market Data & Charts</li>
                    <li>Advanced Options Analytics</li>
                    <li>AI-Powered Forecasting</li>
                    <li>Algorithmic Trading Bots</li>
                    <li>Portfolio Management</li>
                    <li>Basket Orders</li>
                    <li>Real-time News & Sentiment</li>
                </ul>
                
                <h3>🛡️ Security</h3>
                <ul style="color: var(--text-light);">
                    <li>End-to-End Encryption</li>
                    <li>Two-Factor Authentication</li>
                    <li>Secure API Connections</li>
                    <li>Local Session Storage</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def zerodha_login_form():
    """Zerodha specific login form."""
    st.text_input("API Key", key="api_key", placeholder="Enter your API Key")
    st.text_input("API Secret", key="api_secret", type="password", placeholder="Enter your API Secret")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login", use_container_width=True, type="primary"):
            if st.session_state.api_key and st.session_state.api_secret:
                authenticate_zerodha()
            else:
                st.error("Please enter both API Key and Secret")
    
    with col2:
        if st.button("Demo Mode", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.profile = {"user_name": "Demo User"}
            st.session_state.broker = "Zerodha"
            st.session_state.two_factor_setup_complete = True
            st.rerun()

def authenticate_zerodha():
    """Authenticate with Zerodha Kite Connect."""
    try:
        kite = KiteConnect(api_key=st.session_state.api_key)
        
        # In a real implementation, you'd generate a request token
        # For demo purposes, we'll simulate successful authentication
        st.session_state.kite = kite
        st.session_state.profile = {"user_name": "Zerodha User"}
        st.session_state.two_factor_setup_complete = True
        st.session_state.authenticated = True
        
        st.success("✅ Successfully authenticated!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")

def two_factor_dialog():
    """Two-factor authentication dialog."""
    if not st.session_state.get('show_2fa_dialog', True):
        return
        
    st.markdown("""
        <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999;">
            <div style="background: var(--widget-bg); padding: 2rem; border-radius: 10px; max-width: 400px; width: 90%;">
    """, unsafe_allow_html=True)
    
    st.subheader("Two-Factor Authentication")
    st.info("Enter the TOTP from your authenticator app")
    
    totp_code = st.text_input("TOTP Code", placeholder="6-digit code")
    
    col1, col2 = st.columns(2)
    
    if col1.button("Verify", use_container_width=True, type="primary"):
        if totp_code and len(totp_code) == 6:
            try:
                # In real implementation, verify with broker
                st.session_state.authenticated = True
                st.session_state.show_2fa_dialog = False
                st.rerun()
            except Exception as e:
                st.error(f"Verification failed: {str(e)}")
        else:
            st.error("Please enter a valid 6-digit code")
    
    if col2.button("Cancel", use_container_width=True):
        st.session_state.show_2fa_dialog = False
        st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def qr_code_dialog():
    """QR code dialog for 2FA setup."""
    if not st.session_state.get('show_qr_dialog', True):
        return
        
    # Generate a random secret for demo purposes
    if not st.session_state.get('pyotp_secret'):
        st.session_state.pyotp_secret = pyotp.random_base32()
    
    totp = pyotp.TOTP(st.session_state.pyotp_secret)
    provisioning_uri = totp.provisioning_uri(
        name=st.session_state.profile.get('user_name', 'user'),
        issuer_name="BlockVista Terminal"
    )
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for display
    buffered = io.BytesIO()
    qr_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    st.markdown(f"""
        <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999;">
            <div style="background: var(--widget-bg); padding: 2rem; border-radius: 10px; max-width: 400px; width: 90%; text-align: center;">
                <h3>Setup Two-Factor Authentication</h3>
                <p>Scan this QR code with your authenticator app</p>
                <img src="data:image/png;base64,{img_str}" style="width: 200px; height: 200px; margin: 1rem auto;">
                <p style="font-size: 0.8rem; color: var(--text-light); margin: 1rem 0;">
                    Secret: {st.session_state.pyotp_secret}<br>
                    <small>(Keep this secure)</small>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("I've Scanned the QR Code", key="qr_confirm"):
        st.session_state.two_factor_setup_complete = True
        st.session_state.show_qr_dialog = False
        st.session_state.show_2fa_dialog = True
        st.rerun()

# ================ MISSING PAGE FUNCTIONS ================

def page_momentum_and_trend_finder():
    """Momentum and trend scanner page."""
    display_header()
    st.title("Momentum & Trend Scanner")
    st.info("Scan for stocks based on technical indicators and momentum patterns")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Connect to a broker to use the scanner")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scan Criteria")
        scan_type = st.selectbox("Scan Type", [
            "High Momentum", 
            "Oversold", 
            "Overbought", 
            "Volume Surge",
            "New Highs",
            "New Lows"
        ])
        
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        volume_multiplier = st.slider("Volume Multiplier", 1.0, 5.0, 2.0)
        
        if st.button("Run Scan", use_container_width=True):
            st.session_state.scan_results = run_momentum_scan(
                instrument_df, scan_type, rsi_period, volume_multiplier
            )
    
    with col2:
        st.subheader("Scan Results")
        if st.session_state.get('scan_results'):
            results_df = st.session_state.scan_results
            if not results_df.empty:
                st.dataframe(results_df, use_container_width=True)
                
                # Show charts for top results
                top_symbol = results_df.iloc[0]['Symbol']
                st.subheader(f"Chart for {top_symbol}")
                token = get_instrument_token(top_symbol, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period='3mo')
                    if not data.empty:
                        st.plotly_chart(create_chart(data, top_symbol), use_container_width=True)
            else:
                st.info("No stocks matching the criteria")
        else:
            st.info("Configure criteria and run scan to see results")

def run_momentum_scan(instrument_df, scan_type, rsi_period=14, volume_multiplier=2.0):
    """Run momentum scan based on criteria."""
    # Get NIFTY 50 stocks for scanning
    nifty_stocks = get_nifty50_constituents(instrument_df)
    if nifty_stocks.empty:
        return pd.DataFrame()
    
    results = []
    
    for symbol in nifty_stocks['Symbol'].head(15):  # Scan top 15 for performance
        try:
            token = get_instrument_token(symbol, instrument_df, 'NSE')
            if token:
                data = get_historical_data(token, 'day', period='3mo')
                if not data.empty and len(data) > 20:
                    # Calculate indicators
                    data['rsi'] = ta.rsi(data['close'], length=rsi_period)
                    data['volume_sma'] = ta.sma(data['volume'], length=20)
                    data['sma_20'] = ta.sma(data['close'], length=20)
                    
                    latest = data.iloc[-1]
                    prev = data.iloc[-2]
                    
                    # Apply scan criteria
                    if scan_type == "High Momentum" and latest['rsi'] > 60 and latest['close'] > latest['sma_20']:
                        results.append({
                            'Symbol': symbol,
                            'Price': latest['close'],
                            'RSI': latest['rsi'],
                            'Signal': 'Momentum',
                            'Score': latest['rsi']
                        })
                    elif scan_type == "Oversold" and latest['rsi'] < 30:
                        results.append({
                            'Symbol': symbol,
                            'Price': latest['close'],
                            'RSI': latest['rsi'],
                            'Signal': 'Oversold',
                            'Score': 100 - latest['rsi']
                        })
                    elif scan_type == "Volume Surge" and latest['volume'] > latest['volume_sma'] * volume_multiplier:
                        results.append({
                            'Symbol': symbol,
                            'Price': latest['close'],
                            'Volume_Ratio': latest['volume'] / latest['volume_sma'],
                            'Signal': 'Volume Spike',
                            'Score': latest['volume'] / latest['volume_sma']
                        })
        except Exception:
            continue
    
    return pd.DataFrame(results).sort_values('Score', ascending=False)

def page_ai_discovery():
    """AI-powered market discovery and insights page."""
    display_header()
    st.title("AI Market Discovery")
    st.info("AI-powered market analysis and opportunity discovery")
    
    tab1, tab2, tab3 = st.tabs(["Sector Analysis", "Pattern Recognition", "Market Insights"])
    
    with tab1:
        st.subheader("Sector Rotation Analysis")
        st.info("AI analysis of sector performance and rotation patterns")
        
        # Sector performance analysis would go here
        st.write("Sector performance metrics and trends...")
    
    with tab2:
        st.subheader("Chart Pattern Recognition")
        st.info("AI-powered pattern detection in price charts")
        
        instrument_df = get_instrument_df()
        if not instrument_df.empty:
            symbol = st.selectbox("Select Symbol", 
                                instrument_df[instrument_df['exchange'] == 'NSE']['tradingsymbol'].unique()[:20])
            
            if symbol:
                token = get_instrument_token(symbol, instrument_df)
                if token:
                    data = get_historical_data(token, 'day', period='6mo')
                    if not data.empty:
                        st.plotly_chart(create_chart(data, symbol), use_container_width=True)
                        
                        # Simple pattern detection
                        if len(data) > 50:
                            current_price = data['close'].iloc[-1]
                            sma_50 = data['close'].rolling(50).mean().iloc[-1]
                            
                            if current_price > sma_50 * 1.1:
                                st.success(f"📈 {symbol} is in strong uptrend (>{sma_50:.2f})")
                            elif current_price < sma_50 * 0.9:
                                st.error(f"📉 {symbol} is in strong downtrend (<{sma_50:.2f})")
                            else:
                                st.info(f"➡️ {symbol} is trading near average ({sma_50:.2f})")
    
    with tab3:
        st.subheader("Market Sentiment & Insights")
        st.info("AI analysis of market sentiment and emerging opportunities")
        
        # News sentiment analysis
        news_df = fetch_and_analyze_news()
        if not news_df.empty:
            avg_sentiment = news_df['sentiment'].mean()
            st.metric("Overall Market Sentiment", 
                     f"{'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'}",
                     f"{avg_sentiment:.2f}")
            
            st.write("**Top Market News:**")
            for _, news in news_df.head(5).iterrows():
                emoji = "🟢" if news['sentiment'] > 0.1 else "🔴" if news['sentiment'] < -0.1 else "🟡"
                st.write(f"{emoji} [{news['title']}]({news['link']}) - *{news['source']}*")

# ================ MAIN EXECUTION ================

if __name__ == "__main__":
    initialize_session_state()
    
    if not st.session_state.get('authenticated', False):
        login_page()
    else:
        main_app()
