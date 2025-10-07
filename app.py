# Add this to the existing code in the appropriate sections

# ================ ALGO BOTS SECTION ================

def momentum_trader_bot(instrument_df, symbol, capital=100):
    """Momentum trading bot that buys on upward momentum and sells on downward momentum."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, '5minute', period='1d')
    if data.empty or len(data) < 20:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate indicators
    data['RSI'] = ta.rsi(data['close'], length=14)
    data['EMA_20'] = ta.ema(data['close'], length=20)
    data['EMA_50'] = ta.ema(data['close'], length=50)
    
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    signals = []
    
    # Momentum signals
    if (latest['EMA_20'] > latest['EMA_50'] and 
        prev['EMA_20'] <= prev['EMA_50']):
        signals.append("EMA crossover - BULLISH")
    
    if latest['RSI'] < 30:
        signals.append("RSI oversold - BULLISH")
    elif latest['RSI'] > 70:
        signals.append("RSI overbought - BEARISH")
    
    # Price momentum
    price_change_5min = ((latest['close'] - data.iloc[-6]['close']) / data.iloc[-6]['close']) * 100
    if price_change_5min > 0.5:
        signals.append(f"Strong upward momentum: +{price_change_5min:.2f}%")
    
    # Calculate position size
    current_price = latest['close']
    quantity = max(1, int((capital * 0.8) / current_price))  # Use 80% of capital
    
    action = "HOLD"
    if len([s for s in signals if "BULLISH" in s]) >= 2:
        action = "BUY"
    elif len([s for s in signals if "BEARISH" in s]) >= 2:
        action = "SELL"
    
    return {
        "bot_name": "Momentum Trader",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "Medium"
    }

def mean_reversion_bot(instrument_df, symbol, capital=100):
    """Mean reversion bot that trades on price returning to mean levels."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, '15minute', period='5d')
    if data.empty or len(data) < 50:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate Bollinger Bands
    bb = ta.bbands(data['close'], length=20, std=2)
    data = pd.concat([data, bb], axis=1)
    
    latest = data.iloc[-1]
    
    signals = []
    current_price = latest['close']
    bb_lower = latest.get('BBL_20_2.0', current_price)
    bb_upper = latest.get('BBU_20_2.0', current_price)
    bb_middle = latest.get('BBM_20_2.0', current_price)
    
    # Mean reversion signals
    if current_price <= bb_lower * 1.02:  # Within 2% of lower band
        signals.append("Near lower Bollinger Band - BULLISH")
    
    if current_price >= bb_upper * 0.98:  # Within 2% of upper band
        signals.append("Near upper Bollinger Band - BEARISH")
    
    # Distance from mean
    distance_from_mean = ((current_price - bb_middle) / bb_middle) * 100
    if abs(distance_from_mean) > 3:
        signals.append(f"Price {abs(distance_from_mean):.1f}% from mean")
    
    # RSI for confirmation
    data['RSI'] = ta.rsi(data['close'], length=14)
    rsi = data['RSI'].iloc[-1]
    if rsi < 35:
        signals.append("RSI supporting oversold condition")
    elif rsi > 65:
        signals.append("RSI supporting overbought condition")
    
    # Calculate position size
    quantity = max(1, int((capital * 0.6) / current_price))  # Use 60% of capital
    
    action = "HOLD"
    if any("BULLISH" in s for s in signals) and rsi < 40:
        action = "BUY"
    elif any("BEARISH" in s for s in signals) and rsi > 60:
        action = "SELL"
    
    return {
        "bot_name": "Mean Reversion",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "Low"
    }

def volatility_breakout_bot(instrument_df, symbol, capital=100):
    """Volatility breakout bot that trades on breakouts from consolidation."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, '30minute', period='5d')
    if data.empty or len(data) < 30:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate ATR and volatility
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    data['Range'] = data['high'] - data['low']
    avg_range = data['Range'].rolling(window=20).mean()
    
    latest = data.iloc[-1]
    current_price = latest['close']
    current_atr = latest['ATR']
    current_range = latest['Range']
    
    signals = []
    
    # Volatility signals
    if current_range > avg_range.iloc[-1] * 1.5:
        signals.append("High volatility - potential breakout")
    
    # Price action signals
    prev_high = data['high'].iloc[-2]
    prev_low = data['low'].iloc[-2]
    
    if current_price > prev_high + current_atr * 0.5:
        signals.append("Breakout above resistance - BULLISH")
    
    if current_price < prev_low - current_atr * 0.5:
        signals.append("Breakdown below support - BEARISH")
    
    # Volume confirmation (if available)
    if 'volume' in data.columns:
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        if data['volume'].iloc[-1] > avg_volume * 1.2:
            signals.append("High volume confirmation")
    
    # Calculate position size based on ATR
    atr_percentage = (current_atr / current_price) * 100
    risk_per_trade = min(20, max(5, atr_percentage * 2))  # Dynamic position sizing
    quantity = max(1, int((capital * (risk_per_trade / 100)) / current_price))
    
    action = "HOLD"
    if any("BULLISH" in s for s in signals):
        action = "BUY"
    elif any("BEARISH" in s for s in signals):
        action = "SELL"
    
    return {
        "bot_name": "Volatility Breakout",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "High"
    }

def value_investor_bot(instrument_df, symbol, capital=100):
    """Value investor bot focusing on longer-term value signals."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, 'day', period='1y')
    if data.empty or len(data) < 100:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate moving averages and trends
    data['SMA_50'] = ta.sma(data['close'], length=50)
    data['SMA_200'] = ta.sma(data['close'], length=200)
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    latest = data.iloc[-1]
    current_price = latest['close']
    
    signals = []
    
    # Trend analysis
    if latest['SMA_50'] > latest['SMA_200']:
        signals.append("Bullish trend (50 > 200 SMA)")
    else:
        signals.append("Bearish trend (50 < 200 SMA)")
    
    # Support and resistance levels
    support_20 = data['low'].rolling(window=20).min().iloc[-1]
    resistance_20 = data['high'].rolling(window=20).max().iloc[-1]
    
    distance_to_support = ((current_price - support_20) / current_price) * 100
    distance_to_resistance = ((resistance_20 - current_price) / current_price) * 100
    
    if distance_to_support < 5:
        signals.append("Near strong support - BULLISH")
    
    if distance_to_resistance < 5:
        signals.append("Near strong resistance - BEARISH")
    
    # Monthly performance
    monthly_return = ((current_price - data['close'].iloc[-21]) / data['close'].iloc[-21]) * 100
    if monthly_return < -10:
        signals.append("Oversold on monthly basis - BULLISH")
    elif monthly_return > 15:
        signals.append("Overbought on monthly basis - BEARISH")
    
    # Calculate position size for longer term
    quantity = max(1, int((capital * 0.5) / current_price))  # Conservative 50%
    
    action = "HOLD"
    bullish_signals = len([s for s in signals if "BULLISH" in s])
    bearish_signals = len([s for s in signals if "BEARISH" in s])
    
    if bullish_signals >= 2 and bearish_signals == 0:
        action = "BUY"
    elif bearish_signals >= 2 and bullish_signals == 0:
        action = "SELL"
    
    return {
        "bot_name": "Value Investor",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "Low"
    }

def scalper_bot(instrument_df, symbol, capital=100):
    """High-frequency scalping bot for quick, small profits."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, 'minute', period='1d')
    if data.empty or len(data) < 100:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate scalping indicators
    data['RSI_9'] = ta.rsi(data['close'], length=9)
    data['EMA_8'] = ta.ema(data['close'], length=8)
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    latest = data.iloc[-1]
    current_price = latest['close']
    
    signals = []
    
    # Scalping signals
    if latest['EMA_8'] > latest['EMA_21']:
        signals.append("Fast EMA above slow EMA - BULLISH")
    else:
        signals.append("Fast EMA below slow EMA - BEARISH")
    
    rsi_9 = latest['RSI_9']
    if rsi_9 < 25:
        signals.append("Extremely oversold - BULLISH")
    elif rsi_9 > 75:
        signals.append("Extremely overbought - BEARISH")
    
    # Price momentum for scalping
    price_change_3min = ((current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]) * 100
    if abs(price_change_3min) > 0.3:
        signals.append(f"Strong short-term momentum: {price_change_3min:+.2f}%")
    
    # Calculate small position size for scalping
    quantity = max(1, int((capital * 0.3) / current_price))  # Small position for quick exits
    
    action = "HOLD"
    if (any("BULLISH" in s for s in signals) and 
        "BEARISH" not in str(signals) and
        rsi_9 < 70):
        action = "BUY"
    elif (any("BEARISH" in s for s in signals) and 
          "BULLISH" not in str(signals) and
          rsi_9 > 30):
        action = "SELL"
    
    return {
        "bot_name": "Scalper Pro",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "Very High"
    }

def trend_follower_bot(instrument_df, symbol, capital=100):
    """Trend following bot that rides established trends."""
    exchange = 'NSE'
    token = get_instrument_token(symbol, instrument_df, exchange)
    if not token:
        return {"error": f"Could not find instrument for {symbol}"}
    
    data = get_historical_data(token, 'hour', period='1mo')
    if data.empty or len(data) < 100:
        return {"error": "Insufficient data for analysis"}
    
    # Calculate trend indicators
    data['ADX'] = ta.adx(data['high'], data['low'], data['close'], length=14)['ADX_14']
    data['EMA_20'] = ta.ema(data['close'], length=20)
    data['EMA_50'] = ta.ema(data['close'], length=50)
    data['SuperTrend'] = ta.supertrend(data['high'], data['low'], data['close'], length=10, multiplier=3)['SUPERT_10_3.0']
    
    latest = data.iloc[-1]
    current_price = latest['close']
    
    signals = []
    
    # Trend strength
    adx = latest['ADX']
    if adx > 25:
        signals.append(f"Strong trend (ADX: {adx:.1f})")
    else:
        signals.append(f"Weak trend (ADX: {adx:.1f})")
    
    # Trend direction
    if latest['EMA_20'] > latest['EMA_50']:
        signals.append("Uptrend confirmed - BULLISH")
    else:
        signals.append("Downtrend confirmed - BEARISH")
    
    # SuperTrend signals
    if current_price > latest['SuperTrend']:
        signals.append("Price above SuperTrend - BULLISH")
    else:
        signals.append("Price below SuperTrend - BEARISH")
    
    # Pullback opportunities
    if (latest['EMA_20'] > latest['EMA_50'] and 
        current_price < latest['EMA_20'] and 
        current_price > latest['EMA_50']):
        signals.append("Pullback in uptrend - BULLISH")
    
    elif (latest['EMA_20'] < latest['EMA_50'] and 
          current_price > latest['EMA_20'] and 
          current_price < latest['EMA_50']):
        signals.append("Pullback in downtrend - BEARISH")
    
    # Calculate position size
    quantity = max(1, int((capital * 0.7) / current_price))  # Use 70% of capital
    
    action = "HOLD"
    bullish_count = len([s for s in signals if "BULLISH" in s])
    bearish_count = len([s for s in signals if "BEARISH" in s])
    
    if bullish_count >= 2 and adx > 20:
        action = "BUY"
    elif bearish_count >= 2 and adx > 20:
        action = "SELL"
    
    return {
        "bot_name": "Trend Follower",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "current_price": current_price,
        "signals": signals,
        "capital_required": quantity * current_price,
        "risk_level": "Medium"
    }

# Dictionary of all available bots
ALGO_BOTS = {
    "Momentum Trader": momentum_trader_bot,
    "Mean Reversion": mean_reversion_bot,
    "Volatility Breakout": volatility_breakout_bot,
    "Value Investor": value_investor_bot,
    "Scalper Pro": scalper_bot,
    "Trend Follower": trend_follower_bot
}

def execute_bot_trade(instrument_df, bot_result):
    """Executes a trade based on bot recommendation."""
    if bot_result.get("error"):
        st.error(bot_result["error"])
        return
    
    if bot_result["action"] == "HOLD":
        st.info(f"🤖 {bot_result['bot_name']} recommends HOLDING {bot_result['symbol']}")
        return
    
    action = bot_result["action"]
    symbol = bot_result["symbol"]
    quantity = bot_result["quantity"]
    current_price = bot_result["current_price"]
    required_capital = bot_result["capital_required"]
    
    st.success(f"""
    🚀 **{bot_result['bot_name']} Recommendation:**
    - **Action:** {action} {quantity} shares of {symbol}
    - **Current Price:** ₹{current_price:.2f}
    - **Required Capital:** ₹{required_capital:.2f}
    - **Risk Level:** {bot_result['risk_level']}
    """)
    
    col1, col2 = st.columns(2)
    
    if col1.button(f"Execute {action} Order", key=f"execute_{symbol}", use_container_width=True):
        place_order(instrument_df, symbol, quantity, 'MARKET', action, 'MIS')
    
    if col2.button("Ignore Recommendation", key=f"ignore_{symbol}", use_container_width=True):
        st.info("Trade execution cancelled.")

def page_algo_bots():
    """Main algo bots page where users can run different trading bots."""
    display_header()
    st.title("🤖 Algo Trading Bots")
    st.info("Run automated trading bots with minimum capital of ₹100. Each bot uses different strategies and risk profiles.", icon="🤖")
    
    instrument_df = get_instrument_df()
    if instrument_df.empty:
        st.info("Please connect to a broker to use algo bots.")
        return
    
    # Bot selection and configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_bot = st.selectbox(
            "Select Trading Bot",
            list(ALGO_BOTS.keys()),
            help="Choose a trading bot based on your risk appetite and trading style"
        )
        
        # Bot descriptions
        bot_descriptions = {
            "Momentum Trader": "Trades on strong price momentum and trend continuations. Medium risk.",
            "Mean Reversion": "Buys low and sells high based on statistical mean reversion. Low risk.",
            "Volatility Breakout": "Captures breakouts from low volatility periods. High risk.",
            "Value Investor": "Focuses on longer-term value and fundamental trends. Low risk.",
            "Scalper Pro": "High-frequency trading for quick, small profits. Very high risk.",
            "Trend Follower": "Rides established trends with multiple confirmations. Medium risk."
        }
        
        st.markdown(f"**Description:** {bot_descriptions[selected_bot]}")
    
    with col2:
        trading_capital = st.number_input(
            "Trading Capital (₹)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Minimum ₹100 required"
        )
    
    st.markdown("---")
    
    # Symbol selection and bot execution
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("Stock Selection")
        all_symbols = instrument_df[instrument_df['exchange'].isin(['NSE', 'BSE'])]['tradingsymbol'].unique()
        selected_symbol = st.selectbox(
            "Select Stock",
            sorted(all_symbols),
            index=list(all_symbols).index('RELIANCE') if 'RELIANCE' in all_symbols else 0
        )
        
        # Show current price
        quote_data = get_watchlist_data([{'symbol': selected_symbol, 'exchange': 'NSE'}])
        if not quote_data.empty:
            current_price = quote_data.iloc[0]['Price']
            st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col4:
        st.subheader("Bot Execution")
        st.write(f"**Selected Bot:** {selected_bot}")
        st.write(f"**Available Capital:** ₹{trading_capital:,}")
        
        if st.button("🚀 Run Trading Bot", use_container_width=True, type="primary"):
            with st.spinner(f"Running {selected_bot} analysis..."):
                bot_function = ALGO_BOTS[selected_bot]
                bot_result = bot_function(instrument_df, selected_symbol, trading_capital)
                
                if bot_result and not bot_result.get("error"):
                    st.session_state.last_bot_result = bot_result
                    st.rerun()
    
    # Display bot results
    if 'last_bot_result' in st.session_state and st.session_state.last_bot_result:
        bot_result = st.session_state.last_bot_result
        
        if bot_result.get("error"):
            st.error(bot_result["error"])
        else:
            st.markdown("---")
            st.subheader("🤖 Bot Analysis Results")
            
            # Create metrics cards
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                action_color = "green" if bot_result["action"] == "BUY" else "red" if bot_result["action"] == "SELL" else "orange"
                st.markdown(f'<div class="metric-card" style="border-color: {action_color};">'
                           f'<h3 style="color: {action_color};">{bot_result["action"]}</h3>'
                           f'<p>Recommended Action</p></div>', unsafe_allow_html=True)
            
            with col6:
                st.metric("Quantity", bot_result["quantity"])
            
            with col7:
                st.metric("Capital Required", f"₹{bot_result['capital_required']:.2f}")
            
            with col8:
                risk_color = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "darkred"}
                st.markdown(f'<div class="metric-card" style="border-color: {risk_color.get(bot_result["risk_level"], "gray")};">'
                           f'<h3 style="color: {risk_color.get(bot_result["risk_level"], "gray")};">{bot_result["risk_level"]}</h3>'
                           f'<p>Risk Level</p></div>', unsafe_allow_html=True)
            
            # Display signals
            st.subheader("📊 Analysis Signals")
            for signal in bot_result["signals"]:
                if "BULLISH" in signal:
                    st.success(f"✅ {signal}")
                elif "BEARISH" in signal:
                    st.error(f"❌ {signal}")
                else:
                    st.info(f"📈 {signal}")
            
            # Execute trade
            execute_bot_trade(instrument_df, bot_result)
    
    # Bot performance history
    st.markdown("---")
    st.subheader("📈 Bot Performance Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Best Practices:**
        - Start with minimum capital (₹100)
        - Use 'Value Investor' for beginners
        - 'Scalper Pro' requires constant monitoring
        - Always check signals before executing
        - Combine multiple bot recommendations
        """)
    
    with tips_col2:
        st.markdown("""
        **Risk Management:**
        - Never risk more than 2% per trade
        - Use stop losses with every trade
        - Diversify across different bots
        - Monitor performance regularly
        - Adjust capital based on experience
        """)
    
    # Quick bot comparison
    with st.expander("🤖 Bot Comparison Guide"):
        comparison_data = {
            "Bot": list(ALGO_BOTS.keys()),
            "Risk Level": ["Medium", "Low", "High", "Low", "Very High", "Medium"],
            "Holding Period": ["Hours", "Days", "Minutes", "Weeks", "Minutes", "Days"],
            "Capital Recommended": ["₹1,000+", "₹500+", "₹2,000+", "₹2,000+", "₹5,000+", "₹1,500+"],
            "Best For": ["Trend riding", "Safe returns", "Quick profits", "Long term", "Experienced", "Trend following"]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Add the new page to the navigation
# In the main_app function, add "Algo Bots" to the pages dictionary:

# In the 'Cash' section, add:
# "Algo Trading Bots": page_algo_bots,

# So it becomes:
pages = {
    "Cash": {
        "Dashboard": page_dashboard,
        "Algo Trading Bots": page_algo_bots,  # ADD THIS LINE
        "Premarket Pulse": page_premarket_pulse,
        # ... rest of the pages
    },
    # ... other sections
}
