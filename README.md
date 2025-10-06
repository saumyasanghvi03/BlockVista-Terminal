# BlockVista Terminal – Next-Gen Indian Trading Terminal

## 1) Project Overview and Description
BlockVista Terminal is a professional, Streamlit-powered trading and analytics terminal tailored for Indian markets (NSE, BSE, CDS, MCX). It brings institutional-grade features—advanced charting, F&O analytics, ML forecasting, scanners, and portfolio/risk tools—into a modern web UI with optional Zerodha KiteConnect broker integration for quotes and order placement.

Built by students for India’s traders, BlockVista includes a Bharatiya Market Pulse (BMP) score, live market dashboards, options analytics with Greeks and IV, AI discovery and chat, multi-leg strategy payoff, an HFT simulator, and end-to-end portfolio workflows.

---

## 2) Complete Feature List

- Dashboard
  - Bharatiya Market Pulse (BMP): Composite market sentiment score using NIFTY/SENSEX % change and INDIA VIX normalization over lookback window with color-coded labels (Udaan/Pragati/Santulan/Sanket/Mandhi)
  - Index tiles: NIFTY 50, SENSEX, INDIA VIX live metrics; global indices snapshots (S&P 500, Nikkei 225, Hang Seng) via yfinance; GIFT NIFTY proxy intraday chart (ticker IN=F)
  - NIFTY 50 Heatmap: Live treemap of NIFTY constituents sized by price and colored by % change
  - Latest Market News with Sentiment: Aggregated RSS feeds (Economic Times, Moneycontrol, Business Standard, Livemint, Reuters, BBC) with VADER sentiment icons

- Advanced Charting
  - Plotly charts with multiple layouts, dynamic tooltips, custom themes (dark/light), and multi-timeframes
  - 50+ technical indicators (via pandas-ta): RSI, Stochastics, MACD, ADX, EMA/SMA, Bollinger Bands, ATR, etc.
  - Interpretation helper: summarizes latest RSI/Stoch/MACD/ADX states (overbought/oversold, crossovers, trend strength)

- F&O Analytics Hub
  - Options Chain (NIFTY/BANKNIFTY/FINNIFTY): CE/PE OI, LTP, strikes, expiries, Most Active options dialog
  - Greeks & IV: Black–Scholes calculator for price, Delta, Gamma, Vega, Theta, Rho; implied volatility via Newton-Raphson
  - PCR Analysis: Total CE vs PE OI and sentiment bands with metrics
  - Volatility & OI Surface: Per-strike IVs and OI exploration with expiry-based T and risk-free rate inputs

- AI Discovery and Portfolio-Aware Assistant
  - In-app chat that can answer portfolio, holdings, and market questions (broker-connected context)
  - Smart prompts for common workflows (e.g., “show option chain for BANKNIFTY”)

- ML Forecasting
  - Seasonal ARIMA training with seasonal decomposition; backtest fitted values and forecast next N days
  - Confidence intervals with seasonality reintroduced; MAPE utility; combined static CSV + live data loader (Zerodha or yfinance)

- Algo Strategy Hub (Options Strategies)
  - Multi-leg strategy builder (calls/puts, buy/sell, quantities, limit/market)
  - Payoff chart at expiry with breakeven points, max profit, max loss
  - Greeks aggregation and sensitivity exploration

- Portfolio & Risk
  - Live positions and holdings (via broker), order book, and day P&L
  - Allocation pies: stock-wise and sector-wise (sector mapping CSV)

- Market Scanners
  - Momentum (RSI): overbought/oversold signals
  - Trend (EMA alignment): uptrend/downtrend detection
  - Breakout: 20-day high/low breakout/breakdown
  - CSV export and quick add-to-watchlist actions

- HFT Simulator
  - Live tick log, market depth (top bids/asks), latency metric simulation
  - One-click market/limit buy/sell with lot-size aware quantity inputs

- Basket Orders
  - Prepare and place multiple orders in one flow (Zerodha variety=REGULAR), with symbol validation from instruments

- Authentication & Security
  - 2FA with pyotp TOTP, QR provisioning (qrcode), persistent secret derived via SHA-256 + base32

- Utilities
  - India market holiday calendar cache; watchlists; quick trade dialog; session state bootstrap; autorefresh support

---

## 3) Technical Specifications (Libraries)
- UI/Framework: streamlit, streamlit-autorefresh
- Data/Analysis: pandas, numpy, pandas-ta, scipy, statsmodels, tabulate
- Charting: plotly (graph_objects, make_subplots)
- Market/Broker APIs: kiteconnect (Zerodha), yfinance, requests
- Time/Locale: datetime, pytz
- News & NLP: feedparser, vaderSentiment
- Auth/Security: pyotp, qrcode, hashlib, base64
- Imaging/IO: Pillow (PIL), io, qrcode
- Misc: re, random

---

## 4) Installation and Setup

Prerequisites
- Python 3.9+
- Zerodha Kite API key (optional, required for live trading and authenticated quotes)

Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Run
```bash
streamlit run app.py
```

Broker Setup (Optional)
- Generate API key/secret in Zerodha console
- Authenticate session (handled in app); follow on-screen login and 2FA steps
- 2FA: Open the “Generate QR for 2FA” dialog in-app, scan with Google/Microsoft Authenticator, then verify OTP

---

## 5) Usage Guide
- Dashboard
  - Monitor BMP score, index tiles, global indices, and sentiment-tagged news
  - Explore NIFTY 50 heatmap to spot top movers
- Advanced Charting
  - Choose symbols, add indicators, and switch layouts/timeframes; use interpretation helper for quick bias
- F&O Analytics
  - Load options chain for NIFTY/BANKNIFTY/FINNIFTY; compute IV and Greeks; check PCR and Most Active contracts
- Algo Strategy Hub
  - Add option legs; review payoff curve, breakevens, max profit/loss, and leg Greeks
- Portfolio & Risk
  - View holdings/positions with P&L; analyze allocation pies by stock and sector; inspect order book
- Scanners
  - Run Momentum/Trend/Breakout scans; export to CSV; add top candidates to watchlist
- HFT Simulator
  - Watch tick-by-tick changes and depth; place one-click market/limit orders
- ML Forecasting
  - Select instrument; train Seasonal ARIMA; review backtest and forecast with confidence bands

---

## 6) Screenshots / Demo
Place your screenshots or GIFs under docs/screenshots and link them here:
- docs/screenshots/overview.png — Dashboard & BMP
- docs/screenshots/options_chain.png — Options Chain & Greeks
- docs/screenshots/strategy_payoff.png — Strategy Payoff & Risk
- docs/screenshots/hft.png — HFT Simulator & Depth
- docs/screenshots/ml_forecast.png — ML Forecasting

---

## 7) API Integrations
- Zerodha KiteConnect: instruments, quotes, order placement, depth, positions/holdings/orders
- yfinance: global indices (^GSPC, ^N225, ^HSI), GIFT NIFTY proxy (IN=F)
- RSS feeds: Economic Times, Moneycontrol, Business Standard, Livemint, Reuters, BBC

---

## 8) Technical Indicators Supported
Examples via pandas-ta and in-app logic:
- Momentum: RSI(14), Stochastic (14,3,3), MACD(12,26,9), ADX(14)
- Trend: EMA/SMA crossovers, multi-EMA alignment
- Volatility: ATR, Bollinger Bands
- Options: Black–Scholes Greeks (Delta, Gamma, Vega, Theta, Rho), Implied Volatility (Newton)

---

## 9) Contributing
We welcome contributions from students, traders, and developers!
1. Fork the repo
2. Create a feature branch: git checkout -b feature/amazing-feature
3. Commit with clear messages: git commit -m "feat: add amazing feature"
4. Push and open a Pull Request
Areas: new indicators, scanners, ML models, UI polish, docs, tests, broker adapters

---

## 10) License
BlockVista Terminal — Creative Commons Zero v1.0 Universal (CC0-1.0) for authorized collaboration as stated by the project. See repository notices for commercial usage restrictions.

---

## Notes and Disclaimers
- BMP methodology: weighted blend of normalized recent NIFTY/SENSEX returns and inverse-normalized VIX
- Options analytics are for educational purposes; trading involves risk
- Real-time data subject to broker/exchange terms and latency
- Beta software; features may evolve frequently
