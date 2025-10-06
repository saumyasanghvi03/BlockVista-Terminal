# BlockVista Terminal – Next-Gen Indian Trading Terminal

## 1) Project Overview and Description

BlockVista Terminal is a professional, Streamlit-powered trading and analytics terminal tailored for Indian markets (NSE, BSE, CDS, MCX). It brings institutional-grade features—advanced charting, F&O analytics, ML forecasting, scanners, and portfolio/risk tools—into a modern web UI with optional Zerodha KiteConnect broker integration for quotes and order placement.

Built by students for India's traders, BlockVista includes a Bharatiya Market Pulse (BMP) score, live market dashboards, options analytics with Greeks and IV, AI discovery and chat, multi-leg strategy payoff, an HFT simulator, and end-to-end portfolio workflows.

---

## 2) Complete Feature List

- **Dashboard**
  - **Bharatiya Market Pulse (BMP)**: Composite market sentiment score using NIFTY/SENSEX % change and INDIA VIX normalization over lookback window with color-coded labels (Udaan/Pragati/Santulan/Sanket/Mandhi)
  - **Index tiles**: NIFTY 50, SENSEX, INDIA VIX live metrics; global indices snapshots (S&P 500, Nikkei 225, Hang Seng) via yfinance; GIFT NIFTY proxy intraday chart (ticker IN=F)
  - **NIFTY 50 Heatmap**: Live treemap of NIFTY constituents sized by price and colored by % change
  - **Latest Market News with Sentiment**: Aggregated RSS feeds (Economic Times, Moneycontrol, Business Standard, Livemint, Reuters, BBC) with VADER sentiment icons

- **Advanced Charting**
  - Plotly charts with multiple layouts, dynamic tooltips, custom themes (dark/light), and multi-timeframes
  - 50+ technical indicators (via pandas-ta): RSI, Stochastics, MACD, ADX, EMA/SMA, Bollinger Bands, ATR, etc.
  - Interpretation helper: summarizes latest RSI/Stoch/MACD/ADX states (overbought/oversold, crossovers, trend strength)

- **F&O Analytics Hub**
  - **Options Chain** (NIFTY/BANKNIFTY/FINNIFTY): CE/PE OI, LTP, strikes, expiries, Most Active options dialog
  - **Greeks & IV**: Black–Scholes calculator for price, Delta, Gamma, Vega, Theta, Rho; implied volatility via Newton-Raphson
  - **PCR Analysis**: Total CE vs PE OI and sentiment bands with metrics
  - **Volatility & OI Surface**: Per-strike IVs and OI exploration with expiry-based T and risk-free rate inputs

- **AI Discovery and Portfolio-Aware Assistant**
  - In-app chat that can answer portfolio, holdings, and market questions (broker-connected context)
  - Smart prompts for common workflows (e.g., "show option chain for BANKNIFTY")

- **ML Forecasting**
  - Seasonal ARIMA training with seasonal decomposition; backtest fitted values and forecast next N days
  - Confidence intervals with seasonality reintroduced; MAPE utility; combined static CSV + live data loader (Zerodha or yfinance)

- **Algo Strategy Hub (Options Strategies)**
  - Multi-leg strategy builder (calls/puts, buy/sell, quantities, limit/market)
  - Payoff chart at expiry with breakeven points, max profit, max loss
  - Greeks aggregation and sensitivity exploration

- **Portfolio & Risk**
  - Live positions and holdings (via broker), order book, and day P&L
  - Allocation pies: stock-wise and sector-wise (sector mapping CSV)

- **Market Scanners**
  - **Momentum (RSI)**: overbought/oversold signals
  - **Trend (EMA alignment)**: uptrend/downtrend detection
  - **Breakout**: 20-day high/low breakout/breakdown
  - CSV export and quick add-to-watchlist actions

- **HFT Simulator**
  - Live tick log, market depth (top bids/asks), latency metric simulation
  - One-click market/limit buy/sell with lot-size aware quantity inputs

- **Basket Orders**
  - Prepare and place multiple orders in one flow with symbol validation from instruments

- **Authentication & Security**
  - 2FA with TOTP, QR provisioning, persistent secret management

- **Utilities**
  - India market holiday calendar cache; watchlists; quick trade dialog; session state bootstrap; autorefresh support

---

## 3) Trading Technology Stack

### 🏗️ **Trading Infrastructure**
- **UI Framework**: `streamlit` (real-time web interface), `streamlit-autorefresh` (auto-updating dashboards)
- **Time Management**: `datetime`, `pytz` (IST timezone handling, session tracking)

### 📊 **Market Data Engine**
- **Primary Feed**: `kiteconnect` (Zerodha KiteConnect API) – live quotes, tick data, market depth, order book
- **Global Markets**: `yfinance` – international indices (S&P 500, Nikkei, Hang Seng, GIFT NIFTY)
- **Market Intelligence**: `requests` (REST API calls), `feedparser` (RSS news aggregation)

### 🧠 **Analytics Powerhouse**
- **Core Analytics**: `pandas` (data manipulation), `numpy` (numerical computing), `tabulate` (formatted output)
- **Technical Analysis**: `pandas-ta` (50+ indicators – RSI, MACD, EMA, Bollinger Bands, ADX, Stochastics, ATR)
- **Statistical Modeling**: `scipy` (optimization, Greeks calculations), `statsmodels` (ARIMA forecasting, seasonality)

### 📈 **Visualization Suite**
- **Professional Charts**: `plotly` (`graph_objects`, `make_subplots`) – interactive, multi-timeframe, customizable themes

### 🔒 **Security Arsenal**
- **Authentication**: `pyotp` (TOTP 2FA), `qrcode` (authenticator provisioning)
- **Cryptography**: `hashlib` (secure hashing), `base64` (encoding)
- **Image Processing**: `Pillow` (PIL) for QR code generation and manipulation

### 🤖 **AI & Sentiment**
- **NLP Engine**: `vaderSentiment` (real-time news sentiment scoring)

### ⚙️ **Core Utilities**
- Pattern matching: `re` (regex)
- Randomization: `random` (simulation, testing)
- I/O operations: `io` (in-memory file handling)

---

## 4) Getting Started – Deploy Your Trading Edge

### 📋 **Prerequisites**

✅ **Python 3.9+** (recommended: Python 3.10 or 3.11 for optimal performance)  
✅ **Broker Integration** (optional): Zerodha Kite API credentials for live market access and order execution

---

### 🚀 **Installation Steps**

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/saumyasanghvi03/BlockVista-Terminal.git
cd BlockVista-Terminal
```

#### **Step 2: Set Up Virtual Environment**
```bash
# Create isolated Python environment
python -m venv .venv

# Activate environment
# Windows PowerShell/CMD:
.venv\Scripts\activate

# macOS/Linux/WSL:
source .venv/bin/activate
```

#### **Step 3: Install Dependencies**
```bash
# Install all required trading libraries
pip install -r requirements.txt
```

---

### ⚡ **Launch the Terminal**

```bash
streamlit run app.py
```

🎯 **Your trading terminal will launch at** `http://localhost:8501`

---

### 🔐 **Broker Setup (For Live Trading)**

**For Zerodha KiteConnect Integration:**

1️⃣ **Obtain API Credentials**  
   - Login to [Zerodha Kite Developer Console](https://developers.kite.trade/)  
   - Create a new app and note your **API Key** and **API Secret**

2️⃣ **Authenticate Your Session**  
   - Launch BlockVista Terminal  
   - Navigate to broker settings  
   - Enter your API credentials and complete authentication flow

3️⃣ **Enable 2FA Protection**  
   - Open "Generate QR for 2FA" dialog in-app  
   - Scan QR code with Google Authenticator or Microsoft Authenticator  
   - Verify OTP to secure your trading session

💡 **Pro Tip**: Keep your API credentials secure and never share them publicly

---

## 5) Usage Guide

- **Dashboard**
  - Monitor BMP score, index tiles, global indices, and sentiment-tagged news
  - Explore NIFTY 50 heatmap to spot top movers

- **Advanced Charting**
  - Choose symbols, add indicators, and switch layouts/timeframes; use interpretation helper for quick bias

- **F&O Analytics**
  - Load options chain for NIFTY/BANKNIFTY/FINNIFTY; compute IV and Greeks; check PCR and Most Active contracts

- **Algo Strategy Hub**
  - Add option legs; review payoff curve, breakevens, max profit/loss, and leg Greeks

- **Portfolio & Risk**
  - View holdings/positions with P&L; analyze allocation pies by stock and sector; inspect order book

- **Scanners**
  - Run Momentum/Trend/Breakout scans; export to CSV; add top candidates to watchlist

- **HFT Simulator**
  - Watch tick-by-tick changes and depth; place one-click market/limit orders

- **ML Forecasting**
  - Select instrument; train Seasonal ARIMA; review backtest and forecast with confidence bands

---

## 6) Market Data Integrations

### 🔌 **Live Trading & Market Depth**
**Zerodha KiteConnect API** – Your gateway to Indian markets  
✅ Real-time instrument data (NSE, BSE, MCX, CDS)  
✅ Streaming quotes with tick-by-tick precision  
✅ Order placement engine (market, limit, SL, SL-M, bracket orders)  
✅ Level-2 market depth (5-level bid/ask ladder)  
✅ Live positions, holdings, and order book tracking  
✅ Historical data for backtesting and analysis

### 🌏 **Global Market Intelligence**
**yfinance API** – Track worldwide market sentiment  
📍 S&P 500 (^GSPC) – U.S. market benchmark  
📍 Nikkei 225 (^N225) – Japanese market pulse  
📍 Hang Seng (^HSI) – Hong Kong/China exposure  
📍 GIFT NIFTY (IN=F) – Pre-market Indian sentiment proxy

### 📰 **Real-Time News & Sentiment**
**RSS Feeds with AI-Powered Sentiment Analysis**  
📈 **Economic Times** – Breaking market news  
📈 **Moneycontrol** – Stock-specific updates  
📈 **Business Standard** – Policy and regulatory news  
📈 **Livemint** – Sector insights  
📈 **Reuters India** – Global market impact  
📈 **BBC Business** – International perspective

**Powered by VADER Sentiment** – Instant bullish/bearish/neutral scoring on every headline

---

## 7) Technical Indicators Supported

Examples via pandas-ta and in-app logic:

- **Momentum**: RSI(14), Stochastic (14,3,3), MACD(12,26,9), ADX(14)
- **Trend**: EMA/SMA crossovers, multi-EMA alignment
- **Volatility**: ATR, Bollinger Bands
- **Options**: Black–Scholes Greeks (Delta, Gamma, Vega, Theta, Rho), Implied Volatility (Newton)

---

## 8) Contributing

We welcome contributions from students, traders, and developers!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit with clear messages: `git commit -m "feat: add amazing feature"`
4. Push and open a Pull Request

Areas: new indicators, scanners, ML models, UI polish, docs, tests, broker adapters

---

## 9) License

BlockVista Terminal — Creative Commons Zero v1.0 Universal (CC0-1.0) for authorized collaboration as stated by the project. See repository notices for commercial usage restrictions.

---

## Notes and Disclaimers

- BMP methodology: weighted blend of normalized recent NIFTY/SENSEX returns and inverse-normalized VIX
- Options analytics are for educational purposes; trading involves risk
- Real-time data subject to broker/exchange terms and latency
- Beta software; features may evolve frequently
