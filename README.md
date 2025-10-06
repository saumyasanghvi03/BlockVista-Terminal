# 🚀 BlockVista Terminal – Professional Indian Trading Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-CC0--1.0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Market](https://img.shields.io/badge/Market-Indian%20Exchanges-orange.svg)](https://github.com/saumyasanghvi03/BlockVista-Terminal)

> Next-generation trading terminal for Indian capital markets with institutional-grade analytics, ML forecasting, and comprehensive F&O tools.

---

## 📊 Project Overview

BlockVista Terminal is a comprehensive trading and analytics platform built for Indian financial markets (NSE, BSE, MCX, CDS). It combines professional trading tools with advanced analytics, machine learning forecasting, and real-time market intelligence in a modern web interface.

Built by traders, for traders — featuring Zerodha KiteConnect integration, proprietary Bharatiya Market Pulse (BMP) scoring, comprehensive F&O analytics, and AI-powered market discovery.

---

## ✨ Complete Feature Suite

### 📈 Dashboard & Market Intelligence
- 🇮🇳 Bharatiya Market Pulse (BMP): proprietary market sentiment scoring
  - Composite score using NIFTY/SENSEX and India VIX
  - Color-coded bands: Bharat Udaan (80–100) → Bharat Mandhi (0–20)
  - Real-time timing notifications with sound alerts
- 📊 Live Index Tracking: NIFTY 50, SENSEX, India VIX
- 🌏 Global Markets: S&P 500, Nikkei 225, Hang Seng
- 📰 Smart News Feed: RSS aggregation with sentiment scoring (Economic Times, Moneycontrol, Business Standard, Livemint, Reuters, BBC)
- 🔥 NIFTY 50 Heatmap: interactive treemap with real-time data

### 📊 Advanced Charting & Technical Analysis
- 📈 Professional Charts with multiple layouts and themes
- 🔧 50+ Technical Indicators
  - Momentum: RSI, Stochastic, MACD, Williams %R, ROC, MFI
  - Trend: EMA/SMA, ADX, Aroon, PSAR, Supertrend
  - Volatility: Bollinger Bands, ATR, Keltner Channels
  - Volume: OBV, CMF, VWAP
- 🎯 Smart Interpretation: automated signal insights
- ⚡ Multi-timeframe support: 1m to monthly

### 🎲 F&O Analytics Hub
- 📋 Options Chain: real-time CE/PE for NIFTY/BANKNIFTY/FINNIFTY
  - OI analysis, most active contracts, strike-wise LTP/volume/OI
- 🧮 Greeks & IV: Delta, Gamma, Vega, Theta, Rho and IV calculation
- 📊 PCR Analysis with sentiment interpretation
- 🌊 Volatility Surface: 3D IV visualization
- 📈 Strategy Builder: multi-leg strategy construction and payoff analysis

### 🤖 AI & Machine Learning
- 🧠 Portfolio-Aware AI Assistant
  - Natural-language queries, voice-enabled order intents, TA and news context
- 📈 ML Forecasting Engine
  - Seasonal ARIMA, confidence intervals, backtesting (MAPE)
- 🔍 AI Discovery: smart market insights and pattern recognition

### ⚡ Trading & Execution
- 🎯 HFT Simulator: tick-level streaming, order book depth, latency view
- 📦 Basket Orders: multi-symbol workflows with risk checks
- 🎛️ Quick Trade Dialog: instant order placement
- 📊 Futures Terminal: analytics and execution workflows

### 📊 Portfolio & Risk Management
- 💼 Live Portfolio: real-time P&L and positions
- 📈 Holdings Analysis: sector/stock allocation
- 📋 Order Book: full lifecycle tracking
- ⚠️ Risk Metrics: exposure and position sizing

### 🔍 Market Scanners
- ⚡ Momentum (RSI), 📈 Trend (EMA alignment), 💥 Breakout (S/R), 📊 Custom filters
- 📤 Export to CSV and watchlists

### 🔐 Security & Authentication
- 🔒 Two-Factor Authentication (2FA)
- 🔑 Secure sessions and encrypted credential storage
- 📱 Responsive design
- 🛡️ Built-in API rate limiting

### 🕐 Market Timing Features
- ⏰ Smart notifications for market open/close
- 📅 NSE/BSE holiday calendar (2024–2026)
- 🔔 IPO alerts
- ⚠️ 15-minute market close reminders

---

## 🆚 Market Positioning: BlockVista vs. Global Terminals

A concise, buyer-focused comparison across audience fit, India-market relevance, feature depth, pricing, deployment, and innovation.

| Platform | Core Audience | India-Market Relevance | Key Feature Focus | Pricing | Deployment | Innovation Edge |
|---|---|---|---|---|---|---|
| BlockVista Terminal | Active Indian traders, prop desks, advanced retail, small funds | High: BMP sentiment, F&O depth, NSE/BSE first-class | Options analytics, scanners, ML forecasts, AI assistant, basket/HFT sim | Pricing in process | Web app (desktop-first), quick setup | India-first design, AI-native workflows, rapid feature velocity |
| Bloomberg Terminal | Institutional trading/fund managers, research, sales & trading | Medium: India covered within global depth | News, analytics, messaging (IB), fixed income/FX depth | $$$$ (enterprise) | Dedicated desktop with hardware keys; on-prem/cloud access | Unmatched global data + IB network |
| LSEG Refinitiv Eikon | Sell-side/buy-side research, macro, multi-asset desks | Medium: broad India coverage | Cross-asset analytics, charting, news | $$$ (enterprise) | Desktop suite + cloud components | Integrations across LSEG stack |
| TradingView (Pro/Pro+) | Retail/creators, chartists | High: retail adoption in India | Charting, community scripts, alerts | $$ (retail tiers) | Web/mobile apps | Social/community ecosystem |
| Amibroker/NinjaTrader | System developers, quants, discretionary traders | Medium: requires broker/data adapters | Backtesting, strategy dev, execution bridges | $–$$ one-time + add-ons | Windows desktop | Local speed, extensibility |

Notes:
- BlockVista pricing: in process. Early partners and teams can reach out for custom onboarding.
- BlockVista focuses on India-first workflows and pragmatic depth for options/F&O while offering ML/AI insights that complement discretionary decision-making.

---

## 🚀 Quick Start Guide

### Prerequisites
- 🐍 Python 3.9+ (Recommended: 3.10 or 3.11)
- 💡 Zerodha KiteConnect API credentials (optional for live trading)
- 🌐 Internet connection for real-time data

### Installation
```bash
# Clone the repository
git clone https://github.com/saumyasanghvi03/BlockVista-Terminal.git
cd BlockVista-Terminal

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Terminal
```bash
streamlit run app.py
```
🎯 Access your trading terminal at: http://localhost:8501

---

## 📱 Usage Guide

### 🏠 Getting Started
1. Launch: open Dashboard
2. Connect: optional Zerodha broker integration for live data
3. Explore: browse modules and features
4. Customize: watchlists and indicators

### 📊 Key Workflows

#### Market Analysis
1. Check BMP for market sentiment
2. Review NIFTY 50 heatmap for sector moves
3. Scan news sentiment for catalysts
4. Use scanners for setups

#### Options Trading
1. Open F&O Analytics for options chain
2. Calculate Greeks and IV
3. Build multi-leg strategies and review payoff
4. Assess risk metrics

#### Portfolio Management
1. Review live P&L and positions
2. Analyze sector allocation
3. Monitor order execution
4. Set up risk alerts and limits

---

## 🔌 Data Sources & Integrations

### 🇮🇳 Indian Markets
- Zerodha KiteConnect: live quotes, orders, portfolio
  - NSE, BSE, MCX, CDS coverage
  - Real-time tick data and market depth
  - Historical data for backtesting

### 🌍 Global Markets
- yfinance API: S&P 500, Nikkei 225, Hang Seng; GIFT NIFTY (IN=F)

### 📰 News & Sentiment
- RSS feeds + sentiment scoring; multiple business sources

### 📊 Technical Data
- Static CSVs, live streams, and proprietary calculations

---

## 🔧 Configuration

### Broker Setup (Optional)
```python
# For Zerodha KiteConnect integration
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
REQUEST_TOKEN = "your_request_token"
```

### ML Data Sources
Pre-configured instruments for forecasting:
- NIFTY 50, BANK NIFTY, FINNIFTY
- SENSEX, GOLD, USDINR
- S&P 500 for correlation

---

## 🤝 Contributing

We welcome contributions from the trading and developer community!

### How to Contribute
1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/amazing-feature`
3. 💎 Commit changes: `git commit -m "feat: add amazing feature"`
4. 📤 Push to branch: `git push origin feature/amazing-feature`
5. 🎯 Open a Pull Request

### Areas for Contribution
- 📈 Indicators and strategies
- 🤖 ML models and algorithms
- 🎨 UI/UX and themes
- 📱 Mobile responsiveness
- 🔌 Broker integrations
- 📊 New scanners/screeners
- 📚 Docs and tutorials
- 🧪 Test coverage

---

## 📄 License

Creative Commons Zero v1.0 Universal (CC0-1.0)

BlockVista Terminal is released under CC0-1.0 for educational and collaborative purposes. See repository for commercial usage guidelines.

---

## ⚠️ Important Disclaimers
- 🎓 Educational use only
- 📊 Markets involve substantial risk of loss
- 🔍 Real-time data subject to exchange/broker terms
- 🚧 Beta software; features may change
- 💡 BMP methodology uses NIFTY/SENSEX/VIX analysis
- 🔒 Respect broker API rate limits and terms

---

## 📞 Support & Community
- 🐛 Issues: file bugs and feature requests on GitHub
- 💬 Discussions: join the community for insights
- 📧 Contact: reach out for collaboration
- 🌟 Star the Repo if you find it useful

---

<div align="center">

**🚀 Transform Your Trading Experience with BlockVista Terminal**

Professional tools • Real-time data • Advanced analytics • Made in India 🇮🇳

[![GitHub stars](https://img.shields.io/github/stars/saumyasanghvi03/BlockVista-Terminal?style=social)](https://github.com/saumyasanghvi03/BlockVista-Terminal/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/saumyasanghvi03/BlockVista-Terminal?style=social)](https://github.com/saumyasanghvi03/BlockVista-Terminal/network/members)

</div>
