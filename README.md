# 🚀 BlockVista Terminal – Professional Indian Trading Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-CC0--1.0-green.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Trading](https://img.shields.io/badge/Market-Indian%20Exchanges-orange.svg)](https://github.com/saumyasanghvi03/BlockVista-Terminal)

> *Next-generation trading terminal for Indian capital markets with institutional-grade analytics, ML forecasting, and comprehensive F&O tools.*

---

## 📊 Project Overview

BlockVista Terminal is a comprehensive, Streamlit-powered trading and analytics platform specifically designed for Indian financial markets (NSE, BSE, MCX, CDS). It combines professional trading tools with advanced analytics, machine learning forecasting, and real-time market intelligence in a modern web interface.

**Built by traders, for traders** - featuring Zerodha KiteConnect integration, proprietary Bharatiya Market Pulse (BMP) scoring, comprehensive F&O analytics, and AI-powered market discovery.

---

## ✨ Complete Feature Suite

### 📈 **Dashboard & Market Intelligence**
- **🇮🇳 Bharatiya Market Pulse (BMP)**: Proprietary market sentiment scoring system
  - Composite score using NIFTY/SENSEX performance and India VIX
  - Color-coded sentiment bands: Bharat Udaan (80-100) to Bharat Mandhi (0-20)
  - Real-time market timing notifications with sound alerts
- **📊 Live Index Tracking**: NIFTY 50, SENSEX, India VIX with percentage changes
- **🌏 Global Markets**: S&P 500, Nikkei 225, Hang Seng via yfinance integration
- **📰 Smart News Feed**: RSS aggregation with VADER sentiment analysis
  - Sources: Economic Times, Moneycontrol, Business Standard, Livemint, Reuters, BBC
- **🔥 NIFTY 50 Heatmap**: Interactive treemap visualization with real-time data

### 📊 **Advanced Charting & Technical Analysis**
- **📈 Professional Charts**: Plotly-powered with multiple layouts and themes
- **🔧 50+ Technical Indicators** (via pandas-ta):
  - **Momentum**: RSI, Stochastic, MACD, Williams %R, ROC, MFI
  - **Trend**: EMA/SMA, ADX, Aroon, PSAR, Supertrend
  - **Volatility**: Bollinger Bands, ATR, Keltner Channels
  - **Volume**: OBV, CMF, VWAP
- **🎯 Smart Interpretation**: Automated indicator analysis and trading signals
- **⚡ Multi-timeframe Support**: 1min to monthly intervals

### 🎲 **F&O Analytics Hub**
- **📋 Options Chain**: Real-time CE/PE data for NIFTY/BANKNIFTY/FINNIFTY
  - Open Interest analysis and Most Active contracts
  - Strike-wise LTP, volume, and OI tracking
- **🧮 Greeks Calculator**: Complete Black-Scholes implementation
  - Delta, Gamma, Vega, Theta, Rho calculations
  - Implied Volatility using Newton-Raphson method
- **📊 PCR Analysis**: Put-Call Ratio with sentiment interpretation
- **🌊 Volatility Surface**: 3D visualization of implied volatilities
- **📈 Options Strategy Builder**: Multi-leg strategy construction with payoff analysis

### 🤖 **AI & Machine Learning**
- **🧠 Portfolio-Aware AI Assistant**: 
  - Natural language queries about positions and market
  - Order placement via voice commands
  - Technical analysis interpretation
  - News sentiment analysis
- **📈 ML Forecasting Engine**:
  - Seasonal ARIMA models with confidence intervals
  - Backtesting with MAPE accuracy metrics
  - Multiple data sources: Static CSV + Live feeds
- **🔍 AI Discovery**: Smart market insights and pattern recognition

### ⚡ **Trading & Execution**
- **🎯 HFT Terminal**: High-frequency trading simulator
  - Real-time tick data and market depth
  - Latency monitoring and order book analysis
  - One-click market/limit order execution
- **📦 Basket Orders**: Multi-symbol order management
  - Bulk order preparation and validation
  - Risk assessment and position sizing
- **🎛️ Quick Trade Dialog**: Instant order placement from any screen
- **📊 Futures Terminal**: Complete futures analysis and trading

### 📊 **Portfolio & Risk Management**
- **💼 Live Portfolio Tracking**: Real-time P&L and positions
- **📈 Holdings Analysis**: Sector-wise and stock-wise allocation
- **📋 Order Management**: Complete order book with status tracking
- **⚠️ Risk Metrics**: Position sizing and exposure analysis

### 🔍 **Market Scanners**
- **⚡ Momentum Scanner**: RSI-based overbought/oversold detection
- **📈 Trend Scanner**: EMA alignment and trend strength analysis
- **💥 Breakout Scanner**: Support/resistance level breaks
- **📊 Custom Filters**: User-defined screening criteria
- **📤 Export Capabilities**: CSV export and watchlist integration

### 🔐 **Security & Authentication**
- **🔒 Two-Factor Authentication (2FA)**: TOTP implementation with QR codes
- **🔑 Secure Session Management**: Encrypted credential storage
- **📱 Mobile-Friendly**: Responsive design for all devices
- **🛡️ API Rate Limiting**: Built-in protection against over-usage

### 🕐 **Market Timing Features**
- **⏰ Smart Notifications**: Market open/close alerts
- **📅 Holiday Calendar**: NSE/BSE holiday tracking (2024-2026)
- **🔔 IPO Alerts**: Pre-opening and execution notifications
- **⚠️ Closing Warnings**: 15-minute market close reminders

---

## 🛠️ Technology Stack

### **Core Framework**
- **Streamlit**: Modern web app framework
- **Plotly**: Interactive charting and visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### **Trading & Market Data**
- **Zerodha KiteConnect**: Live market data and order execution
- **yfinance**: Global market data and historical prices
- **pandas-ta**: Technical analysis library (50+ indicators)

### **Machine Learning & AI**
- **Seasonal ARIMA**: Time series forecasting
- **VADER Sentiment**: News sentiment analysis
- **scikit-learn**: Statistical modeling

### **Financial Mathematics**
- **Black-Scholes**: Options pricing and Greeks
- **Newton-Raphson**: Implied volatility calculation
- **Statistical Analysis**: Risk metrics and portfolio analysis

---

## 🚀 Quick Start Guide

### **Prerequisites**
- 🐍 Python 3.9+ (Recommended: 3.10 or 3.11)
- 💡 Zerodha KiteConnect API credentials (optional for live trading)
- 🌐 Internet connection for real-time data

### **Installation**

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

### **Launch Terminal**

```bash
streamlit run app.py
```

🎯 **Access your trading terminal at**: `http://localhost:8501`

---

## 📱 Usage Guide

### **🏠 Getting Started**
1. **Launch**: Start the application and navigate to Dashboard
2. **Connect**: Optional Zerodha broker integration for live data
3. **Explore**: Browse through different modules and features
4. **Customize**: Set up watchlists and preferred indicators

### **📊 Key Workflows**

#### **Market Analysis**
1. Check BMP score for overall market sentiment
2. Review NIFTY 50 heatmap for sector rotation
3. Analyze news sentiment for market-moving events
4. Use scanners to identify trading opportunities

#### **Options Trading**
1. Navigate to F&O Analytics for options chain
2. Calculate Greeks and implied volatility
3. Build multi-leg strategies in Strategy Builder
4. Analyze payoff diagrams and risk metrics

#### **Portfolio Management**
1. Review live positions and P&L
2. Analyze sector allocation and concentration
3. Monitor order execution and status
4. Set up risk alerts and position limits

---

## 🔌 Data Sources & Integrations

### **🇮🇳 Indian Markets**
- **Zerodha KiteConnect**: Live quotes, orders, portfolio
  - NSE, BSE, MCX, CDS coverage
  - Real-time tick data and market depth
  - Historical data for backtesting

### **🌍 Global Markets**
- **yfinance API**: International indices and data
  - S&P 500, Nikkei 225, Hang Seng
  - GIFT NIFTY (IN=F) for pre-market sentiment

### **📰 News & Sentiment**
- **RSS Feeds**: Real-time news aggregation
- **VADER Sentiment**: AI-powered sentiment scoring
- **Multiple Sources**: Comprehensive market coverage

### **📊 Technical Data**
- **Static CSV Files**: Historical data for ML training
- **Real-time Streams**: Live price feeds and market data
- **Custom Calculations**: Proprietary indicators and scores

---

## 🔧 Configuration

### **Broker Setup (Optional)**
```python
# For Zerodha KiteConnect integration
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
REQUEST_TOKEN = "your_request_token"
```

### **ML Data Sources**
Pre-configured instruments for ML forecasting:
- NIFTY 50, BANK NIFTY, FINNIFTY
- SENSEX, GOLD, USDINR
- S&P 500 for global correlation

---

## 🤝 Contributing

We welcome contributions from the trading and developer community!

### **How to Contribute**
1. 🍴 Fork the repository
2. 🌿 Create a feature branch: `git checkout -b feature/amazing-feature`
3. 💎 Commit changes: `git commit -m "feat: add amazing feature"`
4. 📤 Push to branch: `git push origin feature/amazing-feature`
5. 🎯 Open a Pull Request

### **Areas for Contribution**
- 📈 New technical indicators and strategies
- 🤖 Enhanced ML models and algorithms
- 🎨 UI/UX improvements and themes
- 📱 Mobile responsiveness enhancements
- 🔌 Additional broker integrations
- 📊 New market scanners and screeners
- 📚 Documentation and tutorials
- 🧪 Test coverage and quality assurance

---

## 📄 License

**Creative Commons Zero v1.0 Universal (CC0-1.0)**

BlockVista Terminal is released under CC0-1.0 license for educational and collaborative purposes. See repository for commercial usage guidelines.

---

## ⚠️ Important Disclaimers

- **🎓 Educational Purpose**: This platform is designed for learning and research
- **📊 Market Risk**: Trading involves substantial risk of loss
- **🔍 Data Accuracy**: Real-time data subject to exchange delays and broker terms
- **🚧 Beta Software**: Features may change frequently during development
- **💡 BMP Methodology**: Proprietary scoring based on NIFTY/SENSEX/VIX analysis
- **🔒 API Limits**: Respect broker API rate limits and terms of service

---

## 📞 Support & Community

- 🐛 **Issues**: Report bugs and feature requests on GitHub
- 💬 **Discussions**: Join the community for trading insights
- 📧 **Contact**: Reach out for collaboration opportunities
- 🌟 **Star the Repo**: Show your support for the project!

---

<div align="center">

**🚀 Transform Your Trading Experience with BlockVista Terminal**

*Professional tools • Real-time data • Advanced analytics • Made in India 🇮🇳*

[![GitHub stars](https://img.shields.io/github/stars/saumyasanghvi03/BlockVista-Terminal?style=social)](https://github.com/saumyasanghvi03/BlockVista-Terminal/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/saumyasanghvi03/BlockVista-Terminal?style=social)](https://github.com/saumyasanghvi03/BlockVista-Terminal/network/members)

</div>
