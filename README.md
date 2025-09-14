# 🚀 BlockVista Terminal™

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-CC0--1.0-lightgrey?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Made for India](https://img.shields.io/badge/Made%20for-India%20🇮🇳-orange?style=for-the-badge)

> **The New-Age Bloomberg Terminal for Indian Stock Market Intraday Trading**
>
> A legendary market cockpit—reborn for India's digital intraday era.
> Power, speed, and insight tailored exclusively for Indian equity traders and analysts.
>
> **Welcome to your BlockVista Terminal.™**

<div align="center">
  <img src="https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/assets/bloomberg-style-banner.png" alt="BlockVista Banner" width="800"/>
</div>

---

## 🏆 What is BlockVista Terminal?

**BlockVista Terminal™** is the next-generation financial command center—the *Bloomberg Terminal reinvented* specifically for **Indian stock market intraday trading**.

Built from the ground up for NSE/BSE equity markets, it delivers a professional-grade interface inspired by Wall Street's finest—but optimized for the unique demands of Indian intraday traders, technical analysts, and market professionals.

### 🇮🇳 **Made for India. Built for Intraday.**

- ✅ **NSE/BSE Live Data**: Real-time prices, volumes, and market depth
- ✅ **Intraday Focus**: Lightning-fast technical analysis for day trading
- ✅ **Indian Broker Integration**: Direct Zerodha/Kite API connectivity
- ✅ **Actionable Intelligence**: Not just charts—execute trades instantly
- ✅ **Professional Grade**: Bloomberg-style interface, browser-native speed

---

## 🌐 Why Choose BlockVista Terminal?

| 🚩 **BlockVista Terminal™** | 🏢 **Legacy Bloomberg** |
|---|---|
| Made for Indian equity markets | US/global focus, expensive Indian data |
| Browser-native, mobile-ready | Requires dedicated terminals |
| **Free & Open Source** | **₹15+ Lakhs yearly** |
| Zerodha/Indian broker integration | Limited Indian broker support |
| Intraday trading optimized | Institution-focused |
| Modern, intuitive interface | Complex, legacy UI |
| Real-time NSE/BSE data | Expensive Indian market feeds |

---

## ✨ Key Features

### 📈 **Live Indian Market Intelligence**
- **Real-time NSE/BSE prices** with millisecond updates
- **Live market depth** and order book visualization  
- **Intraday charts** with 1min, 5min, 15min timeframes
- **Volume analysis** and price action insights
- **Market breadth** indicators (advance/decline, etc.)

### 🔍 **Advanced Technical Analysis**
- **Multi-indicator signals**: RSI, MACD, ADX, Bollinger Bands, Stochastic
- **Candlestick patterns** recognition and alerts
- **Moving averages**: EMA, SMA, VWAP with dynamic crossovers
- **Supertrend & pivot levels** for intraday support/resistance
- **Custom screeners** for breakouts, momentum, and reversal setups

### ⚡ **Instant Order Execution**
- **Direct Zerodha integration** via Kite Connect API
- **One-click trading** from charts and screeners
- **Live P&L tracking** with real-time position monitoring
- **Order management** with stop-loss and target automation
- **Trade alerts** via notifications and webhooks

### 🎯 **Smart Market Screeners**
- **Breakout scanner**: Volume + price breakouts
- **Momentum finder**: High RSI, MACD crossovers
- **Gap scanner**: Opening gaps with follow-through potential
- **Custom filters**: Market cap, sector, technical criteria
- **Intraday watchlists** with real-time updates

### 🔔 **Intelligent Alerts System**
- **Price alerts**: Target/stop-loss notifications
- **Technical alerts**: Indicator crossovers, pattern formations
- **Volume alerts**: Unusual volume spikes
- **News integration**: Market-moving events (roadmap)
- **Multi-channel delivery**: Email, SMS, browser notifications

---

## 📸 Visual Demos

### 📊 **Main Trading Dashboard**
<div align="center">
  <img src="https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/assets/main-dashboard-demo.png" alt="Main Dashboard" width="900"/>
  <br/><em>Live market data, charts, and technical analysis in one powerful interface</em>
</div>

### 📈 **Technical Analysis & Signals**
<div align="center">
  <img src="https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/assets/technical-analysis-demo.png" alt="Technical Analysis" width="900"/>
  <br/><em>Advanced charting with multi-timeframe analysis and signal generation</em>
</div>

### 💹 **Portfolio & P/L Tracking**
<div align="center">
  <img src="https://raw.githubusercontent.com/saumyasanghvi03/BlockVista-Terminal/main/assets/portfolio-tracking-demo.png" alt="Portfolio Tracking" width="900"/>
  <br/><em>Real-time portfolio performance with detailed P&L breakdown</em>
</div>

---

## 🛠️ Tech Stack

**Frontend & UI**
- 🎨 **Streamlit** - Modern web app framework
- 📊 **Plotly** - Interactive financial charts
- 🎯 **Pandas** - Data manipulation and analysis

**Market Data & APIs**
- 📡 **NSE/BSE APIs** - Live market data feeds
- 🔗 **Zerodha Kite Connect** - Order execution and portfolio
- 📈 **Technical Analysis Library** - TA-Lib integration

**Backend & Processing**
- ⚡ **Python 3.9+** - Core application logic
- 🔄 **AsyncIO** - Concurrent data processing
- 💾 **SQLite/PostgreSQL** - Local data storage

---

## 🚀 Quick Getting Started

### 1️⃣ **Clone & Install**
```bash
git clone https://github.com/saumyasanghvi03/BlockVista-Terminal.git
cd BlockVista-Terminal
pip install -r requirements.txt
```

### 2️⃣ **Configure API Secrets**
Create `.streamlit/secrets.toml` with your Zerodha credentials:
```toml
[zerodha]
api_key = "your_kite_api_key"
api_secret = "your_kite_api_secret"
access_token = "your_access_token"

[market_data]
nse_api_key = "your_nse_api_key"  # Optional
```

### 3️⃣ **Launch Terminal**
```bash
streamlit run app.py
```

**🎉 That's it!** Access your Bloomberg-style terminal at `http://localhost:8501`

### 📋 **Prerequisites**
- **Zerodha Account** with Kite Connect API access
- **Python 3.9+** installed
- **NSE/BSE data subscription** (free tier available)

---

## 💡 Modern Usage & User Workflow

### 🌅 **Pre-Market Setup (9:00 AM)**
1. **Launch Terminal** and verify live data connectivity
2. **Review overnight news** and global market moves
3. **Scan for gaps** and pre-market movers
4. **Set watchlists** for the day's potential trades

### 📈 **Market Hours Trading (9:15 AM - 3:30 PM)**
1. **Monitor live charts** with real-time technical indicators
2. **Execute trades** directly from chart interface
3. **Track P&L** and manage positions in real-time
4. **Receive alerts** for breakouts and technical signals

### 📊 **Post-Market Analysis (3:30 PM+)**
1. **Review trade performance** and P&L analytics
2. **Export data** for detailed analysis
3. **Plan next day** watchlists and strategies
4. **Study market patterns** and refine approach

---

## 🛣️ Upgrades & Roadmap

### 🎯 **Phase 1: Core Enhancement** (Q1 2025)
- [ ] **Advanced Indicators**: Ichimoku, Williams %R, Commodity Channel Index
- [ ] **Pattern Recognition**: Head & Shoulders, Double Top/Bottom, Triangles
- [ ] **Enhanced Screeners**: Custom technical combinations, sector rotation
- [ ] **Performance Analytics**: Detailed trade reports, win/loss analysis

### 🚀 **Phase 2: Trading Expansion** (Q2 2025)
- [ ] **Futures & Options**: F&O chain analysis, option Greeks, strategy builder
- [ ] **Multi-Broker Support**: Angel Broking, Upstox, 5Paisa integration
- [ ] **Advanced Orders**: OCO, bracket orders, algorithmic execution
- [ ] **Risk Management**: Position sizing, portfolio risk metrics

### 🌐 **Phase 3: Intelligence Layer** (Q3 2025)
- [ ] **News Sentiment**: AI-powered news analysis affecting stock prices
- [ ] **Social Sentiment**: Twitter, Reddit sentiment for retail stocks
- [ ] **Export & Reporting**: PDF reports, Excel integration, API access
- [ ] **Machine Learning**: Price prediction models, pattern forecasting

### ☁️ **Phase 4: Scale & Mobile** (Q4 2025)
- [ ] **Cloud Deployment**: AWS/Azure hosting with multi-user support
- [ ] **Mobile App**: Native iOS/Android trading interface
- [ ] **Institutional Features**: Multi-account management, team collaboration
- [ ] **Crypto Integration**: Bitcoin, Ethereum as alternative assets (regulatory permitting)

---

## 🤝 Contributing

**BlockVista Terminal™** is built for the Indian fintech ecosystem. We welcome contributions from:

### 🎯 **Target Contributors**
- **Indian traders & analysts** with market expertise
- **Python developers** experienced in financial applications
- **UI/UX designers** familiar with trading interfaces
- **Data engineers** working with market data feeds

### 📝 **How to Contribute**
1. **Fork the repository** and create a feature branch
2. **Follow Indian market standards** and trading conventions
3. **Test with NSE/BSE data** and Zerodha integration
4. **Submit pull requests** with clear documentation

### 🔥 **Priority Areas**
- Enhanced technical indicators and Indian-specific patterns
- Better Zerodha/Indian broker integrations
- Mobile-responsive design improvements
- Performance optimization for real-time data

---

## 📄 License & Copyright

**© 2025 Saumya Sanghvi. All rights reserved.**

This project is licensed under [CC0 1.0 Universal](https://github.com/saumyasanghvi03/BlockVista-Terminal/blob/main/LICENSE) — **PRs and forks are welcome!**

**Built with ❤️ and Mumbai energy by [Saumya Sanghvi](https://github.com/saumyasanghvi03)**

---

<div align="center">

### 🇮🇳 **Experience the Revolution**

**BlockVista Terminal™** — *Your Bloomberg, reinvented for Indian markets.*

**Analyze. Trade. Win.**

The future of Indian trading is not expensive.\
It's open-source.\
And it's live.

**🌐💸 [Get Started Now →](https://github.com/saumyasanghvi03/BlockVista-Terminal)**

</div>
