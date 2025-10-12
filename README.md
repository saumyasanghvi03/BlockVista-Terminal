# 🚀 BlockVista Terminal – Professional Indian Trading Platform

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Commercial-red.svg)](#license)
[![Market](https://img.shields.io/badge/Market-Indian%20Exchanges-orange.svg)](https://github.com/saumyasanghvi03/BlockVista-Terminal)

> Next-generation trading terminal for Indian capital markets with institutional-grade analytics, ML forecasting, and comprehensive F&O tools.

---

## 🎯 What's New in v2025.10

The latest enhancements to BlockVista Terminal include:

- **Advanced Technical Indicators**: Enhanced charting capabilities with additional momentum, volatility, and trend indicators for comprehensive market analysis
- **ML Forecasting Engine**: Advanced machine learning models for price prediction and market trend analysis
- **Enhanced Options Analytics**: Comprehensive options chain analysis with Greeks calculation and risk assessment
- **AI Trading Assistant**: Intelligent assistant for market insights and trading recommendations
- **Paper Trading Mode**: Risk-free simulation environment for strategy testing
- **Multi-Broker Support**: Expanded integration with multiple Indian brokers
- **Economic Calendar**: Real-time economic events and their market impact analysis
- **Enhanced Security & Authentication**: Improved user authentication and data protection
- **Advanced Dashboard**: Redesigned interface with customizable widgets and enhanced user experience

---

## 📊 What is BlockVista Terminal?

BlockVista Terminal is a professional-grade trading platform specifically designed for the Indian capital markets. Built with modern Python architecture and powered by Streamlit, it provides institutional-quality tools for retail and professional traders.

### 🇮🇳 Made for India

- **NSE/BSE Integration**: Direct connectivity to major Indian exchanges
- **Indian Market Hours**: Optimized for IST trading sessions (9:15 AM - 3:30 PM)
- **Rupee-Centric**: All calculations and displays in INR
- **Regulatory Compliance**: Adheres to SEBI guidelines and Indian market regulations
- **Local Broker Support**: Compatible with leading Indian brokers
- **Indian Economic Calendar**: Events relevant to Indian markets

---

## 🏆 Why Choose BlockVista Terminal?

| Feature | BlockVista Terminal | Traditional Platforms | Bloomberg Terminal |
|---------|-------------------|---------------------|-------------------|
| **Indian Market Focus** | ✅ Specialized | ❌ Generic | ✅ Available |
| **Cost** | 💰 Affordable | 💰 Low-Medium | 💰💰💰 Expensive |
| **Real-time Data** | ✅ NSE/BSE | ✅ Basic | ✅ Premium |
| **Technical Analysis** | ✅ 50+ Indicators | ❌ Limited | ✅ Extensive |
| **Options Analytics** | ✅ Advanced Greeks | ❌ Basic | ✅ Professional |
| **ML Forecasting** | ✅ Built-in | ❌ None | ✅ Advanced |
| **Paper Trading** | ✅ Included | ❌ Separate | ✅ Available |
| **AI Assistant** | ✅ Trading AI | ❌ None | ✅ Premium |
| **Multi-timeframe** | ✅ 1min-Monthly | ✅ Basic | ✅ Extensive |
| **Customization** | ✅ Open Source | ❌ Limited | ❌ Proprietary |

---

## 🛠️ Expanded Features

### 📈 Core Trading Features

- **Real-time Market Data**  
  - Live NSE/BSE quotes with microsecond precision
  - Streaming Level-II market depth
  - Real-time order book visualization
  - Tick-by-tick price updates
  - Historical data access (1min to monthly)

- **Advanced Charting**  
  - 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Multiple chart types (Candlestick, Line, Heikin-Ashi, Renko)
  - Multi-timeframe analysis (1min, 5min, 15min, 1H, 1D, 1W, 1M)
  - Custom indicator creation
  - Chart pattern recognition
  - Drawing tools and annotations

- **Options & Derivatives Analytics**  
  - Real-time options chain with bid-ask spreads
  - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
  - Implied volatility surface visualization
  - Options strategy builder (Straddles, Strangles, Spreads)
  - Max pain analysis
  - Put-Call ratio tracking
  - Open interest analysis

- **Machine Learning & AI**  
  - Price prediction models (LSTM, Prophet, ARIMA)
  - Sentiment analysis from news and social media
  - Pattern recognition and anomaly detection
  - Risk assessment algorithms
  - Portfolio optimization
  - AI-powered trading signals

- **Portfolio & Risk Management**  
  - Real-time P&L tracking
  - Position sizing calculators
  - Risk-reward ratio analysis
  - Stop-loss and target management
  - Diversification metrics
  - Performance analytics

### 🤖 AI Trading Assistant

- Natural language queries about market conditions
- Automated trading recommendations
- Risk assessment and portfolio suggestions
- Market sentiment analysis
- News impact prediction
- Personalized trading insights

### 📰 Market Intelligence

- **Economic Calendar**  
  - Indian and global economic events
  - Historical impact analysis
  - Customizable alerts

- **News Integration**  
  - Real-time financial news
  - Sentiment analysis
  - Impact on specific stocks

- **Social Sentiment**  
  - Twitter/Reddit sentiment tracking
  - Community trading sentiment
  - Influencer tracking

### 🎮 Paper Trading

- Risk-free simulation environment
- Real market data
- Virtual portfolio with ₹10,00,000 starting capital
- Complete trading functionality
- Performance tracking and analytics
- Strategy backtesting

### 🔌 Multi-Broker Integration

Supported brokers:
- Zerodha (Kite Connect)
- Angel One (SmartAPI)
- Upstox
- ICICI Direct
- 5Paisa
- Kotak Securities

### 🎨 Customization

- Fully customizable dashboard
- Widget-based interface
- Custom color schemes
- Personalized watchlists
- Saved layouts
- Custom alerts and notifications

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/saumyasanghvi03/BlockVista-Terminal.git
cd BlockVista-Terminal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Broker Configuration

1. Navigate to the Settings page
2. Select your broker
3. Enter API credentials (obtained from your broker's developer portal)
4. Enable desired features (real-time data, trading, etc.)

### Data Sources

The terminal uses multiple data sources:
- **NSE/BSE APIs**: Primary market data
- **Yahoo Finance**: Historical and supplementary data
- **News APIs**: Financial news and sentiment
- **Economic Calendar APIs**: Event data

---

## 📖 Usage Guide

### Dashboard

The main dashboard provides:
- Market overview (Nifty 50, Sensex, sectoral indices)
- Watchlist with real-time quotes
- Active positions and P&L
- Top gainers and losers
- Market breadth indicators

### Charts & Technical Analysis

1. Select a stock from the sidebar
2. Choose timeframe and chart type
3. Add indicators from the toolbar
4. Use drawing tools for analysis
5. Set alerts on price levels

### Options Chain

1. Navigate to Options Analytics
2. Select underlying stock
3. Choose expiry date
4. View options chain with Greeks
5. Analyze strategies using the strategy builder

### ML Forecasting

1. Go to AI Forecasting
2. Select stock and timeframe
3. Choose prediction model (LSTM/Prophet/ARIMA)
4. View predictions with confidence intervals
5. Analyze feature importance

### Paper Trading

1. Enable Paper Trading mode
2. Place virtual orders
3. Track performance
4. Test strategies risk-free

### AI Assistant

1. Open AI Assistant panel
2. Ask questions in natural language
3. Get market insights and recommendations
4. Request analysis on specific stocks

---

## 🔒 Security & Privacy

- **API Key Encryption**: All broker credentials are encrypted
- **Local Storage**: Data stored locally on your machine
- **No Data Collection**: We don't collect or transmit your trading data
- **Secure Connections**: All API calls use HTTPS
- **Open Source**: Full transparency - review the code yourself

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Charting**: Plotly, Matplotlib
- **Machine Learning**: TensorFlow, Scikit-learn, Prophet
- **APIs**: NSE/BSE APIs, Broker APIs, Financial APIs
- **Database**: SQLite (local storage)

---

## 📊 System Requirements

### Minimum
- OS: Windows 10, macOS 10.14, or Linux
- RAM: 4GB
- Storage: 500MB
- Internet: Stable connection for real-time data

### Recommended
- OS: Windows 11, macOS 12+, or Linux
- RAM: 8GB+
- Storage: 2GB
- Internet: High-speed broadband
- Display: 1920x1080 or higher

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

---

## 📞 Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/saumyasanghvi03/BlockVista-Terminal/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/saumyasanghvi03/BlockVista-Terminal/discussions)
- **Email**: Contact us at support@blockvista.in

---

## 📄 Credits & License

### License

© 2025 IDC Fintech Solutions. All rights reserved.

Developers: Saumya Sanghvi & Kanishk Mohan.

This is a commercial product. Unauthorized copying, modification, distribution, or use is strictly prohibited.

### Acknowledgments

- NSE/BSE for market data access
- Streamlit for the amazing framework
- Open-source community for various libraries
- Indian trading community for feedback and support

---

## ⚠️ Disclaimer

BlockVista Terminal is a trading tool designed for educational and informational purposes. Trading in financial markets involves substantial risk of loss. Past performance is not indicative of future results. The developers are not responsible for any financial losses incurred through the use of this software.

Always:
- Do your own research
- Understand the risks
- Trade responsibly
- Consult with a financial advisor
- Never invest more than you can afford to lose

---

**Built with ❤️ for Indian Traders**

[⭐ Star this repo](https://github.com/saumyasanghvi03/BlockVista-Terminal) | [🐛 Report Bug](https://github.com/saumyasanghvi03/BlockVista-Terminal/issues) | [💡 Request Feature](https://github.com/saumyasanghvi03/BlockVista-Terminal/issues)
