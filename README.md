## 🚀 BlockVista Terminal — Professional Indian Trading Platform

<p align="center">
  <img src="image.jpg" height="36" title="Blockchain" />
  <img src="image.jpg" height="36" title="Broker" />
  <img src="image.jpg" height="36" title="Institutional" />
  <img src="image.jpg" height="36" title="Security" />
  <img src="image.jpg" height="36" title="Fintech" />
</p>

> Next-generation trading terminal for Indian capital markets with institutional-grade analytics, ML forecasting, and comprehensive F&O tools.

[![License: Commercial](https://img.shields.io/badge/License-Commercial-blue)](https://github.com/saumyasanghvi03/BlockVista-Terminal/tree/main#license)

---

## 🎯 Key Features

### 📊 **Advanced Dashboards**
Real-time monitoring and analysis tools designed for serious traders:
- **Multi-Market View:** Integrate NSE equities, F&O, commodities (MCX), and bonds in one seamless workspace
- **Custom Widgets:** Arrange your trading desk with drag-and-drop interface for optimal workflow
- **Live Market Depth:** Real-time order book visualization showing bid/ask spreads, volumes, and liquidity
- **AI-Powered Alerts:** Smart notifications for breakouts, unusual volumes, and algorithmic pattern detection
- **Professional Charting:** TradingView-style charts with 100+ technical indicators, drawing tools, and pattern recognition

### 🔐 **Seamless Onboarding**
Professional-grade account setup and trading infrastructure:
- **Direct Broker Integration:** Connect with Angel One, Zerodha, Upstox, 5Paisa, ICICI Direct, and more
- **OAuth 2.0 Security:** Bank-level encrypted authentication with automatic session management
- **Portfolio Import:** Migrate existing holdings and watchlists automatically
- **Risk Profile Setup:** Configure position sizing, stop-loss rules, and margin requirements
- **Paper Trading Mode:** Test strategies with real market data before committing capital

### 🏛️ **Institutional Compliance & Security**
Enterprise-grade protection and regulatory adherence:
- **SEBI Compliance:** Full adherence to Indian capital markets regulations and reporting requirements
- **Encrypted Communications:** TLS 1.3 end-to-end encryption for all trade orders and data transmission
- **Audit Trail:** Complete trade history and decision logs for regulatory compliance
- **Multi-Factor Authentication:** Hardware token support, biometric verification, and time-based OTPs
- **IP Whitelisting:** Restrict access to trusted networks for enhanced security
- **Disaster Recovery:** Automated backups and failover systems for business continuity

### 📈 **F&O Trading Tools**
- **Options Chain Analyzer:** Real-time Greeks (Delta, Gamma, Vega, Theta), OI analysis, and PCR ratio
- **Strategy Builder:** Create multi-leg spreads (straddles, strangles, iron condors) with visual P&L curves
- **Margin Calculator:** Real-time SPAN and exposure margin computation for NSE/MCX derivatives
- **Max Pain & IV Analysis:** Identify key strike prices and implied volatility trends
- **Historical Data:** Access 10+ years of options data for backtesting and research

### 🤖 **ML-Powered Predictions**
- **LSTM Price Forecasting:** Next-day price predictions with confidence intervals
- **Pattern Recognition:** Automated detection of chart patterns, support/resistance, and trend lines
- **Sentiment Analysis:** Real-time news sentiment scoring from 50+ financial sources
- **Correlation Matrix:** Identify inter-market relationships and sector rotation signals
- **Anomaly Detection:** Flag unusual trading activity or potential pump-and-dump schemes

### 💹 **Advanced Analytics**
- **Portfolio Performance:** Track CAGR, Sharpe ratio, maximum drawdown, and risk-adjusted returns
- **Sector Rotation:** Identify leadership shifts across NIFTY sectors with heat maps
- **Relative Strength:** Compare stock performance against sector and benchmark indices
- **Volume Profile:** Analyze volume distribution at different price levels
- **Event Calendar:** Earnings dates, dividend declarations, policy meetings, and macro events

### 🔄 **Semi-Automated Algo Bots**
- **Custom Trade Rules:** Set entry/exit conditions based on technical indicators, time, or external triggers
- **Signal Generation:** Automated alerts for your custom strategies (still requires manual confirmation)
- **Parameter Tuning:** Optimize bot parameters with built-in backtesting over historical data
- **Risk Controls:** Set daily loss limits, position caps, and maximum order size per trade
- **Multi-Instrument Support:** Run bots on equities, futures, options, or commodities simultaneously
- **Scenario Alerts:** Get notified when specific market conditions align with your strategy criteria

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Valid broker account (Angel One, Zerodha, etc.) with API access
- Active internet connection

### Installation

```bash
git clone https://github.com/saumyasanghvi03/BlockVista-Terminal.git
cd BlockVista-Terminal
pip install -r requirements.txt
streamlit run app.py
```

### Configuration

1. **API Credentials:** Add your broker API keys to `config/credentials.json`
2. **Database Setup:** SQLite for local storage (no external DB required)
3. **ML Models:** Pre-trained models included; retrain with your data if needed

---

## 📁 Repository Structure

```
BlockVista-Terminal/
├── app.py                 # Main Streamlit application
├── components/
│   ├── dashboard.py      # Multi-market dashboard
│   ├── fo_tools.py       # F&O analysis modules
│   ├── ml_models.py      # LSTM and prediction engine
│   └── algo_bots.py      # Algorithmic trading bots
├── config/
│   ├── credentials.json  # API keys and settings
│   └── risk_params.json  # Risk management rules
├── data/
│   └── historical/       # Cached market data
├── models/
│   └── lstm_model.h5     # Pre-trained forecasting model
├── utils/
│   ├── broker_api.py     # Broker integration layer
│   ├── data_fetcher.py   # Market data retrieval
│   └── analytics.py      # Technical analysis functions
└── requirements.txt      # Python dependencies
```

---

## 🤝 Contributing

This is a proprietary project. For collaboration or feature requests, contact **IDC Fintech Solutions**.

---

## 📜 License

Copyright © 2025 **IDC Fintech Solutions**. All rights reserved.

Developed by:
- **Saumya Sanghvi** (Lead Developer)
- **Dakshil Madani** (ML Engineering)
- **Chaitya Shah** (Backend & APIs)

**Commercial License:** This software is licensed for commercial use. Unauthorized copying, distribution, or modification is prohibited. Contact **IDC Fintech Solutions** for licensing inquiries.

---

## 📧 Support

For technical support or partnership opportunities:
- 📩 Email: support@idcfintech.com
- 🌐 Website: [idcfintech.com](https://idcfintech.com)
- 💼 LinkedIn: [IDC Fintech Solutions](https://linkedin.com/company/idcfintech)

---

<p align="center">Made with ❤️ for Indian traders by IDC Fintech Solutions</p>
