# app.py - Complete Algorithmic Trading Application with AI Trading Bots
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import requests
import json
from datetime import datetime, timedelta
import time
import random
from textblob import TextBlob
import re

warnings.filterwarnings('ignore')

class TradingBot:
    """Base class for all trading bots"""
    def __init__(self, name, allocated_capital, risk_tolerance="medium"):
        self.name = name
        self.allocated_capital = allocated_capital
        self.risk_tolerance = risk_tolerance
        self.portfolio = {}
        self.trade_history = []
        self.current_cash = allocated_capital
        self.performance_metrics = {}
        
    def analyze_market_conditions(self):
        """Analyze overall market conditions"""
        raise NotImplementedError
        
    def scan_opportunities(self):
        """Scan for trading opportunities"""
        raise NotImplementedError
        
    def execute_trades(self):
        """Execute trades based on analysis"""
        raise NotImplementedError
        
    def calculate_position_size(self, symbol_price, risk_per_trade=0.02):
        """Calculate position size based on risk management"""
        if self.risk_tolerance == "low":
            risk_per_trade = 0.01
        elif self.risk_tolerance == "high":
            risk_per_trade = 0.04
            
        max_risk_amount = self.allocated_capital * risk_per_trade
        position_size = max_risk_amount / symbol_price
        return min(position_size, self.current_cash / symbol_price)

class MomentumTrader(TradingBot):
    """Momentum trading bot that follows trends"""
    
    def __init__(self, allocated_capital, risk_tolerance="medium"):
        super().__init__("Momentum Trader", allocated_capital, risk_tolerance)
        self.momentum_threshold = 0.05  # 5% momentum threshold
        self.volume_spike_threshold = 1.5  # 150% volume spike
        
    def analyze_market_conditions(self):
        """Analyze market momentum conditions"""
        try:
            # Get market indices
            spy_data = yf.download('SPY', period='5d', interval='1d')
            if not spy_data.empty:
                spy_return = (spy_data['Close'][-1] / spy_data['Close'][0] - 1) * 100
                market_trend = "bullish" if spy_return > 0.5 else "bearish" if spy_return < -0.5 else "neutral"
                return {
                    "market_trend": market_trend,
                    "spy_return": spy_return,
                    "volatility": spy_data['Close'].pct_change().std() * 100
                }
        except:
            return {"market_trend": "neutral", "spy_return": 0, "volatility": 0}
        
    def calculate_momentum_score(self, symbol):
        """Calculate momentum score for a symbol"""
        try:
            data = yf.download(symbol, period='30d', interval='1d')
            if data.empty or len(data) < 20:
                return 0
                
            # Price momentum (20-day return)
            price_momentum = (data['Close'][-1] / data['Close'][0] - 1) * 100
            
            # Volume momentum
            avg_volume = data['Volume'][:-5].mean()
            recent_volume = data['Volume'][-5:].mean()
            volume_momentum = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # RSI-like momentum
            gains = data['Close'].pct_change().apply(lambda x: x if x > 0 else 0)
            losses = -data['Close'].pct_change().apply(lambda x: x if x < 0 else 0)
            avg_gain = gains.rolling(window=14).mean().iloc[-1]
            avg_loss = losses.rolling(window=14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi_momentum = 100 - (100 / (1 + rs))
            
            # Combined momentum score
            momentum_score = (price_momentum * 0.5 + 
                            (volume_momentum - 1) * 100 * 0.3 + 
                            (rsi_momentum - 50) * 0.2)
                            
            return momentum_score
            
        except:
            return 0
    
    def scan_opportunities(self, watchlist):
        """Scan for momentum opportunities"""
        opportunities = []
        
        for symbol in watchlist:
            momentum_score = self.calculate_momentum_score(symbol)
            
            if momentum_score > self.momentum_threshold * 100:  # Convert to percentage
                data = yf.download(symbol, period='5d', interval='1d')
                if not data.empty:
                    current_price = data['Close'][-1]
                    position_size = self.calculate_position_size(current_price)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'momentum_score': momentum_score,
                        'current_price': current_price,
                        'position_size': position_size,
                        'action': 'BUY',
                        'confidence': min(momentum_score / 10, 1.0)
                    })
                    
        return sorted(opportunities, key=lambda x: x['momentum_score'], reverse=True)[:5]
    
    def execute_trades(self, opportunities):
        """Execute momentum trades"""
        executed_trades = []
        
        for opp in opportunities:
            if opp['confidence'] > 0.7 and self.current_cash > opp['current_price'] * 100:
                shares = min(opp['position_size'], self.current_cash / opp['current_price'])
                cost = shares * opp['current_price']
                
                if cost > self.current_cash * 0.1:  # Don't use more than 10% cash per trade
                    cost = self.current_cash * 0.1
                    shares = cost / opp['current_price']
                
                trade = {
                    'symbol': opp['symbol'],
                    'action': 'BUY',
                    'shares': int(shares),
                    'price': opp['current_price'],
                    'cost': cost,
                    'timestamp': datetime.now(),
                    'bot': self.name
                }
                
                self.current_cash -= cost
                if opp['symbol'] not in self.portfolio:
                    self.portfolio[opp['symbol']] = 0
                self.portfolio[opp['symbol']] += shares
                self.trade_history.append(trade)
                executed_trades.append(trade)
                
        return executed_trades

class MeanReversionTrader(TradingBot):
    """Mean reversion trading bot"""
    
    def __init__(self, allocated_capital, risk_tolerance="medium"):
        super().__init__("Mean Reversion Trader", allocated_capital, risk_tolerance)
        self.reversion_threshold = 1.5  # Standard deviations from mean
        self.profit_target = 0.05  # 5% profit target
        
    def calculate_mean_reversion_score(self, symbol):
        """Calculate mean reversion score"""
        try:
            data = yf.download(symbol, period='60d', interval='1d')
            if data.empty or len(data) < 20:
                return 0, 0
                
            # Calculate Bollinger Bands
            sma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            current_price = data['Close'][-1]
            current_sma = sma_20.iloc[-1]
            current_std = std_20.iloc[-1]
            
            # Distance from mean in standard deviations
            z_score = (current_price - current_sma) / current_std if current_std > 0 else 0
            
            # RSI for overbought/oversold
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Combined mean reversion score
            if z_score < -self.reversion_threshold:  # Oversold - potential buy
                reversion_score = abs(z_score) * 10 + (30 - current_rsi) * 0.5
                action = 'BUY'
            elif z_score > self.reversion_threshold:  # Overbought - potential sell
                reversion_score = abs(z_score) * 10 + (current_rsi - 70) * 0.5
                action = 'SELL'
            else:
                reversion_score = 0
                action = 'HOLD'
                
            return reversion_score, action
            
        except:
            return 0, 'HOLD'
    
    def scan_opportunities(self, watchlist):
        """Scan for mean reversion opportunities"""
        opportunities = []
        
        for symbol in watchlist:
            reversion_score, action = self.calculate_mean_reversion_score(symbol)
            
            if reversion_score > 10:  # Minimum threshold
                data = yf.download(symbol, period='5d', interval='1d')
                if not data.empty:
                    current_price = data['Close'][-1]
                    position_size = self.calculate_position_size(current_price)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'reversion_score': reversion_score,
                        'current_price': current_price,
                        'position_size': position_size,
                        'action': action,
                        'confidence': min(reversion_score / 20, 1.0)
                    })
                    
        return sorted(opportunities, key=lambda x: x['reversion_score'], reverse=True)[:5]
    
    def execute_trades(self, opportunities):
        """Execute mean reversion trades"""
        executed_trades = []
        
        for opp in opportunities:
            if opp['confidence'] > 0.6:
                if opp['action'] == 'BUY' and self.current_cash > opp['current_price'] * 100:
                    shares = min(opp['position_size'], self.current_cash / opp['current_price'])
                    cost = shares * opp['current_price']
                    
                    trade = {
                        'symbol': opp['symbol'],
                        'action': 'BUY',
                        'shares': int(shares),
                        'price': opp['current_price'],
                        'cost': cost,
                        'timestamp': datetime.now(),
                        'bot': self.name,
                        'profit_target': opp['current_price'] * (1 + self.profit_target)
                    }
                    
                    self.current_cash -= cost
                    if opp['symbol'] not in self.portfolio:
                        self.portfolio[opp['symbol']] = 0
                    self.portfolio[opp['symbol']] += shares
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    
                elif opp['action'] == 'SELL' and opp['symbol'] in self.portfolio:
                    shares = min(self.portfolio[opp['symbol']], opp['position_size'])
                    revenue = shares * opp['current_price']
                    
                    trade = {
                        'symbol': opp['symbol'],
                        'action': 'SELL',
                        'shares': int(shares),
                        'price': opp['current_price'],
                        'revenue': revenue,
                        'timestamp': datetime.now(),
                        'bot': self.name
                    }
                    
                    self.current_cash += revenue
                    self.portfolio[opp['symbol']] -= shares
                    if self.portfolio[opp['symbol']] <= 0:
                        del self.portfolio[opp['symbol']]
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    
        return executed_trades

class ValueInvestor(TradingBot):
    """Value investing bot based on fundamental analysis"""
    
    def __init__(self, allocated_capital, risk_tolerance="low"):
        super().__init__("Value Investor", allocated_capital, risk_tolerance)
        self.pe_threshold = 15
        self.pb_threshold = 1.5
        self.debt_to_equity_threshold = 0.5
        
    def get_fundamental_data(self, symbol):
        """Get fundamental data for a symbol"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            fundamental_data = {
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margins': info.get('profitMargins', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'market_cap': info.get('marketCap', 0)
            }
            
            return fundamental_data
        except:
            return {}
    
    def calculate_value_score(self, symbol):
        """Calculate value investing score"""
        fundamental_data = self.get_fundamental_data(symbol)
        if not fundamental_data:
            return 0
            
        score = 0
        max_score = 0
        
        # P/E Ratio (lower is better)
        if fundamental_data['pe_ratio'] and fundamental_data['pe_ratio'] > 0:
            if fundamental_data['pe_ratio'] < self.pe_threshold:
                score += (self.pe_threshold - fundamental_data['pe_ratio']) / self.pe_threshold * 25
            max_score += 25
            
        # P/B Ratio (lower is better)
        if fundamental_data['pb_ratio'] and fundamental_data['pb_ratio'] > 0:
            if fundamental_data['pb_ratio'] < self.pb_threshold:
                score += (self.pb_threshold - fundamental_data['pb_ratio']) / self.pb_threshold * 25
            max_score += 25
            
        # Debt to Equity (lower is better)
        if fundamental_data['debt_to_equity']:
            if fundamental_data['debt_to_equity'] < self.debt_to_equity_threshold:
                score += (self.debt_to_equity_threshold - fundamental_data['debt_to_equity']) / self.debt_to_equity_threshold * 20
            max_score += 20
            
        # ROE (higher is better)
        if fundamental_data['roe']:
            score += min(fundamental_data['roe'] * 10, 15)  # Cap at 15
            max_score += 15
            
        # Profit Margins (higher is better)
        if fundamental_data['profit_margins']:
            score += min(fundamental_data['profit_margins'] * 100, 15)  # Cap at 15
            max_score += 15
            
        # Normalize score
        if max_score > 0:
            normalized_score = (score / max_score) * 100
        else:
            normalized_score = 0
            
        return normalized_score
    
    def scan_opportunities(self, watchlist):
        """Scan for value investment opportunities"""
        opportunities = []
        
        for symbol in watchlist:
            value_score = self.calculate_value_score(symbol)
            
            if value_score > 60:  # Only consider good value opportunities
                data = yf.download(symbol, period='5d', interval='1d')
                if not data.empty:
                    current_price = data['Close'][-1]
                    position_size = self.calculate_position_size(current_price)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'value_score': value_score,
                        'current_price': current_price,
                        'position_size': position_size,
                        'action': 'BUY',
                        'confidence': value_score / 100
                    })
                    
        return sorted(opportunities, key=lambda x: x['value_score'], reverse=True)[:3]
    
    def execute_trades(self, opportunities):
        """Execute value investment trades"""
        executed_trades = []
        
        for opp in opportunities:
            if opp['confidence'] > 0.7 and self.current_cash > opp['current_price'] * 100:
                shares = min(opp['position_size'], self.current_cash / opp['current_price'])
                cost = shares * opp['current_price']
                
                # Value investors typically take larger positions
                if cost < self.current_cash * 0.2:  # At least 20% of available cash
                    cost = self.current_cash * 0.2
                    shares = cost / opp['current_price']
                
                trade = {
                    'symbol': opp['symbol'],
                    'action': 'BUY',
                    'shares': int(shares),
                    'price': opp['current_price'],
                    'cost': cost,
                    'timestamp': datetime.now(),
                    'bot': self.name,
                    'hold_period': 'long_term'
                }
                
                self.current_cash -= cost
                if opp['symbol'] not in self.portfolio:
                    self.portfolio[opp['symbol']] = 0
                self.portfolio[opp['symbol']] += shares
                self.trade_history.append(trade)
                executed_trades.append(trade)
                
        return executed_trades

class VolatilityBreakoutTrader(TradingBot):
    """Volatility breakout trading bot"""
    
    def __init__(self, allocated_capital, risk_tolerance="high"):
        super().__init__("Volatility Breakout Trader", allocated_capital, risk_tolerance)
        self.volatility_threshold = 0.02  # 2% daily volatility
        self.breakout_threshold = 1.0  # 1 standard deviation breakout
        
    def calculate_volatility_breakout_score(self, symbol):
        """Calculate volatility breakout score"""
        try:
            data = yf.download(symbol, period='30d', interval='1d')
            if data.empty or len(data) < 20:
                return 0, 'HOLD'
                
            # Calculate Average True Range (ATR)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Current volatility
            current_volatility = atr / data['Close'].iloc[-1]
            
            # Bollinger Band width (volatility indicator)
            sma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            bb_width = (std_20 / sma_20).iloc[-1]
            
            # Price position relative to recent range
            recent_high = data['High'][-10:].max()
            recent_low = data['Low'][-10:].min()
            current_price = data['Close'][-1]
            
            range_position = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            
            # Breakout detection
            if current_volatility > self.volatility_threshold:
                if range_position > 0.8:  # Near top of range - potential breakout up
                    breakout_score = current_volatility * 1000 + (range_position - 0.8) * 500
                    action = 'BUY'
                elif range_position < 0.2:  # Near bottom of range - potential breakdown
                    breakout_score = current_volatility * 1000 + (0.2 - range_position) * 500
                    action = 'SELL'
                else:
                    breakout_score = current_volatility * 500
                    action = 'HOLD'
            else:
                breakout_score = 0
                action = 'HOLD'
                
            return breakout_score, action
            
        except:
            return 0, 'HOLD'
    
    def scan_opportunities(self, watchlist):
        """Scan for volatility breakout opportunities"""
        opportunities = []
        
        for symbol in watchlist:
            breakout_score, action = self.calculate_volatility_breakout_score(symbol)
            
            if breakout_score > 10:  # Minimum threshold
                data = yf.download(symbol, period='5d', interval='1d')
                if not data.empty:
                    current_price = data['Close'][-1]
                    position_size = self.calculate_position_size(current_price)
                    
                    opportunities.append({
                        'symbol': symbol,
                        'breakout_score': breakout_score,
                        'current_price': current_price,
                        'position_size': position_size,
                        'action': action,
                        'confidence': min(breakout_score / 50, 1.0)
                    })
                    
        return sorted(opportunities, key=lambda x: x['breakout_score'], reverse=True)[:5]
    
    def execute_trades(self, opportunities):
        """Execute volatility breakout trades"""
        executed_trades = []
        
        for opp in opportunities:
            if opp['confidence'] > 0.7:
                if opp['action'] == 'BUY' and self.current_cash > opp['current_price'] * 100:
                    shares = min(opp['position_size'], self.current_cash / opp['current_price'])
                    cost = shares * opp['current_price']
                    
                    trade = {
                        'symbol': opp['symbol'],
                        'action': 'BUY',
                        'shares': int(shares),
                        'price': opp['current_price'],
                        'cost': cost,
                        'timestamp': datetime.now(),
                        'bot': self.name,
                        'stop_loss': opp['current_price'] * 0.95,  # 5% stop loss
                        'take_profit': opp['current_price'] * 1.08  # 8% profit target
                    }
                    
                    self.current_cash -= cost
                    if opp['symbol'] not in self.portfolio:
                        self.portfolio[opp['symbol']] = 0
                    self.portfolio[opp['symbol']] += shares
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    
                elif opp['action'] == 'SELL' and opp['symbol'] in self.portfolio:
                    shares = min(self.portfolio[opp['symbol']], opp['position_size'])
                    revenue = shares * opp['current_price']
                    
                    trade = {
                        'symbol': opp['symbol'],
                        'action': 'SELL',
                        'shares': int(shares),
                        'price': opp['current_price'],
                        'revenue': revenue,
                        'timestamp': datetime.now(),
                        'bot': self.name
                    }
                    
                    self.current_cash += revenue
                    self.portfolio[opp['symbol']] -= shares
                    if self.portfolio[opp['symbol']] <= 0:
                        del self.portfolio[opp['symbol']]
                    self.trade_history.append(trade)
                    executed_trades.append(trade)
                    
        return executed_trades

class NewsAnalyzer:
    """News analysis for trading bots"""
    
    def __init__(self):
        self.sentiment_cache = {}
        
    def analyze_news_sentiment(self, symbol):
        """Analyze news sentiment for a symbol (simulated)"""
        # In a real implementation, you would use news API like NewsAPI, Alpha Vantage, etc.
        # This is a simulated version for demonstration
        
        # Simulate news sentiment based on price movement and random factors
        try:
            data = yf.download(symbol, period='5d', interval='1d')
            if data.empty:
                return 0
                
            price_change = (data['Close'][-1] / data['Close'][0] - 1) * 100
            
            # Simulate sentiment based on price movement with some randomness
            base_sentiment = np.tanh(price_change / 10)  # Normalize to -1 to 1
            random_factor = random.uniform(-0.2, 0.2)
            sentiment = base_sentiment + random_factor
            
            return max(min(sentiment, 1), -1)  # Clamp between -1 and 1
            
        except:
            return 0

class AlgorithmicTradingApp:
    """Main algorithmic trading application with multiple AI bots"""
    
    def __init__(self, total_capital=100000):
        self.total_capital = total_capital
        self.allocated_capital = {}
        self.trading_bots = {}
        self.news_analyzer = NewsAnalyzer()
        self.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
        self.performance_history = []
        
    def allocate_capital_to_bots(self, allocations):
        """
        Allocate capital to different trading bots
        
        Args:
            allocations (dict): Dictionary with bot names as keys and allocation percentages as values
        """
        self.allocated_capital = allocations
        
        # Initialize trading bots
        if 'momentum' in allocations:
            self.trading_bots['momentum'] = MomentumTrader(
                self.total_capital * allocations['momentum'] / 100
            )
            
        if 'mean_reversion' in allocations:
            self.trading_bots['mean_reversion'] = MeanReversionTrader(
                self.total_capital * allocations['mean_reversion'] / 100
            )
            
        if 'value' in allocations:
            self.trading_bots['value'] = ValueInvestor(
                self.total_capital * allocations['value'] / 100
            )
            
        if 'volatility' in allocations:
            self.trading_bots['volatility'] = VolatilityBreakoutTrader(
                self.total_capital * allocations['volatility'] / 100
            )
        
        print("✅ Capital allocated to trading bots:")
        for bot_name, allocation in allocations.items():
            print(f"   {bot_name}: ${self.total_capital * allocation / 100:,.2f} ({allocation}%)")
    
    def run_trading_cycle(self):
        """Run one complete trading cycle for all bots"""
        print(f"\n🔄 Running trading cycle at {datetime.now()}")
        all_trades = []
        
        for bot_name, bot in self.trading_bots.items():
            print(f"\n🤖 {bot_name.upper()} Bot Scanning...")
            
            # Scan for opportunities
            opportunities = bot.scan_opportunities(self.watchlist)
            
            if opportunities:
                print(f"   Found {len(opportunities)} opportunities:")
                for opp in opportunities[:3]:  # Show top 3
                    print(f"   - {opp['symbol']}: {opp['action']} (Confidence: {opp['confidence']:.2f})")
                
                # Execute trades
                executed_trades = bot.execute_trades(opportunities)
                all_trades.extend(executed_trades)
                
                if executed_trades:
                    print(f"   Executed {len(executed_trades)} trades")
                else:
                    print("   No trades executed")
            else:
                print("   No opportunities found")
        
        # Update performance history
        self.update_performance_metrics()
        
        return all_trades
    
    def update_performance_metrics(self):
        """Update performance metrics for all bots"""
        total_portfolio_value = 0
        performance_data = {'timestamp': datetime.now()}
        
        for bot_name, bot in self.trading_bots.items():
            bot_value = bot.current_cash
            for symbol, shares in bot.portfolio.items():
                try:
                    current_price = yf.download(symbol, period='1d', interval='1d')['Close'][-1]
                    bot_value += shares * current_price
                except:
                    pass
            
            performance_data[bot_name] = bot_value
            performance_data[f'{bot_name}_return'] = (
                (bot_value / (self.total_capital * self.allocated_capital[bot_name] / 100) - 1) * 100
            )
            total_portfolio_value += bot_value
        
        performance_data['total_value'] = total_portfolio_value
        performance_data['total_return'] = (total_portfolio_value / self.total_capital - 1) * 100
        
        self.performance_history.append(performance_data)
    
    def get_portfolio_summary(self):
        """Get complete portfolio summary"""
        summary = {
            'total_capital': self.total_capital,
            'current_value': 0,
            'total_return': 0,
            'bot_performance': {},
            'current_holdings': {}
        }
        
        for bot_name, bot in self.trading_bots.items():
            bot_value = bot.current_cash
            bot_holdings = {}
            
            for symbol, shares in bot.portfolio.items():
                try:
                    current_price = yf.download(symbol, period='1d', interval='1d')['Close'][-1]
                    position_value = shares * current_price
                    bot_value += position_value
                    bot_holdings[symbol] = {
                        'shares': shares,
                        'current_price': current_price,
                        'position_value': position_value
                    }
                except:
                    pass
            
            bot_return = (bot_value / (self.total_capital * self.allocated_capital[bot_name] / 100) - 1) * 100
            
            summary['bot_performance'][bot_name] = {
                'allocated_capital': self.total_capital * self.allocated_capital[bot_name] / 100,
                'current_value': bot_value,
                'return': bot_return,
                'holdings': bot_holdings
            }
            
            summary['current_value'] += bot_value
            
            # Merge holdings
            for symbol, holding in bot_holdings.items():
                if symbol not in summary['current_holdings']:
                    summary['current_holdings'][symbol] = 0
                summary['current_holdings'][symbol] += holding['shares']
        
        summary['total_return'] = (summary['current_value'] / self.total_capital - 1) * 100
        
        return summary
    
    def plot_performance(self):
        """Plot performance of all trading bots"""
        if not self.performance_history:
            print("No performance data available yet")
            return
            
        df = pd.DataFrame(self.performance_history)
        df.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(15, 10))
        
        # Plot individual bot performance
        plt.subplot(2, 1, 1)
        for bot_name in self.trading_bots.keys():
            if f'{bot_name}_return' in df.columns:
                plt.plot(df.index, df[f'{bot_name}_return'], label=f'{bot_name.title()} Bot', linewidth=2)
        
        plt.title('Trading Bots Performance Over Time')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot total portfolio performance
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['total_return'], label='Total Portfolio', linewidth=3, color='black')
        plt.xlabel('Time')
        plt.ylabel('Total Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_continuous_trading(self, cycles=5, interval_hours=24):
        """Run continuous trading simulation"""
        print(f"🚀 Starting continuous trading simulation for {cycles} cycles")
        print(f"   Interval: {interval_hours} hours between cycles")
        
        for cycle in range(cycles):
            print(f"\n{'='*50}")
            print(f"CYCLE {cycle + 1}/{cycles}")
            print(f"{'='*50}")
            
            # Run trading cycle
            trades = self.run_trading_cycle()
            
            # Show portfolio summary
            summary = self.get_portfolio_summary()
            print(f"\n📊 PORTFOLIO SUMMARY (Cycle {cycle + 1}):")
            print(f"   Total Value: ${summary['current_value']:,.2f}")
            print(f"   Total Return: {summary['total_return']:.2f}%")
            
            # Show bot performance
            for bot_name, performance in summary['bot_performance'].items():
                print(f"   {bot_name.title()} Bot: {performance['return']:.2f}%")
            
            # Wait for next cycle (in real implementation, this would be actual time)
            if cycle < cycles - 1:
                print(f"\n⏰ Waiting {interval_hours} hours for next cycle...")
                # time.sleep(interval_hours * 3600)  # Uncomment for real-time waiting

def main():
    """Main function to demonstrate the algorithmic trading application with AI bots"""
    
    # Initialize the trading app with $100,000 capital
    app = AlgorithmicTradingApp(total_capital=100000)
    
    # Allocate capital to different trading bots
    allocations = {
        'momentum': 30,        # 30% to Momentum Trader
        'mean_reversion': 25,  # 25% to Mean Reversion Trader
        'value': 25,           # 25% to Value Investor
        'volatility': 20       # 20% to Volatility Breakout Trader
    }
    
    app.allocate_capital_to_bots(allocations)
    
    # Run continuous trading simulation
    app.run_continuous_trading(cycles=3, interval_hours=24)
    
    # Show final portfolio summary
    print(f"\n{'='*60}")
    print("🎯 FINAL PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    
    final_summary = app.get_portfolio_summary()
    
    print(f"💰 Total Capital: ${final_summary['total_capital']:,.2f}")
    print(f"📈 Current Value: ${final_summary['current_value']:,.2f}")
    print(f"🎯 Total Return: {final_summary['total_return']:.2f}%")
    
    print(f"\n🤖 BOT PERFORMANCE:")
    for bot_name, performance in final_summary['bot_performance'].items():
        print(f"   {bot_name.title():<20} | "
              f"Return: {performance['return']:>7.2f}% | "
              f"Value: ${performance['current_value']:>10,.2f}")
    
    print(f"\n📦 CURRENT HOLDINGS:")
    for symbol, shares in final_summary['current_holdings'].items():
        if shares > 0:
            print(f"   {symbol}: {shares} shares")
    
    # Plot performance
    app.plot_performance()

if __name__ == "__main__":
    main()
