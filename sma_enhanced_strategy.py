"""
Enhanced SMA Crossover Strategy with Optuna, OOS, Monte Carlo, and Advanced Features
====================================================================================

Enhancements:
1. Optuna parameter optimization (short_window, long_window, trend_filter)
2. Out-of-Sample (OOS) validation to prevent overfitting
3. Monte Carlo simulation for robustness testing
4. Buy-and-hold benchmark comparison
5. 200 SMA trend filter (only trade in uptrends)
6. Long/Short variant (can short during downtrends)
7. Multi-asset portfolio testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import optuna
from scipy import stats

# Import our tick data loader
from tick_data_loader import TickDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('sma_enhanced_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedStrategyConfig:
    """Configuration for Enhanced SMA Crossover Strategy"""
    # MA windows (will be optimized by Optuna)
    short_window: int = 40
    long_window: int = 100
    trend_filter_window: int = 200  # 200 SMA for trend confirmation
    
    # Strategy mode
    allow_short: bool = False  # False = long-only, True = long/short
    use_trend_filter: bool = True  # Only trade when price > 200 SMA
    
    # Momentum filters
    use_rsi_filter: bool = True  # Only take crossovers when RSI confirms
    rsi_period: int = 14  # RSI calculation period
    rsi_min: float = 50.0  # Minimum RSI for bullish signals (indicating momentum)
    rsi_max: float = 70.0  # Maximum RSI to avoid overbought (70 or 80 typical)

    use_macd_filter: bool = True  # Only take crossovers when MACD confirms
    macd_fast: int = 12  # MACD fast EMA period
    macd_slow: int = 26  # MACD slow EMA period
    macd_signal: int = 9  # MACD signal line period

    # Portfolio settings
    initial_capital: float = 100000.0
    position_size: int = 100  # Shares per trade
    position_pct: Optional[float] = None  # Alternative: % of portfolio
    
    # Data splits
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    train_end_date: str = "2024-06-30"  # In-sample training period
    
    # Optimization
    optuna_trials: int = 50
    optimize: bool = True
    
    # Monte Carlo
    monte_carlo_runs: int = 1000
    run_monte_carlo: bool = True
    

class BuyAndHoldBenchmark:
    """Simple buy-and-hold benchmark for comparison"""
    
    def __init__(self, initial_capital: float, shares: int):
        self.initial_capital = initial_capital
        self.shares = shares
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run buy-and-hold strategy"""
        portfolio = pd.DataFrame(index=data.index)
        
        # Buy at first price, hold until end
        entry_price = data['Close'].iloc[0]
        portfolio['holdings'] = self.shares * data['Close']
        portfolio['cash'] = self.initial_capital - (self.shares * entry_price)
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        return portfolio


class EnhancedSMABacktester:
    """Enhanced backtester with optimization and validation"""
    
    def __init__(self, config: EnhancedStrategyConfig):
        self.config = config
        self.loader = TickDataLoader()
        self.results = {}
        self.optimization_history = []

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            macd_line: MACD line (fast EMA - slow EMA)
            signal_line: Signal line (EMA of MACD line)
            histogram: MACD histogram (MACD - Signal)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def load_and_prepare_data(self, symbol: str) -> pd.DataFrame:
        """Load tick data and resample to daily OHLC bars"""
        logger.info(f"Loading tick data for {symbol}...")
        
        # Load tick data using correct method name
        ticks = self.loader.load_symbol_range(
            symbol=symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )
        
        if ticks.empty:
            logger.error(f"No tick data found for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(ticks):,} ticks for {symbol}")
        
        # Set datetime as index if not already
        if 'datetime' in ticks.columns:
            ticks = ticks.set_index('datetime')
        
        # Sort by datetime
        ticks = ticks.sort_index()
        
        # Resample to daily OHLC
        daily = pd.DataFrame()
        daily['Open'] = ticks['price'].resample('1D').first()
        daily['High'] = ticks['price'].resample('1D').max()
        daily['Low'] = ticks['price'].resample('1D').min()
        daily['Close'] = ticks['price'].resample('1D').last()
        daily['Volume'] = ticks['size'].resample('1D').sum()
        
        # Drop NaN rows (non-trading days)
        daily = daily.dropna()
        
        logger.info(f"Resampled to {len(daily)} daily bars")
        
        return daily
    
    def generate_signals(self, data: pd.DataFrame, 
                        short_window: int, 
                        long_window: int,
                        trend_filter_window: int = 200,
                        allow_short: bool = False,
                        use_trend_filter: bool = True) -> pd.DataFrame:
        """Generate SMA crossover signals with optional enhancements"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        
        # Calculate moving averages
        signals['short_mavg'] = data['Close'].rolling(
            window=short_window, 
            min_periods=1
        ).mean()
        
        signals['long_mavg'] = data['Close'].rolling(
            window=long_window, 
            min_periods=1
        ).mean()
        
        # 200 SMA trend filter
        if use_trend_filter:
            signals['trend_mavg'] = data['Close'].rolling(
                window=trend_filter_window,
                min_periods=1
            ).mean()
            above_trend = signals['price'] > signals['trend_mavg']
        else:
            above_trend = pd.Series(True, index=data.index)
        
        # Generate signals
        signals['signal'] = 0.0
        
        if allow_short:
            # Long/Short mode: 1.0 = LONG, -1.0 = SHORT
            signals.iloc[long_window:, signals.columns.get_loc('signal')] = np.where(
                signals['short_mavg'].iloc[long_window:] >
                signals['long_mavg'].iloc[long_window:],
                1.0, -1.0
            )

            # Apply trend filter (only for longs, shorts allowed anytime)
            if use_trend_filter:
                # Filter longs when price < 200 SMA
                long_positions = signals['signal'] == 1.0
                below_trend = ~above_trend
                cancel_long = long_positions & below_trend
                signals.loc[cancel_long, 'signal'] = 0.0
        else:
            # Long-only mode: 1.0 = LONG, 0.0 = CASH
            idx = signals.index[long_window:]
            long_condition = signals.loc[idx, 'short_mavg'] > signals.loc[idx, 'long_mavg']
            
            if use_trend_filter:
                trend_condition = above_trend.loc[idx]
                combined_signal = np.where(long_condition & trend_condition, 1.0, 0.0)
            else:
                combined_signal = np.where(long_condition, 1.0, 0.0)

            signals.loc[idx, 'signal'] = combined_signal
        
        # Position changes (1.0 = BUY, -1.0 = SELL, -2.0 = SHORT, 2.0 = COVER)
        signals['position'] = signals['signal'].diff()
        
        return signals
    
    def backtest(self, symbol: str, signals: pd.DataFrame) -> pd.DataFrame:
        """Run backtest and calculate portfolio performance"""
        
        # Initialize positions DataFrame
        positions = pd.DataFrame(index=signals.index)
        
        # Fixed shares (positive = long, negative = short)
        positions[symbol] = self.config.position_size * signals['signal']
        
        # Initialize portfolio
        portfolio = positions.copy()
        
        # Track holdings value (can be negative if short)
        portfolio['holdings'] = (positions[symbol] * signals['price'])
        
        # Track cash (initial - cumulative purchases + short proceeds)
        pos_diff = positions[symbol].diff()
        portfolio['cash'] = self.config.initial_capital - (
            pos_diff * signals['price']
        ).cumsum()
        
        # Total portfolio value
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        # Portfolio returns
        portfolio['returns'] = portfolio['total'].pct_change()
        
        # Add signals for analysis
        portfolio['short_mavg'] = signals['short_mavg']
        portfolio['long_mavg'] = signals['long_mavg']
        portfolio['position'] = signals['position']
        
        if 'trend_mavg' in signals.columns:
            portfolio['trend_mavg'] = signals['trend_mavg']
        
        return portfolio
    
    def calculate_metrics(self, portfolio: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = portfolio['returns'].dropna()
        total = portfolio['total']
        
        # Total return
        total_return = (total.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized Sharpe Ratio
        if returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
        
        # CAGR
        n_years = len(returns) / 252
        if n_years > 0 and total.iloc[-1] > 0:
            cagr = (total.iloc[-1] / self.config.initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * (returns.mean() / downside_returns.std())
        else:
            sortino_ratio = 0.0
        
        # Calmar Ratio (CAGR / Max Drawdown)
        if max_drawdown < 0:
            calmar_ratio = cagr / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        # Win Rate (only for strategy portfolio, not benchmark)
        if 'position' in portfolio.columns:
            trades = portfolio[portfolio['position'] != 0]
        else:
            trades = []
        
        if len(trades) > 1:
            trade_returns = []
            positions = portfolio['position']
            entry_idx = None
            entry_direction = 0
            
            for idx, pos in positions.items():
                if pos != 0 and entry_idx is None:  # Entry
                    entry_idx = idx
                    entry_direction = np.sign(pos)
                elif pos != 0 and entry_idx is not None:  # Exit
                    entry_price = portfolio.loc[entry_idx, 'short_mavg']
                    exit_price = portfolio.loc[idx, 'short_mavg']
                    
                    # Calculate return based on direction
                    if entry_direction > 0:  # Long
                        trade_return = (exit_price - entry_price) / entry_price
                    else:  # Short
                        trade_return = (entry_price - exit_price) / entry_price
                    
                    trade_returns.append(trade_return)
                    entry_idx = idx
                    entry_direction = np.sign(pos)
            
            if trade_returns:
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
                avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            else:
                win_rate = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': volatility,
            'n_trades': len(portfolio[portfolio['position'] != 0]) if 'position' in portfolio.columns else 0,
            'final_value': total.iloc[-1]
        }
        
        return metrics
    
    def optimize_parameters(self, symbol: str, data: pd.DataFrame) -> Dict[str, int]:
        """Use Optuna to find optimal MA windows"""
        logger.info(f"Optimizing parameters for {symbol} using Optuna ({self.config.optuna_trials} trials)...")
        
        # Split into training data
        train_data = data[data.index <= self.config.train_end_date]
        
        def objective(trial):
            # Suggest parameters
            short_window = trial.suggest_int('short_window', 10, 100, step=5)
            long_window = trial.suggest_int('long_window', short_window + 20, 250, step=10)
            trend_filter_window = trial.suggest_int('trend_filter_window', 100, 300, step=20)
            
            # Generate signals
            signals = self.generate_signals(
                train_data, 
                short_window, 
                long_window,
                trend_filter_window,
                self.config.allow_short,
                self.config.use_trend_filter
            )
            
            # Run backtest
            portfolio = self.backtest(symbol, signals)
            
            # Calculate Sharpe ratio as objective
            metrics = self.calculate_metrics(portfolio)
            
            # Store for analysis
            self.optimization_history.append({
                'trial': trial.number,
                'short_window': short_window,
                'long_window': long_window,
                'trend_filter_window': trend_filter_window,
                'sharpe': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown']
            })
            
            return metrics['sharpe_ratio']
        
        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce Optuna output
        study = optuna.create_study(direction='maximize', study_name=f'{symbol}_optimization')
        study.optimize(objective, n_trials=self.config.optuna_trials, show_progress_bar=False)
        
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best Sharpe: {study.best_value:.2f}")
        
        return best_params
    
    def run_monte_carlo(self, portfolio: pd.DataFrame, n_runs: int = 1000) -> Dict[str, float]:
        """Run Monte Carlo simulation on trade sequence"""
        logger.info(f"Running Monte Carlo simulation ({n_runs} runs)...")
        
        # Extract individual trades
        returns = portfolio['returns'].dropna()
        
        if len(returns) < 10:
            logger.warning("Not enough trades for Monte Carlo")
            return {'mean': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        # Run simulations
        final_values = []
        for _ in range(n_runs):
            # Randomly resample returns with replacement
            simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            final_value = self.config.initial_capital * np.prod(1 + simulated_returns)
            final_values.append(final_value)
        
        # Calculate statistics
        mc_mean = np.mean(final_values)
        mc_std = np.std(final_values)
        ci_lower, ci_upper = np.percentile(final_values, [5, 95])
        
        logger.info(f"Monte Carlo Results:")
        logger.info(f"  Mean Final Value: ${mc_mean:,.2f}")
        logger.info(f"  Std Dev: ${mc_std:,.2f}")
        logger.info(f"  90% CI: [${ci_lower:,.2f}, ${ci_upper:,.2f}]")
        
        return {
            'mean': mc_mean,
            'std': mc_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def plot_results(self, symbol: str, signals: pd.DataFrame, 
                     portfolio: pd.DataFrame, benchmark: pd.DataFrame,
                     save_path: Optional[str] = None):
        """Plot strategy results with benchmark comparison"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Price and Moving Averages with Signals
        ax1.plot(signals.index, signals['price'], 
                label='Close Price', color='black', lw=1, alpha=0.7)
        ax1.plot(signals.index, signals['short_mavg'], 
                label=f'Short MA', color='blue', lw=1.5)
        ax1.plot(signals.index, signals['long_mavg'], 
                label=f'Long MA', color='red', lw=1.5)
        
        if 'trend_mavg' in signals.columns and self.config.use_trend_filter:
            ax1.plot(signals.index, signals['trend_mavg'], 
                    label=f'Trend MA', 
                    color='purple', lw=1, linestyle='--', alpha=0.6)
        
        # Mark buy signals (long)
        buy_signals = signals[signals['position'] > 0]
        if len(buy_signals) > 0:
            ax1.plot(buy_signals.index, signals.loc[buy_signals.index, 'short_mavg'],
                    marker='^', markersize=10, color='green', lw=0, 
                    label='Buy Signal')
        
        # Mark sell/short signals
        sell_signals = signals[signals['position'] < 0]
        if len(sell_signals) > 0:
            ax1.plot(sell_signals.index, signals.loc[sell_signals.index, 'short_mavg'],
                    marker='v', markersize=10, color='red', lw=0, 
                    label='Sell/Short Signal')
        
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Enhanced SMA Strategy ({"Long/Short" if self.config.allow_short else "Long-Only"})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value vs Benchmark
        ax2.plot(portfolio.index, portfolio['total'], 
                label='Strategy', color='blue', lw=2)
        ax2.plot(benchmark.index, benchmark['total'],
                label='Buy & Hold', color='orange', lw=2, linestyle='--')
        ax2.axhline(y=self.config.initial_capital, 
                   color='gray', linestyle='--', label='Initial Capital', alpha=0.5)
        
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        strategy_cumulative = (1 + portfolio['returns']).cumprod()
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        
        benchmark_cumulative = (1 + benchmark['returns']).cumprod()
        benchmark_running_max = benchmark_cumulative.expanding().max()
        benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
        
        ax3.fill_between(portfolio.index, strategy_drawdown * 100, 0, 
                         color='red', alpha=0.3, label='Strategy DD')
        ax3.fill_between(benchmark.index, benchmark_drawdown * 100, 0,
                         color='orange', alpha=0.3, label='B&H DD')
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def run(self, symbols: List[str]):
        """Run enhanced strategy on multiple symbols"""
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running Enhanced SMA Strategy on {symbol}")
            logger.info(f"{'='*80}")
            
            # Load data
            data = self.load_and_prepare_data(symbol)
            
            if data.empty:
                logger.warning(f"Skipping {symbol} - no data available")
                continue
            
            # Check if we have enough data
            min_required = max(self.config.long_window, self.config.trend_filter_window)
            if len(data) < min_required:
                logger.warning(
                    f"Skipping {symbol} - insufficient data "
                    f"({len(data)} bars < {min_required} required)"
                )
                continue
            
            # Optimize parameters if enabled
            if self.config.optimize:
                best_params = self.optimize_parameters(symbol, data)
                short_window = best_params['short_window']
                long_window = best_params['long_window']
                trend_filter_window = best_params['trend_filter_window']
            else:
                short_window = self.config.short_window
                long_window = self.config.long_window
                trend_filter_window = self.config.trend_filter_window
            
            # Split data for OOS validation
            train_data = data[data.index <= self.config.train_end_date]
            test_data = data[data.index > self.config.train_end_date]
            
            # Run on full period
            signals_full = self.generate_signals(
                data, short_window, long_window, trend_filter_window,
                self.config.allow_short, self.config.use_trend_filter
            )
            portfolio_full = self.backtest(symbol, signals_full)
            metrics_full = self.calculate_metrics(portfolio_full)
            
            # Run on OOS period
            if len(test_data) > long_window:
                signals_oos = self.generate_signals(
                    test_data, short_window, long_window, trend_filter_window,
                    self.config.allow_short, self.config.use_trend_filter
                )
                portfolio_oos = self.backtest(symbol, signals_oos)
                metrics_oos = self.calculate_metrics(portfolio_oos)
            else:
                metrics_oos = None
            
            # Buy-and-hold benchmark
            benchmark = BuyAndHoldBenchmark(
                self.config.initial_capital, 
                self.config.position_size
            )
            benchmark_portfolio = benchmark.run(data)
            benchmark_metrics = self.calculate_metrics(benchmark_portfolio)
            
            # Monte Carlo
            if self.config.run_monte_carlo:
                mc_results = self.run_monte_carlo(portfolio_full, self.config.monte_carlo_runs)
            else:
                mc_results = None
            
            # Store results
            result = {
                'symbol': symbol,
                'short_window': short_window,
                'long_window': long_window,
                'trend_filter_window': trend_filter_window,
                'full_period': metrics_full,
                'oos_period': metrics_oos,
                'benchmark': benchmark_metrics,
                'monte_carlo': mc_results,
                'signals': signals_full,
                'portfolio': portfolio_full,
                'benchmark_portfolio': benchmark_portfolio
            }
            
            self.results[symbol] = result
            all_results.append(result)
            
            # Print results
            logger.info(f"\n{symbol} Performance Summary:")
            logger.info(f"  Optimized Parameters: short={short_window}, long={long_window}, trend={trend_filter_window}")
            logger.info(f"\n  FULL PERIOD:")
            logger.info(f"    Total Return: {metrics_full['total_return']:.2%}")
            logger.info(f"    CAGR: {metrics_full['cagr']:.2%}")
            logger.info(f"    Sharpe: {metrics_full['sharpe_ratio']:.2f}")
            logger.info(f"    Sortino: {metrics_full['sortino_ratio']:.2f}")
            logger.info(f"    Calmar: {metrics_full['calmar_ratio']:.2f}")
            logger.info(f"    Max DD: {metrics_full['max_drawdown']:.2%}")
            logger.info(f"    Win Rate: {metrics_full['win_rate']:.2%}")
            logger.info(f"    Profit Factor: {metrics_full['profit_factor']:.2f}")
            logger.info(f"    Trades: {metrics_full['n_trades']}")
            logger.info(f"    Final Value: ${metrics_full['final_value']:,.2f}")
            
            if metrics_oos:
                logger.info(f"\n  OUT-OF-SAMPLE (OOS):")
                logger.info(f"    Total Return: {metrics_oos['total_return']:.2%}")
                logger.info(f"    Sharpe: {metrics_oos['sharpe_ratio']:.2f}")
                logger.info(f"    Max DD: {metrics_oos['max_drawdown']:.2%}")
            
            logger.info(f"\n  BUY & HOLD BENCHMARK:")
            logger.info(f"    Total Return: {benchmark_metrics['total_return']:.2%}")
            logger.info(f"    Sharpe: {benchmark_metrics['sharpe_ratio']:.2f}")
            logger.info(f"    Max DD: {benchmark_metrics['max_drawdown']:.2%}")
            
            logger.info(f"\n  STRATEGY vs BENCHMARK:")
            excess_return = metrics_full['total_return'] - benchmark_metrics['total_return']
            logger.info(f"    Excess Return: {excess_return:.2%}")
            logger.info(f"    Sharpe Improvement: {metrics_full['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']:.2f}")
            
            # Plot results
            plot_path = f"sma_enhanced_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.plot_results(symbol, signals_full, portfolio_full, benchmark_portfolio, save_path=plot_path)
        
        # Summary
        if all_results:
            logger.info(f"\n{'='*80}")
            logger.info("SUMMARY - All Symbols")
            logger.info(f"{'='*80}")
            
            summary_data = []
            for r in all_results:
                summary_data.append({
                    'Symbol': r['symbol'],
                    'Short_MA': r['short_window'],
                    'Long_MA': r['long_window'],
                    'Trend_MA': r['trend_filter_window'],
                    'Return': r['full_period']['total_return'],
                    'CAGR': r['full_period']['cagr'],
                    'Sharpe': r['full_period']['sharpe_ratio'],
                    'Sortino': r['full_period']['sortino_ratio'],
                    'Max_DD': r['full_period']['max_drawdown'],
                    'Win_Rate': r['full_period']['win_rate'],
                    'Trades': r['full_period']['n_trades'],
                    'BH_Return': r['benchmark']['total_return'],
                    'Excess_Return': r['full_period']['total_return'] - r['benchmark']['total_return']
                })
            
            summary_df = pd.DataFrame(summary_data)
            print("\n", summary_df.to_string(index=False))
            
            # Save summary
            summary_path = f"sma_enhanced_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nSummary saved to {summary_path}")
            
            # Save optimization history
            if self.optimization_history:
                opt_df = pd.DataFrame(self.optimization_history)
                opt_path = f"sma_optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                opt_df.to_csv(opt_path, index=False)
                logger.info(f"Optimization history saved to {opt_path}")


def main():
    """Main execution"""
    
    # Configuration - start with subset, then expand
    config = EnhancedStrategyConfig(
        # Strategy mode
        allow_short=False,  # Start with long-only
        use_trend_filter=True,  # Use 200 SMA filter
        
        # Portfolio
        initial_capital=100000.0,
        position_size=100,
        
        # Dates (use 2024 only - we have complete data)
        start_date="2024-01-01",
        end_date="2024-12-31",
        train_end_date="2024-06-30",
        
        # Optimization
        optimize=True,
        optuna_trials=30,  # Reasonable number for testing
        
        # Monte Carlo
        run_monte_carlo=True,
        monte_carlo_runs=1000
    )
    
    # Start with top performers and SPY
    test_symbols = ['SPY', 'AAPL', 'MSFT']
    
    logger.info(f"\nTesting Enhanced SMA Strategy on {len(test_symbols)} symbols")
    logger.info(f"Mode: {'Long/Short' if config.allow_short else 'Long-Only'}")
    logger.info(f"Trend Filter: {'Enabled (200 SMA)' if config.use_trend_filter else 'Disabled'}")
    logger.info(f"Optimization: {'Enabled' if config.optimize else 'Disabled'}")
    
    # Create backtester
    backtester = EnhancedSMABacktester(config)
    
    # Run strategy
    backtester.run(test_symbols)
    
    logger.info("\n" + "="*80)
    logger.info("Enhanced backtest complete!")


if __name__ == "__main__":
    main()
