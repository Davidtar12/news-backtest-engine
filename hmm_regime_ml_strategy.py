"""
HMM Regime Detection + ML Signal Generation Strategy
Professional-grade implementation with Optuna, Walk-Forward OOS, and Monte Carlo validation

Architecture:
1. HMM (3 states) identifies market regime: Trending-Down, Ranging, Trending-Up
2. Specialist Random Forest models trained per regime generate trading signals
3. Confidence filtering reduces noise (only trade high-probability signals)
4. ATR-based position sizing and stops for risk management

Data Requirements:
- Tick data cache (from existing .tick_cache/)
- News cache (from .news_cache/) - optional for regime shifts
- VIX data (from tick_data_cache/yfinance_vix/) for regime filtering
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# ML/HMM imports
try:
    from hmmlearn import hmm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except ImportError:
    print("ERROR: Missing dependencies. Install: pip install hmmlearn scikit-learn")
    sys.exit(1)

# Optuna for optimization
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("WARNING: Optuna not installed. Optimization features disabled.")
    print("Install with: pip install optuna")
    OPTUNA_AVAILABLE = False
else:
    OPTUNA_AVAILABLE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('hmm_ml_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HMMRegimeMLStrategy:
    """
    Professional HMM+ML trading strategy with regime detection.
    
    Parameters (for Optuna optimization):
    - bar_size_minutes: Consolidation period (2-10 minutes)
    - hmm_n_components: Number of HMM states (2-4, default 3)
    - roc_window: Lookback for ROC calculation (50-200 bars)
    - sma_period: Fast SMA for momentum (5-30)
    - atr_period: ATR for volatility (10-30)
    - confidence_threshold: ML signal filter (0.51-0.65)
    - rf_n_estimators: Random Forest trees (50-200)
    - position_size_pct: Capital per trade (1-5%)
    - atr_stop_multiplier: Stop distance in ATRs (1.0-3.0)
    """
    
    def __init__(self, params: Dict = None):
        """Initialize strategy with parameters."""
        # RESEARCH-BASED PARAMS (QuantConnect Sharpe 1.9 + QuantInsti 148%)
        self.params = {
            'bar_size_minutes': 1440,      # DAILY bars (1440 min = 24hrs) for better regime detection
            'hmm_n_components': 3,         # CRITICAL: 3 states not 2 (QC research)
            'roc_window': 100,             # Reduced for daily data (252 bars/year, need 200 min)
            'sma_period': 20,              # 20-day SMA (monthly)
            'atr_period': 14,              # Industry standard
            'confidence_threshold': 0.53,  # QuantInsti uses 0.53 minimum (was 0.52)
            'rf_n_estimators': 50,         # Faster training (QC uses 50)
            'position_size_pct': 10.0,     # QC uses 10% equal weight (was 3%)
            'atr_stop_multiplier': 2.5,    # Reasonable stop distance
            'take_profit_multiplier': 6.0, # 6x risk:reward
            'retrain_frequency_days': 90,  # CRITICAL: Quarterly for daily data (was 30 for intraday)
            'min_regime_samples': 30,      # More data per regime (was 10)
            'vix_threshold': 40.0,         # Disabled effectively
        }
        
        if params:
            self.params.update(params)
        
        # State
        self.capital = 100_000  # Starting capital
        self.initial_capital = self.capital
        self.positions = {}  # {ticker: position_info}
        self.closed_trades = []
        self.regime_history = []  # Track regime changes
        
        # Models (will be dict per ticker)
        self.hmm_models = {}  # {ticker: HMM}
        self.rf_models = {}   # {ticker: {regime: RandomForest}}
        self.last_retrain = {}  # {ticker: datetime}
        
        # Data cache
        self.price_history = {}  # {ticker: DataFrame}
        self.vix_data = None  # VIX for regime filter
        
        logger.info(f"Initialized HMMRegimeMLStrategy with params: {self.params}")
    
    def load_vix_data(self, start_date: datetime, end_date: datetime) -> None:
        """Load VIX data for filtering high-volatility regimes."""
        vix_dir = Path('tick_data_cache/yfinance_vix')
        if not vix_dir.exists():
            logger.warning(f"VIX data not found at {vix_dir}. Running without VIX filter.")
            self.vix_data = None
            return
        
        try:
            # Load all VIX parquet files in date range
            vix_chunks = []
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                vix_file = vix_dir / f"VIX_{current_date.strftime('%Y-%m-%d')}.parquet"
                if vix_file.exists():
                    df = pd.read_parquet(vix_file)
                    vix_chunks.append(df)
                current_date += timedelta(days=1)
            
            if not vix_chunks:
                logger.warning("No VIX data found in date range")
                self.vix_data = None
                return
            
            # Combine and prepare
            vix_df = pd.concat(vix_chunks)
            if not isinstance(vix_df.index, pd.DatetimeIndex):
                vix_df.index = pd.to_datetime(vix_df.index)
            
            # Use 'close' column if available, otherwise first numeric column
            if 'close' in vix_df.columns:
                vix_series = vix_df['close'].sort_index()
            elif 'Close' in vix_df.columns:
                vix_series = vix_df['Close'].sort_index()
            else:
                vix_series = vix_df.iloc[:, 0].sort_index()

            # --- FIX: Remove duplicates before resampling ---
            vix_series = vix_series[~vix_series.index.duplicated(keep='last')]

            # --- FIX: Resample to daily for daily bars (eliminates O(n²) per-bar slicing) ---
            vix_series = vix_series.resample('1D').ffill()
            
            # Create timezone-aware date range matching VIX data timezone
            date_range_tz = pd.date_range(start_date, end_date, freq='1D', tz=vix_series.index.tz)
            vix_series = vix_series.reindex(date_range_tz, method='ffill')
            vix_series.name = 'vix'
            self.vix_data = vix_series
            
            logger.info(f"Loaded and resampled VIX data: {len(self.vix_data)} points")
            
        except Exception as e:
            logger.error(f"Failed to load VIX data: {e}")
            self.vix_data = None
    
    def load_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load tick data from cache using existing tick_data_loader infrastructure.
        
        Uses TickDataLoader class to access Finnhub/Alpaca cached parquet files.
        """
        try:
            # --- FIX: Use Path.cwd() instead of __file__ for reliability in interactive environments ---
            sys.path.insert(0, str(Path.cwd()))
            from tick_data_loader import TickDataLoader
            
            # Initialize loader
            loader = TickDataLoader()
            
            # Load data for date range
            logger.info(f"Loading tick data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            tick_data = loader.load_symbol_range(
                ticker,
                start_date.date(),
                end_date.date()
            )
            
            if tick_data.empty:
                logger.warning(f"No tick data found for {ticker}")
                return pd.DataFrame()
            
            # Ensure datetime index with timezone
            if not isinstance(tick_data.index, pd.DatetimeIndex):
                # Try common timestamp column names
                if 'timestamp' in tick_data.columns:
                    tick_data = tick_data.set_index('timestamp')
                elif 'datetime' in tick_data.columns:
                    tick_data = tick_data.set_index('datetime')
                else:
                    logger.error(f"No timestamp/datetime column in tick data for {ticker}")
                    return pd.DataFrame()
            
            # Localize to UTC if naive
            if tick_data.index.tz is None:
                tick_data.index = tick_data.index.tz_localize('UTC')
            
            # Rename 'size' to 'volume' if present (Alpaca format)
            if 'size' in tick_data.columns:
                tick_data.rename(columns={'size': 'volume'}, inplace=True)
            
            logger.info(f"Loaded {len(tick_data):,} ticks for {ticker}")
            
            return tick_data
            
        except ImportError as e:
            logger.error(f"Failed to import tick_data_loader: {e}")
            logger.error("Make sure 'tick_data_loader.py' is in the current directory.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def resample_to_bars(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Resample tick data to OHLCV bars."""
        if tick_data.empty:
            return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(tick_data.index, pd.DatetimeIndex):
            tick_data['timestamp'] = pd.to_datetime(tick_data['timestamp'])
            tick_data.set_index('timestamp', inplace=True)
        
        # Resample
        freq = f"{self.params['bar_size_minutes']}min"  # Use 'min' instead of deprecated 'T'
        bars = tick_data.resample(freq).agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        bars.dropna(inplace=True)
        
        return bars
    
    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features for ML models.
        
        Features:
        1. ROC (Rate of Change) - returns
        2. SMA deviation - price vs moving average
        3. ATR - volatility measure
        4. Volume ratio - current vs average
        5. RSI - momentum oscillator (14-period)
        6. MACD histogram - trend strength
        """
        features = bars.copy()
        
        # ROC (returns)
        features['roc'] = bars['close'].pct_change()
        
        # SMA deviation
        sma_period = self.params['sma_period']
        features['sma'] = bars['close'].rolling(sma_period).mean()
        features['sma_dev'] = (bars['close'] - features['sma']) / features['sma']
        
        # ATR
        atr_period = self.params['atr_period']
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(atr_period).mean()
        features['atr_pct'] = features['atr'] / bars['close']
        
        # Volume ratio
        vol_sma = bars['volume'].rolling(sma_period).mean()
        features['vol_ratio'] = bars['volume'] / vol_sma
        
        # --- NEW: RSI (14-period) for momentum detection ---
        delta = bars['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # --- NEW: MACD histogram for trend strength ---
        ema12 = bars['close'].ewm(span=12, adjust=False).mean()
        ema26 = bars['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features['macd_hist'] = macd - macd_signal
        
        # Clean up
        features.dropna(inplace=True)
        
        return features
    
    def train_hmm(self, ticker: str, features: pd.DataFrame) -> hmm.GaussianHMM:
        """
        Train 3-state HMM on ROC (returns) to identify market regimes.
        
        CRITICAL FIX based on QuantConnect research (Sharpe 1.9):
        - Use ROC (returns) not sma_dev for better regime detection
        - Use 3 states (stable/neutral/volatile) not 2
        - Identify states by volatility to prevent collapse to single state
        
        States:
        - State 0: Low volatility (stable/ranging)
        - State 1: Medium volatility (neutral)
        - State 2: High volatility (trending/breakout)
        """
        n_components = 3  # FIXED: Always 3 states (overrides params)
        
        # Train on ROC (returns) - both successful strategies use this
        hmm_input_values = features['roc'].values.reshape(-1, 1)
        
        # Check minimum samples
        min_samples_required = n_components * 10
        if len(hmm_input_values) < min_samples_required:
            raise ValueError(
                f"Insufficient samples for HMM training: {len(hmm_input_values)} < {min_samples_required}"
            )
        
        # Train HMM with fixed random state for reproducibility
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type='diag',
            n_iter=100,
            random_state=100  # Same as QuantConnect
        )
        
        model.fit(hmm_input_values)
        
        # CRITICAL: Identify states by volatility (not mean)
        # This prevents all data collapsing into one state
        predicted_states = model.predict(hmm_input_values)
        state_stats = []
        
        for state_idx in range(n_components):
            state_data = hmm_input_values[predicted_states == state_idx]
            if len(state_data) > 0:
                state_stats.append({
                    'idx': state_idx,
                    'count': len(state_data),
                    'mean': np.mean(state_data),
                    'vol': np.std(state_data)
                })
            else:
                state_stats.append({
                    'idx': state_idx,
                    'count': 0,
                    'mean': 0,
                    'vol': 0
                })
        
        # Sort by volatility: low=0, medium=1, high=2
        sorted_by_vol = sorted(state_stats, key=lambda x: x['vol'])
        
        # Create volatility-based mapping
        model.vol_state_map = {
            sorted_by_vol[0]['idx']: 0,  # Stable
            sorted_by_vol[1]['idx']: 1,  # Neutral
            sorted_by_vol[2]['idx']: 2   # Volatile
        }
        
        # Log regime characteristics
        means = model.means_.flatten()
        logger.info(f"[{ticker}] Trained 3-state HMM on ROC. Means: {means}")
        
        regime_names = ['Stable', 'Neutral', 'Volatile']
        for orig_idx, mapped_idx in model.vol_state_map.items():
            stats = state_stats[orig_idx]
            pct = (stats['count'] / len(predicted_states)) * 100
            logger.info(f"[{ticker}] State {orig_idx}→{mapped_idx} ({regime_names[mapped_idx]}): "
                       f"{stats['count']} samples ({pct:.1f}%), mean={stats['mean']:.6f}, vol={stats['vol']:.6f}")
        
        return model
    
    def predict_regime(self, ticker: str, features: pd.DataFrame) -> np.ndarray:
        """Predict regime for each time period using ROC."""
        if ticker not in self.hmm_models:
            return np.array([])
        
        model = self.hmm_models[ticker]
        # Use ROC for regime prediction (same as training)
        hmm_input_values = features['roc'].values.reshape(-1, 1)
        
        # Get raw predictions
        raw_regimes = model.predict(hmm_input_values)
        
        # Map to volatility-sorted regimes using vol_state_map
        regimes = np.array([model.vol_state_map[r] for r in raw_regimes])
        
        return regimes
    
    def train_specialist_models(self, ticker: str, features: pd.DataFrame, regimes: np.ndarray):
        """
        Train separate Random Forest models for each regime.
        
        Target: Price movement N bars ahead (default 5 bars = 25 minutes @ 5min bars)
        This filters out bar-to-bar noise and gives trades time to develop.
        """
        if ticker not in self.rf_models:
            self.rf_models[ticker] = {}
        
        # Create target (next-bar return)
        features = features.copy()  # Avoid modifying original
        
        # --- SIMPLIFIED: Predict next bar (not 5 bars ahead) for faster signal ---
        features['target'] = features['roc'].shift(-1)
        features['target_class'] = (features['target'] > 0).astype(int)
        
        # Add regime column
        features['regime'] = regimes
        
        # --- FIX: Drop last row to avoid lookahead from shift(-1) ---
        features_clean = features.iloc[:-1].dropna()
        
        # Feature columns for ML (simplified for speed)
        feature_cols = ['roc', 'sma_dev', 'atr_pct', 'vol_ratio', 'rsi', 'macd_hist']
        
        # Train model for each regime
        n_components = self.params['hmm_n_components']
        min_samples = self.params['min_regime_samples']
        n_estimators = self.params['rf_n_estimators']
        
        for regime_id in range(n_components):
            regime_data = features_clean[features_clean['regime'] == regime_id]
            
            if len(regime_data) < min_samples:
                logger.warning(f"[{ticker}] Regime {regime_id}: Insufficient data ({len(regime_data)} < {min_samples})")
                self.rf_models[ticker].pop(regime_id, None)  # Remove stale model
                continue
            
            # Check if we have both classes
            if len(regime_data['target_class'].unique()) < 2:
                logger.warning(f"[{ticker}] Regime {regime_id}: Only one class present")
                self.rf_models[ticker].pop(regime_id, None)
                continue
            
            # Train model
            X = regime_data[feature_cols].values
            y = regime_data['target_class'].values
            
            # --- FIX: Use TimeSeriesSplit instead of random split for time-series data ---
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Use last fold for validation
            train_idx, val_idx = list(tscv.split(X))[-1]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Check for class imbalance in validation
            if len(np.unique(y_val)) < 2:
                logger.warning(f"[{ticker}] Regime {regime_id}: Validation set has single class, using full data")
                X_train, X_val = X, X
                y_train, y_val = y, y
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Validate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            self.rf_models[ticker][regime_id] = model
            
            logger.info(f"[{ticker}] Regime {regime_id}: Trained RF (n={len(regime_data)}, "
                       f"acc={accuracy:.3f}, next-bar target, train/val={len(X_train)}/{len(X_val)})")
    
    def generate_signal(self, ticker: str, current_features: pd.Series, current_regime: int) -> Tuple[str, float]:
        """
        Generate trading signal using specialist RF model.
        
        Returns:
            (direction, confidence) where direction is 'LONG', 'SHORT', or 'FLAT'
        """
        # Check if we have a model for this regime
        if ticker not in self.rf_models or current_regime not in self.rf_models[ticker]:
            return 'FLAT', 0.5
        
        model = self.rf_models[ticker][current_regime]
        
        # Prepare features (including new RSI and MACD)
        feature_cols = ['roc', 'sma_dev', 'atr_pct', 'vol_ratio', 'rsi', 'macd_hist']
        X = current_features[feature_cols].values.reshape(1, -1)

        # Check for NaNs in features
        if np.isnan(X).any():
            return 'FLAT', 0.5
        
        # Get probability
        prob = model.predict_proba(X)[0]
        
        # Ensure model has two classes (0 and 1)
        if len(prob) < 2:
            return 'FLAT', 0.5

        prob_up = prob[1]  # Probability of class 1 (up move)
        
        # Apply confidence filter
        threshold = self.params['confidence_threshold']
        
        if prob_up > threshold:
            return 'LONG', prob_up
        elif prob_up < (1 - threshold):
            return 'SHORT', 1 - prob_up
        else:
            return 'FLAT', 0.5
    
    def calculate_position_size(self, ticker: str, price: float, atr: float) -> Tuple[int, float]:
        """
        Calculate position size based on ATR risk.
        
        Returns:
            (shares, stop_price)
        """
        # FIX: Use equity (total account value) not capital (available cash)
        # Position sizing should be based on total account size
        equity = self.capital + sum(
            pos['shares'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
        )
        
        # Risk per trade
        risk_amount = equity * (self.params['position_size_pct'] / 100)
        
        # Stop distance in ATR
        stop_distance = atr * self.params['atr_stop_multiplier']
        stop_price = price - stop_distance  # For long positions
        
        # Position size
        risk_per_share = abs(price - stop_distance)
        if risk_per_share == 0:
            logger.warning(f"[{ticker}] DEBUG: risk_per_share=0 (price={price:.2f}, atr={atr:.4f}, stop_distance={stop_distance:.4f})")
            return 0, price
        
        shares = int(risk_amount / risk_per_share)
        
        # DEBUG logging
        if shares == 0:
            logger.warning(f"[{ticker}] DEBUG: equity={equity:.2f}, capital={self.capital:.2f}, "
                          f"risk_amount={risk_amount:.2f}, price={price:.2f}, atr={atr:.4f}, "
                          f"risk_per_share={risk_per_share:.4f}, calc_shares={risk_amount/risk_per_share:.4f}")
        
        # Limit by available capital
        max_shares = int(self.capital / price * 0.99)  # Use 99% max
        shares = min(shares, max_shares)
        
        return shares, stop_price
    
    def open_position(self, ticker: str, direction: str, price: float, atr: float, timestamp: datetime):
        """Open a new position with ATR-based sizing and stop."""
        if ticker in self.positions:
            logger.warning(f"[{ticker}] Attempt to open position while one exists")
            return
        
        shares, stop_price = self.calculate_position_size(ticker, price, atr)
        
        if shares == 0:
            logger.warning(f"[{ticker}] Position size is 0, skipping")
            return
        
        cost = shares * price
        
        # --- FIX: Correct cash flow for LONG vs SHORT ---
        if direction == 'LONG':
            if cost > self.capital:
                logger.warning(f"[{ticker}] Insufficient capital (need ${cost:,.2f}, have ${self.capital:,.2f})")
                return
            self.capital -= cost  # Spend cash to buy
            stop_price = price - (atr * self.params['atr_stop_multiplier'])
        else:  # SHORT
            self.capital += cost  # Receive cash from short sale
            stop_price = price + (atr * self.params['atr_stop_multiplier'])
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_time': timestamp,
            'direction': direction,
            'stop_price': stop_price,
            'atr': atr
        }
        
        logger.info(f"[{timestamp}] OPEN {direction} {shares} {ticker} @ ${price:.2f}, stop=${stop_price:.2f}")
    
    def close_position(self, ticker: str, price: float, timestamp: datetime, reason: str):
        """Close an open position and record the trade."""
        if ticker not in self.positions:
            return
        
        pos = self.positions.pop(ticker)
        proceeds = pos['shares'] * price
        entry_cost = pos['shares'] * pos['entry_price']
        
        # --- FIX: Correct cash flow and P&L for LONG vs SHORT ---
        if pos['direction'] == 'LONG':
            self.capital += proceeds  # Receive cash from sale
            pnl = proceeds - entry_cost
        else:  # SHORT
            self.capital -= proceeds  # Pay cash to buy back
            pnl = entry_cost - proceeds
        
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0
        hold_time = (timestamp - pos['entry_time']).total_seconds() / 60  # minutes
        
        self.closed_trades.append({
            'ticker': ticker,
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'direction': pos['direction'],
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time_min': hold_time,
            'reason': reason
        })
        
        logger.info(f"[{timestamp}] CLOSE {ticker} @ ${price:.2f} → ${pnl:+.2f} ({pnl_pct:+.2f}%) [{reason}]")
    
    def calculate_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate final performance metrics."""
        if equity_curve.empty or len(equity_curve) < 2:
            return {
                'sharpe': 0.0,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0
            }
        
        # P&L
        total_pnl = equity_curve.iloc[-1] - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Returns - resample to daily for proper annualization
        daily_equity = equity_curve.resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        if daily_returns.empty or daily_returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Max Drawdown
        peak = equity_curve.cummax()
        drawdown = equity_curve - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (drawdown / peak).min() * 100
        
        # Trade statistics
        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(self.closed_trades) * 100 if self.closed_trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        # --- IMPROVEMENT: Add Profit Factor and Calmar Ratio ---
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calmar Ratio: Annualized Return / Max Drawdown
        days_traded = (equity_curve.index[-1] - equity_curve.index[0]).days
        annualized_return = (pnl_pct / 100) * (365 / days_traded) if days_traded > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        return {
            'sharpe': sharpe,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        }
    
    def backtest(self, ticker: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Run a walk-forward backtest with periodic model retraining.
        
        This prevents lookahead bias by:
        1. Training models only on past data
        2. Retraining every N days
        3. Bar-by-bar simulation with position management
        """
        logger.info(f"Starting walk-forward backtest: {ticker} from {start_date.date()} to {end_date.date()}")
        
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.regime_history = []
        equity_history = {start_date - timedelta(microseconds=1): self.initial_capital}
        
        # 1. Load VIX and tick data
        self.load_vix_data(start_date, end_date)
        tick_data = self.load_data(ticker, start_date, end_date)
        
        if tick_data.empty:
            logger.warning(f"No data for {ticker}")
            return {'error': 'No data', 'ticker': ticker}
        
        # 2. Resample to bars and compute features
        bars = self.resample_to_bars(tick_data)
        if len(bars) < self.params['roc_window'] * 2:
            logger.warning(f"Insufficient bars for {ticker} ({len(bars)} < {self.params['roc_window'] * 2})")
            return {'error': 'Insufficient data', 'ticker': ticker, 'bars': len(bars)}
        
        features_full = self.compute_features(bars)
        features_full = features_full.dropna()

        # --- IMPROVEMENT: Pre-merge VIX data for O(1) lookup instead of O(n) per-bar slicing ---
        if self.vix_data is not None:
            features_full = pd.merge_asof(
                features_full.sort_index(), 
                self.vix_data.sort_index(), 
                left_index=True, 
                right_index=True, 
                direction='backward'
            )
            features_full = features_full.dropna(subset=['vix'])
            logger.info(f"Merged VIX data. Feature rows: {len(features_full)}")
        
        if len(features_full) < self.params['roc_window']:
            logger.warning(f"Insufficient features after VIX merge for {ticker}")
            return {'error': 'Insufficient features', 'ticker': ticker}
        
        logger.info(f"Loaded {len(bars):,} bars, {len(features_full):,} feature rows for {ticker}")
        
        # 3. Walk-Forward Simulation Loop
        next_retrain_time = features_full.index[self.params['roc_window']]
        retrain_freq = timedelta(days=self.params['retrain_frequency_days'])
        
        for i, (timestamp, row) in enumerate(features_full.iterrows()):
            current_price = row['close']
            current_atr = row['atr']
            current_low = row['low']
            current_high = row['high']
            
            # --- 3.1 Check Stop-Loss & Take-Profit ---
            if ticker in self.positions:
                pos = self.positions[ticker]
                stopped = False
                
                # --- FIX: Use optimizable take_profit_multiplier from params ---
                take_profit_multiplier = self.params.get('take_profit_multiplier', 3.0)
                atr_risk_distance = pos['atr'] * self.params['atr_stop_multiplier']
                
                if pos['direction'] == 'LONG':
                    # Check stop-loss
                    if current_low <= pos['stop_price']:
                        self.close_position(ticker, pos['stop_price'], timestamp, "Stop-Loss")
                        stopped = True
                    # Check take-profit
                    elif current_high >= (pos['entry_price'] + (atr_risk_distance * take_profit_multiplier)):
                        tp_price = pos['entry_price'] + (atr_risk_distance * take_profit_multiplier)
                        self.close_position(ticker, tp_price, timestamp, "Take-Profit")
                        stopped = True
                        
                elif pos['direction'] == 'SHORT':
                    # Check stop-loss
                    if current_high >= pos['stop_price']:
                        self.close_position(ticker, pos['stop_price'], timestamp, "Stop-Loss")
                        stopped = True
                    # Check take-profit
                    elif current_low <= (pos['entry_price'] - (atr_risk_distance * take_profit_multiplier)):
                        tp_price = pos['entry_price'] - (atr_risk_distance * take_profit_multiplier)
                        self.close_position(ticker, tp_price, timestamp, "Take-Profit")
                        stopped = True
                
                if stopped:
                    # Update equity after stop/tp
                    current_value = self.capital
                    equity_history[timestamp] = current_value
                    continue
            
            # --- 3.2 Periodic Model Retraining ---
            if timestamp >= next_retrain_time:
                logger.info(f"[{timestamp.date()}] Re-training models...")
                
                # --- FIX: Train on data strictly before current bar to avoid lookahead ---
                # Exclude current bar by using timestamp - 1 bar
                bar_offset = pd.Timedelta(minutes=self.params['bar_size_minutes'])
                train_cutoff = timestamp - bar_offset
                train_features = features_full.loc[:train_cutoff]
                
                try:
                    # Train HMM
                    self.hmm_models[ticker] = self.train_hmm(ticker, train_features)
                    
                    # Predict regimes for training data
                    train_regimes = self.predict_regime(ticker, train_features)
                    
                    # Train specialist RF models
                    self.train_specialist_models(ticker, train_features, train_regimes)
                    
                except Exception as e:
                    logger.error(f"Training failed at {timestamp}: {e}")
                
                next_retrain_time += retrain_freq
            
            # --- 3.3 VIX Filter (Fast column access from pre-merged data) ---
            if self.vix_data is not None:
                current_vix = row['vix']
                if pd.isna(current_vix):
                    pass  # No VIX data yet, allow trading
                elif current_vix > self.params['vix_threshold']:
                    # High VIX - close positions and skip trading
                    if ticker in self.positions:
                        self.close_position(ticker, current_price, timestamp, "VIX Filter")
                        equity_history[timestamp] = self.capital
                    continue
            
            # --- 3.4 Check if Models are Trained ---
            if ticker not in self.hmm_models:
                continue  # Models not ready yet
            
            # --- 3.5 Predict Regime and Generate Signal ---
            try:
                # Predict current regime using ROC (same as training)
                hmm_input_val = np.array([[row['roc']]])
                if np.isnan(hmm_input_val).any():
                    continue

                current_regime_raw = self.hmm_models[ticker].predict(hmm_input_val)[0]
                current_regime = self.hmm_models[ticker].vol_state_map[current_regime_raw]
                
                self.regime_history.append({
                    'time': timestamp,
                    'regime': current_regime,
                    'price': current_price
                })
                
                # Generate trading signal
                direction, confidence = self.generate_signal(ticker, row, current_regime)
                
            except Exception as e:
                logger.error(f"Signal generation failed at {timestamp}: {e}")
                continue
            
            # --- 3.6 Execute Trades ---
            if ticker not in self.positions:
                # Open new position if signal is strong
                if direction == 'LONG':
                    self.open_position(ticker, 'LONG', current_price, current_atr, timestamp)
                # Enable shorting:
                elif direction == 'SHORT':
                    self.open_position(ticker, 'SHORT', current_price, current_atr, timestamp)
            else:
                # Check for exit signal (signal flip)
                pos_direction = self.positions[ticker]['direction']
                
                if (pos_direction == 'LONG' and direction == 'SHORT'):
                    self.close_position(ticker, current_price, timestamp, "Signal Flip")
                elif (pos_direction == 'SHORT' and direction == 'LONG'):
                    self.close_position(ticker, current_price, timestamp, "Signal Flip")
            
            # --- 3.7 Update Equity Curve (Correct for LONG and SHORT) ---
            current_value = self.capital
            if ticker in self.positions:
                pos = self.positions[ticker]
                if pos['direction'] == 'LONG':
                    current_value += pos['shares'] * current_price  # Add long market value
                else:  # SHORT
                    current_value -= pos['shares'] * current_price  # Subtract short liability
            
            equity_history[timestamp] = current_value
        
        # 4. Close any remaining positions
        if ticker in self.positions:
            final_price = features_full.iloc[-1]['close']
            final_time = features_full.index[-1]
            self.close_position(ticker, final_price, final_time, "End of Backtest")
        
        # 5. Calculate Metrics
        equity_curve = pd.Series(equity_history).sort_index()
        metrics = self.calculate_metrics(equity_curve)
        
        # --- IMPROVEMENT: Calculate buy-and-hold benchmark ---
        bh_return_pct = ((features_full['close'].iloc[-1] / features_full['close'].iloc[0]) - 1) * 100
        metrics['benchmark_bh_pct'] = bh_return_pct
        
        final_results = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'params': self.params.copy(),
            'equity_curve': equity_curve,  # Include for plotting
            'closed_trades': self.closed_trades.copy(),  # Include for analysis
            **metrics
        }
        
        logger.info(f"Backtest complete: {ticker} → {metrics['total_trades']} trades, "
                   f"${metrics['pnl']:+,.2f} ({metrics['pnl_pct']:+.2f}%), Sharpe {metrics['sharpe']:.2f}, "
                   f"B&H: {bh_return_pct:+.2f}%")
        
        return final_results


def create_objective(ticker: str, start_date: datetime, end_date: datetime):
    """
    Factory function to create Optuna objective.
    AGGRESSIVE: Optimize for absolute P&L, not Sharpe.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        """Optuna objective to maximize P&L percentage."""
        
        # --- AGGRESSIVE SEARCH SPACE ---
        params = {
            # 1. Test different timeframes (wider range for 2024 momentum)
            'bar_size_minutes': trial.suggest_categorical('bar_size_minutes', [3, 5, 10, 15]),
            
            # 2. Test WIDER stops for bigger wins (expanded range)
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 3.5, step=0.25),

            # 3. AGGRESSIVE position sizing (expanded to 4%)
            'position_size_pct': trial.suggest_float('position_size_pct', 1.5, 4.0, step=0.5),

            # 4. Lower confidence threshold = more trades (expanded range)
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.50, 0.58, step=0.01),

            # 5. Optimize take-profit (2-6x stop distance)
            'take_profit_multiplier': trial.suggest_float('take_profit_multiplier', 2.0, 6.0, step=1.0),

            # 6. VIX threshold (allow trading through mild vol)
            'vix_threshold': trial.suggest_float('vix_threshold', 25.0, 40.0, step=2.5),

            # 7. Tune SMA for momentum capture
            'sma_period': trial.suggest_int('sma_period', 8, 20, step=2),

            # Keep these stable
            'hmm_n_components': 3,
            'roc_window': 100,
            'atr_period': 14,
            'rf_n_estimators': 100,
            'retrain_frequency_days': 30,
            'min_regime_samples': 10  # Reduced from 20 for more regime data
        }
        
        try:
            strategy = HMMRegimeMLStrategy(params)
            results = strategy.backtest(ticker, start_date, end_date)
            
            pnl_pct = results.get('pnl_pct', -100)
            sharpe = results.get('sharpe', -1.0)
            trades = results.get('total_trades', 0)
            max_dd = abs(results.get('max_drawdown_pct', -100))

            # --- OPTIMIZE FOR P&L with less harsh penalties ---
            # Require: positive P&L, reasonable trade count, acceptable drawdown
            if trades < 20:  # Reduced from 30 for more exploration
                return pnl_pct - 5.0  # Penalty instead of hard failure
            if pnl_pct < 0:  # Must be profitable
                return pnl_pct  # Return negative P&L as penalty
            if max_dd > 20:  # Slightly higher tolerance (was 15%)
                return pnl_pct - (max_dd - 20)  # Penalty for excessive DD
                
            # Return P&L as objective (maximize absolute return)
            objective_value = pnl_pct
            
            logger.info(f"Trial {trial.number}: P&L={pnl_pct:.2f}%, Sharpe={sharpe:.2f}, "
                        f"Trades={trades}, DD={max_dd:.2f}%, BarSize={params['bar_size_minutes']}min, "
                        f"Stop={params['atr_stop_multiplier']:.2f}x, TP={params['take_profit_multiplier']:.1f}x, "
                        f"PosSize={params['position_size_pct']:.1f}%, Conf={params['confidence_threshold']:.2f}, "
                        f"SMA={params['sma_period']}, VIX<{params['vix_threshold']:.0f}")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -100.0
    
    return objective


def run_optuna_optimization(objective_func, n_trials: int = 100):
    """
    Run Optuna optimization to find best parameters.
    
    Returns dict of best parameters.
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not installed. Cannot run optimization.")
        return None
    
    # Callback to log progress
    def logging_callback(study, trial):
        logger.info(f"Trial {trial.number}/{n_trials} complete | "
                   f"Value: {trial.value:.3f} | "
                   f"Best so far: {study.best_value:.3f} (Trial #{study.best_trial.number})")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='hmm_ml_optimization'
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"OPTUNA OPTIMIZATION - ENHANCED FOR 2024 MOMENTUM")
    logger.info(f"{'='*80}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Search Space:")
    logger.info(f"  - Bar size: 3-15 min")
    logger.info(f"  - Stop multiplier: 1.5-3.5x ATR")
    logger.info(f"  - Position size: 1.5-4.0%")
    logger.info(f"  - Confidence: 0.50-0.58")
    logger.info(f"  - Take-profit: 2-6x stop")
    logger.info(f"  - VIX threshold: 25-40")
    logger.info(f"  - SMA period: 8-20")
    logger.info(f"Features: ROC, SMA_dev, ATR, Vol_ratio, RSI, MACD_hist")
    logger.info(f"Target: Maximize P&L% (5-bar prediction horizon)")
    logger.info(f"Estimated time: ~{n_trials * 1.5:.0f} minutes\n")
    logger.info(f"{'='*80}\n")
    
    study.optimize(objective_func, n_trials=n_trials, 
                   callbacks=[logging_callback],
                   show_progress_bar=False)  # Disable progress bar, use logging instead
    
    logger.info(f"\n{'='*80}")
    logger.info(f"OPTUNA OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best P&L: {study.best_value:.2f}%")
    logger.info(f"Best params:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"{'='*80}\n")
    
    # Return complete params (best_params may be missing fixed params)
    best_params_complete = HMMRegimeMLStrategy().params
    best_params_complete.update(study.best_params)
    
    return best_params_complete


def run_walk_forward_oos(
    ticker: str,
    full_start: datetime,
    full_end: datetime,
    is_periods: int = 6,
    oos_periods: int = 1,
    period_days: int = 30,
    optuna_trials: int = 50
):
    """
    Walk-Forward Out-of-Sample validation with optimization.
    
    Process:
    1. Train on IS period (6 months) with Optuna optimization
    2. Test on OOS period (1 month) with best parameters
    3. Roll forward and repeat
    
    This prevents curve-fitting by optimizing on past data only.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"WALK-FORWARD OUT-OF-SAMPLE VALIDATION: {ticker}")
    logger.info(f"{'='*80}")
    logger.info(f"IS: {is_periods} periods × {period_days} days = {is_periods * period_days} days")
    logger.info(f"OOS: {oos_periods} periods × {period_days} days = {oos_periods * period_days} days")
    logger.info(f"Optuna trials per IS period: {optuna_trials}")
    logger.info(f"{'='*80}\n")
    
    results_all = []
    
    current_date = full_start
    is_delta = timedelta(days=period_days * is_periods)
    oos_delta = timedelta(days=period_days * oos_periods)
    
    period_num = 1
    
    while current_date + is_delta + oos_delta <= full_end:
        is_start = current_date
        is_end = is_start + is_delta
        oos_start = is_end
        oos_end = oos_start + oos_delta
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"PERIOD #{period_num}")
        logger.info(f"{'─'*80}")
        logger.info(f"IS:  {is_start.date()} to {is_end.date()} ({(is_end - is_start).days} days)")
        logger.info(f"OOS: {oos_start.date()} to {oos_end.date()} ({(oos_end - oos_start).days} days)")
        
        # 1. Optimize on IS period
        if OPTUNA_AVAILABLE and optuna_trials > 0:
            logger.info(f"\n▶ Running Optuna optimization ({optuna_trials} trials)...")
            objective_func = create_objective(ticker, is_start, is_end)
            best_params = run_optuna_optimization(objective_func, n_trials=optuna_trials)
        else:
            logger.warning("Optuna unavailable or trials=0. Using default params.")
            best_params = HMMRegimeMLStrategy().params
        
        # 2. Test on OOS period with best params
        logger.info(f"\n▶ Testing OOS with optimized params...")
        strategy_oos = HMMRegimeMLStrategy(best_params)
        oos_results = strategy_oos.backtest(ticker, oos_start, oos_end)
        
        # Add period info
        oos_results['period'] = period_num
        oos_results['is_start'] = is_start
        oos_results['is_end'] = is_end
        oos_results['oos_start'] = oos_start
        oos_results['oos_end'] = oos_end
        
        results_all.append(oos_results)
        
        logger.info(f"\n✓ Period #{period_num} OOS Results:")
        logger.info(f"  Trades: {oos_results.get('total_trades', 0)}")
        logger.info(f"  P&L: ${oos_results.get('pnl', 0):+,.2f} ({oos_results.get('pnl_pct', 0):+.2f}%)")
        logger.info(f"  Sharpe: {oos_results.get('sharpe', 0):.3f}")
        logger.info(f"  Win Rate: {oos_results.get('win_rate', 0):.1f}%")
        
        # Roll forward by OOS period
        current_date += oos_delta
        period_num += 1
    
    # Aggregate all OOS results
    logger.info(f"\n{'='*80}")
    logger.info(f"WALK-FORWARD AGGREGATE RESULTS ({len(results_all)} periods)")
    logger.info(f"{'='*80}")
    
    total_trades = sum(r.get('total_trades', 0) for r in results_all)
    total_pnl = sum(r.get('pnl', 0) for r in results_all)
    avg_sharpe = np.mean([r.get('sharpe', 0) for r in results_all if 'sharpe' in r])
    
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Total P&L: ${total_pnl:+,.2f}")
    logger.info(f"Avg Sharpe: {avg_sharpe:.3f}")

    # --- IMPROVEMENT: Combine equity curves for complete picture ---
    full_equity_curve = pd.concat([r['equity_curve'] for r in results_all if 'equity_curve' in r])
    full_equity_curve = full_equity_curve.loc[~full_equity_curve.index.duplicated(keep='last')]
    
    logger.info(f"\nCombined Equity Curve:")
    final_metrics = HMMRegimeMLStrategy().calculate_metrics(full_equity_curve)
    logger.info(f"  Total P&L: ${final_metrics['pnl']:+,.2f} ({final_metrics['pnl_pct']:+.2f}%)")
    logger.info(f"  Sharpe:    {final_metrics['sharpe']:.3f}")
    logger.info(f"  Max DD:    {final_metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"{'='*80}\n")
    
    return results_all, full_equity_curve


def run_monte_carlo_validation(
    base_strategy: HMMRegimeMLStrategy,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    n_simulations: int = 500,
    jitter_pct: float = 0.02
):
    """
    Monte Carlo simulation to test parameter robustness.
    
    Adds random noise to parameters to see if results degrade gracefully
    or if the strategy is overfit to specific parameter values.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"MONTE CARLO VALIDATION: {ticker}")
    logger.info(f"{'='*80}")
    logger.info(f"Simulations: {n_simulations}")
    logger.info(f"Parameter jitter: ±{jitter_pct*100}%")
    logger.info(f"{'='*80}\n")
    
    results = []
    
    for i in range(n_simulations):
        # Add jitter to parameters
        jittered_params = base_strategy.params.copy()
        
        # Jitter continuous parameters
        jittered_params['confidence_threshold'] += np.random.uniform(-jitter_pct, jitter_pct)
        jittered_params['atr_stop_multiplier'] += np.random.uniform(-jitter_pct * 2, jitter_pct * 2)
        jittered_params['position_size_pct'] += np.random.uniform(-jitter_pct * 2, jitter_pct * 2)
        
        # Clip to valid ranges
        jittered_params['confidence_threshold'] = np.clip(jittered_params['confidence_threshold'], 0.51, 0.70)
        jittered_params['atr_stop_multiplier'] = np.clip(jittered_params['atr_stop_multiplier'], 1.0, 3.0)
        jittered_params['position_size_pct'] = np.clip(jittered_params['position_size_pct'], 0.5, 5.0)
        
        # Run backtest
        try:
            strategy = HMMRegimeMLStrategy(jittered_params)
            result = strategy.backtest(ticker, start_date, end_date)
            results.append(result)
        except Exception as e:
            logger.error(f"Simulation {i+1} failed: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{n_simulations} simulations complete")
    
    # Analyze distribution
    sharpes = [r.get('sharpe', 0) for r in results if 'sharpe' in r]
    pnls = [r.get('pnl', 0) for r in results if 'pnl' in r]
    win_rates = [r.get('win_rate', 0) for r in results if 'win_rate' in r]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MONTE CARLO RESULTS (n={len(results)})")
    logger.info(f"{'='*80}")
    
    if sharpes:
        logger.info(f"\nSHARPE RATIO:")
        logger.info(f"  Mean:   {np.mean(sharpes):7.3f}")
        logger.info(f"  Median: {np.median(sharpes):7.3f}")
        logger.info(f"  Std:    {np.std(sharpes):7.3f}")
        logger.info(f"  5th:    {np.percentile(sharpes, 5):7.3f}")
        logger.info(f"  95th:   {np.percentile(sharpes, 95):7.3f}")
    
    if pnls:
        logger.info(f"\nP&L:")
        logger.info(f"  Mean:   ${np.mean(pnls):10,.2f}")
        logger.info(f"  Median: ${np.median(pnls):10,.2f}")
        logger.info(f"  Std:    ${np.std(pnls):10,.2f}")
        logger.info(f"  5th:    ${np.percentile(pnls, 5):10,.2f}")
        logger.info(f"  95th:   ${np.percentile(pnls, 95):10,.2f}")
    
    if win_rates:
        logger.info(f"\nWIN RATE:")
        logger.info(f"  Mean:   {np.mean(win_rates):6.1f}%")
        logger.info(f"  Median: {np.median(win_rates):6.1f}%")
    
    logger.info(f"\n{'='*80}\n")
    
    return results


# Example usage
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("HMM REGIME ML STRATEGY - MULTI-ASSET AGGRESSIVE MODE")
    logger.info("="*80)
    
    # Configuration - START WITH SPY ONLY (test daily data regime detection)
    # SPY has complete data in alpaca_etfs folder
    tickers = [
        'SPY',   # S&P 500 - daily bars for better HMM regime detection
    ]
    
    # Test dates - use 2024 for complete data
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    
    logger.info(f"\nTickers: {', '.join(tickers)} ({len(tickers)} assets)")
    logger.info(f"Period: {start.date()} to {end.date()}")
    logger.info(f"Duration: {(end - start).days} days")
    logger.info(f"Strategy: Aggressive (2 states, 0.52 conf, 3% sizing, 2.5x stops)\n")
    
    # ========================================================================
    # MULTI-ASSET PORTFOLIO BACKTEST
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("MULTI-ASSET PORTFOLIO BACKTEST")
    logger.info("="*80)
    
    # Run backtest for each ticker
    all_results = {}
    total_capital = 100_000
    capital_per_ticker = total_capital / len(tickers)  # Equal weight allocation
    
    logger.info(f"Total Capital: ${total_capital:,}")
    logger.info(f"Per-Asset Allocation: ${capital_per_ticker:,.2f} ({100/len(tickers):.1f}%)\n")
    
    for ticker in tickers:
        logger.info(f"\n{'─'*80}")
        logger.info(f"BACKTESTING: {ticker}")
        logger.info(f"{'─'*80}")
        
        try:
            strategy = HMMRegimeMLStrategy()
            strategy.initial_capital = capital_per_ticker
            strategy.capital = capital_per_ticker
            
            results = strategy.backtest(ticker, start, end)
            
            if results and 'error' not in results:
                all_results[ticker] = results
                logger.info(f"✓ {ticker} Complete: {results['total_trades']} trades, "
                           f"${results['pnl']:+,.2f} ({results['pnl_pct']:+.2f}%)")
            else:
                logger.warning(f"✗ {ticker} Failed: {results.get('error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"✗ {ticker} Error: {e}")
    
    # ========================================================================
    # AGGREGATE PORTFOLIO RESULTS
    # ========================================================================
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info(f"PORTFOLIO AGGREGATE RESULTS")
        logger.info(f"{'='*80}\n")
        
        # Per-ticker summary
        logger.info(f"{'Ticker':<10} {'Trades':>8} {'P&L':>12} {'P&L%':>8} {'Sharpe':>8} {'Max DD':>8} {'B&H%':>8}")
        logger.info(f"{'-'*80}")
        
        total_trades = 0
        total_pnl = 0
        weighted_sharpe = 0
        
        for ticker, res in all_results.items():
            total_trades += res.get('total_trades', 0)
            total_pnl += res.get('pnl', 0)
            weighted_sharpe += res.get('sharpe', 0)
            
            logger.info(f"{ticker:<10} {res.get('total_trades', 0):>8} "
                       f"${res.get('pnl', 0):>10,.2f} {res.get('pnl_pct', 0):>7.2f}% "
                       f"{res.get('sharpe', 0):>7.2f} {res.get('max_drawdown_pct', 0):>7.2f}% "
                       f"{res.get('benchmark_bh_pct', 0):>7.2f}%")
        
        logger.info(f"{'-'*80}")
        
        # Portfolio totals
        total_pnl_pct = (total_pnl / total_capital) * 100
        avg_sharpe = weighted_sharpe / len(all_results) if all_results else 0
        
        logger.info(f"{'PORTFOLIO':<10} {total_trades:>8} ${total_pnl:>10,.2f} {total_pnl_pct:>7.2f}% {avg_sharpe:>7.2f}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PORTFOLIO PERFORMANCE")
        logger.info(f"{'='*80}")
        logger.info(f"  Total Trades:     {total_trades}")
        logger.info(f"  Total P&L:        ${total_pnl:+,.2f}")
        logger.info(f"  Portfolio Return: {total_pnl_pct:+.2f}%")
        logger.info(f"  Avg Sharpe:       {avg_sharpe:.2f}")
        logger.info(f"  Assets Traded:    {len(all_results)}/{len(tickers)}")
        
        # Annualize for semester comparison
        days_traded = (end - start).days
        annualized_return = total_pnl_pct * (365 / days_traded)
        semester_return = total_pnl_pct * (180 / days_traded)
        
        logger.info(f"\n  Annualized Return: {annualized_return:+.2f}%")
        logger.info(f"  Semester Estimate: {semester_return:+.2f}% (target: 11%+)")
        
        if semester_return >= 11.0:
            logger.info(f"\n  ✓ TARGET ACHIEVED: {semester_return:.2f}% ≥ 11%")
        else:
            logger.info(f"\n  ✗ TARGET MISSED: {semester_return:.2f}% < 11%")
            logger.info(f"    Gap: {11.0 - semester_return:.2f}% needed")
        
        logger.info(f"{'='*80}\n")
        
        # Save combined results
        try:
            # Combine all equity curves
            all_equity = []
            for ticker, res in all_results.items():
                eq = res['equity_curve'].to_frame(name=ticker)
                all_equity.append(eq)
            
            if all_equity:
                combined_equity = pd.concat(all_equity, axis=1).fillna(method='ffill')
                combined_equity['PORTFOLIO'] = combined_equity.sum(axis=1)
                combined_equity.to_csv('portfolio_equity_curve.csv')
                logger.info("✓ Saved: portfolio_equity_curve.csv")
            
            # Save all trades
            all_trades = []
            for ticker, res in all_results.items():
                trades_df = pd.DataFrame(res['closed_trades'])
                if not trades_df.empty:
                    trades_df['ticker'] = ticker
                    all_trades.append(trades_df)
            
            if all_trades:
                pd.concat(all_trades).to_csv('portfolio_all_trades.csv', index=False)
                logger.info("✓ Saved: portfolio_all_trades.csv")
            
        except Exception as e:
            logger.warning(f"Could not save portfolio results: {e}")
    
    else:
        logger.error("No successful backtests. Check data availability.")
    
    logger.info("\n" + "="*80)
    logger.info("MULTI-ASSET TESTING COMPLETE")
    logger.info("="*80)
